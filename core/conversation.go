package core

import (
	"encoding/json"
	"fmt"
	"strings"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Provider string

const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
	ProviderGemini    Provider = "gemini"
)

// Part represents a structured segment of message content.
type Part struct {
	Type    string         `json:"type"`
	Content any            `json:"content,omitempty"`
	Meta    map[string]any `json:"meta,omitempty"`
	data    map[string]any
}

// SetMeta sets a metadata key-value pair on the Part.
func (p *Part) SetMeta(key string, val any) {
	if p.Meta == nil {
		p.Meta = make(map[string]any)
	}
	p.Meta[key] = val
}

func (p *Part) Import(blob any) error {
	if m, ok := blob.(map[string]any); ok {
		p.data = m
		p.Type, _ = m["type"].(string)

		// Map text content keys to keep both "content" and "text" compliant
		if p.Type == "text" {
			if val, exists := m["content"]; exists {
				p.Content = val
			} else if val, exists := m["text"]; exists {
				p.Content = val
			}
		} else {
			p.Content = m["content"]
		}

		p.Meta = extractMeta(m["meta"])
		return nil
	}

	if s, ok := blob.(string); ok {
		p.Type = "text"
		p.Content = s
		p.Meta = make(map[string]any)
		p.data = map[string]any{
			"type":    "text",
			"content": s,
		}
		return nil
	}

	return fmt.Errorf("invalid part content type: %T", blob)
}

func (p *Part) Export() (map[string]any, error) {
	if p.data == nil {
		p.data = make(map[string]any)
	}
	p.data["type"] = p.Type
	p.data["content"] = p.Content
	if len(p.Meta) > 0 {
		p.data["meta"] = p.Meta
	} else {
		delete(p.data, "meta")
	}
	return p.data, nil
}

func (p *Part) UnmarshalJSON(b []byte) error {
	var raw any
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	return p.Import(raw)
}

func (p *Part) MarshalJSON() ([]byte, error) {
	m, err := p.Export()
	if err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

// ToolFunction represents a function call specification within a tool call.
type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall represents a tool call invocation emitted by the assistant.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolDefinition represents a declared tool schema provided in a conversation request.
type ToolDefinition struct {
	Type     string         `json:"type"`
	Function map[string]any `json:"function"`
}

// Message represents a single chat turn.
type Message struct {
	Role       Role           `json:"role"`
	Parts      []Part         `json:"parts,omitempty"`
	ToolCalls  []ToolCall     `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
	data       map[string]any
}

// MetaMap returns the full metadata map for the Message.
func (m *Message) MetaMap() map[string]any {
	if m.data == nil {
		return nil
	}
	metaVal, exists := m.data["meta"]
	if !exists {
		return nil
	}
	meta, ok := metaVal.(map[string]any)
	if !ok {
		return nil
	}
	return meta
}

func (m *Message) Import(blob any) error {
	raw, ok := blob.(map[string]any)
	if !ok {
		return fmt.Errorf("message is not an object")
	}
	m.data = raw
	roleStr, _ := raw["role"].(string)
	m.Role = Role(roleStr)

	if name, ok := raw["name"].(string); ok {
		m.Name = name
	}
	if toolCallID, ok := raw["tool_call_id"].(string); ok {
		m.ToolCallID = toolCallID
	}

	if tcBlob, exists := raw["tool_calls"]; exists && tcBlob != nil {
		if tcArr, ok := tcBlob.([]any); ok {
			var toolCalls []ToolCall
			for _, item := range tcArr {
				if itemMap, ok := item.(map[string]any); ok {
					var tc ToolCall
					tc.ID, _ = itemMap["id"].(string)
					tc.Type, _ = itemMap["type"].(string)
					if tc.Type == "" {
						tc.Type = "function"
					}
					if fnMap, ok := itemMap["function"].(map[string]any); ok {
						tc.Function.Name, _ = fnMap["name"].(string)
						if argsStr, ok := fnMap["arguments"].(string); ok {
							tc.Function.Arguments = argsStr
						} else if argsObj, ok := fnMap["arguments"]; ok {
							b, _ := json.Marshal(argsObj)
							tc.Function.Arguments = string(b)
						}
					}
					toolCalls = append(toolCalls, tc)
				}
			}
			m.ToolCalls = toolCalls
		}
	}

	content, exists := raw["content"]
	if !exists || content == nil {
		m.Parts = nil
		return nil
	}

	parts, err := extractContentParts(content)
	if err != nil {
		return err
	}
	m.Parts = parts
	return nil
}

func (m *Message) Export() (map[string]any, error) {
	if m.data == nil {
		m.data = make(map[string]any)
	}
	m.data["role"] = string(m.Role)
	if m.Name != "" {
		m.data["name"] = m.Name
	}
	if m.ToolCallID != "" {
		m.data["tool_call_id"] = m.ToolCallID
	}
	if len(m.ToolCalls) > 0 {
		var rawCalls []any
		for _, tc := range m.ToolCalls {
			tcType := tc.Type
			if tcType == "" {
				tcType = "function"
			}
			rawCalls = append(rawCalls, map[string]any{
				"id":   tc.ID,
				"type": tcType,
				"function": map[string]any{
					"name":      tc.Function.Name,
					"arguments": tc.Function.Arguments,
				},
			})
		}
		m.data["tool_calls"] = rawCalls
	}

	if len(m.Parts) == 0 {
		if len(m.ToolCalls) > 0 {
			m.data["content"] = nil
		} else {
			m.data["content"] = ""
		}
	} else {
		var rawParts []any
		for i := range m.Parts {
			exportedPart, err := m.Parts[i].Export()
			if err != nil {
				return nil, err
			}
			rawParts = append(rawParts, exportedPart)
		}
		m.data["content"] = rawParts
	}
	return m.data, nil
}

func (m *Message) UnmarshalJSON(b []byte) error {
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	return m.Import(raw)
}

func (m *Message) MarshalJSON() ([]byte, error) {
	raw, err := m.Export()
	if err != nil {
		return nil, err
	}
	return json.Marshal(raw)
}

type Parameters struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

type Conversation struct {
	Model      string           `json:"model"`
	System     []string         `json:"system,omitempty"` // Unified system instructions (text-only)
	Messages   []Message        `json:"messages"`         // Non-system interactive turns
	Tools      []ToolDefinition `json:"tools,omitempty"`
	ToolChoice any              `json:"tool_choice,omitempty"`
	Parameters Parameters       `json:"parameters,omitempty"`
	data       map[string]any
}

func (c *Conversation) Import(blob any) error {
	raw, ok := blob.(map[string]any)
	if !ok {
		return fmt.Errorf("conversation is not an object")
	}
	c.data = raw
	c.Model, _ = raw["model"].(string)

	c.System = nil
	if sysPrompt, ok := raw["system_prompt"].(string); ok && sysPrompt != "" {
		c.System = append(c.System, sysPrompt)
	}

	if msgArr, ok := raw["messages"].([]any); ok {
		var messages []Message
		for _, msgItem := range msgArr {
			if msgMap, ok := msgItem.(map[string]any); ok {
				roleStr, _ := msgMap["role"].(string)
				if Role(roleStr) == RoleSystem {
					content, exists := msgMap["content"]
					if exists && content != nil {
						texts, err := extractTextContent(content)
						if err != nil {
							return err
						}
						c.System = append(c.System, texts...)
					}
					continue
				}
			}

			m := Message{}
			if err := m.Import(msgItem); err != nil {
				return err
			}
			messages = append(messages, m)
		}
		c.Messages = messages
	}

	if toolsBlob, exists := raw["tools"]; exists && toolsBlob != nil {
		if tArr, ok := toolsBlob.([]any); ok {
			var tools []ToolDefinition
			for _, item := range tArr {
				if itemMap, ok := item.(map[string]any); ok {
					var td ToolDefinition
					td.Type, _ = itemMap["type"].(string)
					if fnMap, ok := itemMap["function"].(map[string]any); ok {
						td.Function = fnMap
					}
					tools = append(tools, td)
				}
			}
			c.Tools = tools
		}
	}
	if tc, exists := raw["tool_choice"]; exists {
		c.ToolChoice = tc
	}

	c.Parameters = extractParameters(raw)
	return nil
}

func (c *Conversation) Export() (map[string]any, error) {
	if c.data == nil {
		c.data = make(map[string]any)
	}
	c.data["model"] = c.Model

	// Remove legacy system_prompt key to avoid duplicate mapping in output JSON
	delete(c.data, "system_prompt")

	var rawMsgs []any

	if len(c.System) > 0 {
		rawMsgs = append(rawMsgs, map[string]any{
			"role":    string(RoleSystem),
			"content": strings.Join(c.System, "\n\n"),
		})
	}

	for i := range c.Messages {
		exportedMsg, err := c.Messages[i].Export()
		if err != nil {
			return nil, err
		}
		rawMsgs = append(rawMsgs, exportedMsg)
	}
	c.data["messages"] = rawMsgs

	if len(c.Tools) > 0 {
		var rawTools []any
		for _, td := range c.Tools {
			rawTools = append(rawTools, map[string]any{
				"type":     td.Type,
				"function": td.Function,
			})
		}
		c.data["tools"] = rawTools
	}
	if c.ToolChoice != nil {
		c.data["tool_choice"] = c.ToolChoice
	}

	syncParameters(c.data, c.Parameters)
	return c.data, nil
}

func (c *Conversation) UnmarshalJSON(b []byte) error {
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	return c.Import(raw)
}

func (c *Conversation) MarshalJSON() ([]byte, error) {
	raw, err := c.Export()
	if err != nil {
		return nil, err
	}
	return json.Marshal(raw)
}

// SetSystemPrompt replaces the system instruction array with a single text block.
func (c *Conversation) SetSystemPrompt(text string) {
	c.System = []string{text}
}

// AppendSystemPrompt appends a new text block to the system instruction array.
func (c *Conversation) AppendSystemPrompt(text string) {
	c.System = append(c.System, text)
}

// CombinedSystemPrompt concatenates all system instruction blocks into a single string (joined by \n\n).
func (c *Conversation) CombinedSystemPrompt() string {
	return strings.Join(c.System, "\n\n")
}

// AddMessage appends a new message containing a single text part to the conversation.
func (c *Conversation) AddMessage(role Role, text string) *Message {
	msg := Message{
		Role: role,
		Parts: []Part{
			{Type: "text", Content: text},
		},
	}
	c.Messages = append(c.Messages, msg)
	return &c.Messages[len(c.Messages)-1]
}

// AddUserMessage is a helper to append a user message.
func (c *Conversation) AddUserMessage(text string) *Message {
	return c.AddMessage(RoleUser, text)
}

// AddAssistantMessage is a helper to append an assistant message.
func (c *Conversation) AddAssistantMessage(text string) *Message {
	return c.AddMessage(RoleAssistant, text)
}

// AddPart appends a new content part to the message.
func (m *Message) AddPart(pType string, content any) *Part {
	p := Part{
		Type:    pType,
		Content: content,
	}
	m.Parts = append(m.Parts, p)
	return &m.Parts[len(m.Parts)-1]
}

// Helpers
func extractMeta(blob any) map[string]any {
	if m, ok := blob.(map[string]any); ok {
		return m
	}
	return make(map[string]any)
}

func extractContentParts(content any) ([]Part, error) {
	if content == nil {
		return nil, nil
	}

	if s, ok := content.(string); ok {
		p := Part{}
		if err := p.Import(s); err != nil {
			return nil, err
		}
		return []Part{p}, nil
	}

	if arr, ok := content.([]any); ok {
		var parts []Part
		for _, item := range arr {
			p := Part{}
			if err := p.Import(item); err != nil {
				return nil, err
			}
			parts = append(parts, p)
		}
		return parts, nil
	}

	return nil, fmt.Errorf("unsupported content type: %T", content)
}

func extractTextContent(content any) ([]string, error) {
	if content == nil {
		return nil, nil
	}

	if s, ok := content.(string); ok {
		return []string{s}, nil
	}

	if arr, ok := content.([]any); ok {
		var result []string
		for _, item := range arr {
			if s, ok := item.(string); ok {
				result = append(result, s)
			} else if m, ok := item.(map[string]any); ok {
				t, _ := m["type"].(string)
				if t == "text" {
					if txt, ok := m["content"].(string); ok {
						result = append(result, txt)
					} else if txt, ok := m["text"].(string); ok {
						result = append(result, txt)
					}
				}
			}
		}
		return result, nil
	}

	return nil, fmt.Errorf("unsupported system content type: %T", content)
}

func extractParameters(m map[string]any) Parameters {
	var params Parameters
	if t, ok := m["temperature"].(float64); ok {
		params.Temperature = &t
	}
	if p, ok := m["top_p"].(float64); ok {
		params.TopP = &p
	}
	if mt, ok := m["max_tokens"].(float64); ok {
		val := int(mt)
		params.MaxTokens = &val
	}
	if stopBlob, ok := m["stop"]; ok {
		if s, ok := stopBlob.(string); ok {
			params.Stop = []string{s}
		} else if sArr, ok := stopBlob.([]any); ok {
			var stops []string
			for _, item := range sArr {
				if sItem, ok := item.(string); ok {
					stops = append(stops, sItem)
				}
			}
			params.Stop = stops
		}
	}
	return params
}

func syncParameters(m map[string]any, params Parameters) {
	if params.Temperature != nil {
		m["temperature"] = *params.Temperature
	} else {
		delete(m, "temperature")
	}
	if params.TopP != nil {
		m["top_p"] = *params.TopP
	} else {
		delete(m, "top_p")
	}
	if params.MaxTokens != nil {
		m["max_tokens"] = *params.MaxTokens
	} else {
		delete(m, "max_tokens")
	}
	if len(params.Stop) > 0 {
		m["stop"] = params.Stop
	} else {
		delete(m, "stop")
	}
}
