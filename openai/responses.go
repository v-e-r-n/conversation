package openai

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/v-e-r-n/conversation/core"
)

// OpenAIResponseRequest represents a request to the OpenAI Responses API (/v1/responses).
type OpenAIResponseRequest struct {
	Model              string         `json:"model"`
	Input              any            `json:"input"`
	Instructions       string         `json:"instructions,omitempty"`
	PreviousResponseID string         `json:"previous_response_id,omitempty"`
	Tools              []ResponseTool `json:"tools,omitempty"`
	ToolChoice         any            `json:"tool_choice,omitempty"`
	Temperature        *float64       `json:"temperature,omitempty"`
	TopP               *float64       `json:"top_p,omitempty"`
	MaxOutputTokens    *int           `json:"max_output_tokens,omitempty"`
	Stream             *bool          `json:"stream,omitempty"`
	Store              *bool          `json:"store,omitempty"`
	Metadata           map[string]any `json:"metadata,omitempty"`
}

// ResponseTool represents a tool declaration in the OpenAI Responses API.
// In /v1/responses, tool properties are flat at the root of the tool object.
type ResponseTool struct {
	Type        string `json:"type"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

func (t *ResponseTool) UnmarshalJSON(b []byte) error {
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	t.Type, _ = raw["type"].(string)
	if t.Type == "" {
		t.Type = "function"
	}
	if fn, ok := raw["function"].(map[string]any); ok {
		t.Name, _ = fn["name"].(string)
		t.Description, _ = fn["description"].(string)
		t.Parameters = fn["parameters"]
		if s, ok := fn["strict"].(bool); ok {
			t.Strict = &s
		}
	} else {
		t.Name, _ = raw["name"].(string)
		t.Description, _ = raw["description"].(string)
		t.Parameters = raw["parameters"]
		if s, ok := raw["strict"].(bool); ok {
			t.Strict = &s
		}
	}
	return nil
}

func (t ResponseTool) MarshalJSON() ([]byte, error) {
	m := map[string]any{
		"type": t.Type,
	}
	if t.Name != "" {
		m["name"] = t.Name
	}
	if t.Description != "" {
		m["description"] = t.Description
	}
	if t.Parameters != nil {
		m["parameters"] = t.Parameters
	}
	if t.Strict != nil {
		m["strict"] = *t.Strict
	}
	return json.Marshal(m)
}

// ResponseItem represents a single item within the Responses API input array.
type ResponseItem struct {
	Type       string          `json:"type,omitempty"`
	Role       string          `json:"role,omitempty"`
	Content    any             `json:"content,omitempty"`
	CallID     string          `json:"call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
	Arguments  string          `json:"arguments,omitempty"`
	Output     any             `json:"output,omitempty"`
	ToolCalls  []core.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Meta       map[string]any  `json:"meta,omitempty"`
}

// ResponsePart represents a structured multi-modal content part within a Responses API item.
type ResponsePart struct {
	Type       string         `json:"type"`
	Text       string         `json:"text,omitempty"`
	ImageURL   any            `json:"image_url,omitempty"`
	InputAudio any            `json:"input_audio,omitempty"`
	Meta       map[string]any `json:"meta,omitempty"`
}

// FromResponseRequest translates an OpenAIResponseRequest to Vern's canonical core.Conversation.
func FromResponseRequest(req *OpenAIResponseRequest) (*core.Conversation, error) {
	if req == nil {
		return nil, fmt.Errorf("response request cannot be nil")
	}

	c := core.Conversation{
		Model: req.Model,
	}

	// 1. Map top-level instructions into c.System
	if req.Instructions != "" {
		c.System = append(c.System, req.Instructions)
	}

	// 2. Parse input items
	items, err := parseResponseItems(req.Input)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response input: %w", err)
	}

	for _, item := range items {
		callID := item.CallID
		if callID == "" {
			callID = item.ToolCallID
		}

		// Handle standalone function call items
		if item.Type == "function_call" {
			args := item.Arguments
			if args == "" {
				args = "{}"
			}
			tc := core.ToolCall{
				ID:   callID,
				Type: "function",
				Function: core.ToolFunction{
					Name:      item.Name,
					Arguments: args,
				},
			}
			// Merge into immediately preceding assistant turn if present to preserve turn unity
			if len(c.Messages) > 0 && c.Messages[len(c.Messages)-1].Role == core.RoleAssistant {
				lastIdx := len(c.Messages) - 1
				c.Messages[lastIdx].ToolCalls = append(c.Messages[lastIdx].ToolCalls, tc)
			} else {
				c.Messages = append(c.Messages, core.Message{
					Role:      core.RoleAssistant,
					ToolCalls: []core.ToolCall{tc},
				})
			}
			continue
		}

		// Handle standalone function call output items
		if item.Type == "function_call_output" {
			outputStr := ""
			if item.Output != nil {
				outputStr = formatToolOutput(item.Output)
			}
			m := core.Message{
				Role:       core.RoleTool,
				ToolCallID: callID,
				Parts: []core.Part{
					{
						Type:    "text",
						Content: outputStr,
					},
				},
			}
			c.Messages = append(c.Messages, m)
			continue
		}

		// Handle message items
		role := item.Role
		if role == "" && (item.Type == "message" || item.Type == "") {
			role = "user"
		}

		// Modern OpenAI models use developer role alongside system
		if role == "system" || role == "developer" {
			texts, err := extractTextContent(item.Content)
			if err != nil {
				return nil, err
			}
			c.System = append(c.System, texts...)
			continue
		}

		var m core.Message
		m.Role = core.Role(role)
		m.Name = item.Name
		m.ToolCalls = item.ToolCalls
		m.ToolCallID = item.ToolCallID
		if m.ToolCallID == "" && callID != "" && role == "tool" {
			m.ToolCallID = callID
		}

		parts, err := extractResponseContentParts(item.Content, item.Meta)
		if err != nil {
			return nil, err
		}
		m.Parts = parts

		c.Messages = append(c.Messages, m)
	}

	// 3. Preserve previous_response_id, stream, and metadata on core.Conversation
	if req.PreviousResponseID != "" {
		c.SetMeta("previous_response_id", req.PreviousResponseID)
	}
	if req.Stream != nil {
		c.SetMeta("stream", *req.Stream)
	}
	if req.Store != nil {
		c.SetMeta("store", *req.Store)
	}
	if req.Metadata != nil {
		c.SetMeta("metadata", req.Metadata)
	}

	// 4. Tools & Parameters
	for _, rt := range req.Tools {
		td := core.ToolDefinition{
			Type: rt.Type,
		}
		if rt.Name != "" || rt.Description != "" || rt.Parameters != nil || rt.Strict != nil {
			fnMap := map[string]any{}
			if rt.Name != "" {
				fnMap["name"] = rt.Name
			}
			if rt.Description != "" {
				fnMap["description"] = rt.Description
			}
			if rt.Parameters != nil {
				fnMap["parameters"] = rt.Parameters
			}
			if rt.Strict != nil {
				fnMap["strict"] = *rt.Strict
			}
			td.Function = fnMap
		}
		c.Tools = append(c.Tools, td)
	}
	c.ToolChoice = req.ToolChoice

	c.Parameters.Temperature = req.Temperature
	c.Parameters.TopP = req.TopP
	c.Parameters.MaxTokens = req.MaxOutputTokens

	return &c, nil
}

// ToResponseRequest translates a core.Conversation to an OpenAIResponseRequest.
func ToResponseRequest(c *core.Conversation, opts *core.ExportOptions) (*OpenAIResponseRequest, error) {
	if c == nil {
		return nil, fmt.Errorf("conversation cannot be nil")
	}

	model := c.Model
	var renderers map[string]core.Renderer
	if opts != nil {
		if opts.Model != "" {
			model = opts.Model
		}
		renderers = opts.Renderers
	}

	req := &OpenAIResponseRequest{
		Model:      model,
		ToolChoice: c.ToolChoice,
	}

	for _, td := range c.Tools {
		rt := ResponseTool{
			Type: td.Type,
		}
		if td.Function != nil {
			rt.Name, _ = td.Function["name"].(string)
			rt.Description, _ = td.Function["description"].(string)
			rt.Parameters = td.Function["parameters"]
			if s, ok := td.Function["strict"].(bool); ok {
				rt.Strict = &s
			}
		}
		req.Tools = append(req.Tools, rt)
	}

	// 1. Format directives into instructions
	if len(c.System) > 0 {
		req.Instructions = strings.Join(c.System, "\n\n")
	}

	// 2. Map conversation messages to response input items
	var items []ResponseItem

	for _, msg := range c.Messages {
		// Tool call emitted by assistant
		if msg.Role == core.RoleAssistant && len(msg.ToolCalls) > 0 {
			// Chronology fix: assistant text message emitted FIRST
			if len(msg.Parts) > 0 {
				renderedParts, err := renderResponseParts(msg.Parts, renderers)
				if err != nil {
					return nil, err
				}
				items = append(items, ResponseItem{
					Type:    "message",
					Role:    "assistant",
					Content: formatItemContent(renderedParts, msg.Parts),
				})
			}
			// Function calls emitted SECOND
			for _, tc := range msg.ToolCalls {
				args := tc.Function.Arguments
				if args == "" {
					args = "{}"
				}
				items = append(items, ResponseItem{
					Type:      "function_call",
					CallID:    tc.ID,
					Name:      tc.Function.Name,
					Arguments: args,
				})
			}
			continue
		}

		// Tool response
		if msg.Role == core.RoleTool {
			outputStr := ""
			if len(msg.Parts) > 0 {
				outputStr = formatToolOutput(msg.Parts[0].Content)
			}
			items = append(items, ResponseItem{
				Type:   "function_call_output",
				CallID: msg.ToolCallID,
				Output: outputStr,
			})
			continue
		}

		// Standard user or assistant message
		renderedParts, err := renderResponseParts(msg.Parts, renderers)
		if err != nil {
			return nil, err
		}

		item := ResponseItem{
			Type:    "message",
			Role:    string(msg.Role),
			Content: formatItemContent(renderedParts, msg.Parts),
			Name:    msg.Name,
		}
		items = append(items, item)
	}

	req.Input = items

	// 3. Extract previous_response_id, stream, and metadata
	if prevID, ok := c.GetMeta("previous_response_id").(string); ok && prevID != "" {
		req.PreviousResponseID = prevID
	}
	if req.PreviousResponseID == "" {
		for _, msg := range c.Messages {
			for _, part := range msg.Parts {
				if val, ok := part.Meta["previous_response_id"].(string); ok && val != "" {
					req.PreviousResponseID = val
					break
				}
			}
			if req.PreviousResponseID != "" {
				break
			}
		}
	}

	if opts != nil && opts.Stream != nil {
		req.Stream = opts.Stream
	} else if streamVal, ok := c.GetMeta("stream").(bool); ok {
		req.Stream = &streamVal
	}

	if opts != nil && opts.Store != nil {
		req.Store = opts.Store
	} else if storeVal, ok := c.GetMeta("store").(bool); ok {
		req.Store = &storeVal
	}

	if metaVal, ok := c.GetMeta("metadata").(map[string]any); ok && metaVal != nil {
		req.Metadata = metaVal
	}

	// 4. Parameters
	req.Temperature = c.Parameters.Temperature
	req.TopP = c.Parameters.TopP
	req.MaxOutputTokens = c.Parameters.MaxTokens

	return req, nil
}

func formatToolOutput(content any) string {
	if content == nil {
		return ""
	}
	switch v := content.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(b)
	}
}

func renderResponseParts(parts []core.Part, renderers map[string]core.Renderer) ([]map[string]any, error) {
	var renderedParts []map[string]any
	for _, part := range parts {
		rendered, err := renderPart(part.Content, part.Meta, part.Type, renderers)
		if err != nil {
			return nil, err
		}

		switch part.Type {
		case "text":
			renderedParts = append(renderedParts, map[string]any{
				"type": "input_text",
				"text": rendered,
			})
		case "image":
			renderedParts = append(renderedParts, map[string]any{
				"type":      "input_image",
				"image_url": rendered,
			})
		case "audio":
			renderedParts = append(renderedParts, map[string]any{
				"type":        "input_audio",
				"input_audio": rendered,
			})
		default:
			renderedParts = append(renderedParts, map[string]any{
				"type":    part.Type,
				part.Type: rendered,
			})
		}
	}
	return renderedParts, nil
}

func formatItemContent(renderedParts []map[string]any, origParts []core.Part) any {
	if len(renderedParts) == 0 {
		return ""
	}
	// If only 1 text part without metadata, simplify to plain string
	if len(renderedParts) == 1 && origParts[0].Type == "text" && len(origParts[0].Meta) == 0 {
		return renderedParts[0]["text"]
	}
	return renderedParts
}

func parseResponseItems(input any) ([]ResponseItem, error) {
	if input == nil {
		return nil, nil
	}

	// Case 1: Plain string input
	if s, ok := input.(string); ok {
		return []ResponseItem{
			{
				Type:    "message",
				Role:    "user",
				Content: s,
			},
		}, nil
	}

	rv := reflect.ValueOf(input)
	if rv.Kind() == reflect.Slice {
		var items []ResponseItem
		for i := 0; i < rv.Len(); i++ {
			itemBlob := rv.Index(i).Interface()
			switch v := itemBlob.(type) {
			case string:
				items = append(items, ResponseItem{
					Type:    "message",
					Role:    "user",
					Content: v,
				})
			case ResponseItem:
				items = append(items, v)
			case *ResponseItem:
				if v != nil {
					items = append(items, *v)
				}
			default:
				var ri ResponseItem
				b, err := json.Marshal(v)
				if err != nil {
					return nil, err
				}
				if err := json.Unmarshal(b, &ri); err != nil {
					return nil, err
				}
				items = append(items, ri)
			}
		}
		return items, nil
	}

	if rv.Kind() == reflect.Map || rv.Kind() == reflect.Struct {
		var ri ResponseItem
		b, err := json.Marshal(input)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(b, &ri); err != nil {
			return nil, err
		}
		return []ResponseItem{ri}, nil
	}

	return nil, fmt.Errorf("unsupported input type: %T", input)
}

func extractResponseContentParts(content any, itemMeta map[string]any) ([]core.Part, error) {
	if content == nil {
		return nil, nil
	}

	// Simple string content
	if s, ok := content.(string); ok {
		p := core.Part{
			Type:    "text",
			Content: s,
			Meta:    itemMeta,
		}
		return []core.Part{p}, nil
	}

	rv := reflect.ValueOf(content)
	if rv.Kind() == reflect.Slice {
		var parts []core.Part
		for i := 0; i < rv.Len(); i++ {
			item := rv.Index(i).Interface()
			if s, ok := item.(string); ok {
				parts = append(parts, core.Part{Type: "text", Content: s, Meta: itemMeta})
				continue
			}

			var m map[string]any
			if mapVal, ok := item.(map[string]any); ok {
				m = mapVal
			} else {
				b, err := json.Marshal(item)
				if err != nil {
					return nil, err
				}
				if err := json.Unmarshal(b, &m); err != nil {
					return nil, err
				}
			}

			p := core.Part{}
			pType, _ := m["type"].(string)

			switch pType {
			case "input_text", "text", "output_text":
				p.Type = "text"
				if txt, ok := m["text"].(string); ok {
					p.Content = txt
				} else if txt, ok := m["content"].(string); ok {
					p.Content = txt
				}
			case "input_image", "image_url":
				p.Type = "image"
				if img, exists := m["image_url"]; exists {
					p.Content = img
				} else if url, exists := m["url"]; exists {
					p.Content = url
				}
			case "input_audio", "audio":
				p.Type = "audio"
				if aud, exists := m["input_audio"]; exists {
					p.Content = aud
				} else if aud, exists := m["audio"]; exists {
					p.Content = aud
				} else if data, exists := m["data"]; exists {
					p.Content = data
				}
			default:
				if pType != "" {
					p.Type = pType
					p.Content = m[pType]
				} else {
					p.Type = "text"
					p.Content = m["text"]
				}
			}

			partMeta := extractMeta(m["meta"])
			if len(partMeta) == 0 && len(itemMeta) > 0 {
				partMeta = itemMeta
			}
			p.Meta = partMeta
			parts = append(parts, p)
		}
		return parts, nil
	}

	return nil, fmt.Errorf("unsupported response content type: %T", content)
}
