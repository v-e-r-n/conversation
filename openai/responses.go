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
	Model              string                `json:"model"`
	Input              any                   `json:"input"`
	Instructions       string                `json:"instructions,omitempty"`
	PreviousResponseID string                `json:"previous_response_id,omitempty"`
	Tools              []core.ToolDefinition `json:"tools,omitempty"`
	ToolChoice         any                   `json:"tool_choice,omitempty"`
	Temperature        *float64              `json:"temperature,omitempty"`
	TopP               *float64              `json:"top_p,omitempty"`
	MaxOutputTokens    *int                  `json:"max_output_tokens,omitempty"`
	Stream             *bool                 `json:"stream,omitempty"`
	Metadata           map[string]any        `json:"metadata,omitempty"`
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
		// Handle standalone function call items
		if item.Type == "function_call" {
			m := core.Message{
				Role: core.RoleAssistant,
				ToolCalls: []core.ToolCall{
					{
						ID:   item.CallID,
						Type: "function",
						Function: core.ToolFunction{
							Name:      item.Name,
							Arguments: item.Arguments,
						},
					},
				},
			}
			c.Messages = append(c.Messages, m)
			continue
		}

		// Handle standalone function call output items
		if item.Type == "function_call_output" {
			outputStr := ""
			if item.Output != nil {
				if s, ok := item.Output.(string); ok {
					outputStr = s
				} else {
					b, _ := json.Marshal(item.Output)
					outputStr = string(b)
				}
			}
			m := core.Message{
				Role:       core.RoleTool,
				ToolCallID: item.CallID,
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

		if role == "system" {
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
		if m.ToolCallID == "" && item.CallID != "" && role == "tool" {
			m.ToolCallID = item.CallID
		}

		parts, err := extractResponseContentParts(item.Content, item.Meta)
		if err != nil {
			return nil, err
		}
		m.Parts = parts

		c.Messages = append(c.Messages, m)
	}

	// 3. Preserve previous_response_id in metadata
	if req.PreviousResponseID != "" {
		if len(c.Messages) > 0 && len(c.Messages[0].Parts) > 0 {
			c.Messages[0].Parts[0].SetMeta("previous_response_id", req.PreviousResponseID)
		} else if len(c.Messages) > 0 {
			c.Messages[0].Parts = append(c.Messages[0].Parts, core.Part{
				Type:    "text",
				Content: "",
				Meta:    map[string]any{"previous_response_id": req.PreviousResponseID},
			})
		}
	}

	// 4. Tools & Parameters
	c.Tools = req.Tools
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
		Tools:      c.Tools,
		ToolChoice: c.ToolChoice,
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
			for _, tc := range msg.ToolCalls {
				items = append(items, ResponseItem{
					Type:      "function_call",
					CallID:    tc.ID,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				})
			}
			// If message also contains text content, emit it as an assistant message
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
			continue
		}

		// Tool response
		if msg.Role == core.RoleTool {
			outputStr := ""
			if len(msg.Parts) > 0 {
				outputStr = fmt.Sprintf("%v", msg.Parts[0].Content)
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

	// 3. Extract previous_response_id from message metadata if present
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

	// 4. Parameters
	req.Temperature = c.Parameters.Temperature
	req.TopP = c.Parameters.TopP
	req.MaxOutputTokens = c.Parameters.MaxTokens

	return req, nil
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
			m := map[string]any{
				"type": "input_text",
				"text": rendered,
			}
			if len(part.Meta) > 0 {
				m["meta"] = part.Meta
			}
			renderedParts = append(renderedParts, m)
		case "image":
			m := map[string]any{
				"type":      "input_image",
				"image_url": rendered,
			}
			if len(part.Meta) > 0 {
				m["meta"] = part.Meta
			}
			renderedParts = append(renderedParts, m)
		case "audio":
			m := map[string]any{
				"type":        "input_audio",
				"input_audio": rendered,
			}
			if len(part.Meta) > 0 {
				m["meta"] = part.Meta
			}
			renderedParts = append(renderedParts, m)
		default:
			m := map[string]any{
				"type":    part.Type,
				part.Type: rendered,
			}
			if len(part.Meta) > 0 {
				m["meta"] = part.Meta
			}
			renderedParts = append(renderedParts, m)
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
