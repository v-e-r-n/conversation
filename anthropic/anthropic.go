package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/v-e-r-n/conversation/core"
)

type AnthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // can be string or []any parts
}

type AnthropicRequest struct {
	Model         string             `json:"model"`
	Messages      []AnthropicMessage `json:"messages"`
	System        string             `json:"system,omitempty"`
	Tools         []any              `json:"tools,omitempty"`
	ToolChoice    any                `json:"tool_choice,omitempty"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
}

// ToRequest translates core.Conversation to AnthropicRequest.
func ToRequest(c *core.Conversation, opts *core.ExportOptions) (*AnthropicRequest, error) {
	if opts == nil {
		return nil, fmt.Errorf("export options are required and must be non-nil")
	}

	model := c.Model
	if opts.Model != "" {
		model = opts.Model
	}

	maxTokens := opts.DefaultMaxTokens
	if c.Parameters.MaxTokens != nil {
		maxTokens = *c.Parameters.MaxTokens
	}

	var anthropicMsgs []AnthropicMessage

	// 1. Process & collapse all system prompts from c.System
	var systemStr string
	if len(c.System) > 0 {
		systemStr = strings.Join(c.System, "\n\n")
	}

	// 2. Map standard messages
	for _, msg := range c.Messages {
		var antMsg AnthropicMessage
		role := string(msg.Role)
		if msg.Role == core.RoleTool {
			role = "user"
		}
		antMsg.Role = role

		var renderedParts []any
		var hasRichContent bool

		// If role is tool, map to tool_result content part
		if msg.Role == core.RoleTool {
			contentStr := ""
			if len(msg.Parts) > 0 {
				if s, ok := msg.Parts[0].Content.(string); ok {
					contentStr = s
				} else {
					contentStr = fmt.Sprintf("%v", msg.Parts[0].Content)
				}
			}
			renderedParts = append(renderedParts, map[string]any{
				"type":        "tool_result",
				"tool_use_id": msg.ToolCallID,
				"content":     contentStr,
			})
			hasRichContent = true
		} else {
			for _, part := range msg.Parts {
				rendered, err := renderPart(part.Content, part.Meta, part.Type, opts.Renderers)
				if err != nil {
					return nil, err
				}

				switch part.Type {
				case "text":
					renderedParts = append(renderedParts, map[string]any{
						"type": "text",
						"text": rendered,
					})
				case "image":
					renderedParts = append(renderedParts, map[string]any{
						"type":   "image",
						"source": rendered,
					})
					hasRichContent = true
				default:
					renderedParts = append(renderedParts, map[string]any{
						"type":    part.Type,
						part.Type: rendered,
					})
					hasRichContent = true
				}
			}

			// If assistant message has ToolCalls, map to tool_use blocks
			if msg.Role == core.RoleAssistant && len(msg.ToolCalls) > 0 {
				for _, tc := range msg.ToolCalls {
					var inputMap any
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &inputMap); err != nil {
						inputMap = map[string]any{"raw": tc.Function.Arguments}
					}
					renderedParts = append(renderedParts, map[string]any{
						"type":  "tool_use",
						"id":    tc.ID,
						"name":  tc.Function.Name,
						"input": inputMap,
					})
					hasRichContent = true
				}
			}
		}

		if !hasRichContent && len(renderedParts) == 1 {
			if textPart, ok := renderedParts[0].(map[string]any); ok {
				antMsg.Content = textPart["text"]
			}
		} else {
			antMsg.Content = renderedParts
		}

		anthropicMsgs = append(anthropicMsgs, antMsg)
	}

	var antTools []any
	for _, td := range c.Tools {
		fn := td.Function
		toolMap := map[string]any{
			"name":         fn["name"],
			"description":  fn["description"],
			"input_schema": fn["parameters"],
		}
		antTools = append(antTools, toolMap)
	}

	return &AnthropicRequest{
		Model:         model,
		Messages:      anthropicMsgs,
		System:        systemStr,
		Tools:         antTools,
		ToolChoice:    c.ToolChoice,
		MaxTokens:     maxTokens,
		Temperature:   c.Parameters.Temperature,
		TopP:          c.Parameters.TopP,
		StopSequences: c.Parameters.Stop,
	}, nil
}

// FromRequest translates an AnthropicRequest back to core.Conversation.
func FromRequest(req *AnthropicRequest) (*core.Conversation, error) {
	if req == nil {
		return nil, fmt.Errorf("request is nil")
	}

	var c core.Conversation
	c.Model = req.Model
	if req.System != "" {
		c.System = []string{req.System}
	}

	var coreMsgs []core.Message
	for _, msg := range req.Messages {
		var m core.Message
		m.Role = core.Role(msg.Role)

		// Check if content contains tool_result or tool_use blocks
		if arr, ok := msg.Content.([]any); ok {
			var isToolResult bool
			for _, item := range arr {
				if itemMap, ok := item.(map[string]any); ok {
					if itemMap["type"] == "tool_result" {
						isToolResult = true
						m.Role = core.RoleTool
						if id, ok := itemMap["tool_use_id"].(string); ok {
							m.ToolCallID = id
						}
						if cnt, ok := itemMap["content"].(string); ok {
							m.Parts = []core.Part{{Type: "text", Content: cnt}}
						}
						break
					} else if itemMap["type"] == "tool_use" {
						var tc core.ToolCall
						tc.ID, _ = itemMap["id"].(string)
						tc.Type = "function"
						tc.Function.Name, _ = itemMap["name"].(string)
						if inObj, exists := itemMap["input"]; exists {
							b, _ := json.Marshal(inObj)
							tc.Function.Arguments = string(b)
						}
						m.ToolCalls = append(m.ToolCalls, tc)
					}
				}
			}
			if isToolResult {
				coreMsgs = append(coreMsgs, m)
				continue
			}
		}

		parts, err := extractCoreParts(msg.Content)
		if err != nil {
			return nil, err
		}
		m.Parts = parts
		coreMsgs = append(coreMsgs, m)
	}
	c.Messages = coreMsgs

	c.Parameters.Temperature = req.Temperature
	c.Parameters.TopP = req.TopP
	c.Parameters.MaxTokens = &req.MaxTokens
	c.Parameters.Stop = req.StopSequences

	return &c, nil
}

// Helpers
func renderPart(content any, meta map[string]any, pType string, renderers map[string]core.Renderer) (any, error) {
	if renderer, ok := renderers[pType]; ok && renderer != nil {
		return renderer(content, meta)
	}
	return core.DefaultRenderer(content, meta)
}

func extractCoreParts(content any) ([]core.Part, error) {
	if s, ok := content.(string); ok {
		p := core.Part{
			Type:    "text",
			Content: s,
		}
		return []core.Part{p}, nil
	}

	if arr, ok := content.([]any); ok {
		var parts []core.Part
		for _, item := range arr {
			p := core.Part{}
			if m, ok := item.(map[string]any); ok {
				p.Type, _ = m["type"].(string)
				if p.Type == "text" {
					p.Content = m["text"]
				} else if p.Type == "image" {
					p.Content = m["source"]
				} else {
					p.Content = m[p.Type]
				}
				p.Meta = extractMeta(m["meta"])
			} else {
				p.Type = "text"
				p.Content = fmt.Sprintf("%v", item)
			}
			parts = append(parts, p)
		}
		return parts, nil
	}

	return nil, fmt.Errorf("unsupported anthropic content type: %T", content)
}

func extractMeta(blob any) map[string]any {
	if m, ok := blob.(map[string]any); ok {
		return m
	}
	return make(map[string]any)
}
