package openai

import (
	"github.com/v-e-r-n/conversation/core"
	"fmt"
	"strings"
)

type OpenAIMessage struct {
	Role       string          `json:"role"`
	Content    any             `json:"content"` // can be string or []any content parts or nil
	Name       string          `json:"name,omitempty"`
	ToolCalls  []core.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Meta       map[string]any  `json:"meta,omitempty"`
}

type OpenAIRequest struct {
	Model       string                `json:"model"`
	Messages    []OpenAIMessage       `json:"messages"`
	Tools       []core.ToolDefinition `json:"tools,omitempty"`
	ToolChoice  any                   `json:"tool_choice,omitempty"`
	Temperature *float64              `json:"temperature,omitempty"`
	TopP        *float64              `json:"top_p,omitempty"`
	MaxTokens   *int                  `json:"max_tokens,omitempty"`
	Stop        []string              `json:"stop,omitempty"`
}

// ToRequest translates core.Conversation to OpenAIRequest.
func ToRequest(c *core.Conversation, opts *core.ExportOptions) (*OpenAIRequest, error) {
	if opts == nil {
		return nil, fmt.Errorf("export options are required and must be non-nil")
	}

	model := c.Model
	if opts.Model != "" {
		model = opts.Model
	}

	var openAIMsgs []OpenAIMessage

	// 1. Process & collapse all system prompts from c.System
	if len(c.System) > 0 {
		openAIMsgs = append(openAIMsgs, OpenAIMessage{
			Role:    "system",
			Content: strings.Join(c.System, "\n\n"),
		})
	}

	// 2. Map standard messages
	for _, msg := range c.Messages {
		var openAIMsg OpenAIMessage
		openAIMsg.Role = string(msg.Role)
		openAIMsg.Name = msg.Name
		openAIMsg.ToolCalls = msg.ToolCalls
		openAIMsg.ToolCallID = msg.ToolCallID

		var renderedParts []any
		var hasRichContent bool

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
					"type":      "image_url",
					"image_url": rendered,
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

		if len(renderedParts) == 0 {
			if len(msg.ToolCalls) > 0 {
				openAIMsg.Content = nil
			} else {
				openAIMsg.Content = ""
			}
		} else if !hasRichContent && len(renderedParts) == 1 {
			if textPart, ok := renderedParts[0].(map[string]any); ok {
				openAIMsg.Content = textPart["text"]
			}
		} else {
			openAIMsg.Content = renderedParts
		}

		openAIMsgs = append(openAIMsgs, openAIMsg)
	}

	return &OpenAIRequest{
		Model:       model,
		Messages:    openAIMsgs,
		Tools:       c.Tools,
		ToolChoice:  c.ToolChoice,
		Temperature: c.Parameters.Temperature,
		TopP:        c.Parameters.TopP,
		MaxTokens:   c.Parameters.MaxTokens,
		Stop:        c.Parameters.Stop,
	}, nil
}

// FromRequest translates an OpenAIRequest back to core.Conversation.
func FromRequest(req *OpenAIRequest) (*core.Conversation, error) {
	if req == nil {
		return nil, fmt.Errorf("request is nil")
	}

	var c core.Conversation
	c.Model = req.Model

	var coreMsgs []core.Message
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			texts, err := extractTextContent(msg.Content)
			if err != nil {
				return nil, err
			}
			c.System = append(c.System, texts...)
			continue
		}

		var m core.Message
		m.Role = core.Role(msg.Role)
		m.Name = msg.Name
		m.ToolCalls = msg.ToolCalls
		m.ToolCallID = msg.ToolCallID

		if msg.Content != nil {
			parts, err := extractCoreParts(msg.Content, msg.Meta)
			if err != nil {
				return nil, err
			}
			m.Parts = parts
		}
		coreMsgs = append(coreMsgs, m)
	}
	c.Messages = coreMsgs
	c.Tools = req.Tools
	c.ToolChoice = req.ToolChoice

	c.Parameters.Temperature = req.Temperature
	c.Parameters.TopP = req.TopP
	c.Parameters.MaxTokens = req.MaxTokens
	c.Parameters.Stop = req.Stop

	return &c, nil
}

// Helpers
func renderPart(content any, meta map[string]any, pType string, renderers map[string]core.Renderer) (any, error) {
	if renderer, ok := renderers[pType]; ok && renderer != nil {
		return renderer(content, meta)
	}
	return core.DefaultRenderer(content, meta)
}

func extractCoreParts(content any, msgMeta map[string]any) ([]core.Part, error) {
	if content == nil {
		return nil, nil
	}

	if s, ok := content.(string); ok {
		p := core.Part{
			Type:    "text",
			Content: s,
			Meta:    msgMeta,
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
				} else if p.Type == "image_url" {
					p.Content = m["image_url"]
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

	return nil, fmt.Errorf("unsupported openai content type: %T", content)
}

func extractTextContent(content any) ([]string, error) {
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

func extractMeta(blob any) map[string]any {
	if m, ok := blob.(map[string]any); ok {
		return m
	}
	return make(map[string]any)
}
