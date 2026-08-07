package gemini

import (
	"github.com/v-e-r-n/conversation/core"
	"fmt"
	"strings"
)

type GeminiPart struct {
	Text             string         `json:"text,omitempty"`
	InlineData       map[string]any `json:"inlineData,omitempty"`
	FunctionCall     map[string]any `json:"functionCall,omitempty"`
	FunctionResponse map[string]any `json:"functionResponse,omitempty"`
}

type GeminiContent struct {
	Role  string       `json:"role"`
	Parts []GeminiPart `json:"parts"`
}

type SystemInstruction struct {
	Parts []GeminiPart `json:"parts"`
}

type GenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type GeminiRequest struct {
	Contents          []GeminiContent    `json:"contents"`
	SystemInstruction *SystemInstruction `json:"systemInstruction,omitempty"`
	GenerationConfig  *GenerationConfig  `json:"generationConfig,omitempty"`
}

// ToRequest translates core.Conversation to GeminiRequest.
func ToRequest(c *core.Conversation, opts *core.ExportOptions) (*GeminiRequest, error) {
	if opts == nil {
		return nil, fmt.Errorf("export options are required and must be non-nil")
	}

	var contents []GeminiContent

	// 1. Process & collapse all system prompts from c.System
	var systemInstruction *SystemInstruction
	if len(c.System) > 0 {
		systemInstruction = &SystemInstruction{
			Parts: []GeminiPart{
				{Text: strings.Join(c.System, "\n\n")},
			},
		}
	}

	// 2. Map standard messages (user, model)
	for _, msg := range c.Messages {
		var gemMsg GeminiContent
		role := "user"
		if msg.Role == core.RoleAssistant {
			role = "model"
		}
		gemMsg.Role = role

		var gemParts []GeminiPart
		for _, part := range msg.Parts {
			rendered, err := renderPart(part.Content, part.Meta, part.Type, opts.Renderers)
			if err != nil {
				return nil, err
			}

			switch part.Type {
			case "text":
				if s, ok := rendered.(string); ok {
					gemParts = append(gemParts, GeminiPart{Text: s})
				} else {
					gemParts = append(gemParts, GeminiPart{Text: fmt.Sprintf("%v", rendered)})
				}
			case "image":
				if m, ok := rendered.(map[string]any); ok {
					gemParts = append(gemParts, GeminiPart{InlineData: m})
				}
			default:
				gemParts = append(gemParts, GeminiPart{Text: fmt.Sprintf("%v", rendered)})
			}
		}

		gemMsg.Parts = gemParts
		contents = append(contents, gemMsg)
	}

	var config *GenerationConfig
	if c.Parameters.Temperature != nil || c.Parameters.TopP != nil || c.Parameters.MaxTokens != nil || len(c.Parameters.Stop) > 0 {
		config = &GenerationConfig{
			Temperature:     c.Parameters.Temperature,
			TopP:            c.Parameters.TopP,
			MaxOutputTokens: c.Parameters.MaxTokens,
			StopSequences:   c.Parameters.Stop,
		}
	}

	return &GeminiRequest{
		Contents:          contents,
		SystemInstruction: systemInstruction,
		GenerationConfig:  config,
	}, nil
}

// Helpers
func renderPart(content any, meta map[string]any, pType string, renderers map[string]core.Renderer) (any, error) {
	if renderer, ok := renderers[pType]; ok && renderer != nil {
		return renderer(content, meta)
	}
	return core.DefaultRenderer(content, meta)
}
