package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/v-e-r-n/conversation/core"
)

// ContentBlockType represents the discriminator type of an Anthropic content block.
type ContentBlockType string

const (
	BlockTypeText            ContentBlockType = "text"
	BlockTypeThinking        ContentBlockType = "thinking"
	BlockTypeRedactedThinking ContentBlockType = "redacted_thinking"
	BlockTypeToolUse         ContentBlockType = "tool_use"
	BlockTypeToolResult      ContentBlockType = "tool_result"
	BlockTypeImage           ContentBlockType = "image"
)

// ImageSource represents the media source for an image content block.
type ImageSource struct {
	Type      string `json:"type"` // "base64" or "url"
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// ContentBlock represents a typed Anthropic content block.
type ContentBlock struct {
	Type ContentBlockType `json:"type"`

	// Text block
	Text string `json:"text,omitempty"`

	// Thinking & Redacted Thinking block
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"` // For redacted_thinking

	// ToolUse block
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// ToolResult block
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"` // string or []ContentBlock
	IsError   bool   `json:"is_error,omitempty"`

	// Image block
	Source *ImageSource `json:"source,omitempty"`
}

// Usage represents token usage statistics for an Anthropic request/response.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// AnthropicResponse represents a top-level non-streaming response from the Messages API.
type AnthropicResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"` // "message"
	Role         string         `json:"role"` // "assistant"
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"`
	StopReason   *string        `json:"stop_reason,omitempty"`   // "end_turn", "max_tokens", "stop_sequence", "tool_use"
	StopSequence *string        `json:"stop_sequence,omitempty"`
	Usage        Usage          `json:"usage"`
}

// ToResponse constructs an AnthropicResponse from a core.Conversation.
// It inspects the last assistant message in the conversation to build the content blocks.
func ToResponse(c *core.Conversation, id string, model string, usage *Usage) (*AnthropicResponse, error) {
	if c == nil {
		return nil, fmt.Errorf("conversation is nil")
	}

	if id == "" {
		id = "msg_default"
	}
	if model == "" {
		model = c.Model
	}

	var assistantMsg *core.Message
	for i := len(c.Messages) - 1; i >= 0; i-- {
		if c.Messages[i].Role == core.RoleAssistant {
			assistantMsg = &c.Messages[i]
			break
		}
	}

	resp := &AnthropicResponse{
		ID:      id,
		Type:    "message",
		Role:    "assistant",
		Model:   model,
		Content: make([]ContentBlock, 0),
	}

	if usage != nil {
		resp.Usage = *usage
	}

	if assistantMsg == nil {
		// Empty assistant content
		defaultStop := "end_turn"
		resp.StopReason = &defaultStop
		return resp, nil
	}

	// 1. Map parts (text, thinking, image)
	for _, part := range assistantMsg.Parts {
		switch part.Type {
		case "thinking":
			textVal := ""
			if s, ok := part.Content.(string); ok {
				textVal = s
			}
			sigVal := ""
			if part.Meta != nil {
				if sig, ok := part.Meta["signature"].(string); ok {
					sigVal = sig
				}
			}
			resp.Content = append(resp.Content, ContentBlock{
				Type:      BlockTypeThinking,
				Thinking:  textVal,
				Signature: sigVal,
			})
		case "text":
			textVal := ""
			if s, ok := part.Content.(string); ok {
				textVal = s
			} else if part.Content != nil {
				textVal = fmt.Sprintf("%v", part.Content)
			}
			resp.Content = append(resp.Content, ContentBlock{
				Type: BlockTypeText,
				Text: textVal,
			})
		case "image":
			var src *ImageSource
			if s, ok := part.Content.(string); ok {
				src = &ImageSource{Type: "url", URL: s}
			}
			resp.Content = append(resp.Content, ContentBlock{
				Type:   BlockTypeImage,
				Source: src,
			})
		}
	}

	// 2. Map tool calls
	for _, tc := range assistantMsg.ToolCalls {
		rawInput := json.RawMessage("{}")
		if tc.Function.Arguments != "" {
			trimmed := strings.TrimSpace(tc.Function.Arguments)
			if trimmed != "" {
				rawInput = json.RawMessage(trimmed)
			}
		}
		resp.Content = append(resp.Content, ContentBlock{
			Type:  BlockTypeToolUse,
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: rawInput,
		})
	}

	// 3. Determine StopReason
	stopReason := "end_turn"
	if len(assistantMsg.ToolCalls) > 0 {
		stopReason = "tool_use"
	}
	resp.StopReason = &stopReason

	return resp, nil
}

// FromResponse translates an AnthropicResponse into a core.Conversation containing the assistant message turn.
func FromResponse(resp *AnthropicResponse) (*core.Conversation, error) {
	if resp == nil {
		return nil, fmt.Errorf("response is nil")
	}

	c := &core.Conversation{
		Model: resp.Model,
	}

	asstMsg := core.Message{
		Role:      core.RoleAssistant,
		Parts:     make([]core.Part, 0),
		ToolCalls: make([]core.ToolCall, 0),
	}

	for _, block := range resp.Content {
		switch block.Type {
		case BlockTypeText:
			asstMsg.Parts = append(asstMsg.Parts, core.Part{
				Type:    "text",
				Content: block.Text,
			})
		case BlockTypeThinking:
			meta := make(map[string]any)
			if block.Signature != "" {
				meta["signature"] = block.Signature
			}
			asstMsg.Parts = append(asstMsg.Parts, core.Part{
				Type:    "thinking",
				Content: block.Thinking,
				Meta:    meta,
			})
		case BlockTypeToolUse:
			argsStr := "{}"
			if len(block.Input) > 0 {
				argsStr = string(block.Input)
			}
			asstMsg.ToolCalls = append(asstMsg.ToolCalls, core.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: core.ToolFunction{
					Name:      block.Name,
					Arguments: argsStr,
				},
			})
		}
	}

	c.Messages = []core.Message{asstMsg}
	return c, nil
}
