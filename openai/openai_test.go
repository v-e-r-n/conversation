package openai_test

import (
	"testing"

	"github.com/v-e-r-n/conversation/core"
	"github.com/v-e-r-n/conversation/openai"
)

func TestOpenAIToRequest(t *testing.T) {
	c := &core.Conversation{
		Model: "gpt-4o",
	}
	c.SetSystemPrompt("System rule")
	msg := c.AddUserMessage("Hello world!")
	msg.Parts[0].SetMeta("emotion", "excited")

	opts := core.NewExportOptions()
	opts.Renderers["text"] = func(content any, meta map[string]any) (any, error) {
		text := content.(string)
		if emotion, ok := meta["emotion"].(string); ok {
			return text + " [" + emotion + "]", nil
		}
		return text, nil
	}

	req, err := openai.ToRequest(c, opts)
	if err != nil {
		t.Fatalf("ToRequest failed: %v", err)
	}

	if len(req.Messages) != 2 {
		t.Fatalf("Expected 2 messages, got %d", len(req.Messages))
	}

	// 1. Verify system prompt bypassed renderer
	if req.Messages[0].Role != "system" || req.Messages[0].Content != "System rule" {
		t.Errorf("Expected System message to bypass renderer. Got: %v", req.Messages[0].Content)
	}

	// 2. Verify user message was formatted by renderer
	if req.Messages[1].Role != "user" || req.Messages[1].Content != "Hello world! [excited]" {
		t.Errorf("Expected formatted user message. Got: %v", req.Messages[1].Content)
	}
}

func TestOpenAIFromRequest(t *testing.T) {
	req := &openai.OpenAIRequest{
		Model: "gpt-4o",
		Messages: []openai.OpenAIMessage{
			{Role: "system", Content: "Instructions"},
			{Role: "user", Content: "User message"},
		},
	}

	c, err := openai.FromRequest(req)
	if err != nil {
		t.Fatalf("FromRequest failed: %v", err)
	}

	if len(c.System) != 1 || c.System[0] != "Instructions" {
		t.Errorf("Failed to extract system instructions from request: %+v", c.System)
	}

	if len(c.Messages) != 1 || c.Messages[0].Role != core.RoleUser || c.Messages[0].Parts[0].Content != "User message" {
		t.Errorf("Failed to extract standard messages from request: %+v", c.Messages)
	}
}
