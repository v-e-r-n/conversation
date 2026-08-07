package anthropic_test

import (
	"testing"

	"github.com/v-e-r-n/conversation/anthropic"
	"github.com/v-e-r-n/conversation/core"
)

func TestAnthropicToRequest(t *testing.T) {
	c := &core.Conversation{
		Model: "claude-3",
	}
	c.SetSystemPrompt("System 1")
	c.AppendSystemPrompt("System 2")

	// Add messages including RoleTool
	c.AddUserMessage("Hello")
	c.AddMessage(core.RoleTool, "Tool result")

	opts := core.NewExportOptions()
	opts.DefaultMaxTokens = 999

	req, err := anthropic.ToRequest(c, opts)
	if err != nil {
		t.Fatalf("ToRequest failed: %v", err)
	}

	// 1. Verify system prompts collapsed
	if req.System != "System 1\n\nSystem 2" {
		t.Errorf("Expected combined system, got '%s'", req.System)
	}

	// 2. Verify fallback MaxTokens
	if req.MaxTokens != 999 {
		t.Errorf("Expected fallback MaxTokens to be 999, got %d", req.MaxTokens)
	}

	// 3. Verify RoleTool maps to "user"
	if len(req.Messages) != 2 {
		t.Fatalf("Expected 2 messages, got %d", len(req.Messages))
	}
	if req.Messages[1].Role != "user" || req.Messages[1].Content != "Tool result" {
		t.Errorf("Expected Tool role to map to user, got %+v", req.Messages[1])
	}
}

func TestAnthropicFromRequest(t *testing.T) {
	req := &anthropic.AnthropicRequest{
		Model:     "claude-3",
		System:    "Instructions",
		MaxTokens: 50,
		Messages: []anthropic.AnthropicMessage{
			{Role: "user", Content: "Prompt"},
		},
	}

	c, err := anthropic.FromRequest(req)
	if err != nil {
		t.Fatalf("FromRequest failed: %v", err)
	}

	if len(c.System) != 1 || c.System[0] != "Instructions" {
		t.Errorf("Expected System prompt to map to System list")
	}

	if *c.Parameters.MaxTokens != 50 {
		t.Errorf("Expected MaxTokens 50, got %d", *c.Parameters.MaxTokens)
	}
}
