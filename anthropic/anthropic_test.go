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

	// 3. Verify RoleTool maps to "user" with tool_result
	if len(req.Messages) != 2 {
		t.Fatalf("Expected 2 messages, got %d", len(req.Messages))
	}
	if req.Messages[1].Role != "user" {
		t.Errorf("Expected Tool role to map to user, got %+v", req.Messages[1])
	}
	parts, ok := req.Messages[1].Content.([]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("Expected content parts for tool result, got %+v", req.Messages[1].Content)
	}
	partMap := parts[0].(map[string]any)
	if partMap["type"] != "tool_result" || partMap["content"] != "Tool result" {
		t.Errorf("Expected tool_result part, got %+v", partMap)
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

func TestAnthropicToolCalling(t *testing.T) {
	c := &core.Conversation{
		Model: "claude-3-opus",
		Tools: []core.ToolDefinition{
			{
				Type: "function",
				Function: map[string]any{
					"name":        "search",
					"description": "Search web",
					"parameters":  map[string]any{"type": "object"},
				},
			},
		},
		Messages: []core.Message{
			{
				Role:  core.RoleUser,
				Parts: []core.Part{{Type: "text", Content: "Search for Go"}},
			},
			{
				Role: core.RoleAssistant,
				ToolCalls: []core.ToolCall{
					{
						ID:   "toolu_123",
						Type: "function",
						Function: core.ToolFunction{
							Name:      "search",
							Arguments: `{"query":"Go"}`,
						},
					},
				},
			},
			{
				Role:       core.RoleTool,
				ToolCallID: "toolu_123",
				Parts:      []core.Part{{Type: "text", Content: "Found 100 results"}},
			},
		},
	}

	opts := core.NewExportOptions()
	opts.DefaultMaxTokens = 1000
	req, err := anthropic.ToRequest(c, opts)
	if err != nil {
		t.Fatalf("ToRequest failed: %v", err)
	}

	if len(req.Tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(req.Tools))
	}
	if len(req.Messages) != 3 {
		t.Fatalf("Expected 3 messages, got %d", len(req.Messages))
	}

	// Verify assistant tool_use content part
	asstParts, ok := req.Messages[1].Content.([]any)
	if !ok || len(asstParts) != 1 {
		t.Fatalf("Expected assistant to have array of parts: %+v", req.Messages[1].Content)
	}
	toolUsePart := asstParts[0].(map[string]any)
	if toolUsePart["type"] != "tool_use" || toolUsePart["name"] != "search" || toolUsePart["id"] != "toolu_123" {
		t.Errorf("Unexpected tool_use part: %+v", toolUsePart)
	}

	// Verify tool response mapped to user tool_result
	toolParts, ok := req.Messages[2].Content.([]any)
	if !ok || len(toolParts) != 1 {
		t.Fatalf("Expected tool msg to have array of parts: %+v", req.Messages[2].Content)
	}
	toolResultPart := toolParts[0].(map[string]any)
	if toolResultPart["type"] != "tool_result" || toolResultPart["tool_use_id"] != "toolu_123" || toolResultPart["content"] != "Found 100 results" {
		t.Errorf("Unexpected tool_result part: %+v", toolResultPart)
	}

	// FromRequest roundtrip
	cBack, err := anthropic.FromRequest(req)
	if err != nil {
		t.Fatalf("FromRequest failed: %v", err)
	}
	if len(cBack.Messages) != 3 {
		t.Fatalf("Expected 3 messages back, got %d", len(cBack.Messages))
	}
	if len(cBack.Messages[1].ToolCalls) != 1 || cBack.Messages[1].ToolCalls[0].ID != "toolu_123" {
		t.Errorf("Assistant tool call not extracted back: %+v", cBack.Messages[1])
	}
	if cBack.Messages[2].Role != core.RoleTool || cBack.Messages[2].ToolCallID != "toolu_123" {
		t.Errorf("Tool message not extracted back: %+v", cBack.Messages[2])
	}
}
