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

	// Re-export roundtrip: ensure assistant message has exactly 1 tool_use block with id and no ghost parts
	reqAgain, err := anthropic.ToRequest(cBack, opts)
	if err != nil {
		t.Fatalf("ToRequest on roundtrip failed: %v", err)
	}
	asstPartsAgain, ok := reqAgain.Messages[1].Content.([]any)
	if !ok || len(asstPartsAgain) != 1 {
		t.Fatalf("Expected exactly 1 assistant content part on re-export, got %d: %+v", len(asstPartsAgain), reqAgain.Messages[1].Content)
	}
	toolUsePartAgain, ok := asstPartsAgain[0].(map[string]any)
	if !ok || toolUsePartAgain["id"] != "toolu_123" || toolUsePartAgain["name"] != "search" {
		t.Errorf("Unexpected tool_use part on re-export: %+v", asstPartsAgain[0])
	}
}

func TestAnthropicThinkingBlocksRoundTrip(t *testing.T) {
	// Incoming Anthropic request containing thinking blocks from previous turn
	req := &anthropic.AnthropicRequest{
		Model:     "claude-3-7-sonnet-20250219",
		MaxTokens: 1000,
		Messages: []anthropic.AnthropicMessage{
			{
				Role:    "user",
				Content: "Explain quantum computing",
			},
			{
				Role: "assistant",
				Content: []any{
					map[string]any{
						"type":      "thinking",
						"thinking":  "Thinking deeply about superposition...",
						"signature": "sig_quant_xyz",
					},
					map[string]any{
						"type": "text",
						"text": "Quantum computing harnesses superposition and entanglement.",
					},
				},
			},
		},
	}

	conv, err := anthropic.FromRequest(req)
	if err != nil {
		t.Fatalf("FromRequest failed: %v", err)
	}

	if len(conv.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(conv.Messages))
	}
	asst := conv.Messages[1]
	if len(asst.Parts) != 2 {
		t.Fatalf("expected 2 parts in assistant message, got %d", len(asst.Parts))
	}
	if asst.Parts[0].Type != "thinking" || asst.Parts[0].Content != "Thinking deeply about superposition..." {
		t.Errorf("unexpected thinking part: %+v", asst.Parts[0])
	}
	if asst.Parts[0].Meta["signature"] != "sig_quant_xyz" {
		t.Errorf("expected signature sig_quant_xyz, got %v", asst.Parts[0].Meta["signature"])
	}

	// Export back to AnthropicRequest
	opts := core.NewExportOptions()
	reqExported, err := anthropic.ToRequest(conv, opts)
	if err != nil {
		t.Fatalf("ToRequest failed: %v", err)
	}

	asstParts, ok := reqExported.Messages[1].Content.([]any)
	if !ok || len(asstParts) != 2 {
		t.Fatalf("expected 2 exported parts in assistant message, got %+v", reqExported.Messages[1].Content)
	}
	thinkingPart, ok := asstParts[0].(map[string]any)
	if !ok || thinkingPart["type"] != "thinking" || thinkingPart["thinking"] != "Thinking deeply about superposition..." || thinkingPart["signature"] != "sig_quant_xyz" {
		t.Errorf("unexpected exported thinking part: %+v", asstParts[0])
	}
}

func TestAnthropicPolymorphicSystemAndMetadata(t *testing.T) {
	// System prompt as structured content block array
	req := &anthropic.AnthropicRequest{
		Model:     "claude-3-opus",
		MaxTokens: 200,
		System: []any{
			map[string]any{"type": "text", "text": "Base system directive"},
			map[string]any{"type": "text", "text": "Secondary persona directive"},
		},
		Messages: []anthropic.AnthropicMessage{
			{Role: "user", Content: "Hi"},
		},
		Stream: true,
		Metadata: &anthropic.Metadata{
			UserID: "usr_tenant_42",
		},
	}

	conv, err := anthropic.FromRequest(req)
	if err != nil {
		t.Fatalf("FromRequest failed: %v", err)
	}

	if len(conv.System) != 2 {
		t.Fatalf("expected 2 system prompts extracted, got %d", len(conv.System))
	}
	if conv.System[0] != "Base system directive" || conv.System[1] != "Secondary persona directive" {
		t.Errorf("unexpected system contents: %+v", conv.System)
	}
	if conv.Meta["user_id"] != "usr_tenant_42" {
		t.Errorf("expected user_id in conv.Meta, got %v", conv.Meta["user_id"])
	}
	if conv.Meta["stream"] != true {
		t.Errorf("expected stream in conv.Meta, got %v", conv.Meta["stream"])
	}

	// Export back to AnthropicRequest
	opts := core.NewExportOptions()
	reqExported, err := anthropic.ToRequest(conv, opts)
	if err != nil {
		t.Fatalf("ToRequest failed: %v", err)
	}

	if reqExported.System != "Base system directive\n\nSecondary persona directive" {
		t.Errorf("expected collapsed system prompt, got %s", reqExported.System)
	}
	if !reqExported.Stream {
		t.Errorf("expected stream to be true")
	}
	if reqExported.Metadata == nil || reqExported.Metadata.UserID != "usr_tenant_42" {
		t.Errorf("expected metadata user_id preserved, got %+v", reqExported.Metadata)
	}
}

