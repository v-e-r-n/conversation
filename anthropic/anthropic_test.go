package anthropic_test

import (
	"strings"
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

func TestAnthropicResponseRoundTrip(t *testing.T) {
	c := &core.Conversation{
		Model: "claude-3-7-sonnet-20250219",
		Messages: []core.Message{
			{
				Role: core.RoleUser,
				Parts: []core.Part{
					{Type: "text", Content: "Calculate 2+2"},
				},
			},
			{
				Role: core.RoleAssistant,
				Parts: []core.Part{
					{
						Type:    "thinking",
						Content: "Let me compute this arithmetic expression.",
						Meta:    map[string]any{"signature": "sig_valid_12345"},
					},
					{
						Type:    "text",
						Content: "I will calculate 2+2 for you.",
					},
				},
				ToolCalls: []core.ToolCall{
					{
						ID:   "toolu_calc_01",
						Type: "function",
						Function: core.ToolFunction{
							Name:      "calculator",
							Arguments: `{"expr":"2+2"}`,
						},
					},
				},
			},
		},
	}

	usage := &anthropic.Usage{
		InputTokens:              150,
		OutputTokens:             42,
		CacheCreationInputTokens: 50,
		CacheReadInputTokens:     20,
	}

	resp, err := anthropic.ToResponse(c, "msg_test_01", "claude-3-7-sonnet-20250219", usage)
	if err != nil {
		t.Fatalf("ToResponse failed: %v", err)
	}

	if resp.ID != "msg_test_01" {
		t.Errorf("expected ID msg_test_01, got %s", resp.ID)
	}
	if resp.Role != "assistant" || resp.Type != "message" {
		t.Errorf("unexpected role/type: %s / %s", resp.Role, resp.Type)
	}
	if resp.StopReason == nil || *resp.StopReason != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %v", resp.StopReason)
	}
	if resp.Usage.InputTokens != 150 || resp.Usage.OutputTokens != 42 {
		t.Errorf("unexpected usage: %+v", resp.Usage)
	}

	if len(resp.Content) != 3 {
		t.Fatalf("expected 3 content blocks, got %d", len(resp.Content))
	}

	// 1. Thinking block
	if resp.Content[0].Type != anthropic.BlockTypeThinking {
		t.Errorf("expected block 0 to be thinking, got %s", resp.Content[0].Type)
	}
	if resp.Content[0].Thinking != "Let me compute this arithmetic expression." {
		t.Errorf("unexpected thinking text: %s", resp.Content[0].Thinking)
	}
	if resp.Content[0].Signature != "sig_valid_12345" {
		t.Errorf("expected signature sig_valid_12345, got %s", resp.Content[0].Signature)
	}

	// 2. Text block
	if resp.Content[1].Type != anthropic.BlockTypeText || resp.Content[1].Text != "I will calculate 2+2 for you." {
		t.Errorf("unexpected text block: %+v", resp.Content[1])
	}

	// 3. ToolUse block
	if resp.Content[2].Type != anthropic.BlockTypeToolUse || resp.Content[2].Name != "calculator" || resp.Content[2].ID != "toolu_calc_01" {
		t.Errorf("unexpected tool_use block: %+v", resp.Content[2])
	}

	// Verify FromResponse restores the conversation turn
	cBack, err := anthropic.FromResponse(resp)
	if err != nil {
		t.Fatalf("FromResponse failed: %v", err)
	}
	if len(cBack.Messages) != 1 {
		t.Fatalf("expected 1 assistant message, got %d", len(cBack.Messages))
	}
	asst := cBack.Messages[0]
	if asst.Role != core.RoleAssistant {
		t.Errorf("expected assistant role, got %s", asst.Role)
	}
	if len(asst.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(asst.Parts))
	}
	if asst.Parts[0].Type != "thinking" || asst.Parts[0].Meta["signature"] != "sig_valid_12345" {
		t.Errorf("thinking part not restored accurately: %+v", asst.Parts[0])
	}
	if asst.Parts[1].Type != "text" || asst.Parts[1].Content != "I will calculate 2+2 for you." {
		t.Errorf("text part not restored accurately: %+v", asst.Parts[1])
	}
	if len(asst.ToolCalls) != 1 || asst.ToolCalls[0].ID != "toolu_calc_01" || asst.ToolCalls[0].Function.Name != "calculator" {
		t.Errorf("tool call not restored accurately: %+v", asst.ToolCalls)
	}
}

func TestAnthropicStreamingSSE(t *testing.T) {
	// 1. message_start
	startEv := anthropic.MessageStartEvent{
		Type: anthropic.EventMessageStart,
		Message: anthropic.MessageStartPayload{
			ID:    "msg_stream_01",
			Type:  "message",
			Role:  "assistant",
			Model: "claude-3-5-sonnet",
			Usage: anthropic.Usage{InputTokens: 25, OutputTokens: 1},
		},
	}
	raw, err := anthropic.FormatSSE(anthropic.EventMessageStart, startEv)
	if err != nil {
		t.Fatalf("FormatSSE message_start failed: %v", err)
	}
	parsed, err := anthropic.ParseSSE(raw)
	if err != nil {
		t.Fatalf("ParseSSE message_start failed: %v", err)
	}
	if parsed.Event != string(anthropic.EventMessageStart) {
		t.Errorf("expected event message_start, got %s", parsed.Event)
	}

	// 2. content_block_start (thinking)
	blockStartEv := anthropic.ContentBlockStartEvent{
		Type:  anthropic.EventContentBlockStart,
		Index: 0,
		ContentBlock: anthropic.ContentBlock{
			Type:     anthropic.BlockTypeThinking,
			Thinking: "",
		},
	}
	raw, err = anthropic.FormatSSE(anthropic.EventContentBlockStart, blockStartEv)
	if err != nil {
		t.Fatalf("FormatSSE content_block_start failed: %v", err)
	}
	parsed, err = anthropic.ParseSSE(raw)
	if err != nil {
		t.Fatalf("ParseSSE content_block_start failed: %v", err)
	}
	if parsed.Event != string(anthropic.EventContentBlockStart) {
		t.Errorf("expected content_block_start, got %s", parsed.Event)
	}

	// 3. content_block_delta (thinking_delta)
	thinkingDeltaEv := anthropic.ContentBlockDeltaEvent{
		Type:  anthropic.EventContentBlockDelta,
		Index: 0,
		Delta: anthropic.ContentBlockDelta{
			Type:     anthropic.DeltaTypeThinking,
			Thinking: "Deep thought...",
		},
	}
	raw, err = anthropic.FormatSSE(anthropic.EventContentBlockDelta, thinkingDeltaEv)
	if err != nil {
		t.Fatalf("FormatSSE thinking_delta failed: %v", err)
	}
	parsed, err = anthropic.ParseSSE(raw)
	if err != nil {
		t.Fatalf("ParseSSE thinking_delta failed: %v", err)
	}
	if !strings.Contains(string(parsed.Data), "Deep thought...") {
		t.Errorf("delta payload missing expected thinking text: %s", string(parsed.Data))
	}

	// 4. content_block_delta (signature_delta)
	sigDeltaEv := anthropic.ContentBlockDeltaEvent{
		Type:  anthropic.EventContentBlockDelta,
		Index: 0,
		Delta: anthropic.ContentBlockDelta{
			Type:      anthropic.DeltaTypeSignature,
			Signature: "sig_abc123",
		},
	}
	raw, err = anthropic.FormatSSE(anthropic.EventContentBlockDelta, sigDeltaEv)
	if err != nil {
		t.Fatalf("FormatSSE signature_delta failed: %v", err)
	}
	parsed, err = anthropic.ParseSSE(raw)
	if err != nil {
		t.Fatalf("ParseSSE signature_delta failed: %v", err)
	}
	if !strings.Contains(string(parsed.Data), "sig_abc123") {
		t.Errorf("delta payload missing expected signature: %s", string(parsed.Data))
	}

	// 5. content_block_stop
	stopBlockEv := anthropic.ContentBlockStopEvent{
		Type:  anthropic.EventContentBlockStop,
		Index: 0,
	}
	raw, err = anthropic.FormatSSE(anthropic.EventContentBlockStop, stopBlockEv)
	if err != nil {
		t.Fatalf("FormatSSE content_block_stop failed: %v", err)
	}

	// 6. message_delta
	stopReason := "end_turn"
	msgDeltaEv := anthropic.MessageDeltaEvent{
		Type: anthropic.EventMessageDelta,
		Delta: anthropic.MessageDeltaPayload{
			StopReason: &stopReason,
		},
		Usage: &anthropic.MessageDeltaUsage{
			OutputTokens: 30,
		},
	}
	raw, err = anthropic.FormatSSE(anthropic.EventMessageDelta, msgDeltaEv)
	if err != nil {
		t.Fatalf("FormatSSE message_delta failed: %v", err)
	}
	parsed, err = anthropic.ParseSSE(raw)
	if err != nil {
		t.Fatalf("ParseSSE message_delta failed: %v", err)
	}
	if !strings.Contains(string(parsed.Data), "end_turn") || !strings.Contains(string(parsed.Data), "30") {
		t.Errorf("unexpected message_delta payload: %s", string(parsed.Data))
	}

	// 7. message_stop
	raw, err = anthropic.FormatSSE(anthropic.EventMessageStop, anthropic.MessageStopEvent{Type: anthropic.EventMessageStop})
	if err != nil {
		t.Fatalf("FormatSSE message_stop failed: %v", err)
	}

	// 8. ping
	raw, err = anthropic.FormatSSE(anthropic.EventPing, anthropic.PingEvent{Type: anthropic.EventPing})
	if err != nil {
		t.Fatalf("FormatSSE ping failed: %v", err)
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

