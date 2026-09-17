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

func TestOpenAIToolCallingRoundTrip(t *testing.T) {
	req := &openai.OpenAIRequest{
		Model: "gpt-4o",
		Tools: []core.ToolDefinition{
			{
				Type: "function",
				Function: map[string]any{
					"name": "calc",
					"description": "Calculate math",
				},
			},
		},
		ToolChoice: "auto",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "Calculate 2+2"},
			{
				Role:    "assistant",
				Content: nil,
				ToolCalls: []core.ToolCall{
					{
						ID:   "call_abc123",
						Type: "function",
						Function: core.ToolFunction{
							Name:      "calc",
							Arguments: `{"expr":"2+2"}`,
						},
					},
				},
			},
			{
				Role:       "tool",
				ToolCallID: "call_abc123",
				Content:    "4",
			},
		},
	}

	// 1. FromRequest to core.Conversation
	conv, err := openai.FromRequest(req)
	if err != nil {
		t.Fatalf("FromRequest failed: %v", err)
	}

	if len(conv.Tools) != 1 || conv.Tools[0].Function["name"] != "calc" {
		t.Errorf("Tools not parsed: %+v", conv.Tools)
	}
	if len(conv.Messages) != 3 {
		t.Fatalf("Expected 3 messages in conversation, got %d", len(conv.Messages))
	}

	asst := conv.Messages[1]
	if asst.Role != core.RoleAssistant || len(asst.ToolCalls) != 1 || asst.ToolCalls[0].ID != "call_abc123" {
		t.Errorf("Assistant tool call not preserved: %+v", asst)
	}

	toolMsg := conv.Messages[2]
	if toolMsg.Role != core.RoleTool || toolMsg.ToolCallID != "call_abc123" {
		t.Errorf("Tool message not preserved: %+v", toolMsg)
	}

	// 2. ToRequest back to OpenAIRequest
	opts := core.NewExportOptions()
	reqBack, err := openai.ToRequest(conv, opts)
	if err != nil {
		t.Fatalf("ToRequest failed: %v", err)
	}

	if len(reqBack.Tools) != 1 || reqBack.Tools[0].Function["name"] != "calc" {
		t.Errorf("Tools not preserved on export: %+v", reqBack.Tools)
	}
	if len(reqBack.Messages) != 3 {
		t.Fatalf("Expected 3 messages in exported request, got %d", len(reqBack.Messages))
	}

	asstBack := reqBack.Messages[1]
	if asstBack.Role != "assistant" || len(asstBack.ToolCalls) != 1 || asstBack.ToolCalls[0].ID != "call_abc123" {
		t.Errorf("Assistant tool call not exported correctly: %+v", asstBack)
	}
	if asstBack.Content != nil {
		t.Errorf("Expected nil content on assistant tool call, got %v", asstBack.Content)
	}

	toolBack := reqBack.Messages[2]
	if toolBack.Role != "tool" || toolBack.ToolCallID != "call_abc123" || toolBack.Content != "4" {
		t.Errorf("Tool response not exported correctly: %+v", toolBack)
	}
}
