package core_test

import (
	"encoding/json"
	"testing"

	"codeberg.org/v-e-r-n/conversation/core"
)

func TestJSONUnmarshalingAndPolymorphicContent(t *testing.T) {
	jsonStr := `{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Hello!"}
		]
	}`

	var conv1 core.Conversation
	if err := json.Unmarshal([]byte(jsonStr), &conv1); err != nil {
		t.Fatalf("Failed to unmarshal single-string conversation: %v", err)
	}

	if len(conv1.Messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(conv1.Messages))
	}
	if len(conv1.Messages[0].Parts) != 1 {
		t.Fatalf("Expected 1 part, got %d", len(conv1.Messages[0].Parts))
	}
	if conv1.Messages[0].Parts[0].Type != "text" || conv1.Messages[0].Parts[0].Content != "Hello!" {
		t.Errorf("Mismatch in unmarshaled text part: %+v", conv1.Messages[0].Parts[0])
	}

	jsonArrStr := `{
		"model": "gpt-4o",
		"messages": [
			{
				"role": "user",
				"content": [
					"First string block",
					{"type": "text", "text": "Second text block"},
					{"type": "image", "content": {"url": "https://example.com/img.png"}, "meta": {"tracing_id": 456}}
				]
			}
		]
	}`

	var conv2 core.Conversation
	if err := json.Unmarshal([]byte(jsonArrStr), &conv2); err != nil {
		t.Fatalf("Failed to unmarshal array conversation: %v", err)
	}

	parts := conv2.Messages[0].Parts
	if len(parts) != 3 {
		t.Fatalf("Expected 3 parts, got %d", len(parts))
	}

	if parts[0].Type != "text" || parts[0].Content != "First string block" {
		t.Errorf("Expected first part to be text, got %+v", parts[0])
	}
	if parts[1].Type != "text" || parts[1].Content != "Second text block" {
		t.Errorf("Expected second part to be text, got %+v", parts[1])
	}
	if parts[2].Type != "image" {
		t.Errorf("Expected third part to be image, got %s", parts[2].Type)
	}
	imgContent, ok := parts[2].Content.(map[string]any)
	if !ok || imgContent["url"] != "https://example.com/img.png" {
		t.Errorf("Expected image URL, got %+v", parts[2].Content)
	}
	if parts[2].Meta["tracing_id"].(float64) != 456 {
		t.Errorf("Expected tracing_id 456, got %+v", parts[2].Meta["tracing_id"])
	}
}

func TestJSONLosslessPreservation(t *testing.T) {
	jsonStr := `{
		"model": "experimental-model",
		"messages": [
			{"role": "user", "content": "Hi!"}
		],
		"custom_telemetry_key": "some-random-uuid",
		"temperature": 0.8
	}`

	var conv core.Conversation
	if err := json.Unmarshal([]byte(jsonStr), &conv); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if conv.Parameters.Temperature == nil || *conv.Parameters.Temperature != 0.8 {
		t.Errorf("Expected temperature 0.8, got %v", conv.Parameters.Temperature)
	}

	bytes, err := json.Marshal(&conv)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(bytes, &raw); err != nil {
		t.Fatalf("Unmarshal re-check failed: %v", err)
	}

	if raw["custom_telemetry_key"] != "some-random-uuid" {
		t.Errorf("Expected preserved custom_telemetry_key, got %+v", raw["custom_telemetry_key"])
	}
}

func TestHelpers(t *testing.T) {
	c := &core.Conversation{
		Model: "test-model",
	}

	c.SetSystemPrompt("Instruction 1")
	c.AppendSystemPrompt("Instruction 2")
	if len(c.System) != 2 || c.CombinedSystemPrompt() != "Instruction 1\n\nInstruction 2" {
		t.Errorf("Failed system prompt helpers, got system: %+v", c.System)
	}

	msg := c.AddUserMessage("Hello")
	if len(c.Messages) != 1 || c.Messages[0].Role != core.RoleUser {
		t.Errorf("AddUserMessage failed")
	}

	msg.AddPart("image", "some-data").SetMeta("format", "png")
	if len(msg.Parts) != 2 || msg.Parts[1].Type != "image" || msg.Parts[1].Meta["format"] != "png" {
		t.Errorf("AddPart or SetMeta failed")
	}

	c.AddAssistantMessage("Response")
	if len(c.Messages) != 2 || c.Messages[1].Role != core.RoleAssistant {
		t.Errorf("AddAssistantMessage failed")
	}
}
