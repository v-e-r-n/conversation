package openai

import (
	"encoding/json"
	"testing"

	"github.com/v-e-r-n/conversation/core"
)

func TestFromResponseRequest_SimpleStringInput(t *testing.T) {
	temp := 0.7
	maxTokens := 256
	req := &OpenAIResponseRequest{
		Model:              "gpt-4o",
		Instructions:       "You are a helpful assistant.",
		Input:              "What is the capital of France?",
		PreviousResponseID: "resp_init_001",
		Temperature:        &temp,
		MaxOutputTokens:    &maxTokens,
	}

	conv, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if conv.Model != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", conv.Model)
	}
	if len(conv.System) != 1 || conv.System[0] != "You are a helpful assistant." {
		t.Errorf("unexpected system instructions: %v", conv.System)
	}
	if len(conv.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(conv.Messages))
	}

	msg := conv.Messages[0]
	if msg.Role != core.RoleUser {
		t.Errorf("expected user role, got %s", msg.Role)
	}
	if len(msg.Parts) != 1 || msg.Parts[0].Content != "What is the capital of France?" {
		t.Errorf("unexpected message part: %+v", msg.Parts)
	}

	// Verify previous_response_id is preserved in metadata
	if prevID, ok := msg.Parts[0].Meta["previous_response_id"].(string); !ok || prevID != "resp_init_001" {
		t.Errorf("expected previous_response_id 'resp_init_001' in metadata, got %v", msg.Parts[0].Meta)
	}

	if conv.Parameters.Temperature == nil || *conv.Parameters.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %v", conv.Parameters.Temperature)
	}
	if conv.Parameters.MaxTokens == nil || *conv.Parameters.MaxTokens != 256 {
		t.Errorf("expected max tokens 256, got %v", conv.Parameters.MaxTokens)
	}
}

func TestFromResponseRequest_MultiTurnAndTools(t *testing.T) {
	req := &OpenAIResponseRequest{
		Model:        "gpt-4o",
		Instructions: "System directive",
		Input: []ResponseItem{
			{
				Type:    "message",
				Role:    "user",
				Content: "Check order 42",
			},
			{
				Type:      "function_call",
				CallID:    "call_order_42",
				Name:      "get_order",
				Arguments: `{"order_id":42}`,
			},
			{
				Type:   "function_call_output",
				CallID: "call_order_42",
				Output: `{"status":"shipped"}`,
			},
			{
				Type:    "message",
				Role:    "assistant",
				Content: "Your order has been shipped.",
			},
		},
		Tools: []core.ToolDefinition{
			{
				Type: "function",
				Function: map[string]any{
					"name":        "get_order",
					"description": "Look up an order by ID",
				},
			},
		},
	}

	conv, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(conv.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(conv.Messages))
	}

	// Message 0: User turn
	if conv.Messages[0].Role != core.RoleUser || conv.Messages[0].Parts[0].Content != "Check order 42" {
		t.Errorf("unexpected message 0: %+v", conv.Messages[0])
	}

	// Message 1: Tool call
	if conv.Messages[1].Role != core.RoleAssistant || len(conv.Messages[1].ToolCalls) != 1 {
		t.Fatalf("unexpected message 1 tool calls: %+v", conv.Messages[1])
	}
	tc := conv.Messages[1].ToolCalls[0]
	if tc.ID != "call_order_42" || tc.Function.Name != "get_order" || tc.Function.Arguments != `{"order_id":42}` {
		t.Errorf("unexpected tool call details: %+v", tc)
	}

	// Message 2: Tool output
	if conv.Messages[2].Role != core.RoleTool || conv.Messages[2].ToolCallID != "call_order_42" {
		t.Errorf("unexpected message 2 tool output: %+v", conv.Messages[2])
	}
	if conv.Messages[2].Parts[0].Content != `{"status":"shipped"}` {
		t.Errorf("unexpected tool output content: %v", conv.Messages[2].Parts[0].Content)
	}

	// Message 3: Assistant final reply
	if conv.Messages[3].Role != core.RoleAssistant || conv.Messages[3].Parts[0].Content != "Your order has been shipped." {
		t.Errorf("unexpected message 3 assistant reply: %+v", conv.Messages[3])
	}

	// Tools
	if len(conv.Tools) != 1 || conv.Tools[0].Function["name"] != "get_order" {
		t.Errorf("unexpected tools: %+v", conv.Tools)
	}
}

func TestFromResponseRequest_MultiModalParts(t *testing.T) {
	req := &OpenAIResponseRequest{
		Model: "gpt-4o",
		Input: []map[string]any{
			{
				"type": "message",
				"role": "user",
				"content": []any{
					map[string]any{
						"type": "input_text",
						"text": "Analyze this image and audio snippet.",
					},
					map[string]any{
						"type":      "input_image",
						"image_url": "https://example.com/chart.png",
					},
					map[string]any{
						"type": "input_audio",
						"input_audio": map[string]any{
							"data":   "UklGRigAAABXQVZFZm10IBIA...",
							"format": "wav",
						},
					},
				},
			},
		},
	}

	conv, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(conv.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(conv.Messages))
	}

	parts := conv.Messages[0].Parts
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts, got %d", len(parts))
	}

	if parts[0].Type != "text" || parts[0].Content != "Analyze this image and audio snippet." {
		t.Errorf("unexpected part 0: %+v", parts[0])
	}
	if parts[1].Type != "image" || parts[1].Content != "https://example.com/chart.png" {
		t.Errorf("unexpected part 1: %+v", parts[1])
	}
	if parts[2].Type != "audio" {
		t.Errorf("unexpected part 2 type: %s", parts[2].Type)
	}
}

func TestToResponseRequest_EquivalenceRoundTrip(t *testing.T) {
	temp := 0.5
	topP := 0.95
	maxTokens := 1024
	stream := true

	originalReq := &OpenAIResponseRequest{
		Model:              "gpt-4o",
		Instructions:       "You are an expert Vern architect.",
		PreviousResponseID: "resp_prev_999",
		Temperature:        &temp,
		TopP:               &topP,
		MaxOutputTokens:    &maxTokens,
		Stream:             &stream,
		Input: []ResponseItem{
			{
				Type:    "message",
				Role:    "user",
				Content: "Hello Vern!",
			},
			{
				Type:      "function_call",
				CallID:    "call_calc_1",
				Name:      "calculate",
				Arguments: `{"expr":"2+2"}`,
			},
			{
				Type:   "function_call_output",
				CallID: "call_calc_1",
				Output: "4",
			},
			{
				Type:    "message",
				Role:    "assistant",
				Content: "The answer is 4.",
			},
		},
		Tools: []core.ToolDefinition{
			{
				Type: "function",
				Function: map[string]any{
					"name":        "calculate",
					"description": "Perform arithmetic",
				},
			},
		},
	}

	// 1. FromResponseRequest -> core.Conversation
	conv, err := FromResponseRequest(originalReq)
	if err != nil {
		t.Fatalf("FromResponseRequest failed: %v", err)
	}

	// 2. ToResponseRequest -> OpenAIResponseRequest
	exportedReq, err := ToResponseRequest(conv, &core.ExportOptions{})
	if err != nil {
		t.Fatalf("ToResponseRequest failed: %v", err)
	}

	// Verify top-level request properties
	if exportedReq.Model != originalReq.Model {
		t.Errorf("expected model %s, got %s", originalReq.Model, exportedReq.Model)
	}
	if exportedReq.Instructions != originalReq.Instructions {
		t.Errorf("expected instructions %s, got %s", originalReq.Instructions, exportedReq.Instructions)
	}
	if exportedReq.PreviousResponseID != originalReq.PreviousResponseID {
		t.Errorf("expected previous_response_id %s, got %s", originalReq.PreviousResponseID, exportedReq.PreviousResponseID)
	}
	if *exportedReq.Temperature != *originalReq.Temperature {
		t.Errorf("expected temperature %v, got %v", *originalReq.Temperature, *exportedReq.Temperature)
	}
	if *exportedReq.TopP != *originalReq.TopP {
		t.Errorf("expected top_p %v, got %v", *originalReq.TopP, *exportedReq.TopP)
	}
	if *exportedReq.MaxOutputTokens != *originalReq.MaxOutputTokens {
		t.Errorf("expected max_output_tokens %v, got %v", *originalReq.MaxOutputTokens, *exportedReq.MaxOutputTokens)
	}

	// 3. Second round-trip: FromResponseRequest(exportedReq) -> conv2
	conv2, err := FromResponseRequest(exportedReq)
	if err != nil {
		t.Fatalf("second FromResponseRequest failed: %v", err)
	}

	if len(conv2.Messages) != len(conv.Messages) {
		t.Fatalf("expected %d messages in second round-trip, got %d", len(conv.Messages), len(conv2.Messages))
	}
	for i := range conv.Messages {
		if conv2.Messages[i].Role != conv.Messages[i].Role {
			t.Errorf("msg %d expected role %s, got %s", i, conv.Messages[i].Role, conv2.Messages[i].Role)
		}
		if len(conv2.Messages[i].ToolCalls) != len(conv.Messages[i].ToolCalls) {
			t.Errorf("msg %d tool calls length mismatch: %d vs %d", i, len(conv.Messages[i].ToolCalls), len(conv2.Messages[i].ToolCalls))
		}
	}
}

func TestToResponseRequest_JSONSerialization(t *testing.T) {
	conv := &core.Conversation{
		Model:  "gpt-4o-mini",
		System: []string{"System instruction block"},
		Messages: []core.Message{
			{
				Role: core.RoleUser,
				Parts: []core.Part{
					{Type: "text", Content: "Hello world"},
					{Type: "image", Content: "https://example.com/img.jpg"},
				},
			},
		},
	}

	req, err := ToResponseRequest(conv, &core.ExportOptions{})
	if err != nil {
		t.Fatalf("ToResponseRequest failed: %v", err)
	}

	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	if raw["model"] != "gpt-4o-mini" {
		t.Errorf("expected model gpt-4o-mini in JSON, got %v", raw["model"])
	}
	if raw["instructions"] != "System instruction block" {
		t.Errorf("expected instructions in JSON, got %v", raw["instructions"])
	}

	inputs, ok := raw["input"].([]any)
	if !ok || len(inputs) != 1 {
		t.Fatalf("expected 1 input item in JSON, got %+v", raw["input"])
	}
}

func TestValidationAndNilHandling(t *testing.T) {
	if _, err := FromResponseRequest(nil); err == nil {
		t.Error("expected error for nil OpenAIResponseRequest")
	}

	if _, err := ToResponseRequest(nil, nil); err == nil {
		t.Error("expected error for nil Conversation")
	}

	invalidReq := &OpenAIResponseRequest{
		Input: 12345, // invalid type
	}
	if _, err := FromResponseRequest(invalidReq); err == nil {
		t.Error("expected error for invalid input type")
	}
}
