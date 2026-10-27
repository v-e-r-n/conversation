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

	// Verify previous_response_id is preserved in conversation metadata
	if prevID, ok := conv.GetMeta("previous_response_id").(string); !ok || prevID != "resp_init_001" {
		t.Errorf("expected previous_response_id 'resp_init_001' in conversation metadata, got %v", conv.Meta)
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
		Tools: []ResponseTool{
			{
				Type:        "function",
				Name:        "get_order",
				Description: "Look up an order by ID",
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
		Tools: []ResponseTool{
			{
				Type:        "function",
				Name:        "calculate",
				Description: "Perform arithmetic",
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
	if exportedReq.MaxOutputTokens != nil && *exportedReq.MaxOutputTokens != *originalReq.MaxOutputTokens {
		t.Errorf("expected max_output_tokens %v, got %v", *originalReq.MaxOutputTokens, *exportedReq.MaxOutputTokens)
	}
	if exportedReq.Stream == nil || *exportedReq.Stream != *originalReq.Stream {
		t.Errorf("expected stream %v, got %v", *originalReq.Stream, exportedReq.Stream)
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

func TestRenderResponseParts_NoMetaLeakage(t *testing.T) {
	part := core.Part{
		Type:    "text",
		Content: "Hello world with secret metadata",
	}
	part.SetMeta("secret_tracing_id", "trace_xyz_789")
	part.SetMeta("previous_response_id", "resp_secret_123")

	conv := &core.Conversation{
		Model: "gpt-4o",
		Messages: []core.Message{
			{
				Role:  core.RoleUser,
				Parts: []core.Part{part},
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

	// Ensure no "meta" key leaked into JSON payload
	inputs, ok := raw["input"].([]any)
	if !ok || len(inputs) == 0 {
		t.Fatalf("expected input array, got %v", raw["input"])
	}

	for _, item := range inputs {
		itemMap, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if _, exists := itemMap["meta"]; exists {
			t.Errorf("critical schema leakage: found 'meta' key in ResponseItem: %v", itemMap)
		}
		if contentArr, ok := itemMap["content"].([]any); ok {
			for _, partObj := range contentArr {
				if partMap, ok := partObj.(map[string]any); ok {
					if _, exists := partMap["meta"]; exists {
						t.Errorf("critical schema leakage: found 'meta' key in content part: %v", partMap)
					}
				}
			}
		}
	}
}

func TestToResponseRequest_ToolOutputStructuredJSON(t *testing.T) {
	structuredData := map[string]any{
		"status":      "ok",
		"temperature": 72,
	}

	conv := &core.Conversation{
		Model: "gpt-4o",
		Messages: []core.Message{
			{
				Role:       core.RoleTool,
				ToolCallID: "call_weather_1",
				Parts: []core.Part{
					{
						Type:    "text",
						Content: structuredData,
					},
				},
			},
		},
	}

	req, err := ToResponseRequest(conv, &core.ExportOptions{})
	if err != nil {
		t.Fatalf("ToResponseRequest failed: %v", err)
	}

	items, ok := req.Input.([]ResponseItem)
	if !ok || len(items) != 1 {
		t.Fatalf("expected 1 ResponseItem in input, got %v", req.Input)
	}

	item := items[0]
	if item.Type != "function_call_output" {
		t.Errorf("expected type function_call_output, got %s", item.Type)
	}

	outputStr, ok := item.Output.(string)
	if !ok {
		t.Fatalf("expected string output, got %T: %v", item.Output, item.Output)
	}

	// Must be parseable JSON, NOT Go debug string "map[status:ok temperature:72]"
	var parsed map[string]any
	if err := json.Unmarshal([]byte(outputStr), &parsed); err != nil {
		t.Fatalf("tool output is not valid JSON (%q): %v", outputStr, err)
	}
	if parsed["status"] != "ok" || parsed["temperature"] != float64(72) {
		t.Errorf("unexpected parsed tool output content: %+v", parsed)
	}
}

func TestPreviousResponseID_And_Metadata_PreservedOnEmptyMessages(t *testing.T) {
	stream := true
	req := &OpenAIResponseRequest{
		Model:              "gpt-4o",
		Instructions:       "You are an AI assistant.",
		PreviousResponseID: "resp_empty_turn_1",
		Stream:             &stream,
		Metadata: map[string]any{
			"org_id":     "org_12345",
			"project_id": "proj_abcde",
		},
		Input: nil, // zero messages
	}

	conv, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("FromResponseRequest failed: %v", err)
	}

	if len(conv.Messages) != 0 {
		t.Fatalf("expected 0 messages, got %d", len(conv.Messages))
	}

	// Re-export
	exported, err := ToResponseRequest(conv, nil)
	if err != nil {
		t.Fatalf("ToResponseRequest failed: %v", err)
	}

	if exported.PreviousResponseID != "resp_empty_turn_1" {
		t.Errorf("expected PreviousResponseID 'resp_empty_turn_1', got %q", exported.PreviousResponseID)
	}
	if exported.Stream == nil || *exported.Stream != true {
		t.Errorf("expected Stream true, got %v", exported.Stream)
	}
	if exported.Metadata == nil || exported.Metadata["org_id"] != "org_12345" {
		t.Errorf("expected Metadata with org_id 'org_12345', got %v", exported.Metadata)
	}
	if exported.Instructions != "You are an AI assistant." {
		t.Errorf("expected Instructions preserved, got %q", exported.Instructions)
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

func TestResponseTool_FlatJSONSerialization(t *testing.T) {
	strict := true
	tools := []ResponseTool{
		{
			Type:        "function",
			Name:        "get_weather",
			Description: "Get the current weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string"},
				},
				"required": []any{"location"},
			},
			Strict: &strict,
		},
		{
			Type: "web_search",
		},
	}

	req := &OpenAIResponseRequest{
		Model: "gpt-4o",
		Tools: tools,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	rawTools, ok := raw["tools"].([]any)
	if !ok || len(rawTools) != 2 {
		t.Fatalf("expected 2 tools in JSON, got %v", raw["tools"])
	}

	tool0 := rawTools[0].(map[string]any)
	if _, hasNestedFn := tool0["function"]; hasNestedFn {
		t.Errorf("critical schema failure: tool definition contains nested 'function' key: %v", tool0)
	}
	if tool0["type"] != "function" || tool0["name"] != "get_weather" {
		t.Errorf("unexpected tool 0 attributes: %v", tool0)
	}
	if tool0["strict"] != true {
		t.Errorf("expected strict true, got %v", tool0["strict"])
	}

	tool1 := rawTools[1].(map[string]any)
	if tool1["type"] != "web_search" {
		t.Errorf("unexpected built-in tool type: %v", tool1)
	}
	if _, hasName := tool1["name"]; hasName {
		t.Errorf("built-in tool should omit empty name: %v", tool1)
	}

	// Test unmarshaling of legacy nested format as fallback
	legacyJSON := `[
		{
			"type": "function",
			"function": {
				"name": "legacy_lookup",
				"description": "Legacy format description",
				"parameters": {"type": "object"}
			}
		}
	]`
	var unmarshaledTools []ResponseTool
	if err := json.Unmarshal([]byte(legacyJSON), &unmarshaledTools); err != nil {
		t.Fatalf("failed to unmarshal legacy nested tools: %v", err)
	}
	if len(unmarshaledTools) != 1 || unmarshaledTools[0].Name != "legacy_lookup" {
		t.Fatalf("unexpected unmarshaled legacy tool: %+v", unmarshaledTools)
	}
}

func TestAssistantTurnCohesion_TextAndFunctionCall(t *testing.T) {
	// 1. Export: assistant message with text + tool call -> emits message followed by function_call
	conv := &core.Conversation{
		Model: "gpt-4o",
		Messages: []core.Message{
			{
				Role: core.RoleAssistant,
				Parts: []core.Part{
					{Type: "text", Content: "Let me check that order for you."},
				},
				ToolCalls: []core.ToolCall{
					{
						ID: "call_check_99",
						Function: core.ToolFunction{
							Name:      "lookup_order",
							Arguments: `{"id":99}`,
						},
					},
				},
			},
		},
	}

	req, err := ToResponseRequest(conv, nil)
	if err != nil {
		t.Fatalf("ToResponseRequest failed: %v", err)
	}

	items, ok := req.Input.([]ResponseItem)
	if !ok || len(items) != 2 {
		t.Fatalf("expected 2 response items, got %v", req.Input)
	}

	// Must preserve chronological order: message precedes function_call
	if items[0].Type != "message" || items[0].Role != "assistant" {
		t.Errorf("expected first item to be assistant message, got %+v", items[0])
	}
	if items[1].Type != "function_call" || items[1].CallID != "call_check_99" {
		t.Errorf("expected second item to be function_call, got %+v", items[1])
	}

	// 2. Import: message followed by function_call -> collapsed back into single assistant turn
	roundTripConv, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("FromResponseRequest failed: %v", err)
	}

	if len(roundTripConv.Messages) != 1 {
		t.Fatalf("expected 1 consolidated assistant message, got %d", len(roundTripConv.Messages))
	}
	resMsg := roundTripConv.Messages[0]
	if resMsg.Role != core.RoleAssistant {
		t.Errorf("expected role assistant, got %s", resMsg.Role)
	}
	if len(resMsg.Parts) != 1 || resMsg.Parts[0].Content != "Let me check that order for you." {
		t.Errorf("unexpected parts in merged message: %+v", resMsg.Parts)
	}
	if len(resMsg.ToolCalls) != 1 || resMsg.ToolCalls[0].ID != "call_check_99" {
		t.Errorf("unexpected tool calls in merged message: %+v", resMsg.ToolCalls)
	}
}

func TestDeveloperRole_CollapsesIntoSystem(t *testing.T) {
	req := &OpenAIResponseRequest{
		Model:        "gpt-4o",
		Instructions: "Base system prompt.",
		Input: []ResponseItem{
			{
				Type:    "message",
				Role:    "developer",
				Content: "Developer instructions: keep answers brief.",
			},
			{
				Type:    "message",
				Role:    "user",
				Content: "Hello!",
			},
		},
	}

	conv, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("FromResponseRequest failed: %v", err)
	}

	// System should have both Base system prompt and Developer instructions
	if len(conv.System) != 2 {
		t.Fatalf("expected 2 system prompts, got %d: %v", len(conv.System), conv.System)
	}
	if conv.System[0] != "Base system prompt." || conv.System[1] != "Developer instructions: keep answers brief." {
		t.Errorf("unexpected system slice contents: %v", conv.System)
	}

	// Messages should only contain the user message (developer role collapsed into System)
	if len(conv.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(conv.Messages))
	}
	if conv.Messages[0].Role != core.RoleUser {
		t.Errorf("expected user role, got %s", conv.Messages[0].Role)
	}
}

func TestCallIDFallback_ToolCallID(t *testing.T) {
	req := &OpenAIResponseRequest{
		Model: "gpt-4o",
		Input: []ResponseItem{
			{
				Type:       "function_call",
				ToolCallID: "tc_legacy_123", // CallID is empty, ToolCallID populated
				Name:       "ping",
				Arguments:  `{}`,
			},
			{
				Type:       "function_call_output",
				ToolCallID: "tc_legacy_123", // CallID is empty, ToolCallID populated
				Output:     "pong",
			},
		},
	}

	conv, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("FromResponseRequest failed: %v", err)
	}

	if len(conv.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(conv.Messages))
	}

	// Verify function_call ID fallback
	if len(conv.Messages[0].ToolCalls) != 1 || conv.Messages[0].ToolCalls[0].ID != "tc_legacy_123" {
		t.Errorf("expected tool call ID 'tc_legacy_123', got %+v", conv.Messages[0].ToolCalls)
	}

	// Verify function_call_output ToolCallID fallback
	if conv.Messages[1].ToolCallID != "tc_legacy_123" {
		t.Errorf("expected tool output ToolCallID 'tc_legacy_123', got %q", conv.Messages[1].ToolCallID)
	}
}

func TestEmptyToolArguments_DefaultsToEmptyJSON(t *testing.T) {
	// 1. FromResponseRequest defaults empty arguments to "{}"
	req := &OpenAIResponseRequest{
		Model: "gpt-4o",
		Input: []ResponseItem{
			{
				Type:      "function_call",
				CallID:    "call_no_args",
				Name:      "get_current_time",
				Arguments: "", // empty
			},
		},
	}

	conv, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("FromResponseRequest failed: %v", err)
	}

	if len(conv.Messages) != 1 || len(conv.Messages[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 message with 1 tool call, got %+v", conv.Messages)
	}
	if conv.Messages[0].ToolCalls[0].Function.Arguments != "{}" {
		t.Errorf("expected default '{}' arguments, got %q", conv.Messages[0].ToolCalls[0].Function.Arguments)
	}

	// 2. ToResponseRequest defaults empty arguments to "{}"
	convEmptyArgs := &core.Conversation{
		Model: "gpt-4o",
		Messages: []core.Message{
			{
				Role: core.RoleAssistant,
				ToolCalls: []core.ToolCall{
					{
						ID: "call_empty_args",
						Function: core.ToolFunction{
							Name:      "ping",
							Arguments: "", // empty
						},
					},
				},
			},
		},
	}

	exportedReq, err := ToResponseRequest(convEmptyArgs, nil)
	if err != nil {
		t.Fatalf("ToResponseRequest failed: %v", err)
	}

	items, ok := exportedReq.Input.([]ResponseItem)
	if !ok || len(items) != 1 {
		t.Fatalf("expected 1 ResponseItem, got %v", exportedReq.Input)
	}
	if items[0].Arguments != "{}" {
		t.Errorf("expected exported Arguments '{}', got %q", items[0].Arguments)
	}
}

func TestMultiModal_InputAudioExport(t *testing.T) {
	audioPayload := map[string]any{
		"data":   "UklGRigAAABXQVZFZm10IBIA...",
		"format": "wav",
	}

	conv := &core.Conversation{
		Model: "gpt-4o",
		Messages: []core.Message{
			{
				Role: core.RoleUser,
				Parts: []core.Part{
					{
						Type:    "audio",
						Content: audioPayload,
					},
				},
			},
		},
	}

	req, err := ToResponseRequest(conv, nil)
	if err != nil {
		t.Fatalf("ToResponseRequest failed: %v", err)
	}

	items, ok := req.Input.([]ResponseItem)
	if !ok || len(items) != 1 {
		t.Fatalf("expected 1 ResponseItem, got %v", req.Input)
	}

	contentParts, ok := items[0].Content.([]map[string]any)
	if !ok || len(contentParts) != 1 {
		t.Fatalf("expected 1 content part map, got %v", items[0].Content)
	}

	part := contentParts[0]
	if part["type"] != "input_audio" {
		t.Errorf("expected part type input_audio, got %v", part["type"])
	}
	audioObj, ok := part["input_audio"].(map[string]any)
	if !ok || audioObj["format"] != "wav" {
		t.Errorf("expected input_audio map with format 'wav', got %v", part["input_audio"])
	}

	// Test FromResponseRequest parsing back
	conv2, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("FromResponseRequest failed: %v", err)
	}
	if len(conv2.Messages) != 1 || len(conv2.Messages[0].Parts) != 1 {
		t.Fatalf("expected 1 message with 1 part, got %+v", conv2.Messages)
	}
	if conv2.Messages[0].Parts[0].Type != "audio" {
		t.Errorf("expected part type audio, got %s", conv2.Messages[0].Parts[0].Type)
	}
}

func TestStore_ParameterRoundTripAndExport(t *testing.T) {
	storeFalse := false
	req := &OpenAIResponseRequest{
		Model: "gpt-4o",
		Input: "Testing store false",
		Store: &storeFalse,
	}

	conv, err := FromResponseRequest(req)
	if err != nil {
		t.Fatalf("FromResponseRequest failed: %v", err)
	}

	if val, ok := conv.GetMeta("store").(bool); !ok || val != false {
		t.Errorf("expected store=false in conversation meta, got %v", conv.GetMeta("store"))
	}

	// 1. Export without options (uses meta)
	exported, err := ToResponseRequest(conv, nil)
	if err != nil {
		t.Fatalf("ToResponseRequest failed: %v", err)
	}
	if exported.Store == nil || *exported.Store != false {
		t.Errorf("expected exported Store false, got %v", exported.Store)
	}

	// Verify JSON serialization includes "store": false
	b, err := json.Marshal(exported)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if raw["store"] != false {
		t.Errorf("expected JSON payload to contain store: false, got %v", raw["store"])
	}

	// 2. Export with override in ExportOptions
	storeTrue := true
	exportedWithOverride, err := ToResponseRequest(conv, &core.ExportOptions{
		Store: &storeTrue,
	})
	if err != nil {
		t.Fatalf("ToResponseRequest with override failed: %v", err)
	}
	if exportedWithOverride.Store == nil || *exportedWithOverride.Store != true {
		t.Errorf("expected overridden Store true, got %v", exportedWithOverride.Store)
	}
}
