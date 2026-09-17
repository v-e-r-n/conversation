package gemini_test

import (
	"testing"

	"github.com/v-e-r-n/conversation/core"
	"github.com/v-e-r-n/conversation/gemini"
)

func TestGeminiToRequest(t *testing.T) {
	c := &core.Conversation{
		Model: "gemini-1.5",
	}
	c.SetSystemPrompt("Instruction")
	c.AddUserMessage("User query")
	c.AddAssistantMessage("Assistant reply")

	temp := 0.2
	c.Parameters.Temperature = &temp

	opts := core.NewExportOptions()
	req, err := gemini.ToRequest(c, opts)
	if err != nil {
		t.Fatalf("ToRequest failed: %v", err)
	}

	// 1. Verify system instruction mapping
	if req.SystemInstruction == nil || len(req.SystemInstruction.Parts) == 0 || req.SystemInstruction.Parts[0].Text != "Instruction" {
		t.Errorf("System instruction mismatched: %+v", req.SystemInstruction)
	}

	// 2. Verify assistant maps to model role
	if len(req.Contents) != 2 {
		t.Fatalf("Expected 2 contents entries, got %d", len(req.Contents))
	}
	if req.Contents[1].Role != "model" || req.Contents[1].Parts[0].Text != "Assistant reply" {
		t.Errorf("Assistant to model role mapping failed: %+v", req.Contents[1])
	}

	// 3. Verify parameters mapping
	if req.GenerationConfig == nil || *req.GenerationConfig.Temperature != 0.2 {
		t.Errorf("Expected temperature parameter 0.2, got %+v", req.GenerationConfig)
	}
}

func TestGeminiToolCalling(t *testing.T) {
	c := &core.Conversation{
		Model: "gemini-1.5-pro",
		Tools: []core.ToolDefinition{
			{
				Type: "function",
				Function: map[string]any{
					"name":        "lookup",
					"description": "Lookup ID",
					"parameters":  map[string]any{"type": "object"},
				},
			},
		},
		Messages: []core.Message{
			{
				Role:  core.RoleUser,
				Parts: []core.Part{{Type: "text", Content: "Lookup 42"}},
			},
			{
				Role: core.RoleAssistant,
				ToolCalls: []core.ToolCall{
					{
						ID:   "call_99",
						Type: "function",
						Function: core.ToolFunction{
							Name:      "lookup",
							Arguments: `{"id":42}`,
						},
					},
				},
			},
			{
				Role:  core.RoleTool,
				Name:  "lookup",
				Parts: []core.Part{{Type: "text", Content: `{"result":"found"}`}},
			},
		},
	}

	opts := core.NewExportOptions()
	req, err := gemini.ToRequest(c, opts)
	if err != nil {
		t.Fatalf("ToRequest failed: %v", err)
	}

	if len(req.Tools) != 1 {
		t.Fatalf("Expected 1 tool declaration, got %d", len(req.Tools))
	}
	if len(req.Contents) != 3 {
		t.Fatalf("Expected 3 contents entries, got %d", len(req.Contents))
	}

	// Message 2: Model with FunctionCall
	modelTurn := req.Contents[1]
	if modelTurn.Role != "model" || len(modelTurn.Parts) != 1 {
		t.Fatalf("Unexpected model turn: %+v", modelTurn)
	}
	if modelTurn.Parts[0].FunctionCall == nil || modelTurn.Parts[0].FunctionCall["name"] != "lookup" {
		t.Errorf("Expected functionCall 'lookup', got %+v", modelTurn.Parts[0].FunctionCall)
	}

	// Message 3: Function with FunctionResponse
	fnTurn := req.Contents[2]
	if fnTurn.Role != "function" || len(fnTurn.Parts) != 1 {
		t.Fatalf("Unexpected function turn: %+v", fnTurn)
	}
	if fnTurn.Parts[0].FunctionResponse == nil || fnTurn.Parts[0].FunctionResponse["name"] != "lookup" {
		t.Errorf("Expected functionResponse 'lookup', got %+v", fnTurn.Parts[0].FunctionResponse)
	}
}
