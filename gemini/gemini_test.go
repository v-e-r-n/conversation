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
