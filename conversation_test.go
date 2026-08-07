package conversation_test

import (
	"fmt"
	"strings"
	"testing"

	"codeberg.org/v-e-r-n/conversation"
	"codeberg.org/v-e-r-n/conversation/anthropic"
	"codeberg.org/v-e-r-n/conversation/gemini"
	"codeberg.org/v-e-r-n/conversation/openai"
)

func TestRendererFallbackAndSetMeta(t *testing.T) {
	// Create conversation manually using root type aliases
	c := &conversation.Conversation{
		Model: "gpt-4o",
	}
	c.AddUserMessage("Hello world!").Parts[0].SetMeta("emotion", "excited")

	// Create renderer for "text" but none for other modalities
	opts := conversation.NewExportOptions()
	opts.Renderers["text"] = func(content any, meta map[string]any) (any, error) {
		text := content.(string)
		if emotion, ok := meta["emotion"].(string); ok {
			return text + " [" + emotion + "]", nil
		}
		return text, nil
	}

	// Translate to OpenAI
	reqRaw, err := conversation.Translate(c, conversation.ProviderOpenAI, opts)
	if err != nil {
		t.Fatalf("Translate failed: %v", err)
	}

	req := reqRaw.(*openai.OpenAIRequest)
	if len(req.Messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(req.Messages))
	}

	// Verify text renderer successfully formatted content
	if req.Messages[0].Content != "Hello world! [excited]" {
		t.Errorf("Expected rendered text, got '%v'", req.Messages[0].Content)
	}
}

func TestUnifiedSystemInstructionCollapsingAndHelpers(t *testing.T) {
	c := &conversation.Conversation{
		Model: "target-model",
	}
	c.SetSystemPrompt("Root instruction")
	c.AppendSystemPrompt("Second system instruction")
	c.AppendSystemPrompt("Third system instruction")

	c.AddUserMessage("What is my task?")

	expectedSystemCombined := "Root instruction\n\nSecond system instruction\n\nThird system instruction"
	if c.CombinedSystemPrompt() != expectedSystemCombined {
		t.Errorf("CombinedSystemPrompt mismatched: '%s'", c.CombinedSystemPrompt())
	}

	opts := conversation.NewExportOptions()

	// 1. Verify OpenAI collapses system parts and prepends them
	openAIRaw, err := conversation.Translate(c, conversation.ProviderOpenAI, opts)
	if err != nil {
		t.Fatalf("Translate to OpenAI failed: %v", err)
	}
	openAIReq := openAIRaw.(*openai.OpenAIRequest)
	if len(openAIReq.Messages) != 2 {
		t.Fatalf("Expected 2 OpenAI messages, got %d", len(openAIReq.Messages))
	}
	if openAIRaw.(*openai.OpenAIRequest).Messages[0].Role != "system" || openAIReq.Messages[0].Content != expectedSystemCombined {
		t.Errorf("System instructions were not correctly collapsed for OpenAI. Got: %v", openAIReq.Messages[0].Content)
	}

	// 2. Verify Anthropic collapses them into req.System
	anthropicRaw, err := conversation.Translate(c, conversation.ProviderAnthropic, opts)
	if err != nil {
		t.Fatalf("Translate to Anthropic failed: %v", err)
	}
	anthropicReq := anthropicRaw.(*anthropic.AnthropicRequest)
	if anthropicReq.System != expectedSystemCombined {
		t.Errorf("Expected System field to be '%s', got '%s'", expectedSystemCombined, anthropicReq.System)
	}

	// 3. Verify Gemini collapses them into req.SystemInstruction
	geminiRaw, err := conversation.Translate(c, conversation.ProviderGemini, opts)
	if err != nil {
		t.Fatalf("Translate to Gemini failed: %v", err)
	}
	geminiReq := geminiRaw.(*gemini.GeminiRequest)
	if geminiReq.SystemInstruction == nil || len(geminiReq.SystemInstruction.Parts) == 0 || geminiReq.SystemInstruction.Parts[0].Text != expectedSystemCombined {
		t.Errorf("Expected Gemini system instruction text to be '%s'", expectedSystemCombined)
	}
}

func TestProviderImportEquivalence(t *testing.T) {
	temp := 0.7
	maxTokens := 500

	openAIReq := &openai.OpenAIRequest{
		Model: "shared-model-name",
		Messages: []openai.OpenAIMessage{
			{Role: "user", Content: "Hello world!"},
			{Role: "assistant", Content: "Hello indeed!"},
		},
		Temperature: &temp,
		MaxTokens:   &maxTokens,
	}

	anthropicReq := &anthropic.AnthropicRequest{
		Model: "shared-model-name",
		Messages: []anthropic.AnthropicMessage{
			{Role: "user", Content: "Hello world!"},
			{Role: "assistant", Content: "Hello indeed!"},
		},
		Temperature: &temp,
		MaxTokens:   maxTokens,
	}

	convOpenAI, err := conversation.FromOpenAI(openAIReq)
	if err != nil {
		t.Fatalf("FromOpenAI failed: %v", err)
	}

	convAnthropic, err := conversation.FromAnthropic(anthropicReq)
	if err != nil {
		t.Fatalf("FromAnthropic failed: %v", err)
	}

	if convOpenAI.Model != convAnthropic.Model {
		t.Errorf("Model mismatch: '%s' vs '%s'", convOpenAI.Model, convAnthropic.Model)
	}

	if len(convOpenAI.Messages) != len(convAnthropic.Messages) {
		t.Fatalf("Message count mismatch: %d vs %d", len(convOpenAI.Messages), len(convAnthropic.Messages))
	}

	for i := range convOpenAI.Messages {
		msgO := convOpenAI.Messages[i]
		msgA := convAnthropic.Messages[i]

		if msgO.Role != msgA.Role {
			t.Errorf("Message %d role mismatch: %s vs %s", i, msgO.Role, msgA.Role)
		}
		if len(msgO.Parts) != len(msgA.Parts) || msgO.Parts[0].Content != msgA.Parts[0].Content {
			t.Errorf("Message %d content mismatch", i)
		}
	}
}

func TestValidationAndErrorPropagation(t *testing.T) {
	c := &conversation.Conversation{
		Model: "some-model",
	}
	c.AddUserMessage("Hello!")

	// 1. Verify nil options returns an error
	_, err := conversation.Translate(c, conversation.ProviderOpenAI, nil)
	if err == nil {
		t.Error("Expected error when translating with nil options, got nil")
	}

	_, err = conversation.ToOpenAI(c, nil)
	if err == nil {
		t.Error("Expected error when calling ToOpenAI with nil options, got nil")
	}

	// 2. Verify custom renderer error propagation
	opts := conversation.NewExportOptions()
	opts.Renderers["text"] = func(content any, meta map[string]any) (any, error) {
		return nil, fmt.Errorf("simulated formatting failure")
	}

	_, err = conversation.Translate(c, conversation.ProviderOpenAI, opts)
	if err == nil || !strings.Contains(err.Error(), "simulated formatting failure") {
		t.Errorf("Expected simulated formatting error, got %v", err)
	}

	// 3. Verify invalid provider returns an error
	_, err = conversation.Translate(c, conversation.Provider("invalid-provider"), opts)
	if err == nil || !strings.Contains(err.Error(), "unsupported provider") {
		t.Errorf("Expected unsupported provider error, got %v", err)
	}
}
