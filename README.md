# Conversation Intermediate Representation (IR)

`conversation` is a generalized Intermediate Representation (IR) library for chat conversations. It abstracts the differences in API payloads and message structures between major LLM providers (OpenAI, Anthropic, Gemini) into a unified, rich-media-capable Go data structure.

---

## The Core Concept

Different LLM providers expect chat history in different formats:
1. **OpenAI**: Expects system instructions inside the `messages` array under the `"system"` role.
2. **Anthropic (Claude)**: Requires system instructions in a root-level `"system"` parameter, and the `"messages"` array must strictly alternate between `"user"` and `"assistant"` roles.
3. **Google Gemini**: Expects system instructions in a root `"systemInstruction"` object, and message history inside a `"contents"` array using `"user"` and `"model"` roles.

This library acts as an **Intermediate Representation (IR)**. It decouples the core conversation history and system prompts from provider-specific schemas, allowing you to seamlessly translate a single conversation structure to any provider.

---

## Data Models

```go
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Part represents a structured segment of message content (supports multimodal data).
type Part struct {
	Type    string         `json:"type"`              // e.g., "text", "image"
	Content any            `json:"content,omitempty"` // Content string or structured data
	Meta    map[string]any `json:"meta,omitempty"`    // Optional tracking metadata
}

// Message represents a single chat turn containing one or more parts.
type Message struct {
	Role  Role   `json:"role"`
	Parts []Part `json:"parts"`
}

type Parameters struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

type Conversation struct {
	Model      string     `json:"model"`
	System     []string   `json:"system,omitempty"` // Unified system instructions (text-only)
	Messages   []Message  `json:"messages"`         // Interactive chat turns
	Parameters Parameters `json:"parameters,omitempty"`
}
```

---

## Usage

### 1. Ingesting from OpenAI Request
Parse an incoming OpenAI-compatible payload into the generalized IR. This automatically extracts system prompts from the message history:
```go
import (
	"encoding/json"
	"net/http"
	"github.com/v-e-r-n/conversation"
	"github.com/v-e-r-n/conversation/openai"
)

func handleChat(w http.ResponseWriter, r *http.Request) {
	var req openai.OpenAIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Parse to generalized Conversation IR
	conv, err := conversation.FromOpenAI(&req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Access conversation elements
	modelName := conv.Model
	systemPrompt := conv.CombinedSystemPrompt()
}
```

### 2. Exporting to OpenAI
Reconstruct an OpenAI-compatible request payload (re-merging the system prompts back into the messages list):
```go
import "github.com/v-e-r-n/conversation"

// Setup export options
opts := conversation.NewExportOptions()
opts.Model = "gpt-4o"

openaiReq, err := conversation.ToOpenAI(conv, opts)
```

### 3. Exporting to Anthropic Claude
Format the conversation IR into a payload suitable for the Anthropic Messages API:
```go
import "github.com/v-e-r-n/conversation"

opts := conversation.NewExportOptions()
opts.Model = "claude-3-5-sonnet-latest"

anthropicReq, err := conversation.ToAnthropic(conv, opts)
```

### 4. Exporting to Google Gemini
Format the conversation IR into a payload suitable for the Gemini API:
```go
import "github.com/v-e-r-n/conversation"

opts := conversation.NewExportOptions()
opts.Model = "gemini-2.5-flash"

geminiReq, err := conversation.ToGemini(conv, opts)
```

---

## Custom Renderers

Since message content is represented by a list of `Part`s, you can register custom renderers to format specific part types (e.g., custom markdown tags, files, or attachments) during translation to upstream payloads:

```go
import "github.com/v-e-r-n/conversation"

opts := conversation.NewExportOptions()

// Add custom renderer for a custom part type "file_attachment"
opts.Renderers["file_attachment"] = func(content any, meta map[string]any) (any, error) {
	// Format or fetch attachment content dynamically...
	return fmt.Sprintf("[Attachment: %v]", content), nil
}
```

---

## Testing

Run the test suite:
```bash
go test -v ./...
```
