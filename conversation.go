package conversation

import (
	"fmt"

	"github.com/v-e-r-n/conversation/anthropic"
	"github.com/v-e-r-n/conversation/core"
	"github.com/v-e-r-n/conversation/gemini"
	"github.com/v-e-r-n/conversation/openai"
)

// Expose core types to consumer via Go type aliases
type Conversation = core.Conversation
type Message = core.Message
type Part = core.Part
type ExportOptions = core.ExportOptions
type Renderer = core.Renderer
type Provider = core.Provider

// Expose Role constants
const (
	RoleSystem    = core.RoleSystem
	RoleUser      = core.RoleUser
	RoleAssistant = core.RoleAssistant
	RoleTool      = core.RoleTool
)

// Expose Provider constants
const (
	ProviderOpenAI    = core.ProviderOpenAI
	ProviderAnthropic = core.ProviderAnthropic
	ProviderGemini    = core.ProviderGemini
)

// Expose constructors
func NewExportOptions() *ExportOptions {
	return core.NewExportOptions()
}

// Translate dynamic router maps a provider enum to its corresponding sub-package converter.
func Translate(c *Conversation, provider Provider, opts *ExportOptions) (any, error) {
	if opts == nil {
		return nil, fmt.Errorf("export options are required and must be non-nil")
	}
	switch provider {
	case ProviderOpenAI:
		return ToOpenAI(c, opts)
	case ProviderAnthropic:
		return ToAnthropic(c, opts)
	case ProviderGemini:
		return ToGemini(c, opts)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}
}

// ToOpenAI wraps the openai sub-package converter.
func ToOpenAI(c *Conversation, opts *ExportOptions) (*openai.OpenAIRequest, error) {
	if opts == nil {
		return nil, fmt.Errorf("export options are required and must be non-nil")
	}
	return openai.ToRequest(c, opts)
}

// FromOpenAI wraps the openai sub-package parser.
func FromOpenAI(req *openai.OpenAIRequest) (*Conversation, error) {
	return openai.FromRequest(req)
}

// ToAnthropic wraps the anthropic sub-package converter.
func ToAnthropic(c *Conversation, opts *ExportOptions) (*anthropic.AnthropicRequest, error) {
	if opts == nil {
		return nil, fmt.Errorf("export options are required and must be non-nil")
	}
	return anthropic.ToRequest(c, opts)
}

// FromAnthropic wraps the anthropic sub-package parser.
func FromAnthropic(req *anthropic.AnthropicRequest) (*Conversation, error) {
	return anthropic.FromRequest(req)
}

// ToGemini wraps the gemini sub-package converter.
func ToGemini(c *Conversation, opts *ExportOptions) (*gemini.GeminiRequest, error) {
	if opts == nil {
		return nil, fmt.Errorf("export options are required and must be non-nil")
	}
	return gemini.ToRequest(c, opts)
}
