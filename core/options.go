package core

// Renderer is called during translation to compose content and metadata.
type Renderer func(content any, meta map[string]any) (any, error)

// DefaultRenderer is a pass-through that returns the content unaltered.
func DefaultRenderer(content any, meta map[string]any) (any, error) {
	return content, nil
}

// ExportOptions controls translation parameters and customization hooks.
type ExportOptions struct {
	Model            string              // Override target model name
	Renderers        map[string]Renderer // Custom renderers keyed by part type (e.g. "text", "image")
	DefaultMaxTokens int                 // Fallback max tokens if not set on the conversation
}

// NewExportOptions instantiates a pointer to ExportOptions pre-populated with standard defaults.
func NewExportOptions() *ExportOptions {
	return &ExportOptions{
		Renderers:        make(map[string]Renderer),
		DefaultMaxTokens: 1000,
	}
}
