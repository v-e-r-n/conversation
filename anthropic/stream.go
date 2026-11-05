package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// StreamEventType defines the standard Anthropic SSE event types.
type StreamEventType string

const (
	EventMessageStart      StreamEventType = "message_start"
	EventContentBlockStart StreamEventType = "content_block_start"
	EventContentBlockDelta StreamEventType = "content_block_delta"
	EventContentBlockStop  StreamEventType = "content_block_stop"
	EventMessageDelta      StreamEventType = "message_delta"
	EventMessageStop       StreamEventType = "message_stop"
	EventPing              StreamEventType = "ping"
	EventError             StreamEventType = "error"
)

// MessageStartPayload represents the initial message metadata in a message_start event.
type MessageStartPayload struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"` // "message"
	Role         string         `json:"role"` // "assistant"
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"` // Empty initially
	StopReason   *string        `json:"stop_reason"`
	StopSequence *string        `json:"stop_sequence"`
	Usage        Usage          `json:"usage"`
}

// MessageStartEvent is emitted when a new message stream begins.
type MessageStartEvent struct {
	Type    StreamEventType     `json:"type"` // "message_start"
	Message MessageStartPayload `json:"message"`
}

// ContentBlockStartEvent is emitted when a content block (text, thinking, or tool_use) begins.
type ContentBlockStartEvent struct {
	Type         StreamEventType `json:"type"` // "content_block_start"
	Index        int             `json:"index"`
	ContentBlock ContentBlock    `json:"content_block"`
}

// DeltaType represents the discriminator type of a content_block_delta.
type DeltaType string

const (
	DeltaTypeText      DeltaType = "text_delta"
	DeltaTypeThinking  DeltaType = "thinking_delta"
	DeltaTypeSignature DeltaType = "signature_delta"
	DeltaTypeInputJSON DeltaType = "input_json_delta"
)

// ContentBlockDelta represents the incremental chunk in a content_block_delta event.
type ContentBlockDelta struct {
	Type        DeltaType `json:"type"`
	Text        string    `json:"text,omitempty"`
	Thinking    string    `json:"thinking,omitempty"`
	Signature   string    `json:"signature,omitempty"`
	PartialJSON string    `json:"partial_json,omitempty"`
}

// ContentBlockDeltaEvent is emitted for each incremental token/chunk.
type ContentBlockDeltaEvent struct {
	Type  StreamEventType   `json:"type"` // "content_block_delta"
	Index int               `json:"index"`
	Delta ContentBlockDelta `json:"delta"`
}

// ContentBlockStopEvent is emitted when a content block completes.
type ContentBlockStopEvent struct {
	Type  StreamEventType `json:"type"` // "content_block_stop"
	Index int             `json:"index"`
}

// MessageDeltaPayload represents top-level updates (stop reason, usage) at message completion.
type MessageDeltaPayload struct {
	StopReason   *string `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

// MessageDeltaUsage represents token usage emitted in message_delta events.
type MessageDeltaUsage struct {
	OutputTokens int `json:"output_tokens"`
}

// MessageDeltaEvent is emitted before the stream ends with stop reason and final token usage.
type MessageDeltaEvent struct {
	Type  StreamEventType     `json:"type"` // "message_delta"
	Delta MessageDeltaPayload `json:"delta"`
	Usage *MessageDeltaUsage  `json:"usage,omitempty"`
}

// MessageStopEvent is the final event indicating the message stream has completed.
type MessageStopEvent struct {
	Type StreamEventType `json:"type"` // "message_stop"
}

// PingEvent is an occasional heartbeat event.
type PingEvent struct {
	Type StreamEventType `json:"type"` // "ping"
}

// FormatSSE formats an event type and data payload into standard SSE wire format.
// e.g. "event: message_start\ndata: {...}\n\n"
func FormatSSE(eventType StreamEventType, data any) ([]byte, error) {
	b, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal SSE data: %w", err)
	}

	var buf bytes.Buffer
	buf.WriteString("event: ")
	buf.WriteString(string(eventType))
	buf.WriteString("\ndata: ")
	buf.Write(b)
	buf.WriteString("\n\n")

	return buf.Bytes(), nil
}

// RawSSEEvent represents a parsed SSE frame.
type RawSSEEvent struct {
	Event string
	Data  []byte
}

// ParseSSE parses an individual raw SSE frame into its event name and data payload.
func ParseSSE(raw []byte) (*RawSSEEvent, error) {
	lines := strings.Split(string(raw), "\n")
	ev := &RawSSEEvent{}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "event:") {
			ev.Event = strings.TrimSpace(strings.TrimPrefix(trimmed, "event:"))
		} else if strings.HasPrefix(trimmed, "data:") {
			dataStr := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
			ev.Data = []byte(dataStr)
		}
	}

	if ev.Event == "" && len(ev.Data) == 0 {
		return nil, fmt.Errorf("empty or invalid SSE frame")
	}

	return ev, nil
}
