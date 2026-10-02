package reasoner

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// The consolidator decides, per cluster, whether a ReasonStructured failure is
// final for that input (skip the episodes, advance the checkpoint) or
// transient (retry on the next pass). These tests pin which failures are which.

// chatServer serves OpenAI-style chat completions. reply returns the HTTP
// status and, for 200, the assistant message content.
func chatServer(t *testing.T, reply func(call int) (status int, content string)) (*OpenAI, *int32) {
	t.Helper()
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			http.NotFound(w, r)
			return
		}
		n := int(atomic.AddInt32(&calls, 1))
		status, content := reply(n)
		w.Header().Set("Content-Type", "application/json")
		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"error":{"message":"bad request","type":"invalid_request_error"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 0,
			"model":   "test-model",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": content},
			}},
		})
	}))
	t.Cleanup(srv.Close)

	o, err := NewOpenAI(srv.URL+"/v1", "test-key", "test-model")
	if err != nil {
		t.Fatalf("NewOpenAI: %v", err)
	}
	return o, &calls
}

// The prod shape: a 33-character probe episode, and a model that keeps
// paraphrasing it into words the episode does not contain.
func TestReasonStructured_GroundingFailureIsUnusableOutput(t *testing.T) {
	o, calls := chatServer(t, func(int) (int, string) {
		return http.StatusOK, `{"summary":"A connectivity verification was conducted successfully"}`
	})

	_, err := o.ReasonStructured(context.Background(), []string{"Connectivity probe B, 2026-08-25."})

	if err == nil {
		t.Fatal("want a grounding error, got nil")
	}
	if !errors.Is(err, ErrUnusableOutput) {
		t.Fatalf("grounding failure must wrap ErrUnusableOutput, got %v", err)
	}
	if !strings.Contains(err.Error(), "grounding validation") {
		t.Fatalf("error must stay greppable as %q, got %q", "grounding validation", err)
	}
	if got := atomic.LoadInt32(calls); got != 2 {
		t.Fatalf("want 2 attempts before giving up, got %d", got)
	}
}

func TestReasonStructured_BadJSONIsUnusableOutput(t *testing.T) {
	o, calls := chatServer(t, func(int) (int, string) { return http.StatusOK, "not json" })

	_, err := o.ReasonStructured(context.Background(), []string{"Connectivity probe B, 2026-08-25."})

	if !errors.Is(err, ErrUnusableOutput) {
		t.Fatalf("unparseable output must wrap ErrUnusableOutput, got %v", err)
	}
	if !strings.Contains(err.Error(), "parse json") {
		t.Fatalf("error must say what failed, got %q", err)
	}
	if got := atomic.LoadInt32(calls); got != 2 {
		t.Fatalf("want 2 attempts before giving up, got %d", got)
	}
}

// An HTTP/API error says nothing about the input: the same text may well work
// on the next pass. It must NOT be classified as unusable, or a provider
// outage would permanently skip every episode it touched.
func TestReasonStructured_HTTP400IsTransient(t *testing.T) {
	o, _ := chatServer(t, func(int) (int, string) { return http.StatusBadRequest, "" })

	_, err := o.ReasonStructured(context.Background(), []string{"Connectivity probe B, 2026-08-25."})

	if err == nil {
		t.Fatal("want an API error, got nil")
	}
	if errors.Is(err, ErrUnusableOutput) {
		t.Fatalf("an API error must not wrap ErrUnusableOutput, got %v", err)
	}
}

// Control: a grounded answer on the retry is a success, not an error.
func TestReasonStructured_RetrySucceeds(t *testing.T) {
	o, _ := chatServer(t, func(call int) (int, string) {
		if call == 1 {
			return http.StatusOK, "not json"
		}
		return http.StatusOK, `{"summary":"Connectivity probe B on 2026-08-25"}`
	})

	sf, err := o.ReasonStructured(context.Background(), []string{"Connectivity probe B, 2026-08-25."})

	if err != nil {
		t.Fatalf("want success on retry, got %v", err)
	}
	if sf == nil || sf.Summary == "" {
		t.Fatalf("want a summary, got %#v", sf)
	}
}
