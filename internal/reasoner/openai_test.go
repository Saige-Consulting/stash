package reasoner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"
)

// The consolidator decides, per cluster, whether a ReasonStructured failure is
// final for that input (skip the episodes, advance the checkpoint) or
// transient (retry on the next pass). These tests pin which failures are which.

// chatServer serves OpenAI-style chat completions. reply returns the HTTP
// status and, for 200, the assistant message content; for any other status,
// the raw error body (a generic invalid_request_error when empty). Error
// responses carry x-should-retry: false so the SDK answers immediately.
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
			if content == "" {
				content = `{"error":{"message":"bad request","type":"invalid_request_error"}}`
			}
			w.Header().Set("x-should-retry", "false")
			w.WriteHeader(status)
			_, _ = w.Write([]byte(content))
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
	// A plain 400 is about the request, so it is neither an outage (which must
	// never count against an episode) nor a rejection of the input as such.
	if IsUnavailable(err) || errors.Is(err, ErrInputRejected) {
		t.Fatalf("a plain 400 must count against the input, got %v", err)
	}
}

// The provider refusing THIS input (too long, content policy) is permanent for
// that text, but a smaller input may work: callers split a multi-episode
// cluster and give up only on a single episode.
func TestReasonStructured_RejectedInputCodes(t *testing.T) {
	for _, code := range []string{"context_length_exceeded", "content_filter", "string_above_max_length"} {
		t.Run(code, func(t *testing.T) {
			body := `{"error":{"message":"refused","type":"invalid_request_error","param":"messages","code":"` + code + `"}}`
			o, _ := chatServer(t, func(int) (int, string) { return http.StatusBadRequest, body })

			_, err := o.ReasonStructured(context.Background(), []string{"x"})

			if !errors.Is(err, ErrInputRejected) {
				t.Fatalf("code %q must wrap ErrInputRejected, got %v", code, err)
			}
			if IsUnavailable(err) || errors.Is(err, ErrUnusableOutput) {
				t.Fatalf("code %q must be neither an outage nor unusable output, got %v", code, err)
			}
		})
	}
}

// An outage, an auth or quota refusal or a rate limit says nothing about the
// input. It must be recognisable as such, or a long outage would make the
// consolidator give up on every episode it touched.
func TestReasonStructured_ProviderErrorsAreUnavailable(t *testing.T) {
	cases := []struct {
		status int
		body   string
	}{
		{http.StatusUnauthorized, `{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}`},
		{http.StatusForbidden, ""},
		{http.StatusNotFound, `{"error":{"message":"model not found","type":"invalid_request_error","code":"model_not_found"}}`},
		{http.StatusTooManyRequests, `{"error":{"message":"slow down","type":"requests","code":"rate_limit_exceeded"}}`},
		{http.StatusInternalServerError, ""},
		{http.StatusServiceUnavailable, "<html>upstream unavailable</html>"},
	}
	for _, tc := range cases {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			o, _ := chatServer(t, func(int) (int, string) { return tc.status, tc.body })

			_, err := o.ReasonStructured(context.Background(), []string{"x"})

			if !errors.Is(err, ErrUnavailable) || !IsUnavailable(err) {
				t.Fatalf("HTTP %d must be unavailable, got %v", tc.status, err)
			}
			if errors.Is(err, ErrInputRejected) || errors.Is(err, ErrUnusableOutput) {
				t.Fatalf("HTTP %d must not blame the input, got %v", tc.status, err)
			}
		})
	}
}

// IsUnavailable also classifies raw errors from other openai-go clients (the
// embedder) and from the transport, which the reasoner never saw.
func TestIsUnavailable(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"plain", errors.New("reasoner: no response from LLM"), false},
		{"sentinel", fmt.Errorf("x: %w", ErrUnavailable), true},
		{"cancelled", fmt.Errorf("x: %w", context.Canceled), true},
		{"deadline", context.DeadlineExceeded, true},
		{"network", &net.OpError{Op: "dial", Err: errors.New("connection refused")}, true},
		{"api 400", &openai.Error{StatusCode: 400}, false},
		{"api 413", &openai.Error{StatusCode: 413}, false},
		{"api 422", &openai.Error{StatusCode: 422}, false},
		{"api 401", &openai.Error{StatusCode: 401}, true},
		{"api 404", &openai.Error{StatusCode: 404}, true},
		{"api 429", &openai.Error{StatusCode: 429}, true},
		{"api 502", &openai.Error{StatusCode: 502}, true},
	}
	for _, tc := range cases {
		if got := IsUnavailable(tc.err); got != tc.want {
			t.Errorf("IsUnavailable(%s) = %v, want %v", tc.name, got, tc.want)
		}
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
