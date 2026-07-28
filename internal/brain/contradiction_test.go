package brain

import "testing"

// Contradiction detection compares a new fact against every existing fact
// sharing its (entity, property) and spends one LLM call per pair. Properties
// that record an occurrence rather than state must therefore be excluded before
// that loop: a second occurrence never invalidates the first, so every pair is
// both a wasted reasoner call and a junk contradiction row.
func TestIsEventProperty(t *testing.T) {
	events := []string{
		"asked", "asks", "answered", "commented", "mentioned", "message",
		"messaged", "noted", "posted", "question", "replied", "reply",
		"report", "reported", "request", "requested", "said", "stated", "told",
	}
	for _, p := range events {
		if !isEventProperty(p) {
			t.Errorf("isEventProperty(%q) = false, want true", p)
		}
	}

	// State-shaped properties must still be checked for contradictions — these
	// are the cases the feature exists for.
	state := []string{
		"status", "assignee", "division", "role", "owner", "state",
		"apVoucher", "confidence", "", "asked_at_length",
	}
	for _, p := range state {
		if isEventProperty(p) {
			t.Errorf("isEventProperty(%q) = true, want false", p)
		}
	}
}

// Properties arrive from LLM extraction, so casing and stray whitespace are not
// guaranteed. "Asked" and " asked " must be excluded exactly like "asked".
func TestIsEventPropertyNormalises(t *testing.T) {
	for _, p := range []string{"Asked", "ASKED", " asked", "asked ", "  Requested  "} {
		if !isEventProperty(p) {
			t.Errorf("isEventProperty(%q) = false, want true (normalisation)", p)
		}
	}
}
