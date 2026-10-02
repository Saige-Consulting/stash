package main

import (
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/alash3al/stash/internal/brain"
)

// The background ticker used to log only counts and drop result.Errors, so a
// namespace whose checkpoint was pinned by one failing cluster looked healthy
// in the log for months. These pin that the error count and the error text
// both reach the log.

func TestFormatConsolidationLog_IncludesErrorCount(t *testing.T) {
	result := brain.ConsolidationResult{Namespace: "/ns"}
	for i := 0; i < 5; i++ {
		result.Errors = append(result.Errors, fmt.Sprintf("reason structured: boom %d", i))
	}

	lines := formatConsolidationLog(result)

	if len(lines) != 4 {
		t.Fatalf("want 4 lines (1 summary + 3 errors), got %d: %q", len(lines), lines)
	}
	if !strings.Contains(lines[0], "errors=5") {
		t.Fatalf("summary line must carry errors=5, got %q", lines[0])
	}
	for i, l := range lines[1:] {
		if !strings.HasPrefix(l, "Consolidation error for /ns:") {
			t.Fatalf("line %d must start with the error prefix, got %q", i+1, l)
		}
		if !strings.Contains(l, fmt.Sprintf("boom %d", i)) {
			t.Fatalf("line %d must carry error %d's text, got %q", i+1, i, l)
		}
	}
}

func TestFormatConsolidationLog_TruncatesLongErrors(t *testing.T) {
	result := brain.ConsolidationResult{
		Namespace: "/ns",
		Errors:    []string{strings.Repeat("x", 5000)},
	}

	lines := formatConsolidationLog(result)

	if len(lines) != 2 {
		t.Fatalf("want 2 lines, got %d", len(lines))
	}
	const prefix = "Consolidation error for /ns: "
	if !strings.HasPrefix(lines[1], prefix) {
		t.Fatalf("error line must start with %q, got %q", prefix, lines[1][:60])
	}
	if n := len(lines[1]); n > len(prefix)+300 {
		t.Fatalf("error line is %d bytes, want <= %d (prefix + 300)", n, len(prefix)+300)
	}
	if n := utf8.RuneCountInString(lines[1]); n > utf8.RuneCountInString(prefix)+300 {
		t.Fatalf("error line is %d characters, want <= prefix + 300", n)
	}
}

func TestFormatConsolidationLog_NoErrors(t *testing.T) {
	lines := formatConsolidationLog(brain.ConsolidationResult{Namespace: "/ns", FactsCreated: 2})

	if len(lines) != 1 {
		t.Fatalf("want exactly 1 line, got %d: %q", len(lines), lines)
	}
	if !strings.Contains(lines[0], "errors=0") {
		t.Fatalf("summary line must carry errors=0, got %q", lines[0])
	}
	if !strings.HasPrefix(lines[0], "Consolidation completed for /ns:") {
		t.Fatalf("summary line prefix changed, got %q", lines[0])
	}
}

// A multi-line error (a wrapped SQL error, a model response) must not split
// one log record into several lines that no longer carry the namespace.
func TestFormatConsolidationLog_ErrorStaysOnOneLine(t *testing.T) {
	lines := formatConsolidationLog(brain.ConsolidationResult{
		Namespace: "/ns",
		Errors:    []string{"first\nsecond\r\nthird"},
	})
	if len(lines) != 2 {
		t.Fatalf("want 2 lines, got %d", len(lines))
	}
	if strings.ContainsAny(lines[1], "\r\n") {
		t.Fatalf("error line must not contain line breaks, got %q", lines[1])
	}
}

// The per-episode checkpoint's counters are how an operator sees a namespace
// replaying already-mined episodes or giving up on unusable ones.
func TestFormatConsolidationLog_CarriesStageOneCounters(t *testing.T) {
	lines := formatConsolidationLog(brain.ConsolidationResult{
		Namespace:            "/ns",
		EpisodesRead:         7,
		EpisodesAlreadyMined: 4,
		FactsCreated:         2,
		FactsDeduplicated:    3,
		EpisodesSkipped:      1,
	})
	for _, want := range []string{"episodes_read=7", "already_mined=4", "facts=2", "deduped=3", "skipped=1"} {
		if !strings.Contains(lines[0], want) {
			t.Errorf("summary line must contain %q, got %q", want, lines[0])
		}
	}
}
