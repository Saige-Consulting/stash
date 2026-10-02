package main

import (
	"fmt"
	"strings"

	"github.com/alash3al/stash/internal/brain"
)

const (
	// consolidationLogMaxErrors bounds how many error lines one pass logs. The
	// summary line always carries the full count.
	consolidationLogMaxErrors = 3
	// consolidationLogMaxErrorLen bounds each logged error, in characters. A
	// reasoner error can embed a whole model response.
	consolidationLogMaxErrorLen = 300
)

// formatConsolidationLog renders one consolidation pass as log lines: a
// summary line, then up to consolidationLogMaxErrors lines of error text.
//
// The background ticker used to log only counts and drop result.Errors. A
// namespace whose stage-1 checkpoint was pinned by one persistently failing
// cluster therefore looked identical to a healthy one, and the same old
// episodes were re-mined every pass for months without a trace in the log.
func formatConsolidationLog(result brain.ConsolidationResult) []string {
	lines := []string{fmt.Sprintf(
		"Consolidation completed for %s: episodes_read=%d facts=%d deduped=%d relationships=%d errors=%d contradictions=%d auto_resolved=%d goals_annotated=%d failure_repeats=%d hypotheses_updated=%d llm_calls=%d duration=%s",
		result.Namespace,
		result.EpisodesRead,
		result.FactsCreated,
		result.FactsDeduplicated,
		result.RelationshipsFound,
		len(result.Errors),
		result.ContradictionsFound,
		result.ContradictionsAutoResolved,
		result.GoalsAnnotated,
		result.FailureRepeatsDetected,
		result.HypothesesUpdated,
		result.LLMCalls,
		result.Duration,
	)}

	for i, e := range result.Errors {
		if i == consolidationLogMaxErrors {
			break
		}
		lines = append(lines, fmt.Sprintf("Consolidation error for %s: %s",
			result.Namespace, truncateForLog(oneLine(e), consolidationLogMaxErrorLen)))
	}
	return lines
}

// oneLine folds line breaks into spaces so one error stays one log record.
func oneLine(s string) string {
	return strings.NewReplacer("\r\n", " ", "\n", " ", "\r", " ").Replace(s)
}

// truncateForLog cuts s to at most max characters, marking the cut with "...".
// It counts runes, so it never splits a multi-byte character.
func truncateForLog(s string, max int) string {
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	const marker = "..."
	if max <= len(marker) {
		return string(runes[:max])
	}
	return string(runes[:max-len(marker)]) + marker
}
