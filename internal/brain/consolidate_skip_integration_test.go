package brain

// Stage-1 outcomes that need the per-episode checkpoint's own vocabulary:
// reasoner.ErrUnusableOutput and the episodes_already_mined /
// episodes_skipped counters. The old-API tests are in
// consolidate_integration_test.go.

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/alash3al/stash/internal/reasoner"
)

// An extraction that is unusable for this input (bad JSON, or a summary that
// fails the grounding check — the prod probe episodes) will not get better on
// retry. The episode is skipped, the checkpoint moves past it, and it is never
// sent to the reasoner again.
func TestConsolidate_UnusableOutputIsSkippedAndCheckpointAdvances(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	mustEpisode(t, b, e, "/t", "episode A")
	mustEpisode(t, b, e, "/t", "Connectivity probe B/D, 2026-08-25.")
	c := mustEpisode(t, b, e, "/t", "episode C")
	r.script("episode A", summaryScript("A"))
	r.script("Connectivity probe B/D, 2026-08-25.", errorScript(
		fmt.Errorf("%w: grounding validation: field contains ungrounded words", reasoner.ErrUnusableOutput)))
	r.script("episode C", summaryScript("C"))

	res := mustConsolidate(t, b, nsID)

	if cp := episodeCheckpoint(t, b, nsID); cp != c {
		t.Errorf("checkpoint = %d, want %d (C: the skipped episode is final)", cp, c)
	}
	if res.EpisodesSkipped != 1 {
		t.Errorf("EpisodesSkipped = %d, want 1", res.EpisodesSkipped)
	}
	if res.FactsCreated != 2 {
		t.Errorf("FactsCreated = %d, want 2 (A and C)", res.FactsCreated)
	}
	// Still reported, so a skip is visible in the log, but tagged as a skip.
	if len(res.Errors) != 1 || !strings.Contains(res.Errors[0], "skipped episodes") {
		t.Errorf("errors = %q, want one 'skipped episodes' entry", res.Errors)
	}

	structuredBefore := r.callsTo("ReasonStructured")
	factsBefore := queryInt(t, b, "SELECT count(*) FROM facts")

	res2 := mustConsolidate(t, b, nsID)

	if n := r.callsTo("ReasonStructured") - structuredBefore; n != 0 {
		t.Errorf("pass 2 ReasonStructured calls = %d, want 0", n)
	}
	if res2.LLMCalls != 0 {
		t.Errorf("pass 2 LLMCalls = %d, want 0", res2.LLMCalls)
	}
	if n := queryInt(t, b, "SELECT count(*) FROM facts"); n != factsBefore {
		t.Errorf("pass 2 facts = %d, want %d (no new facts)", n, factsBefore)
	}
}

// The episodes_already_mined counter is what tells an operator that a
// namespace was replaying old episodes and is now skipping them.
func TestConsolidate_CountsAlreadyMinedEpisodes(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	mustEpisode(t, b, e, "/t", "episode A")
	mustEpisode(t, b, e, "/t", "episode B")
	mustEpisode(t, b, e, "/t", "episode C")
	r.script("episode A", summaryScript("A"))
	r.script("episode B", errorScript(errors.New("boom")))
	r.script("episode C", summaryScript("C"))

	res1 := mustConsolidate(t, b, nsID)
	res2 := mustConsolidate(t, b, nsID)

	if res1.EpisodesAlreadyMined != 0 || res1.EpisodesRead != 3 {
		t.Errorf("pass 1: read=%d already_mined=%d, want 3 and 0", res1.EpisodesRead, res1.EpisodesAlreadyMined)
	}
	// Pass 2 re-reads B (pinned) and C (above the checkpoint, already mined).
	if res2.EpisodesRead != 2 || res2.EpisodesAlreadyMined != 1 {
		t.Errorf("pass 2: read=%d already_mined=%d, want 2 and 1", res2.EpisodesRead, res2.EpisodesAlreadyMined)
	}
	if res2.EpisodesSkipped != 0 {
		t.Errorf("pass 2: EpisodesSkipped = %d, want 0 (B is transient, not skipped)", res2.EpisodesSkipped)
	}
}

// A cancelled pass leaves the remaining clusters not done, so the checkpoint
// cannot move past an episode that was never extracted.
func TestConsolidate_CancelledPassDoesNotSkipEpisodes(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	a := mustEpisode(t, b, e, "/t", "episode A")
	mustEpisode(t, b, e, "/t", "episode B")
	r.script("episode A", func(context.Context, int) (*reasoner.StructuredFact, error) {
		cancel() // the ticker's context ends while A is being extracted
		return &reasoner.StructuredFact{}, nil
	})
	r.script("episode B", summaryScript("B"))

	if _, err := b.ConsolidateByID(ctx, nsID); err != nil {
		t.Fatalf("ConsolidateByID: %v", err)
	}

	if n := r.structuredCallsFor("episode B"); n != 0 {
		t.Fatalf("B was extracted after cancellation (%d calls)", n)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != a {
		t.Errorf("checkpoint = %d, want %d (A done, B never attempted)", cp, a)
	}
}
