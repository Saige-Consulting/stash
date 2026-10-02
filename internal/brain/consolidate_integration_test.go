package brain

// Stage 1 (episodes -> facts) against a real database. Invariant under test:
// an episode takes part in at most one fact-creating extraction, and the
// checkpoint advances over every episode whose outcome is final, stopping just
// before the first one that failed transiently.
//
// Everything in this file uses only APIs that existed before the per-episode
// checkpoint, so it also runs against the old code: that is how the regression
// test below was shown to fail there. Tests that need the new result counters
// or reasoner.ErrUnusableOutput live in consolidate_skip_integration_test.go.

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/alash3al/stash/internal/reasoner"
)

// The prod bug. One cluster fails on every pass, so the old all-or-nothing
// checkpoint never moved and the clusters around it were re-mined every pass,
// each time into a fresh paraphrase fact.
func TestConsolidate_FailingClusterDoesNotRemineOthers(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	a := mustEpisode(t, b, e, "/t", "episode A")
	bb := mustEpisode(t, b, e, "/t", "episode B")
	c := mustEpisode(t, b, e, "/t", "episode C")
	if !(a < bb && bb < c) {
		t.Fatalf("episode ids must ascend, got %d %d %d", a, bb, c)
	}
	r.script("episode A", summaryScript("A"))
	r.script("episode B", errorScript(errors.New("boom")))
	r.script("episode C", summaryScript("C"))

	for i := 0; i < 3; i++ {
		mustConsolidate(t, b, nsID)
	}

	if n := factsSourcedFrom(t, b, a); n != 1 {
		t.Errorf("facts sourced from A = %d, want 1", n)
	}
	if n := factsSourcedFrom(t, b, c); n != 1 {
		t.Errorf("facts sourced from C = %d, want 1", n)
	}
	if n := r.structuredCallsFor("episode A"); n != 1 {
		t.Errorf("ReasonStructured calls for A = %d, want 1", n)
	}
	if n := r.structuredCallsFor("episode C"); n != 1 {
		t.Errorf("ReasonStructured calls for C = %d, want 1", n)
	}
	if n := r.structuredCallsFor("episode B"); n != 3 {
		t.Errorf("ReasonStructured calls for B = %d, want 3 (transient: retried every pass)", n)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != a {
		t.Errorf("checkpoint = %d, want %d (A: stops just before the failing B)", cp, a)
	}
}

// The shape of /threads/1783634494-027379 on prod: every episode already has a
// fact, but the stage-1 checkpoint is still 0. Later stages are caught up.
func TestConsolidate_StuckProdShape(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	var ids []int64
	for _, text := range []string{"episode 1", "episode 2", "episode 3"} {
		ids = append(ids, mustEpisode(t, b, e, "/t", text))
	}
	var maxFact int64
	for i, eid := range ids {
		vec := make([]float32, testVectorDim)
		vec[testVectorDim-1-i] = 1
		maxFact = mustFact(t, b, nsID, "old fact", vec, nil, nil, nil, eid)
	}
	// Stages 2+ are not behind on prod (measured: 0 namespaces); only stage 1 is.
	if _, err := b.pool.Exec(context.Background(),
		`INSERT INTO consolidation_progress (namespace_id, last_episode_id, last_fact_id, last_pattern_fact_id,
		   last_goal_progress_fact_id, last_hypothesis_fact_id)
		 VALUES ($1, 0, $2, $2, $2, $2)`, nsID, maxFact); err != nil {
		t.Fatalf("seed progress: %v", err)
	}
	factsBefore := queryInt(t, b, "SELECT count(*) FROM facts")

	res := mustConsolidate(t, b, nsID)

	if n := r.totalCalls(); n != 0 {
		t.Errorf("reasoner calls = %d (%v), want 0: every episode was already mined", n, r.calls)
	}
	if res.FactsCreated != 0 {
		t.Errorf("FactsCreated = %d, want 0", res.FactsCreated)
	}
	if n := queryInt(t, b, "SELECT count(*) FROM facts"); n != factsBefore {
		t.Errorf("facts = %d, want %d", n, factsBefore)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != ids[len(ids)-1] {
		t.Errorf("checkpoint = %d, want %d (the max episode id)", cp, ids[len(ids)-1])
	}
	if len(res.Errors) != 0 {
		t.Errorf("errors = %q, want none", res.Errors)
	}
}

// A summary that embeds onto an existing fact is a duplicate: no new fact, but
// the episode is linked to the fact it duplicates, so it reads as mined and is
// never sent to the reasoner again.
func TestConsolidate_DedupLinksEpisodeToExistingFact(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	v := make([]float32, testVectorDim)
	v[testVectorDim-1] = 1
	f := mustFact(t, b, nsID, "existing fact", v, nil, nil, nil)

	ep := mustEpisode(t, b, e, "/t", "episode E")
	r.script("episode E", func(context.Context, int) (*reasoner.StructuredFact, error) {
		return &reasoner.StructuredFact{Summary: "a paraphrase of the existing fact"}, nil
	})
	e.set("a paraphrase of the existing fact", v)

	res := mustConsolidate(t, b, nsID)

	if res.FactsCreated != 0 {
		t.Errorf("FactsCreated = %d, want 0", res.FactsCreated)
	}
	if res.FactsDeduplicated != 1 {
		t.Errorf("FactsDeduplicated = %d, want 1", res.FactsDeduplicated)
	}
	if n := queryInt(t, b, "SELECT count(*) FROM facts"); n != 1 {
		t.Errorf("facts = %d, want 1 (only the existing one)", n)
	}
	if n := queryInt(t, b, "SELECT count(*) FROM fact_sources WHERE fact_id = $1 AND episode_id = $2", f, ep); n != 1 {
		t.Errorf("fact_sources(%d, %d) rows = %d, want 1", f, ep, n)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != ep {
		t.Errorf("checkpoint = %d, want %d", cp, ep)
	}

	before := r.callsTo("ReasonStructured")
	mustConsolidate(t, b, nsID)
	if n := r.callsTo("ReasonStructured") - before; n != 0 {
		t.Errorf("pass 2 ReasonStructured calls = %d, want 0", n)
	}
}

// A fact whose fact_sources insert fails would look unmined and be re-mined
// forever. The fact and its sources must commit together or not at all.
func TestConsolidate_FactAndSourcesAreAtomic(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	a := mustEpisode(t, b, e, "/t", "episode A")
	ep := mustEpisode(t, b, e, "/t", "episode E")
	r.script("episode A", summaryScript("A"))
	r.script("episode E", func(ctx context.Context, _ int) (*reasoner.StructuredFact, error) {
		// Hard-delete E between extraction and insert, so the fact_sources
		// foreign key to episodes fails.
		if _, err := b.pool.Exec(ctx, "DELETE FROM episodes WHERE id = $1", ep); err != nil {
			t.Errorf("delete episode: %v", err)
		}
		return &reasoner.StructuredFact{Summary: "E summary"}, nil
	})

	res := mustConsolidate(t, b, nsID)

	if n := queryInt(t, b, "SELECT count(*) FROM facts WHERE content = 'E summary'"); n != 0 {
		t.Errorf("facts with the orphaned summary = %d, want 0 (rolled back with its sources)", n)
	}
	if n := factsSourcedFrom(t, b, a); n != 1 {
		t.Errorf("facts sourced from A = %d, want 1", n)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp >= ep {
		t.Errorf("checkpoint = %d, want < %d (E's outcome is not final)", cp, ep)
	}
	if len(res.Errors) == 0 {
		t.Error("the failed insert must be reported in result.Errors")
	}

	// E is gone, so the next pass finds nothing to retry and must not panic.
	mustConsolidate(t, b, nsID)
	if n := queryInt(t, b, "SELECT count(*) FROM facts WHERE content = 'E summary'"); n != 0 {
		t.Errorf("pass 2 facts with the orphaned summary = %d, want 0", n)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp > ep {
		t.Errorf("pass 2 checkpoint = %d, want <= %d", cp, ep)
	}
}

func TestConsolidate_ErrorsAreReported(t *testing.T) {
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

	res := mustConsolidate(t, b, nsID)

	if len(res.Errors) != 1 {
		t.Fatalf("errors = %q, want exactly 1", res.Errors)
	}
	if !strings.Contains(res.Errors[0], "boom") {
		t.Errorf("error must carry the cause, got %q", res.Errors[0])
	}
}

// fetch_episodes is shared with the failure-pattern stage, which scans its
// columns positionally. Adding a column for stage 1 must not break that stage.
func TestConsolidate_FailureStageStillReadsEpisodes(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	if _, err := b.CreateFailure(context.Background(), nsID, "deploy failed", "missing secret", "check secrets first", nil); err != nil {
		t.Fatalf("CreateFailure: %v", err)
	}
	mustEpisode(t, b, e, "/t", "episode A")
	r.script("episode A", summaryScript("A"))

	res := mustConsolidate(t, b, nsID)

	for _, msg := range res.Errors {
		if strings.Contains(msg, "failures") {
			t.Errorf("failure stage error: %q", msg)
		}
	}
	if n := r.callsTo("ReasonFailurePatterns"); n != 1 {
		t.Errorf("ReasonFailurePatterns calls = %d, want 1 (the stage must still see the episode)", n)
	}
}
