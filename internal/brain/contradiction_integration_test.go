package brain

// Contradiction detection against a real database. Two facts extracted from
// the same episode are two readings of one source, not two observations: a
// difference between them is paraphrase drift, never a change in the world.
// On prod, episode #835's re-extractions were the old side of 439
// auto/superseded and 25 structured contradiction rows.

import (
	"context"
	"testing"

	"github.com/alash3al/stash/internal/models"
	"github.com/alash3al/stash/internal/reasoner"
)

// replacementReasoner classifies every pair as a confident replacement, the
// answer that auto-supersedes the older fact.
func replacementReasoner(t *testing.T) *fakeReasoner {
	r := newFakeReasoner(t)
	r.contradiction = func(_, _, _, _ string) (*reasoner.ContradictionResult, error) {
		return &reasoner.ContradictionResult{Classification: reasoner.ClassificationReplacement, Confidence: 0.95}, nil
	}
	return r
}

func basis(i int) []float32 {
	v := make([]float32, testVectorDim)
	v[i] = 1
	return v
}

func detect(t *testing.T, b *Brain, nsID, factID int64, entity, property, value string) (detected, autoResolved int) {
	t.Helper()
	d, a, err := b.DetectContradictions(context.Background(), nsID, &models.Fact{
		ID: factID, NamespaceID: nsID, Entity: strp(entity), Property: strp(property), Value: strp(value),
	})
	if err != nil {
		t.Fatalf("DetectContradictions: %v", err)
	}
	return d, a
}

func validUntilIsNull(t *testing.T, b *Brain, factID int64) bool {
	t.Helper()
	return queryInt(t, b, "SELECT count(*) FROM facts WHERE id = $1 AND valid_until IS NULL", factID) == 1
}

func TestDetectContradictions_SameSourceEpisodeIsNotCompared(t *testing.T) {
	r := replacementReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	ep := mustEpisode(t, b, e, "/t", "Opened PR #375 for the NSB fix")
	x := mustFact(t, b, nsID, "PR is #375", basis(6), strp("NSB fix"), strp("pull_request"), strp("PR #375"), ep)
	y := mustFact(t, b, nsID, "PR is https://example/pr/375", basis(7), strp("NSB fix"), strp("pull_request"), strp("https://example/pr/375"), ep)

	detected, autoResolved := detect(t, b, nsID, y, "NSB fix", "pull_request", "https://example/pr/375")

	if n := r.callsTo("ReasonContradiction"); n != 0 {
		t.Errorf("ReasonContradiction calls = %d, want 0 (same source episode)", n)
	}
	if detected != 0 || autoResolved != 0 {
		t.Errorf("detected=%d autoResolved=%d, want 0 and 0", detected, autoResolved)
	}
	if n := queryInt(t, b, "SELECT count(*) FROM contradictions"); n != 0 {
		t.Errorf("contradictions rows = %d, want 0", n)
	}
	if !validUntilIsNull(t, b, x) {
		t.Error("X was superseded by a re-extraction of its own episode")
	}
}

// Control: the guard must not blind detection to genuinely separate sources.
func TestDetectContradictions_DifferentEpisodesStillCompared(t *testing.T) {
	r := replacementReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	e1 := mustEpisode(t, b, e, "/t", "Ticket TR-1 is open")
	e2 := mustEpisode(t, b, e, "/t", "Ticket TR-1 is closed")
	x := mustFact(t, b, nsID, "TR-1 is open", basis(6), strp("TR-1"), strp("status"), strp("open"), e1)
	y := mustFact(t, b, nsID, "TR-1 is closed", basis(7), strp("TR-1"), strp("status"), strp("closed"), e2)

	detected, autoResolved := detect(t, b, nsID, y, "TR-1", "status", "closed")

	if n := r.callsTo("ReasonContradiction"); n != 1 {
		t.Errorf("ReasonContradiction calls = %d, want 1", n)
	}
	if detected != 1 || autoResolved != 1 {
		t.Errorf("detected=%d autoResolved=%d, want 1 and 1", detected, autoResolved)
	}
	if n := queryInt(t, b,
		"SELECT count(*) FROM contradictions WHERE old_fact_id = $1 AND new_fact_id = $2 AND method = 'auto' AND resolution = 'superseded'",
		x, y); n != 1 {
		t.Errorf("auto/superseded contradiction rows for (%d -> %d) = %d, want 1", x, y, n)
	}
	if validUntilIsNull(t, b, x) {
		t.Error("X should be superseded by a fact from a different episode")
	}
}

// Only the candidates that share a source with the new fact drop out; the
// others are still compared in the same call.
func TestDetectContradictions_OnlySameSourceCandidatesAreExcluded(t *testing.T) {
	r := replacementReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	shared := mustEpisode(t, b, e, "/t", "Owner of TR-2 is Ana")
	other := mustEpisode(t, b, e, "/t", "Owner of TR-2 is Ben")
	sameSource := mustFact(t, b, nsID, "TR-2 owner Ana", basis(5), strp("TR-2"), strp("owner"), strp("Ana"), shared)
	otherSource := mustFact(t, b, nsID, "TR-2 owner Ben", basis(6), strp("TR-2"), strp("owner"), strp("Ben"), other)
	newFact := mustFact(t, b, nsID, "TR-2 owner is Ana B.", basis(7), strp("TR-2"), strp("owner"), strp("Ana B."), shared)

	detected, _ := detect(t, b, nsID, newFact, "TR-2", "owner", "Ana B.")

	if n := r.callsTo("ReasonContradiction"); n != 1 {
		t.Errorf("ReasonContradiction calls = %d, want 1 (only the other-source candidate)", n)
	}
	if detected != 1 {
		t.Errorf("detected = %d, want 1", detected)
	}
	if !validUntilIsNull(t, b, sameSource) {
		t.Error("the same-source candidate was superseded")
	}
	if validUntilIsNull(t, b, otherSource) {
		t.Error("the other-source candidate should still be superseded")
	}
}

// A fact extracted from a multi-episode cluster shares a source with a
// candidate if ANY of their episodes match.
func TestDetectContradictions_AnySharedEpisodeExcludes(t *testing.T) {
	r := replacementReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	e1 := mustEpisode(t, b, e, "/t", "Region of NSB-7 is north")
	e2 := mustEpisode(t, b, e, "/t", "NSB-7 moved region")
	e3 := mustEpisode(t, b, e, "/t", "NSB-7 region confirmed")
	x := mustFact(t, b, nsID, "NSB-7 region north", basis(6), strp("NSB-7"), strp("region"), strp("north"), e1, e2)
	y := mustFact(t, b, nsID, "NSB-7 region is North", basis(7), strp("NSB-7"), strp("region"), strp("North"), e2, e3)

	detect(t, b, nsID, y, "NSB-7", "region", "North")

	if n := r.callsTo("ReasonContradiction"); n != 0 {
		t.Errorf("ReasonContradiction calls = %d, want 0 (episode %d is shared)", n, e2)
	}
	if !validUntilIsNull(t, b, x) {
		t.Error("X was superseded by a fact sharing one of its episodes")
	}
}
