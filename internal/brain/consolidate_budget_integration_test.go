package brain

// Stage-1 outcomes that must survive the pass that produced them (review I1,
// I2, M2):
//
//   - a final outcome that creates no fact (nothing to extract, unusable
//     output, input rejected) is recorded, so the episode reads as mined and is
//     never extracted again, even while something below it pins the checkpoint;
//   - an episode that keeps failing is given up on, but only after
//     maxEpisodeAttempts counted failures AND episodeGiveUpAfter since the first
//     one, and an outage never counts, so it cannot make episodes be skipped;
//   - a cluster the provider rejects as too large is split, not skipped.

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/alash3al/stash/internal/reasoner"
)

func unusable() error {
	return fmt.Errorf("%w: grounding validation: field contains ungrounded words", reasoner.ErrUnusableOutput)
}

func rejected() error {
	return fmt.Errorf("%w: chat.completions call failed: 400 context_length_exceeded", reasoner.ErrInputRejected)
}

func emptyScript() structuredScript {
	return func(context.Context, int) (*reasoner.StructuredFact, error) { return &reasoner.StructuredFact{}, nil }
}

// The reviewer's measurement: with P pinned, U (unusable) and Z (empty) were
// re-extracted on every pass (P=5 U=5 Z=5 D=1 over 5 passes), and every pass
// re-reported U's skip.
func TestConsolidate_FinalOutcomesAreNotReExtracted(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	mustEpisode(t, b, e, "/t", "P")
	u := mustEpisode(t, b, e, "/t", "U")
	z := mustEpisode(t, b, e, "/t", "Z")
	mustEpisode(t, b, e, "/t", "D")
	r.script("P", errorScript(errors.New("boom")))
	r.script("U", errorScript(unusable()))
	r.script("Z", emptyScript())
	r.script("D", summaryScript("D"))

	for pass := 1; pass <= 5; pass++ {
		res := mustConsolidate(t, b, nsID)
		if pass == 1 {
			continue
		}
		if len(res.Errors) != 1 || !strings.Contains(res.Errors[0], "boom") {
			t.Errorf("pass %d errors = %q, want only P's failure", pass, res.Errors)
		}
		if res.EpisodesSkipped != 0 {
			t.Errorf("pass %d EpisodesSkipped = %d, want 0 (U was skipped once, in pass 1)", pass, res.EpisodesSkipped)
		}
	}

	for text, want := range map[string]int{"P": 5, "U": 1, "Z": 1, "D": 1} {
		if n := r.structuredCallsFor(text); n != want {
			t.Errorf("ReasonStructured calls for %s = %d, want %d", text, n, want)
		}
	}
	if _, o, _ := episodeState(t, b, u); o != outcomeUnusable {
		t.Errorf("U outcome = %q, want %q", o, outcomeUnusable)
	}
	if _, o, _ := episodeState(t, b, z); o != outcomeEmpty {
		t.Errorf("Z outcome = %q, want %q", o, outcomeEmpty)
	}
}

// The reviewer's starvation case. With BatchSize 3 and P failing on every
// pass, the batch is always [P X Y], so N, written after them, is never
// fetched. P is given up on once it has failed maxEpisodeAttempts times and
// episodeGiveUpAfter has passed since its first failure; then N is mined.
func TestConsolidate_GivesUpAfterAttemptsAndTimeFloor(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	cfg := DefaultConfig()
	cfg.BatchSize = 3
	b := openTestBrainWithConfig(t, r, e, cfg)
	nsID := mustNamespace(t, b, "/t")

	p := mustEpisode(t, b, e, "/t", "P")
	mustEpisode(t, b, e, "/t", "X")
	y := mustEpisode(t, b, e, "/t", "Y")
	r.script("P", errorScript(errors.New("400 bad request")))
	r.script("X", summaryScript("X"))
	r.script("Y", summaryScript("Y"))

	mustConsolidate(t, b, nsID)
	n := mustEpisode(t, b, e, "/t", "N")
	r.script("N", summaryScript("N"))
	mustConsolidate(t, b, nsID)
	mustConsolidate(t, b, nsID)

	// Three counted failures, but within the time floor: still pinned.
	if a, o, _ := episodeState(t, b, p); a != maxEpisodeAttempts || o != "" {
		t.Fatalf("P after 3 passes: attempts=%d outcome=%q, want %d and pending", a, o, maxEpisodeAttempts)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != 0 {
		t.Fatalf("checkpoint = %d, want 0 while P is within the time floor", cp)
	}
	if c := r.structuredCallsFor("N"); c != 0 {
		t.Fatalf("N extracted %d times while outside the batch window", c)
	}

	ageFailures(t, b, episodeGiveUpAfter+time.Hour)
	res4 := mustConsolidate(t, b, nsID)

	if res4.EpisodesGaveUp != 1 {
		t.Errorf("pass 4 EpisodesGaveUp = %d, want 1", res4.EpisodesGaveUp)
	}
	if len(res4.Errors) != 1 || !strings.Contains(res4.Errors[0], "gave up on episodes") {
		t.Errorf("pass 4 errors = %q, want one 'gave up on episodes' entry", res4.Errors)
	}
	if _, o, _ := episodeState(t, b, p); o != outcomeGaveUp {
		t.Errorf("P outcome = %q, want %q", o, outcomeGaveUp)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != y {
		t.Errorf("checkpoint after giving up = %d, want %d (Y)", cp, y)
	}

	mustConsolidate(t, b, nsID)

	if c := r.structuredCallsFor("N"); c != 1 {
		t.Errorf("N extracted %d times, want 1 once the window moved", c)
	}
	if f := factsSourcedFrom(t, b, n); f != 1 {
		t.Errorf("facts from N = %d, want 1", f)
	}
	if c := r.structuredCallsFor("P"); c != 4 {
		t.Errorf("P extracted %d times, want 4 (never again after giving up)", c)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != n {
		t.Errorf("checkpoint = %d, want %d (N)", cp, n)
	}
}

// Both floors are required: an old first failure alone is not enough.
func TestConsolidate_GiveUpNeedsEnoughAttempts(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	p := mustEpisode(t, b, e, "/t", "P")
	r.script("P", errorScript(errors.New("boom")))

	mustConsolidate(t, b, nsID)
	ageFailures(t, b, episodeGiveUpAfter+time.Hour)
	mustConsolidate(t, b, nsID)

	if a, o, _ := episodeState(t, b, p); a != 2 || o != "" {
		t.Fatalf("after 2 attempts: attempts=%d outcome=%q, want 2 and pending", a, o)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != 0 {
		t.Fatalf("checkpoint = %d, want 0 with only 2 attempts", cp)
	}

	mustConsolidate(t, b, nsID)

	if _, o, _ := episodeState(t, b, p); o != outcomeGaveUp {
		t.Errorf("after 3 attempts over the time floor: outcome = %q, want %q", o, outcomeGaveUp)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != p {
		t.Errorf("checkpoint = %d, want %d", cp, p)
	}
}

// An outage, a bad key or a cancelled call says nothing about the episode. It
// must never count, however long it lasts, or a weekend outage would make the
// consolidator give up on every episode it touched.
func TestConsolidate_OutagesNeverCountAgainstEpisodes(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	p := mustEpisode(t, b, e, "/t", "P")
	q := mustEpisode(t, b, e, "/t", "Q")
	r.script("P", errorScript(fmt.Errorf("%w: chat.completions call failed: 503", reasoner.ErrUnavailable)))
	r.script("Q", errorScript(fmt.Errorf("chat.completions call failed: %w", context.DeadlineExceeded)))

	for pass := 0; pass < maxEpisodeAttempts+2; pass++ {
		mustConsolidate(t, b, nsID)
		ageFailures(t, b, episodeGiveUpAfter+time.Hour)
	}

	for _, id := range []int64{p, q} {
		if a, o, ok := episodeState(t, b, id); ok && (a != 0 || o != "") {
			t.Errorf("episode %d: attempts=%d outcome=%q, want no counted attempts", id, a, o)
		}
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != 0 {
		t.Errorf("checkpoint = %d, want 0 (nothing is final during an outage)", cp)
	}
}

// A cluster too large for the model is split into single episodes. Only the
// episode that is rejected on its own is skipped.
func TestConsolidate_RejectedClusterIsSplit(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	g := mustEpisode(t, b, e, "/t", "good")
	big := mustEpisodeLike(t, b, e, "/t", "huge", "good")
	r.cluster = func(texts []string) (*reasoner.StructuredFact, error) { return nil, rejected() }
	r.script("good", summaryScript("good"))
	r.script("huge", errorScript(rejected()))

	res := mustConsolidate(t, b, nsID)

	if n := r.callsTo("ReasonStructured"); n != 3 {
		t.Errorf("ReasonStructured calls = %d, want 3 (the cluster, then each episode)", n)
	}
	if f := factsSourcedFrom(t, b, g); f != 1 {
		t.Errorf("facts from the good episode = %d, want 1", f)
	}
	if _, o, _ := episodeState(t, b, big); o != outcomeRejected {
		t.Errorf("rejected episode outcome = %q, want %q", o, outcomeRejected)
	}
	if res.EpisodesSkipped != 1 {
		t.Errorf("EpisodesSkipped = %d, want 1", res.EpisodesSkipped)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != big {
		t.Errorf("checkpoint = %d, want %d", cp, big)
	}

	before := r.callsTo("ReasonStructured")
	mustConsolidate(t, b, nsID)
	if n := r.callsTo("ReasonStructured") - before; n != 0 {
		t.Errorf("pass 2 ReasonStructured calls = %d, want 0", n)
	}
}

// episodes.embedding and embedding_model are nullable, and the failure stage
// itself writes REPEAT FAILURE episodes with neither. pgvector.Vector rejects
// NULL, so such an episode used to fail the scan and block its namespace.
func TestConsolidate_NullEmbeddingEpisodeIsMined(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	if _, err := b.CreateFailure(context.Background(), nsID, "deploy failed", "missing secret", "check secrets first", nil); err != nil {
		t.Fatalf("CreateFailure: %v", err)
	}
	ep := mustRawEpisode(t, b, nsID, "REPEAT FAILURE [failure #1]: secret missing again")
	r.script("REPEAT FAILURE [failure #1]: secret missing again", summaryScript("repeat"))

	res := mustConsolidate(t, b, nsID)

	if len(res.Errors) != 0 {
		t.Errorf("errors = %q, want none", res.Errors)
	}
	if f := factsSourcedFrom(t, b, ep); f != 1 {
		t.Errorf("facts from the NULL-embedding episode = %d, want 1", f)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != ep {
		t.Errorf("checkpoint = %d, want %d", cp, ep)
	}
	if n := r.callsTo("ReasonFailurePatterns"); n != 1 {
		t.Errorf("ReasonFailurePatterns calls = %d, want 1 (the failure stage reads it too)", n)
	}
}
