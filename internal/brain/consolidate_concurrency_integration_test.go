package brain

// Two passes over the same namespace at once (review M1). On prod the
// consolidate tool on mcp-stash and the stash-consolidator ticker can both run
// a namespace. Without serialisation both mine the same episode into a fact,
// and the same-source contradiction guard then keeps both copies current.

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/alash3al/stash/internal/reasoner"
)

// Deterministic: a concurrent pass commits a fact for E while this pass is
// extracting E. The re-check under the lock must see it and drop this pass's
// copy.
func TestConsolidate_ConcurrentlyMinedEpisodeIsNotMinedTwice(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")

	ep := mustEpisode(t, b, e, "/t", "episode E")
	r.script("episode E", func(context.Context, int) (*reasoner.StructuredFact, error) {
		mustFact(t, b, nsID, "the other pass's fact", basis(7), nil, nil, nil, ep)
		return &reasoner.StructuredFact{Summary: "this pass's fact"}, nil
	})

	res := mustConsolidate(t, b, nsID)

	if n := factsSourcedFrom(t, b, ep); n != 1 {
		t.Errorf("facts sourced from E = %d, want 1 (the concurrent pass's)", n)
	}
	if n := queryInt(t, b, "SELECT count(*) FROM facts WHERE content = 'this pass''s fact'"); n != 0 {
		t.Errorf("this pass's duplicate fact exists (%d rows), want it rolled back", n)
	}
	if res.FactsCreated != 0 {
		t.Errorf("FactsCreated = %d, want 0", res.FactsCreated)
	}
	if len(res.Errors) != 0 {
		t.Errorf("errors = %q, want none: losing the race is not a failure", res.Errors)
	}
	if cp := episodeCheckpoint(t, b, nsID); cp != ep {
		t.Errorf("checkpoint = %d, want %d (E is mined)", cp, ep)
	}
}

// Real concurrency: both passes extract E, then race to insert. The row lock
// serialises them, so exactly one fact is sourced from E. Several rounds, so a
// missing lock shows up even when one interleaving happens to be safe.
func TestConsolidate_ParallelPassesMineAnEpisodeOnce(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)

	const rounds = 5
	for round := 0; round < rounds; round++ {
		slug := "/t" + string(rune('a'+round))
		nsID := mustNamespace(t, b, slug)
		text := "episode E in " + slug
		ep := mustEpisode(t, b, e, slug, text)

		// Both passes reach the reasoner before either may insert.
		var mu sync.Mutex
		arrived := 0
		release := make(chan struct{})
		r.script(text, func(_ context.Context, n int) (*reasoner.StructuredFact, error) {
			mu.Lock()
			arrived++
			if arrived == 2 {
				close(release)
			}
			mu.Unlock()
			select {
			case <-release:
			case <-time.After(10 * time.Second):
				t.Errorf("round %d: second pass never reached the reasoner", round)
			}
			return &reasoner.StructuredFact{Summary: text + " summary " + string(rune('0'+n))}, nil
		})

		var wg sync.WaitGroup
		for i := 0; i < 2; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if _, err := b.ConsolidateByID(context.Background(), nsID); err != nil {
					t.Errorf("ConsolidateByID: %v", err)
				}
			}()
		}
		wg.Wait()

		if n := factsSourcedFrom(t, b, ep); n != 1 {
			t.Errorf("round %d: facts sourced from E = %d, want 1", round, n)
		}
		if cp := episodeCheckpoint(t, b, nsID); cp != ep {
			t.Errorf("round %d: checkpoint = %d, want %d", round, cp, ep)
		}
	}
}

// Two passes can finish in either order. The one that read an older checkpoint
// must not move it back below the other's, or the episodes in between are
// fetched again.
func TestSaveConsolidationProgress_EpisodeCheckpointNeverMovesBackwards(t *testing.T) {
	r := newFakeReasoner(t)
	e := newFakeEmbedder()
	b := openTestBrain(t, r, e)
	nsID := mustNamespace(t, b, "/t")
	ctx := context.Background()

	cp, err := b.GetOrCreateConsolidationProgress(ctx, nsID)
	if err != nil {
		t.Fatalf("GetOrCreateConsolidationProgress: %v", err)
	}
	if _, err := b.pool.Exec(ctx, "UPDATE consolidation_progress SET last_episode_id = 50 WHERE namespace_id = $1", nsID); err != nil {
		t.Fatalf("advance checkpoint: %v", err)
	}

	cp.LastEpisodeID = 10 // a slower pass that started from an older checkpoint
	if err := b.SaveConsolidationProgress(ctx, *cp); err != nil {
		t.Fatalf("SaveConsolidationProgress: %v", err)
	}
	if got := episodeCheckpoint(t, b, nsID); got != 50 {
		t.Fatalf("checkpoint = %d after saving 10 over 50, want 50", got)
	}

	cp.LastEpisodeID = 60
	if err := b.SaveConsolidationProgress(ctx, *cp); err != nil {
		t.Fatalf("SaveConsolidationProgress: %v", err)
	}
	if got := episodeCheckpoint(t, b, nsID); got != 60 {
		t.Fatalf("checkpoint = %d after saving 60, want 60", got)
	}
}
