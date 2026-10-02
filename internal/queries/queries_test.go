package queries

import (
	"reflect"
	"strings"
	"testing"
)

// Stage 1 must be able to tell an episode that already has a fact from one
// that was never mined, so a checkpoint reset or regression cannot turn old
// episodes into new paraphrase facts. The batch itself must stay "the next
// <=limit ids after the checkpoint" — that ordering is what the per-episode
// checkpoint advance relies on.
func TestFetchEpisodesSelectsAlreadyMined(t *testing.T) {
	q, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	sql, args, err := q.FetchEpisodes(7, 42, 100)
	if err != nil {
		t.Fatalf("FetchEpisodes: %v", err)
	}

	for _, want := range []string{"already_mined", "fact_sources", "id >", "ORDER BY id"} {
		if !strings.Contains(sql, want) {
			t.Errorf("rendered SQL must contain %q:\n%s", want, sql)
		}
	}
	if want := []any{int64(7), int64(42), 100}; !reflect.DeepEqual(args, want) {
		t.Fatalf("args = %#v, want %#v", args, want)
	}
}

// An episode whose extraction ended without a fact (nothing to extract,
// unusable output, input rejected, given up) is just as final as one with a
// fact source. Unless the query says so, it is re-extracted on every pass
// while anything below it pins the checkpoint.
func TestFetchEpisodesAlreadyMinedIncludesFinalOutcomes(t *testing.T) {
	q, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	sql, _, err := q.FetchEpisodes(7, 42, 100)
	if err != nil {
		t.Fatalf("FetchEpisodes: %v", err)
	}
	for _, want := range []string{"consolidation_episode_state", "outcome IS NOT NULL"} {
		if !strings.Contains(sql, want) {
			t.Errorf("rendered SQL must contain %q:\n%s", want, sql)
		}
	}
}
