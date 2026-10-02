package queries

import (
	"reflect"
	"strings"
	"testing"

	"github.com/pgvector/pgvector-go"
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

// supersededFilter is the clause that hides facts whose valid_until has
// passed. On prod, 237 of 550 recalled (call, fact) pairs had already been
// superseded when they were served.
const supersededFilter = "f.valid_until IS NULL OR f.valid_until > now()"

func renderRecallFacts(t *testing.T, includeSuperseded bool) string {
	t.Helper()
	q, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	sql, _, err := q.RecallFacts([]int64{1, 2}, pgvector.NewVector([]float32{1, 0}), 10, includeSuperseded)
	if err != nil {
		t.Fatalf("RecallFacts: %v", err)
	}
	return sql
}

func TestRecallFactsTemplate_FiltersSupersededByDefault(t *testing.T) {
	sql := renderRecallFacts(t, false)
	if !strings.Contains(sql, supersededFilter) {
		t.Fatalf("default recall must hide superseded facts; rendered SQL lacks %q:\n%s", supersededFilter, sql)
	}
}

func TestRecallFactsTemplate_IncludeSupersededDropsFilter(t *testing.T) {
	sql := renderRecallFacts(t, true)
	if strings.Contains(sql, supersededFilter) {
		t.Fatalf("include_superseded must drop the validity filter; rendered SQL still has %q:\n%s", supersededFilter, sql)
	}
}

func TestRecallTemplates_JoinNamespaces(t *testing.T) {
	q, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	vec := pgvector.NewVector([]float32{1, 0})
	epSQL, _, err := q.RecallEpisodes([]int64{1}, vec, 10)
	if err != nil {
		t.Fatalf("RecallEpisodes: %v", err)
	}
	for name, sql := range map[string]string{
		"recall_facts":    renderRecallFacts(t, false),
		"recall_episodes": epSQL,
	} {
		for _, want := range []string{"JOIN namespaces", "n.slug"} {
			if !strings.Contains(sql, want) {
				t.Errorf("%s must contain %q:\n%s", name, want, sql)
			}
		}
	}
}

// Every fact result names its source episodes (at most 10, in id order) and
// says how many there are in total.
func TestRecallFactsTemplate_SelectsProvenance(t *testing.T) {
	sql := renderRecallFacts(t, false)
	for _, want := range []string{
		"source_episode_ids", "source_episode_count", "fact_sources",
		"ORDER BY episode_id LIMIT 10", "f.valid_from", "f.valid_until",
	} {
		if !strings.Contains(sql, want) {
			t.Errorf("recall_facts must contain %q:\n%s", want, sql)
		}
	}
}
