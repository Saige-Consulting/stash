package brain

import (
	"context"
	"testing"
)

// query_facts hides superseded facts unless include_superseded is set (DC2),
// and never reads the embedding column.
func TestQueryFacts_ExcludesSupersededByDefault(t *testing.T) {
	e := newFakeEmbedder()
	b := openTestBrain(t, newFakeReasoner(t), e)
	ctx := context.Background()

	ns := mustNamespace(t, b, "/facts")
	live := mustFact(t, b, ns, "live fact", basis(0), nil, nil, nil)
	old := mustFact(t, b, ns, "superseded fact", basis(1), nil, nil, nil)
	expiring := mustFact(t, b, ns, "fact valid until tomorrow", basis(2), nil, nil, nil)
	expireFact(t, b, old, "-1 hour")
	expireFact(t, b, expiring, "1 day")

	got, err := b.QueryFacts(ctx, []string{"/facts"}, nil, nil, Pagination{}, false)
	if err != nil {
		t.Fatalf("QueryFacts: %v", err)
	}
	ids := map[int64]bool{}
	for _, f := range got {
		ids[f.ID] = true
	}
	if len(got) != 2 || !ids[live] || !ids[expiring] {
		t.Fatalf("default query_facts = %v, want live %d and not-yet-expired %d only", ids, live, expiring)
	}

	got, err = b.QueryFacts(ctx, []string{"/facts"}, nil, nil, Pagination{}, true)
	if err != nil {
		t.Fatalf("QueryFacts(include superseded): %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("include_superseded: %d facts, want 3", len(got))
	}
	for _, f := range got {
		switch f.ID {
		case old, expiring:
			if f.ValidUntil == nil {
				t.Errorf("fact %d: ValidUntil is nil, want set", f.ID)
			}
		case live:
			if f.ValidUntil != nil {
				t.Errorf("live fact: ValidUntil = %v, want nil", f.ValidUntil)
			}
		}
		if n := len(f.Embedding.Slice()); n != 0 {
			t.Errorf("fact %d: embedding read back (%d floats); query_facts must not select it", f.ID, n)
		}
		if f.Content == "" || f.EmbeddingModel == "" {
			t.Errorf("fact %d lost a column: %+v", f.ID, f)
		}
	}
}
