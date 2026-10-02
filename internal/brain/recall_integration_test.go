package brain

import (
	"context"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/pgvector/pgvector-go"
)

// Database-backed recall tests. They skip unless STASH_TEST_DSN is set; see
// integration_helpers_test.go.

func resultsByID(results []RecallResult, typ string) map[int64]RecallResult {
	out := map[int64]RecallResult{}
	for _, r := range results {
		if r.Type == typ {
			out[r.ID] = r
		}
	}
	return out
}

func normalized(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x / float32(math.Sqrt(n))
	}
	return out
}

func TestRecall_ExcludesSupersededByDefault(t *testing.T) {
	e := newFakeEmbedder()
	b := openTestBrain(t, newFakeReasoner(t), e)
	ctx := context.Background()

	ns := mustNamespace(t, b, "/findings")
	vec := basis(0)
	e.set("which pr", vec)
	x := mustFact(t, b, ns, "X: the PR is #375", vec, nil, nil, nil)
	y := mustFact(t, b, ns, "Y: the PR is #376", vec, nil, nil, nil)
	// Z expires in the future: still valid now, so it is not superseded.
	z := mustFact(t, b, ns, "Z: the PR is #377 until tomorrow", vec, nil, nil, nil)
	expireFact(t, b, x, "-1 hour")
	expireFact(t, b, z, "1 day")

	got, err := b.Recall(ctx, []string{"/"}, "which pr", 10)
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	facts := resultsByID(got, "fact")
	if _, ok := facts[x]; ok {
		t.Errorf("default recall returned superseded fact X (%d): %+v", x, facts[x])
	}
	if _, ok := facts[y]; !ok {
		t.Errorf("default recall lost live fact Y (%d); got %+v", y, got)
	}
	if r, ok := facts[z]; !ok {
		t.Errorf("default recall lost not-yet-expired fact Z (%d); got %+v", z, got)
	} else if r.Superseded || r.ValidUntil == "" {
		t.Errorf("Z: superseded=%v valid_until=%q, want false and set", r.Superseded, r.ValidUntil)
	}

	got, err = b.RecallWithOptions(ctx, []string{"/"}, "which pr", 10, RecallOptions{IncludeSuperseded: true})
	if err != nil {
		t.Fatalf("RecallWithOptions: %v", err)
	}
	facts = resultsByID(got, "fact")
	if len(facts) != 3 {
		t.Fatalf("include_superseded: %d facts, want 3: %+v", len(facts), got)
	}
	if r := facts[x]; !r.Superseded || r.ValidUntil == "" {
		t.Errorf("X: superseded=%v valid_until=%q, want true and set", r.Superseded, r.ValidUntil)
	}
	if r := facts[y]; r.Superseded || r.ValidUntil != "" {
		t.Errorf("Y: superseded=%v valid_until=%q, want false and empty", r.Superseded, r.ValidUntil)
	}
}

func TestRecall_CarriesProvenance(t *testing.T) {
	e := newFakeEmbedder()
	b := openTestBrain(t, newFakeReasoner(t), e)
	ctx := context.Background()

	nsX := mustNamespace(t, b, "/findings/x")
	nsY := mustNamespace(t, b, "/findings/y")
	e1 := mustEpisode(t, b, e, "/findings/x", "Ticket 146962 was reassigned to the claims team.") // basis 0
	e2 := mustEpisode(t, b, e, "/findings/x", "The claims team closed ticket 146962.")            // basis 1
	fv := normalized([]float32{1, 1, 0, 0, 0, 0, 0, 0})
	f := mustFact(t, b, nsX, "Ticket 146962 went to the claims team, which closed it.", fv, nil, nil, nil, e2, e1)
	d := mustFact(t, b, nsY, "Derived: claims tickets close within a day.", fv, nil, nil, nil)
	e.set("ticket 146962 owner", fv)

	got, err := b.Recall(ctx, []string{"/findings"}, "ticket 146962 owner", 10)
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	facts, episodes := resultsByID(got, "fact"), resultsByID(got, "episode")

	rf, ok := facts[f]
	if !ok {
		t.Fatalf("fact F (%d) not recalled: %+v", f, got)
	}
	if rf.Namespace != "/findings/x" || rf.NamespaceID != nsX {
		t.Errorf("F namespace = %q (%d), want /findings/x (%d)", rf.Namespace, rf.NamespaceID, nsX)
	}
	if rf.Writer != "consolidator" {
		t.Errorf("F writer = %q, want consolidator", rf.Writer)
	}
	if want := []int64{e1, e2}; !reflect.DeepEqual(rf.SourceEpisodeIDs, want) {
		t.Errorf("F source_episode_ids = %v, want %v (ordered by id)", rf.SourceEpisodeIDs, want)
	}
	if rf.SourceEpisodeCount == nil || *rf.SourceEpisodeCount != 2 {
		t.Errorf("F source_episode_count = %v, want 2", rf.SourceEpisodeCount)
	}
	if rf.ValidFrom == "" {
		t.Errorf("F valid_from is empty; mustFact sets valid_from = now()")
	}

	rd, ok := facts[d]
	if !ok {
		t.Fatalf("derived fact D (%d) not recalled: %+v", d, got)
	}
	if rd.Writer != "derived" || rd.Namespace != "/findings/y" {
		t.Errorf("D writer=%q namespace=%q, want derived and /findings/y", rd.Writer, rd.Namespace)
	}
	if rd.SourceEpisodeIDs == nil || len(rd.SourceEpisodeIDs) != 0 {
		t.Errorf("D source_episode_ids = %#v, want an empty, non-nil list", rd.SourceEpisodeIDs)
	}
	if rd.SourceEpisodeCount == nil || *rd.SourceEpisodeCount != 0 {
		t.Errorf("D source_episode_count = %v, want 0", rd.SourceEpisodeCount)
	}

	if len(episodes) != 2 {
		t.Fatalf("want both episodes to fill the remaining slots, got %+v", got)
	}
	for _, id := range []int64{e1, e2} {
		ep, ok := episodes[id]
		if !ok {
			t.Errorf("episode %d not recalled", id)
			continue
		}
		if ep.Writer != "agent" {
			t.Errorf("episode %d writer = %q, want agent", id, ep.Writer)
		}
		if !reflect.DeepEqual(ep.SourceEpisodeIDs, []int64{id}) {
			t.Errorf("episode %d source_episode_ids = %v, want [%d]", id, ep.SourceEpisodeIDs, id)
		}
		if ep.Namespace != "/findings/x" {
			t.Errorf("episode %d namespace = %q, want /findings/x", id, ep.Namespace)
		}
		if ep.SourceEpisodeCount != nil {
			t.Errorf("episode %d carries source_episode_count %d; it is for facts only", id, *ep.SourceEpisodeCount)
		}
	}
}

// A fact mined from many episodes lists the first 10 and counts all of them,
// so one heavily-merged fact cannot blow up the response.
func TestRecall_SourceIDsCappedAtTen(t *testing.T) {
	e := newFakeEmbedder()
	b := openTestBrain(t, newFakeReasoner(t), e)
	ctx := context.Background()

	ns := mustNamespace(t, b, "/findings")
	var eps []int64
	for i := 0; i < 12; i++ {
		eps = append(eps, mustRawEpisode(t, b, ns, "raw episode"))
	}
	vec := basis(3)
	e.set("merged fact", vec)
	// Link in reverse so the cap must sort, not take insertion order.
	rev := make([]int64, len(eps))
	for i, id := range eps {
		rev[len(eps)-1-i] = id
	}
	f := mustFact(t, b, ns, "A fact merged from twelve episodes.", vec, nil, nil, nil, rev...)

	got, err := b.Recall(ctx, []string{"/"}, "merged fact", 10)
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	r, ok := resultsByID(got, "fact")[f]
	if !ok {
		t.Fatalf("fact %d not recalled: %+v", f, got)
	}
	if want := eps[:10]; !reflect.DeepEqual(r.SourceEpisodeIDs, want) {
		t.Errorf("source_episode_ids = %v, want the 10 lowest %v", r.SourceEpisodeIDs, want)
	}
	if r.SourceEpisodeCount == nil || *r.SourceEpisodeCount != 12 {
		t.Errorf("source_episode_count = %v, want 12", r.SourceEpisodeCount)
	}
}

// Prod plans a whole-store recall as an HNSW index scan. Without iterative
// scan, the index hands back ef_search (40) candidates and the validity filter
// then drops the superseded ones, so recall returns fewer than limit, or
// nothing. This test forces that plan (and fails its own setup if the planner
// does not use the index), puts 60 superseded facts nearer the query than 12
// live ones, and asks for 10.
func TestRecall_LimitFilledDespiteSuperseded(t *testing.T) {
	dsn := os.Getenv("STASH_TEST_DSN")
	if dsn == "" {
		t.Skip("STASH_TEST_DSN not set; skipping database-backed test")
	}
	sep := "?"
	if strings.Contains(dsn, "?") {
		sep = "&"
	}
	// Unknown DSN keys become startup parameters on every pooled connection.
	t.Setenv("STASH_TEST_DSN", dsn+sep+"enable_seqscan=off&enable_sort=off")

	e := newFakeEmbedder()
	b := openTestBrain(t, newFakeReasoner(t), e)
	ctx := context.Background()

	ns := mustNamespace(t, b, "/findings")
	query := basis(0)
	e.set("superseded crowd", query)

	// A deterministic spread: superseded facts sit within ~0.2 of the query,
	// live facts at 0.4-0.6, so every superseded fact is nearer.
	near := func(i int, base, step float32) []float32 {
		v := make([]float32, testVectorDim)
		v[0] = 1
		v[1+i%(testVectorDim-1)] = base + step*float32(i)
		v[1+(i/7)%(testVectorDim-1)] += step / 2 * float32(i%5)
		return normalized(v)
	}
	for i := 0; i < 60; i++ {
		id := mustFact(t, b, ns, "superseded paraphrase", near(i, 0.01, 0.003), nil, nil, nil)
		expireFact(t, b, id, "-1 hour")
	}
	live := map[int64]bool{}
	for i := 0; i < 12; i++ {
		live[mustFact(t, b, ns, "live fact", near(i, 0.4, 0.015), nil, nil, nil)] = true
	}

	nsIDs, err := b.resolveNamespaceIDs(ctx, []string{"/"})
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	sql, args, err := b.queries.RecallFacts(nsIDs, pgvector.NewVector(query), 10, false)
	if err != nil {
		t.Fatalf("RecallFacts: %v", err)
	}
	rows, err := b.pool.Query(ctx, "EXPLAIN "+sql, args...)
	if err != nil {
		t.Fatalf("explain: %v", err)
	}
	var plan []string
	for rows.Next() {
		var line string
		if err := rows.Scan(&line); err != nil {
			t.Fatalf("scan plan: %v", err)
		}
		plan = append(plan, line)
	}
	rows.Close()
	if !strings.Contains(strings.Join(plan, "\n"), "facts_embedding_hnsw_idx") {
		t.Fatalf("test setup: the planner did not use the HNSW index, so this test cannot exercise iterative scan:\n%s", strings.Join(plan, "\n"))
	}

	got, err := b.Recall(ctx, []string{"/"}, "superseded crowd", 10)
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(got) != 10 {
		t.Fatalf("recall returned %d results, want 10 (limit) despite 60 nearer superseded facts", len(got))
	}
	for _, r := range got {
		if r.Type != "fact" || !live[r.ID] || r.Superseded {
			t.Errorf("result %+v is not a live fact", r)
		}
	}
	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Errorf("results not in score order at %d: %v > %v", i, got[i].Score, got[i-1].Score)
		}
	}
}
