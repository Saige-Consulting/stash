package brain

// Shared fixtures for the database-backed tests in this package.
//
// These tests need a real pgvector database. They are skipped unless
// STASH_TEST_DSN is set, so `go test ./...` stays pure (no DB, no network) and
// the hub image build, which runs the package tests without a database, is
// unaffected. Every test truncates every stash table first, so give each
// concurrent runner its own database:
//
//	docker run -d --rm --name stash-test-pg-$USER -e POSTGRES_USER=stash \
//	  -e POSTGRES_PASSWORD=stash -e POSTGRES_DB=stash_test \
//	  -p 127.0.0.1:55532:5432 pgvector/pgvector:pg16
//	STASH_TEST_DSN='postgres://stash:stash@127.0.0.1:55532/stash_test?sslmode=disable' \
//	  go test -count=1 ./...

import (
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/alash3al/stash/internal/db"
	"github.com/alash3al/stash/internal/models"
	"github.com/alash3al/stash/internal/queries"
	"github.com/alash3al/stash/internal/reasoner"
	"github.com/jackc/pgx/v5"
	"github.com/pgvector/pgvector-go"
)

const testVectorDim = 8

// openTestBrain opens the test database, empties every stash table and returns
// a Brain wired to the given fakes with the production default config.
func openTestBrain(t *testing.T, r reasoner.Reasoner, e *fakeEmbedder) *Brain {
	t.Helper()
	dsn := os.Getenv("STASH_TEST_DSN")
	if dsn == "" {
		t.Skip("STASH_TEST_DSN not set; skipping database-backed test")
	}
	ctx := context.Background()

	pool, err := db.Open(ctx, dsn, "fake-embed", testVectorDim)
	if err != nil {
		t.Fatalf("db.Open: %v", err)
	}
	t.Cleanup(pool.Close)

	rows, err := pool.Query(ctx,
		`SELECT quote_ident(tablename) FROM pg_tables
		 WHERE schemaname = 'public' AND tablename <> 'goose_db_version'`)
	if err != nil {
		t.Fatalf("list tables: %v", err)
	}
	tables, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		t.Fatalf("collect tables: %v", err)
	}
	if len(tables) == 0 {
		t.Fatal("no stash tables found after migrations")
	}
	if _, err := pool.Exec(ctx, "TRUNCATE "+strings.Join(tables, ", ")+" RESTART IDENTITY CASCADE"); err != nil {
		t.Fatalf("truncate: %v", err)
	}

	q, err := queries.New()
	if err != nil {
		t.Fatalf("queries.New: %v", err)
	}
	b, err := New(pool, e, r, q, DefaultConfig())
	if err != nil {
		t.Fatalf("brain.New: %v", err)
	}
	return b
}

// --- fake embedder ---

// fakeEmbedder returns fixed vectors for registered texts and a fresh vector
// for anything else.
//
// Episodes are registered to orthogonal basis vectors, so each episode is its
// own cluster (cosine 0 < SimilarityThreshold 0.85).
//
// Unregistered texts are the reasoner's summaries. Each call returns a NEW
// vector, which is how prod paraphrases behave: every re-extraction of the
// same episode embeds a little differently and slips past the 0.95 dedup. The
// fresh vectors are the normalised non-zero 0/1 patterns over 8 dimensions.
// Any two of them have cosine <= sqrt(7/8) ~= 0.935, so they never dedup
// against each other.
type fakeEmbedder struct {
	mu        sync.Mutex
	fixed     map[string][]float32
	nextBasis int
	nextFresh int
}

func newFakeEmbedder() *fakeEmbedder {
	return &fakeEmbedder{fixed: map[string][]float32{}}
}

// episodeVector registers text to the next unused basis vector.
func (f *fakeEmbedder) episodeVector(t *testing.T, text string) []float32 {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.nextBasis >= testVectorDim {
		t.Fatalf("fakeEmbedder: only %d orthogonal episode vectors available", testVectorDim)
	}
	v := make([]float32, testVectorDim)
	v[f.nextBasis] = 1
	f.nextBasis++
	f.fixed[text] = v
	return v
}

// set registers text to v, e.g. to make a summary embed onto an existing fact.
func (f *fakeEmbedder) set(text string, v []float32) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fixed[text] = v
}

func (f *fakeEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if v, ok := f.fixed[text]; ok {
		return append([]float32(nil), v...), nil
	}
	f.nextFresh++
	if f.nextFresh >= 1<<testVectorDim {
		return nil, fmt.Errorf("fakeEmbedder: out of fresh vectors")
	}
	v := make([]float32, testVectorDim)
	var n float64
	for i := 0; i < testVectorDim; i++ {
		if f.nextFresh&(1<<i) != 0 {
			v[i] = 1
			n++
		}
	}
	norm := float32(math.Sqrt(n))
	for i := range v {
		v[i] /= norm
	}
	return v, nil
}

func (f *fakeEmbedder) Model() string { return "fake-embed" }
func (f *fakeEmbedder) Dims() int     { return testVectorDim }

// --- fake reasoner ---

// structuredScript answers the n-th (1-based) ReasonStructured call that
// includes a given episode text.
type structuredScript func(ctx context.Context, n int) (*reasoner.StructuredFact, error)

// fakeReasoner implements every Reasoner method. ReasonStructured is scripted
// per episode text; ReasonContradiction by an optional hook; every other
// stage returns nothing. All calls are counted.
type fakeReasoner struct {
	t *testing.T

	mu              sync.Mutex
	structured      map[string]structuredScript
	structuredCalls map[string]int // per episode text
	contradiction   func(entity, property, oldValue, newValue string) (*reasoner.ContradictionResult, error)
	calls           map[string]int // per method
}

func newFakeReasoner(t *testing.T) *fakeReasoner {
	return &fakeReasoner{
		t:               t,
		structured:      map[string]structuredScript{},
		structuredCalls: map[string]int{},
		calls:           map[string]int{},
	}
}

func (r *fakeReasoner) script(text string, s structuredScript) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.structured[text] = s
}

// summaryScript answers "<prefix> v<n>" on the n-th call, with no entity or
// property, so contradiction detection stays out of the way.
func summaryScript(prefix string) structuredScript {
	return func(_ context.Context, n int) (*reasoner.StructuredFact, error) {
		return &reasoner.StructuredFact{Summary: fmt.Sprintf("%s v%d", prefix, n)}, nil
	}
}

func errorScript(err error) structuredScript {
	return func(context.Context, int) (*reasoner.StructuredFact, error) { return nil, err }
}

func (r *fakeReasoner) structuredCallsFor(text string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.structuredCalls[text]
}

func (r *fakeReasoner) callsTo(method string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.calls[method]
}

func (r *fakeReasoner) totalCalls() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	total := 0
	for _, n := range r.calls {
		total += n
	}
	return total
}

func (r *fakeReasoner) count(method string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls[method]++
}

func (r *fakeReasoner) ReasonStructured(ctx context.Context, texts []string) (*reasoner.StructuredFact, error) {
	r.count("ReasonStructured")
	if len(texts) != 1 {
		r.t.Errorf("fakeReasoner: want single-episode clusters, got %d texts: %q", len(texts), texts)
		return nil, fmt.Errorf("unexpected cluster")
	}
	r.mu.Lock()
	r.structuredCalls[texts[0]]++
	n := r.structuredCalls[texts[0]]
	s, ok := r.structured[texts[0]]
	r.mu.Unlock()
	if !ok {
		r.t.Errorf("fakeReasoner: no ReasonStructured script for %q", texts[0])
		return nil, fmt.Errorf("unscripted text")
	}
	return s(ctx, n)
}

func (r *fakeReasoner) ReasonRelationships(context.Context, string) ([]*reasoner.StructuredRelationship, error) {
	r.count("ReasonRelationships")
	return nil, nil
}

func (r *fakeReasoner) ReasonPatterns(context.Context, []models.Fact, []models.Relationship) ([]*reasoner.StructuredPattern, error) {
	r.count("ReasonPatterns")
	return nil, nil
}

func (r *fakeReasoner) ReasonContradiction(_ context.Context, entity, property, oldValue, newValue string) (*reasoner.ContradictionResult, error) {
	r.count("ReasonContradiction")
	if r.contradiction != nil {
		return r.contradiction(entity, property, oldValue, newValue)
	}
	return &reasoner.ContradictionResult{Classification: reasoner.ClassificationCompatible, Confidence: 1}, nil
}

func (r *fakeReasoner) ReasonCausalLinks(context.Context, []models.Fact) ([]*reasoner.StructuredCausalLink, error) {
	r.count("ReasonCausalLinks")
	return nil, nil
}

func (r *fakeReasoner) ReasonGoalProgress(context.Context, []models.Goal, []models.Fact) ([]*reasoner.GoalProgressAssessment, error) {
	r.count("ReasonGoalProgress")
	return nil, nil
}

func (r *fakeReasoner) ReasonFailurePatterns(context.Context, []models.Failure, []string) ([]*reasoner.FailurePatternResult, error) {
	r.count("ReasonFailurePatterns")
	return nil, nil
}

func (r *fakeReasoner) ReasonHypothesisEvidence(context.Context, []models.Hypothesis, []models.Fact) ([]*reasoner.HypothesisEvidenceResult, error) {
	r.count("ReasonHypothesisEvidence")
	return nil, nil
}

// --- data helpers ---

func mustNamespace(t *testing.T, b *Brain, slug string) int64 {
	t.Helper()
	id, err := b.CreateNamespace(context.Background(), slug, slug, "")
	if err != nil {
		t.Fatalf("CreateNamespace(%q): %v", slug, err)
	}
	return id
}

// mustEpisode stores text as an episode in slug, embedded to its own basis
// vector so it forms a single-episode cluster.
func mustEpisode(t *testing.T, b *Brain, e *fakeEmbedder, slug, text string) int64 {
	t.Helper()
	e.episodeVector(t, text)
	id, err := b.Remember(context.Background(), slug, text, nil)
	if err != nil {
		t.Fatalf("Remember(%q): %v", text, err)
	}
	return id
}

// mustFact inserts a fact directly and links it to sourceEpisodes.
func mustFact(t *testing.T, b *Brain, nsID int64, content string, vec []float32, entity, property, value *string, sourceEpisodes ...int64) int64 {
	t.Helper()
	ctx := context.Background()
	var id int64
	err := b.pool.QueryRow(ctx,
		`INSERT INTO facts (namespace_id, content, embedding, embedding_model, confidence, entity, property, value, valid_from)
		 VALUES ($1, $2, $3, 'fake-embed', 0.5, $4, $5, $6, now()) RETURNING id`,
		nsID, content, pgvector.NewVector(vec), entity, property, value,
	).Scan(&id)
	if err != nil {
		t.Fatalf("insert fact %q: %v", content, err)
	}
	for _, eid := range sourceEpisodes {
		if _, err := b.pool.Exec(ctx, "INSERT INTO fact_sources (fact_id, episode_id) VALUES ($1, $2)", id, eid); err != nil {
			t.Fatalf("insert fact_sources(%d, %d): %v", id, eid, err)
		}
	}
	return id
}

func mustConsolidate(t *testing.T, b *Brain, nsID int64) ConsolidationResult {
	t.Helper()
	res, err := b.ConsolidateByID(context.Background(), nsID)
	if err != nil {
		t.Fatalf("ConsolidateByID: %v", err)
	}
	return res
}

func queryInt(t *testing.T, b *Brain, sql string, args ...any) int64 {
	t.Helper()
	var n int64
	if err := b.pool.QueryRow(context.Background(), sql, args...).Scan(&n); err != nil {
		t.Fatalf("query %q: %v", sql, err)
	}
	return n
}

func factsSourcedFrom(t *testing.T, b *Brain, episodeID int64) int64 {
	t.Helper()
	return queryInt(t, b, "SELECT count(*) FROM fact_sources WHERE episode_id = $1", episodeID)
}

func episodeCheckpoint(t *testing.T, b *Brain, nsID int64) int64 {
	t.Helper()
	return queryInt(t, b, "SELECT last_episode_id FROM consolidation_progress WHERE namespace_id = $1", nsID)
}

func strp(s string) *string { return &s }
