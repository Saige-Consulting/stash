package brain

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/alash3al/stash/internal/models"
	"github.com/alash3al/stash/internal/observability"
	"github.com/alash3al/stash/internal/reasoner"
	"github.com/jackc/pgx/v5"
	"github.com/pgvector/pgvector-go"
)

// ConsolidationResult describes the outcome of a consolidation run.
//
// Stage 1 counters: EpisodesRead is every episode in the batch after the checkpoint.
// EpisodesAlreadyMined of them already had a fact source or a recorded final outcome and were
// not sent to the reasoner. EpisodesSkipped were given up on because the output was unusable
// for their text (reasoner.ErrUnusableOutput) or the provider rejected them on their own
// (reasoner.ErrInputRejected). EpisodesGaveUp kept failing until the give-up budget ran out
// (maxEpisodeAttempts over episodeGiveUpAfter). Each skip and give-up is also listed in Errors.
type ConsolidationResult struct {
	Namespace                  string        `json:"namespace"`
	Duration                   time.Duration `json:"duration"`
	EpisodesRead               int           `json:"episodes_read"`
	EpisodesAlreadyMined       int           `json:"episodes_already_mined"`
	EpisodesSkipped            int           `json:"episodes_skipped"`
	EpisodesGaveUp             int           `json:"episodes_gave_up"`
	FactsCreated               int           `json:"facts_created"`
	FactsDeduplicated          int           `json:"facts_deduplicated"`
	RelationshipsFound         int           `json:"relationships_found"`
	CausalLinksFound           int           `json:"causal_links_found"`
	PatternsFound              int           `json:"patterns_found"`
	ContradictionsFound        int           `json:"contradictions_found"`
	ContradictionsAutoResolved int           `json:"contradictions_auto_resolved"`
	GoalsAnnotated             int           `json:"goals_annotated"`
	GoalsSuggestedComplete     int           `json:"goals_suggested_complete"`
	FailureRepeatsDetected     int           `json:"failure_repeats_detected"`
	FailurePatternsFound       int           `json:"failure_patterns_found"`
	HypothesesAutoConfirmed    int           `json:"hypotheses_auto_confirmed"`
	HypothesesAutoRejected     int           `json:"hypotheses_auto_rejected"`
	HypothesesUpdated          int           `json:"hypotheses_updated"`
	FactsDecayed               int           `json:"facts_decayed"`
	FactsExpired               int           `json:"facts_expired"`
	LLMCalls                   int           `json:"llm_calls"`
	Errors                     []string      `json:"errors,omitempty"`
}

// Consolidate runs the full 3-stage consolidation pipeline for a namespace.
func (b *Brain) Consolidate(ctx context.Context, namespaceSlug string) (ConsolidationResult, error) {
	if err := validatePath(namespaceSlug); err != nil {
		return ConsolidationResult{}, err
	}
	nsID, err := b.resolveNamespaceID(ctx, namespaceSlug)
	if err != nil {
		return ConsolidationResult{}, err
	}
	return b.ConsolidateByID(ctx, nsID)
}

// ConsolidateByID runs the full 3-stage consolidation pipeline for a namespace by ID.
// 1. Episodes -> Facts (cluster + synthesize)
// 2. Facts -> Relationships (extract entity edges)
// 3. Facts + Relationships -> Patterns (extract abstractions)
func (b *Brain) ConsolidateByID(ctx context.Context, nsID int64) (ConsolidationResult, error) {
	start := time.Now()

	var namespaceSlug string
	_ = b.pool.QueryRow(ctx, "SELECT slug FROM namespaces WHERE id = $1", nsID).Scan(&namespaceSlug)

	result := ConsolidationResult{Namespace: namespaceSlug}

	cp, err := b.GetOrCreateConsolidationProgress(ctx, nsID)
	if err != nil {
		return result, fmt.Errorf("get progress: %w", err)
	}

	// Stage 1: Episodes -> Facts (+ Stage 4: Contradiction detection)
	if ctx.Err() == nil {
		st := b.consolidateEpisodesToFacts(ctx, nsID, cp)
		result.FactsCreated = st.created
		result.FactsDeduplicated = st.deduped
		result.EpisodesRead = st.read
		result.EpisodesAlreadyMined = st.alreadyMined
		result.EpisodesSkipped = st.skipped
		result.EpisodesGaveUp = st.gaveUp
		result.LLMCalls += st.llmCalls
		result.ContradictionsFound = st.contradictionsFound
		result.ContradictionsAutoResolved = st.contradictionsAutoResolved
		result.Errors = append(result.Errors, st.errs...)
	}

	// Stage 2: Facts -> Relationships
	if ctx.Err() == nil {
		relCount, llmCalls, errs := b.consolidateFactsToRelationships(ctx, nsID, cp)
		result.RelationshipsFound = relCount
		result.LLMCalls += llmCalls
		result.Errors = append(result.Errors, errs...)
	}

	// Stage 3.5: Facts -> Causal Links
	if ctx.Err() == nil {
		causalCount, llmCalls, errs := b.consolidateFactsToCausalLinks(ctx, nsID, cp)
		result.CausalLinksFound = causalCount
		result.LLMCalls += llmCalls
		result.Errors = append(result.Errors, errs...)
	}

	// Stage 6: Goal Progress Inference
	if ctx.Err() == nil {
		annotated, suggestedComplete, llmCalls, errs := b.consolidateGoalProgress(ctx, nsID, cp)
		result.GoalsAnnotated = annotated
		result.GoalsSuggestedComplete = suggestedComplete
		result.LLMCalls += llmCalls
		result.Errors = append(result.Errors, errs...)
	}

	// Stage 7: Failure Pattern Detection
	if ctx.Err() == nil {
		repeats, patterns, llmCalls, errs := b.consolidateFailurePatterns(ctx, nsID, cp)
		result.FailureRepeatsDetected = repeats
		result.FailurePatternsFound = patterns
		result.LLMCalls += llmCalls
		result.Errors = append(result.Errors, errs...)
	}

	// Stage 3: Facts + Relationships -> Patterns
	if ctx.Err() == nil {
		patCount, llmCalls, errs := b.consolidateToPatterns(ctx, nsID, cp)
		result.PatternsFound = patCount
		result.LLMCalls += llmCalls
		result.Errors = append(result.Errors, errs...)
	}

	// Stage 8: Hypothesis Evidence Scanning
	if ctx.Err() == nil {
		autoConfirmed, autoRejected, updated, llmCalls, errs := b.consolidateHypothesisEvidence(ctx, nsID, cp)
		result.HypothesesAutoConfirmed = autoConfirmed
		result.HypothesesAutoRejected = autoRejected
		result.HypothesesUpdated = updated
		result.LLMCalls += llmCalls
		result.Errors = append(result.Errors, errs...)
	}

	// Stage 5: Confidence decay
	if ctx.Err() == nil {
		decayResult, err := b.DecayConfidence(ctx, nsID)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("decay confidence: %v", err))
		} else {
			result.FactsDecayed = decayResult.FactsDecayed
			result.FactsExpired = decayResult.FactsExpired
			// Record that decay ran. Without this the column stays NULL forever
			// even though the stage runs on every pass, so operators inspecting
			// consolidation_progress conclude decay is disabled when it isn't.
			decayedAt := time.Now().UTC()
			cp.LastDecayRun = &decayedAt
		}
	}

	// Save progress
	now := time.Now().UTC()
	cp.LastRun = &now
	saveCtx := ctx
	if ctx.Err() != nil {
		saveCtx = context.Background()
	}
	if err := b.SaveConsolidationProgress(saveCtx, *cp); err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("save progress: %v", err))
	}

	result.Duration = time.Since(start)
	observability.RecordConsolidation(observability.Observation{
		Namespace:          namespaceSlug,
		EventsRead:         result.EpisodesRead,
		EventsProcessed:    result.EpisodesRead - result.EpisodesAlreadyMined,
		FactsCreated:       result.FactsCreated,
		FactsDeduplicated:  result.FactsDeduplicated,
		RelationshipsFound: result.RelationshipsFound,
		LLMCalls:           result.LLMCalls,
		Duration:           result.Duration,
		Errors:             len(result.Errors),
	})

	return result, nil
}

// --- Stage 1: Episodes -> Facts ---

// episodeStageResult is what stage 1 reports back to ConsolidateByID.
type episodeStageResult struct {
	read, alreadyMined, created, deduped, skipped, gaveUp, llmCalls int
	contradictionsFound, contradictionsAutoResolved                 int
	errs                                                            []string
}

// consolidateEpisodesToFacts mines the next batch of episodes after the checkpoint into facts.
//
// Invariant: an episode takes part in at most one fact-creating extraction. An episode that
// already has a fact_sources row or a recorded final outcome (consolidate_outcome.go) is
// "already mined" and never reaches the reasoner again, and the checkpoint advances per episode
// (nextEpisodeCheckpoint) over every episode whose outcome is final, stopping just before the
// first one that is not. An episode that keeps failing is given up on after a bounded number of
// counted failures over a time floor, so it cannot pin its namespace forever.
//
// The batch is still the next <= BatchSize ids after the checkpoint. Before this, the
// checkpoint advanced only when the whole batch had no error, so one persistently failing
// cluster pinned it and every other episode in the batch was re-extracted on every pass, each
// time into a fresh paraphrase that slipped past dedup and superseded the previous one.
func (b *Brain) consolidateEpisodesToFacts(ctx context.Context, nsID int64, cp *models.ConsolidationProgress) (res episodeStageResult) {
	sql, args, err := b.queries.FetchEpisodes(nsID, cp.LastEpisodeID, b.config.BatchSize)
	if err != nil {
		res.errs = append(res.errs, fmt.Sprintf("build fetch episodes: %v", err))
		return
	}

	rows, err := b.pool.Query(ctx, sql, args...)
	if err != nil {
		res.errs = append(res.errs, fmt.Sprintf("fetch episodes: %v", err))
		return
	}
	batch, err := readEpisodeBatch(rows)
	rows.Close()
	if err != nil {
		res.errs = append(res.errs, fmt.Sprintf("episode rows: %v", err))
		return
	}
	if batch.cutErr != nil {
		res.errs = append(res.errs, fmt.Sprintf("scan episode (id unknown; batch ends before it): %v", batch.cutErr))
	}

	res.read = len(batch.ids)
	res.alreadyMined = len(batch.mined)
	if res.read == 0 {
		return
	}

	done := make(map[int64]bool, len(batch.ids))
	for _, id := range batch.mined {
		done[id] = true
	}
	for _, u := range batch.unreadable {
		b.failEpisodes(ctx, []int64{u.id}, fmt.Errorf("scan episode %d: %w", u.id, u.err), done, &res)
	}

	// Cluster by vector similarity. Already-mined episodes stay out: clustering them in again
	// is exactly how an old episode ended up in a second fact.
	for _, cluster := range b.clusterEpisodes(batch.unmined) {
		if ctx.Err() != nil {
			break // the remaining clusters stay not done
		}
		b.mineCluster(ctx, nsID, cluster, done, &res)
	}

	cp.LastEpisodeID = nextEpisodeCheckpoint(cp.LastEpisodeID, batch.ids, done)
	return
}

// mineCluster extracts one fact from a cluster of unmined episodes and marks the cluster's
// episodes in done once their outcome is final and persisted. On any other failure it leaves
// them unmarked, so they are retried on the next pass, and counts the failure toward giving up.
func (b *Brain) mineCluster(ctx context.Context, nsID int64, cluster []models.Episode, done map[int64]bool, res *episodeStageResult) {
	texts := make([]string, 0, len(cluster))
	episodeIDs := make([]int64, 0, len(cluster))
	for _, e := range cluster {
		texts = append(texts, e.Content)
		episodeIDs = append(episodeIDs, e.ID)
	}

	sf, err := b.reasoner.ReasonStructured(ctx, texts)
	res.llmCalls++
	if err != nil {
		switch {
		case errors.Is(err, reasoner.ErrInputRejected) && len(cluster) > 1:
			// Too large, or blocked, as a whole. One episode may be the cause, or only the
			// sum: mine each on its own rather than give up on all of them.
			for _, e := range cluster {
				if ctx.Err() != nil {
					return
				}
				b.mineCluster(ctx, nsID, []models.Episode{e}, done, res)
			}
		case errors.Is(err, reasoner.ErrInputRejected):
			b.finishEpisodes(ctx, episodeIDs, outcomeRejected, err, done, res)
		case errors.Is(err, reasoner.ErrUnusableOutput):
			// Permanent for this text: retrying it every pass only buys a new paraphrase
			// to fail the same check.
			b.finishEpisodes(ctx, episodeIDs, outcomeUnusable, err, done, res)
		default:
			b.failEpisodes(ctx, episodeIDs, fmt.Errorf("reason structured: %w", err), done, res)
		}
		return
	}

	if sf.Summary == "" {
		b.finishEpisodes(ctx, episodeIDs, outcomeEmpty, nil, done, res)
		return
	}

	// Embed the fact content
	vec, err := b.embedder.Embed(ctx, sf.Summary)
	if err != nil {
		b.failEpisodes(ctx, episodeIDs, fmt.Errorf("embed fact: %w", err), done, res)
		return
	}

	// Check for duplicate fact
	dupID, dup, err := b.factExistsByVector(ctx, nsID, vec)
	if err != nil {
		b.failEpisodes(ctx, episodeIDs, fmt.Errorf("check duplicate: %w", err), done, res)
		return
	}
	if dup {
		// Link the episodes to the fact they duplicate, so they read as mined from now on.
		// Without the link they looked unmined and were sent to the reasoner again.
		if _, err := b.pool.Exec(ctx, insertFactSourcesSQL, dupID, episodeIDs); err != nil {
			b.failEpisodes(ctx, episodeIDs, fmt.Errorf("link duplicate fact %d to episodes %v: %w", dupID, episodeIDs, err), done, res)
			return
		}
		res.deduped++
		for _, id := range episodeIDs {
			done[id] = true
		}
		return
	}

	confidence := calculateConfidence(len(cluster))
	factID, err := b.insertFactWithSources(ctx, nsID, sf, vec, confidence, episodeIDs)
	if err != nil {
		b.failEpisodes(ctx, episodeIDs, fmt.Errorf("insert fact for episodes %v: %w", episodeIDs, err), done, res)
		return
	}
	res.created++
	for _, id := range episodeIDs {
		done[id] = true
	}

	// Stage 4: Contradiction detection. Runs after commit, so the new fact's sources are
	// visible to the candidate query.
	newFact := &models.Fact{
		ID:          factID,
		NamespaceID: nsID,
		Content:     sf.Summary,
		Confidence:  confidence,
		Entity:      strPtrOrNull(sf.Entity),
		Property:    strPtrOrNull(sf.Property),
		Value:       strPtrOrNull(sf.Value),
	}
	cd, ca, _ := b.DetectContradictions(ctx, nsID, newFact)
	res.contradictionsFound += cd
	res.contradictionsAutoResolved += ca
}

// insertFactSourcesSQL links one fact to a set of episodes in a single statement, so the links
// land all together or not at all.
const insertFactSourcesSQL = `INSERT INTO fact_sources (fact_id, episode_id)
	SELECT $1, unnest($2::bigint[]) ON CONFLICT DO NOTHING`

// insertFactWithSources inserts a fact and its fact_sources rows in one transaction.
//
// A fact without its sources must not exist: its episodes would look unmined and be mined
// again into another fact. The source inserts used to run outside any transaction with their
// errors discarded.
func (b *Brain) insertFactWithSources(ctx context.Context, nsID int64, sf *reasoner.StructuredFact, vec []float32, confidence float32, episodeIDs []int64) (int64, error) {
	tx, err := b.pool.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("begin: %w", err)
	}
	// No-op after a successful commit. WithoutCancel so a cancelled pass still rolls back
	// cleanly and returns the connection to the pool.
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()

	var factID int64
	err = tx.QueryRow(ctx,
		`INSERT INTO facts (namespace_id, content, embedding, embedding_model, confidence, entity, property, value, valid_from)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id`,
		nsID, sf.Summary, pgvector.NewVector(vec), b.embedder.Model(), confidence,
		strPtrOrNull(sf.Entity), strPtrOrNull(sf.Property), strPtrOrNull(sf.Value), time.Now().UTC(),
	).Scan(&factID)
	if err != nil {
		return 0, fmt.Errorf("insert fact: %w", err)
	}

	if _, err := tx.Exec(ctx, insertFactSourcesSQL, factID, episodeIDs); err != nil {
		return 0, fmt.Errorf("insert fact_sources: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("commit: %w", err)
	}
	return factID, nil
}

func (b *Brain) clusterEpisodes(episodes []models.Episode) [][]models.Episode {
	if len(episodes) == 0 {
		return nil
	}

	clustered := make(map[int64]bool)
	var clusters [][]models.Episode

	for _, seed := range episodes {
		if clustered[seed.ID] {
			continue
		}

		cluster := []models.Episode{seed}
		clustered[seed.ID] = true

		if seed.Embedding.Slice() == nil {
			clusters = append(clusters, cluster)
			continue
		}

		seedVec := seed.Embedding.Slice()
		for _, candidate := range episodes {
			if clustered[candidate.ID] {
				continue
			}
			candVec := candidate.Embedding.Slice()
			if candVec == nil {
				continue
			}
			sim := cosineSimilarity(seedVec, candVec)
			if sim >= float32(b.config.SimilarityThreshold) {
				cluster = append(cluster, candidate)
				clustered[candidate.ID] = true
			}
		}

		clusters = append(clusters, cluster)
	}

	return clusters
}

// factExistsByVector returns the nearest non-deleted fact in the namespace when it is a duplicate of
// vec (cosine >= DedupThreshold). An empty namespace is not a duplicate. Any other error is
// returned: it used to be swallowed as "not a duplicate", which inserted a new fact.
func (b *Brain) factExistsByVector(ctx context.Context, nsID int64, vec []float32) (factID int64, dup bool, err error) {
	var id int64
	var score float32
	err = b.pool.QueryRow(ctx,
		`SELECT id, 1 - (embedding <=> $2) AS score FROM facts
		 WHERE namespace_id = $1 AND deleted_at IS NULL AND embedding IS NOT NULL
		 ORDER BY embedding <=> $2 LIMIT 1`,
		nsID, pgvector.NewVector(vec),
	).Scan(&id, &score)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, fmt.Errorf("nearest fact: %w", err)
	}
	if score >= float32(b.config.DedupThreshold) {
		return id, true, nil
	}
	return 0, false, nil
}

func calculateConfidence(observationCount int) float32 {
	if observationCount == 0 {
		return 0.0
	}
	return float32(observationCount) / float32(observationCount+2)
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func strPtrOrNull(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// --- Stage 2: Facts -> Relationships ---

func (b *Brain) consolidateFactsToRelationships(ctx context.Context, nsID int64, cp *models.ConsolidationProgress) (count, llmCalls int, errs []string) {
	sql, args, err := b.queries.FetchFacts(nsID, cp.LastFactID, 50)
	if err != nil {
		errs = append(errs, fmt.Sprintf("build fetch facts: %v", err))
		return
	}

	rows, err := b.pool.Query(ctx, sql, args...)
	if err != nil {
		errs = append(errs, fmt.Sprintf("fetch facts: %v", err))
		return
	}
	defer rows.Close()

	var facts []models.Fact
	for rows.Next() {
		var f models.Fact
		if err := rows.Scan(&f.ID, &f.NamespaceID, &f.Content, &f.Embedding, &f.EmbeddingModel, &f.Confidence, &f.Entity, &f.Property, &f.Value, &f.ValidFrom, &f.ValidUntil, &f.CreatedAt, &f.UpdatedAt); err != nil {
			errs = append(errs, fmt.Sprintf("scan fact: %v", err))
			continue
		}
		facts = append(facts, f)
	}
	if err := rows.Err(); err != nil {
		errs = append(errs, fmt.Sprintf("fact rows: %v", err))
		return
	}

	if len(facts) == 0 {
		return
	}

	var maxID int64
	for _, fact := range facts {
		if ctx.Err() != nil {
			break
		}
		if fact.ID > maxID {
			maxID = fact.ID
		}

		rels, err := b.reasoner.ReasonRelationships(ctx, fact.Content)
		llmCalls++
		if err != nil {
			errs = append(errs, fmt.Sprintf("reason relationships fact %d: %v", fact.ID, err))
			continue
		}

		for _, rel := range rels {
			if rel.FromEntity == "" || rel.RelationType == "" || rel.ToEntity == "" {
				continue
			}

			// Check for existing relationship from this fact
			exists, _ := b.relationshipExists(ctx, nsID, rel.FromEntity, rel.RelationType, rel.ToEntity, fact.ID)
			if exists {
				continue
			}

			_, err := b.pool.Exec(ctx,
				`INSERT INTO relationships (namespace_id, from_entity, relation_type, to_entity, confidence, source_fact_id)
				 VALUES ($1, $2, $3, $4, $5, $6)`,
				nsID, rel.FromEntity, rel.RelationType, rel.ToEntity, rel.Confidence, fact.ID,
			)
			if err != nil {
				errs = append(errs, fmt.Sprintf("insert relationship: %v", err))
				continue
			}
			count++
		}
	}

	// Only advance checkpoint if no errors occurred (bullet-proof: prevents losing facts)
	if len(errs) == 0 && maxID > cp.LastFactID {
		cp.LastFactID = maxID
	}
	return
}

func (b *Brain) relationshipExists(ctx context.Context, nsID int64, from, relType, to string, sourceFactID int64) (bool, error) {
	var id int64
	err := b.pool.QueryRow(ctx,
		`SELECT id FROM relationships
		 WHERE namespace_id = $1 AND from_entity = $2 AND relation_type = $3 AND to_entity = $4
		 AND source_fact_id = $5 AND deleted_at IS NULL LIMIT 1`,
		nsID, from, relType, to, sourceFactID,
	).Scan(&id)
	if err != nil {
		return false, nil
	}
	return true, nil
}

// --- Stage 3.5: Facts -> Causal Links ---

func (b *Brain) consolidateFactsToCausalLinks(ctx context.Context, nsID int64, cp *models.ConsolidationProgress) (count, llmCalls int, errs []string) {
	sql, args, err := b.queries.FetchFacts(nsID, cp.LastFactID, 30)
	if err != nil {
		errs = append(errs, fmt.Sprintf("build fetch facts for causal: %v", err))
		return
	}

	rows, err := b.pool.Query(ctx, sql, args...)
	if err != nil {
		errs = append(errs, fmt.Sprintf("fetch facts for causal: %v", err))
		return
	}
	defer rows.Close()

	var facts []models.Fact
	for rows.Next() {
		var f models.Fact
		if err := rows.Scan(&f.ID, &f.NamespaceID, &f.Content, &f.Embedding, &f.EmbeddingModel, &f.Confidence, &f.Entity, &f.Property, &f.Value, &f.ValidFrom, &f.ValidUntil, &f.CreatedAt, &f.UpdatedAt); err != nil {
			errs = append(errs, fmt.Sprintf("scan fact for causal: %v", err))
			continue
		}
		facts = append(facts, f)
	}
	if err := rows.Err(); err != nil {
		errs = append(errs, fmt.Sprintf("fact rows for causal: %v", err))
		return
	}

	if len(facts) < 2 {
		return
	}

	llmCalls++
	found, detectErrs := b.DetectCausalLinks(ctx, nsID, facts)
	count = found
	errs = append(errs, detectErrs...)
	return
}

// --- Stage 3: Facts + Relationships -> Patterns ---

func (b *Brain) consolidateToPatterns(ctx context.Context, nsID int64, cp *models.ConsolidationProgress) (count, llmCalls int, errs []string) {
	// Fetch new facts since last pattern extraction
	factSQL, factArgs, err := b.queries.FetchFacts(nsID, cp.LastPatternFactID, 30)
	if err != nil {
		errs = append(errs, fmt.Sprintf("build fetch facts for patterns: %v", err))
		return
	}

	factRows, err := b.pool.Query(ctx, factSQL, factArgs...)
	if err != nil {
		errs = append(errs, fmt.Sprintf("fetch facts for patterns: %v", err))
		return
	}
	defer factRows.Close()

	var facts []models.Fact
	for factRows.Next() {
		var f models.Fact
		if err := factRows.Scan(&f.ID, &f.NamespaceID, &f.Content, &f.Embedding, &f.EmbeddingModel, &f.Confidence, &f.Entity, &f.Property, &f.Value, &f.ValidFrom, &f.ValidUntil, &f.CreatedAt, &f.UpdatedAt); err != nil {
			errs = append(errs, fmt.Sprintf("scan fact for pattern: %v", err))
			continue
		}
		facts = append(facts, f)
	}
	if err := factRows.Err(); err != nil {
		errs = append(errs, fmt.Sprintf("fact rows for patterns: %v", err))
		return
	}

	if len(facts) == 0 {
		return
	}

	// Fetch new relationships since last pattern extraction
	relSQL, relArgs, err := b.queries.FetchRelationships(nsID, cp.LastPatternRelID, 50)
	if err != nil {
		errs = append(errs, fmt.Sprintf("build fetch rels for patterns: %v", err))
		return
	}

	relRows, err := b.pool.Query(ctx, relSQL, relArgs...)
	if err != nil {
		errs = append(errs, fmt.Sprintf("fetch rels for patterns: %v", err))
		return
	}
	defer relRows.Close()

	var rels []models.Relationship
	for relRows.Next() {
		var r models.Relationship
		if err := relRows.Scan(&r.ID, &r.NamespaceID, &r.FromEntity, &r.RelationType, &r.ToEntity, &r.Confidence, &r.SourceFactID, &r.CreatedAt); err != nil {
			errs = append(errs, fmt.Sprintf("scan rel for pattern: %v", err))
			continue
		}
		rels = append(rels, r)
	}
	if err := relRows.Err(); err != nil {
		errs = append(errs, fmt.Sprintf("rel rows for patterns: %v", err))
	}

	if len(rels) == 0 && len(facts) < 3 {
		// Not enough data for pattern extraction
		b.updatePatternCheckpoint(ctx, cp, facts, rels, len(errs) == 0)
		return
	}

	// Call reasoner for pattern extraction
	patterns, err := b.reasoner.ReasonPatterns(ctx, facts, rels)
	llmCalls++
	if err != nil {
		errs = append(errs, fmt.Sprintf("reason patterns: %v", err))
		// Don't update checkpoint on error
		return
	}

	for _, p := range patterns {
		if p.Content == "" {
			continue
		}

		// Confidence = min(source confidences) * coherence_score
		confidence := p.CoherenceScore
		if len(p.SourceFactIDs) > 0 || len(p.SourceRelIDs) > 0 {
			minConf := float32(1.0)
			for _, fid := range p.SourceFactIDs {
				for _, f := range facts {
					if f.ID == fid && f.Confidence < minConf {
						minConf = f.Confidence
					}
				}
			}
			for _, rid := range p.SourceRelIDs {
				for _, r := range rels {
					if r.ID == rid && r.Confidence < minConf {
						minConf = r.Confidence
					}
				}
			}
			confidence = minConf * p.CoherenceScore
		}

		// If no source IDs provided, use all facts/rels as sources
		sourceFactIDs := p.SourceFactIDs
		sourceRelIDs := p.SourceRelIDs
		if len(sourceFactIDs) == 0 && len(sourceRelIDs) == 0 {
			sourceFactIDs = make([]int64, len(facts))
			for i, f := range facts {
				sourceFactIDs[i] = f.ID
			}
			sourceRelIDs = make([]int64, len(rels))
			for i, r := range rels {
				sourceRelIDs[i] = r.ID
			}
		}

		_, err := b.pool.Exec(ctx,
			`INSERT INTO patterns (namespace_id, content, confidence, source_fact_ids, source_rel_ids, coherence_score)
			 VALUES ($1, $2, $3, $4, $5, $6)`,
			nsID, p.Content, confidence, sourceFactIDs, sourceRelIDs, p.CoherenceScore,
		)
		if err != nil {
			errs = append(errs, fmt.Sprintf("insert pattern: %v", err))
			continue
		}
		count++
	}

	// Only update checkpoint if no errors occurred (bullet-proof: prevents losing patterns)
	b.updatePatternCheckpoint(ctx, cp, facts, rels, len(errs) == 0)
	return
}

func (b *Brain) updatePatternCheckpoint(ctx context.Context, cp *models.ConsolidationProgress, facts []models.Fact, rels []models.Relationship, success bool) {
	// Only advance checkpoint if no errors occurred
	if !success {
		return
	}
	for _, f := range facts {
		if f.ID > cp.LastPatternFactID {
			cp.LastPatternFactID = f.ID
		}
	}
	for _, r := range rels {
		if r.ID > cp.LastPatternRelID {
			cp.LastPatternRelID = r.ID
		}
	}
}
