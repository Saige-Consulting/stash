package brain

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/pgvector/pgvector-go"
)

// Writers of a recall result. An episode is the agent's own words, stored
// verbatim by `remember`. A fact with source episodes is the consolidator's
// paraphrase of them. A fact with none was derived by stash itself (a
// confirmed hypothesis or a failure pattern).
const (
	WriterAgent        = "agent"
	WriterConsolidator = "consolidator"
	WriterDerived      = "derived"
)

// recallRollbackTimeout bounds the rollback that ends recall's read-only
// transaction.
const recallRollbackTimeout = 5 * time.Second

// RecallResult is a unified result from semantic search across episodes and facts.
//
// The first block is the original shape and is part of the tool contract: no
// key there may be renamed. Provenance keys are additive and come after it.
type RecallResult struct {
	ID          int64   `json:"id"`
	NamespaceID int64   `json:"namespace_id"`
	Content     string  `json:"content"`
	Confidence  float32 `json:"confidence,omitempty"`
	Score       float32 `json:"score"`
	Type        string  `json:"type"`
	OccurredAt  string  `json:"occurred_at,omitempty"`
	ValidFrom   string  `json:"valid_from,omitempty"`
	CreatedAt   string  `json:"created_at"`

	// Namespace is the slug the result is filed under.
	Namespace string `json:"namespace"`
	// Writer is WriterAgent, WriterConsolidator or WriterDerived.
	Writer string `json:"writer"`
	// SourceEpisodeIDs are the episodes a fact came from (at most 10, lowest
	// first; the cap is in recall_facts); for an episode, its own id.
	SourceEpisodeIDs []int64 `json:"source_episode_ids"`
	// SourceEpisodeCount is the total number of source episodes. Facts only.
	SourceEpisodeCount *int64 `json:"source_episode_count,omitempty"`
	// ValidUntil is set only when the fact has an end of validity.
	ValidUntil string `json:"valid_until,omitempty"`
	// Superseded is true only for a fact whose valid_until has passed; such
	// facts are returned only with RecallOptions.IncludeSuperseded.
	Superseded bool `json:"superseded,omitempty"`
}

// RecallOptions are optional recall switches. The zero value is the default.
type RecallOptions struct {
	// IncludeSuperseded also returns facts whose valid_until has passed, for
	// an audit trail. They are marked Superseded.
	IncludeSuperseded bool
}

// Recall searches episodes and facts by semantic similarity across the given namespaces.
// Each namespace path matches itself and all descendants. Namespaces is required.
// Superseded facts are left out; see RecallWithOptions.
func (b *Brain) Recall(ctx context.Context, namespaces []string, query string, limit int) ([]RecallResult, error) {
	return b.RecallWithOptions(ctx, namespaces, query, limit, RecallOptions{})
}

// RecallWithOptions is Recall with optional switches.
func (b *Brain) RecallWithOptions(ctx context.Context, namespaces []string, query string, limit int, opts RecallOptions) ([]RecallResult, error) {
	if err := validateContent(query); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 10
	}
	if limit > 100 {
		limit = 100
	}

	vec, err := b.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	pgVec := pgvector.NewVector(vec)

	nsIDs, err := b.resolveNamespaceIDs(ctx, namespaces)
	if err != nil {
		return nil, err
	}

	// Both searches run in one read-only transaction so the iterative-scan
	// setting below applies to them and to nothing else on the pooled
	// connection. Nothing is written, so the transaction is always rolled
	// back.
	tx, err := b.pool.BeginTx(ctx, pgx.TxOptions{AccessMode: pgx.ReadOnly})
	if err != nil {
		return nil, fmt.Errorf("begin recall: %w", err)
	}
	defer func() {
		// Roll back even if the caller's context is already cancelled, so the
		// pooled connection is returned clean, but never block on it for long.
		rbCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), recallRollbackTimeout)
		defer cancel()
		_ = tx.Rollback(rbCtx)
	}()

	// A whole-store recall is planned as an HNSW index scan, and the index
	// hands back only ef_search candidates before the validity filter runs.
	// With most near neighbours superseded, that returned fewer than limit
	// facts. Iterative scan keeps reading the index until limit rows pass
	// the filter. relaxed_order may return them slightly out of order; the
	// sort below restores it.
	//
	// Iterative scan gives up after hnsw.max_scan_tuples index tuples
	// (pgvector default 20,000), so a very selective namespace filter on a
	// large store can still return fewer than limit. It is left at the
	// default on purpose: a whole-store recall needs about limit divided by
	// the live fraction (roughly 30 tuples for 10 results at prod's ~66%
	// superseded), and a higher cap only makes a miss slower.
	if _, err := tx.Exec(ctx, "SET LOCAL hnsw.iterative_scan = relaxed_order"); err != nil {
		return nil, fmt.Errorf("enable hnsw iterative scan: %w", err)
	}

	// Search facts first (higher quality, consolidated)
	results, err := b.recallFacts(ctx, tx, nsIDs, pgVec, limit, opts.IncludeSuperseded)
	if err != nil {
		return nil, err
	}

	// Search episodes for remaining slots
	if episodeLimit := limit - len(results); episodeLimit > 0 {
		episodes, err := b.recallEpisodes(ctx, tx, nsIDs, pgVec, episodeLimit)
		if err != nil {
			return nil, err
		}
		results = append(results, episodes...)
	}

	// Sort all results by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

func (b *Brain) recallFacts(ctx context.Context, tx pgx.Tx, nsIDs []int64, vec pgvector.Vector, limit int, includeSuperseded bool) ([]RecallResult, error) {
	sql, args, err := b.queries.RecallFacts(nsIDs, vec, limit, includeSuperseded)
	if err != nil {
		return nil, fmt.Errorf("build fact query: %w", err)
	}
	rows, err := tx.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("query facts: %w", err)
	}
	defer rows.Close()

	var results []RecallResult
	for rows.Next() {
		var (
			r                     RecallResult
			validFrom, validUntil *time.Time
			sourceIDs             []int64
			sourceCount           int64
			createdAt             time.Time
		)
		if err := rows.Scan(
			&r.ID, &r.NamespaceID, &r.Namespace, &r.Content, &r.Confidence,
			&validFrom, &validUntil, &r.Superseded,
			&sourceIDs, &sourceCount,
			&createdAt, &r.Score,
		); err != nil {
			return nil, fmt.Errorf("scan fact: %w", err)
		}
		r.Type = "fact"
		r.CreatedAt = createdAt.Format(time.RFC3339)
		if validFrom != nil {
			r.ValidFrom = validFrom.Format(time.RFC3339)
		}
		if validUntil != nil {
			r.ValidUntil = validUntil.Format(time.RFC3339)
		}
		if sourceIDs == nil {
			sourceIDs = []int64{}
		}
		r.SourceEpisodeIDs = sourceIDs
		r.SourceEpisodeCount = &sourceCount
		r.Writer = WriterDerived
		if sourceCount > 0 {
			r.Writer = WriterConsolidator
		}
		results = append(results, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("fact rows: %w", err)
	}
	return results, nil
}

func (b *Brain) recallEpisodes(ctx context.Context, tx pgx.Tx, nsIDs []int64, vec pgvector.Vector, limit int) ([]RecallResult, error) {
	sql, args, err := b.queries.RecallEpisodes(nsIDs, vec, limit)
	if err != nil {
		return nil, fmt.Errorf("build episode query: %w", err)
	}
	rows, err := tx.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("query episodes: %w", err)
	}
	defer rows.Close()

	var results []RecallResult
	for rows.Next() {
		var (
			r                     RecallResult
			occurredAt, createdAt time.Time
		)
		if err := rows.Scan(&r.ID, &r.NamespaceID, &r.Namespace, &r.Content, &occurredAt, &createdAt, &r.Score); err != nil {
			return nil, fmt.Errorf("scan episode: %w", err)
		}
		r.Type = "episode"
		r.OccurredAt = occurredAt.Format(time.RFC3339)
		r.CreatedAt = createdAt.Format(time.RFC3339)
		r.Writer = WriterAgent
		r.SourceEpisodeIDs = []int64{r.ID}
		results = append(results, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("episode rows: %w", err)
	}
	return results, nil
}
