package brain

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/alash3al/stash/internal/embedder"
	"github.com/alash3al/stash/internal/queries"
	"github.com/alash3al/stash/internal/reasoner"
	"github.com/jackc/pgx/v5/pgxpool"
)

var (
	ErrNamespaceNotFound = fmt.Errorf("brain: namespace not found — call create_namespace first")
	ErrEpisodeNotFound   = fmt.Errorf("brain: episode not found")
	ErrFactNotFound      = fmt.Errorf("brain: fact not found")
	ErrEmptyContent      = fmt.Errorf("brain: content cannot be empty")
	ErrContentTooLong    = fmt.Errorf("brain: content exceeds maximum length")
	ErrInvalidPath = fmt.Errorf("brain: namespace path must start with / and contain valid segments (alphanumeric, hyphens, underscores)")

	// Case-tolerant on purpose: slugs are canonicalised by normalizeSlug at
	// every point that touches the database (create and all resolve paths), so
	// validation must not reject casing that resolution would have handled.
	// It used to be lowercase-only, which made an uppercase path fail validation
	// BEFORE normalisation could run — a caller asking for /people/U08A83MEMKN
	// got ErrInvalidPath even though /people/u08a83memkn existed with 73 facts.
	pathSegmentRe = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`)
	ErrNamespacesRequired = fmt.Errorf("brain: at least one namespace is required")
	maxContentLen = 10000
)

const (
	DefaultLimit = 100
	MaxLimit     = 1000
)

// Pagination controls offset-based pagination for list queries.
type Pagination struct {
	Offset int `json:"offset"`
	Limit  int `json:"limit"`
}

// Sanitize applies defaults and enforces bounds.
func (p Pagination) Sanitize() Pagination {
	if p.Limit <= 0 {
		p.Limit = DefaultLimit
	}
	if p.Limit > MaxLimit {
		p.Limit = MaxLimit
	}
	if p.Offset < 0 {
		p.Offset = 0
	}
	return p
}

type Config struct {
	BatchSize           int
	SimilarityThreshold float64
	DedupThreshold      float64
	Window              time.Duration
	DecayFactor                    float64
	ExpiryThreshold                float32
	HypothesisAutoConfirmThreshold float32
	HypothesisAutoRejectThreshold  float32
}

func DefaultConfig() Config {
	return Config{
		BatchSize:                      100,
		SimilarityThreshold:            0.85,
		DedupThreshold:                 0.95,
		Window:                         7 * 24 * time.Hour,
		DecayFactor:                    0.95,
		ExpiryThreshold:                0.1,
		HypothesisAutoConfirmThreshold: 0.9,
		HypothesisAutoRejectThreshold:  0.9,
	}
}

type Brain struct {
	pool     *pgxpool.Pool
	embedder embedder.Embedder
	reasoner reasoner.Reasoner
	queries  *queries.Queries
	config   Config
}

func New(pool *pgxpool.Pool, e embedder.Embedder, r reasoner.Reasoner, q *queries.Queries, cfg Config) (*Brain, error) {
	if pool == nil {
		return nil, fmt.Errorf("brain: pool is required")
	}
	if e == nil {
		return nil, fmt.Errorf("brain: embedder is required")
	}
	if r == nil {
		return nil, fmt.Errorf("brain: reasoner is required")
	}
	if q == nil {
		return nil, fmt.Errorf("brain: queries is required")
	}
	return &Brain{
		pool:     pool,
		embedder: e,
		reasoner: r,
		queries:  q,
		config:   cfg,
	}, nil
}

func (b *Brain) Close() {
	b.pool.Close()
}

func validateContent(content string) error {
	if content == "" {
		return ErrEmptyContent
	}
	if len(content) > maxContentLen {
		return ErrContentTooLong
	}
	return nil
}

// normalizeSlug canonicalises a namespace path so a slug written in one casing
// still resolves when read in another.
//
// Namespaces are matched exactly (`WHERE slug = $1`) while writers disagree on
// case. Observed live: the NSB agent's Stop hook lowercases Slack user IDs and
// stored `/people/u08a83memkn`, but the agent recalling `/people/<user_id>`
// renders the ID uppercase from its channel tag and asked for
// `/people/U08A83MEMKN`. Every such read missed and returned "namespace not
// found", which the agent is told to treat as no-context-yet — so a per-asker
// profile layer accumulated 229 facts that were never once read back.
//
// Lowercasing both ends fixes it without a migration: every slug already stored
// is lowercase, so normalising the query input can only turn misses into hits.
func normalizeSlug(path string) string {
	return strings.ToLower(strings.TrimSpace(path))
}

func validatePath(path string) error {
	if path == "" || path == "/" {
		return nil
	}
	if !strings.HasPrefix(path, "/") {
		return ErrInvalidPath
	}
	segments := splitPath(path)
	for _, seg := range segments {
		if !pathSegmentRe.MatchString(seg) {
			return ErrInvalidPath
		}
	}
	return nil
}

// splitPath splits a path into its segments.
// "/" → [], "/foo/bar" → ["foo", "bar"]
func splitPath(path string) []string {
	path = strings.TrimPrefix(path, "/")
	path = strings.TrimSuffix(path, "/")
	if path == "" {
		return nil
	}
	return strings.Split(path, "/")
}

// resolveNamespaceID returns the namespace ID for an exact path.
// Returns ErrNamespaceNotFound if no matching namespace exists.
func (b *Brain) resolveNamespaceID(ctx context.Context, path string) (int64, error) {
	path = normalizeSlug(path)
	var id int64
	err := b.pool.QueryRow(ctx,
		"SELECT id FROM namespaces WHERE slug = $1", path,
	).Scan(&id)
	if err != nil {
		return 0, ErrNamespaceNotFound
	}
	return id, nil
}

// ResolveNamespaceIDs resolves a list of paths to namespace IDs.
// Each path matches itself and all descendants. For "/", returns all IDs.
// Returns ErrNamespacesRequired if paths is empty.
func (b *Brain) ResolveNamespaceIDs(ctx context.Context, paths []string) ([]int64, error) {
	return b.resolveNamespaceIDs(ctx, paths)
}

// resolveNamespaceIDs resolves a list of paths to namespace IDs.
// Each path matches itself and all descendants.
func (b *Brain) resolveNamespaceIDs(ctx context.Context, paths []string) ([]int64, error) {
	if len(paths) == 0 {
		return nil, ErrNamespacesRequired
	}

	var allIDs []int64
	seen := make(map[int64]bool)

	for _, p := range paths {
		if err := validatePath(p); err != nil {
			return nil, fmt.Errorf("namespace %q: %w", p, err)
		}
		expanded, err := b.resolveNamespaceIDWithDescendants(ctx, p)
		if err != nil {
			return nil, fmt.Errorf("namespace %q: %w", p, err)
		}
		for _, id := range expanded {
			if !seen[id] {
				seen[id] = true
				allIDs = append(allIDs, id)
			}
		}
	}

	return allIDs, nil
}

// resolveNamespaceIDWithDescendants returns IDs for the exact path plus all descendants.
// For "/", returns all namespace IDs.
func (b *Brain) resolveNamespaceIDWithDescendants(ctx context.Context, path string) ([]int64, error) {
	path = normalizeSlug(path)
	if path == "/" {
		rows, err := b.pool.Query(ctx, "SELECT id FROM namespaces")
		if err != nil {
			return nil, fmt.Errorf("resolve all namespaces: %w", err)
		}
		defer rows.Close()

		var ids []int64
		for rows.Next() {
			var id int64
			if err := rows.Scan(&id); err != nil {
				return nil, fmt.Errorf("scan namespace: %w", err)
			}
			ids = append(ids, id)
		}
		return ids, rows.Err()
	}

	rows, err := b.pool.Query(ctx,
		"SELECT id FROM namespaces WHERE slug = $1 OR slug LIKE $2",
		path, path+"/%",
	)
	if err != nil {
		return nil, fmt.Errorf("resolve namespace descendants: %w", err)
	}
	defer rows.Close()

	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan namespace: %w", err)
		}
		ids = append(ids, id)
	}

	if len(ids) == 0 {
		return nil, ErrNamespaceNotFound
	}

	return ids, rows.Err()
}

func (b *Brain) Health(ctx context.Context) error {
	return b.pool.Ping(ctx)
}

func (b *Brain) Ready(ctx context.Context) error {
	_, err := b.pool.Exec(ctx, "SELECT 1 FROM consolidation_progress LIMIT 0")
	if err != nil {
		return fmt.Errorf("ready: %w", err)
	}
	return b.pool.Ping(ctx)
}
