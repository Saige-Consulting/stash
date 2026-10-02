package brain

// Stage-1 per-episode outcomes and the give-up budget (consolidation_episode_state,
// migration 00022).
//
// An episode's stage-1 outcome is final when it yielded a fact (a fact_sources row), or when
// one of the outcomes below is recorded for it. Either way it reads as already mined and is
// never sent to the reasoner again. Keeping these outcomes only in memory, as the first cut of
// the per-episode checkpoint did, re-extracted them on every pass while anything below them
// pinned the checkpoint.

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/alash3al/stash/internal/models"
	"github.com/alash3al/stash/internal/reasoner"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/pgvector/pgvector-go"
)

const (
	// maxEpisodeAttempts and episodeGiveUpAfter bound how long a failing episode can pin its
	// namespace's checkpoint. Both must hold before giving up: the attempt count alone would
	// let a burst of failures skip an episode within minutes, and the time floor alone would
	// let a single failure do it.
	maxEpisodeAttempts = 3
	episodeGiveUpAfter = 24 * time.Hour

	outcomeEmpty    = "empty"    // the reasoner found nothing to extract
	outcomeUnusable = "unusable" // reasoner.ErrUnusableOutput for this text
	outcomeRejected = "rejected" // reasoner.ErrInputRejected for this episode on its own
	outcomeGaveUp   = "gave_up"  // failed maxEpisodeAttempts times over episodeGiveUpAfter

	maxStoredErrorLen = 1000 // characters of last_error kept per episode
)

// countsAgainstEpisodes reports whether a stage-1 failure says something about the episodes
// themselves, and so counts toward giving up on them.
//
// Anything that is a property of the moment never counts: the model service being down,
// refusing the key or rate limiting (reasoner.IsUnavailable), the network, a cancelled pass,
// or the database being unreachable, overloaded or in a lock conflict. Otherwise an outage
// longer than episodeGiveUpAfter would make the consolidator give up on every episode it
// touched. Of database errors, only data (22) and integrity (23) errors are about the rows.
func countsAgainstEpisodes(err error) bool {
	if reasoner.IsUnavailable(err) {
		return false
	}
	var connErr *pgconn.ConnectError
	if errors.As(err, &connErr) {
		return false
	}
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		return strings.HasPrefix(pgErr.Code, "22") || strings.HasPrefix(pgErr.Code, "23")
	}
	return true
}

// --- reading a batch ---

// unreadableEpisode is a fetched row whose id is known but whose columns would not scan.
type unreadableEpisode struct {
	id  int64
	err error
}

// episodeBatch is one fetch_episodes result, split by what stage 1 must do with each row.
type episodeBatch struct {
	ids        []int64             // every row with a known id, in fetch (ascending id) order
	unmined    []models.Episode    // readable rows with no final outcome yet
	mined      []int64             // rows that already have a fact source or a final outcome
	unreadable []unreadableEpisode // a row that would not scan; it holds the checkpoint
	cutErr     error               // a row whose id could not be recovered; the batch ends before it
}

// readEpisodeBatch reads a fetch_episodes result.
//
// pgx closes the result set on a scan error, so the rows after an unreadable one are not
// available this pass. The unreadable row's id is recovered from its raw value, so it holds
// the checkpoint just before it and its failure counts toward giving up on it, rather than
// blocking the namespace for good. Without an id the batch simply ends before the row, which is
// safe: every id in the batch is lower, so the checkpoint cannot pass it.
func readEpisodeBatch(rows pgx.Rows) (episodeBatch, error) {
	var batch episodeBatch
	for rows.Next() {
		// Capture the id's raw bytes before Scan: a failed Scan closes the rows, and closing
		// drains the connection, which may reuse the buffer the raw values point into.
		var rawID []byte
		var idField pgconn.FieldDescription
		if raw, fds := rows.RawValues(), rows.FieldDescriptions(); len(raw) > 0 && len(fds) > 0 && raw[0] != nil {
			rawID = append([]byte(nil), raw[0]...)
			idField = fds[0]
		}

		e, mined, err := scanFetchedEpisode(rows)
		if err != nil {
			var id int64
			if rawID != nil && pgtype.NewMap().Scan(idField.DataTypeOID, idField.Format, rawID, &id) == nil {
				batch.ids = append(batch.ids, id)
				batch.unreadable = append(batch.unreadable, unreadableEpisode{id: id, err: err})
			} else {
				batch.cutErr = err
			}
			return batch, nil
		}

		batch.ids = append(batch.ids, e.ID)
		if mined {
			batch.mined = append(batch.mined, e.ID)
			continue
		}
		batch.unmined = append(batch.unmined, e)
	}
	return batch, rows.Err()
}

// scanFetchedEpisode scans one fetch_episodes row. embedding and embedding_model are nullable
// (the failure stage writes REPEAT FAILURE episodes without them) and pgvector.Vector rejects
// NULL, so both go through pointers. A missing embedding leaves the episode as its own cluster.
func scanFetchedEpisode(rows pgx.Rows) (e models.Episode, alreadyMined bool, err error) {
	var embedding *pgvector.Vector
	var embeddingModel *string
	if err = rows.Scan(&e.ID, &e.NamespaceID, &e.Content, &embedding, &embeddingModel, &e.OccurredAt, &e.CreatedAt, &alreadyMined); err != nil {
		return e, false, err
	}
	if embedding != nil {
		e.Embedding = *embedding
	}
	if embeddingModel != nil {
		e.EmbeddingModel = *embeddingModel
	}
	return e, alreadyMined, nil
}

// --- recording outcomes ---

// finishEpisodes records a final outcome that created no fact, then marks the episodes done.
// If the outcome cannot be recorded the episodes stay not done: an outcome that lives only in
// memory is re-extracted on the next pass.
func (b *Brain) finishEpisodes(ctx context.Context, ids []int64, outcome string, cause error, done map[int64]bool, res *episodeStageResult) {
	var lastError *string
	if cause != nil {
		s := truncateRunes(cause.Error(), maxStoredErrorLen)
		lastError = &s
	}
	_, err := b.pool.Exec(ctx,
		`INSERT INTO consolidation_episode_state AS st (episode_id, outcome, last_error)
		 SELECT id, $2, $3 FROM episodes WHERE id = ANY($1)
		 ON CONFLICT (episode_id) DO UPDATE SET
		     outcome = EXCLUDED.outcome,
		     last_error = COALESCE(EXCLUDED.last_error, st.last_error),
		     updated_at = now()`,
		ids, outcome, lastError,
	)
	if err != nil {
		res.errs = append(res.errs, fmt.Sprintf("record %s outcome for episodes %v: %v", outcome, ids, err))
		return
	}
	for _, id := range ids {
		done[id] = true
	}
	if outcome == outcomeUnusable || outcome == outcomeRejected {
		res.skipped += len(ids)
		res.errs = append(res.errs, fmt.Sprintf("skipped episodes %v (%s): %v", ids, outcome, cause))
	}
}

// failEpisodes reports a failure that leaves the episodes not done. If the failure says
// something about the episodes (countsAgainstEpisodes), it is counted, and once an episode has
// failed maxEpisodeAttempts times with the first failure at least episodeGiveUpAfter ago, it is
// given up on: recorded as gave_up and marked done, so the checkpoint can move past it.
func (b *Brain) failEpisodes(ctx context.Context, ids []int64, cause error, done map[int64]bool, res *episodeStageResult) {
	if !countsAgainstEpisodes(cause) {
		res.errs = append(res.errs, cause.Error())
		return
	}

	rows, err := b.pool.Query(ctx,
		`WITH prev AS (
		     SELECT episode_id, outcome FROM consolidation_episode_state WHERE episode_id = ANY($1::bigint[])
		 ), up AS (
		     INSERT INTO consolidation_episode_state AS st
		         (episode_id, attempts, first_failed_at, last_failed_at, last_error, outcome, updated_at)
		     SELECT e.id, 1, now(), now(), $2::text,
		            CASE WHEN 1 >= $3::int AND $4::float8 <= 0 THEN 'gave_up' END, now()
		     FROM episodes e WHERE e.id = ANY($1::bigint[])
		     ON CONFLICT (episode_id) DO UPDATE SET
		         attempts        = st.attempts + 1,
		         first_failed_at = COALESCE(st.first_failed_at, now()),
		         last_failed_at  = now(),
		         last_error      = EXCLUDED.last_error,
		         outcome         = CASE
		             WHEN st.outcome IS NOT NULL THEN st.outcome
		             WHEN st.attempts + 1 >= $3::int
		              AND COALESCE(st.first_failed_at, now()) <= now() - make_interval(secs => $4::float8)
		             THEN 'gave_up'
		         END,
		         updated_at      = now()
		     RETURNING episode_id, attempts, first_failed_at, outcome
		 )
		 SELECT up.episode_id, up.attempts, up.first_failed_at, up.outcome IS NOT NULL, prev.outcome IS NOT NULL
		 FROM up LEFT JOIN prev ON prev.episode_id = up.episode_id`,
		ids, truncateRunes(cause.Error(), maxStoredErrorLen), maxEpisodeAttempts, episodeGiveUpAfter.Seconds(),
	)
	if err != nil {
		res.errs = append(res.errs, cause.Error(), fmt.Sprintf("record failed attempt for episodes %v: %v", ids, err))
		return
	}
	defer rows.Close()

	var gaveUp []int64
	var attempts int
	var since time.Time
	for rows.Next() {
		var id int64
		var n int
		var first time.Time
		var final, wasFinal bool
		if err := rows.Scan(&id, &n, &first, &final, &wasFinal); err != nil {
			res.errs = append(res.errs, cause.Error(), fmt.Sprintf("read failed attempt for episodes %v: %v", ids, err))
			return
		}
		if !final {
			continue
		}
		// Final either now or already (an unreadable row given up on earlier).
		done[id] = true
		if !wasFinal {
			gaveUp = append(gaveUp, id)
			attempts, since = n, first
		}
	}
	if err := rows.Err(); err != nil {
		res.errs = append(res.errs, cause.Error(), fmt.Sprintf("failed attempt rows for episodes %v: %v", ids, err))
		return
	}

	if len(gaveUp) == 0 {
		res.errs = append(res.errs, cause.Error())
		return
	}
	res.gaveUp += len(gaveUp)
	res.errs = append(res.errs, fmt.Sprintf("gave up on episodes %v after %d attempts since %s: %v",
		gaveUp, attempts, since.UTC().Format(time.RFC3339), cause))
}

func truncateRunes(s string, limit int) string {
	r := []rune(s)
	if len(r) <= limit {
		return s
	}
	return string(r[:limit])
}
