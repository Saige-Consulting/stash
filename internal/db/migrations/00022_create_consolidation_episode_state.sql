-- Per-episode stage-1 consolidation state.
--
-- outcome records an extraction that ended WITHOUT a fact and is final:
--   empty     the reasoner found nothing to extract
--   unusable  its output was unusable for this text (bad JSON, failed grounding)
--   rejected  the provider refused this episode on its own (too long, content filter)
--   gave_up   it failed attempts >= 3 times, the first at least 24h ago
-- An episode with an outcome, or with a fact_sources row, is "already mined" and is
-- never sent to the reasoner again. A row with outcome NULL is a pending failure history.
-- Undo a give-up by deleting its row (and lowering consolidation_progress.last_episode_id
-- below it): the episode is then retried.
-- +goose Up
CREATE TABLE IF NOT EXISTS consolidation_episode_state (
    episode_id      BIGINT      NOT NULL PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    attempts        INTEGER     NOT NULL DEFAULT 0,
    first_failed_at TIMESTAMPTZ NULL,
    last_failed_at  TIMESTAMPTZ NULL,
    last_error      TEXT        NULL,
    outcome         TEXT        NULL CHECK (outcome IN ('empty', 'unusable', 'rejected', 'gave_up')),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- +goose Down
DROP TABLE IF EXISTS consolidation_episode_state;
