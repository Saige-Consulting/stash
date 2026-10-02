-- Serves the per-episode "already mined" test in stage-1 consolidation
-- (fetch_episodes) and the same-source-episode guard in contradiction
-- detection. The primary key (fact_id, episode_id) cannot serve a lookup by
-- episode_id alone.
-- +goose Up
CREATE INDEX IF NOT EXISTS fact_sources_episode_id_idx ON fact_sources (episode_id);
-- +goose Down
DROP INDEX IF EXISTS fact_sources_episode_id_idx;
