package brain

import (
	"encoding/json"
	"testing"
)

// recallKeysBefore are the keys recall returned before provenance was added.
// Agents and skills read them by name, so none may be renamed or dropped.
var recallKeysBefore = []string{
	"id", "namespace_id", "content", "confidence", "score", "type",
	"occurred_at", "valid_from", "created_at",
}

func recallKeys(t *testing.T, r RecallResult) map[string]json.RawMessage {
	t.Helper()
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return m
}

func TestRecallResultJSON_BackwardCompatibleKeys(t *testing.T) {
	count := int64(2)
	populated := RecallResult{
		ID:                 7,
		NamespaceID:        3,
		Content:            "c",
		Confidence:         0.5,
		Score:              0.9,
		Type:               "fact",
		OccurredAt:         "2026-10-01T00:00:00Z",
		ValidFrom:          "2026-10-01T00:00:00Z",
		CreatedAt:          "2026-10-01T00:00:00Z",
		Namespace:          "/findings/x",
		Writer:             "consolidator",
		SourceEpisodeIDs:   []int64{1, 2},
		SourceEpisodeCount: &count,
		ValidUntil:         "2026-10-02T00:00:00Z",
		Superseded:         true,
	}
	m := recallKeys(t, populated)
	for _, k := range recallKeysBefore {
		if _, ok := m[k]; !ok {
			t.Errorf("existing key %q missing from %v", k, m)
		}
	}
	for _, k := range []string{"namespace", "writer", "source_episode_ids", "source_episode_count", "valid_until", "superseded"} {
		if _, ok := m[k]; !ok {
			t.Errorf("provenance key %q missing from %v", k, m)
		}
	}
	if got := string(m["source_episode_ids"]); got != "[1,2]" {
		t.Errorf("source_episode_ids = %s, want [1,2]", got)
	}

	// A live fact carries no valid_until and no superseded flag; an episode
	// carries no source_episode_count.
	live := recallKeys(t, RecallResult{ID: 1, Type: "episode", Writer: "agent", SourceEpisodeIDs: []int64{1}})
	for _, k := range []string{"superseded", "valid_until", "source_episode_count"} {
		if _, ok := live[k]; ok {
			t.Errorf("key %q must be absent when unset, got %s", k, live[k])
		}
	}
}
