package models

import (
	"encoding/json"
	"testing"

	"github.com/pgvector/pgvector-go"
)

func jsonKeys(t *testing.T, v any) map[string]json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return m
}

// query_facts marshals models.Fact directly. Untagged, the embedding went out
// as 1,536 floats per fact: 40 facts came back as 787,668 characters on prod.
// The other keys stay PascalCase, so no existing key is renamed.
func TestFactJSONOmitsEmbedding(t *testing.T) {
	m := jsonKeys(t, Fact{Content: "c", Embedding: pgvector.NewVector([]float32{1, 2, 3})})
	if _, ok := m["Embedding"]; ok {
		t.Fatalf("Fact JSON must not carry Embedding, got %s", m["Embedding"])
	}
	for _, k := range []string{"ID", "Content", "ValidUntil", "ValidFrom", "EmbeddingModel"} {
		if _, ok := m[k]; !ok {
			t.Errorf("Fact JSON lost key %q: %v", k, m)
		}
	}
}

func TestEpisodeJSONOmitsEmbedding(t *testing.T) {
	m := jsonKeys(t, Episode{Content: "c", Embedding: pgvector.NewVector([]float32{1, 2, 3})})
	if _, ok := m["Embedding"]; ok {
		t.Fatalf("Episode JSON must not carry Embedding, got %s", m["Embedding"])
	}
	for _, k := range []string{"ID", "Content", "OccurredAt"} {
		if _, ok := m[k]; !ok {
			t.Errorf("Episode JSON lost key %q: %v", k, m)
		}
	}
}

func TestOtherEmbeddingCarriersOmitEmbedding(t *testing.T) {
	vec := pgvector.NewVector([]float32{1, 2, 3})
	for name, v := range map[string]any{
		"EmbeddingCache": EmbeddingCache{Text: "t", Embedding: vec},
		"RecallResult":   RecallResult{Content: "c", Embedding: vec},
	} {
		if _, ok := jsonKeys(t, v)["Embedding"]; ok {
			t.Errorf("%s JSON must not carry Embedding", name)
		}
	}
}
