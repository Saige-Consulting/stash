package brain

import (
	"context"
	"testing"
)

// confirm_hypothesis returns the fact it creates. The embedding is never
// output (models.Fact tags it json:"-"), so it must not be read either.
func TestConfirmHypothesis_DoesNotReadEmbedding(t *testing.T) {
	e := newFakeEmbedder()
	b := openTestBrain(t, newFakeReasoner(t), e)
	ctx := context.Background()

	ns := mustNamespace(t, b, "/hypotheses")
	h, err := b.CreateHypothesis(ctx, ns, "Claims tickets close within a day.", "check closed tickets", 0.6, nil)
	if err != nil {
		t.Fatalf("CreateHypothesis: %v", err)
	}
	if _, err := b.UpdateHypothesisStatus(ctx, h.ID, "testing"); err != nil {
		t.Fatalf("UpdateHypothesisStatus: %v", err)
	}
	_, f, err := b.ConfirmHypothesis(ctx, h.ID)
	if err != nil {
		t.Fatalf("ConfirmHypothesis: %v", err)
	}
	if n := len(f.Embedding.Slice()); n != 0 {
		t.Errorf("confirmed fact: embedding read back (%d floats); it is never output", n)
	}
	if f.ID == 0 || f.Content != "Claims tickets close within a day." || f.EmbeddingModel != "fake-embed" || f.ValidFrom == nil {
		t.Errorf("confirmed fact lost a column: %+v", f)
	}
	if got := queryInt(t, b, "SELECT count(*) FROM facts WHERE id = $1 AND embedding IS NOT NULL", f.ID); got != 1 {
		t.Errorf("the stored fact must still have its embedding (for recall); count = %d", got)
	}
}
