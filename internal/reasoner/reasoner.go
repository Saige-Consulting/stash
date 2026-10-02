// Package reasoner synthesizes structured reasoning over text.
package reasoner

import (
	"context"
	"errors"
	"net"

	"github.com/alash3al/stash/internal/models"
	"github.com/openai/openai-go"
)

// ErrUnusableOutput marks a model answer that cannot be used for this input:
// it would not parse, or it failed the grounding check (it contains words the
// input does not). It is permanent for this input: retrying the same text will
// not help, so a caller may give up on the input instead of retrying it on
// every pass. Transport and API errors (HTTP status, timeouts, cancellation)
// are never wrapped with it; they say nothing about the input and are
// transient.
//
// Test with errors.Is; the error text carries the specific cause.
var ErrUnusableOutput = errors.New("reasoner: unusable model output")

// ErrInputRejected marks a request the provider refused because of the input
// itself: it is too long for the model, or a content policy blocked it.
// Retrying the same text gets the same answer, so it is permanent for that
// text, but a smaller input (one episode instead of a cluster) may succeed.
var ErrInputRejected = errors.New("reasoner: input rejected by the provider")

// ErrUnavailable marks a request the provider could not serve at all: an
// outage, an auth, quota or rate-limit refusal, a missing model, a network
// failure or a cancelled context. It says nothing about the input, so callers
// must never count it against the input. See IsUnavailable.
var ErrUnavailable = errors.New("reasoner: model service unavailable")

// StructuredFact represents an extracted fact with entity, property, and value.
type StructuredFact struct {
	Entity   string
	Property string
	Value    string
	Summary  string
}

// StructuredRelationship represents an extracted relationship between two entities.
type StructuredRelationship struct {
	FromEntity   string
	RelationType string
	ToEntity     string
	Confidence   float32
}

// StructuredPattern represents an abstract pattern derived from facts and relationships.
type StructuredPattern struct {
	Content        string
	CoherenceScore float32
	SourceFactIDs  []int64
	SourceRelIDs   []int64
}

// ContradictionClassification is the result of classifying a fact pair.
type ContradictionClassification string

const (
	ClassificationReplacement   ContradictionClassification = "replacement"
	ClassificationContradiction ContradictionClassification = "contradiction"
	ClassificationCompatible    ContradictionClassification = "compatible"
)

// ContradictionResult is the LLM output for classifying two facts about the same entity+property.
type ContradictionResult struct {
	Classification ContradictionClassification
	Confidence     float32
	Explanation    string
}

// StructuredCausalLink represents an extracted cause-effect relationship between two facts.
type StructuredCausalLink struct {
	CauseFactID  int64
	EffectFactID int64
	Confidence   float32
}

// GoalProgressAssessment is the LLM output for one goal against a batch of facts.
type GoalProgressAssessment struct {
	GoalID     int64
	Assessment string  // "progress", "suggested_complete", "contradicted", "irrelevant"
	Note       string
	Confidence float32
}

// FailurePatternResult covers repetition detection and pattern extraction.
type FailurePatternResult struct {
	Type        string  // "repetition" or "pattern"
	FailureID   int64   // For repetition: the original failure ID
	Evidence    string  // For repetition: what evidence suggests the repeat
	PatternFact string  // For pattern: the extracted higher-order fact content
	Confidence  float32
}

// HypothesisEvidenceResult is the LLM output for one hypothesis against a batch of facts.
type HypothesisEvidenceResult struct {
	HypothesisID  int64
	Verdict       string  // "supports", "weakens", "contradicts", "irrelevant"
	Confidence    float32
	Reasoning     string
	NewConfidence float32
}

// Reasoner synthesizes structured reasoning over text input.
type Reasoner interface {
	// ReasonStructured takes a list of text inputs and returns a structured fact.
	ReasonStructured(ctx context.Context, texts []string) (*StructuredFact, error)

	// ReasonRelationships takes a fact and extracts relationships between entities.
	ReasonRelationships(ctx context.Context, factContent string) ([]*StructuredRelationship, error)

	// ReasonPatterns takes facts and relationships and extracts abstract patterns.
	ReasonPatterns(ctx context.Context, facts []models.Fact, relationships []models.Relationship) ([]*StructuredPattern, error)

	// ReasonContradiction classifies whether a new fact replaces, contradicts, or is compatible with an old fact.
	ReasonContradiction(ctx context.Context, entity, property, oldValue, newValue string) (*ContradictionResult, error)

	// ReasonCausalLinks takes a batch of facts and extracts cause-effect pairs.
	ReasonCausalLinks(ctx context.Context, facts []models.Fact) ([]*StructuredCausalLink, error)

	// ReasonGoalProgress assesses whether recent facts indicate progress, completion, or contradiction of active goals.
	ReasonGoalProgress(ctx context.Context, goals []models.Goal, facts []models.Fact) ([]*GoalProgressAssessment, error)

	// ReasonFailurePatterns detects whether recent evidence repeats past failures, and extracts higher-order failure patterns.
	ReasonFailurePatterns(ctx context.Context, failures []models.Failure, evidence []string) ([]*FailurePatternResult, error)

	// ReasonHypothesisEvidence assesses whether new evidence supports, weakens, or contradicts open hypotheses.
	ReasonHypothesisEvidence(ctx context.Context, hypotheses []models.Hypothesis, facts []models.Fact) ([]*HypothesisEvidenceResult, error)
}

// IsUnavailable reports whether err means the model service could not serve
// the request at all (see ErrUnavailable), as opposed to anything about the
// input. It classifies errors already tagged by this package and also raw
// errors from any openai-go client, such as the embedder, and the transport.
func IsUnavailable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrUnavailable) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return !isInputStatus(apiErr.StatusCode)
	}
	var netErr net.Error
	return errors.As(err, &netErr)
}

// isInputStatus reports whether an API status is about the request itself.
// Everything else (401, 402, 403, 404, 408, 409, 429, 5xx, ...) is the
// service, the account or the configuration, and fails every input alike.
func isInputStatus(status int) bool {
	switch status {
	case 400, 413, 422:
		return true
	}
	return false
}
