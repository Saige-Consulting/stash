package brain

import "testing"

// Namespaces are matched with `WHERE slug = $1`, so a slug written in one casing
// must not become unreadable in another. This was a live silent failure: the NSB
// agent's Stop hook stored `/people/u08a83memkn` while the agent recalled
// `/people/U08A83MEMKN` from its Slack channel tag, so every per-asker read
// returned "namespace not found" and the profile layer was write-only.
func TestNormalizeSlug(t *testing.T) {
	cases := map[string]string{
		"/people/U08A83MEMKN":   "/people/u08a83memkn",
		"/people/u08a83memkn":   "/people/u08a83memkn",
		"/FINDINGS/Jira":        "/findings/jira",
		"  /findings/jira  ":    "/findings/jira",
		"/threads/1783634494-1": "/threads/1783634494-1",
		"/":                     "/",
		"":                      "",
	}
	for in, want := range cases {
		if got := normalizeSlug(in); got != want {
			t.Errorf("normalizeSlug(%q) = %q, want %q", in, got, want)
		}
	}
}

// validatePath must accept mixed case on the RAW path, without the caller
// normalising first. This is the ordering bug that made the original fix a
// no-op: resolveNamespaceIDs (and Remember / SetContext / Consolidate) validate
// caller input *before* the resolve step normalises it, so a lowercase-only
// regex rejected /people/U08A83MEMKN with ErrInvalidPath — the normalisation
// downstream never got a chance to match the existing lowercase row.
func TestValidatePathAcceptsMixedCase(t *testing.T) {
	for _, p := range []string{
		"/people/U08A83MEMKN", "/findings/Charter-Costs", "/People/Mixed_Case-1", "/",
	} {
		if err := validatePath(p); err != nil {
			t.Errorf("validatePath(%q) = %v, want nil (case must not be rejected)", p, err)
		}
	}
}

// Relaxing case must not relax anything else.
func TestValidatePathStillRejectsStructuralProblems(t *testing.T) {
	for _, p := range []string{
		"people/no-leading-slash", "/has space", "/-leading-hyphen", "/bad!char",
	} {
		if err := validatePath(p); err == nil {
			t.Errorf("validatePath(%q) = nil, want error", p)
		}
	}
}
