package main

import (
	"reflect"
	"sort"
	"strings"
	"testing"
	"unicode/utf16"

	"github.com/alash3al/stash/internal/bootstrap"
)

// Both tool input schemas must stay backward compatible: every existing
// parameter keeps its name, only optional parameters are added, and nothing
// new becomes required. A caller written against the old schema must keep
// working unchanged.
func TestToolSchemas_BackwardCompatible(t *testing.T) {
	s := newMCPServer(&bootstrap.Context{})

	for _, tc := range []struct {
		tool     string
		props    []string
		required []string
	}{
		{tool: "recall", props: []string{"query", "namespaces", "limit", "include_superseded"}, required: []string{"query"}},
		{tool: "query_facts", props: []string{"namespaces", "limit", "offset", "include_superseded"}, required: nil},
	} {
		t.Run(tc.tool, func(t *testing.T) {
			st := s.GetTool(tc.tool)
			if st == nil {
				t.Fatalf("tool %q not registered", tc.tool)
			}
			schema := st.Tool.InputSchema
			for _, p := range tc.props {
				if _, ok := schema.Properties[p]; !ok {
					t.Errorf("property %q missing; have %v", p, keys(schema.Properties))
				}
			}

			required := append([]string(nil), schema.Required...)
			sort.Strings(required)
			if len(required) == 0 {
				required = nil
			}
			if !reflect.DeepEqual(required, tc.required) {
				t.Errorf("required = %v, want %v", required, tc.required)
			}

			prop, ok := schema.Properties["include_superseded"].(map[string]any)
			if !ok {
				t.Fatalf("include_superseded schema = %#v", schema.Properties["include_superseded"])
			}
			if prop["type"] != "boolean" {
				t.Errorf("include_superseded type = %v, want boolean", prop["type"])
			}
			if prop["default"] != false {
				t.Errorf("include_superseded default = %v, want false", prop["default"])
			}
			// render() returns "" for a template name that does not exist, so
			// a typo in the define name would ship an undocumented parameter.
			if d, _ := prop["description"].(string); strings.TrimSpace(d) == "" {
				t.Errorf("include_superseded has no description")
			}
		})
	}
}

// The recall description must tell the agent what the provenance fields mean,
// in particular that a consolidator fact is a model paraphrase.
func TestRecallDescription_ExplainsResultFields(t *testing.T) {
	d := render("recall_description")
	for _, want := range []string{"RESULT FIELDS", "writer", "consolidator", "source_episode_ids", "include_superseded=true"} {
		if !strings.Contains(d, want) {
			t.Errorf("recall_description must mention %q", want)
		}
	}
	if i, j := strings.Index(d, "RESULT FIELDS"), strings.Index(d, "COST OF NOT CALLING"); i < 0 || j < 0 || i > j {
		t.Errorf("RESULT FIELDS must come just before COST OF NOT CALLING (at %d, %d)", i, j)
	}
}

// maxMCPToolDescription is Claude Code's default cap on an MCP tool
// description (CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH, default 2048). Longer
// descriptions are cut and end in "… [truncated]", so the model never sees
// the tail: for recall, that was the result-field guidance and COST OF NOT
// CALLING. The limit counts JavaScript string length, i.e. UTF-16 code units.
const maxMCPToolDescription = 2048

func TestToolDescriptions_FitClaudeCodeLimit(t *testing.T) {
	s := newMCPServer(&bootstrap.Context{})
	tools := s.ListTools()
	if len(tools) == 0 {
		t.Fatal("no tools registered")
	}
	for name, st := range tools {
		d := st.Tool.Description
		if strings.TrimSpace(d) == "" {
			t.Errorf("%s: empty description (missing template?)", name)
			continue
		}
		if n := len(utf16.Encode([]rune(d))); n > maxMCPToolDescription {
			t.Errorf("%s: description is %d UTF-16 units, over Claude Code's %d; the tail is truncated", name, n, maxMCPToolDescription)
		}
	}
}

func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
