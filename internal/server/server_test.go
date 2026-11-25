package server

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"griz/internal/app"
	"griz/internal/config"
	"griz/internal/memory"
)

func TestToolDefinitionsSize(t *testing.T) {
	t.Logf("tool definitions: ~%d tokens", ToolTokens()) // reported, not limited
}

func TestPinnedRecallIgnoresBudget(t *testing.T) {
	root := t.TempDir()
	cfg := config.Defaults()
	cfg.DataDir, cfg.StateDir = filepath.Join(root, "d"), filepath.Join(root, "s")
	a, err := app.Open(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	a.Stderr = io.Discard
	m := &memory.Memory{ID: "agents/rules/pushy", Pin: true, Body: strings.Repeat("Present options, then wait. ", 50)}
	f := filepath.Join(a.Repo.Dir, "agents/rules/pushy.test-00000001.md")
	os.MkdirAll(filepath.Dir(f), 0o755)
	os.WriteFile(f, m.Format(), 0o644)
	out, ids, failed := callOp(a, nil, "recall", json.RawMessage(`{"pinned":true,"budget":10}`))
	if failed || len(out) < 1000 || len(ids) != 1 {
		t.Fatalf("pinned recall should return everything regardless of budget: %d chars %v", len(out), ids)
	}
}

// The first tool reply of a session carries the pinned memories in a block of
// their own; later replies, and sessions that asked for them, don't repeat it.
func TestPinnedRideFirstReply(t *testing.T) {
	root := t.TempDir()
	cfg := config.Defaults()
	cfg.DataDir, cfg.StateDir = filepath.Join(root, "d"), filepath.Join(root, "s")
	a, err := app.Open(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	a.Stderr = io.Discard
	m := &memory.Memory{ID: "me/profile", Pin: true, Body: "married Robin"}
	f := filepath.Join(a.Repo.Dir, "me/profile.test-00000002.md")
	os.MkdirAll(filepath.Dir(f), 0o755)
	os.WriteFile(f, m.Format(), 0o644)
	texts := func(s *app.Session, args string) []string {
		res, err := handler(a, s, "recall", nil)(context.Background(), &mcp.CallToolRequest{Params: &mcp.CallToolParamsRaw{Arguments: json.RawMessage(args)}})
		if err != nil {
			t.Fatal(err)
		}
		var out []string
		for _, c := range res.Content {
			out = append(out, c.(*mcp.TextContent).Text)
		}
		return out
	}
	s := a.NewSession()
	if got := texts(s, `{"q":"anything"}`); len(got) != 2 || !strings.Contains(got[1], "married Robin") || strings.Contains(got[0], "pinned memories, sent once") {
		t.Fatalf("first reply should carry pinned in a second block: %q", got)
	}
	if got := texts(s, `{"q":"anything"}`); len(got) != 1 {
		t.Fatalf("second reply should not repeat pinned: %q", got)
	}
	s2 := a.NewSession()
	if got := texts(s2, `{"pinned":true}`); len(got) != 1 {
		t.Fatalf("an explicit pinned recall needs no extra block: %q", got)
	}
	if got := texts(s2, `{"q":"anything"}`); len(got) != 1 {
		t.Fatalf("pinned already loaded this session: %q", got)
	}
}
