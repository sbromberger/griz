package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// move works like mv: an existing folder (or a target ending in "/")
// receives the memory under its own name; anything else is its new path.
// The id, the text and the history survive.
func TestMove(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	for _, p := range []string{"home/caddy", "home/infra/dns/notes", "work/plan"} {
		if _, err := a.Store(StoreArgs{ID: p, Body: new("text of " + p), Force: true}); err != nil {
			t.Fatal(err)
		}
	}
	a.Cfg.Store.NewNSGuard = true
	id, _ := a.Repo.Lookup("home/caddy")
	before, _ := a.Repo.Read(id)
	move := func(key, target string, newNS bool) (string, error) {
		return a.Store(StoreArgs{ID: key, Move: &target, NewNS: newNS})
	}
	at := func(want string) {
		t.Helper()
		if got := a.Repo.PathOf(id); got != want {
			t.Fatalf("path %q, want %q", got, want)
		}
	}

	// An existing nested folder, without a slash: moved into it.
	if r, err := move(id, "home/infra", false); err != nil || r != "moved "+id+": home/caddy → home/infra/caddy" {
		t.Fatalf("into a folder: %q %v", r, err)
	}
	at("home/infra/caddy")
	if m, _ := a.Repo.Read(id); m.Body != before.Body || m.Updated != before.Updated {
		t.Errorf("move changed the memory: %+v", m)
	}
	if _, err := os.Stat(filepath.Join(a.Cfg.DataDir, "home", "caddy."+id+".md")); !os.IsNotExist(err) {
		t.Errorf("old file still there: %v", err)
	}
	if log, _ := a.Repo.Git("log", "--format=%s", "-1"); log != "move home/caddy → home/infra/caddy ["+id+"]\n" {
		t.Errorf("commit message %q", log)
	}
	if st, _ := a.Repo.Git("status", "--porcelain"); st != "" {
		t.Errorf("uncommitted after move:\n%s", st)
	}

	// Not a folder: the last part is the new name.
	if _, err := move(id, "home/infra/proxy", false); err != nil {
		t.Fatal(err)
	}
	at("home/infra/proxy")
	// A nested folder name that is not the last part of a memory path.
	if _, err := move(id, "home/infra/dns", false); err != nil {
		t.Fatal(err)
	}
	at("home/infra/dns/proxy")

	// Refusals.
	for _, c := range []struct{ target, want string }{
		{"home/infra/dns/notes", "is taken by home/infra/dns/notes"},
		{"home/infra/dns/proxy", "already at"},
		{"garden/", `new folder "garden"`},
		{"home/infra/dhcp/proxy", `new folder "home/infra/dhcp"`},
		{"Bad Path", "bad path"},
		{"/", "needs a folder"},
	} {
		if _, err := move(id, c.target, false); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("move to %q: %v, want %q", c.target, err, c.want)
		}
	}
	at("home/infra/dns/proxy")
	if _, err := a.Store(StoreArgs{ID: id, Move: new("work"), Pin: new(bool)}); err == nil {
		t.Error("move with other changes should be refused")
	}
	if _, err := move("nothing/here", "work", false); err == nil {
		t.Error("moving a missing memory should fail")
	}

	// A new folder with new_ns; a trailing slash means a folder.
	if _, err := move(id, "garden/", true); err != nil {
		t.Fatal(err)
	}
	at("garden/proxy")

	// Archived memories keep their dot.
	yes := true
	if _, err := a.Store(StoreArgs{ID: id, Archived: &yes}); err != nil {
		t.Fatal(err)
	}
	if _, err := move(id, "work", false); err != nil {
		t.Fatal(err)
	}
	if rel := a.Repo.RelPath(id); rel != "work/.proxy."+id+".md" {
		t.Errorf("archived file %q", rel)
	}

	// The startup folder: the reply says what changes for sessions.
	if r, err := move(id, "agents/startup/proxy", true); err != nil || !strings.Contains(r, "now in the startup folder") {
		t.Errorf("into startup: %q %v", r, err)
	}
	if r, err := move(id, "work", false); err != nil || !strings.Contains(r, "left the startup folder") {
		t.Errorf("out of startup: %q %v", r, err)
	}

	// Search finds it at the new path.
	if r, _, _ := a.Recall("text of home/caddy", 0, true); !strings.Contains(r, "work/proxy") {
		t.Errorf("recall after move:\n%s", r)
	}
}
