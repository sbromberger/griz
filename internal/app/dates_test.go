package app

import (
	"os"
	"regexp"
	"strings"
	"testing"

	"griz/internal/memory"
)

func TestCreatedUpdatedAndAppend(t *testing.T) {
	a := newApp(t, t.TempDir())
	today := memory.Today()
	mustStore(t, a, StoreArgs{ID: "log/house", Body: new("### June 2\nfirst note about the boiler"), NewNS: true}, "ok")
	m, _ := a.Repo.Read("log/house")
	if m.Created != today || m.Updated != today {
		t.Fatalf("created %q updated %q, want %s", m.Created, m.Updated, today)
	}

	// An existing memory with no created date doesn't get one invented.
	put(a, &memory.Memory{ID: "log/old", Body: "old note", Updated: "2025-01-02"})
	if err := a.Repo.CommitAll("add old"); err != nil {
		t.Fatal(err)
	}
	if err := a.Sync(); err != nil {
		t.Fatal(err)
	}
	mustStore(t, a, StoreArgs{ID: "log/old", Old: new("old"), New: new("older")}, "ok")
	if m, _ := a.Repo.Read("log/old"); m.Created != "" || m.Updated != today {
		t.Errorf("edit: created %q updated %q", m.Created, m.Updated)
	}

	// append adds a dated entry at the end.
	mustStore(t, a, StoreArgs{ID: "log/house", Append: new("  boiler serviced  ")}, "ok")
	m, _ = a.Repo.Read("log/house")
	entry := regexp.MustCompile(`\n\n\*\*\d{4}-\d{2}-\d{2} \d{2}:\d{2}\*\* — boiler serviced$`)
	if !entry.MatchString(m.Body) {
		t.Errorf("append body: %q", m.Body)
	}
	if msg, _ := a.Repo.Git("log", "-1", "--format=%s"); noIDs(strings.TrimSpace(msg)) != "append log/house" {
		t.Errorf("commit message %q", msg)
	}

	// append creates a missing memory; the folder guard still applies.
	if _, err := a.Store(StoreArgs{ID: "diary/day", Append: new("hello")}); err == nil || !strings.Contains(err.Error(), "new folder") {
		t.Errorf("append to new folder: %v", err)
	}
	mustStore(t, a, StoreArgs{ID: "log/garden", Append: new("planted tomatoes")}, "ok")
	if m, _ := a.Repo.Read("log/garden"); !strings.HasPrefix(m.Body, "**") || m.Created != today || m.Term == "" {
		t.Errorf("append-created memory: %+v", m)
	}
	for _, bad := range []StoreArgs{
		{ID: "log/house", Append: new("x"), Body: new("y")},
		{ID: "log/house", Append: new("x"), Old: new("a"), New: new("b")},
		{ID: "log/house", Append: new("  ")},
	} {
		if _, err := a.Store(bad); err == nil {
			t.Errorf("store %+v should fail", bad)
		}
	}

	// Recall shows the updated date, and demotes the memory's own ### headings.
	out, _, _ := a.Recall("boiler", 0, false)
	if !strings.Contains(noIDs(out), "### log/house (updated "+today+")\n#### June 2\n") {
		t.Errorf("recall: %q", out)
	}
	if out, _ := a.Get([]string{"log/house"}); !strings.Contains(out, "\n#### June 2\n") {
		t.Errorf("get: %q", out)
	}
	if raw, _ := os.ReadFile(file(a, "log/house")); !strings.Contains(string(raw), "\n### June 2\n") {
		t.Error("the file on disk must keep its headings")
	}
}

func TestRecallExcerptShowsUpdated(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Recall.ExcerptWords = 5
	a.maxWords = 5
	mustStore(t, a, StoreArgs{ID: "log/long", Body: new("alpha one two three four\n\n## Next\nzebra five six seven"), NewNS: true}, "ok")
	out, _, _ := a.Recall("zebra", 0, false)
	if !strings.Contains(out, " of 3, updated "+memory.Today()+")\nzebra") {
		t.Errorf("recall: %q", out)
	}
}

func TestHandEditDates(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Scan.Settle = "0"
	today := memory.Today()
	put(a, &memory.Memory{ID: "notes/a", Body: "a", Created: "2025-01-01", Updated: "2025-01-01"})
	put(a, &memory.Memory{ID: "notes/b", Body: "b", Created: "2025-01-01", Updated: "2025-01-01"})
	if err := a.Repo.CommitAll("seed"); err != nil {
		t.Fatal(err)
	}
	if err := a.Sync(); err != nil {
		t.Fatal(err)
	}

	put(a, &memory.Memory{ID: "notes/a", Body: "a edited", Created: "2025-01-01", Updated: "2025-01-01"})
	put(a, &memory.Memory{ID: "notes/b", Body: "b edited", Created: "2025-01-01", Updated: "2025-03-03"})
	if err := os.WriteFile(file(a, "notes/c"), []byte("new by hand\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := a.Scan(); err != nil {
		t.Fatal(err)
	}
	for id, want := range map[string][2]string{
		"notes/a": {"2025-01-01", today},        // updated stamped
		"notes/b": {"2025-01-01", "2025-03-03"}, // the edit's own date kept
		"notes/c": {today, today},               // new file gets both
	} {
		m, _ := a.Repo.Read(id)
		if m.Created != want[0] || m.Updated != want[1] {
			t.Errorf("%s: created %q updated %q, want %v", id, m.Created, m.Updated, want)
		}
	}
	if clean, _ := a.Repo.Clean(); !clean {
		t.Error("the date fix must be part of the commit")
	}
}
