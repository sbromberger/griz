package app

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"griz/internal/config"
	"griz/internal/memory"
)

func newApp(t *testing.T, root string) *App {
	t.Helper()
	cfg := config.Defaults()
	cfg.DataDir = filepath.Join(root, "data")
	cfg.StateDir = filepath.Join(root, "state")
	a, err := Open(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	a.Stderr = io.Discard
	return a
}

func str(s string) *string { return &s }

func mustStore(t *testing.T, a *App, s StoreArgs, want string) {
	t.Helper()
	got, err := a.Store(s)
	if err != nil {
		got = "ERR " + err.Error()
	}
	if !strings.HasPrefix(noIDs(got), want) {
		t.Fatalf("store %s: got %q, want %q", s.ID, got, want)
	}
}

func TestOperations(t *testing.T) {
	a := newApp(t, t.TempDir())
	mustStore(t, a, StoreArgs{ID: "writing/dashes", Body: str("No em-dashes.")}, "ERR new folder")
	mustStore(t, a, StoreArgs{ID: "writing/dashes", Body: str("No em-dashes in text Claude writes."), NewNS: true, Tags: []string{"Style"}}, "ok writing/dashes")
	mustStore(t, a, StoreArgs{ID: "writing/dashes-again", Body: str("No em-dashes in the text Claude writes.")}, "similar: writing/dashes")
	mustStore(t, a, StoreArgs{ID: "writing/dashes-again", Body: str("No em-dashes in the text Claude writes."), Force: true}, "ok")
	mustStore(t, a, StoreArgs{ID: "writing/dashes", Old: str("Claude"), New: str("the assistant")}, "ok")
	mustStore(t, a, StoreArgs{ID: "writing/dashes", Old: str("nope"), New: str("x")}, "ERR old text not found")
	mustStore(t, a, StoreArgs{ID: "writing/dashes", Body: str("x"), Old: str("a"), New: str("b")}, "ERR give body")
	mustStore(t, a, StoreArgs{ID: "travel/trip", Body: str("Trip to Lisbon."), NewNS: true, Expires: str("2020-01-01")}, "ok")

	m, _ := a.Repo.Read("writing/dashes")
	if m.Body != "No em-dashes in text the assistant writes." || m.Tags[0] != "style" {
		t.Fatalf("after edit: %+v", m)
	}
	out, ids, _ := a.Recall("em-dashes", 0, false)
	if !strings.HasPrefix(out, "### writing/dashes") || len(ids) != 2 {
		t.Fatalf("recall: %q %v", out, ids)
	}
	if out, _, _ := a.Recall("Lisbon", 0, false); noIDs(out) != "archived: travel/trip" {
		t.Fatalf("expired memory: id only, on the archived line: %q", out)
	}
	if out, _ := a.Get([]string{"travel/trip"}); noIDs(out) != "### travel/trip\nTrip to Lisbon." {
		t.Fatalf("get returns archived memories in full: %q", out)
	}
	if out, _, _ := a.Recall("*", 0, false); out != "folders: travel 1, writing 2" {
		t.Fatalf("list: %q", out)
	}
	done, err := a.GC()
	if err != nil || len(done) != 1 || !a.Repo.Exists("travel/trip") {
		t.Fatalf("gc archives, never deletes: %v %v", done, err)
	}
	if m, _ := a.Repo.Read("travel/trip"); !m.Archived || m.Expires != "2020-01-01" {
		t.Fatalf("gc keeps the expiry date as the record: %+v", m)
	}
	if done, _ := a.GC(); len(done) != 0 {
		t.Fatal("already archived memories are left alone")
	}
	f := false
	if _, err := a.Store(StoreArgs{ID: "travel/trip", Archived: &f}); err == nil {
		t.Fatal("unarchiving an expired memory needs a new expires")
	}
	if err := a.Unarchive("travel/trip", ""); err != nil {
		t.Fatal(err)
	}
	if m, _ := a.Repo.Read("travel/trip"); m.Archived || m.Expires != "" {
		t.Fatalf("unarchive clears the flag and the past date: %+v", m)
	}
	if out, _, _ := a.Recall("Lisbon", 0, false); !strings.Contains(out, "### travel/trip") {
		t.Fatalf("unarchived: %q", out)
	}
	if _, err := a.Delete("travel/trip"); err != nil || a.Repo.Exists("travel/trip") {
		t.Fatal(err)
	}
	if err := a.Restore("travel/trip", ""); err != nil || !a.Repo.Exists("travel/trip") {
		t.Fatalf("restore brings back deleted memories: %v", err)
	}
	a.Delete("travel/trip")
	log, _ := a.Repo.Git("log", "--format=%s")
	log = noIDs(log)
	for _, want := range []string{"store writing/dashes", "edit writing/dashes", "expire: travel/trip", "unarchive travel/trip", "delete travel/trip", "restore travel/trip"} {
		if !strings.Contains(log, want) {
			t.Errorf("git log lacks %q:\n%s", want, log)
		}
	}
	if clean, _ := a.Repo.Clean(); !clean {
		t.Error("work tree should be clean")
	}
}

func TestRecallBudgetAndExcerpts(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Recall.ExcerptWords = 50
	a.Cfg.Index.ChunkWords = 50
	a.maxWords = 50
	long := "# Status\n\n" + strings.Repeat("filler words go here. ", 30) + "\n\n# Partners\n\nThe zebra partner is Contoso."
	mustStore(t, a, StoreArgs{ID: "proj/status", Body: str(long), NewNS: true}, "ok")
	for i := 0; i < 6; i++ {
		mustStore(t, a, StoreArgs{ID: "proj/z" + string(rune('a'+i)), Body: str(strings.Repeat("zebra stripes ", 10+i*3) + string(rune('a'+i))), Force: true}, "ok")
	}
	out, _, _ := a.Recall("Contoso partner", 0, false)
	if !strings.Contains(noIDs(out), "### proj/status (") || strings.Contains(out, "filler") {
		t.Fatalf("long memory should return only its matching section:\n%s", out)
	}
	out, full, _ := a.Recall("zebra", 60, false)
	if Tokens(out) > 60 || len(full) == 0 {
		t.Fatalf("budget 60: %d tokens, full=%v\n%s", Tokens(out), full, out)
	}
	out, full, _ = a.Recall("zebra", 0, false)
	if len(full) != 3 || !strings.Contains(out, "\nmore: proj/") || Tokens(out) > 800 {
		t.Fatalf("default budget: full=%v\n%s", full, out)
	}
}

func TestTwoProcesses(t *testing.T) {
	root := t.TempDir()
	a, b := newApp(t, root), newApp(t, root)
	mustStore(t, a, StoreArgs{ID: "x/one", Body: str("alpha bravo"), NewNS: true}, "ok")
	if out, _, _ := b.Recall("bravo", 0, false); !strings.Contains(out, "x/one") {
		t.Fatalf("b should see a's write: %q", out)
	}
	// b writes into the namespace a created, then a sees it.
	mustStore(t, b, StoreArgs{ID: "x/two", Body: str("charlie delta")}, "ok")
	if out, _, _ := a.Recall("delta", 0, false); !strings.Contains(out, "x/two") {
		t.Fatalf("a should see b's write: %q", out)
	}
	b.Delete("x/one")
	if out, _, _ := a.Recall("bravo", 0, false); out != "none" {
		t.Fatalf("a should see b's delete: %q", out)
	}
}

func TestStartupBlockAndPinned(t *testing.T) {
	a := newApp(t, t.TempDir())
	write := func(id, body string, pin, archived bool) {
		m := &memory.Memory{ID: id, Body: body, Pin: pin, Archived: archived}
		os.WriteFile(file(a, id), m.Format(), 0o644)
	}
	// Only startup-folder memories: no generated line.
	write("agents/startup/b-date", "Run date first.", false, false)
	write("agents/startup/a-config", "Ask before editing config files.", false, false)
	write("agents/startup/old", "Archived startup rule.", false, true)
	write("me/other", "Not pinned.", false, false)
	block, err := a.StartupBlock("")
	if err != nil || strings.Contains(block, pinnedLine) || strings.Contains(block, "Archived startup") {
		t.Fatalf("no pinned memories outside startup: no generated line, archived left out:\n%s", block)
	}
	if out, ids, _ := a.Pinned(); out != "none" || ids != nil {
		t.Fatalf("pinned with none: %q", out)
	}

	// Pinned memories elsewhere: the generated line comes first, then the
	// notice, the usage rules, and the startup memories in id order.
	write("agents/rules/pushy", "Present options, then wait.", true, false)
	write("agents/rules/gone", "Archived pinned rule.", true, true)
	block, _ = a.StartupBlock("griz: 1 warning (x).")
	order := []string{pinnedLine, "griz: 1 warning (x).", "griz is the user's long-term memory", "### agents/startup/a-config", "### agents/startup/b-date"}
	last := -1
	for _, part := range order {
		k := strings.Index(block, part)
		if k <= last {
			t.Fatalf("startup block order: %q out of place:\n%s", part, block)
		}
		last = k
	}
	if strings.Contains(block, "Present options") {
		t.Fatal("pinned memories are not in the startup block")
	}
	out, ids, _ := a.Pinned()
	if noIDs(out) != "### agents/rules/pushy\nPresent options, then wait." || len(ids) != 1 {
		t.Fatalf("pinned recall: %q", out)
	}

	// Warnings: pinned+archived, startup block size, pinned set size.
	warns, _ := a.PinWarnings()
	if len(warns) != 1 || warns[0].Label != warnPinArchived {
		t.Fatalf("only pinned+archived so far: %v", warns)
	}
	write("agents/startup/huge", strings.Repeat("word ", 800), false, false)
	write("agents/rules/big", strings.Repeat("word ", 2000), true, false)
	warns, _ = a.PinWarnings()
	labels := map[string]string{}
	for _, w := range warns {
		labels[w.Label] = w.Text
	}
	if !strings.Contains(labels[warnStartup], "agents/startup/huge") || !strings.Contains(labels[warnPinned], "agents/rules/big") {
		t.Fatalf("size warnings naming the largest: %v", warns)
	}
	if block, _ := a.StartupBlock(""); !strings.Contains(block, "agents/startup/huge") {
		t.Fatal("the whole block is still sent")
	}
	a.Cfg.Pin.StartupWarnChars, a.Cfg.Pin.WarnTokens = 0, 0
	if warns, _ := a.PinWarnings(); len(warns) != 1 {
		t.Fatalf("0 turns the size warnings off: %v", warns)
	}
}

func TestArchive(t *testing.T) {
	a := newApp(t, t.TempDir())
	tr, fa := true, false
	mustStore(t, a, StoreArgs{ID: "proj/old-status", Body: str("Status: the walrus project shipped in 2024."), NewNS: true}, "ok")
	mustStore(t, a, StoreArgs{ID: "proj/new-status", Body: str("Status: walrus v2 is in planning."), Force: true}, "ok")
	mustStore(t, a, StoreArgs{ID: "proj/old-status", Archived: &tr}, "ok")
	if log, _ := a.Repo.Git("log", "-1", "--format=%s"); noIDs(log) != "archive proj/old-status\n" {
		t.Fatalf("commit: %q", log)
	}
	out, full, _ := a.Recall("walrus status", 0, false)
	if !strings.Contains(out, "### proj/new-status") || strings.Contains(out, "shipped") ||
		!strings.HasSuffix(noIDs(out), "archived: proj/old-status") || len(full) != 1 {
		t.Fatalf("archived text must not load:\n%s", out)
	}
	out, _, _ = a.Recall("walrus status", 0, true)
	if !strings.Contains(out, "shipped") || strings.Contains(out, "archived:") {
		t.Fatalf("archived: true treats them as current:\n%s", out)
	}
	mustStore(t, a, StoreArgs{ID: "proj/again", Body: str("Status: the walrus project shipped in 2024.")}, "similar: proj/old-status (archived)")
	mustStore(t, a, StoreArgs{ID: "proj/old-status", Archived: &fa}, "ok")
	if m, _ := a.Repo.Read("proj/old-status"); m.Archived {
		t.Fatal("unarchive")
	}
	// Pinned and archived: left out of the pinned set, with a warning.
	m, _ := a.Repo.Read("proj/new-status")
	m.Pin, m.Archived = true, true
	os.WriteFile(file(a, m.ID), m.Format(), 0o644)
	pinned, _, _ := a.Pinned()
	warns, _ := a.PinWarnings()
	if pinned != "none" || len(warns) != 1 || warns[0].Label != warnPinArchived {
		t.Fatalf("pinned+archived: %v %q", warns, pinned)
	}
}

func TestTerms(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Scan.Settle = "0"
	now := time.Now()
	want := func(id, term, expires string) {
		t.Helper()
		m, err := a.Repo.Read(id)
		if err != nil || m.Term != term || m.Expires != expires {
			t.Fatalf("%s: got term %q expires %q, want %q %q (%v)", id, m.Term, m.Expires, term, expires, err)
		}
	}
	six := now.AddDate(0, 6, 0).Format(memory.DateLayout)
	mustStore(t, a, StoreArgs{ID: "x/default", Body: str("a"), NewNS: true}, "ok")
	want("x/default", "medium", six)
	long, short := "long", "short"
	mustStore(t, a, StoreArgs{ID: "x/rule", Body: str("b"), Term: &long, Force: true}, "ok")
	want("x/rule", "long", now.AddDate(1, 0, 0).Format(memory.DateLayout))
	inf := "infinite"
	mustStore(t, a, StoreArgs{ID: "x/forever", Body: str("f"), Term: &inf, Force: true}, "ok")
	want("x/forever", "infinite", "")
	mustStore(t, a, StoreArgs{ID: "x/note", Body: str("c"), Term: &short, Force: true}, "ok")
	want("x/note", "short", now.AddDate(0, 0, 7).Format(memory.DateLayout))
	exp := "2030-01-01"
	mustStore(t, a, StoreArgs{ID: "x/note", Body: str("c2"), Expires: &exp}, "ok")
	want("x/note", "short", "2030-01-01") // explicit expires wins for that write
	mustStore(t, a, StoreArgs{ID: "x/note", Old: str("c2"), New: str("c3")}, "ok")
	want("x/note", "short", now.AddDate(0, 0, 7).Format(memory.DateLayout)) // edits restart the clock
	bad := "forever"
	mustStore(t, a, StoreArgs{ID: "x/bad", Body: str("d"), Term: &bad}, "ERR term")

	// A term changed by hand is picked up by the scan, counted from the last commit.
	m, _ := a.Repo.Read("x/rule")
	m.Term = "short"
	os.WriteFile(file(a, "x/rule"), m.Format(), 0o644)
	a.Scan()
	want("x/rule", "short", now.AddDate(0, 0, 7).Format(memory.DateLayout))
	if clean, _ := a.Repo.Clean(); !clean {
		t.Fatal("hand edit should be committed")
	}
}

func TestNestedFolders(t *testing.T) {
	a := newApp(t, t.TempDir())
	if memory.NS("work/research/survey-design") != "work/research" {
		t.Fatal("folder is everything before the last /")
	}
	mustStore(t, a, StoreArgs{ID: "work/metrics", Body: str("Metrics scoring"), NewNS: true}, "ok")
	mustStore(t, a, StoreArgs{ID: "work/research/survey-design", Body: str("Survey structure")}, `ERR new folder "work/research"`)
	mustStore(t, a, StoreArgs{ID: "work/research/survey-design", Body: str("Survey structure"), NewNS: true}, "ok")
	mustStore(t, a, StoreArgs{ID: "work/research/lab-notebook", Body: str("Lab notebook")}, "ok")
	mustStore(t, a, StoreArgs{ID: "home/garden", Body: str("Garden plan"), NewNS: true}, "ok")
	tr := true
	mustStore(t, a, StoreArgs{ID: "work/metrics", Archived: &tr}, "ok")
	if out, _, _ := a.Recall("*", 0, false); out != "folders: home 1, work 3 (research 2)" {
		t.Fatalf("tree (archived counted): %q", out)
	}
	if out, _, _ := a.Recall("research", 0, false); !strings.Contains(out, "work/research/") {
		t.Fatalf("each folder level is indexed: %q", out)
	}
}
