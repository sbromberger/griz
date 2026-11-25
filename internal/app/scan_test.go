package app

import (
	"os"
	"strings"
	"testing"
	"time"
)

func TestScanCommitsHandEdits(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Scan.Settle = "0"
	mustStore(t, a, StoreArgs{ID: "notes/one", Body: str("original text"), NewNS: true}, "ok")
	mustStore(t, a, StoreArgs{ID: "notes/gone", Body: str("to be removed by hand"), Force: true}, "ok")

	os.WriteFile(file(a, "notes/one"), []byte("edited in vscode with walrus\n"), 0o644)
	os.WriteFile(file(a, "notes/new"), []byte("brand new narwhal\n"), 0o644)
	os.WriteFile(file(a, "notes/broken"), []byte("---\nbogus: 1\n---\nbroken pelican\n"), 0o644)
	os.Remove(file(a, "notes/gone"))
	if err := a.Scan(); err != nil {
		t.Fatal(err)
	}
	msg, _ := a.Repo.Git("log", "-1", "--format=%s")
	for _, id := range []string{"notes/one", "notes/new", "notes/broken", "notes/gone"} {
		if !strings.Contains(msg, id) {
			t.Errorf("commit %q lacks %s", msg, id)
		}
	}
	if !strings.HasPrefix(msg, "edit (manual): ") {
		t.Errorf("message: %q", msg)
	}
	if clean, _ := a.Repo.Clean(); !clean {
		t.Error("tree should be clean")
	}
	for q, want := range map[string]string{"walrus": "notes/one", "narwhal": "notes/new", "pelican": "none", "removed": "none"} {
		out, _, _ := a.Recall(q, 0, false)
		if !strings.Contains(out, want) {
			t.Errorf("recall %s: %q, want %s", q, out, want)
		}
	}
}

func TestScanSettleAndNoCommit(t *testing.T) {
	a := newApp(t, t.TempDir())
	mustStore(t, a, StoreArgs{ID: "notes/one", Body: str("original"), NewNS: true}, "ok")
	os.WriteFile(file(a, "notes/one"), []byte("fresh edit with okapi\n"), 0o644)
	a.Scan() // settle is 1m: indexed, not committed
	if clean, _ := a.Repo.Clean(); clean {
		t.Error("unsettled edit must not be committed")
	}
	if out, _, _ := a.Recall("okapi", 0, false); !strings.Contains(out, "notes/one") {
		t.Errorf("unsettled edit should still be indexed: %q", out)
	}
	old := time.Now().Add(-2 * time.Minute)
	os.Chtimes(file(a, "notes/one"), old, old)
	a.Cfg.Scan.CommitEdits = false
	a.Scan()
	if clean, _ := a.Repo.Clean(); clean {
		t.Error("commit_edits=false must leave edits uncommitted")
	}
	a.Cfg.Scan.CommitEdits = true
	// A store commits pending hand edits first.
	mustStore(t, a, StoreArgs{ID: "notes/two", Body: str("other")}, "ok")
	if clean, _ := a.Repo.Clean(); !clean {
		t.Error("store should commit settled hand edits first")
	}
}
