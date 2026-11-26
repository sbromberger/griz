package app

import (
	"strings"
	"testing"
)

// Pin warnings follow writes that touch pinned or startup memories, and
// only those.
func TestPinWarningsFollowWrites(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	a.Cfg.Pin.StartupWarnChars = 400 // the empty block (rules only) is about 260
	if err := a.RefreshPinWarnings(); err != nil {
		t.Fatal(err)
	} // the server starts on an empty store
	if !strings.Contains(a.Report(), "warnings: none") {
		t.Fatalf("empty store: no warnings:\n%s", a.Report())
	}
	big := strings.Repeat("a long startup rule ", 30)
	if _, err := a.Store(StoreArgs{ID: "agents/startup/big", Body: &big}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(a.Report(), "startup block is") {
		t.Fatalf("a big startup memory should warn:\n%s", a.Report())
	}
	n := a.pinChecks.Load()
	other := "an ordinary note"
	if _, err := a.Store(StoreArgs{ID: "notes/plain", Body: &other}); err != nil {
		t.Fatal(err)
	}
	if a.pinChecks.Load() != n {
		t.Error("a write to a memory that isn't pinned or in startup must not recompute")
	}
	if _, err := a.Delete("agents/startup/big"); err != nil {
		t.Fatal(err)
	}
	if a.pinChecks.Load() == n {
		t.Error("deleting a startup memory must recompute")
	}
	if r := a.Report(); strings.Contains(r, "startup block is") {
		t.Errorf("the warning no longer applies:\n%s", r)
	}
}

// store can pin and unpin without touching the text or the updated date,
// and the pinned-size warning follows.
func TestStorePin(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	a.Cfg.Pin.WarnTokens = 50
	if err := a.RefreshPinWarnings(); err != nil {
		t.Fatal(err)
	}
	body := strings.Repeat("a fact to keep in view ", 20)
	if _, err := a.Store(StoreArgs{ID: "notes/fact", Body: &body}); err != nil {
		t.Fatal(err)
	}
	id, _ := a.Repo.Lookup("notes/fact")
	before, _ := a.Repo.Read(id)
	on, off := true, false
	if r, err := a.Store(StoreArgs{ID: id, Pin: &on}); err != nil || !strings.HasPrefix(r, "ok ") {
		t.Fatalf("pin: %q %v", r, err)
	}
	m, _ := a.Repo.Read(id)
	if !m.Pin || m.Body != before.Body || m.Updated != before.Updated || m.Expires != before.Expires {
		t.Errorf("pin should set only the flag: %+v", m)
	}
	if !strings.Contains(a.Report(), "pinned memories are") {
		t.Errorf("pinning past warn_tokens should warn:\n%s", a.Report())
	}
	if _, err := a.Store(StoreArgs{ID: id, Pin: &off}); err != nil {
		t.Fatal(err)
	}
	if m, _ := a.Repo.Read(id); m.Pin {
		t.Error("unpin")
	}
	if strings.Contains(a.Report(), "pinned memories are") {
		t.Errorf("unpinned: the warning should go:\n%s", a.Report())
	}
	if log, _ := a.Repo.Git("log", "--format=%s", "-2"); !strings.Contains(log, "unpin notes/fact") || !strings.Contains(log, "pin notes/fact") {
		t.Errorf("commit messages: %q", log)
	}
}
