package app

import (
	"strings"
	"testing"

	"griz/internal/memory"
)

// old/new: a unique literal match by default (copies that share text count
// as two), every copy with all, an exact count with count, and a preview
// that writes nothing.
func TestEditMatching(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	body := "Seth Bromberger, data scientist\nha ha ha\nemail Seth about Seth's plan\nthe Sethi family\ncafé café"
	if _, err := a.Store(StoreArgs{ID: "notes/me", Body: &body, Force: true}); err != nil {
		t.Fatal(err)
	}
	id, _ := a.Repo.Lookup("notes/me")
	text := func() string { m, _ := a.Repo.Read(id); return m.Body }
	commits := func() string { log, _ := a.Repo.Git("rev-list", "--count", "HEAD"); return log }
	edit := func(s StoreArgs) (string, error) { s.ID = id; return a.Store(s) }

	refused := func(s StoreArgs, want string) {
		t.Helper()
		before, n := text(), commits()
		if _, err := edit(s); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%+v: %v, want %q", s, err, want)
		}
		if text() != before || commits() != n {
			t.Error("a refused edit changed the memory")
		}
	}
	refused(StoreArgs{Old: new("ha ha"), New: new("x")}, "matches 2 times") // overlapping copies
	refused(StoreArgs{Old: new("ha ha"), New: new("x"), All: true}, "overlap")
	refused(StoreArgs{Old: new("Seth"), New: new("John")}, "matches 4 times")
	refused(StoreArgs{Old: new("Seth"), New: new("John"), All: true, Count: new(3)}, "matches 4 times in notes/me")
	refused(StoreArgs{Old: new("nowhere"), New: new("x"), Preview: true}, "not found")
	refused(StoreArgs{All: true}, "go with old and new")
	refused(StoreArgs{Old: new("ha"), New: new("x"), Preview: true, Tags: []string{"a"}}, "preview shows only")

	// Preview: the lines, before and after; nothing written.
	n := commits()
	p, err := edit(StoreArgs{Old: new("Seth"), New: new("John"), All: true, Preview: true})
	if err != nil {
		t.Fatal(err)
	}
	want := "would replace 4 in notes/me [" + id + "]:\n" +
		"  line 1: Seth Bromberger, data scientist\n    → John Bromberger, data scientist\n" +
		"  line 3: email Seth about Seth's plan\n    → email John about John's plan\n" +
		"  line 4: the Sethi family\n    → the Johni family"
	if p != want {
		t.Errorf("preview:\n%s\nwant:\n%s", p, want)
	}
	if text() != body || commits() != n {
		t.Error("preview wrote")
	}
	// Several copies without all: listed, so old can be made unique.
	if p, _ := edit(StoreArgs{Old: new("Seth"), New: new("John"), Preview: true}); !strings.Contains(p, "matches 4 times") || strings.Contains(p, "→") || !strings.Contains(p, "line 4: the Sethi family") {
		t.Errorf("preview of several:\n%s", p)
	}
	// A copy across lines.
	if p, _ := edit(StoreArgs{Old: new("scientist\nha"), New: new("scientist\nho"), Preview: true}); !strings.Contains(p, "lines 1 to 2: Seth Bromberger, data scientist⏎ha ha ha") {
		t.Errorf("preview across lines:\n%s", p)
	}
	// A small budget lists the first and counts the rest.
	if p, _ := previewEdit(mustRead(t, a, id), "Seth", "John", true, 1); !strings.Contains(p, "line 1:") || !strings.Contains(p, "... and 3 more") {
		t.Errorf("preview over budget:\n%s", p)
	}

	// The edits themselves.
	if r, err := edit(StoreArgs{Old: new("Seth"), New: new("John"), All: true, Count: new(4)}); err != nil || !strings.HasSuffix(r, "(replaced 4)") {
		t.Fatalf("all: %q %v", r, err)
	}
	if got := text(); strings.Contains(got, "Seth") || !strings.Contains(got, "Johni") {
		t.Errorf("after all:\n%s", got)
	}
	if r, err := edit(StoreArgs{Old: new("é"), New: new("e"), All: true}); err != nil || !strings.HasSuffix(r, "(replaced 2)") {
		t.Errorf("multibyte: %q %v", r, err)
	}
	if r, err := edit(StoreArgs{Old: new("data scientist"), New: new("engineer")}); err != nil || strings.Contains(r, "replaced") {
		t.Errorf("unique edit: %q %v", r, err)
	}
}

func mustRead(t *testing.T, a *App, id string) *memory.Memory {
	t.Helper()
	m, err := a.Repo.Read(id)
	if err != nil {
		t.Fatal(err)
	}
	return m
}
