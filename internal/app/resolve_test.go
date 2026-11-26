package app

import (
	"strings"
	"testing"

	"griz/internal/memory"
)

func TestResolveKeys(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	put(a, &memory.Memory{ID: "work/research/grant-policy", Body: "a"})
	put(a, &memory.Memory{ID: "home/maple-street/timeline", Body: "b"})
	put(a, &memory.Memory{ID: "home/maple-street/legal-timeline", Body: "c"})
	syl, _ := a.Repo.Lookup("work/research/grant-policy")
	hexPart := syl[strings.LastIndex(syl, "-")+1:]

	for key, want := range map[string]string{
		syl:                             syl, // exact id
		"work/research/grant-policy":    syl, // exact path
		hexPart[:a.ShortLen(syl)]:       syl, // shortest unique hex prefix
		syl[:len(syl)-4]:                syl, // start of the whole id
		"grant-policy":                  syl, // path ending
		"gran":                          syl, // part of a slug
		"work/research/grant-policy.md": syl,
	} {
		if got, err := a.Resolve(key); err != nil || got != want {
			t.Errorf("Resolve(%q) = %q, %v; want %q", key, got, err, want)
		}
	}
	if _, err := a.Resolve("time"); err == nil || !strings.Contains(err.Error(), "matches 2 memories") {
		t.Errorf("a slug part two memories share must list both: %v", err)
	}
	if _, err := a.Resolve("nothing-like-this"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("no match: %v", err)
	}

	// store: a new path makes a new memory with a fresh id; the same path again updates it.
	got, err := a.Store(StoreArgs{ID: "notes/fresh", Body: new("one")})
	if err != nil || !strings.HasPrefix(got, "ok notes/fresh [") {
		t.Fatalf("store new: %q %v", got, err)
	}
	id, _ := a.Repo.Lookup("notes/fresh")
	if memory.ValidID(id) != nil {
		t.Fatalf("new id %q", id)
	}
	if _, err := a.Store(StoreArgs{ID: "notes/fresh", Body: new("two")}); err != nil {
		t.Fatal(err)
	}
	if again, _ := a.Repo.Lookup("notes/fresh"); again != id {
		t.Errorf("storing to the same path must update, not create: %s vs %s", again, id)
	}
	if _, err := a.Store(StoreArgs{ID: "nopath", Body: new("x")}); err == nil {
		t.Error("a key that is neither a memory nor a path must not create anything")
	}
}
