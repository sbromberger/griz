package app

import (
	"strings"
	"testing"
)

// Recall takes chunks best first: a long memory's other chunks show too
// when they score above the weakest result, in the order they appear in it.
func TestRecallChunks(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	a.maxWords, a.Cfg.Recall.ExcerptWords = 12, 12
	long := strings.Join([]string{
		"# Feeding\nThe walrus eats clams on the sea floor every morning.",
		"# Travel\nIt swims north in spring with the herd along the coast.",
		"# Diet notes\nClams again: the walrus prefers clams to anything else.",
		"# Sleep\nIt hauls out on ice floes and sleeps for hours at a time.",
	}, "\n\n")
	mustStore(t, a, StoreArgs{ID: "zoo/walrus", Body: &long}, "ok")
	mustStore(t, a, StoreArgs{ID: "zoo/seal", Body: new("A seal once ate a clam near the pier."), Force: true}, "ok")

	out, ids, _ := a.Recall("walrus clams", 0, false)
	out = noIDs(out)
	if !strings.HasPrefix(out, "### zoo/walrus (1, ") {
		t.Fatalf("two walrus chunks, the first one first:\n%s", out)
	}
	if !strings.Contains(out, "every morning.\n\n…\n\n# Diet notes") || strings.Contains(out, "Travel") || strings.Contains(out, "Sleep") {
		t.Errorf("the two matching chunks only, with … between:\n%s", out)
	}
	if !strings.Contains(out, "### zoo/seal") || len(ids) != 2 {
		t.Errorf("the other memory still shows: %v\n%s", ids, out)
	}

	// A tight budget still shows the best match first.
	if out, _, _ := a.Recall("walrus clams", 40, false); !strings.HasPrefix(noIDs(out), "### zoo/walrus") {
		t.Errorf("tight budget:\n%s", out)
	}
}
