package app

import (
	"math"
	"strings"
	"testing"
	"time"

	"griz/internal/memory"
)

// daysAgo is the date n days before today.
func daysAgo(n int) string { return time.Now().AddDate(0, 0, -n).Format(memory.DateLayout) }

// backdate rewrites a memory's dates, as if it were last changed n days ago.
func backdate(t *testing.T, a *App, key string, n int) string {
	t.Helper()
	id, err := a.Resolve(key)
	if err != nil {
		t.Fatal(err)
	}
	err = a.write(func() error {
		m, err := a.Repo.Read(id)
		if err != nil {
			return err
		}
		m.Created, m.Updated, m.Renewed = daysAgo(n), daysAgo(n), ""
		if err := a.Repo.Write(m, "backdate "+m.Label()); err != nil {
			return err
		}
		a.indexed(m)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// A memory's weight halves its distance to the floor every half-life of its
// term; infinite and pinned memories don't fade.
func TestFadeWeight(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	today := memory.Today()
	short, inf, yes := "short", "infinite", true
	for _, s := range []StoreArgs{
		{ID: "x/week", Body: new("a"), Term: &short},
		{ID: "x/fortnight", Body: new("b"), Term: &short},
		{ID: "x/forever", Body: new("c"), Term: &inf},
		{ID: "x/pinned", Body: new("d"), Term: &short, Pin: &yes},
		{ID: "x/fresh", Body: new("e"), Term: &short},
	} {
		s.Force = true
		mustStore(t, a, s, "ok")
	}
	week := backdate(t, a, "x/week", 7)
	fortnight := backdate(t, a, "x/fortnight", 14)
	forever := backdate(t, a, "x/forever", 400)
	pinned := backdate(t, a, "x/pinned", 400)
	fresh, _ := a.Resolve("x/fresh")
	for _, c := range []struct {
		id   string
		want float64
	}{{week, 0.75}, {fortnight, 0.625}, {forever, 1}, {pinned, 1}, {fresh, 1}} {
		if w := a.weight(c.id, today); !near(w, c.want) {
			t.Errorf("%s: weight %v, want %v", a.label(c.id), w, c.want)
		}
	}
	a.Cfg.Lint.FadedHalves = 2
	if !a.faded(fortnight, today) || a.faded(week, today) || a.faded(forever, today) {
		t.Error("faded: two half-lives old, and only memories that fade")
	}
}

// The same match ranks lower when it is older; renewing brings it back.
func TestFadeRanking(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	short := "short"
	mustStore(t, a, StoreArgs{ID: "zoo/old", Body: new("the walrus eats clams"), Term: &short}, "ok")
	mustStore(t, a, StoreArgs{ID: "zoo/new", Body: new("the walrus eats clams"), Term: &short, Force: true}, "ok")
	backdate(t, a, "zoo/old", 30)
	first := func() string {
		out, _, _ := a.Recall("walrus clams", 0, false)
		return strings.SplitN(noIDs(out), "\n", 2)[0]
	}
	if got := first(); !strings.HasPrefix(got, "### zoo/new") {
		t.Fatalf("the newer memory should rank first: %q", got)
	}
	mustStore(t, a, StoreArgs{ID: "zoo/old", Renew: true}, "ok")
	if m, _ := a.Repo.Read("zoo/old"); m.Renewed != memory.Today() || m.Updated != daysAgo(30) {
		t.Fatalf("renew sets renewed, not updated: %+v", m)
	}
	if got := first(); !strings.HasPrefix(got, "### zoo/new") && !strings.HasPrefix(got, "### zoo/old") {
		t.Fatalf("recall after renew: %q", got)
	}
	old, _ := a.Resolve("zoo/old")
	if w := a.weight(old, memory.Today()); w != 1 {
		t.Errorf("renewed today: weight %v", w)
	}
}

// A get in the server takes back half a memory's age. The renewal waits
// in memory until another write to the file, a flush, or the server stops.
func TestFadeRenewOnGet(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	short := "short"
	mustStore(t, a, StoreArgs{ID: "x/a", Body: new("alpha"), Term: &short}, "ok")
	mustStore(t, a, StoreArgs{ID: "x/b", Body: new("beta"), Term: &short, Force: true}, "ok")
	ida := backdate(t, a, "x/a", 10)
	idb := backdate(t, a, "x/b", 10)
	commits := func() string { n, _ := a.Repo.Git("rev-list", "--count", "HEAD"); return n }

	a.Get([]string{ida}) // not the server: no renewal
	if a.since(ida) != daysAgo(10) {
		t.Fatal("a get outside the server renewed")
	}
	a.Server = true
	n := commits()
	a.Get([]string{ida, idb})
	if a.since(ida) != daysAgo(5) || a.since(idb) != daysAgo(5) {
		t.Fatalf("since after get: %s %s, want %s", a.since(ida), a.since(idb), daysAgo(5))
	}
	if m, _ := a.Repo.Read(ida); m.Renewed != "" || commits() != n {
		t.Fatal("a get wrote to the file")
	}
	a.Get([]string{ida}) // 5 days back to 2 or 3: still moves
	a.Get([]string{ida})
	a.Get([]string{ida}) // eventually within a day of today: no change
	for range 5 {
		a.Get([]string{ida})
	}
	if s := a.since(ida); s < daysAgo(1) {
		t.Fatalf("repeated gets: since %s", s)
	}

	// Another write to the file carries the renewal along.
	mustStore(t, a, StoreArgs{ID: "x/a", Tags: []string{"t"}}, "ok")
	if m, _ := a.Repo.Read(ida); m.Renewed == "" || a.renew.get(ida) != "" {
		t.Fatalf("a write should take the pending renewal: %+v", m)
	}

	// A flush writes the rest in one commit, and nothing when nothing changed.
	n = commits()
	if err := a.FlushRenewals(); err != nil {
		t.Fatal(err)
	}
	if m, _ := a.Repo.Read(idb); m.Renewed != daysAgo(5) {
		t.Fatalf("flushed: %+v", m)
	}
	if log, _ := a.Repo.Git("log", "-1", "--format=%s"); !strings.HasPrefix(log, "renew: x/b") || commits() == n {
		t.Fatalf("flush commit: %q", log)
	}
	n = commits()
	if err := a.FlushRenewals(); err != nil {
		t.Fatal(err)
	}
	if commits() != n {
		t.Error("an empty flush committed")
	}
	if clean, _ := a.Repo.Clean(); !clean {
		t.Error("work tree should be clean")
	}
}

// The migration drops every expiry date once; dates set later stay.
func TestMigrateExpiry(t *testing.T) {
	a := newApp(t, t.TempDir())
	a.Cfg.Store.NewNSGuard = false
	d := "2031-05-05"
	mustStore(t, a, StoreArgs{ID: "x/a", Body: new("alpha"), Expires: &d}, "ok")
	mustStore(t, a, StoreArgs{ID: "x/b", Body: new("beta"), Force: true}, "ok")
	if n, err := a.MigrateExpiry(); err != nil || n != 1 {
		t.Fatalf("migrate: %d %v", n, err)
	}
	if m, _ := a.Repo.Read("x/a"); m.Expires != "" {
		t.Fatalf("expires kept: %+v", m)
	}
	mustStore(t, a, StoreArgs{ID: "x/b", Expires: &d}, "ok")
	if n, _ := a.MigrateExpiry(); n != 0 {
		t.Fatal("migration ran twice")
	}
	if m, _ := a.Repo.Read("x/b"); m.Expires != d {
		t.Fatalf("a date set after the migration was dropped: %+v", m)
	}

	// An empty repo is marked done too, so a date set later survives.
	b := newApp(t, t.TempDir())
	if n, err := b.MigrateExpiry(); err != nil || n != 0 {
		t.Fatalf("empty: %d %v", n, err)
	}
	b.Cfg.Store.NewNSGuard = false
	mustStore(t, b, StoreArgs{ID: "x/c", Body: new("gamma"), Expires: &d}, "ok")
	if _, err := b.MigrateExpiry(); err != nil {
		t.Fatal(err)
	}
	if m, _ := b.Repo.Read("x/c"); m.Expires != d {
		t.Fatal("second run on a once-empty repo dropped a date")
	}
}
