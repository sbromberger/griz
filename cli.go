package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"sort"
	"strings"
	"time"

	"griz/internal/app"
	"griz/internal/config"
	"griz/internal/memory"
	"griz/internal/server"
)

// ---- lint ----

func runLint([]string) error {
	a, err := open()
	if err != nil {
		return err
	}
	cfg := a.Cfg
	mems, bad, err := a.Repo.LoadAll()
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(out, "search: %s\n%d memories\ntool definitions: ~%d tokens\n", a.Status(), len(mems), server.ToolTokens())
	section := func(title string, lines []string) {
		if len(lines) == 0 {
			return
		}
		_, _ = fmt.Fprintf(out, "\n%s (%d)\n", title, len(lines))
		for _, l := range lines {
			_, _ = fmt.Fprintln(out, "  "+l)
		}
	}
	var lines []string
	for id, e := range bad {
		lines = append(lines, fmt.Sprintf("%s: %v", id, e))
	}
	sort.Strings(lines)
	section("broken files (skipped)", lines)

	lines = nil
	for _, m := range mems {
		if w := memory.Words(m.Body); w > cfg.Lint.LongWords {
			lines = append(lines, fmt.Sprintf("%s: %d words", m.Label(), w))
		}
	}
	section(fmt.Sprintf("long, consider splitting (over %d words)", cfg.Lint.LongWords), lines)

	lines = nil
	byNS := map[string][]string{}
	for _, m := range mems {
		byNS[m.NS()] = append(byNS[m.NS()], m.Label())
	}
	for _, ns := range sortedKeys(byNS) {
		if len(byNS[ns]) == 1 {
			lines = append(lines, byNS[ns][0])
		}
	}
	section("namespaces with one memory", lines)

	section("likely duplicates", a.Duplicates(mems))

	if block, err := a.StartupBlock(""); err == nil {
		pinnedText, _, _ := a.Pinned()
		pinned := 0
		if pinnedText != "none" {
			pinned = app.Tokens(pinnedText)
		}
		_, _ = fmt.Fprintf(out, "\nstartup block: %d characters (warn at %d; the apps cut at about 4096)\n", len(block), cfg.Pin.StartupWarnChars)
		_, _ = fmt.Fprintf(out, "pinned (loaded by recall pinned: true): ~%d tokens (warn at %d)\n", pinned, cfg.Pin.WarnTokens)
		if warns, err := a.PinWarnings(); err == nil {
			for _, w := range warns {
				_, _ = fmt.Fprintln(out, "  warning:", w.Text)
			}
		}
	}

	lines = nil
	last, err := a.Usage.LastUsed()
	if err == nil && len(last) > 0 {
		// Memories count as used when the log starts, so a lost log never flags anything.
		var logStart time.Time
		if es, _ := a.Usage.Read(); len(es) > 0 {
			logStart = es[0].Time
		}
		cut := time.Now().AddDate(0, 0, -cfg.Lint.UnusedDays)
		for _, m := range mems {
			if m.IsArchived(memory.Today()) {
				continue
			}
			t := last[m.ID]
			if t.IsZero() {
				t = logStart
				if _, upd := a.Repo.Dates(m.ID); upd != "" {
					if u, err := time.Parse(memory.DateLayout, upd); err == nil && u.After(t) {
						t = u
					}
				}
			}
			if t.Before(cut) {
				lines = append(lines, fmt.Sprintf("%s: last used %s", m.Label(), t.Format(memory.DateLayout)))
			}
		}
	}
	section(fmt.Sprintf("unused for %d days (consider griz memory archive)", cfg.Lint.UnusedDays), lines)

	lines = nil
	today := memory.Today()
	soon := time.Now().AddDate(0, 0, cfg.Lint.ExpiringDays).Format(memory.DateLayout)
	for _, m := range mems {
		if !m.Archived && m.Expires != "" && m.Expires >= today && m.Expires <= soon {
			lines = append(lines, m.Label()+": expires "+m.Expires)
		}
	}
	section(fmt.Sprintf("expiring within %d days (then archived)", cfg.Lint.ExpiringDays), lines)

	lines = nil
	for _, m := range mems {
		if !m.Archived && a.Faded(m.ID) {
			term := m.Term
			if term == "" {
				term = cfg.Store.DefaultTerm
			}
			lines = append(lines, m.Label()+": "+term+", unchanged since "+a.Since(m.ID))
		}
	}
	section(fmt.Sprintf("faded: %d half-lives or more without an edit or renewal (renew, archive, or leave)", cfg.Lint.FadedHalves), lines)
	return nil
}

func sortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

// ---- stats ----

func runStats([]string) error {
	a, err := open()
	if err != nil {
		return err
	}
	mems, bad, err := a.Repo.LoadAll()
	if err != nil {
		return err
	}
	var archived, pinned int
	terms, folders := map[string]int{}, map[string]int{}
	for _, m := range mems {
		if m.Archived {
			archived++
		}
		if m.Pin {
			pinned++
		}
		t := m.Term
		if t == "" {
			t = "(none)"
		}
		terms[t]++
		folders[strings.SplitN(m.Path, "/", 2)[0]]++
	}
	_, _ = fmt.Fprintf(out, "%d memories: %d current, %d archived, %d pinned", len(mems), len(mems)-archived, archived, pinned)
	if len(bad) > 0 {
		_, _ = fmt.Fprintf(out, "; %d unreadable (griz memory lint)", len(bad))
	}
	count := func(label string, m map[string]int) {
		var parts []string
		for _, k := range sortedKeys(m) {
			parts = append(parts, fmt.Sprintf("%s %d", k, m[k]))
		}
		_, _ = fmt.Fprintf(out, "\n%s: %s", label, strings.Join(parts, ", "))
	}
	count("by term", terms)
	count("by folder", folders)
	_, _ = fmt.Fprint(out, "\n\n")
	es, err := a.Usage.Read()
	if err != nil {
		return err
	}
	if len(es) == 0 {
		_, _ = fmt.Fprintln(out, "usage log is empty:", a.Usage.Path)
		return nil
	}
	type agg struct {
		n, none       int
		in, out, maxO int
		ms, maxMS     int64
	}
	by := map[string]*agg{}
	idCount := map[string]int{}
	for _, e := range es {
		g := by[e.Tool]
		if g == nil {
			g = &agg{}
			by[e.Tool] = g
		}
		g.n++
		g.in += e.In
		g.out += e.Out
		g.ms += e.MS
		g.maxO = max(g.maxO, e.Out)
		g.maxMS = max(g.maxMS, e.MS)
		if e.Tool == "recall" && e.Out <= len("none") {
			g.none++
		}
		for _, id := range e.IDs {
			idCount[id]++
		}
	}
	_, _ = fmt.Fprintf(out, "%s to %s, %d calls\n\n", es[0].Time.Format(memory.DateLayout), es[len(es)-1].Time.Format(memory.DateLayout), len(es))
	_, _ = fmt.Fprintf(out, "%-8s %6s %10s %11s %11s %8s %8s\n", "tool", "calls", "avg in", "avg out", "max out", "avg ms", "max ms")
	for _, t := range sortedKeys(by) {
		g := by[t]
		_, _ = fmt.Fprintf(out, "%-8s %6d %7d tk %8d tk %8d tk %8d %8d\n", t, g.n, g.in/g.n/4, g.out/g.n/4, g.maxO/4, g.ms/int64(g.n), g.maxMS)
	}
	if g := by["recall"]; g != nil {
		_, _ = fmt.Fprintf(out, "\nrecalls returning none: %d of %d\n", g.none, g.n)
	}
	type kv struct {
		id string
		n  int
	}
	var top []kv
	for id, n := range idCount {
		top = append(top, kv{id, n})
	}
	sort.Slice(top, func(i, j int) bool { return top[i].n > top[j].n || (top[i].n == top[j].n && top[i].id < top[j].id) })
	if len(top) > 10 {
		top = top[:10]
	}
	if len(top) > 0 {
		_, _ = fmt.Fprintln(out, "\nmost returned:")
		for _, t := range top {
			_, _ = fmt.Fprintf(out, "  %4d  %s\n", t.n, t.id)
		}
	}
	return nil
}

// ---- purge ----

func runPurge(args []string) error {
	var ids []string
	yes := false
	for _, a := range args {
		switch a {
		case "--yes", "-y":
			yes = true
		default:
			if strings.HasPrefix(a, "-") {
				return fmt.Errorf("unknown flag %s", a)
			}
			ids = append(ids, strings.TrimSuffix(a, ".md"))
		}
	}
	if len(ids) == 0 {
		return usageErr("memory purge")
	}
	if _, err := exec.LookPath("git-filter-repo"); err != nil {
		return fmt.Errorf("purge needs git filter-repo (brew install git-filter-repo); nothing changed")
	}
	a, err := open()
	if err != nil {
		return err
	}
	for i, key := range ids {
		if ids[i], err = a.Resolve(key); err != nil {
			return err
		}
	}
	r := a.Repo
	unlock, err := r.Lock()
	if err != nil {
		return err
	}
	defer unlock()
	plan, paths, err := purgePlan(r, ids)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprint(out, plan)
	if !yes {
		_, _ = fmt.Fprint(out, "\nType yes to continue: ")
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		if strings.TrimSpace(line) != "yes" {
			return fmt.Errorf("cancelled; nothing changed")
		}
	}
	fr := []string{"filter-repo", "--force", "--quiet", "--invert-paths"}
	for _, p := range paths {
		fr = append(fr, "--path", p)
	}
	if _, err := r.Git(fr...); err != nil {
		return err
	}
	if _, err := r.Git("reflog", "expire", "--expire=now", "--all"); err != nil {
		return err
	}
	if _, err := r.Git("gc", "-q", "--prune=now"); err != nil {
		return err
	}
	// Drop the purged text's vectors and usage lines.
	if err := a.PruneVectors(); err != nil {
		return err
	}
	if err := a.Usage.Forget(ids); err != nil {
		return err
	}
	_, _ = fmt.Fprintf(out, "\npurged %d memories from this Mac.\n", len(ids))
	_, _ = fmt.Fprintln(out, "Backups made before now still hold them until those backups expire.")
	return nil
}

// purgePlan checks a purge can run and describes what it removes.
func purgePlan(r *memory.Repo, ids []string) (string, []string, error) {
	if remotes, _ := r.Git("remote"); strings.TrimSpace(remotes) != "" {
		return "", nil, fmt.Errorf("the memory repo has a remote; purge with a remote is not supported in v1")
	}
	if clean, err := r.Clean(); err != nil || !clean {
		return "", nil, fmt.Errorf("the memory repo has uncommitted changes; commit or discard them first")
	}
	// Every name each memory's file has had (renames, archiving, the move to
	// ids), so no copy survives anywhere in history.
	var paths []string
	seen := map[string]bool{}
	for _, id := range ids {
		names, _ := r.Git("log", "--all", "--follow", "--name-only", "--format=", "--", r.RelPath(id))
		for _, n := range append(strings.Fields(names), r.RelPath(id)) {
			if !seen[n] {
				seen[n] = true
				paths = append(paths, n)
			}
		}
	}
	log, err := r.Git(append([]string{"log", "--all", "--format=  %h %cs %s", "--"}, paths...)...)
	if err != nil {
		return "", nil, err
	}
	if strings.TrimSpace(log) == "" {
		return "", nil, fmt.Errorf("no history for %s", strings.Join(paths, ", "))
	}
	return fmt.Sprintf("Permanently remove from all git history:\n  %s\n\nCommits that touch them:\n%s\n",
		strings.Join(paths, "\n  "), strings.TrimRight(log, "\n")), paths, nil
}

// confirmPurge shows the purge plan here and asks for "yes" before the
// server does the purge.
func confirmPurge(args []string) ([]string, error) {
	if slices.Contains(args, "--yes") || slices.Contains(args, "-y") {
		return args, nil
	}
	var ids []string
	for _, a := range args {
		if !strings.HasPrefix(a, "-") {
			ids = append(ids, strings.TrimSuffix(a, ".md"))
		}
	}
	if len(ids) == 0 {
		return nil, usageErr("memory purge")
	}
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}
	a, err := app.Open(cfg)
	if err != nil {
		return nil, err
	}
	for i, key := range ids {
		if ids[i], err = a.Resolve(key); err != nil {
			return nil, err
		}
	}
	plan, _, err := purgePlan(a.Repo, ids)
	if err != nil {
		return nil, err
	}
	_, _ = fmt.Fprint(out, plan, "\nType yes to continue: ")
	line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
	if strings.TrimSpace(line) != "yes" {
		return nil, fmt.Errorf("cancelled; nothing changed")
	}
	return append(args, "--yes"), nil
}
