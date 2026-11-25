package app

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/pprof"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/google/pprof/profile"

	"griz/internal/config"
	"griz/internal/embed"
	"griz/internal/index"
	"griz/internal/memory"
)

// Report describes griz in this process: memories, model, embedding, last
// scan, sizes, warnings and settings that differ from the defaults.
func (a *App) Report() string {
	var b strings.Builder
	line := func(format string, args ...any) { fmt.Fprintf(&b, format+"\n", args...) }

	// Memories.
	mems, bad, _ := a.Repo.LoadAll()
	today := memory.Today()
	soon := time.Now().AddDate(0, 0, a.Cfg.Lint.ExpiringDays).Format(memory.DateLayout)
	top := map[string]int{}
	pinned, archived, expiring := 0, 0, 0
	for _, m := range mems {
		top[strings.SplitN(m.Path, "/", 2)[0]]++
		switch {
		case m.IsArchived(today):
			archived++
		case m.Expires != "" && m.Expires <= soon:
			expiring++
		}
		if m.Pin && !m.IsArchived(today) {
			pinned++
		}
	}
	var folders []string
	for _, f := range sortedKeys(top) {
		folders = append(folders, fmt.Sprintf("%s %d", f, top[f]))
	}
	line("memories: %d (%s); pinned %d, archived %d, expiring within %dd %d", len(mems), strings.Join(folders, ", "),
		pinned, archived, a.Cfg.Lint.ExpiringDays, expiring)
	if len(bad) > 0 {
		line("broken files (not indexed): %d", len(bad))
	}

	// Model.
	line("search: %s", a.Status())
	if a.model != nil {
		w, procs := a.Cfg.Search.Workers(), runtime.GOMAXPROCS(0)
		note := ""
		if w > procs {
			note = fmt.Sprintf(" (note: more than the %d CPUs Go uses, so some wait for a CPU; that's allowed)", procs)
		}
		line("embedding workers: %d at once (search.embed_workers)%s", w, note)
	}
	if a.model != nil && a.model.Failed() == nil {
		loaded, loads := a.model.Loaded()
		state := "not loaded"
		if loaded {
			state = "loaded"
			if at := a.model.UnloadAt(); !at.IsZero() {
				state += fmt.Sprintf(", unloads at %s if idle", at.Format("15:04:05"))
			}
		}
		line("model: %s, %s, %d dims, %d tokens max; %s (loaded %d %s in this process)", a.model.Name(), mb(a.model.Bytes()),
			a.model.Dims(), a.model.MaxTokens(), state, loads, plural(loads, "time", "times"))
	}

	// Embedding.
	if a.model != nil {
		have, total := a.vectorCount()
		done, claimed, since, ids := a.embedding()
		switch {
		case claimed > 0:
			el := time.Since(since)
			left := ""
			if done > 0 {
				left = ", about " + (el * time.Duration(claimed-done) / time.Duration(done)).Round(time.Second).String() + " left"
			}
			line("embedding now: %d of %d chunks after %s%s (%s)", done, claimed, el.Round(time.Second), left, strings.Join(ids, ", "))
		case have == total:
			line("embedding: all %d chunks have current vectors", total)
		default:
			line("embedding: %d of %d chunks have current vectors; the next recall embeds the rest", have, total)
		}
	}

	// Last scan.
	a.mu.Lock()
	sc := a.lastScan
	a.mu.Unlock()
	if every, _ := a.Cfg.Scan.IntervalDuration(); every <= 0 {
		line("scan: off")
	} else if sc.at.IsZero() {
		line("scan: none yet (every %s)", every)
	} else {
		s := fmt.Sprintf("scan: last at %s (every %s); %d changed", sc.at.Format("15:04:05"), every, len(sc.changed))
		if len(sc.changed) > 0 {
			s += " (" + strings.Join(sc.changed, ", ") + ")"
		}
		if len(sc.committed) > 0 {
			s += "; committed " + strings.Join(sc.committed, ", ")
		}
		if sc.err != nil {
			s += "; error: " + sc.err.Error()
		}
		line("%s", s)
	}

	// Sizes.
	vsize := "none"
	if st, err := os.Stat(filepath.Join(a.Cfg.StateDir, "vectors.bin")); err == nil {
		vsize = fmt.Sprintf("%.1f MB", float64(st.Size())/(1<<20))
	}
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	rss := PeakRSS()
	line("sizes: vectors.bin %s; this process: Go heap %s, peak resident %s (the model's mapped weights are shared)",
		vsize, mb(int64(ms.HeapAlloc)), mb(rss))
	line("uptime: %s (pid %d); debug mode %s", time.Since(a.started).Round(time.Second), os.Getpid(), onOff(a.Debugging()))

	// Warnings.
	all := a.current()
	if len(all) == 0 {
		line("warnings: none")
	} else {
		line("warnings:")
		for _, w := range all {
			line("  %s", w.text)
		}
	}

	// Server, sessions and crashes (set by the server, or by the CLI).
	if a.StatusExtra != nil {
		for _, l := range a.StatusExtra() {
			line("%s", l)
		}
	}

	// Settings.
	if diff := configDiff(a.Cfg); len(diff) > 0 {
		line("settings that differ from the defaults:")
		for _, d := range diff {
			line("  %s", d)
		}
	} else {
		line("settings: all defaults")
	}
	return strings.TrimRight(b.String(), "\n")
}

// PeakRSS is this process's peak resident memory in bytes, since it started.
func PeakRSS() int64 {
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	rss := ru.Maxrss // bytes on macOS, KB on Linux
	if runtime.GOOS != "darwin" {
		rss *= 1024
	}
	return rss
}

func onOff(on bool) string {
	if on {
		return "on"
	}
	return "off"
}

// configDiff lists settings that differ from the defaults, as section.key = value.
func configDiff(c *config.Config) []string {
	def := config.Defaults()
	var out []string
	cv, dv := reflect.ValueOf(*c), reflect.ValueOf(def)
	for i := 0; i < cv.NumField(); i++ {
		f := cv.Type().Field(i)
		sec := strings.Split(f.Tag.Get("toml"), ",")[0]
		if sec == "" || sec == "-" || f.Type.Kind() != reflect.Struct {
			continue
		}
		for k := 0; k < f.Type.NumField(); k++ {
			kf := f.Type.Field(k)
			name := strings.Split(kf.Tag.Get("toml"), ",")[0]
			x, y := cv.Field(i).Field(k).Interface(), dv.Field(i).Field(k).Interface()
			if !reflect.DeepEqual(x, y) {
				out = append(out, fmt.Sprintf("%s.%s = %v (default %v)", sec, name, show(x), show(y)))
			}
		}
	}
	return out
}

func show(v any) string {
	if s, ok := v.(string); ok {
		return fmt.Sprintf("%q", s)
	}
	return fmt.Sprint(v)
}

// Explain runs recall's ranking for q and shows, for each result, the
// chosen chunk's keyword rank and score, vector rank and similarity, and the
// fused score; then, if id is given, where that memory landed and why.
func (a *App) Explain(q, id string) (string, error) {
	if err := a.Sync(); err != nil {
		return "", err
	}
	q = strings.TrimSpace(q)
	if q == "" || q == "*" {
		return "", fmt.Errorf("explain needs q")
	}
	op := a.BeginOp(nil, "explain")
	rk := a.rank(q, index.Filter{}, op)
	var b strings.Builder
	fmt.Fprintf(&b, "explain %q: keyword %d chunk hits", q, rk.kwHits)
	if rk.byVector {
		fmt.Fprintf(&b, ", vectors %d of %d chunks at similarity >= %.2f", rk.vecHits, rk.searched, a.Cfg.Search.MinSimilarity)
	} else {
		b.WriteString(", no search by meaning (" + a.Status() + ")")
	}
	fmt.Fprintf(&b, "; each list's top %d chunks collapsed to memories (score of best chunk); blend = %.2f × keyword/best keyword + %.2f × similarity/best similarity (keyword weight from tokens per word)\n",
		a.Cfg.Search.Candidates, rk.alpha, 1-rk.alpha)
	shown := a.Cfg.Recall.FullResults + a.Cfg.Recall.IDResults
	today := memory.Today()
	b.WriteString("\nrank  blend   keyword (rank score chunk)  vector (rank sim chunk)  shown  id\n")
	for i, r := range rk.results {
		if i >= shown {
			fmt.Fprintf(&b, "... %d more below what recall shows (full_results %d + id_results %d)\n",
				len(rk.results)-shown, a.Cfg.Recall.FullResults, a.Cfg.Recall.IDResults)
			break
		}
		note := ""
		if a.ix.ArchivedAt(r.id, today) {
			note = " (archived)"
		}
		fmt.Fprintf(&b, "%-5d %.4f  %-26s  %-23s  %-5d  %s%s\n", i+1, r.score, kwCell(r), vecCell(r), r.seq+1, r.id, note)
	}
	if len(rk.results) == 0 {
		b.WriteString("(no results)\n")
	}
	if id != "" {
		b.WriteString("\n" + a.explainID(q, id, rk))
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

func kwCell(r scored) string {
	if r.kwRank == 0 {
		return "-"
	}
	return fmt.Sprintf("#%d %.2f chunk %d", r.kwRank, r.kwScore, r.kwSeq+1)
}

func vecCell(r scored) string {
	if r.vecRank == 0 {
		return "-"
	}
	return fmt.Sprintf("#%d %.3f chunk %d", r.vecRank, r.vecSim, r.vecSeq+1)
}

// explainID says where one memory landed for q, and why.
func (a *App) explainID(q, id string, rk ranking) string {
	id = strings.TrimSuffix(strings.TrimSpace(id), ".md")
	if !a.ix.Has(id) {
		return id + ": not in the index (no such memory, or its file is broken)"
	}
	var b strings.Builder
	for i, r := range rk.results {
		if r.id == id {
			fmt.Fprintf(&b, "%s: rank %d of %d (blend %.4f; keyword %s; vector %s; shows chunk %d)\n", id, i+1, len(rk.results), r.score, kwCell(r), vecCell(r), r.seq+1)
			break
		}
	}
	if b.Len() == 0 {
		fmt.Fprintf(&b, "%s: not in the results\n", id)
	}
	// Every chunk's similarity, whatever the threshold.
	if rk.byVector {
		if vs, err := a.embed([]string{a.Cfg.Search.QueryPrefix + q}, nil); err == nil && len(vs) == 1 {
			var refs []index.ChunkRef
			for _, r := range a.ix.Chunks(index.Filter{}) {
				if r.ID == id {
					refs = append(refs, r)
				}
			}
			got := map[int]float64{}
			a.vec.Each(a.model.ID(), refs, func(r index.ChunkRef, v []float32) { got[r.Seq] = embed.Cosine(vs[0], v) })
			for _, r := range refs {
				if s, ok := got[r.Seq]; ok {
					fmt.Fprintf(&b, "  chunk %d: similarity %.3f%s\n", r.Seq+1, s, map[bool]string{true: " (below min_similarity)"}[s < a.Cfg.Search.MinSimilarity])
				} else {
					fmt.Fprintf(&b, "  chunk %d: no vector\n", r.Seq+1)
				}
			}
		}
	}
	terms := index.Terms(q)
	if len(terms) == 0 {
		b.WriteString("  keyword: the query has no searchable words (all stopwords)\n")
	} else {
		matched := map[string]bool{}
		for _, t := range index.Terms(a.docText(id)) {
			matched[t] = true
		}
		var hit, miss []string
		for _, t := range terms {
			if matched[t] {
				hit = append(hit, t)
			} else {
				miss = append(miss, t)
			}
		}
		fmt.Fprintf(&b, "  keyword terms found in it: %s; not found: %s\n", listOr(hit), listOr(miss))
	}
	return b.String()
}

func listOr(s []string) string {
	if len(s) == 0 {
		return "none"
	}
	return strings.Join(s, " ")
}

// docText is everything the keyword index sees for id: id, folder, tags, body.
func (a *App) docText(id string) string {
	m, err := a.Repo.Read(id)
	if err != nil {
		return id
	}
	return strings.NewReplacer("/", " ", "-", " ", "_", " ").Replace(id) + " " + strings.Join(m.Tags, " ") + " " + m.Body
}

// Profile records a CPU profile while fn runs (or, with fn nil, for d while
// griz goes on working), and returns the top functions as text. If out is
// set, the raw profile is written there for go tool pprof.
func Profile(d time.Duration, fn func(), out string) (string, error) {
	var buf bytes.Buffer
	if err := pprof.StartCPUProfile(&buf); err != nil {
		return "", fmt.Errorf("profile: %w", err)
	}
	start := time.Now()
	if fn != nil {
		fn()
	} else {
		time.Sleep(d)
	}
	pprof.StopCPUProfile()
	wall := time.Since(start)
	if out != "" {
		if err := os.WriteFile(out, buf.Bytes(), 0o644); err != nil {
			return "", err
		}
	}
	p, err := profile.Parse(bytes.NewReader(buf.Bytes()))
	if err != nil {
		return "", err
	}
	return topFunctions(p, wall), nil
}

// topFunctions lists functions by their own (flat) CPU time, with cumulative time.
func topFunctions(p *profile.Profile, wall time.Duration) string {
	vi := len(p.SampleType) - 1 // cpu nanoseconds
	flat, cum := map[string]int64{}, map[string]int64{}
	var total int64
	for _, s := range p.Sample {
		v := s.Value[vi]
		total += v
		seen := map[string]bool{}
		for li, loc := range s.Location {
			for l := len(loc.Line) - 1; l >= 0; l-- {
				name := loc.Line[l].Function.Name
				if li == 0 && l == len(loc.Line)-1 {
					flat[name] += v
				}
				if !seen[name] {
					seen[name] = true
					cum[name] += v
				}
			}
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "CPU %s over %s wall (%.1f cores busy on average)\n", time.Duration(total).Round(time.Millisecond), wall.Round(time.Millisecond),
		float64(total)/float64(wall))
	if total == 0 {
		b.WriteString("(no samples: griz was idle)")
		return b.String()
	}
	names := make([]string, 0, len(flat))
	for n := range flat {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool { return flat[names[i]] > flat[names[j]] })
	b.WriteString("\n  flat   flat%   cum%   function\n")
	for i, n := range names {
		if i == 25 {
			break
		}
		fmt.Fprintf(&b, "%7s  %5.1f%%  %5.1f%%  %s\n", time.Duration(flat[n]).Round(time.Millisecond), 100*float64(flat[n])/float64(total),
			100*float64(cum[n])/float64(total), n)
	}
	return strings.TrimRight(b.String(), "\n")
}
