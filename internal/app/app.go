// Package app is griz's core: it ties the memory files, the index, the
// embedding cache and the model together. Both the MCP server and the CLI use it.
package app

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"griz/internal/config"
	"griz/internal/embed"
	"griz/internal/index"
	"griz/internal/memory"
	"griz/internal/usage"
	"griz/internal/vecstore"
)

type App struct {
	Cfg    *config.Config
	Repo   *memory.Repo
	ix     *index.Mem
	vec    *vecstore.Store
	Usage  *usage.Log
	Stderr io.Writer // nil = os.Stderr

	model    *embed.Lazy // nil = keyword only (no model set, or the probe failed)
	probeErr string
	warnLoad sync.Once
	warns    warnings
	pend     pending        // chunks being embedded (see vectors.go)
	embedWG  sync.WaitGroup // embedding started by store or scan
	slots    chan struct{}  // one per chunk being embedded (see vectors.go)
	Local    *Session       // the session used without a server
	// StatusExtra adds lines to Report (the server: sessions, crashes).
	StatusExtra func() []string
	sess        sessions
	ops         opLog
	started     time.Time
	lastScan    scanReport

	mu       sync.Mutex // guards head, stamps and pinIDs
	head     string
	stamps   map[string]stamp
	pinIDs   map[string]bool // memories that are pinned or in the startup folder
	pinWatch atomic.Bool     // recompute pin warnings when one of pinIDs changes (the server)
	maxWords int
	// pinChecks counts pin-warning recomputations (tests).
	pinChecks atomic.Int64
}

// ProbeModel is set by the model runner package (to avoid an import cycle).
var ProbeModel func(dir, docPrefix string) (embed.Info, embed.Loader, error)

// Open loads config-driven state: repo, model, index. It never fails because of the model.
func Open(cfg *config.Config) (*App, error) {
	repo, err := memory.Open(cfg.DataDir)
	if err != nil {
		return nil, err
	}
	if repo.Machine = cfg.Store.Machine; repo.Machine == "" {
		host, _ := os.Hostname()
		if repo.Machine = memory.Machine(host); repo.Machine == "" {
			return nil, fmt.Errorf("can't make an id prefix from the host name %q; set store.machine", host)
		}
	}
	if err := os.MkdirAll(cfg.StateDir, 0o755); err != nil {
		return nil, err
	}
	vec, err := vecstore.Open(cfg.StateDir)
	if err != nil {
		return nil, err
	}
	a := &App{
		Cfg: cfg, Repo: repo, vec: vec,
		ix: index.NewWith(index.Params{K1: cfg.Index.BM25K1, B: cfg.Index.BM25B, WID: cfg.Index.WeightID,
			WNS: cfg.Index.WeightNamespace, WTags: cfg.Index.WeightTags, WBody: cfg.Index.WeightBody}),
		Usage:    usage.New(cfg.StateDir, cfg.Log.Usage),
		maxWords: cfg.Index.ChunkWords,
		started:  time.Now(),
		slots:    make(chan struct{}, cfg.Search.Workers()),
		Local:    &Session{Since: time.Now()},
	}
	if a.maxWords <= 0 {
		a.maxWords = config.Defaults().Index.ChunkWords
	}
	a.probeModel()
	if err := a.reloadAll(); err != nil {
		return nil, err
	}
	return a, nil
}

// probeModel checks the model folder (cheap; no weights are read). The model
// itself loads on first use and unloads when idle.
func (a *App) probeModel() {
	dir := a.Cfg.Search.Model
	if dir == "" {
		return
	}
	if ProbeModel == nil {
		a.probeErr = "no model runner built in"
	} else {
		info, load, err := ProbeModel(dir, a.Cfg.Search.DocumentPrefix)
		if err == nil {
			after, _ := a.Cfg.Search.UnloadDuration()
			a.model = embed.NewLazy(info, load, after)
			a.model.OnUnload = func(idle time.Duration) {
				a.debugf("model unloaded after %s idle", idle)
			}
			// Every chunk must fit the model's input whole. Leave room for the
			// id prefix and for words that split into several tokens.
			if w := int(float64(info.MaxTokens-a.Cfg.Index.TokenReserve) * a.Cfg.Index.WordsPerToken); w < a.maxWords {
				a.maxWords = w
			}
			return
		}
		a.probeErr = err.Error()
	}
	a.Warn(warnModel, "search: %s", a.Status())
}

// Status describes the active search mode.
func (a *App) Status() string {
	switch {
	case a.Cfg.Search.Model == "":
		return "keyword only (no model set)"
	case a.probeErr != "":
		return "keyword only (model not loaded: " + a.probeErr + ")"
	}
	if err := a.model.Failed(); err != nil {
		return "keyword only (model not loaded: " + err.Error() + ")"
	}
	return "hybrid (" + a.model.Name() + ")"
}

// HasModel reports whether search by meaning is active.
func (a *App) HasModel() bool { return a.model != nil && a.model.Failed() == nil }

// Model exposes the lazy model (nil if none).
func (a *App) Model() *embed.Lazy { return a.model }

func (a *App) chunk(body string) []string { return chunk(body, a.maxWords) }

func (a *App) doc(m *memory.Memory) index.Doc {
	return index.Doc{ID: m.ID, Path: m.Path, NS: m.NS(), Tags: m.Tags, Expires: m.Expires, Archived: m.Archived, Chunks: a.chunk(m.Body)}
}

// reloadAll rebuilds the index from every file.
func (a *App) reloadAll() error {
	head := a.Repo.Head()
	mems, bad, err := a.Repo.LoadAll()
	if err != nil {
		return err
	}
	for id, e := range bad {
		a.Warn(warnBrokenFile, "skipping %s: %v", id, e)
	}
	docs := make([]index.Doc, len(mems))
	for i, m := range mems {
		docs[i] = a.doc(m)
	}
	a.ix.Load(docs)
	ids, _ := a.Repo.IDs()
	pins := map[string]bool{}
	for _, m := range mems {
		if a.pinRelevant(m) {
			pins[m.ID] = true
		}
	}
	a.mu.Lock()
	a.head = head
	a.stamps = nil
	touched := len(pins) > 0 || len(a.pinIDs) > 0
	a.pinIDs = pins
	a.mu.Unlock()
	a.noteFiles(ids...)
	if touched {
		a.pinsChanged()
	}
	return nil
}

// pinRelevant reports whether m counts toward the pin warnings: pinned, or
// in the startup folder (archived or not).
func (a *App) pinRelevant(m *memory.Memory) bool { return m.Pin || a.inStartup(m.Path) }

// notePin records whether id now counts toward the pin warnings, and
// reports whether it did before or does now.
func (a *App) notePin(id string, now bool) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	was := a.pinIDs[id]
	if a.pinIDs == nil {
		a.pinIDs = map[string]bool{}
	}
	if now {
		a.pinIDs[id] = true
	} else {
		delete(a.pinIDs, id)
	}
	return was || now
}

// indexed puts memories griz just wrote into the index, and recomputes the
// pin warnings if any of them is (or was) pinned or in the startup folder.
func (a *App) indexed(ms ...*memory.Memory) {
	pins := false
	for _, m := range ms {
		a.ix.Add(a.doc(m))
		a.noteFiles(m.ID)
		pins = a.notePin(m.ID, a.pinRelevant(m)) || pins
	}
	if pins {
		a.pinsChanged()
	}
}

// removed takes deleted memories out of the index, like indexed.
func (a *App) removed(ids ...string) {
	pins := false
	for _, id := range ids {
		a.ix.Remove(id)
		a.noteFiles(id)
		pins = a.notePin(id, false) || pins
	}
	if pins {
		a.pinsChanged()
	}
}

// pinsChanged recomputes the pin warnings, once the server has asked for them.
func (a *App) pinsChanged() {
	if a.pinWatch.Load() {
		a.RefreshPinWarnings()
	}
}

// RefreshPinWarnings recomputes the pin warnings (startup block size, pinned
// size, pinned and archived) so they match the startup block a new session
// gets: stale ones are dropped, new ones reported. After the first call,
// they are recomputed whenever a write, scan or sync touches a memory that
// is (or was) pinned or in the startup folder.
func (a *App) RefreshPinWarnings() error {
	a.pinWatch.Store(true)
	a.pinChecks.Add(1)
	ws, err := a.PinWarnings()
	if err != nil {
		return err
	}
	a.setPinWarnings(ws)
	return nil
}

// Sync reloads files changed by other processes since the index was built.
func (a *App) Sync() error {
	head := a.Repo.Head()
	a.mu.Lock()
	old := a.head
	a.mu.Unlock()
	if head == old {
		return nil
	}
	if old == "" {
		return a.reloadAll()
	}
	ids, err := a.Repo.Changed(old, head)
	if err != nil {
		return a.reloadAll() // e.g. history rewritten by purge
	}
	a.reindex(ids...)
	a.mu.Lock()
	a.head = head
	a.mu.Unlock()
	a.vec.Reload()
	return nil
}

// reindex reloads the given ids from disk (removing any that are gone or broken).
func (a *App) reindex(ids ...string) {
	defer a.noteFiles(ids...)
	pins := false
	for _, id := range ids {
		if memory.ValidID(id) != nil {
			continue
		}
		m, err := a.Repo.Read(id)
		if err != nil {
			if a.Repo.Exists(id) {
				a.Warn(warnBrokenFile, "skipping %s: %v", id, err)
			}
			a.ix.Remove(id)
			pins = a.notePin(id, false) || pins // gone or broken
			continue
		}
		a.ix.Add(a.doc(m))
		pins = a.notePin(id, a.pinRelevant(m)) || pins
	}
	if pins {
		a.pinsChanged()
	}
}

// write runs fn under the cross-process lock, after catching up with other
// processes, and afterwards records the new HEAD.
func (a *App) write(fn func() error) error {
	unlock, err := a.Repo.Lock()
	if err != nil {
		return err
	}
	defer unlock()
	// Hand edits are committed before griz writes on top of them.
	if err := a.commitEditsLocked(); err != nil {
		a.Warn(WarnCommit, "committing hand edits: %v", err)
	}
	if err := a.Sync(); err != nil {
		return err
	}
	if err := fn(); err != nil {
		return err
	}
	a.mu.Lock()
	a.head = a.Repo.Head()
	a.mu.Unlock()
	return nil
}

// namespaces returns live namespaces with their memory counts.
func (a *App) namespaces() map[string]int {
	out := map[string]int{}
	for _, r := range a.Repo.Refs() {
		out[memory.NS(r.Path)]++
	}
	return out
}

func sortedKeys(m map[string]int) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

// ---- embedding ----

// embed runs the model, loading it if needed. A load failure is reported once
// and leaves griz keyword-only for the rest of the process.
func (a *App) embed(texts []string, op *Op) ([][]float32, error) {
	_, before := a.model.Loaded()
	t := time.Now()
	vs, err := a.model.Embed(texts)
	if _, after := a.model.Loaded(); after > before {
		op.debugf("model loaded (%s, %s, %s); unloads after %s idle", a.model.Name(), mb(a.model.Bytes()), ms(time.Since(t)), a.model.UnloadAfter())
	}
	if err != nil && a.model.Failed() != nil {
		a.warnLoad.Do(func() { a.Warn(warnModel, "search: %s", a.Status()) })
	}
	return vs, err
}

// ---- expiry ----

// GC archives expired memories (sets archived: true, keeping the expires
// date as the record) with one git commit. Returns the ids archived.
func (a *App) GC() ([]string, error) {
	var done []string
	err := a.write(func() error {
		mems, _, err := a.Repo.LoadAll()
		if err != nil {
			return err
		}
		today := memory.Today()
		var changed []*memory.Memory
		for _, m := range mems {
			if m.Expired(today) && !m.Archived {
				m.Archived = true
				changed = append(changed, m)
				done = append(done, m.ID)
			}
		}
		if len(changed) == 0 {
			return nil
		}
		for _, m := range changed {
			if err := a.Repo.Put(m); err != nil {
				return err
			}
		}
		labels := make([]string, len(done))
		for i, id := range done {
			labels[i] = a.label(id)
		}
		if err := a.Repo.CommitPaths(done, "expire: "+strings.Join(labels, ", ")); err != nil {
			return err
		}
		a.indexed(changed...)
		return nil
	})
	return done, err
}

// Unarchive clears the archived flag. expires "" removes a past expiry date
// (a future one is kept); otherwise it sets a new date, duration or none.
func (a *App) Unarchive(id, expires string) error {
	f := false
	args := StoreArgs{ID: id, Archived: &f}
	if expires != "" {
		args.Expires = &expires
	} else if m, err := a.Repo.Read(id); err == nil && m.Expired(memory.Today()) {
		none := "none"
		args.Expires = &none
	}
	_, err := a.Store(args)
	return err
}

// Restore brings back a deleted memory from git history, by its id or its
// path (ns/slug). It keeps its id; one deleted before ids existed gets a new
// one. expires "" removes the expiry; otherwise it is a date or duration.
func (a *App) Restore(key, expires string) error {
	if memory.ValidID(key) != nil && memory.ValidPath(key) != nil {
		return fmt.Errorf("restore takes a full id (host-1a2b3c4d) or a path (ns/slug), not %q", key)
	}
	exp, err := memory.ParseExpires(expires, time.Now())
	if err != nil {
		return err
	}
	return a.write(func() error {
		if a.Repo.Exists(key) {
			return fmt.Errorf("%s exists", key)
		}
		commit, rel, data, err := a.Repo.Deleted(key)
		if err != nil {
			return err
		}
		path, id, _ := memory.ParseFile(rel)
		if id == "" || a.Repo.Exists(id) {
			id = a.Repo.NewID()
		}
		if a.Repo.Exists(path) {
			return fmt.Errorf("%s exists (another memory has that path now)", path)
		}
		m, err := memory.Parse(id, data)
		if err != nil {
			return err
		}
		m.Path, m.Expires = path, exp
		if err := a.Repo.Write(m, fmt.Sprintf("restore %s (from %.8s)", m.Label(), commit)); err != nil {
			return err
		}
		a.indexed(m)
		return nil
	})
}

// ---- startup block and pinned memories ----

// pinnedLine is written by griz itself whenever pinned memories exist outside
// the startup folder, so editing a memory can't remove it.
const pinnedLine = "Before responding to anything, call `recall` with `pinned: true` and follow what it returns."

const rules = `griz is the user's long-term memory. Call recall once per topic before relying on what you know about the user; store lasting facts, preferences and corrections.
If recall returns only an id (on the "more:" or "archived:" line), get it before acting on it.`

// Warning is a labeled warning (see warn.go for labels).
type Warning struct{ Label, Text string }

func (a *App) inStartup(path string) bool {
	f := strings.Trim(a.Cfg.Pin.StartupFolder, "/")
	ns := memory.NS(path)
	return f != "" && (ns == f || strings.HasPrefix(ns, f+"/"))
}

// alwaysLoaded splits memories into the startup folder's and the pinned rest,
// leaving out archived (and expired) ones. pinnedArchived lists pinned
// memories left out because they are archived.
func (a *App) alwaysLoaded() (startup, pinned []*memory.Memory, pinnedArchived []string, err error) {
	mems, _, err := a.Repo.LoadAll()
	if err != nil {
		return nil, nil, nil, err
	}
	today := memory.Today()
	for _, m := range mems {
		switch {
		case m.IsArchived(today):
			if m.Pin && !a.inStartup(m.Path) {
				pinnedArchived = append(pinnedArchived, m.ID)
			}
		case a.inStartup(m.Path):
			startup = append(startup, m)
		case m.Pin:
			pinned = append(pinned, m)
		}
	}
	byPath := func(ms []*memory.Memory) {
		sort.Slice(ms, func(i, j int) bool { return ms[i].Path < ms[j].Path })
	}
	byPath(startup)
	byPath(pinned)
	return startup, pinned, pinnedArchived, nil
}

func entry(m *memory.Memory) string { return "### " + m.Label() + "\n" + memory.Display(m.Body) }

// largest names the n biggest memories by size, as "id (size)".
func largest(mems []*memory.Memory, n int, size func(*memory.Memory) int) string {
	s := append([]*memory.Memory(nil), mems...)
	sort.Slice(s, func(i, j int) bool { return size(s[i]) > size(s[j]) })
	var names []string
	for i, m := range s {
		if i == n {
			break
		}
		names = append(names, fmt.Sprintf("%s (%d)", m.Path, size(m)))
	}
	return strings.Join(names, ", ")
}

// StartupBlock builds the server instructions, most important first: the
// generated pinned line (only when needed), the warning notice, the usage
// rules, then every startup-folder memory in id order.
func (a *App) StartupBlock(notice string) (string, error) {
	startup, pinned, _, err := a.alwaysLoaded()
	if err != nil {
		return "", err
	}
	var parts []string
	if len(pinned) > 0 {
		parts = append(parts, pinnedLine)
	}
	if notice != "" {
		parts = append(parts, notice)
	}
	parts = append(parts, rules)
	for _, m := range startup {
		parts = append(parts, entry(m))
	}
	return strings.Join(parts, "\n\n"), nil
}

// PinWarnings checks the startup block against pin.startup_warn_chars, the
// pinned set against pin.warn_tokens, and pinned memories that are archived.
func (a *App) PinWarnings() ([]Warning, error) {
	startup, pinned, pinnedArchived, err := a.alwaysLoaded()
	if err != nil {
		return nil, err
	}
	var warns []Warning
	for _, id := range pinnedArchived {
		warns = append(warns, Warning{warnPinArchived, a.label(id) + " is pinned and archived; left out of the pinned memories"})
	}
	block, _ := a.StartupBlock("")
	if max := a.Cfg.Pin.StartupWarnChars; max > 0 && len(block) > max {
		warns = append(warns, Warning{warnStartup, fmt.Sprintf(
			"startup block is %d characters, over pin.startup_warn_chars %d (the apps cut it at about 4096); largest in %s: %s",
			len(block), max, a.Cfg.Pin.StartupFolder, largest(startup, 3, func(m *memory.Memory) int { return len(entry(m)) }))})
	}
	if t := Tokens(a.pinnedText(pinned)); a.Cfg.Pin.WarnTokens > 0 && t > a.Cfg.Pin.WarnTokens {
		warns = append(warns, Warning{warnPinned, fmt.Sprintf(
			"pinned memories are %d tokens, over pin.warn_tokens %d; largest: %s",
			t, a.Cfg.Pin.WarnTokens, largest(pinned, 3, func(m *memory.Memory) int { return Tokens(entry(m)) }))})
	}
	return warns, nil
}

func (a *App) pinnedText(pinned []*memory.Memory) string {
	parts := make([]string, len(pinned))
	for i, m := range pinned {
		parts[i] = entry(m)
	}
	return strings.Join(parts, "\n\n")
}

// Pinned answers recall with pinned: true: every pinned, unarchived memory
// outside the startup folder, in full, ignoring the budget.
func (a *App) Pinned() (string, []string, error) {
	_, pinned, _, err := a.alwaysLoaded()
	if err != nil {
		return "", nil, err
	}
	if len(pinned) == 0 {
		return "none", nil, nil
	}
	ids := make([]string, len(pinned))
	for i, m := range pinned {
		ids[i] = m.ID
	}
	return a.pinnedText(pinned), ids, nil
}

// pinnedHeader opens the pinned block that rides along with a session's first
// tool reply, so it can't be mistaken for results of that call.
const pinnedHeader = "griz: pinned memories, sent once at the start of this session (not results for this call). recall with pinned: true reloads them."

// TakePinnedFor returns the pinned block the first time it is asked for a
// session, and "" after that or when nothing is pinned. MarkPinnedSent
// records an explicit recall with pinned: true, which needs no second copy.
func (a *App) TakePinnedFor(s *Session) string {
	if s == nil || s.pinned.Swap(true) {
		return ""
	}
	text, ids, err := a.Pinned()
	if err != nil || len(ids) == 0 {
		return ""
	}
	return pinnedHeader + "\n\n" + text
}

// MarkPinnedSent records that a session has the pinned memories.
func (a *App) MarkPinnedSent(s *Session) {
	if s != nil {
		s.pinned.Store(true)
	}
}

// Tokens estimates tokens as characters / 4.
func Tokens(s string) int { return (len(s) + 3) / 4 }

var errNotFound = errors.New("not found")
