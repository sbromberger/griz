package app

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

// Warning labels, used to summarize warnings in one line for Claude.
const (
	warnBrokenFile  = "broken file"
	warnPinned      = "pinned memories over budget"
	warnPinArchived = "pinned memory is archived"
	warnStartup     = "startup block near the size limit"
	warnModel       = "model can't load"
	WarnCommit      = "hand-edit commit failed"
	warnEmbed       = "embedding failed"
	warnScan        = "scan failed"
	WarnExpiry      = "expiry cleanup failed"
	WarnCrash       = "an earlier griz server crashed"
)

type warning struct{ label, text string }

// warnings collects warnings once per process: all go to stderr. Each
// session gets each warning once: those from before it connected are
// summarized in its instructions, later ones are added to its next tool
// reply (when warn.via_claude is on).
type warnings struct {
	mu   sync.Mutex
	seen map[string]bool
	all  []warning
	pin  map[string]bool // the pin warnings that still apply (see setPinWarnings)
}

func isPinLabel(l string) bool { return l == warnPinned || l == warnPinArchived || l == warnStartup }

// stale reports a pin warning that no longer applies. Caller holds mu.
func (w *warnings) stale(x warning) bool { return isPinLabel(x.label) && !w.pin[x.text] }

// current returns the warnings that still apply.
func (a *App) current() []warning {
	a.warns.mu.Lock()
	defer a.warns.mu.Unlock()
	var out []warning
	for _, w := range a.warns.all {
		if !a.warns.stale(w) {
			out = append(out, w)
		}
	}
	return out
}

// setPinWarnings replaces the pin warnings. Ones that no longer apply stop
// showing in status and aren't delivered to sessions; one that comes back
// later is reported again.
func (a *App) setPinWarnings(ws []Warning) {
	cur := map[string]bool{}
	for _, w := range ws {
		cur[w.Text] = true
	}
	a.warns.mu.Lock()
	for t := range a.warns.pin {
		if !cur[t] {
			delete(a.warns.seen, t)
		}
	}
	a.warns.pin = cur
	a.warns.mu.Unlock()
	for _, w := range ws {
		a.Warn(w.Label, "%s", w.Text)
	}
}

func stderr(w io.Writer) io.Writer {
	if w == nil {
		return os.Stderr
	}
	return w
}

// Warn reports a problem the user should hear about.
func (a *App) Warn(label, format string, args ...any) {
	text := fmt.Sprintf(format, args...)
	a.warns.mu.Lock()
	defer a.warns.mu.Unlock()
	if a.warns.seen == nil {
		a.warns.seen = map[string]bool{}
	}
	if a.warns.seen[text] {
		return
	}
	a.warns.seen[text] = true
	fmt.Fprintln(stderr(a.Stderr), "griz: "+text)
	a.warns.all = append(a.warns.all, warning{label, text})
}

// Info prints a notice to stderr only.
func (a *App) Info(format string, args ...any) {
	fmt.Fprintf(stderr(a.Stderr), "griz: "+format+"\n", args...)
}

// startupNotice is StartupNoticeFor(App.Local).
func (a *App) startupNotice() string { return a.StartupNoticeFor(a.Local) }

// StartupNoticeFor summarizes the warnings s hasn't had in one line for its
// instructions, and marks them delivered. "" when there are none.
func (a *App) StartupNoticeFor(s *Session) string {
	ws := a.unseen(s)
	if !a.Cfg.Warn.ViaClaude || len(ws) == 0 {
		return ""
	}
	var order []string
	count := map[string]int{}
	for _, w := range ws {
		if count[w.label] == 0 {
			order = append(order, w.label)
		}
		count[w.label]++
	}
	parts := make([]string, len(order))
	for i, l := range order {
		n := count[l]
		switch {
		case l == warnBrokenFile:
			parts[i] = fmt.Sprintf("%d %s with broken frontmatter", n, plural(n, "file", "files"))
		case n > 1:
			parts[i] = fmt.Sprintf("%s (%d)", l, n)
		default:
			parts[i] = l
		}
	}
	return fmt.Sprintf("griz: %d %s (%s). Tell the user once and suggest griz memory lint.",
		len(ws), plural(len(ws), "warning", "warnings"), strings.Join(parts, "; "))
}

// unseen returns the warnings s hasn't had, and marks them delivered.
func (a *App) unseen(s *Session) []warning {
	a.warns.mu.Lock()
	defer a.warns.mu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	var ws []warning
	for _, w := range a.warns.all[s.seen:] {
		if !a.warns.stale(w) {
			ws = append(ws, w)
		}
	}
	s.seen = len(a.warns.all)
	return ws
}

// takeWarnings is TakeWarningsFor(App.Local).
func (a *App) takeWarnings() []string { return a.TakeWarningsFor(a.Local) }

// TakeWarningsFor returns warnings not yet passed to s, as reply lines.
func (a *App) TakeWarningsFor(s *Session) []string {
	ws := a.unseen(s)
	if !a.Cfg.Warn.ViaClaude {
		return nil
	}
	out := make([]string, len(ws))
	for i, w := range ws {
		out[i] = "griz warning: " + w.text
	}
	return out
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}
