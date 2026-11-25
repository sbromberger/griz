package app

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Debug mode and timings live in this process's memory only: no files, no
// signals, no port. Both carry memory ids, never memory text.

// ---- sessions and debug lines ----

// Session is one connected Claude session (one relay connection). Debug mode
// and warning delivery are per session. App.Local is the session used when
// griz runs without a server (tests, CLI).
type Session struct {
	ID     int
	Since  time.Time
	Client string // the relay's pid ("pid 123"), set by the server

	debug  atomic.Bool
	pinned atomic.Bool // pinned memories already sent to this session
	mu     sync.Mutex
	lines  []string
	seen   int // warnings already passed to this session
}

type sessions struct {
	mu   sync.Mutex
	next int
	all  map[*Session]bool
}

// NewSession registers a connected session.
func (a *App) NewSession() *Session {
	a.sess.mu.Lock()
	defer a.sess.mu.Unlock()
	if a.sess.all == nil {
		a.sess.all = map[*Session]bool{}
	}
	a.sess.next++
	s := &Session{ID: a.sess.next, Since: time.Now()}
	s.debug.Store(a.Local.Debugging()) // griz debug on covers sessions that connect later
	a.sess.all[s] = true
	return s
}

// EndSession forgets a session whose connection closed.
func (a *App) EndSession(s *Session) {
	a.sess.mu.Lock()
	defer a.sess.mu.Unlock()
	delete(a.sess.all, s)
}

// Sessions lists connected sessions, oldest first.
func (a *App) Sessions() []*Session {
	a.sess.mu.Lock()
	defer a.sess.mu.Unlock()
	out := make([]*Session, 0, len(a.sess.all))
	for s := range a.sess.all {
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// SetDebug turns debug mode on or off for a session (nil: App.Local).
func (a *App) SetDebug(on bool) { a.Local.SetDebug(on) }

// SetDebugAll turns debug mode on or off for App.Local, every connected
// session, and sessions that connect while it is on (griz debug on|off).
func (a *App) SetDebugAll(on bool) {
	a.Local.SetDebug(on)
	for _, s := range a.Sessions() {
		s.SetDebug(on)
	}
}

// SetDebug turns debug mode on or off for this session.
func (s *Session) SetDebug(on bool) {
	s.debug.Store(on)
	if !on {
		s.mu.Lock()
		s.lines = nil
		s.mu.Unlock()
	}
}

// Debugging reports whether debug mode is on for this session.
func (s *Session) Debugging() bool { return s != nil && s.debug.Load() }

func (s *Session) add(line string) {
	if !s.Debugging() {
		return
	}
	s.mu.Lock()
	s.lines = append(s.lines, "griz debug: "+line)
	s.mu.Unlock()
}

// TakeDebug returns the session's debug lines gathered since the last call.
func (s *Session) TakeDebug() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := s.lines
	s.lines = nil
	return out
}

// TakeDebug returns App.Local's debug lines.
func (a *App) TakeDebug() []string { return a.Local.TakeDebug() }

// Debugging reports whether App.Local is in debug mode.
func (a *App) Debugging() bool { return a.Local.Debugging() }

// debugf records an event not tied to one call (scan, model unload) for
// every session in debug mode.
func (a *App) debugf(format string, args ...any) {
	line := fmt.Sprintf(format, args...)
	a.Local.add(line)
	for _, s := range a.Sessions() {
		s.add(line)
	}
}

// debugf records what this operation did, for its session's next reply.
func (o *Op) debugf(format string, args ...any) {
	if o != nil && o.all != nil {
		o.all.debugf(format, args...)
		return
	}
	if o == nil || !o.Sess.Debugging() {
		return
	}
	o.Sess.add(fmt.Sprintf(format, args...))
}

// ---- timings ----

// Op is one operation (a tool call, or embedding started by store or scan)
// split into timed steps. A nil *Op records nothing.
type Op struct {
	Sess  *Session // whose call this is (debug lines go there)
	all   *App     // set instead of Sess: debug lines go to every session
	Tool  string
	Start time.Time
	total time.Duration
	IDs   []string

	mu    sync.Mutex
	steps []step
}

type step struct {
	name string
	d    time.Duration
}

// Since records step name as taking the time since t0, and returns now, so
// steps can be chained: t = op.Since("keyword", t).
func (o *Op) Since(name string, t0 time.Time) time.Time {
	now := time.Now()
	o.Add(name, now.Sub(t0))
	return now
}

// Add records step name as taking d (added to any earlier time for name).
func (o *Op) Add(name string, d time.Duration) {
	if o == nil {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	for i := range o.steps {
		if o.steps[i].name == name {
			o.steps[i].d += d
			return
		}
	}
	o.steps = append(o.steps, step{name, d})
}

func (o *Op) step(name string) time.Duration {
	o.mu.Lock()
	defer o.mu.Unlock()
	for _, s := range o.steps {
		if s.name == name {
			return s.d
		}
	}
	return 0
}

type opLog struct {
	mu  sync.Mutex
	ops []*Op // oldest first
}

// BeginOp starts timing an operation for session s (nil: App.Local).
func (a *App) BeginOp(s *Session, tool string) *Op {
	if s == nil {
		s = a.Local
	}
	return &Op{Sess: s, Tool: tool, Start: time.Now()}
}

// beginShared starts an operation not tied to one session (the scan's).
func (a *App) beginShared(tool string) *Op { return &Op{all: a, Tool: tool, Start: time.Now()} }

// EndOp records op among the recent operations (debug.timings_kept of them).
func (a *App) EndOp(op *Op, ids []string) {
	if op == nil {
		return
	}
	op.total = time.Since(op.Start)
	op.IDs = ids
	keep := a.Cfg.Debug.TimingsKept
	if keep <= 0 {
		return
	}
	a.ops.mu.Lock()
	defer a.ops.mu.Unlock()
	a.ops.ops = append(a.ops.ops, op)
	if over := len(a.ops.ops) - keep; over > 0 {
		a.ops.ops = append([]*Op(nil), a.ops.ops[over:]...)
	}
}

// Timings summarizes recent operations: per tool and step, the typical
// (median) and slowest time, then the last few operations.
func (a *App) Timings() string {
	a.ops.mu.Lock()
	ops := append([]*Op(nil), a.ops.ops...)
	a.ops.mu.Unlock()
	if len(ops) == 0 {
		return "no operations yet in this process"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "recent operations in this process: %d (since %s)\n", len(ops), ops[0].Start.Format("15:04:05"))
	byTool := map[string][]*Op{}
	var tools []string
	for _, o := range ops {
		if byTool[o.Tool] == nil {
			tools = append(tools, o.Tool)
		}
		byTool[o.Tool] = append(byTool[o.Tool], o)
	}
	sort.Strings(tools)
	for _, t := range tools {
		list := byTool[t]
		totals := make([]time.Duration, len(list))
		var names []string
		seen := map[string]bool{}
		for i, o := range list {
			totals[i] = o.total
			o.mu.Lock()
			for _, s := range o.steps {
				if !seen[s.name] {
					seen[s.name] = true
					names = append(names, s.name)
				}
			}
			o.mu.Unlock()
		}
		med, max := spread(totals)
		fmt.Fprintf(&b, "\n%s ×%d: typical %s, slowest %s\n", t, len(list), ms(med), ms(max))
		for _, n := range names {
			ds := make([]time.Duration, len(list))
			for i, o := range list {
				ds[i] = o.step(n)
			}
			med, max := spread(ds)
			fmt.Fprintf(&b, "  %-16s typical %s, slowest %s\n", n, ms(med), ms(max))
		}
	}
	b.WriteString("\nlast operations:\n")
	for _, o := range ops[max(0, len(ops)-5):] {
		fmt.Fprintf(&b, "  %s %s %s", o.Start.Format("15:04:05"), o.Tool, ms(o.total))
		o.mu.Lock()
		for _, s := range o.steps {
			fmt.Fprintf(&b, ", %s %s", s.name, ms(s.d))
		}
		o.mu.Unlock()
		if len(o.IDs) > 0 {
			fmt.Fprintf(&b, " [%s]", strings.Join(o.IDs, ", "))
		}
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

func spread(ds []time.Duration) (median, max time.Duration) {
	s := append([]time.Duration(nil), ds...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[len(s)/2], s[len(s)-1]
}

// ms formats a duration for people: 0.4 ms, 410 ms, 2.3 s.
func ms(d time.Duration) string {
	switch {
	case d < 10*time.Millisecond:
		return fmt.Sprintf("%.1f ms", float64(d)/float64(time.Millisecond))
	case d < 10*time.Second:
		return fmt.Sprintf("%d ms", d.Milliseconds())
	default:
		return fmt.Sprintf("%.1f s", d.Seconds())
	}
}

func mb(n int64) string { return fmt.Sprintf("%.0f MB", float64(n)/(1<<20)) }
