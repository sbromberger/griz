package app

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"griz/internal/memory"
)

// Fading: recall weighs each memory by how long ago it was last edited or
// renewed, relative to its term's half-life:
//
//	weight = floor + (1 - floor) * 0.5^(age / half-life)
//
// Nothing is archived or forgotten for its age. Edits, renew, unarchive and
// restore renew a memory fully; a get takes back part of its age
// (term.get_renews). Being shown by recall never renews.
//
// Renewals from gets are kept here until they are written: with any other
// write to the file, every term.flush, and when the server stops.

type renewals struct {
	mu      sync.Mutex
	pending map[string]string // id -> renewed date not yet written
}

func (r *renewals) get(id string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.pending[id]
}

func (r *renewals) take(id string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	d := r.pending[id]
	delete(r.pending, id)
	return d
}

func (r *renewals) set(id, date string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.pending == nil {
		r.pending = map[string]string{}
	}
	if date > r.pending[id] {
		r.pending[id] = date
	}
}

func (r *renewals) ids() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	ids := make([]string, 0, len(r.pending))
	for id := range r.pending {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// prepare is the repo's hook before every write: a pending renewal goes
// into the file along with whatever else is being written.
func (a *App) prepare(m *memory.Memory) {
	if d := a.renew.take(m.ID); d > m.Renewed && d > m.Updated {
		m.Renewed = d
	}
}

// halfLife is term's half-life in days (0: it doesn't fade).
func (a *App) halfLife(term string) float64 {
	if term == "" {
		term = a.Cfg.Store.DefaultTerm
	}
	var life string
	switch term {
	case "short":
		life = a.Cfg.Term.Short
	case "medium":
		life = a.Cfg.Term.Medium
	case "long":
		life = a.Cfg.Term.Long
	default: // infinite
		return 0
	}
	base := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
	d, err := memory.ParseExpires(life, base)
	if err != nil {
		return 0
	}
	t, _ := time.Parse(memory.DateLayout, d)
	return t.Sub(base).Hours() / 24
}

// since is the date id's fading counts from, pending renewals included.
func (a *App) since(id string) string {
	_, s, _ := a.ix.Fade(id)
	return max(s, a.renew.get(id))
}

// ageDays is the number of whole days from date to today (0 if unknown).
func ageDays(date, today string) float64 {
	if date == "" {
		return 0
	}
	s, err1 := time.Parse(memory.DateLayout, date)
	t, err2 := time.Parse(memory.DateLayout, today)
	if err1 != nil || err2 != nil || !t.After(s) {
		return 0
	}
	return math.Round(t.Sub(s).Hours() / 24)
}

// weight is id's fading factor for recall on date today.
func (a *App) weight(id, today string) float64 {
	term, _, pin := a.ix.Fade(id)
	h := a.halfLife(term)
	if pin || h <= 0 {
		return 1
	}
	f := a.Cfg.Term.Floor
	return f + (1-f)*math.Pow(0.5, ageDays(a.since(id), today)/h)
}

// faded reports whether id is at least lint.faded_half_lives half-lives old.
func (a *App) faded(id, today string) bool {
	term, _, pin := a.ix.Fade(id)
	h := a.halfLife(term)
	return !pin && h > 0 && a.Cfg.Lint.FadedHalves > 0 && ageDays(a.since(id), today) >= h*float64(a.Cfg.Lint.FadedHalves)
}

// renewOnGet takes back part of id's age (term.get_renews), in whole days.
// Only the server keeps renewals: a get without one doesn't renew.
func (a *App) renewOnGet(id string) {
	if !a.Server || a.Cfg.Term.GetRenews <= 0 {
		return
	}
	today := memory.Today()
	since := a.since(id)
	age := ageDays(since, today)
	back := math.Floor(age * a.Cfg.Term.GetRenews)
	if back < 1 {
		return // already renewed today, or too recently to move by a day
	}
	t, _ := time.Parse(memory.DateLayout, since)
	if since == "" {
		t, _ = time.Parse(memory.DateLayout, today)
	}
	a.renew.set(id, t.AddDate(0, 0, int(back)).Format(memory.DateLayout))
}

// FlushRenewals writes pending renewals from gets in one commit. A memory
// with an uncommitted hand edit is left for the next flush.
func (a *App) FlushRenewals() error {
	ids := a.renew.ids()
	if len(ids) == 0 {
		return nil
	}
	return a.write(func() error {
		var done []*memory.Memory
		var labels []string
		for _, id := range ids {
			if st, err := a.Repo.Git("status", "--porcelain", "--", a.Repo.RelPath(id)); err != nil || strings.TrimSpace(st) != "" {
				continue
			}
			m, err := a.Repo.Read(id)
			if err != nil {
				a.renew.take(id) // gone
				continue
			}
			if d := a.renew.get(id); d <= m.Renewed || d <= m.Updated {
				a.renew.take(id) // nothing newer to write
				continue
			}
			if err := a.Repo.Put(m); err != nil { // prepare adds the renewal
				return err
			}
			done = append(done, m)
			labels = append(labels, m.Label())
		}
		if len(done) == 0 {
			return nil
		}
		dids := make([]string, len(done))
		for i, m := range done {
			dids[i] = m.ID
		}
		if err := a.Repo.CommitPaths(dids, "renew: "+strings.Join(labels, ", ")); err != nil {
			return err
		}
		a.indexed(done...)
		return nil
	})
}

// StartRenewFlusher writes pending renewals every term.flush until stop is
// closed, then once more.
func (a *App) StartRenewFlusher(stop <-chan struct{}) (done <-chan struct{}) {
	every, _ := a.Cfg.Term.FlushDuration()
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		var tick <-chan time.Time
		if every > 0 {
			t := time.NewTicker(every)
			defer t.Stop()
			tick = t.C
		}
		for {
			select {
			case <-stop:
				if err := a.FlushRenewals(); err != nil {
					a.Warn(WarnCommit, "writing renewals: %v", err)
				}
				return
			case <-tick:
				if err := a.FlushRenewals(); err != nil {
					a.Warn(WarnCommit, "writing renewals: %v", err)
				}
			}
		}
	}()
	return ch
}

// fadeNote describes id's fading for explain: its age and weight.
func (a *App) fadeNote(id, today string) string {
	term, _, pin := a.ix.Fade(id)
	switch {
	case pin:
		return "pinned, no fading"
	case a.halfLife(term) <= 0:
		return "infinite, no fading"
	}
	return fmt.Sprintf("%.0f days since %s, weight %.2f", ageDays(a.since(id), today), a.since(id), a.weight(id, today))
}

// migrateMark is the subject of the commit that dropped the expiry dates
// terms used to set; its presence in history means the migration is done.
const migrateMark = "migrate: memories fade instead of expiring; drop every term-derived expires"

// MigrateExpiry runs once per memory repo: before fading, every term set
// an expires date. Those weren't chosen, so all of them go, in one commit
// (an empty one if none had a date, to mark it done). From then on a date
// exists only when someone sets it.
func (a *App) MigrateExpiry() (int, error) {
	if log, err := a.Repo.Git("log", "--format=%s", "--fixed-strings", "--grep", migrateMark); err == nil && strings.Contains(log, migrateMark) {
		return 0, nil
	}
	n := 0
	err := a.write(func() error {
		mems, _, err := a.Repo.LoadAll()
		if err != nil {
			return err
		}
		var changed []*memory.Memory
		var ids []string
		for _, m := range mems {
			if m.Expires != "" {
				m.Expires = ""
				if err := a.Repo.Put(m); err != nil {
					return err
				}
				changed = append(changed, m)
				ids = append(ids, m.ID)
			}
		}
		n = len(changed)
		if n == 0 {
			_, err := a.Repo.Git("commit", "--allow-empty", "-q", "-m", migrateMark)
			return err
		}
		if err := a.Repo.CommitPaths(ids, migrateMark); err != nil {
			return err
		}
		a.indexed(changed...)
		return nil
	})
	return n, err
}

// Faded reports whether id has faded close to the floor (lint lists them).
func (a *App) Faded(id string) bool { return a.faded(id, memory.Today()) }

// Since is the date id's fading counts from, renewals from gets included.
func (a *App) Since(id string) string { return a.since(id) }
