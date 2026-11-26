package app

import (
	"strings"
	"time"

	"griz/internal/memory"
)

// scanReport is what the last scan found, for griz status.
type scanReport struct {
	at        time.Time
	changed   []string
	committed []string
	err       error
}

// stamp is what a memory file looked like when the index last read it.
type stamp struct {
	size  int64
	mtime time.Time
}

// noteFiles records the current stamps of ids (a missing file drops its stamp).
func (a *App) noteFiles(ids ...string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.stamps == nil {
		a.stamps = map[string]stamp{}
	}
	for _, id := range ids {
		if size, mt, ok := a.Repo.Stat(id); ok {
			a.stamps[id] = stamp{size, mt}
		} else {
			delete(a.stamps, id)
		}
	}
}

// changedFiles lists ids whose files were added, changed or removed since the
// index last read them.
func (a *App) changedFiles() []string {
	ids, err := a.Repo.IDs()
	if err != nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	var out []string
	seen := map[string]bool{}
	for _, id := range ids {
		seen[id] = true
		size, mt, ok := a.Repo.Stat(id)
		if s, had := a.stamps[id]; !ok || !had || s.size != size || !s.mtime.Equal(mt) {
			out = append(out, id)
		}
	}
	for id := range a.stamps {
		if !seen[id] {
			out = append(out, id)
		}
	}
	return out
}

// CommitEdits commits hand edits (files changed outside griz) that have
// settled, in one commit. It takes the write lock.
func (a *App) CommitEdits() error {
	if !a.Cfg.Scan.CommitEdits {
		return nil
	}
	unlock, err := a.Repo.Lock()
	if err != nil {
		return err
	}
	defer unlock()
	return a.commitEditsLocked()
}

// commitEditsLocked does the work; the caller holds the write lock, so the
// git status read here can't race another process's commit of the same edit.
func (a *App) commitEditsLocked() error {
	if !a.Cfg.Scan.CommitEdits {
		return nil
	}
	settle, _ := a.Cfg.Scan.SettleDuration()
	if ids, err := a.Repo.AssignIDs(settle); err != nil {
		a.Warn(WarnCommit, "giving new memory files ids: %v", err)
	} else if len(ids) > 0 {
		a.debugf("new memory files got ids: %s", strings.Join(ids, ", "))
	}
	dirty, err := a.Repo.Dirty()
	if err != nil || len(dirty) == 0 {
		return err
	}
	var ready []string
	for _, id := range dirty {
		if _, mt, ok := a.Repo.Stat(id); ok && time.Since(mt) < settle {
			continue // still being edited
		}
		ready = append(ready, id)
		if a.Repo.Exists(id) {
			a.applyHandEdit(id, true)
			if err := memory.ValidID(id); err != nil {
				a.Warn(warnBrokenFile, "committing %s, but not indexing it: %v", id, err)
			} else if m, err := a.Repo.Read(id); err != nil {
				a.Warn(warnBrokenFile, "committing %s, but not indexing it: %v", id, err)
			} else if err := a.Repo.Place(m); err != nil { // archived flag changed by hand: move the file
				a.Warn(WarnCommit, "moving %s to match its archived flag: %v", id, err)
			}
		}
	}
	if len(ready) == 0 {
		return nil
	}
	labels := make([]string, len(ready))
	for i, id := range ready {
		labels[i] = a.label(id)
	}
	if err := a.Repo.CommitPaths(ready, "edit (manual): "+strings.Join(labels, ", ")); err != nil {
		return err
	}
	a.mu.Lock()
	a.lastScan.committed = append(a.lastScan.committed, ready...)
	a.mu.Unlock()
	a.debugf("committed hand edits: %s", strings.Join(ready, ", "))
	return nil
}

// Scan picks up changes made outside this process: commits settled hand
// edits, reloads changed files, rereads vectors.bin if another process
// replaced it, and embeds missing vectors only if the model is already loaded.
func (a *App) Scan() (err error) {
	a.mu.Lock()
	a.lastScan = scanReport{at: time.Now()}
	a.mu.Unlock()
	defer func() {
		a.mu.Lock()
		a.lastScan.err = err
		a.mu.Unlock()
	}()
	if err := a.CommitEdits(); err != nil {
		a.Warn(WarnCommit, "committing hand edits: %v", err)
	}
	if err := a.Sync(); err != nil {
		return err
	}
	if _, err := a.ArchiveExpired(); err != nil {
		a.Warn(WarnExpiry, "archiving expired memories: %v", err)
	}
	ids := a.changedFiles()
	if len(ids) > 0 {
		a.mu.Lock()
		a.lastScan.changed = ids
		a.mu.Unlock()
		a.debugf("scan: reloading %s", strings.Join(ids, ", "))
		if !a.Cfg.Scan.CommitEdits {
			for _, id := range ids {
				a.applyHandEdit(id, false)
			}
		}
		a.reindex(ids...)
	}
	if a.model != nil {
		if loaded, _ := a.model.Loaded(); loaded {
			a.embedAsync(nil, "scan", ids...)
		}
	}
	return nil
}

// StartScanner runs Scan every scan.interval until stop is closed.
func (a *App) StartScanner(stop <-chan struct{}) {
	every, _ := a.Cfg.Scan.IntervalDuration()
	if every <= 0 {
		return
	}
	go func() {
		t := time.NewTicker(every)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				if err := a.Scan(); err != nil {
					a.Warn(warnScan, "scan: %v", err)
				}
			}
		}
	}()
}

// applyHandEdit fixes up a memory changed by hand, comparing it with its
// last committed version. When stamp is set (the edit is about to be
// committed), updated becomes today, and a new file gets created too, unless
// the edit set those dates itself.
func (a *App) applyHandEdit(id string, stamp bool) {
	m, err := a.Repo.Read(id)
	if err != nil {
		return
	}
	var prev *memory.Memory
	if old, err := a.Repo.Git("show", "HEAD:"+a.Repo.RelPath(id)); err == nil {
		prev, _ = memory.Parse(id, []byte(old))
	}
	if prev == nil {
		prev = &memory.Memory{}
	}
	changed := false
	if stamp && m.Created == prev.Created && m.Updated == prev.Updated {
		today := memory.Today()
		if m.Updated != today {
			m.Updated, changed = today, true
		}
		if prev.ID == "" && m.Created == "" {
			m.Created, changed = today, true
		}
	}
	if changed {
		if err := a.Repo.Put(m); err != nil {
			a.Warn(WarnCommit, "updating dates for %s: %v", id, err)
		}
	}
}
