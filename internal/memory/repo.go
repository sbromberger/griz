package memory

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Repo is the data directory: a git repo of Markdown files.
type Repo struct {
	Dir     string
	Machine string // prefix for new ids (see NewID)
	// Prepare, if set, sees every memory just before it is written (griz adds
	// pending renewals there).
	Prepare func(m *Memory)
	writeMu sync.Mutex

	mu    sync.Mutex
	files map[string]string   // id -> repo-relative file
	paths map[string]string   // path (ns/slug) -> id
	dups  map[string][]string // id -> files, when several files carry it
	bad   map[string]error    // repo-relative file -> why it isn't a memory
	loose []string            // memory files with no id yet
	moved map[string][]string // id -> files it left since its last commit
}

// Open creates the directory and git repo if needed.
func Open(dir string) (*Repo, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	// The write lock lives inside .git so it never shows up as a memory or in git status.
	r := &Repo{Dir: dir}
	if _, err := os.Stat(filepath.Join(dir, ".git")); errors.Is(err, fs.ErrNotExist) {
		if _, err := r.Git("init", "-q", "-b", "main"); err != nil {
			return nil, err
		}
	}
	// Commits need an identity; fall back to a local one if none is set.
	if out, _ := r.Git("config", "user.email"); strings.TrimSpace(out) == "" {
		_, _ = r.Git("config", "user.email", "griz@localhost")
		_, _ = r.Git("config", "user.name", "griz")
	}
	return r, nil
}

// Git runs a git command in the repo and returns stdout.
func (r *Repo) Git(args ...string) (string, error) {
	cmd := exec.Command("git", append([]string{"-C", r.Dir}, args...)...)
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		return out.String(), fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(errb.String()))
	}
	return out.String(), nil
}

// Lock takes the write lock. One griz server is the only writer, so this
// only orders writes within it.
func (r *Repo) Lock() (func(), error) {
	r.writeMu.Lock()
	return r.writeMu.Unlock, nil
}

// Files.
//
// A memory is stored as ns/slug.ID.md: the folder is its namespace, the slug
// a label, ID its permanent short id (machine-8hex). An archived memory's
// file starts with a dot (ns/.slug.ID.md), so a directory listing hides it
// and ls -a shows it. Renaming or moving a file keeps its id; the frontmatter
// archived flag decides the dot, and writes move the file to match. A
// memory file with no id (added by hand) gets one from AssignIDs.

// ref is what a memory file's name says.
type ref struct {
	path     string // ns/slug
	id       string // "" if the name has none
	archived bool   // dot prefix
}

// parseRel reads a repo-relative file name. ok is false for files that
// aren't memory files at all (not .md, or a dot file griz doesn't own).
func parseRel(rel string) (ref, bool) {
	rel = filepath.ToSlash(rel)
	if !strings.HasSuffix(rel, ".md") {
		return ref{}, false
	}
	dir, base := path.Split(strings.TrimSuffix(rel, ".md"))
	f := ref{}
	base, f.archived = strings.CutPrefix(base, ".")
	if i := strings.LastIndex(base, "."); i >= 0 {
		base, f.id = base[:i], base[i+1:]
	}
	if base == "" || strings.HasPrefix(base, ".") {
		return ref{}, false
	}
	f.path = dir + base
	return f, true
}

// ParseFile reads a memory file name (repo-relative): its path and id
// ("" if it has none). ok is false if it isn't a memory file name.
func ParseFile(rel string) (path, id string, ok bool) {
	f, ok := parseRel(rel)
	return f.path, f.id, ok
}

// relFor is the file m belongs in.
func relFor(m *Memory) string {
	dir, slug := path.Split(m.Path)
	if m.Archived {
		slug = "." + slug
	}
	return dir + slug + "." + m.ID + ".md"
}

func (r *Repo) abs(rel string) string { return filepath.Join(r.Dir, filepath.FromSlash(rel)) }

func (r *Repo) exists(rel string) bool {
	_, err := os.Stat(r.abs(rel))
	return err == nil
}

// scan walks the directory and rebuilds the id maps.
func (r *Repo) scan() error {
	files, paths := map[string]string{}, map[string]string{}
	dups, bad := map[string][]string{}, map[string]error{}
	var loose []string
	err := filepath.WalkDir(r.Dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p != r.Dir && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(r.Dir, p)
		rel = filepath.ToSlash(rel)
		f, ok := parseRel(rel)
		switch {
		case !ok:
		case ValidPath(f.path) != nil:
			bad[rel] = fmt.Errorf("file name is not a valid memory path")
		case f.id == "":
			loose = append(loose, rel)
		case ValidID(f.id) != nil:
			bad[rel] = fmt.Errorf("%q in the file name is not a valid id", f.id)
		case files[f.id] != "":
			if len(dups[f.id]) == 0 {
				dups[f.id] = []string{files[f.id]}
			}
			dups[f.id] = append(dups[f.id], rel)
		default:
			files[f.id] = rel
			paths[f.path] = f.id
		}
		return nil
	})
	r.mu.Lock()
	r.files, r.paths, r.dups, r.bad, r.loose = files, paths, dups, bad, loose
	r.mu.Unlock()
	return err
}

// lookup finds the id for key, an exact id or path (ns/slug).
func (r *Repo) lookup(key string) (string, bool) {
	find := func() (string, bool) {
		r.mu.Lock()
		defer r.mu.Unlock()
		if _, ok := r.files[key]; ok {
			return key, true
		}
		id, ok := r.paths[key]
		return id, ok
	}
	if id, ok := find(); ok && r.exists(r.relOf(id)) {
		return id, true
	}
	_ = r.scan() // changed on disk since the last walk
	return find()
}

func (r *Repo) relOf(id string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.files[id]
}

// Lookup returns the id for key: an exact id, or an exact path (ns/slug).
func (r *Repo) Lookup(key string) (string, error) {
	if id, ok := r.lookup(key); ok {
		return id, nil
	}
	return "", fmt.Errorf("not found: %s", key)
}

// RelPath is the repo-relative file of key ("" if there is none).
func (r *Repo) RelPath(key string) string {
	id, _ := r.lookup(key)
	return r.relOf(id)
}

// Path is the absolute file of key ("" if there is none).
func (r *Repo) Path(key string) string {
	if rel := r.RelPath(key); rel != "" {
		return r.abs(rel)
	}
	return ""
}

func (r *Repo) Exists(key string) bool { _, ok := r.lookup(key); return ok }

// NewID returns an unused id: Machine, "-", and 8 random hex digits.
func (r *Repo) NewID() string {
	_ = r.scan()
	for {
		var b [4]byte
		rand.Read(b[:])
		id := r.Machine + "-" + hex.EncodeToString(b[:])
		if r.relOf(id) == "" {
			return id
		}
	}
}

// Read loads one memory by id or path.
func (r *Repo) Read(key string) (*Memory, error) {
	id, ok := r.lookup(key)
	if !ok {
		return nil, fmt.Errorf("not found: %s", key)
	}
	r.mu.Lock()
	rel, dup := r.files[id], r.dups[id]
	r.mu.Unlock()
	if len(dup) > 0 {
		return nil, fmt.Errorf("%s: %s all carry this id; keep one", id, strings.Join(dup, ", "))
	}
	b, err := os.ReadFile(r.abs(rel))
	if err != nil {
		return nil, err
	}
	m, err := Parse(id, b)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", id, err)
	}
	f, _ := parseRel(rel)
	m.Path = f.path
	return m, nil
}

// IDs lists every memory id (archived ones included), sorted.
func (r *Repo) IDs() ([]string, error) {
	err := r.scan()
	r.mu.Lock()
	defer r.mu.Unlock()
	ids := make([]string, 0, len(r.files))
	for id := range r.files {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids, err
}

// LoadAll reads every memory. Files that aren't usable memories are
// returned in bad, keyed by id or file name. Files waiting for an id (see
// AssignIDs) are in neither.
func (r *Repo) LoadAll() (mems []*Memory, bad map[string]error, err error) {
	ids, err := r.IDs()
	if err != nil {
		return nil, nil, err
	}
	bad = map[string]error{}
	r.mu.Lock()
	maps.Copy(bad, r.bad)
	r.mu.Unlock()
	for _, id := range ids {
		m, err := r.Read(id)
		if err != nil {
			bad[id] = err
			continue
		}
		mems = append(mems, m)
	}
	return mems, bad, nil
}

// Loose lists memory files (repo-relative) that have no id yet.
func (r *Repo) Loose() []string {
	_ = r.scan()
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.loose...)
}

// AssignIDs gives each memory file without an id, unchanged for at least
// settle, a new one, renaming it to ns/slug.ID.md (not committed; see
// CommitPaths). Returns the new ids.
func (r *Repo) AssignIDs(settle time.Duration) ([]string, error) {
	if err := r.scan(); err != nil {
		return nil, err
	}
	r.mu.Lock()
	loose := append([]string(nil), r.loose...)
	r.mu.Unlock()
	var ids []string
	for _, rel := range loose {
		if st, err := os.Stat(r.abs(rel)); err != nil || time.Since(st.ModTime()) < settle {
			continue // still being written
		}
		f, _ := parseRel(rel)
		id := r.NewID()
		dir, slug := path.Split(f.path)
		if f.archived {
			slug = "." + slug
		}
		to := dir + slug + "." + id + ".md"
		if err := os.Rename(r.abs(rel), r.abs(to)); err != nil {
			return ids, err
		}
		r.noteMove(id, rel)
		ids = append(ids, id)
	}
	if len(ids) > 0 {
		_ = r.scan()
	}
	return ids, nil
}

func (r *Repo) noteMove(id, from string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.moved == nil {
		r.moved = map[string][]string{}
	}
	r.moved[id] = append(r.moved[id], from)
}

// place moves m's file to where it belongs (its path and archived flag),
// without rewriting it.
func (r *Repo) place(m *Memory) error {
	cur, want := r.relOf(m.ID), relFor(m)
	if cur == "" || cur == want || !r.exists(cur) || r.exists(want) {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(r.abs(want)), 0o755); err != nil {
		return err
	}
	if err := os.Rename(r.abs(cur), r.abs(want)); err != nil {
		return err
	}
	r.noteMove(m.ID, cur)
	r.mu.Lock()
	r.files[m.ID] = want
	if f, _ := parseRel(cur); r.paths[f.path] == m.ID {
		delete(r.paths, f.path)
	}
	r.paths[m.Path] = m.ID
	r.mu.Unlock()
	return nil
}

// Place moves m's file to match its archived flag without committing or
// rewriting it (for hand edits that change the flag).
func (r *Repo) Place(m *Memory) error { return r.place(m) }

// put writes m to its file (moving it first if needed).
func (r *Repo) put(m *Memory) error {
	if r.Prepare != nil {
		r.Prepare(m)
	}
	if err := r.place(m); err != nil {
		return err
	}
	rel := relFor(m)
	p := r.abs(rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, m.Format(), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, p); err != nil {
		return err
	}
	r.mu.Lock()
	if r.files == nil {
		r.files, r.paths = map[string]string{}, map[string]string{}
	}
	r.files[m.ID], r.paths[m.Path] = rel, m.ID
	r.mu.Unlock()
	return nil
}

// Put writes a memory file without committing (for bulk changes).
func (r *Repo) Put(m *Memory) error { return r.put(m) }

// Write saves a memory and commits it. Caller holds the lock.
func (r *Repo) Write(m *Memory, msg string) error {
	if err := r.put(m); err != nil {
		return err
	}
	return r.CommitPaths([]string{m.ID}, msg)
}

// rels is every file to stage for ids: the current file, files it moved
// from since its last commit, and any tracked file that carries the id
// (a hand move).
func (r *Repo) rels(ids []string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(rel string) {
		if rel != "" && !seen[rel] {
			seen[rel] = true
			out = append(out, rel)
		}
	}
	var specs []string
	for _, id := range ids {
		r.mu.Lock()
		cur, moved := r.files[id], r.moved[id]
		r.mu.Unlock()
		if cur != "" && r.exists(cur) {
			add(cur)
		}
		specs = append(specs, moved...)
		specs = append(specs, "*."+id+".md")
	}
	if len(specs) > 0 {
		if out2, err := r.Git(append([]string{"ls-files", "-z", "--"}, specs...)...); err == nil {
			for f := range strings.SplitSeq(out2, "\x00") {
				add(f)
			}
		}
	}
	return out
}

// Remove deletes memories with git rm and commits once. Caller holds the lock.
func (r *Repo) Remove(ids []string, msg string) error {
	var rels []string
	for _, id := range ids {
		if rel := r.RelPath(id); rel != "" {
			rels = append(rels, rel)
		}
	}
	if _, err := r.Git(append([]string{"rm", "-q", "--"}, rels...)...); err != nil {
		return err
	}
	_ = r.scan()
	return r.commit(msg, rels...)
}

func (r *Repo) commit(msg string, paths ...string) error {
	// Nothing staged for these paths (e.g. an identical rewrite) is not an error.
	if _, err := r.Git(append([]string{"diff", "--cached", "--quiet", "--"}, paths...)...); err == nil {
		return nil
	}
	_, err := r.Git(append([]string{"commit", "-q", "--no-verify", "-m", msg, "--"}, paths...)...)
	return err
}

// CommitPaths stages the given memories (adds, changes, moves or
// deletions) and commits them once.
func (r *Repo) CommitPaths(ids []string, msg string) error {
	rels := r.rels(ids)
	if len(rels) == 0 {
		return nil
	}
	if _, err := r.Git(append([]string{"add", "-A", "--"}, rels...)...); err != nil {
		return err
	}
	if err := r.commit(msg, rels...); err != nil {
		return err
	}
	r.mu.Lock()
	for _, id := range ids {
		delete(r.moved, id)
	}
	r.mu.Unlock()
	return nil
}

// CommitAll stages everything and commits once.
func (r *Repo) CommitAll(msg string) error {
	if _, err := r.Git("add", "-A"); err != nil {
		return err
	}
	_, err := r.Git("commit", "-q", "--no-verify", "-m", msg)
	r.mu.Lock()
	r.moved = nil
	r.mu.Unlock()
	return err
}

// Head returns the current HEAD commit ("" before the first commit).
// It reads .git directly when it can, since it runs before every tool call.
func (r *Repo) Head() string {
	g := filepath.Join(r.Dir, ".git")
	if b, err := os.ReadFile(filepath.Join(g, "HEAD")); err == nil {
		ref, ok := strings.CutPrefix(strings.TrimSpace(string(b)), "ref: ")
		if !ok {
			return strings.TrimSpace(string(b))
		}
		if b, err := os.ReadFile(filepath.Join(g, filepath.FromSlash(ref))); err == nil {
			return strings.TrimSpace(string(b))
		}
	}
	out, _ := r.Git("rev-parse", "-q", "--verify", "HEAD")
	return strings.TrimSpace(out)
}

// idsOf maps file names to memory ids, once each, in order (files without
// an id are left out).
func idsOf(rels []string) []string {
	seen := map[string]bool{}
	var ids []string
	for _, rel := range rels {
		if f, ok := parseRel(strings.TrimSpace(rel)); ok && f.id != "" && !seen[f.id] {
			seen[f.id] = true
			ids = append(ids, f.id)
		}
	}
	return ids
}

// Changed lists ids of memories whose files differ between two commits.
func (r *Repo) Changed(from, to string) ([]string, error) {
	out, err := r.Git("diff", "--name-only", "--no-renames", from, to, "--", "*.md")
	if err != nil {
		return nil, err
	}
	return idsOf(strings.Split(out, "\n")), nil
}

// Dirty lists memories (by id) with uncommitted changes, including new,
// moved and deleted files. Files without an id aren't listed; AssignIDs
// gives them one first.
func (r *Repo) Dirty() ([]string, error) {
	out, err := r.Git("status", "--porcelain", "-z", "--untracked-files=all", "--no-renames", "--", "*.md")
	if err != nil {
		return nil, err
	}
	var rels []string
	for rec := range strings.SplitSeq(out, "\x00") {
		if len(rec) >= 4 {
			rels = append(rels, rec[3:])
		}
	}
	return idsOf(rels), nil
}

// Dates returns the first and last commit dates (YYYY-MM-DD) touching a memory.
func (r *Repo) Dates(key string) (created, updated string) {
	rel := r.RelPath(key)
	if rel == "" {
		return "", ""
	}
	out, _ := r.Git("log", "--follow", "--format=%cs", "--", rel)
	lines := strings.Fields(out)
	if len(lines) == 0 {
		return "", ""
	}
	return lines[len(lines)-1], lines[0]
}

// Deleted finds the last version of a memory that is gone from the working
// tree, by id or by path (for memories deleted before they had ids): the
// commit that removed it, the file's name, and its contents just before.
func (r *Repo) Deleted(key string) (commit, rel string, data []byte, err error) {
	var specs []string
	if ValidID(key) == nil {
		specs = []string{"*." + key + ".md"}
	} else {
		dir, slug := path.Split(key)
		specs = []string{key + ".md", dir + "." + slug + ".md", key + ".*.md", dir + "." + slug + ".*.md"}
	}
	out, err := r.Git(append([]string{"log", "-n1", "--diff-filter=D", "--no-renames", "--format=%H", "--name-only", "--"}, specs...)...)
	if err != nil {
		return "", "", nil, err
	}
	lines := strings.Fields(out)
	if len(lines) < 2 {
		return "", "", nil, fmt.Errorf("no deleted memory %s in history", key)
	}
	commit, rel = lines[0], lines[1]
	s, err := r.Git("show", commit+"^:"+rel)
	return commit, rel, []byte(s), err
}

// Clean reports whether the working tree has no uncommitted changes.
func (r *Repo) Clean() (bool, error) {
	out, err := r.Git("status", "--porcelain")
	return strings.TrimSpace(out) == "", err
}

// Stat returns size and modification time of a memory file (ok=false if missing).
func (r *Repo) Stat(key string) (size int64, mtime time.Time, ok bool) {
	p := r.Path(key)
	if p == "" {
		return 0, time.Time{}, false
	}
	st, err := os.Stat(p)
	if err != nil {
		return 0, time.Time{}, false
	}
	return st.Size(), st.ModTime(), true
}

// PathOf is id's path (ns/slug): from its file, or for a memory deleted
// but not yet committed, from git. "" if neither knows it.
func (r *Repo) PathOf(id string) string {
	rel := r.relOf(id)
	if rel == "" {
		out, _ := r.Git("ls-files", "-z", "--", "*."+id+".md")
		rel, _, _ = strings.Cut(out, "\x00")
	}
	f, _ := parseRel(rel)
	return f.path
}

// Ref is a memory's id and path, for matching short keys (see app).
type Ref struct{ ID, Path string }

// Refs lists every memory's id and path, sorted by id.
func (r *Repo) Refs() []Ref {
	_ = r.scan()
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Ref, 0, len(r.files))
	for id, rel := range r.files {
		f, _ := parseRel(rel)
		out = append(out, Ref{id, f.path})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
