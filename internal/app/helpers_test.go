package app

import (
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"regexp"

	"griz/internal/memory"
)

// file is where a test writes memory path p by hand: its current file, or a
// new one named with a fresh id (as griz names them).
func file(a *App, p string) string {
	if f := a.Repo.Path(p); f != "" {
		return f
	}
	dir, slug := filepath.Split(p)
	f := filepath.Join(a.Repo.Dir, dir, slug+"."+a.Repo.NewID()+".md")
	os.MkdirAll(filepath.Dir(f), 0o755)
	return f
}

// put writes m without committing. A test gives the path as m.ID; put
// keeps the id the path already has, or makes a new one.
func put(a *App, m *memory.Memory) {
	if m.Path == "" {
		m.Path = m.ID
	}
	if id, err := a.Repo.Lookup(m.Path); err == nil {
		m.ID = id
	} else {
		m.ID = a.Repo.NewID()
	}
	a.Repo.Put(m)
}

var labelIDRE = regexp.MustCompile(` \[[a-z0-9-]+-[0-9a-f]{8}\]`)

// noIDs drops the " [id]" griz adds after paths, for comparing output.
func noIDs(s string) string { return labelIDRE.ReplaceAllString(s, "") }

// testName is a file name for memory path p with an id made from p, for
// tests that write files before griz runs.
func testName(p string) string {
	return fmt.Sprintf("%s.test-%08x.md", p, crc32.ChecksumIEEE([]byte(p)))
}
