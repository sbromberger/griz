package memory

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testRepo(t *testing.T) *Repo {
	r, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	r.Machine = "test"
	return r
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

// Files are ns/slug.ID.md; archived ones get a dot; the id survives moves.
func TestFileNames(t *testing.T) {
	r := testRepo(t)
	unlock, _ := r.Lock()
	defer unlock()
	m := &Memory{ID: r.NewID(), Path: "ns/name", Body: "hello"}
	if ValidID(m.ID) != nil || !strings.HasPrefix(m.ID, "test-") {
		t.Fatalf("new id %q", m.ID)
	}
	plain := filepath.Join(r.Dir, "ns/name."+m.ID+".md")
	dotted := filepath.Join(r.Dir, "ns/.name."+m.ID+".md")
	if err := r.Write(m, "store"); err != nil {
		t.Fatal(err)
	}
	if !exists(plain) {
		t.Fatal("want ns/name.ID.md")
	}
	for _, key := range []string{m.ID, "ns/name"} {
		if got, err := r.Read(key); err != nil || got.ID != m.ID || got.Path != "ns/name" {
			t.Errorf("read %q: %+v %v", key, got, err)
		}
	}
	head := r.Head()
	m.Archived = true
	if err := r.Write(m, "archive"); err != nil {
		t.Fatal(err)
	}
	if exists(plain) || !exists(dotted) {
		t.Fatal("archiving should add the dot")
	}
	if ch, _ := r.Changed(head, r.Head()); len(ch) != 1 || ch[0] != m.ID {
		t.Errorf("changed: %v", ch)
	}
	if clean, _ := r.Clean(); !clean {
		t.Error("the move should be committed")
	}

	// A hand move to another folder and slug keeps the id.
	moved := filepath.Join(r.Dir, "other/renamed."+m.ID+".md")
	os.MkdirAll(filepath.Dir(moved), 0o755)
	os.Rename(dotted, moved)
	got, err := r.Read(m.ID)
	if err != nil || got.Path != "other/renamed" {
		t.Fatalf("after hand move: %+v %v", got, err)
	}
	if d, _ := r.Dirty(); len(d) != 1 || d[0] != m.ID {
		t.Errorf("dirty: %v", d)
	}
	if err := r.Place(got); err != nil || !exists(filepath.Join(r.Dir, "other/.renamed."+m.ID+".md")) {
		t.Fatalf("place should restore the dot (archived): %v", err)
	}
	if err := r.CommitPaths([]string{m.ID}, "edit"); err != nil {
		t.Fatal(err)
	}
	if clean, _ := r.Clean(); !clean {
		t.Error("CommitPaths should stage the move")
	}

	// A file added by hand gets an id.
	os.WriteFile(filepath.Join(r.Dir, "ns/new.md"), []byte("by hand\n"), 0o644)
	ids, err := r.AssignIDs(0)
	if err != nil || len(ids) != 1 || !exists(filepath.Join(r.Dir, "ns/new."+ids[0]+".md")) {
		t.Fatalf("assign: %v %v", ids, err)
	}

	// Two files with one id: broken until one goes.
	cp := filepath.Join(r.Dir, "ns/copy."+ids[0]+".md")
	os.WriteFile(cp, []byte("copy\n"), 0o644)
	r.scan()
	if _, err := r.Read(ids[0]); err == nil || !strings.Contains(err.Error(), "carry this id") {
		t.Errorf("duplicate id: %v", err)
	}
	os.Remove(cp)
	r.scan()

	// Deleted memories can be found by id, for restore.
	if err := r.Remove([]string{m.ID}, "delete"); err != nil {
		t.Fatal(err)
	}
	if _, rel, data, err := r.Deleted(m.ID); err != nil || !strings.Contains(string(data), "hello") || !strings.Contains(rel, m.ID) {
		t.Errorf("deleted: %q %v", rel, err)
	}
}

func TestMachine(t *testing.T) {
	for in, want := range map[string]string{"host": "host", "Alexs-MacBook-Pro.local": "alexs-macbook-pro", "a_b c": "a-b-c", "--x--": "x"} {
		if got := Machine(in); got != want {
			t.Errorf("Machine(%q) = %q, want %q", in, got, want)
		}
	}
}
