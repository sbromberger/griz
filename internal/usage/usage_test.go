package usage

import (
	"os"
	"testing"
	"time"
)

// The log names memories, so only this user may read it: when it is made,
// when it was left looser, and after Forget rewrites it.
func TestLogIsPrivate(t *testing.T) {
	l := New(t.TempDir(), true)
	mode := func() os.FileMode {
		fi, err := os.Stat(l.Path)
		if err != nil {
			t.Fatal(err)
		}
		return fi.Mode().Perm()
	}
	e := Entry{Time: time.Now(), Tool: "recall", IDs: []string{"host-1a2b3c4d"}}
	l.Write(e)
	if m := mode(); m != 0o600 {
		t.Errorf("new log: %v", m)
	}
	os.Chmod(l.Path, 0o644)
	l.Write(e)
	if m := mode(); m != 0o600 {
		t.Errorf("a looser log should be tightened: %v", m)
	}
	os.WriteFile(l.Path+".tmp", nil, 0o644) // a leftover from an interrupted Forget
	if err := l.Forget([]string{"host-1a2b3c4d"}); err != nil {
		t.Fatal(err)
	}
	if m := mode(); m != 0o600 {
		t.Errorf("after Forget: %v", m)
	}
}
