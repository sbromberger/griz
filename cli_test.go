package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"
)

var labelRE = regexp.MustCompile(`notes/walrus \[([a-z0-9-]+)-([0-9a-f]{8})\]`)

// The tool commands, lifetimes, debug and server control, end to end.
func TestCLI(t *testing.T) {
	e := newEnv(t, noScan)
	must := func(args ...string) string {
		t.Helper()
		out, err := e.cli(args...)
		if err != nil {
			t.Fatalf("griz %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return out
	}
	if out := must("memory", "store", "notes/walrus", "--body", "walruses eat clams", "--new-ns"); !strings.HasPrefix(out, "ok notes/walrus [") {
		t.Fatalf("store: %q", out)
	}
	m := labelRE.FindStringSubmatch(must("memory", "recall", "walrus clams"))
	if m == nil {
		t.Fatal("recall should show the label")
	}
	if out := must("memory", "get", m[2][:1]); !strings.Contains(out, "walruses eat clams") {
		t.Errorf("get by one hex digit: %q", out)
	}
	if out := must("memory", "term", "infinite", "walrus"); !strings.HasPrefix(out, "ok") {
		t.Errorf("term: %q", out)
	}
	var data string
	for _, v := range e.vars {
		if d, ok := strings.CutPrefix(v, "XDG_DATA_HOME="); ok {
			data = d
		}
	}
	files, _ := filepath.Glob(filepath.Join(data, "griz", "notes", "walrus.*.md"))
	if len(files) != 1 {
		t.Fatalf("memory file: %v", files)
	}
	if b, _ := os.ReadFile(files[0]); !strings.Contains(string(b), "term: infinite") {
		t.Errorf("term not set:\n%s", b)
	}
	if out := must("memory", "expire", "3w", "walrus"); !strings.HasPrefix(out, "ok") {
		t.Errorf("expire: %q", out)
	}
	if b, _ := os.ReadFile(files[0]); !strings.Contains(string(b), "expires: ") {
		t.Errorf("expire not set:\n%s", b)
	}
	must("memory", "expire", "never", "walrus")
	if out, err := e.cli("memory", "expire", "walrus"); err == nil {
		t.Errorf("expire without a date: %q", out)
	}
	if out := must("memory", "renew", "walrus"); !strings.HasPrefix(out, "ok") {
		t.Errorf("renew: %q", out)
	}
	if out := must("memory", "move", "walrus", "sea/"); out != "moved "+m[1]+"-"+m[2]+": notes/walrus → sea/walrus\n" {
		t.Errorf("move: %q", out)
	}
	if out := must("memory", "store", "walrus", "--append", "they also eat snails"); !strings.HasPrefix(out, "ok") {
		t.Errorf("append: %q", out)
	}
	must("server", "debug", "on")
	if out := must("memory", "recall", "walrus"); !strings.Contains(out, "griz debug: griz memory recall took") {
		t.Errorf("debug on should time commands: %q", out)
	}
	must("server", "debug", "off")
	if out := must("memory", "recall", "walrus"); strings.Contains(out, "griz debug") {
		t.Errorf("debug off: %q", out)
	}
	if out := must("version"); !strings.Contains(out, "server griz "+strings.Fields(out)[1]+" (pid ") || strings.Contains(out, "relay") {
		t.Errorf("version should show the server: %q", out)
	}
	if out := must("server", "profile", "--seconds", "1"); !strings.Contains(out, "sampled the server") {
		t.Errorf("profile: %q", out)
	}
	if out, err := e.cli("server", "profile", "--seconds", "1", "--reembed"); err == nil {
		t.Errorf("profile with both: %q", out)
	}
	pid := e.serverPID()
	if out := must("server", "restart"); !strings.Contains(out, "server  pid") || e.serverPID() == pid {
		t.Errorf("restart should bring up a new server: %q", out)
	}
	if out := must("server", "stop"); !strings.Contains(out, "stopped griz server") || e.serverPID() != 0 {
		t.Errorf("stop: %q", out)
	}
	if out := must("server", "status"); !strings.Contains(out, "server  not running") {
		t.Errorf("status with none: %q", out)
	}
	if out := must("memory", "get", "walrus"); !strings.Contains(out, "snails") {
		t.Errorf("get without a server runs here: %q", out)
	}
	must("memory", "delete", "walrus")
	if out, err := e.cli("memory", "get", "walrus"); err == nil && !strings.Contains(out, "not found") {
		t.Errorf("after delete: %q", out)
	}
	for _, args := range [][]string{{"help"}, {"help", "memory", "term"}, {"memory", "term", "--help"}} {
		if out := must(args...); !strings.Contains(out, "usage: griz") {
			t.Errorf("griz %v: %q", args, out)
		}
	}
	if out, err := e.cli("memory", "nosuch"); err == nil || !strings.Contains(out, "griz memory term") {
		t.Errorf("unknown subcommand should list the group: %q", out)
	}
	// A group alone prints its help; the old names are gone.
	for _, g := range []string{"memory", "server", "relay", "config"} {
		if out := must(g); !strings.HasPrefix(out, "griz "+g+": ") {
			t.Errorf("griz %s alone: %q", g, out)
		}
	}
	for _, old := range [][]string{{"recall", "x"}, {"get", "x"}, {"store", "x"}, {"delete", "x"}, {"reindex"}, {"serve"}, {"debug", "on"}, {"mcp", "status"}} {
		if out, err := e.cli(old...); err == nil || !strings.Contains(out, "unknown command") {
			t.Errorf("griz %v should be unknown: %q", old, out)
		}
	}

	// Several ids at once: pin, unpin, delete. A bad key changes nothing.
	for _, p := range []string{"notes/a", "notes/b"} {
		must("memory", "store", p, "--body", "text of "+p, "--force")
	}
	if out, err := e.cli("memory", "delete", "notes/a", "notes/nosuch"); err == nil {
		t.Errorf("delete with a bad key: %q", out)
	}
	if out := must("memory", "pin", "notes/a", "notes/b"); strings.Count(out, "ok") != 2 {
		t.Errorf("pin: %q", out)
	}
	if out := must("memory", "recall", "--pinned"); !strings.Contains(out, "notes/a") || !strings.Contains(out, "notes/b") {
		t.Errorf("pinned after pin: %q", out)
	}
	must("memory", "unpin", "notes/a", "notes/b")
	if out := must("memory", "recall", "--pinned"); strings.Contains(out, "notes/a") {
		t.Errorf("pinned after unpin: %q", out)
	}
	must("memory", "delete", "notes/a", "notes/b")
	if out, err := e.cli("memory", "get", "notes/b"); err == nil && !strings.Contains(out, "not found") {
		t.Errorf("after deleting two: %q", out)
	}
}

// relay status lists every relay of this user (so it may show others than
// this test's); relay kill signals only pids it lists.
func TestRelays(t *testing.T) {
	e := newEnv(t, noScan)
	a, b := e.session(), e.session()
	a.tool("recall", map[string]any{"q": "x"})
	b.tool("recall", map[string]any{"q": "x"})
	out, err := e.cli("relay", "status")
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	for _, s := range []*session{a, b} {
		if !regexp.MustCompile(`relay pid ` + strconv.Itoa(s.cmd.Process.Pid) + `, from .*, session \d+`).MatchString(out) {
			t.Errorf("relay %d missing or not connected:\n%s", s.cmd.Process.Pid, out)
		}
	}
	if out, err := e.cli("relay", "kill", strconv.Itoa(os.Getpid())); err == nil || !strings.Contains(out, "not a griz relay") {
		t.Errorf("killing a non-relay: %q", out)
	}
	if out, err := e.cli("relay", "kill"); err == nil {
		t.Errorf("kill with nothing: %q", out)
	}
	if out, err := e.cli("relay", "kill", strconv.Itoa(a.cmd.Process.Pid)); err != nil || !strings.Contains(out, "stopped relay pid") {
		t.Fatalf("kill: %v %q", err, out)
	}
	done := make(chan struct{})
	go func() { _ = a.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("killed relay still running")
	}
	if out, _ := b.tool("recall", map[string]any{"q": "x"}); out == "" {
		t.Error("the other relay should still work")
	}
}
