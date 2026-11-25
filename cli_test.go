package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
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
	if out := must("store", "notes/walrus", "--body", "walruses eat clams", "--new-ns"); !strings.HasPrefix(out, "ok notes/walrus [") {
		t.Fatalf("store: %q", out)
	}
	m := labelRE.FindStringSubmatch(must("recall", "walrus clams"))
	if m == nil {
		t.Fatal("recall should show the label")
	}
	if out := must("get", m[2][:1]); !strings.Contains(out, "walruses eat clams") {
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
	if out := must("store", "walrus", "--append", "they also eat snails"); !strings.HasPrefix(out, "ok") {
		t.Errorf("append: %q", out)
	}
	must("debug", "on")
	if out := must("recall", "walrus"); !strings.Contains(out, "griz debug: griz recall took") {
		t.Errorf("debug on should time commands: %q", out)
	}
	must("debug", "off")
	if out := must("recall", "walrus"); strings.Contains(out, "griz debug") {
		t.Errorf("debug off: %q", out)
	}
	if out := must("version"); !strings.Contains(out, "server: pid") || strings.Count(out, "griz "+strings.Fields(out)[1]) != 2 {
		t.Errorf("version should show the server: %q", out)
	}
	pid := e.serverPID()
	if out := must("mcp", "restart"); !strings.Contains(out, "server: pid") || e.serverPID() == pid {
		t.Errorf("restart should bring up a new server: %q", out)
	}
	if out := must("mcp", "stop"); !strings.Contains(out, "stopped griz server") || e.serverPID() != 0 {
		t.Errorf("stop: %q", out)
	}
	if out := must("mcp", "status"); !strings.Contains(out, "no griz server running") {
		t.Errorf("status with none: %q", out)
	}
	if out := must("get", "walrus"); !strings.Contains(out, "snails") {
		t.Errorf("get without a server runs here: %q", out)
	}
	must("delete", "walrus")
	if out, err := e.cli("get", "walrus"); err == nil && !strings.Contains(out, "not found") {
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
}
