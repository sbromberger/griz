package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// These tests build griz and run real relays (griz mcp), servers and CLI
// commands against private directories.

var grizBin string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "grizbin")
	if err != nil {
		panic(err)
	}
	grizBin = filepath.Join(dir, "griz")
	if out, err := exec.Command("go", "build", "-o", grizBin, ".").CombinedOutput(); err != nil {
		panic(string(out))
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

type env struct {
	t    *testing.T
	bin  string
	vars []string
	tmp  string
}

// newEnv makes private config, data, state and socket directories. The
// socket directory is short: macOS caps socket paths at 104 bytes.
func newEnv(t *testing.T, config string) *env {
	root := t.TempDir()
	tmp, err := os.MkdirTemp("/tmp", "gz")
	if err != nil {
		t.Fatal(err)
	}
	// Each test gets its own copy of the binary, so rebuilding one doesn't touch others.
	bin := filepath.Join(root, "griz")
	b, _ := os.ReadFile(grizBin)
	os.WriteFile(bin, b, 0o755)
	os.MkdirAll(filepath.Join(root, "c", "griz"), 0o755)
	os.WriteFile(filepath.Join(root, "c", "griz", "config.toml"), []byte(config), 0o644)
	e := &env{t: t, bin: bin, tmp: tmp, vars: append(os.Environ(),
		"XDG_CONFIG_HOME="+filepath.Join(root, "c"), "XDG_DATA_HOME="+filepath.Join(root, "d"),
		"XDG_STATE_HOME="+filepath.Join(tmp, "s"), "TMPDIR="+tmp)}
	t.Cleanup(func() {
		if pid := e.serverPID(); pid > 0 {
			syscall.Kill(pid, syscall.SIGKILL)
		}
		os.RemoveAll(tmp)
	})
	return e
}

func (e *env) cli(args ...string) (string, error) {
	cmd := exec.Command(e.bin, args...)
	cmd.Env = e.vars
	out, err := cmd.CombinedOutput()
	return string(out), err
}

var pidRE = regexp.MustCompile(`server: pid (\d+)`)

// serverPID asks the running server for its pid (0 if none).
func (e *env) serverPID() int {
	out, _ := e.cli("status")
	m := pidRE.FindStringSubmatch(out)
	if m == nil {
		return 0
	}
	pid, _ := strconv.Atoi(m[1])
	return pid
}

// session is one Claude session: a griz mcp relay process.
type session struct {
	t   *testing.T
	cmd *exec.Cmd
	in  *json.Encoder
	out *bufio.Reader
	id  int
}

func (e *env) session() *session {
	cmd := exec.Command(e.bin, "mcp")
	cmd.Env = e.vars
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	if err := cmd.Start(); err != nil {
		e.t.Fatal(err)
	}
	s := &session{t: e.t, cmd: cmd, in: json.NewEncoder(stdin), out: bufio.NewReader(stdout)}
	e.t.Cleanup(func() { stdin.Close(); cmd.Wait() })
	s.request("initialize", map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{},
		"clientInfo": map[string]any{"name": "test", "version": "1"}})
	s.in.Encode(map[string]any{"jsonrpc": "2.0", "method": "notifications/initialized"})
	return s
}

type reply struct {
	Result struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		IsError      bool   `json:"isError"`
		Instructions string `json:"instructions"`
	} `json:"result"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func (s *session) request(method string, params any) reply {
	s.id++
	id := s.id
	s.in.Encode(map[string]any{"jsonrpc": "2.0", "id": id, "method": method, "params": params})
	for {
		line, err := s.out.ReadBytes('\n')
		if err != nil {
			s.t.Fatalf("%s: relay output ended: %v", method, err)
		}
		var m struct {
			ID json.RawMessage `json:"id"`
		}
		json.Unmarshal(line, &m)
		if string(m.ID) == strconv.Itoa(id) {
			var r reply
			json.Unmarshal(line, &r)
			return r
		}
	}
}

// tool calls a tool and returns its text and whether it was an error.
func (s *session) tool(name string, args map[string]any) (string, bool) {
	r := s.request("tools/call", map[string]any{"name": name, "arguments": args})
	if r.Error != nil {
		return r.Error.Message, true
	}
	var parts []string
	for _, c := range r.Result.Content {
		parts = append(parts, c.Text)
	}
	return strings.Join(parts, "\n"), r.Result.IsError
}

const noScan = "[scan]\ninterval = \"0\"\n[store]\nnew_ns_guard = false\n"

func TestSharedServer(t *testing.T) {
	e := newEnv(t, noScan)
	a, b := e.session(), e.session()
	if out, bad := a.tool("store", map[string]any{"id": "notes/shared", "body": "walrus facts"}); bad || !strings.HasPrefix(out, "ok") {
		t.Fatalf("store: %q", out)
	}
	if out, _ := b.tool("recall", map[string]any{"q": "walrus"}); !strings.Contains(out, "notes/shared") {
		t.Errorf("the other session should see the store at once: %q", out)
	}
	st, _ := e.cli("status")
	if !strings.Contains(st, "2 clients connected") && !strings.Contains(st, "3 clients connected") {
		t.Errorf("status should count both sessions: %q", st)
	}
	if !strings.Contains(st, "relay pid") {
		t.Errorf("status should show each session's relay: %q", st)
	}
	// CLI writes go through the server.
	if out, err := e.cli("memory", "archive", "notes/shared"); err != nil {
		t.Fatalf("archive: %v %s", err, out)
	}
	if out, _ := a.tool("recall", map[string]any{"q": "walrus"}); !strings.Contains(out, "archived: notes/shared") {
		t.Errorf("archive via CLI not seen: %q", out)
	}
	// Debug mode is per session.
	a.tool("griz", map[string]any{"debug": "on"})
	if out, _ := a.tool("recall", map[string]any{"q": "walrus"}); !strings.Contains(out, "griz debug:") {
		t.Errorf("debug lines missing: %q", out)
	}
	if out, _ := b.tool("recall", map[string]any{"q": "walrus"}); strings.Contains(out, "griz debug:") {
		t.Errorf("debug leaked to the other session: %q", out)
	}
}

func TestCrashIsLoud(t *testing.T) {
	e := newEnv(t, noScan)
	a, b := e.session(), e.session()
	a.tool("store", map[string]any{"id": "notes/x", "body": "narwhal"})
	pid := e.serverPID()
	if pid == 0 {
		t.Fatal("no server pid")
	}
	done := make(chan string)
	go func() { // a call in flight when the server dies
		out, bad := a.tool("griz", map[string]any{"show": "profile", "seconds": 5})
		if !bad {
			out = "NOT AN ERROR: " + out
		}
		done <- out
	}()
	time.Sleep(300 * time.Millisecond)
	syscall.Kill(pid, syscall.SIGKILL)
	out := <-done
	for _, want := range []string{"GRIZ SERVER STOPPED", fmt.Sprintf("pid %d", pid), "Tell the user now.", "can be retried"} {
		if !strings.Contains(out, want) {
			t.Errorf("in-flight call (answered by the relay): want %q in %q", want, out)
		}
	}
	// The new server says what happened on each session's next call, which
	// still works: the relays hold no session state, the server resumes it.
	for name, s := range map[string]*session{"in-flight session": a, "other session": b} {
		out, bad := s.tool("recall", map[string]any{"q": "narwhal"})
		if bad || !strings.Contains(out, "GRIZ SERVER CRASHED") || !strings.Contains(out, "disappeared without a crash report (killed?)") || !strings.Contains(out, "notes/x") {
			t.Errorf("%s's next call: %v %q", name, bad, out)
		}
	}
	// Only once.
	if out, _ := b.tool("recall", map[string]any{"q": "narwhal"}); strings.Contains(out, "CRASHED") {
		t.Errorf("crash reported twice: %q", out)
	}
	// status lists the vanished server among the warnings.
	if st, _ := e.cli("status"); !strings.Contains(st, fmt.Sprintf("server pid %d", pid)) || !strings.Contains(st, "disappeared") {
		t.Errorf("status should list the crash: %q", st)
	}
}

func TestCrashReportFile(t *testing.T) {
	e := newEnv(t, noScan)
	a := e.session()
	pid := e.serverPID()
	syscall.Kill(pid, syscall.SIGQUIT) // Go prints a goroutine dump and exits: a crash with output
	time.Sleep(500 * time.Millisecond)
	out, _ := a.tool("recall", map[string]any{"q": "x"})
	if !strings.Contains(out, "GRIZ SERVER CRASHED") || !strings.Contains(out, "SIGQUIT") {
		t.Errorf("want the crash's first line: %q", out)
	}
	// Listing the crash keeps its report (2026-11-25: one was deleted before
	// anyone read the stacks).
	status, _ := e.cli("status")
	if !strings.Contains(status, "CRASHED") {
		t.Errorf("status should list the crash: %q", status)
	}
	kept, _ := filepath.Glob(filepath.Join(e.tmp, "s", "griz", "crashes", fmt.Sprintf("*-%d.shown.txt", pid)))
	if len(kept) != 1 {
		t.Fatalf("the shown report should be kept: %v", kept)
	}
	if b, _ := os.ReadFile(kept[0]); !strings.Contains(string(b), "goroutine") {
		t.Errorf("the kept report should hold the stacks: %d bytes", len(b))
	}
}

func TestRebuiltServerHandsOver(t *testing.T) {
	e := newEnv(t, noScan)
	a := e.session()
	a.tool("store", map[string]any{"id": "notes/y", "body": "okapi"})
	old := e.serverPID()
	// "Rebuild": a different binary at the same path.
	b, _ := os.ReadFile(e.bin)
	tmp := e.bin + ".new"
	os.WriteFile(tmp, append(b, []byte("rebuilt")...), 0o755)
	os.Rename(tmp, e.bin)
	time.Sleep(2500 * time.Millisecond) // the server checks its binary every second
	out, bad := a.tool("recall", map[string]any{"q": "okapi"})
	if bad || strings.Contains(out, "CRASHED") || !strings.Contains(out, "notes/y") {
		t.Errorf("after a rebuild the session should carry on quietly: %v %q", bad, out)
	}
	if now := e.serverPID(); now == old || now == 0 {
		t.Errorf("expected a new server, old %d now %d", old, now)
	}
	if out, bad := a.tool("store", map[string]any{"id": "notes/z", "body": "after the rebuild"}); bad || !strings.HasPrefix(out, "ok") {
		t.Errorf("the session carries on with the new server: %v %q", bad, out)
	}
}

func TestIdleExit(t *testing.T) {
	e := newEnv(t, noScan+"[server]\nidle_exit = \"1s\"\n")
	if out, err := e.cli("memory", "gc"); err != nil { // starts a server
		t.Fatalf("gc: %v %s", err, out)
	}
	sock := filepath.Join(e.tmp, "s", "griz", "server", "griz.sock")
	if _, err := os.Stat(sock); err != nil {
		t.Fatal("server should be running")
	}
	time.Sleep(3 * time.Second)
	if _, err := os.Stat(sock); err == nil {
		t.Error("server should exit when idle")
	}
	if fi, err := os.Stat(filepath.Join(e.tmp, "s", "griz", "server")); err != nil || fi.Mode().Perm() != 0o700 {
		t.Errorf("socket dir must be 0700: %v %v", fi.Mode(), err)
	}
}

func TestOneServerForConcurrentStarts(t *testing.T) {
	e := newEnv(t, noScan)
	ss := make([]*session, 4)
	done := make(chan int)
	for i := range ss {
		go func(i int) { ss[i] = e.session(); done <- i }(i)
	}
	for range ss {
		<-done
	}
	out, _ := exec.Command("pgrep", "-f", e.bin+" serve").Output()
	if n := len(strings.Fields(string(out))); n != 1 {
		t.Errorf("want one server, found %d: %q", n, out)
	}
}

func TestOtherDataDirRefused(t *testing.T) {
	e := newEnv(t, noScan)
	e.session()
	other := append([]string(nil), e.vars...)
	for i, v := range other {
		if strings.HasPrefix(v, "XDG_DATA_HOME=") {
			other[i] = "XDG_DATA_HOME=" + t.TempDir()
		}
	}
	cmd := exec.Command(e.bin, "memory", "gc")
	cmd.Env = other
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "REFUSING") {
		t.Errorf("a client for another data dir must be refused: %v %s", err, out)
	}
}

// Starting griz by a relative path or through a symlink is the same griz.
func TestRelativeAndSymlinkInvocation(t *testing.T) {
	e := newEnv(t, noScan)
	dir := filepath.Dir(e.bin)
	link := filepath.Join(t.TempDir(), "griz-link")
	if err := os.Symlink(e.bin, link); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ name, dir, path string }{
		{"relative", dir, "./griz"},
		{"symlink", "", link},
		{"absolute", "", e.bin},
	} {
		cmd := exec.Command(c.path, "memory", "gc")
		cmd.Dir, cmd.Env = c.dir, e.vars
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Errorf("%s: %v %s", c.name, err, out)
		}
	}
	// A relay started by a relative path works too.
	cmd := exec.Command("./griz", "mcp")
	cmd.Dir, cmd.Env = dir, e.vars
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	cmd.Start()
	defer func() { stdin.Close(); cmd.Wait() }()
	s := &session{t: t, cmd: cmd, in: json.NewEncoder(stdin), out: bufio.NewReader(stdout)}
	r := s.request("initialize", map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{},
		"clientInfo": map[string]any{"name": "test", "version": "1"}})
	if r.Error != nil {
		t.Fatalf("relative-path relay: %s", r.Error.Message)
	}
}

// A refused client hears the server's reason, not a timeout.
func TestRefusalIsExplained(t *testing.T) {
	e := newEnv(t, noScan)
	e.cli("memory", "gc") // start the server
	// A different executable (a copy) is not "this griz".
	other := filepath.Join(t.TempDir(), "griz")
	b, _ := os.ReadFile(e.bin)
	os.WriteFile(other, b, 0o755)
	start := time.Now()
	cmd := exec.Command(other, "memory", "gc")
	cmd.Env = e.vars
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "REFUSED by the griz server") || !strings.Contains(string(out), "not this griz") {
		t.Errorf("want the server's reason: %v %s", err, out)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("refusal took %s: it must not wait for a timeout", d)
	}
}

// rebuild replaces the test binary with a different one at the same path.
func (e *env) rebuild() {
	b, _ := os.ReadFile(e.bin)
	tmp := e.bin + ".new"
	os.WriteFile(tmp, append(b, []byte("rebuilt")...), 0o755)
	os.Rename(tmp, e.bin)
}

// within runs f, failing the test if it takes longer than d.
func within(t *testing.T, d time.Duration, what string, f func()) {
	done := make(chan struct{})
	go func() { f(); close(done) }()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatalf("%s: still waiting after %s", what, d)
	}
}

// A subscription is answered only when its connection closes, so draining
// must not wait for it (2026-11-25: a rebuild hung for good on one).
func TestRebuildWithOpenSubscription(t *testing.T) {
	e := newEnv(t, noScan)
	a := e.session()
	a.tool("store", map[string]any{"id": "notes/z", "body": "quokka"})
	a.in.Encode(map[string]any{"jsonrpc": "2.0", "id": 9999, "method": "subscriptions/listen",
		"params": map[string]any{"notifications": map[string]any{"toolsListChanged": true}}})
	time.Sleep(300 * time.Millisecond)
	old := e.serverPID()
	e.rebuild()
	time.Sleep(2500 * time.Millisecond) // the server checks its binary every second
	within(t, 10*time.Second, "recall after a rebuild", func() {
		out, bad := a.tool("recall", map[string]any{"q": "quokka"})
		if bad || strings.Contains(out, "CRASHED") || !strings.Contains(out, "notes/z") {
			t.Errorf("after a rebuild the session should carry on quietly: %v %q", bad, out)
		}
	})
	if now := e.serverPID(); now == old || now == 0 {
		t.Errorf("expected a new server, old %d now %d", old, now)
	}
}

// A tool call still running after server.drain_wait is given up on: the old
// server exits, and that call fails loudly instead of being resent.
func TestDrainGivesUpOnALongCall(t *testing.T) {
	e := newEnv(t, noScan+"[server]\ndrain_wait = \"1s\"\n")
	a, b := e.session(), e.session()
	old := e.serverPID()
	slow := make(chan string, 1)
	go func() {
		out, _ := b.tool("griz", map[string]any{"show": "profile", "seconds": 30})
		slow <- out
	}()
	time.Sleep(300 * time.Millisecond)
	e.rebuild()
	within(t, 10*time.Second, "the long call's answer", func() {
		if out := <-slow; !strings.Contains(out, "GRIZ SERVER") {
			t.Errorf("the abandoned call should fail loudly: %q", out)
		}
	})
	within(t, 10*time.Second, "recall on the other session", func() {
		if _, bad := a.tool("recall", map[string]any{"q": "x"}); bad {
			t.Error("the other session should carry on")
		}
	})
	if now := e.serverPID(); now == old || now == 0 {
		t.Errorf("expected a new server, old %d now %d", old, now)
	}
}

// Claude Desktop starts griz without $TMPDIR and Claude Code with it; both
// must reach the one server (the socket lives in the state directory).
func TestOneServerWhateverTMPDIR(t *testing.T) {
	e := newEnv(t, noScan)
	a := e.session()
	a.tool("recall", map[string]any{"q": "x"})
	var vars []string
	for _, v := range e.vars {
		if !strings.HasPrefix(v, "TMPDIR=") {
			vars = append(vars, v)
		}
	}
	e.vars = vars
	b := e.session()
	b.tool("recall", map[string]any{"q": "x"})
	if st, _ := e.cli("status"); !strings.Contains(st, "2 clients connected") && !strings.Contains(st, "3 clients connected") {
		t.Errorf("a relay without TMPDIR should join the same server: %s", st)
	}
	if out, _ := exec.Command("pgrep", "-f", e.bin+" serve").Output(); len(strings.Fields(string(out))) != 1 {
		t.Errorf("want one server, got pids %q", out)
	}
}

// A session that carries on with a new server already has the pinned
// memories, so the new server doesn't send them again.
func TestResumedSessionKeepsPinned(t *testing.T) {
	e := newEnv(t, noScan)
	a := e.session()
	if out, bad := a.tool("store", map[string]any{"id": "notes/pinned", "body": "a pinned fact", "pin": true}); bad || !strings.HasPrefix(out, "ok") {
		t.Fatalf("store: %q", out)
	}
	b := e.session()
	if out, _ := b.tool("recall", map[string]any{"q": "anything"}); !strings.Contains(out, "pinned memories, sent once") {
		t.Fatalf("a new session's first reply carries the pinned memories: %q", out)
	}
	if out, err := e.cli("mcp", "restart"); err != nil {
		t.Fatalf("restart: %v %s", err, out)
	}
	if out, bad := b.tool("recall", map[string]any{"q": "pinned fact"}); bad || strings.Contains(out, "pinned memories, sent once") || !strings.Contains(out, "notes/pinned") {
		t.Errorf("after a restart the session carries on without a second copy: %v %q", bad, out)
	}
}
