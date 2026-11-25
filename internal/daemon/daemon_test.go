package daemon

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"griz/internal/app"
	"griz/internal/config"
)

// testState is the current test's state directory: short, since socket
// paths are capped.
var testState string

func shortTmp(t *testing.T) string {
	d, err := os.MkdirTemp("/tmp", "gzt")
	if err != nil {
		t.Fatal(err)
	}
	testState = d
	t.Cleanup(func() { os.RemoveAll(d) })
	return d
}

func testServer(t *testing.T) *Server {
	cfg := config.Defaults()
	cfg.DataDir, cfg.StateDir = filepath.Join(t.TempDir(), "d"), shortTmp(t)
	cfg.Scan.Interval = "0"
	a, err := app.Open(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	a.Stderr = io.Discard
	s := &Server{A: a, Version: "test", StateDir: cfg.StateDir, Check: CheckClient,
		RunCLI: func(args []string, out io.Writer) error {
			io.WriteString(out, "ran "+strings.Join(args, " "))
			return nil
		}}
	if err := s.Listen(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Serve(ctx); close(done) }()
	t.Cleanup(func() { cancel(); <-done })
	return s
}

// raw sends lines on a new connection and returns everything the server
// says until it closes or goes quiet.
func raw(t *testing.T, lines ...string) string {
	c, err := net.Dial("unix", socketPath(testState))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	for _, l := range lines {
		c.Write([]byte(l))
	}
	c.SetReadDeadline(time.Now().Add(time.Second))
	b, _ := io.ReadAll(c)
	return string(b)
}

func TestServerRejectsMalformedInput(t *testing.T) {
	testServer(t)
	bin, _ := executable()
	id, _ := binaryID(bin)
	hello := func(extra string) string {
		return `{"griz":"hello","proto":1,"mode":"cli","binary":"` + id + `"` + extra + "}\n"
	}
	for name, c := range map[string]struct {
		lines []string
		want  string
	}{
		"not json":       {[]string{"hello there\n"}, "bad control message"},
		"unknown field":  {[]string{hello(`,"evil":1`)}, "unknown field"},
		"unknown kind":   {[]string{`{"griz":"exec","args":["rm"]}` + "\n"}, "unknown kind"},
		"wrong proto":    {[]string{`{"griz":"hello","proto":99,"mode":"cli"}` + "\n"}, "protocol 99"},
		"wrong mode":     {[]string{`{"griz":"hello","proto":1,"mode":"shell"}` + "\n"}, `unknown mode`},
		"trailing data":  {[]string{`{"griz":"hello","proto":1,"mode":"cli"} {"x":1}` + "\n"}, "trailing data"},
		"cli not cli":    {[]string{hello(""), `{"griz":"out","text":"x"}` + "\n"}, "expected a cli command"},
		"cli no args":    {[]string{hello(""), `{"griz":"cli"}` + "\n"}, "expected a cli command"},
		"cli bad json":   {[]string{hello(""), "{\n"}, "bad control message"},
		"too long":       {[]string{strings.Repeat("x", maxControl+10) + "\n"}, ""},
		"no newline eof": {[]string{`{"griz":"hello"`}, ""},
	} {
		got := raw(t, c.lines...)
		if c.want != "" && !strings.Contains(got, c.want) {
			t.Errorf("%s: want %q in reply, got %q", name, c.want, got)
		}
		if strings.Contains(got, `"griz":"hello"`) && c.want != "expected a cli command" && c.want != "bad control message" {
			t.Errorf("%s: server should not accept: %q", name, got)
		}
	}
	// Still serving.
	got := raw(t, hello(""), `{"griz":"cli","args":["gc"]}`+"\n")
	if !strings.Contains(got, `"text":"ran gc"`) || !strings.Contains(got, `{"griz":"done"}`) {
		t.Errorf("valid command after bad ones: %q", got)
	}
}

func TestServerSurvivesMalformedMCP(t *testing.T) {
	testServer(t)
	bin, _ := executable()
	id, _ := binaryID(bin)
	got := raw(t, `{"griz":"hello","proto":1,"mode":"mcp","binary":"`+id+`"}`+"\n",
		"not json at all\n", `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"store","arguments":{"id":"../../etc/passwd","body":"x"}}}`+"\n",
		`{"jsonrpc":"2.0","id":2,"method":"no/such"}`+"\n")
	if strings.Contains(got, "ok ../") {
		t.Errorf("must not store outside the repo: %q", got)
	}
	// A normal session still works.
	got = raw(t, `{"griz":"hello","proto":1,"mode":"cli","binary":"`+id+`"}`+"\n", `{"griz":"cli","args":["status"]}`+"\n")
	if !strings.Contains(got, `"griz":"done"`) {
		t.Errorf("server stopped answering: %q", got)
	}
}

func TestStaleServerSaysSo(t *testing.T) {
	testServer(t)
	got := raw(t, `{"griz":"hello","proto":1,"mode":"cli","binary":"0000000000000000"}`+"\n")
	if !strings.Contains(got, `"griz":"stale"`) {
		t.Errorf("a different binary on disk means this server is stale: %q", got)
	}
}

func TestSocketPermissionsChecked(t *testing.T) {
	st := shortTmp(t)
	os.MkdirAll(socketDir(st), 0o755)
	os.Chmod(socketDir(st), 0o755)
	if err := ensureDir(st); err == nil || !strings.Contains(err.Error(), "REFUSING") {
		t.Errorf("a socket dir open to others must be refused: %v", err)
	}
}

// fakeServer answers the handshake and initialize, then drops the
// connection when it sees a tools/call.
func fakeServer(t *testing.T) {
	st := shortTmp(t)
	ensureDir(st)
	ln, err := net.Listen("unix", socketPath(st))
	if err != nil {
		t.Fatal(err)
	}
	os.Chmod(socketPath(st), 0o600)
	t.Cleanup(func() { ln.Close() })
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				br := bufio.NewReader(c)
				readLine(br, maxControl)
				writeMsg(c, msg{Griz: "hello", Proto: proto, PID: 999999, Started: time.Now().Format(time.RFC3339Nano)})
				for {
					line, err := readLine(br, 1<<20)
					if err != nil {
						return
					}
					var m struct {
						ID     json.RawMessage `json:"id"`
						Method string          `json:"method"`
					}
					json.Unmarshal(line, &m)
					switch m.Method {
					case "initialize":
						c.Write([]byte(`{"jsonrpc":"2.0","id":` + string(m.ID) + `,"result":{}}` + "\n"))
					case "tools/call":
						return // crash
					}
				}
			}(c)
		}
	}()
}

type syncBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuf) Write(p []byte) (int, error) { s.mu.Lock(); defer s.mu.Unlock(); return s.b.Write(p) }
func (s *syncBuf) String() string              { s.mu.Lock(); defer s.mu.Unlock(); return s.b.String() }

func TestRelayStoreOutcomeUnknown(t *testing.T) {
	fakeServer(t)
	pr, pw := io.Pipe()
	out := &syncBuf{}
	r := &Relay{Client: Client{StateDir: testState, Check: CheckServer}, In: pr, Out: out}
	go r.Run()
	pw.Write([]byte(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}` + "\n"))
	pw.Write([]byte(`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"store","arguments":{"id":"a/b","body":"x"}}}` + "\n"))
	deadline := time.Now().Add(3 * time.Second)
	for !strings.Contains(out.String(), `"id":2`) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	got := out.String()
	for _, want := range []string{"GRIZ SERVER STOPPED", "pid 999999", "may or may not have been saved; check with get", `"isError":true`} {
		if !strings.Contains(got, want) {
			t.Errorf("want %q in %q", want, got)
		}
	}
	pw.Close()
}

// Sessions on another griz build are summed up in one line, not flagged each.
func TestStatusNotesOlderRelaysOnce(t *testing.T) {
	cfg := config.Defaults()
	cfg.DataDir, cfg.StateDir, cfg.Scan.Interval = filepath.Join(t.TempDir(), "d"), t.TempDir(), "0"
	a, err := app.Open(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{A: a, Version: "test", started: time.Now(), conns: map[*conn]bool{}} // not serving: Status alone
	for _, v := range []string{"test", "old", "older"} {
		s.A.NewSession().Client = "relay pid 1, griz " + v
	}
	out := strings.Join(s.Status(t.TempDir()), "\n")
	if strings.Count(out, "different griz") != 1 || !strings.Contains(out, "2 sessions run a different griz") {
		t.Errorf("want one line for the 2 older relays:\n%s", out)
	}
}
