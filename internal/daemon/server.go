package daemon

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"griz/internal/app"
	"griz/internal/server"
)

// ErrRunning means another griz server holds the server lock.
var ErrRunning = errors.New("a griz server is already running")

// Server is the one griz server: it owns the App and serves every relay and
// CLI client.
type Server struct {
	A        *app.App
	Version  string
	StateDir string        // holds the socket and lock (see socketDir)
	IdleExit time.Duration // exit after this long with no connections; 0 = never
	// DrainWait is how long stopping waits for calls in progress; 0 = no limit.
	DrainWait time.Duration
	// RunCLI runs a CLI command inside the server, writing its output to out.
	// Calls are serialized.
	RunCLI func(args []string, out io.Writer) error
	// Check, if set, vets each connection before the hello (peer checks).
	Check func(c *net.UnixConn) error

	exe, binID, exeStamp string
	started              time.Time
	ln                   *net.UnixListener
	lockF                *os.File

	mu        sync.Mutex
	conns     map[*conn]bool
	idleSince time.Time
	draining  bool
	drainOnce sync.Once
	stop      chan string // reason
	cliMu     sync.Mutex
}

// conn is one client connection.
type conn struct {
	c       *net.UnixConn
	wmu     sync.Mutex
	pending atomic.Int64 // tool calls and CLI commands read but not yet answered
	callMu  sync.Mutex
	calls   map[any]bool  // ids of tool calls read but not yet answered
	drain   chan struct{} // closed when the server starts draining
	closed  chan struct{}
	once    sync.Once
}

func (c *conn) Write(p []byte) (int, error) {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	return c.c.Write(p)
}

func (c *conn) Close() error {
	c.once.Do(func() { close(c.closed) })
	return c.c.Close()
}

// Listen takes the server lock and binds the socket. It returns ErrRunning
// if another server holds the lock. The lock is held until Serve returns.
func (s *Server) Listen() error {
	if err := ensureDir(s.StateDir); err != nil {
		return err
	}
	sock := socketPath(s.StateDir)
	lf, err := os.OpenFile(lockPath(s.StateDir), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	if err := syscall.Flock(int(lf.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		lf.Close()
		return ErrRunning
	}
	s.lockF = lf
	// Holding the lock: any socket file left is stale.
	os.Remove(sock)
	old := syscall.Umask(0o077)
	ln, err := net.ListenUnix("unix", &net.UnixAddr{Name: sock, Net: "unix"})
	syscall.Umask(old)
	if err != nil {
		lf.Close()
		return err
	}
	os.Chmod(sock, 0o600)
	s.ln = ln
	return nil
}

// Serve serves connections until idle, drained (the binary changed), or ctx
// ends; then it releases the lock.
func (s *Server) Serve(ctx context.Context) error {
	defer s.lockF.Close()
	s.started = time.Now()
	s.idleSince = s.started
	s.conns = map[*conn]bool{}
	s.stop = make(chan string, 1)
	var err error
	if s.exe, err = executable(); err == nil {
		s.exeStamp = fileStamp(s.exe)
		s.binID, _ = binaryID(s.exe)
	}
	go s.accept()

	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	reason := ""
	for reason == "" {
		select {
		case <-ctx.Done():
			reason = "stopped"
		case reason = <-s.stop:
		case <-tick.C:
			if s.binaryChanged() {
				reason = "griz was rebuilt"
			} else if s.idle() {
				reason = "idle"
			}
		}
	}
	s.drainAll(reason)
	return nil
}

// Status describes the server for griz status: pid, uptime, connected
// sessions, and crashes since they were last shown.
func (s *Server) Status(stateDir string) []string {
	s.mu.Lock()
	n := len(s.conns)
	s.mu.Unlock()
	lines := []string{fmt.Sprintf("server: pid %d, griz %s, up %s, %d %s connected (exits after %s with none)",
		os.Getpid(), s.Version, time.Since(s.started).Round(time.Second), n, plural(n, "client", "clients"), s.IdleExit)}
	older := 0
	for _, ss := range s.A.Sessions() {
		lines = append(lines, fmt.Sprintf("  session %d: %s, connected %s ago, debug %s", ss.ID, ss.Client, time.Since(ss.Since).Round(time.Second), onOff(ss.Debugging())))
		if !strings.HasSuffix(ss.Client, ", griz "+s.Version) {
			older++
		}
	}
	if older > 0 {
		lines = append(lines, fmt.Sprintf("%d %s a different griz from the server's; restarting their Claude apps updates them",
			older, plural(older, "session runs", "sessions run")))
	}
	if c := CrashReport(stateDir); c != "" {
		lines = append(lines, c)
	}
	return lines
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

func (s *Server) idle() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.IdleExit > 0 && len(s.conns) == 0 && time.Since(s.idleSince) >= s.IdleExit
}

// binaryChanged reports whether the griz executable on disk is no longer
// the one this server runs.
func (s *Server) binaryChanged() bool {
	if s.exe == "" {
		return false
	}
	st := fileStamp(s.exe)
	if st == s.exeStamp {
		return false
	}
	s.exeStamp = st
	id, err := binaryID(s.exe)
	return err == nil && id != s.binID
}

func (s *Server) accept() {
	for {
		c, err := s.ln.AcceptUnix()
		if err != nil {
			return // listener closed
		}
		go s.handle(c)
	}
}

func (s *Server) handle(uc *net.UnixConn) {
	c := &conn{c: uc, drain: make(chan struct{}), closed: make(chan struct{})}
	defer c.Close()
	if s.Check != nil {
		if err := s.Check(uc); err != nil {
			// Say why before hanging up, so the client can report it.
			s.A.Info("refused a connection: %v", err)
			writeMsg(c, msg{Griz: "done", Error: "REFUSED by the griz server (pid " + fmt.Sprint(os.Getpid()) + "): " + err.Error()})
			return
		}
	}
	br := bufio.NewReaderSize(uc, 64<<10)
	uc.SetReadDeadline(time.Now().Add(10 * time.Second))
	line, err := readLine(br, maxControl)
	if err != nil {
		return
	}
	uc.SetReadDeadline(time.Time{})
	hello, err := parseMsg(line)
	switch {
	case err != nil:
	case hello.Griz != "hello":
		err = errors.New("expected hello")
	case hello.Proto != proto:
		err = fmt.Errorf("protocol %d, server speaks %d; restart the Claude app to use the new griz", hello.Proto, proto)
	case hello.Mode != "mcp" && hello.Mode != "cli":
		err = fmt.Errorf("unknown mode %q", hello.Mode)
	case hello.Data != "" && hello.Data != s.A.Cfg.DataDir:
		err = fmt.Errorf("REFUSING: the running griz server serves %s, not %s (different XDG settings?); stop it or use the same settings", s.A.Cfg.DataDir, hello.Data)
	}
	if err != nil {
		writeMsg(c, msg{Griz: "done", Error: "griz server: " + err.Error()})
		return
	}
	if hello.Binary != "" && s.binID != "" && hello.Binary != s.binID {
		// The client hashed the executable on disk just now: this server is stale.
		writeMsg(c, msg{Griz: "stale", Reason: "griz was rebuilt"})
		s.requestStop("griz was rebuilt")
		return
	}
	if !s.add(c) {
		writeMsg(c, msg{Griz: "stale", Reason: "server is shutting down"})
		return
	}
	defer s.remove(c)
	if err := writeMsg(c, msg{Griz: "hello", Proto: proto, Version: s.Version, Binary: s.binID, PID: os.Getpid(),
		Started: s.started.Format(time.RFC3339Nano)}); err != nil {
		return
	}
	switch hello.Mode {
	case "mcp":
		s.serveMCP(c, br, fmt.Sprintf("relay pid %d, griz %s", hello.PID, hello.Version))
	case "cli":
		s.serveCLI(c, br)
	}
}

func (s *Server) requestStop(reason string) {
	select {
	case s.stop <- reason:
	default:
	}
}

func (s *Server) add(c *conn) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.draining {
		return false
	}
	s.conns[c] = true
	return true
}

func (s *Server) remove(c *conn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.conns, c)
	if len(s.conns) == 0 {
		s.idleSince = time.Now()
	}
}

// drainAll stops taking connections and requests, lets every tool call and
// CLI command already read finish and its answer be written, then says bye to
// each client (they reconnect to a new server, resending what wasn't
// answered) and closes. A call still running after DrainWait is given up on:
// its connection closes without a bye, so the relay reports it as failed
// rather than resending work that may have half happened.
func (s *Server) drainAll(reason string) {
	s.mu.Lock()
	s.draining = true
	conns := make([]*conn, 0, len(s.conns))
	for c := range s.conns {
		conns = append(conns, c)
	}
	s.mu.Unlock()
	s.ln.Close() // removes the socket; the lock is held until Run returns
	for _, c := range conns {
		close(c.drain)
	}
	var deadline <-chan time.Time // nil: no limit
	if s.DrainWait > 0 {
		deadline = time.After(s.DrainWait)
	}
	answered := func(c *conn) bool {
		for c.pending.Load() > 0 {
			select {
			case <-c.closed:
				return true
			case <-deadline:
				return false
			case <-time.After(5 * time.Millisecond):
			}
		}
		return true
	}
	for _, c := range conns {
		if answered(c) {
			writeMsg(c, msg{Griz: "bye", Reason: reason})
		} else {
			fmt.Fprintf(s.A.Stderr, "griz: stopping (%s): %d call(s) still running after server.drain_wait %s; closing without waiting\n", reason, c.pending.Load(), s.DrainWait)
		}
		c.Close()
	}
	s.A.WaitEmbeds()
}

// serveMCP runs one Claude session.
func (s *Server) serveMCP(c *conn, br *bufio.Reader, client string) {
	// A session that began with an earlier server carries on without a new
	// initialize: this server makes one up and drops its answer.
	first, err := readLine(br, 64<<20)
	if err != nil {
		return
	}
	var m struct {
		Method string `json:"method"`
	}
	json.Unmarshal(first, &m)
	resumed := m.Method != "initialize"
	in := io.MultiReader(bytes.NewReader(append(first, '\n')), br)
	if resumed {
		in = io.MultiReader(strings.NewReader(resumeInit), in)
	}
	sess := s.A.NewSession()
	sess.Client = client
	defer s.A.EndSession(sess)
	t := &countingTransport{inner: &mcp.IOTransport{Reader: readCloser{in, c}, Writer: c}, c: c}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-c.closed
		cancel()
	}()
	if err := server.Serve(ctx, s.A, sess, t, nil, resumed); err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, context.Canceled) {
		s.A.Info("session %d: %v", sess.ID, err)
	}
}

type readCloser struct {
	io.Reader
	c io.Closer
}

func (r readCloser) Close() error { return r.c.Close() }

// countingTransport counts requests read and answered, so draining can wait
// for the answers. Once draining, requests are not read at all: the client
// resends them to the next server.
type countingTransport struct {
	inner mcp.Transport
	c     *conn
}

func (t *countingTransport) Connect(ctx context.Context) (mcp.Connection, error) {
	inner, err := t.inner.Connect(ctx)
	if err != nil {
		return nil, err
	}
	return &countingConn{Connection: inner, c: t.c}, nil
}

type countingConn struct {
	mcp.Connection
	c *conn
}

func (cc *countingConn) Read(ctx context.Context) (jsonrpc.Message, error) {
	select {
	case <-cc.c.drain:
		<-cc.c.closed // hold here; the relay answers anything unread with "retry"
		return nil, io.EOF
	default:
	}
	msg, err := cc.Connection.Read(ctx)
	if err != nil {
		return msg, err
	}
	if req, ok := msg.(*jsonrpc.Request); ok && req.ID.IsValid() {
		select {
		case <-cc.c.drain:
			<-cc.c.closed // read during the drain: never handled; the relay answers it with "retry"
			return nil, io.EOF
		default:
		}
		// Only tool calls count: they do the work a drain must let finish.
		// Other requests are quick or, like subscriptions/listen, answered
		// only when the connection closes.
		if req.Method == "tools/call" {
			cc.c.callMu.Lock()
			if cc.c.calls == nil {
				cc.c.calls = map[any]bool{}
			}
			cc.c.calls[req.ID.Raw()] = true
			cc.c.callMu.Unlock()
			cc.c.pending.Add(1)
		}
	}
	return msg, nil
}

func (cc *countingConn) Write(ctx context.Context, msg jsonrpc.Message) error {
	if resp, ok := msg.(*jsonrpc.Response); ok && resp.ID.Raw() == resumeID {
		return nil // the answer to the made-up initialize
	}
	err := cc.Connection.Write(ctx, msg)
	if resp, ok := msg.(*jsonrpc.Response); ok {
		cc.c.callMu.Lock()
		counted := cc.c.calls[resp.ID.Raw()]
		delete(cc.c.calls, resp.ID.Raw())
		cc.c.callMu.Unlock()
		if counted {
			cc.c.pending.Add(-1)
		}
	}
	return err
}

// serveCLI runs one CLI command and streams its output.
func (s *Server) serveCLI(c *conn, br *bufio.Reader) {
	line, err := readLine(br, maxControl)
	if err != nil {
		return
	}
	m, err := parseMsg(line)
	if err == nil && (m.Griz != "cli" || len(m.Args) == 0) {
		err = errors.New("expected a cli command")
	}
	if err != nil {
		writeMsg(c, msg{Griz: "done", Error: "griz server: " + err.Error()})
		return
	}
	if m.Args[0] == stopCmd {
		writeMsg(c, msg{Griz: "done"})
		s.requestStop("stopped by griz mcp " + strings.Join(m.Args[1:], " "))
		return
	}
	s.cliMu.Lock()
	defer s.cliMu.Unlock()
	c.pending.Add(1)
	defer c.pending.Add(-1)
	if err := s.RunCLI(m.Args, msgWriter{c}); err != nil {
		writeMsg(c, msg{Griz: "done", Error: err.Error()})
		return
	}
	writeMsg(c, msg{Griz: "done"})
}

// msgWriter sends written text as "out" lines.
type msgWriter struct{ w io.Writer }

func (m msgWriter) Write(p []byte) (int, error) {
	if err := writeMsg(m.w, msg{Griz: "out", Text: string(p)}); err != nil {
		return 0, err
	}
	return len(p), nil
}

// resumeInit is the initialize a resumed session doesn't send again.
const resumeID = "griz-resume"

const resumeInit = `{"jsonrpc":"2.0","id":"` + resumeID + `","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"griz relay (resumed session)","version":"1"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
`
