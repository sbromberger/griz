package daemon

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Client settings shared by the relay and CLI clients.
type Client struct {
	Version      string
	StateDir     string        // for crash reports
	DataDir      string        // the server must serve this data directory
	StartTimeout time.Duration // how long to wait for a new server to listen
	// Check, if set, vets the server end of a new connection (peer checks).
	Check func(c *net.UnixConn) error
}

// server is one connection to a server, after the handshake.
type serverConn struct {
	c       *net.UnixConn
	br      *bufio.Reader
	pid     int
	started time.Time
	version string
}

// dial connects to the server, starting one if none is running, and does
// the handshake. A stale server (griz was rebuilt) is told so; dial waits
// for it to exit and starts the new one.
func (cl *Client) dial(mode string, start bool) (*serverConn, error) {
	if err := ensureDir(cl.StateDir); err != nil {
		return nil, err
	}
	exe, err := executable()
	if err != nil {
		return nil, err
	}
	timeout := cl.StartTimeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	deadline := time.Now().Add(timeout)
	var spawned *exec.Cmd
	var spawnErr chan error
	for {
		sc, err := cl.handshake(exe, mode)
		if err == nil {
			return sc, nil
		}
		var stale errStale
		if !start && !errors.As(err, &stale) {
			return nil, err
		}
		var refused errRefused
		if errors.As(err, &refused) {
			return nil, err
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("no griz server after %s: %v", timeout, err)
		}
		if spawned == nil && start && !errors.As(err, &stale) {
			spawned = exec.Command(exe)
			spawned.Args = []string{ServerName}
			spawned.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
			if err := spawned.Start(); err != nil {
				return nil, fmt.Errorf("starting griz server: %v", err)
			}
			spawnErr = make(chan error, 1)
			go func(c *exec.Cmd) { spawnErr <- c.Wait() }(spawned)
		}
		if spawnErr != nil {
			select {
			case err := <-spawnErr:
				// Exit 0: another server won the race; keep connecting. Otherwise it failed.
				if err != nil {
					return nil, fmt.Errorf("griz server failed to start (%v); run `griz serve` in a terminal to see why", err)
				}
				spawned, spawnErr = nil, nil
			default:
			}
		}
		time.Sleep(20 * time.Millisecond)
		if errors.As(err, &stale) {
			spawned, spawnErr = nil, nil // the old one is exiting; start anew once it's gone
		}
	}
}

type errStale struct{ reason string }

func (e errStale) Error() string { return "server is stale: " + e.reason }

// errRefused is a refusal that retrying won't fix.
type errRefused struct{ msg string }

func (e errRefused) Error() string { return e.msg }

func (cl *Client) handshake(exe, mode string) (*serverConn, error) {
	sock := socketPath(cl.StateDir)
	if _, err := os.Lstat(sock); err == nil {
		if err := checkOwned(sock, false); err != nil {
			return nil, errRefused{err.Error()}
		}
	}
	c, err := net.DialUnix("unix", nil, &net.UnixAddr{Name: sock, Net: "unix"})
	if err != nil {
		return nil, err
	}
	br := bufio.NewReaderSize(c, 64<<10)
	if cl.Check != nil {
		if err := cl.Check(c); err != nil {
			// The server may have refused us and hung up already: show its reason.
			if why := serverReason(c, br); why != "" {
				return nil, errRefused{why}
			}
			c.Close()
			return nil, errRefused{err.Error()}
		}
	}
	id, _ := binaryID(exe)
	if err := writeMsg(c, msg{Griz: "hello", Proto: proto, Version: cl.Version, Binary: id, Mode: mode, Data: cl.DataDir, PID: os.Getpid()}); err != nil {
		if why := serverReason(c, br); why != "" {
			return nil, errRefused{why}
		}
		c.Close()
		return nil, err
	}
	c.SetReadDeadline(time.Now().Add(10 * time.Second))
	line, err := readLine(br, maxControl)
	c.SetReadDeadline(time.Time{})
	if err != nil {
		c.Close()
		return nil, err
	}
	m, err := parseMsg(line)
	if err != nil {
		c.Close()
		return nil, errRefused{"griz server sent " + err.Error()}
	}
	switch m.Griz {
	case "hello":
	case "stale":
		c.Close()
		return nil, errStale{m.Reason}
	case "done":
		c.Close()
		return nil, errRefused{m.Error}
	default:
		c.Close()
		return nil, errRefused{"griz server sent " + m.Griz + " instead of hello"}
	}
	if m.Proto != proto {
		c.Close()
		return nil, errRefused{fmt.Sprintf("griz server speaks protocol %d, this griz %d; restart the Claude app", m.Proto, proto)}
	}
	started, _ := time.Parse(time.RFC3339Nano, m.Started)
	return &serverConn{c: c, br: br, pid: m.PID, started: started, version: m.Version}, nil
}

// serverReason reads a refusal the server sent before hanging up ("" if
// none), and closes c.
func serverReason(c *net.UnixConn, br *bufio.Reader) string {
	defer c.Close()
	c.SetReadDeadline(time.Now().Add(time.Second))
	line, err := readLine(br, maxControl)
	if err != nil {
		return ""
	}
	if m, err := parseMsg(line); err == nil && m.Griz == "done" && m.Error != "" {
		return m.Error
	}
	return ""
}

// ---- CLI client ----

// RunCLI runs a command in the server (starting it if needed) and copies
// its output to out. If start is false and no server is running, it
// returns ErrNoServer.
func (cl *Client) RunCLI(args []string, out io.Writer, start bool) error {
	sc, err := cl.dial("cli", start)
	if err != nil {
		if !start && (errors.Is(err, syscall.ENOENT) || errors.Is(err, syscall.ECONNREFUSED)) {
			return ErrNoServer // nothing listening; refusals and setup errors are reported
		}
		return err
	}
	defer sc.c.Close()
	if err := writeMsg(sc.c, msg{Griz: "cli", Args: args}); err != nil {
		return err
	}
	for {
		line, err := readLine(sc.br, 64<<20)
		if err != nil {
			return fmt.Errorf("GRIZ SERVER CRASHED while running `griz %s` (pid %d, up %s): %s. The command may or may not have finished; check with griz status",
				strings.Join(args, " "), sc.pid, time.Since(sc.started).Round(time.Second), describe(cl.StateDir, sc.pid))
		}
		m, err := parseMsg(line)
		if err != nil {
			return err
		}
		switch m.Griz {
		case "out":
			io.WriteString(out, m.Text)
		case "done":
			if m.Error != "" {
				return errors.New(m.Error)
			}
			return nil
		case "bye":
			return fmt.Errorf("griz server stopped (%s) before running the command; run it again", m.Reason)
		default:
			return fmt.Errorf("griz server sent %q", m.Griz)
		}
	}
}

// stopCmd is the CLI line that asks the server to exit.
const stopCmd = "\x00stop"

// Stop asks the running server to exit (why: "stop" or "restart") and
// waits until its process is gone. It returns the server's pid, or
// ErrNoServer.
func (cl *Client) Stop(why string) (int, error) {
	sc, err := cl.dial("cli", false)
	if err != nil {
		if errors.Is(err, syscall.ENOENT) || errors.Is(err, syscall.ECONNREFUSED) {
			return 0, ErrNoServer
		}
		return 0, err
	}
	defer sc.c.Close()
	pid := sc.pid
	if err := writeMsg(sc.c, msg{Griz: "cli", Args: []string{stopCmd, why}}); err != nil {
		return pid, err
	}
	readLine(sc.br, maxControl) // done
	timeout := cl.StartTimeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	for deadline := time.Now().Add(timeout); alive(pid); time.Sleep(20 * time.Millisecond) {
		if time.Now().After(deadline) {
			return pid, fmt.Errorf("griz server (pid %d) is still running after %s", pid, timeout)
		}
	}
	return pid, nil
}

// SocketPath is where the server for state directory state listens.
func SocketPath(state string) string { return socketPath(state) }

// ErrNoServer means no griz server is running.
var ErrNoServer = errors.New("no griz server running")

// ---- MCP relay ----

// Relay passes MCP messages between stdio (the Claude app) and the server.
// It is a pipe and holds nothing worth keeping: a relay runs as long as its
// Claude app, outliving installs, so anything that changes between griz
// versions belongs in the server. It remembers only which requests are
// waiting for an answer, so that if the server goes away it can answer
// them at once instead of leaving Claude waiting. The next server picks the
// session up without a new initialize (see serveMCP).
type Relay struct {
	Client
	In  io.Reader
	Out io.Writer

	outMu sync.Mutex

	mu      sync.Mutex
	sc      *serverConn
	waiting map[string]call // requests sent and not yet answered, by raw JSON id
}

type call struct {
	id           json.RawMessage
	method, tool string
}

// rpc is the part of a message the relay looks at.
type rpc struct {
	ID     json.RawMessage `json:"id"`
	Method string          `json:"method"`
	Params struct {
		Name string `json:"name"`
	} `json:"params"`
	Griz string `json:"griz"`
}

func (r *Relay) write(b []byte) {
	r.outMu.Lock()
	defer r.outMu.Unlock()
	r.Out.Write(append(b, '\n'))
}

// Run relays until the client closes stdin.
func (r *Relay) Run() error {
	r.waiting = map[string]call{}
	in := bufio.NewReaderSize(r.In, 64<<10)
	for {
		line, err := readLine(in, 64<<20)
		if err != nil {
			r.mu.Lock()
			if r.sc != nil {
				r.sc.c.Close()
			}
			r.mu.Unlock()
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		if len(bytes.TrimSpace(line)) > 0 {
			r.send(line)
		}
	}
}

// send forwards one client message, connecting (and starting a server) if needed.
func (r *Relay) send(line []byte) {
	var m rpc
	json.Unmarshal(line, &m) // a bad message is passed on; the server answers it
	r.mu.Lock()
	defer r.mu.Unlock()
	var c *call
	if len(m.ID) > 0 && string(m.ID) != "null" && m.Method != "" {
		c = &call{id: m.ID, method: m.Method, tool: m.Params.Name}
		r.waiting[string(m.ID)] = *c
	}
	if r.sc == nil {
		sc, err := r.dial("mcp", true)
		if err != nil {
			if c != nil {
				r.failLocked(*c, "GRIZ SERVER UNAVAILABLE: "+err.Error()+". Tell the user now.")
			}
			return
		}
		r.sc = sc
		go r.read(sc)
	}
	if _, err := r.sc.c.Write(append(line, '\n')); err != nil {
		r.lostLocked(r.sc, "")
	}
}

// read passes the server's messages to the client until the connection ends.
func (r *Relay) read(sc *serverConn) {
	bye := ""
	for {
		line, err := readLine(sc.br, 64<<20)
		if err != nil {
			break
		}
		var m rpc
		if json.Unmarshal(line, &m) == nil && m.Griz != "" {
			if cm, err := parseMsg(line); err == nil && cm.Griz == "bye" {
				bye = cm.Reason
				if bye == "" {
					bye = "stopped"
				}
			}
			continue
		}
		if m.Method == "" && len(m.ID) > 0 {
			r.mu.Lock()
			delete(r.waiting, string(m.ID))
			r.mu.Unlock()
		}
		r.write(line)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lostLocked(sc, bye)
}

// lostLocked handles a lost connection: every request still waiting is
// answered now. bye is the server's reason when it shut down on purpose
// (then those requests were never handled), "" if it just went away. The
// next message connects to a new server, which reports any crash itself.
func (r *Relay) lostLocked(sc *serverConn, bye string) {
	if r.sc != sc {
		return // already handled
	}
	r.sc = nil
	sc.c.Close()
	for _, c := range r.waiting {
		msg := fmt.Sprintf("griz restarted (%s) before handling this call; retry it.", bye)
		if bye == "" {
			msg = fmt.Sprintf("GRIZ SERVER STOPPED (pid %d) before answering this call. Tell the user now. "+
				"griz starts a new server on the next call and says what happened.", sc.pid)
			if c.method == "tools/call" && (c.tool == "store" || c.tool == "delete") {
				msg += " This " + c.tool + " may or may not have been saved; check with get before retrying."
			} else {
				msg += " The call can be retried."
			}
		}
		r.failLocked(c, msg)
	}
}

// failLocked answers one call with an error. Tool calls get an MCP error
// result (the model sees its text); other requests a JSON-RPC error.
func (r *Relay) failLocked(c call, msg string) {
	delete(r.waiting, string(c.id))
	var out any
	if c.method == "tools/call" {
		out = map[string]any{"jsonrpc": "2.0", "id": c.id, "result": map[string]any{
			"content": []any{map[string]any{"type": "text", "text": msg}}, "isError": true}}
	} else {
		out = map[string]any{"jsonrpc": "2.0", "id": c.id, "error": map[string]any{"code": -32603, "message": msg}}
	}
	b, _ := json.Marshal(out)
	r.write(b)
}
