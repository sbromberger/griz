// Package daemon runs the one shared griz server and the thin relays and
// CLI clients that talk to it over a Unix socket.
//
// Wire format: newline-delimited JSON. Each side first sends a hello line.
// After that, an MCP connection carries the MCP JSON-RPC messages unchanged,
// plus control lines from the server that have a top-level "griz" key
// (JSON-RPC messages never do). A CLI connection carries one "cli" line from
// the client, then "out" lines and one "done" line from the server.
package daemon

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
)

// proto is the wire protocol version. Relays and servers with different
// versions refuse each other loudly.
const proto = 1

// maxControl caps a hello, cli or other control line. MCP lines use the MCP
// SDK's own cap.
const maxControl = 1 << 20

// msg is every control line. Unknown fields are rejected.
type msg struct {
	Griz    string   `json:"griz"` // hello, stale, bye, cli, out, done
	Proto   int      `json:"proto,omitempty"`
	Version string   `json:"version,omitempty"`
	Binary  string   `json:"binary,omitempty"`  // hash of the griz executable
	Mode    string   `json:"mode,omitempty"`    // hello from a client: mcp or cli
	Data    string   `json:"data,omitempty"`    // hello from a client: its data directory
	PID     int      `json:"pid,omitempty"`     // sender's process id
	Started string   `json:"started,omitempty"` // server hello: RFC 3339 start time
	Args    []string `json:"args,omitempty"`    // cli: the command and its arguments
	Text    string   `json:"text,omitempty"`    // out: output text
	Error   string   `json:"error,omitempty"`   // done: the command's error, "" if none
	Reason  string   `json:"reason,omitempty"`  // bye, stale: why
}

// parseMsg decodes one control line strictly.
func parseMsg(line []byte) (msg, error) {
	var m msg
	dec := json.NewDecoder(bytes.NewReader(line))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return m, fmt.Errorf("bad control message: %v", err)
	}
	if dec.More() {
		return m, errors.New("bad control message: trailing data")
	}
	switch m.Griz {
	case "hello", "stale", "bye", "cli", "out", "done":
	default:
		return m, fmt.Errorf("bad control message: unknown kind %q", m.Griz)
	}
	return m, nil
}

func writeMsg(w io.Writer, m msg) error {
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	_, err = w.Write(append(b, '\n'))
	return err
}

// readLine reads one line of at most max bytes (without the newline).
func readLine(r *bufio.Reader, max int) ([]byte, error) {
	var line []byte
	for {
		chunk, err := r.ReadSlice('\n')
		line = append(line, chunk...)
		if len(line) > max+1 {
			return nil, fmt.Errorf("line longer than %d bytes", max)
		}
		if err == nil {
			return line[:len(line)-1], nil
		}
		if !errors.Is(err, bufio.ErrBufferFull) {
			if err == io.EOF && len(line) > 0 {
				return nil, io.ErrUnexpectedEOF
			}
			return nil, err
		}
	}
}

// Paths.

// socketDir is <state dir>/server. It is found from the state directory,
// never from $TMPDIR: Claude Desktop starts griz without $TMPDIR and Claude
// Code with it, and they must reach the same server.
func socketDir(state string) string { return filepath.Join(state, "server") }

// RelayName and ServerName are the names relays and servers show in ps
// (their argv[0]); a process started under one of them is that role.
const (
	RelayName  = "griz (relay)"
	ServerName = "griz (server)"
)

func socketPath(state string) string { return filepath.Join(socketDir(state), "griz.sock") }
func lockPath(state string) string   { return filepath.Join(socketDir(state), "server.lock") }

// ensureDir creates the socket directory (0700) and checks it is a real
// directory owned by this user that no one else can write to or enter.
func ensureDir(state string) error {
	if state == "" {
		return fmt.Errorf("no state directory")
	}
	dir := socketDir(state)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	if err := checkOwned(dir, true); err != nil {
		return err
	}
	if p := socketPath(state); len(p) > 103 {
		return fmt.Errorf("socket path %s is %d bytes, over the 103 macOS allows; set state_dir (or XDG_STATE_HOME) to a shorter directory", p, len(p))
	}
	return nil
}

// checkOwned refuses a path not owned by this user, or open to others
// (any group or other permission bits on the directory; write bits on the socket).
func checkOwned(path string, dir bool) error {
	fi, err := os.Lstat(path)
	if err != nil {
		return err
	}
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("REFUSING %s: can't read its owner", path)
	}
	switch {
	case int(st.Uid) != os.Getuid():
		return fmt.Errorf("REFUSING %s: owned by user %d, not you (%d); someone else may be impersonating griz", path, st.Uid, os.Getuid())
	case dir && !fi.IsDir():
		return fmt.Errorf("REFUSING %s: not a directory", path)
	case dir && fi.Mode().Perm()&0o077 != 0:
		return fmt.Errorf("REFUSING %s: permissions %v let other users in; it must be 0700", path, fi.Mode().Perm())
	case !dir && fi.Mode()&os.ModeSocket == 0:
		return fmt.Errorf("REFUSING %s: not a socket", path)
	case !dir && fi.Mode().Perm()&0o077 != 0:
		return fmt.Errorf("REFUSING %s: permissions %v let other users in; it must be 0600", path, fi.Mode().Perm())
	}
	return nil
}

// executable returns the path of the running griz binary.
func executable() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(exe)
}

// binaryID is a short hash of the griz executable file at path.
func binaryID(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil))[:16], nil
}

func fileStamp(path string) string {
	fi, err := os.Stat(path)
	if err != nil {
		return ""
	}
	return fmt.Sprint(fi.Size(), fi.ModTime().UnixNano())
}
