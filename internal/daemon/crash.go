package daemon

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const crashLayout = "20060102-150405"

func crashDir(stateDir string) string { return filepath.Join(stateDir, "crashes") }

// StartCrashFile sends Go panics and fatal errors of this process to
// crashes/<time>-<pid>.txt (directory 0700). The file exists, empty, while
// the server runs; done removes it on a clean exit. An empty file left by a
// process that is gone means it was killed without a crash report.
func StartCrashFile(stateDir string) (done func(), err error) {
	dir := crashDir(stateDir)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	os.Chmod(dir, 0o700)
	name := filepath.Join(dir, fmt.Sprintf("%s-%d.txt", time.Now().Format(crashLayout), os.Getpid()))
	f, err := os.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0o600)
	if err != nil {
		return nil, err
	}
	if err := debug.SetCrashOutput(f, debug.CrashOptions{}); err != nil {
		f.Close()
		os.Remove(name)
		return nil, err
	}
	return func() {
		debug.SetCrashOutput(nil, debug.CrashOptions{})
		f.Close()
		os.Remove(name)
	}, nil
}

// crash is one crash file.
type crash struct {
	Path    string
	pid     int
	started time.Time
	first   string // first non-empty line of the crash output ("" if none)
	Size    int64
}

func crashes(stateDir string) []crash {
	ents, _ := os.ReadDir(crashDir(stateDir))
	var out []crash
	for _, e := range ents {
		name := strings.TrimSuffix(e.Name(), ".txt")
		i := strings.LastIndex(name, "-")
		if i < 0 || !strings.HasSuffix(e.Name(), ".txt") {
			continue
		}
		pid, err := strconv.Atoi(name[i+1:])
		if err != nil {
			continue
		}
		t, _ := time.ParseInLocation(crashLayout, name[:i], time.Local)
		c := crash{Path: filepath.Join(crashDir(stateDir), e.Name()), pid: pid, started: t}
		if fi, err := e.Info(); err == nil {
			c.Size = fi.Size()
		}
		c.first = firstLine(c.Path)
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].started.Before(out[j].started) })
	return out
}

func firstLine(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if l := strings.TrimSpace(sc.Text()); l != "" {
			return l
		}
	}
	return ""
}

// alive reports whether a process exists.
func alive(pid int) bool {
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}

// describe says what happened to server pid, from its crash file.
func describe(stateDir string, pid int) string {
	for _, c := range crashes(stateDir) {
		if c.pid == pid && c.Size > 0 {
			if c.first == "" {
				return "crash report " + c.Path
			}
			return c.first + " (crash report " + c.Path + ")"
		}
	}
	if alive(pid) {
		return "the server process is still running but dropped the connection"
	}
	return "disappeared without a crash report (killed?)"
}

// CrashReport lists crashes (and servers that vanished without a report).
// Once listed, a report with content is kept as <name>.shown.txt (which
// crashes skips) so its stacks can still be read; an empty marker is deleted.
// "" if there are none.
func CrashReport(stateDir string) string {
	var lines []string
	for _, c := range crashes(stateDir) {
		switch {
		case c.Size > 0:
			kept := strings.TrimSuffix(c.Path, ".txt") + ".shown.txt"
			lines = append(lines, fmt.Sprintf("  server pid %d (started %s) CRASHED: %s (report: %s)", c.pid, c.started.Format("2006-01-02 15:04:05"), c.first, kept))
			os.Rename(c.Path, kept)
		case c.pid == os.Getpid() || alive(c.pid):
			continue // running
		default:
			lines = append(lines, fmt.Sprintf("  server pid %d (started %s) disappeared without a crash report (killed?)", c.pid, c.started.Format("2006-01-02 15:04:05")))
			os.Remove(c.Path)
		}
	}
	if len(lines) == 0 {
		return ""
	}
	return "crashes since last shown:\n" + strings.Join(lines, "\n")
}
