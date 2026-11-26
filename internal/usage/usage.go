// Package usage writes and reads usage.log: one tab-separated line per tool call.
//
//	time(RFC3339)  tool  bytes-in  bytes-out  ms  ids(comma-separated, returned in full)
package usage

import (
	"bufio"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type Log struct {
	Path    string
	enabled bool
}

func New(stateDir string, enabled bool) *Log {
	return &Log{Path: filepath.Join(stateDir, "usage.log"), enabled: enabled}
}

type Entry struct {
	Time    time.Time
	Tool    string
	In, Out int
	MS      int64
	IDs     []string
}

// Write appends one line. O_APPEND writes of one short line are atomic, so
// several processes can share the file. Errors are ignored: the log is optional.
func (l *Log) Write(e Entry) {
	if !l.enabled {
		return
	}
	f, err := os.OpenFile(l.Path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	defer func() { _ = f.Close() }()
	_ = f.Chmod(0o600) // the log names memories: only this user may read it, even if it was made looser
	_, _ = fmt.Fprintf(f, "%s\t%s\t%d\t%d\t%d\t%s\n", e.Time.UTC().Format(time.RFC3339), e.Tool, e.In, e.Out, e.MS, strings.Join(e.IDs, ","))
}

// Read parses the whole log. A missing log is empty.
func (l *Log) Read() ([]Entry, error) {
	f, err := os.Open(l.Path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	var out []Entry
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		p := strings.Split(sc.Text(), "\t")
		if len(p) < 5 {
			continue
		}
		var e Entry
		e.Time, _ = time.Parse(time.RFC3339, p[0])
		e.Tool = p[1]
		e.In, _ = strconv.Atoi(p[2])
		e.Out, _ = strconv.Atoi(p[3])
		e.MS, _ = strconv.ParseInt(p[4], 10, 64)
		if len(p) > 5 && p[5] != "" {
			e.IDs = strings.Split(p[5], ",")
		}
		out = append(out, e)
	}
	return out, sc.Err()
}

// LastUsed maps each id to the last time it was returned.
func (l *Log) LastUsed() (map[string]time.Time, error) {
	es, err := l.Read()
	out := map[string]time.Time{}
	for _, e := range es {
		for _, id := range e.IDs {
			if e.Time.After(out[id]) {
				out[id] = e.Time
			}
		}
	}
	return out, err
}

// Forget rewrites the log without any line that mentions one of ids.
func (l *Log) Forget(ids []string) error {
	b, err := os.ReadFile(l.Path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var keep []string
	for _, line := range strings.SplitAfter(string(b), "\n") {
		drop := false
		for _, id := range ids {
			if strings.Contains(line, id) {
				drop = true
				break
			}
		}
		if !drop {
			keep = append(keep, line)
		}
	}
	tmp := l.Path + ".tmp"
	_ = os.Remove(tmp) // WriteFile keeps the mode of a file that already exists
	if err := os.WriteFile(tmp, []byte(strings.Join(keep, "")), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, l.Path)
}
