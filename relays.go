package main

import (
	"flag"
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"griz/internal/daemon"
)

// griz relay status and kill. Relays are found with ps, not from the
// server's sessions: a relay that never reached the server (say, one that
// couldn't start it) has no session but is still running.

type relayProc struct {
	pid, ppid int
	up        string // ps etime: [[dd-]hh:]mm:ss
	parent    string // the parent's executable name
}

// listRelays returns this user's relays (ps shows only our own processes).
func listRelays() ([]relayProc, error) {
	b, err := exec.Command("ps", "-x", "-o", "pid=,ppid=,etime=,command=").Output()
	if err != nil {
		return nil, fmt.Errorf("ps: %v", err)
	}
	var rs []relayProc
	var ppids []string
	for line := range strings.SplitSeq(string(b), "\n") {
		f := strings.Fields(line)
		if len(f) < 4 || strings.Join(f[3:], " ") != daemon.RelayName {
			continue
		}
		pid, err1 := strconv.Atoi(f[0])
		ppid, err2 := strconv.Atoi(f[1])
		if err1 != nil || err2 != nil {
			continue
		}
		rs = append(rs, relayProc{pid: pid, ppid: ppid, up: f[2]})
		ppids = append(ppids, f[1])
	}
	if len(rs) == 0 {
		return nil, nil
	}
	// The parents' names, in one more ps. comm is the executable's path,
	// which may hold spaces, so it is the rest of the line.
	names := map[int]string{}
	if b, err := exec.Command("ps", "-o", "pid=,comm=", "-p", strings.Join(ppids, ",")).Output(); err == nil {
		for line := range strings.SplitSeq(string(b), "\n") {
			line = strings.TrimSpace(line)
			num, rest, ok := strings.Cut(line, " ")
			if pid, err := strconv.Atoi(num); ok && err == nil {
				names[pid] = filepath.Base(strings.TrimSpace(rest))
			}
		}
	}
	for i := range rs {
		rs[i].parent = names[rs[i].ppid]
		if rs[i].parent == "" {
			rs[i].parent = "?"
		}
	}
	return rs, nil
}

// stillRelay checks pid right before it is signalled, in case the relay
// exited and its pid went to something else.
func stillRelay(pid int) bool {
	b, err := exec.Command("ps", "-o", "command=", "-p", strconv.Itoa(pid)).Output()
	return err == nil && strings.TrimSpace(string(b)) == daemon.RelayName
}

func (r relayProc) String() string {
	return fmt.Sprintf("pid %d, from %s (pid %d), up %s", r.pid, r.parent, r.ppid, r.up)
}

func runRelayStatus(args []string) error {
	if len(args) > 0 {
		return usageErr("relay status")
	}
	rs, err := listRelays()
	if err != nil {
		return err
	}
	if len(rs) == 0 {
		_, _ = fmt.Fprintln(out, "no relays running")
		return nil
	}
	// In the server, each relay's session; without one, none is connected.
	session := map[int]string{}
	if srvApp != nil {
		for _, s := range srvApp.Sessions() {
			session[s.ClientPID] = fmt.Sprintf("session %d", s.ID)
		}
	}
	for _, r := range rs {
		s := session[r.pid]
		if s == "" {
			s = "not connected"
		}
		_, _ = fmt.Fprintf(out, "relay %s, %s\n", r, s)
	}
	return nil
}

func runRelayKill(args []string) error {
	fs := flag.NewFlagSet("relay kill", flag.ContinueOnError)
	all := fs.Bool("all", false, "every relay")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if *all == (len(pos) > 0) {
		return usageErr("relay kill")
	}
	rs, err := listRelays()
	if err != nil {
		return err
	}
	byPID := map[int]relayProc{}
	for _, r := range rs {
		byPID[r.pid] = r
	}
	targets := rs
	if !*all {
		targets = nil
		for _, p := range pos {
			pid, err := strconv.Atoi(p)
			if err != nil {
				return fmt.Errorf("%q is not a pid", p)
			}
			r, ok := byPID[pid]
			if !ok {
				return fmt.Errorf("pid %d is not a griz relay of yours (griz relay status lists them); nothing stopped", pid)
			}
			targets = append(targets, r)
		}
	}
	if len(targets) == 0 {
		_, _ = fmt.Fprintln(out, "no relays running")
		return nil
	}
	for _, r := range targets {
		if !stillRelay(r.pid) {
			_, _ = fmt.Fprintf(out, "relay pid %d is gone; skipped\n", r.pid)
			continue
		}
		if err := syscall.Kill(r.pid, syscall.SIGTERM); err != nil {
			return fmt.Errorf("relay pid %d: %v", r.pid, err)
		}
		_, _ = fmt.Fprintf(out, "stopped relay %s\n", r)
	}
	return nil
}
