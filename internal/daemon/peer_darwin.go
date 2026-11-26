package daemon

import (
	"bytes"
	"fmt"
	"net"
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// peer returns the user id and process id at the other end of c, from the kernel.
func peer(c *net.UnixConn) (uid, pid int, err error) {
	raw, err := c.SyscallConn()
	if err != nil {
		return 0, 0, err
	}
	var cerr error
	err = raw.Control(func(fd uintptr) {
		var cred *unix.Xucred
		if cred, cerr = unix.GetsockoptXucred(int(fd), unix.SOL_LOCAL, unix.LOCAL_PEERCRED); cerr != nil {
			return
		}
		uid = int(cred.Uid)
		pid, cerr = unix.GetsockoptInt(int(fd), unix.SOL_LOCAL, unix.LOCAL_PEERPID)
	})
	if err == nil {
		err = cerr
	}
	return uid, pid, err
}

// exePath is the absolute, resolved executable path of process pid, from
// the kernel (what libproc's proc_pidpath returns: proc_info with
// PROC_PIDPATHINFO). Unlike the argv path, it doesn't depend on how the
// program was started (relative path, symlink).
func exePath(pid int) (string, error) {
	const (
		procInfoCallPIDInfo = 2
		procPIDPathInfo     = 11
		maxPath             = 4 * 1024 // PROC_PIDPATHINFO_MAXSIZE
	)
	buf := make([]byte, maxPath)
	// libproc's wrapper needs cgo; the raw call is what it makes.
	_, _, errno := syscall.Syscall6(unix.SYS_PROC_INFO, procInfoCallPIDInfo, uintptr(pid), procPIDPathInfo, 0, //nolint:staticcheck // SA1019: no cgo-free wrapper exists
		uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	if errno != 0 {
		return "", errno
	}
	n := bytes.IndexByte(buf, 0)
	if n <= 0 {
		return "", fmt.Errorf("no executable path for pid %d", pid)
	}
	return string(buf[:n]), nil
}

// CheckClient is the server's check on a new connection: the peer must be
// this user, running the griz executable.
func CheckClient(c *net.UnixConn) error {
	uid, pid, err := peer(c)
	if err != nil {
		return fmt.Errorf("can't read the peer's credentials: %v", err)
	}
	if uid != os.Getuid() {
		return fmt.Errorf("peer user %d is not %d", uid, os.Getuid())
	}
	own, err := executable()
	if err != nil {
		return err
	}
	exe, err := exePath(pid)
	if err != nil {
		return fmt.Errorf("can't read the executable of pid %d: %v", pid, err)
	}
	// The same file, however each was invoked.
	a, err1 := os.Stat(exe)
	b, err2 := os.Stat(own)
	if err1 != nil || err2 != nil || !os.SameFile(a, b) {
		return fmt.Errorf("pid %d runs %s, not this griz (%s)", pid, exe, own)
	}
	return nil
}

// CheckServer is a client's check on the server: it must run as this user.
func CheckServer(c *net.UnixConn) error {
	uid, _, err := peer(c)
	if err != nil {
		return fmt.Errorf("REFUSING the griz socket: can't read the server's credentials: %v", err)
	}
	if uid != os.Getuid() {
		return fmt.Errorf("REFUSING the griz socket: the server runs as user %d, not you (%d); someone may be impersonating griz", uid, os.Getuid())
	}
	return nil
}
