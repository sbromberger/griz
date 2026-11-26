// Command griz is a small local memory server for Claude (MCP over stdio),
// and the command line for looking at and changing its memories.
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"syscall"
	"time"

	"griz/internal/app"
	"griz/internal/config"
	"griz/internal/daemon"
	_ "griz/internal/embed/bert" // registers the model runner
	"griz/internal/server"
)

var version = buildVersion()

// buildVersion is the version go build stamps from git: the tag (v0.2.0)
// for a clean build of a tagged commit, otherwise a pseudo-version that
// names the commit (v0.2.1-0.20261125205706-0cf931e07ca9), with +dirty when
// the tree had uncommitted changes. A build outside git is "devel".
func buildVersion() string {
	if bi, ok := debug.ReadBuildInfo(); ok && bi.Main.Version != "" && bi.Main.Version != "(devel)" {
		return bi.Main.Version
	}
	return "devel"
}

var (
	out    io.Writer = os.Stdout
	errOut io.Writer = os.Stderr
	// srvApp is the server's App while a command runs inside the server,
	// and cmdLine describes the griz command that asked.
	srvApp  *app.App
	cmdLine string
)

// Where a command runs.
const (
	local  = iota // in this process
	either        // in the server if one is running, else here
	inSrv         // in the server, started if needed (every write, so it is the only writer)
)

type command struct {
	name, args, short string
	long              string // more for griz help <name>
	where             int
	gap               bool // help starts a new block of its group here
	// prep, if set, runs here before the command is sent to the server
	// (or run locally): reading stdin, asking for confirmation, noting a TTY.
	prep func(args []string) ([]string, error)
	run  func(args []string) error
}

// groups take a subcommand: griz memory term ..., griz server stop.
var groups = map[string]string{
	"memory": "search, change and look after memories",
	"server": "the shared server every session and command talks to",
	"relay":  "one per Claude app, each a griz mcp",
	"config": "settings",
}

var commands []*command

func init() {
	commands = []*command{
		{name: "help", args: "[command]", short: "show usage, or details for one command", run: runHelp},
		{name: "version", short: "print this griz's version, and the running server's", run: runVersion},
		{name: "env", short: "show every file and directory griz uses", run: runEnv},
		{name: "status", short: "processes, memories, model, embedding, scan, sizes, warnings, changed settings", where: either, run: runStatus},
		{name: "mcp", short: "MCP over stdio, relayed to the shared server (started if needed); what Claude runs", run: func([]string) error { return runRelay() }},

		{name: "memory recall", args: "<query> [--archived] [--pinned] [--budget n]", where: either, prep: noteTTY, run: runRecall,
			short: "search memories, as Claude does",
			long: `--archived also searches archived memories; --pinned lists every pinned memory
(no query needed); --budget caps the reply in tokens. Ids show in full, with
the part that is enough to name the memory in bold.`},
		{name: "memory get", args: "<id>...", where: either, prep: noteTTY, run: runGet, short: "print memories in full",
			long: idHelp},
		{name: "memory store", args: "<id> [--body text|-] [--append text|-] [--old text --new text [--all [--count n]] [--preview]] [--tags a,b] [--term t] [--expires when] [--archived[=false]] [--pin[=false]] [--force] [--new-ns]",
			where: inSrv, prep: readStdin, run: runStore, short: "save a new memory, or change one",
			long: `For a new memory give a path (folder/name) and --body; griz assigns the id.
For an existing one give its id (or a key, below) and one of: --body to
replace the text, --old and --new to edit in place, --append for a dated
entry; or only --tags, --term, --expires, --archived or --pin. "-" reads the
text from stdin. --old must match exactly once; --all replaces every copy
(--count n refuses unless there are n); --preview prints the lines that
would change and saves nothing. --term (short, medium, long or infinite)
sets how fast it fades; --expires sets a hard date (2026-12-31, 30d, 6m, or
never) after which it is archived. --force saves even if a similar memory exists;
--new-ns allows a new folder.

` + idHelp},
		{name: "memory delete", args: "<id>...", where: inSrv, run: runDelete, short: "delete memories (griz memory restore brings one back)", long: idHelp},
		{name: "memory move", args: "<id> <folder or ns/slug> [--new-ns]", where: inSrv, run: runMove,
			short: "move or rename a memory, keeping its id",
			long: `Like mv: an existing folder (home/infra), or any target ending in "/",
receives the memory under its own name; otherwise the target is its new
path, and its last part the new name. --new-ns allows a folder that doesn't
exist yet.

` + idHelp},

		{name: "memory term", gap: true, args: "<short|medium|long|infinite> <id>...", where: inSrv, run: runTerm,
			short: "change how fast memories fade in recall (renews them)",
			long: `A memory's weight in recall halves its distance to term.floor every
half-life of its term (term.short, term.medium, term.long in the config),
counted from its last edit or renewal. infinite and pinned memories don't
fade. Nothing is archived for its age.

` + idHelp},
		{name: "memory expire", args: "<date|duration|never> <id>...", where: inSrv, run: runExpire,
			short: "set a hard date: memories are archived after it",
			long:  "A date (2026-12-31), a duration from today (3w, 6m), or never to remove it.\nFading doesn't depend on it.\n\n" + idHelp},
		{name: "memory renew", args: "<id>...", where: inSrv, run: runRenew,
			short: "restart memories' fading today, without editing them",
			long:  "Term and hard date are unchanged. Edits renew too, and each get takes back\npart of a memory's age (term.get_renews).\n\n" + idHelp},
		{name: "memory archive", args: "<id>...", where: inSrv, run: func(a []string) error { return runArchive(a, true) },
			short: "archive memories (found by recall --archived, loaded only on request)", long: idHelp},
		{name: "memory unarchive", args: "<id>...", where: inSrv, run: func(a []string) error { return runArchive(a, false) },
			short: "make archived memories current again, and renew them", long: "A hard date that has passed is removed.\n\n" + idHelp},
		{name: "memory pin", args: "<id>...", where: inSrv, run: func(a []string) error { return runPin(a, true) },
			short: "pin memories: every session loads them at its start", long: idHelp},
		{name: "memory unpin", args: "<id>...", where: inSrv, run: func(a []string) error { return runPin(a, false) },
			short: "unpin memories", long: idHelp},
		{name: "memory restore", args: "<id>", where: inSrv, run: runRestore, short: "bring back a deleted memory, renewed"},
		{name: "memory purge", args: "<id>... [--yes]", where: inSrv, prep: confirmPurge, run: runPurge,
			short: "for mistakes: erase memories from git history for good (asks first)"},

		{name: "memory edit", gap: true, short: "open $EDITOR in the memory directory (changes are committed once they settle)", run: runMemoryEdit},
		{name: "memory stats", short: "counts of memories, and a summary of the usage log", where: either, run: runStats},
		{name: "memory lint", short: "report problems and things worth a look", where: either, run: runLint},
		{name: "memory gc", where: inSrv, run: func([]string) error { return runGC() }, short: "drop vectors no memory uses any more"},
		{name: "memory reindex", where: inSrv, run: func([]string) error { return runReindex() }, short: "recompute every embedding, then drop unused vectors"},
		{name: "memory explain", args: "<query> [--id id]", where: either, run: runExplain,
			short: "show how recall ranks results for a query (and where one memory landed)"},

		{name: "server status", where: either, run: runServerStatus, short: "the server's pid, uptime and sessions"},
		{name: "server stop", run: runServerStop, short: "stop the server; sessions reconnect to a new one on their next call"},
		{name: "server restart", run: runServerRestart, short: "stop the server and start a new one (after a config change)"},
		{name: "server debug", args: "<on|off>", where: inSrv, run: runDebug,
			short: "debug mode for every session and command, until the server exits",
			long:  "Claude's replies gain a note of what happened; griz commands print where their time went."},
		{name: "server timings", where: either, run: runTimings, short: "typical and slowest times of recent operations in the server"},
		{name: "server profile", args: "[--seconds n | --reembed] [--out file]", where: inSrv, prep: absOut, run: runProfile,
			short: "CPU-profile the server and list where the time goes",
			long: `By default it samples the server for --seconds (10), whatever it is doing,
as Claude's griz tool does. --reembed profiles a full re-embed instead.
--out also writes the raw profile, for go tool pprof.`},
		{name: "server run", run: func([]string) error { return runServe() },
			short: "run the server in this terminal, to see why it won't start (griz mcp starts it otherwise)"},

		{name: "relay status", where: either, run: runRelayStatus, short: "every relay of yours: its app, uptime and session"},
		{name: "relay kill", args: "<pid>... | --all", run: runRelayKill,
			short: "stop relays (that app loses griz until it starts griz mcp again)",
			long: `Only pids griz relay status lists are signalled, each checked again just
before. The app's griz connection ends; Claude Code reconnects it with /mcp,
Claude Desktop when it restarts.`},

		{name: "config show", short: "print the config file, directories and effective settings", run: func([]string) error { return runConfig() }},
		{name: "config edit", short: "open the config file in $EDITOR, then check it", run: runConfigEdit},
	}
}

const idHelp = `An id is shown as folder/name [machine-1a2b3c4d]. Commands take the id, the
path, the start of the id's hex part (as few characters as tell it apart),
the end of the path, or part of the name, tried in that order; if a key
matches several memories griz lists them.`

func find(name string) *command {
	for _, c := range commands {
		if c.name == name {
			return c
		}
	}
	return nil
}

// lookup splits a command line into its command and arguments.
func lookup(args []string) (*command, []string, error) {
	if _, ok := groups[args[0]]; ok && len(args) > 1 {
		if c := find(args[0] + " " + args[1]); c != nil {
			return c, args[2:], nil
		}
		return nil, nil, fmt.Errorf("unknown command \"%s %s\"\n\n%s", args[0], args[1], groupUsage(args[0]))
	}
	if c := find(args[0]); c != nil {
		return c, args[1:], nil
	}
	return nil, nil, fmt.Errorf("unknown command %q (griz help lists them)", args[0])
}

func main() {
	server.Version = version
	if os.Getenv(daemon.ServeEnv) == "1" { // a server started by a relay
		_ = os.Unsetenv(daemon.ServeEnv) // not for git or anything else it runs
		exit(runServe())
	}
	if os.Args[0] == daemon.RelayName { // see runRelay
		exit(runRelay())
	}
	args := os.Args[1:]
	if len(args) == 0 {
		_, _ = fmt.Fprint(errOut, usage())
		os.Exit(2)
	}
	switch args[0] {
	case "--help", "-h":
		args[0] = "help"
	case "--version", "-v":
		args[0] = "version"
	}
	if _, grp := groups[args[0]]; grp && (len(args) == 1 || args[1] == "--help" || args[1] == "-h") {
		_, _ = fmt.Fprint(out, groupUsage(args[0])) // a group alone, or asking for help
		return
	}
	c, rest, err := lookup(args)
	if err != nil {
		_, _ = fmt.Fprintln(errOut, "griz:", err)
		os.Exit(2)
	}
	if c.name == "mcp" && len(rest) > 0 { // griz mcp status and the like moved to griz server
		_, _ = fmt.Fprintf(errOut, "griz: unknown command \"mcp %s\": griz mcp takes no arguments (griz server and griz relay control the processes)\n", rest[0])
		os.Exit(2)
	}
	if c.name != "mcp" && (slices.Contains(rest, "--help") || slices.Contains(rest, "-h")) {
		_, _ = fmt.Fprint(out, help(c))
		return
	}
	if err := dispatch(c, rest); err != nil {
		_, _ = fmt.Fprintln(errOut, "griz:", err)
		os.Exit(1)
	}
}

func dispatch(c *command, args []string) error {
	if c.prep != nil {
		var err error
		if args, err = c.prep(args); err != nil {
			return err
		}
	}
	if c.where == local {
		return c.run(args)
	}
	cl, err := client()
	if err != nil {
		return err
	}
	err = cl.RunCLI(append([]string{c.name}, args...), out, c.where == inSrv)
	if c.where == either && errors.Is(err, daemon.ErrNoServer) {
		return c.run(args)
	}
	return err
}

// runInServer runs a CLI command inside the server (one at a time). In
// debug mode it adds where the time went.
func runInServer(a *app.App) func(args []string, pid int, w io.Writer) error {
	return func(args []string, pid int, w io.Writer) error {
		c := find(args[0])
		if c == nil || c.where == local {
			return fmt.Errorf("griz %s doesn't run in the server", args[0])
		}
		out, errOut, srvApp, cmdLine = w, w, a, fmt.Sprintf("pid %d (griz %s)", pid, c.name)
		defer func() { out, errOut, srvApp, cmdLine = os.Stdout, os.Stderr, nil, "" }()
		start := time.Now()
		err := c.run(args[1:])
		if a.Debugging() {
			for _, l := range a.TakeDebug() {
				_, _ = fmt.Fprintln(w, l)
			}
			_, _ = fmt.Fprintf(w, "griz debug: griz %s took %s in the server (pid %d)\n", c.name, time.Since(start).Round(100*time.Microsecond), os.Getpid())
		}
		return err
	}
}

// ---- help ----

// groupOrder is the order help lists the groups in.
var groupOrder = []string{"memory", "server", "relay", "config"}

func usage() string {
	var b strings.Builder
	b.WriteString("usage: griz <command> [args]   (griz help <command> for details)\n\n")
	line := func(c *command) {
		u := c.name
		if c.args != "" {
			u += " " + c.args
		}
		if len(u) > 30 {
			u = c.name + " ..."
		}
		fmt.Fprintf(&b, "  %-30s %s\n", u, c.short)
	}
	for _, c := range commands {
		if !strings.Contains(c.name, " ") {
			line(c)
		}
	}
	for _, g := range groupOrder {
		fmt.Fprintf(&b, "\n%s: %s\n", g, groups[g])
		for _, c := range commands {
			if strings.HasPrefix(c.name, g+" ") {
				if c.gap {
					b.WriteByte('\n')
				}
				line(c)
			}
		}
	}
	return b.String()
}

func groupUsage(g string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "griz %s: %s\n\n", g, groups[g])
	for _, c := range commands {
		if strings.HasPrefix(c.name, g+" ") {
			if c.gap {
				b.WriteByte('\n')
			}
			fmt.Fprintf(&b, "  griz %-18s %s\n", c.name, c.short)
		}
	}
	return b.String()
}

func help(c *command) string {
	u := "usage: griz " + c.name
	if c.args != "" {
		u += " " + c.args
	}
	s := u + "\n\n" + c.short + "\n"
	if c.long != "" {
		s += "\n" + c.long + "\n"
	}
	return s
}

func runHelp(args []string) error {
	if len(args) == 0 {
		_, _ = fmt.Fprint(out, usage())
		return nil
	}
	if _, ok := groups[args[0]]; ok && len(args) == 1 {
		_, _ = fmt.Fprint(out, groupUsage(args[0]))
		return nil
	}
	c, _, err := lookup(args)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprint(out, help(c))
	return nil
}

// runVersion prints this griz's version and the running server's.
func runVersion([]string) error {
	exe, _ := os.Executable()
	_, _ = fmt.Fprintf(out, "griz %s (%s, %s/%s) %s\n", version, runtime.Version(), runtime.GOOS, runtime.GOARCH, exe)
	cl, err := client()
	if err != nil {
		return err
	}
	switch v, pid, err := cl.ServerVersion(); {
	case errors.Is(err, daemon.ErrNoServer):
		_, _ = fmt.Fprintln(out, "server not running")
	case err != nil:
		_, _ = fmt.Fprintln(out, "server:", err)
	default:
		_, _ = fmt.Fprintf(out, "server griz %s (pid %d)\n", v, pid)
	}
	return nil
}

// absOut makes --out absolute before the command goes to the server, whose
// working directory isn't this one.
func absOut(args []string) ([]string, error) {
	for i, a := range args {
		switch {
		case a == "--out" && i+1 < len(args):
			p, err := filepath.Abs(args[i+1])
			if err != nil {
				return nil, err
			}
			args[i+1] = p
		case strings.HasPrefix(a, "--out="):
			p, err := filepath.Abs(strings.TrimPrefix(a, "--out="))
			if err != nil {
				return nil, err
			}
			args[i] = "--out=" + p
		}
	}
	return args, nil
}

// ---- the server ----

func open() (*app.App, error) {
	if srvApp != nil {
		return srvApp, nil
	}
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}
	return app.Open(cfg)
}

func client() (*daemon.Client, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}
	start, _ := cfg.Server.StartDuration()
	return &daemon.Client{Version: version, StateDir: cfg.StateDir, DataDir: cfg.DataDir, StartTimeout: start, Check: daemon.CheckServer}, nil
}

// runRelay is griz mcp: a thin stdio relay to the shared server.
func runRelay() error {
	if os.Args[0] != daemon.RelayName {
		// Show as "griz (relay)" in ps: exec this binary again under that
		// name, before reading anything. Same pid and pipes; if the exec
		// fails, relay as we are.
		if exe, err := os.Executable(); err == nil {
			_ = syscall.Exec(exe, []string{daemon.RelayName}, os.Environ())
		}
	}
	cl, err := client()
	if err != nil {
		return err
	}
	r := &daemon.Relay{Client: *cl, In: os.Stdin, Out: os.Stdout}
	return r.Run()
}

// runServe is the shared server. It exits quietly if one is already running.
func runServe() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	srv := &daemon.Server{Version: version, StateDir: cfg.StateDir, Check: daemon.CheckClient}
	srv.IdleExit, _ = cfg.Server.IdleDuration()
	srv.DrainWait, _ = cfg.Server.DrainDuration()
	if err := srv.Listen(); err != nil {
		if errors.Is(err, daemon.ErrRunning) {
			return nil
		}
		return err
	}
	if done, err := daemon.StartCrashFile(cfg.StateDir); err == nil {
		defer done()
	}
	a, err := app.Open(cfg)
	if err != nil {
		return err
	}
	// A server that died left its crash file; every session hears of it once
	// (relays only say the call failed).
	if c := daemon.CrashReport(cfg.StateDir); c != "" {
		a.Warn(app.WarnCrash, "GRIZ SERVER CRASHED: an earlier griz server stopped without shutting down. Tell the user now.\n%s", c)
	}
	if err := a.CommitEdits(); err != nil {
		a.Warn(app.WarnCommit, "committing hand edits: %v", err)
	}
	if n, err := a.MigrateExpiry(); err != nil {
		a.Warn(app.WarnExpiry, "dropping term-derived expiry dates: %v", err)
	} else if n > 0 {
		a.Info("memories now fade instead of expiring: dropped the expiry dates their terms had set on %d memories", n)
	}
	if gone, err := a.ArchiveExpired(); err != nil {
		a.Warn(app.WarnExpiry, "archiving expired memories: %v", err)
	} else if len(gone) > 0 {
		a.Info("archived (expired): %s", strings.Join(gone, ", "))
	}
	if err := a.RefreshPinWarnings(); err != nil {
		return err
	}
	srv.A = a
	a.Server = true
	srv.RunCLI = runInServer(a)
	a.Processes = func() app.Procs {
		server, relays, crashes := srv.Status(cfg.StateDir)
		return app.Procs{Server: server, Relays: relays, Cmd: cmdLine, Crashes: crashes}
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	a.StartScanner(ctx.Done())
	stopFlush := make(chan struct{})
	flushed := a.StartRenewFlusher(stopFlush)
	err = srv.Serve(ctx)
	close(stopFlush) // write renewals from gets before exiting
	<-flushed
	return err
}

func runServerStop([]string) error {
	cl, err := client()
	if err != nil {
		return err
	}
	pid, err := cl.Stop("stop")
	if errors.Is(err, daemon.ErrNoServer) {
		_, _ = fmt.Fprintln(out, "no griz server running")
		return nil
	}
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(out, "stopped griz server (pid %d)\n", pid)
	return nil
}

func runServerRestart([]string) error {
	cl, err := client()
	if err != nil {
		return err
	}
	if _, err := cl.Stop("restart"); err != nil && !errors.Is(err, daemon.ErrNoServer) {
		return err
	}
	return cl.RunCLI([]string{"server status"}, out, true)
}

func runServerStatus([]string) error {
	if srvApp != nil && srvApp.Processes != nil {
		_, _ = fmt.Fprintln(out, srvApp.Processes().Format(true))
		return nil
	}
	_, _ = fmt.Fprintln(out, app.Procs{Cmd: fmt.Sprintf("pid %d (griz server status)", os.Getpid())}.Format(false))
	return nil
}

func runStatus([]string) error {
	a, err := open()
	if err != nil {
		return err
	}
	if a.Processes == nil { // no server: compute the pin warnings here, and still report crashes
		a.Stderr = io.Discard // they are listed in the report
		if err := a.RefreshPinWarnings(); err != nil {
			return err
		}
		a.Processes = func() app.Procs {
			return app.Procs{Cmd: fmt.Sprintf("pid %d (griz status)", os.Getpid()), Crashes: daemon.CrashReport(a.Cfg.StateDir)}
		}
	}
	_, _ = fmt.Fprintln(out, a.Report())
	return nil
}

// exit ends a role started by name, reporting its error like main does.
func exit(err error) {
	if err != nil {
		_, _ = fmt.Fprintln(errOut, "griz:", err)
		os.Exit(1)
	}
	os.Exit(0)
}
