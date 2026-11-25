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
	"runtime"
	"runtime/debug"
	"slices"
	"sort"
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
	// srvApp is the server's App while a command runs inside the server.
	srvApp *app.App
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
	hidden            bool
	// prep, if set, runs here before the command is sent to the server
	// (or run locally): reading stdin, asking for confirmation, noting a TTY.
	prep func(args []string) ([]string, error)
	run  func(args []string) error
}

// groups take a subcommand: griz memory term ..., griz debug on.
var groups = map[string]string{
	"mcp":    "the MCP server",
	"memory": "housekeeping for memories",
	"config": "settings",
	"debug":  "what griz is doing and how long it takes",
}

var commands []*command

func init() {
	commands = []*command{
		{name: "help", args: "[command]", short: "show usage, or details for one command", run: runHelp},
		{name: "version", short: "print griz's version", run: runVersion},
		{name: "env", short: "show every file and directory griz uses", run: runEnv},
		{name: "status", short: "memories, model, embedding, scan, sizes, warnings, changed settings", where: either, run: runStatus},

		{name: "recall", args: "<query> [--archived] [--pinned] [--budget n]", where: either, prep: noteTTY, run: runRecall,
			short: "search memories, as Claude does",
			long: `--archived also searches archived memories; --pinned lists every pinned memory
(no query needed); --budget caps the reply in tokens. Ids show in full, with
the part that is enough to name the memory in bold.`},
		{name: "get", args: "<id>...", where: either, prep: noteTTY, run: runGet, short: "print memories in full",
			long: idHelp},
		{name: "store", args: "<id> [--body text|-] [--append text|-] [--old text --new text] [--tags a,b] [--term t] [--expires when] [--archived[=false]] [--pin[=false]] [--force] [--new-ns]",
			where: inSrv, prep: readStdin, run: runStore, short: "save a new memory, or change one",
			long: `For a new memory give a path (folder/name) and --body; griz assigns the id.
For an existing one give its id (or a key, below) and one of: --body to
replace the text, --old and --new to edit in place, --append for a dated
entry; or only --tags, --term, --expires, --archived or --pin. "-" reads the text
from stdin. --term is short, medium, long or infinite; --expires is a date,
a duration (30d, 6m) or none. --force saves even if a similar memory exists;
--new-ns allows a new top-level folder.

` + idHelp},
		{name: "delete", args: "<id>", where: inSrv, run: runDelete, short: "delete a memory (griz memory restore brings it back)", long: idHelp},

		{name: "mcp", short: "MCP over stdio, relayed to the shared server (started if needed); what Claude runs", run: func([]string) error { return runRelay() }},
		{name: "mcp stop", short: "stop the server; sessions reconnect to a new one on their next call", run: runMCPStop},
		{name: "mcp restart", short: "stop the server and start a new one (after a config change)", run: runMCPRestart},
		{name: "mcp status", short: "the server's pid, uptime and sessions", where: either, run: runMCPStatus},
		{name: "serve", hidden: true, short: "run the shared server (griz mcp starts it)", run: func([]string) error { return runServe() }},

		{name: "memory edit", short: "open $EDITOR in the memory directory (changes are committed once they settle)", run: runMemoryEdit},
		{name: "memory stats", short: "counts of memories, and a summary of the usage log", where: either, run: runStats},
		{name: "memory lint", short: "report problems and things worth a look", where: either, run: runLint},
		{name: "memory term", args: "<short|medium|long|infinite> <id>... [--expires date|duration|none]", where: inSrv, run: runTerm,
			short: "change memories' lifetimes",
			long:  "The new term's clock starts now. --expires sets the date outright instead.\n\n" + idHelp},
		{name: "memory archive", args: "<id>...", where: inSrv, run: func(a []string) error { return runArchive(a, true) },
			short: "archive memories (found by recall --archived, loaded only on request)", long: idHelp},
		{name: "memory unarchive", args: "<id>... [--expires date|duration|none]", where: inSrv, run: func(a []string) error { return runArchive(a, false) },
			short: "make archived memories current again", long: "An expired memory needs --expires: a new date, or none.\n\n" + idHelp},
		{name: "memory restore", args: "<id> [--expires date|duration]", where: inSrv, run: runRestore, short: "bring back a deleted memory"},
		{name: "memory purge", args: "<id>... [--yes]", where: inSrv, prep: confirmPurge, run: runPurge,
			short: "erase memories from git history for good (asks first)"},
		{name: "memory gc", where: inSrv, run: func([]string) error { return runGC() }, short: "archive expired memories now and drop unused vectors"},

		{name: "reindex", where: inSrv, run: func([]string) error { return runReindex() }, short: "recompute every embedding and drop unused vectors"},

		{name: "config", hidden: true, run: func([]string) error { return runConfig() }},
		{name: "config show", short: "print the config file, directories and effective settings", run: func([]string) error { return runConfig() }},
		{name: "config edit", short: "open the config file in $EDITOR, then check it", run: runConfigEdit},

		{name: "debug on", where: inSrv, run: func([]string) error { return runDebug(true) },
			short: "debug mode for every session and command, until the server exits",
			long:  "Claude's replies gain a note of what happened; griz commands print where their time went."},
		{name: "debug off", where: inSrv, run: func([]string) error { return runDebug(false) }, short: "end debug mode everywhere"},
		{name: "debug explain", args: "<query> [--id id]", where: either, run: runExplain,
			short: "show how recall ranks results for a query (and where one memory landed)"},
		{name: "debug profile", args: "[--out file]", where: inSrv, run: runProfile, short: "CPU-profile a full re-embed and list where the time goes"},
		{name: "debug timings", where: either, run: runTimings, short: "typical and slowest times of recent operations in the server"},
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
		if !strings.HasPrefix(args[1], "-") && args[0] != "mcp" {
			return nil, nil, fmt.Errorf("unknown command \"%s %s\"\n\n%s", args[0], args[1], groupUsage(args[0]))
		}
	}
	if c := find(args[0]); c != nil {
		return c, args[1:], nil
	}
	if _, ok := groups[args[0]]; ok {
		return nil, nil, fmt.Errorf("griz %s needs a command\n\n%s", args[0], groupUsage(args[0]))
	}
	return nil, nil, fmt.Errorf("unknown command %q (griz help lists them)", args[0])
}

func main() {
	server.Version = version
	args := os.Args[1:]
	if len(args) == 0 {
		fmt.Fprint(errOut, usage())
		os.Exit(2)
	}
	switch args[0] {
	case "--help", "-h":
		args[0] = "help"
	case "--version", "-v":
		args[0] = "version"
	}
	c, rest, err := lookup(args)
	if err != nil {
		fmt.Fprintln(errOut, "griz:", err)
		os.Exit(2)
	}
	if c.name != "mcp" && (slices.Contains(rest, "--help") || slices.Contains(rest, "-h")) {
		fmt.Fprint(out, help(c))
		return
	}
	if err := dispatch(c, rest); err != nil {
		fmt.Fprintln(errOut, "griz:", err)
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
func runInServer(a *app.App) func(args []string, w io.Writer) error {
	return func(args []string, w io.Writer) error {
		c := find(args[0])
		if c == nil || c.where == local {
			return fmt.Errorf("griz %s doesn't run in the server", args[0])
		}
		out, errOut, srvApp = w, w, a
		defer func() { out, errOut, srvApp = os.Stdout, os.Stderr, nil }()
		start := time.Now()
		err := c.run(args[1:])
		if a.Debugging() {
			for _, l := range a.TakeDebug() {
				fmt.Fprintln(w, l)
			}
			fmt.Fprintf(w, "griz debug: griz %s took %s in the server (pid %d)\n", c.name, time.Since(start).Round(100*time.Microsecond), os.Getpid())
		}
		return err
	}
}

// ---- help ----

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
		if !c.hidden && !strings.Contains(c.name, " ") {
			if _, grp := groups[c.name]; !grp {
				line(c)
			}
		}
	}
	names := make([]string, 0, len(groups))
	for g := range groups {
		names = append(names, g)
	}
	sort.Strings(names)
	for _, g := range names {
		fmt.Fprintf(&b, "\n%s: %s\n", g, groups[g])
		if c := find(g); c != nil && !c.hidden {
			line(c)
		}
		for _, c := range commands {
			if !c.hidden && strings.HasPrefix(c.name, g+" ") {
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
		if !c.hidden && (c.name == g || strings.HasPrefix(c.name, g+" ")) {
			fmt.Fprintf(&b, "  griz %-16s %s\n", c.name, c.short)
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
		fmt.Fprint(out, usage())
		return nil
	}
	if _, ok := groups[args[0]]; ok && len(args) == 1 {
		fmt.Fprint(out, groupUsage(args[0]))
		return nil
	}
	c, _, err := lookup(args)
	if err != nil {
		return err
	}
	fmt.Fprint(out, help(c))
	return nil
}

// runVersion prints this griz's version, then the running server's and
// its sessions'.
func runVersion([]string) error {
	exe, _ := os.Executable()
	fmt.Fprintf(out, "griz %s (%s, %s/%s) %s\n", version, runtime.Version(), runtime.GOOS, runtime.GOARCH, exe)
	cl, err := client()
	if err != nil {
		return err
	}
	if err := cl.RunCLI([]string{"mcp status"}, out, false); errors.Is(err, daemon.ErrNoServer) {
		fmt.Fprintln(out, "no griz server running")
	} else if err != nil {
		fmt.Fprintln(out, "server:", err)
	}
	return nil
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
	if gone, err := a.GC(); err != nil {
		a.Warn(app.WarnExpiry, "archiving expired memories: %v", err)
	} else if len(gone) > 0 {
		a.Info("archived (expired): %s", strings.Join(gone, ", "))
	}
	if err := a.RefreshPinWarnings(); err != nil {
		return err
	}
	srv.A = a
	srv.RunCLI = runInServer(a)
	a.StatusExtra = func() []string { return srv.Status(cfg.StateDir) }
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	a.StartScanner(ctx.Done())
	return srv.Serve(ctx)
}

func runMCPStop([]string) error {
	cl, err := client()
	if err != nil {
		return err
	}
	pid, err := cl.Stop("stop")
	if errors.Is(err, daemon.ErrNoServer) {
		fmt.Fprintln(out, "no griz server running")
		return nil
	}
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "stopped griz server (pid %d)\n", pid)
	return nil
}

func runMCPRestart([]string) error {
	cl, err := client()
	if err != nil {
		return err
	}
	if _, err := cl.Stop("restart"); err != nil && !errors.Is(err, daemon.ErrNoServer) {
		return err
	}
	return cl.RunCLI([]string{"mcp status"}, out, true)
}

func runMCPStatus([]string) error {
	if srvApp == nil || srvApp.StatusExtra == nil {
		fmt.Fprintln(out, "no griz server running")
		return nil
	}
	for _, l := range srvApp.StatusExtra() {
		fmt.Fprintln(out, l)
	}
	return nil
}

func runStatus([]string) error {
	a, err := open()
	if err != nil {
		return err
	}
	if a.StatusExtra == nil { // no server: compute the pin warnings here, and still report crashes
		fmt.Fprintln(out, "server: not running")
		a.Stderr = io.Discard // they are listed in the report
		if err := a.RefreshPinWarnings(); err != nil {
			return err
		}
		a.StatusExtra = func() []string {
			if c := daemon.CrashReport(a.Cfg.StateDir); c != "" {
				return []string{c}
			}
			return nil
		}
	}
	fmt.Fprintln(out, a.Report())
	return nil
}
