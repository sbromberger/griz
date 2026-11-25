package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"griz/internal/app"
	"griz/internal/config"
	"griz/internal/daemon"
)

// parse parses flags wherever they appear among args and returns the
// other arguments in order.
func parse(fs *flag.FlagSet, args []string) ([]string, error) {
	fs.SetOutput(io.Discard)
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		if args = fs.Args(); len(args) == 0 {
			return pos, nil
		}
		pos, args = append(pos, args[0]), args[1:]
	}
}

func usageErr(name string) error {
	c := find(name)
	return fmt.Errorf("usage: griz %s %s", c.name, c.args)
}

// ---- recall, get, store, delete ----

// noteTTY asks for bold ids when the output is a terminal.
func noteTTY(args []string) ([]string, error) {
	if st, err := os.Stdout.Stat(); err == nil && st.Mode()&os.ModeCharDevice != 0 && os.Getenv("NO_COLOR") == "" {
		args = append(args, "--bold")
	}
	return args, nil
}

var idInText = regexp.MustCompile(`\[([a-z0-9]+(?:-[a-z0-9]+)*-)([0-9a-f]{8})\]`)

// bold marks, in each id in s, the part of its hex that names it.
func bold(a *app.App, s string) string {
	return idInText.ReplaceAllStringFunc(s, func(m string) string {
		sub := idInText.FindStringSubmatch(m)
		n := a.ShortLen(sub[1] + sub[2])
		return "[" + sub[1] + "\x1b[1m" + sub[2][:n] + "\x1b[0m" + sub[2][n:] + "]"
	})
}

func runRecall(args []string) error {
	fs := flag.NewFlagSet("recall", flag.ContinueOnError)
	archived := fs.Bool("archived", false, "")
	pinned := fs.Bool("pinned", false, "")
	budget := fs.Int("budget", 0, "")
	tty := fs.Bool("bold", false, "")
	words, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(words) == 0 && !*pinned {
		return usageErr("recall")
	}
	a, err := open()
	if err != nil {
		return err
	}
	op := a.BeginOp(nil, "recall (cli)")
	var text string
	var ids []string
	if *pinned {
		text, ids, err = a.Pinned()
	} else {
		text, ids, err = a.RecallOp(op, strings.Join(words, " "), *budget, *archived)
	}
	a.EndOp(op, ids)
	if err != nil {
		return err
	}
	if *tty {
		text = bold(a, text)
	}
	fmt.Fprintln(out, text)
	return nil
}

func runGet(args []string) error {
	fs := flag.NewFlagSet("get", flag.ContinueOnError)
	tty := fs.Bool("bold", false, "")
	keys, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(keys) == 0 {
		return usageErr("get")
	}
	a, err := open()
	if err != nil {
		return err
	}
	if err := a.Sync(); err != nil {
		return err
	}
	text, _ := a.Get(keys)
	if *tty {
		text = bold(a, text)
	}
	fmt.Fprintln(out, text)
	return nil
}

// readStdin replaces a "-" given for --body or --append with stdin, which
// the server can't read.
func readStdin(args []string) ([]string, error) {
	args = append([]string(nil), args...)
	var in *string
	stdin := func() (string, error) {
		if in == nil {
			b, err := io.ReadAll(os.Stdin)
			if err != nil {
				return "", err
			}
			s := string(b)
			in = &s
		}
		return *in, nil
	}
	for i, a := range args {
		name, val, hasVal := strings.Cut(strings.TrimLeft(a, "-"), "=")
		if !strings.HasPrefix(a, "-") || (name != "body" && name != "append") {
			continue
		}
		switch {
		case hasVal && val == "-":
			s, err := stdin()
			if err != nil {
				return nil, err
			}
			args[i] = "--" + name + "=" + s
		case !hasVal && i+1 < len(args) && args[i+1] == "-":
			s, err := stdin()
			if err != nil {
				return nil, err
			}
			args[i+1] = s
		}
	}
	return args, nil
}

func runStore(args []string) error {
	fs := flag.NewFlagSet("store", flag.ContinueOnError)
	var s app.StoreArgs
	body, appendText := fs.String("body", "", ""), fs.String("append", "", "")
	old, nw := fs.String("old", "", ""), fs.String("new", "", "")
	tags, term, expires := fs.String("tags", "", ""), fs.String("term", "", ""), fs.String("expires", "", "")
	archived, pin := fs.Bool("archived", false, ""), fs.Bool("pin", false, "")
	fs.BoolVar(&s.Force, "force", false, "")
	fs.BoolVar(&s.NewNS, "new-ns", false, "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return usageErr("store")
	}
	s.ID = pos[0]
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "body":
			s.Body = body
		case "append":
			s.Append = appendText
		case "old":
			s.Old = old
		case "new":
			s.New = nw
		case "tags":
			s.Tags = []string{}
			for _, t := range strings.Split(*tags, ",") {
				if t = strings.TrimSpace(t); t != "" {
					s.Tags = append(s.Tags, t)
				}
			}
		case "term":
			s.Term = term
		case "expires":
			s.Expires = expires
		case "archived":
			s.Archived = archived
		case "pin":
			s.Pin = pin
		}
	})
	a, err := open()
	if err != nil {
		return err
	}
	op := a.BeginOp(nil, "store (cli)")
	reply, err := a.StoreOp(op, s)
	a.EndOp(op, nil)
	if err != nil {
		return err
	}
	if strings.HasPrefix(reply, "similar:") {
		reply += "\nnot saved; --force saves it anyway"
	}
	fmt.Fprintln(out, reply)
	return nil
}

func runDelete(args []string) error {
	if len(args) != 1 {
		return usageErr("delete")
	}
	a, err := open()
	if err != nil {
		return err
	}
	id, err := a.Resolve(args[0])
	if err != nil {
		return err
	}
	reply, err := a.Delete(id)
	if err != nil {
		return err
	}
	fmt.Fprintln(out, reply)
	return nil
}

// ---- memory ----

func runTerm(args []string) error {
	fs := flag.NewFlagSet("term", flag.ContinueOnError)
	expires := fs.String("expires", "", "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(pos) < 2 {
		return usageErr("memory term")
	}
	a, err := open()
	if err != nil {
		return err
	}
	term := pos[0]
	for _, key := range pos[1:] {
		id, err := a.Resolve(key)
		if err != nil {
			return err
		}
		s := app.StoreArgs{ID: id, Term: &term}
		fs.Visit(func(f *flag.Flag) { s.Expires = expires })
		reply, err := a.Store(s)
		if err != nil {
			return err
		}
		fmt.Fprintln(out, reply)
	}
	return nil
}

// edit runs $VISUAL or $EDITOR (vi by default) on path, in dir.
func edit(dir, path string) error {
	ed := os.Getenv("VISUAL")
	if ed == "" {
		ed = os.Getenv("EDITOR")
	}
	if ed == "" {
		ed = "vi"
	}
	cmd := exec.Command("sh", "-c", ed+` "$1"`, "sh", path) // $EDITOR may carry flags (code -w)
	cmd.Dir, cmd.Stdin, cmd.Stdout, cmd.Stderr = dir, os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func runMemoryEdit([]string) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if err := edit(cfg.DataDir, cfg.DataDir); err != nil {
		return err
	}
	fmt.Fprintf(out, "the griz server commits your changes once they have been unchanged for %s\n", cfg.Scan.Settle)
	return nil
}

// ---- config, env, debug ----

func runConfigEdit([]string) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(cfg.ConfigFile), 0o755); err != nil {
		return err
	}
	if err := edit(filepath.Dir(cfg.ConfigFile), cfg.ConfigFile); err != nil {
		return err
	}
	if _, err := config.Load(); err != nil {
		return fmt.Errorf("the config has a problem: %v", err)
	}
	fmt.Fprintln(out, "config ok; griz mcp restart puts it into effect")
	return nil
}

func runEnv([]string) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	exe, _ := os.Executable()
	conf := cfg.ConfigFile
	if !cfg.FileFound {
		conf += " (not found; all defaults)"
	}
	model := cfg.Search.Model
	if model == "" {
		model = "(none: keyword search only)"
	}
	for _, l := range [][2]string{
		{"binary", exe},
		{"config", conf},
		{"data", cfg.DataDir + " (the memories; a git repo)"},
		{"model", model},
		{"state", cfg.StateDir},
	} {
		fmt.Fprintf(out, "%-8s %s\n", l[0], l[1])
	}
	if ents, err := os.ReadDir(cfg.StateDir); err == nil {
		for _, e := range ents {
			fmt.Fprintf(out, "%-8s %s\n", "", filepath.Join(cfg.StateDir, e.Name()))
		}
	}
	fmt.Fprintf(out, "%-8s %s\n", "socket", daemon.SocketPath(cfg.StateDir))
	for _, v := range []string{"XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "VISUAL", "EDITOR", "NO_COLOR"} {
		if s, ok := os.LookupEnv(v); ok {
			fmt.Fprintf(out, "$%s=%s\n", v, s)
		}
	}
	return nil
}

func runDebug(on bool) error {
	a, err := open()
	if err != nil {
		return err
	}
	a.SetDebugAll(on)
	if !on {
		fmt.Fprintln(out, "debug mode off")
		return nil
	}
	n := len(a.Sessions())
	fmt.Fprintf(out, "debug mode on for %d connected %s, new ones, and griz commands, until griz debug off or the server exits\n",
		n, map[bool]string{true: "session", false: "sessions"}[n == 1])
	return nil
}

func runTimings([]string) error {
	if srvApp == nil {
		fmt.Fprintln(out, "no griz server running; timings live in the server")
		return nil
	}
	fmt.Fprintln(out, srvApp.Timings())
	return nil
}
