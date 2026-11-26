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
		return usageErr("memory recall")
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
	_, _ = fmt.Fprintln(out, text)
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
		return usageErr("memory get")
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
	_, _ = fmt.Fprintln(out, text)
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
	fs.BoolVar(&s.All, "all", false, "")
	fs.BoolVar(&s.Preview, "preview", false, "")
	count := fs.Int("count", 0, "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return usageErr("memory store")
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
			for t := range strings.SplitSeq(*tags, ",") {
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
		case "count":
			s.Count = count
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
	_, _ = fmt.Fprintln(out, reply)
	return nil
}

func runDelete(args []string) error {
	if len(args) == 0 {
		return usageErr("memory delete")
	}
	a, err := open()
	if err != nil {
		return err
	}
	ids, err := resolveAll(a, args)
	if err != nil {
		return err
	}
	for _, id := range ids {
		reply, err := a.Delete(id)
		if err != nil {
			return err
		}
		_, _ = fmt.Fprintln(out, reply)
	}
	return nil
}

// resolveAll resolves every key before anything changes, so a typo in the
// third key doesn't leave the first two done.
func resolveAll(a *app.App, keys []string) ([]string, error) {
	ids := make([]string, len(keys))
	for i, k := range keys {
		if strings.HasPrefix(k, "-") {
			return nil, fmt.Errorf("unknown flag %s", k)
		}
		id, err := a.Resolve(k)
		if err != nil {
			return nil, err
		}
		ids[i] = id
	}
	return ids, nil
}

// storeEach runs one store change on each memory named in args, after
// resolving them all. With lead, the first argument is the value the change
// takes (a term, a date) rather than a memory.
func storeEach(name string, args []string, lead bool, change func(value, id string) app.StoreArgs) error {
	value := ""
	if lead && len(args) > 0 {
		value, args = args[0], args[1:]
	}
	if len(args) == 0 {
		return usageErr(name)
	}
	a, err := open()
	if err != nil {
		return err
	}
	ids, err := resolveAll(a, args)
	if err != nil {
		return err
	}
	for _, id := range ids {
		reply, err := a.Store(change(value, id))
		if err != nil {
			return err
		}
		_, _ = fmt.Fprintln(out, reply)
	}
	return nil
}

func runPin(args []string, pin bool) error {
	name := map[bool]string{true: "memory pin", false: "memory unpin"}[pin]
	return storeEach(name, args, false, func(_, id string) app.StoreArgs { return app.StoreArgs{ID: id, Pin: &pin} })
}

func runArchive(args []string, archive bool) error {
	name := map[bool]string{true: "memory archive", false: "memory unarchive"}[archive]
	return storeEach(name, args, false, func(_, id string) app.StoreArgs { return app.StoreArgs{ID: id, Archived: &archive} })
}

func runTerm(args []string) error {
	return storeEach("memory term", args, true, func(t, id string) app.StoreArgs { return app.StoreArgs{ID: id, Term: &t} })
}

func runExpire(args []string) error {
	return storeEach("memory expire", args, true, func(d, id string) app.StoreArgs { return app.StoreArgs{ID: id, Expires: &d} })
}

func runRenew(args []string) error {
	return storeEach("memory renew", args, false, func(_, id string) app.StoreArgs { return app.StoreArgs{ID: id, Renew: true} })
}

// ---- memory ----

func runMove(args []string) error {
	fs := flag.NewFlagSet("move", flag.ContinueOnError)
	newNS := fs.Bool("new-ns", false, "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 2 {
		return usageErr("memory move")
	}
	a, err := open()
	if err != nil {
		return err
	}
	reply, err := a.Store(app.StoreArgs{ID: pos[0], Move: &pos[1], NewNS: *newNS})
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintln(out, reply)
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
	_, _ = fmt.Fprintf(out, "the griz server commits your changes once they have been unchanged for %s\n", cfg.Scan.Settle)
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
	_, _ = fmt.Fprintln(out, "config ok; griz server restart puts it into effect")
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
		_, _ = fmt.Fprintf(out, "%-8s %s\n", l[0], l[1])
	}
	if ents, err := os.ReadDir(cfg.StateDir); err == nil {
		for _, e := range ents {
			_, _ = fmt.Fprintf(out, "%-8s %s\n", "", filepath.Join(cfg.StateDir, e.Name()))
		}
	}
	_, _ = fmt.Fprintf(out, "%-8s %s\n", "socket", daemon.SocketPath(cfg.StateDir))
	for _, v := range []string{"XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "VISUAL", "EDITOR", "NO_COLOR"} {
		if s, ok := os.LookupEnv(v); ok {
			_, _ = fmt.Fprintf(out, "$%s=%s\n", v, s)
		}
	}
	return nil
}

func runDebug(args []string) error {
	if len(args) != 1 || (args[0] != "on" && args[0] != "off") {
		return usageErr("server debug")
	}
	on := args[0] == "on"
	a, err := open()
	if err != nil {
		return err
	}
	a.SetDebugAll(on)
	if !on {
		_, _ = fmt.Fprintln(out, "debug mode off")
		return nil
	}
	n := len(a.Sessions())
	_, _ = fmt.Fprintf(out, "debug mode on for %d connected %s, new ones, and griz commands, until griz server debug off or the server exits\n",
		n, map[bool]string{true: "session", false: "sessions"}[n == 1])
	return nil
}

func runTimings([]string) error {
	if srvApp == nil {
		_, _ = fmt.Fprintln(out, "no griz server running; timings live in the server")
		return nil
	}
	_, _ = fmt.Fprintln(out, srvApp.Timings())
	return nil
}
