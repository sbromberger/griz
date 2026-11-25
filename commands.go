package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"griz/internal/app"
	"griz/internal/config"
)

func runConfig() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	src := cfg.ConfigFile
	if !cfg.FileFound {
		src += " (not found; all defaults)"
	}
	fmt.Fprintf(out, "# config: %s\n# data:   %s\n# state:  %s\n", src, cfg.DataDir, cfg.StateDir)
	a, err := app.Open(cfg)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "# search: %s\n\n%s", a.Status(), cfg)
	return nil
}

func runReindex() error {
	a, err := open()
	if err != nil {
		return err
	}
	if !a.HasModel() {
		fmt.Fprintln(out, "search:", a.Status(), "- nothing to embed")
		return nil
	}
	n, err := a.Reembed(nil)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "embedded %d chunks (%s)\n", n, a.Status())
	return nil
}

func runGC() error {
	a, err := open()
	if err != nil {
		return err
	}
	gone, err := a.GC()
	if err != nil {
		return err
	}
	// gc is one of the two places unused vectors are dropped (reindex is the other).
	if err := a.PruneVectors(); err != nil {
		return err
	}
	if len(gone) == 0 {
		fmt.Fprintln(out, "nothing newly expired")
	} else {
		fmt.Fprintln(out, "archived (expired):", strings.Join(gone, ", "))
	}
	return nil
}

func runRestore(args []string) error {
	fs := flag.NewFlagSet("restore", flag.ContinueOnError)
	exp := fs.String("expires", "", "new expiry date or duration (default: none)")
	id, rest := firstArg(args)
	fs.SetOutput(errOut)
	if err := fs.Parse(rest); err != nil {
		return err
	}
	if id == "" || fs.NArg() > 0 {
		return usageErr("memory restore")
	}
	a, err := open()
	if err != nil {
		return err
	}
	if err := a.Restore(id, *exp); err != nil {
		return err
	}
	fmt.Fprintln(out, "restored", id)
	return nil
}

// firstArg splits off a leading positional argument so flags may follow it.
func firstArg(args []string) (string, []string) {
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		return args[0], args[1:]
	}
	return "", args
}

func runArchive(args []string, archive bool) error {
	var ids []string
	expires := ""
	for i := 0; i < len(args); i++ {
		switch {
		case args[i] == "--expires" && i+1 < len(args):
			expires = args[i+1]
			i++
		case strings.HasPrefix(args[i], "--expires="):
			expires = strings.TrimPrefix(args[i], "--expires=")
		case strings.HasPrefix(args[i], "-"):
			return fmt.Errorf("unknown flag %s", args[i])
		default:
			ids = append(ids, args[i])
		}
	}
	if len(ids) == 0 || (archive && expires != "") {
		if archive {
			return usageErr("memory archive")
		}
		return usageErr("memory unarchive")
	}
	a, err := open()
	if err != nil {
		return err
	}
	for _, key := range ids {
		id, err := a.Resolve(key)
		if err != nil {
			return err
		}
		if archive {
			t := true
			_, err = a.Store(app.StoreArgs{ID: id, Archived: &t})
		} else {
			err = a.Unarchive(id, expires)
		}
		if err != nil {
			return err
		}
		fmt.Fprintln(out, "ok", a.Label(id))
	}
	return nil
}

func runExplain(args []string) error {
	fs := flag.NewFlagSet("explain", flag.ContinueOnError)
	id := fs.String("id", "", "also show where this memory landed and why")
	var words []string
	for len(args) > 0 { // allow flags before or after the query words
		fs.SetOutput(errOut)
		if err := fs.Parse(args); err != nil {
			return err
		}
		args = fs.Args()
		if len(args) > 0 {
			words = append(words, args[0])
			args = args[1:]
		}
	}
	if len(words) == 0 {
		return usageErr("debug explain")
	}
	a, err := open()
	if err != nil {
		return err
	}
	text, err := a.Explain(strings.Join(words, " "), *id)
	if err != nil {
		return err
	}
	fmt.Fprintln(out, text)
	return nil
}

func runProfile(args []string) error {
	fs := flag.NewFlagSet("profile", flag.ContinueOnError)
	outFile := fs.String("out", "", "also write the raw profile here, for go tool pprof")
	fs.SetOutput(errOut)
	if err := fs.Parse(args); err != nil {
		return err
	}
	a, err := open()
	if err != nil {
		return err
	}
	if !a.HasModel() {
		return fmt.Errorf("search: %s - nothing to profile", a.Status())
	}
	var n int
	var rerr error
	start := time.Now()
	text, err := app.Profile(0, func() { n, rerr = a.Reembed(nil) }, *outFile)
	if err != nil {
		return err
	}
	if rerr != nil {
		return rerr
	}
	wall := time.Since(start)
	fmt.Fprintf(out, "re-embedded %d chunks with %s in %s (%.0f ms wall per chunk)\n", n, a.Status(), wall.Round(time.Millisecond),
		float64(wall.Milliseconds())/float64(max(n, 1)))
	fmt.Fprintf(out, "peak resident %.0f MB for this process (pid %d) since it started, including the model's mapped weights\n%s\n",
		float64(app.PeakRSS())/(1<<20), os.Getpid(), text)
	if *outFile != "" {
		fmt.Fprintln(out, "raw profile:", *outFile)
	}
	return nil
}
