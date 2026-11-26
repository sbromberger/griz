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
	_, _ = fmt.Fprintf(out, "# config: %s\n# data:   %s\n# state:  %s\n", src, cfg.DataDir, cfg.StateDir)
	a, err := app.Open(cfg)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(out, "# search: %s\n\n%s", a.Status(), cfg)
	return nil
}

func runReindex() error {
	a, err := open()
	if err != nil {
		return err
	}
	if !a.HasModel() {
		_, _ = fmt.Fprintln(out, "search:", a.Status(), "- nothing to embed")
		return nil
	}
	n, err := a.Reembed(nil)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(out, "embedded %d chunks (%s)\n", n, a.Status())
	return nil
}

func runGC() error {
	a, err := open()
	if err != nil {
		return err
	}
	// gc is one of the two places unused vectors are dropped (reindex is the other).
	if err := a.PruneVectors(); err != nil {
		return err
	}
	_, _ = fmt.Fprintln(out, "dropped vectors no memory uses")
	return nil
}

func runRestore(args []string) error {
	if len(args) != 1 || strings.HasPrefix(args[0], "-") {
		return usageErr("memory restore")
	}
	a, err := open()
	if err != nil {
		return err
	}
	if err := a.Restore(args[0]); err != nil {
		return err
	}
	_, _ = fmt.Fprintln(out, "restored", args[0])
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
		return usageErr("memory explain")
	}
	a, err := open()
	if err != nil {
		return err
	}
	text, err := a.Explain(strings.Join(words, " "), *id)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintln(out, text)
	return nil
}

func runProfile(args []string) error {
	fs := flag.NewFlagSet("profile", flag.ContinueOnError)
	outFile := fs.String("out", "", "also write the raw profile here, for go tool pprof")
	seconds := fs.Int("seconds", 10, "how long to sample the server")
	reembed := fs.Bool("reembed", false, "profile a full re-embed instead")
	fs.SetOutput(errOut)
	if err := fs.Parse(args); err != nil {
		return err
	}
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	if fs.NArg() > 0 || (set["seconds"] && *reembed) || *seconds <= 0 {
		return usageErr("server profile")
	}
	a, err := open()
	if err != nil {
		return err
	}
	if !*reembed {
		text, err := app.Profile(time.Duration(*seconds)*time.Second, nil, *outFile)
		if err != nil {
			return err
		}
		_, _ = fmt.Fprintf(out, "sampled the server (pid %d) for %ds\n%s\n", os.Getpid(), *seconds, text)
		if *outFile != "" {
			_, _ = fmt.Fprintln(out, "raw profile:", *outFile)
		}
		return nil
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
	_, _ = fmt.Fprintf(out, "re-embedded %d chunks with %s in %s (%.0f ms wall per chunk)\n", n, a.Status(), wall.Round(time.Millisecond),
		float64(wall.Milliseconds())/float64(max(n, 1)))
	_, _ = fmt.Fprintf(out, "peak resident %.0f MB for this process (pid %d) since it started, including the model's mapped weights\n%s\n",
		float64(app.PeakRSS())/(1<<20), os.Getpid(), text)
	if *outFile != "" {
		_, _ = fmt.Fprintln(out, "raw profile:", *outFile)
	}
	return nil
}
