package app

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"griz/internal/config"
	"griz/internal/embed"
	"griz/internal/memory"
)

// peakModel records the most texts it was embedding at the same moment.
type peakModel struct{ active, peak *atomic.Int64 }

func (peakModel) ID() string     { return "peak" }
func (peakModel) Name() string   { return "peak" }
func (peakModel) Dims() int      { return 2 }
func (peakModel) MaxTokens() int { return 512 }
func (m peakModel) Embed(t []string) ([][]float32, error) {
	n := m.active.Add(1)
	for p := m.peak.Load(); n > p && !m.peak.CompareAndSwap(p, n); p = m.peak.Load() {
	}
	time.Sleep(10 * time.Millisecond)
	m.active.Add(-1)
	out := make([][]float32, len(t))
	for i, s := range t {
		out[i] = []float32{1, float32(len(s))}
	}
	return out, nil
}

func peakApp(t *testing.T, workers int) (*App, *atomic.Int64) {
	old := ProbeModel
	t.Cleanup(func() { ProbeModel = old })
	m := peakModel{&atomic.Int64{}, &atomic.Int64{}}
	ProbeModel = func(dir, _ string) (embed.Info, embed.Loader, error) {
		return embed.Info{ID: "peak", Name: "peak", Dims: 2, MaxTokens: 512},
			func() (embed.Model, error) { return m, nil }, nil
	}
	root := t.TempDir()
	cfg := config.Defaults()
	cfg.DataDir, cfg.StateDir, cfg.Search.Model = filepath.Join(root, "d"), filepath.Join(root, "s"), "fake"
	cfg.Search.EmbedWorkers = workers
	for i := 0; i < 30; i++ {
		p := filepath.Join(cfg.DataDir, testName(fmt.Sprintf("bulk/m%02d", i)))
		os.MkdirAll(filepath.Dir(p), 0o755)
		os.WriteFile(p, (&memory.Memory{ID: fmt.Sprintf("bulk/m%02d", i), Body: fmt.Sprintf("memory %d", i)}).Format(), 0o644)
	}
	a, err := Open(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	a.Stderr = io.Discard
	t.Cleanup(a.WaitEmbeds)
	return a, m.peak
}

func TestEmbedWorkersCapsConcurrency(t *testing.T) {
	a, peak := peakApp(t, 3)
	if _, err := a.Reembed(nil); err != nil {
		t.Fatal(err)
	}
	if p := peak.Load(); p != 3 {
		t.Errorf("embed_workers = 3: at most 3 chunks at once, and 30 chunks should reach 3; peak was %d", p)
	}
	if !strings.Contains(a.Report(), "embedding workers: 3 at once") {
		t.Errorf("status should show the worker count")
	}
}

func TestEmbedWorkersAboveCPUsIsANote(t *testing.T) {
	over := runtime.GOMAXPROCS(0) + 4
	a, peak := peakApp(t, over)
	if _, err := a.Reembed(nil); err != nil {
		t.Fatal(err)
	}
	if peak.Load() <= int64(runtime.GOMAXPROCS(0)) {
		t.Errorf("a setting above GOMAXPROCS is allowed and used; peak was %d", peak.Load())
	}
	r := a.Report()
	if !strings.Contains(r, fmt.Sprintf("embedding workers: %d at once", over)) || !strings.Contains(r, "note: more than") {
		t.Errorf("status should note the setting: %s", r)
	}
	if w := strings.SplitN(strings.SplitN(r, "warnings:", 2)[1], "\n", 2)[0]; strings.TrimSpace(w) != "none" {
		t.Errorf("above GOMAXPROCS is a note, not a warning: warnings:%s", w)
	}
}

func TestEmbedWorkersDefault(t *testing.T) {
	if w := config.Defaults().Search.Workers(); w != runtime.GOMAXPROCS(0) {
		t.Errorf("default embed_workers 0 means GOMAXPROCS (%d), got %d", runtime.GOMAXPROCS(0), w)
	}
}
