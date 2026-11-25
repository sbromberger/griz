package app

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"griz/internal/config"
	"griz/internal/embed"
	"griz/internal/index"
	"griz/internal/memory"
)

// slowModel takes 20 ms per text, fails on texts containing "boom", and
// counts the texts it embeds.
type slowModel struct{ n *atomic.Int64 }

func (slowModel) ID() string     { return "slow" }
func (slowModel) Name() string   { return "slow" }
func (slowModel) Dims() int      { return 2 }
func (slowModel) MaxTokens() int { return 512 }
func (m slowModel) Embed(t []string) ([][]float32, error) {
	time.Sleep(time.Duration(len(t)) * 20 * time.Millisecond)
	out := make([][]float32, len(t))
	for i, s := range t {
		if strings.Contains(s, "boom") {
			return nil, errors.New("boom")
		}
		out[i] = []float32{1, float32(len(s))}
	}
	m.n.Add(int64(len(t)))
	return out, nil
}

// slowApps opens n griz processes (Apps) on one data and state directory.
func slowApps(t *testing.T, n int, setup func(put func(id, body string))) ([]*App, *atomic.Int64) {
	old := ProbeModel
	t.Cleanup(func() { ProbeModel = old })
	count := &atomic.Int64{}
	ProbeModel = func(dir, _ string) (embed.Info, embed.Loader, error) {
		return embed.Info{ID: "slow", Name: "slow", Dims: 2, MaxTokens: 512},
			func() (embed.Model, error) { return slowModel{count}, nil }, nil
	}
	root := t.TempDir()
	cfg := config.Defaults()
	cfg.DataDir, cfg.StateDir, cfg.Search.Model = filepath.Join(root, "d"), filepath.Join(root, "s"), "fake"
	cfg.Store.NewNSGuard = false
	put := func(id, body string) {
		p := filepath.Join(cfg.DataDir, testName(id))
		os.MkdirAll(filepath.Dir(p), 0o755)
		os.WriteFile(p, (&memory.Memory{ID: id, Body: body}).Format(), 0o644)
	}
	setup(put)
	var apps []*App
	for i := 0; i < n; i++ {
		c := cfg
		a, err := Open(&c)
		if err != nil {
			t.Fatal(err)
		}
		a.Stderr = io.Discard
		t.Cleanup(a.WaitEmbeds)
		apps = append(apps, a)
	}
	return apps, count
}

func TestRecallSearchesCompleteVectors(t *testing.T) {
	apps, count := slowApps(t, 1, func(put func(id, body string)) {
		for i := 0; i < 40; i++ {
			put(fmt.Sprintf("bulk/m%02d", i), fmt.Sprintf("memory number %d", i))
		}
	})
	a := apps[0]
	// No vectors at all: the first recall embeds everything before answering, and says so.
	out, _, _ := a.Recall("memory", 0, false)
	if !strings.Contains(out, "re-embedded all 40 chunks") {
		t.Errorf("first recall should say it re-embedded: %q", out)
	}
	if n := missingN(a); n != 0 {
		t.Fatalf("%d chunks still missing after recall", n)
	}
	// store returns at once; the next recall waits for its embedding.
	a.SetDebug(true)
	a.TakeDebug()
	for i := 0; i < 5; i++ {
		mustStore(t, a, StoreArgs{ID: fmt.Sprintf("bulk/new%d", i), Body: str(fmt.Sprintf("fresh zebra %d", i)), Force: true}, "ok")
	}
	out, _, _ = a.Recall("zebra", 0, false)
	if n := missingN(a); n != 0 {
		t.Fatalf("recall searched with %d chunks missing", n)
	}
	if strings.Contains(out, "re-embedded all") {
		t.Errorf("a few new chunks are not a full re-embed: %q", out)
	}
	if d := strings.Join(a.TakeDebug(), "\n"); !strings.Contains(d, "griz debug: waited") || !strings.Contains(d, "searched 45 chunks") {
		t.Errorf("debug lines: %q", d)
	}
	if n := count.Load(); n != 45+2 { // 45 chunks, plus the two queries
		t.Errorf("embedded %d texts, want 47 (each chunk once)", n)
	}
	// The scan doesn't load an unloaded model.
	a.model.Unload()
	os.WriteFile(file(a, "bulk/hand"), []byte("hand edit\n"), 0o644)
	a.Scan()
	a.WaitEmbeds()
	if loaded, _ := a.model.Loaded(); loaded {
		t.Error("scan must not load the model")
	}
	if n := missingN(a); n != 1 {
		t.Errorf("hand edit should wait for the next recall: %d missing", n)
	}
}

func TestStoreEmbedFailureWarns(t *testing.T) {
	apps, _ := slowApps(t, 1, func(func(id, body string)) {})
	a := apps[0]
	a.Cfg.Store.DupCheck = false
	mustStore(t, a, StoreArgs{ID: "x/bad", Body: str("this will boom")}, "ok")
	a.WaitEmbeds()
	if !a.Repo.Exists("x/bad") {
		t.Fatal("the memory must be saved even though embedding failed")
	}
	if w := strings.Join(a.takeWarnings(), "\n"); !strings.Contains(w, "griz warning: embedding") {
		t.Errorf("warnings: %q", w)
	}
	if out, _, _ := a.Recall("boom", 0, false); !strings.Contains(out, "x/bad") {
		t.Errorf("keyword search should still find it: %q", out)
	}
}

func missingN(a *App) int {
	have, total := a.vectorCount()
	return total - have
}

func TestConcurrentRecallsEmbedOnce(t *testing.T) {
	apps, count := slowApps(t, 1, func(put func(id, body string)) {
		for i := 0; i < 24; i++ {
			put(fmt.Sprintf("bulk/m%02d", i), fmt.Sprintf("memory number %d", i))
		}
	})
	a := apps[0]
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); a.Recall("memory", 0, false) }()
	}
	// A store landing meanwhile is embedded too, once.
	mustStore(t, a, StoreArgs{ID: "bulk/late", Body: str("late arrival"), Force: true}, "ok")
	wg.Wait()
	a.WaitEmbeds()
	a.Recall("memory", 0, false)
	if n := missingN(a); n != 0 {
		t.Fatalf("%d missing", n)
	}
	if n := count.Load(); n != 25+5 { // 25 chunks once each, plus 5 queries
		t.Errorf("embedded %d texts, want 30: some chunk was embedded twice", n)
	}
}

func TestSavesNeverPrune(t *testing.T) {
	apps, _ := slowApps(t, 2, func(put func(id, body string)) { put("x/one", "first memory") })
	a, b := apps[0], apps[1]
	a.Recall("first", 0, false) // a embeds x/one and saves
	// b knows nothing of x/two; a stores it and embeds; b then saves its own vector.
	mustStore(t, a, StoreArgs{ID: "x/two", Body: str("second memory"), Force: true}, "ok")
	a.WaitEmbeds()
	b.vec.Put("slow", [16]byte{9}, []float32{1, 2})
	if err := b.saveVectors(); err != nil {
		t.Fatal(err)
	}
	a.vec.Reload()
	if n := missingN(a); n != 0 {
		t.Fatalf("a save by another process dropped %d of a's vectors", n)
	}
	// Pruning (reindex, gc) keeps only current chunks.
	if err := a.PruneVectors(); err != nil {
		t.Fatal(err)
	}
	if a.vec.Has("slow", [16]byte{9}) || missingN(a) != 0 {
		t.Error("prune should drop the unused vector and keep current ones")
	}
}

// diamondModel embeds text as [number of ♦, 0.1]. The keyword index ignores
// ♦, so tests can steer search by meaning separately from keywords.
type diamondModel struct{}

func (diamondModel) ID() string     { return "diamond" }
func (diamondModel) Name() string   { return "diamond" }
func (diamondModel) Dims() int      { return 2 }
func (diamondModel) MaxTokens() int { return 512 }
func (diamondModel) Embed(t []string) ([][]float32, error) {
	out := make([][]float32, len(t))
	for i, s := range t {
		out[i] = []float32{float32(strings.Count(s, "♦")), 0.1}
	}
	return out, nil
}

// A memory whose best keyword chunk and best meaning chunk differ still gets
// credit from both lists.
func TestMemoryLevelFusion(t *testing.T) {
	old := ProbeModel
	t.Cleanup(func() { ProbeModel = old })
	ProbeModel = func(dir, _ string) (embed.Info, embed.Loader, error) {
		return embed.Info{ID: "diamond", Name: "diamond", Dims: 2, MaxTokens: 512},
			func() (embed.Model, error) { return diamondModel{}, nil }, nil
	}
	root := t.TempDir()
	cfg := config.Defaults()
	cfg.DataDir, cfg.StateDir, cfg.Search.Model = filepath.Join(root, "d"), filepath.Join(root, "s"), "fake"
	cfg.Store.NewNSGuard, cfg.Store.DupCheck = false, false
	a, err := Open(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	a.Stderr = io.Discard
	a.maxWords = 3
	// x/long: chunk 1 is the best keyword chunk, chunk 2 the best meaning chunk.
	mustStore(t, a, StoreArgs{ID: "x/long", Body: str("zebra zebra zebra\n\n♦ quiet words")}, "ok")
	// x/both: one chunk, second in each list.
	mustStore(t, a, StoreArgs{ID: "x/both", Body: str("zebra ♦ ♦ plain")}, "ok")
	a.WaitEmbeds()
	rk := a.rank("zebra ♦", index.Filter{}, nil)
	if long, _ := a.Repo.Lookup("x/long"); len(rk.results) < 2 || rk.results[0].id != long {
		t.Fatalf("x/long should rank first (best in both lists): %+v", rk.results)
	}
	r := rk.results[0]
	if r.kwRank != 1 || r.kwSeq != 0 || r.vecRank != 1 || r.vecSeq != 1 {
		t.Errorf("x/long ranks: %+v", r)
	}
	out, _ := a.Explain("zebra ♦", "x/long")
	if !strings.Contains(out, "#1") || !strings.Contains(out, "chunk 2") {
		t.Errorf("explain: %s", out)
	}
}
