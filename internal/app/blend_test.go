package app

import (
	"io"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"griz/internal/config"
	"griz/internal/embed"
)

// countModel embeds every text the same way and counts tokens like a
// WordPiece vocabulary that knows plain words but splits dotted numbers into
// number, dot, number, ...
type countModel struct{}

func (countModel) ID() string     { return "count" }
func (countModel) Name() string   { return "count" }
func (countModel) Dims() int      { return 2 }
func (countModel) MaxTokens() int { return 512 }
func (countModel) Embed(t []string) ([][]float32, error) {
	out := make([][]float32, len(t))
	for i := range t {
		out[i] = []float32{1, 1}
	}
	return out, nil
}
func (countModel) CountTokens(s string) int { return 2*strings.Count(s, ".") + 1 }

func TestKeywordWeight(t *testing.T) {
	old := ProbeModel
	t.Cleanup(func() { ProbeModel = old })
	ProbeModel = func(dir, _ string) (embed.Info, embed.Loader, error) {
		return embed.Info{ID: "count", Name: "count", Dims: 2, MaxTokens: 512},
			func() (embed.Model, error) { return countModel{}, nil }, nil
	}
	root := t.TempDir()
	cfg := config.Defaults()
	cfg.DataDir, cfg.StateDir, cfg.Search.Model = filepath.Join(root, "d"), filepath.Join(root, "s"), "fake"
	a, err := Open(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	a.Stderr = io.Discard
	for q, want := range map[string]float64{
		"what is my wife's name?":   0,         // plain words: meaning decides
		"who is it?":                0,         // nothing but stopwords
		"10.1.2.3":                  1 - 1.0/7, // one word, seven tokens
		"what is at 10.1.2.3?":      1 - 1.0/7, // stopwords don't dilute it
		"which host is on 10.1.2.3": 1 - 2.0/8, // a mixed query: both count
	} {
		if got := a.keywordWeight(q); math.Abs(got-want) > 1e-9 {
			t.Errorf("%q: %.3f, want %.3f", q, got, want)
		}
	}
}
