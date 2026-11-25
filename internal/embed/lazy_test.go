package embed

import (
	"errors"
	"testing"
	"time"
)

type fake struct{ closed *int }

func (fake) ID() string     { return "f" }
func (fake) Name() string   { return "f" }
func (fake) Dims() int      { return 1 }
func (fake) MaxTokens() int { return 10 }
func (f fake) Close() error { *f.closed++; return nil }
func (fake) Embed(t []string) ([][]float32, error) {
	return make([][]float32, len(t)), nil
}

func TestLazyLoadUnload(t *testing.T) {
	closed := 0
	l := NewLazy(Info{ID: "f"}, func() (Model, error) { return fake{&closed}, nil }, 30*time.Millisecond)
	if ok, n := l.Loaded(); ok || n != 0 {
		t.Fatal("must not load before use")
	}
	l.Embed([]string{"x"})
	if ok, _ := l.Loaded(); !ok {
		t.Fatal("should be loaded")
	}
	time.Sleep(80 * time.Millisecond)
	if ok, _ := l.Loaded(); ok || closed != 1 {
		t.Fatalf("should unload when idle (closed=%d)", closed)
	}
	l.Embed([]string{"x"})
	if _, n := l.Loaded(); n != 2 {
		t.Fatalf("should reload: %d loads", n)
	}
}

func TestLazyFailureIsFinal(t *testing.T) {
	calls := 0
	l := NewLazy(Info{}, func() (Model, error) { calls++; return nil, errors.New("boom") }, 0)
	l.Embed(nil)
	l.Embed(nil)
	if calls != 1 || l.Failed() == nil {
		t.Fatalf("calls=%d", calls)
	}
}
