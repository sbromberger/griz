package embed

import (
	"io"
	"sync"
	"time"
)

// Info describes a model without loading it.
type Info struct {
	ID        string
	Name      string
	Dims      int
	MaxTokens int
	Bytes     int64 // size of the weights file
}

// Loader loads the model for real (maps weights, builds the tokenizer).
type Loader func() (Model, error)

// Lazy holds a model only while it is in use: it loads on the first Embed
// and unloads after idle time with no use. A failed load is final for the
// life of the process. Embeds hold a read lock (so several run at once) and
// loading and unloading the write lock, so a call never sees a half-loaded
// or unloaded model.
type Lazy struct {
	info  Info
	load  Loader
	after time.Duration // 0 = never unload

	mu     sync.RWMutex
	m      Model
	failed error
	timer  *time.Timer
	loads  int
	took   time.Duration // how long the last load took

	lastMu sync.Mutex
	last   time.Time

	// onLoad and OnUnload, if set, are told when the model loads (with how
	// long it took) and when it is released for being idle.
	onLoad   func(took time.Duration)
	OnUnload func(idle time.Duration)
}

func NewLazy(info Info, load Loader, unloadAfter time.Duration) *Lazy {
	return &Lazy{info: info, load: load, after: unloadAfter}
}

func (l *Lazy) ID() string                 { return l.info.ID }
func (l *Lazy) Name() string               { return l.info.Name }
func (l *Lazy) Dims() int                  { return l.info.Dims }
func (l *Lazy) MaxTokens() int             { return l.info.MaxTokens }
func (l *Lazy) Bytes() int64               { return l.info.Bytes }
func (l *Lazy) UnloadAfter() time.Duration { return l.after }

// UnloadAt is when an idle loaded model will be released (zero if it isn't
// loaded or never unloads).
func (l *Lazy) UnloadAt() time.Time {
	l.mu.RLock()
	loaded := l.m != nil
	l.mu.RUnlock()
	if !loaded || l.after <= 0 {
		return time.Time{}
	}
	l.lastMu.Lock()
	defer l.lastMu.Unlock()
	return l.last.Add(l.after)
}

// LoadTook is how long the last load took.
func (l *Lazy) LoadTook() time.Duration {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.took
}

// Failed returns the load error, if loading has failed.
func (l *Lazy) Failed() error {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.failed
}

// Loaded reports whether the model is in memory, and how many times it has been loaded.
func (l *Lazy) Loaded() (bool, int) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.m != nil, l.loads
}

// use runs f with the loaded model (loading it if needed) under the read lock.
func (l *Lazy) use(f func(Model) error) error {
	for {
		l.mu.RLock()
		if l.m != nil {
			l.lastMu.Lock()
			l.last = time.Now()
			l.lastMu.Unlock()
			err := f(l.m)
			l.mu.RUnlock()
			return err
		}
		failed := l.failed
		l.mu.RUnlock()
		if failed != nil {
			return failed
		}
		l.mu.Lock()
		if l.m == nil && l.failed == nil {
			t0 := time.Now()
			m, err := l.load()
			if err != nil {
				l.failed = err
			} else {
				l.m = m
				l.loads++
				l.took = time.Since(t0)
				l.lastMu.Lock()
				l.last = time.Now()
				l.lastMu.Unlock()
				l.arm()
				if l.onLoad != nil {
					l.onLoad(time.Since(t0))
				}
			}
		}
		l.mu.Unlock()
	}
}

func (l *Lazy) Embed(texts []string) ([][]float32, error) {
	var out [][]float32
	err := l.use(func(m Model) (err error) {
		out, err = m.Embed(texts)
		return err
	})
	return out, err
}

// counter is a model that can say how many tokens a text becomes.
type counter interface {
	CountTokens(text string) int
}

// CountTokens returns how many model tokens text becomes, loading the model
// if needed. ok is false when the model can't count tokens.
func (l *Lazy) CountTokens(text string) (n int, ok bool, err error) {
	err = l.use(func(m Model) error {
		if c, yes := m.(counter); yes {
			n, ok = c.CountTokens(text), true
		}
		return nil
	})
	return n, ok, err
}

// serial is a model that can embed one text using only the calling
// goroutine (for callers that run many embeds side by side).
type serial interface {
	EmbedSerial(text string) ([]float32, error)
}

// EmbedOne embeds one text, on the calling goroutine only if the model allows.
func (l *Lazy) EmbedOne(text string) ([]float32, error) {
	var out []float32
	err := l.use(func(m Model) error {
		if s, ok := m.(serial); ok {
			v, err := s.EmbedSerial(text)
			out = v
			return err
		}
		vs, err := m.Embed([]string{text})
		if err == nil && len(vs) == 1 {
			out = vs[0]
		}
		return err
	})
	return out, err
}

// arm schedules an unload check. Caller holds the write lock.
func (l *Lazy) arm() {
	if l.after <= 0 {
		return
	}
	if l.timer == nil {
		l.timer = time.AfterFunc(l.after, l.idle)
	} else {
		l.timer.Reset(l.after)
	}
}

func (l *Lazy) idle() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.m == nil {
		return
	}
	l.lastMu.Lock()
	last := l.last
	l.lastMu.Unlock()
	if wait := l.after - time.Since(last); wait > 0 {
		l.timer.Reset(wait)
		return
	}
	l.unload()
	if l.OnUnload != nil {
		l.OnUnload(l.after)
	}
}

// Unload releases the model now. The next Embed loads it again.
func (l *Lazy) Unload() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.unload()
}

func (l *Lazy) unload() {
	if c, ok := l.m.(io.Closer); ok {
		_ = c.Close()
	}
	l.m = nil
}
