// Package index is the in-memory keyword index over memory chunks (BM25 with
// field weights and English stemming). It holds terms, chunk positions and
// chunk hashes; memory text stays on disk.
package index

import (
	"crypto/sha256"
	"math"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/kljensen/snowball/english"
)

// Doc is one memory as the index sees it.
type Doc struct {
	ID       string // the memory's short id (the index key)
	Path     string // ns/slug: searched and embedded, unlike the id
	NS       string
	Tags     []string
	Expires  string // YYYY-MM-DD or ""
	Archived bool
	Chunks   []string // body split into chunks (text is not kept)
}

// Filter limits results to one namespace (all if NS is empty).
type Filter struct {
	NS string
}

func (f Filter) ok(m *mem) bool { return f.NS == "" || m.ns == f.NS }

// Hit is one ranked chunk.
type Hit struct {
	ID    string
	Seq   int
	Score float64
}

// ChunkRef identifies a chunk and its embedding key.
type ChunkRef struct {
	ID   string
	Seq  int
	Hash Hash
}

// Hash keys a chunk's embedding: a hash of the text it is embedded as.
type Hash [16]byte

// Params tunes BM25 and the field weights (id and tags count more than body text).
type Params struct {
	K1, B                  float64
	WID, WNS, WTags, WBody float64
}

func defaultParams() Params {
	return Params{K1: 1.2, B: 0.75, WID: 4, WNS: 2, WTags: 3, WBody: 1}
}

type mem struct {
	ns, expires string
	archived    bool
	chunks      []int // chunk slots
}

type chunk struct {
	id    string
	seq   int
	hash  Hash
	len   float64 // weighted length
	terms map[string]float64
}

// Mem is the in-memory index. Safe for concurrent use.
type Mem struct {
	p        Params
	mu       sync.RWMutex
	mems     map[string]*mem
	chunks   []*chunk // nil = free slot
	free     []int
	postings map[string]map[int]float64 // term -> chunk slot -> weighted tf
	totalLen float64
	live     int
}

// New makes an index with the default parameters.
func New() *Mem { return NewWith(defaultParams()) }

func NewWith(p Params) *Mem {
	return &Mem{p: p, mems: map[string]*mem{}, postings: map[string]map[int]float64{}}
}

// EmbedText is the text a chunk is embedded as (and hashed by).
func EmbedText(path, text string) string {
	return strings.NewReplacer("/", " ", "-", " ", "_", " ").Replace(path) + ": " + text
}

func HashOf(path, text string) Hash {
	s := sha256.Sum256([]byte(EmbedText(path, text)))
	var h Hash
	copy(h[:], s[:16])
	return h
}

// Add indexes a memory, replacing any earlier version.
func (x *Mem) Add(d Doc) {
	x.mu.Lock()
	defer x.mu.Unlock()
	x.remove(d.ID)
	m := &mem{ns: d.NS, expires: d.Expires, archived: d.Archived}
	head := map[string]float64{}
	for _, t := range Terms(d.Path) {
		head[t] += x.p.WID
	}
	for _, t := range Terms(d.NS) {
		head[t] += x.p.WNS
	}
	for _, t := range Terms(strings.Join(d.Tags, " ")) {
		head[t] += x.p.WTags
	}
	var headLen float64
	for _, v := range head {
		headLen += v
	}
	for i, text := range d.Chunks {
		c := &chunk{id: d.ID, seq: i, hash: HashOf(d.Path, text), terms: map[string]float64{}, len: headLen}
		for t, v := range head {
			c.terms[t] = v
		}
		for _, t := range Terms(text) {
			c.terms[t] += x.p.WBody
			c.len += x.p.WBody
		}
		slot := len(x.chunks)
		if n := len(x.free); n > 0 {
			slot, x.free = x.free[n-1], x.free[:n-1]
			x.chunks[slot] = c
		} else {
			x.chunks = append(x.chunks, c)
		}
		for t, v := range c.terms {
			p := x.postings[t]
			if p == nil {
				p = map[int]float64{}
				x.postings[t] = p
			}
			p[slot] = v
		}
		x.totalLen += c.len
		x.live++
		m.chunks = append(m.chunks, slot)
	}
	x.mems[d.ID] = m
}

// Load replaces the whole index with docs in one step.
func (x *Mem) Load(docs []Doc) {
	fresh := NewWith(x.p)
	for _, d := range docs {
		fresh.Add(d)
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	x.mems, x.chunks, x.free, x.postings = fresh.mems, fresh.chunks, fresh.free, fresh.postings
	x.totalLen, x.live = fresh.totalLen, fresh.live
}

func (x *Mem) Remove(id string) {
	x.mu.Lock()
	defer x.mu.Unlock()
	x.remove(id)
}

func (x *Mem) remove(id string) {
	m := x.mems[id]
	if m == nil {
		return
	}
	for _, slot := range m.chunks {
		c := x.chunks[slot]
		for t := range c.terms {
			if p := x.postings[t]; p != nil {
				delete(p, slot)
				if len(p) == 0 {
					delete(x.postings, t)
				}
			}
		}
		x.totalLen -= c.len
		x.live--
		x.chunks[slot] = nil
		x.free = append(x.free, slot)
	}
	delete(x.mems, id)
}

func (x *Mem) Has(id string) bool {
	x.mu.RLock()
	defer x.mu.RUnlock()
	return x.mems[id] != nil
}

// ArchivedAt reports whether a memory counts as archived on date today:
// flagged archived, or past its expiry date.
func (x *Mem) ArchivedAt(id, today string) bool {
	x.mu.RLock()
	defer x.mu.RUnlock()
	m := x.mems[id]
	return m != nil && (m.archived || (m.expires != "" && m.expires < today))
}

// IDs lists indexed memories passing f.
func (x *Mem) IDs(f Filter) []string {
	x.mu.RLock()
	defer x.mu.RUnlock()
	var out []string
	for id, m := range x.mems {
		if f.ok(m) {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

// Search ranks chunks for q by BM25, best first.
func (x *Mem) Search(q string, f Filter, limit int) []Hit {
	x.mu.RLock()
	defer x.mu.RUnlock()
	if x.live == 0 {
		return nil
	}
	avg := x.totalLen / float64(x.live)
	scores := map[int]float64{}
	seen := map[string]bool{}
	for _, t := range Terms(q) {
		if seen[t] {
			continue
		}
		seen[t] = true
		p := x.postings[t]
		if len(p) == 0 {
			continue
		}
		n := float64(len(p))
		idf := math.Log(1 + (float64(x.live)-n+0.5)/(n+0.5))
		for slot, tf := range p {
			c := x.chunks[slot]
			if !f.ok(x.mems[c.id]) {
				continue
			}
			k1, b := x.p.K1, x.p.B
			scores[slot] += idf * tf * (k1 + 1) / (tf + k1*(1-b+b*c.len/avg))
		}
	}
	hits := make([]Hit, 0, len(scores))
	for slot, s := range scores {
		c := x.chunks[slot]
		hits = append(hits, Hit{ID: c.id, Seq: c.seq, Score: s})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		if hits[i].ID != hits[j].ID {
			return hits[i].ID < hits[j].ID
		}
		return hits[i].Seq < hits[j].Seq
	})
	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return hits
}

// Chunks lists every chunk of memories passing f.
func (x *Mem) Chunks(f Filter) []ChunkRef {
	x.mu.RLock()
	defer x.mu.RUnlock()
	var out []ChunkRef
	for _, c := range x.chunks {
		if c != nil && f.ok(x.mems[c.id]) {
			out = append(out, ChunkRef{ID: c.id, Seq: c.seq, Hash: c.hash})
		}
	}
	return out
}

var wordRE = regexp.MustCompile(`[\p{L}\p{N}]+`)

var stop = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`a an and are as at be but by can did do does for from had has have how i if in
	into is it its me my of on or our so than that the their them then there these they this to was we were what when
	where which who why will with you your about any all also just not no
	s t d ll re ve m`) { // the last line: what's left of possessives and contractions (wife's, don't, I'll)
		stop[w] = true
	}
}

// Terms lowercases, drops stopwords and stems.
func Terms(s string) []string {
	var out []string
	for _, w := range wordRE.FindAllString(strings.ToLower(s), -1) {
		if stop[w] {
			continue
		}
		out = append(out, english.Stem(w, false))
	}
	return out
}
