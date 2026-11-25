package app

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"griz/internal/index"
)

// When vectors are made: store and the scan start embedding in a
// goroutine; recall waits for any of that, embeds whatever is still
// missing, and only then searches, so results never depend on timing.
// Reindex embeds in the foreground.
//
// Every embedding goes through ensure, which keeps a pending map (chunk hash
// -> channel) under one mutex: whoever asks for a chunk first embeds it;
// anyone else asking meanwhile waits on the channel. The vector is stored
// before the channel closes.

type pending struct {
	mu sync.Mutex
	m  map[index.Hash]chan struct{}

	// Progress of the chunks being embedded now, for status.
	claimed, done int
	since         time.Time
	ids           map[string]int
}

// item is one chunk to embed.
type item struct {
	id   string
	hash index.Hash
	text string // what the model sees (document prefix + EmbedText)
}

// embedResult says what an ensure call did.
type embedResult struct {
	waited time.Duration // waiting for chunks someone else was embedding
	Chunks int           // chunks embedded by this call
	took   time.Duration // time spent embedding them
	all    bool          // every chunk asked for was missing (new model, or vectors.bin lost)
}

// items turns chunk refs into embeddable items, reading each memory once.
// Refs whose file has changed since they were indexed are skipped.
func (a *App) items(refs []index.ChunkRef) []item {
	byID := map[string][]index.ChunkRef{}
	var ids []string
	for _, r := range refs {
		if byID[r.ID] == nil {
			ids = append(ids, r.ID)
		}
		byID[r.ID] = append(byID[r.ID], r)
	}
	sort.Strings(ids)
	var out []item
	for _, id := range ids {
		m, err := a.Repo.Read(id)
		if err != nil {
			continue
		}
		parts := a.chunk(m.Body)
		for _, r := range byID[id] {
			if r.Seq < len(parts) && index.HashOf(m.Path, parts[r.Seq]) == r.Hash {
				out = append(out, item{id, r.Hash, a.Cfg.Search.DocumentPrefix + index.EmbedText(m.Path, parts[r.Seq])})
			}
		}
	}
	return out
}

// ensure makes sure every chunk in refs has a vector for the active model:
// it embeds the ones nobody has, waits for the ones someone else is
// embedding, and saves new vectors to vectors.bin (adding only). It loads
// the model only if something is missing. On failure, the chunks it couldn't
// embed stay keyword-only until a later call.
func (a *App) ensure(refs []index.ChunkRef, op *Op) (embedResult, error) {
	var res embedResult
	if !a.HasModel() {
		return res, nil
	}
	mid := a.model.ID()
	var missing []index.ChunkRef
	for _, r := range refs {
		if !a.vec.Has(mid, r.Hash) {
			missing = append(missing, r)
		}
	}
	if len(missing) == 0 {
		return res, nil
	}
	res.all = len(missing) == len(refs) && len(refs) > 1
	var firstErr error
	// Two passes: chunks whose embedder failed are tried once more by us.
	for pass := 0; pass < 2 && len(missing) > 0; pass++ {
		mine, waits := a.claim(mid, missing)
		if len(mine) > 0 {
			its := a.items(mine)
			if len(its) < len(mine) { // changed since indexed: release what we can't embed
				a.release(mine, its)
			}
			n, took, err := a.embedItems(mid, its, op)
			res.Chunks += n
			res.took += took
			if err != nil && firstErr == nil {
				firstErr = err
			}
		}
		t := time.Now()
		for _, ch := range waits {
			<-ch
		}
		res.waited += time.Since(t)
		missing = missing[:0]
		for _, r := range refs {
			if !a.vec.Has(mid, r.Hash) {
				missing = append(missing, r)
			}
		}
		if firstErr != nil {
			break
		}
	}
	op.Add("embed wait", res.waited)
	op.Add("embed", res.took)
	if res.Chunks > 0 {
		op.debugf("embedded %d %s in %s", res.Chunks, plural(res.Chunks, "chunk", "chunks"), ms(res.took))
		if err := a.saveVectors(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return res, firstErr
}

// claim takes the missing chunks nobody is embedding, and returns the
// channels of those someone else is.
func (a *App) claim(mid string, refs []index.ChunkRef) (mine []index.ChunkRef, waits []chan struct{}) {
	p := &a.pend
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.m == nil {
		p.m = map[index.Hash]chan struct{}{}
	}
	seen := map[index.Hash]bool{}
	for _, r := range refs {
		if seen[r.Hash] {
			continue
		}
		seen[r.Hash] = true
		if ch, ok := p.m[r.Hash]; ok {
			waits = append(waits, ch)
			continue
		}
		if a.vec.Has(mid, r.Hash) { // finished between our check and the lock
			continue
		}
		p.m[r.Hash] = make(chan struct{})
		mine = append(mine, r)
	}
	if len(mine) > 0 {
		if p.claimed == p.done {
			p.claimed, p.done, p.since, p.ids = 0, 0, time.Now(), map[string]int{}
		}
		p.claimed += len(mine)
		for _, r := range mine {
			p.ids[r.ID]++
		}
	}
	return mine, waits
}

// finish closes the channels of hashes (their vectors stored, or failed).
func (a *App) finish(hashes []index.Hash) {
	p := &a.pend
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, h := range hashes {
		if ch, ok := p.m[h]; ok {
			close(ch)
			delete(p.m, h)
			p.done++
		}
	}
}

// release gives up claimed chunks that have no item.
func (a *App) release(mine []index.ChunkRef, its []item) {
	have := map[index.Hash]bool{}
	for _, it := range its {
		have[it.hash] = true
	}
	var drop []index.Hash
	for _, r := range mine {
		if !have[r.Hash] {
			drop = append(drop, r.Hash)
		}
	}
	a.finish(drop)
}

// Chunks are embedded one goroutine per chunk, at most search.embed_workers
// at once across the whole server (a.slots). A caller takes a slot before
// starting each goroutine, so it holds at most one blocked send; senders
// waiting on a channel are served in order, so a recall's few chunks
// interleave with a reindex's backlog instead of waiting behind all of it.
// Each goroutine stores the vector, then closes the chunk's channel.
// Recall's query embedding runs on the caller and takes no slot.

// jobsResult collects the outcome of one caller's chunks.
type jobsResult struct {
	wg  sync.WaitGroup
	mu  sync.Mutex
	n   int
	err error
}

func (a *App) embedChunk(mid string, it item, res *jobsResult) {
	defer func() { <-a.slots }()
	defer res.wg.Done()
	v, err := a.model.EmbedOne(it.text)
	if err == nil {
		a.vec.Put(mid, it.hash, v)
	} else if a.model.Failed() != nil {
		a.warnLoad.Do(func() { a.Warn(warnModel, "search: %s", a.Status()) })
	}
	a.finish([]index.Hash{it.hash}) // after the Put: waiters find the vector
	res.mu.Lock()
	if err == nil {
		res.n++
	} else if res.err == nil {
		res.err = err
	}
	res.mu.Unlock()
}

// embedItems embeds claimed items and waits for them. Every claimed
// channel is closed, even on failure.
func (a *App) embedItems(mid string, its []item, op *Op) (int, time.Duration, error) {
	if len(its) == 0 {
		return 0, 0, nil
	}
	perID := map[string]int{}
	for _, it := range its {
		perID[it.id]++
	}
	parts := make([]string, 0, len(perID))
	for _, id := range sortedKeys(perID) {
		parts = append(parts, fmt.Sprintf("%s (%d)", id, perID[id]))
	}
	op.debugf("new or changed since last embed; embedding %d %s: %s",
		len(its), plural(len(its), "chunk", "chunks"), strings.Join(parts, ", "))
	_, loadsBefore := a.model.Loaded()
	start := time.Now()
	res := &jobsResult{}
	res.wg.Add(len(its))
	for _, it := range its {
		a.slots <- struct{}{} // freed by embedChunk
		go a.embedChunk(mid, it, res)
	}
	res.wg.Wait()
	if _, loads := a.model.Loaded(); loads > loadsBefore {
		op.debugf("model loaded (%s, %s, %s); unloads after %s idle", a.model.Name(), mb(a.model.Bytes()), ms(a.model.LoadTook()), a.model.UnloadAfter())
	}
	return res.n, time.Since(start), res.err
}

// embedding reports the chunks being embedded now: done, claimed, since, ids.
func (a *App) embedding() (done, claimed int, since time.Time, ids []string) {
	p := &a.pend
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.claimed == p.done {
		return 0, 0, time.Time{}, nil
	}
	for id := range p.ids {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return p.done, p.claimed, p.since, ids
}

// EmbedMissing makes sure every indexed chunk has a vector.
func (a *App) EmbedMissing(op *Op) (embedResult, error) {
	return a.ensure(a.ix.Chunks(index.Filter{}), op)
}

// embedAsync starts embedding the chunks of ids in a goroutine (for store
// and the scan). A failure becomes a warning in the next tool reply.
func (a *App) embedAsync(sess *Session, why string, ids ...string) {
	if !a.HasModel() || len(ids) == 0 {
		return
	}
	want := map[string]bool{}
	for _, id := range ids {
		want[id] = true
	}
	var refs []index.ChunkRef
	for _, r := range a.ix.Chunks(index.Filter{}) {
		if want[r.ID] {
			refs = append(refs, r)
		}
	}
	a.embedWG.Add(1)
	go func() {
		defer a.embedWG.Done()
		op := a.BeginOp(sess, "embed ("+why+")")
		if sess == nil {
			op = a.beginShared("embed (" + why + ")")
		}
		res, err := a.ensure(refs, op)
		if err != nil && a.HasModel() {
			a.Warn(warnEmbed, "embedding: %v", err)
		}
		if res.Chunks > 0 || err != nil {
			a.EndOp(op, ids)
		}
	}()
}

// WaitEmbeds waits for embedding started by store or the scan.
func (a *App) WaitEmbeds() { a.embedWG.Wait() }

// saveVectors adds new vectors to vectors.bin.
func (a *App) saveVectors() error { return a.vec.Save() }

// PruneVectors drops vectors no current chunk uses (reindex, gc, purge),
// after bringing the index up to date.
func (a *App) PruneVectors() error {
	if err := a.Sync(); err != nil {
		return err
	}
	keep := map[index.Hash]bool{}
	for _, r := range a.ix.Chunks(index.Filter{}) {
		keep[r.Hash] = true
	}
	return a.vec.Prune(keep)
}

// Reembed recomputes the vector of every chunk for the active model, then
// prunes vectors no chunk uses.
func (a *App) Reembed(op *Op) (int, error) {
	if !a.HasModel() {
		return 0, nil
	}
	if err := a.Sync(); err != nil {
		return 0, err
	}
	refs := a.ix.Chunks(index.Filter{})
	mine, waits := a.claim("", refs) // "" never has vectors: claim them all
	for _, ch := range waits {
		<-ch
	}
	n, took, err := a.embedItems(a.model.ID(), a.items(mine), op)
	op.Add("embed", took)
	if err != nil {
		return n, err
	}
	if err := a.saveVectors(); err != nil {
		return n, err
	}
	return n, a.PruneVectors()
}

// vectorCount counts chunks that have a current vector.
func (a *App) vectorCount() (have, total int) {
	if a.model == nil {
		return 0, 0
	}
	mid := a.model.ID()
	refs := a.ix.Chunks(index.Filter{})
	for _, r := range refs {
		if a.vec.Has(mid, r.Hash) {
			have++
		}
	}
	return have, len(refs)
}
