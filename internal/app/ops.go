package app

import (
	"errors"
	"fmt"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"griz/internal/embed"
	"griz/internal/index"
	"griz/internal/memory"
)

type scored struct {
	id     string
	seq    int             // the chunk shown: best by chunk-level fused score
	weight float64         // fading (1: none)
	chunks map[int]float64 // every matching chunk's score (faded), by seq: recall may show more than one
	score  float64         // memory-level fused score
	// Memory-level place in each list (0 = not in it), the score of the
	// memory's best chunk there, and which chunk that was. For explain.
	kwRank  int
	kwScore float64
	kwSeq   int
	vecRank int
	vecSim  float64
	vecSeq  int
}

// ranking is a fused ranking plus what explain and debug need to know.
type ranking struct {
	results  []scored
	kwHits   int
	vecHits  int
	searched int     // chunks compared by meaning
	byVector bool    // search by meaning took part
	alpha    float64 // keyword weight in the blend (1 - alpha goes to meaning)
	embedded embedResult
	embedErr error
}

// rank returns memories best-first, blended from keyword and (if a model is
// set) meaning search. Each search's scores are divided by that search's best
// for the query, and the two are weighted by keywordWeight(q): queries the
// model reads as ordinary words lean on meaning, queries it has to split into
// many tokens (addresses, ids, codes) lean on keywords. A memory scores as its
// best chunk in each search; it shows its best chunk by the same blend.
// Before searching by meaning it waits for embedding in progress and embeds
// whatever is still missing, so the vectors searched are complete.
func (a *App) rank(q string, f index.Filter, op *Op) ranking {
	var rk ranking
	type ck struct {
		id  string
		seq int
	}
	kwChunk, vecChunk := map[ck]float64{}, map[ck]float64{}
	mems := map[string]*scored{}
	get := func(id string) *scored {
		if mems[id] == nil {
			mems[id] = &scored{id: id}
		}
		return mems[id]
	}
	// A fixed snapshot of the chunks: a store landing meanwhile shows up in
	// the next recall, not half in this one.
	refs := a.ix.Chunks(f)
	type chunkKey struct {
		id  string
		seq int
	}
	inSnap := make(map[chunkKey]bool, len(refs))
	for _, r := range refs {
		inSnap[chunkKey{r.ID, r.Seq}] = true
	}
	t := time.Now()
	var kw []index.Hit
	for _, h := range a.ix.Search(q, f, a.Cfg.Search.Candidates) {
		if inSnap[chunkKey{h.ID, h.Seq}] {
			kw = append(kw, h)
		}
	}
	op.Since("keyword", t)
	rk.kwHits = len(kw)
	// Each list collapses to memories, ranked by their best chunk in it.
	n := 0
	for _, h := range kw {
		kwChunk[ck{h.ID, h.Seq}] = h.Score
		if m := get(h.ID); m.kwRank == 0 {
			n++
			m.kwRank, m.kwScore, m.kwSeq = n, h.Score, h.Seq
		}
	}
	if a.HasModel() {
		rk.embedded, rk.embedErr = a.ensure(refs, op)
		if rk.embedErr != nil && a.HasModel() {
			a.Warn(warnEmbed, "embedding: %v", rk.embedErr)
		}
		t = time.Now()
		vs, err := a.embed([]string{a.Cfg.Search.QueryPrefix + q}, op)
		t = op.Since("query embed", t)
		if err == nil && len(vs) == 1 {
			rk.byVector = true
			var vec []index.Hit
			vec, rk.searched = a.similarRefs(vs[0], refs, a.Cfg.Search.MinSimilarity, a.Cfg.Search.Candidates)
			op.Since("vectors", t)
			rk.vecHits = len(vec)
			n := 0
			for _, h := range vec {
				vecChunk[ck{h.ID, h.Seq}] = h.Score
				if m := get(h.ID); m.vecRank == 0 {
					n++
					m.vecRank, m.vecSim, m.vecSeq = n, h.Score, h.Seq
				}
			}
			rk.alpha = a.keywordWeight(q)
		}
	}
	if !rk.byVector {
		rk.alpha = 1
	}
	t = time.Now()
	// Each search relative to its own best for this query (lists are best-first).
	kwBest, vecBest := 0.0, 0.0
	if len(kw) > 0 {
		kwBest = kw[0].Score
	}
	for _, v := range vecChunk {
		vecBest = max(vecBest, v)
	}
	blend := func(k, v float64) float64 {
		s := 0.0
		if kwBest > 0 {
			s += rk.alpha * k / kwBest
		}
		if vecBest > 0 {
			s += (1 - rk.alpha) * max(0, v) / vecBest
		}
		return s
	}
	// Fading multiplies the blend, after normalization: each memory by its own
	// weight, leaving the balance between the two searches alone.
	today := memory.Today()
	for _, m := range mems {
		m.weight = a.weight(m.id, today)
		m.score = m.weight * blend(m.kwScore, m.vecSim)
	}
	best := map[string]float64{}
	pick := func(c ck) {
		m := mems[c.id]
		sc := blend(kwChunk[c], vecChunk[c])
		if m.chunks == nil {
			m.chunks = map[int]float64{}
		}
		m.chunks[c.seq] = m.weight * sc
		if b, ok := best[c.id]; !ok || sc > b || (sc == b && c.seq < m.seq) {
			best[c.id], m.seq = sc, c.seq
		}
	}
	for c := range kwChunk {
		pick(c)
	}
	for c := range vecChunk {
		pick(c)
	}
	for _, m := range mems {
		if m.score >= a.Cfg.Recall.Cutoff {
			rk.results = append(rk.results, *m)
		}
	}
	sort.Slice(rk.results, func(i, j int) bool {
		if rk.results[i].score != rk.results[j].score {
			return rk.results[i].score > rk.results[j].score
		}
		return rk.results[i].id < rk.results[j].id
	})
	op.Since("fuse", t)
	return rk
}

// keywordWeight is 1 - 1/(model tokens per content word of q): 0 when the
// model knows every word whole, near 1 when it has to break them into pieces.
// Stopwords don't count, and a trailing 's is dropped first. With no word left
// to count, or a model that can't count tokens, meaning alone decides.
func (a *App) keywordWeight(q string) float64 {
	var tokens, words int
	for w := range strings.FieldsSeq(q) {
		w = strings.Trim(w, `?!.,;:"()[]{}`)
		w = strings.TrimSuffix(strings.TrimSuffix(w, "'s"), "’s")
		if len(index.Terms(w)) == 0 {
			continue
		}
		n, ok, err := a.model.CountTokens(w)
		if err != nil || !ok {
			return 0
		}
		tokens += n
		words++
	}
	if words == 0 || tokens <= words {
		return 0
	}
	return 1 - float64(words)/float64(tokens)
}

func (a *App) similar(v []float32, f index.Filter, minSim float64, limit int) []index.Hit {
	hits, _ := a.similarRefs(v, a.ix.Chunks(f), minSim, limit)
	return hits
}

// similarRefs ranks refs by similarity to v, also returning how many had a vector to compare.
func (a *App) similarRefs(v []float32, refs []index.ChunkRef, minSim float64, limit int) ([]index.Hit, int) {
	var hits []index.Hit
	n := 0
	a.vec.Each(a.model.ID(), refs, func(r index.ChunkRef, cv []float32) {
		n++
		if s := embed.Cosine(v, cv); s >= minSim {
			hits = append(hits, index.Hit{ID: r.ID, Seq: r.Seq, Score: s})
		}
	})
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, n
}

// Recall searches and renders results within a token budget. It returns the
// ids whose text was returned.
//
// Archived memories (flagged, or past their expiry date) are ranked like any
// other but listed only by id on the "archived:" line, unless withArchived
// treats them as current for this call.
func (a *App) Recall(q string, budget int, withArchived bool) (string, []string, error) {
	return a.RecallOp(nil, q, budget, withArchived)
}

// RecallOp is Recall, timing its steps in op.
func (a *App) RecallOp(op *Op, q string, budget int, withArchived bool) (string, []string, error) {
	if op == nil {
		op = a.BeginOp(nil, "recall") // timed for debug lines, not recorded
	}
	t := time.Now()
	if err := a.Sync(); err != nil {
		return "", nil, err
	}
	op.Since("sync", t)
	if budget <= 0 {
		budget = a.Cfg.Recall.Budget
	}
	q = strings.TrimSpace(q)
	if q == "*" || q == "" {
		return a.list(), nil, nil
	}
	rk := a.rank(q, index.Filter{}, op)
	a.debugRecall(rk, op)
	all := rk.results
	note := ""
	if rk.embedded.all && rk.embedded.Chunks > 0 {
		note = fmt.Sprintf("\n\ngriz: re-embedded all %d chunks for search by meaning before answering (new model, or vectors.bin was missing), so this search was slow. It happens once.", rk.embedded.Chunks)
	}
	if len(all) == 0 {
		return "none" + note, nil, nil
	}
	defer op.Since("format", time.Now())
	today := memory.Today()
	var results []scored
	var archived []string
	for _, r := range all {
		if !withArchived && a.ix.ArchivedAt(r.id, today) {
			archived = append(archived, r.id)
		} else {
			results = append(results, r)
		}
	}
	var b strings.Builder
	var full, more []string
	used := 0
	// The first full_results memories are candidates; the rest go to "more:".
	cands := results
	if len(cands) > a.Cfg.Recall.FullResults {
		for _, r := range cands[a.Cfg.Recall.FullResults:] {
			more = append(more, r.id)
		}
		cands = cands[:a.Cfg.Recall.FullResults]
	}
	// What recall can show, in pieces: each candidate's best chunk (or its
	// whole text, if short), plus any other chunk of a long memory scoring
	// above the weakest candidate. Pieces are taken best first while the
	// budget lasts, so a strong second section of one memory can come before
	// another memory's best chunk.
	type entry struct {
		m     *memory.Memory
		parts []string // chunks, for a long memory (nil: shown whole)
		shown []int    // seqs taken
		cut   string
	}
	type piece struct {
		e     *entry
		seq   int // -1: the whole text
		score float64
		best  bool
	}
	var entries []*entry
	var pieces []piece
	lowest := 0.0
	if len(cands) > 0 {
		lowest = cands[len(cands)-1].score
	}
	for _, r := range cands {
		m, err := a.Repo.Read(r.id)
		if err != nil {
			continue
		}
		e := &entry{m: m}
		entries = append(entries, e)
		if memory.Words(m.Body) <= a.Cfg.Recall.ExcerptWords {
			pieces = append(pieces, piece{e, -1, r.score, true})
			continue
		}
		e.parts = a.chunk(m.Body)
		seq := min(r.seq, len(e.parts)-1)
		pieces = append(pieces, piece{e, seq, r.score, true})
		for s, sc := range r.chunks {
			if s != seq && s < len(e.parts) && sc > lowest {
				pieces = append(pieces, piece{e, s, sc, false})
			}
		}
	}
	sort.SliceStable(pieces, func(i, j int) bool { return pieces[i].score > pieces[j].score })
	header := func(e *entry) string {
		var notes []string
		if e.parts != nil {
			seqs := slices.Clone(e.shown)
			slices.Sort(seqs)
			nums := make([]string, len(seqs))
			for i, s := range seqs {
				nums[i] = strconv.Itoa(s + 1)
			}
			notes = append(notes, strings.Join(nums, ", ")+" of "+strconv.Itoa(len(e.parts)))
		}
		if e.m.Updated != "" {
			notes = append(notes, "updated "+e.m.Updated)
		}
		if s := a.since(e.m.ID); s > e.m.Updated {
			notes = append(notes, "renewed "+s)
		}
		h := "### " + e.m.Label()
		if len(notes) > 0 {
			h += " (" + strings.Join(notes, ", ") + ")"
		}
		return h + "\n"
	}
	text := func(e *entry, seq int) string {
		if seq < 0 {
			return e.m.Body
		}
		return e.parts[seq]
	}
	for _, p := range pieces {
		e := p.e
		if !p.best && len(e.shown) == 0 && e.cut == "" {
			continue // its memory didn't fit
		}
		cost := Tokens(memory.Display(text(e, p.seq)) + "\n\n…\n\n")
		if p.best {
			cost = Tokens(header(e) + memory.Display(text(e, p.seq)) + "\n\n")
		}
		if used+cost > budget {
			if !p.best || len(full) > 0 {
				if p.best {
					more = append(more, e.m.ID)
				}
				continue
			}
			// The best match always shows, cut to fit, leaving room for a
			// short "more:" line when there are other results.
			room := budget
			if len(all) > 1 {
				room -= min(a.Cfg.Recall.MoreReserve, budget/4)
			}
			e.shown = append(e.shown, max(p.seq, 0))
			e.cut = truncate(header(e)+memory.Display(text(e, p.seq))+"\n\n", room*4-8) + " ...\n\n"
			used += Tokens(e.cut)
			full = append(full, e.m.ID)
			continue
		}
		e.shown = append(e.shown, max(p.seq, 0))
		used += cost
		if p.best {
			full = append(full, e.m.ID)
		}
	}
	// Shown in rank order; a memory's chunks in the order they appear in it.
	for _, e := range entries {
		switch {
		case e.cut != "":
			b.WriteString(e.cut)
		case len(e.shown) == 0:
		case e.parts == nil:
			b.WriteString(header(e))
			b.WriteString(memory.Display(e.m.Body))
			b.WriteString("\n\n")
		default:
			seqs := slices.Clone(e.shown)
			slices.Sort(seqs)
			b.WriteString(header(e))
			for i, s := range seqs {
				if i > 0 {
					if s == seqs[i-1]+1 {
						b.WriteString("\n\n")
					} else {
						b.WriteString("\n\n…\n\n")
					}
				}
				b.WriteString(memory.Display(e.parts[s]))
			}
			b.WriteString("\n\n")
		}
	}
	used = a.idLine(&b, "more:", more, a.Cfg.Recall.IDResults, used, budget)
	a.idLine(&b, "archived:", archived, a.Cfg.Recall.ArchivedResults, used, budget)
	out := strings.TrimSpace(b.String())
	if out == "" {
		out = "none"
	}
	return out + note, full, nil
}

// idLine writes "label path [id], path [id], ..." with up to max ids that
// fit the budget, and returns the tokens used so far.
func (a *App) idLine(b *strings.Builder, label string, ids []string, max, used, budget int) int {
	if len(ids) > max {
		ids = ids[:max]
	}
	var line strings.Builder
	line.WriteString(label)
	n := 0
	for _, id := range ids {
		id := a.label(id)
		if used+tokensOf(line.Len()+len(id)+2) > budget {
			break
		}
		if n > 0 {
			line.WriteByte(',')
		}
		line.WriteByte(' ')
		line.WriteString(id)
		n++
	}
	if n == 0 {
		return used
	}
	line.WriteByte('\n')
	b.WriteString(line.String())
	return used + tokensOf(line.Len())
}

// truncate cuts s to at most n bytes at a word boundary.
func truncate(s string, n int) string {
	if len(s) <= n {
		return strings.TrimSpace(s)
	}
	s = s[:max(n, 0)]
	if i := strings.LastIndexAny(s, " \n"); i > 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}

// folders returns every folder at every level, with the number of
// memories under it (archived included).
func (a *App) folders() map[string]int {
	out := map[string]int{}
	for ns, n := range a.namespaces() {
		for p := ns; p != ""; p = memory.NS(p) {
			out[p] += n
		}
	}
	return out
}

// newFolder returns the shallowest level of folder that doesn't exist yet ("" if all do).
func (a *App) newFolder(folder string) string {
	all := a.folders()
	parts := strings.Split(folder, "/")
	for i := range parts {
		if p := strings.Join(parts[:i+1], "/"); all[p] == 0 {
			return p
		}
	}
	return ""
}

// list answers recall "*": the folder tree with counts, e.g. "work 26 (research 19)".
func (a *App) list() string {
	all := a.folders()
	if len(all) == 0 {
		return "none"
	}
	var render func(parent string) string
	render = func(parent string) string {
		var parts []string
		for _, f := range sortedKeys(all) {
			if memory.NS(f) != parent {
				continue
			}
			name := f
			if parent != "" {
				name = strings.TrimPrefix(f, parent+"/")
			}
			part := fmt.Sprintf("%s %d", name, all[f])
			if kids := render(f); kids != "" {
				part += " (" + kids + ")"
			}
			parts = append(parts, part)
		}
		return strings.Join(parts, ", ")
	}
	return "folders: " + render("")
}

// Get returns the full text of memories, archived or not.
func (a *App) Get(ids []string) (string, []string) {
	var parts, found []string
	for _, key := range ids {
		if strings.TrimSpace(key) == "" {
			continue
		}
		id, err := a.Resolve(key)
		var m *memory.Memory
		if err == nil {
			m, err = a.Repo.Read(id)
		}
		if err != nil {
			parts = append(parts, err.Error())
			continue
		}
		parts = append(parts, "### "+m.Label()+"\n"+memory.Display(m.Body))
		found = append(found, id)
		a.renewOnGet(id)
	}
	if len(parts) == 0 {
		return "no id given", nil
	}
	return strings.Join(parts, "\n\n"), found
}

// StoreArgs are the store tool's parameters.
type StoreArgs struct {
	ID       string   `json:"id"`
	Body     *string  `json:"body"`
	Tags     []string `json:"tags"`
	Expires  *string  `json:"expires"`
	Old      *string  `json:"old"`
	New      *string  `json:"new"`
	NewNS    bool     `json:"new_ns"`
	Force    bool     `json:"force"`
	Archived *bool    `json:"archived"`
	Pin      *bool    `json:"pin"`
	Term     *string  `json:"term"`
	Append   *string  `json:"append"`
	Move     *string  `json:"move"`
	All      bool     `json:"all"`     // old/new: replace every copy, not just a unique one
	Preview  bool     `json:"preview"` // old/new: show what would change, write nothing
	Count    *int     `json:"count"`   // old/new: refuse unless this many copies are replaced
	// Renew restarts the memory's fading today (griz memory renew; not an
	// MCP argument: Claude renews by reading).
	Renew bool `json:"-"`
}

// Store creates, replaces or edits a memory. The reply is "ok <id>",
// "similar: <id>", or an error.
func (a *App) Store(s StoreArgs) (string, error) { return a.StoreOp(nil, s) }

// StoreOp is Store, timing its steps in op.
func (a *App) StoreOp(op *Op, s StoreArgs) (string, error) {
	key := strings.TrimSuffix(strings.TrimSpace(s.ID), ".md")
	if key == "" {
		return "", fmt.Errorf("id is required: a memory's id, or a path (ns/slug) for a new one")
	}
	edit := s.Old != nil || s.New != nil
	appendText := s.Append != nil
	if s.Move != nil {
		if edit || appendText || s.Body != nil || s.Tags != nil || s.Term != nil || s.Expires != nil || s.Archived != nil || s.Pin != nil || s.Force || s.Renew {
			return "", fmt.Errorf("give move alone (with new_ns for a new folder)")
		}
		return a.moveOp(op, key, *s.Move, s.NewNS)
	}
	switch {
	case appendText && (edit || s.Body != nil):
		return "", fmt.Errorf("give append alone, without body or old and new")
	case appendText && strings.TrimSpace(*s.Append) == "":
		return "", fmt.Errorf("append is empty")
	case edit && s.Body != nil:
		return "", fmt.Errorf("give body, or old and new, not both")
	case edit && (s.Old == nil || s.New == nil):
		return "", fmt.Errorf("old and new go together")
	case edit && *s.Old == "":
		return "", fmt.Errorf("old is empty")
	case !edit && (s.All || s.Preview || s.Count != nil):
		return "", fmt.Errorf("all, preview and count go with old and new")
	case !edit && !appendText && s.Body == nil && s.Archived == nil && s.Pin == nil && s.Term == nil && s.Expires == nil && s.Tags == nil && !s.Renew:
		return "", fmt.Errorf("body is required")
	case !edit && s.Body != nil && strings.TrimSpace(*s.Body) == "":
		return "", fmt.Errorf("body is empty")
	case s.Preview && (s.Tags != nil || s.Term != nil || s.Expires != nil || s.Archived != nil || s.Pin != nil):
		return "", fmt.Errorf("preview shows only what old and new would change; give the other changes without it")
	}
	if s.Preview {
		id, _, err := a.storeTarget(key)
		if err != nil {
			return "", err
		}
		if id == "" {
			return "", fmt.Errorf("not found: %s", key)
		}
		m, err := a.Repo.Read(id)
		if err != nil {
			return "", err
		}
		return previewEdit(m, *s.Old, *s.New, s.All, a.Cfg.Recall.Budget)
	}
	flagOnly := !edit && !appendText && s.Body == nil // only setting archived, pin, term, tags or expires
	now := time.Now()
	var entryLine string
	if appendText {
		entryLine = memory.EntryLine(now, *s.Append)
	}
	var exp *string
	if s.Expires != nil {
		e, err := memory.ParseExpires(*s.Expires, time.Now())
		if err != nil {
			return "", err
		}
		exp = &e
	}
	if s.Term != nil && !memory.ValidTerm(*s.Term) {
		return "", fmt.Errorf("term %q: want short, medium, long or infinite", *s.Term)
	}
	tags := cleanTags(s.Tags)
	var reply, id, note string
	replaced := 0
	tw := time.Now()
	err := a.write(func() error {
		var m *memory.Memory
		var path string
		var err error
		if id, path, err = a.storeTarget(key); err != nil {
			return err
		}
		exists := id != ""
		verb := "store"
		switch {
		case exists:
			var err error
			if m, err = a.Repo.Read(id); err != nil {
				return err
			}
			switch {
			case appendText:
				verb = "append"
				m.Body = strings.TrimRight(m.Body, "\n") + "\n\n" + entryLine
			case flagOnly && s.Renew && s.Archived == nil && s.Pin == nil && s.Term == nil && s.Expires == nil && s.Tags == nil:
				verb = "renew"
			case flagOnly && s.Archived != nil:
				verb = "archive"
				if !*s.Archived {
					verb = "unarchive"
				}
			case flagOnly && s.Pin != nil && s.Term == nil && s.Expires == nil && s.Tags == nil:
				verb = "pin"
				if !*s.Pin {
					verb = "unpin"
				}
			case flagOnly:
				verb = "set"
			case edit:
				verb = "edit"
				body, n, err := replaceText(m, *s.Old, *s.New, s.All, s.Count)
				if err != nil {
					return err
				}
				m.Body = body
				if s.All {
					replaced = n
				}
			default:
				m.Body = *s.Body
			}
		case edit || flagOnly:
			return fmt.Errorf("not found: %s", key)
		default:
			if nf := a.newFolder(memory.NS(path)); nf != "" && !s.NewNS && a.Cfg.Store.NewNSGuard {
				msg := fmt.Sprintf("new folder %q; pass new_ns: true", nf)
				if all := a.folders(); len(all) > 0 {
					msg += ", or use one of: " + strings.Join(sortedKeys(all), ", ")
				}
				return errors.New(msg)
			}
			if appendText {
				verb = "append"
				id = a.Repo.NewID()
				m = &memory.Memory{ID: id, Path: path, Body: entryLine, Created: now.Format(memory.DateLayout)}
				break
			}
			if a.Cfg.Store.DupCheck && !s.Force {
				if dup := a.duplicate(path, *s.Body); dup != "" {
					reply = "similar: " + a.label(dup)
					if a.ix.ArchivedAt(dup, memory.Today()) {
						reply += " (archived)"
					}
					return nil
				}
			}
			id = a.Repo.NewID()
			m = &memory.Memory{ID: id, Path: path, Body: *s.Body, Created: now.Format(memory.DateLayout)}
		}
		if s.Tags != nil {
			m.Tags = tags
		}
		if s.Pin != nil {
			m.Pin = *s.Pin
		}
		// A content write sets updated; renew, a term change and unarchive renew
		// the memory without an edit. None of them touches the hard date.
		today := now.Format(memory.DateLayout)
		if !flagOnly {
			m.Updated = today
		}
		switch {
		case s.Term != nil:
			m.Term = *s.Term
		case m.Term == "" && !flagOnly:
			m.Term = a.Cfg.Store.DefaultTerm
		}
		if flagOnly && (s.Renew || s.Term != nil || (s.Archived != nil && !*s.Archived)) && today > m.Updated {
			m.Renewed = today
		}
		if exp != nil {
			m.Expires = *exp
		}
		if s.Archived != nil {
			m.Archived = *s.Archived
			if !m.Archived && exp == nil && m.Expired(today) {
				note = fmt.Sprintf(" (its date %s had passed; cleared)", m.Expires)
				m.Expires = ""
			}
		}
		if strings.TrimSpace(m.Body) == "" {
			return fmt.Errorf("body would be empty; use delete")
		}
		if err := a.Repo.Write(m, verb+" "+m.Label()); err != nil {
			return err
		}
		a.indexed(m)
		reply = "ok " + m.Label() + note
		if replaced > 0 {
			reply += fmt.Sprintf(" (replaced %d)", replaced)
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	op.Since("write and commit", tw)
	if strings.HasPrefix(reply, "ok ") {
		op.debugf("stored %s; embedding its chunks in the background", reply[3:])
		// Embed the new chunks in the background; recall waits for them.
		var sess *Session
		if op != nil {
			sess = op.Sess
		}
		a.embedAsync(sess, "store", id)
	}
	return reply, nil
}

// moveOp moves the memory key names to target, like mv: an existing folder
// (or any target ending in "/") receives it under its own slug; otherwise
// target is its new ns/slug. The id, file contents and history stay.
func (a *App) moveOp(op *Op, key, target string, newNS bool) (string, error) {
	target = strings.TrimSuffix(strings.TrimSpace(target), ".md")
	if strings.Trim(target, "/") == "" {
		return "", fmt.Errorf("move needs a folder or a path (ns/slug)")
	}
	var reply, id string
	tw := time.Now()
	err := a.write(func() error {
		var err error
		if id, _, err = a.storeTarget(key); err != nil {
			return err
		}
		if id == "" {
			return fmt.Errorf("not found: %s", key)
		}
		m, err := a.Repo.Read(id)
		if err != nil {
			return err
		}
		from := m.Path
		to := target
		if folder := strings.Trim(target, "/"); strings.HasSuffix(target, "/") || a.folders()[folder] > 0 {
			to = folder + "/" + from[strings.LastIndex(from, "/")+1:]
		}
		if err := memory.ValidPath(to); err != nil {
			return err
		}
		if to == from {
			return fmt.Errorf("%s is already at %s", m.Label(), to)
		}
		if other, err := a.Repo.Lookup(to); err == nil {
			return fmt.Errorf("%s is taken by %s", to, a.label(other))
		}
		if nf := a.newFolder(memory.NS(to)); nf != "" && !newNS && a.Cfg.Store.NewNSGuard {
			return fmt.Errorf("new folder %q; pass new_ns: true", nf)
		}
		m.Path = to
		if err := a.Repo.Write(m, fmt.Sprintf("move %s → %s [%s]", from, to, id)); err != nil {
			return err
		}
		a.indexed(m)
		reply = fmt.Sprintf("moved %s: %s → %s", id, from, to)
		switch was, is := a.inStartup(from), a.inStartup(to); {
		case is && !was:
			reply += "\nnow in the startup folder: new sessions load it; running ones don't"
		case was && !is:
			reply += "\nleft the startup folder: new sessions no longer load it; running ones keep it"
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	op.Since("move and commit", tw)
	// The path is part of what is embedded, so the chunks are embedded again.
	var sess *Session
	if op != nil {
		sess = op.Sess
	}
	a.embedAsync(sess, "move", id)
	return reply, nil
}

func cleanTags(in []string) []string {
	var out []string
	seen := map[string]bool{}
	for _, t := range in {
		t = strings.ToLower(strings.TrimSpace(t))
		if t != "" && !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	return out
}

// duplicate finds an existing memory saying nearly the same thing as body.
func (a *App) duplicate(path, body string) string {
	if a.HasModel() {
		parts := a.chunk(body)
		vs, err := a.embed([]string{a.Cfg.Search.DocumentPrefix + index.EmbedText(path, parts[0])}, nil)
		if err == nil && len(vs) == 1 {
			hits := a.similar(vs[0], index.Filter{}, a.Cfg.Store.DupThreshold, 1)
			if len(hits) > 0 {
				return hits[0].ID
			}
			return ""
		}
	}
	mems, _, err := a.Repo.LoadAll()
	if err != nil {
		return ""
	}
	want := termSet(body)
	bestID, best := "", 0.0
	for _, m := range mems {
		if j := jaccard(want, termSet(m.Body)); j > best {
			bestID, best = m.ID, j
		}
	}
	if best >= a.Cfg.Store.DupWordsThreshold {
		return bestID
	}
	return ""
}

func termSet(s string) map[string]bool {
	out := map[string]bool{}
	for _, t := range index.Terms(s) {
		out[t] = true
	}
	return out
}

func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	n := 0
	for t := range a {
		if b[t] {
			n++
		}
	}
	return float64(n) / float64(len(a)+len(b)-n)
}

// Delete removes a memory with git rm (undo with griz memory restore).
// Delete takes an exact id or path; callers resolve short keys first.
func (a *App) Delete(key string) (string, error) {
	key = strings.TrimSuffix(strings.TrimSpace(key), ".md")
	var label string
	err := a.write(func() error {
		id, err := a.Repo.Lookup(key)
		if err != nil {
			return err
		}
		label = a.label(id)
		if err := a.Repo.Remove([]string{id}, "delete "+label); err != nil {
			return err
		}
		a.removed(id)
		return nil
	})
	if err != nil {
		return "", err
	}
	return "ok deleted " + label, nil
}

// Duplicates lists pairs of memories that look like near-duplicates.
func (a *App) Duplicates(mems []*memory.Memory) []string {
	var out []string
	if _, err := a.EmbedMissing(nil); a.HasModel() && err == nil {
		// Compare whole memories by their mean chunk vector.
		mid := a.model.ID()
		byID := map[string][]float32{}
		a.vec.Each(mid, a.ix.Chunks(index.Filter{}), func(r index.ChunkRef, v []float32) {
			acc := byID[r.ID]
			if acc == nil {
				acc = make([]float32, len(v))
				byID[r.ID] = acc
			}
			for i := range v {
				acc[i] += v[i]
			}
		})
		for i, m := range mems {
			for _, n := range mems[i+1:] {
				x, y := byID[m.ID], byID[n.ID]
				if x != nil && y != nil {
					if c := embed.Cosine(x, y); c >= a.Cfg.Store.DupThreshold {
						out = append(out, fmt.Sprintf("%s ~ %s (meaning %.2f)", m.Label(), n.Label(), c))
					}
				}
			}
		}
		return out
	}
	sets := make([]map[string]bool, len(mems))
	for i, m := range mems {
		sets[i] = termSet(m.Body)
	}
	for i := range mems {
		for j := i + 1; j < len(mems); j++ {
			if s := jaccard(sets[i], sets[j]); s >= a.Cfg.Store.DupWordsThreshold {
				out = append(out, fmt.Sprintf("%s ~ %s (words %.2f)", mems[i].Label(), mems[j].Label(), s))
			}
		}
	}
	return out
}

// debugRecall describes a search for debug mode.
func (a *App) debugRecall(rk ranking, op *Op) {
	if op == nil || !op.Sess.Debugging() {
		return
	}
	if !rk.byVector {
		op.debugf("keyword search only: %d chunk hits (keyword %s)", rk.kwHits, ms(op.step("keyword")))
		return
	}
	op.debugf("waited %s for embedding, then searched %d chunks (keyword %s, query embed %s, vectors %s)",
		ms(op.step("embed wait")), rk.searched, ms(op.step("keyword")), ms(op.step("query embed")), ms(op.step("vectors")))
}
