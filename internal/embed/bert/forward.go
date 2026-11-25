package bert

import (
	"math"
	"runtime"
	"sync"
)

// Embed returns one vector per text. Texts longer than MaxTokens are cut.
func (m *Model) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	if len(texts) == 1 {
		// One text: split the work inside each step across CPUs.
		out[0] = m.embedOne(m.tok.encode(texts[0], m.maxTokens), workers)
		return out, nil
	}
	// Several texts: one text per CPU, each run serially (less coordination).
	parallel(len(texts), workers, func(i int) {
		out[i] = m.embedOne(m.tok.encode(texts[i], m.maxTokens), 1)
	})
	return out, nil
}

// EmbedSerial embeds one text using only the calling goroutine.
func (m *Model) EmbedSerial(text string) ([]float32, error) {
	return m.embedOne(m.tok.encode(text, m.maxTokens), 1), nil
}

// scratch holds one embedding's working buffers. Buffers are reused across
// layers and, through scratchPool, across calls, so a steady stream of
// embeddings allocates almost nothing.
type scratch struct {
	x, o, q, k, v, ctx, a, inter, scores []float32
}

var scratchPool = sync.Pool{New: func() any { return new(scratch) }}

// grow returns buf resized to n, reallocating only when it is too small.
func grow(buf []float32, n int) []float32 {
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

func (m *Model) embedOne(ids []int, par int) []float32 {
	n, H := len(ids), m.hidden
	s := scratchPool.Get().(*scratch)
	defer scratchPool.Put(s)
	s.x = grow(s.x, n*H)
	x := s.x
	for i, id := range ids {
		row := x[i*H : (i+1)*H]
		w := m.wordEmb[id*H : (id+1)*H]
		p := m.posEmb[i*H : (i+1)*H]
		for j := range row {
			row[j] = w[j] + p[j]
		}
		if m.typeEmb != nil {
			t := m.typeEmb[:H] // single-sequence input: token type 0
			for j := range row {
				row[j] += t[j]
			}
		}
	}
	m.layerNorm(x, n, m.embLN)
	for li := range m.layers {
		m.layer(&m.layers[li], s, n, par)
		s.x, s.o = s.o, s.x // this layer's output is the next one's input
	}
	x = s.x
	vec := make([]float32, H)
	if m.pooling == "cls" {
		copy(vec, x[:H])
	} else {
		for i := 0; i < n; i++ {
			for j := 0; j < H; j++ {
				vec[j] += x[i*H+j]
			}
		}
		for j := range vec {
			vec[j] /= float32(n)
		}
	}
	if m.normalize {
		var s float64
		for _, v := range vec {
			s += float64(v) * float64(v)
		}
		if s > 0 {
			inv := float32(1 / math.Sqrt(s))
			for j := range vec {
				vec[j] *= inv
			}
		}
	}
	return vec
}

// layer runs one encoder layer on s.x (n rows) and leaves the result in s.o.
func (m *Model) layer(L *layer, s *scratch, n, par int) {
	H, heads := m.hidden, m.heads
	dh := H / heads
	x := s.x
	s.q = apply(L.q, x, n, par, s.q)
	s.k = apply(L.k, x, n, par, s.k)
	s.v = apply(L.v, x, n, par, s.v)
	q, k, v := s.q, s.k, s.v
	s.ctx = grow(s.ctx, n*H)
	ctx := s.ctx
	clear(ctx) // accumulated into below
	s.scores = grow(s.scores, heads*n)
	scale := float32(1 / math.Sqrt(float64(dh)))
	parallel(heads, par, func(h int) {
		scores := s.scores[h*n : (h+1)*n]
		off := h * dh
		for i := 0; i < n; i++ {
			qi := q[i*H+off : i*H+off+dh]
			kj := func(j int) []float32 { return k[j*H+off : j*H+off+dh] }
			j := 0
			for ; j+4 <= n; j += 4 { // four keys per pass: one lane reduction each, not one per dot
				scores[j], scores[j+1], scores[j+2], scores[j+3] = dot4(kj(j), kj(j+1), kj(j+2), kj(j+3), qi)
			}
			for ; j < n; j++ {
				scores[j] = dot(qi, kj(j))
			}
			mx := float32(math.Inf(-1))
			for j := range scores {
				scores[j] *= scale
				mx = max(mx, scores[j])
			}
			var sum float32
			for j := range scores {
				e := float32(math.Exp(float64(scores[j] - mx)))
				scores[j] = e
				sum += e
			}
			ci := ctx[i*H+off : i*H+off+dh]
			inv := 1 / sum
			for j := 0; j < n; j++ {
				axpy(scores[j]*inv, v[j*H+off:j*H+off+dh], ci)
			}
		}
	})
	s.a = apply(L.attnOut, ctx, n, par, s.a)
	a := s.a
	for i := range a {
		a[i] += x[i]
	}
	m.layerNorm(a, n, L.ln1)
	s.inter = apply(L.inter, a, n, par, s.inter)
	inter := s.inter
	m.activate(inter)
	s.o = apply(L.out, inter, n, par, s.o)
	o := s.o
	for i := range o {
		o[i] += a[i]
	}
	m.layerNorm(o, n, L.ln2)
}

func (m *Model) layerNorm(x []float32, n int, ln layerNorm) {
	H := m.hidden
	for i := 0; i < n; i++ {
		row := x[i*H : (i+1)*H]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(H)
		var vr float64
		for _, v := range row {
			d := float64(v) - mean
			vr += d * d
		}
		inv := 1 / math.Sqrt(vr/float64(H)+float64(m.eps))
		for j, v := range row {
			row[j] = float32((float64(v)-mean)*inv)*ln.w[j] + ln.b[j]
		}
	}
}

// apply computes x·Wᵀ + b for n rows into dst (resized as needed) and
// returns it, splitting output columns across CPUs. Four input rows are
// processed per pass so each weight row is read once per four.
func apply(l linear, x []float32, n, par int, dst []float32) []float32 {
	y := grow(dst, n*l.o)
	const block = 16
	nb := (l.o + block - 1) / block
	in := l.in
	parallel(nb, par, func(bi int) {
		for o := bi * block; o < min((bi+1)*block, l.o); o++ {
			w := l.w[o*in : (o+1)*in]
			b := l.b[o]
			i := 0
			for ; i+4 <= n; i += 4 {
				s0, s1, s2, s3 := dot4(x[i*in:(i+1)*in], x[(i+1)*in:(i+2)*in], x[(i+2)*in:(i+3)*in], x[(i+3)*in:(i+4)*in], w)
				y[i*l.o+o] = s0 + b
				y[(i+1)*l.o+o] = s1 + b
				y[(i+2)*l.o+o] = s2 + b
				y[(i+3)*l.o+o] = s3 + b
			}
			for ; i < n; i++ {
				y[i*l.o+o] = dot(x[i*in:(i+1)*in], w) + b
			}
		}
	})
	return y
}

var workers = runtime.GOMAXPROCS(0)

// parallel runs fn(0..n-1) on up to par goroutines.
func parallel(n, par int, fn func(int)) {
	w := min(par, n)
	if w <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Add(w)
	for g := 0; g < w; g++ {
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}
