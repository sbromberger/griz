//go:build goexperiment.simd

package bert

import (
	"math"
	"simd"
)

// With GOEXPERIMENT=simd the matmul inner loops use the portable simd
// package (NEON on arm64, AVX on amd64), with vector width from Len().

const simdBuild = true

// sum adds a vector's lanes (the 1.27 API has no ReduceSum).
func sum(v simd.Float32s) float32 {
	var buf [16]float32 // wide enough for any vector length Go supports
	n := v.Len()
	v.Store(buf[:n])
	var s float32
	for _, x := range buf[:n] {
		s += x
	}
	return s
}

// dot4 computes four dot products against the same w, two accumulators
// per row so eight multiply-adds are in flight.
func dot4(a0, a1, a2, a3, w []float32) (float32, float32, float32, float32) {
	n := len(w)
	a0, a1, a2, a3 = a0[:n], a1[:n], a2[:n], a3[:n]
	var z simd.Float32s
	L := z.Len()
	var s0, s1, s2, s3, t0, t1, t2, t3 simd.Float32s
	k := 0
	for ; k+2*L <= n; k += 2 * L {
		w0 := simd.LoadFloat32s(w[k : k+L])
		w1 := simd.LoadFloat32s(w[k+L : k+2*L])
		s0 = simd.LoadFloat32s(a0[k:k+L]).MulAdd(w0, s0)
		s1 = simd.LoadFloat32s(a1[k:k+L]).MulAdd(w0, s1)
		s2 = simd.LoadFloat32s(a2[k:k+L]).MulAdd(w0, s2)
		s3 = simd.LoadFloat32s(a3[k:k+L]).MulAdd(w0, s3)
		t0 = simd.LoadFloat32s(a0[k+L:k+2*L]).MulAdd(w1, t0)
		t1 = simd.LoadFloat32s(a1[k+L:k+2*L]).MulAdd(w1, t1)
		t2 = simd.LoadFloat32s(a2[k+L:k+2*L]).MulAdd(w1, t2)
		t3 = simd.LoadFloat32s(a3[k+L:k+2*L]).MulAdd(w1, t3)
	}
	r0, r1, r2, r3 := sum(s0.Add(t0)), sum(s1.Add(t1)), sum(s2.Add(t2)), sum(s3.Add(t3))
	for ; k < n; k++ {
		r0 += a0[k] * w[k]
		r1 += a1[k] * w[k]
		r2 += a2[k] * w[k]
		r3 += a3[k] * w[k]
	}
	return r0, r1, r2, r3
}

// dot is one dot product with four accumulators.
func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var z simd.Float32s
	L := z.Len()
	var s0, s1, s2, s3 simd.Float32s
	i := 0
	for ; i+4*L <= len(a); i += 4 * L {
		s0 = simd.LoadFloat32s(a[i:i+L]).MulAdd(simd.LoadFloat32s(b[i:i+L]), s0)
		s1 = simd.LoadFloat32s(a[i+L:i+2*L]).MulAdd(simd.LoadFloat32s(b[i+L:i+2*L]), s1)
		s2 = simd.LoadFloat32s(a[i+2*L:i+3*L]).MulAdd(simd.LoadFloat32s(b[i+2*L:i+3*L]), s2)
		s3 = simd.LoadFloat32s(a[i+3*L:i+4*L]).MulAdd(simd.LoadFloat32s(b[i+3*L:i+4*L]), s3)
	}
	for ; i+L <= len(a); i += L {
		s0 = simd.LoadFloat32s(a[i:i+L]).MulAdd(simd.LoadFloat32s(b[i:i+L]), s0)
	}
	r := sum(s0.Add(s1).Add(s2.Add(s3)))
	for ; i < len(a); i++ {
		r += a[i] * b[i]
	}
	return r
}

// geluErf applies exact-GELU with the erf32 rational approximation, a
// vector of values at a time (same formula as erf32).
func geluErf(x []float32) {
	var z simd.Float32s
	L := z.Len()
	b := func(v float32) simd.Float32s { return simd.BroadcastFloat32s(v) }
	lo, hi, rs2, half, one := b(-4), b(4), b(math.Sqrt2/2), b(0.5), b(1)
	a13, a11, a9, a7, a5, a3, a1 := b(erfA13), b(erfA11), b(erfA9), b(erfA7), b(erfA5), b(erfA3), b(erfA1)
	b8, b6, b4, b2, b0 := b(erfB8), b(erfB6), b(erfB4), b(erfB2), b(erfB0)
	i := 0
	for ; i+L <= len(x); i += L {
		u := simd.LoadFloat32s(x[i : i+L])
		t := u.Mul(rs2).Max(lo).Min(hi)
		t2 := t.Mul(t)
		p := t2.MulAdd(a13, a11)
		p = p.MulAdd(t2, a9)
		p = p.MulAdd(t2, a7)
		p = p.MulAdd(t2, a5)
		p = p.MulAdd(t2, a3)
		p = p.MulAdd(t2, a1)
		q := t2.MulAdd(b8, b6)
		q = q.MulAdd(t2, b4)
		q = q.MulAdd(t2, b2)
		q = q.MulAdd(t2, b0)
		erf := t.Mul(p).Div(q)
		u.Mul(half).Mul(one.Add(erf)).Store(x[i : i+L])
	}
	geluErfScalar(x[i:])
}

// axpy adds a·x to y.
func axpy(a float32, x, y []float32) {
	x = x[:len(y)]
	var z simd.Float32s
	L := z.Len()
	av := simd.BroadcastFloat32s(a)
	i := 0
	for ; i+L <= len(y); i += L {
		simd.LoadFloat32s(x[i:i+L]).MulAdd(av, simd.LoadFloat32s(y[i:i+L])).Store(y[i : i+L])
	}
	for ; i < len(y); i++ {
		y[i] += a * x[i]
	}
}
