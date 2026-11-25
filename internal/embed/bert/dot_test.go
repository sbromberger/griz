package bert

import (
	"math"
	"math/rand"
	"testing"
)

// The simd and plain versions must agree (run the tests with and without
// GOEXPERIMENT=simd; without it they are the same code).
func TestDotMatchesScalar(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	vec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(r.NormFloat64())
		}
		return v
	}
	close := func(a, b float32) bool {
		return math.Abs(float64(a-b)) <= 1e-4*math.Max(1, math.Abs(float64(b)))
	}
	for _, n := range []int{0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 384, 1536} {
		a0, a1, a2, a3, w := vec(n), vec(n), vec(n), vec(n), vec(n)
		if got, want := dot(a0, w), dotScalar(a0, w); !close(got, want) {
			t.Errorf("dot n=%d: %v, scalar %v", n, got, want)
		}
		g0, g1, g2, g3 := dot4(a0, a1, a2, a3, w)
		w0, w1, w2, w3 := dot4Scalar(a0, a1, a2, a3, w)
		if !close(g0, w0) || !close(g1, w1) || !close(g2, w2) || !close(g3, w3) {
			t.Errorf("dot4 n=%d: %v %v %v %v, scalar %v %v %v %v", n, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
	t.Logf("simd build: %v", simdBuild)
}

func BenchmarkDot384(b *testing.B) {
	x, y := make([]float32, 384), make([]float32, 384)
	for i := range x {
		x[i], y[i] = float32(i%7), float32(i%5)
	}
	var s float32
	for b.Loop() {
		s += dot(x, y)
	}
	_ = s
}

func BenchmarkDot4x384(b *testing.B) {
	x, w := make([]float32, 4*384), make([]float32, 384)
	var s float32
	for b.Loop() {
		r0, r1, r2, r3 := dot4(x[:384], x[384:768], x[768:1152], x[1152:], w)
		s += r0 + r1 + r2 + r3
	}
	_ = s
}
