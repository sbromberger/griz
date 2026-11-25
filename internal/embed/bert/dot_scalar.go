package bert

// Plain Go dot products. They are always compiled: the default build uses
// them, and the simd build's test compares against them.

// dot4Scalar computes four dot products against the same w.
func dot4Scalar(a0, a1, a2, a3, w []float32) (float32, float32, float32, float32) {
	n := len(w)
	a0, a1, a2, a3 = a0[:n], a1[:n], a2[:n], a3[:n]
	var s0, s1, s2, s3, t0, t1, t2, t3 float32
	k := 0
	for ; k+2 <= n; k += 2 {
		w0, w1 := w[k], w[k+1]
		s0 += a0[k] * w0
		s1 += a1[k] * w0
		s2 += a2[k] * w0
		s3 += a3[k] * w0
		t0 += a0[k+1] * w1
		t1 += a1[k+1] * w1
		t2 += a2[k+1] * w1
		t3 += a3[k+1] * w1
	}
	for ; k < n; k++ {
		s0 += a0[k] * w[k]
		s1 += a1[k] * w[k]
		s2 += a2[k] * w[k]
		s3 += a3[k] * w[k]
	}
	return s0 + t0, s1 + t1, s2 + t2, s3 + t3
}

func dotScalar(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+8 <= len(a); i += 8 {
		s0 += a[i]*b[i] + a[i+4]*b[i+4]
		s1 += a[i+1]*b[i+1] + a[i+5]*b[i+5]
		s2 += a[i+2]*b[i+2] + a[i+6]*b[i+6]
		s3 += a[i+3]*b[i+3] + a[i+7]*b[i+7]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}
