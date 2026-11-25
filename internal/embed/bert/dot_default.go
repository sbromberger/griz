//go:build !goexperiment.simd

package bert

// Without GOEXPERIMENT=simd the matmul uses the plain Go loops.

func dot4(a0, a1, a2, a3, w []float32) (float32, float32, float32, float32) {
	return dot4Scalar(a0, a1, a2, a3, w)
}

func dot(a, b []float32) float32 { return dotScalar(a, b) }

const simdBuild = false

func geluErf(x []float32) { geluErfScalar(x) }

func axpy(a float32, x, y []float32) { axpyScalar(a, x, y) }
