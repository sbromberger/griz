// Package embed turns text into vectors with a locally loaded model.
package embed

import "math"

// Model is any embedding model griz can run.
type Model interface {
	ID() string     // name plus a hash of the weights; vectors from different IDs are never compared
	Name() string   // short display name
	Dims() int      // vector size
	MaxTokens() int // maximum input length in model tokens
	Embed(texts []string) ([][]float32, error)
}

// Cosine similarity of two vectors (normalized or not).
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / math.Sqrt(na*nb)
}
