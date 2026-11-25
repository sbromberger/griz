package bert

import "math"

// Activations, from config.json hidden_act.
//
//   - "gelu" (MiniLM and most BERTs): exact GELU, x/2·(1+erf(x/√2)). erf is a
//     float32 rational approximation (erf32), accurate to float32 precision;
//     the tanh formula is not used here because it drifts from what the
//     model was trained with.
//   - "gelu_new", "gelu_pytorch_tanh": the tanh approximation those models
//     were trained with.
const (
	actGELU = iota
	actGELUTanh
)

// actImpl names each activation's implementation. It is part of the model
// identity, so changing how an activation is computed (and so, slightly,
// the vectors) makes griz re-embed. Bump the version on any change.
var actImpl = map[string]string{
	"gelu":              "gelu:erf32-rational-v1",
	"gelu_new":          "gelu-tanh:v1",
	"gelu_pytorch_tanh": "gelu-tanh:v1",
}

var actKind = map[string]int{"gelu": actGELU, "gelu_new": actGELUTanh, "gelu_pytorch_tanh": actGELUTanh}

// activate applies the model's activation to x in place.
func (m *Model) activate(x []float32) {
	switch {
	case m.act == actGELUTanh:
		geluTanh(x)
	case m.exactErf: // tests only: the reference the fast path is checked against
		for i, u := range x {
			x[i] = float32(0.5 * float64(u) * (1 + math.Erf(float64(u)/math.Sqrt2)))
		}
	default:
		geluErf(x)
	}
}

// erf32 coefficients: erf(x) ≈ x·P(x²)/Q(x²) on [-4, 4] (beyond that erf is
// ±1 to float32 precision). This is the rational approximation used by
// Eigen and XLA for float32 erf.
const (
	erfA1  = -1.60960333262415e-02
	erfA3  = -2.95459980854025e-03
	erfA5  = -7.34990630326855e-04
	erfA7  = -5.69250639462346e-05
	erfA9  = -2.10102402082508e-06
	erfA11 = 2.77068142495902e-08
	erfA13 = -2.72614225801306e-10
	erfB0  = -1.42647390514189e-02
	erfB2  = -7.37332916720468e-03
	erfB4  = -1.68282697438203e-03
	erfB6  = -2.13374055278905e-04
	erfB8  = -1.45660718464996e-05
)

func erf32(x float32) float32 {
	x = min(max(x, -4), 4)
	x2 := x * x
	p := x2*erfA13 + erfA11
	p = p*x2 + erfA9
	p = p*x2 + erfA7
	p = p*x2 + erfA5
	p = p*x2 + erfA3
	p = p*x2 + erfA1
	q := x2*erfB8 + erfB6
	q = q*x2 + erfB4
	q = q*x2 + erfB2
	q = q*x2 + erfB0
	return x * p / q
}

// geluErfScalar is exact-GELU with erf32, one value at a time.
func geluErfScalar(x []float32) {
	for i, u := range x {
		x[i] = 0.5 * u * (1 + erf32(u*math.Sqrt2/2))
	}
}

// geluTanh is the tanh approximation, for models trained with it.
func geluTanh(x []float32) {
	const c = 0.7978845608028654 // √(2/π)
	for i, u := range x {
		v := float64(u)
		x[i] = float32(0.5 * v * (1 + math.Tanh(c*(v+0.044715*v*v*v))))
	}
}

// axpyScalar adds a·x to y.
func axpyScalar(a float32, x, y []float32) {
	x = x[:len(y)]
	for i := range y {
		y[i] += a * x[i]
	}
}
