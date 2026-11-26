package bert

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"griz/internal/embed"
)

// erf32 against math.Erf over the whole useful range (and past the clamp).
func TestErf32Accuracy(t *testing.T) {
	var worst, at float64
	for x := -6.0; x <= 6.0; x += 1e-5 {
		f := float32(x)
		e := math.Abs(float64(erf32(f)) - math.Erf(float64(f)))
		if e > worst {
			worst, at = e, x
		}
	}
	t.Logf("erf32: max abs error %.3g at x=%.5f", worst, at)
	if worst > 5e-7 {
		t.Errorf("erf32 max abs error %.3g at %.5f; want float32 precision (about 1e-7)", worst, at)
	}
}

// The build's geluErf (SIMD or not) against the scalar erf32 path and exact GELU.
func TestGELUMatches(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	x := make([]float32, 1003) // not a multiple of any vector width
	for i := range x {
		x[i] = float32(r.NormFloat64() * 3)
	}
	fast, scalar := append([]float32(nil), x...), append([]float32(nil), x...)
	geluErf(fast)
	geluErfScalar(scalar)
	var vsScalar, vsExact float64
	for i, u := range x {
		exact := 0.5 * float64(u) * (1 + math.Erf(float64(u)/math.Sqrt2))
		vsScalar = max(vsScalar, math.Abs(float64(fast[i]-scalar[i])))
		vsExact = max(vsExact, math.Abs(float64(fast[i])-exact))
	}
	t.Logf("gelu (simd build %v): max abs diff vs scalar %.3g, vs exact %.3g", simdBuild, vsScalar, vsExact)
	if vsScalar > 1e-6 || vsExact > 2e-6 {
		t.Errorf("gelu differs: vs scalar %.3g, vs exact %.3g", vsScalar, vsExact)
	}
}

func TestAxpyMatchesScalar(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, n := range []int{1, 3, 4, 31, 32, 33, 384} {
		x, y := make([]float32, n), make([]float32, n)
		for i := range x {
			x[i], y[i] = r.Float32()-0.5, r.Float32()-0.5
		}
		got, want := append([]float32(nil), y...), append([]float32(nil), y...)
		axpy(0.37, x, got)
		axpyScalar(0.37, x, want)
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 1e-6 {
				t.Fatalf("n=%d: element %d is %v, want %v", n, i, got[i], want[i])
			}
		}
	}
}

// The fast GELU must leave real MiniLM embeddings essentially unchanged.
func TestFastGELUKeepsEmbeddings(t *testing.T) {
	m, err := Load(miniLM(t))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = m.Close() }()
	texts := []string{
		"the cat upstairs meowing",
		"Our project partner is Contoso.",
		"**2025-03-14 09:12** — Called the library about the late fee; they waived it.",
		strings.Repeat("Prefer short sentences and plain words in anything written for others. ", 10),
		"go test -race ./... passes; GOEXPERIMENT=simd halves embedding time",
	}
	fast, _ := m.Embed(texts)
	m.exactErf = true
	exact, _ := m.Embed(texts)
	m.exactErf = false
	worst := 1.0
	for i := range texts {
		worst = min(worst, float64(embed.Cosine(fast[i], exact[i])))
	}
	t.Logf("fast vs exact GELU (simd build %v): lowest cosine %.9f", simdBuild, worst)
	if worst < 0.99999 {
		t.Errorf("fast GELU changes embeddings: lowest cosine %.9f, want ≥ 0.99999", worst)
	}
}

// hidden_act picks the activation, and its implementation is in the identity.
func TestHiddenActDispatch(t *testing.T) {
	src := miniLM(t)
	ids := map[string]string{}
	for _, act := range []string{"gelu", "gelu_new", "gelu_pytorch_tanh", "relu"} {
		dir := t.TempDir()
		for _, f := range []string{"tokenizer.json", "model.safetensors", "modules.json", "1_Pooling", "sentence_bert_config.json"} {
			if _, err := os.Stat(filepath.Join(src, f)); err == nil {
				if err := os.Symlink(filepath.Join(src, f), filepath.Join(dir, f)); err != nil {
					t.Fatal(err)
				}
			}
		}
		raw, _ := os.ReadFile(filepath.Join(src, "config.json"))
		var c map[string]any
		if err := json.Unmarshal(raw, &c); err != nil {
			t.Fatal(err)
		}
		c["hidden_act"] = act
		raw, _ = json.Marshal(c)
		if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o644); err != nil {
			t.Fatal(err)
		}
		info, load, err := probe(dir, "")
		if act == "relu" {
			if err == nil || !strings.Contains(err.Error(), "relu") {
				t.Errorf("relu should be refused: %v", err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", act, err)
		}
		ids[act] = info.ID
		em, err := load()
		if err != nil {
			t.Fatal(err)
		}
		m := em.(*Model)
		if want := actKind[act]; m.act != want {
			t.Errorf("%s: activation %d, want %d", act, m.act, want)
		}
		if err := m.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if ids["gelu"] == ids["gelu_new"] {
		t.Errorf("a different activation must change the model identity")
	}
	// Changing the implementation version changes the identity too.
	old := actImpl["gelu"]
	actImpl["gelu"] = old + "-next"
	defer func() { actImpl["gelu"] = old }()
	info, _, _ := probe(src, "")
	actImpl["gelu"] = old
	again, _, _ := probe(src, "")
	if info.ID == again.ID {
		t.Errorf("the activation implementation version must be part of the identity")
	}
}

// geluTanh is the tanh formula itself.
func TestGELUTanh(t *testing.T) {
	x := []float32{-3, -1, -0.5, 0, 0.5, 1, 3}
	got := append([]float32(nil), x...)
	geluTanh(got)
	for i, u := range x {
		v := float64(u)
		want := 0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v)))
		if math.Abs(float64(got[i])-want) > 1e-6 {
			t.Errorf("geluTanh(%v) = %v, want %v", u, got[i], want)
		}
	}
}
