// Package bert runs BERT-style sentence-transformers models (all-MiniLM,
// bge-small, e5-small, ...) in pure Go. Weights are memory-mapped from
// model.safetensors and used in place, so griz processes share one copy.
package bert

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"griz/internal/app"
	"griz/internal/embed"
)

func init() { app.ProbeModel = probe }

type linear struct {
	w     []float32 // [out][in], row-major
	b     []float32
	in, o int
}

type layerNorm struct{ w, b []float32 }

type layer struct {
	q, k, v, attnOut linear
	ln1              layerNorm
	inter, out       linear
	ln2              layerNorm
}

type Model struct {
	name, id  string
	hidden    int
	heads     int
	eps       float32
	maxTokens int
	pooling   string // "mean" or "cls"
	normalize bool
	act       int  // actGELU or actGELUTanh (act.go)
	exactErf  bool // tests only: GELU with math.Erf instead of erf32
	tok       *tokenizer
	wordEmb   []float32
	posEmb    []float32
	typeEmb   []float32
	embLN     layerNorm
	layers    []layer
	data      []byte // mmapped weights file
}

var _ embed.Model = (*Model)(nil)

func (m *Model) ID() string     { return m.id }
func (m *Model) Name() string   { return m.name }
func (m *Model) Dims() int      { return m.hidden }
func (m *Model) MaxTokens() int { return m.maxTokens }

type bertConfig struct {
	ModelType  string  `json:"model_type"`
	Hidden     int     `json:"hidden_size"`
	Layers     int     `json:"num_hidden_layers"`
	Heads      int     `json:"num_attention_heads"`
	Inter      int     `json:"intermediate_size"`
	Act        string  `json:"hidden_act"`
	Eps        float64 `json:"layer_norm_eps"`
	MaxPos     int     `json:"max_position_embeddings"`
	PosEmbType string  `json:"position_embedding_type"`
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

func need(dir, name string) (string, error) {
	p := filepath.Join(dir, name)
	if _, err := os.Stat(p); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return "", fmt.Errorf("%s missing", name)
		}
		return "", err
	}
	return p, nil
}

// probe checks a sentence-transformers model folder without reading the
// weights, and returns the model's description and a loader for it. The
// identity covers config.json, tokenizer.json, the safetensors header, the
// weights file size, docPrefix (which changes what gets embedded) and the
// activation's implementation version (actImpl).
func probe(dir, docPrefix string) (embed.Info, embed.Loader, error) {
	var info embed.Info
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		return info, nil, fmt.Errorf("folder %s not found", dir)
	}
	cfgPath, err := need(dir, "config.json")
	if err != nil {
		return info, nil, err
	}
	tokPath, err := need(dir, "tokenizer.json")
	if err != nil {
		return info, nil, err
	}
	wPath, err := need(dir, "model.safetensors")
	if err != nil {
		return info, nil, err
	}
	cfgRaw, err := os.ReadFile(cfgPath)
	if err != nil {
		return info, nil, err
	}
	var c bertConfig
	if err := json.Unmarshal(cfgRaw, &c); err != nil {
		return info, nil, fmt.Errorf("config.json: %w", err)
	}
	switch {
	case c.ModelType != "bert":
		return info, nil, fmt.Errorf("model_type %q is not supported (BERT only)", c.ModelType)
	case actImpl[c.Act] == "":
		return info, nil, fmt.Errorf("hidden_act %q is not supported (gelu, gelu_new or gelu_pytorch_tanh)", c.Act)
	case c.PosEmbType != "" && c.PosEmbType != "absolute":
		return info, nil, fmt.Errorf("position_embedding_type %q is not supported", c.PosEmbType)
	case c.Hidden == 0 || c.Heads == 0 || c.Hidden%c.Heads != 0 || c.MaxPos == 0:
		return info, nil, fmt.Errorf("config.json: bad hidden_size, num_attention_heads or max_position_embeddings")
	}
	tokRaw, err := os.ReadFile(tokPath)
	if err != nil {
		return info, nil, err
	}
	tokMax, err := probeTokenizer(tokRaw)
	if err != nil {
		return info, nil, err
	}
	pooling, normalize, err := readPooling(dir)
	if err != nil {
		return info, nil, err
	}
	hdr, size, err := safetensorsHeader(wPath)
	if err != nil {
		return info, nil, err
	}
	h := sha256.New()
	for _, b := range [][]byte{cfgRaw, tokRaw, hdr, []byte(fmt.Sprint(size)), []byte(docPrefix), []byte(actImpl[c.Act])} {
		_, _ = fmt.Fprintf(h, "%d:", len(b))
		h.Write(b)
	}
	info = embed.Info{
		Name:      modelName(dir),
		Dims:      c.Hidden,
		MaxTokens: maxTokens(dir, c.MaxPos, tokMax),
		Bytes:     size,
	}
	info.ID = info.Name + "@" + hex.EncodeToString(h.Sum(nil))[:16]
	load := func() (embed.Model, error) {
		tok, err := loadTokenizer(tokPath)
		if err != nil {
			return nil, err
		}
		m := &Model{name: info.Name, id: info.ID, hidden: c.Hidden, heads: c.Heads, eps: float32(c.Eps),
			tok: tok, maxTokens: info.MaxTokens, pooling: pooling, normalize: normalize, act: actKind[c.Act]}
		if m.eps == 0 {
			m.eps = 1e-12
		}
		if err := m.loadWeights(wPath, c); err != nil {
			return nil, err
		}
		return m, nil
	}
	return info, load, nil
}

// Load probes and loads a model folder in one step.
func Load(dir string) (*Model, error) {
	_, load, err := probe(dir, "")
	if err != nil {
		return nil, err
	}
	m, err := load()
	if err != nil {
		return nil, err
	}
	return m.(*Model), nil
}

// Close unmaps the weights. The model must not be used afterwards.
func (m *Model) Close() error {
	if m.data == nil {
		return nil
	}
	err := syscall.Munmap(m.data)
	m.data, m.wordEmb, m.posEmb, m.typeEmb, m.layers, m.tok = nil, nil, nil, nil, nil, nil
	return err
}

func safetensorsHeader(path string) ([]byte, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = f.Close() }()
	st, err := f.Stat()
	if err != nil {
		return nil, 0, err
	}
	var nb [8]byte
	if _, err := io.ReadFull(f, nb[:]); err != nil {
		return nil, 0, fmt.Errorf("model.safetensors: %w", err)
	}
	n := binary.LittleEndian.Uint64(nb[:])
	if n > uint64(st.Size()-8) || n > 100<<20 {
		return nil, 0, errors.New("model.safetensors: bad header")
	}
	hdr := make([]byte, n)
	if _, err := io.ReadFull(f, hdr); err != nil {
		return nil, 0, fmt.Errorf("model.safetensors: %w", err)
	}
	return hdr, st.Size(), nil
}

// modelName gives "all-MiniLM-L6-v2" for a Hugging Face cache snapshot path,
// otherwise the folder name.
func modelName(dir string) string {
	for part := range strings.SplitSeq(filepath.ToSlash(filepath.Clean(dir)), "/") {
		if rest, ok := strings.CutPrefix(part, "models--"); ok {
			if i := strings.LastIndex(rest, "--"); i >= 0 {
				return rest[i+2:]
			}
			return rest
		}
	}
	return filepath.Base(dir)
}

// maxTokens: sentence_bert_config.json max_seq_length, else the tokenizer's
// truncation length, else the position-embedding count.
func maxTokens(dir string, maxPos, tokMax int) int {
	var sb struct {
		MaxSeqLength int `json:"max_seq_length"`
	}
	n := maxPos
	if tokMax > 0 && tokMax < n {
		n = tokMax
	}
	if readJSON(filepath.Join(dir, "sentence_bert_config.json"), &sb) == nil && sb.MaxSeqLength > 0 {
		n = sb.MaxSeqLength
	}
	if n > maxPos {
		n = maxPos
	}
	return n
}

// readPooling reads modules.json and the pooling config. With neither,
// sentence-transformers' default applies: mean pooling, no normalization.
func readPooling(dir string) (pooling string, normalize bool, err error) {
	pooling = "mean"
	var mods []struct {
		Path string `json:"path"`
		Type string `json:"type"`
	}
	poolDir := "1_Pooling"
	if readJSON(filepath.Join(dir, "modules.json"), &mods) == nil {
		for _, md := range mods {
			switch {
			case strings.HasSuffix(md.Type, ".Pooling"):
				poolDir = md.Path
			case strings.HasSuffix(md.Type, ".Normalize"):
				normalize = true
			case strings.HasSuffix(md.Type, ".Transformer"):
			default:
				return "", false, fmt.Errorf("modules.json: module %s is not supported", md.Type)
			}
		}
	}
	var p struct {
		CLS  bool `json:"pooling_mode_cls_token"`
		Mean bool `json:"pooling_mode_mean_tokens"`
		Max  bool `json:"pooling_mode_max_tokens"`
	}
	if readJSON(filepath.Join(dir, poolDir, "config.json"), &p) == nil {
		switch {
		case p.CLS && !p.Mean && !p.Max:
			pooling = "cls"
		case p.Mean && !p.CLS && !p.Max:
			pooling = "mean"
		default:
			return "", false, fmt.Errorf("pooling config: only one of cls or mean pooling is supported")
		}
	}
	return pooling, normalize, nil
}

func (m *Model) loadWeights(path string, c bertConfig) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	st, err := f.Stat()
	if err != nil {
		return err
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	m.data = data
	fail := func(err error) error { _ = syscall.Munmap(data); m.data = nil; return err }
	if len(data) < 8 {
		return fail(errors.New("model.safetensors: too short"))
	}
	n := binary.LittleEndian.Uint64(data)
	if n > uint64(len(data)-8) {
		return fail(errors.New("model.safetensors: bad header"))
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(data[8:8+n], &hdr); err != nil {
		return fail(fmt.Errorf("model.safetensors: %w", err))
	}
	base := 8 + int(n)
	type info struct {
		Dtype   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	prefix := ""
	if _, ok := hdr["bert.embeddings.word_embeddings.weight"]; ok {
		prefix = "bert."
	}
	var lerr error
	get := func(name string, shape ...int) []float32 {
		if lerr != nil {
			return nil
		}
		raw, ok := hdr[prefix+name]
		if !ok {
			lerr = fmt.Errorf("model.safetensors: %s missing", name)
			return nil
		}
		var t info
		_ = json.Unmarshal(raw, &t)
		if t.Dtype != "F32" {
			lerr = fmt.Errorf("model.safetensors: %s is %s; only F32 is supported", name, t.Dtype)
			return nil
		}
		want := 1
		for _, d := range shape {
			want *= d
		}
		got := 1
		for _, d := range t.Shape {
			got *= d
		}
		off, end := base+t.Offsets[0], base+t.Offsets[1]
		if got != want || end-off != 4*want || end > len(data) || off%4 != 0 {
			lerr = fmt.Errorf("model.safetensors: %s has shape %v, want %v", name, t.Shape, shape)
			return nil
		}
		return unsafe.Slice((*float32)(unsafe.Pointer(&data[off])), want)
	}
	lin := func(name string, out, in int) linear {
		return linear{w: get(name+".weight", out, in), b: get(name+".bias", out), in: in, o: out}
	}
	ln := func(name string) layerNorm {
		return layerNorm{w: get(name+".weight", c.Hidden), b: get(name+".bias", c.Hidden)}
	}
	rows := func(name string) int { // first dimension of a tensor, 0 if absent
		var t info
		if raw, ok := hdr[prefix+name]; ok && json.Unmarshal(raw, &t) == nil && len(t.Shape) == 2 {
			return t.Shape[0]
		}
		return 0
	}
	H := c.Hidden
	vocab := rows("embeddings.word_embeddings.weight")
	m.wordEmb = get("embeddings.word_embeddings.weight", vocab, H)
	m.posEmb = get("embeddings.position_embeddings.weight", c.MaxPos, H)
	if n := rows("embeddings.token_type_embeddings.weight"); n > 0 {
		m.typeEmb = get("embeddings.token_type_embeddings.weight", n, H)
	}
	m.embLN = ln("embeddings.LayerNorm")
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		m.layers = append(m.layers, layer{
			q:       lin(p+"attention.self.query", H, H),
			k:       lin(p+"attention.self.key", H, H),
			v:       lin(p+"attention.self.value", H, H),
			attnOut: lin(p+"attention.output.dense", H, H),
			ln1:     ln(p + "attention.output.LayerNorm"),
			inter:   lin(p+"intermediate.dense", c.Inter, H),
			out:     lin(p+"output.dense", H, c.Inter),
			ln2:     ln(p + "output.LayerNorm"),
		})
	}
	if lerr != nil {
		return fail(lerr)
	}
	for _, id := range m.tok.vocab {
		if id >= vocab {
			return fail(fmt.Errorf("tokenizer vocab is larger than the model's"))
		}
	}
	return nil
}
