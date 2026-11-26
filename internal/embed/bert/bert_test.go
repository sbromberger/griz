package bert

import (
	"os"
	"path/filepath"
	"testing"

	"griz/internal/embed"
)

func miniLM(t *testing.T) string { return miniLMB(t) }

func miniLMB(t testing.TB) string {
	dirs, _ := filepath.Glob(filepath.Join(os.Getenv("HOME"), ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"))
	if len(dirs) == 0 {
		t.Skip("all-MiniLM-L6-v2 not in the Hugging Face cache")
	}
	return dirs[0]
}

func TestTokenizer(t *testing.T) {
	tok, err := loadTokenizer(filepath.Join(miniLM(t), "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	// "Héllo, world!" -> [CLS] hello , world ! [SEP]
	ids := tok.encode("Héllo, world!", 128)
	want := []int{101, 7592, 1010, 2088, 999, 102}
	if len(ids) != len(want) {
		t.Fatalf("got %v want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("got %v want %v", ids, want)
		}
	}
	if n := len(tok.encode(string(make([]byte, 0))+repeat("word ", 500), 128)); n != 128 {
		t.Fatalf("truncation: %d", n)
	}
}

func repeat(s string, n int) string {
	out := ""
	for range n {
		out += s
	}
	return out
}

func TestProbeAndEmbed(t *testing.T) {
	dir := miniLM(t)
	info, load, err := probe(dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if info.Name != "all-MiniLM-L6-v2" || info.Dims != 384 || info.MaxTokens != 128 {
		t.Fatalf("info: %+v", info)
	}
	if i2, _, _ := probe(dir, "search_document: "); i2.ID == info.ID {
		t.Fatal("document prefix must change the model identity")
	}
	m, err := load()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = m.(*Model).Close() }()
	vs, _ := m.Embed([]string{"Our project partner is Contoso.", "the Contoso folks", "My favorite soup is minestrone."})
	near, far := embed.Cosine(vs[0], vs[1]), embed.Cosine(vs[0], vs[2])
	if near <= far || near < 0.4 {
		t.Fatalf("similar texts should score higher: near %.3f far %.3f", near, far)
	}
}

func TestProbeErrors(t *testing.T) {
	if _, _, err := probe(t.TempDir(), ""); err == nil || err.Error() != "config.json missing" {
		t.Fatalf("got %v", err)
	}
}

// Scratch buffers are reused across calls and layers; a stale or shared
// buffer would make the same text embed differently depending on what ran
// before it, or on how the work was split.
func TestBufferReuseIsInvisible(t *testing.T) {
	_, load, err := probe(miniLM(t), "")
	if err != nil {
		t.Fatal(err)
	}
	e, err := load()
	if err != nil {
		t.Fatal(err)
	}
	m := e.(*Model)
	defer func() { _ = m.Close() }()
	short, long := "the cat upstairs meowing", repeat("a much longer chunk of text about many things ", 20)
	first, _ := m.EmbedSerial(short)
	if _, err := m.EmbedSerial(long); err != nil {
		t.Fatal(err)
	}
	again, _ := m.EmbedSerial(short)
	par, _ := m.Embed([]string{short}) // one text: work split inside each step
	many, _ := m.Embed([]string{long, short, long, short})
	for name, v := range map[string][]float32{"after a longer text": again, "parallel": par[0], "batched": many[1], "batched twice": many[3]} {
		for i := range first {
			if v[i] != first[i] {
				t.Fatalf("%s: element %d is %v, want %v", name, i, v[i], first[i])
			}
		}
	}
}

func TestCountTokens(t *testing.T) {
	tok, err := loadTokenizer(filepath.Join(miniLM(t), "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{tok: tok}
	for text, want := range map[string]int{"wife": 1, "partner": 1, "192.0.2.111": 7, "02:00:5e:10:00:01": 0} {
		got := m.CountTokens(text)
		if want == 0 {
			if got < 8 {
				t.Errorf("%q: %d tokens, want many", text, got)
			}
		} else if got != want {
			t.Errorf("%q: %d tokens, want %d", text, got, want)
		}
	}
}
