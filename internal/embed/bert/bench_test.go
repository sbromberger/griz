package bert

import (
	"strings"
	"testing"
)

func BenchmarkEmbed128(b *testing.B) {
	m, err := Load(miniLMB(b))
	if err != nil {
		b.Fatal(err)
	}
	text := strings.Repeat("memory ", 200)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Embed([]string{text})
	}
}

func BenchmarkEmbedBatch8(b *testing.B) {
	m, err := Load(miniLMB(b))
	if err != nil {
		b.Fatal(err)
	}
	texts := make([]string, 8)
	for i := range texts {
		texts[i] = strings.Repeat("memory ", 200)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Embed(texts)
	}
}
