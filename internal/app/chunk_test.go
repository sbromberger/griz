package app

import (
	"strings"
	"testing"
)

func TestChunkShort(t *testing.T) {
	if c := chunk("one two three", 10); len(c) != 1 {
		t.Fatal(c)
	}
}

func TestChunkLong(t *testing.T) {
	para := strings.TrimSpace(strings.Repeat("word ", 30))
	body := "# Intro\n\n" + para + "\n\n" + para + "\n\n# Next\n\n" + para + "\n\n" + strings.Repeat("long sentence here. ", 40)
	chunks := chunk(body, 50)
	total := 0
	for _, c := range chunks {
		if n := words(c); n > 50 {
			t.Errorf("chunk has %d words", n)
		}
		total += words(c)
	}
	if total != words(body) {
		t.Errorf("lost words: %d vs %d", total, words(body))
	}
	found := false
	for _, c := range chunks {
		if strings.HasPrefix(c, "# Next") {
			found = true
		}
	}
	if !found {
		t.Errorf("a heading should start a chunk: %q", chunks)
	}
}
