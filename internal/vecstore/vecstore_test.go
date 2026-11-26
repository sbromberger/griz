package vecstore

import (
	"testing"

	"griz/internal/index"
)

func TestSaveReload(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	h1, h2 := index.Hash{1}, index.Hash{2}
	s.Put("m-odd", h1, []float32{1, 2, 3})
	s.Put("m", h2, []float32{4, 5})
	if err := s.Save(); err != nil {
		t.Fatal(err)
	}
	// A second "process" sees the file and adds to it.
	s2, _ := Open(dir)
	if !s2.Has("m-odd", h1) || !s2.Has("m", h2) || s2.Has("m", h1) {
		t.Fatal("keys after reload")
	}
	s2.Put("m", index.Hash{3}, []float32{6})
	if err := s2.Save(); err != nil {
		t.Fatal(err)
	}
	// The first saves again without losing the second's entry, and prunes h1.
	s.Put("m", index.Hash{4}, []float32{7})
	if err := s.Prune(map[index.Hash]bool{h2: true, {3}: true, {4}: true}); err != nil {
		t.Fatal(err)
	}
	var got []float32
	s.Each("m", []index.ChunkRef{{Hash: h2}, {Hash: index.Hash{3}}, {Hash: index.Hash{4}}}, func(_ index.ChunkRef, v []float32) {
		got = append(got, v...)
	})
	if len(got) != 4 || got[0] != 4 || got[1] != 5 || got[2] != 6 || got[3] != 7 {
		t.Fatalf("vectors: %v", got)
	}
	if s.Has("m-odd", h1) {
		t.Fatal("h1 should be pruned")
	}
}
