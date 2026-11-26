// Package vecstore is the embedding cache, vectors.bin: a memory-mapped,
// read-only file of vectors keyed by model identity plus chunk hash.
// Updates rewrite the whole file to a temp name and rename it into place.
//
// Layout (little endian):
//
//	"GRIZVEC1"
//	entries until EOF, each 4-byte aligned:
//	  u16 model length, model bytes, [16]byte hash, u32 dims,
//	  zero padding to 4-byte alignment, dims × float32
package vecstore

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"math"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"unsafe"

	"griz/internal/index"
)

const magic = "GRIZVEC1"

type key struct {
	model string
	hash  index.Hash
}

type loc struct{ off, dims int }

// Store is safe for concurrent use.
type Store struct {
	path string
	mu   sync.RWMutex
	data []byte // mmapped file (nil if none)
	locs map[key]loc
	mem  map[key][]float32 // vectors not yet written
	seen string            // size and mtime of the file when last mapped
}

func (s *Store) fileStamp() string {
	st, err := os.Stat(s.path)
	if err != nil {
		return ""
	}
	return fmt.Sprint(st.Size(), st.ModTime().UnixNano())
}

// Changed reports whether vectors.bin was replaced since it was last read.
func (s *Store) Changed() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.fileStamp() != s.seen
}

func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	s := &Store{path: filepath.Join(dir, "vectors.bin"), mem: map[key][]float32{}}
	return s, s.Reload()
}

// Reload maps the current file, picking up writes from other processes.
func (s *Store) Reload() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reload()
}

func (s *Store) reload() error {
	if s.data != nil {
		_ = syscall.Munmap(s.data)
		s.data = nil
	}
	s.locs = map[key]loc{}
	s.seen = s.fileStamp()
	f, err := os.Open(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	st, err := f.Stat()
	if err != nil || st.Size() < int64(len(magic)) {
		return err
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	if string(data[:len(magic)]) != magic {
		_ = syscall.Munmap(data)
		return nil // unknown file: treat as empty; the next write replaces it
	}
	s.data = data
	p := len(magic)
	for p+2 <= len(data) {
		ml := int(binary.LittleEndian.Uint16(data[p:]))
		p += 2
		if p+ml+16+4 > len(data) {
			break
		}
		var k key
		k.model = string(data[p : p+ml])
		p += ml
		copy(k.hash[:], data[p:p+16])
		p += 16
		dims := int(binary.LittleEndian.Uint32(data[p:]))
		p += 4
		p = align4(p)
		if p+4*dims > len(data) {
			break
		}
		s.locs[k] = loc{p, dims}
		p += 4 * dims
	}
	return nil
}

func align4(p int) int { return (p + 3) &^ 3 }

// Has reports whether a vector is cached.
func (s *Store) Has(model string, h index.Hash) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.get(key{model, h}) != nil
}

// Each calls fn with the cached vector of every ref that has one. The vector
// aliases the mapped file and is valid only during the call.
func (s *Store) Each(model string, refs []index.ChunkRef, fn func(index.ChunkRef, []float32)) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, r := range refs {
		if v := s.get(key{model, r.Hash}); v != nil {
			fn(r, v)
		}
	}
}

func (s *Store) get(k key) []float32 {
	if v, ok := s.mem[k]; ok {
		return v
	}
	l, ok := s.locs[k]
	if !ok {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&s.data[l.off])), l.dims)
}

// Put adds a vector in memory; Save writes it out.
func (s *Store) Put(model string, h index.Hash, v []float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.mem[key{model, h}] = v
}

// Dirty reports whether there are unsaved vectors.
func (s *Store) Dirty() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.mem) > 0
}

// Save writes unsaved vectors out, adding them to what is on disk. It never
// drops an entry.
func (s *Store) Save() error { return s.save(nil) }

// Prune rewrites the file keeping only entries whose hash is in keep (plus
// any unsaved vectors in keep). Only reindex and gc prune, after bringing the
// index up to date.
func (s *Store) Prune(keep map[index.Hash]bool) error {
	if keep == nil {
		keep = map[index.Hash]bool{}
	}
	return s.save(keep)
}

// save merges unsaved vectors with the file on disk, keeps only entries
// whose hash is in keep (all if keep is nil), and atomically replaces the file.
func (s *Store) save(keep map[index.Hash]bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	pending := s.mem
	if err := s.reload(); err != nil {
		return err
	}
	all := map[key][]float32{}
	for k := range s.locs {
		all[k] = s.get(k)
	}
	maps.Copy(all, pending)
	tmp, err := os.CreateTemp(filepath.Dir(s.path), ".vectors-*.tmp")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	w := bufio.NewWriter(tmp)
	_, _ = w.WriteString(magic)
	p := len(magic)
	var hdr bytes.Buffer
	for k, v := range all {
		if keep != nil && !keep[k.hash] {
			continue
		}
		hdr.Reset()
		_ = binary.Write(&hdr, binary.LittleEndian, uint16(len(k.model)))
		hdr.WriteString(k.model)
		hdr.Write(k.hash[:])
		_ = binary.Write(&hdr, binary.LittleEndian, uint32(len(v)))
		for (p+hdr.Len())%4 != 0 {
			hdr.WriteByte(0)
		}
		_, _ = w.Write(hdr.Bytes())
		p += hdr.Len()
		var buf [4]byte
		for _, f := range v {
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(f))
			_, _ = w.Write(buf[:])
		}
		p += 4 * len(v)
	}
	if err := w.Flush(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp.Name(), s.path); err != nil {
		return err
	}
	s.mem = map[key][]float32{}
	return s.reload()
}

func (s *Store) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.data != nil {
		_ = syscall.Munmap(s.data)
		s.data = nil
	}
}
