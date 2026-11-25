package bert

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// tokenizer is a BERT WordPiece tokenizer read from tokenizer.json.
type tokenizer struct {
	vocab       map[string]int
	unk         int
	cls, sep    int
	prefix      string
	maxChars    int
	lowercase   bool
	stripAccent bool
	chinese     bool
	maxLen      int // from truncation, 0 if none
}

func loadTokenizer(path string) (*tokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f struct {
		Truncation *struct {
			MaxLength int `json:"max_length"`
		} `json:"truncation"`
		Normalizer *struct {
			Type         string `json:"type"`
			Lowercase    *bool  `json:"lowercase"`
			StripAccents *bool  `json:"strip_accents"`
			Chinese      *bool  `json:"handle_chinese_chars"`
		} `json:"normalizer"`
		Model struct {
			Type     string         `json:"type"`
			Vocab    map[string]int `json:"vocab"`
			Unk      string         `json:"unk_token"`
			Prefix   string         `json:"continuing_subword_prefix"`
			MaxChars int            `json:"max_input_chars_per_word"`
		} `json:"model"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if f.Model.Type != "WordPiece" {
		return nil, fmt.Errorf("tokenizer.json: model type %q is not WordPiece", f.Model.Type)
	}
	t := &tokenizer{vocab: f.Model.Vocab, prefix: f.Model.Prefix, maxChars: f.Model.MaxChars,
		lowercase: true, chinese: true}
	if t.prefix == "" {
		t.prefix = "##"
	}
	if t.maxChars == 0 {
		t.maxChars = 100
	}
	if n := f.Normalizer; n != nil {
		if n.Type != "BertNormalizer" {
			return nil, fmt.Errorf("tokenizer.json: normalizer %q is not supported", n.Type)
		}
		if n.Lowercase != nil {
			t.lowercase = *n.Lowercase
		}
		if n.Chinese != nil {
			t.chinese = *n.Chinese
		}
	}
	// strip_accents unset follows lowercase, as in the BERT tokenizer.
	t.stripAccent = t.lowercase
	if f.Normalizer != nil && f.Normalizer.StripAccents != nil {
		t.stripAccent = *f.Normalizer.StripAccents
	}
	if f.Truncation != nil {
		t.maxLen = f.Truncation.MaxLength
	}
	var ok1, ok2, ok3 bool
	unk := f.Model.Unk
	if unk == "" {
		unk = "[UNK]"
	}
	t.unk, ok1 = t.vocab[unk]
	t.cls, ok2 = t.vocab["[CLS]"]
	t.sep, ok3 = t.vocab["[SEP]"]
	if !ok1 || !ok2 || !ok3 {
		return nil, fmt.Errorf("tokenizer.json: vocab lacks [UNK], [CLS] or [SEP]")
	}
	return t, nil
}

// probeTokenizer checks tokenizer.json's settings without building the
// vocabulary, and returns its truncation length (0 if none).
func probeTokenizer(raw []byte) (int, error) {
	var f struct {
		Truncation *struct {
			MaxLength int `json:"max_length"`
		} `json:"truncation"`
		Normalizer *struct {
			Type string `json:"type"`
		} `json:"normalizer"`
		Model struct {
			Type string `json:"type"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return 0, fmt.Errorf("tokenizer.json: %w", err)
	}
	if f.Model.Type != "WordPiece" {
		return 0, fmt.Errorf("tokenizer.json: model type %q is not WordPiece", f.Model.Type)
	}
	if f.Normalizer != nil && f.Normalizer.Type != "BertNormalizer" {
		return 0, fmt.Errorf("tokenizer.json: normalizer %q is not supported", f.Normalizer.Type)
	}
	if f.Truncation != nil {
		return f.Truncation.MaxLength, nil
	}
	return 0, nil
}

// CountTokens returns how many model tokens text becomes, without [CLS] and
// [SEP] and without truncation.
func (m *Model) CountTokens(text string) int {
	n := 0
	for _, w := range m.tok.words(text) {
		n += len(m.tok.wordpiece(w))
	}
	return n
}

// encode returns token ids with [CLS] and [SEP], truncated to maxLen tokens.
func (t *tokenizer) encode(text string, maxLen int) []int {
	ids := []int{t.cls}
	for _, w := range t.words(text) {
		ids = append(ids, t.wordpiece(w)...)
		if len(ids) >= maxLen-1 {
			ids = ids[:maxLen-1]
			break
		}
	}
	return append(ids, t.sep)
}

// words applies the BertNormalizer and BertPreTokenizer.
func (t *tokenizer) words(text string) []string {
	var b strings.Builder
	for _, r := range text {
		switch {
		case r == 0 || r == 0xfffd || isControl(r):
			continue
		case unicode.IsSpace(r):
			b.WriteByte(' ')
		case t.chinese && isCJK(r):
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
		default:
			b.WriteRune(r)
		}
	}
	s := b.String()
	if t.stripAccent {
		s = norm.NFD.String(s)
		s = strings.Map(func(r rune) rune {
			if unicode.Is(unicode.Mn, r) {
				return -1
			}
			return r
		}, s)
	}
	if t.lowercase {
		s = strings.ToLower(s)
	}
	var out []string
	for _, f := range strings.Fields(s) {
		start := 0
		for i, r := range f {
			if isPunct(r) {
				if i > start {
					out = append(out, f[start:i])
				}
				out = append(out, string(r))
				start = i + len(string(r))
			}
		}
		if start < len(f) {
			out = append(out, f[start:])
		}
	}
	return out
}

func (t *tokenizer) wordpiece(w string) []int {
	rs := []rune(w)
	if len(rs) > t.maxChars {
		return []int{t.unk}
	}
	var ids []int
	for start := 0; start < len(rs); {
		end := len(rs)
		found := -1
		for end > start {
			sub := string(rs[start:end])
			if start > 0 {
				sub = t.prefix + sub
			}
			if id, ok := t.vocab[sub]; ok {
				found = id
				break
			}
			end--
		}
		if found < 0 {
			return []int{t.unk}
		}
		ids = append(ids, found)
		start = end
	}
	return ids
}

func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.In(r, unicode.Cc, unicode.Cf)
}

func isPunct(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) || (r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) || (r >= 0x2B740 && r <= 0x2B81F) || (r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) || (r >= 0x2F800 && r <= 0x2FA1F)
}
