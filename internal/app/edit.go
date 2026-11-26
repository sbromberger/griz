package app

import (
	"fmt"
	"strings"

	"griz/internal/memory"
)

// Literal find-and-replace for store's old and new. Every place old starts
// counts, so copies that share text ("ha ha" in "ha ha ha") count as two
// and a unique edit refuses them.

// matchesAt returns every byte offset where old starts in body, overlapping
// ones included. old is valid UTF-8, so a match can't start mid-rune.
func matchesAt(body, old string) []int {
	var pos []int
	for i := 0; i <= len(body); {
		j := strings.Index(body[i:], old)
		if j < 0 {
			break
		}
		pos = append(pos, i+j)
		i += j + 1
	}
	return pos
}

// editMatches picks the copies of old an edit replaces: the only one, or,
// with all, every one (which must not share text).
func editMatches(m *memory.Memory, old string, all bool) ([]int, error) {
	pos := matchesAt(m.Body, old)
	switch {
	case len(pos) == 0:
		return nil, fmt.Errorf("old text not found in %s", m.Label())
	case !all && len(pos) > 1:
		return nil, fmt.Errorf("old text matches %d times in %s; give more of it, or pass all to replace every copy (preview shows them)", len(pos), m.Label())
	}
	for i := 1; i < len(pos); i++ {
		if pos[i] < pos[i-1]+len(old) {
			return nil, fmt.Errorf("copies of old text overlap in %s (at line %d); give more of it", m.Label(), lineOf(m.Body, pos[i]))
		}
	}
	return pos, nil
}

// replaceText applies an edit: the body after it, and how many copies it
// replaced. count, if given, must equal that number.
func replaceText(m *memory.Memory, old, new string, all bool, count *int) (string, int, error) {
	pos, err := editMatches(m, old, all)
	if err != nil {
		return "", 0, err
	}
	if count != nil && *count != len(pos) {
		return "", 0, fmt.Errorf("old text matches %d times in %s, not %d; nothing changed", len(pos), m.Label(), *count)
	}
	return splice(m.Body, 0, pos, len(old), new), len(pos), nil
}

// splice replaces the n bytes at each of pos (offsets into the whole body;
// s starts at offset base) with new.
func splice(s string, base int, pos []int, n int, new string) string {
	var b strings.Builder
	last := 0
	for _, p := range pos {
		b.WriteString(s[last : p-base])
		b.WriteString(new)
		last = p - base + n
	}
	b.WriteString(s[last:])
	return b.String()
}

func lineOf(body string, off int) int { return 1 + strings.Count(body[:off], "\n") }

// previewEdit shows what an edit would change, a line at a time, without
// writing: every copy of old with the whole lines it sits on, before and
// after. With several copies and no all, it lists them so old can be made
// unique. The listing stops at about budget tokens.
func previewEdit(m *memory.Memory, old, new string, all bool, budget int) (string, error) {
	body := m.Body
	pos := matchesAt(body, old)
	several := !all && len(pos) > 1 // listed, so old can be made unique
	if !several {
		var err error
		if pos, err = editMatches(m, old, all); err != nil {
			return "", err
		}
	}
	var b strings.Builder
	if several {
		fmt.Fprintf(&b, "old text matches %d times in %s; an edit would be refused (give more of it, or pass all):\n", len(pos), m.Label())
	} else {
		fmt.Fprintf(&b, "would replace %d in %s:\n", len(pos), m.Label())
	}
	shown := 0
	for i := 0; i < len(pos); {
		// One group: the lines the copy at pos[i] spans, plus every later
		// copy that starts inside them (several on one line show once).
		start := strings.LastIndex(body[:pos[i]], "\n") + 1
		end := lineEnd(body, pos[i]+len(old))
		j := i + 1
		for j < len(pos) && pos[j] < end {
			end = max(end, lineEnd(body, pos[j]+len(old)))
			j++
		}
		before := body[start:end]
		where := fmt.Sprintf("line %d", lineOf(body, start))
		if l := lineOf(body, end); l != lineOf(body, start) {
			where = fmt.Sprintf("lines %d to %d", lineOf(body, start), l)
		}
		entry := fmt.Sprintf("  %s: %s\n", where, oneLine(before))
		if !several {
			entry += fmt.Sprintf("    → %s\n", oneLine(splice(before, start, pos[i:j], len(old), new)))
		}
		if shown > 0 && tokensOf(b.Len()+len(entry)) > budget {
			fmt.Fprintf(&b, "  ... and %d more\n", len(pos)-i)
			break
		}
		b.WriteString(entry)
		shown++
		i = j
	}
	return strings.TrimRight(b.String(), "\n"), nil
}

// lineEnd is the offset of the end of the line holding off (before its "\n").
func lineEnd(body string, off int) int {
	if off > 0 && off <= len(body) && body[off-1] == '\n' {
		off-- // the copy ends with a newline: its line ends there
	}
	if i := strings.IndexByte(body[off:], '\n'); i >= 0 {
		return off + i
	}
	return len(body)
}

// oneLine shows line breaks inside a preview entry as ⏎.
func oneLine(s string) string { return strings.ReplaceAll(s, "\n", "⏎") }
