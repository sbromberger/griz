package app

import (
	"regexp"
	"strings"
)

var paraRE = regexp.MustCompile(`\n[ \t]*\n+`)

// chunk splits a body into pieces of at most maxWords words. Short bodies are
// one chunk. Long ones split at paragraphs (a heading always starts a new
// chunk), merging small paragraphs and splitting big ones by lines, then words.
func chunk(body string, maxWords int) []string {
	body = strings.TrimSpace(body)
	if maxWords < 1 {
		maxWords = 1
	}
	if words(body) <= maxWords {
		return []string{body}
	}
	var out []string
	var cur []string
	n := 0
	flush := func() {
		if len(cur) > 0 {
			out = append(out, strings.Join(cur, "\n\n"))
			cur, n = nil, 0
		}
	}
	for _, p := range paraRE.Split(body, -1) {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		w := words(p)
		if strings.HasPrefix(p, "#") {
			flush()
		}
		if w > maxWords {
			flush()
			out = append(out, splitBig(p, maxWords)...)
			continue
		}
		if n+w > maxWords {
			flush()
		}
		cur = append(cur, p)
		n += w
	}
	flush()
	return out
}

// splitBig splits one paragraph by lines, and lines by words.
func splitBig(p string, maxWords int) []string {
	var units []string
	for _, line := range strings.Split(p, "\n") {
		if words(line) <= maxWords {
			units = append(units, line)
			continue
		}
		units = append(units, splitWords(line, maxWords)...)
	}
	var out, cur []string
	n := 0
	for _, u := range units {
		w := words(u)
		if n+w > maxWords && len(cur) > 0 {
			out = append(out, strings.Join(cur, "\n"))
			cur, n = nil, 0
		}
		cur = append(cur, u)
		n += w
	}
	if len(cur) > 0 {
		out = append(out, strings.Join(cur, "\n"))
	}
	return out
}

// splitWords cuts a long line into pieces, preferring sentence ends.
func splitWords(line string, maxWords int) []string {
	f := strings.Fields(line)
	var out []string
	for len(f) > maxWords {
		cut := maxWords
		for i := maxWords; i > maxWords/2; i-- {
			if strings.ContainsAny(f[i-1][len(f[i-1])-1:], ".!?") {
				cut = i
				break
			}
		}
		out = append(out, strings.Join(f[:cut], " "))
		f = f[cut:]
	}
	if len(f) > 0 {
		out = append(out, strings.Join(f, " "))
	}
	return out
}

func words(s string) int { return len(strings.Fields(s)) }
