// Package memory reads and writes memory files and commits them with git.
package memory

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"time"

	"go.yaml.in/yaml/v3"
)

const DateLayout = "2006-01-02"

type Memory struct {
	ID       string // short id, machine-8hex (host-1a2b3c4d); never changes
	Path     string // namespace/slug: the folder is the namespace, the slug a label
	Tags     []string
	Pin      bool
	Archived bool   // matches searches but is only loaded on request
	Term     string // short, medium, long, infinite, or "" (none set)
	Created  string // YYYY-MM-DD (local) or ""
	Updated  string // YYYY-MM-DD (local) or ""
	Renewed  string // YYYY-MM-DD or "": last renewed without an edit (renew, unarchive, restore, or reads)
	Expires  string // YYYY-MM-DD or "": a hard date, set only on purpose; archived after it
	Body     string
}

// NS is the namespace: the folder, everything before the last "/" of the path (folders nest).
func (m *Memory) NS() string { return NS(m.Path) }

// Label is how a memory is shown: its path and id, "ns/slug [host-1a2b3c4d]".
func (m *Memory) Label() string { return m.Path + " [" + m.ID + "]" }

func NS(path string) string {
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[:i]
	}
	return ""
}

// Since is the date its fading counts from: the latest of created, updated
// and renewed ("" if none is set).
func (m *Memory) Since() string { return max(m.Created, m.Updated, m.Renewed) }

// IsArchived reports whether the memory counts as archived: flagged, or expired.
func (m *Memory) IsArchived(today string) bool { return m.Archived || m.Expired(today) }

// Expired reports whether the memory is past its expiry date.
func (m *Memory) Expired(today string) bool {
	return m.Expires != "" && m.Expires < today
}

func Today() string { return time.Now().Format(DateLayout) }

var (
	pathRE = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*(/[a-z0-9][a-z0-9_-]*)+$`)
	idRE   = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*-[0-9a-f]{8}$`)
)

// ValidPath checks a memory path: namespace/slug, lowercase a-z 0-9 - _.
func ValidPath(p string) error {
	if !pathRE.MatchString(p) {
		return fmt.Errorf("bad path %q: want namespace/name, lowercase a-z 0-9 - _", p)
	}
	return nil
}

// ValidID checks a short id: a machine prefix, "-", and 8 hex digits.
func ValidID(id string) error {
	if !idRE.MatchString(id) {
		return fmt.Errorf("bad id %q: want machine-8hex, like host-1a2b3c4d", id)
	}
	return nil
}

// Machine turns a host name into an id prefix: lowercase, letters, digits
// and "-", up to the first dot ("Alexs-Mac.local" becomes "alexs-mac").
func Machine(host string) string {
	host, _, _ = strings.Cut(strings.ToLower(host), ".")
	var b strings.Builder
	for _, r := range host {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case b.Len() > 0 && !strings.HasSuffix(b.String(), "-"):
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-")
}

type front struct {
	Tags     []string `yaml:"tags,omitempty"`
	Pin      bool     `yaml:"pin,omitempty"`
	Archived bool     `yaml:"archived,omitempty"`
	Term     string   `yaml:"term,omitempty"`
	Created  string   `yaml:"created,omitempty"`
	Updated  string   `yaml:"updated,omitempty"`
	Renewed  string   `yaml:"renewed,omitempty"`
	Expires  string   `yaml:"expires,omitempty"`
}

// Parse reads a memory file's contents. Unknown frontmatter keys are errors.
func Parse(id string, data []byte) (*Memory, error) {
	m := &Memory{ID: id}
	s := strings.ReplaceAll(string(data), "\r\n", "\n")
	if strings.HasPrefix(s, "---\n") {
		rest := s[4:]
		var fm string
		if strings.HasPrefix(rest, "---\n") || rest == "---" {
			fm, s = "", strings.TrimPrefix(strings.TrimPrefix(rest, "---"), "\n")
		} else {
			i := strings.Index(rest, "\n---\n")
			if i < 0 {
				if strings.HasSuffix(rest, "\n---") {
					i = len(rest) - 4
				} else {
					return nil, errors.New("frontmatter has no closing ---")
				}
			}
			fm = rest[:i+1]
			s = rest[min(len(rest), i+5):]
		}
		var f front
		dec := yaml.NewDecoder(strings.NewReader(fm))
		dec.KnownFields(true)
		if err := dec.Decode(&f); err != nil && !errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("frontmatter: %w", err)
		}
		for k, v := range map[string]string{"expires": f.Expires, "created": f.Created, "updated": f.Updated, "renewed": f.Renewed} {
			if v != "" {
				if _, err := time.Parse(DateLayout, v); err != nil {
					return nil, fmt.Errorf("frontmatter: %s %q is not YYYY-MM-DD", k, v)
				}
			}
		}
		if f.Term != "" && !ValidTerm(f.Term) {
			return nil, fmt.Errorf("frontmatter: term %q is not short, medium, long or infinite", f.Term)
		}
		m.Tags, m.Pin, m.Archived, m.Term, m.Expires = f.Tags, f.Pin, f.Archived, f.Term, f.Expires
		m.Created, m.Updated, m.Renewed = f.Created, f.Updated, f.Renewed
	}
	m.Body = strings.TrimSpace(s)
	return m, nil
}

// Format renders the file contents.
func (m *Memory) Format() []byte {
	var b bytes.Buffer
	if len(m.Tags) > 0 || m.Pin || m.Archived || m.Term != "" || m.Created != "" || m.Updated != "" || m.Renewed != "" || m.Expires != "" {
		b.WriteString("---\n")
		if len(m.Tags) > 0 {
			q := make([]string, len(m.Tags))
			for i, t := range m.Tags {
				q[i] = yamlScalar(t)
			}
			fmt.Fprintf(&b, "tags: [%s]\n", strings.Join(q, ", "))
		}
		if m.Pin {
			b.WriteString("pin: true\n")
		}
		if m.Archived {
			b.WriteString("archived: true\n")
		}
		if m.Term != "" {
			fmt.Fprintf(&b, "term: %s\n", m.Term)
		}
		if m.Created != "" {
			fmt.Fprintf(&b, "created: %s\n", m.Created)
		}
		if m.Updated != "" {
			fmt.Fprintf(&b, "updated: %s\n", m.Updated)
		}
		if m.Renewed != "" {
			fmt.Fprintf(&b, "renewed: %s\n", m.Renewed)
		}
		if m.Expires != "" {
			fmt.Fprintf(&b, "expires: %s\n", m.Expires)
		}
		b.WriteString("---\n")
	}
	b.WriteString(strings.TrimSpace(m.Body))
	b.WriteString("\n")
	return b.Bytes()
}

var plainTag = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*$`)

func yamlScalar(s string) string {
	if plainTag.MatchString(s) && s != "true" && s != "false" && s != "null" && s != "yes" && s != "no" {
		if _, err := strconv.ParseFloat(s, 64); err != nil {
			return s
		}
	}
	return strconv.Quote(s)
}

// ParseExpires accepts YYYY-MM-DD or a duration like 30d, 2w, 6m, 1y.
// "none" or "never" returns "" (no expiry).
func ParseExpires(s string, now time.Time) (string, error) {
	s = strings.TrimSpace(strings.ToLower(s))
	switch s {
	case "", "none", "never":
		return "", nil
	}
	if t, err := time.Parse(DateLayout, s); err == nil {
		return t.Format(DateLayout), nil
	}
	if len(s) >= 2 {
		n, err := strconv.Atoi(s[:len(s)-1])
		if err == nil && n > 0 {
			switch s[len(s)-1] {
			case 'd':
				return now.AddDate(0, 0, n).Format(DateLayout), nil
			case 'w':
				return now.AddDate(0, 0, 7*n).Format(DateLayout), nil
			case 'm':
				return now.AddDate(0, n, 0).Format(DateLayout), nil
			case 'y':
				return now.AddDate(n, 0, 0).Format(DateLayout), nil
			}
		}
	}
	return "", fmt.Errorf("bad expires %q: want YYYY-MM-DD or 30d, 2w, 6m, 1y", s)
}

// Words counts whitespace-separated words.
func Words(s string) int { return len(strings.Fields(s)) }

// ValidTerm reports whether t is short, medium, long or infinite.
func ValidTerm(t string) bool { return t == "short" || t == "medium" || t == "long" || t == "infinite" }

var deepHeading = regexp.MustCompile(`^#{3,6}[ \t]`)

// Display returns body text for showing to Claude: every heading of level 3
// or deeper gets one more "#", so only griz's own result lines start with
// exactly "###". Fenced code is left alone. The file is not changed.
func Display(body string) string {
	lines := strings.Split(body, "\n")
	fence := false
	for i, l := range lines {
		t := strings.TrimSpace(l)
		if strings.HasPrefix(t, "```") || strings.HasPrefix(t, "~~~") {
			fence = !fence
			continue
		}
		if !fence && deepHeading.MatchString(l) {
			lines[i] = "#" + l
		}
	}
	return strings.Join(lines, "\n")
}

// EntryLine is the dated line store's append writes: **YYYY-MM-DD HH:MM** — text.
func EntryLine(t time.Time, text string) string {
	return "**" + t.Format("2006-01-02 15:04") + "** — " + strings.TrimSpace(text)
}
