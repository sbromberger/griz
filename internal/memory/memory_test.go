package memory

import (
	"strings"
	"testing"
	"time"
)

func TestParseFormatRoundTrip(t *testing.T) {
	m := &Memory{ID: "writing/x", Tags: []string{"style", "true", "a b"}, Pin: true, Expires: "2027-01-15", Body: "Hello.\n\nWorld."}
	got, err := Parse(m.ID, m.Format())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got.Tags, "|") != "style|true|a b" || !got.Pin || got.Expires != m.Expires || got.Body != m.Body {
		t.Fatalf("round trip: %+v\n%s", got, m.Format())
	}
	plain := &Memory{ID: "a/b", Body: "just text"}
	if string(plain.Format()) != "just text\n" {
		t.Fatalf("no frontmatter expected: %q", plain.Format())
	}
}

func TestParseStrict(t *testing.T) {
	for name, in := range map[string]string{
		"unknown key":  "---\ntitle: x\n---\nbody",
		"bad date":     "---\nexpires: soon\n---\nbody",
		"unterminated": "---\ntags: [a]\nbody",
	} {
		if _, err := Parse("a/b", []byte(in)); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
	m, err := Parse("a/b", []byte("---\n---\nbody\n"))
	if err != nil || m.Body != "body" {
		t.Fatalf("empty frontmatter: %v %+v", err, m)
	}
}

func TestParseExpires(t *testing.T) {
	now := time.Date(2026, 1, 31, 0, 0, 0, 0, time.UTC)
	for in, want := range map[string]string{"2027-02-03": "2027-02-03", "10d": "2026-02-10", "2w": "2026-02-14",
		"1m": "2026-03-03", "1y": "2027-01-31", "none": "", "": ""} {
		got, err := ParseExpires(in, now)
		if err != nil || got != want {
			t.Errorf("%q: got %q %v, want %q", in, got, err, want)
		}
	}
	if _, err := ParseExpires("tomorrow", now); err == nil {
		t.Error("expected error")
	}
}

func TestValidPathAndID(t *testing.T) {
	for _, p := range []string{"a/b", "class101/policy_changes", "x/y/z-1"} {
		if ValidPath(p) != nil {
			t.Errorf("path %s should be valid", p)
		}
	}
	for _, p := range []string{"a", "A/b", "a/", "/a", "a/../b", "a/b.md", "a b/c"} {
		if ValidPath(p) == nil {
			t.Errorf("path %s should be invalid", p)
		}
	}
	for _, id := range []string{"host-1a2b3c4d", "alexs-mac-00000000"} {
		if ValidID(id) != nil {
			t.Errorf("id %s should be valid", id)
		}
	}
	for _, id := range []string{"1a2b3c4d", "host-1a2b3c4", "host-1A2B3C4D", "host-1a2b3c4dd", "a/b", "-1a2b3c4d"} {
		if ValidID(id) == nil {
			t.Errorf("id %s should be invalid", id)
		}
	}
}

func TestDisplayAndDates(t *testing.T) {
	in := "## Keep\n### Three\n#### Four\n###### Six\n###not\n```\n### code\n```\ntext ### mid"
	want := "## Keep\n#### Three\n##### Four\n####### Six\n###not\n```\n### code\n```\ntext ### mid"
	if got := Display(in); got != want {
		t.Errorf("Display:\n%s\nwant\n%s", got, want)
	}
	m := &Memory{ID: "a/b", Body: "x", Created: "2026-01-02", Updated: "2026-03-04"}
	back, err := Parse("a/b", m.Format())
	if err != nil || back.Created != m.Created || back.Updated != m.Updated {
		t.Errorf("round trip: %+v %v", back, err)
	}
	if _, err := Parse("a/b", []byte("---\nupdated: June 2\n---\nx\n")); err == nil {
		t.Error("bad updated date should fail")
	}
}
