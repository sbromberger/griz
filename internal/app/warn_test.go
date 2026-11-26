package app

import (
	"os"
	"testing"
)

func TestWarningsReachClaudeOnce(t *testing.T) {
	root := t.TempDir()
	a := newApp(t, root)
	if err := os.MkdirAll(a.Repo.Dir+"/notes", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file(a, "notes/bad1"), []byte("---\nbogus: 1\n---\nx\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file(a, "notes/bad2"), []byte("---\nbogus: 2\n---\nx\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	b := newApp(t, root) // a fresh process sees the broken files at startup
	b.setPinWarnings([]Warning{{warnPinned, "pinned memories are 900 tokens"}})
	n := b.startupNotice()
	if n != "griz: 3 warnings (2 files with broken frontmatter; pinned memories over budget). Tell the user once and suggest griz memory lint." {
		t.Fatalf("notice: %q", n)
	}
	if ws := b.takeWarnings(); len(ws) != 0 {
		t.Fatalf("startup warnings were already delivered: %v", ws)
	}
	b.Warn(warnEmbed, "embedding: boom")
	b.Warn(warnEmbed, "embedding: boom") // same warning again: reported once
	if ws := b.takeWarnings(); len(ws) != 1 || ws[0] != "griz warning: embedding: boom" {
		t.Fatalf("in-session: %v", ws)
	}
	if ws := b.takeWarnings(); len(ws) != 0 {
		t.Fatalf("second take: %v", ws)
	}
	c := newApp(t, root)
	c.Cfg.Warn.ViaClaude = false
	if c.startupNotice() != "" {
		t.Fatal("via_claude=false: no notice")
	}
	if fresh := newApp(t, t.TempDir()); fresh.startupNotice() != "" {
		t.Fatal("no warnings: no notice")
	}
}
