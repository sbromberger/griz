package index

import (
	"strings"
	"testing"
)

func TestSearch(t *testing.T) {
	x := New()
	x.Add(Doc{ID: "writing/no-em-dashes", NS: "writing", Tags: []string{"style"}, Chunks: []string{"No em-dashes in text Claude writes."}})
	x.Add(Doc{ID: "project/partners", NS: "project", Chunks: []string{"Partners are Northwind and Fabrikam."}})
	x.Add(Doc{ID: "travel/trip", NS: "travel", Expires: "2020-01-01", Chunks: []string{"Trip to Fabrikam."}})

	if h := x.Search("Fabrikam", Filter{}, 10); len(h) != 2 {
		t.Fatalf("expired memories still match: %+v", h)
	}
	if !x.ArchivedAt("travel/trip", "2026-01-01") || x.ArchivedAt("travel/trip", "2019-12-31") || x.ArchivedAt("project/partners", "2026-01-01") {
		t.Fatal("ArchivedAt")
	}
	if h := x.Search("writing style", Filter{}, 10); len(h) == 0 || h[0].ID != "writing/no-em-dashes" {
		t.Fatalf("stemming / id / tags: %+v", h)
	}
	if h := x.Search("Fabrikam", Filter{NS: "writing"}, 10); len(h) != 0 {
		t.Fatalf("namespace filter: %+v", h)
	}
	if h := x.Search("the and of", Filter{}, 10); len(h) != 0 {
		t.Fatalf("stopwords only: %+v", h)
	}
	x.Remove("project/partners")
	x.Add(Doc{ID: "project/partners", NS: "project", Chunks: []string{"Partners: Contoso."}})
	if h := x.Search("northwind", Filter{}, 10); len(h) != 0 {
		t.Fatalf("replaced text still found: %+v", h)
	}
	if h := x.Search("contoso", Filter{}, 10); len(h) != 1 {
		t.Fatalf("new text not found: %+v", h)
	}
	if n := len(x.Chunks(Filter{})); n != 3 {
		t.Fatalf("chunks: %d", n)
	}
}

func TestLoadReplaces(t *testing.T) {
	x := New()
	x.Add(Doc{ID: "a/b", NS: "a", Chunks: []string{"alpha"}})
	x.Load([]Doc{{ID: "c/d", NS: "c", Chunks: []string{"gamma"}}})
	if x.Has("a/b") || !x.Has("c/d") || len(x.Search("alpha", Filter{}, 5)) != 0 {
		t.Fatal("Load should replace everything")
	}
}

func TestTermsDropContractionFragments(t *testing.T) {
	if got := strings.Join(Terms("what is my wife's name? I'll check, don't worry"), " "); got != "wife name check don worri" {
		t.Errorf("got %q", got)
	}
}
