package app

import (
	"fmt"
	"regexp"
	"strings"

	"griz/internal/memory"
)

var hexRE = regexp.MustCompile(`^[0-9a-f]+$`)

// Resolve finds the one memory a key names, trying in order:
//  1. an exact id (host-1a2b3c4d) or path (ns/slug);
//  2. the start of an id: its hex part ("1a2b") or the whole id ("host-1a");
//  3. the end of a path ("slug", "ns/slug");
//  4. part of a slug.
//
// The first rule with any match decides: one match is the answer, several
// are an error that lists them.
func (a *App) Resolve(key string) (string, error) {
	key = strings.TrimSuffix(strings.TrimSpace(key), ".md")
	if key == "" {
		return "", fmt.Errorf("no id given")
	}
	if id, err := a.Repo.Lookup(key); err == nil {
		return id, nil
	}
	refs := a.Repo.Refs()
	rules := []func(memory.Ref) bool{
		func(r memory.Ref) bool {
			hexPart := r.ID[strings.LastIndex(r.ID, "-")+1:]
			return (hexRE.MatchString(key) && strings.HasPrefix(hexPart, key)) ||
				(strings.Contains(key, "-") && strings.HasPrefix(r.ID, key))
		},
		func(r memory.Ref) bool { return strings.HasSuffix("/"+r.Path, "/"+key) },
		func(r memory.Ref) bool { return strings.Contains(r.Path[strings.LastIndex(r.Path, "/")+1:], key) },
	}
	for _, match := range rules {
		var hits []memory.Ref
		for _, r := range refs {
			if match(r) {
				hits = append(hits, r)
			}
		}
		switch {
		case len(hits) == 1:
			return hits[0].ID, nil
		case len(hits) > 1:
			lines := make([]string, 0, len(hits))
			for i, h := range hits {
				if i == 10 {
					lines = append(lines, fmt.Sprintf("  ... and %d more", len(hits)-10))
					break
				}
				lines = append(lines, "  "+h.ID+"  "+h.Path)
			}
			return "", fmt.Errorf("%q matches %d memories:\n%s", key, len(hits), strings.Join(lines, "\n"))
		}
	}
	return "", fmt.Errorf("not found: %s", key)
}

// storeTarget decides what a store call's id names: an existing memory (its
// id), or, for a path no memory has, a new memory at that path.
func (a *App) storeTarget(key string) (id, path string, err error) {
	if id, err := a.Repo.Lookup(key); err == nil {
		return id, "", nil
	}
	if memory.ValidPath(key) == nil {
		return "", key, nil
	}
	if id, err := a.Resolve(key); err == nil {
		return id, "", nil
	} else if strings.Contains(err.Error(), "matches") {
		return "", "", err
	}
	return "", "", fmt.Errorf("not found: %s (to make a new memory, give a path: namespace/name)", key)
}

// Label is "ns/slug [id]" for an id (the id alone if it can't be read).
func (a *App) Label(id string) string { return a.label(id) }

func (a *App) label(id string) string {
	if p := a.Repo.PathOf(id); p != "" {
		return p + " [" + id + "]"
	}
	return id
}

// ShortLen is how many characters of id's hex part tell it apart from
// every other memory's (at least 1).
func (a *App) ShortLen(id string) int {
	hexOf := func(id string) string { return id[strings.LastIndex(id, "-")+1:] }
	h, n := hexOf(id), 1
	for _, r := range a.Repo.Refs() {
		if r.ID == id {
			continue
		}
		o := hexOf(r.ID)
		k := 0
		for k < len(h) && k < len(o) && h[k] == o[k] {
			k++
		}
		n = max(n, k+1)
	}
	return min(n, len(h))
}
