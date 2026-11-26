package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

var (
	uuidRe = regexp.MustCompile(`(?m)^([0-9a-f]{8}-[0-9a-f-]{27})\|`)
	hdrRe  = regexp.MustCompile(`(?m)^### ([a-z0-9_-]+(?:/[a-z0-9_-]+)+)`)
	moreRe = regexp.MustCompile(`(?m)^more: (.*)$`)
)

func lines(p string) []string {
	b, _ := os.ReadFile(p)
	return strings.Split(strings.TrimRight(string(b), "\n"), "\n")
}

func main() {
	dir := os.Args[1]
	qs := lines(dir + "/questions.tsv")
	titles := map[string]string{}
	for _, l := range lines(dir + "/aim/titles.txt") {
		p := strings.SplitN(l, "|", 2)
		if len(p) == 2 {
			titles[p[0]] = p[1]
		}
	}
	load := func(f string) map[int]string {
		m := map[int]string{}
		fh, _ := os.Open(dir + "/" + f)
		sc := bufio.NewScanner(fh)
		sc.Buffer(make([]byte, 1<<22), 1<<22)
		for sc.Scan() {
			var r struct {
				Kind, Text string
				I          int
			}
			_ = json.Unmarshal(sc.Bytes(), &r)
			if r.Kind == "q" {
				m[r.I] = r.Text
			}
		}
		return m
	}
	aim, kw, hy := load("aim.jsonl"), load("kw.jsonl"), load("hy3.jsonl")
	gr := func(text, pat string) string {
		re := regexp.MustCompile("^(" + pat + ")")
		hs := hdrRe.FindAllStringSubmatch(text, -1)
		for i, h := range hs {
			if re.MatchString(h[1]) {
				return fmt.Sprint(i + 1)
			}
		}
		if m := moreRe.FindStringSubmatch(text); m != nil {
			for j, id := range strings.Split(m[1], ",") {
				if re.MatchString(strings.TrimSpace(id)) {
					return fmt.Sprintf("more(%d)", len(hs)+j+1)
				}
			}
		}
		return "miss"
	}
	ga := func(text, pat string) string {
		re := regexp.MustCompile(pat)
		for i, m := range uuidRe.FindAllStringSubmatch(text, -1) {
			if re.MatchString(titles[m[1]]) {
				return fmt.Sprint(i + 1)
			}
		}
		return "miss"
	}
	for i, l := range qs {
		p := strings.Split(l, "\t")
		fmt.Printf("%s\t%s\t%s\t%s\n", ga(aim[i], p[2]), gr(kw[i], p[1]), gr(hy[i], p[1]), p[0])
	}
}
