// mcpbench: start an MCP stdio server, call one recall tool per question, record timing and output.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

type rpc struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  any             `json:"params,omitempty"`
}

func main() {
	name := flag.String("name", "", "label")
	tool := flag.String("tool", "", "recall tool name")
	arg := flag.String("arg", "", "query argument name")
	extra := flag.String("extra", "{}", "extra JSON args")
	qfile := flag.String("q", "", "questions tsv")
	out := flag.String("out", "", "output jsonl")
	flag.Parse()
	cmdline := flag.Args()

	var ex map[string]any
	if err := json.Unmarshal([]byte(*extra), &ex); err != nil {
		panic(err)
	}
	qs := []string{}
	b, _ := os.ReadFile(*qfile)
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		qs = append(qs, strings.SplitN(l, "\t", 2)[0])
	}

	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Env = os.Environ()
	cmd.Stderr = os.Stderr
	in, _ := cmd.StdinPipe()
	op, _ := cmd.StdoutPipe()
	start := time.Now()
	if err := cmd.Start(); err != nil {
		panic(err)
	}
	rd := bufio.NewReaderSize(op, 1<<20)
	enc := json.NewEncoder(in)
	id := 0
	call := func(method string, params any) map[string]any {
		id++
		enc.Encode(rpc{JSONRPC: "2.0", ID: id, Method: method, Params: params})
		for {
			line, err := rd.ReadBytes('\n')
			if err != nil {
				panic(fmt.Sprintf("%s: %v", method, err))
			}
			var m map[string]any
			if json.Unmarshal(line, &m) != nil {
				continue
			}
			if v, ok := m["id"].(float64); ok && int(v) == id {
				return m
			}
		}
	}
	f, _ := os.Create(*out)
	w := json.NewEncoder(f)

	init := call("initialize", map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{}, "clientInfo": map[string]any{"name": "bench", "version": "1"}})
	initMs := time.Since(start).Milliseconds()
	enc.Encode(rpc{JSONRPC: "2.0", Method: "notifications/initialized"})
	instr, _ := init["result"].(map[string]any)["instructions"].(string)
	tl := call("tools/list", map[string]any{})
	tj, _ := json.Marshal(tl["result"].(map[string]any)["tools"])
	w.Encode(map[string]any{"kind": "header", "name": *name, "init_ms": initMs, "instructions": instr, "tools_json_bytes": len(tj)})

	for i, q := range qs {
		args := map[string]any{*arg: q}
		for k, v := range ex {
			args[k] = v
		}
		t0 := time.Now()
		r := call("tools/call", map[string]any{"name": *tool, "arguments": args})
		ms := float64(time.Since(t0).Microseconds()) / 1000
		var text strings.Builder
		if res, ok := r["result"].(map[string]any); ok {
			for _, c := range res["content"].([]any) {
				text.WriteString(c.(map[string]any)["text"].(string))
			}
		} else {
			e, _ := json.Marshal(r["error"])
			text.WriteString("ERROR " + string(e))
		}
		w.Encode(map[string]any{"kind": "q", "i": i, "q": q, "ms": ms, "text": text.String()})
	}
	f.Close()
	in.Close()
	cmd.Wait()
}
