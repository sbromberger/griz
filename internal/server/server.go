// Package server exposes griz over MCP (stdio) with five small tools.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"griz/internal/app"
	"griz/internal/usage"
)

// Version is set by main.
var Version = "dev"

// Tool definitions are written by hand: generated schemas cost several times
// the tokens. Keep every description to one line.
var tools = []struct {
	name, desc, schema string
}{
	{"recall", `Search the user's memories: one idea per q, in the user's words (search synonyms separately, not in one q). q "*" lists folders; pinned: true loads all pinned memories.`,
		`{"type":"object","properties":{"q":{"type":"string"},"budget":{"type":"integer","description":"max tokens"},"archived":{"type":"boolean","description":"true when the user asks to search their memories, including old ones"},"pinned":{"type":"boolean"}}}`},
	{"get", `Full text of memories by id.`,
		`{"type":"object","properties":{"id":{"type":["string","array"],"items":{"type":"string"}}},"required":["id"]}`},
	{"store", `Save a new memory at id "folder/name" with body (griz assigns its id), or change one by id: new body, old and new to edit in place (old must match once; all: true replaces every copy; preview: true shows the matching lines and writes nothing), or append for a dated entry. term sets how fast a memory fades in recall (nothing is forgotten): long for lasting facts and preferences, infinite for what must never fade, short for what matters only for days. expires (a date or 30d/6m, or never) is a hard date: archived after it; set it only for a known end. pin only when the user asks: pinned memories load in every session. move (alone) relocates a memory, keeping its id: an existing folder or one ending in / keeps its name, else "folder/name".`,
		`{"type":"object","properties":{"id":{"type":"string"},"body":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"},"all":{"type":"boolean"},"preview":{"type":"boolean"},"count":{"type":"integer","description":"with all: refuse unless this many copies"},"append":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}},"term":{"enum":["short","medium","long","infinite"]},"expires":{"type":"string"},"new_ns":{"type":"boolean"},"archived":{"type":"boolean"},"pin":{"type":"boolean"},"move":{"type":"string"},"force":{"type":"boolean","description":"save despite similar"}},"required":["id"]}`},
	{"delete", `Delete a memory (can be restored).`,
		`{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}`},
	{"griz", `About griz itself, for questions on what it's doing or why a search ranked as it did. debug: on adds what happened to every reply.`,
		`{"type":"object","properties":{"show":{"enum":["status","timings","profile","explain"]},"debug":{"enum":["on","off"]},"q":{"type":"string","description":"for explain"},"id":{"type":"string","description":"explain: where this memory landed"},"seconds":{"type":"integer","description":"for profile; default 10"}}}`},
}

// ToolTokens estimates the size of the tool definitions in tokens (characters / 4).
func ToolTokens() int {
	var defs []map[string]any
	for _, t := range tools {
		defs = append(defs, map[string]any{"name": t.name, "description": t.desc, "inputSchema": json.RawMessage(t.schema)})
	}
	b, _ := json.Marshal(defs)
	return app.Tokens(string(b))
}

// New builds the MCP server for one session. inFlight, if set, counts tool
// calls in progress (so a draining server can wait for them).
func New(a *app.App, sess *app.Session, instructions string, inFlight *sync.WaitGroup) *mcp.Server {
	s := mcp.NewServer(&mcp.Implementation{Name: "griz", Version: Version}, &mcp.ServerOptions{Instructions: instructions})
	for _, t := range tools {
		h := handler(a, sess, t.name, inFlight)
		s.AddTool(&mcp.Tool{Name: t.name, Description: t.desc, InputSchema: json.RawMessage(t.schema)}, h)
	}
	return s
}

// callOp runs one tool by name on raw JSON arguments, timing its steps in
// op (nil for none). It returns the reply text, the ids returned in full
// (for the usage log), and whether it failed.
func callOp(a *app.App, op *app.Op, name string, args json.RawMessage) (string, []string, bool) {
	out, ids, err := call(a, op, name, args)
	if err != nil {
		return err.Error(), nil, true
	}
	return out, ids, false
}

func call(a *app.App, op *app.Op, name string, raw json.RawMessage) (string, []string, error) {
	sess := a.Local
	if op != nil {
		sess = op.Sess
	}
	if len(raw) == 0 {
		raw = []byte("{}")
	}
	switch name {
	case "recall":
		var p struct {
			Q        string `json:"q"`
			Budget   int    `json:"budget"`
			Archived bool   `json:"archived"`
			Pinned   bool   `json:"pinned"`
		}
		if err := json.Unmarshal(raw, &p); err != nil {
			return "", nil, err
		}
		if p.Pinned {
			return a.Pinned()
		}
		return a.RecallOp(op, p.Q, p.Budget, p.Archived)
	case "get":
		var p struct {
			ID json.RawMessage `json:"id"`
		}
		if err := json.Unmarshal(raw, &p); err != nil {
			return "", nil, err
		}
		var ids []string
		var one string
		if err := json.Unmarshal(p.ID, &ids); err != nil {
			if err := json.Unmarshal(p.ID, &one); err != nil {
				return "", nil, fmt.Errorf("id: want a string or a list of strings")
			}
			ids = strings.FieldsFunc(one, func(r rune) bool { return r == ',' || r == ' ' || r == '\n' })
		}
		if err := a.Sync(); err != nil {
			return "", nil, err
		}
		out, found := a.Get(ids)
		return out, found, nil
	case "store":
		var p app.StoreArgs
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&p); err != nil {
			return "", nil, err
		}
		out, err := a.StoreOp(op, p)
		return out, nil, err
	case "delete":
		var p struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(raw, &p); err != nil {
			return "", nil, err
		}
		out, err := a.Delete(p.ID)
		return out, nil, err
	case "griz":
		var p struct {
			Show    string `json:"show"`
			Debug   string `json:"debug"`
			Q       string `json:"q"`
			ID      string `json:"id"`
			Seconds int    `json:"seconds"`
		}
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&p); err != nil {
			return "", nil, err
		}
		return griz(a, sess, p.Show, p.Debug, p.Q, p.ID, p.Seconds)
	}
	return "", nil, fmt.Errorf("unknown tool %q", name)
}

// griz answers the griz tool.
func griz(a *app.App, sess *app.Session, show, debug, q, id string, seconds int) (string, []string, error) {
	var parts []string
	switch debug {
	case "":
	case "on", "off":
		sess.SetDebug(debug == "on")
		parts = append(parts, "debug mode "+debug+" (this session only; ends when it does)")
	default:
		return "", nil, fmt.Errorf("debug: want on or off")
	}
	switch show {
	case "":
		if debug == "" {
			return "", nil, fmt.Errorf("give show (status, timings, profile or explain) or debug (on or off)")
		}
	case "status":
		parts = append(parts, a.Report())
	case "timings":
		parts = append(parts, a.Timings())
	case "profile":
		if seconds <= 0 {
			seconds = 10
		}
		out, err := app.Profile(time.Duration(seconds)*time.Second, nil, "")
		if err != nil {
			return "", nil, err
		}
		parts = append(parts, out)
	case "explain":
		out, err := a.Explain(q, id)
		if err != nil {
			return "", nil, err
		}
		parts = append(parts, out)
	default:
		return "", nil, fmt.Errorf("show: want status, timings, profile or explain")
	}
	return strings.Join(parts, "\n\n"), nil, nil
}

func handler(a *app.App, sess *app.Session, name string, inFlight *sync.WaitGroup) mcp.ToolHandler {
	return func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		if inFlight != nil {
			inFlight.Add(1)
			defer inFlight.Done()
		}
		start := time.Now()
		args := req.Params.Arguments
		op := a.BeginOp(sess, name)
		out, ids, failed := callOp(a, op, name, args)
		a.EndOp(op, ids)
		if ws := a.TakeWarningsFor(sess); len(ws) > 0 {
			out += "\n\n" + strings.Join(ws, "\n")
		}
		if ds := sess.TakeDebug(); len(ds) > 0 {
			out += "\n\n" + strings.Join(ds, "\n")
		}
		content := []mcp.Content{&mcp.TextContent{Text: out}}
		if wantsPinned(name, args) {
			a.MarkPinnedSent(sess)
		} else if p := a.TakePinnedFor(sess); p != "" {
			content = append(content, &mcp.TextContent{Text: p}) // its own block, outside this call's results and usage
		}
		a.Usage.Write(usage.Entry{Time: start, Tool: name, In: len(args), Out: len(out), MS: time.Since(start).Milliseconds(), IDs: ids})
		return &mcp.CallToolResult{Content: content, IsError: failed}, nil
	}
}

// wantsPinned reports a recall with pinned: true, which already returns them.
func wantsPinned(name string, args json.RawMessage) bool {
	var p struct {
		Pinned bool `json:"pinned"`
	}
	return name == "recall" && json.Unmarshal(args, &p) == nil && p.Pinned
}

// Serve runs one session's MCP over t until the connection ends.
// A resumed session is one that began with an earlier server: Claude never
// sees this server's instructions, so its warnings wait for the next reply.
func Serve(ctx context.Context, a *app.App, sess *app.Session, t mcp.Transport, inFlight *sync.WaitGroup, resumed bool) error {
	notice := ""
	if !resumed {
		notice = a.StartupNoticeFor(sess)
	}
	instr, err := a.StartupBlock(notice)
	if err != nil {
		return err
	}
	ss, err := New(a, sess, instr, inFlight).Connect(ctx, t, nil)
	if err != nil {
		return err
	}
	return ss.Wait()
}
