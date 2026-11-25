# griz

griz is a local, long-term memory server for Claude. It speaks MCP over
stdio, so Claude Desktop and Claude Code can both use it, and it keeps every
memory as a Markdown file in a git repository on your own machine. Nothing
leaves the machine: search, embeddings and storage all run locally.

## Features

- **Plain files you can edit.** Each memory is a Markdown file with a little
  YAML frontmatter, in folders you choose (`work/projects/apollo.host-1a2b3c4d.md`).
  Every change griz makes is a git commit, and edits you make by hand are
  committed for you once they settle.
- **Hybrid search.** Keyword ranking (BM25) catches exact names, numbers and
  addresses; an optional local embedding model catches meaning ("spouse"
  finds "married Robin"). The two are blended per query. The model runs in
  pure Go, with no Python and no network, and uses Go's SIMD package when
  built with `GOEXPERIMENT=simd`.
- **Replies sized for a context window.** `recall` returns the best matches
  in full, then a short list of further ids, within a token budget.
  Archived memories appear only as ids, on a separate line.
- **Memories that load themselves.** Memories in a startup folder go into
  the instructions every session receives; pinned memories load on the first
  `recall`. griz warns when either grows past a size you set.
- **Lifetimes.** A memory is `short`, `medium`, `long` or `infinite` term.
  When its term runs out it is archived, never deleted: still searchable,
  just out of the way. Deleted memories can be restored from git history,
  and `griz memory purge` erases one from history for good.
- **Permanent ids.** Each memory has an id like `host-1a2b3c4d` that
  survives renames and moves. Commands accept the id, its first few hex
  digits (as many as make it unique, like git), the path, or part of the
  name.
- **One shared server.** Every Claude app starts `griz mcp`, a small relay
  that connects to one shared `griz serve` process, so all sessions see the
  same memories at once and the model is loaded only once. The server starts
  on demand, exits when idle, hands over to a new build after `go install`,
  and reports crashes to every session.
- **Guard rails.** Storing near-duplicates is refused unless forced, new
  top-level folders need an explicit flag, and `griz memory lint` reports
  long memories, likely duplicates, unused ones and broken files.
- **Local only.** The server listens on a Unix socket in a directory only
  you can open, and accepts a connection only from your own user running
  the same griz binary as the server.

## Requirements

- macOS (the connection checks are macOS-specific for now)
- Go 1.27
- git
- Optional: a sentence-transformers model for search by meaning, such as
  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
  griz needs the folder holding `config.json`, `tokenizer.json` and
  `model.safetensors`. Without a model, griz uses keyword search only.
- Optional: [git-filter-repo](https://github.com/newren/git-filter-repo),
  for `griz memory purge`

## Install

```sh
make install          # go install, with GOEXPERIMENT=simd
make test             # tests with and without SIMD
```

Register griz with Claude Code (this uses the griz on your PATH after `make install`):

```sh
claude mcp add --scope user griz -- "$(command -v griz)" mcp
```

and with Claude Desktop, in
`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "griz": { "command": "/absolute/path/to/griz", "args": ["mcp"] }
  }
}
```

After a later `make install`, the server switches to the new build by
itself; sessions carry on.

## MCP tools

| Tool | What it does |
|---|---|
| `recall` | Search memories (`q`); `pinned: true` loads every pinned memory; `archived: true` includes old ones |
| `get` | Full text of memories by id |
| `store` | Create a memory at a path, or change one: new body, an in-place edit (`old`/`new`), a dated `append`, or just its tags, term, expiry, pin or archived flag |
| `delete` | Delete a memory (it can be restored) |
| `griz` | Status, timings, a ranking explanation for a query, or debug mode for the session |

## Command line

The same binary is a CLI. `griz help` lists everything; `griz help <command>`
gives details. The common ones:

```sh
griz recall "where is the spare key?"     # search, as Claude does
griz get 1a2b                             # a memory by id, path or name
griz store notes/garden --body "Plant garlic in October" --term long
griz memory term infinite 1a2b            # change a lifetime
griz memory archive 1a2b                  # or unarchive, restore, purge
griz memory edit                          # $EDITOR in the memory folder
griz memory stats                         # counts, and the usage log
griz memory lint                          # long, duplicate, unused or broken memories
griz status                               # memories, model, warnings
griz version                              # this build, the server's, and each session's
griz env                                  # every file and folder griz uses
griz mcp restart                          # after a config change
griz debug on                             # timings for every session and command
```

## Memory files

```markdown
---
tags: [garden]
pin: true
term: long
updated: 2026-10-02
expires: 2027-10-02
---
Plant garlic in October; harvest when the lower leaves brown.
```

All frontmatter is optional. Archived memories are the same files with a
dot in front of the name, so they stand out in a directory listing. Files
you add by hand get an id on the next scan.

## Configuration

griz reads `~/.config/griz/config.toml` (or `$XDG_CONFIG_HOME/griz/config.toml`)
over its defaults. Every option is optional. `griz config show` prints the
effective settings, and `griz config edit` opens the file and checks it.
Changes take effect after `griz mcp restart`.

The options you are most likely to want:

```toml
# Where memories and griz's own files live (XDG defaults shown).
# data_dir  = "~/.local/share/griz"
# state_dir = "~/.local/state/griz"

[search]
# Folder holding a sentence-transformers model. Unset: keyword search only.
# model = "~/models/all-MiniLM-L6-v2"
# query_prefix    = "search_query: "      # for models trained with prefixes (nomic, e5)
# document_prefix = "search_document: "
unload_after  = "10m"   # free the model after this long unused; "0" = never
embed_workers = 0       # chunks embedded at once; 0 = one per CPU

[recall]
budget       = 800      # tokens per recall reply
full_results = 3        # results shown in full; more are listed by id

[term]                  # lifetimes; infinite never expires
short  = "7d"
medium = "6m"
long   = "1y"

[store]
default_term = "medium"
machine      = ""       # id prefix for memories made here; "" = the host name
dup_check    = true     # refuse near-duplicates unless forced
new_ns_guard = true     # new top-level folders need new_ns / --new-ns

[pin]
startup_folder = "agents/startup"   # memories here load in every session
warn_tokens    = 2000               # warn when pinned memories exceed this

[scan]
interval     = "10m"    # look for hand edits this often; "0" = off
settle       = "1m"     # commit a hand edit once unchanged this long
commit_edits = true

[server]
idle_exit = "10m"       # the shared server exits after this long with no sessions

[log]
usage = true            # log each tool call's size and timing (no memory text)
```

Sizes and limits in the config are warnings, never hard caps: griz tells
you when something grows past them and leaves the decision to you.
