# ai-memory vs griz comparison (2026-11-24)

60 questions in `questions.tsv` (not included: they come from a private store; write your own in the same format): question, expected griz id regex, expected ai-memory title regex. Lines 1-24 are the original set, 25-32 ask about one person in different words, 33-44 look up IP addresses (bare and as a question), 45-60 look up exact strings (MAC, phone, hex, ports, RFC1918) and mix them with prose.

- `mcpbench`: starts an MCP stdio server, calls one recall tool per question, writes JSONL (timing + text).
  `go run ./bench/mcpbench -name X -tool recall -arg q -q bench/questions.tsv -out OUT.jsonl -- /abs/path/griz mcp`
  ai-memory: `-tool memory_recall -arg context -- ai-memory --db COPY.db mcp` (always on a copy of the DB).
  Use griz's absolute path, and sandbox XDG_CONFIG_HOME / XDG_DATA_HOME / XDG_STATE_HOME.
- `grader`: `go run ./bench/grader DIR` expects in DIR: `questions.tsv`, `aim/titles.txt`
  (`sqlite3 -readonly COPY.db "select id||'|'||title from memories"`), `aim.jsonl`, `kw.jsonl`, `hy3.jsonl`.
  Prints rank of the expected memory per question: ai-memory, griz keyword-only, griz hybrid.

Results so far (right memory first / top 3 / found, of 24):
ai-memory 10/15/18; griz keyword 19/21/22; griz hybrid 19/23/23 (memory-level fusion).
