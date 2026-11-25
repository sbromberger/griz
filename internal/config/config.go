// Package config loads griz's options and resolves its XDG directories.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

type Recall struct {
	Budget          int     `toml:"budget"`
	FullResults     int     `toml:"full_results"`
	IDResults       int     `toml:"id_results"`
	Cutoff          float64 `toml:"cutoff"`
	ExcerptWords    int     `toml:"excerpt_words"`
	ArchivedResults int     `toml:"archived_results"`    // max archived ids listed on the "archived:" line
	MoreReserve     int     `toml:"more_reserve_tokens"` // budget kept for the "more:" line when the top result is cut
}

type Search struct {
	Model          string  `toml:"model"`
	QueryPrefix    string  `toml:"query_prefix"`
	DocumentPrefix string  `toml:"document_prefix"`
	MinSimilarity  float64 `toml:"min_similarity"` // vector hits below this cosine are ignored; 0 keeps every positive match and lets ranking decide
	Candidates     int     `toml:"candidates"`     // chunks taken from each ranked list before blending
	UnloadAfter    string  `toml:"unload_after"`   // release the model after this long unused; "0" = never
	EmbedWorkers   int     `toml:"embed_workers"`  // chunks embedded at once; 0 = GOMAXPROCS
}

// Workers is the number of chunks embedded at once: search.embed_workers,
// or GOMAXPROCS when it is 0.
func (s Search) Workers() int {
	if s.EmbedWorkers > 0 {
		return s.EmbedWorkers
	}
	return runtime.GOMAXPROCS(0)
}

// UnloadDuration parses search.unload_after.
func (s Search) UnloadDuration() (time.Duration, error) {
	return duration("search.unload_after", s.UnloadAfter)
}

type Scan struct {
	Interval    string `toml:"interval"`     // check for outside changes this often; "0" = off
	Settle      string `toml:"settle"`       // only auto-commit files unchanged this long
	CommitEdits bool   `toml:"commit_edits"` // commit hand edits automatically
}

func duration(key, v string) (time.Duration, error) {
	if v == "" || v == "0" {
		return 0, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil || d < 0 {
		return 0, fmt.Errorf("%s %q: want a duration like 10m, or 0", key, v)
	}
	return d, nil
}

func (s Scan) IntervalDuration() (time.Duration, error) { return duration("scan.interval", s.Interval) }
func (s Scan) SettleDuration() (time.Duration, error)   { return duration("scan.settle", s.Settle) }

type Term struct {
	Short  string `toml:"short"`  // lifetime of short-term memories
	Medium string `toml:"medium"` // lifetime of medium-term memories
	Long   string `toml:"long"`   // lifetime of long-term memories (infinite never expires)
}

var machineRE = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

var lifetimeRE = regexp.MustCompile(`^[1-9][0-9]*[dwmy]$`)

type Index struct {
	ChunkWords      int     `toml:"chunk_words"`     // target chunk size (also capped by the model's max input)
	TokenReserve    int     `toml:"token_reserve"`   // tokens kept free in each chunk for the id prefix
	WordsPerToken   float64 `toml:"words_per_token"` // converts the model's max tokens into chunk words
	BM25K1          float64 `toml:"bm25_k1"`
	BM25B           float64 `toml:"bm25_b"`
	WeightID        float64 `toml:"weight_id"`
	WeightNamespace float64 `toml:"weight_namespace"`
	WeightTags      float64 `toml:"weight_tags"`
	WeightBody      float64 `toml:"weight_body"`
}

type Store struct {
	Machine           string  `toml:"machine"`      // id prefix for memories made here; "" = the host name
	DefaultTerm       string  `toml:"default_term"` // term for memories stored without one
	DupCheck          bool    `toml:"dup_check"`
	DupThreshold      float64 `toml:"dup_threshold"`       // similarity by meaning that counts as a near-duplicate
	DupWordsThreshold float64 `toml:"dup_words_threshold"` // shared-word overlap that counts, without a model
	NewNSGuard        bool    `toml:"new_ns_guard"`        // require new_ns: true to create a namespace
}

type Pin struct {
	StartupFolder    string `toml:"startup_folder"`     // memories here go in full into the startup block
	StartupWarnChars int    `toml:"startup_warn_chars"` // warn when the startup block nears the apps' ~4096-character cut; 0 = never
	WarnTokens       int    `toml:"warn_tokens"`        // warn when pinned memories loaded by recall exceed this; 0 = never
}

type Lint struct {
	UnusedDays   int `toml:"unused_days"`
	ExpiringDays int `toml:"expiring_days"`
	LongWords    int `toml:"long_words"`
}

type Warn struct {
	ViaClaude bool `toml:"via_claude"` // add warnings to the server instructions / tool replies
}

type Log struct {
	Usage bool `toml:"usage"`
}

type Debug struct {
	TimingsKept int `toml:"timings_kept"` // recent operations kept in memory for griz show: timings
}

type Server struct {
	IdleExit     string `toml:"idle_exit"`     // the shared server exits after this long with no connected sessions; "0" = never
	StartTimeout string `toml:"start_timeout"` // how long a session waits for a newly started server to listen
	DrainWait    string `toml:"drain_wait"`    // how long a stopping server waits for tool calls in progress before giving up on them; "0" = no limit
}

func (s Server) IdleDuration() (time.Duration, error) {
	return duration("server.idle_exit", s.IdleExit)
}
func (s Server) DrainDuration() (time.Duration, error) {
	return duration("server.drain_wait", s.DrainWait)
}
func (s Server) StartDuration() (time.Duration, error) {
	return duration("server.start_timeout", s.StartTimeout)
}

type Config struct {
	DataDir  string `toml:"data_dir"`
	StateDir string `toml:"state_dir"`
	Recall   Recall `toml:"recall"`
	Term     Term   `toml:"term"`
	Index    Index  `toml:"index"`
	Search   Search `toml:"search"`
	Scan     Scan   `toml:"scan"`
	Store    Store  `toml:"store"`
	Pin      Pin    `toml:"pin"`
	Lint     Lint   `toml:"lint"`
	Warn     Warn   `toml:"warn"`
	Log      Log    `toml:"log"`
	Debug    Debug  `toml:"debug"`
	Server   Server `toml:"server"`

	// Not from the file.
	ConfigDir  string `toml:"-"`
	ConfigFile string `toml:"-"`
	FileFound  bool   `toml:"-"`
}

func Defaults() Config {
	return Config{
		Recall: Recall{Budget: 800, FullResults: 3, IDResults: 5, Cutoff: 0, ExcerptWords: 200, MoreReserve: 16, ArchivedResults: 3},
		Index: Index{ChunkWords: 200, TokenReserve: 16, WordsPerToken: 0.67,
			BM25K1: 1.2, BM25B: 0.75, WeightID: 4, WeightNamespace: 2, WeightTags: 3, WeightBody: 1},
		Search: Search{MinSimilarity: 0, UnloadAfter: "10m", Candidates: 50},
		Scan:   Scan{Interval: "10m", Settle: "1m", CommitEdits: true},
		Term:   Term{Short: "7d", Medium: "6m", Long: "1y"},
		Store:  Store{DefaultTerm: "medium", DupCheck: true, DupThreshold: 0.9, DupWordsThreshold: 0.6, NewNSGuard: true},
		Pin:    Pin{StartupFolder: "agents/startup", StartupWarnChars: 3800, WarnTokens: 2000},
		Lint:   Lint{UnusedDays: 180, ExpiringDays: 14, LongWords: 500},
		Warn:   Warn{ViaClaude: true},
		Log:    Log{Usage: true},
		Debug:  Debug{TimingsKept: 200},
		Server: Server{IdleExit: "10m", StartTimeout: "30s", DrainWait: "1m"},
	}
}

func xdg(env, fallback string) string {
	if v := os.Getenv(env); v != "" && filepath.IsAbs(v) {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, fallback)
}

// expandHome turns a leading ~ into the home directory.
func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, p[1:])
	}
	return p
}

// Load reads $XDG_CONFIG_HOME/griz/config.toml (if present) over the defaults.
func Load() (*Config, error) {
	c := Defaults()
	c.ConfigDir = filepath.Join(xdg("XDG_CONFIG_HOME", ".config"), "griz")
	c.ConfigFile = filepath.Join(c.ConfigDir, "config.toml")
	b, err := os.ReadFile(c.ConfigFile)
	switch {
	case errors.Is(err, fs.ErrNotExist):
	case err != nil:
		return nil, err
	default:
		c.FileFound = true
		dec := toml.NewDecoder(bytes.NewReader(b))
		md, err := dec.Decode(&c)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", c.ConfigFile, err)
		}
		if und := md.Undecoded(); len(und) > 0 {
			keys := make([]string, len(und))
			for i, k := range und {
				keys[i] = k.String()
			}
			return nil, fmt.Errorf("%s: unknown keys: %s", c.ConfigFile, strings.Join(keys, ", "))
		}
	}
	if c.DataDir == "" {
		c.DataDir = filepath.Join(xdg("XDG_DATA_HOME", ".local/share"), "griz")
	}
	if c.StateDir == "" {
		c.StateDir = filepath.Join(xdg("XDG_STATE_HOME", ".local/state"), "griz")
	}
	c.DataDir = expandHome(c.DataDir)
	c.StateDir = expandHome(c.StateDir)
	c.Search.Model = expandHome(c.Search.Model)
	for key, v := range map[string]string{"term.short": c.Term.Short, "term.medium": c.Term.Medium, "term.long": c.Term.Long} {
		if !lifetimeRE.MatchString(v) {
			return nil, fmt.Errorf("%s: %s %q: want a length like 7d, 2w, 6m or 1y", c.ConfigFile, key, v)
		}
	}
	if c.Store.Machine != "" && !machineRE.MatchString(c.Store.Machine) {
		return nil, fmt.Errorf("%s: store.machine %q: want lowercase letters, digits and single dashes, like host or alexs-mac", c.ConfigFile, c.Store.Machine)
	}
	switch c.Store.DefaultTerm {
	case "short", "medium", "long", "infinite":
	default:
		return nil, fmt.Errorf("%s: store.default_term %q: want short, medium, long or infinite", c.ConfigFile, c.Store.DefaultTerm)
	}
	if c.Search.EmbedWorkers < 0 {
		return nil, fmt.Errorf("%s: search.embed_workers %d: want 0 (one per CPU) or a positive number", c.ConfigFile, c.Search.EmbedWorkers)
	}
	for _, check := range []func() (time.Duration, error){c.Search.UnloadDuration, c.Scan.IntervalDuration, c.Scan.SettleDuration, c.Server.IdleDuration, c.Server.StartDuration, c.Server.DrainDuration} {
		if _, err := check(); err != nil {
			return nil, fmt.Errorf("%s: %w", c.ConfigFile, err)
		}
	}
	return &c, nil
}

// String prints the effective settings as TOML.
func (c *Config) String() string {
	var b bytes.Buffer
	toml.NewEncoder(&b).Encode(c)
	return b.String()
}
