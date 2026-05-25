# hermes-weaviate-engram

Standalone [Weaviate Engram](https://weaviate.io/engram) memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

> Per Hermes [CONTRIBUTING.md](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#memory-providers-ship-as-a-standalone-plugin), new memory providers are shipped as standalone repos that drop into `~/.hermes/plugins/`. This is that repo.

## What it does

- Long-term memory backed by Weaviate Engram (semantic recall + server-side extract / reconcile / commit pipelines).
- Three tools exposed to the agent: `engram_search`, `engram_store`, `engram_fetch`.
- Per-turn ingest (`sync_turn`) into Engram's async pipeline.
- Per-turn prefetch (`prefetch`) into the system prompt as a `<engram-context>` block.
- Per-profile scoping by default via Engram's `user_id` multi-tenancy.

**No `engram_forget` tool.** Engram is designed around *purposeful forgetting* — deletion and expiry are server-side concerns. The agent corrects memories by storing a new correcting memory; Engram's reconcile pipeline supersedes the old one.

## Install

```bash
# 1. Clone the repo anywhere.
git clone https://github.com/weaviate/hermes-weaviate-engram.git
cd hermes-weaviate-engram

# 2. Symlink the plugin into $HERMES_HOME/plugins/.
./install.sh

# 3. Install the Engram SDK.
pip install weaviate-engram

# 4. Set credentials.
echo 'ENGRAM_API_KEY=...' >> ~/.hermes/.env

# 5. Activate.
hermes memory setup        # pick weaviate_engram
# (or:  hermes config set memory.provider weaviate_engram)
```

`install.sh` honors `$HERMES_HOME` if set (defaults to `~/.hermes`). It
creates a symlink so plugin updates require nothing more than `git pull`
inside the repo. Hermes discovers the plugin on next start.

### Repo layout

```
hermes-weaviate-engram/
├── weaviate_engram/           # the plugin (symlinked into ~/.hermes/plugins/)
│   ├── __init__.py
│   ├── client.py
│   └── plugin.yaml
├── tests/
├── install.sh
├── pyproject.toml             # for tests / CI; not a publishable wheel
└── README.md
```

## Configuration

Optional settings in `~/.hermes/weaviate_engram.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `user_id_template` | `{identity}` | Template for the Engram `user_id`. `{identity}` is replaced with the Hermes profile name. Set to a literal string for shared memory across profiles. |
| `auto_recall` | `true` | Inject relevant memory context before each turn. |
| `auto_capture` | `true` | Store each completed turn after the response. |
| `max_recall_results` | `10` | Max recalled items, bounded 1..20. |
| `min_capture_chars` | `10` | Skip trivial turns shorter than this. |
| `api_timeout` | `10.0` | Engram request timeout (seconds), bounded 0.5..60. |
| `pipeline_hint` | `""` | Optional note injected into the system prompt so the model knows which Engram pipeline is active. |

Environment variables:

| Variable | Description |
|----------|-------------|
| `ENGRAM_API_KEY` | Engram API key (required). |
| `ENGRAM_BASE_URL` | Optional. Overrides the default `https://api.engram.weaviate.io` (use for staging or self-hosted endpoints). |
| `WEAVIATE_ENGRAM_USER_ID` | Pin a single `user_id`. Overrides `user_id_template`. |

## Tools

| Tool | Description |
|------|-------------|
| `engram_search` | Search memories by semantic similarity. |
| `engram_store` | Store an explicit memory. **Also the forgetting mechanism** — store a correcting memory rather than expecting a delete. |
| `engram_fetch` | Profile-shaped recall ("what do you know about me?"). |

## Purposeful forgetting

The `engram` SDK exposes `.memories.delete` and `.memories.get`, but this plugin deliberately does not surface them as agent tools. Engram is designed around *purposeful forgetting* — deletion and expiry are first-class server-side operations handled by the same pipelines that extract and reconcile memories.

To correct a memory, store a new memory that explicitly states the correction:

```
engram_store(content="Correction: the user moved from Berlin to Lisbon in 2026.")
```

Engram's reconcile pipeline supersedes the older memory. A regression test in this repo locks the design choice in (the plugin must never expose `engram_forget` / `engram_delete` / `engram_remove` / `engram_purge`).

## Development

```bash
git clone https://github.com/weaviate/hermes-weaviate-engram.git
cd hermes-weaviate-engram
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

Tests run fully mocked (no network) against a `FakeEngramClient`. They depend on Hermes Agent's `MemoryProvider` ABC and `tool_error` helper, which are installed as a dev dependency via `hermes-agent`.

## Roadmap (Phase 2)

- `on_session_end` full-conversation ingest
- `on_memory_write` mirror (correction/retraction phrasing for replace/remove)
- `on_session_switch` state reset
- `queue_prefetch` + cached `prefetch` for sub-turn latency
- `on_pre_compress` extraction
- Optional property-scoped tagging (`scope_properties: true` — uses Engram's `properties`/`group` parameters)

## License

MIT — see [LICENSE](LICENSE).

## Links

- [Hermes Agent](https://github.com/NousResearch/hermes-agent)
- [Weaviate Engram](https://weaviate.io/engram) / [deep-dive blog](https://weaviate.io/blog/engram-deep-dive)
- [`weaviate-engram` on PyPI](https://pypi.org/project/weaviate-engram/)
