# RLM Technique

A lightweight implementation of the **Recursive Language Model (RLM)** inference paradigm, applied to grounded Q&A over a corpus of PDF documents.

## Background

https://arxiv.org/abs/2512.24601

Based on [**Recursive Language Models**](https://arxiv.org/abs/2512.24601) (Zhang, Kraska & Khattab — MIT CSAIL, 2025). RLMs let the model drive its own research over an external environment instead of stuffing everything into the context window, scaling to 10M+ tokens without quality loss. See **[RLM_PAPER.md](RLM_PAPER.md)** for a full summary of the paper, its results, and how this repo maps to the original architecture.

## How It Works

1. **PDF ingestion** — PDFs are loaded page-by-page into an in-memory library (`build_library`).
2. **Recursive loop** — For each user question the system enters a bounded loop (up to `RLM_MAX_STEPS` iterations):
   - The LLM is prompted with the question, chat history, and evidence collected so far.
   - It responds with either a `NEXT_QUERY` (search keyword/phrase) or `DONE` plus a final `ANSWER`.
   - If a query is returned, the system runs a regex search across all document pages, retrieves the top snippet hits, and appends them as evidence.
   - The loop repeats until the model declares `DONE` or the step budget is exhausted.
3. **Citation** — The final answer references evidence as `[doc_id p#]` so users can verify claims.

## Project Structure

| File | Purpose |
|---|---|
| `rlm_core.py` | Config, PDF loading, search/snippet tools, LLM calling, the RLM loop (`answer_question`), and data classes (`ChatMessage`, `Evidence`, `AnswerResult`) |
| `rlm_cli.py` | Interactive chat CLI with session save/load, history, and per-turn tracing |
| `rlm_tracer.py` | Tree-structured tracing system (`TraceTree`) that records every LLM call, tool invocation, and step for debugging/analysis |
| `pyproject.toml` | Project metadata and dependencies |

## Requirements

- Python ≥ 3.13
- **One** of the following LLM backends:
  - [Ollama](https://ollama.com) running locally (default) — e.g. `ollama pull qwen2.5:7b`
  - An [OpenAI](https://platform.openai.com) API key
  - An [Anthropic](https://console.anthropic.com) API key
- A `data/` directory containing the PDF documents you want to query

### Python Dependencies

Managed via `pyproject.toml`:

- `pypdf` — PDF text extraction
- `requests` — HTTP calls to the Ollama API
- `openai` — OpenAI provider support
- `anthropic` — Anthropic provider support
- `jupyterlab` — notebook support (optional for CLI usage)

## Installation

```bash
# clone the repo
git clone <repo-url> && cd rlm_technique

# create a virtual environment & install deps
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Configuration

All configuration is driven by environment variables with sensible defaults.

### LLM Settings

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or `anthropic` |
| `LLM_MODEL` | *(per-provider)* | Model name — defaults to `qwen2.5:7b` / `gpt-4o-mini` / `claude-sonnet-4-20250514` |
| `LLM_BASE_URL` | *(per-provider)* | API base URL (auto-set per provider, override for proxies) |
| `OPENAI_API_KEY` | — | API key when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | — | API key when `LLM_PROVIDER=anthropic` |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |

### RLM Loop Settings

| Variable | Default | Description |
|---|---|---|
| `RLM_MAX_STEPS` | `8` | Maximum search-read-decide iterations per question |
| `RLM_MAX_HITS` | `6` | Max search hits returned per query |
| `RLM_MAX_EVIDENCE` | `8` | Max evidence snippets retained (oldest are dropped) |
| `RLM_SNIPPET_CHARS` | `1600` | Characters to extract per snippet |
| `RLM_MAX_HISTORY_MESSAGES` | `8` | Chat history messages included in the prompt |

### Tracing & Debug

| Variable | Default | Description |
|---|---|---|
| `RLM_VERBOSE` | `false` | Print trace events to stdout |
| `RLM_TRACE` | `false` | Write per-turn JSON trace files |
| `RLM_TRACE_DIR` | `traces` | Directory for trace output files |

## Usage

### CLI Chat

```bash
python rlm_cli.py
```

You'll be dropped into an interactive prompt:

```
RLM Chat CLI (type /help). Loading PDFs...
Loaded 4 documents. Ready.

you> What are the key agentic coding trends for 2026?
```

### CLI Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/docs` | List loaded document IDs |
| `/history` | Show last 10 chat messages |
| `/save [path]` | Save session history to JSON (default: `session.json`) |
| `/load [path]` | Load session history from JSON (default: `session.json`) |
| `/clear` | Clear chat history |
| `/quit` | Exit the CLI |

### Enabling Tracing

```bash
RLM_TRACE=true RLM_VERBOSE=true python rlm_cli.py
```

Each conversation turn writes a JSON trace tree to the `traces/` directory, capturing every LLM call, search, and snippet retrieval with timestamps — useful for debugging and understanding the model's research path.

## Architecture

Context is a **variable, not input** — the LLM never sees the full corpus. It interacts with the document environment through three tools:

```
User Question (query only — context stays external)
     │
     ▼
┌──────────────────────────┐
│  decide_next_step (LLM)  │◄─────────────────────────┐
│  "Which tool should I    │                           │
│   call next?"            │                           │
└──────────┬───────────────┘                           │
           │                                           │
     ┌─────┼──────────┬──────────┐                     │
     │     │          │          │                     │
     ▼     ▼          ▼          ▼                     │
   peek   grep     lm_call     done                   │
   │       │          │          │                     │
   │       │          │          ▼                     │
   │       │          │     ANSWER ──► return          │
   │       │          │                                │
   │       │          ▼                                │
   │       │     sub-LM call                           │
   │       │     (read chunk,                          │
   │       │      answer sub-Q)                        │
   │       │          │                                │
   ▼       ▼          ▼                                │
  append to evidence ──────────────────────────────────┘

RUNTIME ENVIRONMENT (external)
┌──────────────────────────────┐
│  context: str = ...          │  ◄── PDFs stored as variables
│  ██████████████████████████  │      (lazy-loaded from data/)
│  ██████████████████████████  │
└──────────────────────────────┘
```

| Tool | Maps to paper | What it does |
|---|---|---|
| `peek` | `peek(context[:2000])` | Reveal structure of a document page |
| `grep` | `grep(context, pattern)` | Search all docs for a keyword → filtered subset |
| `lm_call` | `lm_call(sub_query, chunk)` | Recursive sub-LM call on a page chunk → sub-answer |
| `done` | — | Enough evidence gathered → final answer |

The LLM context window stays lean — it only ever contains the query, evidence summaries, and tool results. The full document text lives in the `DocumentLibrary` (runtime environment) and is only accessed through tool calls.

## License

See project files for license details.
