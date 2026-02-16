# RLM Technique

An educational implementation of **Recursive Language Models (RLMs)** — the inference paradigm from [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) where the LLM writes code to explore documents in a REPL sandbox instead of stuffing everything into the context window.

Includes a **TUI (Terminal UI)** that visualizes the RLM process in real-time and compares it against the traditional approach side-by-side.

## Background

See **[`docs/RLM_PAPER.md`](docs/RLM_PAPER.md)** for a full summary of the paper, its results, and how this repo maps to the original architecture.
See **[`docs/TWEAKS_EXPLAINED.md`](docs/TWEAKS_EXPLAINED.md)** for a beginner-friendly explanation of practical implementation tweaks and why they were made.

**TL;DR:** Traditional LLM apps cram all your documents into one prompt. This causes *context rot* — performance degrades as input grows. RLMs fix this by storing documents externally and letting the LLM write code to explore only what it needs.

## Quick Start

```bash
# clone and install
git clone <repo-url> && cd rlm_technique
uv sync

# create env file
cp env.sample .env
# then edit .env with your real API keys

# drop PDFs in data/
cp your-documents/*.pdf data/

# run the TUI
uv run python rlm_tui.py

# or the plain CLI
uv run python rlm_cli.py
```

On first run, PDFs are automatically extracted to `processed_data/` (text per page). Subsequent runs load instantly.

## Evaluation Harness

Run a repeatable benchmark to compare RLM vs a baseline approach across fixed questions:

```bash
uv run python rlm_eval.py --questions-file eval_questions.txt --min-cited-docs 8
```

For a live step stream similar to the TUI inspector:

```bash
uv run python rlm_eval.py --questions-file eval_questions.txt --min-cited-docs 8 --verbose-trace
```

What it reports per question:
- traditional tokens/cost + truncation stats
- RLM tokens/cost
- token savings percentage (RLM vs baseline)
- cited-document count (RLM and traditional)
- answer overlap ratio (surface similarity)
- semantic similarity (configurable: LLM judge or vector cosine)
- topic overlap score (LLM judge)
- citation booleans from judge (`baseline_has_citations`, `rlm_has_citations`)

LLM judge prompt is editable at:
- `prompts/semantic_judge.txt`

Optional judge-model overrides (separate from main answer model):
- `JUDGE_LLM_PROVIDER`
- `JUDGE_LLM_MODEL`
- `JUDGE_LLM_BASE_URL`
- `JUDGE_LLM_API_KEY`
- `JUDGE_LLM_TEMPERATURE`

Semantic similarity backend config:
- `SEMANTIC_SIMILARITY_BACKEND` = `llm` or `vector`
- `SEMANTIC_EMBED_PROVIDER` = `openai` or `ollama` (for `vector`)
- `SEMANTIC_EMBED_MODEL` (for `vector`, e.g. `text-embedding-3-small`)
- `SEMANTIC_EMBED_BASE_URL`
- `SEMANTIC_EMBED_API_KEY`

To disable semantic judging:

```bash
uv run python rlm_eval.py --questions-file eval_questions.txt --no-semantic-judge
```

It saves machine-readable output to `eval_results.json` by default.

## How It Works

The LLM gets a **Python REPL sandbox** with 4 functions:

| Function | What it does |
|---|---|
| `peek(doc_id, page)` | Read a document page — returns the text |
| `search(query)` | Regex search across all docs — returns hits |
| `llm_query(question, text)` | Recursive sub-LM call — ask a question about a chunk |
| `answer(text)` | Set the final answer — ends the loop |

Each iteration:
1. The LLM writes a Python code block
2. The code executes in the sandbox (persistent state, like Jupyter)
3. Output feeds back to the LLM as context
4. Repeat until `answer()` is called

```
User Question (query only — documents stay external)
     │
     ▼
┌──────────────────────────┐
│  LLM writes Python code  │◄──────────────────────────┐
│  in the REPL sandbox     │                            │
└──────────┬───────────────┘                            │
           │                                            │
     Code executes:                                     │
     peek() / search() / llm_query() / answer()         │
           │                                            │
     stdout + return values                             │
           │                                            │
     Feed output back to LLM ───────────────────────────┘

RUNTIME ENVIRONMENT (external — documents live here)
┌──────────────────────────────┐
│  docs = {doc_id: pages}      │  ◄── text extracted from PDFs
│  peek(), search() read from  │      never sent to LLM directly
│  this, not the prompt         │
└──────────────────────────────┘
```

## The TUI

```bash
uv run python rlm_tui.py
```

Two-panel layout:
- **Left: Chat** — clean Q&A with markdown answers, token tables, and a side-by-side comparison of RLM vs traditional
- **Right: Inspector** — live view of the code the LLM writes, REPL output, and paper concept annotations

Features:
- 📚 **Paper concepts** — each step shows the relevant paper section explaining *why* that operation matters
- 💡 **Cost comparison** — after each answer, shows traditional vs RLM token usage
- 🐌 **Traditional answer** — runs the naive "all docs in one prompt" approach for comparison
- 🚀 **RLM answer** — the agent explores documents via code, with citations
- 📊 **Per-iteration token table** — see exactly where tokens were spent
- 🧪 **`/eval` benchmark in-chat** — run fixed-question evaluation with live traditional + RLM traces
- `/copy` — copies the full turn (question + answers + steps + tokens) as JSON to clipboard

Keyboard shortcuts:
| Key | Action |
|---|---|
| `ctrl+]` | Inspector wider |
| `ctrl+[` | Inspector narrower |
| `ctrl+h` | Toggle inspector on/off |
| `ctrl+l` | Clear chat |
| `ctrl+c` | Clear input first; press again quickly to quit |

## Project Structure

| File | Purpose |
|---|---|
| `rlm_core.py` | Core types/config/providers + REPL sandbox (`RLMSandbox`) |
| `approaches/rlm.py` | RLM approach implementation (router + recursive loop) |
| `approaches/traditional.py` | Traditional baseline implementation |
| `approaches/rag.py` | RAG baseline implementation (Weaviate retrieval + synthesis) |
| `rlm_tui.py` | TUI launcher entrypoint |
| `tui/app.py` | Main TUI app logic (chat, inspector, commands, workers) |
| `tui/constants.py` | TUI constants and slash-command/help text |
| `tui/styles.py` | TUI CSS/styling |
| `rlm_cli.py` | Plain CLI (no panels, just text output) |
| `rlm_trace.py` | Tree-structured tracing (`TraceTree`) |
| `prompts/rlm_system.txt` | System prompt telling the LLM how to use the sandbox |
| `scripts/legacy/process-pdfs.py` | Legacy full PDF processor with embeddings (optional, advanced use) |
| `notebooks/build-data.ipynb` | Experimental Weaviate notebook workflow |
| `docs/assets/` | Paper PDF and visual/reference assets |
| `data/` | Drop your PDFs here |
| `processed_data/` | Auto-generated text extraction (page-by-page) |
| `.env` | API keys and config |

## Configuration

All via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai`, or `anthropic` |
| `LLM_MODEL` | *(per-provider)* | `qwen2.5:7b` / `gpt-4o-mini` / `claude-sonnet-4-20250514` |
| `LLM_BASE_URL` | *(per-provider)* | API base URL |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `ENABLED_APPROACHES` | `rlm,traditional` | Ordered, comma-separated approaches to run (`rlm`, `traditional`, `rag`) |
| `RLM_MAX_ITERATIONS` | `10` | Max REPL iterations per question |
| `RLM_HARD_MAX_ITERATIONS` | `60` | Safety cap used when `RLM_MAX_ITERATIONS=0` (until-done mode) |
| `RLM_SANDBOX_EXEC_TIMEOUT_S` | `5.0` | Per-step Python execution timeout in the sandbox |
| `RLM_SANDBOX_MAX_EXEC_LINES` | `50000` | Per-step Python line budget in the sandbox |
| `RLM_MAX_HISTORY_MESSAGES` | `8` | Chat history included in prompts |
| `RLM_DATA_DIR` | `data` | PDF source directory |
| `RLM_PROCESSED_DIR` | `processed_data` | Extracted text directory |
| `RAG_WEAVIATE_MODE` | `local` | Weaviate connection mode for RAG baseline |
| `RAG_WEAVIATE_COLLECTION` | `RLMChunk` | Weaviate collection to query |
| `RAG_TOP_K` | `10` | Number of chunks to retrieve |
| `RAG_RETRIEVAL_MODE` | `semantic` | Retrieval strategy: `semantic` (vector only) or `hybrid` (BM25 + vector) |
| `RAG_HYBRID_ALPHA` | `0.7` | Hybrid blend weight toward vector search (0.0-1.0; used only in `hybrid` mode) |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model used for query vectors |
| `RAG_AUTO_BOOTSTRAP` | `true` | If collection is missing/empty, auto-create and ingest from `processed_data` on first RAG run |
| `RAG_INGEST_MAX_CHARS` | `1800` | Max characters per ingested chunk |
| `RAG_INGEST_BATCH_SIZE` | `64` | Embedding batch size during bootstrap ingestion |
| `SEMANTIC_SIMILARITY_BACKEND` | `vector` | Semantic metric backend for eval/autoeval: `llm` or `vector` |
| `SEMANTIC_EMBED_PROVIDER` | `openai` | Embedding provider for vector backend: `openai` or `ollama` |
| `SEMANTIC_EMBED_MODEL` | `text-embedding-3-small` | Embedding model used by vector semantic similarity |
| `SEMANTIC_EMBED_BASE_URL` | *(provider default)* | Base URL for vector embedding provider |
| `SEMANTIC_EMBED_API_KEY` | — | Optional override API key for vector embedding provider |

## Why RLM > Traditional

| | Traditional | RLM |
|---|---|---|
| **4 docs** | Works OK — fits in context | Similar cost, better citations |
| **50 docs** | Truncates most content, loses info | Searches + reads only relevant pages |
| **500 docs** | Impossible — exceeds context window | Same as 50. Agent finds what it needs. |
| **Citations** | None — it's all one blob | `[doc_id p#]` — knows exactly where each fact came from |
| **Cost scaling** | Linear with corpus size | Logarithmic — reads only what's needed |
| **Context rot** | Yes — quality degrades with length | No — each prompt is small and focused |

## License

See project files for license details.
