# RLM Technique

An educational implementation of **Recursive Language Models (RLMs)** — the inference-time scaling paradigm from [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) where an LLM writes and executes code to explore documents in a REPL sandbox, rather than stuffing all context into a single prompt.

The primary interface is a **TUI (Terminal User Interface)** that visualizes the RLM process in real-time and compares it against baseline approaches (Traditional context-stuffing, RAG vector retrieval) side-by-side.

## Background

- **[`docs/RLM_PAPER.md`](docs/RLM_PAPER.md)** — full summary of the paper, its results, and how this repo maps to the original architecture.
- **[`docs/TWEAKS_EXPLAINED.md`](docs/TWEAKS_EXPLAINED.md)** — beginner-friendly explanation of practical implementation tweaks and rationale.
- **[`docs/BEGINNER_SETUP_AND_EVAL.md`](docs/BEGINNER_SETUP_AND_EVAL.md)** — step-by-step setup, evaluation workflow, complete env-variable table, and troubleshooting.

**TL;DR:** Traditional LLM apps cram all documents into one prompt. This causes *context rot* — answer quality degrades as input grows. RLMs fix this by storing documents externally and letting the LLM write code to explore only what it needs.

## Quick Start

```bash
git clone <repo-url> && cd rlm_technique
uv sync

cp env.sample .env
# edit .env — set your API keys and ENABLED_APPROACHES

# drop PDFs into data/
cp your-documents/*.pdf data/

# launch the TUI
uv run python rlm_tui.py

# or the plain CLI
uv run python rlm_cli.py
```

On first run, PDFs are automatically extracted to `processed_data/` (one text file per page). Subsequent runs load instantly.

## Architecture: How RLM Works

The LLM receives a **Python REPL sandbox** with four functions:

| Function | Purpose |
|---|---|
| `peek(doc_id, page)` | Read a specific document page — returns the raw text |
| `search(query)` | Regex search across all documents — returns `(doc_id, page, snippet)` hits |
| `llm_query(question, text)` | Recursive sub-LM call — ask a focused question about a text chunk |
| `answer(text)` | Set the final answer — terminates the exploration loop |

Each iteration:
1. The LLM writes a Python code block.
2. The code executes in the sandbox (persistent state across iterations, like Jupyter).
3. `stdout` and return values feed back to the LLM as context.
4. Repeat until `answer()` is called or the iteration budget is exhausted.

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

### Core Concept: Context as Variable

Documents are an external variable the LLM accesses programmatically. The prompt contains only the question and a document index — not document content. This decouples answer quality from corpus size.

## The TUI

```bash
uv run python rlm_tui.py
```

Two-panel layout:
- **Left: Chat** — clean Q&A with markdown answers, token tables, cost comparison, and post-turn evaluation metrics.
- **Right: Inspector** — live view of the code the LLM writes, REPL output, search results, and paper concept annotations.

Features:
- 📚 **Paper concepts** — each step shows the relevant paper section explaining *why* that operation matters.
- 💡 **Cost comparison** — after each answer, shows baseline vs RLM token usage and dollar cost.
- 🐌 **Baseline approaches** — Traditional (all docs in one prompt) and/or RAG (vector retrieval + synthesis) run for comparison.
- 🚀 **RLM answer** — the agent explores documents via code, with `[doc_id p#]` citations.
- 📊 **Per-iteration token table** — see exactly where tokens were spent per step.
- 🧪 **Post-turn eval** — automatic semantic similarity, TF-IDF, lexical overlap, evidence-doc overlap, and LLM judge metrics.
- `/eval` — run fixed-question benchmark in-chat with live traces.
- `/copy` — copies the full turn (question + answers + steps + tokens) as JSON to clipboard.
- `/test` — run `RLM_TEST_QUERY` from `.env` automatically.

Keyboard shortcuts:

| Key | Action |
|---|---|
| `ctrl+]` | Inspector wider |
| `ctrl+[` | Inspector narrower |
| `ctrl+h` | Toggle inspector on/off |
| `ctrl+l` | Clear chat |
| `ctrl+c` | Clear input first; press again quickly to quit |

### RAG in the TUI

The RAG approach requires a **locally running Weaviate vector database** (`RAG_WEAVIATE_MODE=local`). Remote/managed vector backends (Weaviate Cloud, Pinecone, etc.) are a planned enhancement.

Runtime RAG controls via slash commands:
- `/rag status` — current collection stats, retrieval mode, top_k, alpha.
- `/rag topk <N>` — set retrieval chunk count.
- `/rag mode semantic|hybrid` — switch retrieval strategy.
- `/rag alpha <0.0-1.0>` — set hybrid BM25/vector blend weight.
- `/rag retry` — re-initialize RAG connection.
- `/rag reset` — delete and rebuild the Weaviate collection from `processed_data/`.

### Startup Flags

```bash
# Run a specific query on TUI startup
uv run python rlm_tui.py --query "What governance controls are recommended?"

# Run the RLM_TEST_QUERY from .env on startup
uv run python rlm_tui.py --test
```

## Evaluation Harness

Run a repeatable benchmark comparing approaches across fixed questions:

```bash
uv run python rlm_eval.py --questions-file eval_questions.txt --min-cited-docs 8
```

Verbose mode (streams approach traces like the TUI inspector):

```bash
uv run python rlm_eval.py --questions-file eval_questions.txt --min-cited-docs 8 --verbose-trace
```

Disable semantic judging:

```bash
uv run python rlm_eval.py --questions-file eval_questions.txt --no-semantic-judge
```

### Metrics reported per question

- Token counts and estimated USD cost per approach.
- Token savings percentage (RLM vs baseline).
- Cited-document counts (both approaches).
- Answer similarity: lexical overlap, TF-IDF cosine, semantic similarity (configurable: LLM judge or vector cosine).
- LLM judge metrics (when `SEMANTIC_SIMILARITY_BACKEND=llm`): topic overlap, factuality/groundedness, evidence sufficiency, hallucination risk, winner, confidence.
- Citation booleans (`baseline_has_citations`, `rlm_has_citations`).
- Evidence-document overlap: shared docs, baseline-only, RLM-only, Jaccard index.

Results are saved to `eval_results.json`.

### Judge and Semantic Backend Configuration

The semantic similarity metric backend is configurable:

| Variable | Default | Description |
|---|---|---|
| `SEMANTIC_SIMILARITY_BACKEND` | `vector` | `vector` (embedding cosine) or `llm` (full LLM judge with structured JSON output) |
| `SEMANTIC_EMBED_PROVIDER` | `openai` | Embedding provider for vector backend: `openai` or `ollama` |
| `SEMANTIC_EMBED_MODEL` | `text-embedding-3-small` | Embedding model for vector similarity |

LLM judge overrides (independent of the main answer model):

| Variable | Default | Description |
|---|---|---|
| `JUDGE_LLM_PROVIDER` | main provider | Judge provider override |
| `JUDGE_LLM_MODEL` | main model | Judge model override |
| `JUDGE_LLM_TEMPERATURE` | `0.0` | Keep low for scoring consistency |
| `JUDGE_JSON_RETRIES` | `1` | Auto-retry count for strict JSON output validation |

The judge prompt is editable at `prompts/semantic_judge.txt`.

## Project Structure

| File / Directory | Purpose |
|---|---|
| `rlm_core.py` | Core types, `Config`, LLM provider abstraction, `RLMSandbox` REPL, `AnswerResult` |
| `approaches/base.py` | `BaseApproach` ABC + approach registry (`register_approach`, `get_approach`, `list_approaches`) |
| `approaches/rlm.py` | RLM approach: conversational router + iterative REPL exploration loop |
| `approaches/traditional.py` | Traditional baseline: full corpus context-stuffing with truncation reporting |
| `approaches/rag.py` | RAG baseline: Weaviate vector retrieval, embedding cache, auto-bootstrap, synthesis |
| `approaches/__init__.py` | Package exports and approach registration |
| `rlm_tui.py` | TUI launcher entrypoint (argument parsing, startup query flags) |
| `tui/app.py` | Main TUI application class: chat, inspector, commands, workers, post-turn eval |
| `tui/constants.py` | TUI constants: data paths, concept notes, slash commands, suggestions |
| `tui/styles.py` | TUI CSS/styling |
| `rlm_cli.py` | Plain CLI (no panels, text output) |
| `rlm_eval.py` | Evaluation harness: fixed-question benchmarks, metrics, semantic judge, JSON output |
| `rlm_docs.py` | Document discovery, PDF preprocessing, recursive directory scanning |
| `rlm_event_stream.py` | Event callback helpers for structured approach tracing |
| `rlm_trace.py` | Tree-structured tracing (`TraceTree`) |
| `prompts/` | All externalized LLM prompt templates (editable `.txt` files) |
| `scripts/legacy/` | Legacy PDF processor and utilities |
| `notebooks/` | Experimental Weaviate notebook workflow |
| `docs/` | Paper summary, tweaks documentation, beginner guide, assets |
| `data/` | Drop your PDFs here |
| `processed_data/` | Auto-generated text extraction (page-by-page) |
| `env.sample` | Example `.env` with all supported variables |

## Extending Approaches

Approaches follow a registry/ABC pattern defined in `approaches/base.py`:

1. Subclass `BaseApproach`.
2. Implement `run(doc_map, question, cfg, on_event) -> ApproachRun`.
3. Register with `register_approach(YourApproach())`.
4. Add your approach id to `ENABLED_APPROACHES` in `.env`.

This supports arbitrary comparison pairs (e.g. `traditional,rag`, `rlm,your_custom`, or all three) without modifying the TUI or eval runner.

## Configuration Reference

All configuration is via environment variables or `.env`. See **[`docs/BEGINNER_SETUP_AND_EVAL.md`](docs/BEGINNER_SETUP_AND_EVAL.md)** for the complete table with defaults and descriptions, or refer to `env.sample` for a working template.

## Why RLM Outperforms at Scale

| | Traditional | RAG | RLM |
|---|---|---|---|
| **4 docs** | Works — fits in context | Works — retrieves relevant chunks | Similar cost, better citations |
| **50 docs** | Truncates most content, loses information | Retrieves top-k; misses cross-doc synthesis | Searches + reads only relevant pages across all docs |
| **500 docs** | Impossible — exceeds context window | Same top-k budget; recall limited by embedding quality | Same as 50. Agent programmatically finds what it needs. |
| **Citations** | None — all content is one blob | Chunk-level `[doc_id p#]` | `[doc_id p#]` — knows exactly where each fact came from |
| **Cost scaling** | Linear with corpus size | Fixed retrieval + synthesis call | Logarithmic — reads only what's needed |
| **Context rot** | Yes — quality degrades with input length | Mitigated by retrieval window | No — each prompt is small and focused |
| **Cross-doc reasoning** | Limited by truncation boundary | Limited by top-k retrieval window | Unlimited — agent can chain searches and sub-queries |

## License

See project files for license details.
