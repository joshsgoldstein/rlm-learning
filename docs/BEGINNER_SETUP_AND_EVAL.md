# Beginner Setup and Evaluation Guide

This guide is the practical, step-by-step path for running the project with the **TUI as the main interface**.

## What You Get

- A terminal app where you ask questions over a PDF corpus.
- Side-by-side approach runs (for example `traditional` vs `rag`, or baseline vs `rlm`).
- Post-turn metrics (semantic, TF-IDF, lexical overlap, evidence overlap).
- A repeatable offline evaluation command (`rlm_eval.py`).

## 1) Prerequisites

- Python 3.11+ (project currently uses 3.13 in development).
- `uv` installed.
- API key for your chosen provider (`OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`).
- PDFs placed in `data/`.

Optional for RAG:

- A **local Weaviate instance** running and reachable from the app process (`RAG_WEAVIATE_MODE=local`).
- At this stage, RAG is implemented against local Weaviate only; support for remote/managed vector backends is a planned enhancement.

## 2) Install

```bash
uv sync
```

## 3) Configure `.env`

```bash
cp env.sample .env
```

Then edit `.env` and set real keys.

Minimum recommended first run:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
ENABLED_APPROACHES=rlm,rag
```

If you want baseline-only comparison:

```env
ENABLED_APPROACHES=traditional,rag
```

## 4) Run the TUI (Main Experience)

```bash
uv run python rlm_tui.py
```

On first run, PDFs are extracted from `data/` into `processed_data/`.

### Useful TUI actions

- `ctrl+c`: clear input first; press again quickly to quit.
- `ctrl+h`: toggle inspector.
- `ctrl+]` / `ctrl+[`: resize inspector.
- `/help`: show commands.
- `/test`: run `RLM_TEST_QUERY` from `.env`.

### RAG controls in TUI

- `/rag status`
- `/rag topk 12`
- `/rag mode semantic`
- `/rag mode hybrid`
- `/rag alpha 0.55`
- `/rag retry`
- `/rag reset`

## 5) Run Evaluation from CLI

Basic:

```bash
uv run python rlm_eval.py --questions-file eval_questions.txt --min-cited-docs 8
```

Verbose (streams approach traces):

```bash
uv run python rlm_eval.py --questions-file eval_questions.txt --min-cited-docs 8 --verbose-trace
```

Output is written to:

- `eval_results.json`

## 6) Environment Variables (Complete App/Eval Table)

The table below covers active configuration used by current app/eval paths.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Main answer model provider: `ollama`, `openai`, `anthropic`. |
| `LLM_MODEL` | provider-specific | Main answer model name. |
| `LLM_BASE_URL` | provider-specific | Base URL for the main provider API. |
| `LLM_API_KEY` | empty | Generic API key fallback for providers that use `LLM_API_KEY`. |
| `LLM_CONTEXT_WINDOW` | `0` | `0` means auto-detect; set explicit context window to override. |
| `OPENAI_API_KEY` | empty | OpenAI key for answering, RAG embeddings, and optional semantic embeddings. |
| `ANTHROPIC_API_KEY` | empty | Anthropic key when provider/judge is Anthropic. |
| `LLM_TEMPERATURE` | `0.3` | Main answer model sampling temperature. |
| `ENABLED_APPROACHES` | `rlm,traditional` | Ordered approaches to run. Example: `traditional,rag` or `rlm,rag`. |
| `RLM_MAX_ITERATIONS` | `10` | Max RLM planning iterations. Use `0` for until-done mode. |
| `RLM_HARD_MAX_ITERATIONS` | `60` | Safety cap used when `RLM_MAX_ITERATIONS=0`. |
| `RLM_SANDBOX_EXEC_TIMEOUT_S` | `5.0` | Per-step sandbox execution timeout. |
| `RLM_SANDBOX_MAX_EXEC_LINES` | `50000` | Per-step sandbox line budget. |
| `RLM_MAX_HISTORY_MESSAGES` | `8` | Chat history messages included in prompts. |
| `RLM_TEST_QUERY` | sample text | Query used by `/test` and `--test` startup mode. |
| `RLM_DATA_DIR` | `data` | Source PDF directory. |
| `RLM_PROCESSED_DIR` | `processed_data` | Extracted page-text directory. |
| `RAG_WEAVIATE_MODE` | `local` | Weaviate connection mode. Current implementation supports local Weaviate only. |
| `RAG_WEAVIATE_COLLECTION` | `RLMChunk` | Collection name used for RAG chunks. |
| `RAG_TOP_K` | `10` | Number of retrieved chunks for RAG answering. |
| `RAG_RETRIEVAL_MODE` | `semantic` | `semantic` (vector only) or `hybrid` (BM25 + vector). |
| `RAG_HYBRID_ALPHA` | `0.7` | Hybrid weighting toward vector signal (0.0-1.0). |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for RAG indexing/query vectors. |
| `RAG_AUTO_BOOTSTRAP` | `true` | Auto-create/ingest collection when missing or empty. |
| `RAG_INGEST_MAX_CHARS` | `1800` | Chunk character cap used during ingest/cache logic. |
| `RAG_INGEST_BATCH_SIZE` | `64` | Batch size for embedding during ingest. |
| `OPENAI_BASE_URL` | `https://api.openai.com` | Optional OpenAI endpoint override (used by embeddings too). |
| `SEMANTIC_SIMILARITY_BACKEND` | `vector` | Semantic metric backend: `vector` or `llm`. |
| `SEMANTIC_EMBED_PROVIDER` | `openai` | Embedding provider for vector semantic metric: `openai` or `ollama`. |
| `SEMANTIC_EMBED_MODEL` | `text-embedding-3-small` | Embedding model for vector semantic similarity. |
| `SEMANTIC_EMBED_BASE_URL` | provider-specific | Base URL for semantic-embedding provider. |
| `SEMANTIC_EMBED_API_KEY` | empty | Optional API key override for semantic embeddings. |
| `JUDGE_LLM_PROVIDER` | main provider | LLM judge provider override (used when semantic backend is `llm`). |
| `JUDGE_LLM_MODEL` | main model | LLM judge model override. |
| `JUDGE_LLM_BASE_URL` | provider-specific | Judge provider base URL override. |
| `JUDGE_LLM_API_KEY` | provider key | Judge key override. |
| `JUDGE_LLM_TEMPERATURE` | `0.0` | Judge temperature; keep low for consistency. |
| `JUDGE_JSON_RETRIES` | `1` | Retry count for strict judge JSON repair/validation. |
| `RLM_TRACE_PATH` | `rlm_trace_tree.json` | Output path for trace-tree tooling. |

## 7) Common Beginner Workflows

### A) TUI-first with RLM + RAG

1. Set `ENABLED_APPROACHES=rlm,rag`.
2. Start TUI.
3. Ask a question.
4. Check post-turn metrics in chat.
5. Tune retrieval with `/rag topk`, `/rag mode`, `/rag alpha`.

### B) Baseline comparison without RLM

1. Set `ENABLED_APPROACHES=traditional,rag`.
2. Start TUI.
3. Ask question.
4. Observe direct baseline-vs-baseline metrics.

### C) Reproducible benchmark run

1. Edit `eval_questions.txt`.
2. Run `rlm_eval.py`.
3. Inspect `eval_results.json`.

## 8) Troubleshooting

- **`ModuleNotFoundError: weaviate`**: run dependency sync in the active env (`uv sync --active`).
- **RAG initialization/connection failures**: verify local Weaviate is running and accessible; current RAG path does not yet target remote/managed vector DB endpoints.
- **RAG empty results**: run `/rag status`, then `/rag retry` or `/rag reset`.
- **`SandboxLimitExceeded`**: increase `RLM_SANDBOX_EXEC_TIMEOUT_S` and/or `RLM_SANDBOX_MAX_EXEC_LINES`.
- **No API calls working**: verify provider, base URL, and matching API key are set in `.env`.
