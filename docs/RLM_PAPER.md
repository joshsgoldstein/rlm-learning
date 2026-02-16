# The RLM Paper — Recursive Language Models

> Zhang, A. L., Kraska, T., & Khattab, O. (2025). *Recursive Language Models.* arXiv:2512.24601.

- **Paper:** <https://arxiv.org/abs/2512.24601>
- **Official code:** <https://github.com/alexzhang13/rlm>
- **Minimal implementation:** <https://github.com/alexzhang13/rlm-minimal>
- **Blog post:** <https://alexzhang13.github.io/blog/2025/rlm/>

## The Problem: Context Rot

Even with 100K+ token context windows, LLM performance **degrades as input length grows**. The authors call this *context rot* — the query drowns in noise, the model's attention spreads thin, and answer quality drops. Traditional approaches either:
- **Summarize** → lose information
- **RAG (retrieval)** → require embeddings, vector DBs, chunking strategies — the *system* decides what's relevant, not the LLM

## The Core Idea: Context as Variable, Not Input

RLMs flip the paradigm. Instead of feeding documents into the prompt:

```python
# Traditional (context rot)
llm.complete(prompt="Answer this", context=huge_document)

# RLM (context as variable)
rlm.complete(query="Answer this", context=huge_document)
# → context stored in REPL, LLM writes code to explore it
```

The LLM gets a **Python REPL** where documents are stored as variables. It writes code to:
1. **Peek** at sections (`context[:2000]`) → understand structure
2. **Search** with regex (`grep(context, pattern)`) → find relevant parts
3. **Recursively call itself** (`llm_query(sub_question, chunk)`) → extract facts from small pieces
4. **Compose an answer** from the evidence gathered

The model only ever sees metadata and execution results — never the raw bulk text.

## Architecture (§3)

Four components:

| Component | What it does | Our implementation |
|---|---|---|
| **Environment** — Python REPL storing context as variables | Documents live externally, LLM accesses via code | `RLMSandbox` in `rlm_core.py` — persistent `exec()` with `peek()`, `search()`, `llm_query()`, `answer()` |
| **Root LM** — Decides what to inspect, writes code | Controls the decomposition strategy | The LLM generates Python each iteration, prompted by `prompts/rlm_system.txt` |
| **Sub-LM Calls** — Recursive calls on chunks | `llm_query(question, text)` invokes the LLM on a small piece | `RLMSandbox.llm_query()` — makes a focused LLM call on a page chunk |
| **Code Execution** — Runs in the REPL | State persists across iterations like a Jupyter notebook | `RLMSandbox.execute_code()` — `exec()` with persistent globals |

### Key Design Principle

**Language for reasoning, code for memory.** LLMs are good at reasoning but bad at storing/retrieving large amounts of text. RLMs separate these concerns:
- The LLM *reasons* about what to look at next
- *Code* handles the memory (storing, slicing, searching documents)

## Key Results (§5)

- Processes inputs **up to 10M+ tokens** — 100× beyond model context windows — without quality degradation
- **RLM + GPT-5-mini outperforms vanilla GPT-5** on the OOLONG benchmark (>2× correct answers) while being **cheaper per query**
- Fine-tuned **RLM-Qwen3-8B outperforms base Qwen3-8B by 28.3%** on average
- Comparable or lower cost than ReAct + retrieval baselines
- Works across four diverse long-context benchmarks

## Why It Matters

| Approach | Scales to | Cost | Quality at scale |
|---|---|---|---|
| Traditional (stuff in context) | ~100K tokens, then truncate | Linear with corpus | Degrades (context rot) |
| RAG (embeddings + retrieval) | Large, but system decides relevance | Infrastructure + API | Depends on chunking strategy |
| **RLM (agent explores via code)** | **Arbitrary** — 10M+ tokens | **Logarithmic** — reads only what's needed | **Maintained** — each prompt is small and focused |

## Paper Sections Referenced in the TUI

The educational TUI (`rlm_tui.py`) shows paper concepts at each step:

| Agent action | TUI annotation | Paper section |
|---|---|---|
| New iteration | "RLM Loop — the LLM writes code to explore the external environment" | §3 |
| `peek()` called | "Context as Variable — documents stored externally, not in the context window" | §3.1 |
| `search()` called | "Programmatic Search — the LLM wrote code to grep the environment" | §3.2 |
| `llm_query()` called | "Recursive Sub-Call — the LLM invokes itself on a small chunk" | §3.3 |
| `answer()` called | "Evidence Synthesis — cited answer without seeing the full corpus" | §4 |
| Cost comparison | "Traditional: ~38K tokens / RLM: 12K tokens (69% less)" | §5 |
