from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rlm_core import (
    Config,
    AnswerResult,
    IterationStats,
    TokenUsage,
    TraditionalStats,
    call_llm,
    load_document_pages,
    _render_prompt,
)


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
TRADITIONAL_PROMPT_PATH = PROMPTS_DIR / "traditional_answer.txt"


def answer_traditional(
    doc_map: Dict[str, str],
    question: str,
    cfg: Config,
    on_event: Optional[Callable] = None,
) -> AnswerResult:
    """Traditional approach: stuff ALL document text into one prompt."""
    usage = TokenUsage()
    start_total = time.perf_counter()
    _on_event = on_event or (lambda *a: None)
    _on_event("traditional_stage", f"Preparing traditional run over {len(doc_map)} documents")

    all_docs: List[str] = []
    doc_ids_ordered = list(doc_map.keys())
    doc_page_counts: Dict[str, int] = {}
    total_chars = 0
    total_docs = max(1, len(doc_map))
    for idx, (doc_id, path) in enumerate(doc_map.items(), start=1):
        if idx == 1 or idx % 10 == 0 or idx == total_docs:
            _on_event("traditional_stage", f"Loading document text: {idx}/{total_docs}")
        pages = load_document_pages(path)
        doc_page_counts[doc_id] = len(pages)
        full_text = "\n".join(pages)
        all_docs.append(f"=== DOCUMENT: {doc_id} ===\n{full_text}")
        total_chars += len(full_text)

    _on_event("traditional_stage", f"Loaded corpus: {total_chars:,} chars from {len(all_docs)} docs")
    combined = "\n\n".join(all_docs)

    max_context_tokens = cfg.context_window - 2_000
    max_chars = max_context_tokens * 4
    truncated = False
    docs_included = len(all_docs)
    docs_excluded: List[str] = []
    if len(combined) > max_chars:
        combined = combined[:max_chars]
        truncated = True
        docs_included = combined.count("=== DOCUMENT:")
        docs_excluded = doc_ids_ordered[docs_included:]
        _on_event(
            "traditional_stage",
            f"Context limit hit: {docs_included}/{len(all_docs)} docs fit (truncated to {max_chars:,} chars)",
        )
    else:
        _on_event("traditional_stage", "All documents fit in context window")

    chars_sent = min(len(combined), max_chars)
    context_used_pct = (chars_sent / 4) / cfg.context_window * 100 if cfg.context_window > 0 else 0

    trad_stats = TraditionalStats(
        total_docs=len(doc_map),
        docs_included=docs_included,
        docs_excluded=docs_excluded,
        total_chars=total_chars,
        chars_sent=chars_sent,
        truncated=truncated,
        context_window=cfg.context_window,
        context_used_pct=min(context_used_pct, 100.0),
    )

    prompt = _render_prompt(
        TRADITIONAL_PROMPT_PATH,
        DOCUMENTS=combined,
        TRUNCATION_NOTE="(NOTE: Documents were truncated to fit in context window — information was lost.)" if truncated else "",
        QUESTION=question,
    )

    est_tokens = max(1, len(prompt) // 4)
    _on_event("traditional_stage", f"Calling LLM (~{est_tokens:,} input tokens)")
    llm_t0 = time.perf_counter()
    answer = call_llm(prompt, cfg, usage_accum=usage)
    llm_ms = (time.perf_counter() - llm_t0) * 1000.0
    _on_event("traditional_stage", "Traditional answer complete")

    if truncated:
        answer += (
            f"\n\n*⚠️ Traditional approach truncated: {total_chars:,} chars → {max_chars:,} chars "
            f"({docs_included}/{len(all_docs)} docs fit in {cfg.context_window:,} token context window of {cfg.llm_model})*"
        )

    return AnswerResult(
        answer=answer,
        usage=usage,
        iterations=1,
        iteration_stats=[
            IterationStats(
                iteration=1,
                action="traditional",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                planner_llm_ms=llm_ms,
                iteration_ms=(time.perf_counter() - start_total) * 1000.0,
            )
        ],
        code_history=[f"# Full corpus: {total_chars:,} chars across {len(doc_map)} docs" + (f" (TRUNCATED to {max_chars:,})" if truncated else "")],
        traditional_stats=trad_stats,
        evidence_manifest={
            "type": "traditional",
            "docs_included": doc_ids_ordered[:docs_included],
            "docs_excluded": docs_excluded,
            "doc_page_counts": {doc_id: doc_page_counts.get(doc_id, 0) for doc_id in doc_ids_ordered[:docs_included]},
            "truncated": truncated,
        },
    )
