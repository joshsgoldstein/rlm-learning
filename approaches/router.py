from __future__ import annotations

from typing import List

from rlm_core import (
    ChatMessage,
    Config,
    CONVERSATIONAL_ROUTER_PROMPT_PATH,
    _render_prompt,
    call_llm,
)


def build_router_corpus_context(doc_ids: List[str], max_items: int = 12) -> tuple[str, str]:
    """Build a compact corpus summary for router prompts."""
    ids = doc_ids or []
    doc_count = str(len(ids))
    if not ids:
        return doc_count, "(empty corpus)"
    if len(ids) <= max_items:
        return doc_count, "\n".join(f"  - {did}" for did in ids)
    sample = ids[:max_items]
    sample_text = "\n".join(f"  - {did}" for did in sample)
    return doc_count, sample_text + f"\n  ... and {len(ids) - max_items} more"


def classify_route(question: str, history: List[ChatMessage], cfg: Config, doc_ids: List[str] | None = None) -> str:
    """
    Route a user question as:
      - CHAT (no corpus research required)
      - RESEARCH (corpus research required)
    """
    recent = "\n".join(f"{m.role}: {m.content[:200]}" for m in history[-4:]) if history else "(no prior conversation)"
    doc_count, doc_sample = build_router_corpus_context(doc_ids or [])
    prompt = _render_prompt(
        CONVERSATIONAL_ROUTER_PROMPT_PATH,
        DOC_COUNT=doc_count,
        DOC_SAMPLE=doc_sample,
        RECENT=recent,
        QUESTION=question,
    )
    result = call_llm(prompt, cfg, usage_accum=None)  # routing tokens intentionally not counted
    return "CHAT" if "CHAT" in result.upper() else "RESEARCH"
