from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, List

from dotenv import load_dotenv

from rlm_core import Config, TokenUsage, answer_question, answer_traditional, call_llm, estimate_cost
from rlm_docs import discover_docs, list_pdf_files, missing_processed_pdfs, preprocess_pdfs
from rlm_event_stream import make_event_callback


load_dotenv()


DATA_DIR = Path(os.getenv("RLM_DATA_DIR", "data"))
PROCESSED_DIR = Path(os.getenv("RLM_PROCESSED_DIR", "processed_data"))

DEFAULT_QUESTIONS = [
    "How did Deloitte's position on enterprise AI evolve from 2025 to 2026?",
    "What are the key differences between build, buy, and borrow approaches for AI agents?",
    "What governance and risk controls are repeatedly recommended across reports?",
]
SEMANTIC_JUDGE_PROMPT_PATH = Path("prompts/semantic_judge.txt")
DEFAULT_PROVIDER_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
}


def preprocess_missing_pdfs(verbose: bool = True, emit: Callable[[str], None] = print) -> None:
    src, dst = DATA_DIR, PROCESSED_DIR
    if not src.is_dir():
        return

    all_pdfs = list_pdf_files(src)
    if not all_pdfs:
        return

    dst.mkdir(parents=True, exist_ok=True)
    missing = missing_processed_pdfs(all_pdfs, src, dst)
    if not missing:
        if verbose:
            emit(f"[init] processed_data ready ({len(all_pdfs)}/{len(all_pdfs)} already extracted)")
        return

    if verbose:
        emit(f"[init] {len(missing)}/{len(all_pdfs)} PDFs missing in {PROCESSED_DIR}/, processing now...")

    def _on_progress(evt: str, detail: str) -> None:
        if not verbose:
            return
        if evt == "file_start":
            emit(f"  extracting {detail}")
        elif evt == "file_done":
            emit(f"    done {detail}")
        elif evt == "file_error":
            emit(f"    skipped {detail}")

    preprocess_pdfs(pdfs=missing, data_dir=src, processed_dir=dst, on_progress=_on_progress)


def load_questions(path: str | None) -> List[str]:
    if not path:
        return DEFAULT_QUESTIONS
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Questions file not found: {p}")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def extract_cited_docs(answer: str) -> List[str]:
    cited: List[str] = []
    # Matches [doc_id], [doc_id p12], [doc_id p 12]
    for m in re.finditer(r"\[([^\]]+)\]", answer):
        text = m.group(1).strip()
        text = re.sub(r"\s+p\s*\d+\s*$", "", text, flags=re.IGNORECASE).strip()
        if text and text not in cited:
            cited.append(text)
    return cited


def lexical_overlap(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _judge_config(base_cfg: Config) -> Config:
    """Build judge config, allowing override of provider/model independent of main run."""
    provider = os.getenv("JUDGE_LLM_PROVIDER", base_cfg.llm_provider).lower().strip()
    model = os.getenv("JUDGE_LLM_MODEL", base_cfg.llm_model).strip()

    base_url = os.getenv("JUDGE_LLM_BASE_URL", "").strip()
    if not base_url:
        if provider == base_cfg.llm_provider:
            base_url = base_cfg.llm_base_url
        else:
            base_url = DEFAULT_PROVIDER_URLS.get(provider, base_cfg.llm_base_url)

    # Judge-specific key wins; else fallback by provider.
    api_key = os.getenv("JUDGE_LLM_API_KEY", "").strip()
    if not api_key:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        else:
            api_key = os.getenv("LLM_API_KEY", base_cfg.llm_api_key)

    try:
        judge_temp = float(os.getenv("JUDGE_LLM_TEMPERATURE", "0.0"))
    except ValueError:
        judge_temp = 0.0

    return Config(
        llm_provider=provider,
        llm_model=model,
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_temperature=judge_temp,
        max_iterations=base_cfg.max_iterations,
        hard_max_iterations=base_cfg.hard_max_iterations,
        sandbox_exec_timeout_s=base_cfg.sandbox_exec_timeout_s,
        sandbox_max_exec_lines=base_cfg.sandbox_max_exec_lines,
        max_history_messages=base_cfg.max_history_messages,
        context_window=base_cfg.context_window,
    )


@dataclass
class SemanticJudgeResult:
    semantic_score: float
    topic_overlap_score: float
    factuality_groundedness_score: float
    evidence_sufficiency_score: float
    hallucination_risk_flag: str
    winner: str
    confidence: float
    answer_a_has_citations: bool
    answer_b_has_citations: bool
    semantic_reason: str
    topic_reason: str
    usage: TokenUsage


def _build_semantic_judge_prompt(reference: str, candidate: str) -> str:
    template = SEMANTIC_JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{{ANSWER_A}}", reference[:8000]).replace("{{ANSWER_B}}", candidate[:8000])


def semantic_similarity_llm(reference: str, candidate: str, cfg: Config) -> SemanticJudgeResult:
    """LLM judge for semantic and topic overlap (0.0-1.0)."""
    usage = TokenUsage()
    prompt = _build_semantic_judge_prompt(reference, candidate)
    judge_cfg = _judge_config(cfg)
    raw = call_llm(prompt, judge_cfg, usage_accum=usage).strip()

    semantic_score = 0.0
    topic_score = 0.0
    factuality_score = 0.0
    evidence_score = 0.0
    hallucination_risk_flag = "med"
    winner = "tie"
    confidence = 0.0
    a_has_citations = False
    b_has_citations = False
    semantic_reason = "Could not parse semantic reason."
    topic_reason = "Could not parse topic overlap reason."
    try:
        # Handle plain JSON or JSON embedded in markdown.
        m = re.search(r"\{[\s\S]*\}", raw)
        obj = json.loads(m.group(0) if m else raw)
        semantic_score = float(obj.get("semantic_score", obj.get("score", 0.0)))
        topic_score = float(obj.get("topic_overlap_score", 0.0))
        factuality_score = float(obj.get("factuality_groundedness_score", 0.0))
        evidence_score = float(obj.get("evidence_sufficiency_score", 0.0))
        hallucination_risk_flag = str(obj.get("hallucination_risk_flag", "med")).strip().lower() or "med"
        winner = str(obj.get("winner", "tie")).strip().lower() or "tie"
        confidence = float(obj.get("confidence", 0.0))
        a_has_citations = bool(obj.get("answer_a_has_citations", False))
        b_has_citations = bool(obj.get("answer_b_has_citations", False))
        semantic_reason = str(obj.get("semantic_reason", obj.get("reason", ""))).strip() or semantic_reason
        topic_reason = str(obj.get("topic_reason", "")).strip() or topic_reason
    except Exception:
        # Best-effort fallback parse.
        m_scores = re.findall(r"([01](?:\.\d+)?)", raw)
        if m_scores:
            try:
                semantic_score = float(m_scores[0])
                if len(m_scores) > 1:
                    topic_score = float(m_scores[1])
            except Exception:
                semantic_score = 0.0
                topic_score = 0.0
        semantic_reason = raw[:200].replace("\n", " ").strip() or semantic_reason
        # Fallback citation detection by bracketed citation pattern.
        a_has_citations = bool(re.search(r"\[[^\]]+\]", reference))
        b_has_citations = bool(re.search(r"\[[^\]]+\]", candidate))

    semantic_score = max(0.0, min(1.0, semantic_score))
    topic_score = max(0.0, min(1.0, topic_score))
    factuality_score = max(0.0, min(1.0, factuality_score))
    evidence_score = max(0.0, min(1.0, evidence_score))
    confidence = max(0.0, min(1.0, confidence))
    if hallucination_risk_flag not in {"low", "med", "high"}:
        hallucination_risk_flag = "med"
    if winner not in {"baseline", "rlm", "tie"}:
        winner = "tie"
    return SemanticJudgeResult(
        semantic_score=semantic_score,
        topic_overlap_score=topic_score,
        factuality_groundedness_score=factuality_score,
        evidence_sufficiency_score=evidence_score,
        hallucination_risk_flag=hallucination_risk_flag,
        winner=winner,
        confidence=confidence,
        answer_a_has_citations=a_has_citations,
        answer_b_has_citations=b_has_citations,
        semantic_reason=semantic_reason,
        topic_reason=topic_reason,
        usage=usage,
    )


@dataclass
class EvalRow:
    question: str
    traditional_tokens: int
    rlm_tokens: int
    token_savings_pct: float
    traditional_cost_usd: float
    rlm_cost_usd: float
    traditional_truncated: bool
    traditional_docs_included: int
    traditional_total_docs: int
    rlm_cited_docs: int
    traditional_cited_docs: int
    overlap_ratio: float
    semantic_similarity_score: float
    topic_overlap_score: float
    factuality_groundedness_score: float
    evidence_sufficiency_score: float
    hallucination_risk_flag: str
    judge_winner: str
    judge_confidence: float
    judge_traditional_has_citations: bool
    judge_rlm_has_citations: bool
    semantic_similarity_reason: str
    topic_overlap_reason: str
    semantic_judge_tokens: int
    semantic_judge_cost_usd: float
    pass_min_cited_docs: bool
    pass_token_efficiency: bool


def run_eval(
    questions: List[str],
    cfg: Config,
    min_cited_docs: int,
    require_rlm_token_savings: bool,
    verbose_trace: bool = False,
    use_semantic_judge: bool = True,
    emit: Callable[[str], None] = print,
) -> List[EvalRow]:
    doc_map = discover_docs(DATA_DIR, PROCESSED_DIR)
    if not doc_map:
        raise RuntimeError(f"No documents found in {DATA_DIR}/ or {PROCESSED_DIR}/")

    emit(f"[eval] provider={cfg.llm_provider} model={cfg.llm_model}")
    emit(f"[eval] docs={len(doc_map)} questions={len(questions)}")

    rows: List[EvalRow] = []
    for i, q in enumerate(questions, start=1):
        emit(f"\n=== Q{i}: {q}")
        trad_cb = make_event_callback(emit=emit, prefix="[traditional]", show_section_headers=True) if verbose_trace else None
        rlm_cb = make_event_callback(emit=emit, prefix="[rlm]", show_section_headers=True) if verbose_trace else None

        trad = answer_traditional(doc_map=doc_map, question=q, cfg=cfg, on_event=trad_cb)
        rlm = answer_question(doc_map=doc_map, question=q, history=[], cfg=cfg, on_event=rlm_cb)

        trad_t = trad.usage.total_tokens
        rlm_t = rlm.usage.total_tokens
        savings_pct = (1 - (rlm_t / max(1, trad_t))) * 100

        trad_cites = extract_cited_docs(trad.answer)
        rlm_cites = extract_cited_docs(rlm.answer)
        overlap = lexical_overlap(trad.answer, rlm.answer)
        sem_score = 0.0
        topic_score = 0.0
        factuality_score = 0.0
        evidence_score = 0.0
        hallucination_risk_flag = "med"
        judge_winner = "tie"
        judge_confidence = 0.0
        sem_reason = "(semantic judge disabled)"
        topic_reason = "(semantic judge disabled)"
        judge_a_has_cites = False
        judge_b_has_cites = False
        sem_usage = TokenUsage()
        if use_semantic_judge:
            judge = semantic_similarity_llm(trad.answer, rlm.answer, cfg)
            sem_score = judge.semantic_score
            topic_score = judge.topic_overlap_score
            factuality_score = judge.factuality_groundedness_score
            evidence_score = judge.evidence_sufficiency_score
            hallucination_risk_flag = judge.hallucination_risk_flag
            judge_winner = judge.winner
            judge_confidence = judge.confidence
            judge_a_has_cites = judge.answer_a_has_citations
            judge_b_has_cites = judge.answer_b_has_citations
            sem_reason = judge.semantic_reason
            topic_reason = judge.topic_reason
            sem_usage = judge.usage

        ts = trad.traditional_stats
        pass_cites = len(rlm_cites) >= min_cited_docs
        pass_tokens = (savings_pct > 0) if require_rlm_token_savings else True

        emit(f"  traditional: {trad_t:,} tok (${estimate_cost(trad.usage, cfg.llm_model):.4f})")
        emit(f"  rlm:         {rlm_t:,} tok (${estimate_cost(rlm.usage, cfg.llm_model):.4f})")
        emit(f"  token delta: {savings_pct:+.1f}% (positive means RLM cheaper)")
        emit(f"  rlm cited docs: {len(rlm_cites)} (target >= {min_cited_docs})")
        if ts:
            trunc_msg = "yes" if ts.truncated else "no"
            emit(f"  traditional truncated: {trunc_msg} ({ts.docs_included}/{ts.total_docs} docs fit)")
        emit(f"  answer overlap ratio (RLM vs traditional): {overlap:.2f}")
        if use_semantic_judge:
            emit(
                f"  semantic similarity (LLM judge): {sem_score:.2f} "
                f"(judge: {sem_usage.total_tokens:,} tok, ${estimate_cost(sem_usage, cfg.llm_model):.4f})"
            )
            emit(f"  semantic note: {sem_reason}")
            emit(f"  topic overlap (LLM judge): {topic_score:.2f}")
            emit(f"  topic note: {topic_reason}")
            emit(f"  factuality/groundedness (LLM judge): {factuality_score:.2f}")
            emit(f"  evidence sufficiency (LLM judge): {evidence_score:.2f}")
            emit(f"  hallucination risk (LLM judge): {hallucination_risk_flag}")
            emit(f"  judge winner/confidence: {judge_winner}/{judge_confidence:.2f}")
            emit(f"  judge citation booleans (traditional/RLM): {judge_a_has_cites}/{judge_b_has_cites}")

        rows.append(
            EvalRow(
                question=q,
                traditional_tokens=trad_t,
                rlm_tokens=rlm_t,
                token_savings_pct=savings_pct,
                traditional_cost_usd=estimate_cost(trad.usage, cfg.llm_model),
                rlm_cost_usd=estimate_cost(rlm.usage, cfg.llm_model),
                traditional_truncated=bool(ts.truncated) if ts else False,
                traditional_docs_included=ts.docs_included if ts else 0,
                traditional_total_docs=ts.total_docs if ts else len(doc_map),
                rlm_cited_docs=len(rlm_cites),
                traditional_cited_docs=len(trad_cites),
                overlap_ratio=overlap,
                semantic_similarity_score=sem_score,
                topic_overlap_score=topic_score,
                factuality_groundedness_score=factuality_score,
                evidence_sufficiency_score=evidence_score,
                hallucination_risk_flag=hallucination_risk_flag,
                judge_winner=judge_winner,
                judge_confidence=judge_confidence,
                judge_traditional_has_citations=judge_a_has_cites,
                judge_rlm_has_citations=judge_b_has_cites,
                semantic_similarity_reason=sem_reason,
                topic_overlap_reason=topic_reason,
                semantic_judge_tokens=sem_usage.total_tokens,
                semantic_judge_cost_usd=estimate_cost(sem_usage, cfg.llm_model),
                pass_min_cited_docs=pass_cites,
                pass_token_efficiency=pass_tokens,
            )
        )
    return rows


def summarize(rows: List[EvalRow], out_json: str, emit: Callable[[str], None] = print) -> None:
    total = len(rows)
    pass_cites = sum(1 for r in rows if r.pass_min_cited_docs)
    pass_tokens = sum(1 for r in rows if r.pass_token_efficiency)
    avg_savings = sum(r.token_savings_pct for r in rows) / max(1, total)
    avg_overlap = sum(r.overlap_ratio for r in rows) / max(1, total)
    avg_semantic = sum(r.semantic_similarity_score for r in rows) / max(1, total)
    avg_topic = sum(r.topic_overlap_score for r in rows) / max(1, total)
    avg_factuality = sum(r.factuality_groundedness_score for r in rows) / max(1, total)
    avg_evidence = sum(r.evidence_sufficiency_score for r in rows) / max(1, total)
    avg_confidence = sum(r.judge_confidence for r in rows) / max(1, total)
    risk_counts = {
        "low": sum(1 for r in rows if r.hallucination_risk_flag == "low"),
        "med": sum(1 for r in rows if r.hallucination_risk_flag == "med"),
        "high": sum(1 for r in rows if r.hallucination_risk_flag == "high"),
    }
    winner_counts = {
        "baseline": sum(1 for r in rows if r.judge_winner == "baseline"),
        "rlm": sum(1 for r in rows if r.judge_winner == "rlm"),
        "tie": sum(1 for r in rows if r.judge_winner == "tie"),
    }
    judge_tokens = sum(r.semantic_judge_tokens for r in rows)
    judge_cost = sum(r.semantic_judge_cost_usd for r in rows)

    emit("\n=== SUMMARY ===")
    emit(f"questions: {total}")
    emit(f"pass cited-doc threshold: {pass_cites}/{total}")
    emit(f"pass token-efficiency gate: {pass_tokens}/{total}")
    emit(f"avg token savings (RLM vs traditional): {avg_savings:+.1f}%")
    emit(f"avg answer overlap ratio: {avg_overlap:.2f}")
    emit(f"avg semantic similarity (LLM judge): {avg_semantic:.2f}")
    emit(f"avg topic overlap (LLM judge): {avg_topic:.2f}")
    emit(f"avg factuality/groundedness (LLM judge): {avg_factuality:.2f}")
    emit(f"avg evidence sufficiency (LLM judge): {avg_evidence:.2f}")
    emit(f"avg judge confidence: {avg_confidence:.2f}")
    emit(f"hallucination risk counts (low/med/high): {risk_counts['low']}/{risk_counts['med']}/{risk_counts['high']}")
    emit(f"winner counts (baseline/rlm/tie): {winner_counts['baseline']}/{winner_counts['rlm']}/{winner_counts['tie']}")
    emit(f"semantic judge overhead: {judge_tokens:,} tokens (${judge_cost:.4f})")

    payload = {
        "summary": {
            "questions": total,
            "pass_cited_docs": pass_cites,
            "pass_token_efficiency": pass_tokens,
            "avg_token_savings_pct": avg_savings,
            "avg_overlap_ratio": avg_overlap,
            "avg_semantic_similarity": avg_semantic,
            "avg_topic_overlap": avg_topic,
            "avg_factuality_groundedness": avg_factuality,
            "avg_evidence_sufficiency": avg_evidence,
            "avg_judge_confidence": avg_confidence,
            "hallucination_risk_counts": risk_counts,
            "winner_counts": winner_counts,
            "semantic_judge_tokens": judge_tokens,
            "semantic_judge_cost_usd": judge_cost,
        },
        "rows": [asdict(r) for r in rows],
    }
    Path(out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    emit(f"saved: {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RLM vs traditional on fixed questions.")
    parser.add_argument("--questions-file", default="", help="Path to .txt file with one question per line")
    parser.add_argument("--out-json", default="eval_results.json", help="Where to write metrics JSON")
    parser.add_argument("--min-cited-docs", type=int, default=8, help="RLM must cite at least this many docs per question")
    parser.add_argument(
        "--no-token-gate",
        action="store_true",
        help="Disable pass/fail gate requiring RLM token savings over traditional",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip startup check for missing processed PDFs",
    )
    parser.add_argument(
        "--verbose-trace",
        action="store_true",
        help="Stream traditional stages and RLM iteration events (TUI-like) during evaluation runs",
    )
    parser.add_argument(
        "--no-semantic-judge",
        action="store_true",
        help="Disable LLM-based semantic similarity judging",
    )
    args = parser.parse_args()

    cfg = Config.from_env()
    if cfg.llm_provider in ("openai", "anthropic") and (not cfg.llm_api_key or "REPLACE" in cfg.llm_api_key):
        key = "OPENAI_API_KEY" if cfg.llm_provider == "openai" else "ANTHROPIC_API_KEY"
        raise RuntimeError(f"{key} not set in environment/.env")

    if not args.skip_preprocess:
        preprocess_missing_pdfs(verbose=True, emit=print)

    questions = load_questions(args.questions_file or None)
    rows = run_eval(
        questions=questions,
        cfg=cfg,
        min_cited_docs=args.min_cited_docs,
        require_rlm_token_savings=not args.no_token_gate,
        verbose_trace=args.verbose_trace,
        use_semantic_judge=not args.no_semantic_judge,
        emit=print,
    )
    summarize(rows, args.out_json, emit=print)


if __name__ == "__main__":
    main()
