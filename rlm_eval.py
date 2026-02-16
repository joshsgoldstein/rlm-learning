from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
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
JUDGE_REQUIRED_KEYS = {
    "semantic_score",
    "topic_overlap_score",
    "factuality_groundedness_score",
    "evidence_sufficiency_score",
    "hallucination_risk_flag",
    "winner",
    "confidence",
    "baseline_has_citations",
    "rlm_has_citations",
    "semantic_reason",
    "topic_reason",
}
SEMANTIC_BACKENDS = {"llm", "vector"}


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


def _extract_citation_chunks(answer: str) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(r"\[([^\]]+)\]", answer)]


def has_doc_page_citation(answer: str) -> bool:
    """True when at least one citation includes an explicit page reference."""
    for chunk in _extract_citation_chunks(answer):
        if re.search(r"\bp(?:age)?\s*\d+(?:\s*-\s*\d+)?\b", chunk, flags=re.IGNORECASE):
            return True
    return False


def extract_cited_docs(answer: str) -> List[str]:
    cited: List[str] = []
    for text in _extract_citation_chunks(answer):
        text = re.sub(r"\s+p(?:age)?\s*\d+(?:\s*-\s*\d+)?\s*$", "", text, flags=re.IGNORECASE).strip()
        if text and text not in cited:
            cited.append(text)
    return cited


def extract_evidence_docs(manifest: object) -> set[str]:
    """Extract doc ids touched/retrieved by an approach from evidence manifest."""
    if not isinstance(manifest, dict):
        return set()

    out: set[str] = set()

    def _add_many(items: object) -> None:
        if not isinstance(items, list):
            return
        for x in items:
            s = str(x).strip()
            if s:
                out.add(s)

    _add_many(manifest.get("docs_touched"))
    _add_many(manifest.get("docs_included"))
    _add_many(manifest.get("retrieved_doc_ids"))

    for key in ("peeked_pages", "search_hit_pages", "retrieved_chunks"):
        rows = manifest.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    did = str(row.get("doc_id", "")).strip()
                    if did:
                        out.add(did)
    return out


def format_evidence_manifest(manifest: object) -> str:
    if isinstance(manifest, dict):
        return json.dumps(manifest, indent=2, ensure_ascii=True)[:8000]
    return "{}"


def _extract_json_obj(raw: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", raw)
    obj = json.loads(m.group(0) if m else raw)
    if not isinstance(obj, dict):
        raise ValueError("judge output is not a JSON object")
    return obj


def _validate_judge_obj(obj: dict) -> dict:
    missing = sorted(JUDGE_REQUIRED_KEYS - set(obj.keys()))
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")

    # Validate numeric fields.
    for k in (
        "semantic_score",
        "topic_overlap_score",
        "factuality_groundedness_score",
        "evidence_sufficiency_score",
        "confidence",
    ):
        v = float(obj[k])
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{k} out of range 0..1")

    # Validate enums.
    risk = str(obj["hallucination_risk_flag"]).strip().lower()
    if risk not in {"low", "med", "high"}:
        raise ValueError("hallucination_risk_flag must be low|med|high")
    winner = str(obj["winner"]).strip().lower()
    if winner not in {"baseline", "rlm", "tie"}:
        raise ValueError("winner must be baseline|rlm|tie")

    # Validate booleans.
    if not isinstance(obj["baseline_has_citations"], bool):
        raise ValueError("baseline_has_citations must be true/false")
    if not isinstance(obj["rlm_has_citations"], bool):
        raise ValueError("rlm_has_citations must be true/false")

    # Validate strings.
    if not str(obj["semantic_reason"]).strip():
        raise ValueError("semantic_reason must be non-empty")
    if not str(obj["topic_reason"]).strip():
        raise ValueError("topic_reason must be non-empty")

    return obj


def lexical_overlap(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def tfidf_cosine_similarity(a: str, b: str) -> float:
    """Compute cosine similarity on TF-IDF vectors for two texts."""
    def _tokens(s: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_]{2,}", (s or "").lower())

    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0

    tf_a = Counter(ta)
    tf_b = Counter(tb)
    vocab = set(tf_a.keys()) | set(tf_b.keys())
    if not vocab:
        return 0.0

    n_docs = 2
    df = {}
    for t in vocab:
        df[t] = (1 if t in tf_a else 0) + (1 if t in tf_b else 0)

    # Smooth IDF commonly used in sklearn-style variants.
    idf = {t: math.log((1 + n_docs) / (1 + df[t])) + 1.0 for t in vocab}

    len_a = max(1, sum(tf_a.values()))
    len_b = max(1, sum(tf_b.values()))
    wa = {t: (tf_a[t] / len_a) * idf[t] for t in vocab}
    wb = {t: (tf_b[t] / len_b) * idf[t] for t in vocab}

    dot = sum(wa[t] * wb[t] for t in vocab)
    na = math.sqrt(sum(v * v for v in wa.values()))
    nb = math.sqrt(sum(v * v for v in wb.values()))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


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


def semantic_backend_from_env() -> str:
    raw = os.getenv("SEMANTIC_SIMILARITY_BACKEND", "vector").strip().lower()
    return raw if raw in SEMANTIC_BACKENDS else "llm"


def _semantic_embed_config(base_cfg: Config) -> tuple[str, str, str, str]:
    provider = os.getenv("SEMANTIC_EMBED_PROVIDER", "openai").strip().lower() or "openai"
    model = os.getenv("SEMANTIC_EMBED_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"
    base_url = os.getenv("SEMANTIC_EMBED_BASE_URL", "").strip()
    api_key = os.getenv("SEMANTIC_EMBED_API_KEY", "").strip()

    if provider == "openai":
        if not base_url:
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "")
    elif provider == "ollama":
        if not base_url:
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434").rstrip("/")
    else:
        raise ValueError(f"Unsupported SEMANTIC_EMBED_PROVIDER '{provider}' (use openai|ollama)")
    return provider, model, base_url, api_key


def _embed_text(text: str, base_cfg: Config) -> List[float]:
    provider, model, base_url, api_key = _semantic_embed_config(base_cfg)
    payload = (text or "").strip()[:12000] or "(empty)"
    if provider == "openai":
        from openai import OpenAI

        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        client = OpenAI(api_key=api_key, base_url=base)
        resp = client.embeddings.create(model=model, input=payload)
        return list(resp.data[0].embedding)

    # ollama
    import requests

    r = requests.post(
        f"{base_url}/api/embeddings",
        json={"model": model, "prompt": payload},
        timeout=120,
    )
    r.raise_for_status()
    body = r.json()
    emb = body.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise ValueError("Invalid embedding response from Ollama")
    return emb


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    na = math.sqrt(sum(float(a[i]) * float(a[i]) for i in range(n)))
    nb = math.sqrt(sum(float(b[i]) * float(b[i]) for i in range(n)))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    cos = dot / (na * nb)
    return max(-1.0, min(1.0, cos))


@dataclass
class SemanticJudgeResult:
    backend: str
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


def _build_semantic_judge_prompt(
    reference: str,
    candidate: str,
    reference_label: str = "Baseline",
    candidate_label: str = "RLM",
    reference_evidence: str = "{}",
    candidate_evidence: str = "{}",
) -> str:
    template = SEMANTIC_JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template
        .replace("{{BASELINE_LABEL}}", reference_label)
        .replace("{{RLM_LABEL}}", candidate_label)
        .replace("{{BASELINE_ANSWER}}", reference[:8000])
        .replace("{{RLM_ANSWER}}", candidate[:8000])
        .replace("{{BASELINE_EVIDENCE}}", reference_evidence[:8000])
        .replace("{{RLM_EVIDENCE}}", candidate_evidence[:8000])
    )


def semantic_similarity_llm(
    reference: str,
    candidate: str,
    cfg: Config,
    reference_label: str = "Baseline",
    candidate_label: str = "RLM",
    reference_evidence: str = "{}",
    candidate_evidence: str = "{}",
) -> SemanticJudgeResult:
    """Semantic comparison with configurable backend: llm or vector cosine."""
    backend = semantic_backend_from_env()
    if backend == "vector":
        usage = TokenUsage()
        try:
            ref_vec = _embed_text(reference, cfg)
            cand_vec = _embed_text(candidate, cfg)
            semantic_score = (_cosine_similarity(ref_vec, cand_vec) + 1.0) / 2.0
            semantic_reason = "Cosine similarity from embedding vectors."
        except Exception as e:
            semantic_score = 0.0
            semantic_reason = f"Vector similarity failed: {type(e).__name__}: {e}"
        return SemanticJudgeResult(
            backend="vector",
            semantic_score=max(0.0, min(1.0, semantic_score)),
            topic_overlap_score=0.0,
            factuality_groundedness_score=0.0,
            evidence_sufficiency_score=0.0,
            hallucination_risk_flag="med",
            winner="tie",
            confidence=0.0,
            answer_a_has_citations=has_doc_page_citation(reference),
            answer_b_has_citations=has_doc_page_citation(candidate),
            semantic_reason=semantic_reason,
            topic_reason="Not computed in vector backend.",
            usage=usage,
        )

    # LLM backend
    usage = TokenUsage()
    prompt = _build_semantic_judge_prompt(
        reference,
        candidate,
        reference_label,
        candidate_label,
        reference_evidence,
        candidate_evidence,
    )
    judge_cfg = _judge_config(cfg)
    raw = call_llm(prompt, judge_cfg, usage_accum=usage).strip()
    retries = max(0, int(os.getenv("JUDGE_JSON_RETRIES", "1")))

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
    parse_error = ""
    obj = None
    for attempt in range(retries + 1):
        try:
            obj = _validate_judge_obj(_extract_json_obj(raw))
            break
        except Exception as e:
            parse_error = str(e)
            if attempt >= retries:
                break
            raw = call_llm(
                (
                    "Your previous output failed strict JSON validation.\n"
                    f"Validation error: {parse_error}\n"
                    "Return ONLY one valid JSON object that exactly matches the required schema keys/types.\n"
                    "Do not include markdown or extra text.\n\n"
                    f"Previous output:\n{raw[:4000]}"
                ),
                judge_cfg,
                usage_accum=usage,
            ).strip()

    if obj is not None:
        semantic_score = float(obj["semantic_score"])
        topic_score = float(obj["topic_overlap_score"])
        factuality_score = float(obj["factuality_groundedness_score"])
        evidence_score = float(obj["evidence_sufficiency_score"])
        hallucination_risk_flag = str(obj["hallucination_risk_flag"]).strip().lower()
        winner = str(obj["winner"]).strip().lower()
        confidence = float(obj["confidence"])
        a_has_citations = bool(obj["baseline_has_citations"])
        b_has_citations = bool(obj["rlm_has_citations"])
        semantic_reason = str(obj["semantic_reason"]).strip() or semantic_reason
        topic_reason = str(obj["topic_reason"]).strip() or topic_reason
    else:
        semantic_reason = f"Judge JSON parse failed: {parse_error or 'unknown error'}"
        topic_reason = "No valid structured JSON returned by judge."
        a_has_citations = has_doc_page_citation(reference)
        b_has_citations = has_doc_page_citation(candidate)

    # Deterministic guardrail: if explicit doc+page citations exist in text, mark true.
    a_has_citations = a_has_citations or has_doc_page_citation(reference)
    b_has_citations = b_has_citations or has_doc_page_citation(candidate)

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
        backend="llm",
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
    baseline_evidence_docs: int
    rlm_evidence_docs: int
    evidence_doc_overlap_count: int
    evidence_doc_baseline_only_count: int
    evidence_doc_rlm_only_count: int
    evidence_doc_jaccard: float
    overlap_ratio: float
    tfidf_similarity_score: float
    semantic_backend: str
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
        baseline_ev_docs = extract_evidence_docs(trad.evidence_manifest)
        rlm_ev_docs = extract_evidence_docs(rlm.evidence_manifest)
        if not baseline_ev_docs and trad_cites:
            baseline_ev_docs = set(trad_cites)
        if not rlm_ev_docs and rlm_cites:
            rlm_ev_docs = set(rlm_cites)
        ev_overlap = baseline_ev_docs & rlm_ev_docs
        ev_baseline_only = baseline_ev_docs - rlm_ev_docs
        ev_rlm_only = rlm_ev_docs - baseline_ev_docs
        ev_union_n = len(baseline_ev_docs | rlm_ev_docs)
        ev_jaccard = (len(ev_overlap) / ev_union_n) if ev_union_n else 0.0
        overlap = lexical_overlap(trad.answer, rlm.answer)
        tfidf_score = tfidf_cosine_similarity(trad.answer, rlm.answer)
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
            judge = semantic_similarity_llm(
                trad.answer,
                rlm.answer,
                cfg,
                reference_label="Traditional",
                candidate_label="RLM",
                reference_evidence=format_evidence_manifest(trad.evidence_manifest),
                candidate_evidence=format_evidence_manifest(rlm.evidence_manifest),
            )
            sem_score = judge.semantic_score
            sem_backend = judge.backend
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
        else:
            sem_backend = "disabled"

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
        emit(f"  TF-IDF cosine similarity: {tfidf_score:.2f}")
        emit(
            "  evidence docs overlap "
            f"(baseline/RLM/overlap/baseline-only/RLM-only): "
            f"{len(baseline_ev_docs)}/{len(rlm_ev_docs)}/{len(ev_overlap)}/{len(ev_baseline_only)}/{len(ev_rlm_only)}"
        )
        emit(f"  evidence docs jaccard: {ev_jaccard:.2f}")
        if use_semantic_judge:
            backend_label = "LLM judge" if sem_backend == "llm" else "vector cosine"
            emit(
                f"  semantic similarity ({backend_label}): {sem_score:.2f} "
                f"(judge: {sem_usage.total_tokens:,} tok, ${estimate_cost(sem_usage, cfg.llm_model):.4f})"
            )
            emit(f"  semantic note: {sem_reason}")
            if sem_backend == "llm":
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
                baseline_evidence_docs=len(baseline_ev_docs),
                rlm_evidence_docs=len(rlm_ev_docs),
                evidence_doc_overlap_count=len(ev_overlap),
                evidence_doc_baseline_only_count=len(ev_baseline_only),
                evidence_doc_rlm_only_count=len(ev_rlm_only),
                evidence_doc_jaccard=ev_jaccard,
                overlap_ratio=overlap,
                tfidf_similarity_score=tfidf_score,
                semantic_backend=sem_backend,
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
    avg_tfidf = sum(r.tfidf_similarity_score for r in rows) / max(1, total)
    avg_ev_overlap = sum(r.evidence_doc_overlap_count for r in rows) / max(1, total)
    avg_ev_baseline_only = sum(r.evidence_doc_baseline_only_count for r in rows) / max(1, total)
    avg_ev_rlm_only = sum(r.evidence_doc_rlm_only_count for r in rows) / max(1, total)
    avg_ev_jaccard = sum(r.evidence_doc_jaccard for r in rows) / max(1, total)
    avg_semantic = sum(r.semantic_similarity_score for r in rows) / max(1, total)
    backend_counts = {
        "llm": sum(1 for r in rows if r.semantic_backend == "llm"),
        "vector": sum(1 for r in rows if r.semantic_backend == "vector"),
        "disabled": sum(1 for r in rows if r.semantic_backend == "disabled"),
    }
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
    emit(f"avg TF-IDF cosine similarity: {avg_tfidf:.2f}")
    emit(
        f"avg evidence docs (overlap/baseline-only/RLM-only): "
        f"{avg_ev_overlap:.2f}/{avg_ev_baseline_only:.2f}/{avg_ev_rlm_only:.2f}"
    )
    emit(f"avg evidence-doc jaccard: {avg_ev_jaccard:.2f}")
    emit(f"avg semantic similarity: {avg_semantic:.2f} (backends llm/vector/disabled={backend_counts['llm']}/{backend_counts['vector']}/{backend_counts['disabled']})")
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
            "avg_tfidf_similarity": avg_tfidf,
            "avg_evidence_doc_overlap_count": avg_ev_overlap,
            "avg_evidence_doc_baseline_only_count": avg_ev_baseline_only,
            "avg_evidence_doc_rlm_only_count": avg_ev_rlm_only,
            "avg_evidence_doc_jaccard": avg_ev_jaccard,
            "avg_semantic_similarity": avg_semantic,
            "semantic_backend_counts": backend_counts,
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
