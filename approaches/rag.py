from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import uuid
import hashlib

from rlm_core import (
    Config,
    AnswerResult,
    IterationStats,
    TokenUsage,
    call_llm,
    load_document_pages,
    _render_prompt,
)


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_answer.txt"


@dataclass
class RagStats:
    total_docs: int = 0
    retrieved_chunks: int = 0
    unique_docs_retrieved: int = 0
    retrieval_backend: str = "weaviate"
    collection_name: str = ""
    retrieval_ms: float = 0.0


@dataclass
class RagCollectionStatus:
    collection_name: str
    exists: bool
    chunk_objects: int
    unique_docs: int


@dataclass
class RagConfig:
    top_k: int = 10
    collection_name: str = "RLMChunk"
    embedding_model: str = "text-embedding-3-small"
    weaviate_mode: str = "local"
    auto_bootstrap: bool = True
    ingest_max_chars: int = 1800
    ingest_batch_size: int = 64
    retrieval_mode: str = "semantic"  # semantic | hybrid
    hybrid_alpha: float = 0.7

    @staticmethod
    def from_env() -> "RagConfig":
        return RagConfig(
            top_k=max(1, int(os.getenv("RAG_TOP_K", "10"))),
            collection_name=os.getenv("RAG_WEAVIATE_COLLECTION", "RLMChunk").strip() or "RLMChunk",
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small",
            weaviate_mode=os.getenv("RAG_WEAVIATE_MODE", "local").strip().lower() or "local",
            auto_bootstrap=os.getenv("RAG_AUTO_BOOTSTRAP", "true").strip().lower() in ("1", "true", "yes", "on"),
            ingest_max_chars=max(500, int(os.getenv("RAG_INGEST_MAX_CHARS", "1800"))),
            ingest_batch_size=max(1, int(os.getenv("RAG_INGEST_BATCH_SIZE", "64"))),
            retrieval_mode=(
                os.getenv("RAG_RETRIEVAL_MODE", "semantic").strip().lower()
                if os.getenv("RAG_RETRIEVAL_MODE", "semantic").strip().lower() in {"semantic", "hybrid"}
                else "semantic"
            ),
            hybrid_alpha=max(0.0, min(1.0, float(os.getenv("RAG_HYBRID_ALPHA", "0.7")))),
        )


def _openai_client():
    OpenAI = importlib.import_module("openai").OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for RAG query embeddings.")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return OpenAI(api_key=api_key, base_url=base)


def _embed_texts_openai(texts: List[str], model: str) -> List[List[float]]:
    client = _openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [list(d.embedding) for d in resp.data]


def _embed_query_openai(question: str, model: str) -> List[float]:
    client = _openai_client()
    resp = client.embeddings.create(model=model, input=question)
    return list(resp.data[0].embedding)


def _connect_weaviate(weaviate_mode: str):
    weaviate = importlib.import_module("weaviate")
    if weaviate_mode != "local":
        raise RuntimeError(f"Unsupported RAG_WEAVIATE_MODE='{weaviate_mode}'. Currently only 'local' is supported.")
    return weaviate.connect_to_local()


def _count_collection(coll) -> Tuple[int, int]:
    total = 0
    unique_docs = set()
    for obj in coll.iterator(return_properties=["doc_id"]):
        total += 1
        props = getattr(obj, "properties", {}) or {}
        did = props.get("doc_id")
        if did:
            unique_docs.add(str(did))
    return total, len(unique_docs)


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    total = max(1, total)
    done = min(max(0, done), total)
    fill = int((done / total) * width)
    return "[" + ("#" * fill) + ("-" * (width - fill)) + f"] {done}/{total}"


def _make_chunks(doc_map: Dict[str, str], max_chars: int) -> List[dict]:
    chunks: List[dict] = []
    _ = max_chars  # kept for cache/version compatibility; chunking is page-level.
    for doc_id, path in doc_map.items():
        pages = load_document_pages(path)
        for page_idx, page_text in enumerate(pages, start=1):
            text = (page_text or "").strip()
            if not text:
                continue
            # Page-level chunking: exactly one vector chunk per page.
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{page_idx}:{text[:64]}"))
            text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "uuid": chunk_id,
                    "doc_id": doc_id,
                    "source_path": path,
                    "text": text,
                    "text_hash": text_hash,
                    "page_start": page_idx,
                    "page_end": page_idx,
                }
            )
    return chunks


def _metadata_cache_path(source_path: str) -> Optional[Path]:
    p = Path(source_path)
    if p.is_dir():
        return p / "metadata.json"
    return None


def _load_doc_cache(path: Optional[Path], embedding_model: str, max_chars: int) -> Dict[str, Tuple[str, List[float]]]:
    if not path or not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if obj.get("embedding_model") != embedding_model or int(obj.get("ingest_max_chars", -1)) != int(max_chars):
        return {}
    out: Dict[str, Tuple[str, List[float]]] = {}
    for row in obj.get("chunks", []):
        cid = str(row.get("chunk_id", ""))
        th = str(row.get("text_hash", ""))
        emb = row.get("embedding")
        if cid and th and isinstance(emb, list) and emb:
            out[cid] = (th, emb)
    return out


def _write_doc_cache(path: Optional[Path], embedding_model: str, max_chars: int, doc_chunks: List[dict], vectors_by_uuid: Dict[str, List[float]]) -> None:
    if not path:
        return
    rows = []
    for c in doc_chunks:
        vec = vectors_by_uuid.get(c["uuid"])
        if vec is None:
            continue
        rows.append(
            {
                "chunk_id": c["uuid"],
                "text_hash": c["text_hash"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "embedding": vec,
            }
        )
    payload = {
        "version": 1,
        "embedding_model": embedding_model,
        "ingest_max_chars": max_chars,
        "updated_at_epoch_s": int(time.time()),
        "chunks": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prepare_vectors_with_cache(chunks: List[dict], rag_cfg: RagConfig, on_event: Callable[[str, str], None]) -> Tuple[Dict[str, List[float]], int, int]:
    by_doc: Dict[str, List[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    vectors_by_uuid: Dict[str, List[float]] = {}
    cached = 0
    embedded = 0
    total = len(chunks)

    for doc_id, doc_chunks in by_doc.items():
        cache_path = _metadata_cache_path(doc_chunks[0]["source_path"])
        cached_map = _load_doc_cache(cache_path, rag_cfg.embedding_model, rag_cfg.ingest_max_chars)

        missing_texts: List[str] = []
        missing_ids: List[str] = []

        for c in doc_chunks:
            hit = cached_map.get(c["uuid"])
            if hit and hit[0] == c["text_hash"]:
                vectors_by_uuid[c["uuid"]] = hit[1]
                cached += 1
            else:
                missing_ids.append(c["uuid"])
                missing_texts.append(c["text"])

        for i in range(0, len(missing_texts), rag_cfg.ingest_batch_size):
            bt = missing_texts[i:i + rag_cfg.ingest_batch_size]
            bids = missing_ids[i:i + rag_cfg.ingest_batch_size]
            vecs = _embed_texts_openai(bt, rag_cfg.embedding_model)
            for cid, v in zip(bids, vecs):
                vectors_by_uuid[cid] = v
                embedded += 1
            done = cached + embedded
            on_event("rag_stage", f"Embedding progress {_progress_bar(done, total)} (cached {cached})")

        _write_doc_cache(cache_path, rag_cfg.embedding_model, rag_cfg.ingest_max_chars, doc_chunks, vectors_by_uuid)
        on_event("rag_stage", f"Cache updated for {doc_id} ({len(doc_chunks)} chunks)")

    return vectors_by_uuid, cached, embedded


def _ensure_collection_populated(client, doc_map: Dict[str, str], rag_cfg: RagConfig, on_event: Callable[[str, str], None]) -> int:
    wcfg = importlib.import_module("weaviate.classes.config")
    if not client.collections.exists(rag_cfg.collection_name):
        on_event("rag_stage", f"Collection '{rag_cfg.collection_name}' not found. Creating...")
        client.collections.create(
            name=rag_cfg.collection_name,
            vectorizer_config=wcfg.Configure.Vectorizer.none(),
            properties=[
                wcfg.Property(name="doc_id", data_type=wcfg.DataType.TEXT),
                wcfg.Property(name="text", data_type=wcfg.DataType.TEXT),
                wcfg.Property(name="page_start", data_type=wcfg.DataType.INT),
                wcfg.Property(name="page_end", data_type=wcfg.DataType.INT),
            ],
        )

    coll = client.collections.get(rag_cfg.collection_name)
    try:
        probe = coll.query.fetch_objects(limit=1, return_properties=[])
        if getattr(probe, "objects", None):
            on_event("rag_stage", f"Collection '{rag_cfg.collection_name}' already has vectors.")
            return 0
    except (RuntimeError, ValueError, AttributeError, TypeError, OSError):
        # If probe fails, continue with bootstrap attempt.
        pass

    if not rag_cfg.auto_bootstrap:
        raise RuntimeError(
            f"Collection '{rag_cfg.collection_name}' is empty and RAG_AUTO_BOOTSTRAP is disabled."
        )

    chunks = _make_chunks(doc_map, max_chars=rag_cfg.ingest_max_chars)
    if not chunks:
        on_event("rag_stage", "No text chunks found to ingest.")
        return 0

    on_event("rag_stage", f"Collection empty. Ingesting {len(chunks)} chunks from {len(doc_map)} docs...")
    vectors_by_uuid, cached_count, embedded_count = _prepare_vectors_with_cache(chunks, rag_cfg, on_event)
    on_event("rag_stage", f"Embeddings ready: {cached_count} cached, {embedded_count} newly embedded")
    inserted = 0
    bs = rag_cfg.ingest_batch_size
    for i in range(0, len(chunks), bs):
        batch = chunks[i:i + bs]
        for c in batch:
            vec = vectors_by_uuid.get(c["uuid"])
            if vec is None:
                continue
            coll.data.insert(
                uuid=c["uuid"],
                properties={
                    "doc_id": c["doc_id"],
                    "text": c["text"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                },
                vector=vec,
            )
            inserted += 1
        if inserted % max(100, bs) == 0 or inserted == len(chunks):
            on_event("rag_stage", f"Ingest progress {_progress_bar(inserted, len(chunks))}")
    on_event("rag_stage", f"Ingestion complete: {inserted} chunks inserted.")
    return inserted


def _query_weaviate(
    question: str,
    top_k: int,
    collection_name: str,
    embedding_model: str,
    doc_map: Dict[str, str],
    rag_cfg: RagConfig,
    on_event: Callable[[str, str], None],
) -> Tuple[List[dict], float]:
    wq = importlib.import_module("weaviate.classes.query")

    started = time.perf_counter()
    client = _connect_weaviate(rag_cfg.weaviate_mode)
    try:
        _ensure_collection_populated(client, doc_map=doc_map, rag_cfg=rag_cfg, on_event=on_event)
        coll = client.collections.get(collection_name)
        query_vec = _embed_query_openai(question, embedding_model)
        if rag_cfg.retrieval_mode == "hybrid":
            on_event("rag_stage", f"Retrieval mode: hybrid (alpha={rag_cfg.hybrid_alpha:.2f})")
            resp = coll.query.hybrid(
                query=question,
                vector=query_vec,
                alpha=rag_cfg.hybrid_alpha,
                limit=top_k,
                return_metadata=wq.MetadataQuery(distance=True),
            )
        else:
            on_event("rag_stage", "Retrieval mode: semantic (near_vector)")
        resp = coll.query.near_vector(
            near_vector=query_vec,
            limit=top_k,
            return_metadata=wq.MetadataQuery(distance=True),
        )
        out: List[dict] = []
        for obj in getattr(resp, "objects", []) or []:
            props = getattr(obj, "properties", {}) or {}
            md = getattr(obj, "metadata", None)
            out.append(
                {
                    "doc_id": str(props.get("doc_id", "")),
                    "text": str(props.get("text", "")),
                    "page_start": props.get("page_start"),
                    "page_end": props.get("page_end"),
                    "distance": getattr(md, "distance", None) if md else None,
                }
            )
        return out, (time.perf_counter() - started) * 1000.0
    finally:
        client.close()


def ensure_rag_ready(
    doc_map: Dict[str, str],
    on_event: Optional[Callable] = None,
) -> RagStats:
    """Ensure RAG collection exists and is populated before question-time retrieval."""
    _on_event = on_event or (lambda *a: None)
    rag_cfg = RagConfig.from_env()
    _on_event("rag_stage", f"Checking RAG collection '{rag_cfg.collection_name}' ({rag_cfg.weaviate_mode})")
    client = _connect_weaviate(rag_cfg.weaviate_mode)
    try:
        _ensure_collection_populated(client, doc_map=doc_map, rag_cfg=rag_cfg, on_event=_on_event)
        coll = client.collections.get(rag_cfg.collection_name)
        probe = coll.query.fetch_objects(limit=1, return_properties=[])
        has_any = bool(getattr(probe, "objects", None))
        _on_event(
            "rag_stage",
            f"RAG ready: collection '{rag_cfg.collection_name}' {'has vectors' if has_any else 'is empty'}",
        )
        return RagStats(
            total_docs=len(doc_map),
            retrieval_backend="weaviate",
            collection_name=rag_cfg.collection_name,
            retrieval_ms=0.0,
        )
    finally:
        client.close()


def get_rag_collection_status(
    on_event: Optional[Callable] = None,
) -> RagCollectionStatus:
    """Return current collection status (exists, chunks, unique docs)."""
    _on_event = on_event or (lambda *a: None)
    rag_cfg = RagConfig.from_env()
    client = _connect_weaviate(rag_cfg.weaviate_mode)
    try:
        exists = client.collections.exists(rag_cfg.collection_name)
        if not exists:
            _on_event("rag_stage", f"Collection '{rag_cfg.collection_name}' does not exist.")
            return RagCollectionStatus(
                collection_name=rag_cfg.collection_name,
                exists=False,
                chunk_objects=0,
                unique_docs=0,
            )
        coll = client.collections.get(rag_cfg.collection_name)
        chunk_objects, unique_docs = _count_collection(coll)
        _on_event("rag_stage", f"Collection '{rag_cfg.collection_name}': {chunk_objects} chunks across {unique_docs} docs")
        return RagCollectionStatus(
            collection_name=rag_cfg.collection_name,
            exists=True,
            chunk_objects=chunk_objects,
            unique_docs=unique_docs,
        )
    finally:
        client.close()


def reset_rag_collection(
    doc_map: Dict[str, str],
    on_event: Optional[Callable] = None,
) -> RagCollectionStatus:
    """Delete and recreate the collection, then bootstrap ingest using cache-aware embedding prep."""
    _on_event = on_event or (lambda *a: None)
    rag_cfg = RagConfig.from_env()
    client = _connect_weaviate(rag_cfg.weaviate_mode)
    try:
        if client.collections.exists(rag_cfg.collection_name):
            _on_event("rag_stage", f"Deleting collection '{rag_cfg.collection_name}'...")
            client.collections.delete(rag_cfg.collection_name)
        _on_event("rag_stage", f"Recreating collection '{rag_cfg.collection_name}'...")
        _ensure_collection_populated(client, doc_map=doc_map, rag_cfg=rag_cfg, on_event=_on_event)
        coll = client.collections.get(rag_cfg.collection_name)
        chunk_objects, unique_docs = _count_collection(coll)
        _on_event("rag_stage", f"Reset complete: {chunk_objects} chunks across {unique_docs} docs")
        return RagCollectionStatus(
            collection_name=rag_cfg.collection_name,
            exists=True,
            chunk_objects=chunk_objects,
            unique_docs=unique_docs,
        )
    finally:
        client.close()


def answer_rag(
    doc_map: Dict[str, str],
    question: str,
    cfg: Config,
    on_event: Optional[Callable] = None,
) -> Tuple[AnswerResult, RagStats]:
    """RAG baseline using Weaviate vector search + one synthesis call."""
    _on_event = on_event or (lambda *a: None)
    usage = TokenUsage()

    rag_cfg = RagConfig.from_env()
    _on_event("rag_stage", f"Embedding query with {rag_cfg.embedding_model}")
    _on_event(
        "rag_stage",
        f"Retrieving top {rag_cfg.top_k} chunks from Weaviate collection '{rag_cfg.collection_name}' ({rag_cfg.weaviate_mode})",
    )
    try:
        chunks, retrieval_ms = _query_weaviate(
            question,
            top_k=rag_cfg.top_k,
            collection_name=rag_cfg.collection_name,
            embedding_model=rag_cfg.embedding_model,
            doc_map=doc_map,
            rag_cfg=rag_cfg,
            on_event=_on_event,
        )
    except (ModuleNotFoundError, ImportError, RuntimeError, OSError, ValueError) as e:
        _on_event("rag_stage", f"RAG retrieval failed: {e}")
        fallback = AnswerResult(
            answer=(
                "RAG baseline could not run. "
                "Install `weaviate-client` in your environment and ensure Weaviate is running "
                f"with collection '{rag_cfg.collection_name}'."
            ),
            usage=usage,
            iterations=1,
            iteration_stats=[IterationStats(iteration=1, action="rag_error")],
            code_history=[f"# RAG error: {type(e).__name__}: {e}"],
        )
        return (
            fallback,
            RagStats(
                total_docs=len(doc_map),
                retrieval_backend="weaviate",
                collection_name=rag_cfg.collection_name,
                retrieval_ms=0.0,
            ),
        )

    if not chunks:
        _on_event("rag_stage", "No chunks returned from Weaviate")
        fallback = AnswerResult(
            answer="I couldn't retrieve any RAG chunks from Weaviate. Run ingestion first, then retry.",
            usage=usage,
            iterations=1,
            iteration_stats=[IterationStats(iteration=1, action="rag")],
            code_history=["# No chunks returned from Weaviate"],
        )
        return fallback, RagStats(total_docs=len(doc_map), retrieval_backend="weaviate", collection_name=rag_cfg.collection_name, retrieval_ms=retrieval_ms)

    context_blocks = []
    cited_docs = set()
    for i, c in enumerate(chunks, start=1):
        doc_id = c.get("doc_id", "unknown")
        cited_docs.add(doc_id)
        page_start = c.get("page_start")
        page_end = c.get("page_end")
        page_label = f"p{page_start}" if page_start == page_end else f"p{page_start}-{page_end}"
        context_blocks.append(
            f"[{i}] doc={doc_id} {page_label}\n{c.get('text', '')[:2400]}"
        )
    rag_context = "\n\n".join(context_blocks)

    prompt = _render_prompt(
        RAG_PROMPT_PATH,
        QUESTION=question,
        TOP_K=str(rag_cfg.top_k),
        RETRIEVED_CONTEXT=rag_context,
    )

    _on_event("rag_stage", f"Calling LLM with {len(chunks)} retrieved chunks")
    llm_t0 = time.perf_counter()
    answer = call_llm(prompt, cfg, usage_accum=usage)
    llm_ms = (time.perf_counter() - llm_t0) * 1000.0
    _on_event("rag_stage", "RAG answer complete")

    result = AnswerResult(
        answer=answer,
        usage=usage,
        iterations=1,
        iteration_stats=[
            IterationStats(
                iteration=1,
                action="rag",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                planner_llm_ms=llm_ms,
                iteration_ms=llm_ms + retrieval_ms,
            )
        ],
        code_history=[f"# Retrieved {len(chunks)} chunks from Weaviate collection {rag_cfg.collection_name}"],
    )

    stats = RagStats(
        total_docs=len(doc_map),
        retrieved_chunks=len(chunks),
        unique_docs_retrieved=len(cited_docs),
        retrieval_backend="weaviate",
        collection_name=rag_cfg.collection_name,
        retrieval_ms=retrieval_ms,
    )
    return result, stats
