from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


ProgressCallback = Optional[Callable[[str, str], None]]


def _sanitize(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _doc_extensions_from_env() -> set[str]:
    raw = os.getenv(
        "RLM_DOC_EXTENSIONS",
        "pdf,md,markdown,txt,json,csv,tsv,yaml,yml,xml,html,log",
    )
    exts = {x.strip().lower().lstrip(".") for x in raw.split(",") if x.strip()}
    return exts or {
        "pdf",
        "md",
        "markdown",
        "txt",
        "json",
        "csv",
        "tsv",
        "yaml",
        "yml",
        "xml",
        "html",
        "log",
    }


def _doc_ignore_dirs_from_env() -> set[str]:
    raw = os.getenv(
        "RLM_DOC_IGNORE_DIRS",
        ".git,.obsidian,node_modules,venv,.venv,dist,build,__pycache__",
    )
    return {x.strip() for x in raw.split(",") if x.strip()}


def _should_ignore(path: Path, data_root: Path, ignore_dirs: set[str]) -> bool:
    try:
        rel = path.relative_to(data_root)
        parts = rel.parts[:-1] if path.is_file() else rel.parts
    except Exception:
        parts = path.parts
    for part in parts:
        if part in ignore_dirs:
            return True
        if part.startswith("."):
            return True
    return False


def default_processed_dir(data_dir: Path) -> Path:
    """Keep PDF extracted pages local to the active corpus folder by default."""
    return data_dir / "processed_data"


def resolve_processed_dir(data_dir: Path | None = None, processed_dir: Path | None = None) -> Path:
    src = data_dir or Path(os.getenv("RLM_DATA_DIR", "data"))
    if processed_dir is not None:
        return processed_dir
    configured = os.getenv("RLM_PROCESSED_DIR", "").strip()
    if configured:
        return Path(configured)
    return default_processed_dir(src)


def doc_id_from_pdf(pdf: Path, data_root: Path) -> str:
    rel = pdf.relative_to(data_root).with_suffix("")
    return _sanitize(str(rel).replace("/", "_"))


def doc_id_from_source(path: Path, data_root: Path) -> str:
    rel = path.relative_to(data_root).with_suffix("")
    return _sanitize(str(rel).replace("/", "_"))


def list_pdf_files(data_dir: Path | None = None) -> List[Path]:
    root = data_dir or Path(os.getenv("RLM_DATA_DIR", "data"))
    if not root.is_dir():
        return []
    if "pdf" not in _doc_extensions_from_env():
        return []
    ignore_dirs = _doc_ignore_dirs_from_env()
    return sorted(
        p for p in root.rglob("*.pdf") if p.is_file() and not _should_ignore(p, root, ignore_dirs)
    )


def list_markdown_files(data_dir: Path | None = None) -> List[Path]:
    root = data_dir or Path(os.getenv("RLM_DATA_DIR", "data"))
    if not root.is_dir():
        return []
    exts = _doc_extensions_from_env()
    ignore_dirs = _doc_ignore_dirs_from_env()
    out: List[Path] = []
    if "md" in exts:
        out.extend(p for p in root.rglob("*.md") if p.is_file() and not _should_ignore(p, root, ignore_dirs))
    if "markdown" in exts:
        out.extend(
            p for p in root.rglob("*.markdown") if p.is_file() and not _should_ignore(p, root, ignore_dirs)
        )
    return sorted(out)


def list_text_like_files(data_dir: Path | None = None) -> List[Path]:
    root = data_dir or Path(os.getenv("RLM_DATA_DIR", "data"))
    if not root.is_dir():
        return []
    exts = _doc_extensions_from_env() - {"pdf", "md", "markdown"}
    if not exts:
        return []
    ignore_dirs = _doc_ignore_dirs_from_env()
    out: List[Path] = []
    for ext in sorted(exts):
        out.extend(
            p
            for p in root.rglob(f"*.{ext}")
            if p.is_file() and not _should_ignore(p, root, ignore_dirs)
        )
    # Deduplicate and keep stable sorted output.
    return sorted(set(out))


def missing_processed_pdfs(
    pdfs: List[Path],
    data_root: Path,
    processed_root: Path,
) -> List[Path]:
    missing: List[Path] = []
    for pdf in pdfs:
        doc_dir = processed_root / doc_id_from_pdf(pdf, data_root)
        if not (doc_dir.exists() and any(doc_dir.glob("page_*/text.txt"))):
            missing.append(pdf)
    return missing


def preprocess_pdfs(
    pdfs: Optional[List[Path]] = None,
    data_dir: Path | None = None,
    processed_dir: Path | None = None,
    on_progress: ProgressCallback = None,
) -> Tuple[int, int]:
    """Extract text pages for PDFs into a processed directory.

    Returns: (processed_count, skipped_count)
    Progress callback events:
      - start/detail, file_start/detail, file_done/detail, file_skip/detail, file_error/detail, done/detail
    """
    from pypdf import PdfReader

    src = data_dir or Path(os.getenv("RLM_DATA_DIR", "data"))
    dst = resolve_processed_dir(src, processed_dir)
    dst.mkdir(parents=True, exist_ok=True)

    to_process = sorted(pdfs) if pdfs is not None else list_pdf_files(src)
    if on_progress:
        on_progress("start", f"{len(to_process)} PDFs")

    processed = 0
    skipped = 0
    for pdf in to_process:
        rel = pdf.relative_to(src)
        doc_dir = dst / doc_id_from_pdf(pdf, src)
        if doc_dir.exists() and any(doc_dir.glob("page_*/text.txt")):
            skipped += 1
            if on_progress:
                on_progress("file_skip", f"{rel}")
            continue

        if on_progress:
            on_progress("file_start", f"{rel}")
        try:
            reader = PdfReader(pdf)
            for i, page in enumerate(reader.pages):
                pd = doc_dir / f"page_{i + 1}"
                pd.mkdir(parents=True, exist_ok=True)
                (pd / "text.txt").write_text(page.extract_text() or "", encoding="utf-8")
            processed += 1
            if on_progress:
                on_progress("file_done", f"{rel} ({len(reader.pages)} pages)")
        except Exception as e:
            skipped += 1
            if on_progress:
                on_progress("file_error", f"{rel} -> {e}")

    if on_progress:
        on_progress("done", f"{processed} processed, {skipped} skipped")
    return processed, skipped


def discover_docs(
    data_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> Dict[str, str]:
    """Discover docs. Source mode:
    - auto (default): text-like files are loaded from data/, PDFs prefer processed pages
    - prefer_processed: same as auto, but strictly prefer processed PDFs when present
    - data_only: only raw files in data/
    - processed_only: only processed_data
    """
    src = data_dir or Path(os.getenv("RLM_DATA_DIR", "data"))
    dst = resolve_processed_dir(src, processed_dir)
    source_mode = os.getenv("RLM_DOC_SOURCE_MODE", "auto").strip().lower()

    docs: Dict[str, str] = {}
    if source_mode in ("auto", "prefer_processed", "data_only") and src.is_dir():
        # Non-PDF text-like sources always come straight from the corpus folder.
        for md in list_markdown_files(src):
            did = doc_id_from_source(md, src)
            if did not in docs:
                docs[did] = str(md)
        for fp in list_text_like_files(src):
            did = doc_id_from_source(fp, src)
            if did not in docs:
                docs[did] = str(fp)

        # PDFs can use processed pages when available.
        for pdf in list_pdf_files(src):
            did = doc_id_from_pdf(pdf, src)
            processed_doc = dst / did
            has_processed = processed_doc.is_dir() and any(processed_doc.glob("page_*/text.txt"))
            if source_mode == "data_only":
                docs[did] = str(pdf)
            elif has_processed:
                docs[did] = str(processed_doc)
            else:
                docs[did] = str(pdf)

    if source_mode == "processed_only" and dst.is_dir():
        for d in sorted(dst.iterdir()):
            if d.is_dir() and any(d.glob("page_*/text.txt")):
                docs[_sanitize(d.name)] = str(d)

    return docs
