from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


ProgressCallback = Optional[Callable[[str, str], None]]


def _sanitize(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def doc_id_from_pdf(pdf: Path, data_root: Path) -> str:
    rel = pdf.relative_to(data_root).with_suffix("")
    return _sanitize(str(rel).replace("/", "_"))


def list_pdf_files(data_dir: Path | None = None) -> List[Path]:
    root = data_dir or Path(os.getenv("RLM_DATA_DIR", "data"))
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.pdf"))


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
    """Extract text pages for PDFs into processed_data.

    Returns: (processed_count, skipped_count)
    Progress callback events:
      - start/detail, file_start/detail, file_done/detail, file_skip/detail, file_error/detail, done/detail
    """
    from pypdf import PdfReader

    src = data_dir or Path(os.getenv("RLM_DATA_DIR", "data"))
    dst = processed_dir or Path(os.getenv("RLM_PROCESSED_DIR", "processed_data"))
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
    """Discover docs preferring processed_data dirs, fallback to raw PDFs."""
    src = data_dir or Path(os.getenv("RLM_DATA_DIR", "data"))
    dst = processed_dir or Path(os.getenv("RLM_PROCESSED_DIR", "processed_data"))

    docs: Dict[str, str] = {}
    if dst.is_dir():
        for d in sorted(dst.iterdir()):
            if d.is_dir() and any(d.glob("page_*/text.txt")):
                docs[_sanitize(d.name)] = str(d)

    if src.is_dir():
        for pdf in sorted(src.rglob("*.pdf")):
            did = doc_id_from_pdf(pdf, src)
            if did not in docs:
                docs[did] = str(pdf)
    return docs
