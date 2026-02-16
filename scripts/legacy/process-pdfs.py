import argparse
import json
import os
import time
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm # For markdown extraction
import requests
from tqdm import tqdm

# yes
# Imported sanitize_name and corrected function name
from utils import (
    embed_chunk_image,
    embed_chunk_text,
    sentence_transformers_embed,
    sanitize_name,
    openai_embed,
)
import concurrent.futures
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Model name configuration
IMG_MODEL_NAME = os.getenv("IMG_LATE_INTERACTION_MODEL", "colpali-1.3")
TEXT_MODEL_NAME = os.getenv("TEXT_LATE_INTERACTION_MODEL", "colpali-1.3")
EMBED_SENTENCE_TRANSFORMERS_URL = os.getenv(
    "EMBED_SENTANCE_TRANSFORMERS_URL",  # keep env var name for compatibility
    "http://jarvita-agx:8123/embed",
)


def fetch_st_model_id(base_url: str) -> Optional[str]:
    """Fetch the model ID from the /info endpoint."""
    if base_url.endswith("/embed"):
        info_url = base_url[:-6] + "/info"
    else:
        info_url = base_url.replace("/embed", "/info")

    try:
        resp = requests.get(info_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("model_id")
    except Exception as e:
        print(f"⚠️ Could not fetch model ID from {info_url}: {e}")
        return None

def embed_chunk_parallel(
    png_path: str, 
    text: str,
    *,
    do_image: bool = True,
    do_text: bool = True,
    do_sentence_transformers: bool = True,
    do_openai: bool = False,
) -> Tuple[Optional[list], Optional[list], Optional[list], Optional[list]]:
    """
    Run requested embedding calls in parallel using threads.
    Returns: (img_vector, text_late_interaction_vector, sentence_transformer_vector, openai_vector)
    """
    img_vector: Optional[list] = None
    text_li_vector: Optional[list] = None
    st_vector: Optional[list] = None
    openai_vector: Optional[list] = None
    
    futures = {}
    
    # Calculate max_workers based on how many embeddings we're actually doing
    num_tasks = sum([do_image, do_text, do_sentence_transformers, do_openai])
    
    if num_tasks == 0:
        return None, None, None, None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
        # Submit only the requested embedding tasks
        if do_image:
            futures["img"] = executor.submit(embed_chunk_image, png_path)
        if do_text:
            futures["text_li"] = executor.submit(embed_chunk_text, text)
        if do_sentence_transformers:
            futures["st"] = executor.submit(sentence_transformers_embed, text)
        if do_openai:
            futures["openai"] = executor.submit(openai_embed, text)

        # Wait for all to complete
        for name, future in futures.items():
            try:
                result = future.result()
            except Exception as e:
                print(f"Error in embedding {name}: {e}")
                result = None
            
            if name == "img":
                img_vector = result
            elif name == "text_li":
                text_li_vector = result
            elif name == "st":
                st_vector = result
            elif name == "openai":
                openai_vector = result

    return img_vector, text_li_vector, st_vector, openai_vector

def process_pdf(
    pdf_path: str, 
    output_dir: str = None,
    *,
    do_image: bool = True,
    do_text: bool = True,
    do_sentence_transformers: bool = True,
    do_openai: bool = False,
    st_model_name: str = "qwen-06b",
) -> None:
    pdf_path = Path(pdf_path)
    safe_name = sanitize_name(pdf_path.name)
    safe_stem = sanitize_name(pdf_path.stem)

    # Use output_dir if provided, otherwise default to source parent
    if output_dir:
        output_root = Path(output_dir) / safe_stem
    else:
        output_root = pdf_path.parent / safe_stem
        
    output_root.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    num_pages = doc.page_count

    print(f"Processing PDF: {pdf_path.name} ({num_pages} pages)")
    print(f"Saving into folder: {output_root}")
    
    embeddings_enabled = []
    if do_image:
        embeddings_enabled.append("image")
    if do_text:
        embeddings_enabled.append("text")
    if do_sentence_transformers:
        embeddings_enabled.append("sentence_transformers")
    if do_openai:
        embeddings_enabled.append("openai")
    print(f"Embeddings enabled: {', '.join(embeddings_enabled) if embeddings_enabled else 'none'}")

    for i in tqdm(range(num_pages), desc=f"Processing {safe_stem}"):
        page_number = i + 1
        page = doc[i]

        page_dir = output_root / f"page_{page_number}"
        page_dir.mkdir(exist_ok=True)

        # --- Save page PDF ---
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        pdf_output_path = page_dir / f"page_{page_number}.pdf"
        new_doc.save(pdf_output_path)
        new_doc.close()

        # --- Save page PNG ---
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        png_output_path = page_dir / f"page_{page_number}.png"
        pix.save(png_output_path)

        # --- Extract text ---
        text = page.get_text("text") or ""
        try:
            # Use pymupdf4llm for high-quality markdown
            # to_markdown returns text for specific pages (0-indexed)
            text_markdown = pymupdf4llm.to_markdown(doc, pages=[i]) or ""
        except Exception as e:
            print(f"⚠️ Markdown extraction failed for page {page_number}: {e}")
            text_markdown = text # Fallback to plain text
        
        text_output_path = page_dir / "text.txt"
        text_output_path.write_text(text, encoding="utf-8")

        # --- Call embedding API (only for requested types) ---
        vector, late_interaction_txt_vector, sentance_transformers_embeddings, openai_embeddings = embed_chunk_parallel(
            str(png_output_path), 
            text,
            do_image=do_image,
            do_text=do_text,
            do_sentence_transformers=do_sentence_transformers,
            do_openai=do_openai,
        )

        # --- Metadata (base fields) ---
        metadata_path = page_dir / "metadata.json"
        
        # 1. Try to load existing metadata
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        else:
            metadata = {}

        # 2. Update base fields (always safe to refresh these)
        metadata.update({
            "original_pdf_name": safe_stem,
            "original_pdf_path": safe_name,
            "pdf_name": f"{safe_stem}_{page_number}",
            "page_number": page_number,
            "total_pages": num_pages,
            "chunk_png_file": f"{safe_stem}/page_{page_number}/page_{page_number}.png",
            "chunk_pdf_file": f"{safe_stem}/page_{page_number}/page_{page_number}.pdf",
            "chunk_text_file": f"{safe_stem}/page_{page_number}/text.txt",
            "content": text,
            "content_markdown": text_markdown,
            "text_length": len(text),
            "timestamp": int(time.time()),
            "width": page.rect.width,
            "height": page.rect.height,
        })
        
        # 3. Add ONLY the requested embeddings (preserving others)
        if do_image and vector is not None:
            metadata["img_late_interaction_vector"] = vector
            metadata["img_late_interaction_model"] = IMG_MODEL_NAME
            
        if do_text and late_interaction_txt_vector is not None:
            metadata["text_late_interaction_vector"] = late_interaction_txt_vector
            metadata["text_late_interaction_model"] = TEXT_MODEL_NAME
            
        if do_sentence_transformers and sentance_transformers_embeddings is not None:
            metadata["sentence_transformers_embeddings"] = sentance_transformers_embeddings
            metadata["sentence_transformers_model"] = st_model_name
        
        if do_openai and openai_embeddings is not None:
            metadata["openai_embeddings"] = openai_embeddings
            metadata["openai_model"] = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

        # 4. Save merged metadata
        metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    doc.close()
    print(f"Done processing {safe_stem}!\n")

def process_all_pdfs_in_directory(
    directory_path: str, 
    output_dir: str = None,
    *,
    do_image: bool = True,
    do_text: bool = True,
    do_sentence_transformers: bool = True,
    do_openai: bool = False,
    st_model_name: str = "qwen-06b",
):
    root = Path(directory_path)   
    pdf_files = list(root.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {directory_path}...")

    # Ensure output directory exists if provided
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_files:
        try:
            print(f"--- Starting {pdf_file.name} ---")
            process_pdf(
                str(pdf_file), 
                output_dir=output_dir,
                do_image=do_image,
                do_text=do_text,
                do_sentence_transformers=do_sentence_transformers,
                do_openai=do_openai,
                st_model_name=st_model_name,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Failed to process {pdf_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process PDF files and attach embeddings.")
    parser.add_argument("--input-dir", type=str, default="./data", help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, default="./processed_data", help="Output directory for processed data")
    
    parser.add_argument(
        "--embeddings",
        nargs="+",
        choices=["image", "text", "st", "openai"],
        default=["image", "text", "st"],
        help="Specific embeddings to generate. Default: all.",
    )
    parser.add_argument("--metadata-only", action="store_true", help="Skip all embeddings and only generate file artifacts and base metadata.")

    parser.add_argument(
        "--st-model-name",
        type=str,
        default=None,
        help="Sentence-transformers model name. If None, fetch from /info.",
    )
    
    args = parser.parse_args()

    # Determine which embeddings to run
    if args.metadata_only:
        active_embeddings = set()
    else:
        active_embeddings = set(args.embeddings)

    do_image = "image" in active_embeddings
    do_text = "text" in active_embeddings
    do_st = "st" in active_embeddings
    do_openai = "openai" in active_embeddings

    st_model_name = args.st_model_name
    # Only fetch if we are actually running ST and didn't provide a name
    if do_st and st_model_name is None:
        fetched_id = fetch_st_model_id(EMBED_SENTENCE_TRANSFORMERS_URL)
        st_model_name = fetched_id if fetched_id else "qwen-06b"
    elif st_model_name is None:
        st_model_name = "qwen-06b"
    
    process_all_pdfs_in_directory(
        args.input_dir, 
        output_dir=args.output_dir,
        do_image=do_image,
        do_text=do_text,
        do_sentence_transformers=do_st,
        do_openai=do_openai,
        st_model_name=st_model_name,
    )

if __name__ == "__main__":
    main()


