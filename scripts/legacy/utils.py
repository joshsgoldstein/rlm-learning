import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import unicodedata
from pathlib import Path
import os
from IPython.display import display, HTML, Image
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Use environment variables for URLs to make the code portable
EMBED_IMAGE_URL = os.getenv("EMBED_IMAGE_URL", "http://jarvita-agx:8124/embed")
EMBED_TEXT_URL = os.getenv("EMBED_TEXT_URL", "http://jarvita-agx:8124/embed")
# Note: Keeping the Env Var name as is to avoid breaking external config, but fixing variable name
EMBED_SENTENCE_TRANSFORMERS_URL = os.getenv("EMBED_SENTANCE_TRANSFORMERS_URL", "http://jarvita-agx:8123/embed")



# Cloud Based LLMS


OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

def openai_embed(text, model: str = None):
    """
    Create an OpenAI embedding for a string (or list of strings).
    Returns a single vector for a string, or a list of vectors for a list.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=model or OPENAI_EMBED_MODEL,
        input=text,
    )
    if isinstance(text, list):
        return [item.embedding for item in response.data]
    return response.data[0].embedding




# Jetson Orin API's

def get_session():
    session = requests.Session()
    retries = Retry(
        total=3,  # Increased to 3 for better reliability
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session

# Initialize a global session to reuse connections
_session = get_session()

def display_results_three_per_row(results, base_path=None):
    """
    Display a list of result objects with exactly 3 images per row, arranged horizontally.
    
    Args:
        results: List of result objects (e.g., response.objects from Weaviate query)
        base_path: Optional base path to prepend to image paths if they are relative.
                   Defaults to "processed_data" if not provided.
    """
    # Default to processed_data if no path provided, preserving original behavior
    if base_path is None:
        base_path = "processed_data"
    
    print(f"\n{'='*60}")
    print(f"📄 SEARCH RESULTS")
    print(f"📊 Number of results: {len(results)}")
    print(f"{'='*60}\n")
    
    # Display images in groups of 3
    for i in range(0, len(results), 3):
        row_results = results[i:i + 3]
        
        # Create HTML to display images horizontally
        html_content = '<div style="display: flex; gap: 10px; margin: 10px 0;">'
        
        for obj in row_results:
            # Get image path from properties
            image_path = obj.properties.get("chunk_png_file") or obj.properties.get("image_path")
            
            if image_path:
                # Construct full path if relative
                if not os.path.isabs(image_path):
                    full_image_path = os.path.join(base_path, image_path)
                else:
                    full_image_path = image_path
                
                # Check if file exists
                if os.path.exists(full_image_path):
                    page_num = obj.properties.get('page_number', 'N/A')
                    pdf_name = obj.properties.get('original_pdf_name', 'N/A')
                    
                    # Display metadata if available
                    score_info = ""
                    if hasattr(obj, 'metadata'):
                        if hasattr(obj.metadata, 'distance') and obj.metadata.distance is not None:
                            score_info = f"<div style='font-size: 10px; color: #666;'>Distance: {obj.metadata.distance:.4f}</div>"
                        elif hasattr(obj.metadata, 'score') and obj.metadata.score is not None:
                            score_info = f"<div style='font-size: 10px; color: #666;'>Score: {obj.metadata.score:.4f}</div>"
                    
                    html_content += f'''
                    <div style="text-align: center; border: 1px solid #ddd; border-radius: 4px; padding: 5px; background: #f9f9f9;">
                        <div style="font-size: 11px; margin-bottom: 3px; font-weight: bold;">Page {page_num}</div>
                        <div style="font-size: 9px; margin-bottom: 3px; color: #666;">{pdf_name}</div>
                        {score_info}
                        <!-- Use full_image_path for display to ensure it links correctly -->
                        <img src="{full_image_path}" width="250" style="border: 1px solid #ccc; border-radius: 4px; margin-top: 5px;">
                    </div>
                    '''
                else:
                    html_content += f'''
                    <div style="text-align: center; border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: #fff3cd;">
                        <div style="color: #856404;">⚠️ Image not found</div>
                        <div style="font-size: 10px; color: #666;">{full_image_path}</div>
                    </div>
                    '''
            else:
                page_num = obj.properties.get('page_number', 'N/A')
                html_content += f'''
                <div style="text-align: center; border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: #f8d7da;">
                    <div style="color: #721c24;">❌ No image path</div>
                    <div style="font-size: 10px; color: #666;">Page {page_num}</div>
                </div>
                '''
        
        html_content += '</div>'
        display(HTML(html_content))
        
        # Add spacing between rows
        if i + 3 < len(results):
            print("\n" + "-"*40 + "\n")

def sanitize_name(name: str) -> str:
    """
    Normalize & sanitize names: drop accents, keep ASCII letters/numbers/._-
    """
    norm = unicodedata.normalize("NFKD", name)
    ascii_only = norm.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", ascii_only)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "file"

def embed_chunk_image(image_path: str):
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/png")}
            response = _session.post(EMBED_IMAGE_URL, files=files, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Error embedding image {image_path}: {e}")
        return None

def embed_chunk_text(text: str):
    try:
        response = _session.post(EMBED_TEXT_URL, data={"text": text}, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

def sentence_transformers_embed(text: str):
    try:
        response = _session.post(EMBED_SENTENCE_TRANSFORMERS_URL, data={"text": text}, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None


