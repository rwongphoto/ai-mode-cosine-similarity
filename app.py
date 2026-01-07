import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import openai
from openai import OpenAI
import re
import ast
import time
import random
import base64
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from bs4 import BeautifulSoup
import textwrap
from selenium_stealth import stealth
from huggingface_hub import login as hf_login

st.set_page_config(layout="wide", page_title="AI Mode Query Fan-Out Analyzer")

# --- Session State Initialization ---
if "all_url_metrics_list" not in st.session_state: st.session_state.all_url_metrics_list = None
if "url_processed_units_dict" not in st.session_state: st.session_state.url_processed_units_dict = None
if "all_queries_for_analysis" not in st.session_state: st.session_state.all_queries_for_analysis = None
if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
if "selenium_driver_instance" not in st.session_state: st.session_state.selenium_driver_instance = None
if "gemini_api_key_to_persist" not in st.session_state: st.session_state.gemini_api_key_to_persist = ""
if "gemini_api_configured" not in st.session_state: st.session_state.gemini_api_configured = False
if "openai_api_key_to_persist" not in st.session_state: st.session_state.openai_api_key_to_persist = ""
if "openai_api_configured" not in st.session_state: st.session_state.openai_api_configured = False
if "openai_client" not in st.session_state: st.session_state.openai_client = None
if "selected_embedding_model" not in st.session_state: st.session_state.selected_embedding_model = 'all-mpnet-base-v2'
if "processing" not in st.session_state: st.session_state.processing = False
if "zyte_api_key_to_persist" not in st.session_state: st.session_state.zyte_api_key_to_persist = ""
if "zyte_api_configured" not in st.session_state: st.session_state.zyte_api_configured = False
if "huggingface_api_key_to_persist" not in st.session_state: st.session_state.huggingface_api_key_to_persist = ""
if "huggingface_api_configured" not in st.session_state: st.session_state.huggingface_api_configured = False

# Check for environment variable API keys (for deployment environments like Posit Connect)
env_hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or os.getenv('hf_login')
if env_hf_token and not st.session_state.huggingface_api_configured:
    try:
        hf_login(token=env_hf_token, add_to_git_credential=False)
        st.session_state.huggingface_api_key_to_persist = env_hf_token
        st.session_state.huggingface_api_configured = True
    except Exception:
        pass

REQUEST_INTERVAL = 3.0
last_request_time = 0

# User Agents List
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36"
]

def get_random_user_agent(): 
    return random.choice(USER_AGENTS)

def enforce_rate_limit():
    global last_request_time
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < REQUEST_INTERVAL: 
        time.sleep(REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

# --- Sidebar API Configuration ---
st.sidebar.header("🔑 API Configuration")

with st.sidebar.expander("OpenAI API", expanded=not st.session_state.get("openai_api_configured", False)):
    openai_api_key_input = st.text_input("Enter OpenAI API Key:", type="password", value=st.session_state.get("openai_api_key_to_persist", ""), disabled=st.session_state.processing)
    if st.button("Set & Verify OpenAI Key", disabled=st.session_state.processing):
        if openai_api_key_input:
            try:
                test_client = OpenAI(api_key=openai_api_key_input)
                test_client.embeddings.create(input=["test"], model="text-embedding-3-small")
                st.session_state.openai_api_key_to_persist = openai_api_key_input
                st.session_state.openai_api_configured = True
                st.session_state.openai_client = test_client
                st.success("OpenAI API Key Configured!")
                st.rerun()
            except Exception as e:
                st.session_state.openai_api_key_to_persist = ""
                st.session_state.openai_api_configured = False
                st.session_state.openai_client = None
                st.error(f"OpenAI Key Failed: {str(e)[:200]}")
        else: 
            st.warning("Please enter OpenAI API Key.")

with st.sidebar.expander("Gemini API", expanded=not st.session_state.get("gemini_api_configured", False)):
    gemini_api_key_input = st.text_input("Enter Google Gemini API Key:", type="password", value=st.session_state.get("gemini_api_key_to_persist", ""), disabled=st.session_state.processing)
    if st.button("Set & Verify Gemini Key", disabled=st.session_state.processing):
        if gemini_api_key_input:
            try:
                genai.configure(api_key=gemini_api_key_input)
                if not any('generateContent' in m.supported_generation_methods for m in genai.list_models()): 
                    raise Exception("No usable models found for this API key.")
                st.session_state.gemini_api_key_to_persist = gemini_api_key_input
                st.session_state.gemini_api_configured = True
                st.success("Gemini API Key Configured!")
                st.rerun()
            except Exception as e:
                st.session_state.gemini_api_key_to_persist = ""
                st.session_state.gemini_api_configured = False
                st.error(f"API Key Failed: {str(e)[:200]}")
        else: 
            st.warning("Please enter API Key.")

with st.sidebar.expander("Hugging Face API (for Gemma models)", expanded=not st.session_state.get("huggingface_api_configured", False)):
    # Check if configured via environment variable
    env_hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN') or os.getenv('hf_login')
    if env_hf_token and st.session_state.huggingface_api_configured:
        st.info("🔐 Hugging Face API configured via environment variable")
        st.markdown("*Using key from environment*")
    else:
        hf_api_key_input = st.text_input("Enter Hugging Face API Key:", type="password", value=st.session_state.get("huggingface_api_key_to_persist", "") if not env_hf_token else "", disabled=st.session_state.processing)
        if st.button("Set & Verify Hugging Face Key", disabled=st.session_state.processing):
            if hf_api_key_input:
                try:
                    hf_login(token=hf_api_key_input, add_to_git_credential=False)
                    st.session_state.huggingface_api_key_to_persist = hf_api_key_input
                    st.session_state.huggingface_api_configured = True
                    st.success("Hugging Face API Key Configured!")
                    st.rerun()
                except Exception as e:
                    st.session_state.huggingface_api_key_to_persist = ""
                    st.session_state.huggingface_api_configured = False
                    st.error(f"Hugging Face Key Failed: {str(e)[:200]}")
            else:
                st.warning("Please enter Hugging Face API Key.")

with st.sidebar.expander("Zyte API (Web Scraping)", expanded=not st.session_state.get("zyte_api_configured", False)):
    zyte_api_key_input = st.text_input("Enter Zyte API Key:", type="password", value=st.session_state.get("zyte_api_key_to_persist", ""), disabled=st.session_state.processing)
    if st.button("Set & Verify Zyte Key", disabled=st.session_state.processing):
        if zyte_api_key_input:
            try:
                response = requests.post(
                    "https://api.zyte.com/v1/extract",
                    auth=(zyte_api_key_input, ''),
                    json={'url': 'https://toscrape.com/', 'httpResponseBody': True},
                    timeout=20
                )
                if response.status_code == 200:
                    st.session_state.zyte_api_key_to_persist = zyte_api_key_input
                    st.session_state.zyte_api_configured = True
                    st.success("Zyte API Key Configured!")
                    st.rerun()
                else:
                    st.session_state.zyte_api_configured = False
                    st.error(f"Zyte Key Failed. Status: {response.status_code}")
            except Exception as e:
                st.session_state.zyte_api_configured = False
                st.error(f"Zyte API Request Failed: {str(e)[:200]}")
        else:
            st.warning("Please enter Zyte API Key.")

st.sidebar.markdown("---")
if st.session_state.get("openai_api_configured"): 
    st.sidebar.markdown("✅ OpenAI API: **Configured**")
else: 
    st.sidebar.markdown("⚠️ OpenAI API: **Not Configured**")

if st.session_state.get("gemini_api_configured"): 
    st.sidebar.markdown("✅ Gemini API: **Configured**")
else: 
    st.sidebar.markdown("⚠️ Gemini API: **Not Configured**")

st.sidebar.markdown(f"✅ Hugging Face API: **{'Configured' if st.session_state.huggingface_api_configured else 'Not Configured'}**")
st.sidebar.markdown(f"✅ Zyte API: **{'Configured' if st.session_state.zyte_api_configured else 'Not Configured'}**")

# Initialize clients if keys are available
if st.session_state.get("openai_api_key_to_persist") and not st.session_state.get("openai_client"):
    st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key_to_persist)

if st.session_state.get("gemini_api_key_to_persist"):
    try: 
        genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
    except Exception: 
        st.session_state.gemini_api_configured = False

# Initialize Hugging Face authentication if key is available
if st.session_state.get("huggingface_api_key_to_persist") and st.session_state.get("huggingface_api_configured"):
    try:
        hf_login(token=st.session_state.huggingface_api_key_to_persist, add_to_git_credential=False)
    except Exception:
        st.session_state.huggingface_api_configured = False

# --- Embedding Functions ---
@st.cache_resource
def load_local_sentence_transformer_model(model_name):
    try:
        # Use authentication token if available and model requires it
        if st.session_state.get("huggingface_api_configured") and st.session_state.get("huggingface_api_key_to_persist"):
            # Re-authenticate in case it was reset
            hf_login(token=st.session_state.huggingface_api_key_to_persist, add_to_git_credential=False)
        
        return SentenceTransformer(model_name, use_auth_token=st.session_state.get("huggingface_api_key_to_persist", None))
    except Exception as e: 
        st.error(f"Failed to load local model '{model_name}': {e}")
        if "gated" in str(e).lower() or "authentication" in str(e).lower():
            st.warning("This model may be gated. Please ensure your Hugging Face API key has access to this model.")
        return None

def get_openai_embeddings(texts: list, client: OpenAI, model: str):
    if not texts or not client: 
        return np.array([])
    try:
        texts = [text.replace("\n", " ") for text in texts]
        response = client.embeddings.create(input=texts, model=model)
        return np.array([item.embedding for item in response.data])
    except Exception as e: 
        st.error(f"OpenAI embedding failed: {e}")
        return np.array([])

def get_gemini_embeddings(texts: list, model: str):
    if not texts: 
        return np.array([])
    try:
        result = genai.embed_content(model=model, content=texts, task_type="RETRIEVAL_DOCUMENT")
        return np.array(result['embedding'])
    except Exception as e: 
        st.error(f"Gemini embedding failed: {e}")
        return np.array([])

def get_embeddings(texts, local_model_instance=None):
    model_choice = st.session_state.selected_embedding_model
    if model_choice.startswith("openai-"): 
        return get_openai_embeddings(texts, client=st.session_state.openai_client, model=model_choice.replace("openai-", ""))
    elif model_choice.startswith("gemini-"): 
        return get_gemini_embeddings(texts, model="models/" + model_choice.replace("gemini-", ""))
    else:
        if local_model_instance is None: 
            st.error("Local embedding model not loaded.")
            return np.array([])
        return local_model_instance.encode(list(texts) if isinstance(texts, tuple) else texts)

# --- Content Processing Functions ---
def initialize_selenium_driver():
    options = ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    try:
        driver = webdriver.Chrome(service=ChromeService(), options=options)
        stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32", webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)
        return driver
    except Exception as e:
        st.error(f"Selenium initialization failed: {e}")
        return None

def fetch_content_with_zyte(url, api_key):
    """Fetches HTML content using Zyte API."""
    if not api_key:
        st.error("Zyte API key not configured.")
        return None

    enforce_rate_limit()
    st.write(f"_Fetching with Zyte API: {url}..._")

    try:
        response = requests.post(
            "https://api.zyte.com/v1/extract",
            auth=(api_key, ''),
            json={'url': url, 'httpResponseBody': True},
            timeout=45
        )
        response.raise_for_status()

        data = response.json()
        if data.get('httpResponseBody'):
            html_content = base64.b64decode(data['httpResponseBody']).decode('utf-8', 'ignore')
            st.write(f"_✓ Got HTML ({len(html_content)} chars)_")
            return html_content
        else:
            st.error(f"Zyte API did not return content for {url}")
            return None

    except Exception as e:
        st.error(f"Zyte API error for {url}: {e}")
        return None

def fetch_content_with_selenium(url, driver_instance):
    if not driver_instance: 
        return fetch_content_with_requests(url)
    try:
        enforce_rate_limit()
        driver_instance.get(url)
        time.sleep(5)
        return driver_instance.page_source
    except Exception as e:
        st.error(f"Selenium error for {url}: {e}")
        st.session_state.selenium_driver_instance = None
        st.warning(f"Selenium failed for {url}. Falling back to requests.")
        return fetch_content_with_requests(url)

def fetch_content_with_requests(url):
    enforce_rate_limit()
    headers = {'User-Agent': get_random_user_agent()}
    try:
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Requests error for {url}: {e}")
        return None

def clean_text_for_display(text):
    if not text: 
        return ""
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = re.sub(r'([.?!])([a-zA-Z])', r'\1 \2', cleaned_text)
    cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)
    return cleaned_text

def split_text_into_sentences(text):
    if not text: 
        return []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]

def extract_structural_passages_with_full_text(html_content, chunk_size=500):
    if not html_content:
        return [], ""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract image alt text before removing images (valuable for galleries)
    image_alt_texts = []
    for img in soup.find_all('img'):
        alt = img.get('alt', '').strip()
        if alt and len(alt.split()) >= 3:  # Only meaningful alt text
            image_alt_texts.append(alt)

    # Remove unwanted elements (keep figure/figcaption for gallery captions)
    for el in soup(["script", "style", "noscript", "iframe", "link", "meta", 'nav', 'header', 'footer', 'aside', 'form', 'menu', 'banner', 'dialog', 'img', 'svg']):
        if el.name:
            el.decompose()
    
    # Remove elements by selectors
    selectors = ["[class*='menu']", "[id*='nav']", "[class*='header']", "[id*='footer']", "[class*='sidebar']", "[class*='cookie']", "[class*='consent']", "[class*='popup']", "[class*='modal']", "[class*='social']", "[class*='share']", "[class*='advert']", "[id*='ad']", "[aria-hidden='true']"]
    for sel in selectors:
        try:
            for element in soup.select(sel):
                if not any(p.name in ['main', 'article', 'body'] for p in element.parents): 
                    element.decompose()
        except Exception: 
            pass
    
    # Extract content from semantic HTML tags first (including figcaption for gallery captions)
    semantic_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table', 'article', 'section', 'figcaption']
    content_elements = list(soup.find_all(semantic_tags))

    # Also extract content from divs/spans with content-related classes (e-commerce, collection pages, etc.)
    # BUT only if they don't contain semantic child elements (to avoid duplication)
    content_selectors = [
        # E-commerce / product content
        "[class*='product']",
        "[class*='item']",
        "[class*='card']",
        "[class*='listing']",
        # Text content
        "[class*='title']",
        "[class*='name']",
        "[class*='description']",
        "[class*='content']",
        "[class*='text']",
        "[class*='hero']",
        "[class*='collection']",
        "[class*='category']",
        "[class*='info']",
        "[class*='detail']",
        "[class*='summary']",
        "[class*='body']",
        "[class*='main']",
        "[class*='rte']",  # Rich text editor content (common in Shopify)
        # Gallery / image captions
        "[class*='caption']",
        "[class*='gallery']",
        "[class*='photo']",
        "[class*='image-title']",
        "[class*='img-caption']",
        "[class*='thumbnail']",
        "[class*='thumb']",
        "[class*='lightbox']",
        "[class*='slide']",
        "[class*='fig']",
        "[data-caption]",
        "[alt]",  # Image alt text often contains descriptions
        # Data attributes
        "[data-product]",
        "[data-content]",
        "[data-title]"
    ]

    for sel in content_selectors:
        try:
            for element in soup.select(sel):
                # Skip if this element contains semantic child elements (they're already extracted)
                has_semantic_children = element.find(semantic_tags)
                if has_semantic_children:
                    continue

                # Get text from these content elements (leaf nodes or containers without semantic children)
                text = element.get_text(separator=' ', strip=True)
                if text and len(text.split()) > 2 and element not in content_elements:
                    content_elements.append(element)
        except Exception:
            pass

    # Deduplicate: remove elements that are ancestors/descendants of each other
    # Keep the more specific (child) element when there's nesting
    filtered_elements = []
    for el in content_elements:
        # Check if any other element in our list is a parent of this one
        is_ancestor_in_list = False
        for other_el in content_elements:
            if other_el != el and el in other_el.descendants:
                is_ancestor_in_list = True
                break

        # Only keep if no ancestor is already in the list (prefer children over parents)
        if not is_ancestor_in_list:
            filtered_elements.append(el)

    # Final deduplication by element id
    seen_elements = set()
    unique_content_elements = []
    for el in filtered_elements:
        if id(el) not in seen_elements:
            seen_elements.add(id(el))
            unique_content_elements.append(el)
    content_elements = unique_content_elements

    merged_passages = []
    i = 0

    while i < len(content_elements):
        current_element = content_elements[i]
        current_text = current_element.get_text(separator=' ', strip=True)

        # Merge headers with following content
        if current_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and (i + 1) < len(content_elements):
            next_text = content_elements[i+1].get_text(separator=' ', strip=True)
            combined_text = f"{current_text}. {next_text}"
            if combined_text.strip():
                merged_passages.append(combined_text)
            i += 2
        else:
            if current_text.strip():
                merged_passages.append(current_text)
            i += 1
    
    # Include image alt texts as additional passages (valuable for gallery pages)
    merged_passages.extend(image_alt_texts)

    # Clean all passages
    cleaned_passages = [clean_text_for_display(p) for p in merged_passages if p and len(p.split()) > 2]

    # Chunk passages by character count (~500 chars per chunk, similar to LLM chunking)
    chunked_passages = chunk_passages_by_size(cleaned_passages, target_chunk_size=chunk_size)

    full_text = clean_text_for_display(soup.get_text(separator=' '))
    return chunked_passages, full_text


def chunk_passages_by_size(passages, target_chunk_size=500, min_chunk_size=100):
    """
    Groups small passages together into larger chunks of approximately target_chunk_size characters.
    Keeps larger passages intact. This improves semantic similarity for galleries/product listings.
    """
    if not passages:
        return []

    chunked = []
    current_chunk = []
    current_size = 0

    for passage in passages:
        passage_len = len(passage)

        # If this single passage is already large enough, flush current chunk and add it separately
        if passage_len >= target_chunk_size:
            # First, flush any accumulated small passages
            if current_chunk:
                chunked.append(" | ".join(current_chunk))
                current_chunk = []
                current_size = 0
            # Add the large passage as its own chunk
            chunked.append(passage)
        else:
            # Small passage - accumulate into current chunk
            current_chunk.append(passage)
            current_size += passage_len + 3  # +3 for " | " separator

            # If current chunk is large enough, flush it
            if current_size >= target_chunk_size:
                chunked.append(" | ".join(current_chunk))
                current_chunk = []
                current_size = 0

    # Don't forget any remaining passages in the current chunk
    if current_chunk:
        # If it's too small and we have previous chunks, try to merge with the last one
        remaining_text = " | ".join(current_chunk)
        if len(remaining_text) < min_chunk_size and chunked:
            # Merge with last chunk if combined size is reasonable
            last_chunk = chunked[-1]
            if len(last_chunk) + len(remaining_text) < target_chunk_size * 1.5:
                chunked[-1] = last_chunk + " | " + remaining_text
            else:
                chunked.append(remaining_text)
        else:
            chunked.append(remaining_text)

    return chunked

def add_sentence_overlap_to_passages(structural_passages, overlap_count=2):
    if not structural_passages or overlap_count == 0: 
        return structural_passages
    
    def _get_n_sentences(text, n, from_start=True):
        sentences = split_text_into_sentences(text)
        if not sentences: 
            return ""
        return " ".join(sentences[:n]) if from_start else " ".join(sentences[-n:])
    
    expanded_passages = []
    num_passages = len(structural_passages)
    
    for i in range(num_passages):
        current_passage = structural_passages[i]
        prefix = _get_n_sentences(structural_passages[i-1], overlap_count, from_start=False) if i > 0 else ""
        suffix = _get_n_sentences(structural_passages[i+1], overlap_count, from_start=True) if i < num_passages - 1 else ""
        expanded_passages.append(" ".join(filter(None, [prefix, current_passage, suffix])))
    
    return [p for p in expanded_passages if p.strip()]

def render_safe_highlighted_html(html_content, unit_scores_map):
    if not html_content or not unit_scores_map: 
        return "<p>Could not generate highlighted HTML.</p>"
    
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract image alt text before removing images
    image_alt_parts = []
    for img in soup.find_all('img'):
        alt = img.get('alt', '').strip()
        if alt and len(alt.split()) >= 3:
            # Find score for this alt text
            alt_score = unit_scores_map.get(alt, 0.5)
            for unit_text, score in unit_scores_map.items():
                if alt in unit_text or unit_text in alt:
                    alt_score = max(alt_score, score)
            color = "green" if alt_score >= 0.75 else "red" if alt_score < 0.60 else "inherit"
            style = f"color:{color}; border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 0.5em; font-style: italic;"
            image_alt_parts.append(f"<p style='{style}'>[Image: {alt}]</p>")

    tags_to_remove = ["script", "style", "noscript", "iframe", "link", "meta", "button", "a", "img", "svg", "video", "audio", "canvas", 'form', 'nav', 'header', 'footer', 'aside', 'menu', 'banner', 'dialog']

    for tag in tags_to_remove:
        for el in soup.find_all(tag):
            el.decompose()

    html_parts = []
    semantic_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table', 'article', 'section', 'figcaption']

    # Collect elements from semantic tags
    all_elements = list(soup.find_all(semantic_tags))

    # Also collect content-related elements (e-commerce, collection pages, galleries, etc.)
    # BUT only if they don't contain semantic child elements (to avoid duplication)
    content_selectors = [
        # E-commerce / product content
        "[class*='product']",
        "[class*='item']",
        "[class*='card']",
        "[class*='listing']",
        # Text content
        "[class*='title']",
        "[class*='name']",
        "[class*='description']",
        "[class*='content']",
        "[class*='text']",
        "[class*='hero']",
        "[class*='collection']",
        "[class*='category']",
        "[class*='info']",
        "[class*='detail']",
        "[class*='summary']",
        "[class*='body']",
        "[class*='main']",
        "[class*='rte']",
        # Gallery / image captions
        "[class*='caption']",
        "[class*='gallery']",
        "[class*='photo']",
        "[class*='image-title']",
        "[class*='img-caption']",
        "[class*='thumbnail']",
        "[class*='thumb']",
        "[class*='lightbox']",
        "[class*='slide']",
        "[class*='fig']",
        "[data-caption]",
        # Data attributes
        "[data-product]",
        "[data-content]",
        "[data-title]"
    ]
    for sel in content_selectors:
        try:
            for element in soup.select(sel):
                # Skip if this element contains semantic child elements
                has_semantic_children = element.find(semantic_tags)
                if has_semantic_children:
                    continue

                text = element.get_text(separator=' ', strip=True)
                if text and len(text.split()) > 2 and element not in all_elements:
                    all_elements.append(element)
        except Exception:
            pass

    # Deduplicate: remove elements that are ancestors/descendants of each other
    filtered_elements = []
    for el in all_elements:
        is_ancestor_in_list = False
        for other_el in all_elements:
            if other_el != el and el in other_el.descendants:
                is_ancestor_in_list = True
                break
        if not is_ancestor_in_list:
            filtered_elements.append(el)

    # Final deduplication by element id
    seen_ids = set()
    unique_elements = []
    for el in filtered_elements:
        if id(el) not in seen_ids:
            seen_ids.add(id(el))
            unique_elements.append(el)
    all_elements = unique_elements

    for element in all_elements:
        element_text = clean_text_for_display(element.get_text(separator=' '))
        passage_score = 0.5  # Default score
        
        if not element_text: 
            continue
        
        # Try multiple matching strategies to find the best score
        best_score = 0.5
        
        # Strategy 1: Exact match
        if element_text in unit_scores_map:
            best_score = unit_scores_map[element_text]
        else:
            # Strategy 2: Check if element is contained in any passage
            for unit_text, score in unit_scores_map.items():
                if element_text in unit_text:
                    best_score = max(best_score, score)
            
            # Strategy 3: Check if any passage is contained in element (for very long elements)
            if best_score == 0.5:
                for unit_text, score in unit_scores_map.items():
                    if unit_text in element_text:
                        best_score = max(best_score, score)
            
            # Strategy 4: Fuzzy matching for partial overlaps
            if best_score == 0.5:
                element_words = set(element_text.lower().split())
                if len(element_words) > 3:  # Only for substantial text
                    for unit_text, score in unit_scores_map.items():
                        unit_words = set(unit_text.lower().split())
                        if len(unit_words) > 3:
                            # Calculate word overlap percentage
                            overlap = len(element_words.intersection(unit_words))
                            overlap_ratio = overlap / min(len(element_words), len(unit_words))
                            
                            # If substantial overlap (>60%), use a weighted score
                            if overlap_ratio > 0.6:
                                weighted_score = score * overlap_ratio
                                best_score = max(best_score, weighted_score)
        
        passage_score = best_score
        
        # Color coding based on score
        color = "green" if passage_score >= 0.75 else "red" if passage_score < 0.60 else "inherit"
        style = f"color:{color}; border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 1em; margin-top: 1em;"
        
        if element.name in ['ul', 'ol']:
            list_items_html = "".join([f"<li>{clean_text_for_display(li.get_text())}</li>" for li in element.find_all('li')])
            html_parts.append(f"<{element.name} style='{style}'>{list_items_html}</{element.name}>")
        else:
            html_parts.append(f"<{element.name} style='{style}'>{element_text}</{element.name}>")

    # Append image alt text passages at the end (for gallery pages)
    if image_alt_parts:
        html_parts.append("<h4 style='margin-top: 2em; color: #666;'>Image Descriptions:</h4>")
        html_parts.extend(image_alt_parts)

    return "".join(html_parts)

def get_sentence_highlighted_html_flat(page_text_content, unit_scores_map):
    if not page_text_content or not unit_scores_map: 
        return "<p>No content to highlight.</p>"
    
    sentences = split_text_into_sentences(page_text_content)
    if not sentences: 
        return "<p>No sentences to highlight.</p>"
    
    highlighted_html = ""
    for sentence in sentences:
        sentence_score = 0.5
        cleaned_sentence = clean_text_for_display(sentence)
        if cleaned_sentence in unit_scores_map: 
            sentence_score = unit_scores_map[cleaned_sentence]
        color = "green" if sentence_score >= 0.75 else "red" if sentence_score < 0.60 else "black"
        highlighted_html += f'<p style="color:{color}; margin-bottom: 2px;">{cleaned_sentence}</p>'
    
    return highlighted_html

def generate_seo_recommendations(url, analysis_results, processed_units_data):
    """
    Uses Gemini to generate SEO/AI Search recommendations based on the similarity analysis results.
    """
    if not st.session_state.get("gemini_api_configured", False):
        st.error("Gemini API not configured for recommendations.")
        return None

    # Prepare analysis summary for Gemini
    url_metrics = [m for m in analysis_results if m["URL"] == url]
    if not url_metrics:
        return None

    # Get overall scores
    avg_overall_sim = np.mean([m["Overall Similarity (Weighted)"] for m in url_metrics])
    avg_max_sim = np.mean([m["Max Unit Sim."] for m in url_metrics])

    # Identify weak and strong queries
    weak_queries = [(m["Query"], m["Overall Similarity (Weighted)"], m["Max Similarity Passage"])
                    for m in url_metrics if m["Overall Similarity (Weighted)"] < 0.6]
    strong_queries = [(m["Query"], m["Overall Similarity (Weighted)"], m["Max Similarity Passage"])
                      for m in url_metrics if m["Overall Similarity (Weighted)"] >= 0.75]

    # Get content passages for context
    content_passages = []
    if url in processed_units_data:
        units = processed_units_data[url].get("units", [])
        unit_sims = processed_units_data[url].get("unit_similarities")
        if unit_sims is not None and len(units) > 0:
            # Get average similarity for each passage across all queries
            avg_passage_sims = np.mean(unit_sims, axis=1)
            # Get top and bottom passages
            sorted_indices = np.argsort(avg_passage_sims)
            bottom_passages = [(units[i], avg_passage_sims[i]) for i in sorted_indices[:5] if i < len(units)]
            top_passages = [(units[i], avg_passage_sims[i]) for i in sorted_indices[-5:] if i < len(units)]
            content_passages = {"top": top_passages, "bottom": bottom_passages}

    # Build prompt for Gemini
    prompt = f"""You are an expert SEO consultant specializing in AI Search optimization (Google AI Overviews, ChatGPT Search, Perplexity, etc.).

Analyze this content similarity report and provide actionable SEO recommendations.

## Target URL
{url}

## Overall Performance
- Average Similarity Score: {avg_overall_sim:.3f} (scale 0-1, higher is better)
- Average Best-Match Score: {avg_max_sim:.3f}

## Queries with WEAK Content Coverage (< 0.6 similarity):
"""

    if weak_queries:
        for query, score, passage in weak_queries[:10]:
            prompt += f"\n- Query: \"{query}\" (Score: {score:.3f})"
            if passage:
                prompt += f"\n  Best matching content: \"{passage[:200]}...\""
    else:
        prompt += "\nNo significant content gaps detected."

    prompt += "\n\n## Queries with STRONG Content Coverage (>= 0.75 similarity):\n"

    if strong_queries:
        for query, score, passage in strong_queries[:10]:
            prompt += f"\n- Query: \"{query}\" (Score: {score:.3f})"
    else:
        prompt += "\nNo queries with strong coverage detected."

    if content_passages:
        prompt += "\n\n## Lowest Scoring Content Passages (potential improvement areas):\n"
        for passage, score in content_passages.get("bottom", []):
            prompt += f"\n- (Score: {score:.3f}) \"{passage[:300]}...\"\n"

        prompt += "\n\n## Highest Scoring Content Passages (strengths to maintain):\n"
        for passage, score in content_passages.get("top", []):
            prompt += f"\n- (Score: {score:.3f}) \"{passage[:300]}...\"\n"

    prompt += """

## Your Task
Provide specific, actionable SEO recommendations to improve this page's visibility in AI Search results. Focus on:

1. **Content Gaps**: What topics or information should be added to better answer the weak queries?
2. **Content Structure**: How can the existing content be restructured for better AI understanding?
3. **Semantic Coverage**: What related concepts, entities, or long-tail variations are missing?
4. **AI Search Optimization**: Specific tips for appearing in AI Overviews, featured snippets, and conversational AI results.
5. **Quick Wins**: 3-5 immediate changes that could improve scores.

Format your response with clear headers and bullet points. Be specific and reference the actual queries and content where relevant.
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini recommendation generation failed: {e}")
        return None


def generate_synthetic_queries(user_query, num_queries=7):
    if not st.session_state.get("gemini_api_configured", False):
        st.error("Gemini API not configured for query generation.")
        return []
    
    model = genai.GenerativeModel("gemini-2.5-pro")
    prompt = f"""
    Based on the user's initial search query: "{user_query}"
    Generate {num_queries} diverse synthetic search queries using the "Query Fan Out" technique. These queries should explore different facets, intents, or related concepts. Aim for a mix of the following query types, ensuring variety:
    
    1. Related Queries: Queries on closely associated topics or entities.
    2. Implicit Queries: Queries that unearth underlying assumptions or unstated needs related to the original query.
    3. Comparative Queries: Queries that seek to compare aspects of the original query's subject with alternatives.
    4. Recent Queries: Logical next steps or follow-up queries someone might search for.
    5. Personalized Queries: How the query might be personalized for different user contexts.
    6. Reformulation Queries: Different ways of phrasing the original query.
    7. Entity-Expanded Queries: Queries that broaden scope by including related entities.
    
    CRITICAL INSTRUCTIONS:
    - Ensure queries span multiple categories above for semantic diversity
    - Make queries distinct and aimed at retrieving diverse content types
    - Do NOT number queries or add prefixes like "Query 1:"
    - Return ONLY a Python-parseable list of strings
    - Example format: ["synthetic query 1", "another synthetic query", "a third query"]
    - Each query should be complete and self-contained
    """
    
    try:
        response = model.generate_content(prompt)
        content_text = response.text.strip()
        
        # Clean up response format
        for prefix in ["```python", "```json", "```"]:
            if content_text.startswith(prefix): 
                content_text = content_text.split(prefix, 1)[1].rsplit("```", 1)[0].strip()
                break
        
        # Parse response
        if content_text.startswith('['):
            queries = ast.literal_eval(content_text)
        else:
            # Fallback parsing for non-list format
            queries = [re.sub(r'^\s*[-\*\d\.]+\s*', '', q.strip().strip('"\'')) 
                      for q in content_text.split('\n') if q.strip()]
        
        if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries): 
            raise ValueError("Invalid format returned")
            
        return [q for q in queries if q.strip()]
        
    except Exception as e:
        st.error(f"Query generation failed: {e}")
        # Extract queries manually as fallback
        lines = content_text.split('\n')
        extracted = [re.sub(r'^\s*[-\*\d\.]+\s*', '', line.strip().strip('"\'')) 
                    for line in lines if line.strip()]
        if extracted: 
            st.warning("Using fallback query parsing")
            return extracted
        return []

# --- Main UI ---
st.title("🚀 AI Mode Query Fan-Out Analyzer")
st.markdown("**Generate diverse synthetic queries and analyze content similarity across multiple search intents.**")

# --- Sidebar Configuration ---
st.sidebar.subheader("🤖 Embedding Model Configuration")
embedding_model_options = { 
    "Local: MPNet (Quality Focus)": "all-mpnet-base-v2", 
    "Local: MiniLM (Speed Focus)": "all-MiniLM-L6-v2", 
    "Local: DistilRoBERTa (Balanced)": "all-distilroberta-v1", 
    "Local: MixedBread (Large & Powerful)": "mixedbread-ai/mxbai-embed-large-v1",
    "Google: Gemma Embedding (300M)": "google/embeddinggemma-300m",
    "OpenAI: text-embedding-3-small": "openai-text-embedding-3-small", 
    "OpenAI: text-embedding-3-large": "openai-text-embedding-3-large", 
    "Gemini: embedding-001": "gemini-embedding-001"
}
selected_embedding_label = st.sidebar.selectbox("Select Embedding Model:", options=list(embedding_model_options.keys()), index=0, disabled=st.session_state.processing)
st.session_state.selected_embedding_model = embedding_model_options[selected_embedding_label]

st.sidebar.subheader("📄 Text Extraction & Processing")
analysis_granularity = st.sidebar.selectbox("Analysis Granularity:", ("Passage-based (HTML Tags)", "Sentence-based"), index=0, disabled=st.session_state.processing)

scraping_method_options = ["Zyte API (best for tough sites)", "Selenium (for dynamic sites)", "Requests (lightweight)"]
scraping_method = st.sidebar.selectbox(
    "Scraping Method:",
    options=scraping_method_options,
    index=0,
    disabled=st.session_state.processing,
    help="Zyte API avoids most blocks. Selenium renders JavaScript. Requests is fastest but easily blocked."
)

st.sidebar.divider()
st.sidebar.header("⚙️ Query Configuration")
initial_query_val = st.sidebar.text_input("Initial Search Query:", "benefits of server-side rendering", disabled=st.session_state.processing)
num_sq_val = st.sidebar.slider("Number of Synthetic Queries:", 3, 20, 7, disabled=st.session_state.processing)

st.sidebar.subheader("📊 Input Mode")
input_mode = st.sidebar.radio("Choose Input Mode:", ("Analyze URLs", "Analyze Pasted Text"), disabled=st.session_state.processing)

if input_mode == "Analyze URLs":
    urls_text_area_val = st.sidebar.text_area(
        "Enter URLs (one per line):", 
        "https://cloudinary.com/guides/automatic-image-cropping/server-side-rendering-benefits-use-cases-and-best-practices\nhttps://prismic.io/blog/what-is-ssr", 
        height=100, 
        disabled=st.session_state.processing
    )
else: 
    pasted_content_text = st.sidebar.text_area("Paste content here:", height=200, disabled=st.session_state.processing)

if analysis_granularity.startswith("Passage"):
    st.sidebar.subheader("Passage Context Settings")
    chunk_size_val = st.sidebar.slider(
        "Chunk Size (characters):",
        200, 1000, 500,
        step=100,
        help="Groups small passages (like gallery captions, product titles) into chunks of this size. ~500 chars is optimal for embedding models.",
        disabled=st.session_state.processing
    )
    s_overlap_val = st.sidebar.slider("Context Sentence Overlap:", 0, 5, 2, help="Adds context from neighboring passages for better similarity calculation.", disabled=st.session_state.processing)
else:
    chunk_size_val = 500
    s_overlap_val = 0

analyze_disabled = not (st.session_state.gemini_api_configured or st.session_state.openai_api_configured)

if st.sidebar.button("🚀 Analyze Content", key="analyze_button", type="primary", disabled=st.session_state.processing or analyze_disabled):
    st.session_state.processing = True
    st.session_state.scraping_method = scraping_method
    st.rerun()

# --- Main Processing Block ---
if st.session_state.processing:
    try:
        # Validate inputs
        if not initial_query_val: 
            st.warning("Initial Search Query is required.")
            st.stop()
        
        # Prepare jobs
        jobs = []
        if input_mode == "Analyze URLs":
            if not urls_text_area_val: 
                st.warning("Please enter URLs to analyze.")
                st.stop()
            jobs = [{'type': 'url', 'identifier': url.strip()} for url in urls_text_area_val.split('\n') if url.strip()]
        else: 
            if not pasted_content_text:
                st.warning("Please paste content to analyze.")
                st.stop()
            jobs.append({'type': 'paste', 'identifier': "Pasted Content", 'content': pasted_content_text})
        
        # Load embedding model
        local_embedding_model_instance = None
        if not st.session_state.selected_embedding_model.startswith(("openai-", "gemini-")):
            with st.spinner(f"Loading embedding model..."): 
                local_embedding_model_instance = load_local_sentence_transformer_model(st.session_state.selected_embedding_model)
            if not local_embedding_model_instance: 
                st.stop()
        
        # Reset session state
        st.session_state.all_url_metrics_list = []
        st.session_state.url_processed_units_dict = {}
        st.session_state.all_queries_for_analysis = []
        st.session_state.analysis_done = False
        
        # Initialize Selenium if needed
        if st.session_state.get("scraping_method", "").startswith("Selenium") and input_mode == "Analyze URLs" and not st.session_state.selenium_driver_instance:
            with st.spinner("Initializing Selenium WebDriver..."): 
                st.session_state.selenium_driver_instance = initialize_selenium_driver()
        
        # Generate synthetic queries
        with st.spinner("Generating synthetic queries..."): 
            synthetic_queries = generate_synthetic_queries(initial_query_val, num_sq_val)
        
        if not synthetic_queries:
            st.warning("No synthetic queries generated. Continuing with initial query only.")
        
        local_all_queries = [f"Initial: {initial_query_val}"] + (synthetic_queries or [])
        
        # Embed all queries
        with st.spinner("Computing query embeddings..."): 
            local_all_query_embs = get_embeddings(local_all_queries, local_embedding_model_instance)
        
        if local_all_query_embs.size == 0:
            st.error("Query embedding failed. Please check your API configuration.")
            st.stop()
        
        initial_query_embedding = local_all_query_embs[0]
        
        # Fetch content
        local_all_metrics = []
        local_processed_units_data = {}
        fetched_content = {}
        
        if input_mode == "Analyze URLs":
            with st.spinner(f"Fetching content from {len(jobs)} URL(s)..."):
                for job in jobs:
                    url = job['identifier']
                    method = st.session_state.get("scraping_method", "Requests (lightweight)")
                    
                    if method.startswith("Zyte"):
                        fetched_content[url] = fetch_content_with_zyte(url, st.session_state.zyte_api_key_to_persist)
                    elif method.startswith("Selenium"):
                        fetched_content[url] = fetch_content_with_selenium(url, st.session_state.selenium_driver_instance)
                    else: 
                        fetched_content[url] = fetch_content_with_requests(url)

            # Clean up Selenium
            if st.session_state.selenium_driver_instance:
                st.session_state.selenium_driver_instance.quit()
                st.session_state.selenium_driver_instance = None
        else: 
            fetched_content[jobs[0]['identifier']] = jobs[0]['content']

        # Process content
        with st.spinner(f"Processing {len(fetched_content)} content source(s)..."):
            for identifier, content in fetched_content.items():
                st.markdown(f"**Processing:** `{identifier}`")
                
                if not content or len(content.strip()) < 20: 
                    st.warning(f"Insufficient content from {identifier}. Skipping.")
                    continue
                
                # Determine if this is HTML content
                is_url_source = any(job['identifier'] == identifier for job in jobs if job['type'] == 'url')
                raw_html_for_highlighting = content if is_url_source else "".join([f"<p>{clean_text_for_display(p)}</p>" for p in content.split('\n\n')])
                
                # Extract text units based on granularity
                if analysis_granularity.startswith("Sentence"):
                    _, page_text_for_highlight = extract_structural_passages_with_full_text(raw_html_for_highlighting, chunk_size=chunk_size_val)
                    units_for_display = split_text_into_sentences(page_text_for_highlight) if page_text_for_highlight else []
                    units_for_embedding = units_for_display
                else:
                    units_for_display, page_text_for_highlight = extract_structural_passages_with_full_text(raw_html_for_highlighting, chunk_size=chunk_size_val)
                    units_for_embedding = add_sentence_overlap_to_passages(units_for_display, s_overlap_val)
                    if not units_for_display and page_text_for_highlight:
                        units_for_display = [page_text_for_highlight]

                if not units_for_embedding: 
                    st.warning(f"No processable content units found for {identifier}. Skipping.")
                    continue
                
                st.write(f"Found {len(units_for_embedding)} content units")
                
                # Generate embeddings for content units
                unit_embeddings = get_embeddings(units_for_embedding, local_embedding_model_instance)
                
                # Store processed data
                local_processed_units_data[identifier] = {
                    "units": units_for_display, 
                    "embeddings": unit_embeddings, 
                    "unit_similarities": None, 
                    "page_text_for_highlight": page_text_for_highlight, 
                    "raw_html": raw_html_for_highlighting
                }
                
                # Calculate similarities
                if unit_embeddings.size > 0:
                    # Calculate similarity to initial query for weighting
                    unit_sims_to_initial = cosine_similarity(unit_embeddings, initial_query_embedding.reshape(1, -1)).flatten()
                    weights = np.maximum(0, unit_sims_to_initial)
                    
                    # Calculate weighted overall embedding
                    if np.sum(weights) > 1e-6:
                        weighted_overall_emb = np.average(unit_embeddings, axis=0, weights=weights).reshape(1, -1)
                    else:
                        weighted_overall_emb = np.mean(unit_embeddings, axis=0).reshape(1, -1)
                    
                    # Calculate overall similarities to all queries
                    overall_sims = cosine_similarity(weighted_overall_emb, local_all_query_embs)[0]
                    
                    # Calculate unit-level similarities to all queries
                    unit_q_sims = cosine_similarity(unit_embeddings, local_all_query_embs)
                    local_processed_units_data[identifier]["unit_similarities"] = unit_q_sims
                    
                    # Generate metrics for each query
                    for sq_idx, query_text in enumerate(local_all_queries):
                        current_q_unit_sims = unit_q_sims[:, sq_idx]
                        max_sim_passage_text = ""
                        
                        if current_q_unit_sims.size > 0: 
                            max_sim_idx = np.argmax(current_q_unit_sims)
                            max_sim_passage_text = units_for_display[max_sim_idx]
                        
                        local_all_metrics.append({ 
                            "URL": identifier, 
                            "Query": query_text, 
                            "Overall Similarity (Weighted)": overall_sims[sq_idx], 
                            "Max Unit Sim.": np.max(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0, 
                            "Avg. Unit Sim.": np.mean(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0, 
                            "Num Units": len(units_for_display), 
                            "Max Similarity Passage": max_sim_passage_text 
                        })
        
        # Save results
        if local_all_metrics:
            st.session_state.all_url_metrics_list = local_all_metrics
            st.session_state.url_processed_units_dict = local_processed_units_data
            st.session_state.all_queries_for_analysis = local_all_queries
            st.session_state.analysis_done = True
            st.session_state.last_analysis_granularity = analysis_granularity
            st.success(f"Analysis complete! Processed {len(local_processed_units_data)} sources with {len(local_all_queries)} queries.")

    finally:
        st.session_state.processing = False
        st.rerun()

# --- Results Display Section ---
if st.session_state.get("analysis_done") and st.session_state.all_url_metrics_list:
    st.markdown("---")
    st.subheader("📋 Generated Queries")
    
    # Display all queries in a clean format
    query_display = []
    for i, query in enumerate(st.session_state.all_queries_for_analysis):
        if query.startswith("Initial: "):
            query_display.append(f"🎯 **Initial Query:** {query.replace('Initial: ', '')}")
        else:
            query_display.append(f"🔄 **Synthetic {i}:** {query}")
    
    with st.expander("View All Generated Queries", expanded=False):
        for q in query_display:
            st.markdown(q)
    
    unit_label = "Sentence" if st.session_state.last_analysis_granularity.startswith("Sentence") else "Passage"
    
    st.markdown("---")
    st.subheader(f"📊 Content Similarity Analysis")
    
    # Create summary dataframe
    df_summary = pd.DataFrame(st.session_state.all_url_metrics_list)
    df_display = df_summary.rename(columns={
        "URL": "Source", 
        "Max Unit Sim.": f"Max {unit_label} Sim.", 
        "Avg. Unit Sim.": f"Avg. {unit_label} Sim.", 
        "Num Units": f"Num {unit_label}s"
    })
    
    # Display summary table
    st.dataframe(
        df_display, 
        use_container_width=True,
        column_config={
            "Overall Similarity (Weighted)": st.column_config.ProgressColumn(
                "Overall Similarity",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            f"Max {unit_label} Sim.": st.column_config.ProgressColumn(
                f"Max {unit_label} Sim.",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
            f"Avg. {unit_label} Sim.": st.column_config.ProgressColumn(
                f"Avg. {unit_label} Sim.",
                format="%.3f",
                min_value=0,
                max_value=1,
            ),
        }
    )
    
    st.markdown("---")
    st.subheader("📈 Similarity Visualization")
    
    # Create bar chart
    fig_bar = px.bar(
        df_display, 
        x="Query", 
        y="Overall Similarity (Weighted)", 
        color="Source", 
        barmode="group", 
        title="Content Similarity Across All Queries",
        height=max(500, 60 * len(st.session_state.all_queries_for_analysis))
    )
    fig_bar.update_yaxes(range=[0, 1])
    fig_bar.update_xaxes(tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    st.subheader(f"🔥 Detailed {unit_label} Analysis")
    
    # Detailed analysis for each source
    for item_idx, (identifier, p_data) in enumerate(st.session_state.url_processed_units_dict.items()):
        with st.expander(f"📄 Analysis for: {identifier}", expanded=(item_idx == 0)):
            if p_data.get("unit_similarities") is None: 
                st.write(f"No {unit_label.lower()} similarity data available.")
                continue
            
            unit_sims = p_data["unit_similarities"]
            units = p_data["units"]
            all_queries = st.session_state.all_queries_for_analysis
            
            # Create heatmap
            short_queries = [q.replace("Initial: ", "(I) ")[:40] + ('...' if len(q) > 40 else '') for q in all_queries]
            unit_labels = [f"{unit_label[0]}{i+1}" for i in range(len(units))]
            
            # Create hover text for heatmap
            wrapped_units = [textwrap.fill(unit, width=80, replace_whitespace=False).replace('\n', '<br>') for unit in units]
            hover_text = [
                [f"<b>{unit_labels[i]}</b> vs <b>{all_queries[j][:35]}...</b><br>Similarity: {unit_sims[i, j]:.3f}<br><hr><b>Text:</b><br>{wrapped_units[i]}" 
                 for i in range(unit_sims.shape[0])] 
                for j in range(unit_sims.shape[1])
            ]
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=unit_sims.T, 
                x=unit_labels, 
                y=short_queries, 
                colorscale='Viridis', 
                zmin=0, 
                zmax=1, 
                text=hover_text, 
                hoverinfo='text',
                colorbar=dict(title="Similarity Score")
            ))
            
            fig_heat.update_layout(
                title=f"{unit_label} Similarity Heatmap for {identifier}", 
                height=max(400, 35 * len(short_queries) + 100), 
                yaxis_autorange='reversed', 
                xaxis_title=f"{unit_label}s", 
                yaxis_title="Queries"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
            st.markdown("---")
            
            # Query-specific analysis
            key_base = f"{item_idx}_{identifier.replace('/', '_').replace(' ', '_')}"
            selected_query = st.selectbox(
                f"Select Query for Detailed Analysis:", 
                options=st.session_state.all_queries_for_analysis, 
                key=f"q_sel_{key_base}"
            )
            
            if selected_query:
                query_idx = st.session_state.all_queries_for_analysis.index(selected_query)
                scored_units = sorted(zip(p_data["units"], unit_sims[:, query_idx]), key=lambda item: item[1], reverse=True)
                query_display_name = selected_query.replace("Initial: ", "(Initial) ")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.checkbox(f"Show Highlighted Content", key=f"cb_hl_{key_base}_{query_idx}"):
                        with st.spinner("Generating highlights..."):
                            unit_scores_for_query = {unit_text: score for unit_text, score in scored_units}
                            if st.session_state.last_analysis_granularity.startswith("Passage"):
                                highlighted_html = render_safe_highlighted_html(p_data.get("raw_html"), unit_scores_for_query)
                            else:
                                highlighted_html = get_sentence_highlighted_html_flat(p_data["page_text_for_highlight"], unit_scores_for_query)
                            st.markdown(highlighted_html, unsafe_allow_html=True)
                
                with col2:
                    if st.checkbox(f"Show Top/Bottom {unit_label}s", key=f"cb_tb_{key_base}_{query_idx}"):
                        n_val = st.slider("Number to show:", 1, 10, 3, key=f"sl_tb_{key_base}_{query_idx}")
                        
                        st.markdown(f"**🔥 Top {n_val} {unit_label}s for:** {query_display_name}")
                        for i, (u_t, u_s) in enumerate(scored_units[:n_val]):
                            st.markdown(f"**#{i+1} - Score: {u_s:.3f}**")
                            st.markdown(f"> {u_t}")
                            st.divider()

                        st.markdown(f"**❄️ Bottom {n_val} {unit_label}s for:** {query_display_name}")
                        for i, (u_t, u_s) in enumerate(scored_units[-n_val:]):
                            st.markdown(f"**#{len(scored_units)-n_val+i+1} - Score: {u_s:.3f}**")
                            st.markdown(f"> {u_t}")
                            st.divider()

# --- SEO Recommendations Section ---
if st.session_state.get("analysis_done") and st.session_state.all_url_metrics_list:
    st.markdown("---")
    st.subheader("🎯 AI-Powered SEO Recommendations")

    if not st.session_state.get("gemini_api_configured", False):
        st.warning("Gemini API required for SEO recommendations. Please configure your Gemini API key in the sidebar.")
    else:
        # Allow user to select which URL to get recommendations for
        available_urls = list(st.session_state.url_processed_units_dict.keys())

        if len(available_urls) == 1:
            selected_url_for_recs = available_urls[0]
            st.markdown(f"**Target URL:** `{selected_url_for_recs}`")
        else:
            selected_url_for_recs = st.selectbox(
                "Select URL for SEO Recommendations:",
                options=available_urls,
                key="seo_rec_url_select"
            )

        # Initialize session state for recommendations
        if "seo_recommendations" not in st.session_state:
            st.session_state.seo_recommendations = {}

        col_rec1, col_rec2 = st.columns([1, 4])

        with col_rec1:
            generate_recs = st.button(
                "🚀 Generate Recommendations",
                key="gen_seo_recs_btn",
                type="primary",
                help="Uses Gemini to analyze your content gaps and provide actionable SEO recommendations"
            )

        with col_rec2:
            if selected_url_for_recs in st.session_state.seo_recommendations:
                if st.button("🔄 Regenerate", key="regen_seo_recs_btn"):
                    generate_recs = True

        if generate_recs:
            with st.spinner("Analyzing content and generating SEO recommendations..."):
                recommendations = generate_seo_recommendations(
                    url=selected_url_for_recs,
                    analysis_results=st.session_state.all_url_metrics_list,
                    processed_units_data=st.session_state.url_processed_units_dict
                )
                if recommendations:
                    st.session_state.seo_recommendations[selected_url_for_recs] = recommendations

        # Display recommendations if available
        if selected_url_for_recs in st.session_state.seo_recommendations:
            recommendations = st.session_state.seo_recommendations[selected_url_for_recs]

            # Calculate summary stats for context
            url_metrics = [m for m in st.session_state.all_url_metrics_list if m["URL"] == selected_url_for_recs]
            avg_sim = np.mean([m["Overall Similarity (Weighted)"] for m in url_metrics]) if url_metrics else 0
            weak_count = len([m for m in url_metrics if m["Overall Similarity (Weighted)"] < 0.6])
            strong_count = len([m for m in url_metrics if m["Overall Similarity (Weighted)"] >= 0.75])

            # Summary metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Avg. Similarity", f"{avg_sim:.2%}")
            with metric_col2:
                st.metric("Content Gaps", weak_count, help="Queries with < 60% similarity")
            with metric_col3:
                st.metric("Strong Matches", strong_count, help="Queries with >= 75% similarity")

            st.markdown("---")

            # Display the recommendations
            with st.container():
                st.markdown(recommendations)

            # Download button for recommendations
            st.download_button(
                label="📥 Download Recommendations",
                data=f"# SEO Recommendations for {selected_url_for_recs}\n\n{recommendations}",
                file_name=f"seo_recommendations_{selected_url_for_recs.replace('https://', '').replace('/', '_')[:50]}.md",
                mime="text/markdown"
            )

# Footer
st.sidebar.divider()
st.sidebar.info("🚀 AI Mode Query Fan-Out Analyzer v2.0")

