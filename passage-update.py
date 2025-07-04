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
import os
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from bs4 import BeautifulSoup
import textwrap
from selenium_stealth import stealth
import json

from google.cloud import language_v1
from google.oauth2 import service_account

st.set_page_config(layout="wide", page_title="AI Semantic Analyzer")

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
if "selected_embedding_model" not in st.session_state: st.session_state.selected_embedding_model = 'mixedbread-ai/mxbai-embed-large-v1'
if "processing" not in st.session_state: st.session_state.processing = False
if "gcp_nlp_configured" not in st.session_state: st.session_state.gcp_nlp_configured = False
if "gcp_credentials_info" not in st.session_state: st.session_state.gcp_credentials_info = None
if "entity_analysis_results" not in st.session_state: st.session_state.entity_analysis_results = None


REQUEST_INTERVAL = 3.0
last_request_time = 0
# ### RESTORED ### - Full User Agents List from your original script
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0"
]
def get_random_user_agent(): return random.choice(USER_AGENTS)
def enforce_rate_limit():
    global last_request_time
    now = time.time(); elapsed = now - last_request_time
    if elapsed < REQUEST_INTERVAL: time.sleep(REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

# --- Sidebar API Configuration ---
st.sidebar.header("🔑 API Configuration")
with st.sidebar.expander("OpenAI API", expanded=not st.session_state.get("openai_api_configured", False)):
    openai_api_key_input = st.text_input("Enter OpenAI API Key:", type="password", value=st.session_state.get("openai_api_key_to_persist", ""), disabled=st.session_state.processing)
    if st.button("Set & Verify OpenAI Key", disabled=st.session_state.processing):
        if openai_api_key_input:
            try:
                test_client = OpenAI(api_key=openai_api_key_input); test_client.embeddings.create(input=["test"], model="text-embedding-3-small")
                st.session_state.openai_api_key_to_persist, st.session_state.openai_api_configured, st.session_state.openai_client = openai_api_key_input, True, test_client
                st.success("OpenAI API Key Configured!"); st.rerun()
            except Exception as e:
                st.session_state.openai_api_key_to_persist, st.session_state.openai_api_configured, st.session_state.openai_client = "", False, None
                st.error(f"OpenAI Key Failed: {str(e)[:200]}")
        else: st.warning("Please enter OpenAI API Key.")

with st.sidebar.expander("Gemini API", expanded=not st.session_state.get("gemini_api_configured", False)):
    gemini_api_key_input = st.text_input("Enter Google Gemini API Key:", type="password", value=st.session_state.get("gemini_api_key_to_persist", ""), disabled=st.session_state.processing)
    if st.button("Set & Verify Gemini Key", disabled=st.session_state.processing):
        if gemini_api_key_input:
            try:
                genai.configure(api_key=gemini_api_key_input)
                if not any('generateContent' in m.supported_generation_methods for m in genai.list_models()): raise Exception("No usable models found for this API key.")
                st.session_state.gemini_api_key_to_persist, st.session_state.gemini_api_configured = gemini_api_key_input, True
                st.success("Gemini API Key Configured!"); st.rerun()
            except Exception as e:
                st.session_state.gemini_api_key_to_persist, st.session_state.gemini_api_configured = "", False
                st.error(f"API Key Failed: {str(e)[:200]}")
        else: st.warning("Please enter API Key.")

with st.sidebar.expander("Google Cloud NLP API", expanded=not st.session_state.gcp_nlp_configured):
    uploaded_gcp_key = st.file_uploader(
        "Upload Google Cloud Service Account JSON", type="json",
        help="Upload the JSON key file for a service account with 'Cloud Natural Language API User' role.",
        disabled=st.session_state.processing
    )
    if uploaded_gcp_key is not None:
        try:
            credentials_info = json.load(uploaded_gcp_key)
            if "project_id" in credentials_info and "private_key" in credentials_info:
                st.session_state.gcp_credentials_info, st.session_state.gcp_nlp_configured = credentials_info, True
                st.success(f"GCP Key for project '{credentials_info['project_id']}' loaded!")
            else:
                st.error("Invalid JSON key file format."); st.session_state.gcp_nlp_configured, st.session_state.gcp_credentials_info = False, None
        except Exception as e:
            st.error(f"Failed to process GCP key file: {e}"); st.session_state.gcp_nlp_configured, st.session_state.gcp_credentials_info = False, None

st.sidebar.markdown("---")
if st.session_state.get("openai_api_configured"): st.sidebar.markdown("✅ OpenAI API: **Configured**")
else: st.sidebar.markdown("⚠️ OpenAI API: **Not Configured**")
if st.session_state.get("gemini_api_configured"): st.sidebar.markdown("✅ Gemini API: **Configured**")
else: st.sidebar.markdown("⚠️ Gemini API: **Not Configured**")
if st.session_state.get("gcp_nlp_configured"): st.sidebar.markdown("✅ Google NLP API: **Configured**")
else: st.sidebar.markdown("⚠️ Google NLP API: **Not Configured**")
if st.session_state.get("openai_api_key_to_persist") and not st.session_state.get("openai_client"):
    st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key_to_persist)
if st.session_state.get("gemini_api_key_to_persist"):
    try: genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
    except Exception: st.session_state.gemini_api_configured = False

@st.cache_data(show_spinner="Extracting entities from text...")
def extract_entities_with_google_nlp(text: str, _credentials_info: dict):
    if not _credentials_info: st.error("Google Cloud credentials are not configured."); return {}
    if not text: return {}
    try:
        credentials = service_account.Credentials.from_service_account_info(_credentials_info)
        client = language_v1.LanguageServiceClient(credentials=credentials)
        max_bytes = 1000000; text_bytes = text.encode('utf-8')
        if len(text_bytes) > max_bytes:
            text = text_bytes[:max_bytes].decode('utf-8', 'ignore'); st.warning("Text was truncated to fit Google NLP API size limit.")
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = client.analyze_entities(document=document, encoding_type=language_v1.EncodingType.UTF8)
        entities_dict = {}
        for entity in response.entities:
            key = entity.name.lower()
            if key not in entities_dict or entity.salience > entities_dict[key]['salience']:
                 entities_dict[key] = {'name': entity.name, 'type': language_v1.Entity.Type(entity.type_).name, 'salience': entity.salience, 'mentions': len(entity.mentions)}
        return entities_dict
    except Exception as e: st.error(f"Google Cloud NLP API Error: {e}"); return {}

# --- Embedding Functions ---
@st.cache_resource
def load_local_sentence_transformer_model(model_name):
    try: return SentenceTransformer(model_name)
    except Exception as e: st.error(f"Failed to load local model '{model_name}': {e}"); return None
def get_openai_embeddings(texts: list, client: OpenAI, model: str):
    if not texts or not client: return np.array([])
    try:
        texts = [text.replace("\n", " ") for text in texts]; response = client.embeddings.create(input=texts, model=model); return np.array([item.embedding for item in response.data])
    except Exception as e: st.error(f"OpenAI embedding failed: {e}"); return np.array([])
def get_gemini_embeddings(texts: list, model: str):
    if not texts: return np.array([])
    try:
        result = genai.embed_content(model=model, content=texts, task_type="RETRIEVAL_DOCUMENT"); return np.array(result['embedding'])
    except Exception as e: st.error(f"Gemini embedding failed: {e}"); return np.array([])
def get_embeddings(texts, local_model_instance=None):
    model_choice = st.session_state.selected_embedding_model
    if model_choice.startswith("openai-"): return get_openai_embeddings(texts, client=st.session_state.openai_client, model=model_choice.replace("openai-", ""))
    elif model_choice.startswith("gemini-"): return get_gemini_embeddings(texts, model="models/" + model_choice.replace("gemini-", ""))
    else:
        if local_model_instance is None: st.error("Local embedding model not loaded."); return np.array([])
        return local_model_instance.encode(list(texts) if isinstance(texts, tuple) else texts)

# --- Core Content Processing Functions ---
def initialize_selenium_driver():
    options = ChromeOptions()
    options.add_argument("--headless"); options.add_argument("--no-sandbox"); options.add_argument("--disable-dev-shm-usage"); options.add_argument("--disable-gpu")
    try:
        driver = webdriver.Chrome(service=ChromeService(), options=options)
        stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32", webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)
        return driver
    except Exception as e:
        st.error(f"Selenium with Stealth init failed: {e}"); return None
def fetch_content_with_selenium(url, driver_instance):
    if not driver_instance: return fetch_content_with_requests(url)
    try:
        enforce_rate_limit(); driver_instance.get(url); time.sleep(5); return driver_instance.page_source
    except Exception as e:
        st.error(f"Selenium fetch error for {url}: {e}"); st.session_state.selenium_driver_instance = None
        st.warning(f"Selenium failed for {url}. Falling back to requests.")
        try: return fetch_content_with_requests(url)
        except Exception as req_e: st.error(f"Requests fallback also failed for {url}: {req_e}"); return None
def fetch_content_with_requests(url):
    enforce_rate_limit(); headers={'User-Agent':get_random_user_agent()}; resp=requests.get(url,timeout=20,headers=headers); resp.raise_for_status(); return resp.text

# ### RESTORED ### - Original clean_text_for_display function
def clean_text_for_display(text):
    if not text: return ""
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = re.sub(r'([.?!])([a-zA-Z])', r'\1 \2', cleaned_text)
    cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)
    return cleaned_text

def split_text_into_sentences(text):
    if not text: return []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]

def extract_structural_passages_with_full_text(html_content, use_trafilatura=True, favor_recall=False):
    if not html_content: return [], ""
    soup_input = html_content
    if use_trafilatura:
        cleaned_html = trafilatura.extract(
            html_content, include_comments=False, output_format='xml', favor_recall=favor_recall
        )
        if cleaned_html:
            soup_input = cleaned_html
        else:
            st.warning("Trafilatura extracted no content, falling back to raw HTML.")

    soup = BeautifulSoup(soup_input, 'lxml')
    for el in soup(["script", "style", "noscript", "iframe", "link", "meta", 'nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption', 'menu', 'banner', 'dialog', 'img', 'svg']):
        if el.name: el.decompose()

    target_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table']
    content_elements = soup.find_all(target_tags)
    merged_passages, i = [], 0
    while i < len(content_elements):
        current_element = content_elements[i]
        if current_element.name == 'table':
            rows = current_element.find_all('tr')
            table_text = ". ".join(
                " | ".join(cell.get_text(separator=' ', strip=True) for cell in row.find_all(['td', 'th']))
                for row in rows
            )
            current_text = table_text
        else:
            current_text = current_element.get_text(separator=' ', strip=True)

        if current_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and (i + 1) < len(content_elements) and content_elements[i+1].name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            next_element = content_elements[i+1]
            if next_element.name == 'table':
                 rows = next_element.find_all('tr')
                 next_text = ". ".join(" | ".join(cell.get_text(separator=' ', strip=True) for cell in row.find_all(['td', 'th'])) for row in rows)
            else:
                next_text = next_element.get_text(separator=' ', strip=True)
            
            combined_text = f"{current_text}. {next_text}"
            if combined_text.strip(): merged_passages.append(combined_text)
            i += 2
        else:
            if current_text.strip(): merged_passages.append(current_text)
            i += 1

    final_passages = [clean_text_for_display(p) for p in merged_passages if p and len(p.split()) > 2]
    full_text = clean_text_for_display(soup.get_text(separator=' '))
    return final_passages, full_text

def add_sentence_overlap_to_passages(structural_passages, overlap_count=2):
    if not structural_passages or overlap_count == 0: return structural_passages
    def _get_n_sentences(text, n, from_start=True):
        sentences = split_text_into_sentences(text)
        if not sentences: return ""
        return " ".join(sentences[:n]) if from_start else " ".join(sentences[-n:])
    expanded_passages = []
    num_passages = len(structural_passages)
    for i in range(num_passages):
        current_passage, prefix, suffix = structural_passages[i], _get_n_sentences(structural_passages[i-1], overlap_count, from_start=False) if i > 0 else "", _get_n_sentences(structural_passages[i+1], overlap_count, from_start=True) if i < num_passages - 1 else ""
        expanded_passages.append(" ".join(filter(None, [prefix, current_passage, suffix])))
    return [p for p in expanded_passages if p.strip()]

def render_safe_highlighted_html(html_content, unit_scores_map):
    if not html_content or not unit_scores_map: return "<p>Could not generate highlighted HTML.</p>"
    soup = BeautifulSoup(html_content, 'lxml')
    tags_to_remove = ["script", "style", "noscript", "iframe", "link", "meta", "button", "a", "img", "svg", "video", "audio", "canvas", "figure", "figcaption", 'form', 'nav', 'header', 'footer', 'aside', 'menu', 'banner', 'dialog']
    for tag in tags_to_remove:
        for el in soup.find_all(tag): el.decompose()
    html_parts, target_tags = [], ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table']
    for element in soup.find_all(target_tags):
        element_text = clean_text_for_display(element.get_text(separator=' ')); passage_score = 0.5
        if not element_text: continue
        for unit_text, score in unit_scores_map.items():
            if element_text in unit_text: passage_score = score; break
        color = "green" if passage_score >= 0.70 else "red" if passage_score < 0.50 else "inherit"
        style = f"color:{color}; border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 1em; margin-top: 1em;"
        if element.name in ['ul', 'ol']:
            list_items_html = "".join([f"<li>{clean_text_for_display(li.get_text())}</li>" for li in element.find_all('li')])
            html_parts.append(f"<{element.name} style='{style}'>{list_items_html}</{element.name}>")
        else: html_parts.append(f"<{element.name} style='{style}'>{element_text}</{element.name}>")
    return "".join(html_parts)

def get_sentence_highlighted_html_flat(page_text_content, unit_scores_map):
    if not page_text_content or not unit_scores_map: return "<p>No content to highlight.</p>"
    sentences = split_text_into_sentences(page_text_content)
    if not sentences: return "<p>No sentences to highlight.</p>"
    highlighted_html = ""
    for sentence in sentences:
        sentence_score, cleaned_sentence = 0.5, clean_text_for_display(sentence)
        if cleaned_sentence in unit_scores_map: sentence_score = unit_scores_map[cleaned_sentence]
        color = "green" if sentence_score >= 0.70 else "red" if sentence_score < 0.50 else "black"
        highlighted_html += f'<p style="color:{color}; margin-bottom: 2px;">{cleaned_sentence}</p>'
    return highlighted_html

# ### RESTORED ### - Full, detailed Gemini prompt and robust parsing from your original script
def generate_synthetic_queries(user_query, num_queries=7):
    if not st.session_state.get("gemini_api_configured", False): st.error("Gemini API not configured."); return []
    model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")
    prompt = f"""
    Based on the user's initial search query: "{user_query}"
    Generate {num_queries} diverse synthetic search queries using the "Query Fan Out" technique. These queries should explore different facets, intents, or related concepts. Aim for a mix of the following query types, ensuring variety:
    1. Related Queries: Queries on closely associated topics or entities.
    2. Implicit Queries: Queries that unearth underlying assumptions or unstated needs related to the original query.
    3. Comparative Queries: Queries that seek to compare aspects of the original query's subject with alternatives or other facets.
    4. Recent Queries (Hypothetical): If this were part of an ongoing search session, what related queries might have come before or could follow? (Focus on logical next steps).
    5. Personalized Queries (Hypothetical): Example queries that reflect how the original query might be personalized based on hypothetical user contexts, preferences, or history (e.g., "best [topic] for families with young children," or "[topic] near me now").
    6. Reformulation Queries: Different ways of phrasing the original query to capture the same or slightly nuanced intent.
    7. Entity-Expanded Queries: Queries that broaden the scope by including related entities or exploring the original entity in more contexts (e.g., if original is "Eiffel Tower", expand to "history of Eiffel Tower construction", "restaurants near Eiffel Tower", "Eiffel Tower controversy").
    CRITICAL INSTRUCTIONS:
    - Ensure the generated queries span multiple query categories from the list above. - The queries should be distinct and aim to retrieve diverse content types or aspects. - Avoid overfitting to the exact same semantic zone; seek semantic diversity. - Do NOT number the queries or add any prefix like "Query 1:". - Return ONLY a Python-parseable list of strings. For example: ["synthetic query 1", "another synthetic query", "a third different query"] - Each query in the list should be a complete, self-contained search query string.
    """
    try:
        response = model.generate_content(prompt); content_text = response.text.strip()
        try:
            for pf in ["```python","```json","```"]:
                if content_text.startswith(pf): content_text=content_text.split(pf,1)[1].rsplit("```",1)[0].strip(); break
            queries = ast.literal_eval(content_text) if content_text.startswith('[') else [re.sub(r'^\s*[-\*\d\.]+\s*','',q.strip().strip('"\'')) for q in content_text.split('\n') if q.strip()]
            if not isinstance(queries,list) or not all(isinstance(qs,str) for qs in queries): raise ValueError("Not list of str.")
            return [qs for qs in queries if qs.strip()]
        except (SyntaxError,ValueError) as e:
            st.error(f"Gemini response parse error: {e}. Raw: {content_text[:300]}..."); extracted=[re.sub(r'^\s*[-\*\d\.]+\s*','',l.strip().strip('"\'')) for l in content_text.split('\n') if l.strip()]
            if extracted: st.warning("Fallback parsing used."); return extracted
            return []
    except Exception as e: st.error(f"Gemini API call error: {e}"); return []

st.title("✨ AI Mode Simulator ✨")
st.markdown("Fetch, clean, analyze web content against initial & AI-generated queries. Features advanced text extraction and weighted scoring.")

# --- Sidebar Configuration Widgets ---
st.sidebar.subheader("🤖 Embedding Model Configuration")
embedding_model_options = {
    "Local: MixedBread (Large & Powerful)": "mixedbread-ai/mxbai-embed-large-v1",
    "Local: MPNet (Quality Focus)": "all-mpnet-base-v2",
    "Local: MiniLM (Speed Focus)": "all-MiniLM-L6-v2",
    "Local: DistilRoBERTa (Balanced)": "all-distilroberta-v1",
    "OpenAI: text-embedding-3-small": "openai-text-embedding-3-small",
    "OpenAI: text-embedding-3-large": "openai-text-embedding-3-large",
    "Gemini: embedding-001": "gemini-embedding-001"
}
selected_embedding_label = st.sidebar.selectbox("Select Embedding Model:", options=list(embedding_model_options.keys()), index=0, disabled=st.session_state.processing)
st.session_state.selected_embedding_model = embedding_model_options[selected_embedding_label]
st.sidebar.subheader("📄 Text Extraction & Processing")
analysis_granularity = st.sidebar.selectbox("Analysis Granularity:", ("Passage-based (HTML Tags)", "Sentence-based"), index=0, disabled=st.session_state.processing)
use_selenium_opt = st.sidebar.checkbox("Use Selenium for fetching (for URL mode)", value=True, disabled=st.session_state.processing)

st.sidebar.divider()
st.sidebar.header("⚙️ Input & Query Configuration")
input_mode = st.sidebar.radio("Choose Input Mode:", ("Fetch from URLs", "Paste Raw Text"), disabled=st.session_state.processing)
initial_query_val = st.sidebar.text_input("Initial Search Query:", "benefits of server-side rendering", disabled=st.session_state.processing)
if input_mode == "Fetch from URLs":
    urls_text_area_val = st.sidebar.text_area("Enter URLs:", "https://cloudinary.com/guides/automatic-image-cropping/server-side-rendering-benefits-use-cases-and-best-practices\nhttps://prismic.io/blog/what-is-ssr", height=100, disabled=st.session_state.processing)
    use_trafilatura_opt = st.sidebar.checkbox("Use Trafilatura (main content)", value=True, help="Attempt to use Trafilatura for primary content extraction. If it fails, a fallback BeautifulSoup method is used.", disabled=st.session_state.processing)
    trafilatura_favor_recall = st.sidebar.checkbox("Trafilatura: Favor Recall", value=False, help="Trafilatura option to get more text, potentially at the cost of precision.", disabled=st.session_state.processing)
    run_entity_gap_analysis = st.sidebar.checkbox(
        "☑️ Run Entity Gap Analysis", value=False,
        help="Requires Google Cloud NLP to be configured. Extracts entities to find content gaps.",
        disabled=(st.session_state.processing or not st.session_state.gcp_nlp_configured)
    )
else:
    pasted_content_label = st.sidebar.text_input("Content Label:", value="Pasted Content", disabled=st.session_state.processing)
    pasted_content_text = st.sidebar.text_area("Paste content here:", height=200, disabled=st.session_state.processing)
    urls_text_area_val, use_trafilatura_opt, trafilatura_favor_recall, run_entity_gap_analysis = "", False, False, False

num_sq_val = st.sidebar.slider("Num Synthetic Queries:", 3, 50, 5, disabled=st.session_state.processing)
if analysis_granularity.startswith("Passage"):
    st.sidebar.subheader("Passage Context Settings")
    s_overlap_val = st.sidebar.slider("Context Sentence Overlap:", 0, 10, 2, help="For each core passage, include N sentences from adjacent passages for contextual similarity calculation. This overlap is NOT shown in the results display.", disabled=st.session_state.processing)
else: s_overlap_val = 0
analyze_disabled = not (st.session_state.get("gemini_api_configured", False) or st.session_state.get("openai_api_configured", False))

if st.sidebar.button("🚀 Analyze Content", type="primary", disabled=st.session_state.processing or analyze_disabled):
    st.session_state.processing = True
    st.session_state.run_entity_gap_analysis_flag = run_entity_gap_analysis
    st.session_state.use_trafilatura_opt = use_trafilatura_opt
    st.session_state.trafilatura_favor_recall = trafilatura_favor_recall
    st.rerun()

# --- Main processing block ---
if st.session_state.processing:
    try:
        jobs = []
        if not initial_query_val: st.warning("Initial Search Query is required."); st.stop()
        if input_mode == "Fetch from URLs":
            if not urls_text_area_val: st.warning("Please enter URLs."); st.stop()
            jobs = [{'type': 'url', 'identifier': url.strip()} for url in urls_text_area_val.split('\n') if url.strip()]
        else:
            st.warning("Entity Gap Analysis & Trafilatura are only available for URL mode."); st.session_state.run_entity_gap_analysis_flag = False
            if not pasted_content_text: st.warning("Please paste content."); st.stop()
            jobs.append({'type': 'paste', 'identifier': pasted_content_label or "Pasted Content", 'content': pasted_content_text})
        
        local_embedding_model_instance = None
        if not st.session_state.selected_embedding_model.startswith(("openai-", "gemini-")):
            with st.spinner(f"Loading local model '{st.session_state.selected_embedding_model}'..."):
                local_embedding_model_instance = load_local_sentence_transformer_model(st.session_state.selected_embedding_model)
            if not local_embedding_model_instance: st.stop()
        
        st.session_state.all_url_metrics_list, st.session_state.url_processed_units_dict = [], {}
        st.session_state.all_queries_for_analysis, st.session_state.analysis_done = [], False
        st.session_state.entity_analysis_results = None
        
        if use_selenium_opt and input_mode == "Fetch from URLs" and not st.session_state.selenium_driver_instance:
            with st.spinner("Initializing Selenium WebDriver..."): st.session_state.selenium_driver_instance = initialize_selenium_driver()
        
        with st.spinner("Generating synthetic queries..."): synthetic_queries = generate_synthetic_queries(initial_query_val, num_sq_val)
        local_all_queries = [f"Initial: {initial_query_val}"] + (synthetic_queries or [])
        
        with st.spinner("Embedding all queries..."): local_all_query_embs = get_embeddings(local_all_queries, local_embedding_model_instance)
        if local_all_query_embs.size == 0: st.error("Query embedding failed. Halting analysis."); st.stop()
        initial_query_embedding = local_all_query_embs[0].reshape(1, -1)
        
        local_all_metrics, local_processed_units_data = [], {}
        fetched_content = {}
        if input_mode == "Fetch from URLs":
            with st.spinner(f"Fetching content from {len(jobs)} URL(s)..."):
                for job in jobs:
                    fetched_content[job['identifier']] = fetch_content_with_selenium(job['identifier'], st.session_state.selenium_driver_instance) if use_selenium_opt else fetch_content_with_requests(job['identifier'])
            if st.session_state.selenium_driver_instance:
                st.session_state.selenium_driver_instance.quit(); st.session_state.selenium_driver_instance = None
        else: fetched_content[jobs[0]['identifier']] = jobs[0]['content']

        with st.spinner(f"Processing {len(fetched_content)} content source(s)..."):
            for identifier, content in fetched_content.items():
                st.markdown(f"--- \n#### Processing: {identifier}")
                if not content or len(content.strip()) < 20: st.warning(f"Insufficient text from {identifier}. Skipping."); continue
                is_url_source = any(job['identifier'] == identifier for job in jobs if job['type'] == 'url')
                raw_html_for_highlighting = content if is_url_source else "".join([f"<p>{clean_text_for_display(p)}</p>" for p in content.split('\n\n')])
                
                units_for_display, page_text_for_highlight = extract_structural_passages_with_full_text(
                    raw_html_for_highlighting,
                    use_trafilatura=st.session_state.get('use_trafilatura_opt', False) and is_url_source,
                    favor_recall=st.session_state.get('trafilatura_favor_recall', False)
                )

                if analysis_granularity.startswith("Sentence"):
                    units_for_embedding = split_text_into_sentences(page_text_for_highlight) if page_text_for_highlight else []
                    units_for_display = units_for_embedding
                else: # Passage-based
                    units_for_embedding = add_sentence_overlap_to_passages(units_for_display, s_overlap_val)
                    if not units_for_display and page_text_for_highlight: units_for_display = [page_text_for_highlight]

                if not units_for_embedding: st.warning(f"No processable content units found for {identifier}. Skipping."); continue
                
                unit_embeddings = get_embeddings(units_for_embedding, local_embedding_model_instance)
                local_processed_units_data[identifier] = {"units": units_for_display, "embeddings": unit_embeddings, "unit_similarities": None, "page_text_for_highlight": page_text_for_highlight, "raw_html": raw_html_for_highlighting}
                
                if unit_embeddings.size > 0:
                    unit_sims_to_initial = cosine_similarity(unit_embeddings, initial_query_embedding).flatten()
                    weights = np.maximum(0, unit_sims_to_initial)
                    weighted_overall_emb = np.average(unit_embeddings, axis=0, weights=weights if np.sum(weights) > 1e-6 else None).reshape(1, -1)
                    overall_sims = cosine_similarity(weighted_overall_emb, local_all_query_embs)[0]
                    unit_q_sims = cosine_similarity(unit_embeddings, local_all_query_embs)
                    local_processed_units_data[identifier]["unit_similarities"] = unit_q_sims
                    for sq_idx, query_text in enumerate(local_all_queries):
                        max_sim_passage_text = ""
                        if unit_q_sims[:, sq_idx].size > 0: max_sim_idx = np.argmax(unit_q_sims[:, sq_idx]); max_sim_passage_text = units_for_display[max_sim_idx]
                        local_all_metrics.append({ "URL": identifier, "Query": query_text, "Overall Similarity (Weighted)": overall_sims[sq_idx], "Max Unit Sim.": np.max(unit_q_sims[:, sq_idx]) if unit_q_sims[:, sq_idx].size > 0 else 0.0, "Avg. Unit Sim.": np.mean(unit_q_sims[:, sq_idx]) if unit_q_sims[:, sq_idx].size > 0 else 0.0, "Num Units": len(units_for_display), "Max Similarity Passage": max_sim_passage_text })
        
        if local_all_metrics:
            st.session_state.all_url_metrics_list, st.session_state.url_processed_units_dict = local_all_metrics, local_processed_units_data
            st.session_state.all_queries_for_analysis, st.session_state.analysis_done = local_all_queries, True
            st.session_state.last_analysis_granularity = analysis_granularity

        if st.session_state.get('run_entity_gap_analysis_flag'):
            if not st.session_state.gcp_nlp_configured: st.warning("Entity Gap Analysis was skipped because Google Cloud NLP is not configured.")
            elif input_mode != "Fetch from URLs" or len(jobs) < 2: st.warning("Entity Gap Analysis requires at least two URLs to compare.")
            else:
                entity_results = {}
                with st.spinner("Performing Entity Analysis on all URLs..."):
                    for identifier, data in st.session_state.url_processed_units_dict.items():
                        full_text = data.get("page_text_for_highlight", "")
                        if full_text:
                            entities = extract_entities_with_google_nlp(full_text, st.session_state.gcp_credentials_info)
                            entity_results[identifier] = entities
                st.session_state.entity_analysis_results = entity_results
    finally:
        st.session_state.processing = False
        st.rerun()

# --- Results Display Section ---
if st.session_state.get("analysis_done") and st.session_state.all_url_metrics_list:
    st.subheader("Analysed Queries (Initial + Synthetic)"); st.expander("View All Analysed Queries").json([q.replace("Initial: ", "(Initial) ") for q in st.session_state.all_queries_for_analysis])
    unit_label = "Sentence" if st.session_state.last_analysis_granularity.startswith("Sentence") else "Passage"
    st.markdown("---"); st.subheader(f"📈 Overall Similarity & {unit_label} Metrics Summary")
    df_summary = pd.DataFrame(st.session_state.all_url_metrics_list); df_display = df_summary.rename(columns={"URL": "Source / URL", "Max Unit Sim.": f"Max {unit_label} Sim.", "Avg. Unit Sim.": f"Avg. {unit_label} Sim.", "Num Units": f"Num {unit_label}s"})
    st.dataframe(df_display, use_container_width=True)
    st.markdown("---"); st.subheader("📊 Visual: Overall Similarity to All Queries (Weighted)")
    fig_bar = px.bar(df_display, x="Query", y="Overall Similarity (Weighted)", color="Source / URL", barmode="group", title="Overall Content Similarity to Queries", height=max(600, 80 * len(st.session_state.all_queries_for_analysis)))
    fig_bar.update_yaxes(range=[0,1]); st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("---"); st.subheader(f"🔥 {unit_label} Heatmaps vs. All Queries")
    for item_idx, (identifier, p_data) in enumerate(st.session_state.url_processed_units_dict.items()):
        with st.expander(f"Heatmap & Details for: {identifier}", expanded=(item_idx==0)):
            if p_data.get("unit_similarities") is None: st.write(f"No {unit_label.lower()} similarity data."); continue
            unit_sims, units, all_queries = p_data["unit_similarities"], p_data["units"], st.session_state.all_queries_for_analysis
            short_queries = [q.replace("Initial: ", "(I) ")[:50] + ('...' if len(q) > 50 else '') for q in all_queries]
            unit_labels = [f"{unit_label[0]}{i+1}" for i in range(len(units))]
            wrapped_units = [textwrap.fill(unit, width=100, replace_whitespace=False).replace('\n', '<br>') for unit in units]
            hover_text = [[f"<b>{unit_labels[i]}</b> vs Q:'{all_queries[j][:45]}...'<br>Similarity:{unit_sims[i, j]:.3f}<hr><b>Text:</b><br>{wrapped_units[i]}" for i in range(unit_sims.shape[0])] for j in range(unit_sims.shape[1])]
            fig_heat = go.Figure(data=go.Heatmap(z=unit_sims.T, x=unit_labels, y=short_queries, colorscale='Viridis', zmin=0, zmax=1, text=hover_text, hoverinfo='text'))
            fig_heat.update_layout(title=f"{unit_label} Similarity for {identifier}", height=max(400, 25 * len(short_queries) + 100), yaxis_autorange='reversed', xaxis_title=f"{unit_label}s", yaxis_title="Queries")
            st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown("---")
            key_base = f"{item_idx}_{identifier.replace('/', '_').replace(' ', '_')}"
            selected_query = st.selectbox(f"Select Query for Details:", options=st.session_state.all_queries_for_analysis, key=f"q_sel_{key_base}")
            if selected_query:
                query_idx = st.session_state.all_queries_for_analysis.index(selected_query); scored_units = sorted(zip(p_data["units"], unit_sims[:, query_idx]), key=lambda item: item[1], reverse=True)
                query_display_name = selected_query.replace("Initial: ", "(I) ")[:30] + "..."
                if st.checkbox(f"Show highlighted text for '{query_display_name}'?", key=f"cb_hl_{key_base}_{query_idx}"):
                    with st.spinner("Highlighting..."):
                        unit_scores_for_query = {unit_text: score for unit_text, score in scored_units}
                        if st.session_state.last_analysis_granularity.startswith("Passage"): highlighted_html = render_safe_highlighted_html(p_data.get("raw_html"), unit_scores_for_query)
                        else: highlighted_html = get_sentence_highlighted_html_flat(p_data["page_text_for_highlight"], unit_scores_for_query)
                        st.markdown(highlighted_html, unsafe_allow_html=True)
                if st.checkbox(f"Show top/bottom {unit_label.lower()}s for '{query_display_name}'?", key=f"cb_tb_{key_base}_{query_idx}"):
                    n_val = st.slider("N:", 1, 10, 3, key=f"sl_tb_{key_base}_{query_idx}")
                    st.markdown(f"**Top {n_val} {unit_label}s:**")
                    for u_t, u_s in scored_units[:n_val]:
                        st.markdown(f"**Score: {u_s:.3f}**")
                        st.markdown(f"> {u_t}")
                        st.divider()

                    st.markdown(f"**Bottom {n_val} {unit_label}s:**")
                    for u_t, u_s in scored_units[-n_val:]:
                        st.markdown(f"**Score: {u_s:.3f}**")
                        st.markdown(f"> {u_t}")
                        st.divider()

if st.session_state.get("entity_analysis_results"):
    st.markdown("---")
    with st.expander("🔎 Entity Gap Analysis", expanded=True):
        results = st.session_state.entity_analysis_results
        url_options = list(results.keys())
        if len(url_options) < 2: st.info("Entity Gap Analysis requires at least two URLs to compare.")
        else:
            primary_url = st.selectbox("Select your Primary URL to check for missing entities:", options=url_options, index=0, key="primary_url_selector")
            if primary_url:
                primary_entities = set(results[primary_url].keys()); st.write(f"Found **{len(primary_entities)}** unique entities on `{primary_url}`.")
                competitor_entities_map = {}
                for url, entities_data in results.items():
                    if url != primary_url:
                        for entity_key, entity_info in entities_data.items():
                            if entity_key not in competitor_entities_map: competitor_entities_map[entity_key] = {'info': entity_info, 'found_on': []}
                            competitor_entities_map[entity_key]['found_on'].append(url)
                missing_entities = [{'Entity': data['info']['name'], 'Type': data['info']['type'], 'Salience': data['info']['salience'], 'Found On (Competitors)': ", ".join([f"`{os.path.basename(u)}`" for u in data['found_on']])} for entity_key, data in competitor_entities_map.items() if entity_key not in primary_entities]
                if not missing_entities: st.success(f"✅ **No Gaps Found!** Your primary URL covers all entities found on the competitor pages.")
                else:
                    st.subheader(f"❗️ Found {len(missing_entities)} entities present in other URLs but MISSING from your page:")
                    df_missing = pd.DataFrame(missing_entities).sort_values(by='Salience', ascending=False).reset_index(drop=True)
                    st.dataframe(df_missing, use_container_width=True, column_config={"Salience": st.column_config.ProgressColumn("Salience (Importance)", format="%.3f", min_value=0, max_value=1)})

st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v5.27 | Core Functionality Restored")
