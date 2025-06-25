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
if "selected_embedding_model" not in st.session_state: st.session_state.selected_embedding_model = 'all-mpnet-base-v2'

REQUEST_INTERVAL = 3.0
last_request_time = 0
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
    openai_api_key_input = st.text_input("Enter OpenAI API Key:", type="password", value=st.session_state.get("openai_api_key_to_persist", ""))
    if st.button("Set & Verify OpenAI Key"):
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
    gemini_api_key_input = st.text_input("Enter Google Gemini API Key:", type="password", value=st.session_state.get("gemini_api_key_to_persist", ""))
    if st.button("Set & Verify Gemini Key"):
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

st.sidebar.markdown("---")
if st.session_state.get("openai_api_configured"): st.sidebar.markdown("✅ OpenAI API: **Configured**")
else: st.sidebar.markdown("⚠️ OpenAI API: **Not Configured**")
if st.session_state.get("gemini_api_configured"): st.sidebar.markdown("✅ Gemini API: **Configured**")
else: st.sidebar.markdown("⚠️ Gemini API: **Not Configured**")
if st.session_state.get("openai_api_key_to_persist") and not st.session_state.get("openai_client"):
    st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key_to_persist)
if st.session_state.get("gemini_api_key_to_persist"):
    try: genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
    except Exception: st.session_state.gemini_api_configured = False


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
    options = ChromeOptions(); options.add_argument("--headless"); options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage"); options.add_argument("--disable-gpu"); options.add_argument(f"user-agent={get_random_user_agent()}")
    try: return webdriver.Chrome(service=ChromeService(), options=options)
    except Exception as e: st.error(f"Selenium init failed: {e}"); return None
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

# --- UPDATED: Text Cleaning Function ---
def clean_text_for_display(text):
    """Collapses whitespace and ensures space after punctuation and between camelCase-like words."""
    if not text: return ""
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = re.sub(r'([.?!])([a-zA-Z])', r'\1 \2', cleaned_text)
    # Add a space between a lowercase letter and an uppercase letter (camelCase helper)
    cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)
    return cleaned_text

def split_text_into_sentences(text):
    if not text: return []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]

def extract_structural_passages_with_full_text(html_content):
    if not html_content: return [], ""
    soup = BeautifulSoup(html_content, 'html.parser')
    for el in soup(["script", "style", "noscript", "iframe", "link", "meta", 'nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption', 'menu', 'banner', 'dialog']):
        if el.name: el.decompose()
    selectors = ["[class*='menu']", "[id*='nav']", "[class*='header']", "[id*='footer']", "[class*='sidebar']", "[class*='cookie']", "[class*='consent']", "[class*='popup']", "[class*='modal']", "[class*='social']", "[class*='share']", "[class*='advert']", "[id*='ad']", "[aria-hidden='true']"]
    for sel in selectors:
        try:
            for element in soup.select(sel):
                if not any(p.name in ['main', 'article', 'body'] for p in element.parents): element.decompose()
        except Exception: pass
    target_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table']
    content_elements = soup.find_all(target_tags)
    merged_passages = []
    i = 0
    while i < len(content_elements):
        current_element = content_elements[i]
        # --- KEY CHANGE: Use separator=' ' for better text joining ---
        current_text = current_element.get_text(separator=' ', strip=True)
        if current_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and (i + 1) < len(content_elements):
            next_element = content_elements[i+1]
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
        current_passage = structural_passages[i]
        prefix = _get_n_sentences(structural_passages[i-1], overlap_count, from_start=False) if i > 0 else ""
        suffix = _get_n_sentences(structural_passages[i+1], overlap_count, from_start=True) if i < num_passages - 1 else ""
        expanded_passages.append(" ".join(filter(None, [prefix, current_passage, suffix])))
    return [p for p in expanded_passages if p.strip()]

def get_passage_highlighted_html(html_content, unit_scores_map):
    if not html_content or not unit_scores_map: return "<p>Could not generate highlighted HTML.</p>"
    soup = BeautifulSoup(html_content, 'html.parser')
    for el in soup(["script", "style", "noscript", "iframe", "link", "meta", 'nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption', 'menu', 'banner', 'dialog']):
        if el.name: el.decompose()
    target_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table']
    for element in soup.find_all(target_tags):
        element_text = clean_text_for_display(element.get_text(separator=' '))
        if not element_text: continue
        passage_score = 0.5
        for unit_text, score in unit_scores_map.items():
            if element_text in unit_text:
                passage_score = score
                break
        color = "green" if passage_score >= 0.75 else "red" if passage_score < 0.35 else "inherit"
        element['style'] = f"color:{color}; border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 12px; margin-top: 12px;"
    return str(soup)

def get_sentence_highlighted_html_flat(page_text_content, unit_scores_map):
    if not page_text_content or not unit_scores_map: return "<p>No content to highlight.</p>"
    sentences = split_text_into_sentences(page_text_content)
    if not sentences: return "<p>No sentences to highlight.</p>"
    highlighted_html = ""
    for sentence in sentences:
        sentence_score = 0.5
        cleaned_sentence = clean_text_for_display(sentence)
        if cleaned_sentence in unit_scores_map:
            sentence_score = unit_scores_map[cleaned_sentence]
        color = "green" if sentence_score >= 0.75 else "red" if sentence_score < 0.35 else "black"
        highlighted_html += f'<p style="color:{color}; margin-bottom: 2px;">{cleaned_sentence}</p>'
    return highlighted_html
    
def generate_synthetic_queries(user_query, num_queries=7):
    if not st.session_state.get("gemini_api_configured", False): st.error("Gemini API not configured."); return []
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
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
st.sidebar.subheader("🤖 Embedding Model Configuration")
embedding_model_options = { "Local: MPNet (Quality Focus)": "all-mpnet-base-v2", "Local: MiniLM (Speed Focus)": "all-MiniLM-L6-v2", "Local: DistilRoBERTa (Balanced)": "all-distilroberta-v1", "OpenAI: text-embedding-3-small": "openai-text-embedding-3-small", "OpenAI: text-embedding-3-large": "openai-text-embedding-3-large", "Gemini: embedding-001": "gemini-embedding-001"}
selected_embedding_label = st.sidebar.selectbox("Select Embedding Model:", options=list(embedding_model_options.keys()), index=0)
st.session_state.selected_embedding_model = embedding_model_options[selected_embedding_label]
st.sidebar.subheader("📄 Text Extraction & Processing")
analysis_granularity = st.sidebar.selectbox("Analysis Granularity:", ("Passage-based (HTML Tags)", "Sentence-based"), index=0)
use_selenium_opt = st.sidebar.checkbox("Use Selenium for fetching (for URL mode)", value=True)
st.sidebar.divider()
st.sidebar.header("⚙️ Input & Query Configuration")
input_mode = st.sidebar.radio("Choose Input Mode:", ("Fetch from URLs", "Paste Raw Text"))
initial_query_val = st.sidebar.text_input("Initial Search Query:", "benefits of server-side rendering")
if input_mode == "Fetch from URLs":
    urls_text_area_val = st.sidebar.text_area("Enter URLs:", "https://vercel.com/blog/understanding-rendering-in-react\nhttps://www.patterns.dev/posts/rendering-patterns/", height=100)
    use_trafilatura_opt = st.sidebar.checkbox("Use Trafilatura (main content)", value=True, help="Attempt to use Trafilatura for primary content extraction. If it fails, a fallback BeautifulSoup method is used.")
    st.session_state.trafilatura_favor_recall = st.sidebar.checkbox("Trafilatura: Favor Recall", value=False, help="Trafilatura option to get more text, potentially at the cost of precision.")
else:
    pasted_content_label = st.sidebar.text_input("Content Label:", value="Pasted Content")
    pasted_content_text = st.sidebar.text_area("Paste content here:", height=200)
    urls_text_area_val, use_trafilatura_opt = "", False
num_sq_val = st.sidebar.slider("Num Synthetic Queries:", 3, 50, 5)
if analysis_granularity.startswith("Passage"):
    st.sidebar.subheader("Passage Context Settings")
    s_overlap_val = st.sidebar.slider("Context Sentence Overlap:", 0, 10, 2, help="For each core passage, include N sentences from adjacent passages for contextual similarity calculation. This overlap is NOT shown in the results display.")
else: s_overlap_val = 0
analyze_disabled = not st.session_state.get("gemini_api_configured", False) and not st.session_state.get("openai_api_configured")

if st.sidebar.button("🚀 Analyze Content", type="primary", disabled=analyze_disabled):
    jobs = []
    if not initial_query_val: st.warning("Initial Search Query is required."); st.stop()
    if input_mode == "Fetch from URLs":
        if not urls_text_area_val: st.warning("Please enter URLs."); st.stop()
        jobs = [{'type': 'url', 'identifier': url.strip()} for url in urls_text_area_val.split('\n') if url.strip()]
    else:
        if not pasted_content_text: st.warning("Please paste content."); st.stop()
        jobs.append({'type': 'paste', 'identifier': pasted_content_label or "Pasted Content", 'content': pasted_content_text})
    local_embedding_model_instance = None
    if not st.session_state.selected_embedding_model.startswith(("openai-", "gemini-")):
        with st.spinner(f"Loading local model..."): local_embedding_model_instance = load_local_sentence_transformer_model(st.session_state.selected_embedding_model)
        if not local_embedding_model_instance: st.stop()
    st.session_state.all_url_metrics_list, st.session_state.url_processed_units_dict = [], {}
    st.session_state.all_queries_for_analysis, st.session_state.analysis_done = [], False
    if use_selenium_opt and input_mode == "Fetch from URLs" and not st.session_state.selenium_driver_instance:
        with st.spinner("Initializing Selenium..."): st.session_state.selenium_driver_instance = initialize_selenium_driver()
    with st.spinner("Generating synthetic queries..."): synthetic_queries = generate_synthetic_queries(initial_query_val, num_sq_val)
    local_all_queries = [f"Initial: {initial_query_val}"] + (synthetic_queries or [])
    with st.spinner("Embedding all queries..."): local_all_query_embs = get_embeddings(local_all_queries, local_embedding_model_instance)
    initial_query_embedding = local_all_query_embs[0]
    local_all_metrics, local_processed_units_data = [], {}
    with st.spinner(f"Processing {len(jobs)} content source(s)..."):
        for i, job in enumerate(jobs):
            identifier = job['identifier']; st.markdown(f"--- \n#### Processing: {identifier}")
            raw_html_for_highlighting = None
            content = job.get('content') or (fetch_content_with_selenium(identifier, st.session_state.selenium_driver_instance) if use_selenium_opt else fetch_content_with_requests(identifier))
            if not content or len(content.strip()) < 20: st.warning(f"Insufficient text from {identifier}. Skipping."); continue
            if job['type'] == 'url': raw_html_for_highlighting = content
            else: paragraphs = content.split('\n\n'); raw_html_for_highlighting = "".join([f"<p>{clean_text_for_display(p)}</p>" for p in paragraphs])
            if analysis_granularity.startswith("Sentence"):
                page_text_for_highlight = clean_text_for_display(parse_and_clean_html(raw_html_for_highlighting, "", False))
                units_for_display = split_text_into_sentences(page_text_for_highlight) if page_text_for_highlight else []
                units_for_embedding = units_for_display
            else:
                units_for_display, page_text_for_highlight = extract_structural_passages_with_full_text(raw_html_for_highlighting)
                units_for_embedding = add_sentence_overlap_to_passages(units_for_display, s_overlap_val)
                if not units_for_display and page_text_for_highlight: units_for_display = [page_text_for_highlight]
            if not units_for_embedding: st.warning(f"No processable content units found for {identifier}. Skipping."); continue
            unit_embeddings = get_embeddings(units_for_embedding, local_embedding_model_instance)
            local_processed_units_data[identifier] = {"units": units_for_display, "embeddings": unit_embeddings, "unit_similarities": None, "page_text_for_highlight": page_text_for_highlight, "raw_html": raw_html_for_highlighting}
            if unit_embeddings.size > 0:
                unit_sims_to_initial = cosine_similarity(unit_embeddings, initial_query_embedding.reshape(1, -1)).flatten()
                weights = np.maximum(0, unit_sims_to_initial)
                weighted_overall_emb = np.average(unit_embeddings, axis=0, weights=weights if np.sum(weights) > 1e-6 else None).reshape(1, -1)
                overall_sims = cosine_similarity(weighted_overall_emb, local_all_query_embs)[0]
                unit_q_sims = cosine_similarity(unit_embeddings, local_all_query_embs)
                local_processed_units_data[identifier]["unit_similarities"] = unit_q_sims
                for sq_idx, query_text in enumerate(local_all_queries):
                    current_q_unit_sims = unit_q_sims[:, sq_idx]; max_sim_passage_text = ""
                    if current_q_unit_sims.size > 0: max_sim_idx = np.argmax(current_q_unit_sims); max_sim_passage_text = units_for_display[max_sim_idx]
                    local_all_metrics.append({ "URL": identifier, "Query": query_text, "Overall Similarity (Weighted)": overall_sims[sq_idx], "Max Unit Sim.": np.max(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0, "Avg. Unit Sim.": np.mean(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0, "Num Units": len(units_for_display), "Max Similarity Passage": max_sim_passage_text })
    if local_all_metrics:
        st.session_state.all_url_metrics_list = local_all_metrics; st.session_state.url_processed_units_dict = local_processed_units_data
        st.session_state.all_queries_for_analysis = local_all_queries; st.session_state.analysis_done = True; st.session_state.last_analysis_granularity = analysis_granularity

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
            unit_sims, units = p_data["unit_similarities"], p_data["units"]; all_queries = st.session_state.all_queries_for_analysis
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
                        highlighted_html = ""
                        if st.session_state.last_analysis_granularity.startswith("Passage"):
                            highlighted_html = get_passage_highlighted_html(p_data.get("raw_html"), unit_scores_for_query)
                        else:
                            highlighted_html = get_sentence_highlighted_html_flat(p_data["page_text_for_highlight"], unit_scores_for_query)
                        st.markdown(highlighted_html, unsafe_allow_html=True)
                if st.checkbox(f"Show top/bottom {unit_label.lower()}s for '{query_display_name}'?", key=f"cb_tb_{key_base}_{query_idx}"):
                    n_val = st.slider("N:", 1, 10, 3, key=f"sl_tb_{key_base}_{query_idx}")
                    st.markdown(f"**Top {n_val} {unit_label}s:**")
                    for u_t, u_s in scored_units[:n_val]:
                        st.markdown(f"**Score: {u_s:.3f}**"); st.markdown(f"> {u_t}"); st.divider()
                    st.markdown(f"**Bottom {n_val} {unit_label}s:**")
                    for u_t, u_s in scored_units[-n_val:]:
                        st.markdown(f"**Score: {u_s:.3f}**"); st.markdown(f"> {u_t}"); st.divider()
st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v5.15 | Final Text Cleaning")
