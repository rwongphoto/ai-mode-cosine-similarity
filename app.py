import streamlit as st
import requests
# from bs4 import BeautifulSoup # trafilatura might handle most of this
from sentence_transformers import SentenceTransformer, CrossEncoder # Added CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import re
import ast
import time
import random
import os

# --- NEW: Trafilatura for advanced text extraction ---
import trafilatura

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Semantic Analyzer")

# --- Session State Initialization ---
if "all_url_metrics_list" not in st.session_state: st.session_state.all_url_metrics_list = None
if "url_processed_units_dict" not in st.session_state: st.session_state.url_processed_units_dict = None
if "all_queries_for_analysis" not in st.session_state: st.session_state.all_queries_for_analysis = None
if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
if "gemini_api_key_to_persist" not in st.session_state: st.session_state.gemini_api_key_to_persist = ""
if "gemini_api_configured" not in st.session_state: st.session_state.gemini_api_configured = False
if "selenium_driver_instance" not in st.session_state: st.session_state.selenium_driver_instance = None


# --- Web Fetching Enhancements ---
REQUEST_INTERVAL = 3.0
last_request_time = 0
USER_AGENTS = [ # Your full list
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

# --- Models & Driver Setup ---
@st.cache_resource
def load_embedding_models():
    st.write("Loading embedding models...") # For feedback as this can take time
    bi_encoder = SentenceTransformer(st.session_state.get("selected_bi_encoder_model", 'all-mpnet-base-v2'))
    # Conditionally load cross-encoder if the option is selected to use it later
    cross_encoder_model_name = st.session_state.get("selected_cross_encoder_model")
    cross_encoder = None
    if cross_encoder_model_name:
        try:
            cross_encoder = CrossEncoder(cross_encoder_model_name)
        except Exception as e:
            st.warning(f"Could not load CrossEncoder model '{cross_encoder_model_name}': {e}. Re-ranking will be disabled.")
    return bi_encoder, cross_encoder

# Initialize models based on potential selection later
# These will be loaded when `load_embedding_models` is called after UI setup for model selection
# embedding_model, cross_encoder_model = None, None # Will be set by load_embedding_models

def initialize_selenium_driver():
    options = ChromeOptions()
    options.add_argument("--headless"); options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage"); options.add_argument("--disable-gpu")
    options.add_argument(f"user-agent={get_random_user_agent()}")
    try: return webdriver.Chrome(service=ChromeService(), options=options)
    except Exception as e: st.error(f"Selenium init failed: {e}"); return None

# --- API Key Config ---
st.sidebar.header("🔑 Gemini API Configuration")
api_key_input = st.sidebar.text_input("Enter Google Gemini API Key:", type="password", value=st.session_state.gemini_api_key_to_persist)
if st.sidebar.button("Set & Verify API Key"):
    if api_key_input:
        try:
            genai.configure(api_key=api_key_input)
            if not [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]:
                raise Exception("No usable models found.")
            st.session_state.gemini_api_key_to_persist = api_key_input
            st.session_state.gemini_api_configured = True
            st.sidebar.success("Gemini API Key Configured!")
        except Exception as e:
            st.session_state.gemini_api_key_to_persist = ""
            st.session_state.gemini_api_configured = False
            st.sidebar.error(f"API Key Failed: {str(e)[:200]}")
    else: st.sidebar.warning("Please enter API Key.")

if st.session_state.get("gemini_api_configured"):
    st.sidebar.markdown("✅ Gemini API: **Configured**")
    if st.session_state.gemini_api_key_to_persist:
        try: genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
        except Exception: st.session_state.gemini_api_configured = False
else: st.sidebar.markdown("⚠️ Gemini API: **Not Configured**")


# --- Helper Functions ---
def fetch_content_with_selenium(url, driver_instance):
    if not driver_instance: st.warning(f"Selenium N/A for {url}. Fallback."); return fetch_content_with_requests(url)
    enforce_rate_limit(); driver_instance.get(url); time.sleep(5); return driver_instance.page_source

def fetch_content_with_requests(url):
    enforce_rate_limit(); headers={'User-Agent':get_random_user_agent()}; resp=requests.get(url,timeout=20,headers=headers); resp.raise_for_status(); return resp.text

# --- UPDATED: parse_and_clean_html to use trafilatura ---
def parse_and_clean_html(html_content, url, use_trafilatura=True):
    if not html_content: return None
    text_content = None
    if use_trafilatura:
        # Favor_recall=True tries to get more text, False is more precise (less boilerplate)
        # include_comments=False, include_tables=False are common settings
        # You might want to expose these trafilatura settings as advanced options to the user
        downloaded = trafilatura.fetch_url(url) # Trafilatura can also fetch, but we provide HTML
        if downloaded: # If trafilatura fetched something (it might if html_content was poor)
             text_content = trafilatura.extract(downloaded, include_comments=False, include_tables=False, favor_recall=st.session_state.get("trafilatura_favor_recall", False))
        if not text_content and html_content: # Fallback to extracting from provided html if fetch_url failed or returned nothing
            text_content = trafilatura.extract(html_content, include_comments=False, include_tables=False, favor_recall=st.session_state.get("trafilatura_favor_recall", False))
    
    if not text_content: # Fallback to BeautifulSoup if trafilatura fails or returns nothing
        st.warning(f"Trafilatura failed for {url} or returned no content. Falling back to BeautifulSoup cleaning.")
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for el in soup(["script","style","noscript","iframe","link","meta",'nav','header','footer','aside','form','figure','figcaption','menu','banner','dialog']): el.decompose()
            selectors = ["[class*='menu']","[id*='nav']","[class*='header']","[id*='footer']","[class*='sidebar']","[class*='cookie']","[class*='consent']","[class*='popup']","[class*='modal']","[class*='social']","[class*='share']","[class*='advert']","[id*='ad']","[aria-hidden='true']"]
            for sel in selectors:
                try:
                    for element in soup.select(sel):
                        is_main = element.name in ['body','main','article'] or any(c in element.get('class',[]) for c in ['content','main-content','article-body'])
                        if not is_main or element.name not in ['body','main','article']:
                            if element.parent: element.decompose()
                except: pass
            text_content = soup.get_text(separator=' ', strip=True)
        except Exception as e_bs4:
            st.error(f"BeautifulSoup fallback parsing error ({url}): {e_bs4}")
            return None

    if text_content:
        text_content = re.sub(r'\s+',' ',text_content); text_content = re.sub(r'\.{3,}','.',text_content); text_content = re.sub(r'( \.){2,}','.',text_content)
        return text_content.strip() if text_content.strip() else None
    return None


def split_text_into_sentences(text):
    if not text: return []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]

def split_text_into_passages(text, s_per_p=7, s_overlap=2):
    if not text: return []
    sentences = split_text_into_sentences(text) 
    if not sentences: return []
    passages, step = [], max(1, s_per_p - s_overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + s_per_p])
        if chunk.strip() and len(chunk.split()) > 10: passages.append(chunk)
    return [p for p in passages if p.strip()]

def get_bi_embeddings(_texts, bi_encoder_model): # Now takes the model as arg
    if not _texts or bi_encoder_model is None: return np.array([])
    return bi_encoder_model.encode(_texts)

# --- NEW: Cross-Encoder Scoring ---
def get_cross_encoder_scores(query_passage_pairs, cross_encoder_model_instance):
    if not query_passage_pairs or cross_encoder_model_instance is None:
        return []
    return cross_encoder_model_instance.predict(query_passage_pairs, show_progress_bar=False)


def get_ranked_sentences_for_display(page_text_content, query_text, bi_encoder, top_n=5): # Pass bi_encoder
    if not page_text_content or not query_text: return [], []
    sentences = split_text_into_sentences(page_text_content)
    if not sentences: return [], []
    sentence_embeddings = get_bi_embeddings(sentences, bi_encoder)
    query_embedding = get_bi_embeddings([query_text], bi_encoder)[0]
    if sentence_embeddings.size == 0 or query_embedding.size == 0: return [], []
    similarities = cosine_similarity(sentence_embeddings, query_embedding.reshape(1, -1)).flatten()
    sentence_scores = sorted(list(zip(sentences, similarities)), key=lambda item: item[1], reverse=True)
    return sentence_scores[:top_n], sentence_scores[-top_n:]

def get_highlighted_sentence_html(page_text_content, query_text, bi_encoder): # Pass bi_encoder
    if not page_text_content or not query_text: return ""
    sentences = split_text_into_sentences(page_text_content)
    if not sentences: return "<p>No sentences to highlight.</p>"
    sentence_embeddings = get_bi_embeddings(sentences, bi_encoder)
    query_embedding = get_bi_embeddings([query_text], bi_encoder)[0]
    if sentence_embeddings.size == 0 or query_embedding.size == 0: return "<p>Could not generate embeddings.</p>"
    similarities = cosine_similarity(sentence_embeddings, query_embedding.reshape(1, -1)).flatten()
    highlighted_html = ""
    if not similarities.size: return "<p>No similarity scores.</p>"
    min_sim, max_sim = np.min(similarities), np.max(similarities)
    for sentence, sim in zip(sentences, similarities):
        norm_sim = (sim - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 0.5 
        color = "green" if norm_sim >= 0.65 else "red" if norm_sim < 0.35 else "black"
        highlighted_html += f'<p style="color:{color}; margin-bottom: 2px;">{sentence}</p>'
    return highlighted_html

def generate_synthetic_queries(user_query, num_queries=7): # Keep your full fan-out prompt
    # ... (Your full, detailed fan-out prompt and logic - ensure it's the correct one) ...
    if not st.session_state.get("gemini_api_configured", False): st.error("Gemini API not configured."); return []
    model_name = "gemini-2.5-flash-preview-05-20"
    try: model = genai.GenerativeModel(model_name)
    except Exception as e: st.error(f"Gemini model init error ({model_name}): {e}"); return []
    prompt = f"""
    Based on the user's initial search query: "{user_query}"
    Generate {num_queries} diverse synthetic search queries using the "Query Fan Out" technique.
    These queries should explore different facets, intents, or related concepts.
    Aim for a mix of the following query types, ensuring variety:
    1.  Related Queries: Queries on closely associated topics or entities.
    2.  Implicit Queries: Queries that unearth underlying assumptions or unstated needs related to the original query.
    3.  Comparative Queries: Queries that seek to compare aspects of the original query's subject with alternatives or other facets.
    4.  Recent Queries (Hypothetical): If this were part of an ongoing search session, what related queries might have come before or could follow? (Focus on logical next steps).
    5.  Personalized Queries (Hypothetical): Example queries that reflect how the original query might be personalized based on hypothetical user contexts, preferences, or history (e.g., "best [topic] for families with young children," or "[topic] near me now").
    6.  Reformulation Queries: Different ways of phrasing the original query to capture the same or slightly nuanced intent.
    7.  Entity-Expanded Queries: Queries that broaden the scope by including related entities or exploring the original entity in more contexts (e.g., if original is "Eiffel Tower", expand to "history of Eiffel Tower construction", "restaurants near Eiffel Tower", "Eiffel Tower controversy").

    CRITICAL INSTRUCTIONS:
    - Ensure the generated queries span multiple query categories from the list above.
    - The queries should be distinct and aim to retrieve diverse content types or aspects.
    - Avoid overfitting to the exact same semantic zone; seek semantic diversity.
    - Do NOT number the queries or add any prefix like "Query 1:".
    - Return ONLY a Python-parseable list of strings. For example:
      ["synthetic query 1", "another synthetic query", "a third different query"]
    - Each query in the list should be a complete, self-contained search query string.
    """
    try:
        response = model.generate_content(prompt)
        content_text = "".join(p.text for p in response.parts if hasattr(p,'text')) if hasattr(response,'parts') and response.parts else response.text
        content_text = content_text.strip()
        try:
            for pf in ["```python","```json","```"]:
                if content_text.startswith(pf): content_text=content_text.split(pf,1)[1].rsplit("```",1)[0].strip(); break
            queries = ast.literal_eval(content_text) if content_text.startswith('[') else \
                      [re.sub(r'^\s*[-\*\d\.]+\s*','',q.strip().strip('"\'')) for q in content_text.split('\n') if q.strip() and len(q.strip())>3]
            if not isinstance(queries,list) or not all(isinstance(qs,str) for qs in queries): raise ValueError("Not list of str.")
            return [qs for qs in queries if qs.strip()]
        except (SyntaxError,ValueError) as e:
            st.error(f"Gemini response parse error: {e}. Raw: {content_text[:300]}...")
            extracted=[re.sub(r'^\s*[-\*\d\.]+\s*','',l.strip().strip('"\'')) for l in content_text.split('\n') if l.strip() and len(l.strip())>3]
            if extracted: st.warning("Fallback parsing used."); return extracted
            return []
    except Exception as e: st.error(f"Gemini API call error: {e}"); return []


# --- Main UI ---
st.title("✨ AI Semantic Search Analyzer ✨") # New Title
st.markdown("Fetch, clean, analyze web content against initial & AI-generated queries. Features advanced text extraction, weighted scoring, and optional cross-encoder re-ranking.")

# --- NEW: Model Selection & Advanced Options in Sidebar ---
st.sidebar.subheader("🤖 Embedding Model Configuration")
# Bi-Encoder (for initial retrieval and overall embeddings)
bi_encoder_options = {
    "MPNet (Quality Focus)": "all-mpnet-base-v2",
    "MiniLM (Speed Focus)": "all-MiniLM-L6-v2",
    "DistilRoBERTa (Balanced)": "all-distilroberta-v1"
}
selected_bi_encoder_label = st.sidebar.selectbox(
    "Select Bi-Encoder Model:",
    options=list(bi_encoder_options.keys()),
    index=0, # Default to MPNet
    help="Model for generating initial embeddings of passages/sentences and queries."
)
st.session_state.selected_bi_encoder_model = bi_encoder_options[selected_bi_encoder_label]

# Cross-Encoder (for re-ranking, optional)
use_cross_encoder_rerank = st.sidebar.checkbox("Enable Cross-Encoder Re-ranking (More Accurate, Slower)", value=False)
cross_encoder_options = {
    "MS MARCO MiniLM L6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "MS MARCO MiniLM L12": "cross-encoder/ms-marco-MiniLM-L-12-v2", # Larger, more accurate
    "TinyBERT (General)": "cross-encoder/stsb-tinybert-L-4" # For general semantic similarity
}
if use_cross_encoder_rerank:
    selected_cross_encoder_label = st.sidebar.selectbox(
        "Select Cross-Encoder Model:",
        options=list(cross_encoder_options.keys()),
        index=0,
        help="Model for re-ranking top results. Applied if checkbox is enabled."
    )
    st.session_state.selected_cross_encoder_model = cross_encoder_options[selected_cross_encoder_label]
else:
    st.session_state.selected_cross_encoder_model = None


st.sidebar.subheader("📄 Text Extraction & Processing")
use_trafilatura_opt = st.sidebar.checkbox("Use Trafilatura for Main Content Extraction", value=True, help="Recommended for cleaner text by removing boilerplate.")
st.session_state.trafilatura_favor_recall = st.sidebar.checkbox("Trafilatura: Favor Recall (more text, less precise)", value=False, help="If Trafilatura is used, this gets more content but might include some noise.")

analysis_granularity = st.sidebar.selectbox(
    "Analysis Granularity:",
    ("Passage-based (Groups of sentences)", "Sentence-based (Individual sentences)"),
    index=0, help="Choose to analyze content as passages or individual sentences."
)
use_selenium_opt = st.sidebar.checkbox("Use Selenium for fetching", value=True, help="More robust. Requires Chromedriver.") # Moved down slightly


st.sidebar.divider()
st.sidebar.header("⚙️ Query & URL Configuration")
initial_query_val = st.sidebar.text_input("Initial Search Query:", "benefits of server-side rendering")
urls_text_area_val = st.sidebar.text_area("Enter URLs (one per line):", "https://vercel.com/blog/react-server-components\nhttps://www.patterns.dev/posts/react-server-components/", height=100)
num_sq_val = st.sidebar.slider("Num Synthetic Queries:", min_value=3, max_value=50, value=5)

s_per_p_val_default = 7
s_overlap_val_default = 2
if analysis_granularity == "Passage-based (Groups of sentences)":
    st.sidebar.subheader("Passage Settings:")
    s_per_p_val = st.sidebar.slider("Sentences/Passage:", min_value=2, max_value=20, value=s_per_p_val_default)
    s_overlap_val = st.sidebar.slider("Sentence Overlap:", min_value=0, max_value=10, value=s_overlap_val_default)
else:
    s_per_p_val = 1
    s_overlap_val = 0

analyze_disabled = not st.session_state.get("gemini_api_configured", False)

if st.sidebar.button("🚀 Analyze Content", type="primary", disabled=analyze_disabled):
    if not initial_query_val or not urls_text_area_val: st.warning("Need initial query and URLs."); st.stop()

    # --- Load models based on selection ---
    # This will run only once if models change or first time, due to @st.cache_resource
    # Or if we explicitly clear its cache. For now, it loads on first Analyze click or if model selection changes & reruns.
    with st.spinner("Loading embedding models..."):
        embedding_model, cross_encoder_model = load_embedding_models() 
        if embedding_model is None:
            st.error("Bi-Encoder model failed to load. Cannot proceed.")
            st.stop()
        if use_cross_encoder_rerank and cross_encoder_model is None:
            st.warning("Cross-Encoder model failed to load. Re-ranking will be disabled.")
            # use_cross_encoder_rerank = False # Optionally disable it if loading failed


    st.session_state.all_url_metrics_list = []
    st.session_state.url_processed_units_dict = {}
    st.session_state.all_queries_for_analysis = []
    st.session_state.analysis_done = False

    current_selenium_driver = None
    if use_selenium_opt:
        if st.session_state.get("selenium_driver_instance") is None:
            with st.spinner("Initializing Selenium WebDriver..."):
                st.session_state.selenium_driver_instance = initialize_selenium_driver()
        current_selenium_driver = st.session_state.selenium_driver_instance
        if not current_selenium_driver: st.warning("Selenium driver failed. Using 'requests'.")

    local_urls = [url.strip() for url in urls_text_area_val.split('\n') if url.strip()]
    actual_overlap = 0
    if analysis_granularity == "Passage-based (Groups of sentences)":
        actual_overlap = max(0, s_per_p_val - 1) if s_overlap_val >= s_per_p_val else s_overlap_val
    
    synthetic_queries_only = generate_synthetic_queries(initial_query_val, num_sq_val)
    local_all_queries = [f"Initial: {initial_query_val}"]
    if synthetic_queries_only: local_all_queries.extend(synthetic_queries_only)
    else: st.warning("Synthetic query generation failed. Analyzing against initial query only.")
    if not local_all_queries: st.error("No queries for analysis."); st.stop()
    
    local_all_query_embs = get_bi_embeddings(local_all_queries, embedding_model) # Use selected bi-encoder

    # --- NEW: Get embedding for the initial query separately for weighting ---
    initial_query_text_only = initial_query_val # The pure text
    initial_query_embedding_for_weighting = get_bi_embeddings([initial_query_text_only], embedding_model)[0]


    local_all_metrics = []
    local_processed_units_data = {}

    with st.spinner(f"Processing {len(local_urls)} URLs..."):
        for i, url in enumerate(local_urls):
            st.markdown(f"--- \n#### Processing URL {i+1}: {url}")
            html = None
            if use_selenium_opt and current_selenium_driver:
                html = fetch_content_with_selenium(url, current_selenium_driver)
            if not html: html = fetch_content_with_requests(url)
            
            text = parse_and_clean_html(html, url, use_trafilatura=use_trafilatura_opt) # Use trafilatura option
            processed_units = []
            if not text or len(text.strip()) < 20:
                st.warning(f"Insufficient text from {url}.")
                if text: processed_units = [text]
                else:
                    for sq_idx, query_text in enumerate(local_all_queries):
                        local_all_metrics.append({"URL":url,"Query":query_text,"Overall Similarity":0.0,"Max Unit Sim.":0.0,"Avg. Unit Sim.":0.0, "ReRanked Max Sim.":0.0, "Num Units":0})
                    continue 
            else:
                if analysis_granularity == "Sentence-based (Individual sentences)":
                    processed_units = split_text_into_sentences(text)
                else:
                    processed_units = split_text_into_passages(text, s_per_p_val, actual_overlap)
                if not processed_units:
                    st.info(f"No distinct text units from {url}. Using entire content.")
                    processed_units = [text]
            
            unit_embeddings = get_bi_embeddings(processed_units, embedding_model) # Use selected bi-encoder
            local_processed_units_data[url] = {"units":processed_units, "embeddings":unit_embeddings, "unit_similarities":None, "page_text_for_highlight": text, "reranked_scores": {}}

            if unit_embeddings.size > 0:
                calc_unit_embs = unit_embeddings.reshape(1,-1) if unit_embeddings.ndim==1 else unit_embeddings
                if local_all_query_embs is None or local_all_query_embs.size==0:
                    st.error("Query embeddings missing."); continue
                
                # --- NEW: Weighted Averaging for Overall URL Embedding ---
                # Calculate similarity of each unit to the *initial query* to get weights
                if initial_query_embedding_for_weighting is not None and initial_query_embedding_for_weighting.size > 0:
                    unit_sims_to_initial_query = cosine_similarity(calc_unit_embs, initial_query_embedding_for_weighting.reshape(1, -1)).flatten()
                    weights = unit_sims_to_initial_query / np.sum(unit_sims_to_initial_query) if np.sum(unit_sims_to_initial_query) > 0 else np.ones(len(calc_unit_embs)) / len(calc_unit_embs)
                    weighted_overall_url_emb = np.average(calc_unit_embs, axis=0, weights=weights).reshape(1,-1)
                else: # Fallback to simple mean if initial query embedding failed
                    st.warning("Could not get initial query embedding for weighting. Using simple mean.")
                    weighted_overall_url_emb = np.mean(calc_unit_embs,axis=0).reshape(1,-1)
                # --- END Weighted Averaging ---

                overall_sims = cosine_similarity(weighted_overall_url_emb, local_all_query_embs)[0] # Use weighted emb
                
                bi_encoder_unit_q_sims = cosine_similarity(calc_unit_embs, local_all_query_embs)
                local_processed_units_data[url]["unit_similarities"] = bi_encoder_unit_q_sims # Store bi-encoder sims for heatmap

                for sq_idx, query_text in enumerate(local_all_queries):
                    current_q_unit_bi_sims = bi_encoder_unit_q_sims[:, sq_idx]
                    max_s_bi = np.max(current_q_unit_bi_sims) if current_q_unit_bi_sims.size > 0 else 0.0
                    avg_s_bi = np.mean(current_q_unit_bi_sims) if current_q_unit_bi_sims.size > 0 else 0.0
                    
                    # --- NEW: Cross-Encoder Re-ranking (Optional) ---
                    max_s_cross = 0.0
                    if use_cross_encoder_rerank and cross_encoder_model and current_q_unit_bi_sims.size > 0:
                        # Get top N units based on bi-encoder to re-rank
                        top_k_indices = np.argsort(current_q_unit_bi_sims)[-st.session_state.get("cross_encoder_top_n_rerank", 5):][::-1] # Top N for re-ranking
                        rerank_units_text = [processed_units[k] for k in top_k_indices]
                        
                        # Prepare pairs for cross-encoder: (query, unit_text)
                        # Need the actual text of the current query (without "Initial:" prefix)
                        current_query_text_for_cross_encoder = query_text.replace("Initial: ", "")
                        query_passage_pairs = [[current_query_text_for_cross_encoder, unit_t] for unit_t in rerank_units_text]
                        
                        if query_passage_pairs:
                            ce_scores = get_cross_encoder_scores(query_passage_pairs, cross_encoder_model)
                            if ce_scores: # Check if scores were returned
                                max_s_cross = np.max(ce_scores)
                                # Store all re-ranked scores for this query and URL if needed for detailed view
                                local_processed_units_data[url]["reranked_scores"][query_text] = sorted(list(zip(rerank_units_text, ce_scores)), key=lambda x:x[1], reverse=True)
                    # --- END Cross-Encoder ---

                    local_all_metrics.append({
                        "URL":url,"Query":query_text,
                        "Overall Similarity (Weighted)":overall_sims[sq_idx], # Now weighted
                        "Max Unit Sim. (Bi-Enc)":max_s_bi,
                        "Avg. Unit Sim. (Bi-Enc)":avg_s_bi,
                        "ReRanked Max Sim. (Cross-Enc)": max_s_cross if use_cross_encoder_rerank else "N/A",
                        "Num Units":len(processed_units)
                    })
            else: # No unit_embeddings
                st.warning(f"No text unit embeddings for {url}.")
                for sq_idx, query_text in enumerate(local_all_queries):
                    local_all_metrics.append({"URL":url,"Query":query_text,"Overall Similarity (Weighted)":0.0,"Max Unit Sim. (Bi-Enc)":0.0,"Avg. Unit Sim. (Bi-Enc)":0.0, "ReRanked Max Sim. (Cross-Enc)":"N/A", "Num Units":0})
    
    if local_all_metrics:
        st.session_state.all_url_metrics_list = local_all_metrics
        st.session_state.url_processed_units_dict = local_processed_units_data
        st.session_state.all_queries_for_analysis = local_all_queries
        st.session_state.analysis_done = True
        st.session_state.last_analysis_granularity = analysis_granularity
    else:
        st.info("No data processed for summary."); st.session_state.analysis_done = False

# --- Display Results ---
if st.session_state.get("analysis_done") and st.session_state.all_url_metrics_list:
    st.subheader("Analysed Queries (Initial + Synthetic)")
    if st.session_state.all_queries_for_analysis:
        query_display_list = [q.replace("Initial: ", "(Initial Query) ") if q.startswith("Initial: ") else q for q in st.session_state.all_queries_for_analysis]
        st.expander("View All Analysed Queries").json(query_display_list)

    unit_label = "Sentence" if st.session_state.get("last_analysis_granularity") == "Sentence-based (Individual sentences)" else "Passage"
    
    st.markdown("---"); st.subheader(f"📈 Overall Similarity & {unit_label} Metrics Summary")
    df_summary = pd.DataFrame(st.session_state.all_url_metrics_list)
    # Update column names for display
    summary_cols_display = ['URL', 'Query', 'Overall Similarity (Weighted)', f'Max {unit_label} Sim. (Bi-Enc)', f'Avg. {unit_label} Sim. (Bi-Enc)', f'ReRanked Max Sim. (Cross-Enc)', f'Num {unit_label}s']
    df_summary_display = df_summary.rename(columns={
        "Max Unit Sim. (Bi-Enc)": f"Max {unit_label} Sim. (Bi-Enc)", 
        "Avg. Unit Sim. (Bi-Enc)": f"Avg. {unit_label} Sim. (Bi-Enc)",
        "Num Units": f"Num {unit_label}s"
    })
    # Select only the columns we want to display from the potentially renamed DataFrame
    df_to_show_in_table = df_summary_display[summary_cols_display]

    st.dataframe(df_to_show_in_table.style.format({
        "Overall Similarity (Weighted)":"{:.3f}",
        f"Max {unit_label} Sim. (Bi-Enc)":"{:.3f}",
        f"Avg. {unit_label} Sim. (Bi-Enc)":"{:.3f}",
        "ReRanked Max Sim. (Cross-Enc)": lambda x: f"{x:.3f}" if isinstance(x, float) else x # Handle "N/A"
    }), use_container_width=True, height=(min(len(df_summary)*38+38,700)))
    
    st.markdown("---"); st.subheader("📊 Visual: Overall URL vs. All Queries Similarity (Weighted)")
    df_overall_bar = df_summary.drop_duplicates(subset=['URL','Query'])
    fig_bar = px.bar(df_overall_bar,x="Query",y="Overall Similarity (Weighted)",color="URL",barmode="group", title="Overall Webpage Similarity (Weighted) to All Analysed Queries",height=max(600,100*num_sq_val))
    fig_bar.update_xaxes(tickangle=30,automargin=True,title_text=None)
    fig_bar.update_yaxes(range=[0,1]); fig_bar.update_layout(legend_title_text='Webpage URL')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---"); st.subheader(f"🔥 {unit_label} Heatmaps (Bi-Encoder Scores) vs. All Queries")
    if st.session_state.url_processed_units_dict and st.session_state.all_queries_for_analysis:
        embedding_model, cross_encoder_model = load_embedding_models() # Ensure models are loaded for display functions
        for url_idx, (url, p_data) in enumerate(st.session_state.url_processed_units_dict.items()):
            with st.expander(f"Heatmap & Details for: {url}", expanded=(url_idx==0)):
                units, unit_bi_sims = p_data["units"], p_data.get("unit_similarities") # These are bi-encoder similarities
                page_full_text = p_data.get("page_text_for_highlight", "")
                reranked_scores_for_url = p_data.get("reranked_scores", {})


                if unit_bi_sims is None or unit_bi_sims.size==0: st.write(f"No {unit_label.lower()} similarity data."); continue
                
                hover = [[f"<b>{unit_label[0]}{i+1}</b> vs Q:'{st.session_state.all_queries_for_analysis[j][:45]}...'<br>Bi-Sim:{unit_bi_sims[i,j]:.3f}<hr>Txt:{units[i][:120]}..." for j in range(unit_bi_sims.shape[1])] for i in range(unit_bi_sims.shape[0])]
                short_queries_display = [q.replace("Initial: ", "(Initial) ")[:50] + ('...' if len(q.replace("Initial: ", "(Initial) ")) > 50 else '') for q in st.session_state.all_queries_for_analysis]
                unit_labels = [f"{unit_label[0]}{i+1}" for i in range(len(units))]
                ticks = (list(range(0,len(unit_labels),max(1,len(unit_labels)//15))),[unit_labels[k] for k in range(0,len(unit_labels),max(1,len(unit_labels)//15))]) if len(unit_labels)>25 else (unit_labels,unit_labels)
                
                fig_heat = go.Figure(data=go.Heatmap(z=unit_bi_sims.T,x=unit_labels,y=short_queries_display,colorscale='Viridis', hoverongaps=False,text=np.array(hover).T,hoverinfo='text',zmin=0,zmax=1))
                fig_heat.update_layout(title=f"{unit_label} Similarity (Bi-Encoder) for {url}", xaxis_title=f"{unit_label}s",yaxis_title="Analysed Queries",height=max(400,50*len(short_queries_display)+100), yaxis_autorange='reversed',xaxis=dict(tickmode='array',tickvals=ticks[0],ticktext=ticks[1],automargin=True))
                st.plotly_chart(fig_heat, use_container_width=True)
                
                st.markdown("---")
                selected_query_for_detail = st.selectbox(f"Select Query for {unit_label}-level details:", options=st.session_state.all_queries_for_analysis, key=f"q_sel_{url_idx}_{url.replace('/', '_')}")
                
                if selected_query_for_detail:
                    query_display_name = selected_query_for_detail.replace("Initial: ", "(Initial) ")[:30] + "..."
                    
                    # Display Highlighted Text (uses Bi-Encoder for coloring as Cross-Encoder doesn't give per-sentence scores for all sentences easily)
                    if page_full_text and st.checkbox(f"Show highlighted text for '{query_display_name}'?", key=f"cb_hl_{url_idx}_{url.replace('/', '_')}_{selected_query_for_detail[:10].replace(' ','_')}"):
                        with st.spinner("Highlighting..."): 
                            actual_query_text = selected_query_for_detail.replace("Initial: ", "")
                            st.markdown(get_highlighted_sentence_html(page_full_text, actual_query_text, embedding_model), unsafe_allow_html=True)
                    
                    # Display Top/Bottom N (can show both Bi-Encoder and Cross-Encoder results if available)
                    if st.checkbox(f"Show top/bottom N {unit_label.lower()}s for '{query_display_name}'?", key=f"cb_tb_{url_idx}_{url.replace('/', '_')}_{selected_query_for_detail[:10].replace(' ','_')}"):
                        n_val = st.slider("N for top/bottom:", 1, 10, 3, key=f"sl_tb_{url_idx}_{url.replace('/', '_')}_{selected_query_for_detail[:10].replace(' ','_')}")
                        
                        # Bi-Encoder Top/Bottom
                        q_idx_bi = st.session_state.all_queries_for_analysis.index(selected_query_for_detail)
                        current_q_unit_bi_sims = unit_bi_sims[:, q_idx_bi]
                        scored_units_bi = sorted(list(zip(units, current_q_unit_bi_sims)), key=lambda item: item[1], reverse=True)
                        
                        st.markdown(f"**Bi-Encoder Top {n_val} {unit_label}s:**")
                        for u_t, u_s in scored_units_bi[:n_val]: st.caption(f"Score: {u_s:.3f} - {u_t[:200]}...")
                        st.markdown(f"**Bi-Encoder Bottom {n_val} {unit_label}s:**")
                        for u_t, u_s in scored_units_bi[-n_val:]: st.caption(f"Score: {u_s:.3f} - {u_t[:200]}...")

                        # Cross-Encoder Top/Bottom (if enabled and scores exist for this query)
                        if use_cross_encoder_rerank and selected_query_for_detail in reranked_scores_for_url:
                            st.markdown("---")
                            st.markdown(f"**Cross-Encoder Re-ranked Top {n_val} {unit_label}s (from initial top Bi-Encoder hits):**")
                            reranked_list = reranked_scores_for_url[selected_query_for_detail]
                            for u_t, u_s in reranked_list[:n_val]: st.caption(f"Score: {u_s:.3f} - {u_t[:200]}...")
                            # Note: Bottom N for cross-encoder re-ranking of *top bi-encoder hits* might not be as meaningful
                            # as they are already pre-selected high-scoring candidates.

elif st.session_state.get("analysis_done"):
    st.info("Analysis complete, but no data to display. Check inputs or logs.")

st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v3.0 (Advanced Features)")
