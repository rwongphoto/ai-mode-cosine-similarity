import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import openai # --- NEW ---
from openai import OpenAI # --- NEW ---
import re
import ast
import time
import random
import os
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

st.set_page_config(layout="wide", page_title="AI Semantic Analyzer")

# --- Session State Initialization (Original + New) ---
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
# --- FULLY RESTORED USER AGENTS LIST ---
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

# --- NEW: OpenAI API Configuration ---
st.sidebar.header("🔑 OpenAI API Configuration")
openai_api_key_input = st.sidebar.text_input(
    "Enter OpenAI API Key:", type="password", value=st.session_state.get("openai_api_key_to_persist", "")
)
if st.sidebar.button("Set & Verify OpenAI Key"):
    if openai_api_key_input:
        try:
            test_client = OpenAI(api_key=openai_api_key_input)
            test_client.embeddings.create(input=["test"], model="text-embedding-3-small")
            st.session_state.openai_api_key_to_persist = openai_api_key_input
            st.session_state.openai_api_configured = True
            st.session_state.openai_client = test_client
            st.sidebar.success("OpenAI API Key Configured!")
        except Exception as e:
            st.session_state.openai_api_key_to_persist = ""
            st.session_state.openai_api_configured = False
            st.session_state.openai_client = None
            st.sidebar.error(f"OpenAI Key Failed: {str(e)[:200]}")
    else:
        st.sidebar.warning("Please enter OpenAI API Key.")

if st.session_state.get("openai_api_configured"):
    st.sidebar.markdown("✅ OpenAI API: **Configured**")
    if st.session_state.openai_client is None and st.session_state.openai_api_key_to_persist:
        try: st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key_to_persist)
        except Exception: st.session_state.openai_api_configured = False
else:
    st.sidebar.markdown("⚠️ OpenAI API: **Not Configured**")

st.sidebar.header("🔑 Gemini API Configuration")
api_key_input = st.sidebar.text_input("Enter Google Gemini API Key:", type="password", value=st.session_state.gemini_api_key_to_persist)
if st.sidebar.button("Set & Verify API Key"):
    if api_key_input:
        try:
            genai.configure(api_key=api_key_input)
            if not [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods or 'embedContent' in m.supported_generation_methods]:
                 raise Exception("No usable models found for this API key.")
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


# --- Embedding Functions ---

@st.cache_resource
def load_local_sentence_transformer_model(model_name):
    try: return SentenceTransformer(model_name)
    except Exception as e: st.error(f"Failed to load local model '{model_name}': {e}"); return None

def get_openai_embeddings(texts: list, client: OpenAI, model: str):
    if not texts or not client: return np.array([])
    try:
        texts = [text.replace("\n", " ") for text in texts]
        response = client.embeddings.create(input=texts, model=model)
        return np.array([item.embedding for item in response.data])
    except Exception as e: st.error(f"OpenAI embedding failed: {e}"); return np.array([])

def get_gemini_embeddings(texts: list, model: str):
    if not texts: return np.array([])
    try:
        result = genai.embed_content(model=model, content=texts, task_type="RETRIEVAL_DOCUMENT")
        return np.array(result['embedding'])
    except Exception as e: st.error(f"Gemini embedding failed: {e}"); return np.array([])

def get_embeddings(texts, local_model_instance=None):
    """--- NEW MASTER FUNCTION --- Routes to the correct embedding provider."""
    model_choice = st.session_state.selected_embedding_model
    if model_choice.startswith("openai-"):
        model_name = model_choice.replace("openai-", "")
        return get_openai_embeddings(texts, client=st.session_state.openai_client, model=model_name)
    elif model_choice.startswith("gemini-"):
        model_name = "models/" + model_choice.replace("gemini-", "")
        return get_gemini_embeddings(texts, model=model_name)
    else:
        if local_model_instance is None: st.error("Local embedding model not loaded."); return np.array([])
        return local_model_instance.encode(list(texts) if isinstance(texts, tuple) else texts)

# --- Original Core Functions (Restored) ---

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

def parse_and_clean_html(html_content, url, use_trafilatura=True):
    if not html_content: return None
    text_content = None
    if use_trafilatura:
        downloaded = None
        if html_content: text_content = trafilatura.extract(html_content, include_comments=False, include_tables=False, favor_recall=st.session_state.get("trafilatura_favor_recall", False))
        if not text_content or len(text_content) < 50:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                 new_text_content = trafilatura.extract(downloaded, include_comments=False, include_tables=False, favor_recall=st.session_state.get("trafilatura_favor_recall", False))
                 if new_text_content and (not text_content or len(new_text_content) > len(text_content)): text_content = new_text_content
    if not text_content:
        try:
            from bs4 import BeautifulSoup
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
        except Exception as e_bs4: st.error(f"BS4 fallback error ({url}): {e_bs4}"); return None
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

def get_highlighted_sentence_html(page_text_content, query_text, local_model_instance=None):
    if not page_text_content or not query_text: return ""
    sentences = split_text_into_sentences(page_text_content)
    if not sentences: return "<p>No sentences to highlight.</p>"
    
    sentence_embeddings = get_embeddings(sentences, local_model_instance)
    query_embedding = get_embeddings([query_text], local_model_instance)
    
    if sentence_embeddings.size == 0 or query_embedding.size == 0: return "<p>Could not generate embeddings.</p>"
    
    query_embedding = query_embedding[0].reshape(1, -1)
    similarities = cosine_similarity(sentence_embeddings, query_embedding).flatten()
    
    if not similarities.size: return "<p>No similarity scores.</p>"
    
    highlighted_html = ""
    # --- FIX APPLIED HERE ---
    # The min/max normalization has been removed.
    # We now check the raw similarity score 'sim' directly against your thresholds.
    for sentence, sim in zip(sentences, similarities):
        color = "green" if sim >= 0.75 else "red" if sim < 0.35 else "black"
        highlighted_html += f'<p style="color:{color}; margin-bottom: 2px;">{sentence}</p>'
        
    return highlighted_html

# --- FULLY RESTORED: Your original synthetic query function ---
def generate_synthetic_queries(user_query, num_queries=7):
    if not st.session_state.get("gemini_api_configured", False): st.error("Gemini API not configured."); return []
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
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
        content_text = response.text.strip()
        try:
            for pf in ["```python","```json","```"]:
                if content_text.startswith(pf): content_text=content_text.split(pf,1)[1].rsplit("```",1)[0].strip(); break
            queries = ast.literal_eval(content_text) if content_text.startswith('[') else \
                      [re.sub(r'^\s*[-\*\d\.]+\s*','',q.strip().strip('"\'')) for q in content_text.split('\n') if q.strip()]
            if not isinstance(queries,list) or not all(isinstance(qs,str) for qs in queries): raise ValueError("Not list of str.")
            return [qs for qs in queries if qs.strip()]
        except (SyntaxError,ValueError) as e:
            st.error(f"Gemini response parse error: {e}. Raw: {content_text[:300]}...")
            extracted=[re.sub(r'^\s*[-\*\d\.]+\s*','',l.strip().strip('"\'')) for l in content_text.split('\n') if l.strip()]
            if extracted: st.warning("Fallback parsing used."); return extracted
            return []
    except Exception as e: st.error(f"Gemini API call error: {e}"); return []


st.title("✨ AI Mode Simulator ✨")
st.markdown("Fetch, clean, analyze web content against initial & AI-generated queries. Features advanced text extraction and weighted scoring.")

st.sidebar.subheader("🤖 Embedding Model Configuration")
# --- UPDATED: Unified Model Selector ---
embedding_model_options = {
    "Local: MPNet (Quality Focus)": "all-mpnet-base-v2",
    "Local: MiniLM (Speed Focus)": "all-MiniLM-L6-v2",
    "Local: DistilRoBERTa (Balanced)": "all-distilroberta-v1",
    "OpenAI: text-embedding-3-small": "openai-text-embedding-3-small",
    "OpenAI: text-embedding-3-large": "openai-text-embedding-3-large",
    "Gemini: embedding-001": "gemini-embedding-001",
}
selected_embedding_label = st.sidebar.selectbox("Select Embedding Model:", options=list(embedding_model_options.keys()), index=0)
st.session_state.selected_embedding_model = embedding_model_options[selected_embedding_label]

st.sidebar.subheader("📄 Text Extraction & Processing")
use_trafilatura_opt = st.sidebar.checkbox("Use Trafilatura for Main Content Extraction", value=True)
st.session_state.trafilatura_favor_recall = st.sidebar.checkbox("Trafilatura: Favor Recall (more text, less precise)", value=False)
analysis_granularity = st.sidebar.selectbox("Analysis Granularity:", ("Passage-based (Groups of sentences)", "Sentence-based (Individual sentences)"), index=0)
use_selenium_opt = st.sidebar.checkbox("Use Selenium for fetching", value=True)

st.sidebar.divider()
st.sidebar.header("⚙️ Query & URL Configuration")
initial_query_val = st.sidebar.text_input("Initial Search Query:", "benefits of server-side rendering")
urls_text_area_val = st.sidebar.text_area("Enter URLs (one per line):", "https://vercel.com/blog/understanding-rendering-in-react\nhttps://www.patterns.dev/posts/rendering-patterns/", height=100)
num_sq_val = st.sidebar.slider("Num Synthetic Queries:", min_value=3, max_value=50, value=5)
if analysis_granularity == "Passage-based (Groups of sentences)":
    st.sidebar.subheader("Passage Settings:")
    s_per_p_val = st.sidebar.slider("Sentences/Passage:", 2, 20, 4)
    s_overlap_val = st.sidebar.slider("Sentence Overlap:", 0, 10, 2)
else: s_per_p_val = 1; s_overlap_val = 0

analyze_disabled = not st.session_state.get("gemini_api_configured", False)

if st.sidebar.button("🚀 Analyze Content", type="primary", disabled=analyze_disabled):
    if not initial_query_val or not urls_text_area_val: st.warning("Need initial query and URLs."); st.stop()

    local_embedding_model_instance = None
    model_choice = st.session_state.selected_embedding_model
    if not model_choice.startswith(("openai-", "gemini-")):
        with st.spinner(f"Loading local embedding model: {model_choice}..."):
            local_embedding_model_instance = load_local_sentence_transformer_model(model_choice)
        if local_embedding_model_instance is None: st.error("Local model failed to load."); st.stop()
    else:
        if model_choice.startswith("openai-") and not st.session_state.openai_api_configured: st.error("OpenAI API not configured."); st.stop()
        if model_choice.startswith("gemini-") and not st.session_state.gemini_api_configured: st.error("Gemini API not configured."); st.stop()

    # Reset state variables as in original
    st.session_state.all_url_metrics_list, st.session_state.url_processed_units_dict = [], {}
    st.session_state.all_queries_for_analysis, st.session_state.analysis_done = [], False

    if use_selenium_opt and st.session_state.get("selenium_driver_instance") is None:
        with st.spinner("Initializing Selenium WebDriver..."):
            st.session_state.selenium_driver_instance = initialize_selenium_driver()

    local_urls = [url.strip() for url in urls_text_area_val.split('\n') if url.strip()]
    
    synthetic_queries = generate_synthetic_queries(initial_query_val, num_sq_val)
    local_all_queries = [f"Initial: {initial_query_val}"] + (synthetic_queries or [])
    
    # --- UPDATED to use master embedding function ---
    local_all_query_embs = get_embeddings(local_all_queries, local_embedding_model_instance)
    initial_query_embedding = get_embeddings([initial_query_val], local_embedding_model_instance)[0]

    local_all_metrics, local_processed_units_data = [], {}

    with st.spinner(f"Processing {len(local_urls)} URLs..."):
        for i, url in enumerate(local_urls):
            st.markdown(f"--- \n#### Processing URL {i+1}: {url}")
            html = fetch_content_with_selenium(url, st.session_state.selenium_driver_instance) if use_selenium_opt else fetch_content_with_requests(url)
            text = parse_and_clean_html(html, url, use_trafilatura=use_trafilatura_opt)
            
            if not text or len(text.strip()) < 20:
                st.warning(f"Insufficient text from {url}."); continue

            processed_units = (split_text_into_passages(text, s_per_p_val, s_overlap_val) if analysis_granularity.startswith("Passage") 
                               else split_text_into_sentences(text))
            if not processed_units: processed_units = [text]
            
            unit_embeddings = get_embeddings(processed_units, local_embedding_model_instance)
            local_processed_units_data[url] = {"units":processed_units, "embeddings":unit_embeddings, "unit_similarities":None, "page_text_for_highlight": text}

            if unit_embeddings.size > 0:
                unit_sims_to_initial = cosine_similarity(unit_embeddings, initial_query_embedding.reshape(1, -1)).flatten()
                weights = np.maximum(0, unit_sims_to_initial)
                weighted_overall_url_emb = np.average(unit_embeddings, axis=0, weights=weights if np.sum(weights) > 1e-6 else None).reshape(1, -1)

                overall_sims = cosine_similarity(weighted_overall_url_emb, local_all_query_embs)[0]
                unit_q_sims = cosine_similarity(unit_embeddings, local_all_query_embs)
                local_processed_units_data[url]["unit_similarities"] = unit_q_sims

                for sq_idx, query_text in enumerate(local_all_queries):
                    current_q_unit_sims = unit_q_sims[:, sq_idx]
                    local_all_metrics.append({
                        "URL":url, "Query":query_text,
                        "Overall Similarity (Weighted)": overall_sims[sq_idx],
                        "Max Unit Sim.": np.max(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0,
                        "Avg. Unit Sim.": np.mean(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0,
                        "Num Units":len(processed_units)
                    })
    
    if local_all_metrics:
        st.session_state.all_url_metrics_list = local_all_metrics
        st.session_state.url_processed_units_dict = local_processed_units_data
        st.session_state.all_queries_for_analysis = local_all_queries
        st.session_state.analysis_done = True
        st.session_state.last_analysis_granularity = analysis_granularity

# --- Display Results (Restored original structure) ---
if st.session_state.get("analysis_done") and st.session_state.all_url_metrics_list:
    st.subheader("Analysed Queries (Initial + Synthetic)")
    st.expander("View All Analysed Queries").json([q.replace("Initial: ", "(Initial) ") for q in st.session_state.all_queries_for_analysis])

    unit_label = "Sentence" if st.session_state.last_analysis_granularity.startswith("Sentence") else "Passage"
    
    st.markdown("---"); st.subheader(f"📈 Overall Similarity & {unit_label} Metrics Summary")
    df_summary = pd.DataFrame(st.session_state.all_url_metrics_list)
    df_display = df_summary.rename(columns={
        "Max Unit Sim.": f"Max {unit_label} Sim.", "Avg. Unit Sim.": f"Avg. {unit_label} Sim.", "Num Units": f"Num {unit_label}s"
    })
    st.dataframe(df_display.style.format("{:.3f}", subset=["Overall Similarity (Weighted)", f"Max {unit_label} Sim.", f"Avg. {unit_label} Sim."]), use_container_width=True)
    
    st.markdown("---"); st.subheader("📊 Visual: Overall URL vs. All Queries Similarity (Weighted)")
    fig_bar = px.bar(df_display, x="Query", y="Overall Similarity (Weighted)", color="URL", barmode="group", title="Overall Page Similarity to Queries", height=max(600, 80 * len(st.session_state.all_queries_for_analysis)))
    fig_bar.update_yaxes(range=[0,1]); st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---"); st.subheader(f"🔥 {unit_label} Heatmaps vs. All Queries")
    local_model_instance_for_display = None
    if not st.session_state.selected_embedding_model.startswith(('openai-', 'gemini-')):
        local_model_instance_for_display = load_local_sentence_transformer_model(st.session_state.selected_embedding_model)

    for url_idx, (url, p_data) in enumerate(st.session_state.url_processed_units_dict.items()):
        with st.expander(f"Heatmap & Details for: {url}", expanded=(url_idx==0)):
            if p_data.get("unit_similarities") is None: st.write(f"No {unit_label.lower()} similarity data."); continue
            
            unit_sims, units = p_data["unit_similarities"], p_data["units"]
            all_queries = st.session_state.all_queries_for_analysis
            short_queries = [q.replace("Initial: ", "(I) ")[:50] + ('...' if len(q) > 50 else '') for q in all_queries]
            unit_labels = [f"{unit_label[0]}{i+1}" for i in range(len(units))]
            
            # --- FIX IS HERE: HOVER TEXT LOGIC RESTORED ---
            hover_text = [[f"<b>{unit_label[0]}{i+1}</b> vs Q:'{all_queries[j][:45]}...'<br>Similarity:{unit_sims[i, j]:.3f}<hr>Text:{units[i][:120]}..."
                           for i in range(unit_sims.shape[0])]
                          for j in range(unit_sims.shape[1])]

            fig_heat = go.Figure(data=go.Heatmap(
                z=unit_sims.T, 
                x=unit_labels, 
                y=short_queries, 
                colorscale='Viridis', 
                zmin=0, 
                zmax=1,
                text=hover_text,
                hoverinfo='text'
            ))
            
            fig_heat.update_layout(title=f"{unit_label} Similarity for {url}", height=max(400, 40 * len(short_queries) + 100), yaxis_autorange='reversed')
            st.plotly_chart(fig_heat, use_container_width=True)
            
            st.markdown("---")
            # Restored original unique keys for widgets
            key_base = f"{url_idx}_{url.replace('/', '_')}"
            selected_query = st.selectbox(f"Select Query for Details:", options=st.session_state.all_queries_for_analysis, key=f"q_sel_{key_base}")
            
            if selected_query:
                query_idx = st.session_state.all_queries_for_analysis.index(selected_query)
                scored_units = sorted(zip(p_data["units"], unit_sims[:, query_idx]), key=lambda item: item[1], reverse=True)
                
                query_display_name = selected_query.replace("Initial: ", "(I) ")[:30] + "..."
                if st.checkbox(f"Show highlighted text for '{query_display_name}'?", key=f"cb_hl_{key_base}_{query_idx}"):
                    with st.spinner("Highlighting..."):
                        actual_query = selected_query.replace("Initial: ", "")
                        st.markdown(get_highlighted_sentence_html(p_data["page_text_for_highlight"], actual_query, local_model_instance_for_display), unsafe_allow_html=True)
                
                if st.checkbox(f"Show top/bottom {unit_label.lower()}s for '{query_display_name}'?", key=f"cb_tb_{key_base}_{query_idx}"):
                    n_val = st.slider("N:", 1, 10, 3, key=f"sl_tb_{key_base}_{query_idx}")
                    st.markdown(f"**Top {n_val} {unit_label}s:**")
                    for u_t, u_s in scored_units[:n_val]: st.caption(f"Score: {u_s:.3f} - {u_t[:200]}...")
                    st.markdown(f"**Bottom {n_val} {unit_label}s:**")
                    for u_t, u_s in scored_units[-n_val:]: st.caption(f"Score: {u_s:.3f} - {u_t[:200]}...")

st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v5.0 | Gemini + OpenAI")
