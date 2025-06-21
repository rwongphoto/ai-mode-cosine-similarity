import streamlit as st
import requests
from sentence_transformers import SentenceTransformer # No CrossEncoder, as removed previously
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import openai # Added
from openai import OpenAI # Added
import re
import ast
import time
import random
import os
import trafilatura
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
if "selenium_driver_instance" not in st.session_state: st.session_state.selenium_driver_instance = None
# API Keys
if "gemini_api_key_to_persist" not in st.session_state: st.session_state.gemini_api_key_to_persist = ""
if "gemini_api_configured" not in st.session_state: st.session_state.gemini_api_configured = False
if "openai_api_key_to_persist" not in st.session_state: st.session_state.openai_api_key_to_persist = "" # New
if "openai_api_configured" not in st.session_state: st.session_state.openai_api_configured = False # New
if "openai_client" not in st.session_state: st.session_state.openai_client = None # New
# Model Selection
if "selected_embedding_model" not in st.session_state: st.session_state.selected_embedding_model = 'Local: MPNet (Quality Focus)' # Default local model
if "loaded_local_model_name" not in st.session_state: st.session_state.loaded_local_model_name = None
if "local_embedding_model_instance" not in st.session_state: st.session_state.local_embedding_model_instance = None


# --- Web Fetching Enhancements ---
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

# --- Models & Driver Setup ---
@st.cache_resource # Cache local model loading
def load_local_sentence_transformer_model(model_name_key):
    # model_name_key is like "Local: MPNet (Quality Focus)"
    # We need to extract the actual model name like "all-mpnet-base-v2"
    actual_model_name = EMBEDDING_MODEL_OPTIONS[model_name_key]
    st.info(f"Loading local Sentence Transformer model: {actual_model_name}")
    try:
        model = SentenceTransformer(actual_model_name)
        st.session_state.loaded_local_model_name = model_name_key # Store which model is loaded
        return model
    except Exception as e:
        st.error(f"Failed to load local model '{actual_model_name}': {e}")
        return None

def initialize_selenium_driver():
    options = ChromeOptions(); options.add_argument("--headless"); options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage"); options.add_argument("--disable-gpu"); options.add_argument(f"user-agent={get_random_user_agent()}")
    try: return webdriver.Chrome(service=ChromeService(), options=options)
    except Exception as e: st.error(f"Selenium init failed: {e}"); return None

# --- API Key Configuration ---
st.sidebar.header("🔑 API Key Configuration")

# OpenAI API Key
openai_api_key_input = st.sidebar.text_input(
    "OpenAI API Key (for embedding models like text-embedding-ada-002):",
    type="password",
    value=st.session_state.openai_api_key_to_persist,
    key="openai_api_key"
)
if st.sidebar.button("Set & Verify OpenAI Key"):
    if openai_api_key_input:
        try:
            test_client = OpenAI(api_key=openai_api_key_input)
            # Test with a small, common model
            test_client.embeddings.create(input=["test"], model="text-embedding-3-small") # Use a known model
            st.session_state.openai_api_key_to_persist = openai_api_key_input
            st.session_state.openai_api_configured = True
            st.session_state.openai_client = test_client # Store the initialized client
            st.sidebar.success("OpenAI API Key Configured!")
        except Exception as e:
            st.session_state.openai_api_key_to_persist = ""
            st.session_state.openai_api_configured = False
            st.session_state.openai_client = None
            st.sidebar.error(f"OpenAI Key Failed: {e}")
    else:
        st.sidebar.warning("Please enter OpenAI API Key.")

if st.session_state.get("openai_api_configured"):
    st.sidebar.markdown("✅ OpenAI API: **Configured**")
    # Ensure client is re-initialized if key exists but client is None (e.g. after script rerun)
    if st.session_state.openai_client is None and st.session_state.openai_api_key_to_persist:
        try:
            st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key_to_persist)
        except Exception: # If re-init fails, mark as not configured
            st.session_state.openai_api_configured = False
            st.sidebar.markdown("⚠️ OpenAI API: Re-init failed. Please re-verify.")
else:
    st.sidebar.markdown("⚠️ OpenAI API: **Not Configured** (Needed for OpenAI embedding models)")


# Gemini API Key
gemini_api_key_input_val = st.sidebar.text_input("Google Gemini API Key (for embeddings & fan-out):", type="password", value=st.session_state.gemini_api_key_to_persist, key="gemini_api_key")
if st.sidebar.button("Set & Verify Gemini Key"): # Changed button label for uniqueness
    if gemini_api_key_input_val:
        try:
            genai.configure(api_key=gemini_api_key_input_val)
            # Check for both content generation and embedding models
            models = genai.list_models()
            can_generate = any('generateContent' in m.supported_generation_methods for m in models)
            can_embed = any('embedContent' in m.supported_generation_methods for m in models)
            if not (can_generate and can_embed): # Need both for this app
                 raise Exception("Key valid, but required models (generate/embed) might not be available/enabled for this key/project.")
            st.session_state.gemini_api_key_to_persist = gemini_api_key_input_val
            st.session_state.gemini_api_configured = True
            st.sidebar.success("Gemini API Key Configured!")
        except Exception as e:
            st.session_state.gemini_api_key_to_persist = ""
            st.session_state.gemini_api_configured = False
            st.sidebar.error(f"Gemini Key Failed: {str(e)[:200]}")
    else: st.sidebar.warning("Please enter Gemini API Key.")

if st.session_state.get("gemini_api_configured"):
    st.sidebar.markdown("✅ Gemini API: **Configured**")
    if st.session_state.gemini_api_key_to_persist: # Re-apply config on rerun
        try: genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
        except Exception: st.session_state.gemini_api_configured = False
else: st.sidebar.markdown("⚠️ Gemini API: **Not Configured** (Needed for query fan-out & Gemini embeddings)")


# --- Embedding Functions ---
def get_openai_embeddings(texts: list, client: OpenAI, model_id: str = "text-embedding-3-small"):
    if not texts or client is None : return np.array([])
    try:
        texts = [text.replace("\n", " ") for text in texts] # OpenAI recommendation
        response = client.embeddings.create(input=texts, model=model_id)
        return np.array([item.embedding for item in response.data])
    except Exception as e: st.error(f"OpenAI embedding error with model {model_id}: {e}"); return np.array([])

def get_gemini_embeddings(texts: list, model_id: str = "models/embedding-001"):
    if not texts or not st.session_state.get("gemini_api_configured"): return np.array([])
    try:
        # Gemini's embed_content can take a list of strings directly for 'RETRIEVAL_DOCUMENT'
        result = genai.embed_content(model=model_id, content=texts, task_type="RETRIEVAL_DOCUMENT")
        return np.array(result['embedding'])
    except Exception as e: st.error(f"Gemini embedding error with model {model_id}: {e}"); return np.array([])

def get_embeddings_master(texts_to_embed: list): # Renamed to avoid conflict
    """Master function to route to the correct embedding provider."""
    selected_model_key = st.session_state.selected_embedding_model # This is the user-friendly label
    
    if not texts_to_embed: return np.array([])

    if selected_model_key.startswith("OpenAI: "):
        model_api_name = EMBEDDING_MODEL_OPTIONS[selected_model_key] # Get the actual API model name
        if not st.session_state.get("openai_api_configured") or st.session_state.openai_client is None:
            st.error("OpenAI API not configured. Cannot get OpenAI embeddings.")
            return np.array([])
        return get_openai_embeddings(texts_to_embed, client=st.session_state.openai_client, model_id=model_api_name)
    
    elif selected_model_key.startswith("Gemini: "):
        model_api_name = EMBEDDING_MODEL_OPTIONS[selected_model_key] # Get the actual API model name
        if not st.session_state.get("gemini_api_configured"):
            st.error("Gemini API not configured. Cannot get Gemini embeddings.")
            return np.array([])
        return get_gemini_embeddings(texts_to_embed, model_id=model_api_name)
    
    elif selected_model_key.startswith("Local: "):
        # Ensure the correct local model is loaded if selection changed
        if st.session_state.loaded_local_model_name != selected_model_key or st.session_state.local_embedding_model_instance is None:
            with st.spinner(f"Loading {selected_model_key}..."):
                st.session_state.local_embedding_model_instance = load_local_sentence_transformer_model(selected_model_key)
        
        if st.session_state.local_embedding_model_instance is None:
            st.error(f"Local model {selected_model_key} failed to load.")
            return np.array([])
        return st.session_state.local_embedding_model_instance.encode(list(texts_to_embed) if isinstance(texts_to_embed, tuple) else texts_to_embed)
    else:
        st.error(f"Unknown embedding model selection: {selected_model_key}")
        return np.array([])


# --- Original Core Functions (Restored and using master get_embeddings) ---
def fetch_content_with_selenium(url, driver_instance):
    # ... (your existing selenium fetch)
    if not driver_instance: st.warning(f"Selenium N/A for {url}. Fallback."); return fetch_content_with_requests(url)
    enforce_rate_limit(); driver_instance.get(url); time.sleep(5); return driver_instance.page_source

def fetch_content_with_requests(url):
    # ... (your existing requests fetch)
    enforce_rate_limit(); headers={'User-Agent':get_random_user_agent()}; resp=requests.get(url,timeout=20,headers=headers); resp.raise_for_status(); return resp.text

def parse_and_clean_html(html_content, url, use_trafilatura=True):
    # ... (your existing trafilatura/bs4 parse and clean)
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

def get_highlighted_sentence_html(page_text_content, query_text, unit_scores_map=None):
    if not page_text_content or not query_text: return ""
    sentences = split_text_into_sentences(page_text_content)
    if not sentences: return "<p>No sentences to highlight.</p>"
    highlighted_html = ""

    if unit_scores_map: # If passage scores are provided, use them
        for sentence in sentences:
            sentence_score = 0.5 
            for unit_text, score in unit_scores_map.items():
                if sentence in unit_text: sentence_score = score; break
            color = "green" if sentence_score >= 0.65 else "red" if sentence_score < 0.35 else "black"
            highlighted_html += f'<p style="color:{color}; margin-bottom: 2px;">{sentence}</p>'
    else: # Fallback to sentence-by-sentence scoring if no unit_scores_map (e.g., sentence granularity)
        sentence_embeddings = get_embeddings_master(sentences) # Use master embedding func
        query_embedding = get_embeddings_master([query_text])
        if sentence_embeddings.size == 0 or query_embedding.size == 0: return "<p>Could not generate embeddings for highlighting.</p>"
        query_embedding = query_embedding[0].reshape(1, -1)
        similarities = cosine_similarity(sentence_embeddings, query_embedding).flatten()
        if not similarities.size: return "<p>No similarity scores.</p>"
        for sentence, sim in zip(sentences, similarities):
            color = "green" if sim >= 0.65 else "red" if sim < 0.35 else "black"
            highlighted_html += f'<p style="color:{color}; margin-bottom: 2px;">{sentence}</p>'
    return highlighted_html


def generate_synthetic_queries(user_query, num_queries=7):
    # --- Your FULLY RESTORED Gemini prompt ---
    if not st.session_state.get("gemini_api_configured", False): st.error("Gemini API not configured."); return []
    model_name = "gemini-1.5-flash-latest" # Use your preferred model here
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
st.title("✨ AI Semantic Search Analyzer ✨")
st.markdown("Analyze web content or pasted text against initial & AI-generated queries.")

# --- NEW: Content Input Method ---
st.sidebar.subheader("📜 Content Input Method")
input_method = st.sidebar.radio(
    "Choose content source:",
    ("Fetch URLs", "Paste Text Content"),
    key="input_method_selector"
)

st.sidebar.subheader("🤖 Embedding Model Configuration")
EMBEDDING_MODEL_OPTIONS = { # Define this globally or near the top
    "Local: MPNet (Quality Focus)": "all-mpnet-base-v2",
    "Local: MiniLM (Speed Focus)": "all-MiniLM-L6-v2",
    "Local: DistilRoBERTa (Balanced)": "all-distilroberta-v1",
    "OpenAI: text-embedding-3-small": "text-embedding-3-small", # API name
    "OpenAI: text-embedding-3-large": "text-embedding-3-large", # API name
    "Gemini: embedding-001": "models/embedding-001", # API name including "models/"
}
selected_embedding_label = st.sidebar.selectbox(
    "Select Embedding Model:",
    options=list(EMBEDDING_MODEL_OPTIONS.keys()),
    index=0, # Default to first local model
    key="embedding_model_selector"
)
st.session_state.selected_embedding_model = EMBEDDING_MODEL_OPTIONS[selected_embedding_label] # Store actual model name/id


st.sidebar.subheader("📄 Text Extraction & Processing")
if input_method == "Fetch URLs": # Only show these if fetching URLs
    use_trafilatura_opt = st.sidebar.checkbox("Use Trafilatura for Main Content Extraction", value=True)
    st.session_state.trafilatura_favor_recall = st.sidebar.checkbox("Trafilatura: Favor Recall", value=False)
    use_selenium_opt = st.sidebar.checkbox("Use Selenium for fetching", value=True)
else: # Defaults if pasting text
    use_trafilatura_opt = False 
    st.session_state.trafilatura_favor_recall = False
    use_selenium_opt = False


analysis_granularity = st.sidebar.selectbox(
    "Analysis Granularity:",
    ("Passage-based (Groups of sentences)", "Sentence-based (Individual sentences)"),
    index=0, key="granularity_selector"
)

st.sidebar.divider()
st.sidebar.header("⚙️ Query & Content Configuration") # Renamed
initial_query_val = st.sidebar.text_input("Initial Search Query:", "benefits of server-side rendering")

# Conditional input for URLs or Pasted Text
urls_to_process_input = [] # This will hold URLs or ["Pasted Content 1"]
pasted_content_input = ""

if input_method == "Fetch URLs":
    urls_text_area_val = st.sidebar.text_area("Enter URLs (one per line):", "https://vercel.com/blog/understanding-rendering-in-react\nhttps://www.patterns.dev/posts/rendering-patterns/", height=100, key="url_input_area")
else: # Paste Text Content
    pasted_content_input = st.sidebar.text_area("Paste Text Content Here:", height=200, key="pasted_text_area")


num_sq_val = st.sidebar.slider("Num Synthetic Queries:", min_value=0, max_value=50, value=5, help="Set to 0 to only use initial query.") # Allow 0 synthetic

s_per_p_val_default = 4 # Adjusted default
s_overlap_val_default = 1 # Adjusted default
if analysis_granularity == "Passage-based (Groups of sentences)":
    st.sidebar.subheader("Passage Settings:")
    s_per_p_val = st.sidebar.slider("Sentences/Passage:", min_value=1, max_value=20, value=s_per_p_val_default, key="s_per_p_slider")
    s_overlap_val = st.sidebar.slider("Sentence Overlap:", min_value=0, max_value=10, value=s_overlap_val_default, key="s_overlap_slider")
else: s_per_p_val = 1; s_overlap_val = 0

analyze_disabled = not (st.session_state.get("gemini_api_configured", False) and \
                        (st.session_state.selected_embedding_model.startswith("Local:") or \
                         (st.session_state.selected_embedding_model.startswith("OpenAI:") and st.session_state.get("openai_api_configured", False)) or \
                         (st.session_state.selected_embedding_model.startswith("Gemini:") and st.session_state.get("gemini_api_configured", False))
                        ))


if st.sidebar.button("🚀 Analyze Content", type="primary", disabled=analyze_disabled):
    # --- Input Validation ---
    if not initial_query_val: st.warning("Initial query is required."); st.stop()
    if input_method == "Fetch URLs" and not urls_text_area_val.strip():
        st.warning("Please enter URLs to fetch."); st.stop()
    if input_method == "Paste Text Content" and not pasted_content_input.strip():
        st.warning("Please paste text content to analyze."); st.stop()

    # --- Load Local Model if Selected ---
    local_embedding_model_instance = None
    if st.session_state.selected_embedding_model.startswith("Local:"):
        # Check if the selected model is already loaded or different from last load
        if st.session_state.loaded_local_model_name != st.session_state.selected_embedding_model or \
           st.session_state.local_embedding_model_instance is None:
            with st.spinner(f"Loading {st.session_state.selected_embedding_model}..."):
                st.session_state.local_embedding_model_instance = load_local_sentence_transformer_model(st.session_state.selected_embedding_model)
        local_embedding_model_instance = st.session_state.local_embedding_model_instance
        if local_embedding_model_instance is None:
            st.error(f"Local embedding model ({st.session_state.selected_embedding_model}) failed to load. Cannot proceed."); st.stop()


    # --- Reset Session State for Results ---
    st.session_state.all_url_metrics_list, st.session_state.url_processed_units_dict = [], {}
    st.session_state.all_queries_for_analysis, st.session_state.analysis_done = [], False

    # --- Prepare Content Source List ---
    content_sources = [] # List of tuples: (identifier, text_content_provider_func or text_itself)
    if input_method == "Fetch URLs":
        urls_list = [url.strip() for url in urls_text_area_val.split('\n') if url.strip()]
        for url in urls_list:
            content_sources.append({"id": url, "type": "url"})
    else: # Pasted content
        content_sources.append({"id": "Pasted Content 1", "type": "pasted", "text": pasted_content_input})
    
    if not content_sources: st.warning("No content sources (URLs or pasted text) to process."); st.stop()

    # --- Selenium Driver Init (if needed for URL fetching) ---
    current_selenium_driver = None
    if input_method == "Fetch URLs" and use_selenium_opt:
        if st.session_state.get("selenium_driver_instance") is None:
            with st.spinner("Initializing Selenium WebDriver..."):
                st.session_state.selenium_driver_instance = initialize_selenium_driver()
        current_selenium_driver = st.session_state.selenium_driver_instance
        if not current_selenium_driver: st.warning("Selenium driver failed. Using 'requests' for URL fetching.")

    # --- Generate Queries ---
    synthetic_queries_only = []
    if num_sq_val > 0: # Only call Gemini if user wants synthetic queries
        synthetic_queries_only = generate_synthetic_queries(initial_query_val, num_sq_val)
    
    local_all_queries = [f"Initial: {initial_query_val}"]
    if synthetic_queries_only: local_all_queries.extend(synthetic_queries_only)
    elif num_sq_val > 0: st.warning("Synthetic query generation failed or returned empty. Analyzing against initial query only.")
    
    if not local_all_queries: st.error("No queries for analysis."); st.stop()
    
    local_all_query_embs = get_embeddings_master(local_all_queries)
    initial_query_embedding = get_embeddings_master([initial_query_val])[0]

    # --- Main Processing Loop ---
    local_all_metrics = []
    local_processed_units_data = {}

    with st.spinner(f"Processing {len(content_sources)} content source(s)..."):
        for i, source_info in enumerate(content_sources):
            content_id = source_info["id"]
            st.markdown(f"--- \n#### Processing Source {i+1}: {content_id}")
            
            text_to_analyze = None
            if source_info["type"] == "url":
                html = fetch_content_with_selenium(content_id, current_selenium_driver) if use_selenium_opt else fetch_content_with_requests(content_id)
                text_to_analyze = parse_and_clean_html(html, content_id, use_trafilatura=use_trafilatura_opt)
            elif source_info["type"] == "pasted":
                text_to_analyze = source_info["text"]
                # Optionally, clean pasted text too if desired (e.g. basic whitespace normalization)
                text_to_analyze = re.sub(r'\s+', ' ', text_to_analyze).strip()
            
            processed_units = []
            if not text_to_analyze or len(text_to_analyze.strip()) < 20:
                st.warning(f"Insufficient text from {content_id}.")
                if text_to_analyze: processed_units = [text_to_analyze] # Use whatever little text is there
                else: # No text at all
                    for sq_idx, query_text in enumerate(local_all_queries):
                        local_all_metrics.append({"URL":content_id,"Query":query_text,"Overall Similarity (Weighted)":0.0,"Max Unit Sim.":0.0,"Avg. Unit Sim.":0.0,"Num Units":0})
                    continue 
            else:
                if analysis_granularity == "Sentence-based (Individual sentences)":
                    processed_units = split_text_into_sentences(text_to_analyze)
                else:
                    processed_units = split_text_into_passages(text_to_analyze, s_per_p_val, s_overlap_val)
                if not processed_units:
                    st.info(f"No distinct units from {content_id}. Using entire content.")
                    processed_units = [text_to_analyze]
            
            unit_embeddings = get_embeddings_master(processed_units)
            local_processed_units_data[content_id] = {"units":processed_units, "embeddings":unit_embeddings, "unit_similarities":None, "page_text_for_highlight": text_to_analyze}

            if unit_embeddings.size > 0:
                calc_unit_embs = unit_embeddings.reshape(1,-1) if unit_embeddings.ndim==1 else unit_embeddings
                if local_all_query_embs is None or local_all_query_embs.size==0: st.error("Query embeddings missing."); continue
                
                if initial_query_embedding is not None and initial_query_embedding.size > 0:
                    unit_sims_to_initial = cosine_similarity(calc_unit_embs, initial_query_embedding.reshape(1, -1)).flatten()
                    weights = np.maximum(0, unit_sims_to_initial); sum_weights = np.sum(weights)
                    weights = weights / sum_weights if sum_weights > 1e-6 else np.ones(len(calc_unit_embs)) / len(calc_unit_embs)
                    weighted_overall_url_emb = np.average(calc_unit_embs, axis=0, weights=weights).reshape(1,-1)
                else: weighted_overall_url_emb = np.mean(calc_unit_embs,axis=0).reshape(1,-1)

                overall_sims = cosine_similarity(weighted_overall_url_emb, local_all_query_embs)[0]
                unit_q_sims = cosine_similarity(calc_unit_embs, local_all_query_embs)
                local_processed_units_data[content_id]["unit_similarities"] = unit_q_sims

                for sq_idx, query_text in enumerate(local_all_queries):
                    current_q_unit_sims = unit_q_sims[:, sq_idx]
                    local_all_metrics.append({
                        "URL":content_id, "Query":query_text,
                        "Overall Similarity (Weighted)": overall_sims[sq_idx],
                        "Max Unit Sim.": np.max(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0,
                        "Avg. Unit Sim.": np.mean(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0,
                        "Num Units":len(processed_units)
                    })
            else: # No unit embeddings
                st.warning(f"No text unit embeddings for {content_id}.")
                for sq_idx, query_text in enumerate(local_all_queries):
                     local_all_metrics.append({"URL":content_id,"Query":query_text,"Overall Similarity (Weighted)":0.0,"Max Unit Sim.":0.0,"Avg. Unit Sim.":0.0,"Num Units":0})
    
    # Store final results in session state
    if local_all_metrics:
        st.session_state.all_url_metrics_list = local_all_metrics
        st.session_state.url_processed_units_dict = local_processed_units_data
        st.session_state.all_queries_for_analysis = local_all_queries
        st.session_state.analysis_done = True
        st.session_state.last_analysis_granularity = analysis_granularity
    else: st.info("No data processed."); st.session_state.analysis_done = False


# --- Display Results (Restored) ---
if st.session_state.get("analysis_done") and st.session_state.all_url_metrics_list:
    st.subheader("Analysed Queries (Initial + Synthetic)")
    st.expander("View All Analysed Queries").json([q.replace("Initial: ", "(I) ") for q in st.session_state.all_queries_for_analysis])
    unit_label = "Sentence" if st.session_state.last_analysis_granularity.startswith("Sentence") else "Passage"
    st.markdown("---"); st.subheader(f"📈 Overall Similarity & {unit_label} Metrics Summary")
    df_summary = pd.DataFrame(st.session_state.all_url_metrics_list)
    df_display = df_summary.rename(columns={"Max Unit Sim.":f"Max {unit_label} Sim.","Avg. Unit Sim.":f"Avg. {unit_label} Sim.","Num Units":f"Num {unit_label}s"})
    st.dataframe(df_display.style.format("{:.3f}",subset=["Overall Similarity (Weighted)",f"Max {unit_label} Sim.",f"Avg. {unit_label} Sim."]),use_container_width=True)
    st.markdown("---"); st.subheader("📊 Visual: Overall URL vs. All Queries Similarity (Weighted)")
    fig_bar = px.bar(df_display,x="Query",y="Overall Similarity (Weighted)",color="URL",barmode="group",title="Page Similarity to Queries",height=max(600,80*len(st.session_state.all_queries_for_analysis)))
    fig_bar.update_yaxes(range=[0,1]); st.plotly_chart(fig_bar,use_container_width=True)
    st.markdown("---"); st.subheader(f"🔥 {unit_label} Heatmaps vs. All Queries")
    if st.session_state.url_processed_units_dict:
        for url_idx, (url, p_data) in enumerate(st.session_state.url_processed_units_dict.items()):
            with st.expander(f"Heatmap & Details for: {url}", expanded=(url_idx==0)):
                if p_data.get("unit_similarities") is None: st.write(f"No {unit_label.lower()} similarity data."); continue
                unit_sims, units = p_data["unit_similarities"], p_data["units"]
                short_queries = [q.replace("Initial: ", "(I) ")[:50]+('...' if len(q)>50 else '') for q in st.session_state.all_queries_for_analysis]
                unit_labels = [f"{unit_label[0]}{i+1}" for i in range(len(units))]
                hover_text = [[f"<b>{unit_labels[i]}</b> vs Q:'{st.session_state.all_queries_for_analysis[j][:45]}...'<br>Sim:{unit_sims[i,j]:.3f}<hr>Txt:{units[i][:120]}..." for j in range(unit_sims.shape[1])] for i in range(unit_sims.shape[0])]
                fig_heat = go.Figure(data=go.Heatmap(z=unit_sims.T,x=unit_labels,y=short_queries,colorscale='Viridis',zmin=0,zmax=1,text=np.array(hover_text).T if hover_text else None, hoverinfo='text' if hover_text else 'all'))
                fig_heat.update_layout(title=f"{unit_label} Similarity for {url}",height=max(400,40*len(short_queries)+100),yaxis_autorange='reversed')
                st.plotly_chart(fig_heat,use_container_width=True)
                st.markdown("---")
                key_base = f"{url_idx}_{url.replace('/','_').replace(':','_')}"
                selected_query = st.selectbox(f"Select Query for Details:",options=st.session_state.all_queries_for_analysis,key=f"q_sel_{key_base}")
                if selected_query:
                    query_idx = st.session_state.all_queries_for_analysis.index(selected_query)
                    scored_units = sorted(zip(units, unit_sims[:, query_idx]),key=lambda x:x[1],reverse=True)
                    query_disp_name = selected_query.replace("Initial: ","(I) ")[:30]+"..."
                    if st.checkbox(f"Highlight text for '{query_disp_name}'?",key=f"cb_hl_{key_base}_{query_idx}"):
                        # Prepare unit_scores_map for consistent highlighting if passage-based
                        unit_scores_for_highlight = None
                        if st.session_state.last_analysis_granularity.startswith("Passage"):
                            unit_scores_for_highlight = {unit_text: score for unit_text, score in scored_units}

                        st.markdown(get_highlighted_sentence_html(p_data["page_text_for_highlight"],selected_query.replace("Initial: ",""), unit_scores_map=unit_scores_for_highlight),unsafe_allow_html=True)
                    if st.checkbox(f"Top/bottom {unit_label.lower()}s for '{query_disp_name}'?",key=f"cb_tb_{key_base}_{query_idx}"):
                        n = st.slider("N:",1,10,3,key=f"sl_tb_{key_base}_{query_idx}")
                        st.markdown(f"**Top {n} {unit_label}s:**"); [st.caption(f"Score: {s:.3f} - {t[:200]}...") for t,s in scored_units[:n]]
                        st.markdown(f"**Bottom {n} {unit_label}s:**"); [st.caption(f"Score: {s:.3f} - {t[:200]}...") for t,s in scored_units[-n:]]

st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v5.0 | Gemini + OpenAI + Paste")
