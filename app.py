import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import re
import nltk
import ast
import time
import random

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Query Fan-Out Analyzer")

# --- Session State Initialization ---
if "all_url_metrics_list" not in st.session_state:
    st.session_state.all_url_metrics_list = None
if "url_passage_data_dict" not in st.session_state:
    st.session_state.url_passage_data_dict = None
if "synthetic_queries_list" not in st.session_state:
    st.session_state.synthetic_queries_list = None
if "analysis_done" not in st.session_state: # Flag to indicate if analysis has been run
    st.session_state.analysis_done = False
if "gemini_api_key_to_persist" not in st.session_state: st.session_state.gemini_api_key_to_persist = ""
if "gemini_api_configured" not in st.session_state: st.session_state.gemini_api_configured = False
if "selenium_driver_instance" not in st.session_state: st.session_state.selenium_driver_instance = None # For Selenium driver


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
]
def get_random_user_agent(): return random.choice(USER_AGENTS)
def enforce_rate_limit():
    global last_request_time
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < REQUEST_INTERVAL: time.sleep(REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

# --- NLTK ---
@st.cache_resource # Safe to cache
def download_nltk_resources():
    try: nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Downloading NLTK 'punkt'...")
        try: nltk.download('punkt', quiet=True)
        except Exception as e: st.sidebar.error(f"NLTK download error: {e}")
download_nltk_resources()

# --- Models & Driver Setup ---
@st.cache_resource # Safe to cache
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedding_model = load_sentence_transformer_model()

# @st.cache_resource # Keep Selenium setup uncached for now if driver needs to be fresh or re-init on demand
def initialize_selenium_driver():
    # st.write("DEBUG: Initializing Selenium Driver...")
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"user-agent={get_random_user_agent()}")
    try:
        service = ChromeService() # Assumes chromedriver is in PATH
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        st.error(f"Selenium WebDriver init failed: {e}. Ensure chromedriver is set up.")
        return None

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
    if st.session_state.gemini_api_key_to_persist: # Re-apply config if key exists in session
        try: genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
        except Exception: st.session_state.gemini_api_configured = False
else: st.sidebar.markdown("⚠️ Gemini API: **Not Configured**")

# --- Helper Functions (NO CACHING on these for now after previous issues) ---
def fetch_content_with_selenium(url, driver_instance):
    if not driver_instance:
        st.warning(f"Selenium driver N/A for {url}. Falling back to requests.")
        return fetch_content_with_requests(url)
    enforce_rate_limit()
    try:
        driver_instance.get(url)
        time.sleep(5) # Consider WebDriverWait
        return driver_instance.page_source
    except Exception as e: st.error(f"Selenium error ({url}): {e}"); return None

def fetch_content_with_requests(url):
    enforce_rate_limit() # Ensure rate limit is also applied for requests fallback
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e: st.error(f"Requests error ({url}): {e}"); return None

def parse_and_clean_html(html_content, url):
    if not html_content: return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for el in soup(["script", "style", "noscript", "iframe", "link", "meta", 'nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption', 'menu', 'banner', 'dialog']): el.decompose()
        selectors = ["[class*='menu']","[id*='nav']","[class*='header']","[id*='footer']","[class*='sidebar']","[class*='cookie']","[class*='consent']","[class*='popup']","[class*='modal']","[class*='social']","[class*='share']","[class*='advert']","[id*='ad']","[aria-hidden='true']"]
        for sel in selectors:
            try:
                for element in soup.select(sel):
                    is_main = element.name in ['body','main','article'] or any(c in element.get('class',[]) for c in ['content','main-content','article-body'])
                    if not is_main or element.name not in ['body','main','article']:
                        if element.parent: element.decompose()
            except: pass
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+',' ',text); text = re.sub(r'\.{3,}','.',text); text = re.sub(r'( \.){2,}','.',text)
        return text.strip() if text.strip() else None
    except Exception as e: st.error(f"HTML parsing error ({url}): {e}"); return None

def split_text_into_passages(text, s_per_p=7, s_overlap=2):
    if not text: return []
    try: sentences = nltk.sent_tokenize(text)
    except: sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 3] or \
                        [s.strip() for s in text.split('\n\n') if s.strip() and len(s.split()) > 5]
    if not sentences: return []
    passages, step = [], max(1, s_per_p - s_overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + s_per_p])
        if chunk.strip() and len(chunk.split()) > 10: passages.append(chunk)
    return [p for p in passages if p.strip()]

def get_embeddings(_texts): # NO CACHING - this was the source of the identical scores bug
    if not _texts: return np.array([])
    return embedding_model.encode(_texts)

def generate_synthetic_queries(user_query, num_queries=7): # NO CACHING
    if not st.session_state.get("gemini_api_configured", False): st.error("Gemini API not configured."); return []
    model_name = "gemini-2.5-flash-preview-05-20"
    try: model = genai.GenerativeModel(model_name)
    except Exception as e: st.error(f"Gemini model init error ({model_name}): {e}"); return []
    
    # --- THIS IS THE CORRECT, DETAILED FAN-OUT PROMPT ---
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
    # --- END OF DETAILED PROMPT ---
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
st.title("🌐 AI Query Fan-Out & Webpage Analyzer")
st.markdown("Fetch content, generate diverse queries, visualize alignment. **Requires Gemini API Key & Selenium.**")

use_selenium_opt = st.sidebar.checkbox("Use Selenium for fetching", value=True, help="More robust. Requires Chromedriver.")
st.sidebar.divider()
st.sidebar.header("⚙️ Analysis Configuration")
initial_query_val = st.sidebar.text_input("Initial Search Query:", "benefits of server-side rendering")
urls_text_area_val = st.sidebar.text_area("Enter URLs (one per line):", "https://vercel.com/blog/react-server-components\nhttps://www.patterns.dev/posts/react-server-components/", height=100)
num_sq_val = st.sidebar.slider("Num Synthetic Queries:", 3, 10, 5)
st.sidebar.subheader("Passage Settings:")
s_per_p_val = st.sidebar.slider("Sentences/Passage:", 2, 20, 7)
s_overlap_val = st.sidebar.slider("Sentence Overlap:", 0, 10, 2)

analyze_disabled = not st.session_state.get("gemini_api_configured", False)

if st.sidebar.button("🚀 Analyze Content", type="primary", disabled=analyze_disabled):
    if not initial_query_val or not urls_text_area_val: st.warning("Need initial query and URLs."); st.stop()

    # Clear previous results from session state for a fresh run
    st.session_state.all_url_metrics_list = []
    st.session_state.url_passage_data_dict = {}
    st.session_state.synthetic_queries_list = []
    st.session_state.analysis_done = False

    current_selenium_driver = None
    if use_selenium_opt:
        if st.session_state.get("selenium_driver_instance") is None:
            with st.spinner("Initializing Selenium WebDriver..."):
                st.session_state.selenium_driver_instance = initialize_selenium_driver()
        current_selenium_driver = st.session_state.selenium_driver_instance
        if not current_selenium_driver: st.warning("Selenium driver failed. Using 'requests'.")

    local_urls = [url.strip() for url in urls_text_area_val.split('\n') if url.strip()]
    actual_overlap = max(0, s_per_p_val - 1) if s_overlap_val >= s_per_p_val else s_overlap_val
    
    # --- Store results in local variables first, then assign to session_state ---
    local_syn_queries = generate_synthetic_queries(initial_query_val, num_sq_val)
    if not local_syn_queries: st.error("No synthetic queries generated."); st.stop()
    
    local_syn_query_embs = get_embeddings(local_syn_queries)

    local_all_metrics = []
    local_passage_data = {}

    with st.spinner(f"Processing {len(local_urls)} URLs..."):
        for i, url in enumerate(local_urls):
            st.markdown(f"--- \n#### Processing URL {i+1}: {url}")
            html = None
            if use_selenium_opt and current_selenium_driver:
                html = fetch_content_with_selenium(url, current_selenium_driver)
            if not html: html = fetch_content_with_requests(url)
            
            text = parse_and_clean_html(html, url)
            passages = []
            if not text or len(text.strip()) < 30:
                st.warning(f"Insufficient text from {url}.")
                if text: passages = [text] # Use whole text if some exists
                else: # No text at all
                    for sq_idx, sq in enumerate(local_syn_queries):
                        local_all_metrics.append({"URL":url,"Query":sq,"Overall Similarity":0.0,"Max Passage Sim.":0.0,"Avg. Passage Sim.":0.0,"Num Passages":0})
                    continue 
            else:
                passages = split_text_into_passages(text, s_per_p_val, actual_overlap)
                if not passages:
                    st.info(f"No distinct passages from {url}. Using entire content.")
                    passages = [text]
            
            passage_embs = get_embeddings(passages)
            local_passage_data[url] = {"passages":passages, "embeddings":passage_embs, "passage_similarities":None}

            if passage_embs.size > 0:
                calc_p_embs = passage_embs.reshape(1,-1) if passage_embs.ndim==1 else passage_embs
                if local_syn_query_embs is None or local_syn_query_embs.size==0:
                    st.error("Synthetic query embeddings missing."); continue
                
                overall_url_emb = np.mean(calc_p_embs,axis=0).reshape(1,-1)
                overall_sims = cosine_similarity(overall_url_emb, local_syn_query_embs)[0]
                
                # Your debug block
                st.write(f"--- DEBUG FOR URL: {url} (Loop Iteration: {i}) ---")
                st.write(f"Num passages: {len(passages)}")
                if passages: st.write(f"Sample passage: '{passages[0][:100]}...'")
                st.write(f"Shape of passage_embs: {passage_embs.shape}")
                st.write(f"Mean URL Emb (first 5): {overall_url_emb[0, :5]}")
                st.write(f"Overall Sims: {overall_sims}")
                st.write(f"--- END DEBUG FOR URL: {url} ---")

                passage_q_sims = cosine_similarity(calc_p_embs, local_syn_query_embs)
                local_passage_data[url]["passage_similarities"] = passage_q_sims

                for sq_idx, sq in enumerate(local_syn_queries):
                    current_q_passage_sims = passage_q_sims[:, sq_idx]
                    max_s = np.max(current_q_passage_sims) if current_q_passage_sims.size > 0 else 0.0
                    avg_s = np.mean(current_q_passage_sims) if current_q_passage_sims.size > 0 else 0.0
                    local_all_metrics.append({
                        "URL":url,"Query":sq,"Overall Similarity":overall_sims[sq_idx],
                        "Max Passage Sim.":max_s,"Avg. Passage Sim.":avg_s,
                        "Num Passages":len(passages)
                    })
            else: # No passage embeddings
                st.warning(f"No passage embeddings for {url}.")
                for sq_idx, sq in enumerate(local_syn_queries):
                    local_all_metrics.append({"URL":url,"Query":sq,"Overall Similarity":0.0,"Max Passage Sim.":0.0,"Avg. Passage Sim.":0.0,"Num Passages":0})
    
    # Store results in session state AFTER processing all URLs
    if local_all_metrics:
        st.session_state.all_url_metrics_list = local_all_metrics
        st.session_state.url_passage_data_dict = local_passage_data
        st.session_state.synthetic_queries_list = local_syn_queries
        st.session_state.analysis_done = True
    else:
        st.info("No data processed for summary.")
        st.session_state.analysis_done = False


# --- Display Results (reads from session state) ---
if st.session_state.get("analysis_done") and st.session_state.all_url_metrics_list:
    st.subheader("🤖 Generated Synthetic Queries")
    st.expander("View Queries").json(st.session_state.synthetic_queries_list)

    st.markdown("---"); st.subheader("📈 Overall Similarity & Passage Metrics Summary")
    df_summary = pd.DataFrame(st.session_state.all_url_metrics_list)
    st.dataframe(df_summary[['URL', 'Query', 'Overall Similarity', 'Max Passage Sim.', 'Avg. Passage Sim.', 'Num Passages']].style.format({
        "Overall Similarity":"{:.3f}","Max Passage Sim.":"{:.3f}","Avg. Passage Sim.":"{:.3f}"
    }), use_container_width=True, height=(min(len(df_summary)*38+38,700)))

    st.markdown("---"); st.subheader("📊 Visual: Overall URL vs. Synthetic Query Similarity")
    df_overall_bar = df_summary.drop_duplicates(subset=['URL','Query'])
    fig_bar = px.bar(df_overall_bar,x="Query",y="Overall Similarity",color="URL",barmode="group",
                     title="Overall Webpage Similarity to Synthetic Queries",height=max(600,100*num_sq_val))
    fig_bar.update_xaxes(tickangle=30,automargin=True,title_text=None)
    fig_bar.update_yaxes(range=[0,1]); fig_bar.update_layout(legend_title_text='Webpage URL')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---"); st.subheader("🔥 Passage Heatmaps vs. Synthetic Queries")
    if st.session_state.url_passage_data_dict and st.session_state.synthetic_queries_list:
        for url_idx, (url, p_data) in enumerate(st.session_state.url_passage_data_dict.items()):
            with st.expander(f"Heatmap for: {url}", expanded=(url_idx==0)):
                passages, passage_sims = p_data["passages"], p_data.get("passage_similarities")
                if passage_sims is None or passage_sims.size==0: st.write("No passage similarity data."); continue
                
                hover = [[f"<b>P{i+1}</b> vs Q:'{st.session_state.synthetic_queries_list[j][:45]}...'<br>Sim:{passage_sims[i,j]:.3f}<hr>Txt:{passages[i][:120]}..."
                          for j in range(passage_sims.shape[1])] for i in range(passage_sims.shape[0])]
                short_sq = [q[:50]+'...' if len(q)>50 else q for q in st.session_state.synthetic_queries_list]
                p_labels = [f"P{i+1}" for i in range(len(passages))]
                ticks = (list(range(0,len(p_labels),max(1,len(p_labels)//15))),[p_labels[k] for k in range(0,len(p_labels),max(1,len(p_labels)//15))]) if len(p_labels)>25 else (p_labels,p_labels)

                fig_heat = go.Figure(data=go.Heatmap(z=passage_sims.T,x=p_labels,y=short_sq,colorscale='Viridis',
                                                     hoverongaps=False,text=np.array(hover).T,hoverinfo='text',zmin=0,zmax=1))
                fig_heat.update_layout(title=f"Passage Similarity for {url}",
                                       xaxis_title="Passages",yaxis_title="Queries",height=max(400,50*len(short_sq)+100),
                                       yaxis_autorange='reversed',xaxis=dict(tickmode='array',tickvals=ticks[0],ticktext=ticks[1],automargin=True))
                st.plotly_chart(fig_heat, use_container_width=True)

                if st.checkbox("Show highest/lowest similarity passages?", key=f"cb_passages_{url.replace('/', '_').replace(':', '_')}"): # Made key more robust
                    for q_idx, sq_txt in enumerate(st.session_state.synthetic_queries_list):
                        st.markdown(f"##### Query: '{sq_txt}'")
                        if passage_sims.shape[1] > q_idx:
                            sims_q = passage_sims[:, q_idx]
                            if sims_q.size > 0:
                                idx_max,idx_min = np.argmax(sims_q),np.argmin(sims_q)
                                st.markdown(f"**Most Similar (P{idx_max+1} - Score: {sims_q[idx_max]:.3f}):**"); st.caption(passages[idx_max])
                                st.markdown(f"**Least Similar (P{idx_min+1} - Score: {sims_q[idx_min]:.3f}):**"); st.caption(passages[idx_min])
elif st.session_state.get("analysis_done"):
    st.info("Analysis complete, but no data to display. Check inputs or logs.")

st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v1.9 (Session State & Full Prompt)")
