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
# import nltk # NLTK is removed
import ast
import time
import random
import os

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Query Fan-Out Analyzer")

# --- Session State Initialization ---
if "all_url_metrics_list" not in st.session_state: st.session_state.all_url_metrics_list = None
if "url_processed_units_dict" not in st.session_state: st.session_state.url_processed_units_dict = None
if "all_queries_for_analysis" not in st.session_state: st.session_state.all_queries_for_analysis = None # Will include initial + synthetic
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
def load_sentence_transformer_model(): return SentenceTransformer('all-mpnet-base-v2')
embedding_model = load_sentence_transformer_model()

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
def parse_and_clean_html(html, url):
    if not html: return None
    try:
        soup = BeautifulSoup(html, 'html.parser')
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
        text_content = re.sub(r'\s+',' ',text_content); text_content = re.sub(r'\.{3,}','.',text_content); text_content = re.sub(r'( \.){2,}','.',text_content)
        return text_content.strip() if text_content.strip() else None
    except Exception as e: st.error(f"HTML parsing error ({url}): {e}"); return None

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

def get_embeddings(_texts):
    if not _texts: return np.array([])
    return embedding_model.encode(_texts)

def get_ranked_sentences_for_display(page_text_content, query_text, top_n=5):
    if not page_text_content or not query_text: return [], []
    sentences = split_text_into_sentences(page_text_content)
    if not sentences: return [], []
    sentence_embeddings = get_embeddings(sentences)
    query_embedding = get_embeddings([query_text])[0] # Note: [query_text] for single item
    if sentence_embeddings.size == 0 or query_embedding.size == 0: return [], []
    similarities = cosine_similarity(sentence_embeddings, query_embedding.reshape(1, -1)).flatten()
    sentence_scores = sorted(list(zip(sentences, similarities)), key=lambda item: item[1], reverse=True)
    return sentence_scores[:top_n], sentence_scores[-top_n:]

def get_highlighted_sentence_html(page_text_content, query_text):
    if not page_text_content or not query_text: return ""
    sentences = split_text_into_sentences(page_text_content)
    if not sentences: return "<p>No sentences to highlight.</p>"
    sentence_embeddings = get_embeddings(sentences)
    query_embedding = get_embeddings([query_text])[0] # Note: [query_text] for single item
    if sentence_embeddings.size == 0 or query_embedding.size == 0: return "<p>Could not generate embeddings for highlighting.</p>"
    similarities = cosine_similarity(sentence_embeddings, query_embedding.reshape(1, -1)).flatten()
    highlighted_html = ""
    if not similarities.size: return "<p>No similarity scores for highlighting.</p>"
    min_sim, max_sim = np.min(similarities), np.max(similarities)
    for sentence, sim in zip(sentences, similarities):
        norm_sim = (sim - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 0.5 
        color = "green" if norm_sim >= 0.75 else "red" if norm_sim < 0.35 else "black"
        highlighted_html += f'<p style="color:{color}; margin-bottom: 2px;">{sentence}</p>'
    return highlighted_html

def generate_synthetic_queries(user_query, num_queries=7):
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
st.title("🌐 AI Query Fan-Out & Webpage Analyzer")
st.markdown("Fetch content, generate diverse queries, visualize alignment. **Requires Gemini API Key & Selenium.**")

use_selenium_opt = st.sidebar.checkbox("Use Selenium for fetching", value=True, help="More robust. Requires Chromedriver.")
analysis_granularity = st.sidebar.selectbox(
    "Analysis Granularity:",
    ("Passage-based (Groups of sentences)", "Sentence-based (Individual sentences)"),
    index=0, help="Choose to analyze content as passages or individual sentences."
)
st.sidebar.divider()
st.sidebar.header("⚙️ Analysis Configuration")
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

    st.session_state.all_url_metrics_list = []
    st.session_state.url_processed_units_dict = {}
    # st.session_state.synthetic_queries_list = [] # This will be set by all_queries_for_analysis now
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
    
    # Generate synthetic queries
    synthetic_queries_only = generate_synthetic_queries(initial_query_val, num_sq_val)
    
    # --- Create the full list of queries for analysis (Initial + Synthetic) ---
    local_all_queries = [f"Initial: {initial_query_val}"] # Label the initial query
    if synthetic_queries_only:
        local_all_queries.extend(synthetic_queries_only)
    else: # If Gemini fails, at least analyze against the initial query
        st.warning("Synthetic query generation failed or returned empty. Analyzing against initial query only.")
    
    if not local_all_queries: st.error("No queries available for analysis (initial or synthetic)."); st.stop()
    
    local_all_query_embs = get_embeddings(local_all_queries) # Embed all queries (initial + synthetic)

    local_all_metrics = []
    local_processed_units_data = {}

    with st.spinner(f"Processing {len(local_urls)} URLs..."):
        for i, url in enumerate(local_urls):
            st.markdown(f"--- \n#### Processing URL {i+1}: {url}")
            html = None
            if use_selenium_opt and current_selenium_driver:
                html = fetch_content_with_selenium(url, current_selenium_driver)
            if not html: html = fetch_content_with_requests(url)
            
            text = parse_and_clean_html(html, url)
            processed_units = []
            if not text or len(text.strip()) < 20:
                st.warning(f"Insufficient text from {url}.")
                if text: processed_units = [text]
                else:
                    for sq_idx, query_text in enumerate(local_all_queries):
                        local_all_metrics.append({"URL":url,"Query":query_text,"Overall Similarity":0.0,"Max Unit Sim.":0.0,"Avg. Unit Sim.":0.0,"Num Units":0})
                    continue 
            else:
                if analysis_granularity == "Sentence-based (Individual sentences)":
                    processed_units = split_text_into_sentences(text)
                else:
                    processed_units = split_text_into_passages(text, s_per_p_val, actual_overlap)
                if not processed_units:
                    st.info(f"No distinct text units from {url}. Using entire content.")
                    processed_units = [text]
            
            unit_embeddings = get_embeddings(processed_units)
            local_processed_units_data[url] = {"units":processed_units, "embeddings":unit_embeddings, "unit_similarities":None, "page_text_for_highlight": text}

            if unit_embeddings.size > 0:
                calc_unit_embs = unit_embeddings.reshape(1,-1) if unit_embeddings.ndim==1 else unit_embeddings
                if local_all_query_embs is None or local_all_query_embs.size==0:
                    st.error("Query embeddings missing."); continue
                
                overall_url_emb = np.mean(calc_unit_embs,axis=0).reshape(1,-1)
                overall_sims = cosine_similarity(overall_url_emb, local_all_query_embs)[0]
                
                unit_q_sims = cosine_similarity(calc_unit_embs, local_all_query_embs)
                local_processed_units_data[url]["unit_similarities"] = unit_q_sims

                for sq_idx, query_text in enumerate(local_all_queries): # Iterate over all queries
                    current_q_unit_sims = unit_q_sims[:, sq_idx]
                    max_s = np.max(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0
                    avg_s = np.mean(current_q_unit_sims) if current_q_unit_sims.size > 0 else 0.0
                    local_all_metrics.append({
                        "URL":url,"Query":query_text,"Overall Similarity":overall_sims[sq_idx],
                        "Max Unit Sim.":max_s,"Avg. Unit Sim.":avg_s,
                        "Num Units":len(processed_units)
                    })
            else:
                st.warning(f"No text unit embeddings for {url}.")
                for sq_idx, query_text in enumerate(local_all_queries):
                    local_all_metrics.append({"URL":url,"Query":query_text,"Overall Similarity":0.0,"Max Unit Sim.":0.0,"Avg. Unit Sim.":0.0,"Num Units":0})
    
    if local_all_metrics:
        st.session_state.all_url_metrics_list = local_all_metrics
        st.session_state.url_processed_units_dict = local_processed_units_data
        st.session_state.all_queries_for_analysis = local_all_queries # Store all queries
        st.session_state.analysis_done = True
        st.session_state.last_analysis_granularity = analysis_granularity
    else:
        st.info("No data processed for summary."); st.session_state.analysis_done = False

# --- Display Results ---
if st.session_state.get("analysis_done") and st.session_state.all_url_metrics_list:
    st.subheader("Analysed Queries (Initial + Synthetic)") # Updated title
    # Display all queries including the initial one
    if st.session_state.all_queries_for_analysis:
        query_display_list = [q.replace("Initial: ", "(Initial Query) ") if q.startswith("Initial: ") else q for q in st.session_state.all_queries_for_analysis]
        st.expander("View All Analysed Queries").json(query_display_list)


    unit_label = "Sentence" if st.session_state.get("last_analysis_granularity") == "Sentence-based (Individual sentences)" else "Passage"
    st.markdown("---"); st.subheader(f"📈 Overall Similarity & {unit_label} Metrics Summary")
    df_summary = pd.DataFrame(st.session_state.all_url_metrics_list)
    summary_cols = ['URL', 'Query', 'Overall Similarity', f'Max {unit_label} Sim.', f'Avg. {unit_label} Sim.', f'Num {unit_label}s']
    df_summary_display = df_summary.rename(columns={"Max Unit Sim.": f"Max {unit_label} Sim.", "Avg. Unit Sim.": f"Avg. {unit_label} Sim.", "Num Units": f"Num {unit_label}s"})
    st.dataframe(df_summary_display[summary_cols].style.format({"Overall Similarity":"{:.3f}",f"Max {unit_label} Sim.":"{:.3f}",f"Avg. {unit_label} Sim.":"{:.3f}"}), use_container_width=True, height=(min(len(df_summary)*38+38,700)))
    
    st.markdown("---"); st.subheader("📊 Visual: Overall URL vs. All Queries Similarity") # Updated title
    df_overall_bar = df_summary.drop_duplicates(subset=['URL','Query'])
    fig_bar = px.bar(df_overall_bar,x="Query",y="Overall Similarity",color="URL",barmode="group", title="Overall Webpage Similarity to All Analysed Queries",height=max(600,100*num_sq_val))
    fig_bar.update_xaxes(tickangle=30,automargin=True,title_text=None)
    fig_bar.update_yaxes(range=[0,1]); fig_bar.update_layout(legend_title_text='Webpage URL')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---"); st.subheader(f"🔥 {unit_label} Heatmaps vs. All Queries") # Updated title
    if st.session_state.url_processed_units_dict and st.session_state.all_queries_for_analysis:
        for url_idx, (url, p_data) in enumerate(st.session_state.url_processed_units_dict.items()):
            with st.expander(f"Heatmap & Details for: {url}", expanded=(url_idx==0)):
                units, unit_sims = p_data["units"], p_data.get("unit_similarities")
                page_full_text = p_data.get("page_text_for_highlight", "")
                if unit_sims is None or unit_sims.size==0: st.write(f"No {unit_label.lower()} similarity data."); continue
                
                hover = [[f"<b>{unit_label[0]}{i+1}</b> vs Q:'{st.session_state.all_queries_for_analysis[j][:45]}...'<br>Sim:{unit_sims[i,j]:.3f}<hr>Txt:{units[i][:120]}..." for j in range(unit_sims.shape[1])] for i in range(unit_sims.shape[0])]
                short_queries_display = [q.replace("Initial: ", "(Initial) ")[:50] + ('...' if len(q.replace("Initial: ", "(Initial) ")) > 50 else '') for q in st.session_state.all_queries_for_analysis]

                unit_labels = [f"{unit_label[0]}{i+1}" for i in range(len(units))]
                ticks = (list(range(0,len(unit_labels),max(1,len(unit_labels)//15))),[unit_labels[k] for k in range(0,len(unit_labels),max(1,len(unit_labels)//15))]) if len(unit_labels)>25 else (unit_labels,unit_labels)
                
                fig_heat = go.Figure(data=go.Heatmap(z=unit_sims.T,x=unit_labels,y=short_queries_display,colorscale='Viridis', hoverongaps=False,text=np.array(hover).T,hoverinfo='text',zmin=0,zmax=1))
                fig_heat.update_layout(title=f"{unit_label} Similarity for {url}", xaxis_title=f"{unit_label}s",yaxis_title="Analysed Queries",height=max(400,50*len(short_queries_display)+100), yaxis_autorange='reversed',xaxis=dict(tickmode='array',tickvals=ticks[0],ticktext=ticks[1],automargin=True))
                st.plotly_chart(fig_heat, use_container_width=True)
                
                st.markdown("---")
                # Update selectbox to use all_queries_for_analysis
                selected_query_for_detail = st.selectbox(f"Select Query for {unit_label}-level details:", options=st.session_state.all_queries_for_analysis, key=f"q_sel_{url_idx}_{url.replace('/', '_')}")
                
                if selected_query_for_detail and page_full_text:
                    query_display_name = selected_query_for_detail.replace("Initial: ", "(Initial) ")[:30] + "..."
                    if st.checkbox(f"Show highlighted text for '{query_display_name}'?", key=f"cb_hl_{url_idx}_{url.replace('/', '_')}_{selected_query_for_detail[:10].replace(' ','_')}"): # Shorter key part
                        with st.spinner("Highlighting..."): st.markdown(get_highlighted_sentence_html(page_full_text, selected_query_for_detail.replace("Initial: ", "")), unsafe_allow_html=True) # Pass original query text if it's the initial one
                    
                    if st.checkbox(f"Show top/bottom N {unit_label.lower()}s for '{query_display_name}'?", key=f"cb_tb_{url_idx}_{url.replace('/', '_')}_{selected_query_for_detail[:10].replace(' ','_')}"):
                        n_val = st.slider("N for top/bottom:", 1, 10, 3, key=f"sl_tb_{url_idx}_{url.replace('/', '_')}_{selected_query_for_detail[:10].replace(' ','_')}")
                        with st.spinner(f"Ranking..."):
                            q_idx = st.session_state.all_queries_for_analysis.index(selected_query_for_detail)
                            current_q_unit_sims = unit_sims[:, q_idx]
                            scored_units = sorted(list(zip(units, current_q_unit_sims)), key=lambda item: item[1], reverse=True)
                            
                            st.markdown(f"**Top {n_val} {unit_label}s:**")
                            for u_t, u_s in scored_units[:n_val]:
                                st.caption(f"Score: {u_s:.3f} - {u_t[:200]}...")
                            
                            st.markdown(f"**Bottom {n_val} {unit_label}s:**")
                            for u_t, u_s in scored_units[-n_val:]:
                                st.caption(f"Score: {u_s:.3f} - {u_t[:200]}...")
                elif st.session_state.get("last_analysis_granularity") == "Passage-based (Groups of sentences)": # Fallback for passage details
                     if st.checkbox(f"Show highest/lowest similarity passages?", key=f"cb_passages_old_{url_idx}_{url.replace('/', '_')}"):
                        for q_idx, sq_txt in enumerate(st.session_state.all_queries_for_analysis): # Use all_queries here
                            st.markdown(f"##### Query: '{sq_txt.replace('Initial: ', '(Initial) ')}'")
                            if unit_sims.shape[1] > q_idx:
                                sims_q = unit_sims[:, q_idx]
                                if sims_q.size > 0:
                                    idx_max,idx_min = np.argmax(sims_q),np.argmin(sims_q)
                                    st.markdown(f"**Most Similar ({unit_label[0]}{idx_max+1} - Score: {sims_q[idx_max]:.3f}):**"); st.caption(units[idx_max])
                                    st.markdown(f"**Least Similar ({unit_label[0]}{idx_min+1} - Score: {sims_q[idx_min]:.3f}):**"); st.caption(units[idx_min])
elif st.session_state.get("analysis_done"):
    st.info("Analysis complete, but no data to display. Check inputs or logs.")

st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v2.5 (Initial Query Integrated)")
