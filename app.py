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
# from webdriver_manager.chrome import ChromeDriverManager

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Query Fan-Out Analyzer")

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

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def enforce_rate_limit():
    global last_request_time
    now = time.time()
    elapsed_since_last_request = now - last_request_time
    if elapsed_since_last_request < REQUEST_INTERVAL:
        sleep_time = REQUEST_INTERVAL - elapsed_since_last_request
        time.sleep(sleep_time)
    last_request_time = time.time()

# --- NLTK ---
# @st.cache_resource # REMOVED CACHING
def download_nltk_resources():
    resource_id = 'tokenizers/punkt'
    try: nltk.data.find(resource_id)
    except LookupError:
        st.info(f"Downloading NLTK '{resource_id}'...")
        try: nltk.download('punkt', quiet=True)
        except Exception as e: st.sidebar.error(f"NLTK download failed: {e}")
    except Exception as e: st.warning(f"NLTK check error: {e}")
download_nltk_resources()

# --- Models ---
# @st.cache_resource # REMOVED CACHING
def load_sentence_transformer():
    st.write("DEBUG: load_sentence_transformer EXECUTING (NO CACHE)") # Debug print
    return SentenceTransformer('all-MiniLM-L6-v2')
embedding_model = load_sentence_transformer()

# @st.cache_resource # REMOVED CACHING
def setup_selenium_driver():
    st.write("DEBUG: setup_selenium_driver EXECUTING (NO CACHE)") # Debug print
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"user-agent={get_random_user_agent()}")
    try:
        service = ChromeService()
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        st.error(f"Selenium WebDriver init failed: {e}. Ensure setup. May not work in all cloud envs.")
        return None

# --- Session State & API Key ---
if "gemini_api_key_to_persist" not in st.session_state: st.session_state.gemini_api_key_to_persist = ""
if "gemini_api_configured" not in st.session_state: st.session_state.gemini_api_configured = False
if "selenium_driver" not in st.session_state: st.session_state.selenium_driver = None

st.sidebar.header("🔑 Gemini API Configuration")
api_key_input_val = st.sidebar.text_input("Enter Google Gemini API Key:", type="password", value=st.session_state.gemini_api_key_to_persist)

if st.sidebar.button("Set & Verify API Key"):
    if api_key_input_val:
        try:
            genai.configure(api_key=api_key_input_val)
            if not [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]:
                raise Exception("No usable models found.")
            st.session_state.gemini_api_key_to_persist = api_key_input_val
            st.session_state.gemini_api_configured = True
            st.sidebar.success("Gemini API Key Configured!")
        except Exception as e:
            st.session_state.gemini_api_key_to_persist = ""
            st.session_state.gemini_api_configured = False
            st.sidebar.error(f"API Key Failed: {str(e)[:200]}")
    else: st.sidebar.warning("Please enter an API Key.")

if st.session_state.get("gemini_api_configured"):
    st.sidebar.markdown("✅ Gemini API: **Configured**")
    if st.session_state.gemini_api_key_to_persist:
        try: genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
        except Exception: st.session_state.gemini_api_configured = False
else: st.sidebar.markdown("⚠️ Gemini API: **Not Configured**")

# --- Helper Functions ---
# @st.cache_data(show_spinner="Fetching URL with Selenium...") # REMOVED CACHING
def fetch_content_with_selenium(url, _driver):
    st.write(f"DEBUG: fetch_content_with_selenium EXECUTING for {url} (NO CACHE)") # Debug print
    if not _driver:
        st.warning(f"Selenium driver not available for {url}. Falling back to requests.")
        return fetch_content_with_requests(url)
    enforce_rate_limit()
    try:
        _driver.get(url)
        time.sleep(5)
        return _driver.page_source
    except Exception as e:
        st.error(f"Selenium error ({url}): {e}")
        return None

def fetch_content_with_requests(url):
    st.write(f"DEBUG: fetch_content_with_requests EXECUTING for {url} (NO CACHE)") # Debug print
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.Timeout: st.error(f"Requests timeout ({url})."); return None
    except requests.exceptions.RequestException as e: st.error(f"Requests error ({url}): {e}"); return None

# @st.cache_data(show_spinner="Parsing and cleaning URL content...") # REMOVED CACHING
def parse_and_clean_html(html_content, url):
    st.write(f"DEBUG: parse_and_clean_html EXECUTING for {url} (NO CACHE)") # Debug print
    if not html_content: return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for el_to_remove in soup(["script", "style", "noscript", "iframe", "link", "meta"]): el_to_remove.decompose()
        tags_to_remove_list = ['nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption', 'menu', 'banner', 'dialog']
        for tag_name in tags_to_remove_list:
            for el in soup.find_all(tag_name): el.decompose()
        
        selectors_to_remove_list = [
            "[class*='menu']", "[class*='navbar']", "[id*='nav']", "[class*='header']",
            "[class*='footer']", "[id*='footer']", "[class*='sidebar']",
            "[class*='cookie']", "[id*='cookie']", "[class*='consent']",
            "[class*='popup']", "[class*='modal']", "[id*='modal']",
            "[class*='social']", "[class*='share']", "[class*='breadcrumb']",
            "[class*='advert']", "[class*='ads']", "[class*='sponsor']", "[id*='ad']",
            "[aria-hidden='true']", "[role='dialog']", "[role='alert']"
        ]
        for selector_str in selectors_to_remove_list:
            try:
                for element_found in soup.select(selector_str):
                    is_main_block = element_found.name in ['body', 'main', 'article'] or \
                                    any(c in element_found.get('class', []) for c in ['content', 'main-content', 'article-body'])
                    if not is_main_block or element_found.name not in ['body', 'main', 'article']:
                        if element_found.parent: element_found.decompose()
            except Exception: pass

        text_content = soup.get_text(separator=' ', strip=True)
        text_content = re.sub(r'\s+', ' ', text_content)
        text_content = re.sub(r'\.{3,}', '.', text_content)
        text_content = re.sub(r'( \.){2,}', '.', text_content)
        if not text_content.strip(): return None
        return text_content
    except Exception as e: st.error(f"HTML parsing error ({url}): {e}"); return None

# @st.cache_data(show_spinner="Splitting text into passages...") # REMOVED CACHING
def split_text_into_passages(text, sentences_per_passage=7, sentence_overlap=2):
    st.write(f"DEBUG: split_text_into_passages EXECUTING (NO CACHE). Input text len: {len(text) if text else 0}") # Debug print
    if not text: return []
    try: sentences_list = nltk.sent_tokenize(text)
    except Exception:
        sentences_list = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 3]
        if not sentences_list: sentences_list = [s.strip() for s in text.split('\n\n') if s.strip() and len(s.split()) > 5]
    if not sentences_list: return []
    passages_list, step_size = [], max(1, sentences_per_passage - sentence_overlap)
    for i_idx in range(0, len(sentences_list), step_size):
        passage_text_chunk = " ".join(sentences_list[i_idx : i_idx + sentences_per_passage])
        if passage_text_chunk.strip() and len(passage_text_chunk.split()) > 10: passages_list.append(passage_text_chunk)
    return [p for p in passages_list if p.strip()]

# @st.cache_data # REMOVED CACHING
def get_embeddings(_texts):
    if not _texts:
        st.write("DEBUG: get_embeddings called with empty _texts (NO CACHE). Returning empty array.")
        return np.array([])
    st.write(f"DEBUG: get_embeddings EXECUTING (NO CACHE). Num texts: {len(_texts)}. First text sample: '{_texts[0][:70] if _texts else 'N/A'}'")
    embeddings = embedding_model.encode(_texts)
    st.write(f"DEBUG: get_embeddings (NO CACHE) produced embeddings of shape: {embeddings.shape}")
    return embeddings


# @st.cache_data(show_spinner="Generating synthetic queries with Gemini...") # REMOVED CACHING
def generate_synthetic_queries(user_query, num_queries=7):
    st.write(f"DEBUG: generate_synthetic_queries EXECUTING for '{user_query}' (NO CACHE)") # Debug print
    if not st.session_state.get("gemini_api_configured", False):
        st.error("Gemini API not configured.")
        return []

    model_name = "gemini-1.5-flash-latest" 
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Could not initialize Gemini model ({model_name}): {e}")
        return []

    prompt = f"""
    Based on the user's initial search query: "{user_query}"
    Generate {num_queries} diverse synthetic search queries using the "Query Fan Out" technique.
    These queries should explore different facets, intents, or related concepts.
    Aim for a mix of the following query types, ensuring variety:
    1.  Related Queries: Broader, narrower, or tangentially related topics.
    2.  Implicit Queries: Underlying questions, needs, or assumptions the user might have.
    3.  Comparative Queries: Queries that would compare the initial topic with alternatives or similar concepts.
    4.  Recent Queries (Hypothetical): If this were part of an ongoing search session, what related queries might have come before or could follow? (Focus on logical next steps).
    5.  Personalized Queries (Hypothetical): What might a user with a specific interest (e.g., technical depth, practical application, historical context) search for related to the initial query?
    6.  Reformulation Queries: Different ways of phrasing the same core intent or asking slightly different questions about the main topic.
    7.  Entity-Expanded Queries: Queries that focus on specific entities, people, organizations, or concepts mentioned or implied by the initial query, or expand to related entities.
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
        content_text = "".join(part.text for part in response.parts if hasattr(part, 'text')) if hasattr(response, 'parts') and response.parts else response.text
        content_text = content_text.strip()
        try:
            for prefix_str in ["```python", "```json", "```"]:
                if content_text.startswith(prefix_str):
                    content_text = content_text.split(prefix_str, 1)[1].rsplit("```", 1)[0].strip()
                    break
            if not content_text.startswith('['): 
                queries_list = [re.sub(r'^\s*[-\*\d\.]+\s*', '', q.strip().strip('"\'')) for q in content_text.split('\n') if q.strip() and len(q.strip()) > 3]
            else:
                queries_list = ast.literal_eval(content_text)
            if not isinstance(queries_list, list) or not all(isinstance(q_str, str) for q_str in queries_list):
                raise ValueError("Parsed result is not a list of strings.")
            return [q_str for q_str in queries_list if q_str.strip()]
        except (SyntaxError, ValueError) as e: 
            st.error(f"Error parsing Gemini's response: {e}. Raw: \n{content_text[:300]}...")
            lines = content_text.split('\n')
            extracted_queries_list = [re.sub(r'^\s*[-\*\d\.]+\s*', '', line.strip().strip('"\'')) for line in lines if line.strip() and len(line.strip()) > 3]
            if extracted_queries_list:
                st.warning("Used fallback parsing for synthetic queries.")
                return extracted_queries_list
            return []
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        if hasattr(e, 'message'): st.error(f"Gemini API Error Message: {e.message}")
        return []


# --- Main UI ---
st.title("🌐 AI Query Fan-Out & Webpage Analyzer")
st.markdown("Fetch content, generate diverse queries, visualize alignment. **Requires Gemini API Key & Selenium setup for best results.**")

use_selenium_opt = st.sidebar.checkbox("Use Selenium for fetching (more robust, needs setup)", value=True, help="May bypass anti-bot measures. Requires Chromedriver.")

st.sidebar.divider()
st.sidebar.header("⚙️ Analysis Configuration")
initial_query_input = st.sidebar.text_input("Initial Search Query:", "understanding cloud security posture management")
url_inputs_text_area = st.sidebar.text_area("Enter URLs (one per line):", "https://www.paloaltonetworks.com/cyberpedia/what-is-cspm\nhttps://www.crowdstrike.com/cybersecurity-101/cloud-security/cloud-security-posture-management-cspm/", height=100)
num_synthetic_queries_slider = st.sidebar.slider("Num Synthetic Queries:", 3, 10, 5)
st.sidebar.subheader("Passage Settings:")
sentences_per_passage_slider = st.sidebar.slider("Sentences/Passage:", 2, 20, 7)
sentence_overlap_slider = st.sidebar.slider("Sentence Overlap:", 0, 10, 2)

analyze_button_disabled_flag = not st.session_state.get("gemini_api_configured", False)
analyze_button_tooltip_text = "Configure Gemini API Key." if analyze_button_disabled_flag else ""

if st.sidebar.button("🚀 Analyze Content", type="primary", disabled=analyze_button_disabled_flag, help=analyze_button_tooltip_text):
    if not initial_query_input or not url_inputs_text_area: st.warning("Need initial query and URLs."); st.stop()

    active_driver = None
    if use_selenium_opt:
        # No caching for setup_selenium_driver, so it might re-initialize if not in session_state
        if "selenium_driver" not in st.session_state or st.session_state.selenium_driver is None:
             with st.spinner("Initializing Selenium WebDriver..."):
                st.session_state.selenium_driver = setup_selenium_driver()
        active_driver = st.session_state.selenium_driver
        if not active_driver: st.warning("Selenium driver failed. Using 'requests'.")


    urls_list = [url_str.strip() for url_str in url_inputs_text_area.split('\n') if url_str.strip()]
    if sentence_overlap_slider >= sentences_per_passage_slider: sentence_overlap_slider = max(0, sentences_per_passage_slider - 1)

    # No caching for generate_synthetic_queries, so it will run every time
    synthetic_queries_list = generate_synthetic_queries(initial_query_input, num_synthetic_queries_slider)
    if not synthetic_queries_list: st.error("No synthetic queries generated."); st.stop()

    st.subheader("🤖 Generated Synthetic Queries")
    st.expander("View Queries").json(synthetic_queries_list)
    
    # No caching for get_embeddings on synthetic queries
    synthetic_query_embeddings_arr = get_embeddings(synthetic_queries_list)


    all_url_metrics_list = []
    url_passage_data_dict = {}

    with st.spinner(f"Processing {len(urls_list)} URLs..."):
        for i_url, current_url in enumerate(urls_list):
            st.markdown(f"--- \n#### Processing URL {i_url+1}: {current_url}")
            
            page_html_content = None
            # No caching for fetch_content_with_selenium / fetch_content_with_requests
            if use_selenium_opt and active_driver:
                page_html_content = fetch_content_with_selenium(current_url, active_driver)
            if not page_html_content:
                page_html_content = fetch_content_with_requests(current_url)

            # No caching for parse_and_clean_html
            page_text_content = parse_and_clean_html(page_html_content, current_url)
            current_passages_list = []

            if not page_text_content or len(page_text_content.strip()) < 30:
                st.warning(f"Insufficient text from {current_url}. Attempting overall score if any text.")
                if page_text_content: current_passages_list = [page_text_content]
                else:
                    for sq_idx_val, syn_query_val in enumerate(synthetic_queries_list):
                        all_url_metrics_list.append({"URL": current_url, "Query": syn_query_val, "Overall Similarity": 0.0, "Max Passage Sim.": 0.0, "Avg. Passage Sim.": 0.0, "Num Passages": 0})
                    continue
            else:
                # No caching for split_text_into_passages
                current_passages_list = split_text_into_passages(page_text_content, sentences_per_passage_slider, sentence_overlap_slider)
                if not current_passages_list:
                    st.info(f"No distinct passages from {current_url}. Using entire content as one passage.")
                    current_passages_list = [page_text_content]
            
            # No caching for get_embeddings on passages
            passage_embeddings_arr = get_embeddings(current_passages_list)
            url_passage_data_dict[current_url] = {"passages": current_passages_list, "embeddings": passage_embeddings_arr, "passage_similarities": None}

            if passage_embeddings_arr.size > 0:
                calc_passage_embs = passage_embeddings_arr.reshape(1, -1) if passage_embeddings_arr.ndim == 1 else passage_embeddings_arr
                
                if synthetic_query_embeddings_arr is None or synthetic_query_embeddings_arr.size == 0:
                    st.error("Synthetic query embeddings are missing. Cannot calculate similarities.")
                    for sq_idx_val, syn_query_val in enumerate(synthetic_queries_list):
                        all_url_metrics_list.append({"URL": current_url, "Query": syn_query_val, "Overall Similarity": 0.0, "Max Passage Sim.": 0.0, "Avg. Passage Sim.": 0.0, "Num Passages": len(current_passages_list)})
                    continue

                overall_url_emb_arr = np.mean(calc_passage_embs, axis=0).reshape(1, -1)
                overall_sims_to_queries_arr = cosine_similarity(overall_url_emb_arr, synthetic_query_embeddings_arr)[0]
                
                # +++ START DEBUG BLOCK for overall similarity +++
                st.write(f"--- DEBUG FOR URL: {current_url} (Loop Iteration: {i_url}) ---")
                st.write(f"Number of passages processed: {len(current_passages_list)}")
                if current_passages_list:
                     st.write(f"Sample of first passage: '{current_passages_list[0][:100]}...'")
                st.write(f"Shape of `passage_embeddings_arr` for this URL: {passage_embeddings_arr.shape}")
                st.write(f"Mean of `passage_embeddings_arr` (Overall URL Embedding - first 5 values): {overall_url_emb_arr[0, :5]}")
                st.write(f"`overall_sims_to_queries_arr` (scores against synthetic queries): {overall_sims_to_queries_arr}")
                st.write(f"--- END DEBUG FOR URL: {current_url} ---")
                # +++ END DEBUG BLOCK +++
                
                passage_sims_to_queries_arr = cosine_similarity(calc_passage_embs, synthetic_query_embeddings_arr)
                url_passage_data_dict[current_url]["passage_similarities"] = passage_sims_to_queries_arr

                for sq_idx_val, syn_query_val in enumerate(synthetic_queries_list):
                    query_passage_sims_arr = passage_sims_to_queries_arr[:, sq_idx_val]
                    max_sim = np.max(query_passage_sims_arr) if query_passage_sims_arr.size > 0 else 0.0
                    avg_sim = np.mean(query_passage_sims_arr) if query_passage_sims_arr.size > 0 else 0.0
                    
                    all_url_metrics_list.append({
                        "URL": current_url, "Query": syn_query_val,
                        "Overall Similarity": overall_sims_to_queries_arr[sq_idx_val],
                        "Max Passage Sim.": max_sim,
                        "Avg. Passage Sim.": avg_sim,
                        "Num Passages": len(current_passages_list)
                    })
            else:
                st.warning(f"No passage embeddings for {current_url}.")
                for sq_idx_val, syn_query_val in enumerate(synthetic_queries_list):
                    all_url_metrics_list.append({"URL": current_url, "Query": syn_query_val, "Overall Similarity": 0.0, "Max Passage Sim.": 0.0, "Avg. Passage Sim.": 0.0, "Num Passages": 0})

    if not all_url_metrics_list: st.info("No data for summary."); st.stop()

    st.markdown("---"); st.subheader("📈 Overall Similarity & Passage Metrics Summary")
    df_summary_table = pd.DataFrame(all_url_metrics_list)
    st.dataframe(df_summary_table[['URL', 'Query', 'Overall Similarity', 'Max Passage Sim.', 'Avg. Passage Sim.', 'Num Passages']].style.format({
        "Overall Similarity": "{:.3f}", "Max Passage Sim.": "{:.3f}", "Avg. Passage Sim.": "{:.3f}"
    }), use_container_width=True, height=(min(len(df_summary_table) * 38 + 38, 700)))


    st.markdown("---"); st.subheader("📊 Visual: Overall URL vs. Synthetic Query Similarity")
    df_overall_sim_bar_chart = df_summary_table.drop_duplicates(subset=['URL', 'Query'])
    fig_bar_chart = px.bar(df_overall_sim_bar_chart, x="Query", y="Overall Similarity", color="URL", barmode="group",
                           title="Overall Webpage Similarity to Synthetic Queries", height=max(600, 100 * num_synthetic_queries_slider))
    fig_bar_chart.update_xaxes(tickangle=30, automargin=True, title_text=None)
    fig_bar_chart.update_yaxes(range=[0,1])
    fig_bar_chart.update_layout(legend_title_text='Webpage URL')
    st.plotly_chart(fig_bar_chart, use_container_width=True)

    st.markdown("---"); st.subheader("🔥 Passage Heatmaps vs. Synthetic Queries")
    for url_idx_val, (current_url_val, passage_data_val) in enumerate(url_passage_data_dict.items()):
        with st.expander(f"Heatmap for: {current_url_val}", expanded=(url_idx_val==0)):
            passages_list_val, passage_similarities_arr_val = passage_data_val["passages"], passage_data_val.get("passage_similarities")
            if passage_similarities_arr_val is None or passage_similarities_arr_val.size == 0: st.write("No passage similarity data."); continue
            
            hover_matrix_text = [[f"<b>P{i_p+1}</b> vs Q: '{synthetic_queries_list[j_q][:45]}...'<br>Sim: {passage_similarities_arr_val[i_p,j_q]:.3f}<hr>Txt: {passages_list_val[i_p][:120]}..."
                                  for j_q in range(passage_similarities_arr_val.shape[1])] for i_p in range(passage_similarities_arr_val.shape[0])]
            short_synthetic_queries = [q_str[:50] + '...' if len(q_str) > 50 else q_str for q_str in synthetic_queries_list]
            passage_labels_list = [f"P{i_p+1}" for i_p in range(len(passages_list_val))]
            heatmap_ticks_info = (list(range(0,len(passages_list_val),max(1,len(passages_list_val)//15))), [passage_labels_list[i_tk] for i_tk in range(0,len(passages_list_val),max(1,len(passages_list_val)//15))]) if len(passages_list_val)>25 else (passage_labels_list,passage_labels_list)

            fig_heatmap_obj = go.Figure(data=go.Heatmap(z=passage_similarities_arr_val.T, x=passage_labels_list, y=short_synthetic_queries, colorscale='Viridis',
                                                         hoverongaps=False, text=np.array(hover_matrix_text).T, hoverinfo='text', zmin=0, zmax=1))
            fig_heatmap_obj.update_layout(title=f"Passage Similarity for {current_url_val}",
                                           xaxis_title="Passages", yaxis_title="Queries", height=max(400,50*len(synthetic_queries_list)+100),
                                           yaxis_autorange='reversed', xaxis=dict(tickmode='array',tickvals=heatmap_ticks_info[0],ticktext=heatmap_ticks_info[1],automargin=True))
            st.plotly_chart(fig_heatmap_obj, use_container_width=True)

            if st.checkbox("Show highest/lowest similarity passages?", key=f"cb_passages_{current_url_val}"):
                for q_idx_val, sq_text_val in enumerate(synthetic_queries_list):
                    st.markdown(f"##### Query: '{sq_text_val}'")
                    if passage_similarities_arr_val.shape[1] > q_idx_val:
                        sims_for_query_arr = passage_similarities_arr_val[:, q_idx_val]
                        if sims_for_query_arr.size > 0:
                            idx_max_val, idx_min_val = np.argmax(sims_for_query_arr), np.argmin(sims_for_query_arr)
                            st.markdown(f"**Most Similar (P{idx_max_val+1} - Score: {sims_for_query_arr[idx_max_val]:.3f}):**"); st.caption(passages_list_val[idx_max_val])
                            st.markdown(f"**Least Similar (P{idx_min_val+1} - Score: {sims_for_query_arr[idx_min_val]:.3f}):**"); st.caption(passages_list_val[idx_min_val])

st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v1.7.2 (All Caching Removed)")
