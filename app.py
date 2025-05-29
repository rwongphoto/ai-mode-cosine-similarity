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
import time # For rate limiting
import random # For user-agent rotation

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Query Fan-Out Analyzer")

# --- Web Fetching Enhancements ---
REQUEST_INTERVAL = 2.0  # Seconds between requests
last_request_time = 0   # Global variable to track the last request time

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.2151.97",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)",
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

# --- NLTK Resource Download ---
@st.cache_resource
def download_nltk_resources():
    resource_id = 'tokenizers/punkt'
    try:
        nltk.data.find(resource_id)
    except LookupError:
        st.info(f"Downloading NLTK '{resource_id}' resource...")
        try:
            nltk.download('punkt', quiet=True)
            # st.sidebar.success(f"NLTK resource '{resource_id}' downloaded.") # Optional success message
        except Exception as e:
            st.sidebar.error(f"Failed to download NLTK '{resource_id}': {e}")
    except Exception as e:
        st.warning(f"Error checking NLTK resource '{resource_id}': {e}")
download_nltk_resources()

# --- Sentence Transformer Model ---
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedding_model = load_sentence_transformer()

# --- Session State Initialization for API Key ---
if "gemini_api_key_to_persist" not in st.session_state:
    st.session_state.gemini_api_key_to_persist = ""
if "gemini_api_configured" not in st.session_state:
    st.session_state.gemini_api_configured = False

# --- Sidebar API Key Configuration ---
st.sidebar.header("🔑 Gemini API Configuration")
api_key_from_input = st.sidebar.text_input(
    "Enter your Google Gemini API Key:",
    type="password",
    key="gemini_api_key_input_widget",
    value=st.session_state.gemini_api_key_to_persist
)

if st.sidebar.button("Set & Verify API Key"):
    if api_key_from_input:
        try:
            genai.configure(api_key=api_key_from_input)
            models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if not models:
                raise Exception("No usable models found. API key might be invalid or restricted.")
            st.session_state.gemini_api_key_to_persist = api_key_from_input
            st.session_state.gemini_api_configured = True
            st.sidebar.success("Gemini API Key configured successfully!")
        except Exception as e:
            st.session_state.gemini_api_key_to_persist = ""
            st.session_state.gemini_api_configured = False
            st.sidebar.error(f"API Key Configuration Failed: {str(e)[:300]}")
    else:
        st.sidebar.warning("Please enter an API Key.")

if st.session_state.get("gemini_api_configured", False):
    st.sidebar.markdown("✅ Gemini API: **Configured**")
    # Ensure genai is configured if flag is true but somehow genai lost its internal config
    # This can happen if Streamlit reruns the script in a way that doesn't preserve genai's global state
    # (though less common for genai.configure compared to other types of global state).
    # Calling configure again is idempotent.
    if st.session_state.gemini_api_key_to_persist:
        try:
            genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
        except Exception:
            st.session_state.gemini_api_configured = False # Mark as not configured if it fails
            st.sidebar.markdown("⚠️ Gemini API: **Re-configuration failed**. Please re-verify.")


else:
    st.sidebar.markdown("⚠️ Gemini API: **Not Configured**. Please enter your key and click 'Set & Verify'.")

# --- Helper Functions ---
@st.cache_data(show_spinner="Fetching and cleaning URL content...")
def fetch_and_parse_url(url):
    enforce_rate_limit()
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        tags_to_remove = ['nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption']
        for tag_name in tags_to_remove:
            for tag_element in soup.find_all(tag_name):
                tag_element.decompose()
        selectors_to_remove = [
            "[class*='menu']", "[class*='navbar']", "[id*='nav']", "[class*='header']", "[id*='header']",
            "[class*='footer']", "[id*='footer']", "[class*='sidebar']", "[id*='sidebar']",
            "[class*='widget']", "[class*='cookie']", "[id*='cookie']", "[class*='consent']",
            "[class*='banner']", "[id*='banner']", "[class*='popup']", "[id*='popup']",
            "[class*='social']", "[class*='share']", "[class*='breadcrumb']", "[class*='pagination']",
            "[class*='advert']", "[class*='ads']", "[class*='sponsor']",
            "[aria-label*='menu']", "[role='navigation']", "[role='banner']",
            "[role='contentinfo']", "[role='complementary']"
        ]
        for selector in selectors_to_remove:
            try:
                for element in soup.select(selector):
                    # More refined check to avoid removing main content
                    is_likely_main_content_container = element.name in ['body', 'main', 'article'] or \
                                                       any(cls in element.get('class', []) for cls in ['content', 'main-content', 'article-body']) or \
                                                       element.get('id', '') in ['content', 'main-content']
                    if not is_likely_main_content_container or element.name not in ['body', 'main', 'article']:
                        if element.parent: element.decompose()
            except Exception: pass

        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        if not text.strip(): return None
        return text
    except requests.exceptions.Timeout: st.error(f"Timeout fetching {url}."); return None
    except requests.exceptions.RequestException as e: st.error(f"Error fetching {url}: {e}"); return None
    except Exception as e: st.error(f"Error parsing {url}: {e}"); return None

@st.cache_data(show_spinner="Splitting text into passages...")
def split_text_into_passages(text, sentences_per_passage=7, sentence_overlap=2):
    if not text: return []
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 3]
        if not sentences: sentences = [s.strip() for s in text.split('\n\n') if s.strip() and len(s.split()) > 5]
    if not sentences: return []
    passages, effective_step = [], max(1, sentences_per_passage - sentence_overlap)
    for i in range(0, len(sentences), effective_step):
        passage_text = " ".join(sentences[i : i + sentences_per_passage])
        if passage_text.strip() and len(passage_text.split()) > 10: passages.append(passage_text)
    return [p for p in passages if p.strip()]

@st.cache_data
def get_embeddings(_texts):
    if not _texts: return np.array([])
    return embedding_model.encode(_texts)

@st.cache_data(show_spinner="Generating synthetic queries with Gemini...")
def generate_synthetic_queries(user_query, num_queries=7):
    if not st.session_state.get("gemini_api_configured", False):
        st.error("Gemini API not configured.")
        return []

    model_name = "gemini-2.5-flash-preview-05-20"
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Could not initialize Gemini model ({model_name}): {e}")
        return []

    # THIS IS THE DETAILED FAN-OUT PROMPT
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
            for prefix in ["```python", "```json", "```"]:
                if content_text.startswith(prefix):
                    content_text = content_text.split(prefix, 1)[1].rsplit("```", 1)[0].strip()
                    break
            if not content_text.startswith('['):
                queries = [re.sub(r'^\s*[-\*\d\.]+\s*', '', q.strip().strip('"\'')) for q in content_text.split('\n') if q.strip() and len(q.strip()) > 3]
            else:
                queries = ast.literal_eval(content_text)
            if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
                raise ValueError("Parsed result is not a list of strings.")
            return [q for q in queries if q.strip()]
        except (SyntaxError, ValueError) as e:
            st.error(f"Error parsing Gemini response: {e}. Raw: {content_text[:300]}...")
            extracted_queries = [re.sub(r'^[\s\-\*\d\.]+\s*', '', line.strip().strip('"\'')) for line in content_text.split('\n') if line.strip() and len(line.strip()) > 3]
            if extracted_queries: st.warning("Used fallback parsing."); return extracted_queries
            return []
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        if hasattr(e, 'message'): st.error(f"Gemini API Message: {e.message}")
        return []

# --- Main Application UI ---
st.title("🌐 Webpage Query Fan-Out Analyzer")
st.markdown("""
This app fetches content from URLs, generates synthetic queries using Google Gemini based on your initial search,
and visualizes how webpages align with these diverse queries. **Requires your own Google Gemini API Key.**
""")

st.sidebar.divider()
st.sidebar.header("⚙️ Analysis Configuration")
initial_query = st.sidebar.text_input("Enter your initial search query:", "benefits of server-side rendering")
url_inputs_str = st.sidebar.text_area("Enter URLs (one per line):",
                                      "https://vercel.com/blog/react-server-components\nhttps://www.patterns.dev/posts/react-server-components/",
                                      height=120)
num_synthetic_queries = st.sidebar.slider("Number of Synthetic Queries:", 3, 10, 5)
st.sidebar.subheader("Passage Creation Settings:")
sentences_per_passage = st.sidebar.slider("Sentences per Passage:", 2, 20, 7)
sentence_overlap = st.sidebar.slider("Sentence Overlap:", 0, 10, 2)

analyze_button_disabled = not st.session_state.get("gemini_api_configured", False)
analyze_button_tooltip = "Configure Gemini API Key first." if analyze_button_disabled else ""

if st.sidebar.button("🚀 Analyze Content", type="primary", disabled=analyze_button_disabled, help=analyze_button_tooltip):
    if not initial_query or not url_inputs_str:
        st.warning("Please enter an initial query and at least one URL.")
    else:
        urls = [url.strip() for url in url_inputs_str.split('\n') if url.strip()]
        if sentence_overlap >= sentences_per_passage:
            st.warning("Sentence overlap adjusted.")
            sentence_overlap = max(0, sentences_per_passage - 1)

        synthetic_queries = generate_synthetic_queries(initial_query, num_synthetic_queries)
        if not synthetic_queries:
            st.error("Failed to generate synthetic queries. Check API or query.")
        else:
            st.subheader("🤖 Generated Synthetic Queries")
            st.markdown(f"From: **'{initial_query}'**")
            st.expander("View Generated Queries").json(synthetic_queries)
            synthetic_query_embeddings = get_embeddings(synthetic_queries)
            all_url_data, url_passage_data = [], {}

            with st.spinner(f"Processing {len(urls)} URLs..."):
                for i, url in enumerate(urls):
                    st.markdown(f"--- \n#### Processing URL {i+1}: {url}")
                    page_text = fetch_and_parse_url(url)
                    if not page_text or len(page_text.strip()) < 50:
                        st.warning(f"Insufficient text from {url}. Skipping."); continue
                    passages = split_text_into_passages(page_text, sentences_per_passage, sentence_overlap)
                    if not passages:
                        st.warning(f"No passages from {url}. Skipping."); continue
                    
                    url_passage_data[url] = {"passages": passages, "embeddings": get_embeddings(passages)}
                    passage_embeddings = url_passage_data[url]["embeddings"]
                    if passage_embeddings.ndim == 1: passage_embeddings = passage_embeddings.reshape(1, -1)

                    if passage_embeddings.size > 0:
                        overall_url_emb = np.mean(passage_embeddings, axis=0).reshape(1, -1)
                        url_q_sims = cosine_similarity(overall_url_emb, synthetic_query_embeddings)[0]
                        for sq_idx, score in enumerate(url_q_sims):
                            all_url_data.append({"URL": url, "Synthetic Query": synthetic_queries[sq_idx], "Similarity": score})
                    else: st.warning(f"No passage embeddings for {url}.")

            if not all_url_data: st.info("No similarity data. Check URLs/parsing.")
            else:
                st.markdown("---"); st.subheader("📊 URL vs. Synthetic Query Similarity")
                df_url_sim = pd.DataFrame(all_url_data)
                df_url_sim['URL_Short'] = df_url_sim['URL'].apply(lambda x: x.split('//')[-1].split('/')[0][:30] + ('...' if len(x.split('//')[-1].split('/')[0]) > 30 else ''))
                fig_bar = px.bar(df_url_sim, x="Synthetic Query", y="Similarity", color="URL_Short", barmode="group",
                                 title="Overall Webpage Similarity to Synthetic Queries", height=max(500, 90 * num_synthetic_queries))
                fig_bar.update_xaxes(tickangle=25, automargin=True); fig_bar.update_yaxes(range=[0,1])
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("---"); st.subheader("🔥 Passage Heatmaps vs. Synthetic Queries")
                for url_idx, (url, p_data) in enumerate(url_passage_data.items()):
                    with st.expander(f"Heatmap for: {url}", expanded=(url_idx==0)):
                        passages, passage_embs = p_data["passages"], p_data["embeddings"]
                        if passage_embs is None or passage_embs.size == 0: st.write("No embeddings."); continue
                        if passage_embs.ndim == 1: passage_embs = passage_embs.reshape(1, -1)
                        sq_embs_r = synthetic_query_embeddings.reshape(1, -1) if synthetic_query_embeddings.ndim == 1 else synthetic_query_embeddings
                        
                        passage_q_sims = cosine_similarity(passage_embs, sq_embs_r)
                        hover_matrix = [[f"<b>P{i+1}</b> vs Q: '{synthetic_queries[j][:45]}...'<br>Sim: {passage_q_sims[i,j]:.3f}<hr>Txt: {passages[i][:120]}..."
                                         for j in range(passage_q_sims.shape[1])] for i in range(passage_q_sims.shape[0])]
                        short_sq = [q[:50] + '...' if len(q) > 50 else q for q in synthetic_queries]
                        p_labels = [f"P{i+1}" for i in range(len(passages))]
                        ticks = (list(range(0,len(passages),max(1,len(passages)//15))),[p_labels[i] for i in range(0,len(passages),max(1,len(passages)//15))]) if len(passages)>25 else (p_labels,p_labels)

                        fig_heat = go.Figure(data=go.Heatmap(z=passage_q_sims.T, x=p_labels, y=short_sq, colorscale='Viridis',
                                                             hoverongaps=False, text=np.array(hover_matrix).T, hoverinfo='text', zmin=0, zmax=1))
                        fig_heat.update_layout(title=f"Passage Similarity for {url.split('//')[-1].split('/')[0]}",
                                               xaxis_title="Passages", yaxis_title="Queries", height=max(400,50*len(synthetic_queries)+100),
                                               yaxis_autorange='reversed', xaxis=dict(tickmode='array',tickvals=ticks[0],ticktext=ticks[1],automargin=True))
                        st.plotly_chart(fig_heat, use_container_width=True)

                        if st.checkbox("Show highest/lowest similarity passages?", key=f"cb_{url}"):
                            for q_idx, sq_txt in enumerate(synthetic_queries):
                                st.markdown(f"##### Query: '{sq_txt}'")
                                if passage_q_sims.shape[1] > q_idx:
                                    sims_for_q = passage_q_sims[:, q_idx]
                                    if sims_for_q.size > 0:
                                        idx_max, idx_min = np.argmax(sims_for_q), np.argmin(sims_for_q)
                                        st.markdown(f"**Most Similar (P{idx_max+1} - Score: {sims_for_q[idx_max]:.3f}):**")
                                        st.caption(passages[idx_max])
                                        st.markdown(f"**Least Similar (P{idx_min+1} - Score: {sims_for_q[idx_min]:.3f}):**")
                                        st.caption(passages[idx_min])
st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v1.3")
