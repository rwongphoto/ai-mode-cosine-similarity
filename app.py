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
import nltk # Make sure nltk is imported
import ast

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Query Fan-Out Analyzer")

# --- NLTK Resource Download ---
@st.cache_resource
def download_nltk_resources():
    resource_id = 'tokenizers/punkt'
    try:
        # Check if the resource is already available
        nltk.data.find(resource_id)
        # st.sidebar.info(f"NLTK resource '{resource_id}' found.") # Optional: for debugging
    except LookupError:
        # If not found, download it
        st.info(f"Downloading NLTK '{resource_id}' resource...")
        try:
            nltk.download('punkt', quiet=True)
            st.sidebar.success(f"NLTK resource '{resource_id}' downloaded successfully.")
        except Exception as e: # Catch any error during download itself
            st.sidebar.error(f"Failed to download NLTK '{resource_id}': {e}")
            st.warning(f"Could not download NLTK '{resource_id}'. Sentence tokenization might be impaired.")
    except Exception as e: # Catch other unexpected errors
        st.warning(f"An error occurred while checking for NLTK resource '{resource_id}': {e}")

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

# --- Session State Initialization for API Key ---
if "gemini_api_key_to_persist" not in st.session_state:
    st.session_state.gemini_api_key_to_persist = ""
if "gemini_api_configured" not in st.session_state:
    st.session_state.gemini_api_configured = False

# The genai.configure() call will happen when the user clicks "Set & Verify API Key".
# Our st.session_state.gemini_api_configured flag will then reflect the status.


# --- Sidebar API Key Configuration ---
st.sidebar.header("🔑 Gemini API Configuration")
api_key_from_input = st.sidebar.text_input(
    "Enter your Google Gemini API Key:",
    type="password",
    key="gemini_api_key_input_widget", # Unique key for this widget
    value=st.session_state.gemini_api_key_to_persist # Pre-fill if key exists in session
)

if st.sidebar.button("Set & Verify API Key"):
    if api_key_from_input:
        try:
            genai.configure(api_key=api_key_from_input)
            # Verify by listing models (lightweight check)
            models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if not models:
                raise Exception("No usable models found. API key might be invalid, restricted, or models are unavailable in your region/project.")
            
            st.session_state.gemini_api_key_to_persist = api_key_from_input
            st.session_state.gemini_api_configured = True
            st.sidebar.success("Gemini API Key configured successfully!")
        except Exception as e:
            st.session_state.gemini_api_key_to_persist = "" # Clear invalid key attempt from persisting
            st.session_state.gemini_api_configured = False
            st.sidebar.error(f"API Key Configuration Failed: {str(e)[:300]}")
    else:
        st.sidebar.warning("Please enter an API Key.")

if st.session_state.get("gemini_api_configured", False):
    st.sidebar.markdown("✅ Gemini API: **Configured**")
else:
    st.sidebar.markdown("⚠️ Gemini API: **Not Configured**. Please enter your key and click 'Set & Verify'.")


# --- Helper Functions ---
@st.cache_data(show_spinner="Fetching and cleaning URL content...")
def fetch_and_parse_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        tags_to_remove = ['nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption']
        for tag_name in tags_to_remove:
            for tag_element in soup.find_all(tag_name):
                tag_element.decompose()

        selectors_to_remove = [
            "[class*='menu']", "[class*='navbar']", "[class*='nav-']", "[id*='nav']",
            "[class*='header']", "[id*='header']",
            "[class*='footer']", "[id*='footer']",
            "[class*='sidebar']", "[id*='sidebar']", "[class*='widget']",
            "[class*='cookie']", "[id*='cookie']", "[class*='consent']", "[id*='consent']",
            "[class*='banner']", "[id*='banner']", "[class*='popup']", "[id*='popup']",
            "[class*='social']", "[class*='share']", "[class*='breadcrumb']", "[class*='pagination']",
            "[class*='advert']", "[class*='ads']", "[class*='sponsor']",
            "[aria-label*='menu']", "[aria-label*='navigation']", "[role='navigation']",
            "[role='banner']", "[role='contentinfo']", "[role='complementary']"
        ]
        for selector in selectors_to_remove:
            try:
                for element in soup.select(selector):
                    if element.parent: element.decompose()
            except Exception: pass
        
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        return text
    except requests.RequestException as e:
        st.error(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Error parsing content from {url}: {e}")
        return None

@st.cache_data(show_spinner="Splitting text into passages...")
def split_text_into_passages(text, sentences_per_passage=7, sentence_overlap=2):
    if not text: return []
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        st.warning(f"NLTK sentence tokenization failed: {e}. Falling back to simple splitting.")
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.split()) > 3]
        if not sentences: sentences = [s.strip() for s in text.split('\n\n') if s.strip() and len(s.split()) > 5]

    if not sentences: return []
    passages = []
    effective_step = max(1, sentences_per_passage - sentence_overlap)
    for i in range(0, len(sentences), effective_step):
        passage_text = " ".join(sentences[i : i + sentences_per_passage])
        if passage_text.strip() and len(passage_text.split()) > 10:
            passages.append(passage_text)
    return [p for p in passages if p.strip()]

@st.cache_data
def get_embeddings(_texts):
    if not _texts: return np.array([])
    return embedding_model.encode(_texts)

@st.cache_data(show_spinner="Generating synthetic queries with Gemini...")
def generate_synthetic_queries(user_query, num_queries=7):
    # Assumes genai is configured globally by the sidebar logic
    if not st.session_state.get("gemini_api_configured", False):
        st.error("Gemini API is not configured. Cannot generate queries.")
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
    Aim for a mix of: Related, Implicit, Comparative, Recent (Hypothetical), Personalized (Hypothetical), Reformulation, Entity-Expanded Queries.
    CRITICAL INSTRUCTIONS:
    - Ensure diversity across query categories and semantic zones.
    - Do NOT number or prefix queries.
    - Return ONLY a Python-parseable list of strings. Example: ["query 1", "query 2"]
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
            
            if not content_text.startswith('['): # If not a list string, try splitting by newline
                queries = [q.strip().replace('"', '').replace("'", "") for q in content_text.split('\n') if q.strip()]
                queries = [re.sub(r'^\s*[-\*\d\.]+\s*', '', q) for q in queries if q and len(q) > 3]
            else:
                queries = ast.literal_eval(content_text) # Main parsing attempt
            
            if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
                raise ValueError("Parsed result is not a list of strings.")
            return [q for q in queries if q.strip()]
        
        except (SyntaxError, ValueError) as e: # Fallback parsing
            st.error(f"Error parsing Gemini's response: {e}. Raw response: \n{content_text[:500]}...")
            lines = content_text.split('\n')
            extracted_queries = [re.sub(r'^[\s\-\*\d\.]+\s*', '', line.strip().strip('"\'')) for line in lines if line.strip() and len(line.strip()) > 3]
            if extracted_queries:
                st.warning("Used fallback parsing for synthetic queries.")
                return extracted_queries
            return []
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        if hasattr(e, 'message'): st.error(f"Gemini API Error Message: {e.message}")
        return []

# --- Main Application UI ---
st.title("🌐 Webpage Query Fan-Out Analyzer")
st.markdown("""
This app fetches cleaned content from URLs, tokenizes it into semantic passages, generates synthetic queries 
based on your initial search query using Google Gemini, and visualizes content alignment.
**Important:** You need to provide your own Google Gemini API Key in the sidebar.
""")

st.sidebar.divider()
st.sidebar.header("⚙️ Analysis Configuration")
initial_query = st.sidebar.text_input("Enter your initial search query:", "understanding React Server Components")
url_inputs_str = st.sidebar.text_area("Enter URLs (one per line):", 
                                      "https://vercel.com/blog/react-server-components\nhttps://www.patterns.dev/posts/react-server-components/\nhttps://nextjs.org/docs/app/building-your-application/rendering/server-components", 
                                      height=150)
num_synthetic_queries = st.sidebar.slider("Number of Synthetic Queries:", min_value=3, max_value=10, value=5)

st.sidebar.subheader("Passage Creation Settings:")
sentences_per_passage = st.sidebar.slider("Sentences per Passage:", min_value=2, max_value=20, value=7)
sentence_overlap = st.sidebar.slider("Sentence Overlap:", min_value=0, max_value=10, value=2)

# Disable button if API key not configured
analyze_button_disabled = not st.session_state.get("gemini_api_configured", False)
analyze_button_tooltip = "Please configure your Gemini API Key in the sidebar first." if analyze_button_disabled else ""

if st.sidebar.button("🚀 Analyze Content", type="primary", disabled=analyze_button_disabled, help=analyze_button_tooltip):
    if not initial_query:
        st.warning("Please enter an initial search query.")
    elif not url_inputs_str:
        st.warning("Please enter at least one URL.")
    # API key configured check is implicitly handled by the button's disabled state,
    # but an explicit check here is good practice if called programmatically.
    elif not st.session_state.get("gemini_api_configured", False):
         st.error("Gemini API is not configured. Please set your API key in the sidebar.")
    else:
        urls = [url.strip() for url in url_inputs_str.split('\n') if url.strip()]
        
        if sentence_overlap >= sentences_per_passage:
            st.warning("Sentence overlap adjusted to be less than sentences per passage.")
            sentence_overlap = max(0, sentences_per_passage - 1)
        
        synthetic_queries = generate_synthetic_queries(initial_query, num_synthetic_queries)

        if not synthetic_queries:
            st.error("Failed to generate synthetic queries. Check Gemini API status or your query.")
        else:
            st.subheader("🤖 Generated Synthetic Queries:")
            st.markdown(f"Based on: **'{initial_query}'**")
            st.expander("View Synthetic Queries").json(synthetic_queries) # Display in a compact way
            
            synthetic_query_embeddings = get_embeddings(synthetic_queries)
            all_url_data = []
            url_passage_data = {} 

            with st.spinner(f"Processing {len(urls)} URLs... This may take a moment."):
                for i, url in enumerate(urls):
                    st.markdown(f"--- \n### Processing URL {i+1}: {url}")
                    page_text = fetch_and_parse_url(url)
                    if not page_text or len(page_text.strip()) < 100:
                        st.warning(f"Insufficient text from {url}. Skipping.")
                        continue

                    passages = split_text_into_passages(page_text, sentences_per_passage, sentence_overlap)
                    if not passages:
                        st.warning(f"No passages extracted from {url}. Skipping.")
                        continue
                    
                    url_passage_data[url] = {"passages": passages, "embeddings": None}
                    passage_embeddings = get_embeddings(passages)
                    url_passage_data[url]["embeddings"] = passage_embeddings

                    if passage_embeddings.ndim == 1: passage_embeddings = passage_embeddings.reshape(1, -1)
                    
                    if passage_embeddings.size > 0:
                        overall_url_embedding = np.mean(passage_embeddings, axis=0).reshape(1, -1)
                        url_query_similarities = cosine_similarity(overall_url_embedding, synthetic_query_embeddings)[0]
                        for sq_idx, sim_score in enumerate(url_query_similarities):
                            all_url_data.append({
                                "URL": url, "Synthetic Query": synthetic_queries[sq_idx], "Similarity": sim_score
                            })
                    else:
                        st.warning(f"No passage embeddings for {url}.")

            if not all_url_data:
                st.info("No similarity data to display. Check URL fetching/parsing or embeddings.")
            else:
                st.markdown("---")
                st.subheader("📊 Cosine Similarity: URLs vs. Synthetic Queries")
                df_url_similarity = pd.DataFrame(all_url_data)
                df_url_similarity['URL_Short'] = df_url_similarity['URL'].apply(lambda x: x.split('//')[-1].split('/')[0][:30] + ('...' if len(x.split('//')[-1].split('/')[0]) > 30 else ''))
                fig_bar = px.bar(df_url_similarity, x="Synthetic Query", y="Similarity", color="URL_Short",
                                 barmode="group", title="Overall Webpage Similarity to Synthetic Queries",
                                 labels={"Similarity": "Cosine Similarity Score", "Synthetic Query": "Synthetic Query"},
                                 height=max(500, 100 * num_synthetic_queries))
                fig_bar.update_xaxes(tickangle=30, automargin=True)
                fig_bar.update_yaxes(range=[0,1])
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("---")
                st.subheader("🔥 Heatmaps: URL Passage Similarity to Synthetic Queries")
                for url_idx, (url, data) in enumerate(url_passage_data.items()):
                    with st.expander(f"Heatmap for: {url}", expanded=(url_idx==0)):
                        passages, passage_embeddings = data["passages"], data["embeddings"]
                        if passage_embeddings is None or passage_embeddings.size == 0:
                            st.write("No passage embeddings for heatmap."); continue
                        
                        if passage_embeddings.ndim == 1: passage_embeddings = passage_embeddings.reshape(1, -1)
                        s_q_embeddings_r = synthetic_query_embeddings.reshape(1, -1) if synthetic_query_embeddings.ndim == 1 else synthetic_query_embeddings
                        
                        passage_query_similarities = cosine_similarity(passage_embeddings, s_q_embeddings_r)
                        hover_text_matrix = [[
                            f"<b>Passage {i+1}</b> vs Q: '<i>{synthetic_queries[j][:50]}...</i>'<br>Sim: {passage_query_similarities[i,j]:.3f}<hr>Preview: {passages[i][:150]}..."
                            for j in range(passage_query_similarities.shape[1])]
                            for i in range(passage_query_similarities.shape[0])]

                        short_sq = [q[:60] + '...' if len(q) > 60 else q for q in synthetic_queries]
                        p_labels = [f"P{i+1}" for i in range(len(passages))]
                        tickvals, ticktext = (list(range(0, len(passages), max(1, len(passages)//20))), [p_labels[i] for i in range(0, len(passages), max(1, len(passages)//20))]) if len(passages) > 30 else (p_labels, p_labels)

                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=passage_query_similarities.T, x=p_labels, y=short_sq, colorscale='Viridis',
                            hoverongaps=False, text=np.array(hover_text_matrix).T, hoverinfo='text', zmin=0, zmax=1 ))
                        fig_heatmap.update_layout(
                            title=f"Passage Similarity for {url.split('//')[-1].split('/')[0]}",
                            xaxis_title="Content Passages (P#)", yaxis_title="Synthetic Queries",
                            height=max(400, 60 * len(synthetic_queries) + 150), yaxis_autorange='reversed',
                            xaxis=dict(tickmode='array', tickvals=tickvals, ticktext=ticktext, automargin=True))
                        st.plotly_chart(fig_heatmap, use_container_width=True)

                        if st.checkbox("Show passages with highest/lowest similarity?", key=f"show_passages_{url}"):
                            for q_idx, syn_query in enumerate(synthetic_queries):
                                st.markdown(f"#### Query: '{syn_query}'")
                                if passage_query_similarities.shape[1] > q_idx:
                                    sims_for_q = passage_query_similarities[:, q_idx]
                                    if sims_for_q.size > 0:
                                        idx_max, idx_min = np.argmax(sims_for_q), np.argmin(sims_for_q)
                                        st.markdown(f"**Most Similar (P{idx_max+1} - Score: {sims_for_q[idx_max]:.3f}):**")
                                        st.caption(passages[idx_max])
                                        st.markdown(f"**Least Similar (P{idx_min+1} - Score: {sims_for_q[idx_min]:.3f}):**")
                                        st.caption(passages[idx_min])
                                    else: st.write("No similarity scores.")
                                else: st.write(f"Not enough data for query: {syn_query}")
st.sidebar.divider()
st.sidebar.info("Query Fan-Out Analyzer | v1.2")
