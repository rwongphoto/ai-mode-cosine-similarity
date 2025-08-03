import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import random
import base64
import json
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from bs4 import BeautifulSoup
from selenium_stealth import stealth
from sklearn.cluster import KMeans
from collections import Counter
import networkx as nx

# Google Cloud NLP libraries
from google.cloud import language_v1
from google.oauth2 import service_account

st.set_page_config(layout="wide", page_title="Entity Gap Analysis Tool")

# --- Session State Initialization ---
if "processing" not in st.session_state: st.session_state.processing = False
if "selenium_driver_instance" not in st.session_state: st.session_state.selenium_driver_instance = None
if "gcp_nlp_configured" not in st.session_state: st.session_state.gcp_nlp_configured = False
if "gcp_credentials_info" not in st.session_state: st.session_state.gcp_credentials_info = None
if "zyte_api_key_to_persist" not in st.session_state: st.session_state.zyte_api_key_to_persist = ""
if "zyte_api_configured" not in st.session_state: st.session_state.zyte_api_configured = False
if "entity_analysis_results" not in st.session_state: st.session_state.entity_analysis_results = None
if "content_passages" not in st.session_state: st.session_state.content_passages = None
if "embedding_model" not in st.session_state: st.session_state.embedding_model = None
if "content_relationships" not in st.session_state: st.session_state.content_relationships = None
if "gemini_api_configured" not in st.session_state: st.session_state.gemini_api_configured = False
if "gemini_api_key_to_persist" not in st.session_state: st.session_state.gemini_api_key_to_persist = ""

REQUEST_INTERVAL = 3.0
last_request_time = 0

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def enforce_rate_limit():
    global last_request_time
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

# --- Sidebar API Configuration ---
st.sidebar.header("🔑 API Configuration")
with st.sidebar.expander("Google Cloud NLP API", expanded=not st.session_state.gcp_nlp_configured):
    uploaded_gcp_key = st.file_uploader("Upload Google Cloud Service Account JSON", type="json", help="Upload the JSON key file for a service account with 'Cloud Natural Language API User' role.", disabled=st.session_state.processing, key="gcp_key_uploader")
    if uploaded_gcp_key is not None:
        try:
            credentials_info = json.load(uploaded_gcp_key)
            if "project_id" in credentials_info and "private_key" in credentials_info:
                st.session_state.gcp_credentials_info = credentials_info
                st.session_state.gcp_nlp_configured = True
                st.success(f"GCP Key for project '{credentials_info['project_id']}' loaded!")
            else: st.error("Invalid JSON key file format."); st.session_state.gcp_nlp_configured = False; st.session_state.gcp_credentials_info = None
        except Exception as e: st.error(f"Failed to process GCP key file: {e}"); st.session_state.gcp_nlp_configured = False; st.session_state.gcp_credentials_info = None
with st.sidebar.expander("Zyte API (Optional)", expanded=False):
    zyte_api_key_input = st.text_input("Enter Zyte API Key:", type="password", value=st.session_state.get("zyte_api_key_to_persist", ""), disabled=st.session_state.processing, key="zyte_api_key_input")
    if st.button("Set & Verify Zyte Key", disabled=st.session_state.processing, key="zyte_verify_btn"):
        if zyte_api_key_input:
            try:
                response = requests.post("https://api.zyte.com/v1/extract", auth=(zyte_api_key_input, ''), json={'url': 'https://toscrape.com/', 'httpResponseBody': True}, timeout=20)
                if response.status_code == 200: st.session_state.zyte_api_key_to_persist = zyte_api_key_input; st.session_state.zyte_api_configured = True; st.success("Zyte API Key Configured!"); st.rerun()
                else: st.session_state.zyte_api_configured = False; st.error(f"Zyte Key Failed. Status: {response.status_code}")
            except Exception as e: st.session_state.zyte_api_configured = False; st.error(f"Zyte API Request Failed: {str(e)[:200]}")
        else: st.warning("Please enter Zyte API Key.")
with st.sidebar.expander("Gemini API (Optional)", expanded=False):
    gemini_api_key_input = st.text_input("Enter Gemini API Key:", type="password", value=st.session_state.get("gemini_api_key_to_persist", ""), disabled=st.session_state.processing, help="Required for AI-powered implementation strategies and content analysis", key="gemini_api_key_input")
    if st.button("Set & Verify Gemini Key", disabled=st.session_state.processing, key="gemini_verify_btn"):
        if gemini_api_key_input:
            try:
                import google.generativeai as genai; genai.configure(api_key=gemini_api_key_input); model = genai.GenerativeModel("gemini-1.5-flash"); test_response = model.generate_content("Hello, respond with 'API key works'")
                if test_response and test_response.text: st.session_state.gemini_api_key_to_persist = gemini_api_key_input; st.session_state.gemini_api_configured = True; st.success("Gemini API Key Configured!"); st.rerun()
                else: st.session_state.gemini_api_configured = False; st.error("Gemini API Key verification failed - no response received")
            except Exception as e: st.session_state.gemini_api_configured = False; st.error(f"Gemini API Key verification failed: {str(e)}")
        else: st.warning("Please enter a Gemini API Key.")
    if st.session_state.get('gemini_api_configured', False):
        if st.button("🗑️ Clear Gemini Key", disabled=st.session_state.processing, key="gemini_clear_btn"): st.session_state.gemini_api_key_to_persist = ""; st.session_state.gemini_api_configured = False; st.success("Gemini API key cleared!"); st.rerun()
st.sidebar.markdown("---"); st.sidebar.markdown(f"✅ Google NLP API: **{'Required - Configured' if st.session_state.gcp_nlp_configured else 'Required - Not Configured'}**"); st.sidebar.markdown(f"🔧 Zyte API: **{'Configured' if st.session_state.zyte_api_configured else 'Optional - Not Configured'}**"); st.sidebar.markdown(f"🤖 Gemini API: **{'Configured' if st.session_state.get('gemini_api_configured', False) else 'Optional - Not Configured'}**")

# --- Core Functions ---
def is_number_entity(entity_name):
    if not entity_name: return True
    cleaned = re.sub(r'[,\s\-\.]', '', entity_name)
    if cleaned.isdigit(): return True
    if entity_name.strip().endswith('%') and re.sub(r'[%,\s\-\.]', '', entity_name).isdigit(): return True
    if re.match(r'^\d{4}$', cleaned): return True
    digit_count = sum(1 for char in entity_name if char.isdigit())
    total_chars = len(re.sub(r'\s', '', entity_name))
    if total_chars > 0 and (digit_count / total_chars) > 0.7: return True
    if len(entity_name.strip()) <= 4 and any(char.isdigit() for char in entity_name): return True
    return False

@st.cache_data(show_spinner="Extracting entities...")
def extract_entities_with_google_nlp(text: str, _credentials_info: dict):
    if not _credentials_info or not text: return {}
    try:
        credentials = service_account.Credentials.from_service_account_info(_credentials_info); client = language_v1.LanguageServiceClient(credentials=credentials)
        max_bytes = 900000; text_bytes = text.encode('utf-8')
        if len(text_bytes) > max_bytes: text = text_bytes[:max_bytes].decode('utf-8', 'ignore'); st.warning("Text was truncated to fit Google NLP API size limit.")
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT); response = client.analyze_entities(document=document, encoding_type=language_v1.EncodingType.UTF8)
        entities_dict = {}
        for entity in response.entities:
            entity_name = entity.name.strip()
            if is_number_entity(entity_name): continue
            key = entity_name.lower()
            if key not in entities_dict or entity.salience > entities_dict[key]['salience']: entities_dict[key] = {'name': entity_name, 'type': language_v1.Entity.Type(entity.type_).name, 'salience': entity.salience, 'mentions': len(entity.mentions)}
        return entities_dict
    except Exception as e: st.error(f"Google Cloud NLP API Error: {e}"); return {}

@st.cache_resource
def load_embedding_model():
    try: return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e: st.error(f"Failed to load embedding model: {e}"); return None

def calculate_entity_query_relevance(entity_name, query_text, model):
    if not model or not entity_name or not query_text: return 0.0
    try: entity_embedding = model.encode([entity_name]); query_embedding = model.encode([query_text]); similarity = cosine_similarity(entity_embedding, query_embedding)[0][0]; return float(similarity)
    except Exception as e: st.warning(f"Failed to calculate similarity for '{entity_name}': {e}"); return 0.0

@st.cache_data(show_spinner="Analyzing content structure...")
def cluster_content_passages(_passages, _model, num_clusters=5):
    if not _passages or not _model or len(_passages) < num_clusters: return []
    try:
        passage_embeddings = _model.encode(_passages); actual_num_clusters = min(num_clusters, len(_passages))
        kmeans = KMeans(n_clusters=actual_num_clusters, random_state=0, n_init='auto').fit(passage_embeddings)
        clusters = []
        for i in range(actual_num_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) == 0: continue
            cluster_passages_text = " ".join([_passages[j] for j in cluster_indices]); words = re.findall(r'\b\w{4,15}\b', cluster_passages_text.lower())
            stop_words = set(['this', 'that', 'with', 'your', 'from', 'what', 'which', 'will', 'also', 'just', 'they', 'have', 'been', 'about', 'more', 'like', 'content', 'using', 'server', 'client'])
            words = [word for word in words if word not in stop_words]; most_common = [word for word, count in Counter(words).most_common(3)]; cluster_name = " ".join(most_common).title() if most_common else f"Topic {i+1}"
            clusters.append({"id": f"cluster_{i}", "name": f"Section: {cluster_name}", "centroid": kmeans.cluster_centers_[i], "passage_indices": cluster_indices})
        return clusters
    except Exception as e: st.error(f"Failed to cluster content: {e}"); return []

@st.cache_data(show_spinner="Mapping entities to content structure...")
def calculate_content_relationships(_primary_entities, _missing_entities, _clusters, _model):
    if not _model or not _clusters: return {}
    primary_entity_list = [e for e in _primary_entities if isinstance(e, dict) and 'Entity' in e]; missing_entity_list = [e for e in _missing_entities if isinstance(e, dict) and 'Entity' in e]
    all_entities = primary_entity_list + missing_entity_list
    if not all_entities: return {}
    entity_names = [e['Entity'] for e in all_entities]; entity_embeddings = _model.encode(entity_names); cluster_centroids = np.array([c['centroid'] for c in _clusters])
    similarity_matrix = cosine_similarity(entity_embeddings, cluster_centroids)
    nodes, edges = [], []
    for cluster in _clusters: nodes.append({"id": cluster['id'], "label": cluster['name'], "type": "cluster"})
    for i, entity in enumerate(all_entities):
        entity_id = f"entity_{i}_{entity['Entity'].replace(' ', '_')}"; entity_type = "primary" if entity in primary_entity_list else "missing"
        best_cluster_index = np.argmax(similarity_matrix[i]); best_cluster_id = _clusters[best_cluster_index]['id']
        nodes.append({"id": entity_id, "label": entity['Entity'], "type": entity_type, "score": entity.get('Combined Score', 0)}); edges.append({"source": entity_id, "target": best_cluster_id})
    return {"nodes": nodes, "edges": edges}

def create_content_structure_graph(relationships, highlighted_entity=None):
    if not relationships or not relationships.get('nodes'): return None
    nodes, edges = relationships['nodes'], relationships['edges']
    G = nx.Graph(); [G.add_node(node['id']) for node in nodes]; [G.add_edge(edge['source'], edge['target']) for edge in edges]
    clusters = [n for n in nodes if n['type'] == 'cluster']; center_nodes_ids = [c['id'] for c in clusters]
    pos_center = nx.circular_layout(G.subgraph(center_nodes_ids)); pos = nx.spring_layout(G, pos=pos_center, fixed=center_nodes_ids, iterations=200, k=1.8/np.sqrt(len(G.nodes())))
    node_x, node_y, node_text, node_color, node_size, node_hover, node_line_color, node_line_width = [], [], [], [], [], [], [], []
    for node in nodes:
        x, y = pos[node['id']]; node_x.append(x); node_y.append(y); node_text.append(node['label'])
        is_highlighted = highlighted_entity and node['label'] == highlighted_entity
        if node['type'] == 'cluster':
            node_color.append('rgba(220, 220, 220, 0.8)'); node_size.append(50); node_hover.append(f"<b>{node['label']}</b><br>A main theme of your content."); node_line_color.append('black'); node_line_width.append(1)
        elif node['type'] == 'primary':
            node_color.append('rgba(28, 117, 215, 0.9)'); node_size.append(15 + node.get('score', 0) * 20); node_hover.append(f"<b>{node['label']}</b><br>Type: In your content"); node_line_color.append('blue'); node_line_width.append(1)
        elif node['type'] == 'missing':
            node_color.append('rgba(255, 100, 100, 0.9)'); node_size.append(20 + node.get('score', 0) * 25); node_hover.append(f"<b>GAP: {node['label']}</b><br>Type: Missing from this section"); node_line_color.append('red' if is_highlighted else 'darkred'); node_line_width.append(3 if is_highlighted else 1)
    
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(color=node_color, size=node_size, line=dict(width=node_line_width, color=node_line_color)), hoverinfo='text', hovertext=node_hover)
    edge_x, edge_y = [], []; [edge_x.extend([pos[edge['source']][0], pos[edge['target']][0], None]) or edge_y.extend([pos[edge['source']][1], pos[edge['target']][1], None]) for edge in edges]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.7, color='#888'), hoverinfo='none', mode='lines')
    
    annotations = [dict(x=pos[node['id']][0], y=pos[node['id']][1], text=node['label'], showarrow=False, font=dict(size=9, color='black'), xanchor='center', yshift= -node_size[i]/2 - 10) for i, node in enumerate(nodes)]
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="Content Structure & Entity Gap Map", titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), annotations=annotations, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor='white', height=800))
    fig.add_annotation(x=0.01, y=0.98, xref="paper", yref="paper", text="<b>Legend:</b>", showarrow=False, xanchor="left"); fig.add_annotation(x=0.02, y=0.94, xref="paper", yref="paper", text="<span style='font-size: 18px; color:rgba(220, 220, 220, 1);'>●</span> Content Section", showarrow=False, xanchor="left"); fig.add_annotation(x=0.02, y=0.90, xref="paper", yref="paper", text="<span style='font-size: 18px; color:rgba(28, 117, 215, 1);'>●</span> Your Entity", showarrow=False, xanchor="left"); fig.add_annotation(x=0.02, y=0.86, xref="paper", yref="paper", text="<span style='font-size: 18px; color:rgba(255, 100, 100, 1);'>●</span> Missing Entity (Gap)", showarrow=False, xanchor="left")
    return fig

def split_text_into_passages(text, passage_length=200):
    if not text: return []
    words = text.split(); passages = [' '.join(words[i:i + passage_length]) for i in range(0, len(words), passage_length)]
    return [p.strip() for p in passages if p.strip()]

def find_entity_best_passage(entity_name, passages, model):
    if not passages or not model: return {"passage": "", "similarity": 0.0, "index": -1}
    try: entity_embedding = model.encode([entity_name]); passage_embeddings = model.encode(passages); similarities = cosine_similarity(entity_embedding, passage_embeddings)[0]; best_idx = np.argmax(similarities); return {"passage": passages[best_idx], "similarity": float(similarities[best_idx]), "index": int(best_idx)}
    except Exception as e: st.warning(f"Failed to find best passage for '{entity_name}': {e}"); return {"passage": "", "similarity": 0.0, "index": -1}

def generate_semantic_implementation_analysis(entity_name, primary_content, best_passage, target_query, entity_info):
    if not st.session_state.get('gemini_api_configured', False): return None
    try:
        import google.generativeai as genai; genai.configure(api_key=st.session_state.gemini_api_key_to_persist); model = genai.GenerativeModel("gemini-1.5-flash")
        tone_analysis_prompt = f"""Analyze the writing tone, style, and voice of this content sample: "{primary_content[:2000]}" Provide a brief analysis of: 1. Tone 2. Style 3. Audience. Keep it concise."""
        tone_response = model.generate_content(tone_analysis_prompt); tone_analysis = tone_response.text.strip()
        implementation_prompt = f"""Based on the target query "{target_query}" and analysis, provide recommendations for implementing entity "{entity_name}". TONE: {tone_analysis}. ENTITY: {entity_name} (Type: {entity_info.get('Type', 'Unknown')}, Query Relevance: {entity_info.get('Query Relevance', 0):.3f}). CONTEXT: "{best_passage}". Provide: 1. Why to Implement. 2. Where to Implement. 3. How to Implement (2-3 actionable examples)."""
        implementation_response = model.generate_content(implementation_prompt); return implementation_response.text.strip()
    except Exception as e: st.error(f"Gemini analysis failed: {e}"); return None

# --- Content Fetching Functions ---
def initialize_selenium_driver():
    options = ChromeOptions(); options.add_argument("--headless"); options.add_argument("--no-sandbox"); options.add_argument("--disable-dev-shm-usage"); options.add_argument("--disable-gpu")
    try: driver = webdriver.Chrome(service=ChromeService(), options=options); stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32", webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True); return driver
    except Exception as e: st.error(f"Selenium initialization failed: {e}"); return None
def fetch_content_with_zyte(url, api_key):
    if not api_key: st.error("Zyte API key not configured."); return None
    enforce_rate_limit(); st.write(f"_Fetching with Zyte API: {url}..._")
    try:
        response = requests.post("https://api.zyte.com/v1/extract", auth=(api_key, ''), json={'url': url, 'httpResponseBody': True}, timeout=45); response.raise_for_status(); data = response.json()
        if data.get('httpResponseBody'): return base64.b64decode(data['httpResponseBody']).decode('utf-8', 'ignore')
        else: st.error(f"Zyte API did not return content for {url}"); return None
    except Exception as e: st.error(f"Zyte API error for {url}: {e}"); return None
def fetch_content_with_selenium(url, driver_instance):
    if not driver_instance: return fetch_content_with_requests(url)
    try: enforce_rate_limit(); driver_instance.get(url); time.sleep(5); return driver_instance.page_source
    except Exception as e: st.error(f"Selenium fetch error for {url}: {e}"); st.warning(f"Falling back to requests."); return fetch_content_with_requests(url)
def fetch_content_with_requests(url):
    enforce_rate_limit(); headers = {'User-Agent': get_random_user_agent()}
    try: response = requests.get(url, timeout=20, headers=headers); response.raise_for_status(); return response.text
    except Exception as e: st.error(f"Requests error for {url}: {e}"); return None
def extract_clean_text(html_content):
    if not html_content: return ""
    soup = BeautifulSoup(html_content, 'html.parser'); [el.decompose() for el in soup(["script", "style", "noscript", "iframe", "link", "meta", 'nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption', 'menu', 'banner', 'dialog', 'img', 'svg']) if el.name]; return re.sub(r'\s+', ' ', soup.get_text(separator=' ')).strip()

# --- Main UI ---
st.title("🎯 Entity Gap Analysis Tool"); st.markdown("**Analyze entity gaps between your content and competitors, with query-specific relevance scoring.**")
st.sidebar.subheader("📊 Analysis Configuration"); target_query = st.sidebar.text_input("Target Search Query:", "server-side rendering benefits", help="The search query to measure entity relevance against", disabled=st.session_state.processing, key="target_query_input"); scraping_method = st.sidebar.selectbox("Scraping Method:", ["Requests (fast)", "Zyte API (best)", "Selenium (for JS)"], index=0, disabled=st.session_state.processing, key="scraping_method_select"); st.sidebar.subheader("🔗 URL Configuration"); primary_url = st.sidebar.text_input("Your Primary URL:", "https://cloudinary.com/guides/automatic-image-cropping/server-side-rendering-benefits-use-cases-and-best-practices", disabled=st.session_state.processing, key="primary_url_input"); competitor_urls = st.sidebar.text_area("Competitor URLs (one per line):", "https://prismic.io/blog/what-is-ssr\nhttps://nextjs.org/docs/pages/building-your-application/rendering/server-side-rendering", height=100, disabled=st.session_state.processing, key="competitor_urls_input")
analysis_disabled = not st.session_state.gcp_nlp_configured or st.session_state.processing
if st.sidebar.button("🚀 Analyze Entity Gaps", disabled=analysis_disabled, type="primary", key="analyze_btn"):
    if not target_query or not primary_url or not competitor_urls: st.error("Please fill in all required fields.")
    else: st.session_state.processing = True; st.rerun()

# --- Main Processing ---
if st.session_state.processing:
    try:
        if not st.session_state.embedding_model:
            with st.spinner("Loading embedding model..."): st.session_state.embedding_model = load_embedding_model()
        if not st.session_state.embedding_model: st.error("Failed to load embedding model."); st.stop()
        competitor_url_list = [url.strip() for url in competitor_urls.split('\n') if url.strip()]; all_urls = [primary_url] + competitor_url_list; url_content, url_entities = {}, {}
        if scraping_method.startswith("Selenium") and not st.session_state.selenium_driver_instance:
            with st.spinner("Initializing Selenium..."): st.session_state.selenium_driver_instance = initialize_selenium_driver()
        with st.spinner(f"Fetching content from {len(all_urls)} URLs..."):
            for i, url in enumerate(all_urls):
                st.write(f"Fetching {i+1}/{len(all_urls)}: {url}")
                if scraping_method.startswith("Zyte") and st.session_state.zyte_api_configured: content = fetch_content_with_zyte(url, st.session_state.zyte_api_key_to_persist)
                elif scraping_method.startswith("Selenium"): content = fetch_content_with_selenium(url, st.session_state.selenium_driver_instance)
                else: content = fetch_content_with_requests(url)
                if content:
                    clean_text = extract_clean_text(content)
                    if clean_text and len(clean_text) > 100: url_content[url] = clean_text; st.success(f"✅ Fetched {len(clean_text):,} characters")
                    else: st.warning(f"⚠️ Insufficient content from {url}")
                else: st.error(f"❌ Failed to fetch {url}")
        if st.session_state.selenium_driver_instance: st.session_state.selenium_driver_instance.quit(); st.session_state.selenium_driver_instance = None
        if not url_content: st.error("No content fetched. Check URLs."); st.stop()
        with st.spinner("Extracting entities from all URLs..."):
            for url, content in url_content.items():
                st.write(f"Extracting entities from: {url}"); entities = extract_entities_with_google_nlp(content, st.session_state.gcp_credentials_info)
                if entities: url_entities[url] = entities; st.success(f"✅ Found {len(entities)} entities")
                else: st.warning(f"⚠️ No entities extracted from {url}")
        st.session_state.entity_analysis_results = url_entities; st.session_state.content_passages = {}
        if primary_url in url_content: passages = split_text_into_passages(url_content[primary_url]); st.session_state.content_passages[primary_url] = passages
        st.success("✅ Entity analysis complete!")
    finally: st.session_state.processing = False; st.rerun()

# --- Results Display ---
if st.session_state.entity_analysis_results:
    st.markdown("---"); st.subheader("📊 Entity Gap Analysis Results")
    primary_entities = st.session_state.entity_analysis_results.get(primary_url, {})
    primary_entity_analysis, missing_entities = [], []
    if not primary_entities: st.error("No entities found in primary URL.")
    else:
        competitor_entities = {}; [competitor_entities.setdefault(key, {'data': data, 'found_on': []}).get('found_on').append(url) for url, entities in st.session_state.entity_analysis_results.items() if url != primary_url for key, data in entities.items()]
        primary_entity_keys = set(primary_entities.keys()); primary_entity_names = {data['name'].lower() for data in primary_entities.values()}
        for key, comp_data in competitor_entities.items():
            entity_name_lower = comp_data['data']['name'].lower()
            if key not in primary_entity_keys and entity_name_lower not in primary_entity_names and not any(entity_name_lower in pn or pn in entity_name_lower for pn in primary_entity_names):
                query_sim = calculate_entity_query_relevance(comp_data['data']['name'], target_query, st.session_state.embedding_model); combined_score = (comp_data['data']['salience'] + query_sim) / 2
                if combined_score > 0.15: missing_entities.append({'Entity': comp_data['data']['name'], 'Type': comp_data['data']['type'], 'Document Salience': comp_data['data']['salience'], 'Query Relevance': query_sim, 'Combined Score': combined_score, 'Found On': len(comp_data['found_on']), 'URLs': ', '.join([f"`{url.split('//')[-1].split('/')[0]}`" for url in comp_data['found_on'][:2]])})
        if missing_entities:
            st.subheader("❗ Missing Entities (Found in Competitors)"); gap_df = pd.DataFrame(missing_entities).sort_values('Combined Score', ascending=False)
            st.dataframe(gap_df, use_container_width=True, column_config={"Combined Score": st.column_config.ProgressColumn("Combined Score",format="%.3f",min_value=0,max_value=1), "Query Relevance": st.column_config.ProgressColumn("Query Relevance",format="%.3f",min_value=0,max_value=1), "Document Salience": st.column_config.ProgressColumn("Document Salience",format="%.3f",min_value=0,max_value=1)})
        else: st.success("✅ No entity gaps found! Your content covers all entities found in competitors.")
        
        st.subheader("📋 Your Content Entities vs Target Query")
        for key, data in primary_entities.items(): query_sim = calculate_entity_query_relevance(data['name'], target_query, st.session_state.embedding_model); primary_entity_analysis.append({'Entity': data['name'], 'Type': data['type'], 'Document Salience': data['salience'], 'Query Relevance': query_sim, 'Combined Score': (data['salience'] + query_sim) / 2})
        primary_df = pd.DataFrame(primary_entity_analysis).sort_values('Combined Score', ascending=False)
        st.dataframe(primary_df, use_container_width=True, column_config={"Combined Score": st.column_config.ProgressColumn("Combined Score",format="%.3f",min_value=0,max_value=1), "Query Relevance": st.column_config.ProgressColumn("Query Relevance",format="%.3f",min_value=0,max_value=1), "Document Salience": st.column_config.ProgressColumn("Document Salience",format="%.3f",min_value=0,max_value=1)})

        if st.session_state.content_passages.get(primary_url):
            st.markdown("---"); st.subheader("🗺️ Content Structure & Entity Gap Map"); st.markdown("_This map visualizes the main thematic sections of your content. It shows which entities you've covered in each section and identifies gaps where missing entities should be added._")
            with st.expander("📊 Graph & Filter Controls", expanded=True):
                col1, col2, col3 = st.columns(3); num_clusters = col1.slider("Content Sections to Identify", min_value=3, max_value=8, value=5, help="How many distinct topics to find in your content?"); max_primary = col2.slider("Max 'Your' Entities", 3, 25, 15, help="Show top N entities from your content"); max_missing = col3.slider("Max 'Missing' Entities", 3, 20, 15, help="Show top N missing entities")
                filtered_primary = sorted(primary_entity_analysis, key=lambda x: x.get('Combined Score', 0), reverse=True)[:max_primary]
                filtered_missing = sorted(missing_entities, key=lambda x: x.get('Combined Score', 0), reverse=True)[:max_missing]
                missing_entity_names = [e['Entity'] for e in filtered_missing]; selected_entity_for_highlight = st.selectbox("🎯 Highlight a Missing Entity:", ["None"] + missing_entity_names, help="Choose a missing entity to highlight on the graph.")
            
            content_clusters = cluster_content_passages(st.session_state.content_passages[primary_url], st.session_state.embedding_model, num_clusters)
            if content_clusters:
                with st.spinner("Mapping entities to content structure..."):
                    content_relationships = calculate_content_relationships(filtered_primary, filtered_missing, content_clusters, st.session_state.embedding_model)
                    highlight = selected_entity_for_highlight if selected_entity_for_highlight != "None" else None
                    content_graph = create_content_structure_graph(content_relationships, highlighted_entity=highlight)
                if content_graph: st.plotly_chart(content_graph, use_container_width=True)
                else: st.warning("Could not generate the content structure graph.")
            else: st.warning("Could not analyze content structure. The article might be too short.")

        if missing_entities and st.session_state.content_passages.get(primary_url):
            st.markdown("---"); st.subheader("📍 Where to Add Missing Entities in Your Content"); st.markdown("_Select a missing entity to find the best insertion point and get AI-powered implementation advice._")
            all_sorted_missing = sorted(missing_entities, key=lambda x: x.get('Query Relevance', 0), reverse=True)
            entity_options = [""] + [f"{e['Entity']} (Relevance: {e.get('Query Relevance', 0):.2f})" for e in all_sorted_missing]; entity_lookup = {f"{e['Entity']} (Relevance: {e.get('Query Relevance', 0):.2f})": e for e in all_sorted_missing}
            selected_entity_display = st.selectbox("🔍 Select a missing entity to analyze:", options=entity_options, index=0, key="selected_entity_display")
            if selected_entity_display:
                entity_info = entity_lookup[selected_entity_display]; entity_name = entity_info['Entity']; best_passage_info = find_entity_best_passage(entity_name, st.session_state.content_passages[primary_url], st.session_state.embedding_model)
                col_a, col_b = st.columns([1, 2])
                with col_a: st.metric("Query Relevance", f"{entity_info.get('Query Relevance', 0):.3f}"); st.metric("Combined Score", f"{entity_info.get('Combined Score', 0):.3f}")
                with col_b:
                    st.text_area("Best passage for adding this entity:", value=best_passage_info["passage"], height=120, disabled=True, key=f"passage_{entity_name}")
                    if st.session_state.get('gemini_api_configured', False):
                        if st.button(f"🤖 Get AI Implementation Strategy for '{entity_name}'", key=f"gemini_{entity_name}"):
                            with st.spinner("Generating AI strategy..."):
                                primary_content = " ".join(st.session_state.content_passages[primary_url]); semantic_analysis = generate_semantic_implementation_analysis(entity_name, primary_content, best_passage_info["passage"], target_query, entity_info)
                                if semantic_analysis: st.markdown("---"); st.markdown(semantic_analysis)
                    else: st.info("💡 **Enable the Gemini API** in the sidebar to get AI-powered implementation strategies.")
st.sidebar.divider(); st.sidebar.info("🎯 Entity Gap Analysis Tool v4.0"); st.sidebar.markdown("**Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)**")

