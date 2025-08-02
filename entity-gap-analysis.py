# Entity Relationship Graph
        if st.session_state.entity_relationships:
            st.markdown("---")
            st.subheader("🕸️ Entity Relationship Graph")
            st.markdown("_Visualize how your existing entities relate to missing entities and your target query_")
            
            # Entity selection for highlighting
            missing_entity_names = [entity['Entity'] for entity in missing_entities]
            import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
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
if "entity_relationships" not in st.session_state: st.session_state.entity_relationships = None

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
    uploaded_gcp_key = st.file_uploader(
        "Upload Google Cloud Service Account JSON",
        type="json",
        help="Upload the JSON key file for a service account with 'Cloud Natural Language API User' role.",
        disabled=st.session_state.processing
    )
    if uploaded_gcp_key is not None:
        try:
            credentials_info = json.load(uploaded_gcp_key)
            if "project_id" in credentials_info and "private_key" in credentials_info:
                st.session_state.gcp_credentials_info = credentials_info
                st.session_state.gcp_nlp_configured = True
                st.success(f"GCP Key for project '{credentials_info['project_id']}' loaded!")
            else:
                st.error("Invalid JSON key file format.")
                st.session_state.gcp_nlp_configured = False
                st.session_state.gcp_credentials_info = None
        except Exception as e:
            st.error(f"Failed to process GCP key file: {e}")
            st.session_state.gcp_nlp_configured = False
            st.session_state.gcp_credentials_info = None

with st.sidebar.expander("Zyte API (Optional - for tough sites)", expanded=False):
    zyte_api_key_input = st.text_input("Enter Zyte API Key:", type="password", value=st.session_state.get("zyte_api_key_to_persist", ""), disabled=st.session_state.processing)
    if st.button("Set & Verify Zyte Key", disabled=st.session_state.processing):
        if zyte_api_key_input:
            try:
                response = requests.post(
                    "https://api.zyte.com/v1/extract",
                    auth=(zyte_api_key_input, ''),
                    json={'url': 'https://toscrape.com/', 'httpResponseBody': True},
                    timeout=20
                )
                if response.status_code == 200:
                    st.session_state.zyte_api_key_to_persist = zyte_api_key_input
                    st.session_state.zyte_api_configured = True
                    st.success("Zyte API Key Configured!")
                    st.rerun()
                else:
                    st.session_state.zyte_api_configured = False
                    st.error(f"Zyte Key Failed. Status: {response.status_code}")
            except Exception as e:
                st.session_state.zyte_api_configured = False
                st.error(f"Zyte API Request Failed: {str(e)[:200]}")
        else:
            st.warning("Please enter Zyte API Key.")

st.sidebar.markdown("---")
if st.session_state.get("gcp_nlp_configured"): 
    st.sidebar.markdown("✅ Google NLP API: **Required - Configured**")
else: 
    st.sidebar.markdown("⚠️ Google NLP API: **Required - Not Configured**")

st.sidebar.markdown(f"🔧 Zyte API: **{'Configured' if st.session_state.zyte_api_configured else 'Optional - Not Configured'}**")

# --- Core Functions ---
@st.cache_data(show_spinner="Extracting entities...")
def extract_entities_with_google_nlp(text: str, _credentials_info: dict):
    """Extracts entities from text using Google Cloud Natural Language API."""
    if not _credentials_info or not text:
        return {}

    try:
        credentials = service_account.Credentials.from_service_account_info(_credentials_info)
        client = language_v1.LanguageServiceClient(credentials=credentials)
        
        # Truncate if needed
        max_bytes = 900000  # Leave some buffer
        text_bytes = text.encode('utf-8')
        if len(text_bytes) > max_bytes:
            text = text_bytes[:max_bytes].decode('utf-8', 'ignore')
            st.warning("Text was truncated to fit Google NLP API size limit.")

        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = client.analyze_entities(document=document, encoding_type=language_v1.EncodingType.UTF8)
        
        entities_dict = {}
        for entity in response.entities:
            key = entity.name.lower()
            if key not in entities_dict or entity.salience > entities_dict[key]['salience']:
                entities_dict[key] = {
                    'name': entity.name,
                    'type': language_v1.Entity.Type(entity.type_).name,
                    'salience': entity.salience,
                    'mentions': len(entity.mentions)
                }
        return entities_dict

    except Exception as e:
        st.error(f"Google Cloud NLP API Error: {e}")
        return {}

def calculate_entity_relationships(primary_entities, missing_entities, embedding_model, target_query):
    """Calculate relationships between primary entities and missing entities."""
    if not embedding_model:
        return {}
    
    relationships = {
        'primary_entities': [],
        'missing_entities': [],
        'edges': [],
        'query_entity': target_query
    }
    
    try:
        # Prepare entity lists
        primary_names = [entity['name'] for entity in primary_entities]
        missing_names = [entity['name'] for entity in missing_entities]
        all_entity_names = primary_names + missing_names + [target_query]
        
        # Calculate embeddings for all entities
        if not all_entity_names:
            return relationships
            
        entity_embeddings = embedding_model.encode(all_entity_names)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(entity_embeddings)
        
        # Add primary entities to graph
        for i, entity in enumerate(primary_entities):
            relationships['primary_entities'].append({
                'id': f"primary_{i}",
                'name': entity['name'],
                'type': entity.get('type', 'UNKNOWN'),
                'salience': entity.get('document_salience', entity.get('salience', 0)),
                'query_relevance': entity.get('query_relevance', 0),
                'combined_score': entity.get('combined_score', 0),
                'node_type': 'primary'
            })
        
        # Add missing entities to graph
        for i, entity in enumerate(missing_entities):
            relationships['missing_entities'].append({
                'id': f"missing_{i}",
                'name': entity['name'],
                'type': entity.get('type', 'UNKNOWN'),
                'salience': entity.get('document_salience', entity.get('salience', 0)),
                'query_relevance': entity.get('query_relevance', 0),
                'combined_score': entity.get('combined_score', 0),
                'node_type': 'missing'
            })
        
        # Calculate edges (relationships) with similarity threshold
        similarity_threshold = 0.3  # Only show meaningful relationships
        
        # Primary to missing entity relationships
        for i, primary_entity in enumerate(primary_entities):
            primary_idx = i
            for j, missing_entity in enumerate(missing_entities):
                missing_idx = len(primary_names) + j
                similarity = similarity_matrix[primary_idx][missing_idx]
                
                if similarity > similarity_threshold:
                    relationships['edges'].append({
                        'source': f"primary_{i}",
                        'target': f"missing_{j}",
                        'weight': float(similarity),
                        'type': 'primary_to_missing'
                    })
        
        # Query to entity relationships
        query_idx = len(all_entity_names) - 1
        
        # Query to primary entities
        for i, primary_entity in enumerate(primary_entities):
            similarity = similarity_matrix[i][query_idx]
            if similarity > similarity_threshold:
                relationships['edges'].append({
                    'source': 'query',
                    'target': f"primary_{i}",
                    'weight': float(similarity),
                    'type': 'query_to_primary'
                })
        
        # Query to missing entities
        for j, missing_entity in enumerate(missing_entities):
            missing_idx = len(primary_names) + j
            similarity = similarity_matrix[missing_idx][query_idx]
            if similarity > similarity_threshold:
                relationships['edges'].append({
                    'source': 'query',
                    'target': f"missing_{j}",
                    'weight': float(similarity),
                    'type': 'query_to_missing'
                })
        
        return relationships
        
    except Exception as e:
        st.error(f"Error calculating entity relationships: {e}")
        return relationships

def create_entity_relationship_graph(relationships, selected_missing_entity=None):
    """Create an interactive entity relationship graph using Plotly."""
    
    if not relationships or (not relationships['primary_entities'] and not relationships['missing_entities']):
        return None
    
    # Create NetworkX graph for layout calculation
    G = nx.Graph()
    
    # Add nodes
    all_entities = relationships['primary_entities'] + relationships['missing_entities']
    
    # Add query node
    G.add_node('query', node_type='query')
    
    for entity in all_entities:
        G.add_node(entity['id'], **entity)
    
    # Add edges
    for edge in relationships['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    # Calculate layout
    try:
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    except:
        # Fallback to circular layout if spring layout fails
        pos = nx.circular_layout(G)
    
    # Prepare data for Plotly
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in relationships['edges']:
        source = edge['source']
        target = edge['target']
        
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{source} → {target}<br>Similarity: {edge['weight']:.3f}")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    node_info = []
    
    # Add query node
    if 'query' in pos:
        qx, qy = pos['query']
        node_x.append(qx)
        node_y.append(qy)
        node_text.append(f"🎯 {relationships['query_entity']}")
        node_colors.append('gold')
        node_sizes.append(25)
        node_info.append(f"<b>Target Query</b><br>{relationships['query_entity']}")
    
    # Add entity nodes
    for entity in all_entities:
        if entity['id'] in pos:
            x, y = pos[entity['id']]
            node_x.append(x)
            node_y.append(y)
            
            # Node styling based on type and selection
            if entity['node_type'] == 'primary':
                color = 'lightblue'
                icon = '🔵'
                size = 15 + (entity.get('combined_score', 0) * 10)
            else:  # missing entity
                if selected_missing_entity and entity['name'] == selected_missing_entity:
                    color = 'red'  # Highlight selected missing entity
                    icon = '🔴'
                    size = 25
                else:
                    color = 'orange'
                    icon = '🟠'
                    size = 15 + (entity.get('combined_score', 0) * 10)
            
            # Truncate long names for display
            display_name = entity['name']
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."
            
            node_text.append(f"{icon} {display_name}")
            node_colors.append(color)
            node_sizes.append(size)
            
            # Hover info
            node_info.append(
                f"<b>{entity['name']}</b><br>"
                f"Type: {entity['type']}<br>"
                f"Combined Score: {entity.get('combined_score', 0):.3f}<br>"
                f"Query Relevance: {entity.get('query_relevance', 0):.3f}<br>"
                f"Status: {'In Your Content' if entity['node_type'] == 'primary' else 'Missing (Competitor)'}"
            )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_info,
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        textfont=dict(size=10),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                        title=dict(
                            text=f"Entity Relationship Graph for Primary URL",
                            x=0.5,
                            font=dict(size=16)
                        ),
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="🔵 Your Content | 🟠 Missing (Competitors) | 🔴 Selected Missing | 🎯 Target Query<br>Node size = Combined Score | Lines = Semantic Similarity",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(size=10)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                        ))
    
    return fig

@st.cache_resource
def load_embedding_model():
    """Load a lightweight embedding model for query similarity."""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

def calculate_entity_query_similarity(entity_name, query_text, model):
    """Calculate similarity between entity and target query."""
    if not model or not entity_name or not query_text:
        return 0.0
    
    try:
        entity_embedding = model.encode([entity_name])
        query_embedding = model.encode([query_text])
        similarity = cosine_similarity(entity_embedding, query_embedding)[0][0]
        return float(similarity)
    except Exception as e:
        st.warning(f"Failed to calculate similarity for '{entity_name}': {e}")
        return 0.0

def split_text_into_passages(text, passage_length=200):
    """Split text into passages of roughly equal length."""
    if not text:
        return []
    
    words = text.split()
    passages = []
    
    for i in range(0, len(words), passage_length):
        passage = ' '.join(words[i:i + passage_length])
        if passage.strip():
            passages.append(passage.strip())
    
    return passages

def find_entity_best_passage(entity_name, passages, model):
    """Find the passage where the entity has highest semantic relevance."""
    if not passages or not model:
        return {"passage": "", "similarity": 0.0, "index": -1}
    
    try:
        entity_embedding = model.encode([entity_name])
        passage_embeddings = model.encode(passages)
        
        similarities = cosine_similarity(entity_embedding, passage_embeddings)[0]
        best_idx = np.argmax(similarities)
        
        return {
            "passage": passages[best_idx],
            "similarity": float(similarities[best_idx]),
            "index": int(best_idx)
        }
    except Exception as e:
        st.warning(f"Failed to find best passage for '{entity_name}': {e}")
        return {"passage": "", "similarity": 0.0, "index": -1}

# --- Content Fetching Functions ---
def initialize_selenium_driver():
    """Initialize Selenium driver with stealth configuration."""
    options = ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    try:
        driver = webdriver.Chrome(service=ChromeService(), options=options)
        stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32", webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)
        return driver
    except Exception as e:
        st.error(f"Selenium initialization failed: {e}")
        return None

def fetch_content_with_zyte(url, api_key):
    """Fetch content using Zyte API."""
    if not api_key:
        st.error("Zyte API key not configured.")
        return None
    
    enforce_rate_limit()
    st.write(f"_Fetching with Zyte API: {url}..._")
    
    try:
        response = requests.post(
            "https://api.zyte.com/v1/extract",
            auth=(api_key, ''),
            json={'url': url, 'httpResponseBody': True},
            timeout=45
        )
        response.raise_for_status()
        
        data = response.json()
        if data.get('httpResponseBody'):
            return base64.b64decode(data['httpResponseBody']).decode('utf-8', 'ignore')
        else:
            st.error(f"Zyte API did not return content for {url}")
            return None
            
    except requests.exceptions.HTTPError as e:
        st.error(f"Zyte API HTTP Error for {url}: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"Zyte API error for {url}: {e}")
        return None

def fetch_content_with_selenium(url, driver_instance):
    """Fetch content using Selenium with proper fallback."""
    if not driver_instance: 
        return fetch_content_with_requests(url)
    try:
        enforce_rate_limit()
        driver_instance.get(url)
        time.sleep(5)
        return driver_instance.page_source
    except Exception as e:
        st.error(f"Selenium fetch error for {url}: {e}")
        st.session_state.selenium_driver_instance = None
        st.warning(f"Selenium failed for {url}. Falling back to requests.")
        try: 
            return fetch_content_with_requests(url)
        except Exception as req_e: 
            st.error(f"Requests fallback also failed for {url}: {req_e}")
            return None

def fetch_content_with_requests(url):
    """Fetch content using requests with user agent rotation."""
    enforce_rate_limit()
    headers = {'User-Agent': get_random_user_agent()}
    try:
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Requests error for {url}: {e}")
        return None

def extract_clean_text(html_content):
    """Extract clean text from HTML."""
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted elements
    for el in soup(["script", "style", "noscript", "iframe", "link", "meta", 'nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption', 'menu', 'banner', 'dialog', 'img', 'svg']):
        if el.name: 
            el.decompose()
    
    # Get clean text
    text = soup.get_text(separator=' ')
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main UI ---
st.title("🎯 Entity Gap Analysis Tool")
st.markdown("**Analyze entity gaps between your content and competitors, with query-specific relevance scoring.**")

# --- Configuration ---
st.sidebar.subheader("📊 Analysis Configuration")
target_query = st.sidebar.text_input(
    "Target Search Query:", 
    "server-side rendering benefits", 
    help="The search query to measure entity relevance against",
    disabled=st.session_state.processing
)

scraping_method = st.sidebar.selectbox(
    "Scraping Method:",
    ["Requests (fast)", "Zyte API (best)", "Selenium (for JS)"],
    index=0,
    disabled=st.session_state.processing
)

st.sidebar.subheader("🔗 URL Configuration")
primary_url = st.sidebar.text_input(
    "Your Primary URL:", 
    "https://cloudinary.com/guides/automatic-image-cropping/server-side-rendering-benefits-use-cases-and-best-practices",
    disabled=st.session_state.processing
)

competitor_urls = st.sidebar.text_area(
    "Competitor URLs (one per line):",
    "https://prismic.io/blog/what-is-ssr\nhttps://nextjs.org/docs/pages/building-your-application/rendering/server-side-rendering",
    height=100,
    disabled=st.session_state.processing
)

# --- Analysis Button ---
analysis_disabled = not st.session_state.gcp_nlp_configured or st.session_state.processing

if st.sidebar.button("🚀 Analyze Entity Gaps", disabled=analysis_disabled, type="primary"):
    if not target_query or not primary_url or not competitor_urls:
        st.error("Please fill in all required fields.")
    else:
        st.session_state.processing = True
        st.rerun()

# --- Main Processing ---
if st.session_state.processing:
    try:
        # Load embedding model
        if not st.session_state.embedding_model:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = load_embedding_model()
        
        if not st.session_state.embedding_model:
            st.error("Failed to load embedding model.")
            st.stop()
        
        # Parse URLs
        competitor_url_list = [url.strip() for url in competitor_urls.split('\n') if url.strip()]
        all_urls = [primary_url] + competitor_url_list
        
        # Fetch content
        url_content = {}
        url_entities = {}
        
        # Initialize Selenium if needed
        if scraping_method.startswith("Selenium") and not st.session_state.selenium_driver_instance:
            with st.spinner("Initializing Selenium WebDriver..."): 
                st.session_state.selenium_driver_instance = initialize_selenium_driver()
        
        with st.spinner(f"Fetching content from {len(all_urls)} URLs..."):
            for i, url in enumerate(all_urls):
                st.write(f"Fetching {i+1}/{len(all_urls)}: {url}")
                
                # Choose fetching method based on user selection and API availability
                if scraping_method.startswith("Zyte") and st.session_state.zyte_api_configured:
                    content = fetch_content_with_zyte(url, st.session_state.zyte_api_key_to_persist)
                elif scraping_method.startswith("Selenium"):
                    content = fetch_content_with_selenium(url, st.session_state.selenium_driver_instance)
                else:
                    content = fetch_content_with_requests(url)
                
                if content:
                    clean_text = extract_clean_text(content)
                    if clean_text and len(clean_text) > 100:
                        url_content[url] = clean_text
                        st.success(f"✅ Fetched {len(clean_text):,} characters")
                    else:
                        st.warning(f"⚠️ Insufficient content from {url}")
                else:
                    st.error(f"❌ Failed to fetch {url}")
        
        # Clean up Selenium if used
        if st.session_state.selenium_driver_instance:
            st.session_state.selenium_driver_instance.quit()
            st.session_state.selenium_driver_instance = None
        
        if not url_content:
            st.error("No content was successfully fetched. Please check your URLs and try again.")
            st.stop()
        
        # Extract entities
        with st.spinner("Extracting entities from all URLs..."):
            for url, content in url_content.items():
                st.write(f"Extracting entities from: {url}")
                entities = extract_entities_with_google_nlp(content, st.session_state.gcp_credentials_info)
                
                if entities:
                    url_entities[url] = entities
                    st.success(f"✅ Found {len(entities)} entities")
                else:
                    st.warning(f"⚠️ No entities extracted from {url}")
        
        # Store results
        st.session_state.entity_analysis_results = url_entities
        st.session_state.content_passages = {}
        
        # Split primary URL content into passages for highlighting
        if primary_url in url_content:
            passages = split_text_into_passages(url_content[primary_url])
            st.session_state.content_passages[primary_url] = passages
        
        st.success("✅ Entity analysis complete!")
        
    finally:
        st.session_state.processing = False
        st.rerun()

# --- Results Display ---
if st.session_state.entity_analysis_results:
    st.markdown("---")
    st.subheader("📊 Entity Gap Analysis Results")
    
    # Get primary URL entities
    primary_entities = st.session_state.entity_analysis_results.get(primary_url, {})
    
    if not primary_entities:
        st.error("No entities found in primary URL.")
    else:
        # Collect all competitor entities
        competitor_entities = {}
        for url, entities in st.session_state.entity_analysis_results.items():
            if url != primary_url:
                for entity_key, entity_data in entities.items():
                    if entity_key not in competitor_entities:
                        competitor_entities[entity_key] = {
                            'data': entity_data,
                            'found_on': []
                        }
                    competitor_entities[entity_key]['found_on'].append(url)
        
        # Calculate entity gaps
        primary_entity_keys = set(primary_entities.keys())
        missing_entities = []
        
        for entity_key, comp_data in competitor_entities.items():
            if entity_key not in primary_entity_keys:
                # Calculate query relevance
                query_similarity = calculate_entity_query_similarity(
                    comp_data['data']['name'], 
                    target_query, 
                    st.session_state.embedding_model
                )
                
                # Calculate combined score (consistent with primary URL entities)
                combined_score = (comp_data['data']['salience'] + query_similarity) / 2
                
                missing_entities.append({
                    'Entity': comp_data['data']['name'],
                    'Type': comp_data['data']['type'],
                    'Document Salience': comp_data['data']['salience'],
                    'Query Relevance': query_similarity,
                    'Combined Score': combined_score,
                    'Found On': len(comp_data['found_on']),
                    'URLs': ', '.join([f"`{url.split('//')[-1].split('/')[0]}`" for url in comp_data['found_on'][:2]])
                })
        
        # Display gaps
        if missing_entities:
            st.subheader("❗ Missing Entities (Found in Competitors)")
            
            gap_df = pd.DataFrame(missing_entities)
            gap_df = gap_df.sort_values('Combined Score', ascending=False)
            
            st.dataframe(
                gap_df,
                use_container_width=True,
                column_config={
                    "Document Salience": st.column_config.ProgressColumn(
                        "Document Salience",
                        format="%.3f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Query Relevance": st.column_config.ProgressColumn(
                        "Query Relevance",
                        format="%.3f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Combined Score": st.column_config.ProgressColumn(
                        "Combined Score",
                        format="%.3f",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )
        else:
            st.success("✅ No entity gaps found! Your content covers all entities found in competitors.")
        
        # Display primary URL entities with query relevance
        st.subheader("📋 Your Content Entities vs Target Query")
        
        primary_entity_analysis = []
        for entity_key, entity_data in primary_entities.items():
            query_similarity = calculate_entity_query_similarity(
                entity_data['name'], 
                target_query, 
                st.session_state.embedding_model
            )
            
            primary_entity_analysis.append({
                'Entity': entity_data['name'],
                'Type': entity_data['type'],
                'Document Salience': entity_data['salience'],
                'Query Relevance': query_similarity,
                'Combined Score': (entity_data['salience'] + query_similarity) / 2
            })
        
        primary_df = pd.DataFrame(primary_entity_analysis)
        primary_df = primary_df.sort_values('Combined Score', ascending=False)
        
        st.dataframe(
            primary_df,
            use_container_width=True,
            column_config={
                "Document Salience": st.column_config.ProgressColumn(
                    "Document Salience",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
                "Query Relevance": st.column_config.ProgressColumn(
                    "Query Relevance",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
                "Combined Score": st.column_config.ProgressColumn(
                    "Combined Score",
                    format="%.3f",
                    min_value=0,
                    max_value=1,
                ),
            }
        )
        
        # Missing Entity Location Analysis
        if missing_entities and st.session_state.content_passages.get(primary_url):
            st.subheader("📍 Where to Add Missing Entities in Your Content")
            st.markdown("_Find the most relevant passages in your content for each missing entity, sorted by query relevance._")
            
            passages = st.session_state.content_passages[primary_url]
            
            # Sort missing entities by query relevance (highest first)
            sorted_missing = sorted(missing_entities, key=lambda x: x['Query Relevance'], reverse=True)
            
            # Create searchable dropdown for entity selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create entity options with scores for the dropdown
                entity_options = []
                entity_lookup = {}
                
                for entity_info in sorted_missing:
                    entity_name = entity_info['Entity']
                    query_relevance = entity_info['Query Relevance']
                    combined_score = entity_info['Combined Score']
                    
                    # Create display name with scores
                    display_name = f"{entity_name} (Query: {query_relevance:.3f}, Combined: {combined_score:.3f})"
                    entity_options.append(display_name)
                    entity_lookup[display_name] = entity_info
                
                # Searchable selectbox
                selected_entity_display = st.selectbox(
                    "🔍 Search and select missing entity to analyze:",
                    options=[""] + entity_options,
                    index=0,
                    help="Type to search through missing entities. Sorted by Query Relevance (highest first)."
                )
            
            with col2:
                # Show bulk analysis option
                if st.button("📊 Show Top 5 Entities", help="Display analysis for top 5 entities by query relevance"):
                    selected_entity_display = "SHOW_TOP_5"
            
            # Display analysis based on selection
            if selected_entity_display and selected_entity_display != "":
                if selected_entity_display == "SHOW_TOP_5":
                    # Show top 5 entities
                    st.markdown("**🔥 Top 5 Missing Entities by Query Relevance:**")
                    
                    for i, entity_info in enumerate(sorted_missing[:5]):
                        entity_name = entity_info['Entity']
                        combined_score = entity_info['Combined Score']
                        query_relevance = entity_info['Query Relevance']
                        document_salience = entity_info['Document Salience']
                        
                        # Find best passage for this missing entity
                        best_passage_info = find_entity_best_passage(
                            entity_name, 
                            passages, 
                            st.session_state.embedding_model
                        )
                        
                        # Create expandable section for each entity
                        relevance_icon = "🔥" if query_relevance > 0.7 else "🟡" if query_relevance > 0.4 else "🔵"
                        
                        with st.expander(f"{relevance_icon} **#{i+1}: {entity_name}** (Query Relevance: {query_relevance:.3f})", expanded=(i < 2)):
                            col_a, col_b = st.columns([1, 2])
                            
                            with col_a:
                                st.markdown("**📊 Entity Info:**")
                                st.markdown(f"- **Type:** {entity_info['Type']}")
                                st.markdown(f"- **Query Relevance:** {query_relevance:.3f}")
                                st.markdown(f"- **Combined Score:** {combined_score:.3f}")
                                st.markdown(f"- **Document Salience:** {document_salience:.3f}")
                                st.markdown(f"- **Found on:** {entity_info['Found On']} competitor site(s)")
                                st.markdown(f"- **Competitors:** {entity_info['URLs']}")
                                
                                # Add Wikipedia link if filtering was enabled
                                if st.session_state.get('enable_wikipedia_filtering', False):
                                    wikipedia_info = st.session_state.get('wikipedia_mappings', {})
                                    for url, mapping in wikipedia_info.items():
                                        if url != primary_url:
                                            entity_key = entity_name.lower()
                                            if entity_key in mapping:
                                                wiki_url = mapping[entity_key]['url']
                                                wiki_title = mapping[entity_key]['title']
                                                st.markdown(f"- **Wikipedia:** [📖 {wiki_title}]({wiki_url})")
                                                break
                            
                            with col_b:
                                if best_passage_info["similarity"] > 0.1:
                                    st.markdown("**🎯 Best place to add this entity:**")
                                    st.markdown(f"**Content Relevance:** {best_passage_info['similarity']:.3f}")
                                    
                                    passage_text = best_passage_info["passage"]
                                    st.markdown("**📝 Suggested insertion location:**")
                                    st.text_area(
                                        "Passage text:",
                                        value=passage_text,
                                        height=80,
                                        disabled=True,
                                        key=f"bulk_passage_{i}_{entity_name.replace(' ', '_')}"
                                    )
                                    
                                    if best_passage_info["similarity"] > 0.5:
                                        st.success("✅ High semantic relevance")
                                    elif best_passage_info["similarity"] > 0.3:
                                        st.info("ℹ️ Moderate relevance")
                                    else:
                                        st.warning("⚠️ Lower relevance")
                                else:
                                    st.markdown("**🤔 No strongly relevant passage found.**")
                                    st.info(f"Consider adding a new section about '{entity_name}'.")
                            
                            st.divider()
                
                else:
                    # Show detailed analysis for selected entity
                    entity_info = entity_lookup[selected_entity_display]
                    entity_name = entity_info['Entity']
                    combined_score = entity_info['Combined Score']
                    query_relevance = entity_info['Query Relevance']
                    document_salience = entity_info['Document Salience']
                    
                    # Find best passage for this missing entity
                    best_passage_info = find_entity_best_passage(
                        entity_name, 
                        passages, 
                        st.session_state.embedding_model
                    )
                    
                    st.markdown(f"### 🎯 Analysis for: **{entity_name}**")
                    
                    col_a, col_b = st.columns([1, 2])
                    
                    with col_a:
                        st.markdown("**📊 Entity Metrics:**")
                        
                        # Progress bars for scores
                        st.metric("Query Relevance", f"{query_relevance:.3f}")
                        st.progress(query_relevance)
                        
                        st.metric("Combined Score", f"{combined_score:.3f}")
                        st.progress(combined_score)
                        
                        st.metric("Document Salience", f"{document_salience:.3f}")
                        st.progress(document_salience)
                        
                        st.markdown("**🏢 Competitor Info:**")
                        st.markdown(f"- **Type:** {entity_info['Type']}")
                        st.markdown(f"- **Found on:** {entity_info['Found On']} competitor site(s)")
                        st.markdown(f"- **Competitors:** {entity_info['URLs']}")
                        
                        # Add Wikipedia link if available and filtering was enabled
                        if st.session_state.get('enable_wikipedia_filtering', False):
                            wikipedia_info = st.session_state.get('wikipedia_mappings', {})
                            for url, mapping in wikipedia_info.items():
                                if url != primary_url:  # Check competitor URLs
                                    entity_key = entity_name.lower()
                                    if entity_key in mapping:
                                        wiki_url = mapping[entity_key]['url']
                                        wiki_title = mapping[entity_key]['title']
                                        st.markdown(f"- **Wikipedia:** [📖 {wiki_title}]({wiki_url})")
                                        break
                    
                    with col_b:
                        if best_passage_info["similarity"] > 0.1:
                            st.markdown("**🎯 Recommended Insertion Location:**")
                            st.markdown(f"**Content Relevance Score:** {best_passage_info['similarity']:.3f}")
                            
                            passage_text = best_passage_info["passage"]
                            
                            st.text_area(
                                "Best passage for adding this entity:",
                                value=passage_text,
                                height=120,
                                disabled=True,
                                key=f"selected_passage_{entity_name.replace(' ', '_')}"
                            )
                            
                            # Quality assessment
                            if best_passage_info["similarity"] > 0.5:
                                st.success("✅ **Excellent insertion point** - High semantic relevance")
                                st.markdown("💡 **Recommendation:** This entity fits naturally into this section.")
                            elif best_passage_info["similarity"] > 0.3:
                                st.info("ℹ️ **Good insertion point** - Moderate relevance")
                                st.markdown("💡 **Recommendation:** Consider expanding this section to include this entity.")
                            else:
                                st.warning("⚠️ **Lower relevance** - Consider content fit")
                                st.markdown("💡 **Recommendation:** Evaluate if this entity aligns with your content theme.")
                        else:
                            st.markdown("**🤔 No strongly relevant passage found in your content.**")
                            st.info(f"**Recommendation:** Consider adding a new section specifically about '{entity_name}' or expanding existing content to cover this topic.")
                            
                            # Show entity importance context
                            if query_relevance > 0.7:
                                st.error("⚠️ **High Priority Gap:** This entity is very relevant to your target query but missing from your content.")
                            elif query_relevance > 0.4:
                                st.warning("⚠️ **Medium Priority Gap:** This entity has moderate relevance to your target query.")
            
            else:
                st.info("👆 Select a missing entity above to see detailed insertion recommendations, or click 'Show Top 5 Entities' for a quick overview.")
        
        # Existing Entity Location Finder (for entities already in your content)
        st.subheader("📍 Locate Existing Entities in Your Content")
        
        if st.session_state.content_passages.get(primary_url):
            entity_options = [entity['Entity'] for entity in primary_entity_analysis]
            
            selected_entity = st.selectbox(
                "Select existing entity to locate in your content:",
                options=entity_options,
                key="existing_entity_selector"
            )
            
            if selected_entity:
                passages = st.session_state.content_passages[primary_url]
                best_passage_info = find_entity_best_passage(
                    selected_entity, 
                    passages, 
                    st.session_state.embedding_model
                )
                
                if best_passage_info["similarity"] > 0:
                    st.markdown(f"**🎯 Best mention of '{selected_entity}' in your content:**")
                    st.markdown(f"**Relevance Score:** {best_passage_info['similarity']:.3f}")
                    
                    # Highlight the entity name in the passage
                    highlighted_passage = best_passage_info["passage"]
                    if selected_entity.lower() in highlighted_passage.lower():
                        highlighted_passage = re.sub(
                            f"({re.escape(selected_entity)})", 
                            r"**\1**", 
                            highlighted_passage, 
                            flags=re.IGNORECASE
                        )
                    
                    st.markdown(f"> {highlighted_passage}")
                else:
                    st.info(f"No strong semantic connection found for '{selected_entity}' in your content passages.")

# Footer
st.sidebar.divider()
st.sidebar.info("🎯 Entity Gap Analysis Tool v1.0")
st.sidebar.markdown("---")
st.sidebar.markdown("**Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)**")

# Help section
with st.sidebar.expander("❓ How it works", expanded=False):
    st.markdown("""
    **This tool helps you find content gaps by:**
    
    1. **Extracting entities** from your content and competitors
    2. **Measuring relevance** to your target search query
    3. **Identifying gaps** - entities competitors have but you don't
    4. **Locating entities** in your content for optimization
    
    **Entity Salience:** How important an entity is to the document
    **Query Relevance:** How relevant an entity is to your target query
    **Combined Score:** Average of both metrics
    """)
