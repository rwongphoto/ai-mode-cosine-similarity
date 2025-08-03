import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
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

with st.sidebar.expander("Zyte API (Optional)", expanded=False):
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
st.sidebar.markdown(f"🤖 Gemini API: **{'Configured' if st.session_state.get('gemini_api_configured', False) else 'Optional - Not Configured'}**")

# --- Core Functions ---
def is_number_entity(entity_name):
    """Check if an entity is primarily numeric and should be filtered out."""
    if not entity_name:
        return True
    
    # Remove common separators and whitespace
    cleaned = re.sub(r'[,\s\-\.]', '', entity_name)
    
    # Check if it's purely numeric
    if cleaned.isdigit():
        return True
    
    # Check if it's a percentage
    if entity_name.strip().endswith('%') and re.sub(r'[%,\s\-\.]', '', entity_name).isdigit():
        return True
    
    # Check if it's a year (4 digits)
    if re.match(r'^\d{4}$', cleaned):
        return True
    
    # Check if it's mostly numeric (>70% digits)
    digit_count = sum(1 for char in entity_name if char.isdigit())
    total_chars = len(re.sub(r'\s', '', entity_name))
    
    if total_chars > 0 and (digit_count / total_chars) > 0.7:
        return True
    
    # Filter out very short numeric-heavy entities
    if len(entity_name.strip()) <= 4 and any(char.isdigit() for char in entity_name):
        return True
    
    return False

@st.cache_data(show_spinner="Extracting entities...")
def extract_entities_with_google_nlp(text: str, _credentials_info: dict):
    """Extracts entities from text using Google Cloud Natural Language API."""
    if not _credentials_info or not text:
        return {}

    try:
        credentials = service_account.Credentials.from_service_account_info(_credentials_info)
        client = language_v1.LanguageServiceClient(credentials=credentials)
        
        max_bytes = 900000
        text_bytes = text.encode('utf-8')
        if len(text_bytes) > max_bytes:
            text = text_bytes[:max_bytes].decode('utf-8', 'ignore')
            st.warning("Text was truncated to fit Google NLP API size limit.")

        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = client.analyze_entities(document=document, encoding_type=language_v1.EncodingType.UTF8)
        
        entities_dict = {}
        for entity in response.entities:
            entity_name = entity.name.strip()
            
            # Filter out number entities
            if is_number_entity(entity_name):
                continue
                
            key = entity_name.lower()
            if key not in entities_dict or entity.salience > entities_dict[key]['salience']:
                entities_dict[key] = {
                    'name': entity_name,
                    'type': language_v1.Entity.Type(entity.type_).name,
                    'salience': entity.salience,
                    'mentions': len(entity.mentions)
                }
        return entities_dict

    except Exception as e:
        st.error(f"Google Cloud NLP API Error: {e}")
        return {}

@st.cache_resource
def load_embedding_model():
    """Load a lightweight embedding model for query similarity."""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

def calculate_entity_query_relevance(entity_name, query_text, model):
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

def calculate_entity_relationships(primary_entities, missing_entities, embedding_model, target_query):
    """Calculate relationships between primary entities and missing entities."""
    if not embedding_model:
        return {}
    
    # Filter to top entities for cleaner, more readable graph
    top_primary = sorted(primary_entities, key=lambda x: x['Combined Score'], reverse=True)[:15]  # Top 15 primary
    top_missing = sorted(missing_entities, key=lambda x: x['Combined Score'], reverse=True)[:10]   # Top 10 missing
    
    relationships = {
        'primary_entities': [],
        'missing_entities': [],
        'edges': [],
        'query_entity': target_query
    }
    
    try:
        # Prepare entity lists
        primary_names = [entity['Entity'] for entity in top_primary]
        missing_names = [entity['Entity'] for entity in top_missing]
        all_entity_names = primary_names + missing_names + [target_query]
        
        # Calculate embeddings for all entities
        if not all_entity_names:
            return relationships
            
        entity_embeddings = embedding_model.encode(all_entity_names)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(entity_embeddings)
        
        # Add primary entities to graph
        for i, entity in enumerate(top_primary):
            relationships['primary_entities'].append({
                'id': f"primary_{i}",
                'name': entity['Entity'],
                'type': entity.get('Type', 'UNKNOWN'),
                'salience': entity.get('Document Salience', 0),
                'query_relevance': entity.get('Query Relevance', 0),
                'combined_score': entity.get('Combined Score', 0),
                'node_type': 'primary'
            })
        
        # Add missing entities to graph
        for i, entity in enumerate(top_missing):
            relationships['missing_entities'].append({
                'id': f"missing_{i}",
                'name': entity['Entity'],
                'type': entity.get('Type', 'UNKNOWN'),
                'salience': entity.get('Document Salience', 0),
                'query_relevance': entity.get('Query Relevance', 0),
                'combined_score': entity.get('Combined Score', 0),
                'node_type': 'missing'
            })
        
        # Calculate edges with higher similarity thresholds for cleaner graph
        similarity_threshold = 0.4  # Higher threshold for fewer, stronger connections
        
        # Primary to missing entity relationships
        for i, primary_entity in enumerate(top_primary):
            primary_idx = i
            for j, missing_entity in enumerate(top_missing):
                missing_idx = len(primary_names) + j
                similarity = similarity_matrix[primary_idx][missing_idx]
                
                if similarity > similarity_threshold:
                    relationships['edges'].append({
                        'source': f"primary_{i}",
                        'target': f"missing_{j}",
                        'weight': float(similarity),
                        'type': 'primary_to_missing'
                    })
        
        # Query to entity relationships with higher threshold
        query_idx = len(all_entity_names) - 1
        query_threshold = 0.5  # Higher threshold for query connections
        
        # Query to primary entities
        for i, primary_entity in enumerate(top_primary):
            similarity = similarity_matrix[i][query_idx]
            if similarity > query_threshold:
                relationships['edges'].append({
                    'source': 'query',
                    'target': f"primary_{i}",
                    'weight': float(similarity),
                    'type': 'query_to_primary'
                })
        
        # Query to missing entities
        for j, missing_entity in enumerate(top_missing):
            missing_idx = len(primary_names) + j
            similarity = similarity_matrix[missing_idx][query_idx]
            if similarity > query_threshold:
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
    
    # Add query node with connections
    if 'query' in pos:
        qx, qy = pos['query']
        node_x.append(qx)
        node_y.append(qy)
        node_text.append(f"🎯 {relationships['query_entity']}")
        node_colors.append('gold')
        node_sizes.append(25)
        
        # Find query connections
        query_connections = []
        for edge in relationships['edges']:
            if edge['source'] == 'query':
                target_entity = next((e for e in all_entities if e['id'] == edge['target']), None)
                if target_entity:
                    query_connections.append(f"→ {target_entity['name']} ({edge['weight']:.3f})")
            elif edge['target'] == 'query':
                source_entity = next((e for e in all_entities if e['id'] == edge['source']), None)
                if source_entity:
                    query_connections.append(f"← {source_entity['name']} ({edge['weight']:.3f})")
        
        connections_text = "<br>".join(query_connections) if query_connections else "No connections above threshold"
        node_info.append(
            f"<b>Target Query</b><br>"
            f"{relationships['query_entity']}<br>"
            f"<br><b>Connected to:</b><br>{connections_text}"
        )
    
    # Add entity nodes
    for entity in all_entities:
        if entity['id'] in pos:
            x, y = pos[entity['id']]
            node_x.append(x)
            node_y.append(y)
            
            # Find all connections for this entity
            connections = []
            for edge in relationships['edges']:
                if edge['source'] == entity['id']:
                    # Find target entity name
                    if edge['target'] == 'query':
                        connections.append(f"🎯 {relationships['query_entity']} ({edge['weight']:.3f})")
                    else:
                        target_entity = next((e for e in all_entities if e['id'] == edge['target']), None)
                        if target_entity:
                            connections.append(f"→ {target_entity['name']} ({edge['weight']:.3f})")
                elif edge['target'] == entity['id']:
                    # Find source entity name
                    if edge['source'] == 'query':
                        connections.append(f"🎯 {relationships['query_entity']} ({edge['weight']:.3f})")
                    else:
                        source_entity = next((e for e in all_entities if e['id'] == edge['source']), None)
                        if source_entity:
                            connections.append(f"← {source_entity['name']} ({edge['weight']:.3f})")
            
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
            
            # Enhanced hover info with connections
            connections_text = "<br>".join(connections) if connections else "No connections above threshold"
            node_info.append(
                f"<b>{entity['name']}</b><br>"
                f"Type: {entity['type']}<br>"
                f"Combined Score: {entity.get('combined_score', 0):.3f}<br>"
                f"Query Relevance: {entity.get('query_relevance', 0):.3f}<br>"
                f"Status: {'In Your Content' if entity['node_type'] == 'primary' else 'Missing (Competitor)'}<br>"
                f"<br><b>Connected to:</b><br>{connections_text}"
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
                        title=f"Entity Relationship Graph for Primary URL",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
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

def generate_semantic_implementation_analysis(entity_name, primary_content, best_passage, target_query, entity_info):
    """Generate Gemini-powered analysis for implementing missing entities."""
    
    if not st.session_state.get('gemini_api_configured', False):
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
        model = genai.GenerativeModel("gemini-2.5-pro")
        
        # Analyze content tone and style
        tone_analysis_prompt = f"""
        Analyze the writing tone, style, and voice of this content sample:
        
        "{primary_content[:2000]}"
        
        Provide a brief analysis of:
        1. Writing tone (formal/informal, technical/accessible, etc.)
        2. Style characteristics (sentence structure, vocabulary level, etc.)
        3. Target audience level
        
        Keep your analysis concise and focused on actionable style guidelines.
        """
        
        tone_response = model.generate_content(tone_analysis_prompt)
        tone_analysis = tone_response.text.strip()
        
        # Generate implementation strategy
        implementation_prompt = f"""
        Based on this content analysis and the target query "{target_query}", provide strategic recommendations for implementing the entity "{entity_name}" into the content.
        
        CONTENT TONE ANALYSIS:
        {tone_analysis}
        
        ENTITY TO IMPLEMENT: {entity_name}
        - Type: {entity_info.get('Type', 'Unknown')}
        - Query Relevance: {entity_info.get('Query Relevance', 0):.3f}
        - Found on {entity_info.get('Found On', 0)} competitor sites
        
        BEST INSERTION LOCATION:
        "{best_passage}"
        
        CURRENT CONTENT SAMPLE:
        "{primary_content[:1500]}"
        
        Provide a strategic analysis with:
        
        ## WHY Implement This Entity
        - Strategic value for the target query
        - SEO and content gap benefits
        - User value proposition
        
        ## WHERE to Implement
        - Specific section recommendations
        - Integration with existing content flow
        - Placement strategy for maximum impact
        
        ## HOW to Implement
        - 2-3 specific, actionable content additions
        - Maintain the identified tone and style
        - Natural integration techniques
        - Suggested word count and depth
        
        Keep recommendations practical, specific, and aligned with the content's existing voice. Focus on seamless integration rather than forced insertion.
        """
        
        implementation_response = model.generate_content(implementation_prompt)
        return implementation_response.text.strip()
        
    except Exception as e:
        st.error(f"Gemini analysis failed: {e}")
        return None
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
        st.error(f"Zyte API HTTP Error for {url}: {e.response.status_code}")
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
                
                # Choose fetching method
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
        
        # Split primary URL content into passages
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
        
        # Calculate entity gaps with better matching
        primary_entity_keys = set(primary_entities.keys())
        primary_entity_names = {entity_data['name'].lower() for entity_data in primary_entities.values()}
        missing_entities = []
        
        for entity_key, comp_data in competitor_entities.items():
            entity_name = comp_data['data']['name']
            entity_name_lower = entity_name.lower()
            
            # Check both exact key match and name similarity
            is_missing = (entity_key not in primary_entity_keys and 
                         entity_name_lower not in primary_entity_names and
                         not any(entity_name_lower in primary_name or primary_name in entity_name_lower 
                                for primary_name in primary_entity_names))
            
            if is_missing:
                # Calculate query relevance
                query_similarity = calculate_entity_query_relevance(
                    entity_name, 
                    target_query, 
                    st.session_state.embedding_model
                )
                
                # Calculate combined score
                combined_score = (comp_data['data']['salience'] + query_similarity) / 2
                
                # Only include entities with meaningful scores
                if combined_score > 0.15:  # Filter out very low relevance entities
                    missing_entities.append({
                        'Entity': entity_name,
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
            query_similarity = calculate_entity_query_relevance(
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
        
        # Calculate and store entity relationships for the graph
        if primary_entity_analysis and missing_entities:
            with st.spinner("Calculating entity relationships..."):
                st.session_state.entity_relationships = calculate_entity_relationships(
                    primary_entity_analysis,
                    missing_entities,
                    st.session_state.embedding_model,
                    target_query
                )
        
        # Entity Relationship Graph
        if st.session_state.entity_relationships:
            st.markdown("---")
            st.subheader("🕸️ Entity Relationship Graph")
            st.markdown("_Visualize how your existing entities relate to missing entities and your target query_")
            
            # Entity selection for highlighting
            missing_entity_names = [entity['Entity'] for entity in missing_entities]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_entity_for_graph = st.selectbox(
                    "🎯 Select missing entity to highlight in graph:",
                    options=["None"] + missing_entity_names,
                    index=0,
                    help="Choose a missing entity to highlight its relationships in red"
                )
            
            with col2:
                if st.button("🔄 Refresh Graph Layout"):
                    # Force recalculation of graph layout
                    st.rerun()
            
            # Create and display the graph
            selected_entity = selected_entity_for_graph if selected_entity_for_graph != "None" else None
            
            entity_graph = create_entity_relationship_graph(
                st.session_state.entity_relationships,
                selected_missing_entity=selected_entity
            )
            
            if entity_graph:
                st.plotly_chart(entity_graph, use_container_width=True, height=600)
                
                # Graph insights
                with st.expander("📊 Graph Insights", expanded=False):
                    relationships = st.session_state.entity_relationships
                    
                    total_primary = len(relationships['primary_entities'])
                    total_missing = len(relationships['missing_entities'])
                    total_edges = len(relationships['edges'])
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Your Entities", total_primary)
                        st.metric("Missing Entities", total_missing)
                    
                    with col_b:
                        st.metric("Total Relationships", total_edges)
                        query_connections = len([e for e in relationships['edges'] if 'query' in e['source'] or 'query' in e['target']])
                        st.metric("Query Connections", query_connections)
                    
                    with col_c:
                        if total_edges > 0:
                            avg_similarity = np.mean([e['weight'] for e in relationships['edges']])
                            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                        
                        strong_connections = len([e for e in relationships['edges'] if e['weight'] > 0.5])
                        st.metric("Strong Connections", strong_connections)
                    
                    # Recommendations based on graph analysis
                    st.markdown("**🎯 Key Insights:**")
                    
                    # Find most connected missing entities
                    missing_connections = {}
                    for edge in relationships['edges']:
                        if edge['type'] == 'primary_to_missing':
                            target = edge['target']
                            if target not in missing_connections:
                                missing_connections[target] = []
                            missing_connections[target].append(edge['weight'])
                    
                    if missing_connections:
                        # Find missing entity with strongest connections
                        best_connected = max(missing_connections.items(), 
                                           key=lambda x: (len(x[1]), max(x[1])))
                        entity_id = best_connected[0]
                        entity_name = next(e['name'] for e in relationships['missing_entities'] 
                                         if e['id'] == entity_id)
                        
                        st.success(f"**Most Connected Missing Entity:** {entity_name} "
                                 f"({len(best_connected[1])} connections, max similarity: {max(best_connected[1]):.3f})")
                    
                    # Find query-relevant missing entities
                    query_missing_edges = [e for e in relationships['edges'] if e['type'] == 'query_to_missing']
                    if query_missing_edges:
                        best_query_edge = max(query_missing_edges, key=lambda x: x['weight'])
                        target_id = best_query_edge['target']
                        entity_name = next(e['name'] for e in relationships['missing_entities'] 
                                         if e['id'] == target_id)
                        
                        st.info(f"**Most Query-Relevant Missing Entity:** {entity_name} "
                               f"(similarity to query: {best_query_edge['weight']:.3f})")
            else:
                st.warning("Could not generate entity relationship graph. Please check if entities were extracted successfully.")
        
        # Missing Entity Location Analysis
        if missing_entities and st.session_state.content_passages.get(primary_url):
            st.subheader("📍 Where to Add Missing Entities in Your Content")
            st.markdown("_Select from all missing entities to find optimal insertion locations_")
            
            passages = st.session_state.content_passages[primary_url]
            
            # Sort missing entities by query relevance (show all, not just top 25)
            sorted_missing = sorted(missing_entities, key=lambda x: x['Query Relevance'], reverse=True)
            
            # Entity selection dropdown
            entity_options = []
            entity_lookup = {}
            
            for entity_info in sorted_missing:
                entity_name = entity_info['Entity']
                query_relevance = entity_info['Query Relevance']
                combined_score = entity_info['Combined Score']
                
                display_name = f"{entity_name} (Query: {query_relevance:.3f})"
                entity_options.append(display_name)
                entity_lookup[display_name] = entity_info
            
            selected_entity_display = st.selectbox(
                "🔍 Select missing entity to analyze:",
                options=[""] + entity_options,
                index=0,
                help=f"All {len(sorted_missing)} missing entities sorted by Query Relevance (highest first)."
            )
            
            if selected_entity_display and selected_entity_display != "":
                entity_info = entity_lookup[selected_entity_display]
                entity_name = entity_info['Entity']
                
                # Find best passage
                best_passage_info = find_entity_best_passage(
                    entity_name, 
                    passages, 
                    st.session_state.embedding_model
                )
                
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.markdown("**📊 Entity Metrics:**")
                    st.metric("Query Relevance", f"{entity_info['Query Relevance']:.3f}")
                    st.metric("Combined Score", f"{entity_info['Combined Score']:.3f}")
                    st.metric("Document Salience", f"{entity_info['Document Salience']:.3f}")
                    
                    st.markdown("**🏢 Details:**")
                    st.markdown(f"- **Type:** {entity_info['Type']}")
                    st.markdown(f"- **Found on:** {entity_info['Found On']} site(s)")
                
                with col_b:
                    if best_passage_info["similarity"] > 0.1:
                        st.markdown("**🎯 Recommended Insertion Location:**")
                        st.markdown(f"**Content Relevance:** {best_passage_info['similarity']:.3f}")
                        
                        st.text_area(
                            "Best passage for adding this entity:",
                            value=best_passage_info["passage"],
                            height=120,
                            disabled=True,
                            key=f"passage_{entity_name.replace(' ', '_')}"
                        )
                        
                        if best_passage_info["similarity"] > 0.5:
                            st.success("✅ **Excellent insertion point**")
                        elif best_passage_info["similarity"] > 0.3:
                            st.info("ℹ️ **Good insertion point**")
                        else:
                            st.warning("⚠️ **Lower relevance**")
                            
                        # Add Gemini semantic analysis
                        if st.session_state.get('gemini_api_configured', False):
                            if st.button(f"🤖 Get AI Implementation Strategy for '{entity_name}'", key=f"gemini_{entity_name.replace(' ', '_')}"):
                                with st.spinner("Generating strategic implementation analysis..."):
                                    # Get primary URL content for tone analysis
                                    primary_content = ""
                                    if primary_url in st.session_state.get('entity_analysis_results', {}):
                                        # Reconstruct content from passages
                                        passages = st.session_state.content_passages.get(primary_url, [])
                                        primary_content = " ".join(passages[:5])  # First 5 passages for tone analysis
                                    
                                    semantic_analysis = generate_semantic_implementation_analysis(
                                        entity_name,
                                        primary_content,
                                        best_passage_info["passage"],
                                        target_query,
                                        entity_info
                                    )
                                    
                                    if semantic_analysis:
                                        st.markdown("---")
                                        st.markdown("### 🤖 AI Implementation Strategy")
                                        st.markdown(semantic_analysis)
                                    else:
                                        st.error("Failed to generate implementation strategy.")
                        else:
                            st.info("💡 **Enable Gemini API** in the sidebar to get AI-powered implementation strategies that match your content tone.")
                    else:
                        st.markdown("**🤔 No strongly relevant passage found.**")
                        st.info(f"Consider adding a new section about '{entity_name}'.")
                        
                        # Gemini analysis for new section creation
                        if st.session_state.get('gemini_api_configured', False):
                            if st.button(f"🤖 Get AI Strategy for New '{entity_name}' Section", key=f"gemini_new_{entity_name.replace(' ', '_')}"):
                                with st.spinner("Generating new section strategy..."):
                                    # Generate strategy for creating new section
                                    primary_content = ""
                                    if primary_url in st.session_state.get('entity_analysis_results', {}):
                                        passages = st.session_state.content_passages.get(primary_url, [])
                                        primary_content = " ".join(passages[:5])
                                    
                                    semantic_analysis = generate_semantic_implementation_analysis(
                                        entity_name,
                                        primary_content,
                                        "No existing relevant passage found - new section needed",
                                        target_query,
                                        entity_info
                                    )
                                    
                                    if semantic_analysis:
                                        st.markdown("---")
                                        st.markdown("### 🤖 AI Strategy for New Section")
                                        st.markdown(semantic_analysis)
                                    else:
                                        st.error("Failed to generate new section strategy.")

# Footer
st.sidebar.divider()
st.sidebar.info("🎯 Entity Gap Analysis Tool v2.0")
st.sidebar.markdown("---")
st.sidebar.markdown("**Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)**")
