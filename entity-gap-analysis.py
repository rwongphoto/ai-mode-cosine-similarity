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
        disabled=st.session_state.processing,
        key="gcp_key_uploader"
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
    zyte_api_key_input = st.text_input(
        "Enter Zyte API Key:", 
        type="password", 
        value=st.session_state.get("zyte_api_key_to_persist", ""), 
        disabled=st.session_state.processing,
        key="zyte_api_key_input"
    )
    if st.button("Set & Verify Zyte Key", disabled=st.session_state.processing, key="zyte_verify_btn"):
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

with st.sidebar.expander("Gemini API (Optional)", expanded=False):
    gemini_api_key_input = st.text_input(
        "Enter Gemini API Key:", 
        type="password", 
        value=st.session_state.get("gemini_api_key_to_persist", ""), 
        disabled=st.session_state.processing,
        help="Required for AI-powered implementation strategies and content analysis",
        key="gemini_api_key_input"
    )
    
    if st.button("Set & Verify Gemini Key", disabled=st.session_state.processing, key="gemini_verify_btn"):
        if gemini_api_key_input:
            try:
                # Test the Gemini API key
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key_input)
                model = genai.GenerativeModel("gemini-2.0-flash-exp")
                
                # Simple test to verify the API key works
                test_response = model.generate_content("Hello, respond with 'API key works'")
                
                if test_response and test_response.text:
                    st.session_state.gemini_api_key_to_persist = gemini_api_key_input
                    st.session_state.gemini_api_configured = True
                    st.success("Gemini API Key Configured!")
                    st.rerun()
                else:
                    st.session_state.gemini_api_configured = False
                    st.error("Gemini API Key verification failed - no response received")
                    
            except Exception as e:
                st.session_state.gemini_api_configured = False
                st.error(f"Gemini API Key verification failed: {str(e)}")
        else:
            st.warning("Please enter a Gemini API Key.")
    
    # Clear key button
    if st.session_state.get('gemini_api_configured', False):
        if st.button("🗑️ Clear Gemini Key", disabled=st.session_state.processing, key="gemini_clear_btn"):
            st.session_state.gemini_api_key_to_persist = ""
            st.session_state.gemini_api_configured = False
            st.success("Gemini API key cleared!")
            st.rerun()

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

def calculate_entity_relationships(primary_entities, missing_entities, embedding_model, target_query, similarity_threshold=0.4):
    """Calculate relationships between primary entities and missing entities with custom threshold."""
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
        
        # Calculate edges with custom similarity threshold
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
        
        # Query to entity relationships with slightly lower threshold
        query_idx = len(all_entity_names) - 1
        query_threshold = max(0.3, similarity_threshold - 0.1)  # Slightly lower for query connections
        
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
    """Create an improved entity relationship graph with hierarchical layout."""
    
    if not relationships or (not relationships['primary_entities'] and not relationships['missing_entities']):
        return None
    
    # Prepare data
    primary_entities = relationships['primary_entities']
    missing_entities = relationships['missing_entities']
    edges = relationships['edges']
    query_entity = relationships['query_entity']
    
    # Create hierarchical layout - three main rows
    # Row 1: Missing entities (top)
    # Row 2: Query (center) 
    # Row 3: Primary entities (bottom)
    
    pos = {}
    
    # Layout missing entities in top row
    missing_count = len(missing_entities)
    if missing_count > 0:
        for i, entity in enumerate(missing_entities):
            x = (i - (missing_count - 1) / 2) * 1.5  # Spread horizontally
            y = 2.0  # Top row
            pos[entity['id']] = (x, y)
    
    # Layout query in center
    pos['query'] = (0, 0)  # Center position
    
    # Layout primary entities in bottom row
    primary_count = len(primary_entities)
    if primary_count > 0:
        for i, entity in enumerate(primary_entities):
            x = (i - (primary_count - 1) / 2) * 1.2  # Spread horizontally, slightly tighter
            y = -2.0  # Bottom row
            pos[entity['id']] = (x, y)
    
    # Create edge traces with different styles for different relationship types
    edge_traces = []
    
    # Query to Missing (red dashed lines)
    query_missing_x, query_missing_y = [], []
    query_missing_weights = []
    for edge in edges:
        if edge['type'] == 'query_to_missing':
            source_pos = pos.get(edge['source'])
            target_pos = pos.get(edge['target'])
            if source_pos and target_pos:
                query_missing_x.extend([source_pos[0], target_pos[0], None])
                query_missing_y.extend([source_pos[1], target_pos[1], None])
                query_missing_weights.append(edge['weight'])
    
    if query_missing_x:
        edge_traces.append(go.Scatter(
            x=query_missing_x, y=query_missing_y,
            line=dict(width=3, color='red', dash='dash'),
            mode='lines',
            name='Query → Missing Entities',
            hoverinfo='none',
            showlegend=True
        ))
    
    # Query to Primary (blue solid lines)
    query_primary_x, query_primary_y = [], []
    for edge in edges:
        if edge['type'] == 'query_to_primary':
            source_pos = pos.get(edge['source'])
            target_pos = pos.get(edge['target'])
            if source_pos and target_pos:
                query_primary_x.extend([source_pos[0], target_pos[0], None])
                query_primary_y.extend([source_pos[1], target_pos[1], None])
    
    if query_primary_x:
        edge_traces.append(go.Scatter(
            x=query_primary_x, y=query_primary_y,
            line=dict(width=3, color='blue'),
            mode='lines',
            name='Query → Your Content',
            hoverinfo='none',
            showlegend=True
        ))
    
    # Primary to Missing (orange dotted lines)
    primary_missing_x, primary_missing_y = [], []
    for edge in edges:
        if edge['type'] == 'primary_to_missing':
            source_pos = pos.get(edge['source'])
            target_pos = pos.get(edge['target'])
            if source_pos and target_pos:
                primary_missing_x.extend([source_pos[0], target_pos[0], None])
                primary_missing_y.extend([source_pos[1], target_pos[1], None])
    
    if primary_missing_x:
        edge_traces.append(go.Scatter(
            x=primary_missing_x, y=primary_missing_y,
            line=dict(width=2, color='orange', dash='dot'),
            mode='lines',
            name='Your Content ↔ Missing',
            hoverinfo='none',
            showlegend=True
        ))
    
    # Create node traces for each type
    node_traces = []
    
    # Query node (center, large, gold)
    query_connections = len([e for e in edges if 'query' in e['source'] or 'query' in e['target']])
    
    # Find query connections for hover
    query_connected_entities = []
    for edge in edges:
        if edge['source'] == 'query':
            target_entity = next((e for e in missing_entities + primary_entities if e['id'] == edge['target']), None)
            if target_entity:
                entity_type = "Missing" if target_entity in missing_entities else "Your Content"
                query_connected_entities.append(f"→ {target_entity['name']} ({entity_type}, {edge['weight']:.3f})")
        elif edge['target'] == 'query':
            source_entity = next((e for e in missing_entities + primary_entities if e['id'] == edge['source']), None)
            if source_entity:
                entity_type = "Missing" if source_entity in missing_entities else "Your Content"
                query_connected_entities.append(f"← {source_entity['name']} ({entity_type}, {edge['weight']:.3f})")
    
    query_connections_text = "<br>".join(query_connected_entities) if query_connected_entities else "No connections above threshold"
    
    query_trace = go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=40, color='gold', line=dict(width=3, color='darkgoldenrod')),
        text=[f"🎯"],
        textposition="middle center",
        textfont=dict(size=16, color='black'),
        name="Target Query",
        hovertemplate=f"<b>Target Query</b><br>{query_entity}<br><br><b>Connected to {query_connections} entities:</b><br>{query_connections_text}<extra></extra>",
        showlegend=True
    )
    node_traces.append(query_trace)
    
    # Missing entities (top row, orange/red)
    if missing_entities:
        missing_x = [pos[entity['id']][0] for entity in missing_entities]
        missing_y = [pos[entity['id']][1] for entity in missing_entities]
        missing_colors = []
        missing_sizes = []
        missing_text = []
        missing_hover = []
        
        for entity in missing_entities:
            # Highlight selected entity in red
            if selected_missing_entity and entity['name'] == selected_missing_entity:
                missing_colors.append('red')
                missing_sizes.append(30)
                icon = '🔴'
            else:
                missing_colors.append('orange')
                missing_sizes.append(15 + (entity.get('combined_score', 0) * 20))
                icon = '⚠️'
            
            missing_text.append(f"{icon}")
            
            # Find all connections for this entity with details
            entity_connections = []
            connection_count = 0
            for edge in edges:
                if edge['source'] == entity['id']:
                    if edge['target'] == 'query':
                        entity_connections.append(f"→ Query: {query_entity} ({edge['weight']:.3f})")
                    else:
                        target_entity = next((e for e in primary_entities if e['id'] == edge['target']), None)
                        if target_entity:
                            entity_connections.append(f"→ {target_entity['name']} (Your Content, {edge['weight']:.3f})")
                    connection_count += 1
                elif edge['target'] == entity['id']:
                    if edge['source'] == 'query':
                        entity_connections.append(f"← Query: {query_entity} ({edge['weight']:.3f})")
                    else:
                        source_entity = next((e for e in primary_entities if e['id'] == edge['source']), None)
                        if source_entity:
                            entity_connections.append(f"← {source_entity['name']} (Your Content, {edge['weight']:.3f})")
                    connection_count += 1
            
            connections_text = "<br>".join(entity_connections) if entity_connections else "No connections above threshold"
            
            # Enhanced hover info
            missing_hover.append(
                f"<b>{entity['name']}</b><br>"
                f"Type: {entity['type']}<br>"
                f"Combined Score: {entity.get('combined_score', 0):.3f}<br>"
                f"Query Relevance: {entity.get('query_relevance', 0):.3f}<br>"
                f"Status: Missing from your content<br>"
                f"Found on {entity.get('salience', 0):.3f} competitor sites<br>"
                f"<br><b>Connected to {connection_count} entities:</b><br>{connections_text}"
            )
        
        missing_trace = go.Scatter(
            x=missing_x, y=missing_y,
            mode='markers+text',
            marker=dict(size=missing_sizes, color=missing_colors, line=dict(width=2, color='white')),
            text=missing_text,
            textposition="middle center",
            textfont=dict(size=12),
            name="Missing Entities",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=missing_hover,
            showlegend=True
        )
        node_traces.append(missing_trace)
    
    # Primary entities (bottom row, blue)
    if primary_entities:
        primary_x = [pos[entity['id']][0] for entity in primary_entities]
        primary_y = [pos[entity['id']][1] for entity in primary_entities]
        primary_sizes = [15 + (entity.get('combined_score', 0) * 20) for entity in primary_entities]
        primary_text = []
        primary_hover = []
        
        for entity in primary_entities:
            primary_text.append("✅")
            
            # Find all connections for this entity with details
            entity_connections = []
            connection_count = 0
            for edge in edges:
                if edge['source'] == entity['id']:
                    if edge['target'] == 'query':
                        entity_connections.append(f"→ Query: {query_entity} ({edge['weight']:.3f})")
                    else:
                        target_entity = next((e for e in missing_entities if e['id'] == edge['target']), None)
                        if target_entity:
                            entity_connections.append(f"→ {target_entity['name']} (Missing, {edge['weight']:.3f})")
                    connection_count += 1
                elif edge['target'] == entity['id']:
                    if edge['source'] == 'query':
                        entity_connections.append(f"← Query: {query_entity} ({edge['weight']:.3f})")
                    else:
                        source_entity = next((e for e in missing_entities if e['id'] == edge['source']), None)
                        if source_entity:
                            entity_connections.append(f"← {source_entity['name']} (Missing, {edge['weight']:.3f})")
                    connection_count += 1
            
            connections_text = "<br>".join(entity_connections) if entity_connections else "No connections above threshold"
            
            # Enhanced hover info
            primary_hover.append(
                f"<b>{entity['name']}</b><br>"
                f"Type: {entity['type']}<br>"
                f"Combined Score: {entity.get('combined_score', 0):.3f}<br>"
                f"Query Relevance: {entity.get('query_relevance', 0):.3f}<br>"
                f"Status: In your content<br>"
                f"Document Salience: {entity.get('salience', 0):.3f}<br>"
                f"<br><b>Connected to {connection_count} entities:</b><br>{connections_text}"
            )
        
        primary_trace = go.Scatter(
            x=primary_x, y=primary_y,
            mode='markers+text',
            marker=dict(size=primary_sizes, color='lightblue', line=dict(width=2, color='blue')),
            text=primary_text,
            textposition="middle center",
            textfont=dict(size=12),
            name="Your Content Entities",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=primary_hover,
            showlegend=True
        )
        node_traces.append(primary_trace)
    
    # Add entity name labels as annotations
    annotations = []
    
    # Missing entity labels (above nodes)
    for entity in missing_entities:
        if entity['id'] in pos:
            x, y = pos[entity['id']]
            display_name = entity['name']
            if len(display_name) > 15:
                display_name = display_name[:12] + "..."
            
            annotations.append(dict(
                x=x, y=y + 0.3,
                text=display_name,
                showarrow=False,
                font=dict(size=10, color='red'),
                xanchor='center'
            ))
    
    # Primary entity labels (below nodes)
    for entity in primary_entities:
        if entity['id'] in pos:
            x, y = pos[entity['id']]
            display_name = entity['name']
            if len(display_name) > 15:
                display_name = display_name[:12] + "..."
            
            annotations.append(dict(
                x=x, y=y - 0.3,
                text=display_name,
                showarrow=False,
                font=dict(size=10, color='blue'),
                xanchor='center'
            ))
    
    # Query label
    annotations.append(dict(
        x=0, y=0.5,
        text=f"<b>{query_entity}</b>",
        showarrow=False,
        font=dict(size=12, color='darkgoldenrod'),
        xanchor='center'
    ))
    
    # Section labels
    annotations.extend([
        dict(
            text="<b>Missing Entities (Competitors Have)</b>",
            x=0, y=2.7,
            xref='x', yref='y',
            showarrow=False,
            font=dict(size=14, color='red'),
            xanchor='center'
        ),
        dict(
            text="<b>Your Content Entities</b>",
            x=0, y=-2.7,
            xref='x', yref='y',
            showarrow=False,
            font=dict(size=14, color='blue'),
            xanchor='center'
        )
    ])
    
    # Combine all traces
    all_traces = edge_traces + node_traces
    
    # Calculate dynamic range based on entity count
    max_entities = max(len(missing_entities), len(primary_entities))
    x_range = max(8, max_entities * 0.8)
    
    # Create figure with improved layout
    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            title=f"Entity Relationship Analysis: {query_entity}",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            hovermode='closest',
            margin=dict(b=80, l=40, r=40, t=100),
            annotations=annotations + [
                # Instructions
                dict(
                    text="🔴 Red dashed: Query gaps (high priority) | 🔵 Blue solid: Current coverage | 🟠 Orange dotted: Related opportunities<br><b>Hover for details • Click missing entities to highlight</b>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.12,
                    xanchor='center', yanchor='bottom',
                    font=dict(size=11)
                )
            ],
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[-x_range, x_range]
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[-3.5, 3.5]
            ),
            plot_bgcolor='white',
            height=700
        )
    )
    
    return fig

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
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
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
    disabled=st.session_state.processing,
    key="target_query_input"
)

scraping_method = st.sidebar.selectbox(
    "Scraping Method:",
    ["Requests (fast)", "Zyte API (best)", "Selenium (for JS)"],
    index=0,
    disabled=st.session_state.processing,
    key="scraping_method_select"
)

st.sidebar.subheader("🔗 URL Configuration")
primary_url = st.sidebar.text_input(
    "Your Primary URL:", 
    "https://cloudinary.com/guides/automatic-image-cropping/server-side-rendering-benefits-use-cases-and-best-practices",
    disabled=st.session_state.processing,
    key="primary_url_input"
)

competitor_urls = st.sidebar.text_area(
    "Competitor URLs (one per line):",
    "https://prismic.io/blog/what-is-ssr\nhttps://nextjs.org/docs/pages/building-your-application/rendering/server-side-rendering",
    height=100,
    disabled=st.session_state.processing,
    key="competitor_urls_input"
)

# --- Analysis Button ---
analysis_disabled = not st.session_state.gcp_nlp_configured or st.session_state.processing

if st.sidebar.button("🚀 Analyze Entity Gaps", disabled=analysis_disabled, type="primary", key="analyze_btn"):
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

        # Entity Relationship Graph with Enhanced Controls
        if primary_entity_analysis and missing_entities:
            st.markdown("---")
            st.subheader("🕸️ Entity Relationship Graph")
            st.markdown("_Visualize how your existing entities relate to missing entities and your target query_")
            
            # Graph Configuration Controls
            st.markdown("### 📊 Graph Controls")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_primary = st.slider(
                    "Max Primary Entities", 
                    min_value=3, 
                    max_value=25, 
                    value=12, 
                    step=1,
                    help="Show top N entities from your content",
                    key="max_primary_slider"
                )
            
            with col2:
                max_missing = st.slider(
                    "Max Missing Entities", 
                    min_value=3, 
                    max_value=20, 
                    value=8, 
                    step=1,
                    help="Show top N missing entities from competitors",
                    key="max_missing_slider"
                )
            
            with col3:
                similarity_threshold = st.slider(
                    "Connection Threshold", 
                    min_value=0.2, 
                    max_value=0.7, 
                    value=0.35, 
                    step=0.05,
                    help="Minimum similarity to show connections",
                    key="similarity_threshold_slider"
                )
            
            # Advanced Filters
            with st.expander("🔍 Advanced Filters", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Entity Type Filters
                    entity_types = set()
                    for entity in primary_entity_analysis + missing_entities:
                        entity_types.add(entity.get('Type', 'UNKNOWN'))
                    
                    selected_types = st.multiselect(
                        "Show entity types:",
                        options=sorted(entity_types),
                        default=list(entity_types),
                        help="Select which entity types to include in the graph",
                        key="entity_types_multiselect"
                    )
                
                with col_b:
                    # Score Threshold
                    min_score = st.slider(
                        "Minimum Combined Score",
                        min_value=0.0,
                        max_value=0.8,
                        value=0.1,
                        step=0.05,
                        help="Only show entities above this relevance threshold",
                        key="min_score_slider"
                    )
            
            # Quick Filter Presets
            st.markdown("**⚡ Quick Presets:**")
            col_q1, col_q2, col_q3, col_q4, col_q5 = st.columns(5)
            
            preset_applied = False
            
            with col_q1:
                if st.button("🎯 High Priority", help="Show only high-scoring entities", key="preset_high_priority"):
                    st.session_state.graph_preset = {
                        'max_primary': 8,
                        'max_missing': 5,
                        'min_score': 0.4,
                        'threshold': 0.5
                    }
                    preset_applied = True
            
            with col_q2:
                if st.button("📈 Balanced", help="Recommended balanced view", key="preset_balanced"):
                    st.session_state.graph_preset = {
                        'max_primary': 12,
                        'max_missing': 8,
                        'min_score': 0.2,
                        'threshold': 0.35
                    }
                    preset_applied = True
            
            with col_q3:
                if st.button("🔍 Detailed", help="Show more entities", key="preset_detailed"):
                    st.session_state.graph_preset = {
                        'max_primary': 18,
                        'max_missing': 12,
                        'min_score': 0.1,
                        'threshold': 0.3
                    }
                    preset_applied = True
            
            with col_q4:
                if st.button("🌐 Complete", help="Show all entities", key="preset_complete"):
                    st.session_state.graph_preset = {
                        'max_primary': 25,
                        'max_missing': 20,
                        'min_score': 0.05,
                        'threshold': 0.25
                    }
                    preset_applied = True
            
            with col_q5:
                if st.button("🔄 Reset", help="Reset to defaults", key="preset_reset"):
                    if 'graph_preset' in st.session_state:
                        del st.session_state.graph_preset
                    st.rerun()
            
            # Apply preset if selected
            if preset_applied:
                st.rerun()
            
            # Use preset values if available
            if 'graph_preset' in st.session_state:
                preset = st.session_state.graph_preset
                max_primary = preset.get('max_primary', max_primary)
                max_missing = preset.get('max_missing', max_missing)
                min_score = preset.get('min_score', min_score)
                similarity_threshold = preset.get('threshold', similarity_threshold)
            
            # Filter entities based on user selections
            filtered_primary = [
                e for e in primary_entity_analysis 
                if e.get('Type', 'UNKNOWN') in selected_types 
                and e.get('Combined Score', 0) >= min_score
            ][:max_primary]
            
            filtered_missing = [
                e for e in missing_entities 
                if e.get('Type', 'UNKNOWN') in selected_types 
                and e.get('Combined Score', 0) >= min_score
            ][:max_missing]
            
            # Entity selection for highlighting
            if filtered_missing:
                missing_entity_names = [entity['Entity'] for entity in filtered_missing]
                
                col_sel1, col_sel2 = st.columns([2, 1])
                
                with col_sel1:
                    selected_entity_for_graph = st.selectbox(
                        "🎯 Highlight missing entity:",
                        options=["None"] + missing_entity_names,
                        index=0,
                        help="Choose a missing entity to highlight in the graph",
                        key="selected_entity_for_graph"
                    )
                
                with col_sel2:
                    if st.button("🔄 Refresh Layout", key="refresh_layout_btn"):
                        # Force recalculation
                        if 'entity_relationships' in st.session_state:
                            del st.session_state.entity_relationships
                        st.rerun()
            else:
                selected_entity_for_graph = None
                st.warning("No entities match the selected filters.")
            
            # Calculate and display relationships
            if filtered_primary or filtered_missing:
                with st.spinner("Calculating entity relationships..."):
                    relationships = calculate_entity_relationships(
                        filtered_primary,
                        filtered_missing,
                        st.session_state.embedding_model,
                        target_query,
                        similarity_threshold=similarity_threshold
                    )
                    
                    if relationships and (relationships['primary_entities'] or relationships['missing_entities']):
                        # Create and display the graph
                        selected_entity = selected_entity_for_graph if selected_entity_for_graph != "None" else None
                        
                        entity_graph = create_entity_relationship_graph(
                            relationships,
                            selected_missing_entity=selected_entity
                        )
                        
                        if entity_graph:
                            st.plotly_chart(entity_graph, use_container_width=True)
                            
                            # Graph Statistics
                            with st.expander("📊 Graph Statistics", expanded=False):
                                total_primary = len(relationships['primary_entities'])
                                total_missing = len(relationships['missing_entities'])
                                total_edges = len(relationships['edges'])
                                
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                
                                with col_stat1:
                                    st.metric("Your Entities", total_primary)
                                    st.metric("Missing Entities", total_missing)
                                
                                with col_stat2:
                                    st.metric("Total Connections", total_edges)
                                    query_connections = len([e for e in relationships['edges'] if 'query' in e['source'] or 'query' in e['target']])
                                    st.metric("Query Connections", query_connections)
                                
                                with col_stat3:
                                    if total_edges > 0:
                                        avg_similarity = np.mean([e['weight'] for e in relationships['edges']])
                                        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                                    
                                    strong_connections = len([e for e in relationships['edges'] if e['weight'] > 0.5])
                                    st.metric("Strong Connections", strong_connections)
                                
                                with col_stat4:
                                    # Gap analysis
                                    gap_connections = len([e for e in relationships['edges'] if e['type'] == 'query_to_missing'])
                                    coverage_connections = len([e for e in relationships['edges'] if e['type'] == 'query_to_primary'])
                                    
                                    st.metric("High Priority Gaps", gap_connections)
                                    st.metric("Current Coverage", coverage_connections)
                                
                                # Insights
                                st.markdown("**🎯 Key Insights:**")
                                
                                # Find most connected missing entities
                                missing_connections = {}
                                for edge in relationships['edges']:
                                    if edge['type'] in ['primary_to_missing', 'query_to_missing']:
                                        target_id = edge['target'] if edge['type'] == 'primary_to_missing' else edge['target']
                                        if target_id.startswith('missing_'):
                                            if target_id not in missing_connections:
                                                missing_connections[target_id] = []
                                            missing_connections[target_id].append(edge['weight'])
                                
                                if missing_connections:
                                    # Find missing entity with strongest connections
                                    best_connected = max(missing_connections.items(), 
                                                       key=lambda x: (len(x[1]), max(x[1])))
                                    entity_id = best_connected[0]
                                    entity_name = next(e['name'] for e in relationships['missing_entities'] 
                                                     if e['id'] == entity_id)
                                    
                                    st.success(f"**Most Connected Gap:** {entity_name} "
                                             f"({len(best_connected[1])} connections, max: {max(best_connected[1]):.3f})")
                                
                                # Find query-relevant missing entities
                                query_missing_edges = [e for e in relationships['edges'] if e['type'] == 'query_to_missing']
                                if query_missing_edges:
                                    best_query_edge = max(query_missing_edges, key=lambda x: x['weight'])
                                    target_id = best_query_edge['target']
                                    entity_name = next(e['name'] for e in relationships['missing_entities'] 
                                                     if e['id'] == target_id)
                                    
                                    st.info(f"**Highest Priority Gap:** {entity_name} "
                                           f"(query similarity: {best_query_edge['weight']:.3f})")
                        else:
                            st.warning("Could not generate entity relationship graph.")
                    else:
                        st.warning("No relationships found with current filter settings. Try lowering the thresholds.")
            else:
                st.warning("No entities match the selected filters. Please adjust your criteria.")
        else:
            st.info("Entity relationship graph will appear once both primary and missing entities are analyzed.")
        
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
                help=f"All {len(sorted_missing)} missing entities sorted by Query Relevance (highest first).",
                key="selected_entity_display"
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
                                        passages_for_content = st.session_state.content_passages.get(primary_url, [])
                                        primary_content = " ".join(passages_for_content[:5])  # First 5 passages for tone analysis
                                    
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
                                        passages_for_content = st.session_state.content_passages.get(primary_url, [])
                                        primary_content = " ".join(passages_for_content[:5])
                                    
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
