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
                model = genai.GenerativeModel("gemini-1.5-flash") # Using 1.5-flash for speed

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
    """Calculate relationships between the given lists of primary and missing entities."""
    if not embedding_model:
        return {}

    # The entities passed to this function are already filtered by the UI.
    # We use them directly.
    top_primary = primary_entities
    top_missing = missing_entities

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
    """Create an improved entity relationship graph with a clear hierarchical layout."""
    if not relationships or (not relationships['primary_entities'] and not relationships['missing_entities']):
        st.info("Not enough data to generate a relationship graph.")
        return None

    primary_entities = relationships['primary_entities']
    missing_entities = relationships['missing_entities']
    edges = relationships['edges']
    query_entity = relationships['query_entity']

    # --- 1. Create Hierarchical Layout Positions ---
    pos = {}
    # Top row for Missing Entities
    missing_count = len(missing_entities)
    if missing_count > 0:
        for i, entity in enumerate(missing_entities):
            x = (i - (missing_count - 1) / 2) * 1.5
            y = 2.0
            pos[entity['id']] = (x, y)

    # Center for the Query
    pos['query'] = (0, 0)

    # Bottom row for Primary Entities
    primary_count = len(primary_entities)
    if primary_count > 0:
        for i, entity in enumerate(primary_entities):
            x = (i - (primary_count - 1) / 2) * 1.2
            y = -2.0
            pos[entity['id']] = (x, y)

    # --- 2. Create Edge Traces with Different Styles ---
    edge_traces = []

    # Type 1: Query to Missing (High-Priority Gaps) - Red Dashed
    q_to_m_x, q_to_m_y = [], []
    for edge in [e for e in edges if e['type'] == 'query_to_missing']:
        source_pos, target_pos = pos.get(edge['source']), pos.get(edge['target'])
        if source_pos and target_pos:
            q_to_m_x.extend([source_pos[0], target_pos[0], None])
            q_to_m_y.extend([source_pos[1], target_pos[1], None])
    if q_to_m_x:
        edge_traces.append(go.Scatter(x=q_to_m_x, y=q_to_m_y, line=dict(width=2, color='red', dash='dash'), hoverinfo='none', mode='lines', name='Query → Missing (Gap)'))

    # Type 2: Query to Primary (Current Coverage) - Blue Solid
    q_to_p_x, q_to_p_y = [], []
    for edge in [e for e in edges if e['type'] == 'query_to_primary']:
        source_pos, target_pos = pos.get(edge['source']), pos.get(edge['target'])
        if source_pos and target_pos:
            q_to_p_x.extend([source_pos[0], target_pos[0], None])
            q_to_p_y.extend([source_pos[1], target_pos[1], None])
    if q_to_p_x:
        edge_traces.append(go.Scatter(x=q_to_p_x, y=q_to_p_y, line=dict(width=1.5, color='blue'), hoverinfo='none', mode='lines', name='Query → Your Content'))

    # Type 3: Primary to Missing (Bridge Opportunity) - Orange Dotted
    p_to_m_x, p_to_m_y = [], []
    for edge in [e for e in edges if e['type'] == 'primary_to_missing']:
        source_pos, target_pos = pos.get(edge['source']), pos.get(edge['target'])
        if source_pos and target_pos:
            p_to_m_x.extend([source_pos[0], target_pos[0], None])
            p_to_m_y.extend([source_pos[1], target_pos[1], None])
    if p_to_m_x:
        edge_traces.append(go.Scatter(x=p_to_m_x, y=p_to_m_y, line=dict(width=1, color='orange', dash='dot'), hoverinfo='none', mode='lines', name='Your Content ↔ Missing'))

    # --- 3. Create Node Traces with Rich Hover Info ---
    node_traces = []

    # Helper to build connection text for hover info
    def get_connection_text(entity_id, all_nodes, all_edges, query_name):
        connections = []
        for edge in all_edges:
            if entity_id == edge['source']:
                if edge['target'] == 'query':
                    connections.append(f"→ Query ({edge['weight']:.2f})")
                else:
                    target_node = next((n for n in all_nodes if n['id'] == edge['target']), None)
                    if target_node: connections.append(f"→ {target_node['name']} ({edge['weight']:.2f})")
            elif entity_id == edge['target']:
                if edge['source'] == 'query':
                    connections.append(f"← Query ({edge['weight']:.2f})")
                else:
                    source_node = next((n for n in all_nodes if n['id'] == edge['source']), None)
                    if source_node: connections.append(f"← {source_node['name']} ({edge['weight']:.2f})")
        return "<br>".join(connections) if connections else "No direct connections"

    # Query Node
    query_connections = get_connection_text('query', primary_entities + missing_entities, edges, query_entity)
    query_hover = f"<b>🎯 Target Query: {query_entity}</b><br><br><b>Connections:</b><br>{query_connections}<extra></extra>"
    node_traces.append(go.Scatter(x=[0], y=[0], text=['🎯'], mode='markers+text', marker=dict(size=40, color='gold', line=dict(width=3, color='darkgoldenrod')), textposition='middle center', textfont=dict(size=20), hovertemplate=query_hover, name='Target Query'))

    # Missing Entity Nodes
    if missing_entities:
        m_x = [pos[e['id']][0] for e in missing_entities]
        m_y = [pos[e['id']][1] for e in missing_entities]
        m_sizes = [15 + (e.get('combined_score', 0) * 25) for e in missing_entities]
        m_colors = ['red' if e['name'] == selected_missing_entity else 'orange' for e in missing_entities]
        m_hover = [f"<b>⚠️ Missing: {e['name']}</b><br>Type: {e['type']}<br>Combined Score: {e['combined_score']:.3f}<br>Query Relevance: {e['query_relevance']:.3f}<br><br><b>Connections:</b><br>{get_connection_text(e['id'], primary_entities, edges, query_entity)}<extra></extra>" for e in missing_entities]
        node_traces.append(go.Scatter(x=m_x, y=m_y, mode='markers', marker=dict(size=m_sizes, color=m_colors, line=dict(width=2, color='white')), hovertemplate=m_hover, customdata=m_hover, name='Missing Entities'))

    # Primary Entity Nodes
    if primary_entities:
        p_x = [pos[e['id']][0] for e in primary_entities]
        p_y = [pos[e['id']][1] for e in primary_entities]
        p_sizes = [15 + (e.get('combined_score', 0) * 25) for e in primary_entities]
        p_hover = [f"<b>✅ In Your Content: {e['name']}</b><br>Type: {e['type']}<br>Combined Score: {e['combined_score']:.3f}<br>Query Relevance: {e['query_relevance']:.3f}<br><br><b>Connections:</b><br>{get_connection_text(e['id'], missing_entities, edges, query_entity)}<extra></extra>" for e in primary_entities]
        node_traces.append(go.Scatter(x=p_x, y=p_y, mode='markers', marker=dict(size=p_sizes, color='lightblue', line=dict(width=2, color='blue')), hovertemplate=p_hover, customdata=p_hover, name='Your Content Entities'))

    # --- 4. Create Annotations for Labels ---
    annotations = []
    # Section Titles
    if missing_entities: annotations.append(dict(x=0, y=2.7, text="<b>Missing Entities (Content Gaps)</b>", showarrow=False, font=dict(size=16, color='red'), xanchor='center'))
    if primary_entities: annotations.append(dict(x=0, y=-2.7, text="<b>Your Content's Entities</b>", showarrow=False, font=dict(size=16, color='blue'), xanchor='center'))
    # Entity Name Labels
    for entity_list, y_offset, color in [(missing_entities, 0.4, 'red'), (primary_entities, -0.4, 'blue')]:
        for entity in entity_list:
            if entity['id'] in pos:
                x, y = pos[entity['id']]
                display_name = entity['name'] if len(entity['name']) < 20 else entity['name'][:17] + '...'
                annotations.append(dict(x=x, y=y + y_offset, text=display_name, showarrow=False, font=dict(size=10, color=color), xanchor='center'))

    # --- 5. Assemble the Figure ---
    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title=f"Entity Relationship Map for '{query_entity}'",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=100),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
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

        BEST INSERTION LOCATION (Context):
        "{best_passage}"

        CURRENT CONTENT SAMPLE (for broader context):
        "{primary_content[:1500]}"

        Provide a strategic analysis with the following structure:

        ### 1. Why Implement This Entity
        - **Strategic Value:** Explain its importance for the target query "{target_query}".
        - **SEO & User Benefit:** How does it close a content gap and improve user experience?

        ### 2. Where to Implement
        - **Optimal Placement:** Recommend a specific section or paragraph for insertion, referencing the provided "Best Insertion Location" context.
        - **Integration:** How can it be woven into the existing content flow naturally?

        ### 3. How to Implement (Actionable Examples)
        - **Content Suggestions:** Provide 2-3 specific, actionable examples of sentences or short paragraphs that could be added. These examples **must** match the tone and style identified in the analysis.
        - **Depth & Angle:** Suggest the angle to take (e.g., definition, example, benefit, comparison).

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
            st.markdown("_Visualize how your existing entities relate to missing entities and your target query._")

            # Graph Configuration Controls
            with st.expander("📊 Graph & Filter Controls", expanded=True):
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
                        help="Minimum similarity to show a connection line between entities.",
                        key="similarity_threshold_slider"
                    )

                # Entity Type Filters
                entity_types = sorted(list(set(e.get('Type', 'UNKNOWN') for e in primary_entity_analysis + missing_entities)))

                selected_types = st.multiselect(
                    "Filter by Entity Type:",
                    options=entity_types,
                    default=entity_types,
                    help="Select which entity types to include in the graph",
                    key="entity_types_multiselect"
                )

            # Filter entities based on user selections
            # Sort first, then filter by type, then slice
            sorted_primary = sorted(primary_entity_analysis, key=lambda x: x['Combined Score'], reverse=True)
            sorted_missing = sorted(missing_entities, key=lambda x: x['Combined Score'], reverse=True)

            filtered_primary = [e for e in sorted_primary if e.get('Type', 'UNKNOWN') in selected_types][:max_primary]
            filtered_missing = [e for e in sorted_missing if e.get('Type', 'UNKNOWN') in selected_types][:max_missing]

            # Entity selection for highlighting
            if filtered_missing:
                missing_entity_names = [entity['Entity'] for entity in filtered_missing]
                selected_entity_for_graph = st.selectbox(
                    "🎯 Highlight a Missing Entity in the Graph:",
                    options=["None"] + missing_entity_names,
                    index=0,
                    help="Choose a missing entity to highlight in red on the graph below.",
                    key="selected_entity_for_graph"
                )
            else:
                selected_entity_for_graph = None
                st.warning("No missing entities match the selected filters.")

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
                        selected_entity = selected_entity_for_graph if selected_entity_for_graph != "None" else None
                        entity_graph = create_entity_relationship_graph(relationships, selected_missing_entity=selected_entity)
                        if entity_graph:
                            st.plotly_chart(entity_graph, use_container_width=True)
                        else:
                            st.warning("Could not generate entity relationship graph with current settings.")
                    else:
                        st.warning("No relationships found with current filter settings. Try lowering the thresholds.")
            else:
                st.warning("No entities to display. Please adjust your filters.")
        else:
            st.info("Entity relationship graph will appear once both primary and missing entities are analyzed.")

        # Missing Entity Location Analysis
        if missing_entities and st.session_state.content_passages.get(primary_url):
            st.markdown("---")
            st.subheader("📍 Where to Add Missing Entities in Your Content")
            st.markdown("_Select from all missing entities to find optimal insertion locations and get AI-powered implementation advice._")

            passages = st.session_state.content_passages[primary_url]

            # Sort missing entities by query relevance (show all, not just top N)
            all_sorted_missing = sorted(missing_entities, key=lambda x: x['Query Relevance'], reverse=True)

            # Entity selection dropdown
            entity_options = [""] + [f"{e['Entity']} (Query Relevance: {e['Query Relevance']:.2f})" for e in all_sorted_missing]
            entity_lookup = {f"{e['Entity']} (Query Relevance: {e['Query Relevance']:.2f})": e for e in all_sorted_missing}

            selected_entity_display = st.selectbox(
                "🔍 Select a missing entity to analyze:",
                options=entity_options,
                index=0,
                help=f"All {len(all_sorted_missing)} missing entities are listed, sorted by relevance to your target query.",
                key="selected_entity_display"
            )

            if selected_entity_display:
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
                    st.markdown(f"**📊 Metrics for '{entity_name}'**")
                    st.metric("Query Relevance", f"{entity_info['Query Relevance']:.3f}")
                    st.metric("Combined Score", f"{entity_info['Combined Score']:.3f}")
                    st.metric("Competitor Salience", f"{entity_info['Document Salience']:.3f}")
                    st.markdown(f"- **Type:** `{entity_info['Type']}`")
                    st.markdown(f"- **Found on:** `{entity_info['Found On']}` competitor site(s)")

                with col_b:
                    if best_passage_info["similarity"] > 0.1:
                        st.markdown("**🎯 Recommended Insertion Context**")
                        st.markdown(f"The most semantically similar passage in your content (Relevance: **{best_passage_info['similarity']:.2f}**):")

                        st.text_area(
                            "Best passage for adding this entity:",
                            value=best_passage_info["passage"],
                            height=150,
                            disabled=True,
                            key=f"passage_{entity_name.replace(' ', '_')}"
                        )
                    else:
                        st.markdown("**🤔 No strongly relevant passage found.**")
                        st.info(f"This may indicate a need for a completely new section about '{entity_name}'.")

                    # Add Gemini semantic analysis
                    if st.session_state.get('gemini_api_configured', False):
                        button_text = f"🤖 Get AI Implementation Strategy for '{entity_name}'"
                        if st.button(button_text, key=f"gemini_{entity_name.replace(' ', '_')}"):
                            with st.spinner("Generating strategic implementation analysis..."):
                                passages_for_content = st.session_state.content_passages.get(primary_url, [])
                                primary_content_sample = " ".join(passages_for_content)

                                semantic_analysis = generate_semantic_implementation_analysis(
                                    entity_name,
                                    primary_content_sample,
                                    best_passage_info["passage"],
                                    target_query,
                                    entity_info
                                )
                                if semantic_analysis:
                                    st.markdown("---")
                                    st.markdown(semantic_analysis)
                                else:
                                    st.error("Failed to generate implementation strategy.")
                    else:
                        st.info("💡 **Enable the Gemini API** in the sidebar to get AI-powered implementation strategies that match your content's tone and style.")

# Footer
st.sidebar.divider()
st.sidebar.info("🎯 Entity Gap Analysis Tool v2.1")
st.sidebar.markdown("**Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)**")

