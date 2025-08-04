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
if "graph_config" not in st.session_state: st.session_state.graph_config = {
    'max_primary': 15,
    'max_missing': 10,
    'min_score': 0.15,
    'threshold': 0.4
}

REQUEST_INTERVAL = 3.0
last_request_time = 0

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.2151.97", #Edge
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1", #Safari iPhone
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1", #Safari iPad
    "Mozilla/5.0 (Android 14; Mobile; rv:121.0) Gecko/121.0 Firefox/121.0", #Firefox Android
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36", #Chrome Android
]

# --- Core Functions ---

@st.cache_data
def get_image_as_base64(url):
    """Fetches an image from a URL and returns it as a Base64 encoded string."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
    except Exception as e:
        st.error(f"Failed to fetch logo from {url}: {e}")
        return ""

def create_navigation_menu(logo_data_uri):
    """Creates a branded header with a logo and navigation links."""
    menu_options = {
        "Home": "https://theseoconsultant.ai/", "About": "https://theseoconsultant.ai/about/",
        "Services": "https://theseoconsultant.ai/seo-services/", "Blog": "https://theseoconsultant.ai/blog/",
        "Contact": "https://theseoconsultant.ai/contact/"
    }
    st.markdown("""
        <style>
        #MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}
        .topnav {overflow: hidden; background-color: #f1f1f1; display: flex; justify-content: center; margin-bottom: 35px; border-radius: 10px; border: 1px solid #e0e0e0;}
        .topnav a {float: left; display: block; color: black; text-align: center; padding: 14px 16px; text-decoration: none; font-size: 17px;}
        .topnav a:hover {background-color: #ddd; color: black;}
        .logo-container {display: flex; justify-content: center; margin-bottom: 20px;}
        </style>""", unsafe_allow_html=True)
    if logo_data_uri:
        st.markdown(f'<div class="logo-container"><img src="data:image/png;base64,{logo_data_uri}" width="350"></div>', unsafe_allow_html=True)
    menu_html = "<div class='topnav'>" + "".join([f"<a href='{value}' target='_blank'>{key}</a>" for key, value in menu_options.items()]) + "</div>"
    st.markdown(menu_html, unsafe_allow_html=True)

def get_random_user_agent(): return random.choice(USER_AGENTS)

def enforce_rate_limit():
    global last_request_time; now = time.time(); elapsed = now - last_request_time
    if elapsed < REQUEST_INTERVAL: time.sleep(REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

# --- Sidebar API Configuration ---
st.sidebar.header("🔑 API Configuration")
with st.sidebar.expander("Google Cloud NLP API", expanded=not st.session_state.gcp_nlp_configured):
    uploaded_gcp_key = st.file_uploader("Upload Google Cloud Service Account JSON", type="json", help="Upload the JSON key file for a service account with 'Cloud Natural Language API User' role.", disabled=st.session_state.processing)
    if uploaded_gcp_key:
        try:
            credentials_info = json.load(uploaded_gcp_key)
            if "project_id" in credentials_info and "private_key" in credentials_info:
                st.session_state.gcp_credentials_info = credentials_info; st.session_state.gcp_nlp_configured = True
                st.success(f"GCP Key for project '{credentials_info['project_id']}' loaded!")
            else: st.error("Invalid JSON key file format."); st.session_state.gcp_nlp_configured = False; st.session_state.gcp_credentials_info = None
        except Exception as e: st.error(f"Failed to process GCP key file: {e}"); st.session_state.gcp_nlp_configured = False; st.session_state.gcp_credentials_info = None
with st.sidebar.expander("Zyte API (Optional)", expanded=False):
    zyte_api_key_input = st.text_input("Enter Zyte API Key:", type="password", value=st.session_state.get("zyte_api_key_to_persist", ""), disabled=st.session_state.processing)
    if st.button("Set & Verify Zyte Key", disabled=st.session_state.processing):
        if zyte_api_key_input:
            try:
                response = requests.post("https://api.zyte.com/v1/extract", auth=(zyte_api_key_input, ''), json={'url': 'https://toscrape.com/', 'httpResponseBody': True}, timeout=20)
                if response.status_code == 200: st.session_state.zyte_api_key_to_persist = zyte_api_key_input; st.session_state.zyte_api_configured = True; st.success("Zyte API Key Configured!"); st.rerun()
                else: st.session_state.zyte_api_configured = False; st.error(f"Zyte Key Failed. Status: {response.status_code}")
            except Exception as e: st.session_state.zyte_api_configured = False; st.error(f"Zyte API Request Failed: {str(e)[:200]}")
        else: st.warning("Please enter Zyte API Key.")
with st.sidebar.expander("Gemini API (Optional)", expanded=False):
    gemini_api_key_input = st.text_input("Enter Gemini API Key:", type="password", value=st.session_state.get("gemini_api_key_to_persist", ""), disabled=st.session_state.processing, help="Required for AI-powered implementation strategies and content analysis")
    if st.button("Set & Verify Gemini Key", disabled=st.session_state.processing):
        if gemini_api_key_input:
            try:
                import google.generativeai as genai; genai.configure(api_key=gemini_api_key_input); model = genai.GenerativeModel("gemini-2.5-pro")
                test_response = model.generate_content("Hello, respond with 'API key works'")
                if test_response and test_response.text: st.session_state.gemini_api_key_to_persist = gemini_api_key_input; st.session_state.gemini_api_configured = True; st.success("Gemini API Key Configured!"); st.rerun()
                else: st.session_state.gemini_api_configured = False; st.error("Gemini API Key verification failed - no response received")
            except Exception as e: st.session_state.gemini_api_configured = False; st.error(f"Gemini API Key verification failed: {str(e)}")
        else: st.warning("Please enter a Gemini API Key.")
    if st.session_state.get('gemini_api_configured', False):
        if st.button("🗑️ Clear Gemini Key", disabled=st.session_state.processing): st.session_state.gemini_api_key_to_persist = ""; st.session_state.gemini_api_configured = False; st.success("Gemini API key cleared!"); st.rerun()
st.sidebar.markdown("---")
if st.session_state.get("gcp_nlp_configured"): st.sidebar.markdown("✅ Google NLP API: **Required - Configured**")
else: st.sidebar.markdown("⚠️ Google NLP API: **Required - Not Configured**")
st.sidebar.markdown(f"🔧 Zyte API: **{'Configured' if st.session_state.zyte_api_configured else 'Optional - Not Configured'}**")
st.sidebar.markdown(f"🤖 Gemini API: **{'Configured' if st.session_state.get('gemini_api_configured', False) else 'Optional - Not Configured'}**")

# --- (Core Functions Start Here) ---
def is_number_entity(entity_name):
    if not entity_name: return True
    cleaned = re.sub(r'[,\s\-\.]', '', entity_name)
    if cleaned.isdigit(): return True
    if entity_name.strip().endswith('%') and re.sub(r'[%,\s\-\.]', '', entity_name).isdigit(): return True
    if re.match(r'^\d{4}$', cleaned): return True
    digit_count = sum(1 for char in entity_name if char.isdigit()); total_chars = len(re.sub(r'\s', '', entity_name))
    if total_chars > 0 and (digit_count / total_chars) > 0.7: return True
    if len(entity_name.strip()) <= 4 and any(char.isdigit() for char in entity_name): return True
    return False

@st.cache_data(show_spinner="Extracting entities...")
def extract_entities_with_google_nlp(text: str, _credentials_info: dict):
    if not _credentials_info or not text: return {}
    try:
        credentials = service_account.Credentials.from_service_account_info(_credentials_info); client = language_v1.LanguageServiceClient(credentials=credentials)
        text_bytes = text.encode('utf-8')
        if len(text_bytes) > 900000: text = text_bytes[:900000].decode('utf-8', 'ignore'); st.warning("Text was truncated.")
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = client.analyze_entities(document=document, encoding_type=language_v1.EncodingType.UTF8)
        entities_dict = {}
        for entity in response.entities:
            entity_name = entity.name.strip()
            if is_number_entity(entity_name): continue
            key = entity_name.lower()
            if key not in entities_dict or entity.salience > entities_dict[key]['salience']:
                entities_dict[key] = {'name': entity_name, 'type': language_v1.Entity.Type(entity.type_).name, 'salience': entity.salience, 'mentions': len(entity.mentions)}
        return entities_dict
    except Exception as e: st.error(f"Google Cloud NLP API Error: {e}"); return {}

@st.cache_resource
def load_embedding_model():
    try: return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e: st.error(f"Failed to load embedding model: {e}"); return None

def calculate_entity_query_relevance(entity_name, query_text, model):
    if not model or not entity_name or not query_text: return 0.0
    try:
        entity_embedding = model.encode([entity_name]); query_embedding = model.encode([query_text])
        return float(cosine_similarity(entity_embedding, query_embedding)[0][0])
    except Exception as e: st.warning(f"Failed to calculate similarity for '{entity_name}': {e}"); return 0.0

def calculate_entity_relationships(primary_entities, missing_entities, embedding_model, target_query, primary_content_passages, similarity_threshold=0.4):
    if not embedding_model or not primary_content_passages: return {}
    top_primary = sorted(primary_entities, key=lambda x: x['Combined Score'], reverse=True)
    top_missing = sorted(missing_entities, key=lambda x: x['Combined Score'], reverse=True)
    relationships = {'primary_entities': [], 'missing_entities': [], 'content_sections': [], 'edges': [], 'query_entity': target_query}
    try:
        primary_names = [e['Entity'] for e in top_primary]; missing_names = [e['Entity'] for e in top_missing]
        all_entity_names = primary_names + missing_names
        if not all_entity_names: return relationships
        entity_embeddings = embedding_model.encode(all_entity_names); passage_embeddings = embedding_model.encode(primary_content_passages)
        query_embedding = embedding_model.encode([target_query]); entity_embedding_map = {name: emb for name, emb in zip(all_entity_names, entity_embeddings)}
        connected_passages = {}
        for i, entity in enumerate(top_primary):
            if entity['Entity'] in entity_embedding_map:
                sims = cosine_similarity([entity_embedding_map[entity['Entity']]], passage_embeddings)[0]
                idx, score = np.argmax(sims), np.max(sims)
                if score > similarity_threshold:
                    if idx not in connected_passages: connected_passages[idx] = {'entities': [], 'query_similarity': 0.0}
                    connected_passages[idx]['entities'].append({'type': 'primary', 'id': f"primary_{i}", 'score': float(score)})
        for i, entity in enumerate(top_missing):
            if entity['Entity'] in entity_embedding_map:
                sims = cosine_similarity([entity_embedding_map[entity['Entity']]], passage_embeddings)[0]
                idx, score = np.argmax(sims), np.max(sims)
                if score > similarity_threshold:
                    if idx not in connected_passages: connected_passages[idx] = {'entities': [], 'query_similarity': 0.0}
                    connected_passages[idx]['entities'].append({'type': 'missing', 'id': f"missing_{i}", 'score': float(score)})
        query_passage_sims = cosine_similarity(query_embedding, passage_embeddings)[0]
        for i, score in enumerate(query_passage_sims):
            if score > similarity_threshold:
                 if i not in connected_passages: connected_passages[i] = {'entities': [], 'query_similarity': 0.0}
                 connected_passages[i]['query_similarity'] = float(score)
        for i, entity in enumerate(top_primary): relationships['primary_entities'].append({'id': f"primary_{i}", 'name': entity['Entity'], 'type': entity.get('Type', 'UNKNOWN'), 'combined_score': entity.get('Combined Score', 0), 'node_type': 'primary'})
        for i, entity in enumerate(top_missing): relationships['missing_entities'].append({'id': f"missing_{i}", 'name': entity['Entity'], 'type': entity.get('Type', 'UNKNOWN'), 'combined_score': entity.get('Combined Score', 0), 'node_type': 'missing'})
        for idx, conns in connected_passages.items():
            sec_id = f"section_{idx}"; relationships['content_sections'].append({'id': sec_id, 'name': f"Section {idx + 1}", 'text': primary_content_passages[idx], 'node_type': 'section', 'connection_count': len(conns['entities'])})
            for e_conn in conns['entities']: relationships['edges'].append({'source': e_conn['id'], 'target': sec_id, 'weight': e_conn['score'], 'type': 'entity_to_section'})
            if conns['query_similarity'] > similarity_threshold: relationships['edges'].append({'source': 'query', 'target': sec_id, 'weight': conns['query_similarity'], 'type': 'query_to_section'})
        query_entity_sims = cosine_similarity(query_embedding, entity_embeddings)[0]; query_threshold = max(similarity_threshold, 0.5)
        all_entities_with_ids = relationships['primary_entities'] + relationships['missing_entities']
        for i, entity in enumerate(all_entities_with_ids):
            if query_entity_sims[i] > query_threshold: relationships['edges'].append({'source': 'query', 'target': entity['id'], 'weight': float(query_entity_sims[i]), 'type': 'query_to_entity'})
        return relationships
    except Exception as e: st.error(f"Error calculating entity relationships: {e}"); return {}

def create_entity_relationship_graph(relationships, selected_missing_entity=None):
    if not relationships or (not relationships.get('primary_entities') and not relationships.get('missing_entities') and not relationships.get('content_sections')): return None
    G = nx.Graph(); all_nodes = relationships['primary_entities'] + relationships['missing_entities'] + relationships['content_sections']
    G.add_node('query', node_type='query', name=relationships['query_entity'])
    for node in all_nodes: G.add_node(node['id'], **node)
    for edge in relationships.get('edges', []):
        if G.has_node(edge['source']) and G.has_node(edge['target']): G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    try: pos = nx.spring_layout(G, k=(3 / np.sqrt(G.order()) if G.order() > 1 else 1.0), iterations=75, seed=42)
    except Exception as e: st.warning(f"Spring layout failed ({e}), falling back to circular."); pos = nx.circular_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges(): x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='rgba(125,125,125,0.5)'), hoverinfo='none', mode='lines', showlegend=False)
    node_x, node_y, node_text, node_colors, node_sizes, node_info, node_border_colors, node_border_widths = [], [], [], [], [], [], [], []
    for node_id, node_data in G.nodes(data=True):
        if node_id not in pos: continue
        x, y = pos[node_id]; node_x.append(x); node_y.append(y)
        display_name = node_data.get('name', 'Unknown')[:22] + "..." if len(node_data.get('name', 'Unknown')) > 25 else node_data.get('name', 'Unknown')
        info = f"<b>{node_data.get('name', 'Unknown')}</b>"; border_color, border_width = 'white', 2
        node_type = node_data.get('node_type')
        if node_type == 'query': color, icon, size = 'gold', '🎯', 30; node_text.append(f"{icon}")
        elif node_type == 'primary': color, icon, size = 'lightblue', '🔵', 15 + (node_data.get('combined_score', 0) * 15); node_text.append(f"{icon} {display_name}"); info += f" (Your Content)<br>Type: {node_data['type']}<br>Score: {node_data.get('combined_score', 0):.3f}"
        elif node_type == 'missing':
            color, icon, size = 'orange', '🟠', 15 + (node_data.get('combined_score', 0) * 15)
            if selected_missing_entity and node_data['name'] == selected_missing_entity: color, icon, size = 'red', '🔴', 35; border_color, border_width = 'black', 3
            node_text.append(f"{icon} {display_name}"); info += f" (Missing Entity)<br>Type: {node_data['type']}<br>Score: {node_data.get('combined_score', 0):.3f}"
        elif node_type == 'section':
            color, icon, size = '#C8A2C8', '📄', 15 + (node_data.get('connection_count', 0) * 3); node_text.append(f"{icon} {display_name}")
            passage_snippet = node_data.get('text', '')[:197] + "..." if len(node_data.get('text', '')) > 200 else node_data.get('text', '')
            info += f"<br><i>Passage Snippet:</i><br>...{passage_snippet}..."
        else: color, icon, size = 'grey', '❓', 10; node_text.append(f"{icon} {display_name}")
        connections = []
        for neighbor in G.neighbors(node_id):
            n_data = G.nodes[neighbor]; weight = G.get_edge_data(node_id, neighbor)['weight']; n_name, n_type = n_data.get('name', 'Unknown'), n_data.get('node_type')
            n_icon = {'primary': '🔵', 'missing': '🟠', 'section': '📄', 'query': '🎯'}.get(n_type, '❓'); connections.append(f"{n_icon} {n_name} ({weight:.2f})")
        connections_html = "<br>".join(sorted(connections, key=lambda x: x.split(' ')[-1], reverse=True)) or "No connections"
        info += f"<br><br><b>Connected to:</b><br>{connections_html}"; node_colors.append(color); node_sizes.append(size); node_info.append(info); node_border_colors.append(border_color); node_border_widths.append(border_width)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', hovertext=node_info, text=node_text, textposition="top center", marker=dict(size=node_sizes, color=node_colors, line=dict(width=node_border_widths, color=node_border_colors)), textfont=dict(size=10, color='black'), showlegend=False)
    annotation_text = "🔵 Your Entity | 🟠 Missing Entity | 🔴 **Selected Missing** | 📄 Content Section | 🎯 Target Query<br><b>Blue/Orange → Purple:</b> Shows entity relevance to content sections. | <b>Gold → All:</b> Shows query relevance."
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="Entity to Content Section Relationship Graph", showlegend=False, hovermode='closest', margin=dict(b=60,l=5,r=5,t=40), annotations=[dict(text=annotation_text, showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.07, xanchor='left', yanchor='bottom', font=dict(size=11))], xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor='white'))
    return fig

def split_text_into_passages(text, passage_length=200):
    if not text: return []
    words = text.split(); return [' '.join(words[i:i + passage_length]).strip() for i in range(0, len(words), passage_length) if ' '.join(words[i:i + passage_length]).strip()]

def find_entity_best_passage(entity_name, passages, model):
    if not passages or not model: return {"passage": "", "similarity": 0.0, "index": -1}
    try:
        entity_embedding = model.encode([entity_name]); passage_embeddings = model.encode(passages)
        similarities = cosine_similarity(entity_embedding, passage_embeddings)[0]; best_idx = np.argmax(similarities)
        return {"passage": passages[best_idx], "similarity": float(similarities[best_idx]), "index": int(best_idx)}
    except Exception as e: st.warning(f"Failed to find best passage for '{entity_name}': {e}"); return {"passage": "", "similarity": 0.0, "index": -1}

def generate_semantic_implementation_analysis(entity_name, primary_content, best_passage, target_query, entity_info):
    """Generate Gemini-powered analysis for implementing missing entities."""
    if not st.session_state.get('gemini_api_configured', False):
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
        model = genai.GenerativeModel("gemini-2.5-pro")
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
        implementation_prompt = f"""
        Based on this content analysis and the target query "{target_query}", provide strategic recommendations for implementing the entity "{entity_name}" into the content.
        CONTENT TONE ANALYSIS:
        {tone_analysis}
        ENTITY TO IMPLEMENT: {entity_name}
        - Type: {entity_info.get('Type', 'Unknown')}
        - Query Relevance: {entity_info.get('Query Relevance', 0):.3f}
        - Found on {entity_info.get('Found On', 0)} competitor sites
        BEST INSERTION LOCATION (Identified by semantic analysis):
        "{best_passage}"
        CURRENT CONTENT SAMPLE:
        "{primary_content[:1500]}"
        Provide a strategic analysis with:
        ## Why Implement This Entity
        - Strategic value for the target query "{target_query}".
        - SEO and content gap benefits (topical authority).
        - How it improves the user's understanding.
        ## Where to Implement
        - Recommend the best way to integrate into the suggested insertion location.
        - Suggest how to blend it with the existing content flow.
        - Propose placement strategy for maximum impact (e.g., as a new paragraph, a list item, or an expansion of an existing sentence).
        ## How to Implement
        - Provide 2-3 specific, actionable content addition examples (e.g., "Add a sentence like '...'").
        - Ensure suggestions maintain the identified tone and style.
        - Offer natural integration techniques (e.g., using transition phrases).
        - Suggest an appropriate word count and depth for the addition.
        Keep recommendations practical, specific, and aligned with the content's existing voice. Focus on seamless integration rather than forced insertion.
        """
        implementation_response = model.generate_content(implementation_prompt)
        return implementation_response.text.strip()
    except Exception as e:
        st.error(f"Gemini analysis failed: {e}")
        return None

def generate_strengthening_analysis(entity_name, primary_content, current_passage, target_query, entity_info):
    """Generate Gemini-powered analysis for strengthening existing entities."""
    if not st.session_state.get('gemini_api_configured', False): return None
    try:
        import google.generativeai as genai; genai.configure(api_key=st.session_state.gemini_api_key_to_persist); model = genai.GenerativeModel("gemini-2.5-pro")
        tone_analysis_prompt = f'Analyze the writing tone, style, and voice of this content sample:\n"{primary_content[:2000]}"\nProvide a brief analysis of: tone, style, and target audience level.'
        tone_response = model.generate_content(tone_analysis_prompt); tone_analysis = tone_response.text.strip()
        strengthening_prompt = f"""
        **Objective:** Strengthen an existing entity in the content to improve its relevance to a target query.
        **Target Query:** "{target_query}"
        **Entity to Strengthen:** "{entity_name}"
        - Current Query Relevance: {entity_info.get('Query Relevance', 0):.3f} (This is low, we need to increase it)
        - Current Document Salience: {entity_info.get('Document Salience', 0):.3f}
        **Current Context (Where the entity appears now):**
        "{current_passage}"
        **Content Tone Analysis:**
        {tone_analysis}
        **Your Task:**
        Provide a strategic analysis with the following sections:
        ## 1. Diagnosis: Why is the connection weak?
        - Briefly analyze why the entity "{entity_name}" in its current context isn't strongly related to the query "{target_query}". Is it too brief? Lacking context? Not framed correctly?
        ## 2. Strategy: How to strengthen the connection?
        - Propose a clear strategy. Should we expand the definition? Add an example? Connect it to another concept?
        ## 3. Actionable Recommendations: What to write?
        - Provide 2-3 specific, actionable rewrite or addition examples.
        - The new content should seamlessly integrate into the current passage.
        - All suggestions must maintain the identified content tone and style.
        - Example format: "Rewrite the sentence '...' to be '...'." or "After the sentence '...', add a new sentence like '...'."
        """
        response = model.generate_content(strengthening_prompt)
        return response.text.strip()
    except Exception as e: st.error(f"Gemini strengthening analysis failed: {e}"); return None

def initialize_selenium_driver():
    options = ChromeOptions(); options.add_argument("--headless"); options.add_argument("--no-sandbox"); options.add_argument("--disable-dev-shm-usage"); options.add_argument("--disable-gpu")
    try:
        driver = webdriver.Chrome(service=ChromeService(), options=options)
        stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32", webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)
        return driver
    except Exception as e: st.error(f"Selenium initialization failed: {e}"); return None

def fetch_content_with_zyte(url, api_key):
    if not api_key: st.error("Zyte API key not configured."); return None
    enforce_rate_limit(); st.write(f"_Fetching with Zyte API: {url}..._")
    try:
        response = requests.post("https://api.zyte.com/v1/extract", auth=(api_key, ''), json={'url': url, 'httpResponseBody': True}, timeout=45)
        response.raise_for_status(); data = response.json()
        if data.get('httpResponseBody'): return base64.b64decode(data['httpResponseBody']).decode('utf-8', 'ignore')
        else: st.error(f"Zyte API did not return content for {url}"); return None
    except requests.exceptions.HTTPError as e: st.error(f"Zyte API HTTP Error for {url}: {e.response.status_code}"); return None
    except Exception as e: st.error(f"Zyte API error for {url}: {e}"); return None

def fetch_content_with_selenium(url, driver_instance):
    if not driver_instance: return fetch_content_with_requests(url)
    try:
        enforce_rate_limit(); driver_instance.get(url); time.sleep(5)
        return driver_instance.page_source
    except Exception as e:
        st.error(f"Selenium fetch error for {url}: {e}"); st.session_state.selenium_driver_instance = None
        st.warning(f"Selenium failed for {url}. Falling back to requests.")
        try: return fetch_content_with_requests(url)
        except Exception as req_e: st.error(f"Requests fallback also failed for {url}: {req_e}"); return None

def fetch_content_with_requests(url):
    enforce_rate_limit(); headers = {'User-Agent': get_random_user_agent()}
    try:
        response = requests.get(url, timeout=20, headers=headers); response.raise_for_status()
        return response.text
    except Exception as e: st.error(f"Requests error for {url}: {e}"); return None

def extract_clean_text(html_content):
    if not html_content: return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    for el in soup(["script", "style", "noscript", "iframe", "link", "meta", 'nav', 'header', 'footer', 'aside', 'form', 'figure', 'figcaption', 'menu', 'banner', 'dialog', 'img', 'svg']):
        if el.name: el.decompose()
    return re.sub(r'\s+', ' ', soup.get_text(separator=' ')).strip()


# --- Main UI ---
LOGO_URL = "https://theseoconsultant.ai/wp-content/uploads/2024/12/cropped-theseoconsultant-logo-2.jpg"
logo_b64 = get_image_as_base64(LOGO_URL)
create_navigation_menu(logo_b64)
st.title("🎯 Entity Gap Analysis Tool")
st.markdown("**Analyze entity gaps between your content and competitors, with query-specific relevance scoring.**")
st.sidebar.subheader("📊 Analysis Configuration")
target_query = st.sidebar.text_input("Target Search Query:", "server-side rendering benefits", help="The search query to measure entity relevance against", disabled=st.session_state.processing)
scraping_method = st.sidebar.selectbox("Scraping Method:", ["Requests (fast)", "Zyte API (best)", "Selenium (for JS)"], index=0, disabled=st.session_state.processing)
st.sidebar.subheader("🔗 URL Configuration")
primary_url = st.sidebar.text_input("Your Primary URL:", "https://cloudinary.com/guides/automatic-image-cropping/server-side-rendering-benefits-use-cases-and-best-practices", disabled=st.session_state.processing)
competitor_urls = st.sidebar.text_area("Competitor URLs (one per line):", "https://prismic.io/blog/what-is-ssr\nhttps://nextjs.org/docs/pages/building-your-application/rendering/server-side-rendering", height=100, disabled=st.session_state.processing)
analysis_disabled = not st.session_state.gcp_nlp_configured or st.session_state.processing
if st.sidebar.button("🚀 Analyze Entity Gaps", disabled=analysis_disabled, type="primary"):
    if not target_query or not primary_url or not competitor_urls: st.error("Please fill in all required fields.")
    else: st.session_state.processing = True; st.rerun()

# --- Main Processing ---
if st.session_state.processing:
    try:
        if not st.session_state.embedding_model:
            with st.spinner("Loading embedding model..."): st.session_state.embedding_model = load_embedding_model()
        if not st.session_state.embedding_model: st.error("Failed to load embedding model."); st.stop()
        competitor_url_list = [url.strip() for url in competitor_urls.split('\n') if url.strip()]; all_urls = [primary_url] + competitor_url_list
        url_content, url_entities = {}, {}
        if scraping_method.startswith("Selenium") and not st.session_state.selenium_driver_instance:
            with st.spinner("Initializing Selenium WebDriver..."): st.session_state.selenium_driver_instance = initialize_selenium_driver()
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
        if not url_content: st.error("No content was successfully fetched. Please check your URLs and try again."); st.stop()
        with st.spinner("Extracting entities from all URLs..."):
            for url, content in url_content.items():
                st.write(f"Extracting entities from: {url}")
                entities = extract_entities_with_google_nlp(content, st.session_state.gcp_credentials_info)
                if entities: url_entities[url] = entities; st.success(f"✅ Found {len(entities)} entities")
                else: st.warning(f"⚠️ No entities extracted from {url}")
        st.session_state.entity_analysis_results = url_entities; st.session_state.content_passages = {}
        if primary_url in url_content: st.session_state.content_passages[primary_url] = split_text_into_passages(url_content[primary_url])
        st.success("✅ Entity analysis complete!")
    finally: st.session_state.processing = False; st.rerun()

# --- Results Display ---
if st.session_state.entity_analysis_results:
    st.markdown("---"); st.subheader("📊 Entity Gap Analysis Results")
    primary_entities = st.session_state.entity_analysis_results.get(primary_url, {})
    if not primary_entities: st.error("No entities found in primary URL.")
    else:
        primary_entity_analysis = [{'Entity': data['name'], 'Type': data['type'], 'Document Salience': data['salience'], 'Query Relevance': calculate_entity_query_relevance(data['name'], target_query, st.session_state.embedding_model), 'Combined Score': (data['salience'] + calculate_entity_query_relevance(data['name'], target_query, st.session_state.embedding_model)) / 2} for key, data in primary_entities.items()]
        primary_df = pd.DataFrame(primary_entity_analysis).sort_values('Combined Score', ascending=False)
        st.subheader("📋 Your Content Entities vs Target Query")
        st.dataframe(primary_df, use_container_width=True, column_config={"Document Salience": st.column_config.ProgressColumn("Salience",format="%.3f",min_value=0,max_value=1),"Query Relevance": st.column_config.ProgressColumn("Query Relevance",format="%.3f",min_value=0,max_value=1),"Combined Score": st.column_config.ProgressColumn("Combined Score",format="%.3f",min_value=0,max_value=1)})

        st.markdown("---"); st.subheader("💡 Content Optimization: Strengthen Your Existing Entities")
        st.markdown("_Identify entities on your page that are weakly connected to the target query and get AI-powered suggestions to improve them._")
        weak_entities = sorted([e for e in primary_entity_analysis if e['Query Relevance'] < 0.4 and e['Combined Score'] > 0.1], key=lambda x: x['Query Relevance'])
        if weak_entities and st.session_state.content_passages.get(primary_url):
            entity_options = {f"{e_info['Entity']} (Query Relevance: {e_info['Query Relevance']:.3f})": e_info for e_info in weak_entities}
            selected_weak_entity_display = st.selectbox("🔍 Select a weak entity to strengthen:", options=[""] + list(entity_options.keys()), index=0, help="Entities from your content with the lowest query relevance are listed first.")
            if selected_weak_entity_display:
                entity_info = entity_options[selected_weak_entity_display]; entity_name = entity_info['Entity']
                passages = st.session_state.content_passages[primary_url]; best_passage_info = find_entity_best_passage(entity_name, passages, st.session_state.embedding_model)
                col_a, col_b = st.columns([1, 2])
                with col_a: st.markdown("**📊 Entity Metrics:**"); st.metric("Query Relevance", f"{entity_info['Query Relevance']:.3f}", delta_color="inverse"); st.metric("Document Salience", f"{entity_info['Document Salience']:.3f}")
                with col_b: st.markdown("**🎯 Current Context:**"); st.text_area("The entity currently appears in this passage:", value=best_passage_info["passage"], height=120, disabled=True, key=f"passage_strengthen_{entity_name.replace(' ', '_')}")
                if st.session_state.get('gemini_api_configured', False):
                    if st.button(f"🤖 Get AI Strategy to Strengthen '{entity_name}'", key=f"gemini_strengthen_{entity_name.replace(' ', '_')}"):
                        with st.spinner(f"Generating optimization strategy for '{entity_name}'..."):
                            primary_content_sample = " ".join(passages[:5]); strengthening_analysis = generate_strengthening_analysis(entity_name, primary_content_sample, best_passage_info["passage"], target_query, entity_info)
                            if strengthening_analysis: st.markdown("---"); st.markdown("### 🤖 AI Optimization Strategy"); st.markdown(strengthening_analysis)
                            else: st.error("Failed to generate optimization strategy.")
                else: st.info("💡 **Enable Gemini API** in the sidebar to get AI-powered optimization strategies.")
        else: st.success("✅ All your existing entities seem well-connected to the target query!")

        competitor_entities = {}; primary_entity_keys = set(primary_entities.keys()); primary_entity_names = {data['name'].lower() for data in primary_entities.values()}
        for url, entities in st.session_state.entity_analysis_results.items():
            if url != primary_url:
                for key, data in entities.items():
                    if key not in competitor_entities: competitor_entities[key] = {'data': data, 'found_on': []}
                    competitor_entities[key]['found_on'].append(url)
        missing_entities = []
        for key, comp_data in competitor_entities.items():
            name, name_lower = comp_data['data']['name'], comp_data['data']['name'].lower()
            is_missing = (key not in primary_entity_keys and name_lower not in primary_entity_names and not any(name_lower in p_name or p_name in name_lower for p_name in primary_entity_names))
            if is_missing:
                q_sim = calculate_entity_query_relevance(name, target_query, st.session_state.embedding_model); c_score = (comp_data['data']['salience'] + q_sim) / 2
                if c_score > 0.15: missing_entities.append({'Entity': name, 'Type': comp_data['data']['type'], 'Document Salience': comp_data['data']['salience'], 'Query Relevance': q_sim, 'Combined Score': c_score, 'Found On': len(comp_data['found_on']), 'URLs': ', '.join([f"`{url.split('//')[-1].split('/')[0]}`" for url in comp_data['found_on'][:2]])})
        if missing_entities:
            st.markdown("---"); st.subheader("❗ Missing Entities (Found in Competitors)")
            gap_df = pd.DataFrame(missing_entities).sort_values('Combined Score', ascending=False)
            st.dataframe(gap_df, use_container_width=True, column_config={"Document Salience": st.column_config.ProgressColumn("Salience",format="%.3f",min_value=0,max_value=1),"Query Relevance": st.column_config.ProgressColumn("Query Relevance",format="%.3f",min_value=0,max_value=1),"Combined Score": st.column_config.ProgressColumn("Combined Score",format="%.3f",min_value=0,max_value=1)})
        else: st.markdown("---"); st.success("✅ No entity gaps found! Your content covers all entities found in competitors.")
        
        if primary_entity_analysis and missing_entities:
            st.markdown("---"); st.subheader("🕸️ Content Semantic Structure Graph")
            st.markdown("""_This graph visualizes your content's semantic architecture..._""")
            col1, col2, col3 = st.columns(3)
            with col1: max_primary = st.slider("Max Primary Entities", 5, 25, st.session_state.graph_config['max_primary'], 5, help="Show top N entities")
            with col2: max_missing = st.slider("Max Missing Entities", 5, 20, st.session_state.graph_config['max_missing'], 5, help="Show top N missing entities")
            with col3: similarity_threshold = st.slider("Connection Threshold", 0.3, 0.7, st.session_state.graph_config['threshold'], 0.05, help="Minimum similarity")
            st.markdown("**🏷️ Filter by Entity Types:**"); entity_types = sorted(list(set(e.get('Type', 'UNKNOWN') for e in primary_entity_analysis + missing_entities)))
            selected_types = st.multiselect("Show entity types:", options=entity_types, default=entity_types, help="Select entity types for the graph")
            min_score = st.slider("Minimum Combined Score", 0.0, 0.8, st.session_state.graph_config['min_score'], 0.05, help="Only show entities above this score")
            st.session_state.graph_config.update({'max_primary': max_primary, 'max_missing': max_missing, 'min_score': min_score, 'threshold': similarity_threshold})
            missing_entity_names_for_graph = [e['Entity'] for e in sorted(missing_entities, key=lambda x: x['Combined Score'], reverse=True)]
            col1, col2 = st.columns([2, 1])
            with col1: selected_entity_for_graph = st.selectbox("🎯 Select missing entity to highlight in graph:", ["None"] + missing_entity_names_for_graph, 0, help="Highlight a missing entity on the graph.")
            with col2:
                 if st.button("🔄 Refresh Graph Layout"): st.rerun()
            filtered_primary = [e for e in primary_entity_analysis if e.get('Type', 'UNKNOWN') in selected_types and e.get('Combined Score', 0) >= min_score][:max_primary]
            filtered_missing = [e for e in missing_entities if e.get('Type', 'UNKNOWN') in selected_types and e.get('Combined Score', 0) >= min_score][:max_missing]
            primary_url_passages = st.session_state.content_passages.get(primary_url, [])
            if (filtered_primary or filtered_missing) and primary_url_passages:
                with st.spinner("Calculating entity-to-content relationships..."): st.session_state.entity_relationships = calculate_entity_relationships(filtered_primary, filtered_missing, st.session_state.embedding_model, target_query, primary_url_passages, similarity_threshold=similarity_threshold)
                selected_entity = selected_entity_for_graph if selected_entity_for_graph != "None" else None
                entity_graph = create_entity_relationship_graph(st.session_state.entity_relationships, selected_missing_entity=selected_entity)
                if entity_graph: st.plotly_chart(entity_graph, use_container_width=True, config={'displayModeBar': True}, height=700)
                else: st.warning("Could not generate graph. Not enough connections found.")
            else: st.warning("No entities match filters, or no content was found for the primary URL.")

        # --- WHERE TO ADD MISSING ENTITIES SECTION (CORRECTED) ---
        if missing_entities and st.session_state.content_passages.get(primary_url):
            st.markdown("---")
            st.subheader("📍 Where to Add Missing Entities")
            st.markdown("_Select from missing entities to find optimal insertion locations._")
            
            passages = st.session_state.content_passages[primary_url]
            sorted_missing = sorted(missing_entities, key=lambda x: x['Query Relevance'], reverse=True)
            
            # FIX IS HERE: Use a single, correctly named dictionary.
            entity_lookup = {}
            for entity_info in sorted_missing:
                display_name = f"{entity_info['Entity']} (Query Relevance: {entity_info['Query Relevance']:.3f})"
                entity_lookup[display_name] = entity_info
            
            selected_entity_display = st.selectbox(
                "🔍 Select missing entity to analyze:",
                options=[""] + list(entity_lookup.keys()), # Use the keys from the lookup dict
                index=0,
                help=f"All {len(sorted_missing)} missing entities sorted by Query Relevance."
            )

            if selected_entity_display:
                # This will now work because we are looking up in the populated dictionary.
                entity_info = entity_lookup[selected_entity_display]
                entity_name = entity_info['Entity']
                
                best_passage_info = find_entity_best_passage(entity_name, passages, st.session_state.embedding_model)
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown("**📊 Entity Metrics:**")
                    st.metric("Query Relevance", f"{entity_info['Query Relevance']:.3f}")
                    st.metric("Combined Score", f"{entity_info['Combined Score']:.3f}")
                    st.metric("Document Salience", f"{entity_info['Document Salience']:.3f}")
                    st.markdown("**🏢 Details:**")
                    st.markdown(f"- **Type:** {entity_info['Type']}\n- **Found on:** {entity_info['Found On']} competitor site(s)")
                with col_b:
                    if best_passage_info["similarity"] > 0.1:
                        st.markdown("**🎯 Recommended Insertion Location:**")
                        st.markdown(f"**Content Relevance:** {best_passage_info['similarity']:.3f}")
                        st.text_area("Best passage for adding this entity:", value=best_passage_info["passage"], height=120, disabled=True, key=f"passage_add_{entity_name.replace(' ', '_')}")
                        if best_passage_info["similarity"] > 0.5: st.success("✅ **Excellent insertion point**")
                        elif best_passage_info["similarity"] > 0.3: st.info("ℹ️ **Good insertion point**")
                        else: st.warning("⚠️ **Lower relevance** - consider if a new section is needed.")
                        if st.session_state.get('gemini_api_configured', False):
                            if st.button(f"🤖 Get AI Implementation Strategy for '{entity_name}'", key=f"gemini_add_{entity_name.replace(' ', '_')}"):
                                with st.spinner("Generating implementation strategy..."):
                                    primary_content = " ".join(passages[:5])
                                    semantic_analysis = generate_semantic_implementation_analysis(entity_name, primary_content, best_passage_info["passage"], target_query, entity_info)
                                    if semantic_analysis: st.markdown("---"); st.markdown("### 🤖 AI Implementation Strategy"); st.markdown(semantic_analysis)
                                    else: st.error("Failed to generate implementation strategy.")
                        else: st.info("💡 **Enable Gemini API** to get AI-powered implementation strategies.")
                    else:
                        st.markdown("**🤔 No strongly relevant passage found.**")
                        st.info(f"Consider adding a new section about '{entity_name}'.")
                        if st.session_state.get('gemini_api_configured', False):
                            if st.button(f"🤖 Get AI Strategy for New '{entity_name}' Section", key=f"gemini_new_{entity_name.replace(' ', '_')}"):
                                with st.spinner("Generating new section strategy..."):
                                    primary_content = " ".join(passages[:5])
                                    semantic_analysis = generate_semantic_implementation_analysis(entity_name, primary_content, "No existing relevant passage found - a new section is required", target_query, entity_info)
                                    if semantic_analysis: st.markdown("---"); st.markdown("### 🤖 AI Strategy for New Section"); st.markdown(semantic_analysis)
                                    else: st.error("Failed to generate new section strategy.")
# Footer
st.sidebar.divider()
st.sidebar.info("🎯 Entity Gap Analysis Tool v2.6")
st.sidebar.markdown("---")
st.sidebar.markdown("**Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)**")
