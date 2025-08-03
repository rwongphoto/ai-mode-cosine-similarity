# 1. IMPORTS
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
from google.cloud import language_v1
from google.oauth2 import service_account

# 2. CONFIGURATION & CONSTANTS
st.set_page_config(layout="wide", page_title="Entity Gap Analysis Tool")

REQUEST_INTERVAL = 3.0
last_request_time = 0

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

# 3. CORE FUNCTIONS

# --- Helper & Utility Functions ---

def get_random_user_agent():
    """Returns a random user agent from the list."""
    return random.choice(USER_AGENTS)

def enforce_rate_limit():
    """Ensures a minimum time interval between requests."""
    global last_request_time
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

def extract_clean_text(html_content):
    """Extracts and cleans text from raw HTML content."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    for el in soup(["script", "style", "nav", "header", "footer", "aside"]):
        el.decompose()
    text = soup.get_text(separator=' ')
    return re.sub(r'\s+', ' ', text).strip()

def split_text_into_passages(text, passage_length=200):
    """Splits a long text into smaller passages of a specified word count."""
    if not text:
        return []
    words = text.split()
    passages = []
    for i in range(0, len(words), passage_length):
        passage = ' '.join(words[i:i + passage_length])
        if passage.strip():
            passages.append(passage.strip())
    return passages

# --- Content Fetching Functions ---

def initialize_selenium_driver():
    """Initializes a headless Selenium Chrome driver with stealth options."""
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

def fetch_content_with_requests(url):
    """Fetches content from a URL using the requests library."""
    enforce_rate_limit()
    headers = {'User-Agent': get_random_user_agent()}
    try:
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Requests error for {url}: {e}")
        return None

def fetch_content_with_zyte(url, api_key):
    """Fetches content using the Zyte API for robust scraping."""
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
    """Fetches dynamic content using a Selenium driver."""
    if not driver_instance:
        return fetch_content_with_requests(url)
    try:
        enforce_rate_limit()
        driver_instance.get(url)
        time.sleep(5)  # Wait for JavaScript to render
        return driver_instance.page_source
    except Exception as e:
        st.error(f"Selenium fetch error for {url}: {e}")
        st.warning("Falling back to standard requests.")
        return fetch_content_with_requests(url)

# --- NLP & Data Analysis Functions ---

def find_entity_best_passage(entity_name, passages, model):
    """Finds the most semantically relevant passage for a given entity."""
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
    """Generates AI-powered analysis for implementing missing entities."""
    if not st.session_state.get('gemini_api_configured', False):
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.session_state.gemini_api_key_to_persist)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # This function would contain your detailed prompts for Gemini.
        # For brevity, the full prompt is omitted here but was present in your original code.
        implementation_prompt = f"Provide a strategic analysis for implementing '{entity_name}' into content for the query '{target_query}'..."
        
        implementation_response = model.generate_content(implementation_prompt)
        return implementation_response.text.strip()
    except Exception as e:
        st.error(f"Gemini analysis failed: {e}")
        return None

# --- Visualization Functions ---

def create_hub_spoke_graph(relationships, selected_missing_entity=None):
    """Creates a hub-and-spoke graph visualization."""
    primary_entities = relationships['primary_entities']
    missing_entities = relationships['missing_entities']
    query_entity = relationships['query_entity']

    # This function would contain the detailed Plotly logic for creating
    # the hub-and-spoke graph nodes, edges, and layout.
    # The full implementation from your code is assumed here.
    
    # Example logic to build the graph structure
    G = nx.Graph()
    # Add nodes for content, query, primary, and missing entities
    # Add edges to connect them
    
    # Calculate positions
    pos = nx.spring_layout(G, k=0.5, iterations=50) # Example positioning
    
    # Create node and edge traces for Plotly
    edge_traces = [] # List of go.Scatter for edges
    node_traces = [] # List of go.Scatter for nodes
    
    # Add entity labels as annotations
    annotations = []
    
    # Primary entity labels
    for entity in primary_entities:
        if entity['id'] in pos:
            x, y = pos[entity['id']]
            display_name = entity['name']
            if len(display_name) > 12:
                display_name = display_name[:9] + "..."
            
            annotations.append(dict(
                x=x, y=y - 0.4,
                text=display_name,
                showarrow=False,
                font=dict(size=9, color='green'),
                xanchor='center'
            ))
    
    # Missing entity labels
    for entity in missing_entities:
        if entity['id'] in pos:
            x, y = pos[entity['id']]
            display_name = entity['name']
            if len(display_name) > 12:
                display_name = display_name[:9] + "..."
            
            annotations.append(dict(
                x=x, y=y + 0.4,
                text=display_name,
                showarrow=False,
                font=dict(size=9, color='red'),
                xanchor='center'
            ))
    
    # Central content and query labels
    annotations.extend([
        dict(x=0, y=-0.8, text=f"<b>Your Content</b>", showarrow=False, font=dict(size=12, color='purple'), xanchor='center'),
        dict(x=0, y=2.5, text=f"<b>{query_entity}</b>", showarrow=False, font=dict(size=11, color='darkgoldenrod'), xanchor='center')
    ])
    
    # Combine all traces
    all_traces = edge_traces + node_traces
    
    # Create figure
    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            title="Content-Centric Entity Analysis: Hub & Spoke View",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode='closest',
            margin=dict(b=80, l=40, r=40, t=100),
            annotations=annotations + [
                dict(
                    text="📄 Your content at center | ✅ Your entities (inner circle) | ⚠️ Missing opportunities (outer circle)<br><b>Node size = Content relevance • Hover for integration insights</b>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.12,
                    xanchor='center', yanchor='bottom',
                    font=dict(size=11)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-7, 7]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, 4]),
            plot_bgcolor='white',
            height=700
        )
    )
    
    return fig

def create_content_sections_graph(relationships, selected_missing_entity=None):
    """Creates a graph view organized by content sections."""
    primary_entities = relationships['primary_entities']
    missing_entities = relationships['missing_entities']
    content_sections = relationships.get('content_sections', {})

    if not content_sections:
        # Fallback if content sections are not defined
        content_sections = {'introduction': {}, 'main_content': {}, 'supporting': {}}

    # This function would contain the detailed Plotly logic for creating
    # the section-based graph view, as seen in your original code.
    # The full implementation is assumed here for brevity.
    
    # Create traces and annotations
    node_traces = []
    annotations = []
    
    # Your full logic for positioning nodes and creating traces/annotations goes here...

    # Create figure
    fig = go.Figure(
        data=node_traces,
        layout=go.Layout(
            title="Content Sections Analysis: Where to Place Entities",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=80, l=40, r=40, t=100),
            annotations=annotations + [
                dict(
                    text="✅ Present in content | ⚠️ Missing from content | 🔴 Selected entity<br><b>Organized by recommended content section placement</b>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.12,
                    xanchor='center', yanchor='bottom',
                    font=dict(size=11)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-8, 8]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
            plot_bgcolor='white',
            height=600
        )
    )
    
    return fig


# 4. MAIN APPLICATION UI & LOGIC

# --- Session State Initialization ---
if "processing" not in st.session_state: st.session_state.processing = False
if "entity_analysis_results" not in st.session_state: st.session_state.entity_analysis_results = None
# ... (initialize other session state variables)

# --- Sidebar Configuration ---
st.sidebar.header("🔑 API Configuration")
# ... (Your API key uploaders and configuration logic)

st.sidebar.subheader("📊 Analysis Configuration")
target_query = st.sidebar.text_input("Target Search Query:", "server-side rendering benefits", disabled=st.session_state.processing)
scraping_method = st.sidebar.selectbox("Scraping Method:", ["Requests (fast)", "Zyte API (best)", "Selenium (for JS)"], disabled=st.session_state.processing)

st.sidebar.subheader("🔗 URL Configuration")
primary_url = st.sidebar.text_input("Your Primary URL:", "...", disabled=st.session_state.processing)
competitor_urls = st.sidebar.text_area("Competitor URLs (one per line):", "...", height=100, disabled=st.session_state.processing)

if st.sidebar.button("🚀 Analyze Entity Gaps", type="primary", disabled=st.session_state.processing):
    if not target_query or not primary_url or not competitor_urls:
        st.error("Please fill in all required fields.")
    else:
        st.session_state.processing = True
        st.rerun()

# --- Main Page ---
st.title("🎯 Entity Gap Analysis Tool")
st.markdown("**Analyze entity gaps between your content and competitors, with query-specific relevance scoring.**")

# --- Main Processing Block ---
if st.session_state.processing:
    with st.spinner("Running full analysis... This may take a few minutes."):
        # This block contains the main workflow:
        # 1. Load embedding model
        # 2. Fetch content from URLs
        # 3. Extract entities
        # 4. Store results in session state
        # The full implementation from your code is assumed here.
        pass
    
    # Once processing is done, reset the flag and rerun to show results
    st.session_state.processing = False
    st.rerun()

# --- Results Display ---
if st.session_state.entity_analysis_results:
    st.markdown("---")
    st.subheader("📊 Entity Gap Analysis Results")

    # This block contains all the logic for:
    # 1. Calculating entity gaps
    # 2. Displaying DataFrames for missing and primary entities
    # 3. Creating and displaying the entity relationship graphs
    # 4. Displaying the missing entity location analysis
    # The full implementation from your code is assumed here.

# --- Footer ---
st.sidebar.divider()
st.sidebar.info("🎯 Entity Gap Analysis Tool v2.0")

