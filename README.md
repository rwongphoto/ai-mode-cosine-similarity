# AI Mode Cosine Similarity

Streamlit app that scores how well a URL's content covers Google AI Mode query fan-outs using cosine similarity between embeddings.

## What it does

Given a target URL and a seed query, the tool:

1. Fetches the URL (Selenium with stealth + user-agent rotation) and chunks the content into passages.
2. Generates fan-out sub-queries from the seed query via Gemini and/or OpenAI.
3. Embeds both passages and sub-queries with Sentence Transformers (default `all-mpnet-base-v2`).
4. Computes cosine similarity between every passage and every sub-query.
5. Renders interactive Plotly visualizations — coverage matrix, per-query max similarity, weakest-covered sub-queries.

Also includes a **Prompt Ranking** mode that scores how likely an LLM would be to cite a given URL for a set of prompts.

## Stack

- Streamlit UI
- `sentence-transformers` for embeddings
- Gemini (`google-genai`) + OpenAI for fan-out generation
- Selenium + `selenium-stealth` for content fetching
- Trafilatura for text extraction
- Hugging Face Hub auth for gated embedding models
- Plotly for visualizations

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configure API keys in the sidebar: Gemini, OpenAI, Hugging Face, Zyte (optional for JS-heavy pages). Or set env vars `HF_TOKEN`, `GEMINI_API_KEY`, `OPENAI_API_KEY`.

## Files

- `app.py` — full Streamlit application
- `requirements.txt` — Python dependencies
