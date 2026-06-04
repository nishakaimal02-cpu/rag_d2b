# AI Research Assistant — LangChain + ChromaDB + BM25 + UMAP

Day 2b of my 20-day AI learning journey. A semantic search app built with LangChain and ChromaDB, with three search modes and an embedding visualisation.

## What it does
Ask questions over AI research papers and get answers with page-level citations, confidence scores, and source document references. Switch between three search modes to compare retrieval strategies.

## Search modes
- **Semantic search** — meaning-based, uses OpenAI embeddings. Best for conceptual questions and synonyms.
- **BM25 keyword search** — exact word matching, no API call needed. Best for specific terminology and section names.
- **Hybrid** — combines both. Best for most real-world queries.

## Key features
- Page number citations — every answer shows exactly which page it came from
- Confidence scoring — warns when similarity score is low
- Intent classification — greetings skip the database entirely
- UMAP visualisation — see all document chunks plotted in 2D space, coloured by source document
- Password protection — prevents unauthorised API usage

## How it works
1. `ingest.py` — loads TXT and PDF files, chunks with RecursiveCharacterTextSplitter, embeds via OpenAI, stores in ChromaDB
2. `retriever.py` — three search functions (semantic, BM25, hybrid), confidence check, GPT-4o-mini answer generation
3. `app.py` — Streamlit UI with search mode toggle, chat interface, UMAP tab
4. `umap_viz.py` — pulls embeddings from ChromaDB, reduces to 2D with UMAP, plots with Plotly

## BM25 vs Semantic — key findings
From 5 test queries across 3 AI research papers:
- BM25 wins for exact domain terminology — finds right pages faster
- Semantic wins for conceptual questions — understands meaning across documents
- Neither wins when documents don't contain plain English explanations
- Hybrid needs Reciprocal Rank Fusion for proper combination

## Stack
- Python, LangChain, ChromaDB, OpenAI API, Streamlit
- BM25 via rank-bm25, visualisation via UMAP + Plotly
- GPT-4o-mini for generation, text-embedding-3-small for embeddings

## Running locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Add OPENAI_API_KEY and APP_PASSWORD to .streamlit/secrets.toml
python3 ingest.py  # Build index first
streamlit run app.py
```

## Note on deployment
Click 'Build / Rebuild Index' after opening — index rebuilds each session due to Streamlit Cloud's ephemeral filesystem. Pinecone integration coming in Day 4 for persistent cloud storage.

