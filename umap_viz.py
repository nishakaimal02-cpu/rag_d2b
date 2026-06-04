# umap_viz.py
# PURPOSE: Pull all embeddings from ChromaDB, reduce to 2D using UMAP,
# and create an interactive plotly scatter plot showing how chunks cluster by document

import os
import numpy as np
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
import chromadb

load_dotenv()

CHROMA_PATH = "chroma_db"

def get_embeddings_from_chroma():
    """Pull all vectors, documents and metadata from ChromaDB."""
    print("Loading embeddings from ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("langchain")
    
    # Get everything — vectors, texts, metadata
    results = collection.get(include=["embeddings", "documents", "metadatas"])
    
    embeddings = np.array(results["embeddings"])
    documents = results["documents"]
    metadatas = results["metadatas"]
    
    print(f"   Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings, documents, metadatas

def reduce_with_umap(embeddings, n_neighbors=15, min_dist=0.1):
    """Reduce 1536-dimensional vectors to 2D using UMAP."""
    print("Running UMAP dimensionality reduction...")
    import umap
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"   Reduced to shape {reduced.shape}")
    return reduced

def get_short_source(source):
    """Extract just the filename from full path."""
    return os.path.basename(source).replace(".pdf", "").replace(".txt", "")

def build_dataframe(reduced, documents, metadatas):
    """Build a pandas dataframe for plotting."""
    sources = [get_short_source(m.get("source", "unknown")) for m in metadatas]
    pages = [m.get("page", 0) + 1 for m in metadatas]
    
    # Truncate document text for hover display
    hover_texts = [doc[:200] + "..." if len(doc) > 200 else doc for doc in documents]
    
    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "source": sources,
        "page": pages,
        "text": hover_texts
    })
    return df

def plot_embeddings(df):
    """Create interactive plotly scatter plot."""
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        hover_data={"x": False, "y": False, "page": True, "text": True},
        title="Document embeddings visualised in 2D — UMAP projection",
        labels={"source": "Document", "x": "UMAP 1", "y": "UMAP 2"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=900,
        height=600
    )
    
    return fig

def run_umap_viz():
    """Main function — pull embeddings, reduce, plot."""
    if not os.path.exists(CHROMA_PATH):
        print("No ChromaDB index found. Run ingest.py first.")
        return
    
    embeddings, documents, metadatas = get_embeddings_from_chroma()
    reduced = reduce_with_umap(embeddings)
    df = build_dataframe(reduced, documents, metadatas)
    fig = plot_embeddings(df)
    
    # Save as HTML file you can open in browser
    output_path = "umap_embeddings.html"
    fig.write_html(output_path)
    print(f"✅ Saved to {output_path} — open in your browser!")
    
    # Also show it directly
    fig.show()

if __name__ == "__main__":
    run_umap_viz()