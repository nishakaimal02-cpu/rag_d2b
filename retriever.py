# retriever.py
# PURPOSE: Three search modes — semantic, BM25 keyword, hybrid
# Also handles intent classification and answer generation

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import os

# Load API key from .env file
load_dotenv()

# Must match the path used in ingest.py
CHROMA_PATH = "/tmp/chroma_db"


def load_vectorstore():
    import chromadb
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    IS_CLOUD = not os.path.exists("/Users")
    
    if IS_CLOUD:
        chroma_client = chromadb.EphemeralClient()
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="langchain",
            embedding_function=embeddings
        )
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
    return vectorstore

def get_semantic_results(query, vectorstore, k=3):
    """
    Semantic search — meaning based, uses embeddings.
    Converts query to vector, finds closest chunk vectors in ChromaDB.
    Score = distance (lower = more similar).
    Best for: conceptual questions, synonyms, paraphrasing.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "score": round(float(score), 4),
            "method": "semantic"
        }
        for doc, score in results
    ]

def get_bm25_results(query, vectorstore, k=3):
    """
    BM25 keyword search — exact word matching, no embeddings, no API call.
    Scores based on term frequency + inverse document frequency + length normalisation.
    Score = relevance (higher = more relevant). Opposite direction to semantic scores.
    Best for: exact terms, product codes, legal clauses, proper nouns.
    """
    # Pull all documents and metadata out of ChromaDB
    all_docs = vectorstore.get()
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    
    # Tokenize — split each document into lowercase words
    # BM25 works on word lists, not raw text
    tokenized = [doc.lower().split() for doc in documents]
    
    # Build BM25 index from all tokenized documents
    bm25 = BM25Okapi(tokenized)
    
    # Score every document against the query words
    scores = bm25.get_scores(query.lower().split())
    
    # Sort by score descending, take top k
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]
    
    return [
        {
            "text": documents[idx],
            "source": metadatas[idx].get("source", "unknown"),
            "page": metadatas[idx].get("page", "N/A"),
            "score": round(float(scores[idx]), 4),
            "method": "bm25"
        }
        for idx in top_indices
    ]

def get_hybrid_results(query, vectorstore, k=3):
    """
    Hybrid search — combines semantic and BM25 results.
    Simple approach: semantic results first, BM25 fills remaining slots.
    Deduplicates so same chunk never appears twice.
    Production systems use Reciprocal Rank Fusion for smarter combining.
    Best for: most real-world queries where you want both coverage and precision.
    """
    semantic = get_semantic_results(query, vectorstore, k=k)
    bm25 = get_bm25_results(query, vectorstore, k=k)
    
    # Deduplicate — if same chunk appears in both, keep it once
    seen = set()
    combined = []
    for result in semantic + bm25:
        if result["text"] not in seen:
            seen.add(result["text"])
            combined.append(result)
    
    return combined[:k]

def build_answer(query, chunks):
    """
    Takes retrieved chunks and generates an answer using GPT-4o-mini.
    Prompt explicitly grounds the model in the provided context only.
    Includes source filename and page number in context for citations.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Build context string with source citations
    # Now includes page numbers — "From paper.pdf (page 4):"
    context = "\n\n".join([
        f"From {c['source']} (page {c['page']}):\n{c['text']}"
        for c in chunks
    ])
    
    # Prompt grounds GPT in context only — prevents hallucination
    prompt = f"""You are a helpful research assistant.
Answer the question using only the context provided below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}
Answer:"""

    response = llm.invoke(prompt)
    return response.content

def classify_intent(query):
    """
    Classifies user input as 'research' or 'casual' before hitting the database.
    Prevents greetings and small talk from triggering unnecessary API calls.
    Returns single word: 'research' or 'casual'.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(
        f"Classify this as 'research' or 'casual'. Reply with one word only: {query}"
    )
    return response.content.strip().lower()

def answer_question(query, vectorstore, search_mode="hybrid", k=3):
    """
    Main function called by app.py.
    1. Classify intent — skip database if casual
    2. Retrieve chunks using selected search mode
    3. Check confidence — warn if similarity score is too high (too distant)
    4. Generate answer from retrieved chunks
    Returns: answer text, sources list, confidence warning if applicable
    """
    # Step 1: Intent classification
    intent = classify_intent(query)
    if intent == "casual":
        return {
            "answer": "Hi! I'm your AI research assistant. Ask me anything about the research papers in my knowledge base.",
            "sources": [],
            "confidence": "high",
            "chunks": []
        }
    
    # Step 2: Retrieve chunks based on selected search mode
    if search_mode == "semantic":
        chunks = get_semantic_results(query, vectorstore, k=k)
    elif search_mode == "bm25":
        chunks = get_bm25_results(query, vectorstore, k=k)
    else:
        chunks = get_hybrid_results(query, vectorstore, k=k)
    
    # Step 3: Confidence check
    # For semantic search, score = distance (lower = better)
    # If best score > 1.5, the match is weak — warn the user
    confidence = "high"
    if search_mode in ["semantic", "hybrid"] and chunks:
        best_score = chunks[0].get("score", 0)
        if best_score > 1.5:
            confidence = "low"
    
    # Step 4: Generate answer
    answer = build_answer(query, chunks)
    
    return {
        "answer": answer,
        "sources": [
            f"{c['source']} (page {c['page']}) — score: {c['score']} via {c['method']}"
            for c in chunks
        ],
        "confidence": confidence,
        "chunks": chunks
    }