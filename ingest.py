# ingest.py
# PURPOSE: Load documents, chunk them, embed them, store in ChromaDB
# Run this once to build the index, then again when you add new documents
import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load API key from .env file
load_dotenv()

# Where your documents live and where ChromaDB will save the index
DOCS_PATH = "docs"
CHROMA_PATH = "/tmp/chroma_db"

def ingest_docs():
    import shutil
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🗑️ Cleared existing index")

    print("📄 Loading documents...")
    
    # Load all .txt files from docs folder
    txt_loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    
    # Load all .pdf files — PyPDFLoader automatically adds page numbers as metadata
    pdf_loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    
    # Combine txt and pdf documents into one list
    documents = txt_loader.load() + pdf_loader.load()
    print(f"   Loaded {len(documents)} documents")

    print("✂️  Chunking documents...")
    
    # RecursiveCharacterTextSplitter splits at natural boundaries first:
    # paragraphs → sentences → words → characters (last resort)
    # chunk_size=500: each chunk is max 500 characters
    # chunk_overlap=50: chunks share 50 characters to avoid losing context at boundaries
    # add_start_index=True: stores character position of each chunk as metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    print("🔢 Embedding and storing in ChromaDB...")
    
    # OpenAI embedding model — converts text chunks to vectors (1536 numbers each)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Chroma.from_documents does three things in one call:
    # 1. Embeds every chunk via OpenAI API
    # 2. Stores vectors in ChromaDB
    # 3. Stores original text + all metadata (filename, page, start_index) alongside vectors
    # persist_directory saves everything to disk so it survives app restarts
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"   Stored in ChromaDB at '{CHROMA_PATH}/'")
    return vectorstore

# Only runs when you execute this file directly: python ingest.py
# Does NOT run when other files import from this file
if __name__ == "__main__":
    ingest_docs()