"""
Store Index Script
==================
Run this script ONCE to:
1. Load the Medical PDF from the data/ directory
2. Split it into text chunks
3. Generate embeddings using HuggingFace sentence-transformers
4. Upload all embeddings to a Pinecone vector index

Usage:
    python store_index.py
"""

from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Please add it to your .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "medical-chatbot"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dim vectors


def main():
    print("=" * 60)
    print("  Medical Chatbot - Pinecone Index Builder")
    print("=" * 60)

    # Step 1: Load PDF documents
    print("\n[1/4] Loading PDF documents from data/ directory...")
    extracted_data = load_pdf_file(data_path="data/")
    print(f"       [OK] Loaded {len(extracted_data)} pages")

    # Step 2: Split into chunks
    print("\n[2/4] Splitting documents into text chunks...")
    text_chunks = text_split(extracted_data)
    print(f"       [OK] Created {len(text_chunks)} text chunks")

    # Step 3: Initialize embeddings
    print("\n[3/4] Downloading HuggingFace embedding model...")
    embeddings = download_huggingface_embeddings()
    print("       [OK] Embedding model ready (all-MiniLM-L6-v2)")

    # Step 4: Create Pinecone index and upload
    print("\n[4/4] Setting up Pinecone index and uploading embeddings...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"       Creating new index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        print("       Waiting for index to be ready...")
        while True:
            desc = pc.describe_index(INDEX_NAME)
            is_ready = desc.status.get("ready", False) if isinstance(desc.status, dict) else getattr(desc.status, "ready", False)
            if is_ready:
                break
            time.sleep(2)
        print("       Index is ready!")
    else:
        print(f"       Index '{INDEX_NAME}' already exists, using existing index.")

    # Upload embeddings to Pinecone in batches to avoid network timeouts / payload limits
    print("       Uploading embeddings to Pinecone in batches...")
    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i : i + batch_size]
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=INDEX_NAME,
        )
        print(f"       [Progress] Uploaded {min(i + batch_size, len(text_chunks))}/{len(text_chunks)} chunks")

    print("\n" + "=" * 60)
    print("  [OK] SUCCESS! All embeddings uploaded to Pinecone.")
    print(f"  Index Name: {INDEX_NAME}")
    print(f"  Total Chunks: {len(text_chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
