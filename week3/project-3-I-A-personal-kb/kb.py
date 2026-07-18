"""
kb.py — Project 3-I-A: Personal Knowledge Base

Architecture:
  Document Loader (multi-format: .pdf, .txt, .md)
    -> Text Splitter (char-based chunking)
    -> TF-IDF Embedder (custom, avoids broken torch/onnxruntime on this machine)
    -> ChromaDB (persistent, on disk)
    -> Retrieval + Groq generation (streamed)
    -> CLI

Usage:
    python kb.py ingest              # (re)ingest everything in documents/
    python kb.py ask "question"                     # ask across all sources
    python kb.py ask "question" --source notes.pdf   # filter to one source
    python kb.py sources             # list ingested sources
"""

import os
import sys
import argparse
import glob
from pypdf import PdfReader
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DOCS_DIR = "documents"
DB_DIR = "chroma_store"          # persistent on-disk storage
COLLECTION_NAME = "personal_kb"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 4
GROQ_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=os.environ["GROQ_API_KEY"])


# ---------- Embedding function (TF-IDF, pure Python — see README for why) ----------

class TfidfEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, corpus: list[str]):
        self.vectorizer.fit(corpus)
        self._fitted = True

    def __call__(self, input: Documents) -> Embeddings:
        if not self._fitted:
            self.vectorizer.fit(input)
            self._fitted = True
        return self.vectorizer.transform(input).toarray().tolist()


# ---------- Document loading ----------

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_md(path: str) -> str:
    return load_txt(path)


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_md,
}


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def discover_documents() -> list[str]:
    files = []
    for ext in LOADERS:
        files.extend(glob.glob(os.path.join(DOCS_DIR, f"**/*{ext}"), recursive=True))
    return sorted(files)


# ---------- Ingestion ----------

def ingest():
    files = discover_documents()
    if not files:
        print(f"No documents found in {DOCS_DIR}/. Add .pdf, .txt, or .md files and re-run.")
        return

    print(f"Found {len(files)} documents.")

    all_chunks, all_ids, all_metas = [], [], []
    chunk_counter = 0

    for path in files:
        ext = os.path.splitext(path)[1].lower()
        loader = LOADERS[ext]
        try:
            text = loader(path)
        except Exception as e:
            print(f"  Skipped {path}: {e}")
            continue

        source_name = os.path.basename(path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  {source_name}: {len(chunks)} chunks")

        for c in chunks:
            all_chunks.append(c)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metas.append({"source": source_name, "path": path})
            chunk_counter += 1

    if not all_chunks:
        print("No text could be extracted from any document.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}. Building embeddings + persistent store...")

    embed_fn = TfidfEmbeddingFunction()
    embed_fn.fit(all_chunks)

    persistent_client = chromadb.PersistentClient(path=DB_DIR)
    try:
        persistent_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = persistent_client.create_collection(COLLECTION_NAME, embedding_function=embed_fn)

    # Chroma has batch size limits; add in chunks of 500 to be safe
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        collection.add(
            documents=all_chunks[i:i + BATCH],
            ids=all_ids[i:i + BATCH],
            metadatas=all_metas[i:i + BATCH],
        )

    print(f"Ingested {len(files)} documents, {len(all_chunks)} chunks, into '{DB_DIR}'.")


# ---------- Query ----------

def get_collection():
    persistent_client = chromadb.PersistentClient(path=DB_DIR)
    # NOTE: embedding_function must be re-attached on load; Chroma stores
    # the embeddings themselves, but query-time embedding needs the same fn.
    # We refit on the stored documents to reconstruct a matching vectorizer.
    collection = persistent_client.get_collection(COLLECTION_NAME)
    all_docs = collection.get()["documents"]
    embed_fn = TfidfEmbeddingFunction()
    embed_fn.fit(all_docs)
    collection._embedding_function = embed_fn
    return collection


def list_sources():
    collection = get_collection()
    metas = collection.get()["metadatas"]
    sources = sorted(set(m["source"] for m in metas))
    print(f"{len(sources)} sources ingested:")
    for s in sources:
        print(f"  - {s}")


def ask(query: str, source_filter: str | None = None):
    collection = get_collection()

    where = {"source": source_filter} if source_filter else None
    results = collection.query(query_texts=[query], n_results=TOP_K, where=where)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        print("No relevant chunks found (check --source spelling if used).")
        return

    print("\n--- Retrieved chunks ---")
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        preview = doc[:100].replace("\n", " ")
        print(f"{i}. [{meta['source']}] {preview}...")

    context = "\n\n---\n\n".join(
        f"(Source: {m['source']})\n{d}" for d, m in zip(docs, metas)
    )
    prompt = f"""Answer the question using ONLY the context below. Cite which source(s)
you used in your answer (by filename). If the context doesn't contain the
answer, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    print("\n--- Answer (streaming) ---")
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()  # newline after stream ends


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Personal Knowledge Base CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Ingest all documents in documents/")

    ask_parser = sub.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", type=str)
    ask_parser.add_argument("--source", type=str, default=None,
                             help="Filter to a specific source filename")

    sub.add_parser("sources", help="List ingested sources")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest()
    elif args.command == "ask":
        ask(args.query, args.source)
    elif args.command == "sources":
        list_sources()


if __name__ == "__main__":
    main()