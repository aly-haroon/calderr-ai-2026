"""
naive_rag.py — Lab 3.2: Naive RAG Pipeline

Pipeline: load PDF -> split into chunks -> embed (ChromaDB default,
ONNX-based, no torch needed) -> store in ChromaDB -> retrieve on query
-> generate answer with Groq.

Usage:
    python naive_rag.py
"""

import os
from pypdf import PdfReader
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

PDF_PATH = "CalderR_Week-3.pdf"
CHUNK_SIZE = 512     # characters per chunk (simple char-based chunking for speed)
CHUNK_OVERLAP = 50
TOP_K = 3
GROQ_MODEL = "llama-3.3-70b-versatile"


class TfidfEmbeddingFunction(EmbeddingFunction):
    """
    Pure-Python embedding function for ChromaDB using TF-IDF.
    Avoids onnxruntime/torch, which hit a DLL load error on this machine.
    The vectorizer is fit once on the full corpus, then reused for queries.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, corpus: list[str]):
        self.vectorizer.fit(corpus)
        self._fitted = True

    def __call__(self, input: Documents) -> Embeddings:
        if not self._fitted:
            # Fallback: fit on whatever is passed in (shouldn't normally happen
            # since we fit explicitly before adding documents)
            self.vectorizer.fit(input)
            self._fitted = True
        vectors = self.vectorizer.transform(input).toarray()
        return vectors.tolist()


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_chroma_collection(chunks: list[str]):
    client = chromadb.Client()  # in-memory for this lab

    # Fit TF-IDF on the full corpus BEFORE adding, so query vectors
    # later are comparable (same vocabulary/dimensions)
    embed_fn = TfidfEmbeddingFunction()
    embed_fn.fit(chunks)

    # If a stale collection exists from a previous run, drop it first
    try:
        client.delete_collection("week3_rag")
    except Exception:
        pass
    collection = client.create_collection("week3_rag", embedding_function=embed_fn)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_index": i, "source": PDF_PATH} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return collection


def retrieve(collection, query: str, top_k: int = TOP_K):
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


def generate_answer(query: str, retrieved: list[tuple[str, dict]]) -> str:
    context = "\n\n---\n\n".join([doc for doc, _ in retrieved])

    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def main():
    print(f"Loading PDF: {PDF_PATH}")
    text = load_pdf_text(PDF_PATH)
    print(f"  Extracted {len(text)} characters")

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("Building ChromaDB collection (embedding chunks)...")
    collection = build_chroma_collection(chunks)
    print("  Done.\n")

    print("RAG ready. Type a question (or 'quit' to exit).")
    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        retrieved = retrieve(collection, query)
        print("\n--- Retrieved chunks ---")
        for i, (doc, meta) in enumerate(retrieved, start=1):
            preview = doc[:120].replace("\n", " ")
            print(f"{i}. [chunk {meta['chunk_index']}] {preview}...")

        answer = generate_answer(query, retrieved)
        print(f"\n--- Answer ---\n{answer}")


if __name__ == "__main__":
    main()