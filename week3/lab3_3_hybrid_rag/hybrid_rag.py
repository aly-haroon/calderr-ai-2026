"""
hybrid_rag.py — Lab 3.3: Hybrid Retrieval & Evaluation

Pipeline: load PDF -> chunk -> build BM25 index + TF-IDF semantic index
-> hybrid retrieval (weighted score combo) -> Groq re-ranks the hybrid
top results -> generate answer -> Groq-based evaluation (faithfulness,
answer relevancy, context precision), mirroring RAGAS's core metrics
without the torch/onnxruntime dependency (broken via DLL error on
this machine).

Usage:
    python hybrid_rag.py
"""

import os
import re
import json
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

PDF_PATH = "CalderR_Week-3.pdf"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
HYBRID_TOP_K = 8      # candidates pulled before re-ranking
FINAL_TOP_K = 3        # chunks kept after re-ranking
BM25_WEIGHT = 0.5      # hybrid combination weight (semantic gets 1 - this)
GROQ_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def normalize(scores):
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridIndex:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c) for c in chunks])
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K):
        bm25_scores = self.bm25.get_scores(tokenize(query))
        q_vec = self.vectorizer.transform([query])
        semantic_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        bm25_norm = normalize(list(bm25_scores))
        sem_norm = normalize(list(semantic_scores))

        hybrid_scores = [
            BM25_WEIGHT * b + (1 - BM25_WEIGHT) * s
            for b, s in zip(bm25_norm, sem_norm)
        ]

        ranked = sorted(
            range(len(self.chunks)), key=lambda i: hybrid_scores[i], reverse=True
        )[:top_k]

        return [(self.chunks[i], hybrid_scores[i], i) for i in ranked]


def groq_rerank(query: str, candidates: list[tuple[str, float, int]], top_k: int = FINAL_TOP_K):
    """
    Ask Groq to score each candidate chunk's relevance to the query on 0-10.
    Substitutes for a cross-encoder re-ranker (which needs torch).
    """
    scored = []
    for chunk, hybrid_score, idx in candidates:
        prompt = f"""Rate how relevant this passage is to the question, on a scale of 0-10.
Respond with ONLY a number, nothing else.

Question: {query}

Passage: {chunk}

Relevance score (0-10):"""
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        try:
            score = float(re.search(r"\d+(\.\d+)?", resp.choices[0].message.content).group())
        except (AttributeError, ValueError):
            score = 0.0
        scored.append((chunk, score, idx))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def generate_answer(query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def groq_eval(query: str, context_chunks: list[str], answer: str) -> dict:
    """
    RAGAS-style evaluation using Groq as judge instead of RAGAS's package
    (which depends on torch/embeddings). Mirrors three core RAGAS metrics:
    faithfulness, answer relevancy, context precision.
    """
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are evaluating a RAG system's output. Score each metric 0-10.

Question: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Score these three metrics:
1. Faithfulness: Is the answer fully supported by the context, with no hallucinated facts?
2. Answer Relevancy: Does the answer directly address the question asked?
3. Context Precision: Is the retrieved context actually relevant and useful for answering, without irrelevant filler?

Respond ONLY in this exact JSON format, nothing else:
{{"faithfulness": <0-10>, "answer_relevancy": <0-10>, "context_precision": <0-10>, "notes": "<one short sentence>"}}"""

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences if the model adds them
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithfulness": None, "answer_relevancy": None,
                "context_precision": None, "notes": f"Could not parse: {raw}"}


def main():
    print(f"Loading PDF: {PDF_PATH}")
    text = load_pdf_text(PDF_PATH)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  {len(chunks)} chunks created")

    print("Building hybrid index (BM25 + TF-IDF)...")
    index = HybridIndex(chunks)
    print("  Done.\n")

    print("Hybrid RAG ready. Type a question (or 'quit' to exit).")
    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        candidates = index.search(query)
        print(f"\n--- Hybrid candidates (top {len(candidates)}) ---")
        for chunk, score, idx in candidates:
            print(f"  [chunk {idx}] hybrid_score={score:.3f}")

        reranked = groq_rerank(query, candidates)
        print(f"\n--- After Groq re-ranking (top {len(reranked)}) ---")
        for chunk, score, idx in reranked:
            preview = chunk[:100].replace("\n", " ")
            print(f"  [chunk {idx}] relevance={score:.1f}  {preview}...")

        final_chunks = [c for c, _, _ in reranked]
        answer = generate_answer(query, final_chunks)
        print(f"\n--- Answer ---\n{answer}")

        print("\nEvaluating (RAGAS-style, via Groq judge)...")
        scores = groq_eval(query, final_chunks, answer)
        print(f"--- Evaluation ---\n{json.dumps(scores, indent=2)}")


if __name__ == "__main__":
    main()