"""
semantic_search_fallback.py — TEMPORARY stopgap for Lab 3.1

torch/sentence-transformers is broken (DLL load error) on this machine.
This version uses TF-IDF + cosine similarity from scikit-learn instead,
so you have a working demo. Swap to real embeddings once torch is fixed.

Usage:
    python semantic_search_fallback.py
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SENTENCES_FILE = "sentences.json"
TOP_K = 5


def load_sentences():
    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    sentences = load_sentences()
    print(f"Loaded {len(sentences)} sentences.\n")
    print("NOTE: using TF-IDF (keyword-based) fallback, not real embeddings.")
    print("Real embedding models are blocked by a torch DLL error on this machine.\n")

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(sentences)

    print("Search ready. Type a query (or 'quit' to exit).")
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, doc_matrix).flatten()
        top_idx = sims.argsort()[::-1][:TOP_K]

        print("\n--- TF-IDF Results ---")
        for rank, i in enumerate(top_idx, start=1):
            print(f"{rank}. ({sims[i]:.4f}) {sentences[i]}")


if __name__ == "__main__":
    main()