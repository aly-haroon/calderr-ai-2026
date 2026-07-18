"""
fetch_sentences.py
Pulls sentences from a handful of Wikipedia articles and caches
them to sentences.json so we don't hit the API every run.

Usage:
    python fetch_sentences.py
"""

import json
import re
import wikipedia

TOPICS = [
    "Artificial intelligence",
    "Python (programming language)",
    "Islamabad",
    "Quantum computing",
    "Climate change",
    "Ancient Rome",
    "Football",
    "The Beatles",
]

OUTPUT_FILE = "sentences.json"
TARGET_COUNT = 100


def clean_split(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", "", text)
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if len(s.strip()) > 30]


def main():
    all_sentences = []

    for topic in TOPICS:
        try:
            print(f"Fetching: {topic}")
            page = wikipedia.page(topic, auto_suggest=False)
            sentences = clean_split(page.summary + " " + page.content[:3000])
            all_sentences.extend(sentences)
        except Exception as e:
            print(f"  Skipped '{topic}': {e}")

    seen = set()
    unique_sentences = []
    for s in all_sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    final = unique_sentences[:TARGET_COUNT]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(final)} sentences to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()