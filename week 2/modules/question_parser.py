"""
question_parser.py

Lightweight validation/normalization layer for the user's natural language
question. It does NOT try to understand the question semantically (that is
the LLM's job in code_generator.py) — it only guards against empty, absurdly
long, or clearly non-analytical input before we spend an API call on it.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_QUESTION_LENGTH = 500
MIN_QUESTION_LENGTH = 3


class QuestionParserError(Exception):
    """Raised when the user's question fails basic validation."""


@dataclass
class ParsedQuestion:
    """A cleaned, validated question ready to be sent to the LLM."""

    raw_text: str
    cleaned_text: str

    @property
    def is_visual_request(self) -> bool:
        """
        Heuristic: does the question explicitly ask for a chart/plot?
        Used later to bias the visualization generator, but visualization
        is still attempted even when this is False, based on result shape.
        """
        keywords = (
            "plot", "chart", "graph", "visuali", "trend", "distribution",
            "histogram", "scatter", "pie", "bar chart", "line chart",
        )
        return any(k in self.cleaned_text.lower() for k in keywords)


def parse_question(raw_text: str) -> ParsedQuestion:
    """
    Validate and normalize a raw question string.

    Args:
        raw_text: The exact text the user typed.

    Returns:
        ParsedQuestion with whitespace-normalized text.

    Raises:
        QuestionParserError: If the question is empty, too short, or too long.
    """
    if raw_text is None:
        raise QuestionParserError("Please enter a question about your dataset.")

    cleaned = " ".join(raw_text.strip().split())

    if len(cleaned) < MIN_QUESTION_LENGTH:
        raise QuestionParserError(
            "Your question is too short. Please describe what you'd like to know."
        )
    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise QuestionParserError(
            f"Your question is too long (max {MAX_QUESTION_LENGTH} characters). "
            "Please make it more concise."
        )

    return ParsedQuestion(raw_text=raw_text, cleaned_text=cleaned)
