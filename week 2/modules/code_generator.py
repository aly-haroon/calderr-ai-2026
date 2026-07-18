"""
code_generator.py

Wraps the Groq LLM (via LangChain's ChatGroq) to turn a natural-language
question + schema description into a single block of executable pandas
code. The model is instructed to:
    * assume a DataFrame named `df` already exists,
    * store its final answer in a variable named `result`,
    * return ONLY code (no prose, no markdown fences).

We still defensively strip markdown fences and validate the output before
it ever reaches the executor, because LLMs occasionally ignore instructions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from modules.schema_analyzer import SchemaReport
from modules.question_parser import ParsedQuestion

MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0

SYSTEM_PROMPT = """You are an expert Python data analyst that writes pandas code.

STRICT RULES:
1. A pandas DataFrame named `df` already exists in scope. Never redefine or reload it.
2. Store your final answer in a variable named exactly `result`.
   - If the answer is a number, string, or short text, `result` should hold that value.
   - If the answer is tabular, `result` should be a DataFrame or Series.
3. You may use: pandas (as pd), numpy (as np), matplotlib.pyplot (as plt).
   Do NOT import anything else. Do NOT use `os`, `sys`, `subprocess`, `open`,
   `eval`, `exec`, `__import__`, or any networking/file-system module.
4. Do not read or write files. Do not access the network.
5. If you create a chart, build it with matplotlib on a figure called `fig`
   (e.g. `fig, ax = plt.subplots()`) instead of calling `plt.show()`.
6. Only reference columns that exist in the schema provided below.
7. Return ONLY raw Python code. No markdown fences, no explanations, no comments
   about what you are about to do outside of inline `#` comments in the code itself.
"""

USER_PROMPT_TEMPLATE = """Dataset schema:
{schema}

Sample rows:
{sample_rows}

User question:
{question}

Write pandas code that answers this question, following all rules above.
"""

# Any of these substrings appearing in generated code causes immediate rejection,
# regardless of what the LLM was told. This is a second, independent layer of
# defense on top of the prompt instructions (see code_executor.py for the
# runtime sandbox, which is the third layer).
FORBIDDEN_PATTERNS = [
    r"\bos\.", r"\bsys\.", r"\bsubprocess\b", r"\b__import__\b",
    r"\bopen\s*\(", r"\beval\s*\(", r"\bexec\s*\(", r"\bimport\s+os\b",
    r"\bimport\s+sys\b", r"\bimport\s+subprocess\b", r"\bimport\s+socket\b",
    r"\bimport\s+shutil\b", r"\brequests\.", r"\burllib\b", r"\bglobals\s*\(",
    r"\blocals\s*\(", r"\bcompile\s*\(", r"\binput\s*\(", r"\bpickle\b",
    r"\.system\s*\(", r"\bremove\s*\(", r"\bunlink\s*\(", r"\brmdir\s*\(",
]


class CodeGenerationError(Exception):
    """Raised when the LLM fails to return valid, safe-looking code."""


@dataclass
class GeneratedCode:
    """The result of a code-generation call."""

    code: str
    raw_llm_output: str


def _strip_markdown_fences(text: str) -> str:
    """Remove ```python / ``` fences if the model added them anyway."""
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _validate_code_is_safe(code: str) -> None:
    """
    First-pass static check: reject obviously dangerous code before it is
    ever executed. This is a defense-in-depth measure — the *authoritative*
    protection is the restricted execution environment in code_executor.py,
    which has no access to these modules/functions even if a pattern slips
    past this regex check.
    """
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            raise CodeGenerationError(
                f"Generated code was rejected for containing a disallowed "
                f"operation matching pattern: {pattern}"
            )
    if "result" not in code:
        raise CodeGenerationError(
            "Generated code does not assign a `result` variable as required."
        )


def _build_llm(api_key: str) -> ChatGroq:
    """Construct the ChatGroq client with deterministic settings."""
    return ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=api_key,
    )


def generate_pandas_code(
    question: ParsedQuestion,
    schema: SchemaReport,
    api_key: str | None = None,
) -> GeneratedCode:
    """
    Call Groq to generate pandas code answering `question` against `schema`.

    Args:
        question: The validated user question.
        schema: The dataset's schema report (used to ground the prompt).
        api_key: Groq API key. Falls back to the GROQ_API_KEY env var.

    Returns:
        GeneratedCode containing the cleaned, validated code string.

    Raises:
        CodeGenerationError: If no API key is available, the LLM call fails,
            or the returned code fails safety validation.
    """
    resolved_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_key:
        raise CodeGenerationError(
            "No Groq API key found. Set GROQ_API_KEY in your environment or .env file."
        )

    sample_rows_str = "\n".join(str(row) for row in schema.sample_rows[:5])
    user_prompt = USER_PROMPT_TEMPLATE.format(
        schema=schema.to_prompt_string(),
        sample_rows=sample_rows_str,
        question=question.cleaned_text,
    )

    try:
        llm = _build_llm(resolved_key)
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        raw_output = str(response.content)
    except Exception as exc:  # noqa: BLE001 - surface any LLM/network failure cleanly
        raise CodeGenerationError(f"Groq API call failed: {exc}") from exc

    code = _strip_markdown_fences(raw_output)
    if not code:
        raise CodeGenerationError("The LLM returned an empty response.")

    _validate_code_is_safe(code)

    return GeneratedCode(code=code, raw_llm_output=raw_output)
