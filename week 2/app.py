"""
app.py

Streamlit entry point for the Automated Data Analysis Agent.

Run with:
    streamlit run app.py

Pipeline wired up here:
    CSV Upload -> Schema Analyzer -> Question Parser -> Groq Code Generator
    -> Safe Code Executor -> Visualization Generator -> Report Builder -> UI
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.schema_analyzer import analyze_schema, SchemaAnalyzerError, SchemaReport
from modules.question_parser import parse_question, QuestionParserError
from modules.code_generator import generate_pandas_code, CodeGenerationError
from modules.code_executor import execute_pandas_code, CodeExecutionError
from modules.visualization import build_chart_from_result, VisualizationError
from modules.report_builder import build_report

load_dotenv()

st.set_page_config(
    page_title="Automated Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)

MAX_UPLOAD_MB = 50


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse uploaded CSV bytes into a DataFrame. Cached on file content so
    re-running analyses on the same file doesn't re-parse it.
    """
    return pd.read_csv(io.BytesIO(file_bytes))


def _render_schema_panel(schema: SchemaReport) -> None:
    """Render dataset statistics + schema in the sidebar-style panel."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{schema.row_count:,}")
    c2.metric("Columns", schema.column_count)
    c3.metric("Missing cells", f"{schema.total_missing_cells:,}")
    c4.metric("Memory", f"{schema.memory_usage_kb:,.1f} KB")

    with st.expander("📋 Full schema details", expanded=False):
        schema_df = pd.DataFrame(
            [
                {
                    "Column": col.name,
                    "Type": col.dtype,
                    "Missing": col.missing_count,
                    "Missing %": col.missing_pct,
                    "Unique": col.unique_count,
                    "Category": "Numeric" if col.is_numeric else "Categorical",
                }
                for col in schema.columns
            ]
        )
        st.dataframe(schema_df, use_container_width=True)


def _render_result(result: Any) -> None:
    """Display the execution result using the most appropriate widget."""
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True)
    elif isinstance(result, pd.Series):
        st.dataframe(result.to_frame(name="value"), use_container_width=True)
    else:
        st.success(f"Result: {result}")


def _init_session_state() -> None:
    defaults = {
        "df": None,
        "schema": None,
        "dataset_name": None,
        "last_report_md": None,
        "history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def main() -> None:
    _init_session_state()

    st.title("📊 Automated Data Analysis Agent")
    st.caption(
        "Upload a CSV, ask a question in plain English, and get pandas code, "
        "results, and charts — generated and executed automatically."
    )

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Falls back to GROQ_API_KEY from your .env file if left blank.",
        )
        st.markdown("---")
        st.markdown(
            "**Pipeline**\n\n"
            "1. CSV Upload\n2. Schema Analyzer\n3. Question Parser\n"
            "4. Groq Code Generator\n5. Safe Code Executor\n"
            "6. Visualization Generator\n7. Report Builder\n8. Streamlit UI"
        )
        st.markdown("---")
        st.caption(
            "🔒 Generated code runs in a restricted sandbox: no file access, "
            "no network, no OS calls. See README for details."
        )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_UPLOAD_MB}MB upload limit.")
            return

        try:
            with st.status("Reading and analyzing dataset...", expanded=False) as status:
                df = _load_csv(file_bytes)
                schema = analyze_schema(df)
                status.update(label="Dataset loaded successfully.", state="complete")
        except pd.errors.EmptyDataError:
            st.error("This CSV file appears to be empty or invalid.")
            return
        except pd.errors.ParserError as exc:
            st.error(f"Could not parse this CSV file: {exc}")
            return
        except SchemaAnalyzerError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error while reading the file: {exc}")
            return

        st.session_state.df = df
        st.session_state.schema = schema
        st.session_state.dataset_name = uploaded_file.name

        st.success(f"Loaded **{uploaded_file.name}** — {schema.row_count:,} rows × {schema.column_count} columns")

        tab_preview, tab_schema = st.tabs(["🔍 Preview", "🧬 Schema & Stats"])
        with tab_preview:
            st.dataframe(df.head(20), use_container_width=True)
        with tab_schema:
            _render_schema_panel(schema)
            with st.expander("Descriptive statistics (numeric columns)"):
                if schema.numerical_columns:
                    st.dataframe(df[schema.numerical_columns].describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found in this dataset.")

        st.markdown("---")
        st.subheader("💬 Ask a question about your data")
        question_text = st.text_area(
            "Question",
            placeholder="e.g. What is the average sales by region? Show me the trend of revenue over time.",
            label_visibility="collapsed",
        )
        analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=False)

        if analyze_clicked:
            _run_analysis(question_text, df, schema, uploaded_file.name, api_key)

    else:
        st.info("👆 Upload a CSV file to get started.")
        st.markdown(
            "Don't have a dataset handy? Try the sample files in "
            "`project_2_p_c/sample_data/` from this project."
        )


def _run_analysis(
    question_text: str,
    df: pd.DataFrame,
    schema: SchemaReport,
    dataset_name: str,
    api_key: str,
) -> None:
    """Execute the full question -> code -> result -> chart -> report pipeline."""
    progress = st.progress(0, text="Starting analysis...")

    # Step 1: Parse/validate question
    try:
        parsed_question = parse_question(question_text)
        progress.progress(15, text="Question validated...")
    except QuestionParserError as exc:
        progress.empty()
        st.error(str(exc))
        return

    # Step 2: Generate code via Groq
    try:
        progress.progress(35, text="Generating pandas code with Groq (Llama 3.3 70B)...")
        generated = generate_pandas_code(parsed_question, schema, api_key=api_key or None)
    except CodeGenerationError as exc:
        progress.empty()
        st.error(f"Code generation failed: {exc}")
        return

    with st.expander("🧾 Generated Python code", expanded=True):
        st.code(generated.code, language="python")

    # Step 3: Execute code safely
    try:
        progress.progress(65, text="Executing code in the sandbox...")
        exec_result = execute_pandas_code(generated.code, df)
    except CodeExecutionError as exc:
        progress.empty()
        st.error(f"Code execution failed: {exc}")
        st.info(
            "Tip: try rephrasing your question, or check that it references "
            "columns that actually exist in the dataset."
        )
        return

    if exec_result.stdout.strip():
        with st.expander("🖨️ Captured output (print statements)"):
            st.text(exec_result.stdout)

    st.markdown("### 📈 Result")
    _render_result(exec_result.result)

    # Step 4: Visualization
    progress.progress(85, text="Building visualization...")
    chart_type = None
    figure = exec_result.figure
    if figure is not None:
        chart_type = "custom (generated by LLM code)"
    else:
        try:
            chart = build_chart_from_result(exec_result.result)
            if chart is not None:
                figure = chart.figure
                chart_type = chart.chart_type
        except VisualizationError as exc:
            st.warning(f"Could not auto-generate a chart: {exc}")

    if figure is not None:
        st.markdown("### 📊 Chart")
        st.pyplot(figure, use_container_width=True)
    else:
        st.caption("No chart was generated for this result type.")

    # Step 5: Report builder
    progress.progress(100, text="Building report...")
    report = build_report(
        question=parsed_question.cleaned_text,
        generated_code=generated.code,
        result=exec_result.result,
        schema=schema,
        chart_type=chart_type,
        dataset_name=dataset_name,
    )
    st.session_state.last_report_md = report.markdown
    progress.empty()

    st.markdown("### 📄 Report")
    with st.expander("Preview full report", expanded=False):
        st.markdown(report.markdown)

    st.download_button(
        "⬇️ Download report (Markdown)",
        data=report.markdown,
        file_name="data_analysis_report.md",
        mime="text/markdown",
    )

    st.session_state.history.append(
        {"question": parsed_question.cleaned_text, "code": generated.code}
    )


if __name__ == "__main__":
    main()
