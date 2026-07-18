# 📊 Automated Data Analysis Agent

An AI agent that accepts a CSV file and a natural-language question, generates
pandas code using Groq's `llama-3.3-70b-versatile`, safely executes it in a
restricted sandbox, auto-generates visualizations, and produces a downloadable
report — all inside a Streamlit app.

## Architecture

```
CSV Upload
   -> Schema Analyzer      (modules/schema_analyzer.py)
   -> Question Parser      (modules/question_parser.py)
   -> Groq Code Generator  (modules/code_generator.py)
   -> Safe Code Executor   (modules/code_executor.py)
   -> Visualization Gen.   (modules/visualization.py)
   -> Report Builder       (modules/report_builder.py)
   -> Streamlit UI         (app.py)
```

## Project Structure

```
project_2_p_c/
├── app.py                     # Streamlit entry point, wires the pipeline together
├── modules/
│   ├── __init__.py
│   ├── schema_analyzer.py     # Extracts columns, dtypes, missing values, samples
│   ├── question_parser.py     # Validates/normalizes the user's question
│   ├── code_generator.py      # Groq/LangChain pandas code generation
│   ├── code_executor.py       # Restricted sandbox execution
│   ├── visualization.py       # Auto chart-type selection (bar/line/hist/scatter/pie)
│   └── report_builder.py      # Assembles the final Markdown report
├── sample_data/
│   └── sales_sample.csv       # Example dataset to try the agent with
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
cd project_2_p_c
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY (get a free key at https://console.groq.com/keys)

streamlit run app.py
```

You can also paste your Groq API key directly into the sidebar of the running
app instead of using a `.env` file.

## Try it

1. Upload `sample_data/sales_sample.csv`.
2. Ask something like:
   - "What is the total revenue by region?"
   - "Show me the trend of revenue over time."
   - "Which product category has the highest average unit price?"
   - "What is the distribution of quantity ordered?"

## Security

Generated code is **never** run with a raw, unrestricted `exec()`. Three
layers of defense are used (see detailed comments in
`modules/code_generator.py` and `modules/code_executor.py`):

1. **Prompt-level constraints** — the system prompt instructs the model to
   only use `pandas`/`numpy`/`matplotlib`, never touch files/network/OS, and
   always assign to a `result` variable. A regex pre-filter additionally
   rejects any generated code containing tokens like `os.`, `subprocess`,
   `open(`, `eval(`, `exec(`, `__import__`, `requests.`, etc. **before** it
   is ever executed.

2. **Restricted execution namespace** — code runs via `exec(code, namespace)`
   where `namespace["__builtins__"]` is replaced with a small allow-list of
   safe built-ins (`len`, `range`, `sum`, `print`, ...). Dangerous built-ins
   such as `open`, `eval`, `exec`, `__import__`, `input`, and `compile` are
   simply **not present**, so any attempt to use them raises a `NameError`.
   The only external names available are `df` (a defensive copy — generated
   code cannot mutate your original data), `pd`, `np`, and `plt`.

3. **Timeout watchdog** — execution runs on a daemon thread with a
   15-second timeout, so an inefficient or infinite loop in generated code
   cannot hang the server.

**What this does *not* do:** this is process-level sandboxing within the
same Python interpreter, appropriate for a trusted single-user demo (e.g.
Streamlit Cloud with your own API key). It is **not** OS-level isolation.
If you deploy this for untrusted/multi-tenant users, additionally run the
whole app (or at minimum the execution step) inside a container with
`--network=none`, a read-only filesystem, a non-root user, and a tool such
as `gVisor`, `firejail`, or a microVM (Firecracker) for kernel-level
isolation.

## Error Handling Covered

- Invalid / unparseable CSV files
- Empty datasets or datasets with zero columns
- Empty, too-short, or too-long questions
- LLM/API failures (missing key, network errors, empty responses)
- Generated code that fails the safety pre-filter
- Runtime exceptions during execution (caught per-exception, shown cleanly)
- Execution timeouts
- Missing `result` variable after execution
- Chart-generation failures (caught independently so a bad chart never
  blocks the rest of the report)

## Deployment

The app runs with a single command:

```bash
streamlit run app.py
```

To deploy on **Streamlit Community Cloud**: push this project to a GitHub
repo, create a new app pointing at `app.py`, and add `GROQ_API_KEY` under
"Secrets" in the app settings.

## Tech Stack

Python 3.12 · Streamlit · LangChain · langchain-groq · Groq API (Llama 3.3
70B Versatile) · Pandas · NumPy · Matplotlib · python-dotenv · Pydantic
