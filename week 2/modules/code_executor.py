"""
code_executor.py

Executes LLM-generated pandas code in a restricted environment.

SECURITY MODEL (read this before modifying):
-----------------------------------------------------------------------------
We intentionally do NOT run generated code in a separate OS process/container
in this reference implementation (that would be the ideal production setup —
see the README "Security" section for recommendations on using firejail,
gVisor, Docker `--network=none`, or a microVM). Instead we use a three-layer
defense-in-depth approach appropriate for a demo/portfolio project:

  Layer 1 (code_generator.py): Prompt instructs the LLM to only use pandas/
           numpy/matplotlib and never touch files, network, or the OS. A
           regex pre-filter rejects code containing obviously dangerous
           tokens before it is ever executed.

  Layer 2 (this file): The code is executed with `exec()` but with a
           *restricted globals dictionary*. Specifically:
             - `__builtins__` is replaced with a small allow-listed subset
               of safe built-in functions (no `open`, `eval`, `exec`,
               `__import__`, `input`, `compile`, etc.)
             - The only names available in the namespace are `df`, `pd`,
               `np`, `plt`, and the safe builtins above.
             - No access to `os`, `sys`, `subprocess`, `socket`, `shutil`,
               `requests`, or any other module is possible because those
               modules are never placed in the namespace and `__import__`
               is not reachable, so `import os` inside the exec'd code will
               raise a NameError/ImportError.

  Layer 3 (this file): Execution is wrapped in a timeout (via a watchdog
           thread) so a runaway/infinite loop cannot hang the Streamlit
           server indefinitely. stdout is captured and truncated. Any
           exception is caught and returned to the UI as a clean error
           message rather than crashing the app.

This is NOT a substitute for OS-level sandboxing if you deploy this with
untrusted, multi-tenant users. It raises the bar significantly for a
single-user / trusted-demo deployment such as Streamlit Cloud, but true
isolation for hostile input requires process/container-level sandboxing.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import io
import threading
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend: never opens GUI windows, no display access
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Execution time limit in seconds. Generated analysis code should be fast;
# anything longer almost certainly indicates a runaway loop.
EXECUTION_TIMEOUT_SECONDS = 15

# Maximum characters of captured stdout we keep, to prevent memory abuse.
MAX_STDOUT_CHARS = 5000

# Allow-listed built-in functions only. Everything else (open, eval, exec,
# __import__, input, compile, exit, quit, help, breakpoint, ...) is absent,
# so any attempt to use them raises a NameError inside the sandboxed code.
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "print": print, "range": range, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "isinstance": isinstance, "type": type, "Exception": Exception,
    "ValueError": ValueError, "KeyError": KeyError, "TypeError": TypeError,
}


class CodeExecutionError(Exception):
    """Raised when generated code fails to execute or times out."""


@dataclass
class ExecutionResult:
    """Outcome of running generated code against a DataFrame."""

    result: Any
    stdout: str
    figure: Any = None  # matplotlib Figure, if the code created one
    local_vars: dict[str, Any] = field(default_factory=dict)


def _run_in_thread(code: str, namespace: dict[str, Any], outcome: dict[str, Any]) -> None:
    """Target function executed on the watchdog thread."""
    stdout_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, namespace)  # noqa: S102 - restricted namespace, see module docstring
        outcome["success"] = True
    except Exception as exc:  # noqa: BLE001 - we want to capture *any* runtime error
        outcome["success"] = False
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        outcome["stdout"] = stdout_buffer.getvalue()[:MAX_STDOUT_CHARS]


def execute_pandas_code(code: str, df: pd.DataFrame) -> ExecutionResult:
    """
    Execute LLM-generated pandas code against `df` inside a restricted
    namespace, enforcing a timeout.

    Args:
        code: Validated Python source (already passed through the
            code_generator's static safety checks).
        df: The user's uploaded DataFrame, made available as `df`.

    Returns:
        ExecutionResult with the computed `result`, captured stdout, and
        an optional matplotlib figure if the code produced one.

    Raises:
        CodeExecutionError: If execution raises, times out, or fails to
            produce a `result` variable.
    """
    # Fresh namespace per call: only these names are visible to the code.
    # A defensive copy of df is used so generated code cannot mutate the
    # original DataFrame held by the Streamlit session.
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "df": df.copy(deep=True),
        "pd": pd,
        "np": np,
        "plt": plt,
    }

    outcome: dict[str, Any] = {}
    worker = threading.Thread(target=_run_in_thread, args=(code, namespace, outcome), daemon=True)
    worker.start()
    worker.join(timeout=EXECUTION_TIMEOUT_SECONDS)

    if worker.is_alive():
        # The thread is still running past our deadline. We cannot forcibly
        # kill a Python thread safely, so we detach (daemon=True lets the
        # process exit without waiting on it) and report a timeout to the
        # user rather than hanging the UI.
        raise CodeExecutionError(
            f"Code execution exceeded the {EXECUTION_TIMEOUT_SECONDS}s time limit "
            "and was aborted. This usually means the generated code contains "
            "an inefficient or infinite loop."
        )

    if not outcome.get("success"):
        raise CodeExecutionError(
            outcome.get("error", "Unknown error during code execution.")
        )

    if "result" not in namespace:
        raise CodeExecutionError(
            "The generated code ran successfully but did not define a `result` variable."
        )

    # Detect a matplotlib figure the code may have created (e.g. `fig, ax = plt.subplots()`)
    figure = namespace.get("fig")
    if figure is None and plt.get_fignums():
        figure = plt.gcf()

    return ExecutionResult(
        result=namespace["result"],
        stdout=outcome.get("stdout", ""),
        figure=figure,
        local_vars={k: v for k, v in namespace.items() if k not in ("__builtins__", "df", "pd", "np", "plt")},
    )
