from __future__ import annotations

import os
from pathlib import Path


DATA_DIR = Path(os.getenv("RLM_DATA_DIR", "data"))
PROCESSED_DIR = Path(os.getenv("RLM_PROCESSED_DIR", "processed_data"))

CONCEPT_NOTES = {
    "iteration": "🔄 [bold]RLM Loop[/] — Each iteration, the LLM writes code to explore the external environment. The context never enters the prompt directly. (Paper §3)",
    "peek": "📖 [bold]Context as Variable[/] — The LLM reads a page from the environment. Documents are stored externally, not in the context window. This prevents context rot. (Paper §3.1)",
    "search": "🔎 [bold]Programmatic Search[/] — The LLM wrote code to grep the environment. It decides what to search for — the system doesn't pre-chunk or pre-embed. (Paper §3.2)",
    "llm_query": "🧠 [bold]Recursive Sub-Call[/] — The LLM invokes itself on a small chunk to extract specific facts. This is the 'recursive' in RLM — sub-LM calls on filtered context. (Paper §3.3)",
    "answer": "✅ [bold]Evidence Synthesis[/] — The LLM has gathered enough evidence from targeted reads and sub-calls. It composes a cited answer without ever seeing the full corpus. (Paper §4)",
    "code": "💻 [bold]Code Generation[/] — The LLM writes Python to interact with the REPL. It controls the decomposition strategy — not the system. (Paper §3.1)",
    "output": "📤 [bold]REPL Output[/] — Results from code execution feed back to the LLM as context for the next iteration. The state persists like a Jupyter notebook.",
}

INSPECTOR_SIZES = [0, 50, 70, 90, 110]

SLASH_COMMANDS = [
    ("/help", "show commands"),
    ("/docs", "list all documents"),
    ("/test", "run query from RLM_TEST_QUERY in .env"),
    ("/config [N]", "show/set max iterations"),
    ("/traditional", "show last traditional stats"),
    ("/rag status|retry|reset|topk N|mode semantic|hybrid|alpha X", "RAG controls"),
    ("/eval [file]", "run benchmark in chat"),
    ("/autoeval [on|off|status]", "toggle post-turn eval"),
    ("/copy", "copy last turn as JSON"),
    ("/clear", "clear chat and inspector"),
    ("/quit", "exit app"),
]

INPUT_SUGGESTIONS = ["/help", "/docs", "/test", "/config", "/traditional", "/rag", "/eval", "/autoeval", "/copy", "/clear", "/quit", "/exit"]
