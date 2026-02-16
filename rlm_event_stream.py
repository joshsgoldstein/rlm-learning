from __future__ import annotations

from typing import Callable


def make_event_callback(
    emit: Callable[[str], None] = print,
    prefix: str = "",
    show_section_headers: bool = True,
) -> Callable[[str, str], None]:
    """Return a callback(event_type, detail) that emits trace events in a TUI-like stream."""

    icons = {
        "traditional_stage": "🐌",
        "iteration": "🔄",
        "peek": "🔍",
        "search": "🔎",
        "llm_query": "🧠",
        "code": "💻",
        "output": "📤",
        "answer": "✅",
        "warning": "⚠️",
        "tokens": "📊",
    }
    pfx = f"{prefix} " if prefix else ""
    state = {"section": None}

    def _cb(event_type: str, detail: str) -> None:
        if show_section_headers:
            if event_type == "traditional_stage" and state["section"] != "traditional":
                emit(f"{pfx}=== Traditional ===")
                state["section"] = "traditional"
            elif event_type != "traditional_stage" and state["section"] != "rlm":
                emit(f"{pfx}=== RLM ===")
                state["section"] = "rlm"

        icon = icons.get(event_type, "•")
        if event_type == "code":
            emit(f"{pfx}{icon} code:")
            code = detail.strip().strip("`").strip()
            for line in code.splitlines()[:25]:
                emit(f"{pfx}   {line}")
            if len(code.splitlines()) > 25:
                emit(f"{pfx}   ...")
            return
        if event_type == "output":
            emit(f"{pfx}{icon} output:")
            lines = detail.strip().splitlines()
            for line in lines[:20]:
                emit(f"{pfx}   {line}")
            if len(lines) > 20:
                emit(f"{pfx}   ... ({len(lines) - 20} more)")
            return
        if event_type == "tokens":
            try:
                i, o, t = [int(x) for x in detail.split(",")]
                emit(f"{pfx}{icon} tokens: {i:,} in / {o:,} out / {t:,} total")
            except Exception:
                emit(f"{pfx}{icon} tokens: {detail}")
            return
        emit(f"{pfx}{icon} {event_type}: {detail}")

    return _cb


def make_console_event_callback(prefix: str = "") -> Callable[[str, str], None]:
    """Compatibility helper: print to stdout."""
    return make_event_callback(emit=print, prefix=prefix, show_section_headers=True)
