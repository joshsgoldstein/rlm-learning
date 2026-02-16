# rlm_cli.py — RLM Chat CLI
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from rlm_core import Config, ChatMessage, TokenUsage, answer_question
from rlm_docs import (
    discover_docs,
    list_pdf_files,
    missing_processed_pdfs,
    preprocess_pdfs,
    resolve_processed_dir,
)


# ────────────────────────── Settings ──────────────────────────

DATA_DIR = Path(os.getenv("RLM_DATA_DIR", "data"))
PROCESSED_DIR = resolve_processed_dir(DATA_DIR)

HELP = """
Commands:
  /help       show this help
  /docs       list documents
  /stats      session token usage
  /history    recent chat history
  /save [f]   save session (default: session.json)
  /load [f]   load session (default: session.json)
  /clear      clear history
  /quit       exit
"""


# ────────────────────────── Preprocessing ──────────────────────────


# ────────────────────────── Session ──────────────────────────

def save_session(history: List[ChatMessage], path: str) -> None:
    Path(path).write_text(json.dumps([{"role": m.role, "content": m.content} for m in history], indent=2), encoding="utf-8")


def load_session(path: str) -> List[ChatMessage]:
    p = Path(path)
    if not p.exists():
        return []
    return [ChatMessage(role=d["role"], content=d["content"]) for d in json.loads(p.read_text(encoding="utf-8"))]


# ────────────────────────── Display ──────────────────────────

ICONS = {"iteration": "🔄", "peek": "🔍", "search": "🔎", "llm_query": "🧠", "code": "💻", "output": "📤", "answer": "✅", "warning": "⚠️"}


def print_event(event_type: str, detail: str) -> None:
    icon = ICONS.get(event_type, "•")
    if event_type == "code":
        print(f"  {icon} code:")
        for line in detail.strip().strip("`").strip().splitlines():
            print(f"     {line}")
    elif event_type == "output":
        # show the full output so you can see the REPL state
        lines = detail.strip().splitlines()
        if len(lines) == 1:
            print(f"  {icon} → {lines[0]}")
        else:
            print(f"  {icon} output:")
            for line in lines[:20]:  # cap at 20 lines
                print(f"     {line}")
            if len(lines) > 20:
                print(f"     ... ({len(lines) - 20} more lines)")
    elif event_type == "iteration":
        print(f"\n  {icon} {detail}")
    else:
        print(f"  {icon} {event_type}: {detail}")


def print_token_table(result, session: TokenUsage) -> None:
    """Print per-iteration breakdown + totals."""
    w = 58
    sep = "─" * w
    print(f"\n  {sep}")
    print(f"  {'step':>5}  {'action':<16}  {'input':>8}  {'output':>8}  {'total':>8}  {'sub':>3}  {'ms':>6}")
    print(f"  {sep}")
    for s in result.iteration_stats:
        print(f"  {s.iteration:>5}  {s.action:<16}  {s.input_tokens:>8,}  {s.output_tokens:>8,}  {s.total_tokens:>8,}  {s.sub_calls:>3}  {int(getattr(s, 'iteration_ms', 0)):>6}")
    print(f"  {sep}")
    t = result.usage
    turn_ms = int(sum(getattr(s, "iteration_ms", 0.0) for s in result.iteration_stats))
    print(f"  {'turn':>5}  {'':16}  {t.input_tokens:>8,}  {t.output_tokens:>8,}  {t.total_tokens:>8,}  {'':>3}  {turn_ms:>6}")
    print(f"  {'total':>5}  {'':16}  {session.input_tokens:>8,}  {session.output_tokens:>8,}  {session.total_tokens:>8,}")
    print(f"  {sep}\n")


# ────────────────────────── Main ──────────────────────────

def main() -> None:
    cfg = Config.from_env()

    # validate API key
    if cfg.llm_provider in ("openai", "anthropic") and (not cfg.llm_api_key or "REPLACE" in cfg.llm_api_key):
        key = "OPENAI_API_KEY" if cfg.llm_provider == "openai" else "ANTHROPIC_API_KEY"
        print(f"Error: {key} not set. Edit .env file.")
        return

    print(f"Provider: {cfg.llm_provider} / {cfg.llm_model}")

    # preprocess only missing PDFs
    if DATA_DIR.is_dir():
        all_pdfs = list_pdf_files(DATA_DIR)
        if all_pdfs:
            missing = missing_processed_pdfs(all_pdfs, DATA_DIR, PROCESSED_DIR)
            if missing:
                print(
                    f"Preprocessing PDFs from {DATA_DIR}/ (recursive): "
                    f"{len(missing)}/{len(all_pdfs)} missing in {PROCESSED_DIR}/ ..."
                )
                def _on_progress(evt: str, detail: str) -> None:
                    if evt == "file_start":
                        print(f"  extracting {detail} ...")
                    elif evt == "file_done":
                        print(f"    → {detail}")
                    elif evt == "file_error":
                        print(f"    ⚠️ skipped: {detail}")

                preprocess_pdfs(missing, data_dir=DATA_DIR, processed_dir=PROCESSED_DIR, on_progress=_on_progress)
                print()
            else:
                print(
                    f"Processed data check: {len(all_pdfs)}/{len(all_pdfs)} PDFs already extracted in {PROCESSED_DIR}/."
                )

    doc_map = discover_docs(DATA_DIR, PROCESSED_DIR)
    if not doc_map:
        print(f"No supported documents found in {DATA_DIR}/")
        return

    print(f"RLM Chat (type /help). {len(doc_map)} docs ready.\n")
    for did in doc_map:
        print(f"  • {did}")
    print()

    history: List[ChatMessage] = []
    session_usage = TokenUsage()

    while True:
        try:
            user_in = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return

        if not user_in:
            continue

        # commands
        if user_in.startswith("/"):
            parts = user_in.split(maxsplit=1)
            cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
            if cmd == "/help":
                print(HELP)
            elif cmd == "/docs":
                for did, path in doc_map.items():
                    print(f"  {did}  ({path})")
            elif cmd == "/stats":
                print(f"  provider: {cfg.llm_provider} / {cfg.llm_model}")
                print(f"  tokens:   {session_usage.input_tokens:,} in / {session_usage.output_tokens:,} out / {session_usage.total_tokens:,} total")
            elif cmd == "/history":
                for m in history[-10:]:
                    print(f"  {m.role}> {m.content[:200]}")
            elif cmd == "/clear":
                history = []
                print("Cleared.")
            elif cmd == "/save":
                p = arg or "session.json"
                save_session(history, p)
                print(f"Saved: {p}")
            elif cmd == "/load":
                p = arg or "session.json"
                history = load_session(p)
                print(f"Loaded: {p} ({len(history)} messages)")
            elif cmd == "/quit":
                print("bye.")
                return
            else:
                print("Unknown command. /help")
            continue

        # chat turn
        history.append(ChatMessage(role="user", content=user_in))
        print()

        result = answer_question(
            doc_map=doc_map,
            question=user_in,
            history=history,
            cfg=cfg,
            on_event=print_event,
        )
        session_usage += result.usage

        from rich.console import Console
        from rich.markdown import Markdown as RichMarkdown
        console = Console()
        print("\nassistant>")
        console.print(RichMarkdown(result.answer))
        print()
        print(f"  ({result.iterations} iterations)")
        print_token_table(result, session_usage)

        # cost comparison
        total_doc_chars = sum(
            f.stat().st_size
            for path in doc_map.values()
            for f in (Path(path).glob("page_*/text.txt") if Path(path).is_dir() else [])
        )
        est_trad = total_doc_chars // 4
        t = result.usage
        savings = max(0, 100 - int(t.input_tokens / max(1, est_trad) * 100))
        print(f"  💡 Why RLM matters (Paper §4):")
        print(f"     Traditional (all docs in prompt): ~{est_trad:,} input tokens + context rot")
        print(f"     RLM (targeted exploration):        {t.input_tokens:,} input tokens, {savings}% less, no context rot")
        print()

        history.append(ChatMessage(role="assistant", content=result.answer))


if __name__ == "__main__":
    main()
