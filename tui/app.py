# rlm_tui.py — Educational RLM Terminal UI
#
# Visualizes the Recursive Language Model process in real-time.
# Left: clean chat. Right: inspector showing code, REPL output, tokens, and paper concepts.
#
# Run: uv run python rlm_tui.py

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Input, RichLog, Label
from textual.binding import Binding
from textual.suggester import SuggestFromList
from textual import work
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from rlm_core import Config, ChatMessage, TokenUsage, IterationStats, estimate_cost, enabled_approaches_from_env
from rlm_docs import discover_docs, list_pdf_files, missing_processed_pdfs, preprocess_pdfs
from rlm_eval import load_questions, run_eval, summarize
from rlm_eval import (
    extract_cited_docs,
    lexical_overlap,
    semantic_similarity_llm,
    format_evidence_manifest,
    semantic_backend_from_env,
    tfidf_cosine_similarity,
    extract_evidence_docs,
)
from approaches.rag import (
    ensure_rag_ready,
    get_rag_collection_status,
    reset_rag_collection,
    RagConfig,
)
from approaches import get_approach
from .constants import (
    DATA_DIR,
    PROCESSED_DIR,
    CONCEPT_NOTES,
    INSPECTOR_SIZES as DEFAULT_INSPECTOR_SIZES,
    SLASH_COMMANDS as DEFAULT_SLASH_COMMANDS,
    INPUT_SUGGESTIONS,
)
from .styles import APP_CSS


# ────────────────────────── TUI App ──────────────────────────

class RLMApp(App):
    """Educational RLM Terminal UI."""

    CSS = APP_CSS

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Clear / Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+right_square_bracket", "inspector_wider", "Inspector +"),
        Binding("ctrl+left_square_bracket", "inspector_narrower", "Inspector -"),
        Binding("ctrl+h", "toggle_inspector", "Toggle Inspector"),
    ]

    TITLE = "RLM Explorer"
    SUB_TITLE = "Recursive Language Models — Educational Demo  |  ctrl+] wider  ctrl+[ narrower  ctrl+h toggle"

    # inspector width presets
    INSPECTOR_SIZES = DEFAULT_INSPECTOR_SIZES
    SLASH_COMMANDS = DEFAULT_SLASH_COMMANDS

    def __init__(self, startup_query: str = ""):
        super().__init__()
        self.cfg = Config.from_env()
        self.enabled_approaches = enabled_approaches_from_env()
        self._inspector_idx = 2  # start at 70
        self.auto_eval_after_chat = True
        # state for /copy — tracks the last turn's data
        self._last_result = None          # AnswerResult
        self._last_question: str = ""
        self._inspector_events: List[dict] = []  # [{type, detail}, ...]
        self.doc_map: Dict[str, str] = {}
        self.history: List[ChatMessage] = []
        self.session_usage = TokenUsage()
        self.current_stats: List[IterationStats] = []
        self._last_ctrl_c_ts: float = 0.0
        self.rag_ready: bool = False
        self.rag_last_error: str = ""
        self.startup_query: str = startup_query.strip()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="chat-panel"):
                yield Label("💬 Chat", id="chat-title")
                yield RichLog(id="chat-log", wrap=True, highlight=True, markup=True)
                yield Static("", id="slash-popup")
                yield Input(
                    placeholder="Ask a question about your documents...",
                    id="chat-input",
                    suggester=SuggestFromList(INPUT_SUGGESTIONS, case_sensitive=False),
                )
            with Vertical(id="inspector-panel"):
                yield Label("🔬 Inspector", id="inspector-title")
                yield RichLog(id="inspector-log", wrap=True, highlight=True, markup=True, auto_scroll=True)
                with Vertical(id="bottom-info"):
                    yield Static("", id="token-panel")
                    yield Static("", id="concept-panel")
                    yield Static("", id="docs-label")
        yield Footer()

    def on_mount(self) -> None:
        # set initial inspector width
        self._apply_inspector_size()

        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text("Welcome to RLM Explorer!", style="bold"))
        chat.write(Text(f"Provider: {self.cfg.llm_provider} / {self.cfg.llm_model}", style="dim"))
        chat.write("")

        # show concept intro
        self.query_one("#concept-panel", Static).update(
            Panel(
                "[bold]RLM Concept[/]\nThe LLM writes code to explore documents stored externally.\n"
                "It never sees the full text — only what it requests.\n"
                "Ask a question to see it in action!",
                border_style="yellow",
            )
        )

        self.query_one("#inspector-log", RichLog).write(Text("Initializing...", style="dim"))

        # kick off doc loading in background so we can show progress
        self._init_docs()

    @work(thread=True)
    def _init_docs(self) -> None:
        """Load/preprocess documents with progress shown in chat."""
        chat = self.query_one("#chat-log", RichLog)
        inspector = self.query_one("#inspector-log", RichLog)

        # process only PDFs missing from processed_data/
        if DATA_DIR.is_dir():
            pdfs = list_pdf_files(DATA_DIR)
            if pdfs:
                src = DATA_DIR
                dst = PROCESSED_DIR
                dst.mkdir(parents=True, exist_ok=True)
                missing = missing_processed_pdfs(pdfs, src, dst)
                if missing:
                    self.call_from_thread(
                        chat.write,
                        Text(
                            f"📂 Found {len(pdfs)} PDFs in {DATA_DIR}/ (recursive) — "
                            f"{len(missing)}/{len(pdfs)} missing in {PROCESSED_DIR}/, processing now...",
                            style="bold yellow",
                        ),
                    )
                    idx_map = {p: i + 1 for i, p in enumerate(missing)}

                    def _on_progress(evt: str, detail: str) -> None:
                        if evt == "file_start":
                            try:
                                rel = Path(detail)
                                num = idx_map.get(src / rel, 0)
                                self.call_from_thread(chat.write, Text(f"  📄 [{num}/{len(missing)}] {detail}...", style="dim"))
                            except Exception:
                                self.call_from_thread(chat.write, Text(f"  📄 {detail}...", style="dim"))
                        elif evt == "file_done":
                            self.call_from_thread(chat.write, Text(f"     ✓ {detail}", style="dim green"))
                        elif evt == "file_error":
                            self.call_from_thread(chat.write, Text(f"     ⚠️ skipped: {detail}", style="dim red"))

                    preprocess_pdfs(pdfs=missing, data_dir=src, processed_dir=dst, on_progress=_on_progress)

                    self.call_from_thread(chat.write, Text(""))
                    self.call_from_thread(chat.write, Text("✅ Missing documents processed.", style="bold green"))
                    self.call_from_thread(chat.write, Text(""))
                else:
                    self.call_from_thread(
                        chat.write,
                        Text(
                            f"📂 Found {len(pdfs)} PDFs in {DATA_DIR}/ (recursive) — all already processed.",
                            style="dim green",
                        ),
                    )
                    self.call_from_thread(chat.write, Text(""))

        # discover docs
        self.doc_map = discover_docs(DATA_DIR, PROCESSED_DIR)

        if not self.doc_map:
            self.call_from_thread(chat.write, Text(f"No documents found. Add PDFs to {DATA_DIR}/", style="bold red"))
            return

        # show docs (cap at 10 in chat, full list via /docs)
        doc_ids = list(self.doc_map.keys())
        self.call_from_thread(chat.write, Text(f"📄 {len(doc_ids)} documents ready:", style="bold"))
        for did in doc_ids[:10]:
            self.call_from_thread(chat.write, Text(f"  • {did}", style="dim"))
        if len(doc_ids) > 10:
            self.call_from_thread(chat.write, Text(f"  ... and {len(doc_ids) - 10} more (type /docs to see all)", style="dim"))
        self.call_from_thread(chat.write, Text(""))
        self.call_from_thread(chat.write, Text(f"Context window: {self.cfg.context_window:,} tokens ({self.cfg.llm_model})", style="dim"))
        self.call_from_thread(chat.write, Text("Ask a question to see RLM in action! Type /help for commands.", style="dim"))
        self.call_from_thread(chat.write, Text(""))

        docs_text = f"📄 {len(self.doc_map)} docs"
        for did in self.doc_map:
            docs_text += f"\n  • {did}"
        self.call_from_thread(self.query_one("#docs-label", Static).update, docs_text)

        # If RAG is enabled, ensure collection/index is ready at startup.
        if "rag" in self.enabled_approaches:
            self.call_from_thread(chat.write, Text("📚 Initializing RAG collection/index...", style="bold magenta"))
            self.call_from_thread(inspector.clear)
            self.call_from_thread(inspector.write, Text("📚 RAG Initialization", style="bold"))
            self.call_from_thread(inspector.write, Text(""))

            def _on_rag_init(evt: str, detail: str) -> None:
                if evt == "rag_stage":
                    self.call_from_thread(inspector.write, Text(f"  • {detail}", style="dim"))

            try:
                ensure_rag_ready(self.doc_map, on_event=_on_rag_init)
                self.rag_ready = True
                self.rag_last_error = ""
                self.call_from_thread(chat.write, Text("✅ RAG is ready.", style="green"))
            except Exception as e:
                # Keep app usable: disable rag for this session if init fails.
                self.rag_ready = False
                self.rag_last_error = str(e)
                self.enabled_approaches = [a for a in self.enabled_approaches if a != "rag"]
                self.call_from_thread(chat.write, Text(f"⚠️ RAG init failed; disabled for this session: {e}", style="yellow"))
            self.call_from_thread(chat.write, Text(""))

        self.call_from_thread(inspector.clear)
        self.call_from_thread(inspector.write, Text("Waiting for a question...", style="dim"))
        self.call_from_thread(self.query_one("#chat-input", Input).focus)

        # Optional startup query (from launcher flag) runs once after init.
        if self.startup_query:
            q = self.startup_query
            self.startup_query = ""
            self.call_from_thread(chat.write, Text(f"startup> {q}", style="bold cyan"))
            self.history.append(ChatMessage(role="user", content=q))
            self.call_from_thread(inspector.clear)
            self.current_stats = []
            self.call_from_thread(self.run_rlm, q)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_in = event.value.strip()
        if not user_in:
            return

        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""
        self._update_slash_popup("")

        chat = self.query_one("#chat-log", RichLog)
        inspector = self.query_one("#inspector-log", RichLog)

        # handle commands
        if user_in.startswith("/"):
            cmd = user_in.split()[0].lower()
            if cmd in ("/quit", "/exit"):
                self.exit()
                return
            elif cmd == "/clear":
                self.history = []
                chat.clear()
                inspector.clear()
                chat.write(Text("Chat cleared.", style="dim"))
                return
            elif cmd == "/help":
                chat.write(Text("Commands:", style="bold"))
                chat.write(Text("  /docs         — list all documents", style="dim"))
                chat.write(Text("  /test         — run query from RLM_TEST_QUERY in .env", style="dim"))
                chat.write(Text("  /config [N]   — show/set max iterations (0 = until done, e.g. /config 15)", style="dim"))
                chat.write(Text("  /traditional  — show last traditional run stats (context window, truncation)", style="dim"))
                chat.write(Text("  /rag status|retry|reset|topk N|mode semantic|hybrid|alpha X", style="dim"))
                chat.write(Text("  /eval [file]  — run benchmark in chat (optional questions .txt path)", style="dim"))
                chat.write(Text("  /autoeval [on|off|status] — run post-turn semantic eval after each chat", style="dim"))
                chat.write(Text("  /copy         — copy last turn as JSON to clipboard", style="dim"))
                chat.write(Text("  /clear        — clear chat and inspector", style="dim"))
                chat.write(Text("  /quit         — exit", style="dim"))
                chat.write(Text("  ctrl+] / [    — inspector wider / narrower", style="dim"))
                chat.write(Text("  ctrl+h        — toggle inspector", style="dim"))
                return
            elif cmd == "/docs":
                chat.write(Text(f"📄 {len(self.doc_map)} documents:", style="bold"))
                for did, path in self.doc_map.items():
                    chat.write(Text(f"  • {did}  ({path})", style="dim"))
                return
            elif cmd == "/test":
                test_query = os.getenv("RLM_TEST_QUERY", "").strip()
                if not test_query:
                    chat.write(Text("Set RLM_TEST_QUERY in .env, then run /test.", style="dim red"))
                    return
                chat.write(Text(f"🧪 test> {test_query}", style="bold cyan"))
                self.history.append(ChatMessage(role="user", content=test_query))
                inspector.clear()
                self.current_stats = []
                self.run_rlm(test_query)
                return
            elif cmd == "/config":
                parts = user_in.split()
                if len(parts) > 1:
                    try:
                        new_val = int(parts[1])
                        if new_val < 0:
                            raise ValueError
                        self.cfg.max_iterations = new_val
                        if new_val == 0:
                            chat.write(Text(f"✅ max_iterations set to 0 (until done mode, hard cap {self.cfg.hard_max_iterations})", style="green"))
                        else:
                            chat.write(Text(f"✅ max_iterations set to {new_val}", style="green"))
                    except ValueError:
                        chat.write(Text(f"Usage: /config <number>=0+  (0=until done, e.g. /config 15)", style="dim red"))
                else:
                    chat.write(Text(f"  provider:       {self.cfg.llm_provider} / {self.cfg.llm_model}", style="dim"))
                    chat.write(Text(f"  approaches:     {', '.join(self.enabled_approaches)}", style="dim"))
                    chat.write(Text(f"  rag status:     {'ready' if self.rag_ready else 'not ready'}", style="dim"))
                    chat.write(Text(f"  context window: {self.cfg.context_window:,} tokens", style="dim"))
                    max_iters_label = (
                        f"until done (hard cap {self.cfg.hard_max_iterations})"
                        if self.cfg.max_iterations == 0 else str(self.cfg.max_iterations)
                    )
                    chat.write(Text(f"  max iterations: {max_iters_label}", style="dim"))
                    chat.write(Text(f"  session tokens: {self.session_usage.total_tokens:,} (${estimate_cost(self.session_usage, self.cfg.llm_model):.4f})", style="dim"))
                return
            elif cmd == "/rag":
                parts = user_in.split()
                arg = parts[1].strip().lower() if len(parts) > 1 else "status"
                if arg in ("retry", "reconnect", "init"):
                    if not self.doc_map:
                        chat.write(Text("No documents loaded yet; cannot initialize RAG.", style="dim red"))
                        return
                    chat.write(Text("📚 Retrying RAG initialization...", style="bold magenta"))
                    self.run_rag_init(manual=True)
                elif arg == "reset":
                    if not self.doc_map:
                        chat.write(Text("No documents loaded yet; cannot reset RAG.", style="dim red"))
                        return
                    chat.write(Text("♻️ Resetting RAG collection (cache-aware rebootstrap)...", style="bold magenta"))
                    self.run_rag_init(manual=True, reset=True)
                elif arg == "topk":
                    if len(parts) < 3:
                        chat.write(Text("Usage: /rag topk <int>", style="dim"))
                        return
                    try:
                        v = max(1, int(parts[2]))
                    except ValueError:
                        chat.write(Text("Usage: /rag topk <int>", style="dim red"))
                        return
                    os.environ["RAG_TOP_K"] = str(v)
                    chat.write(Text(f"✅ RAG_TOP_K set to {v} (applies next query)", style="green"))
                elif arg == "mode":
                    if len(parts) < 3:
                        chat.write(Text("Usage: /rag mode semantic|hybrid", style="dim"))
                        return
                    mode = parts[2].strip().lower()
                    if mode not in {"semantic", "hybrid"}:
                        chat.write(Text("Usage: /rag mode semantic|hybrid", style="dim red"))
                        return
                    os.environ["RAG_RETRIEVAL_MODE"] = mode
                    chat.write(Text(f"✅ RAG_RETRIEVAL_MODE set to {mode} (applies next query)", style="green"))
                elif arg == "alpha":
                    if len(parts) < 3:
                        chat.write(Text("Usage: /rag alpha <0.0-1.0>", style="dim"))
                        return
                    try:
                        a = float(parts[2])
                    except ValueError:
                        chat.write(Text("Usage: /rag alpha <0.0-1.0>", style="dim red"))
                        return
                    a = max(0.0, min(1.0, a))
                    os.environ["RAG_HYBRID_ALPHA"] = f"{a:.3f}"
                    chat.write(Text(f"✅ RAG_HYBRID_ALPHA set to {a:.3f} (applies next query)", style="green"))
                else:
                    rag_cfg = RagConfig.from_env()
                    chat.write(Text(f"RAG enabled in this session: {'yes' if 'rag' in self.enabled_approaches else 'no'}", style="dim"))
                    chat.write(Text(f"RAG ready: {'yes' if self.rag_ready else 'no'}", style="dim"))
                    chat.write(Text(f"RAG retrieval mode: {rag_cfg.retrieval_mode}", style="dim"))
                    chat.write(Text(f"RAG top_k: {rag_cfg.top_k}", style="dim"))
                    if rag_cfg.retrieval_mode == "hybrid":
                        chat.write(Text(f"RAG hybrid alpha: {rag_cfg.hybrid_alpha:.2f}", style="dim"))
                    if self.rag_last_error:
                        chat.write(Text(f"Last RAG error: {self.rag_last_error}", style="dim red"))
                    self.run_rag_status()
                    chat.write(Text("Usage: /rag status|retry|reset|topk N|mode semantic|hybrid|alpha X", style="dim"))
                return
            elif cmd == "/traditional":
                if not self._last_result or not self._last_result.traditional_stats:
                    chat.write(Text("No traditional run yet — ask a question first.", style="dim red"))
                else:
                    ts = self._last_result.traditional_stats
                    fill = int(ts.context_used_pct / 5)
                    bar = "█" * fill + "░" * (20 - fill)
                    chat.write(Text("🐌 Last Traditional Run Stats:", style="bold"))
                    chat.write(Text(""))
                    chat.write(Text(f"  Context window: {ts.context_window:,} tokens ({self.cfg.llm_model})", style="dim"))
                    chat.write(Text(f"  [{bar}] {ts.context_used_pct:.0f}% full", style="yellow"))
                    chat.write(Text(""))
                    chat.write(Text(f"  Total corpus:   {ts.total_chars:,} chars across {ts.total_docs} docs", style="dim"))
                    chat.write(Text(f"  Actually sent:  {ts.chars_sent:,} chars ({ts.docs_included} docs fit)", style="dim"))
                    if ts.truncated:
                        lost_pct = int((1 - ts.chars_sent / max(1, ts.total_chars)) * 100)
                        chat.write(Text(f"  ⚠️ Truncated — {lost_pct}% of content lost", style="red"))
                        chat.write(Text(""))
                        chat.write(Text(f"  ✅ Included ({ts.docs_included}):", style="green"))
                        included = [d for d in self.doc_map if d not in ts.docs_excluded]
                        for did in included:
                            chat.write(Text(f"     • {did}", style="dim green"))
                        chat.write(Text(f"  ❌ Excluded ({len(ts.docs_excluded)}):", style="red"))
                        for did in ts.docs_excluded:
                            chat.write(Text(f"     • {did}", style="dim red"))
                    else:
                        chat.write(Text(f"  ✅ All {ts.total_docs} docs fit — no truncation", style="green"))
                return
            elif cmd == "/copy":
                self._handle_copy(chat)
                return
            elif cmd == "/eval":
                parts = user_in.split(maxsplit=1)
                qfile = parts[1].strip() if len(parts) > 1 else "eval_questions.txt"
                chat.write(Text(f"🧪 Running eval benchmark (questions: {qfile})...", style="bold yellow"))
                chat.write(Text("   Streaming traditional + RLM traces below.", style="dim"))
                self.run_eval_benchmark(qfile)
                return
            elif cmd == "/autoeval":
                parts = user_in.split(maxsplit=1)
                arg = parts[1].strip().lower() if len(parts) > 1 else "status"
                if arg in ("on", "true", "1"):
                    self.auto_eval_after_chat = True
                    chat.write(Text("✅ auto-eval is ON (post-turn semantic eval enabled)", style="green"))
                elif arg in ("off", "false", "0"):
                    self.auto_eval_after_chat = False
                    chat.write(Text("✅ auto-eval is OFF", style="yellow"))
                else:
                    state = "ON" if self.auto_eval_after_chat else "OFF"
                    chat.write(Text(f"auto-eval status: {state}", style="dim"))
                    chat.write(Text("Usage: /autoeval on|off|status", style="dim"))
                return
            else:
                chat.write(Text(f"Unknown command: {cmd}. Type /help", style="dim red"))
                return

        # handle bare exit/quit without slash
        if user_in.lower() in ("exit", "quit"):
            self.exit()
            return

        # show user message
        chat.write(Text(f"you> {user_in}", style="bold cyan"))
        self.history.append(ChatMessage(role="user", content=user_in))

        # clear inspector for new turn
        inspector.clear()
        self.current_stats = []

        # run RLM in background
        self.run_rlm(user_in)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Show slash-command popup as the user types, similar to Slack."""
        if event.input.id != "chat-input":
            return
        self._update_slash_popup(event.value)

    def _update_slash_popup(self, text: str) -> None:
        popup = self.query_one("#slash-popup", Static)
        q = (text or "").strip()
        if not q.startswith("/"):
            popup.display = False
            popup.update("")
            return

        token = q.split()[0].lower()
        matches = [cmd for cmd in self.SLASH_COMMANDS if cmd[0].startswith(token)]
        if not matches:
            popup.display = False
            popup.update("")
            return

        lines = ["[bold]Slash commands[/]"]
        for cmd, desc in matches[:8]:
            lines.append(f"  {cmd:<20} [dim]- {desc}[/]")
        popup.update("\n".join(lines))
        popup.display = True

    @work(thread=True)
    def run_rag_init(self, manual: bool = False, reset: bool = False) -> None:
        """Retry/check or reset RAG readiness from command without restarting app."""
        chat = self.query_one("#chat-log", RichLog)
        inspector = self.query_one("#inspector-log", RichLog)

        self.call_from_thread(inspector.clear)
        self.call_from_thread(inspector.write, Text("📚 RAG Initialization", style="bold"))
        self.call_from_thread(inspector.write, Text(""))

        def _on_rag_evt(evt: str, detail: str) -> None:
            if evt == "rag_stage":
                self.call_from_thread(inspector.write, Text(f"  • {detail}", style="dim"))

        try:
            if reset:
                status = reset_rag_collection(self.doc_map, on_event=_on_rag_evt)
            else:
                ensure_rag_ready(self.doc_map, on_event=_on_rag_evt)
                status = get_rag_collection_status(on_event=_on_rag_evt)
            self.rag_ready = True
            self.rag_last_error = ""
            if "rag" not in self.enabled_approaches:
                self.enabled_approaches.append("rag")
            self.call_from_thread(
                chat.write,
                Text(
                    f"✅ RAG is ready. Collection '{status.collection_name}': "
                    f"{status.unique_docs} docs, {status.chunk_objects} chunks.",
                    style="green",
                ),
            )
        except Exception as e:
            self.rag_ready = False
            self.rag_last_error = str(e)
            if manual:
                self.call_from_thread(chat.write, Text(f"⚠️ RAG retry failed: {e}", style="yellow"))

    @work(thread=True)
    def run_rag_status(self) -> None:
        """Fetch and print live RAG collection stats."""
        chat = self.query_one("#chat-log", RichLog)
        inspector = self.query_one("#inspector-log", RichLog)

        def _on_rag_evt(evt: str, detail: str) -> None:
            if evt == "rag_stage":
                self.call_from_thread(inspector.write, Text(f"  • {detail}", style="dim"))

        try:
            status = get_rag_collection_status(on_event=_on_rag_evt)
            self.rag_ready = status.exists and status.chunk_objects > 0
            self.call_from_thread(
                chat.write,
                Text(
                    f"RAG collection '{status.collection_name}': "
                    f"{status.unique_docs} docs, {status.chunk_objects} chunks.",
                    style="dim",
                ),
            )
        except Exception as e:
            self.rag_ready = False
            self.rag_last_error = str(e)
            self.call_from_thread(chat.write, Text(f"⚠️ RAG status check failed: {e}", style="dim red"))

    @work(thread=True)
    def run_rlm(self, question: str) -> None:
        """Run the RLM agent loop in a background thread."""
        chat = self.query_one("#chat-log", RichLog)
        inspector = self.query_one("#inspector-log", RichLog)

        self._live_iterations: List[tuple] = []
        self._inspector_events = []  # reset for this turn
        self._live_step = 0

        def _update_live_tokens(step: int, usage: TokenUsage) -> None:
            """Update the token panel with a running total."""
            t = Table(show_header=True, header_style="bold", border_style="cyan", expand=True, title="⏱ Live Tokens")
            t.add_column("", width=10)
            t.add_column("Input", justify="right", width=8)
            t.add_column("Output", justify="right", width=8)
            t.add_column("Total", justify="right", width=8)
            t.add_row(f"step {step}", f"{usage.input_tokens:,}", f"{usage.output_tokens:,}", f"{usage.total_tokens:,}", style="bold")
            sess = self.session_usage
            t.add_row("session", f"{sess.input_tokens + usage.input_tokens:,}", f"{sess.output_tokens + usage.output_tokens:,}", f"{sess.total_tokens + usage.total_tokens:,}", style="dim")
            self.call_from_thread(self.query_one("#token-panel", Static).update, t)

        def on_event(event_type: str, detail: str) -> None:
            """Callback from the sandbox — update the inspector panel."""
            # save event for /copy
            self._inspector_events.append({"type": event_type, "detail": detail})

            # track iterations for live token display
            if event_type == "iteration":
                self._live_iterations.append(detail)
                self._live_step += 1

            # update concept note
            if event_type in CONCEPT_NOTES:
                self.call_from_thread(
                    self.query_one("#concept-panel", Static).update,
                    Panel(CONCEPT_NOTES[event_type], border_style="yellow", title="📚 Paper Concept")
                )

            if event_type == "iteration":
                self.call_from_thread(inspector.write, Text(f"\n{'─'*40}", style="dim"))
                self.call_from_thread(inspector.write, Text(f"🔄 {detail}", style="bold white"))

            elif event_type == "code":
                code_text = detail.strip().strip("`").strip()
                self.call_from_thread(inspector.write, Text("💻 Code:", style="bold magenta"))
                for line in code_text.splitlines():
                    self.call_from_thread(inspector.write, Text(f"  {line}", style="bright_green"))

            elif event_type == "output":
                lines = detail.strip().splitlines()
                self.call_from_thread(inspector.write, Text("📤 Output:", style="bold blue"))
                for line in lines[:15]:
                    self.call_from_thread(inspector.write, Text(f"  {line}", style="white"))
                if len(lines) > 15:
                    self.call_from_thread(inspector.write, Text(f"  ... ({len(lines)-15} more)", style="dim"))

            elif event_type == "peek":
                self.call_from_thread(inspector.write, Text(f"  🔍 peek: {detail}", style="yellow"))

            elif event_type == "search":
                self.call_from_thread(inspector.write, Text(f"  🔎 search: {detail}", style="cyan"))

            elif event_type == "llm_query":
                self.call_from_thread(inspector.write, Text(f"  🧠 sub-call: {detail[:80]}...", style="bright_magenta"))

            elif event_type == "answer":
                self.call_from_thread(inspector.write, Text(f"  ✅ {detail}", style="bold green"))

            elif event_type == "warning":
                self.call_from_thread(inspector.write, Text(f"  ⚠️ {detail}", style="bold red"))

        # wrap on_event to update live tokens
        def on_event_with_tokens(event_type: str, detail: str) -> None:
            on_event(event_type, detail)
            if event_type == "tokens":
                # detail is "input,output,total"
                parts = detail.split(",")
                live_usage = TokenUsage(int(parts[0]), int(parts[1]), int(parts[2]))
                _update_live_tokens(self._live_step, live_usage)

        # --- Run enabled approaches and compare ---
        enabled = self.enabled_approaches
        is_rlm_run = False
        result = None
        trad_result = None
        baseline_for_eval = None
        baseline_name_for_eval = "baseline"
        candidate_for_eval = None
        candidate_name_for_eval = "candidate"
        baseline_results = []

        if "traditional" in enabled:
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(chat.write, Text("🐌 Traditional (all docs in one prompt):", style="bold dim"))
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(inspector.clear)
            self.call_from_thread(inspector.write, Text("🐌 Traditional Approach", style="bold"))
            self.call_from_thread(inspector.write, Text(""))
            self.call_from_thread(inspector.write, Text("⏳ Starting...", style="dim"))

            def on_traditional_event(event_type: str, detail: str) -> None:
                if event_type == "traditional_stage":
                    self.call_from_thread(inspector.write, Text(f"  • {detail}", style="dim"))

            trad_exec = get_approach("traditional").run(
                doc_map=self.doc_map,
                question=question,
                history=self.history,
                cfg=self.cfg,
                on_event=on_traditional_event,
            )
            trad_result = trad_exec.answer
            trad_t = trad_result.usage
            baseline_results.append(("Traditional", trad_result, "dim red"))
            baseline_for_eval = baseline_for_eval or trad_result
            if baseline_for_eval is trad_result:
                baseline_name_for_eval = "Traditional"
            ts = trad_result.traditional_stats

            self.call_from_thread(inspector.clear)
            self.call_from_thread(inspector.write, Text("🐌 Traditional Approach", style="bold"))
            self.call_from_thread(inspector.write, Text(""))
            if ts:
                fill = int(ts.context_used_pct / 5)
                bar = "█" * fill + "░" * (20 - fill)
                fill_color = "green" if ts.context_used_pct < 50 else "yellow" if ts.context_used_pct < 90 else "bold red"
                self.call_from_thread(inspector.write, Text(f"  Context window: {ts.context_window:,} tokens", style="dim"))
                self.call_from_thread(inspector.write, Text(f"  [{bar}] {ts.context_used_pct:.0f}% full", style=fill_color))
                self.call_from_thread(inspector.write, Text(""))
                self.call_from_thread(inspector.write, Text(f"  Corpus: {ts.total_chars:,} chars ({ts.total_docs} docs)", style="dim"))
                self.call_from_thread(inspector.write, Text(f"  Sent:   {ts.chars_sent:,} chars ({ts.docs_included} docs fit)", style="dim"))

            self.call_from_thread(chat.write, Markdown(trad_result.answer))
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(chat.write, Text(f"  ({trad_t.input_tokens:,} in / {trad_t.output_tokens:,} out / {trad_t.total_tokens:,} tokens  ${estimate_cost(trad_t, self.cfg.llm_model):.4f})", style="dim red"))
            self.call_from_thread(chat.write, Text(""))

        if "rag" in enabled:
            self.call_from_thread(chat.write, Text("─" * 50, style="dim"))
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(chat.write, Text("📚 RAG (vector retrieval baseline):", style="bold magenta"))
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(inspector.clear)
            self.call_from_thread(inspector.write, Text("📚 RAG Approach", style="bold"))
            self.call_from_thread(inspector.write, Text(""))

            def on_rag_event(event_type: str, detail: str) -> None:
                if event_type == "rag_stage":
                    self.call_from_thread(inspector.write, Text(f"  • {detail}", style="dim"))

            try:
                rag_exec = get_approach("rag").run(
                    doc_map=self.doc_map,
                    question=question,
                    history=self.history,
                    cfg=self.cfg,
                    on_event=on_rag_event,
                )
                rag_result = rag_exec.answer
                rag_stats = rag_exec.metadata.get("rag_stats")
            except Exception as e:
                # Never let baseline failures kill the full run.
                self.call_from_thread(inspector.write, Text(f"  ⚠️ RAG failed: {e}", style="bold red"))
                self.rag_ready = False
                self.rag_last_error = str(e)
                rag_result = None
                rag_stats = None
            if rag_result is None:
                self.call_from_thread(chat.write, Text("⚠️ RAG baseline failed; continuing with remaining approaches.", style="yellow"))
            else:
                baseline_results.append(("RAG", rag_result, "magenta"))
                baseline_for_eval = baseline_for_eval or rag_result
                if baseline_for_eval is rag_result:
                    baseline_name_for_eval = "RAG"
                self.call_from_thread(chat.write, Markdown(rag_result.answer))
                self.call_from_thread(chat.write, Text(""))
                self.call_from_thread(
                    chat.write,
                    Text(
                        f"  ({rag_result.usage.input_tokens:,} in / {rag_result.usage.output_tokens:,} out / "
                        f"{rag_result.usage.total_tokens:,} tokens  ${estimate_cost(rag_result.usage, self.cfg.llm_model):.4f})",
                        style="dim magenta",
                    ),
                )
                if isinstance(rag_stats, dict):
                    self.call_from_thread(
                        chat.write,
                        Text(
                            f"  (retrieved {int(rag_stats.get('retrieved_chunks', 0))} chunks from "
                            f"{int(rag_stats.get('unique_docs_retrieved', 0))} docs in "
                            f"{int(rag_stats.get('retrieval_ms', 0.0))}ms)",
                            style="dim",
                        ),
                    )
                self.call_from_thread(chat.write, Text(""))

        if "rlm" in enabled:
            self.call_from_thread(chat.write, Text("─" * 50, style="dim"))
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(chat.write, Text("🚀 RLM (targeted exploration):", style="bold green"))
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(inspector.write, Text("\n🚀 Starting RLM agent...", style="bold"))

            rlm_exec = get_approach("rlm").run(
                doc_map=self.doc_map,
                question=question,
                history=self.history,
                cfg=self.cfg,
                on_event=on_event_with_tokens,
            )
            result = rlm_exec.answer
            self.session_usage += result.usage
            self.history.append(ChatMessage(role="assistant", content=result.answer))
            is_rlm_run = any(s.action != "direct" for s in result.iteration_stats)
        else:
            if baseline_results:
                result = baseline_results[-1][1]
                self.session_usage += result.usage
                self.history.append(ChatMessage(role="assistant", content=result.answer))
            is_rlm_run = False

        # save state for /copy and /traditional
        self._last_question = question
        self._last_result = result
        if result is not None and trad_result is not None:
            result.traditional_stats = trad_result.traditional_stats

        if result is None:
            self.call_from_thread(chat.write, Text("No approach is enabled. Set ENABLED_APPROACHES in .env.", style="bold red"))
            return

        if "rlm" in enabled:
            # show RLM answer
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(chat.write, Markdown(result.answer))
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(chat.write, Text(""))

            # 3. Token breakdown table (RLM iterations)
            table = Table(show_header=True, header_style="bold", border_style="dim", padding=(0, 1), expand=False)
            table.add_column("#", justify="right", style="dim", width=3)
            table.add_column("Action", width=14)
            table.add_column("In", justify="right", width=7)
            table.add_column("Out", justify="right", width=7)
            table.add_column("Total", justify="right", width=7)
            table.add_column("Sub", justify="right", width=3)
            table.add_column("ms", justify="right", width=6)

            for s in result.iteration_stats:
                table.add_row(
                    str(s.iteration), s.action,
                    f"{s.input_tokens:,}", f"{s.output_tokens:,}", f"{s.total_tokens:,}",
                    str(s.sub_calls) if s.sub_calls else "—",
                    f"{int(getattr(s, 'iteration_ms', 0)):,}",
                )
            table.add_section()
            t = result.usage
            turn_ms = int(sum(getattr(s, "iteration_ms", 0.0) for s in result.iteration_stats))
            table.add_row("", "turn total", f"{t.input_tokens:,}", f"{t.output_tokens:,}", f"{t.total_tokens:,}", "", f"{turn_ms:,}", style="bold")
            su = self.session_usage
            table.add_row("", "session", f"{su.input_tokens:,}", f"{su.output_tokens:,}", f"{su.total_tokens:,}", "", "", style="dim")
            self.call_from_thread(chat.write, table)
            self.call_from_thread(chat.write, Text(""))

        # 4. Comparison and post-turn eval pair selection
        if is_rlm_run and result is not None and baseline_results:
            baseline_for_eval = baseline_results[0][1]
            baseline_name_for_eval = baseline_results[0][0]
            candidate_for_eval = result
            candidate_name_for_eval = "RLM"
        elif len(baseline_results) >= 2:
            baseline_for_eval = baseline_results[0][1]
            baseline_name_for_eval = baseline_results[0][0]
            candidate_for_eval = baseline_results[1][1]
            candidate_name_for_eval = baseline_results[1][0]

        # 4b. Comparison box with dollar costs
        if baseline_for_eval is not None and candidate_for_eval is not None:
            cand_t = candidate_for_eval.usage
            cand_cost = estimate_cost(cand_t, self.cfg.llm_model)
            ref_name = baseline_name_for_eval
            ref_res = baseline_for_eval
            ref_cost = estimate_cost(ref_res.usage, self.cfg.llm_model)

            self.call_from_thread(chat.write, Text("  ┌─ 💡 Comparison ────────────────────────────────────────────────", style="dim"))
            for name, bres, style in baseline_results:
                btok = bres.usage.total_tokens
                bcost = estimate_cost(bres.usage, self.cfg.llm_model)
                self.call_from_thread(chat.write, Text(f"  │ {name:<12}: {btok:>8,} tokens  ${bcost:.4f}", style=style))
            icon = "🚀" if candidate_name_for_eval == "RLM" else "✅"
            self.call_from_thread(chat.write, Text(f"  │ {icon} {candidate_name_for_eval:<10}: {cand_t.total_tokens:>8,} tokens  ${cand_cost:.4f}", style="green"))
            savings_pct = max(0, 100 - int(cand_t.total_tokens / max(1, ref_res.usage.total_tokens) * 100))
            if savings_pct > 0:
                saved = ref_cost - cand_cost
                self.call_from_thread(chat.write, Text(f"  │ 💰 {candidate_name_for_eval} vs {ref_name}: {savings_pct}% fewer tokens, saved ${saved:.4f}", style="bold green"))
            else:
                self.call_from_thread(chat.write, Text(f"  │ 💡 At {len(self.doc_map)} docs, costs can be similar by question type.", style="yellow"))
            self.call_from_thread(chat.write, Text("  └────────────────────────────────────────────────────────────────", style="dim"))
            self.call_from_thread(chat.write, Text(""))
            self.call_from_thread(inspector.write, Text("✅ Comparison complete", style="dim"))

        # 5. Optional post-turn eval (semantic + citation signal) right in chat
        if self.auto_eval_after_chat and baseline_for_eval is not None and candidate_for_eval is not None:
            self.call_from_thread(chat.write, Text("🧪 Post-turn eval:", style="bold yellow"))
            sem_backend = semantic_backend_from_env()
            self.call_from_thread(
                chat.write,
                Text(
                    f"  Running semantic similarity ({sem_backend}) ({baseline_name_for_eval} vs {candidate_name_for_eval})...",
                    style="dim",
                ),
            )

            baseline_cites = extract_cited_docs(baseline_for_eval.answer)
            candidate_cites = extract_cited_docs(candidate_for_eval.answer)
            baseline_ev_docs = extract_evidence_docs(baseline_for_eval.evidence_manifest)
            candidate_ev_docs = extract_evidence_docs(candidate_for_eval.evidence_manifest)
            if not baseline_ev_docs and baseline_cites:
                baseline_ev_docs = set(baseline_cites)
            if not candidate_ev_docs and candidate_cites:
                candidate_ev_docs = set(candidate_cites)
            ev_overlap = baseline_ev_docs & candidate_ev_docs
            ev_baseline_only = baseline_ev_docs - candidate_ev_docs
            ev_candidate_only = candidate_ev_docs - baseline_ev_docs
            ev_union_n = len(baseline_ev_docs | candidate_ev_docs)
            ev_jaccard = (len(ev_overlap) / ev_union_n) if ev_union_n else 0.0
            lexical = lexical_overlap(baseline_for_eval.answer, candidate_for_eval.answer)
            tfidf_sim = tfidf_cosine_similarity(baseline_for_eval.answer, candidate_for_eval.answer)

            try:
                judge = semantic_similarity_llm(
                    baseline_for_eval.answer,
                    candidate_for_eval.answer,
                    self.cfg,
                    reference_label=baseline_name_for_eval,
                    candidate_label=candidate_name_for_eval,
                    reference_evidence=format_evidence_manifest(baseline_for_eval.evidence_manifest),
                    candidate_evidence=format_evidence_manifest(candidate_for_eval.evidence_manifest),
                )
                sem_cost = estimate_cost(judge.usage, self.cfg.llm_model)
                backend_label = "LLM judge" if judge.backend == "llm" else "vector cosine"
                self.call_from_thread(chat.write, Text(f"  • Semantic similarity ({backend_label}): {judge.semantic_score:.2f}  (judge {judge.usage.total_tokens:,} tok, ${sem_cost:.4f})", style="yellow"))
                if judge.backend == "llm":
                    self.call_from_thread(chat.write, Text(f"  • Topic overlap: {judge.topic_overlap_score:.2f}", style="yellow"))
                    self.call_from_thread(chat.write, Text(f"  • Factuality/groundedness: {judge.factuality_groundedness_score:.2f}", style="yellow"))
                    self.call_from_thread(chat.write, Text(f"  • Evidence sufficiency: {judge.evidence_sufficiency_score:.2f}", style="yellow"))
                    self.call_from_thread(chat.write, Text(f"  • Hallucination risk: {judge.hallucination_risk_flag}", style="yellow"))
                    self.call_from_thread(chat.write, Text(f"  • Judge winner/confidence: {judge.winner}/{judge.confidence:.2f}", style="yellow"))
                self.call_from_thread(chat.write, Text(f"  • Semantic note: {judge.semantic_reason}", style="dim"))
                if judge.backend == "llm":
                    self.call_from_thread(chat.write, Text(f"  • Topic note: {judge.topic_reason}", style="dim"))
            except Exception as e:
                self.call_from_thread(chat.write, Text(f"  • Semantic judge failed: {e}", style="dim red"))

            self.call_from_thread(chat.write, Text(f"  • Lexical overlap: {lexical:.2f}", style="dim"))
            self.call_from_thread(chat.write, Text(f"  • TF-IDF cosine similarity: {tfidf_sim:.2f}", style="dim"))
            self.call_from_thread(chat.write, Text(f"  • Cited docs ({baseline_name_for_eval}/{candidate_name_for_eval}): {len(baseline_cites)}/{len(candidate_cites)}", style="dim"))
            self.call_from_thread(
                chat.write,
                Text(
                    "  • Evidence docs overlap "
                    f"({baseline_name_for_eval}/{candidate_name_for_eval}/overlap/{baseline_name_for_eval}-only/{candidate_name_for_eval}-only): "
                    f"{len(baseline_ev_docs)}/{len(candidate_ev_docs)}/{len(ev_overlap)}/{len(ev_baseline_only)}/{len(ev_candidate_only)}",
                    style="dim",
                ),
            )
            self.call_from_thread(chat.write, Text(f"  • Evidence docs jaccard: {ev_jaccard:.2f}", style="dim"))
            self.call_from_thread(chat.write, Text(""))

    @work(thread=True)
    def run_eval_benchmark(self, questions_file: str) -> None:
        """Run benchmark inside TUI chat window with live traces."""
        chat = self.query_one("#chat-log", RichLog)

        def emit(msg: str) -> None:
            style = "dim"
            if msg.startswith("=== "):
                style = "bold"
            elif "SUMMARY" in msg:
                style = "bold green"
            elif msg.startswith("[traditional] ==="):
                style = "bold yellow"
            elif msg.startswith("[rlm] ==="):
                style = "bold cyan"
            self.call_from_thread(chat.write, Text(msg, style=style))

        try:
            questions = load_questions(questions_file if questions_file else None)
            rows = run_eval(
                questions=questions,
                cfg=self.cfg,
                min_cited_docs=8,
                require_rlm_token_savings=True,
                verbose_trace=True,
                emit=emit,
            )
            summarize(rows, "eval_results.json", emit=emit)
            self.call_from_thread(chat.write, Text("✅ Eval complete. Saved eval_results.json", style="bold green"))
        except Exception as e:
            self.call_from_thread(chat.write, Text(f"❌ Eval failed: {e}", style="bold red"))

    def _handle_copy(self, chat: RichLog) -> None:
        """Copy last turn as JSON to clipboard via pbcopy."""
        import subprocess

        if not self._last_result:
            chat.write(Text("Nothing to copy yet — ask a question first.", style="dim red"))
            return

        result = self._last_result
        payload = {
            "question": self._last_question,
            "answer": result.answer,
            "iterations": result.iterations,
            "tokens": result.usage.to_dict(),
            "per_iteration": [
                {
                    "step": s.iteration,
                    "action": s.action,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "total_tokens": s.total_tokens,
                    "sub_calls": s.sub_calls,
                }
                for s in result.iteration_stats
            ],
            "code_history": result.code_history,
            "inspector_events": self._inspector_events,
        }
        json_str = json.dumps(payload, indent=2, ensure_ascii=False)

        try:
            proc = subprocess.run(["pbcopy"], input=json_str.encode("utf-8"), check=True)
            chat.write(Text(f"✅ Copied to clipboard ({len(json_str):,} chars JSON)", style="bold green"))
        except Exception as e:
            # fallback: save to file
            out_path = Path("last_turn.json")
            out_path.write_text(json_str, encoding="utf-8")
            chat.write(Text(f"⚠️ pbcopy failed, saved to {out_path} ({len(json_str):,} chars)", style="yellow"))

    def _apply_inspector_size(self) -> None:
        panel = self.query_one("#inspector-panel")
        w = self.INSPECTOR_SIZES[self._inspector_idx]
        if w == 0:
            panel.display = False
        else:
            panel.display = True
            panel.styles.width = w

    def action_inspector_wider(self) -> None:
        if self._inspector_idx < len(self.INSPECTOR_SIZES) - 1:
            self._inspector_idx += 1
            self._apply_inspector_size()

    def action_inspector_narrower(self) -> None:
        if self._inspector_idx > 0:
            self._inspector_idx -= 1
            self._apply_inspector_size()

    def action_toggle_inspector(self) -> None:
        panel = self.query_one("#inspector-panel")
        if panel.display:
            self._prev_inspector_idx = self._inspector_idx
            self._inspector_idx = 0
        else:
            self._inspector_idx = getattr(self, "_prev_inspector_idx", 2)
        self._apply_inspector_size()

    def action_clear_chat(self) -> None:
        self.history = []
        self.query_one("#chat-log", RichLog).clear()
        self.query_one("#inspector-log", RichLog).clear()

    def action_interrupt(self) -> None:
        """Ctrl+C behavior:
        1) If input has text -> clear it.
        2) If input is empty -> press twice quickly to quit.
        """
        input_widget = self.query_one("#chat-input", Input)
        chat = self.query_one("#chat-log", RichLog)

        if input_widget.value:
            input_widget.value = ""
            chat.write(Text("Input cleared. Press ctrl+c again to quit.", style="dim"))
            self._last_ctrl_c_ts = 0.0
            return

        now = time.monotonic()
        if (now - self._last_ctrl_c_ts) <= 1.2:
            self.exit()
            return

        self._last_ctrl_c_ts = now
        chat.write(Text("Press ctrl+c again to quit.", style="dim"))

    def action_quit(self) -> None:
        self.exit()


if __name__ == "__main__":
    app = RLMApp()
    app.run()
