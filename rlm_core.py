# rlm_core.py — RLM: Recursive Language Model (paper-faithful implementation)
#
# The LLM gets a Python REPL sandbox with documents loaded as variables.
# It writes code to explore, search, and recursively call itself on chunks.
from __future__ import annotations

import json
import os
import re
import io
import contextlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

from rlm_trace import TraceTree


# ────────────────────────── Config ──────────────────────────

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic")
SUPPORTED_APPROACHES = ("rlm", "traditional", "rag")


def enabled_approaches_from_env(default: str = "rlm,traditional") -> List[str]:
    """Parse ENABLED_APPROACHES from env as ordered, deduplicated list."""
    raw = os.getenv("ENABLED_APPROACHES", default)
    items = [s.strip().lower() for s in raw.split(",") if s.strip()]
    out: List[str] = []
    for item in items:
        if item in SUPPORTED_APPROACHES and item not in out:
            out.append(item)
    return out or [s.strip() for s in default.split(",") if s.strip()]

# Known context window sizes (tokens) for common models
MODEL_CONTEXT_WINDOWS = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    # Anthropic
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    # Ollama (common models)
    "qwen2.5:7b": 32_768,
    "qwen2.5:14b": 32_768,
    "qwen2.5:32b": 32_768,
    "llama3.1:8b": 128_000,
    "llama3.1:70b": 128_000,
    "mistral:7b": 32_768,
    "gemma2:9b": 8_192,
}


def get_context_window(model: str) -> int:
    """Get the context window size for a model. Falls back to 32K if unknown."""
    # exact match
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    # partial match (e.g. "gpt-4o-mini-2024-07-18" → "gpt-4o-mini")
    for known, size in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(known):
            return size
    return 32_768  # conservative default


# Pricing per 1M tokens (USD) — input / output
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "o1": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    # Ollama (local — free)
    "qwen2.5:7b": (0.0, 0.0),
    "llama3.1:8b": (0.0, 0.0),
    "mistral:7b": (0.0, 0.0),
}


def estimate_cost(usage: "TokenUsage", model: str) -> float:
    """Estimate USD cost for a token usage based on model pricing."""
    pricing = None
    if model in MODEL_PRICING:
        pricing = MODEL_PRICING[model]
    else:
        for known, p in MODEL_PRICING.items():
            if model.startswith(known):
                pricing = p
                break
    if not pricing:
        return 0.0
    input_cost = (usage.input_tokens / 1_000_000) * pricing[0]
    output_cost = (usage.output_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


@dataclass
class Config:
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_temperature: float
    max_iterations: int
    hard_max_iterations: int
    sandbox_exec_timeout_s: float
    sandbox_max_exec_lines: int
    max_history_messages: int
    context_window: int = 0  # 0 = auto-detect from model

    @staticmethod
    def from_env() -> Config:
        provider = os.getenv("LLM_PROVIDER", "ollama")
        default_models = {"ollama": "qwen2.5:7b", "openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-20250514"}
        default_urls = {"ollama": "http://localhost:11434", "openai": "https://api.openai.com", "anthropic": "https://api.anthropic.com"}
        api_key = (
            os.getenv("OPENAI_API_KEY", "") if provider == "openai"
            else os.getenv("ANTHROPIC_API_KEY", "") if provider == "anthropic"
            else os.getenv("LLM_API_KEY", "")
        )
        model = os.getenv("LLM_MODEL", default_models.get(provider, "gpt-4o-mini"))
        context_window = int(os.getenv("LLM_CONTEXT_WINDOW", "0")) or get_context_window(model)
        return Config(
            llm_provider=provider,
            llm_model=model,
            llm_base_url=os.getenv("LLM_BASE_URL", default_urls.get(provider, "http://localhost:11434")),
            llm_api_key=api_key,
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", "10")),
            hard_max_iterations=int(os.getenv("RLM_HARD_MAX_ITERATIONS", "60")),
            sandbox_exec_timeout_s=float(os.getenv("RLM_SANDBOX_EXEC_TIMEOUT_S", "5.0")),
            sandbox_max_exec_lines=int(os.getenv("RLM_SANDBOX_MAX_EXEC_LINES", "50000")),
            max_history_messages=int(os.getenv("RLM_MAX_HISTORY_MESSAGES", "8")),
            context_window=context_window,
        )


# ────────────────────────── Token tracking ──────────────────────────

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        return self

    def to_dict(self) -> Dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "total_tokens": self.total_tokens}


@dataclass
class LLMResult:
    text: str
    usage: TokenUsage


# ────────────────────────── LLM providers ──────────────────────────

def _call_ollama(prompt: str, cfg: Config) -> LLMResult:
    r = requests.post(
        f"{cfg.llm_base_url}/api/generate",
        json={"model": cfg.llm_model, "prompt": prompt, "stream": False, "options": {"temperature": cfg.llm_temperature}},
        timeout=180,
    )
    r.raise_for_status()
    body = r.json()
    return LLMResult(
        text=body["response"],
        usage=TokenUsage(body.get("prompt_eval_count", 0), body.get("eval_count", 0), body.get("prompt_eval_count", 0) + body.get("eval_count", 0)),
    )


def _call_openai(prompt: str, cfg: Config) -> LLMResult:
    from openai import OpenAI
    base = cfg.llm_base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    client = OpenAI(api_key=cfg.llm_api_key, base_url=base)
    resp = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.llm_temperature,
    )
    u = resp.usage
    return LLMResult(
        text=resp.choices[0].message.content or "",
        usage=TokenUsage(u.prompt_tokens if u else 0, u.completion_tokens if u else 0, u.total_tokens if u else 0),
    )


def _call_anthropic(prompt: str, cfg: Config) -> LLMResult:
    from anthropic import Anthropic
    client = Anthropic(api_key=cfg.llm_api_key)
    resp = client.messages.create(model=cfg.llm_model, max_tokens=4096, temperature=cfg.llm_temperature, messages=[{"role": "user", "content": prompt}])
    return LLMResult(
        text=resp.content[0].text,
        usage=TokenUsage(resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.input_tokens + resp.usage.output_tokens),
    )


_PROVIDERS = {"ollama": _call_ollama, "openai": _call_openai, "anthropic": _call_anthropic}


def call_llm(prompt: str, cfg: Config, usage_accum: Optional[TokenUsage] = None) -> str:
    if cfg.llm_provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {cfg.llm_provider}")
    result = _PROVIDERS[cfg.llm_provider](prompt, cfg)
    if usage_accum is not None:
        usage_accum += result.usage
    return result.text


# ────────────────────────── Document Library ──────────────────────────

def load_document_pages(path: str) -> List[str]:
    """Load pages from a processed dir (text.txt per page) or raw PDF."""
    p = Path(path)
    if p.is_dir():
        page_dirs = sorted(p.glob("page_*"), key=lambda x: int(x.name.split("_")[1]))
        pages = []
        for pd in page_dirs:
            txt = pd / "text.txt"
            pages.append(txt.read_text(encoding="utf-8") if txt.exists() else "")
        return pages
    else:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return [(pg.extract_text() or "") for pg in reader.pages]


PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "rlm_system.txt"
LLM_QUERY_PROMPT_PATH = PROMPTS_DIR / "llm_query.txt"
TRADITIONAL_PROMPT_PATH = PROMPTS_DIR / "traditional_answer.txt"
CONVERSATIONAL_ROUTER_PROMPT_PATH = PROMPTS_DIR / "conversational_router.txt"
DIRECT_ANSWER_PROMPT_PATH = PROMPTS_DIR / "direct_answer.txt"
RLM_ITER_INITIAL_PROMPT_PATH = PROMPTS_DIR / "rlm_iteration_initial.txt"
RLM_ITER_CONTINUE_PROMPT_PATH = PROMPTS_DIR / "rlm_iteration_continue.txt"
RLM_SYNTH_PROMPT_PATH = PROMPTS_DIR / "rlm_synthesis_fallback.txt"


def _render_prompt(path: Path, **kwargs: object) -> str:
    """Render a prompt template with {{PLACEHOLDER}} substitutions."""
    template = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        template = template.replace(f"{{{{{key}}}}}", str(value))
    return template


@dataclass
class ChatMessage:
    role: str
    content: str


# ────────────────────────── REPL Sandbox ──────────────────────────

class RLMSandbox:
    """Python REPL sandbox that the LLM writes code against.

    Available to the LLM as built-in functions:
      - docs                  → dict of {doc_id: {"pages": int, "path": str}}
      - peek(doc_id, page)    → str (read page text, 1-indexed)
      - search(query)         → list of (doc_id, page, context_snippet)
      - llm_query(question, text) → str (recursive sub-LM call)
      - answer(text)          → sets the final answer
    """

    def __init__(self, doc_map: Dict[str, str], cfg: Config, on_event: Optional[Callable] = None):
        self.cfg = cfg
        self.on_event = on_event or (lambda *a: None)
        self.usage = TokenUsage()

        # load all documents into memory (text pages)
        self._pages: Dict[str, List[str]] = {}
        self._paths = doc_map
        for doc_id, path in doc_map.items():
            self._pages[doc_id] = load_document_pages(path)

        # metadata the LLM can see
        self.docs: Dict[str, dict] = {}
        for doc_id, pages in self._pages.items():
            self.docs[doc_id] = {"pages": len(pages), "path": doc_map[doc_id]}

        self.final_answer: Optional[str] = None
        self._iteration = 0
        self._sub_call_count = 0
        self._sub_call_time_ms = 0.0

        # persistent REPL state — variables survive across exec() calls
        self._globals: Dict[str, object] = {
            "docs": self.docs,
            "peek": self.peek,
            "search": self.search,
            "llm_query": self.llm_query,
            "answer": self.answer,
            "print": print,  # will be captured per-call
            "len": len, "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "range": range, "enumerate": enumerate,
            "sorted": sorted, "min": min, "max": max, "sum": sum, "zip": zip,
            "isinstance": isinstance, "type": type,
            "True": True, "False": False, "None": None,
        }

    def peek(self, doc_id: str, page: int, max_chars: int = 3000) -> str:
        """Read a document page (1-indexed). Returns the text."""
        pages = self._pages.get(doc_id)
        if not pages:
            return f"ERROR: unknown doc_id '{doc_id}'"
        if page < 1 or page > len(pages):
            return f"ERROR: page {page} out of range (doc has {len(pages)} pages)"
        text = pages[page - 1]
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + f"\n... (truncated, {len(pages[page-1])} total chars)"
        self.on_event("peek", f"[{doc_id} p{page}] {len(text)} chars")
        return text

    def search(self, query: str, max_hits: Optional[int] = None, max_per_doc: Optional[int] = None) -> List[Tuple[str, int, str]]:
        """Search all documents for a keyword/phrase. Returns [(doc_id, page, context)].

        Spreads results across documents (max_per_doc hits per document)
        so the agent discovers content from many sources, not just the first match.
        """
        q = query.strip()
        if not q:
            return []

        # Adaptive defaults for large corpora:
        # - fewer hits per doc to increase breadth
        # - more total hits so 50-100+ docs are actually discoverable
        total_docs = len(self._pages)
        if max_per_doc is None:
            max_per_doc = 1 if total_docs >= 40 else 2
        if max_hits is None:
            if total_docs >= 80:
                max_hits = 120
            elif total_docs >= 40:
                max_hits = 80
            elif total_docs >= 20:
                max_hits = 50
            else:
                max_hits = 20

        try:
            pat = re.compile(q, re.IGNORECASE)
        except re.error:
            pat = re.compile(re.escape(q), re.IGNORECASE)

        hits: List[Tuple[str, int, str]] = []
        for doc_id, pages in self._pages.items():
            doc_hits = 0
            for i, text in enumerate(pages):
                m = pat.search(text)
                if not m:
                    continue
                start = max(0, m.start() - 200)
                end = min(len(text), m.end() + 200)
                ctx = text[start:end].replace("\n", " ").strip()
                hits.append((doc_id, i + 1, ctx))
                if len(hits) >= max_hits:
                    self.on_event("search", f'"{q}" → {len(hits)} hits across {len(set(h[0] for h in hits))} docs')
                    return hits
                doc_hits += 1
                if doc_hits >= max_per_doc:
                    break  # move to next document
        self.on_event("search", f'"{q}" → {len(hits)} hits across {len(set(h[0] for h in hits))} docs')
        return hits

    def llm_query(self, question: str, text: str) -> str:
        """Recursive sub-LM call: ask a question about a chunk of text."""
        prompt = _render_prompt(
            LLM_QUERY_PROMPT_PATH,
            TEXT=text[:4000],
            QUESTION=question,
        )
        t0 = time.perf_counter()
        result = call_llm(prompt, self.cfg, usage_accum=self.usage)
        self._sub_call_time_ms += (time.perf_counter() - t0) * 1000.0
        self._sub_call_count += 1
        preview = result.replace("\n", " ")[:100]
        self.on_event("llm_query", f'"{question[:60]}" → {preview}...')
        return result

    def answer(self, text: str) -> None:
        """Set the final answer. Call this when done."""
        self.final_answer = text
        self.on_event("answer", f"Final answer set ({len(text)} chars)")

    def execute_code(self, code: str) -> str:
        """Execute Python code in the persistent REPL sandbox.

        Behaves like a Jupyter cell:
        - print() output is captured
        - If the last statement is an expression, its repr is appended (like a REPL)
        - Variables persist across calls
        """
        class _SandboxLimitExceeded(Exception):
            pass

        stdout = io.StringIO()
        self._globals["print"] = lambda *a, **kw: print(*a, **kw, file=stdout)
        started = time.monotonic()
        line_budget = max(1000, int(self.cfg.sandbox_max_exec_lines))
        timeout_s = max(0.5, float(self.cfg.sandbox_exec_timeout_s))
        line_count = 0

        def _trace(frame, event, arg):  # type: ignore[no-untyped-def]
            nonlocal line_count
            if event == "line":
                line_count += 1
                if line_count > line_budget:
                    raise _SandboxLimitExceeded(
                        f"line budget exceeded ({line_budget} lines). "
                        f"Try smaller loops / fewer docs per step."
                    )
                if (time.monotonic() - started) > timeout_s:
                    raise _SandboxLimitExceeded(
                        f"time limit exceeded ({timeout_s:.1f}s). "
                        f"Step likely stuck in long loop."
                    )
            return _trace

        try:
            sys.settrace(_trace)
            # Try to detect if last line is an expression — if so, eval it and print repr
            # (mimics Python REPL / Jupyter behavior)
            import ast as _ast
            tree = _ast.parse(code)
            if tree.body and isinstance(tree.body[-1], _ast.Expr):
                # split: exec everything except last, then eval last expression
                last_expr = tree.body.pop()
                if tree.body:
                    exec(compile(tree, "<rlm>", "exec"), self._globals)
                val = eval(compile(_ast.Expression(body=last_expr.value), "<rlm>", "eval"), self._globals)
                if val is not None:
                    print(repr(val), file=stdout)
            else:
                exec(code, self._globals)
        except _SandboxLimitExceeded as e:
            return f"ERROR: SandboxLimitExceeded: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
        finally:
            sys.settrace(None)
            self._globals["print"] = print
        return stdout.getvalue()


# ────────────────────────── RLM Agent Loop ──────────────────────────


@dataclass
class IterationStats:
    """Token usage for a single iteration of the RLM loop."""
    iteration: int
    action: str           # "peek", "search", "llm_query", "code", etc.
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    sub_calls: int = 0    # number of llm_query sub-calls in this iteration
    planner_llm_ms: float = 0.0
    sub_llm_ms: float = 0.0
    exec_ms: float = 0.0
    iteration_ms: float = 0.0


@dataclass
class TraditionalStats:
    """Metadata about what happened during a traditional (all-in-context) call."""
    total_docs: int = 0
    docs_included: int = 0
    docs_excluded: List[str] = field(default_factory=list)
    total_chars: int = 0
    chars_sent: int = 0
    truncated: bool = False
    context_window: int = 0
    context_used_pct: float = 0.0


@dataclass
class AnswerResult:
    answer: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    iterations: int = 0
    iteration_stats: List[IterationStats] = field(default_factory=list)
    code_history: List[str] = field(default_factory=list)
    traditional_stats: Optional[TraditionalStats] = None


def answer_traditional(
    doc_map: Dict[str, str],
    question: str,
    cfg: Config,
    on_event: Optional[Callable] = None,
) -> AnswerResult:
    """Compatibility wrapper: implementation lives in rlm_approach_traditional.py."""
    from approaches.traditional import answer_traditional as _answer_traditional

    return _answer_traditional(doc_map=doc_map, question=question, cfg=cfg, on_event=on_event)


def _is_conversational(question: str, history: List[ChatMessage], cfg: Config) -> bool:
    """Compatibility wrapper: implementation lives in approaches/rlm.py."""
    from approaches.rlm import is_conversational

    return is_conversational(question=question, history=history, cfg=cfg)


def answer_direct(
    question: str,
    history: List[ChatMessage],
    cfg: Config,
    on_event: Optional[Callable] = None,
) -> AnswerResult:
    """Compatibility wrapper: implementation lives in approaches/rlm.py."""
    from approaches.rlm import answer_direct as _answer_direct

    return _answer_direct(question=question, history=history, cfg=cfg, on_event=on_event)


def answer_question(
    doc_map: Dict[str, str],
    question: str,
    history: List[ChatMessage],
    cfg: Config,
    on_event: Optional[Callable] = None,
) -> AnswerResult:
    """Compatibility wrapper: implementation lives in approaches/rlm.py."""
    from approaches.rlm import answer_question as _answer_question

    return _answer_question(doc_map=doc_map, question=question, history=history, cfg=cfg, on_event=on_event)


def _extract_code(text: str) -> str:
    """Compatibility wrapper: implementation lives in approaches/rlm.py."""
    from approaches.rlm import extract_code

    return extract_code(text)
