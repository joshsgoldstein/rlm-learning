from __future__ import annotations

import re
import time
from typing import Callable, Dict, List, Optional

from rlm_core import (
    AnswerResult,
    ChatMessage,
    Config,
    DIRECT_ANSWER_PROMPT_PATH,
    CONVERSATIONAL_ROUTER_PROMPT_PATH,
    RLM_ITER_CONTINUE_PROMPT_PATH,
    RLM_ITER_INITIAL_PROMPT_PATH,
    RLM_SYNTH_PROMPT_PATH,
    RLMSandbox,
    SYSTEM_PROMPT_PATH,
    IterationStats,
    TokenUsage,
    _render_prompt,
    call_llm,
)
from .router import classify_route


def is_conversational(question: str, history: List[ChatMessage], cfg: Config, doc_ids: List[str] | None = None) -> bool:
    """Let the LLM decide if this question needs document research or is just conversational/general."""
    return classify_route(question=question, history=history, cfg=cfg, doc_ids=doc_ids) == "CHAT"


def answer_direct(
    question: str,
    history: List[ChatMessage],
    cfg: Config,
    on_event: Optional[Callable] = None,
) -> AnswerResult:
    """Answer a conversational question directly from chat history — no document research."""
    _on_event = on_event or (lambda *a: None)
    _on_event("iteration", "Direct answer (no document research needed)")

    usage = TokenUsage()
    tail = history[-(cfg.max_history_messages):] if cfg.max_history_messages > 0 else []
    chat_ctx = "\n".join(f"{m.role.upper()}: {m.content[:800]}" for m in tail if m.content.strip()) or "(none)"

    prompt = _render_prompt(
        DIRECT_ANSWER_PROMPT_PATH,
        CHAT_CTX=chat_ctx,
        QUESTION=question,
    )

    t0 = time.perf_counter()
    answer = call_llm(prompt, cfg, usage_accum=usage)
    llm_ms = (time.perf_counter() - t0) * 1000.0
    _on_event("answer", f"Direct answer ({len(answer)} chars)")

    return AnswerResult(
        answer=answer,
        usage=usage,
        iterations=1,
        iteration_stats=[
            IterationStats(
                iteration=1,
                action="direct",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                planner_llm_ms=llm_ms,
                iteration_ms=llm_ms,
            )
        ],
        code_history=[],
        evidence_manifest={"type": "direct_chat", "docs_touched": []},
    )


def answer_question(
    doc_map: Dict[str, str],
    question: str,
    history: List[ChatMessage],
    cfg: Config,
    on_event: Optional[Callable] = None,
) -> AnswerResult:
    """Smart router: conversational/general questions get a direct answer, research questions get the RLM sandbox."""
    if is_conversational(question, history, cfg, doc_ids=list(doc_map.keys())):
        return answer_direct(question, history, cfg, on_event)

    _on_event = on_event or (lambda *a: None)
    sandbox = RLMSandbox(doc_map, cfg, on_event=_on_event)
    doc_count = len(doc_map)

    unbounded_mode = cfg.max_iterations <= 0
    effective_max_iterations = cfg.max_iterations
    if unbounded_mode:
        effective_max_iterations = max(1, cfg.hard_max_iterations)
        _on_event(
            "warning",
            f"Until-done mode enabled (max_iterations=0). Hard safety cap: {effective_max_iterations} iterations.",
        )
    else:
        if doc_count >= 60 and effective_max_iterations < 18:
            effective_max_iterations = 18
            _on_event("warning", f"Large corpus detected ({doc_count} docs): auto-increasing max iterations to {effective_max_iterations}")
        elif doc_count >= 30 and effective_max_iterations < 14:
            effective_max_iterations = 14
            _on_event("warning", f"Medium corpus detected ({doc_count} docs): auto-increasing max iterations to {effective_max_iterations}")

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    tail = history[-(cfg.max_history_messages):] if cfg.max_history_messages > 0 else []
    chat_ctx = "\n".join(f"{m.role.upper()}: {m.content[:600]}" for m in tail if m.content.strip()) or "(none)"

    doc_items = list(sandbox.docs.items())
    if doc_count > 30:
        sample = doc_items[:20]
        sampled_summary = "\n".join(f"  - {did}: {info['pages']} pages" for did, info in sample)
        doc_summary = (
            f"{doc_count} total documents available.\n"
            f"Sample of first {len(sample)} doc ids:\n{sampled_summary}\n"
            "Use `docs.keys()` and `docs[doc_id]['pages']` in code to enumerate all documents."
        )
    else:
        doc_summary = "\n".join(f"  - {did}: {info['pages']} pages" for did, info in doc_items)

    conversation: List[tuple[str, str]] = []
    code_history: List[str] = []
    iteration_stats: List[IterationStats] = []

    history_window = 2 if doc_count >= 30 else 3
    per_code_chars = 800 if doc_count >= 30 else 1200
    per_output_chars = 700 if doc_count >= 30 else 1000

    iteration_total_label = "∞" if unbounded_mode else str(effective_max_iterations)
    for iteration in range(effective_max_iterations):
        iter_started = time.perf_counter()
        _on_event("iteration", f"Step {iteration + 1}/{iteration_total_label}")

        tokens_before = TokenUsage(sandbox.usage.input_tokens, sandbox.usage.output_tokens, sandbox.usage.total_tokens)
        sub_calls_before = sandbox._sub_call_count
        sub_time_before = sandbox._sub_call_time_ms

        if iteration == 0:
            prompt = _render_prompt(
                RLM_ITER_INITIAL_PROMPT_PATH,
                SYSTEM_PROMPT=system_prompt,
                DOC_SUMMARY=doc_summary,
                CHAT_CTX=chat_ctx,
                QUESTION=question,
            )
        else:
            recent_iters = []
            for i, (code_i, output_i) in enumerate(conversation[-history_window:]):
                trimmed_code = code_i[:per_code_chars]
                trimmed_out = output_i[:per_output_chars]
                if len(code_i) > per_code_chars:
                    trimmed_code += "\n# ... truncated ..."
                if len(output_i) > per_output_chars:
                    trimmed_out += "\n... (truncated)"
                recent_iters.append(
                    f"--- Recent Iteration {len(conversation) - history_window + i + 1} ---\n"
                    f"Code:\n{trimmed_code}\nOutput:\n{trimmed_out}"
                )
            recent_trace = "\n".join(recent_iters) if recent_iters else "(no prior steps)"

            is_last = iteration == effective_max_iterations - 1
            if is_last:
                force_msg = (
                    "THIS IS YOUR FINAL ITERATION. You MUST call answer() NOW with the best answer you can give "
                    "based on the evidence you have gathered so far. Do NOT explore further. Just synthesize and call answer()."
                )
            else:
                force_msg = ("The answer has NOT been set yet. " if sandbox.final_answer is None else "") + (
                    "Write the next block of Python code. Call answer(\"your final answer\") when you have enough evidence."
                )

            prompt = _render_prompt(
                RLM_ITER_CONTINUE_PROMPT_PATH,
                SYSTEM_PROMPT=system_prompt,
                DOC_COUNT=doc_count,
                QUESTION=question,
                RECENT_TRACE=recent_trace,
                FORCE_MSG=force_msg,
            )

        llm_t0 = time.perf_counter()
        raw = call_llm(prompt, cfg, usage_accum=sandbox.usage)
        planner_llm_ms = (time.perf_counter() - llm_t0) * 1000.0

        code = extract_code(raw)
        if not code.strip():
            _on_event("warning", "LLM returned no code, retrying...")
            continue

        code_history.append(code)
        _on_event("code", f"```\n{code}\n```")

        exec_t0 = time.perf_counter()
        output = sandbox.execute_code(code)
        exec_ms = (time.perf_counter() - exec_t0) * 1000.0
        if output.strip():
            _on_event("output", output.strip()[:300])
            if output.startswith("ERROR: SandboxLimitExceeded:"):
                _on_event("warning", output.strip())

        conversation.append((code, output))
        _on_event("tokens", f"{sandbox.usage.input_tokens},{sandbox.usage.output_tokens},{sandbox.usage.total_tokens}")

        iter_input = sandbox.usage.input_tokens - tokens_before.input_tokens
        iter_output = sandbox.usage.output_tokens - tokens_before.output_tokens
        iter_sub = sandbox._sub_call_count - sub_calls_before
        iter_sub_ms = sandbox._sub_call_time_ms - sub_time_before
        iter_ms = (time.perf_counter() - iter_started) * 1000.0

        tools_used = []
        if "peek(" in code:
            tools_used.append("peek")
        if "search(" in code:
            tools_used.append("search")
        if "llm_query(" in code:
            tools_used.append("llm_query")
        if "answer(" in code:
            tools_used.append("answer")
        action_label = "+".join(tools_used) if tools_used else "code"

        iteration_stats.append(
            IterationStats(
                iteration=iteration + 1,
                action=action_label,
                input_tokens=iter_input,
                output_tokens=iter_output,
                total_tokens=iter_input + iter_output,
                sub_calls=iter_sub,
                planner_llm_ms=planner_llm_ms,
                sub_llm_ms=iter_sub_ms,
                exec_ms=exec_ms,
                iteration_ms=iter_ms,
            )
        )

        if sandbox.final_answer is not None:
            return AnswerResult(
                answer=sandbox.final_answer,
                usage=sandbox.usage,
                iterations=iteration + 1,
                iteration_stats=iteration_stats,
                code_history=code_history,
                evidence_manifest=sandbox.evidence_manifest(),
            )

    if sandbox.final_answer is None:
        if unbounded_mode:
            _on_event("warning", "Safety cap reached in until-done mode — synthesizing answer from gathered evidence...")
        else:
            _on_event("warning", "Max iterations reached — synthesizing answer from gathered evidence...")
        all_output = "\n".join(out for _, out in conversation if out.strip())
        if all_output.strip():
            synth_prompt = _render_prompt(
                RLM_SYNTH_PROMPT_PATH,
                QUESTION=question,
                EVIDENCE=all_output[:6000],
            )
            sandbox.final_answer = call_llm(synth_prompt, cfg, usage_accum=sandbox.usage)
            _on_event("answer", f"Synthesized answer from evidence ({len(sandbox.final_answer)} chars)")
        else:
            sandbox.final_answer = (
                "I wasn't able to find enough information in the documents to answer this question. "
                "Try rephrasing or asking about a more specific topic."
            )

    return AnswerResult(
        answer=sandbox.final_answer,
        usage=sandbox.usage,
        iterations=len(iteration_stats),
        iteration_stats=iteration_stats,
        code_history=code_history,
        evidence_manifest=sandbox.evidence_manifest(),
    )


def extract_code(text: str) -> str:
    """Extract Python code from LLM output (handles ```python blocks or raw code)."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = text.strip().splitlines()
    code_lines = []
    for line in lines:
        if line.strip().startswith("```"):
            continue
        code_lines.append(line)
    return "\n".join(code_lines).strip()
