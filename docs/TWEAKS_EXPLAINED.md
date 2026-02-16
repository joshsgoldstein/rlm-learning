# Why We Made These Tweaks

This document explains what changed in this project and why, in plain language.

Audience: beginner to intermediate builders who want to understand the system behavior and the tradeoffs.

## The Core Problem

The original RLM demo worked, but real use with many PDFs exposed common issues:

- answers could be shallow or repetitive
- traditional baseline looked "better" for some questions
- long runs could feel stuck
- tracing and evaluation were hard to compare across runs
- prompt logic was spread across code and harder to edit

The goal of these changes was to make the system:

- more reliable
- easier to inspect and debug
- easier to tune
- easier to evaluate in a repeatable way

## What We Changed and Why

## 1) Document Loading and Preprocessing

What changed:

- recursive PDF discovery under `data/` (includes subfolders)
- missing-only processing into `processed_data/`
- fallback to raw PDF when processed text is not present

Why:

- avoids manual cleanup when adding many files
- prevents reprocessing everything every run
- keeps startup faster and more predictable

## 2) Prompt Management

What changed:

- prompt text moved out of Python strings into `prompts/*.txt`
- prompt templates now use `{{PLACEHOLDER}}` replacement

Why:

- easier to iterate on behavior without code edits
- safer prompt tuning for non-engineers
- clearer audit trail for "why the model did that"

## 3) RLM Loop Control

What changed:

- support for "until done" mode (`max_iterations = 0`)
- hard safety cap (`RLM_HARD_MAX_ITERATIONS`)
- autoscaling iteration floor for larger corpora

Why:

- small fixed step counts underperform with many docs
- unbounded loops need a cap for safety
- gives better balance between exploration depth and cost

## 4) Sandbox Safety for Generated Code

What changed:

- per-step execution timeout (`RLM_SANDBOX_EXEC_TIMEOUT_S`)
- per-step line budget (`RLM_SANDBOX_MAX_EXEC_LINES`)
- clear `SandboxLimitExceeded` errors when code hangs

Why:

- model-written Python can accidentally create long loops
- protects the app from appearing frozen
- keeps progress visible and recoverable

## 5) Better Traditional Baseline Transparency

What changed:

- staged traditional progress events in the inspector
- explicit context-window usage and truncation reporting
- stronger instruction that traditional answers must cite sources

Why:

- users can see the baseline is working, not stuck
- makes context truncation visible and measurable
- improves fairness in side-by-side comparison

## 6) Repeatable Evaluation

What changed:

- `rlm_eval.py` benchmark runner
- `eval_questions.txt` fixed question set
- metrics for tokens, cost, truncation, citations, and overlap
- live trace option (`--verbose-trace`)

Why:

- moves evaluation from subjective impressions to measurable runs
- makes regressions easier to detect after prompt/tuning changes

## 7) Semantic Judge Improvements

What changed:

- judge prompt moved to `prompts/semantic_judge.txt`
- separate scores for:
  - semantic similarity
  - topic overlap
- citation booleans in structured output:
  - `answer_a_has_citations`
  - `answer_b_has_citations`
- judge model/provider can be overridden independently via `JUDGE_*` env vars

Why:

- lexical overlap alone is too shallow
- topic overlap captures "did it cover the same themes"
- citation booleans make source behavior explicit
- independent judge model lets you control cost/strictness

## 8) TUI Quality of Life

What changed:

- `/eval` command in chat
- `/autoeval` toggle for post-turn evaluation
- `/traditional` for last baseline stats
- layout and padding tweaks in inspector

Why:

- makes analysis available without leaving the chat flow
- faster iteration when tuning prompts and parameters
- better readability during long runs

## Practical Defaults to Start With

For large corpora (50 to 100 docs), these are solid starting settings:

- `RLM_MAX_ITERATIONS=0` (until done mode)
- `RLM_HARD_MAX_ITERATIONS=60`
- `RLM_SANDBOX_EXEC_TIMEOUT_S=5.0`
- `RLM_SANDBOX_MAX_EXEC_LINES=50000`

If answers are still shallow:

- increase hard cap to 80
- keep timeout and line limits in place
- tune prompt templates in `prompts/`

## Tradeoffs to Know

- More iterations can improve depth, but increase token cost.
- Stronger citation requirements can reduce fluency if evidence is sparse.
- Semantic judging gives better quality signals, but adds judge-token overhead.
- Tight sandbox limits reduce hangs, but may cut off very heavy analysis code.

## Suggested Workflow

1. Edit prompts in `prompts/`.
2. Run a few normal questions in the TUI.
3. Run `/eval` on fixed questions.
4. Compare metrics and traces.
5. Adjust one variable at a time.

This keeps tuning systematic and easier to reason about.
