# Potential TUI Layout Designs

## Overview

This document contains proposed ASCII layout designs for the RLM Explorer TUI. The goal is to improve the user experience by:

1. **Showing the winning response** prominently in the chat window
2. **Displaying a comparison table** for easy evaluation
3. **Using tabs/subtabs** for process details and final outputs
4. **Better visualizing context** to show what each approach "sees"

---

## Current Layout Analysis

### Current Structure:
- Shows **all 3 outputs** (Traditional, RAG, RLM) sequentially in the chat panel
- Has inspector tabs for each approach (Overview, Traditional, RAG, RLM)
- Shows comparison table at the end of all outputs
- Context/token/concept info in inspector bottom panel

### Issues to Address:
- All responses shown sequentially makes it hard to identify the winner
- Comparison requires scrolling to the end
- Process details mixed with outputs
- Context visualization is minimal

---

## OPTION 1: Winner-First with Tabbed Deep Dive

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RLM Explorer - Question: "Where do Deloitte and KPMG disagree..."          │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ 💬 Chat - Winning Response           │ 📊 Analysis & Comparison             │
│                                      │                                      │
│ 🏆 WINNER: RLM (81% token savings)  │ ┌──────────────────────────────────┐ │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │ │ COMPARISON TABLE                 │ │
│ [RLM's answer with citations here]   │ ├──────────┬────────┬───────┬──────┤ │
│ ...full response...                  │ │ Approach │ Tokens │  Cost │ Qual │ │
│                                      │ ├──────────┼────────┼───────┼──────┤ │
│ Evidence docs: 86 docs explored      │ │ 🐌 Trad  │ 113.9K │ $0.02 │ 0.48 │ │
│ Cited: Deloitte_2024, KPMG_2024...  │ │ 📚 RAG   │  45.2K │ $0.01 │ 0.61 │ │
│                                      │ │ 🚀 RLM   │  21.4K │ $0.00 │ 0.92 │ │
│                                      │ └──────────┴────────┴───────┴──────┘ │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │                                      │
│                                      │ ┌──────────────────────────────────┐ │
│ 💡 Quick Stats:                      │ │ Tabs: [Traditional] [RAG] [RLM]  │ │
│   • RLM: 21.4K tokens ($0.004)       │ ├──────────────────────────────────┤ │
│   • Traditional: 113.9K ($0.017)     │ │ Subtabs: [Process] [Output]      │ │
│   • Savings: 81% fewer tokens        │ ├──────────────────────────────────┤ │
│   • Semantic similarity: 0.92        │ │ [Process tab selected]           │ │
│                                      │ │                                  │ │
│ [See tabs for detailed processes →] │ │ 🔄 Iteration 1: Router decision  │ │
│                                      │ │ 💻 Code: search("governance")    │ │
│                                      │ │ 📤 Output: Found 8 docs...       │ │
│                                      │ │                                  │ │
│                                      │ │ 🔄 Iteration 2: Deep analysis    │ │
│                                      │ │ 💻 Code: peek(Deloitte_2024)     │ │
│                                      │ │ ...                              │ │
│                                      │ └──────────────────────────────────┘ │
│                                      │                                      │
│ > [Input box here]                   │ 📚 Context: 86 docs loaded          │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### Pros:
- Winner is immediately visible
- Comparison table always in view
- Deep dive available in tabs
- Clean separation of "what won" vs "how it worked"

### Cons:
- Can't see other responses without switching tabs
- Requires user to explore tabs to understand alternative approaches

---

## OPTION 2: Tabbed Approaches with Comparison Header

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RLM Explorer - Question: "Where do Deloitte and KPMG disagree..."          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 📊 COMPARISON DASHBOARD (always visible)                                    │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ 🏆 Winner: RLM (81% savings) │ Semantic: 0.92 │ Tokens: 21.4K │ $0.004│   │
│ ├───────┬─────────────┬──────────┬─────────┬──────────┬─────────────────┤   │
│ │       │ Tokens      │ Cost     │ Quality │ Citations│ Status          │   │
│ ├───────┼─────────────┼──────────┼─────────┼──────────┼─────────────────┤   │
│ │ 🐌 Traditional │ 113,929 │ $0.017   │ 0.48    │ 3 docs   │ ⚠️ Truncated│   │
│ │ 📚 RAG        │  45,283 │ $0.010   │ 0.61    │ 5 docs   │ ✅ Complete │   │
│ │ 🚀 RLM        │  21,394 │ $0.004   │ 0.92    │ 4 docs   │ ✅ Complete │   │
│ └───────┴─────────────┴──────────┴─────────┴──────────┴─────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─ APPROACH DETAILS ─────────────────────────────────────────────────────┐  │
│ │ Tabs: [🏆 Winner] [🐌 Traditional] [📚 RAG] [🚀 RLM] [📊 All Metrics]  │  │
│ ├────────────────────────────────────────────────────────────────────────┤  │
│ │ Subtabs for selected: [📝 Output] [⚙️ Process] [📈 Stats] [🔍 Context]│  │
│ ├────────────────────────────────────────────────────────────────────────┤  │
│ │                                                                         │  │
│ │ [📝 Output tab for RLM approach shown]                                 │  │
│ │                                                                         │  │
│ │ Deloitte and KPMG disagree on governance readiness in several key      │  │
│ │ areas. Deloitte emphasizes [citation: Deloitte_2024_p42] that...       │  │
│ │                                                                         │  │
│ │ Evidence from exploration:                                              │  │
│ │   • Searched: 86 documents                                             │  │
│ │   • Targeted: 12 documents for deep analysis                           │  │
│ │   • Cited: 4 primary sources                                           │  │
│ │                                                                         │  │
│ │ [Scroll for full response...]                                           │  │
│ │                                                                         │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│ > [Input box here]                                                            │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Pros:
- Comparison always visible at top
- Easy to switch between approaches
- Subtabs provide structured views (Output, Process, Stats, Context)
- Can compare outputs by switching tabs quickly

### Cons:
- Comparison header takes vertical space
- Need to switch tabs to compare actual response content

---

## OPTION 3: Split View with Side-by-Side Winner + Runner-up

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RLM Explorer                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 💬 CHAT - Responses                                                         │
│ ┌─────────────────────────────────┬─────────────────────────────────────┐   │
│ │ 🏆 RLM (Winner) - $0.004         │ 🥈 Traditional (Baseline) - $0.017  │   │
│ │ 21.4K tokens │ Quality: 0.92     │ 113.9K tokens │ Quality: 0.48      │   │
│ ├─────────────────────────────────┼─────────────────────────────────────┤   │
│ │ [RLM Output]                     │ [Traditional Output]                │   │
│ │                                  │                                     │   │
│ │ Deloitte and KPMG disagree on... │ Based on the documents, there is... │   │
│ │                                  │                                     │   │
│ │ Evidence: 86 docs searched       │ ⚠️ Note: Only 8 of 86 docs fit     │   │
│ │ Citations: [Deloitte_2024...]    │ Citations: [Deloitte_2024...]       │   │
│ │                                  │                                     │   │
│ │ [Click to expand full response]  │ [Click to expand full response]     │   │
│ └─────────────────────────────────┴─────────────────────────────────────┘   │
│                                                                               │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│                                                                               │
│ 📊 Detailed Analysis                                                         │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Tabs: [⚙️ RLM Process] [⚙️ Trad Process] [⚙️ RAG Process] [📈 Metrics] │ │
│ ├─────────────────────────────────────────────────────────────────────────┤ │
│ │                                                                          │ │
│ │ [⚙️ RLM Process selected]                                                │ │
│ │                                                                          │ │
│ │ 🔄 Iteration 1: Router (CHAT vs DOC)                                    │ │
│ │   💻 Code: classify_route(question, docs)                               │ │
│ │   📤 Output: DOC (requires document analysis)                           │ │
│ │                                                                          │ │
│ │ 🔄 Iteration 2: Document discovery                                      │ │
│ │   💻 Code: search(keywords=["governance", "readiness"])                 │ │
│ │   📤 Output: Found 12 relevant docs                                     │ │
│ │   🔍 Peek: Deloitte_2024.txt lines 1-50                                 │ │
│ │                                                                          │ │
│ │ 🔄 Iteration 3: Evidence extraction                                     │ │
│ │   💻 Code: extract_quotes(doc_ids=[...])                                │ │
│ │   🧠 Sub-LLM call: Summarize position on governance                     │ │
│ │                                                                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│ > [Input box here]                                                            │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Pros:
- Direct visual comparison of top 2 approaches
- Process tabs separate from outputs
- Easy to see why one won vs the other
- Side-by-side makes differences immediately obvious

### Cons:
- Each response gets less horizontal space
- Only shows top 2, not all approaches
- May be cramped on smaller terminals

---

## OPTION 4: Context-First Design with Winner Emphasis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RLM Explorer                                                                │
├───────────────────────────────────┬─────────────────────────────────────────┤
│ 💬 Chat & Results                 │ 🔍 Context & Process Explorer           │
│                                   │                                         │
│ ❓ Question:                       │ 📚 CONTEXT VISUALIZATION                │
│ "Where do Deloitte and KPMG..."   │ ┌───────────────────────────────────┐   │
│                                   │ │ Total Corpus: 86 docs               │   │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │ │                                     │   │
│ 🏆 WINNER: RLM                    │ │ Traditional: Used 8/86 (truncated) │   │
│                                   │ │ ████░░░░░░░░░░░░░░░░ (9%)           │   │
│ [Answer with citations]           │ │                                     │   │
│                                   │ │ RAG: Retrieved 5 docs               │   │
│ Deloitte emphasizes...            │ │ ██████░░░░░░░░░░░░░░ (12%)          │   │
│ KPMG argues...                    │ │                                     │   │
│ The key disagreement is...        │ │ RLM: Explored 86, used 12          │   │
│                                   │ │ ████████████████████ (100% → 14%)   │   │
│ [Full response...]                │ └───────────────────────────────────┘   │
│                                   │                                         │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │ ┌───────────────────────────────────┐   │
│                                   │ │ PROCESS TABS:                       │   │
│ 📊 Quick Comparison:              │ │ [Traditional] [RAG] [RLM]          │   │
│                                   │ ├───────────────────────────────────┤   │
│ ┌─────────┬────────┬────────┐    │ │ [RLM selected]                      │   │
│ │Approach │ Tokens │ Cost   │    │ │                                     │   │
│ ├─────────┼────────┼────────┤    │ │ 🧭 Router: Classified as DOC query │   │
│ │Trad     │113.9K  │ $0.017 │    │ │ 🔍 Search: "governance readiness"  │   │
│ │RAG      │ 45.2K  │ $0.010 │    │ │    → Found 12 docs                  │   │
│ │🏆 RLM   │ 21.4K  │ $0.004 │    │ │ 📖 Peek: Deloitte_2024 pg 42-45    │   │
│ └─────────┴────────┴────────┘    │ │ 📖 Peek: KPMG_2024 pg 18-21        │   │
│                                   │ │ 🧠 Sub-call: Extract positions     │   │
│ See tabs for other responses →    │ │ ✅ Answer: Synthesized response    │   │
│                                   │ │                                     │   │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │ │ Subtabs: [Code] [Output] [Stats]  │   │
│                                   │ └───────────────────────────────────┘   │
│ Tabs: [🏆Winner] [📚All Responses] │                                        │
│       [📊Metrics] [🔬Analysis]     │                                        │
│                                   │                                         │
│ > [Input box]                     │                                         │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

### Pros:
- **Context visualization shows what each approach "saw"**
- Winner is primary but other responses accessible via tabs
- Process explorer shows step-by-step execution
- Visual indication of corpus coverage with progress bars
- Makes it immediately clear why RLM can be more efficient

### Cons:
- More complex to implement
- Requires calculating and tracking document usage per approach

---

## Recommendation

**Option 4 (Context-First Design)** or a hybrid of **Option 1 + 4** is recommended because:

1. ✅ **Context visualization** is key to understanding RLM vs Traditional vs RAG
2. ✅ Winner is prominently displayed (addresses primary requirement)
3. ✅ Comparison table is compact but informative
4. ✅ Process details are available but not overwhelming
5. ✅ The visual corpus coverage bars make it instantly clear why RLM wins
6. ✅ Educational value: shows the "explore all vs explore targeted" difference

### Hybrid Option 1+4 Features:
- Left panel: Winner response (from Option 1)
- Right panel top: Context visualization (from Option 4)
- Right panel bottom: Process tabs with subtabs
- Always visible comparison table
- Clear winner indication

---

## Implementation Considerations

### Tab Structure Recommendations:

#### Main Tabs:
- **🏆 Winner** - Shows winning approach output
- **📊 Comparison** - Side-by-side or table view
- **🐌 Traditional** - Full traditional output + process
- **📚 RAG** - Full RAG output + process
- **🚀 RLM** - Full RLM output + process
- **📈 Metrics** - Detailed evaluation metrics

#### Subtabs (per approach):
- **📝 Output** - Final response text
- **⚙️ Process** - Step-by-step execution
- **📈 Stats** - Token counts, costs, timing
- **🔍 Context** - What docs/chunks were used

### Data to Track:
- Winner determination (by tokens, cost, or quality score)
- Document coverage per approach (for visualization)
- Step-by-step process logs
- Comparative metrics (semantic similarity, citation overlap, etc.)

### Visual Elements:
- Progress bars for context usage
- Color coding (green=winner, yellow=runner-up, red=truncated)
- Icons for each approach (🐌, 📚, 🚀)
- Syntax highlighting for code blocks in process view

---

## Next Steps

1. Choose preferred layout (or request variations)
2. Design detailed component structure
3. Implement tab/subtab navigation
4. Add context visualization tracking
5. Update comparison display logic
6. Test with real eval results
