# Plan: AI chart notes

**Branch:** `feature/ai-chart-notes` · **Status:** built (steps 1–4 and 6–8); step 5, the live OpenRouter run, still to do · **Written:** 2026-11-17

> **As built.** The implementation and the decisions made while building it are in [DESIGN.md section 7b](DESIGN.md#7b-ai-chart-notes-ga4appannotatepy-charts_draw_notes-uichart_notes_). Differences from this plan:
>
> - `ui.chart_notes_controls(anchors, rule_notes=None, *, key_prefix)`: the chart ID travels inside `annotate.AnchorSet`, which also holds the scope, the empty-chart caption and the anchor → product key map for Inspect (never sent to the model).
> - Raw/Adjusted is not purely a display control here: quadrants are classified on the plotted metric. On both snapshots it gives the same coded products, so the anchors and the cache key do not change; if it ever re-codes products, the notes are rewritten, which is correct.
> - "Write AI notes" also turns the toggle on.
> - Heatmap anchor IDs come from `charts.heatmap_cells(matrix, codes=...)`, so `device_heatmap` takes `codes` too; `R<row>` is the 1-based row number. The 30-anchor cap keeps strong-lean products' cells first, so they are never cut.
> - The Holiday Shift scope carries `significance_level: 0.05`, so a note may say "p < 0.05"; rule-based bar notes write the lift rounded like the bar's own label.
> - Validation keeps the first four notes before the number check, as in 3.3, and shortens text at a word boundary so a number is never cut.

## 1. Goal

Today the AI writes one paragraph (the verdict) under two pages. This feature puts AI notes **on the charts themselves**: numbered callouts that point at the bubbles, bars or cells that matter for the homepage decision.

The split of work is the safety rule:

- **The AI chooses what to point at and what to say**, from a list of anchors the app offers.
- **The app decides where the note goes** (it owns the chart coordinates).
- **The app checks what the note says**: the anchor must exist and every number must come from that anchor's data.

The three charts:

| Page | Chart | Function |
|---|---|---|
| Feature Picks | quadrant bubble scatter | `charts.quadrant_scatter` |
| Holiday Shift | conversion by category, focus vs base | `charts.period_bars` |
| Device Split | candidates × device heatmap | `charts.device_heatmap` |

Rule-based notes work without an API key, as the rule-based verdict does today.

**Not in v1:** the other charts (daily line, Trends small multiples, transition matrix, Slot Plan candidates, relative-risk dot plot), auto-writing on filter change, and free-text questions.

## 2. Before step 1: the uncommitted work on `main` (done)

This plan was written against work that was uncommitted on `main` at the time (`ga4app/charts.py`, `ui.py`, `narrative.py`, the views, tests, README, DESIGN). That work is now commit `1414094` on `main`, and this branch is rebased onto it, so the branch holds only this plan. Start at step 1.

## 3. Design

### 3.1 Anchors: the only things a note can point at

Each chart gets an anchor builder in `ga4app/annotate.py`. An anchor is an ID plus the facts behind it. The model can only choose anchor IDs from that list.

| Chart | Anchor ID | Facts | Never offered |
|---|---|---|---|
| Scatter | product code: `G1`, `W2`, `X1` | the `narrative._entry` fields (name, category, views, purchases, `conversion_pct`, `adjusted_conversion_pct`, `interval_pct`, `vs_avg`, `confidence`, `lift_per_1k` or `forgone_purchases`, `views_undercount`, `device_lean`) plus `quadrant` and `cannibal_pair` | uncoded and ineligible products |
| Holiday bars | `Apparel·cyber_week` (group · period) | window label, `conversion_pct`, `base_conversion_pct`, 95% `interval_pct`, `lift_pct`, `p_value`, `significant`, `lift_vs_rest_pct`, `p_value_vs_rest` | base-window bars; `low_sample` bars (not tested) |
| Heatmap | `G1·mobile` (code, or `R<row>` for an uncoded row, · device) | name, device, `conversion_pct`, `x_usual` (the cell's colour value as a multiple), views, purchases, the product's `lean` | sparse tablet cells (the "–" ones); cells within ±19% of usual (abs(log2) < 0.25) unless the product has a strong lean |

Rules for all anchors:

- **Anchors depend only on the data and on data controls, never on display controls.**
  - Data controls: sidebar filters, the high-traffic cutoff, the Holiday Shift comparison, and the Device Split candidate list.
  - Display controls: Raw/Adjusted axis, Log/Linear, bubble size and colour.
  - Switching a display control re-places the notes without a new AI call (the cache key is built from the anchors).
- **Facts are stored in display units and rounded** the way `narrative.py` rounds them (percent with 2 decimals, counts as integers, multiples with 2 decimals), so the number check compares like with like.
- **Scatter anchors reuse the verdict payload.** Feature Picks already builds it with `narrative.build_payload`. `scatter_anchors(payload, products, conflict_keys)` takes the winners, hidden gems and wasted exposure entries (up to 15), adds `quadrant`, and adds `cannibal_pair` from `data.conflicts`.
- **The heatmap's numbers must match what the cells show.**
  - Move the per-cell computation (row order, `rel`, `sparse`) out of `device_heatmap` into a public `charts.heatmap_cells(matrix)`.
  - The chart and the anchor builder both use it.
  - Cap at `HEATMAP_ANCHORS_MAX = 30` anchors, keeping the strongest abs(log2(`rel`)).

The user message is compact JSON:

```json
{"chart": "feature_picks_scatter",
 "question": "Which products earn a holiday homepage slot, and which waste one?",
 "scope": {"window": "Cyber Week", "filters": "...", "avg_conversion_pct": 0.71, "views_threshold": 2100},
 "anchors": [{"id": "G1", "name": "Google Tee", "quadrant": "Hidden gems", "views": 410, "...": "..."}]}
```

### 3.2 The model call

- **Module:** `ga4app/annotate.py` with its own `NOTES_PROMPT_VERSION = "notes-2026-11-17.1"`.
- **System prompt outline:**
  1. Role: the same merchandising analyst as the verdict. You are annotating one chart for the Head of Merchandising.
  2. Pick **at most 4 anchors** that matter most for the homepage-slot decision.
  3. For each anchor, return:
     - `anchor`
     - `kind`: `highlight` or `caution`
     - `label`: 5 words or fewer, for the chart
     - `detail`: 25 words or fewer, a sentence for the list and hover
  4. Copy numbers exactly from that anchor's facts or the scope. Never compute new numbers (no differences, sums or ratios).
  5. If any anchor carries `views_undercount`, `cannibal_pair` or `confidence: "uncertain"`, include at least one `caution`.
  6. Name a product in `detail` with its code in brackets, as in the verdict.
  7. Reply with JSON only: `{"notes": [...]}`.
- **Transport:** reuse `llm.stream_chat` unchanged, with the same model, fallbacks and key as the verdict (`state.ai_settings()`), joining the `text` events. Don't send `response_format`, because structured-output support varies by model on OpenRouter. Parse leniently instead (3.3).

### 3.3 Validation (`annotate.validate_notes`)

Every note goes through these steps, rule-based ones included:

1. **Parse.** Remove ```` ``` ```` fences, take the first `{...}` block and run `json.loads`. If that fails, show the rule-based notes and the warning "AI notes could not be read".
2. **Shape.** Keep dicts whose `anchor`, `label` and `detail` are strings. An unknown `kind` becomes `highlight`.
3. **Anchor.** Drop notes whose anchor is not in the list, remove duplicates by anchor, and keep the first `NOTES_MAX = 4`.
4. **Number check.**
   - Remove product codes and the anchor ID from the text, then find every number (`\d[\d,]*(?:\.\d+)?`).
   - Each number `x`, written with `d` decimals, must be a rounding of some value `v` in the allowed set: abs(x − v) ≤ 0.5 · 10^−d.
   - The allowed set is every number in this anchor's facts and in the scope, including numbers inside strings such as window dates.
   - Integers 1–5 are allowed as counting words ("top 3").
   - Signs are compared by absolute value, because text often says "drops 23%". Direction words are not checked.
   - A note with any unmatched number is dropped, and the reason is recorded.
5. **Length.** Shorten `label` to `NOTE_LABEL_MAX = 40` characters and `detail` to `NOTE_DETAIL_MAX = 200`, ending with "...".
6. **Escaping happens at render time, not in storage:** `charts._esc` for Plotly text (annotations accept HTML) and `narrative.md_safe` for the list.
7. **No survivors.** Show the rule-based notes with a warning that says how many notes were dropped and why.

It returns a `NotesResult` with `notes`, `source` (`"ai"` or `"rule"`), `model`, `usage`, `dropped` (`[(anchor, reason)]`) and `error`.

### 3.4 Rule-based notes (`annotate.rule_based_notes`)

These are deterministic and built from the same anchors. They are the default and the fallback.

- **Scatter:**
  - `G1` highlight, "Best hidden gem": "{name} [G1] converts at {vs_avg}× the average on {views} views."
  - `W1` highlight, "Top winner".
  - `X1` caution, "Most purchases forgone": "{forgone_purchases} purchases forgone."
  - The first featured pick with `views_undercount`: caution, "Views may be under-counted".
- **Holiday bars:** the largest significant rise and the largest significant drop. If nothing is significant, show no notes and the caption "No significant shift to point out."
- **Heatmap:** the favoured-device cell of up to 2 strong-lean products, then the strongest remaining cell.

A test checks that every rule-based note passes `validate_notes`, which keeps the number check honest.

### 3.5 Placement (in `charts.py`)

- **New type:** `ChartNote(anchor, kind, label, detail, n)` in `ga4app/types.py`. `n` is the 1-based number shown on the chart and in the list.
- **Shared helper `_draw_notes(fig, placed)`.** Each note is `fig.add_annotation(name=f"note:{source}", ...)`:
  - text `<b>{n}</b> {label}`, with `hovertext=detail`, an arrow, `bgcolor=WHITE_BOX` and font `INK`
  - highlight notes get an `INK` border
  - caution notes get a **coral border** plus a "!" prefix (the palette says coral is never used for text)
- **Scatter (`quadrant_scatter(..., notes=None)`):**
  - Find the drawn point whose code equals the anchor. Skip notes whose point is not drawn.
  - `x = log10(views)`, because annotations on a log axis take log10 values (see the existing comment above the threshold annotation).
  - `y` = the **drawn** value (`y_plot`, log10 when the y axis is log), so a bubble clipped to the top or bottom edge still gets its arrow.
  - Arrow offset points toward the plot's interior: `ax = -40` in the right half, else `+40`; `ay = +30` in the top third, else `-30`.
- **Bars (`period_bars(..., notes=None)`):**
  - Anchor `group·period` maps to the bar's numeric `x[j]` (already computed) and `y = tops[j]`.
  - Use `ay = -40` so the label sits above the existing lift label.
  - When notes are present, raise the y headroom from `1.18` to `1.35`.
- **Heatmap (`device_heatmap(..., notes=None)`):**
  - Draw a rectangle shape around the cell, coral for cautions and `INK` otherwise.
  - Put a small numbered badge in the cell's corner.
  - The label is not drawn (the grid is too dense). Label and detail go in the list under the chart.
- **Existing annotations keep their names and positions** (`badge:*`, `threshold:*`, `lift`, `lean`, `overflow`, `underflow`). With `notes=None`, every figure is unchanged.

### 3.6 Page UI (`ui.py`)

Streamlit draws top to bottom, so there are two helpers.

**`ui.chart_notes_controls(chart_id, anchors, rule_notes, *, key_prefix) -> NotesResult | None`, drawn above the chart:**

- A toggle, "Point out what matters", per page (`{P}notes_on`, **off by default**). When it's on, the rule-based notes show at once.
- A "Write AI notes" button, disabled when no model is set (same rule as the verdict button).
- **Cache:**
  - Key: `sha256(NOTES_PROMPT_VERSION, model, chart_id, canonical JSON of scope + anchors)`.
  - Stored in `st.session_state[state.NOTES_CACHE]` and in a shared `st.cache_resource` dict, as `verdict_block` does.
  - Only `source == "ai"` results are cached.
- **On click:** `with st.spinner("Writing notes…")`, then `annotate.ai_notes(...)`, then `validate_notes`. The call is not streamed: the chart is drawn once the notes are back, so no placeholder swapping is needed.

**`ui.chart_notes_list(result, *, inspect=False)`, drawn under the chart:**

- A caption: "AI notes · {model} · every number checked against the data", or "Rule-based notes from the numbers on this chart".
- A warning when notes were dropped (count and reasons) or the call failed (the reason, as for the verdict).
- A numbered list: **n** label: detail.
- With `inspect=True` (Feature Picks only, the one page with the product inspector), an **Inspect** button per product note that sets `state.INSPECT`.

**Accessibility:** add an `alt_extra` parameter to `ui.chart_with_table` so the note sentences join the chart's `alt` text without repeating them in the visible caption.

### 3.7 State, config and the fake transport

- **`state.py`:** `NOTES_CACHE = "notes_cache"`, set with `setdefault` in `init_state`.
- **`config.py`:**
  - `NOTES_MAX = 4`
  - `NOTE_LABEL_MAX = 40`, `NOTE_DETAIL_MAX = 200`
  - `HEATMAP_NOTE_MIN_LOG2 = 0.25`, `HEATMAP_ANCHORS_MAX = 30`
- **`llm.py`:**
  - `fake_transport(respond=None)` takes an optional function from the request body to text chunks. The default returns `FAKE_TEXT_CHUNKS`, so verdict tests are unchanged.
  - `_fake_sse(chunks)` takes the chunks as an argument.
- **`annotate.fake_respond(body)`** reads the anchors from the request's user message and returns notes for the first two. It uses no numbers, so it passes validation on any data.

### 3.8 Page wiring

- **`views/feature_picks.py`:**
  - Move the `conflict_keys` computation above the chart.
  - Build `annotate.scatter_anchors(payload, products, conflict_keys)` and call `ui.chart_notes_controls` before `charts.quadrant_scatter(..., notes=...)`.
  - Call `ui.chart_notes_list(result, inspect=True)` after `chart_with_table`.
- **`views/holiday_shift.py`:** build anchors from `groups` before `charts.period_bars(groups, notes=...)`. `_fade_low_sample` still wraps the figure.
- **`views/device_split.py`:** build anchors from `charts.heatmap_cells(matrix)`, `lean`, and codes from the page's `products` table, then call `charts.device_heatmap(matrix, lean, notes=...)`.

## 4. Files

| File | Change |
|---|---|
| `ga4app/annotate.py` (new) | anchor builders (3), prompt, `ai_notes`, `validate_notes`, number check, `rule_based_notes`, `fake_respond`, `cache_key` |
| `ga4app/types.py` | `ChartNote`, `NotesResult` |
| `ga4app/charts.py` | `_draw_notes`; `notes=` on the 3 chart functions; `heatmap_cells` taken out of `device_heatmap` |
| `ga4app/ui.py` | `chart_notes_controls`, `chart_notes_list`, `alt_extra` on `chart_with_table` |
| `ga4app/state.py` | `NOTES_CACHE` |
| `ga4app/config.py` | note limits |
| `ga4app/llm.py` | `fake_transport(respond=None)`, `_fake_sse(chunks)` |
| `views/feature_picks.py`, `views/holiday_shift.py`, `views/device_split.py` | wiring (a few lines each) |
| `tests/test_annotate.py` (new), `tests/test_charts.py`, `tests/test_llm.py`, `tests/test_app_pages.py`, `tests/test_pages_q3_q4.py` | tests (section 5) |
| `docs/DESIGN.md`, `README.md` | new design section, feature description, screenshot, "What is still missing" |

## 5. Tests

**`tests/test_annotate.py` (new):**

- Anchors:
  - scatter: only coded, eligible products; ties to the payload codes
  - holiday: no base-window or `low_sample` bars
  - heatmap: no sparse tablet cells; capped at 30; keeps strong-lean cells
  - the cache key is the same across Raw/Adjusted and Log/Linear
- Validation:
  - unknown anchor → dropped
  - duplicate anchors → one kept
  - more than 4 notes → trimmed to 4
  - unknown `kind` → `highlight`
  - fenced or prose-wrapped JSON → parsed
  - broken JSON → rule-based result with an error
- Number check:
  - `2.4` for 2.37 passes and `2.6` fails
  - a computed difference fails
  - "top 3" passes
  - `G1` in the text is ignored
  - `1,234` with a comma passes
  - "drops 23%" for a −23.4 lift passes
- Rule-based notes:
  - deterministic
  - pass `validate_notes` on the synthetic snapshot for all 3 charts
  - pass on the real snapshot (`real_data_dir` fixture)
- `ai_notes` with `fake_respond`: notes come back for the first two anchors.
- `ai_notes` errors:
  - `OpenRouterError` gives a rule-based result whose `error` holds the reason
  - an empty reply does the same

**`tests/test_charts.py`:**

- Scatter on a log x axis: the note annotation's `x == log10(views)`.
- A clipped bubble (Linear scale): the note's `y` is the drawn cap, not the actual rate.
- A bar note sits on the bar's `x[j]`, and the y range grows when notes are present.
- A heatmap note adds a cell rectangle and a badge.
- Note text with `<b>` or `&` is escaped.
- With `notes=None`, annotation names and counts match today's figures (a regression guard).

**`tests/test_llm.py`:** `fake_transport(respond=...)` returns the responder's chunks, and the default still returns `FAKE_TEXT_CHUNKS`.

**AppTest (`test_app_pages.py`, `test_pages_q3_q4.py`):**

- The toggle adds `note:rule` annotations to the figure and a numbered list.
- With `GA4APP_LLM_FAKE=1` and a key in `at.secrets`, clicking "Write AI notes" adds `note:ai` annotations and fills `notes_cache`.
- Switching Log/Linear afterwards makes no new call (a counting responder).
- Inspect sets `inspect_key`.

CI runs all of these through the existing `python -m pytest -q` step.

## 6. Build order

Each step is one commit, and the suite stays green after each.

| Step | Work | Done when |
|---|---|---|
| 0 | ~~Commit the work in progress to `main`, then rebase this branch onto it (section 2).~~ Done. | the branch has only this plan on top of `main` |
| 1 | ~~`ChartNote` type, `_draw_notes`, `notes=` on `quadrant_scatter`~~ Done. | placement tests pass; figures unchanged without notes |
| 2 | ~~`annotate.py`: scatter anchors, `rule_based_notes`, `validate_notes` with the number check~~ Done. | `test_annotate.py` scatter tests pass |
| 3 | ~~UI helpers and Feature Picks wiring, rule-based only~~ Done. | AppTest: the toggle draws notes and the list |
| 4 | ~~AI call: prompt, `ai_notes`, fake responder, cache, "Write AI notes" button~~ Done. | AppTest with the fake transport passes |
| 5 | **Manual, still to do:** live run against OpenRouter with Claude Opus 5.5 and one free model. Record the cost per call, how often notes are dropped and why, and the latency. Tune the prompt and bump `NOTES_PROMPT_VERSION`. No key was available while building; the table to fill is in DESIGN.md 7b. | numbers recorded in DESIGN.md; the cost note shown in the UI |
| 6 | ~~Holiday Shift: anchors, bar placement, wiring, tests~~ Done; the screenshot shows no overlap with the lift labels. | tests pass; screenshot checked for overlap with lift labels |
| 7 | ~~Device Split: `heatmap_cells` refactor, anchors, cell placement, wiring, tests~~ Done; the heatmap figure is byte-identical. | tests pass; heatmap colours unchanged |
| 8 | ~~Docs: DESIGN.md section "AI chart notes", README feature and screenshots, "What is still missing"~~ Done. | then open a PR to `main` |

Step 5 needs an `OPENROUTER_API_KEY`. The current verdict has not been run live either, so do both in the same session.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Clutter on the scatter (code labels, badges and notes together) | at most 4 notes, labels of 5 words or fewer, arrows pushed inward, notes off by default; check in screenshots |
| The number check drops too many notes | the warning lists the reasons; the prompt says to copy numbers exactly; tune in step 5 |
| The model replies in prose, not JSON (more likely on free models) | lenient parse, then rule-based fallback with a warning |
| Latency (a non-streamed call) | spinner; low reasoning effort as in the verdict; cache shared across sessions |
| Notes quoting numbers that differ from the chart | anchors use the chart's own numbers (`heatmap_cells`, the payload entries), with the same rounding |
| Log-axis coordinate mistakes | placement lives in the chart function and is tested on log axes |

## 8. Open questions for the team

1. Should notes be **on by default** (rule-based) or off until toggled? This plan says off.
2. Should "Auto-write when filters change" also write notes? This plan says no: it would triple the calls.
3. Should there be 4 notes per chart, or 3?
4. Should the Slot Plan candidates chart get notes in a later version?

## 9. Done when

- All three charts show rule-based notes without a key, and AI notes with one.
- Every AI note on screen has passed the anchor and number checks, and tests show the checks reject unknown anchors and invented numbers.
- Changing a display control (Raw/Adjusted, Log/Linear, bubble size, colour) never triggers a new AI call.
- Without notes, every existing chart is unchanged.
- CI passes; DESIGN.md and README are updated with screenshots; the live run in step 5 is recorded.
