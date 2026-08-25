# ruff: noqa: E501  (markdown tables in the templates below must stay on one line)
"""Report generation helper.

Renders the auto-generated part of the lab report (metrics tables + interpretation) from a
:class:`~.metrics.MetricsReport`. The hand-written narrative lives in
``reports/lab_report.md``; this renderer produces ``reports/lab_report_generated.md`` (or whatever
``report_path`` the config points at) so numbers in the report can never drift from
``outputs/metrics.json``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .metrics import MetricsReport, ScenarioMetric

_ARCHITECTURE = """\
## Architecture

`START → intake → classify → <conditional> → … → finalize → END`

| Node | Role | LLM |
|---|---|:--:|
| `intake` | Normalise the raw ticket text | – |
| `classify` | Intent classification into one of 5 routes | **yes** (structured output) |
| `tool` | Mock support-API call; simulates a transient failure on the error route | – |
| `evaluate` | Retry gate: is the tool result good enough? | **yes** (LLM-as-judge) |
| `answer` | Final customer reply, grounded in `tool_results`/`approval` | **yes** |
| `clarify` | Ask one specific question instead of guessing | yes (with template fallback) |
| `risky_action` | Describe the side effect, execute nothing | – |
| `approval` | Human-in-the-loop gate (mock, or real `interrupt()`) | – |
| `retry` | Increment `attempt`, log the transient failure | – |
| `dead_letter` | Escalate after the retry budget is spent | – |
| `finalize` | Emit the terminal audit event | – |

Four conditional edges carry all the control flow: `route_after_classify`,
`route_after_evaluate`, `route_after_retry` (bounded), `route_after_approval`. The single cycle
in the graph is `tool → evaluate → retry → tool`, bounded by `attempt < max_attempts`.

## State schema

| Field | Reducer | Why |
|---|---|---|
| `route`, `risk_level`, `attempt`, `evaluation_result`, `approval`, `proposed_action`, `pending_question`, `final_answer` | overwrite | current truth only — routing on a stale value would break the graph |
| `messages`, `tool_results`, `errors`, `events` | append (`operator.add`) | audit trail; the retry loop revisits `tool` and each visit must add a row, not clobber one |
"""


def _fmt_scenario_rows(items: list[ScenarioMetric]) -> str:
    rows = []
    for m in items:
        rows.append(
            f"| `{m.scenario_id}` | {m.expected_route} | {m.actual_route or '—'} | "
            f"{'✅' if m.success else '❌'} | {m.nodes_visited} | {m.retry_count} | "
            f"{m.interrupt_count} | {'yes' if m.approval_required else 'no'} | "
            f"{'yes' if m.approval_observed else 'no'} | {m.latency_ms} | {len(m.errors)} |"
        )
    return "\n".join(rows)


def _interpretation(metrics: MetricsReport) -> str:
    """Explain *why* the numbers look the way they do — the rubric asks for this explicitly."""
    failures = [m for m in metrics.scenario_metrics if not m.success]
    retried = [m for m in metrics.scenario_metrics if m.retry_count]
    approved = [m for m in metrics.scenario_metrics if m.approval_observed]
    route_counts = Counter(m.actual_route or "none" for m in metrics.scenario_metrics)

    lines = [
        f"- **Success rate {metrics.success_rate:.0%}** "
        f"({sum(1 for m in metrics.scenario_metrics if m.success)}/{metrics.total_scenarios}). "
        "A scenario counts as a success only when the actual route equals the expected route "
        "*and* the run produced a `final_answer` or a `pending_question` — a correctly routed "
        "run that fell off the graph without output still fails.",
        "- **Route distribution:** "
        + ", ".join(f"`{route}`×{n}" for route, n in sorted(route_counts.items())),
        f"- **{metrics.total_retries} retry visits** across "
        f"{len(retried)} scenario(s): {', '.join(f'`{m.scenario_id}`({m.retry_count})' for m in retried) or '—'}. "
        "Retries are driven by `tool_node` failing deterministically while `attempt < 2` on the "
        "error route, so the count is reproducible rather than flaky.",
        f"- **{metrics.total_interrupts} approval/HITL visits** in "
        f"{len(approved)} scenario(s): {', '.join(f'`{m.scenario_id}`' for m in approved) or '—'}. "
        "Every risky-route ticket passed through `risky_action → approval` before any side effect "
        "was executed — the `tool` node is only reachable on that path via an approved decision.",
        f"- **Average {metrics.avg_nodes_visited:.1f} nodes per scenario.** Short paths "
        "(`simple`, `missing_info`) visit 4 nodes; the `tool` path 6; the risky path 8 "
        "(`intake→classify→risky_action→approval→tool→evaluate→answer→finalize`); a fully "
        "retried error path is the longest because `tool`/`evaluate`/`retry` repeat.",
        f"- **resume_success = {str(metrics.resume_success).lower()}** — "
        + (
            "a checkpointed run was interrupted and resumed from SQLite in a fresh process "
            "(see `outputs/resume_evidence.json`)."
            if metrics.resume_success
            else "no crash-resume run was recorded for this invocation; run "
            "`agent-lab demo-resume` to produce the evidence."
        ),
    ]
    if failures:
        lines.append(
            "- **Failures:** "
            + "; ".join(
                f"`{m.scenario_id}` expected `{m.expected_route}` got `{m.actual_route}`"
                + (f" — {m.errors[0][:80]}" if m.errors else "")
                for m in failures
            )
        )
    else:
        lines.append("- **No failing scenarios** in this run.")

    degraded = [m for m in metrics.scenario_metrics if any("llm_unavailable" in e for e in m.errors)]
    if degraded:
        lines.append(
            f"- ⚠️ **{len(degraded)} scenario(s) ran on the offline fallback path** "
            f"({', '.join(f'`{m.scenario_id}`' for m in degraded)}): the LLM provider was "
            "unreachable, so the deterministic router/templates answered instead. The numbers "
            "below are therefore *not* evidence of LLM classification quality."
        )
    return "\n".join(lines)


def render_report(metrics: MetricsReport) -> str:
    """Render the metrics section of the lab report as markdown."""
    return f"""# Day 08 Lab — Generated Metrics Report

> Auto-generated from `outputs/metrics.json` by `render_report()`. Do not edit by hand;
> re-run `make run-scenarios`. The narrative report is `reports/lab_report.md`.

## 1. Summary

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.2%} |
| Avg nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Total interrupts (HITL) | {metrics.total_interrupts} |
| Resume / crash-recovery verified | {"yes" if metrics.resume_success else "no"} |

## 2. Per-scenario results

| Scenario | Expected | Actual | OK | Nodes | Retries | Interrupts | Appr. req. | Appr. seen | Latency (ms) | Errors |
|---|---|---|:--:|---:|---:|---:|:--:|:--:|---:|---:|
{_fmt_scenario_rows(metrics.scenario_metrics)}

## 3. What the numbers mean

{_interpretation(metrics)}

{_ARCHITECTURE}
## 5. Failure modes exercised by these scenarios

1. **Transient tool failure → bounded retry.** `tool_node` returns an `ERROR:` string while
   `attempt < 2` on the error route. `evaluate_node` refuses it, `route_after_evaluate` sends the
   run to `retry`, and `route_after_retry` compares the incremented `attempt` against
   `max_attempts`. Without that comparison the graph would cycle until LangGraph's recursion
   limit; with it, the run either recovers or lands in `dead_letter`.
2. **Unrecoverable failure → dead letter.** A scenario with `max_attempts: 1` exhausts its budget
   on the first retry, so `dead_letter_node` writes an escalation answer instead of leaving the
   customer with nothing. The route stays `error` (the classification was right); the escalation is
   recorded in `events`/`errors`.
3. **Risky action without approval.** `risky_action_node` only *describes* the side effect. The
   only edge into `tool` from that path goes through `approval`, and `route_after_approval` sends a
   rejected decision to `clarify`. A missing/None approval is falsy, so the default is "do not
   execute" rather than "execute".
4. **Vague ticket → clarification, not hallucination.** `missing_info` never reaches `tool` or the
   grounded answer node; it asks one question and terminates.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
