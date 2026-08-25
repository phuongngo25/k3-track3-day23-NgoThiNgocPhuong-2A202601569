# Day 08 Lab — Generated Metrics Report

> Auto-generated from `outputs/metrics.json` by `render_report()`. Do not edit by hand;
> re-run `make run-scenarios`. The narrative report is `reports/lab_report.md`.

## 1. Summary

| Metric | Value |
|---|---:|
| Total scenarios | 12 |
| Success rate | 100.00% |
| Avg nodes visited | 6.42 |
| Total retries | 5 |
| Total interrupts (HITL) | 3 |
| Resume / crash-recovery verified | yes |

## 2. Per-scenario results

| Scenario | Expected | Actual | OK | Nodes | Retries | Interrupts | Appr. req. | Appr. seen | Latency (ms) | Errors |
|---|---|---|:--:|---:|---:|---:|:--:|:--:|---:|---:|
| `S01_simple` | simple | simple | ✅ | 4 | 0 | 0 | no | no | 3130 | 0 |
| `S02_tool` | tool | tool | ✅ | 6 | 0 | 0 | no | no | 3484 | 0 |
| `S03_missing` | missing_info | missing_info | ✅ | 4 | 0 | 0 | no | no | 2738 | 0 |
| `S04_risky` | risky | risky | ✅ | 8 | 0 | 1 | yes | yes | 3253 | 0 |
| `S05_error` | error | error | ✅ | 10 | 2 | 0 | no | no | 3436 | 2 |
| `S06_delete` | risky | risky | ✅ | 8 | 0 | 1 | yes | yes | 4709 | 0 |
| `S07_dead_letter` | error | error | ✅ | 5 | 1 | 0 | no | no | 1190 | 2 |
| `S08_cancel_sub` | risky | risky | ✅ | 8 | 0 | 1 | yes | yes | 37943 | 0 |
| `S09_howto_delete` | simple | simple | ✅ | 4 | 0 | 0 | no | no | 3171 | 0 |
| `S10_tracking` | tool | tool | ✅ | 6 | 0 | 0 | no | no | 3369 | 0 |
| `S11_vague` | missing_info | missing_info | ✅ | 4 | 0 | 0 | no | no | 2572 | 0 |
| `S12_service_down` | error | error | ✅ | 10 | 2 | 0 | no | no | 4058 | 2 |

## 3. What the numbers mean

- **Success rate 100%** (12/12). A scenario counts as a success only when the actual route equals the expected route *and* the run produced a `final_answer` or a `pending_question` — a correctly routed run that fell off the graph without output still fails.
- **Route distribution:** `error`×3, `missing_info`×2, `risky`×3, `simple`×2, `tool`×2
- **5 retry visits** across 3 scenario(s): `S05_error`(2), `S07_dead_letter`(1), `S12_service_down`(2). Retries are driven by `tool_node` failing deterministically while `attempt < 2` on the error route, so the count is reproducible rather than flaky.
- **3 approval/HITL visits** in 3 scenario(s): `S04_risky`, `S06_delete`, `S08_cancel_sub`. Every risky-route ticket passed through `risky_action → approval` before any side effect was executed — the `tool` node is only reachable on that path via an approved decision.
- **Average 6.4 nodes per scenario.** Short paths (`simple`, `missing_info`) visit 4 nodes; the `tool` path 6; the risky path 8 (`intake→classify→risky_action→approval→tool→evaluate→answer→finalize`); a fully retried error path is the longest because `tool`/`evaluate`/`retry` repeat.
- **resume_success = true** — a checkpointed run was interrupted and resumed from SQLite in a fresh process (see `outputs/resume_evidence.json`).
- **No failing scenarios** in this run.

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
