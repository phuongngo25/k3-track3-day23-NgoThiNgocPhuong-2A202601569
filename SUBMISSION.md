# Submission — Day 08 LangGraph Agent Lab

**Ngô Thị Ngọc Phượng — 2A202601569**

The narrative report is [reports/lab_report.md](reports/lab_report.md). Auto-generated metrics
live in [reports/lab_report_generated.md](reports/lab_report_generated.md).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,google,sqlite]'

cp .env.example .env          # then paste your key into .env
#   GEMINI_API_KEY=AIza...    (or OPENAI_API_KEY / ANTHROPIC_API_KEY)

make lint typecheck test      # 57 offline tests in ~0.2s (+6 live-LLM smoke tests with a key)
make run-scenarios            # -> outputs/metrics.json + reports/lab_report_generated.md
make grade-local              # schema validation
```

Extensions:

```bash
make diagram        # docs/graph.mmd — exported from the compiled graph
make history        # checkpoint replay of the retry loop (time travel)
make demo-resume    # interrupt() + crash-resume from SQLite -> outputs/resume_evidence.json
```

## Where each rubric item lives

| Rubric category | Pts | Where |
|---|---:|---|
| Architecture & state schema | 15 | [state.py](src/langgraph_agent_lab/state.py) — 5 added fields, documented overwrite vs append reducers; report §3 |
| Graph construction & wiring | 15 | [graph.py](src/langgraph_agent_lab/graph.py) — 11 nodes, 4 conditional edges with explicit path maps, node names shared with [routing.py](src/langgraph_agent_lab/routing.py) |
| LLM integration | 15 | [nodes.py](src/langgraph_agent_lab/nodes.py) — `classify_node` (`.with_structured_output(IntentClassification)`), `answer_node` (grounded), `evaluate_node` (LLM-as-judge, bonus) |
| Graph behavior | 20 | 12/12 scenarios route correctly; bounded retry (`route_after_retry`); HITL gate; every route ends `finalize → END` |
| Persistence & recovery | 10 | [persistence.py](src/langgraph_agent_lab/persistence.py) SQLite+WAL; per-run `thread_id`; `make history`; `make demo-resume` → [outputs/resume_evidence.json](outputs/resume_evidence.json) |
| Metrics & tests | 15 | [outputs/metrics.json](outputs/metrics.json) validates; 57 offline tests pass; 38 written for this lab |
| Report & demo | 10 | [reports/lab_report.md](reports/lab_report.md) — architecture, metrics, 8 failure modes (3 found by running it for real), improvement plan |

## Demo script (2 minutes)

1. **One route** — `make history` shows `S05_error` climbing `attempt` 0→1→2 through the
   `retry → tool → evaluate` cycle and exiting the moment `evaluate` returns `success`.
2. **One failure mode** — `S07_dead_letter` has `max_attempts: 1`, so the first retry exhausts the
   budget and `dead_letter_node` escalates instead of looping. `route` stays `error` because the
   *classification* was right; only the outcome changed.
3. **The HITL guarantee** — `risky_action_node` performs no side effect, its only edge is to
   `approval`, and `route_after_approval` fails closed on a missing/falsy decision.
   `make demo-resume --reject` shows the rejected path never touching `tool`.

## Result

`outputs/metrics.json` was generated with **`gemini-3.5-flash-lite` actually driving all four
LLM-backed nodes** — **12/12 (100%), zero scenarios on a fallback path**, 5 retries, 3 HITL
interrupts, `resume_success: true`, median latency 3.3 s.

Note on the model: the starter's `gemini-2.5-flash` default is retired for new API keys, so
`llm.py` now defaults to `gemini-3.6-flash`, and `.env` pins `gemini-3.5-flash-lite` because the
free tier allows only **20 requests/day/model** while one 12-scenario run needs ~30. If a run
reports `*_llm_unavailable` with a 429, switch `LLM_MODEL` in `.env` to another Gemini model.

### Three bugs the real LLM found that the offline run hid

The offline run scored 100% while hiding all three. Details in report §5.5–5.7.

1. **The mock tool answered the wrong question.** Its success payload for an error ticket was a
   shipping lookup. The substring evaluator accepted it; the LLM judge rejected it and
   dead-lettered the scenario. The judge was right.
2. **`.content` is not the text.** Reasoning models return content as a *list of blocks*, so
   `.content.strip()` raised `AttributeError` and silently disabled `answer_node` and
   `ask_clarification_node` on 10 of 12 scenarios — while the success rate stayed at 100%.
3. **A status string leaked into a prompt, masking a double-executed side effect.**
   `proposed_action` carried "Not executed yet — awaiting human approval", which `tool_node`
   interpolated into "executed approved action '…not executed yet…'". The judge rejected the
   contradiction every time, and each retry **re-ran the account deletion — four times.**
   `tool_node` is now idempotent.

The lesson I would defend in the demo: an offline suite proves the graph is wired correctly and
proves almost nothing about whether the agent works.
