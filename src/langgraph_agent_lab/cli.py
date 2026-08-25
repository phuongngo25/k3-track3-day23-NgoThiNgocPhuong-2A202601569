"""CLI for the lab."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from .graph import RECURSION_LIMIT, build_graph, render_mermaid
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import AgentState, initial_state

app = typer.Typer(no_args_is_help=True)

RESUME_EVIDENCE_PATH = Path("outputs/resume_evidence.json")


def new_run_id() -> str:
    """Short id identifying one *invocation* of the CLI."""
    return uuid.uuid4().hex[:8]


def scope_thread(state: AgentState, run_id: str) -> AgentState:
    """Give this run its own checkpoint lineage.

    The scenario-derived `thread-<id>` from `initial_state` is stable across runs, so with a
    durable checkpointer a second run would *resume* the first one's thread — and the append-only
    reducers would concatenate the old audit trail onto the new one (doubling `nodes_visited` and
    `retry_count`). Suffixing a per-run id keeps each run a clean lineage while the scenario stays
    identifiable in the thread name.
    """
    state["thread_id"] = f"{state['thread_id']}-{run_id}"
    return state


def _run_config(state: AgentState) -> dict[str, Any]:
    """One thread_id per run: every run gets its own checkpoint lineage."""
    return {
        "configurable": {"thread_id": state["thread_id"]},
        "recursion_limit": RECURSION_LIMIT,
    }


def _resume_success() -> bool:
    """True only if a real resume run left verified evidence behind."""
    if not RESUME_EVIDENCE_PATH.exists():
        return False
    try:
        return bool(json.loads(RESUME_EVIDENCE_PATH.read_text(encoding="utf-8")).get("resumed_ok"))
    except (json.JSONDecodeError, OSError):
        return False


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    run_id = new_run_id()
    typer.echo(f"run_id={run_id}  checkpointer={cfg.get('checkpointer', 'memory')}\n")
    for scenario in scenarios:
        state = scope_thread(initial_state(scenario), run_id)
        run_config = _run_config(state)
        started = time.perf_counter()
        final_state = graph.invoke(state, config=run_config)
        latency_ms = int((time.perf_counter() - started) * 1000)
        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
                latency_ms=latency_ms,
            )
        )
        status = "ok " if metrics[-1].success else "FAIL"
        typer.echo(
            f"[{status}] {scenario.id:<16} expected={scenario.expected_route.value:<12}"
            f" actual={metrics[-1].actual_route!s:<12} nodes={metrics[-1].nodes_visited:>2}"
            f" retries={metrics[-1].retry_count} {latency_ms}ms"
        )
    report = summarize_metrics(metrics, resume_success=_resume_success())
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(
        f"\nWrote metrics to {output} — success_rate={report.success_rate:.2%}, "
        f"retries={report.total_retries}, interrupts={report.total_interrupts}"
    )


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("diagram")
def diagram(
    output: Annotated[Path, typer.Option("--output")] = Path("docs/graph.mmd"),
) -> None:
    """Extension: export the compiled graph as a Mermaid diagram."""
    mermaid = render_mermaid()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(mermaid, encoding="utf-8")
    typer.echo(mermaid)
    typer.echo(f"\nWrote diagram to {output}")


@app.command("history")
def history(
    scenario_id: Annotated[str, typer.Option("--scenario-id")] = "S04_risky",
    config: Annotated[Path, typer.Option("--config")] = Path("configs/lab.yaml"),
) -> None:
    """Persistence evidence: replay the checkpoint history of one thread (time travel)."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = {s.id: s for s in load_scenarios(cfg["scenarios_path"])}
    if scenario_id not in scenarios:
        raise typer.BadParameter(f"Unknown scenario {scenario_id}. Have: {sorted(scenarios)}")

    checkpointer = build_checkpointer("sqlite", cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    state = scope_thread(initial_state(scenarios[scenario_id]), new_run_id())
    run_config = _run_config(state)
    graph.invoke(state, config=run_config)

    snapshots = list(graph.get_state_history(run_config))
    typer.echo(f"thread_id={state['thread_id']}  checkpoints={len(snapshots)}")
    for snap in reversed(snapshots):
        step = snap.metadata.get("step") if snap.metadata else None
        nxt = ",".join(snap.next) or "END"
        route = snap.values.get("route", "")
        attempt = snap.values.get("attempt", 0)
        typer.echo(
            f"  step={str(step):>3}  next={nxt:<14} route={route:<12} attempt={attempt} "
            f"checkpoint=…{snap.config['configurable']['checkpoint_id'][-8:]}"
        )


@app.command("demo-resume")
def demo_resume(
    scenario_id: Annotated[str, typer.Option("--scenario-id")] = "S04_risky",
    config: Annotated[Path, typer.Option("--config")] = Path("configs/lab.yaml"),
    approve: Annotated[bool, typer.Option("--approve/--reject")] = True,
) -> None:
    """Extension: real HITL interrupt + crash-resume from a SQLite checkpoint.

    Phase 1 runs the graph with ``LANGGRAPH_INTERRUPT=true`` until ``approval_node`` pauses it.
    Phase 2 throws that graph object away, builds a *fresh* graph over the same SQLite file, and
    resumes the same ``thread_id`` with a human decision — which is exactly what a process restart
    after a crash would do.
    """
    from langgraph.types import Command

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = {s.id: s for s in load_scenarios(cfg["scenarios_path"])}
    if scenario_id not in scenarios:
        raise typer.BadParameter(f"Unknown scenario {scenario_id}. Have: {sorted(scenarios)}")
    scenario = scenarios[scenario_id]

    db = cfg.get("database_url") or "outputs/checkpoints.sqlite"
    os.environ["LANGGRAPH_INTERRUPT"] = "true"

    # ── phase 1: run until the human gate ─────────────────────────────
    graph_a = build_graph(checkpointer=build_checkpointer("sqlite", db))
    state = initial_state(scenario)
    state["thread_id"] = f"resume-{scenario.id}-{new_run_id()}"
    run_config = _run_config(state)
    paused = graph_a.invoke(state, config=run_config)
    interrupts = paused.get("__interrupt__") or ()
    snapshot_a = graph_a.get_state(run_config)
    typer.echo(
        f"phase 1: paused at next={snapshot_a.next} with "
        f"{len(interrupts)} interrupt(s), proposed_action="
        f"{(snapshot_a.values.get('proposed_action') or '')[:70]!r}"
    )
    del graph_a  # simulate the process dying: nothing but SQLite survives

    # ── phase 2: fresh graph, same DB, same thread_id ─────────────────
    graph_b = build_graph(checkpointer=build_checkpointer("sqlite", db))
    recovered = graph_b.get_state(run_config)
    typer.echo(f"phase 2: recovered from disk, next={recovered.next}")
    final = graph_b.invoke(
        Command(
            resume={
                "approved": approve,
                "reviewer": "human-operator",
                "comment": "resumed after restart",
            }
        ),
        config=run_config,
    )
    checkpoints = len(list(graph_b.get_state_history(run_config)))

    resumed_ok = bool(final.get("final_answer")) and bool(final.get("approval"))
    evidence = {
        "scenario_id": scenario.id,
        "thread_id": run_config["configurable"]["thread_id"],
        "database": db,
        "paused_at": list(snapshot_a.next),
        "interrupt_payload": [str(i.value) for i in interrupts],
        "recovered_next_after_restart": list(recovered.next),
        "approval": final.get("approval"),
        "checkpoints_written": checkpoints,
        "final_answer": final.get("final_answer"),
        "nodes_visited": [e.get("node") for e in final.get("events", [])],
        "resumed_ok": resumed_ok,
    }
    RESUME_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESUME_EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(
        f"phase 2: resumed and finished — {checkpoints} checkpoints, "
        f"approved={(final.get('approval') or {}).get('approved')}"
    )
    typer.echo(f"Evidence written to {RESUME_EVIDENCE_PATH} (resumed_ok={resumed_ok})")


if __name__ == "__main__":
    app()
