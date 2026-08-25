"""Checkpointer construction and offline end-to-end graph behaviour.

No API key needed: the LLM nodes take their fallback path, which is exactly what CI exercises.
"""

import importlib.util

import pytest

from langgraph_agent_lab.graph import RECURSION_LIMIT, build_graph, render_mermaid
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("langgraph") is None, reason="langgraph not installed"
    ),
    # These assert graph *structure* (termination, retry bound, HITL ordering), which must hold
    # without a provider — and must not become slow or flaky when one is configured.
    pytest.mark.usefixtures("offline_llm"),
]


def test_memory_checkpointer():
    assert build_checkpointer("memory") is not None


def test_none_checkpointer():
    assert build_checkpointer("none") is None


def test_unknown_checkpointer_raises():
    with pytest.raises(ValueError, match="Unknown checkpointer"):
        build_checkpointer("redis")


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph.checkpoint.sqlite") is None,
    reason="langgraph-checkpoint-sqlite not installed",
)
def test_sqlite_checkpointer_writes_history(tmp_path):
    """A run against SQLite must leave an inspectable checkpoint lineage on disk."""
    db = tmp_path / "cp.sqlite"
    graph = build_graph(checkpointer=build_checkpointer("sqlite", str(db)))
    state = initial_state(Scenario(id="persist", query="How do I reset my password?",
                                   expected_route=Route.SIMPLE))
    config = {"configurable": {"thread_id": state["thread_id"]}, "recursion_limit": RECURSION_LIMIT}
    graph.invoke(state, config=config)

    assert db.exists()
    snapshots = list(graph.get_state_history(config))
    assert len(snapshots) > 1
    assert snapshots[0].next == ()  # newest snapshot is terminal


def test_sqlite_url_forms_are_accepted(tmp_path):
    from langgraph_agent_lab.persistence import _sqlite_path

    assert _sqlite_path(None).endswith(".sqlite")
    assert _sqlite_path("sqlite:///a/b.sqlite") == "a/b.sqlite"
    assert _sqlite_path("plain/path.sqlite") == "plain/path.sqlite"


def test_bounded_retry_terminates_offline():
    """The error route must finish without hitting the recursion limit, even with no LLM."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="bound", query="Timeout failure while processing request",
                        expected_route=Route.ERROR, max_attempts=3)
    state = initial_state(scenario)
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]},
                "recursion_limit": RECURSION_LIMIT},
    )
    assert result["final_answer"]
    assert result["attempt"] <= scenario.max_attempts
    assert [e for e in result["events"] if e["node"] == "finalize"]


def test_dead_letter_reached_when_budget_is_one():
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="dl", query="System failure cannot recover after multiple attempts",
                        expected_route=Route.ERROR, max_attempts=1)
    state = initial_state(scenario)
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]},
                "recursion_limit": RECURSION_LIMIT},
    )
    nodes = [e["node"] for e in result["events"]]
    assert "dead_letter" in nodes
    assert nodes[-1] == "finalize"
    assert result["route"] == Route.ERROR.value  # dead letter must not rewrite the route


def test_risky_route_cannot_reach_tool_without_approval():
    """Structural guarantee: no `tool` event may precede the `approval` event."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="hitl", query="Refund this customer and send confirmation email",
                        expected_route=Route.RISKY, requires_approval=True)
    state = initial_state(scenario)
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]},
                "recursion_limit": RECURSION_LIMIT},
    )
    nodes = [e["node"] for e in result["events"]]
    assert "approval" in nodes
    if "tool" in nodes:
        assert nodes.index("approval") < nodes.index("tool")


def test_all_nodes_present_in_mermaid_diagram():
    mermaid = render_mermaid()
    for node in ("intake", "classify", "tool", "evaluate", "answer", "clarify",
                 "risky_action", "approval", "retry", "dead_letter", "finalize"):
        assert node in mermaid
