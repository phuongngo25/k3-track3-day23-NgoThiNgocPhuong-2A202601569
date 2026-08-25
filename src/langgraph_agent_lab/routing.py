"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node. These strings
MUST match the node names registered in :mod:`.graph`; the node-name constants live here so the
graph and the routers cannot drift apart.
"""

from __future__ import annotations

from .state import AgentState, Route

# ── node names (single source of truth, imported by graph.py) ─────────
N_INTAKE = "intake"
N_CLASSIFY = "classify"
N_TOOL = "tool"
N_EVALUATE = "evaluate"
N_ANSWER = "answer"
N_CLARIFY = "clarify"
N_RISKY = "risky_action"
N_APPROVAL = "approval"
N_RETRY = "retry"
N_DEAD_LETTER = "dead_letter"
N_FINALIZE = "finalize"

#: classify route -> next node. Unknown routes fall through to `answer` so an unexpected
#: classifier output still terminates instead of hanging the graph.
ROUTE_TO_NODE: dict[str, str] = {
    Route.SIMPLE.value: N_ANSWER,
    Route.TOOL.value: N_TOOL,
    Route.MISSING_INFO.value: N_CLARIFY,
    Route.RISKY.value: N_RISKY,
    Route.ERROR.value: N_RETRY,
}


def route_after_classify(state: AgentState) -> str:
    """Map the classified route to the next graph node (default: ``answer``)."""
    return ROUTE_TO_NODE.get(state.get("route", ""), N_ANSWER)


def route_after_evaluate(state: AgentState) -> str:
    """The 'done?' check that creates the retry loop — LangGraph's edge over a linear chain."""
    if state.get("evaluation_result") == "needs_retry":
        return N_RETRY
    return N_ANSWER


def route_after_retry(state: AgentState) -> str:
    """Bounded retry: try the tool again while budget remains, otherwise dead-letter.

    ``attempt`` has already been incremented by ``retry_or_fallback_node``, so the comparison is
    against the post-increment value: attempt 3 of max 3 means the budget is spent.
    """
    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 3) or 3)
    return N_TOOL if attempt < max_attempts else N_DEAD_LETTER


def route_after_approval(state: AgentState) -> str:
    """Execute the action only on an explicit approval; otherwise go back to the customer."""
    approval = state.get("approval") or {}
    return N_TOOL if approval.get("approved") else N_CLARIFY
