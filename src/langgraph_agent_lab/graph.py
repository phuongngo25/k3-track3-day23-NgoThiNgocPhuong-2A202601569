"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit
tests that check schema/metrics can run even if the graph wiring is still being debugged.

Topology
--------
::

    START → intake → classify → ┬ simple       → answer → finalize → END
                                ├ tool         → tool → evaluate ─┬ ok    → answer → finalize
                                │                                 └ retry → retry ─┬ tool (loop)
                                │                                                  └ dead_letter
                                ├ missing_info → clarify → finalize → END
                                ├ risky        → risky_action → approval ─┬ ok → tool → …
                                │                                         └ no → clarify
                                └ error        → retry → (bounded loop above)

Every path terminates at ``finalize → END``; the only cycle is ``tool → evaluate → retry → tool``
and it is bounded by ``route_after_retry`` comparing ``attempt`` to ``max_attempts``.
"""

from __future__ import annotations

from typing import Any

from .routing import (
    N_ANSWER,
    N_APPROVAL,
    N_CLARIFY,
    N_CLASSIFY,
    N_DEAD_LETTER,
    N_EVALUATE,
    N_FINALIZE,
    N_INTAKE,
    N_RETRY,
    N_RISKY,
    N_TOOL,
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from .state import AgentState

#: Safety net for the bounded retry cycle. Worst realistic path is
#: intake→classify→retry→tool→evaluate→retry→tool→evaluate→answer→finalize (~12 steps); 60 leaves
#: generous headroom while still turning a wiring bug into a clear GraphRecursionError.
RECURSION_LIMIT = 60


def build_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the LangGraph workflow described in the module docstring."""
    from langgraph.graph import END, START, StateGraph

    from .nodes import (
        answer_node,
        approval_node,
        ask_clarification_node,
        classify_node,
        dead_letter_node,
        evaluate_node,
        finalize_node,
        intake_node,
        retry_or_fallback_node,
        risky_action_node,
        tool_node,
    )

    builder = StateGraph(AgentState)

    # 1. Register all 11 nodes.
    builder.add_node(N_INTAKE, intake_node)
    builder.add_node(N_CLASSIFY, classify_node)
    builder.add_node(N_TOOL, tool_node)
    builder.add_node(N_EVALUATE, evaluate_node)
    builder.add_node(N_ANSWER, answer_node)
    builder.add_node(N_CLARIFY, ask_clarification_node)
    builder.add_node(N_RISKY, risky_action_node)
    builder.add_node(N_APPROVAL, approval_node)
    builder.add_node(N_RETRY, retry_or_fallback_node)
    builder.add_node(N_DEAD_LETTER, dead_letter_node)
    builder.add_node(N_FINALIZE, finalize_node)

    # 2. Fixed edges.
    builder.add_edge(START, N_INTAKE)
    builder.add_edge(N_INTAKE, N_CLASSIFY)
    builder.add_edge(N_TOOL, N_EVALUATE)
    builder.add_edge(N_RISKY, N_APPROVAL)
    builder.add_edge(N_ANSWER, N_FINALIZE)
    builder.add_edge(N_CLARIFY, N_FINALIZE)
    builder.add_edge(N_DEAD_LETTER, N_FINALIZE)
    builder.add_edge(N_FINALIZE, END)

    # 3. Conditional edges. The explicit path maps double as documentation and as a guard:
    #    a router returning a name outside its map raises instead of silently dead-ending.
    builder.add_conditional_edges(
        N_CLASSIFY,
        route_after_classify,
        {
            N_ANSWER: N_ANSWER,
            N_TOOL: N_TOOL,
            N_CLARIFY: N_CLARIFY,
            N_RISKY: N_RISKY,
            N_RETRY: N_RETRY,
        },
    )
    builder.add_conditional_edges(
        N_EVALUATE,
        route_after_evaluate,
        {N_ANSWER: N_ANSWER, N_RETRY: N_RETRY},
    )
    builder.add_conditional_edges(
        N_RETRY,
        route_after_retry,
        {N_TOOL: N_TOOL, N_DEAD_LETTER: N_DEAD_LETTER},
    )
    builder.add_conditional_edges(
        N_APPROVAL,
        route_after_approval,
        {N_TOOL: N_TOOL, N_CLARIFY: N_CLARIFY},
    )

    return builder.compile(checkpointer=checkpointer)


def render_mermaid(checkpointer: Any | None = None) -> str:
    """Return the compiled graph as a Mermaid diagram (extension: graph visualisation)."""
    return build_graph(checkpointer=checkpointer).get_graph().draw_mermaid()
