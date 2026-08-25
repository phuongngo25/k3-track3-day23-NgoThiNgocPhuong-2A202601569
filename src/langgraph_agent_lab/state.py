"""State schema for the Day 08 LangGraph lab.

Design rule: state stays lean and JSON-serializable so any checkpointer (memory, SQLite,
Postgres) can persist it without custom encoders. Fields fall into two families:

* **overwrite** — "current truth" scalars (route, attempt, evaluation_result...). LangGraph's
  default reducer replaces them, which is what we want: reading a stale route would break routing.
* **append-only** (``Annotated[list, add]``) — audit trails (messages, tool_results, errors,
  events). Nodes return single-element lists and the reducer concatenates, so the retry loop can
  visit ``tool`` three times without any node needing to read-modify-write a list.
"""

from __future__ import annotations

from enum import StrEnum
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    DEAD_LETTER = "dead_letter"
    DONE = "done"


#: Routes that ``classify_node`` is allowed to emit. DEAD_LETTER/DONE are terminal bookkeeping
#: values produced by nodes, never by the classifier.
CLASSIFIABLE_ROUTES: tuple[str, ...] = (
    Route.RISKY.value,
    Route.TOOL.value,
    Route.MISSING_INFO.value,
    Route.ERROR.value,
    Route.SIMPLE.value,
)


class LabEvent(BaseModel):
    """Append-only audit event for grading and debugging."""

    node: str
    event_type: str
    message: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool = False
    reviewer: str = "mock-reviewer"
    comment: str = ""


class AgentState(TypedDict, total=False):
    """LangGraph state.

    See module docstring for the overwrite vs append-only rationale.
    """

    # ── identity / inputs (overwrite) ──────────────────────────────────
    thread_id: str
    scenario_id: str
    query: str

    # ── classification + control flow (overwrite) ─────────────────────
    route: str
    risk_level: str
    attempt: int
    max_attempts: int

    # ── student-added control fields (overwrite) ──────────────────────
    #: "success" | "needs_retry" — gate read by route_after_evaluate.
    evaluation_result: str
    #: Clarification question emitted on the missing_info / rejected-approval paths.
    pending_question: str | None
    #: Human-readable description of the side-effecting action awaiting approval.
    proposed_action: str | None
    #: ApprovalDecision as a plain dict (serializable) — read by route_after_approval.
    approval: dict[str, Any] | None
    #: Set by classify_node when the LLM call failed and the heuristic fallback was used.
    classifier_mode: str

    final_answer: str | None

    # ── append-only audit trails ──────────────────────────────────────
    messages: Annotated[list[str], add]
    tool_results: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    events: Annotated[list[dict[str, Any]], add]


class Scenario(BaseModel):
    id: str
    query: str
    expected_route: Route
    requires_approval: bool = False
    should_retry: bool = False
    max_attempts: int = 3
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


def initial_state(scenario: Scenario) -> AgentState:
    """Create a serializable initial state for one scenario."""
    return {
        "thread_id": f"thread-{scenario.id}",
        "scenario_id": scenario.id,
        "query": scenario.query,
        "route": "",
        "risk_level": "unknown",
        "attempt": 0,
        "max_attempts": scenario.max_attempts,
        "evaluation_result": "",
        "pending_question": None,
        "proposed_action": None,
        "approval": None,
        "classifier_mode": "",
        "final_answer": None,
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def make_event(node: str, event_type: str, message: str, **metadata: Any) -> dict[str, Any]:
    """Create a normalized event payload."""
    event = LabEvent(node=node, event_type=event_type, message=message, metadata=metadata)
    return event.model_dump()
