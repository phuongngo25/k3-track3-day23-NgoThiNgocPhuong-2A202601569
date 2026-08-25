"""Checkpointer adapter.

The checkpointer is what makes this a *durable* workflow rather than a function call: every
super-step writes the full state to the backend keyed by ``thread_id``, so a run can be inspected
(``get_state_history``), resumed after an ``interrupt()``, or recovered after a process crash.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_SQLITE_PATH = "outputs/checkpoints.sqlite"


def _sqlite_path(database_url: str | None) -> str:
    """Accept a bare path or a ``sqlite:///path`` URL."""
    if not database_url:
        return DEFAULT_SQLITE_PATH
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///") :]
    if database_url.startswith("sqlite://"):
        return database_url[len("sqlite://") :]
    return database_url


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer for the requested backend."""
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError("Install: pip install langgraph-checkpoint-sqlite") from exc

        path = _sqlite_path(database_url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: LangGraph may touch the connection from a worker thread.
        conn = sqlite3.connect(path, check_same_thread=False)
        # WAL + NORMAL sync: concurrent readers during a run, and a checkpoint write that survives
        # a process kill without paying a full fsync per super-step.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return SqliteSaver(conn=conn)

    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError("Install: pip install langgraph-checkpoint-postgres") from exc
        if not database_url:
            raise ValueError("postgres checkpointer requires database_url")
        saver = PostgresSaver.from_conn_string(database_url)
        saver.setup()
        return saver

    raise ValueError(f"Unknown checkpointer kind: {kind}")
