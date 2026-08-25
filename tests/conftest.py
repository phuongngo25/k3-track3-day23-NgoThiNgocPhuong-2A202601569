"""Shared test fixtures.

Importing the llm helper runs its dotenv loader, so the API-key skip markers in
`test_graph_smoke.py` see the same configuration the CLI does.
"""

import pytest

import langgraph_agent_lab.llm  # noqa: F401  (imported for its .env side effect)

_LLM_KEYS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture
def offline_llm(monkeypatch):
    """Force the documented offline fallback path for the duration of a test.

    Without this, a developer with a key in `.env` runs the "offline" suite against a live
    provider: slow, non-deterministic, and it stalls for minutes behind rate-limit retries. Tests
    that *want* a real model ask for one explicitly via the skip markers in
    `test_graph_smoke.py`.
    """
    for key in _LLM_KEYS:
        monkeypatch.delenv(key, raising=False)
