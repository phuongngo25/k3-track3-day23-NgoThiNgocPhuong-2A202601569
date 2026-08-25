"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any


def _load_dotenv() -> None:
    """Load `.env` once so `get_llm()` works from pytest, the CLI and notebooks alike.

    Uses python-dotenv when available and falls back to a tiny parser so the lab has no hard
    dependency on it. Existing environment variables always win over the file.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY → ChatGoogleGenerativeAI
    2. OPENAI_API_KEY → ChatOpenAI
    3. ANTHROPIC_API_KEY → ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        # Gemini 3.x models use fixed sampling and warn on every call that `temperature` is
        # ignored. The argument is still correct for the other providers, so silence the notice
        # rather than branching the public signature on the model name.
        warnings.filterwarnings(
            "ignore", message=".*fixed sampling defaults.*", category=UserWarning
        )
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL") or "gemini-3.6-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL") or "gpt-4o-mini",
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        # ChatAnthropic's constructor overload requires timeout/stop explicitly; None means
        # "use the client defaults".
        return ChatAnthropic(
            model_name=model or os.getenv("LLM_MODEL") or "claude-sonnet-4-5-20250929",
            temperature=temperature,
            timeout=None,
            stop=None,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
