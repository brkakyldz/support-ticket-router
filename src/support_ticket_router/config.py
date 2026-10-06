"""Environment and model setup, shared by the dev server, the scripts and the tests.

Keys live in `.env` at the repository root (see `env.example`). The LangSmith
project is pinned here *before* anything traced runs: langsmith caches its
environment lookups, so setting them later has no effect.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

LANGSMITH_PROJECT = "support-ticket-router"
DEFAULT_MODEL = "gpt-6-luna"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")


@lru_cache(maxsize=1)
def get_model():
    """Built lazily, so importing the graph (tests, Studio's schema reads) needs no key.
    gpt-6 models only call tools on Chat Completions at reasoning effort `none`, so the
    Responses API is requested explicitly; structured output uses it too."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        use_responses_api=True,
        reasoning={"effort": "low"},
    )
