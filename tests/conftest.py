"""Shared test fixtures.

Loads `.env` so `network`-marked tests can reach the real APIs. Open-Meteo needs no
credentials, so those run anywhere; the EIA ones require a key and **skip** rather than
fail when it is absent — a missing optional credential is not a broken build, and CI
deliberately has no key (it runs `-m "not network"`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Populate os.environ from .env without overriding anything already set.

    Hand-rolled rather than importing python-dotenv so the offline suite does not depend
    on the `pipeline` extra being installed. Strips surrounding quotes, because a quoted
    value in .env is a common paste artifact and produces a confusing 403 downstream.
    """
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        value = value.strip().strip("\"'").strip()
        if value and name not in os.environ:
            os.environ[name] = value


_load_dotenv()


@pytest.fixture(scope="session")
def eia_key() -> str:
    """The EIA API key, or skip the test.

    Requested only by tests that genuinely need live EIA access.
    """
    key = os.environ.get("EIA_API_KEY")
    if not key:
        pytest.skip("No EIA_API_KEY set; see .env.example")
    return key
