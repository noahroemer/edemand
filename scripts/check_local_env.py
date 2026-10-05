"""Local environment doctor (CLAUDE.md section 3).

Checks the Windows/venv side only. It says nothing about what Databricks serverless can
reach or install — that is section 13's job and has to run as a notebook, because under
Databricks Connect `requests` executes on this machine and would falsely pass.

    python scripts/check_local_env.py
    python scripts/check_local_env.py --spark    # also open a serverless Spark session
    python scripts/check_local_env.py --apis     # also probe EIA and Open-Meteo from here
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path

OK = "  ok  "
WARN = " warn "
FAIL = " FAIL "

REQUIRED = {
    "core": ["fastapi", "uvicorn", "pydantic", "numpy", "holidays"],
    "pipeline": ["databricks-connect", "databricks-sdk", "mlflow", "requests", "pandas", "pyarrow"],
    "modeling": ["pymc", "arviz", "pygam", "statsmodels", "properscoring", "scipy"],
    "dev": ["pytest", "responses", "ruff"],
}

# External CLIs, with the milestone where each first blocks work.
TOOLS = {
    "git": "needed now",
    "docker": "needed day 2 (containerized API)",
    "databricks": "needed day 7/11 (Asset Bundle)",
    "gh": "optional (Actions secrets, releases)",
}

failures: list[str] = []
warnings: list[str] = []


def report(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))
    if status == FAIL:
        failures.append(label)
    elif status == WARN:
        warnings.append(label)


def check_python() -> None:
    major, minor, micro = sys.version_info[:3]
    version = f"{major}.{minor}.{micro}"
    if (major, minor) == (3, 12):
        report(OK, "Python version", version)
    else:
        report(
            FAIL,
            "Python version",
            f"{version} - section 3 requires 3.12 (3.13 breaks databricks-connect)",
        )

    expected = Path(__file__).resolve().parent.parent / ".venv"
    if sys.prefix == sys.base_prefix:
        report(FAIL, "Virtualenv", "not active - run .\\.venv\\Scripts\\Activate.ps1")
    elif Path(sys.prefix).resolve() != expected.resolve():
        report(WARN, "Virtualenv", f"active at {sys.prefix}, expected {expected}")
    else:
        report(OK, "Virtualenv", sys.prefix)


def check_packages() -> None:
    for group, names in REQUIRED.items():
        missing = [n for n in names if not _installed(n)]
        if missing:
            report(
                FAIL,
                f"Packages [{group}]",
                f"missing {', '.join(missing)} - pip install -e .[{group}]",
            )
        else:
            report(OK, f"Packages [{group}]", f"{len(names)} present")


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def check_tools() -> None:
    for tool, when in TOOLS.items():
        if shutil.which(tool):
            report(OK, f"CLI: {tool}")
        else:
            report(FAIL if when == "needed now" else WARN, f"CLI: {tool}", f"not found - {when}")


def check_credentials() -> None:
    cfg = Path.home() / ".databrickscfg"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8", errors="replace")
        if "host" in text and ("token" in text or "client_id" in text):
            report(OK, "~/.databrickscfg", "host + auth present")
        else:
            report(FAIL, "~/.databrickscfg", "present but missing host or auth")
    elif os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"):
        report(OK, "Databricks credentials", "from environment")
    else:
        report(FAIL, "Databricks credentials", "no ~/.databrickscfg and no env vars")

    if _eia_key():
        report(OK, "EIA_API_KEY", "set")
    else:
        report(
            WARN,
            "EIA_API_KEY",
            "not set - register at eia.gov/opendata/register.php, then put it in .env (blocks demand ingestion)",
        )


def _eia_key() -> str | None:
    """Read the key from the environment or .env, without needing python-dotenv."""
    if os.getenv("EIA_API_KEY"):
        return os.environ["EIA_API_KEY"]
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == "EIA_API_KEY" and value.strip():
            return value.strip()
    return None


def check_apis() -> None:
    """Probe the data APIs from this machine.

    Note what this does and does not prove: it shows your laptop can reach them. It says
    nothing about Databricks serverless, whose outbound access is restricted (section 13
    question 1) and must be tested in a notebook.
    """
    import requests

    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": 34.05,
                "longitude": -118.24,
                "start_date": "2024-06-01",
                "end_date": "2024-06-01",
                "hourly": "temperature_2m",
                "timezone": "GMT",
            },
            timeout=20,
        )
        hours = len(resp.json().get("hourly", {}).get("time", []))
        report(
            OK if hours == 24 else WARN, "Open-Meteo archive", f"{resp.status_code}, {hours} hours"
        )
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "Open-Meteo archive", f"{type(exc).__name__}: {exc}")

    key = _eia_key()
    if not key:
        report(WARN, "EIA API", "skipped - no key")
        return
    try:
        resp = requests.get(
            "https://api.eia.gov/v2/electricity/rto/region-data/data/",
            params={
                "api_key": key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": "CISO",
                "facets[type][]": "D",
                "length": 5,
            },
            timeout=30,
        )
        if resp.status_code == 403:
            report(FAIL, "EIA API", "403 - key rejected")
        elif resp.ok:
            total = resp.json().get("response", {}).get("total")
            report(OK, "EIA API", f"200, {total} rows available for CISO")
        else:
            report(FAIL, "EIA API", f"{resp.status_code}: {resp.text[:120]}")
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "EIA API", f"{type(exc).__name__}: {exc}")


def check_spark() -> None:
    try:
        from databricks.connect import DatabricksSession
    except ImportError:
        report(FAIL, "Spark session", "databricks-connect not installed")
        return
    try:
        spark = DatabricksSession.builder.serverless(True).getOrCreate()
        value = spark.sql("SELECT 1 AS ok").collect()[0]["ok"]
        report(OK, "Spark session", f"serverless responded ({value}); Spark {spark.version}")
        catalogs = [r[0] for r in spark.sql("SHOW CATALOGS").collect()]
        report(OK, "Unity Catalog", f"catalogs: {', '.join(catalogs)}")
        if "main" not in catalogs:
            report(
                WARN,
                "Catalog name",
                "no 'main' catalog - section 12 names main.default.demand_forecast; see section 13 question 3",
            )
    except Exception as exc:  # noqa: BLE001
        report(FAIL, "Spark session", f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spark", action="store_true", help="start a serverless Spark session (~1 min cold)"
    )
    parser.add_argument(
        "--apis", action="store_true", help="probe EIA and Open-Meteo from this machine"
    )
    args = parser.parse_args()

    print("=== local environment (CLAUDE.md section 3) ===")
    check_python()
    check_packages()
    check_tools()
    check_credentials()

    if args.apis:
        print("\n=== data APIs, from this machine ===")
        check_apis()
    if args.spark:
        print("\n=== databricks connect (serverless) ===")
        check_spark()
    if not (args.apis or args.spark):
        print("\n(skipped API and Spark probes; add --apis --spark)")

    print(f"\n{len(failures)} failure(s), {len(warnings)} warning(s)")
    if failures:
        print("blocking: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
