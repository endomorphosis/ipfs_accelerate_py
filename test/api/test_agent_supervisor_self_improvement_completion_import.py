"""Cold-import regressions for the self-improvement completion public API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_MODULE = "ipfs_accelerate_py.agent_supervisor"
COMPLETION_MODULE = (
    "ipfs_accelerate_py.agent_supervisor.self_improvement."
    "self_improvement_completion"
)
LEGACY_COLD_MODULE = (
    "ipfs_accelerate_py.agent_supervisor._self_improvement_completion_cold"
)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["IPFS_ACCEL_SKIP_CORE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def test_agent_supervisor_cold_import_is_deprecation_warning_clean() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            f"import {SUPERVISOR_MODULE}",
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DeprecationWarning" not in completed.stderr


def test_root_completion_export_is_canonical_owner_symbol() -> None:
    script = f"""
import importlib
import json
import sys

api = importlib.import_module({SUPERVISOR_MODULE!r})
owner = importlib.import_module({COMPLETION_MODULE!r})
root_symbol = api.evaluate_self_improvement_root_completion
owner_symbol = owner.evaluate_self_improvement_root_completion
print(json.dumps({{
    "identical": root_symbol is owner_symbol,
    "legacy_cold_loaded": {LEGACY_COLD_MODULE!r} in sys.modules,
    "owner_module": owner_symbol.__module__,
    "owner_package": owner.__package__,
    "owner_spec_parent": owner.__spec__.parent,
    "root_module": root_symbol.__module__,
}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "identical": True,
        "legacy_cold_loaded": False,
        "owner_module": COMPLETION_MODULE,
        "owner_package": "ipfs_accelerate_py.agent_supervisor.self_improvement",
        "owner_spec_parent": (
            "ipfs_accelerate_py.agent_supervisor.self_improvement"
        ),
        "root_module": COMPLETION_MODULE,
    }
