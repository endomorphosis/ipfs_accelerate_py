"""Compatibility checks for the lazy :mod:`p2p_tasks` package surface."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


ACCELERATE_ROOT = Path(__file__).resolve().parents[2]


def test_historical_module_aliases_resolve_only_on_access() -> None:
    environment = dict(os.environ)
    environment["IPFS_ACCEL_SKIP_CORE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(ACCELERATE_ROOT),
            environment.get("PYTHONPATH", ""),
        )
        if part
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import ipfs_accelerate_py.p2p_tasks as package

                aliases = (
                    "client",
                    "peer_trust",
                    "protocol",
                    "service",
                    "task_queue",
                    "worker",
                )
                assert all(
                    f"{package.__name__}.{name}" not in sys.modules
                    for name in aliases
                )
                assert all(name not in package.__all__ for name in aliases)
                for name in aliases:
                    module = getattr(package, name)
                    assert module.__name__ == f"{package.__name__}.{name}"
                    assert getattr(package, name) is module
                """
            ),
        ],
        cwd=ACCELERATE_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
