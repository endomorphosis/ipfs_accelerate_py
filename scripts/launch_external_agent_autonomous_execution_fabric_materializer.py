#!/usr/bin/python3.12
"""Stdlib-only, fail-closed launcher for the EAAEF bootstrap materializer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER = ROOT / "scripts/materialize_external_agent_autonomous_execution_fabric_control_plane.py"
CONFIGURED_BOARD_SCHEDULER = (
    ROOT / "scripts/ops/agent_supervisor/configured_board_scheduler.py"
)
CONFIGURED_BOARD_CONFIG = (
    ROOT / "config/external_agent_autonomous_execution_fabric_scheduler.json"
)
APPROVED_IMPORT_ROOT = Path("/home/barberb/.local/lib/python3.12/site-packages")
PYCACHE_PREFIX = Path("/nonexistent/eaaef-bootstrap-runtime-pycache")
COMMANDS = {
    "build",
    "runtime-check",
    "materialize",
    "verify",
    "launch-plan",
    "configured-board-launch",
}


def main() -> int:
    flags = sys.flags
    if not (
        flags.isolated == 1
        and flags.no_site == 1
        and flags.dont_write_bytecode == 1
        and flags.no_user_site == 1
        and flags.ignore_environment == 1
        and flags.safe_path is True
    ):
        raise SystemExit("EAAEF bootstrap launcher requires exact -I -S -B flags")
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(
            "usage: launcher {build|runtime-check|materialize|verify|launch-plan|"
            "configured-board-launch}"
        )
    if APPROVED_IMPORT_ROOT.resolve(strict=True) != APPROVED_IMPORT_ROOT:
        raise SystemExit("approved import root is missing or noncanonical")
    if PYCACHE_PREFIX.exists() or PYCACHE_PREFIX.resolve(strict=False) != PYCACHE_PREFIX:
        raise SystemExit("bootstrap pycache prefix must be canonical and absent")
    sys.pycache_prefix = str(PYCACHE_PREFIX)
    sys.path.insert(0, str(APPROVED_IMPORT_ROOT))
    command = sys.argv[1]
    if command == "configured-board-launch":
        try:
            namespace = runpy.run_path(str(MATERIALIZER))
            config = namespace["_load_object"](CONFIGURED_BOARD_CONFIG)
            launch_plan = namespace["launch_plan"](
                config,
                invocation_command="configured-board-launch",
            )
        except Exception as exc:
            raise SystemExit(
                "EAAEF configured-board launch is not admitted"
            ) from exc
        expected_argv = [str(item) for item in sys.orig_argv]
        if (
            launch_plan.get("allowed") is not True
            or launch_plan.get("process_started") is not False
            or launch_plan.get("argv") != expected_argv
        ):
            raise SystemExit("EAAEF configured-board launch is not admitted")
        sys.argv = [
            str(CONFIGURED_BOARD_SCHEDULER),
            "--repo-root",
            str(ROOT),
            "--config",
            str(CONFIGURED_BOARD_CONFIG),
            "launch",
            "--implement",
        ]
        runpy.run_path(str(CONFIGURED_BOARD_SCHEDULER), run_name="__main__")
        return 0
    sys.argv = [str(MATERIALIZER), command]
    runpy.run_path(str(MATERIALIZER), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
