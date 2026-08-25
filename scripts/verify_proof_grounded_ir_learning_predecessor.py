#!/usr/bin/env python3
"""Replay the immutable PGIR predecessor proof in its exact source forest.

The historical verifier intentionally reads its working tree.  Current main has
advanced since that evidence was frozen, so this wrapper creates temporary
shared clones at the commits named by the freeze and invokes the unmodified
verifier there.  It performs no network access and never treats the historical
``no_go`` result as descendant execution authority.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "ipfs_datasets_py"
FREEZE_RELATIVE = Path("data/agent_supervisor/proof_grounded_ir_learning/freeze")
FREEZE_DIR = ROOT / FREEZE_RELATIVE
SOURCE_COMMIT = "04fbb09b4a8b34e77d11bd8da6642e0978baa02c"
DATASETS_COMMIT = "b20bd9e3cfae79e8888929daf64f52b2f8a5689a"
EXPECTED_SCHEMA = "PGIRFreezeIndependentVerificationOutcome@1"


class ReplayError(RuntimeError):
    """Raised when the historical forest or verifier cannot be reproduced."""


def run(
    args: Sequence[str], *, cwd: Path | None = None, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.run(
            tuple(args),
            cwd=cwd,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReplayError(
            f"command timed out after {timeout}s: {' '.join(args)}"
        ) from exc
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip() or "no diagnostic"
        raise ReplayError(
            f"command failed ({process.returncode}): {' '.join(args)}: {detail}"
        )
    return process


def require_commit(repository: Path, commit: str) -> None:
    run(("git", "cat-file", "-e", f"{commit}^{{commit}}"), cwd=repository)


def replay() -> dict[str, Any]:
    if not FREEZE_DIR.is_dir():
        raise ReplayError(f"historical freeze is absent: {FREEZE_DIR}")
    if not DATASETS.is_dir():
        raise ReplayError(f"datasets checkout is absent: {DATASETS}")
    require_commit(ROOT, SOURCE_COMMIT)
    require_commit(DATASETS, DATASETS_COMMIT)

    with tempfile.TemporaryDirectory(prefix="pgir-predecessor-replay-") as temporary:
        replay_root = Path(temporary) / "ipfs_accelerate_py"
        run(
            (
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                "--quiet",
                str(ROOT),
                str(replay_root),
            )
        )
        run(("git", "checkout", "--detach", "--quiet", SOURCE_COMMIT), cwd=replay_root)

        replay_datasets = replay_root / "ipfs_datasets_py"
        if replay_datasets.exists():
            if not replay_datasets.is_dir() or any(replay_datasets.iterdir()):
                raise ReplayError(
                    "historical datasets gitlink did not materialize as an empty directory"
                )
            replay_datasets.rmdir()
        run(
            (
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                "--quiet",
                str(DATASETS),
                str(replay_datasets),
            )
        )
        run(
            ("git", "checkout", "--detach", "--quiet", DATASETS_COMMIT),
            cwd=replay_datasets,
        )

        replay_freeze = replay_root / FREEZE_RELATIVE
        shutil.copytree(FREEZE_DIR, replay_freeze, dirs_exist_ok=True)
        verifier = replay_freeze / "verify_freeze.py"
        verified = run((sys.executable, str(verifier), "--json"), cwd=replay_root)
        try:
            outcome = json.loads(verified.stdout)
        except json.JSONDecodeError as exc:
            raise ReplayError("historical verifier did not emit valid JSON") from exc
        if not isinstance(outcome, dict):
            raise ReplayError("historical verifier outcome is not an object")
        if (
            outcome.get("schema") != EXPECTED_SCHEMA
            or outcome.get("verified") is not True
            or outcome.get("campaign_decision") != "no_go"
            or outcome.get("authorizes_execution") is not False
        ):
            raise ReplayError(
                f"historical verifier emitted an unsafe outcome: {outcome}"
            )

    return {
        "schema": "proof-grounded-ir-learning-predecessor-replay@1",
        "verified": True,
        "source_commit": SOURCE_COMMIT,
        "datasets_commit": DATASETS_COMMIT,
        "network_accessed": False,
        "descendant_execution_authorized": False,
        "historical_outcome": outcome,
    }


def main() -> int:
    try:
        payload = replay()
    except (OSError, ReplayError, ValueError) as exc:
        payload = {
            "schema": "proof-grounded-ir-learning-predecessor-replay@1",
            "verified": False,
            "source_commit": SOURCE_COMMIT,
            "datasets_commit": DATASETS_COMMIT,
            "error": str(exc),
        }
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
