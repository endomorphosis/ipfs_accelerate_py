"""Qualify an inactive source tree without relocating a live state owner.

This is the source-only phase of a native cutover. Its report never authorizes
launch, owner recovery, claim mutation, or replacement of executing files.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from . import configured_board_scheduler as scheduler


SCHEMA = "ipfs_accelerate_py/agent-supervisor/source-candidate-preflight@1"


def qualify_source_candidate(*, repo_root: Path, config_path: Path,
                             candidate_root: Path) -> dict[str, Any]:
    # Load against the real repository first. In particular the existing
    # managed-owner confinement and typed program checks are not bypassed.
    board = scheduler.load_configured_board(config_path, repo_root=repo_root)
    root = scheduler._canonical_no_symlink_root(candidate_root)
    if root == board.repo_root:
        raise scheduler.ConfiguredBoardError("source candidate must be a separate checkout")
    relative = board.config_path.relative_to(board.repo_root)
    candidate_config = scheduler._contained_path(root, relative.as_posix())
    original = board.config_path.read_bytes()
    revision = scheduler._identity({
        "path": relative.as_posix(),
        "bytes_sha256": hashlib.sha256(original).hexdigest(),
    })
    if revision != board.configuration_revision:
        raise scheduler.ConfiguredBoardError("board configuration changed since load")
    if candidate_config.read_bytes() != original:
        raise scheduler.ConfiguredBoardError("source candidate changed the board configuration")
    # This view is used only by the existing read-only source preflight. It
    # cannot be returned to a launcher: the original program retains its real
    # owner paths and the normal loader still rejects launch from this root.
    source_view = replace(board, repo_root=root, config_path=candidate_config)
    before = scheduler._git_identity(root)
    result = scheduler.preflight_configured_board(source_view)
    after = scheduler._git_identity(root)
    status = scheduler._git(source_view, "status", "--porcelain=v1", "--untracked-files=all")
    if (before != after or status.returncode != 0 or status.stdout.strip()
            or board.config_path.read_bytes() != original
            or candidate_config.read_bytes() != original):
        raise scheduler.ConfiguredBoardError("source candidate or configuration changed during qualification")
    return {
        "schema": SCHEMA,
        "valid": result.get("valid") is True,
        "phase": "source_only",
        "authorizes_launch": False,
        "authorizes_recovery": False,
        "authorizes_source_replacement": False,
        "repository_root": str(board.repo_root),
        "candidate_root": str(root),
        "source_head": after[0],
        "source_tree": after[1],
        "configuration_root": board.configuration_root,
        "configuration_sha256": hashlib.sha256(original).hexdigest(),
        "configured_owner_binding_changed": False,
        "source_preflight": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = qualify_source_candidate(repo_root=args.repo_root,
            config_path=args.config, candidate_root=args.candidate_root)
    except (OSError, ValueError, RuntimeError) as exc:
        report = {"schema": SCHEMA, "valid": False, "phase": "source_only",
                  "authorizes_launch": False, "authorizes_recovery": False,
                  "authorizes_source_replacement": False,
                  "errors": [str(exc)]}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
