"""Pick the live extra-gate status, ignoring stale run-* copies."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


def _payload(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, UnicodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _birth_pid(identity: Mapping[str, Any]) -> int | None:
    birth = identity.get("process_birth")
    if not isinstance(birth, dict):
        return None
    pid = birth.get("pid")
    return pid if isinstance(pid, int) and pid > 1 else None


def status_is_live_ready(
    payload: Mapping[str, Any],
    *,
    quack_pid: int | None = None,
) -> bool:
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        return False
    pid = _birth_pid(identity)
    if pid is None or not Path(f"/proc/{pid}").exists():
        return False
    if quack_pid is not None and pid != quack_pid:
        return False
    lifecycle = str(payload.get("lifecycle") or "")
    status = str(identity.get("status") or "")
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    ready = lifecycle == "ready" or status == "ready" or readiness.get("ready") is True
    return ready and status != "stopped" and lifecycle != "stopped"


def iter_owner_status_paths(source_root: Path) -> list[Path]:
    root = Path(source_root)
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in (
        "data/agent_supervisor/*/run-*/quack-owner/quack-state-server.status.json",
        "data/agent_supervisor/*/quack-owner/quack-state-server.status.json",
    ):
        for path in root.glob(pattern):
            resolved = path if path.is_absolute() else (root / path)
            try:
                resolved = resolved.resolve()
            except OSError:
                continue
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            found.append(resolved)
    return found


def find_ready_owner_status(
    source_root: Path,
    *,
    quack_pid: int | None = None,
) -> Path | None:
    matched: list[tuple[float, Path]] = []
    for path in iter_owner_status_paths(source_root):
        payload = _payload(path)
        if payload is None:
            continue
        if not status_is_live_ready(payload, quack_pid=quack_pid):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        matched.append((mtime, path))
    if not matched:
        return None
    matched.sort(key=lambda item: item[0], reverse=True)
    return matched[0][1]


def state_root_for_ready_owner(
    source_root: Path,
    *,
    quack_pid: int | None = None,
) -> Path | None:
    status = find_ready_owner_status(source_root, quack_pid=quack_pid)
    if status is None:
        return None
    run_dir = status.parent.parent
    state = run_dir / "state"
    if (state / "lane-0").is_dir():
        return state
    return state if state.is_dir() else None


def owner_is_ready(source_root: Path, quack_pid: int | None = None) -> bool:
    return find_ready_owner_status(source_root, quack_pid=quack_pid) is not None


def _parse_pid(text: str) -> int | None:
    try:
        pid = int(text)
    except (TypeError, ValueError):
        return None
    return pid if pid > 1 else None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        return 2
    command, root_text = args[0], args[1]
    quack_pid = _parse_pid(args[2]) if len(args) > 2 else None
    root = Path(root_text)
    if command == "ready":
        return 0 if owner_is_ready(root, quack_pid) else 1
    if command == "state-root":
        state = state_root_for_ready_owner(root, quack_pid=quack_pid)
        if state is None:
            return 1
        sys.stdout.write(str(state))
        sys.stdout.write(os.linesep)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
