"""Hold-aware, integrity-pinned adapter for native pre-start recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("loader", "loader-sha256", "helper-sha256", "source-root", "inventory", "fleet-config", "board"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    inventory = json.loads(Path(args.inventory).read_text())
    fleet = json.loads(Path(args.fleet_config).read_text())
    if inventory.get("schema") != "ipfs_accelerate_py/taskboard-fleet-inventory@1" or fleet.get("schema") != "agent-supervisor/fleet-watchdog-config@1":
        raise ValueError("unexpected native inventory or fleet configuration")
    def select(payload):
        entries = [entry for entry in payload["boards"] if entry.get("id") == args.board]
        if len(entries) != 1:
            raise ValueError("board absent or ambiguous")
        return entries[0]
    board, managed = select(inventory), select(fleet)
    root = Path(args.source_root).resolve(strict=True)
    if any(Path(item["cwd"]).resolve(strict=True) != root for item in (board, managed)):
        raise ValueError("native source differs from inventory or fleet source")
    runtime = Path(board["runtime_root"])
    holds = {*(Path(p) for p in board.get("hold_paths", [])),
             *(Path(p) for p in managed.get("hold_files", [])), runtime / "HOLD", runtime / "OPERATOR_STOP"}
    present = sorted(str(path) for path in holds if os.path.lexists(path))
    if present:
        print(json.dumps({"board": args.board, "held": True, "hold_files": present, "recovered": 0}))
        return 0
    owner = json.loads(Path(board["owner_status_path"]).read_text())
    if owner.get("lifecycle") != "stopped":
        print(json.dumps({"board": args.board, "recovered": 0, "reason": "native_owner_not_stopped"}))
        return 0
    loader = Path(args.loader)
    for path, digest in ((loader, args.loader_sha256), (loader.with_name("verification_deferral_recovery.py"), args.helper_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("immutable recovery source integrity mismatch")
    import sys
    command = [sys.executable, "-P", str(loader), str(root), "--inventory", args.inventory, "--board", args.board, "--apply"]
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
