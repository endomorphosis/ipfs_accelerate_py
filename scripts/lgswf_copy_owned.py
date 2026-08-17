"""Copy owned LGSWF implementation files into an attempt worktree."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable


def copy_owned(relpaths: Iterable[str], *, dest: Path | None = None) -> dict[str, object]:
    src_root = Path(__file__).resolve().parents[1]
    dest_root = Path(dest or Path.cwd())
    copied: list[str] = []
    for rel in relpaths:
        source = src_root / rel
        target = dest_root / rel
        if not source.is_file():
            raise FileNotFoundError(f"owned source missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(rel)
    add = subprocess.run(
        ["git", "--literal-pathspecs", "add", "--force", "--", *copied],
        cwd=dest_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "copied": copied,
        "staged": add.returncode == 0,
        "stage_returncode": add.returncode,
        "stage_stderr": (add.stderr or "")[-500:],
        "dest": str(dest_root),
    }


def emit(result: dict[str, object]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
