"""Thin process entry: Grok CLI agent via llm_router.grok_cli.

The implementation daemon feeds the task prompt on stdin. Grok expects
``--prompt-file``, so this runner materializes stdin and execs the command
built by :func:`ipfs_accelerate_py.llm_router.build_grok_cli_command`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Authorized Grok CLI agent entry (llm_router.grok_cli)."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--grok-bin", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-turns", default="")
    parser.add_argument(
        "--mode",
        default="agent",
        choices=("agent", "chat"),
        help="agent enables tool approvals for implementation work",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from ipfs_accelerate_py.llm_router import (
        LLMRouterError,
        build_grok_cli_command,
        build_grok_cli_env,
        find_grok_cli,
    )

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    grok_bin = str(args.grok_bin).strip() or find_grok_cli() or ""
    if not grok_bin:
        print("grok CLI not found on PATH", file=sys.stderr)
        return 127

    prompt = sys.stdin.read()
    if not str(prompt).strip():
        print("implementation prompt on stdin is empty", file=sys.stderr)
        return 2

    prompt_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="asref-grok-prompt-",
            suffix=".txt",
            delete=False,
        ) as handle:
            handle.write(prompt)
            prompt_path = handle.name

        try:
            cmd = build_grok_cli_command(
                mode=str(args.mode),
                workspace=workspace,
                model_name=str(args.model).strip() or None,
                max_turns=int(args.max_turns) if str(args.max_turns).strip() else None,
                grok_bin=grok_bin,
                prompt_file=prompt_path,
            )
            env = build_grok_cli_env()
        except LLMRouterError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        os.chdir(workspace)
        completed = subprocess.run(cmd, env=env, check=False)
        return int(completed.returncode)
    finally:
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
