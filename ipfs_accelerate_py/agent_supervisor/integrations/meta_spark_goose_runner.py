"""Thin process entry: goose agent via llm_router.goose_cli (Meta Spark backend)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Authorized goose agent entry (llm_router.goose_cli / Meta Spark)."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--goose-bin", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--base-path", default="")
    parser.add_argument("--max-turns", default="")
    parser.add_argument("--max-tokens", default="")
    parser.add_argument("--no-developer", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from ipfs_accelerate_py.llm_router import (
        LLMRouterError,
        build_goose_cli_command,
        build_goose_cli_env,
        find_goose_cli,
    )

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2
    goose_bin = str(args.goose_bin).strip() or find_goose_cli() or ""
    if not goose_bin:
        print("goose CLI not found on PATH", file=sys.stderr)
        return 127
    mode = "chat" if args.no_developer else "agent"
    try:
        cmd = build_goose_cli_command(
            mode=mode,
            workspace=workspace,
            model_name=str(args.model).strip() or None,
            max_turns=int(args.max_turns) if str(args.max_turns).strip() else None,
            with_developer=not args.no_developer,
            goose_bin=goose_bin,
        )
        env = build_goose_cli_env(
            mode=mode,
            model_name=str(args.model).strip() or None,
            max_tokens=int(args.max_tokens) if str(args.max_tokens).strip() else None,
        )
    except LLMRouterError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if str(args.host).strip():
        env["OPENAI_HOST"] = str(args.host).strip()
    if str(args.base_path).strip():
        env["OPENAI_BASE_PATH"] = str(args.base_path).strip()
    os.chdir(workspace)
    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError as exc:
        print(f"failed to exec goose: {exc}", file=sys.stderr)
        return 126
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
