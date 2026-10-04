#!/usr/bin/env python3
"""Prepare source-bound semantic and world context for an existing intent task.

Prints artifact nominations for the daemon. Canonical plans, task status and
execution authority stay owned by their existing native services.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--intent-database", type=Path, required=True)
    parser.add_argument("--task-cid", required=True)
    parser.add_argument("--file", action="append", required=True, dest="paths")
    parser.add_argument("--required-raw-file", action="append", required=True, dest="required")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--security-source-program-config", type=Path,
        help="Explicit local 384D Security checkpoint configuration; optional inference fails open")
    args = parser.parse_args(argv)
    from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import (
        prepare_supervised_task_context,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
        open_intent_repository,
    )

    repository = args.repository.resolve(strict=True)
    database = args.intent_database.resolve(strict=True)
    with open_intent_repository(database, install_schema=False) as intent:
        result = prepare_supervised_task_context(
            repository=repository, intent=intent, task_cid=args.task_cid,
            paths=args.paths, required_raw_paths=args.required,
            output=args.output.resolve(),
            **({"security_source_program_config": args.security_source_program_config.absolute()}
               if args.security_source_program_config is not None else {}),
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
