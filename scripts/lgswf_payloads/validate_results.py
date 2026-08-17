#!/usr/bin/env python3
"""LGSWF-131 benchmark results validator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args(argv)
    root = Path(args.results)
    payload = json.loads((root / "results.json").read_text(encoding="utf-8"))
    if payload.get("schema") != "lgswf/benchmark-results@1":
        print("unexpected schema", payload.get("schema"), file=sys.stderr)
        return 1
    if "suites" not in payload:
        print("missing suites", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
