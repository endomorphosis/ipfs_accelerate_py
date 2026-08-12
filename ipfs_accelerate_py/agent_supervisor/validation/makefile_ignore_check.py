#!/usr/bin/env python3
"""Check that a Makefile ignore fragment is present or absent.

Used as a fail-closed replacement for shell forms like::

    test -z "$(rg -n 'ignore=tests/foo' Makefile || true)"

so proposal validation does not need shell expansion in argv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--makefile",
        default="Makefile",
        help="Path to the Makefile to inspect (default: Makefile)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--absent",
        metavar="FRAGMENT",
        help="Succeed only when FRAGMENT does not appear in the Makefile",
    )
    group.add_argument(
        "--present",
        metavar="FRAGMENT",
        help="Succeed only when FRAGMENT appears in the Makefile",
    )
    args = parser.parse_args(argv)
    path = Path(args.makefile)
    try:
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2
    if args.absent is not None:
        fragment = str(args.absent)
        if fragment in text:
            print(f"forbidden fragment still present: {fragment}", file=sys.stderr)
            return 1
        return 0
    fragment = str(args.present)
    if fragment not in text:
        print(f"required fragment missing: {fragment}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
