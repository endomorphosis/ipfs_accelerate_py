#!/usr/bin/env python3
"""Fail if board-prefix ticket IDs leak into primary agent-supervisor docs.

Primary surfaces must stay product-vocabulary first. Ticket IDs are allowed in:
  - *.todo.md / *.objectives.md
  - architecture appendix sections named historical/program evidence
  - programs/ glossary intentionally listing prefixes
  - fenced code blocks inside primary docs (limited)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TICKET = re.compile(r"\b(ASREF|CBP|ASI|AICAT|PLAT2|GOOSE|IRF|REF)-(G?\d+[A-Z0-9]*)\b")

PRIMARY = [
    ROOT / "docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md",
    ROOT / "docs/architecture/agent_supervisor/README.md",
    ROOT / "docs/architecture/agent_supervisor/PACKAGE_MAP.md",
    ROOT / "docs/architecture/agent_supervisor/FOR_AGENTS.md",
    ROOT / "docs/architecture/agent_supervisor/FOR_CONTRIBUTORS.md",
    ROOT / "docs/architecture/agent_supervisor/packages",
]

# PROGRAMS.md intentionally defines prefixes; architecture body ban is separate.
ARCH = ROOT / "docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md"


def strip_fenced(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def body_before_historical_appendix(text: str) -> str:
    marker = "## Appendix: historical program evidence tags"
    if marker in text:
        return text.split(marker, 1)[0]
    return text


def check_file(path: Path, text: str) -> list[str]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if TICKET.search(line):
            hits.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:120]}")
    return hits


def main() -> int:
    errors: list[str] = []
    files: list[Path] = []
    for p in PRIMARY:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        elif p.is_file():
            files.append(p)

    for path in files:
        raw = path.read_text(encoding="utf-8")
        # Allow PROGRAMS.md fully
        if path.name == "PROGRAMS.md":
            continue
        # packages may mention programs in optional evidence section only after marker
        text = strip_fenced(raw)
        if "## Program evidence" in text:
            text = text.split("## Program evidence", 1)[0]
        errors.extend(check_file(path, text))

    if ARCH.is_file():
        body = strip_fenced(body_before_historical_appendix(ARCH.read_text(encoding="utf-8")))
        errors.extend(check_file(ARCH, body))

    if errors:
        print("Ticket IDs found in primary agent-supervisor docs:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    print("OK: primary agent-supervisor docs are free of board-prefix ticket IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
