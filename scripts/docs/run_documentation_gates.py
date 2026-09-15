#!/usr/bin/env python3
"""Run the offline documentation-gates required on GitHub ``main``.

Same four checks as ``.github/workflows/documentation-gates.yml``. Used by
fleet publication when GitHub Actions cannot start (billing lock).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _need(path: Path, pattern: str, label: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text, flags=re.M)
    if not match:
        print(f"FAIL: {label} not found in {path}", file=sys.stderr)
        raise SystemExit(1)
    return match.group(1)


def main(root: Path | None = None) -> int:
    root = (root or Path.cwd()).resolve()
    subprocess.run(
        [sys.executable, str(root / "scripts/docs/check_agent_supervisor_docs.py")],
        cwd=root, check=True,
    )
    subprocess.run(
        [sys.executable, str(root / "scripts/docs/check_current_docs_links.py")],
        cwd=root, check=True,
    )
    pyproject = _need(
        root / "pyproject.toml",
        r'^version\s*=\s*"([^"]+)"\s*$',
        "pyproject version",
    )
    setup = _need(
        root / "setup.py",
        r'^\s*version\s*=\s*"([^"]+)"\s*,?\s*$',
        "setup.py version",
    )
    packaging = _need(
        root / "ipfs_accelerate_py" / "__init__.py",
        r'^_PACKAGING_VERSION\s*=\s*"([^"]+)"\s*$',
        "_PACKAGING_VERSION",
    )
    init_text = (root / "ipfs_accelerate_py" / "__init__.py").read_text(encoding="utf-8")
    if not re.search(r'^__version__\s*=\s*_PACKAGING_VERSION\s*$', init_text, flags=re.M):
        print("FAIL: __version__ must be assigned from _PACKAGING_VERSION", file=sys.stderr)
        return 1
    if not (pyproject == setup == packaging):
        print(
            "FAIL: version pins disagree: "
            f"pyproject={pyproject!r} setup={setup!r} "
            f"_PACKAGING_VERSION={packaging!r}",
            file=sys.stderr,
        )
        return 1
    index = (root / "docs/INDEX.md").read_text(encoding="utf-8")
    current = (root / "docs/development/DOCUMENTATION_CURRENT_STATE.md").read_text(
        encoding="utf-8"
    )
    if "Documentation baseline" not in index:
        print("FAIL: Documentation baseline missing from docs/INDEX.md", file=sys.stderr)
        return 1
    if "Last verified" not in current:
        print(
            "FAIL: Last verified missing from docs/development/DOCUMENTATION_CURRENT_STATE.md",
            file=sys.stderr,
        )
        return 1
    if not (root / "docs/development/DOCUMENTATION_VALIDATION_2026_08.md").is_file():
        print("FAIL: docs/development/DOCUMENTATION_VALIDATION_2026_08.md missing", file=sys.stderr)
        return 1
    print(f"OK: documentation-gates aligned at {pyproject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
