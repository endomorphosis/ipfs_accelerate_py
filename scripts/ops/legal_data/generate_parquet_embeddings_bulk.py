#!/usr/bin/env python3
"""Compatibility entrypoint for bulk parquet embeddings generation.

This wrapper preserves legacy invocation from the repository root:
`scripts/ops/legal_data/generate_parquet_embeddings_bulk.py`

It forwards execution to the canonical implementation under
`ipfs_datasets_py/scripts/ops/legal_data/`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    pkg_root = repo_root / "ipfs_datasets_py"
    # Ensure canonical package path wins over namespace-shadowed imports.
    sys.path.insert(0, str(pkg_root)) if str(pkg_root) not in sys.path else None
    cached = sys.modules.get("ipfs_datasets_py")
    if cached is not None and not hasattr(cached, "embeddings_router"):
        sys.modules.pop("ipfs_datasets_py", None)
    target = repo_root / "ipfs_datasets_py" / "scripts" / "ops" / "legal_data" / "generate_parquet_embeddings_bulk.py"
    if not target.exists():
        raise FileNotFoundError(f"Canonical script not found: {target}")
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
