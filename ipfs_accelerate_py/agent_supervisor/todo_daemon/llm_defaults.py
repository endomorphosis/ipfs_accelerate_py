"""LLM default constants for the accelerator todo daemon port."""

from __future__ import annotations

try:  # Keep compatibility with the source daemon when ipfs_datasets_py is installed.
    from ipfs_datasets_py.optimizers.common.llm_defaults import (  # type: ignore[import-not-found]
        DEFAULT_CODEX_MODEL,
        DEFAULT_CODEX_PROVIDER,
    )
except Exception:  # pragma: no cover - exercised when the optional bridge is absent.
    DEFAULT_CODEX_MODEL = "gpt-5-codex"
    DEFAULT_CODEX_PROVIDER = "codex_cli"
