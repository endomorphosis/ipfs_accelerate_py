"""Operator-reviewed LLM defaults for the accelerator supervisor.

These values are owned locally. Importing them from whichever editable
``ipfs_datasets_py`` checkout happens to be installed made the supervisor's
runtime model drift with ambient package resolution.
"""

from __future__ import annotations

DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_CODEX_PROVIDER = "codex_cli"
