"""Operator-reviewed LLM defaults for the accelerator supervisor.

These values are owned locally.  Importing them from whichever editable
``ipfs_datasets_py`` checkout happens to be installed made the supervisor's
runtime model drift with ambient package resolution.
"""

from __future__ import annotations

DEFAULT_GROK_PRIMARY_MODEL = "grok-4.6"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
DEFAULT_CODEX_PROVIDER = "codex_cli"

# The ordinary implementation route keeps its direct-Codex default separate
# from the narrowly authorized Grok quota fallback.  A fallback is a distinct
# policy decision: it must not drift when an operator changes the model used by
# an explicitly selected Codex lane or by an independent review provider.
DEFAULT_CODEX_QUOTA_FALLBACK_MODEL = "gpt-5.6-terra"
DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT = "medium"
