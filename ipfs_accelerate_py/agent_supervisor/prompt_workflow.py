"""ASREF compatibility path for prompt-workflow contracts.

Canonical implementation lives at::

    ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow

This surface exists so path-based import probes and historical fixture
references continue to resolve without reintroducing a second implementation.
It is provider-free: it only re-executes the leaf contract module.
"""

from __future__ import annotations

from pathlib import Path
import runpy

_CANONICAL_PATH = (
    Path(__file__).resolve().parent / "prompt" / "prompt_workflow.py"
)
_loaded = runpy.run_path(str(_CANONICAL_PATH), run_name=__name__)
globals().update(
    {key: value for key, value in _loaded.items() if not key.startswith("_")}
)

# Preserve dual-copy provenance for inventory / ASREF probes.
PROMPT_WORKFLOW_LANDED_PATH = (
    "ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py"
)
PROMPT_WORKFLOW_COMPATIBILITY_PATH = (
    "ipfs_accelerate_py/agent_supervisor/prompt_workflow.py"
)
PROMPT_WORKFLOW_MIGRATED = True
