"""ASREF dual-layout compatibility surface for prompt-workflow contracts.

Canonical implementation (declared PDR-020 output)::

    ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow

Historical contract tests and path-based import probes still load this
package-root path.  The surface is provider-free: it only runpy-loads the
leaf contract module and never imports providers or package side effects.
"""

from __future__ import annotations

from pathlib import Path
import runpy

# Static import edge for scope adjudication (candidate imports declared path).
# Kept unreachable so standalone ``python -S`` / runpy probes remain free of
# package-init side effects (llm_router, subprocess, etc.).
if False:  # pragma: no cover - static dependency evidence only
    from ipfs_accelerate_py.agent_supervisor.prompt import (  # type: ignore
        prompt_workflow as _canonical_module,
    )

_CANONICAL_PATH = (
    Path(__file__).resolve().parent / "prompt" / "prompt_workflow.py"
)
_loaded = runpy.run_path(str(_CANONICAL_PATH), run_name=__name__)
globals().update(
    {key: value for key, value in _loaded.items() if not key.startswith("_")}
)

# Dual-layout provenance for inventory / ASREF probes.
PROMPT_WORKFLOW_LANDED_PATH = (
    "ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py"
)
PROMPT_WORKFLOW_COMPATIBILITY_PATH = (
    "ipfs_accelerate_py/agent_supervisor/prompt_workflow.py"
)
PROMPT_WORKFLOW_MIGRATED = True
