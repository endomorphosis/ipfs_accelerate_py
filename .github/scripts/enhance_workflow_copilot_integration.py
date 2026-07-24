#!/usr/bin/env python3
"""Compatibility entry point for the workflow Copilot integration tool.

The implementation is owned by the nested ``ipfs_datasets_py`` checkout.
Keeping this small delegating module at the supervisor-facing path lets tools
import, compile, or execute the script from the outer worktree without
duplicating the implementation.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


CANONICAL_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ipfs_datasets_py"
    / ".github"
    / "scripts"
    / "enhance_workflow_copilot_integration.py"
)


def _load_canonical_module() -> ModuleType:
    """Load the canonical script, preserving import and execution failures."""
    spec = importlib.util.spec_from_file_location(
        "_canonical_enhance_workflow_copilot_integration", CANONICAL_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load workflow Copilot integration from {CANONICAL_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_canonical = _load_canonical_module()
WorkflowCopilotIntegration = _canonical.WorkflowCopilotIntegration
main = _canonical.main

__all__ = ["WorkflowCopilotIntegration", "main"]


if __name__ == "__main__":
    sys.exit(main())
