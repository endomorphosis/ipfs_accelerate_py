"""Retired flat path: load the canonical package prompt_workflow module.

ASE3-025 repairs this standalone fixture so contract probes and legacy
imports resolve the package implementation without process effects.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_module = importlib.import_module(
    "ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow"
)

# Re-export public names for star-import and attribute compatibility.
for _name in dir(_module):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_module, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
