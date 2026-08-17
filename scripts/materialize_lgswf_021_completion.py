#!/usr/bin/env python3
"""Deterministic LGSWF-021 completion-contract writer."""

from __future__ import annotations

from pathlib import Path

from lgswf_copy_owned import copy_owned, emit

OWNED = (
    "ipfs_accelerate_py/agent_supervisor/objectives/completion_contracts.py",
    "test/api/test_agent_supervisor_semantic_completion_contracts.py",
)

if __name__ == "__main__":
    emit(copy_owned(OWNED, dest=Path.cwd()))
