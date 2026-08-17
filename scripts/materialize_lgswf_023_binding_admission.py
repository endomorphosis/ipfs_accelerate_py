#!/usr/bin/env python3
"""Deterministic LGSWF-023 binding-admission writer."""

from __future__ import annotations

from pathlib import Path

from lgswf_copy_owned import copy_owned, emit

OWNED = (
    "ipfs_accelerate_py/agent_supervisor/planning/semantic_binding_admission.py",
    "test/api/test_agent_supervisor_semantic_binding_admission.py",
)

if __name__ == "__main__":
    emit(copy_owned(OWNED, dest=Path.cwd()))
