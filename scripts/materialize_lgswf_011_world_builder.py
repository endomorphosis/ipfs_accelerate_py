#!/usr/bin/env python3
"""Deterministic LGSWF-011 WorldSnapshotBuilder writer."""

from __future__ import annotations

from pathlib import Path

from lgswf_copy_owned import copy_owned, emit

OWNED = (
    "ipfs_accelerate_py/agent_supervisor/semantic_state/world_snapshot_builder.py",
    "test/api/semantic_state/test_world_snapshot_builder.py",
)

if __name__ == "__main__":
    emit(copy_owned(OWNED, dest=Path.cwd()))
