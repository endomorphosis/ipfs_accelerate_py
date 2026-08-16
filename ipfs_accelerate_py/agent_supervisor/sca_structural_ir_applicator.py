"""SCA-facing re-export of structural IR application.

Canonical implementation:
:mod:`ipfs_accelerate_py.agent_supervisor.proof.ir_structural_application`

This module remains for SCA scripts and historical imports. Prefer the proof
module for new, domain-agnostic planner/doctor/repair code.
"""

from __future__ import annotations

from .proof.ir_structural_application import *  # noqa: F403
from .proof.ir_structural_application import (
    DEFAULT_STRUCTURAL_FAMILIES,
    DEFAULT_VECTOR_DIMS,
    STRUCTURAL_IR_INTERFACE,
    apply_ast_logic,
    apply_knowledge_graph_logic,
    apply_structural_logic,
    apply_vector_index_logic,
    project_ast_blob_record,
)

__all__ = [
    "DEFAULT_STRUCTURAL_FAMILIES",
    "DEFAULT_VECTOR_DIMS",
    "STRUCTURAL_IR_INTERFACE",
    "apply_ast_logic",
    "apply_knowledge_graph_logic",
    "apply_structural_logic",
    "apply_vector_index_logic",
    "project_ast_blob_record",
]
