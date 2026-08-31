#!/usr/bin/env python3
"""Thin operator facade for the existing SAWM supervisor authorities.

Import and ``--help`` are side-effect free.  All operational work is delegated
to the landed validators, DatabaseTaskSource, Quack owner, provider router and
configured-board scheduler; this module is not a second agent framework.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
_M34_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.duckdb"
)
_M34_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.coordination.duckdb"
)
_M34_WORKTREE_ROOT = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/worktrees"
)
_M34_GENERATION = 29
_M34_TARGET_PLAN_REVISION = 28
_M34_TARGET_EVENT_WATERMARK = 285
_M34_TARGET_PROJECTION_CID = (
    "baguqeeragsizyo6v4izu7qfvjbj5l5bjkuycw2xyaf3vhlzx2nai7xyrvd4q"
)
_M34_TARGET_QUACK_PORT = 24_070
_M34_INITIAL_CONTROL_COMMIT = "0000000000000000000000000000000000000000"
_M34_INITIAL_CONTROL_TREE = "0000000000000000000000000000000000000000"
_M33_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.duckdb"
)
_M33_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.coordination.duckdb"
)
_M33_WORKTREE_ROOT = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/worktrees"
)
_M33_GENERATION = 29
_M33_TARGET_PLAN_REVISION = 28
_M33_TARGET_EVENT_WATERMARK = 284
_M33_TARGET_PROJECTION_CID = (
    "baguqeera5wkenkpg5zpndh5whgwrqkvpq2e7qz6xv6rflrajynrpqf7dtmla"
)
_M33_TARGET_QUACK_PORT = 24_070
_M33_INITIAL_CONTROL_COMMIT = "b0526d4085b231f1eca0cf638743e8d455debc08"
_M33_INITIAL_CONTROL_TREE = "61054b8cdf793a6c9ed4f326215e60b9799a4a6d"
_M32_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.duckdb"
)
_M32_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.coordination.duckdb"
)
_M32_WORKTREE_ROOT = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/worktrees"
)
_M32_GENERATION = 29
_M32_TARGET_PLAN_REVISION = 28
_M32_TARGET_EVENT_WATERMARK = 283
_M32_TARGET_PROJECTION_CID = (
    "baguqeerab3m6k3ea4ulaouojsdazccvepipcfryyblps7676ymqrtbyc5tiq"
)
_M32_TARGET_QUACK_PORT = 24_070
_M32_INITIAL_CONTROL_COMMIT = "547485ffacd636c6046b00ed23dc0f4c53de4315"
_M32_INITIAL_CONTROL_TREE = "10e153a78f5709e8b841567ab4475f60eff05db3"
_M31_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.duckdb"
)
_M31_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.coordination.duckdb"
)
_M31_WORKTREE_ROOT = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/worktrees"
)
_M31_GENERATION = 29
_M31_TARGET_PLAN_REVISION = 28
_M31_TARGET_EVENT_WATERMARK = 282
_M31_TARGET_PROJECTION_CID = (
    "baguqeerakjradc5sa5dmflygtfh2birrd5onygnt6q2pkvoomsxrspi22jaa"
)
_M31_TARGET_QUACK_PORT = 24_070
_M31_INITIAL_CONTROL_COMMIT = "07aed87e3ebc4ef5667541435fd04f2d62a39b25"
_M31_INITIAL_CONTROL_TREE = "c9fe9a657d2d0e1e05172688dbc44dea6f699265"
_M30_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.duckdb"
)
_M30_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.coordination.duckdb"
)
_M30_WORKTREE_ROOT = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/worktrees"
)
_M30_GENERATION = 28
_M30_TARGET_PLAN_REVISION = 28
_M30_TARGET_EVENT_WATERMARK = 281
_M30_TARGET_PROJECTION_CID = (
    "baguqeeragvf7yhfecuccjs3fqqg7azmwukg63sbgkbivnczfv6uljqd4js4q"
)
_M30_TARGET_QUACK_PORT = 24_070
_M29_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.duckdb"
)
_M29_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.coordination.duckdb"
)
_M29_WORKTREE_ROOT = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/worktrees"
)
_M29_GENERATION = 27
_M29_TARGET_PLAN_REVISION = 28
_M29_TARGET_EVENT_WATERMARK = 274
_M29_TARGET_PROJECTION_CID = (
    "baguqeera4z7aafxgr5xb7tfnb4mdhih4d2ynzkugodxrmrwpisi427hxnz7a"
)
_M29_TARGET_QUACK_PORT = 24_070
_M28_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.duckdb"
)
_M28_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.coordination.duckdb"
)
_M28_WORKTREE_ROOT = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/worktrees"
)
_M28_GENERATION = 27
_M28_TARGET_PLAN_REVISION = 28
_M28_TARGET_EVENT_WATERMARK = 273
_M28_TARGET_PROJECTION_CID = (
    "baguqeerazyzybkjnlihpazozv7xjiign23bfius47mpb3iol6fciuonvzd7a"
)
_M28_TARGET_QUACK_PORT = 24_070
_M27_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.duckdb"
)
_M27_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/control.coordination.duckdb"
)
_M27_WORKTREE_ROOT = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m27/worktrees"
)
_M27_GENERATION = 26
_M27_TARGET_PLAN_REVISION = 28
_M27_TARGET_EVENT_WATERMARK = 268
_M27_TARGET_QUACK_PORT = 24_070
_M26_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m26/control.duckdb"
)
_M26_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m26/control.coordination.duckdb"
)
_M26_GENERATION = 25
_M26_TARGET_PLAN_REVISION = 27
_M26_TARGET_EVENT_WATERMARK = 262
_M26_TARGET_QUACK_PORT = 24_069
_M25_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m25/control.duckdb"
)
_M25_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m25/control.coordination.duckdb"
)
# M24 failed before publishing a generation-24 state-server identity.  The
# source-only M25 successor therefore realizes the still-unconsumed generation
# rather than inventing a generation gap.
_M25_GENERATION = 24
_M25_TARGET_PLAN_REVISION = 26
_M25_TARGET_EVENT_WATERMARK = 251
_M25_TARGET_QUACK_PORT = 24_068
_M24_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m24/control.duckdb"
)
_M24_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m24/control.coordination.duckdb"
)
_M24_GENERATION = 24
_M24_TARGET_PLAN_REVISION = 25
_M24_TARGET_EVENT_WATERMARK = 249
_M24_TARGET_QUACK_PORT = 24_067
_M23_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m23/control.duckdb"
)
_M23_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m23/control.coordination.duckdb"
)
_M23_GENERATION = 23
_M23_TARGET_PLAN_REVISION = 24
_M23_TARGET_EVENT_WATERMARK = 244
_M23_TARGET_QUACK_PORT = 24_066
_M22_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m22/control.duckdb"
)
_M22_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m22/control.coordination.duckdb"
)
_M22_GENERATION = 22
_M22_TARGET_PLAN_REVISION = 23
_M22_TARGET_EVENT_WATERMARK = 233
_M22_TARGET_QUACK_PORT = 24_065
_M21_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m21/control.duckdb"
)
_M21_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m21/control.coordination.duckdb"
)
_M21_GENERATION = 21
_M21_TARGET_PLAN_REVISION = 22
_M21_TARGET_EVENT_WATERMARK = 231
_M21_TARGET_QUACK_PORT = 24_064
_M20_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m20/control.duckdb"
)
_M20_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m20/control.coordination.duckdb"
)
_M20_GENERATION = 21
_M20_TARGET_PLAN_REVISION = 21
_M20_TARGET_EVENT_WATERMARK = 229
_M20_TARGET_QUACK_PORT = 24_063
_M19_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m19/control.duckdb"
)
_M19_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m19/control.coordination.duckdb"
)
_M19_GENERATION = 20
_M19_TARGET_PLAN_REVISION = 20
_M19_TARGET_EVENT_WATERMARK = 227
_M19_TARGET_QUACK_PORT = 24_062
_M18_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m18/control.duckdb"
)
_M18_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m18/control.coordination.duckdb"
)
_M18_GENERATION = 19
_M18_TARGET_PLAN_REVISION = 19
_M18_TARGET_EVENT_WATERMARK = 225
_M18_TARGET_PROJECTION_CID = (
    "baguqeeraqsmnfs6rzwc6bvjjdszmryjtdrtg5aosnk2wdsppaxymsrrpqyxa"
)
_M18_TARGET_QUACK_PORT = 24_061
_M17_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m17/control.duckdb"
)
_M17_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m17/control.coordination.duckdb"
)
_M17_GENERATION = 18
_M17_TARGET_PLAN_REVISION = 18
_M17_TARGET_EVENT_WATERMARK = 211
_M17_TARGET_PROJECTION_CID = (
    "baguqeerat5ph3demwcyfmvtxjsq4dxdei5lf6xva2jhylwvgh2s4ewttscza"
)
_M17_TARGET_QUACK_PORT = 24_060
_M16_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m16/control.duckdb"
)
_M16_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m16/control.coordination.duckdb"
)
_M16_GENERATION = 17
_M16_TARGET_PLAN_REVISION = 17
_M16_TARGET_EVENT_WATERMARK = 209
_M16_TARGET_PROJECTION_CID = (
    "baguqeerakzd5xe55z5l6nifvwumea7unzobnhokoigbfkkdzg2chmrl4xa6q"
)
_M16_TARGET_QUACK_PORT = 24_059
_M16_RECEIPT_KEYS = frozenset(
    {
        "schema", "authoritative", "control_database_is_authority",
        "coordination_database_is_authority", "receipt_is_final_pair_commit_marker",
        "migration_revision", "program_definition_cid",
        "current_source_binding_cid", "validation_digest", "migration_digest",
        "migration_evidence_id", "plan_migration_event_id",
        "migration_evidence_event_id", "task_rearm_event_ids",
        "task_rearm_receipt_cids", "coordination_rearm_event_ids",
        "coordination_rearm_ids", "plan_projection_cid",
        "migration_projection_cid", "projection_cid",
        "migration_event_watermark", "target_event_watermark",
        "target_generation", "target_quack_port", "target_runtime_root",
        "database_path", "coordination_path", "control_store_sha256",
        "control_store_size", "coordination_store_sha256",
        "coordination_store_size", "semantic_authority_digest",
        "frozen_base_authority_digest", "append_surface_digest",
        "catalog_digest", "coordination_projection_digest",
        "coordination_event_count", "prior_database_path",
        "prior_coordination_path", "prior_control_store_sha256",
        "prior_control_store_size", "prior_coordination_store_sha256",
        "prior_coordination_store_size", "prior_event_watermark",
        "prior_event_prefix_sha256", "prior_projection_cid",
        "prior_semantic_authority_digest",
        "prior_frozen_base_authority_digest", "prior_append_surface_digest",
        "prior_catalog_digest", "prior_coordination_projection_digest",
        "prior_coordination_event_count", "prior_generation",
        "prior_plan_revision", "prior_server_id", "prior_process_birth_id",
        "prior_startup_epoch", "prior_started_at", "prior_stopped_at",
        "prior_stopped_status_projection_path",
        "prior_stopped_status_projection_sha256",
        "prior_stopped_status_projection_size", "prior_migration_receipt_path",
        "prior_migration_receipt_sha256", "prior_migration_receipt_size",
        "prior_migration_receipt_cid", "repair_source_commit",
        "accepted_source_repair", "failure_receipts_preserved",
        "operator_task_rearms", "runtime_root_changed",
        "control_and_coordination_bases_copied", "prior_control_store_mutated",
        "prior_coordination_store_mutated", "prior_lifecycle_artifacts_preserved",
        "plan_revision_changes", "evidence_node_changes",
        "coordination_semantic_changes", "task_revision_changes",
        "task_status_changes", "accepted_definition_changes",
        "accepted_completion_changes", "implementation_provider_invocations",
        "execution_sidecar_copied", "read_replica_sidecar_copied",
        "effect_claim_changes", "implementation_commit_changes",
        "merge_attempt_changes", "worker_self_approval",
        "accepted_source_retry_successor_materialization_cid", "receipt_cid",
    }
)
_M18_RECEIPT_KEYS = frozenset(
    (_M16_RECEIPT_KEYS - {"accepted_source_retry_successor_materialization_cid"})
    | {"portal_completion_persistence_successor_materialization_cid"}
)
_M13_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m13/control.duckdb"
)
_M14_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m14/control.duckdb"
)
_M14_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m14/control.coordination.duckdb"
)
_M14_GENERATION = 15
_M14_TARGET_PLAN_REVISION = 15
_M14_TARGET_EVENT_WATERMARK = 199
_M15_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m15/control.duckdb"
)
_M15_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m15/control.coordination.duckdb"
)
_M15_GENERATION = 16
_M15_TARGET_PLAN_REVISION = 16
_M15_TARGET_EVENT_WATERMARK = 201
_M15_RECEIPT_KEYS = frozenset(
    {
        "schema", "authoritative", "control_database_is_authority",
        "coordination_database_is_authority", "receipt_is_final_pair_commit_marker",
        "migration_revision", "program_definition_cid",
        "current_source_binding_cid", "validation_digest", "migration_digest",
        "migration_evidence_id", "plan_migration_event_id",
        "migration_evidence_event_id", "plan_projection_cid",
        "migration_projection_cid", "projection_cid", "migration_event_watermark",
        "target_event_watermark", "target_generation", "target_quack_port",
        "target_runtime_root", "database_path", "coordination_path",
        "control_store_sha256", "control_store_size", "coordination_store_sha256",
        "coordination_store_size", "semantic_authority_digest",
        "frozen_base_authority_digest", "append_surface_digest", "catalog_digest",
        "coordination_projection_digest", "coordination_event_count",
        "prior_database_path", "prior_coordination_path",
        "prior_control_store_sha256", "prior_control_store_size",
        "prior_coordination_store_sha256", "prior_coordination_store_size",
        "prior_event_watermark", "prior_event_prefix_sha256", "prior_projection_cid",
        "prior_semantic_authority_digest", "prior_frozen_base_authority_digest",
        "prior_append_surface_digest", "prior_catalog_digest", "prior_generation",
        "prior_plan_revision", "prior_server_id", "prior_process_birth_id",
        "prior_startup_epoch", "prior_started_at", "prior_stopped_at",
        "prior_stopped_status_projection_path",
        "prior_stopped_status_projection_sha256",
        "prior_stopped_status_projection_size", "prior_migration_receipt_path",
        "prior_migration_receipt_sha256", "prior_migration_receipt_size",
        "prior_migration_receipt_cid", "detached_launch_blocker",
        "runtime_root_changed", "runtime_root_collision_removed",
        "control_and_coordination_bases_copied", "prior_control_store_mutated",
        "prior_coordination_store_mutated", "prior_lifecycle_artifacts_preserved",
        "plan_revision_changes", "evidence_node_changes",
        "coordination_semantic_changes", "task_revision_changes",
        "task_status_changes", "accepted_definition_changes",
        "accepted_completion_changes", "implementation_provider_invocations",
        "execution_sidecar_copied", "read_replica_sidecar_copied",
        "effect_claim_changes", "implementation_commit_changes",
        "merge_attempt_changes", "worker_self_approval",
        "runtime_root_rebind_successor_materialization_cid", "receipt_cid",
    }
)
_M14_RECEIPT_KEYS = frozenset(
    {
        "schema", "authoritative", "control_database_is_authority",
        "coordination_database_is_authority", "receipt_is_final_pair_commit_marker",
        "migration_revision", "database_path", "coordination_path",
        "prior_database_path", "prior_coordination_path", "program_definition_cid",
        "current_source_binding_cid", "validation_digest", "migration_digest",
        "migration_projection_cid", "target_event_watermark", "target_generation",
        "target_quack_port", "control_store_sha256", "control_store_size",
        "coordination_store_sha256", "coordination_store_size",
        "plan_migration_event_id", "migration_evidence_event_id",
        "migration_evidence_id", "prior_control_store_sha256",
        "prior_coordination_store_sha256", "prior_stale_owner_recovery_cid",
        "prior_stopped_status_projection_present",
        "control_and_coordination_bases_copied", "prior_wals_absent",
        "prior_owner_marker_absent", "coordination_semantic_changes",
        "task_revision_changes", "task_status_changes",
        "accepted_definition_changes", "accepted_completion_changes",
        "implementation_provider_invocations", "effect_claim_changes",
        "implementation_commit_changes", "merge_attempt_changes",
        "worker_self_approval", "receipt_cid",
    }
)
_M14_RECEIPT_KEYS = _M14_RECEIPT_KEYS | frozenset(
    {
        "plan_projection_cid", "migration_event_watermark", "projection_cid",
        "coordination_projection_digest", "coordination_event_count",
        "semantic_authority_digest", "frozen_base_authority_digest",
        "append_surface_digest", "catalog_digest", "prior_control_store_size",
        "prior_coordination_store_size", "prior_event_watermark",
        "prior_event_prefix_sha256", "prior_projection_cid",
        "prior_semantic_authority_digest", "prior_frozen_base_authority_digest",
        "prior_append_surface_digest", "prior_catalog_digest",
        "prior_database_uuid", "prior_generation", "prior_server_id",
        "prior_process_birth_id", "prior_startup_epoch",
        "prior_state_server_revision", "prior_stopped_at", "prior_source_head",
        "prior_source_tree", "post_stop_source_head", "post_stop_source_tree",
        "prior_stopped_status_projection_path",
        "prior_stopped_status_projection_sha256",
        "prior_stopped_status_projection_size",
        "prior_stale_owner_recovery_receipt_path",
        "prior_stale_owner_recovery_receipt_present",
        "prior_stale_owner_recovery_receipt_sha256",
        "prior_stale_owner_recovery_receipt_size",
        "prior_migration_receipt_path", "prior_migration_receipt_sha256",
        "prior_migration_receipt_size", "prior_migration_receipt_cid",
        "stale_owner_restart_successor_materialization_cid",
        "prior_control_store_mutated", "prior_coordination_store_mutated",
        "prior_lifecycle_artifacts_preserved", "plan_revision_changes",
        "evidence_node_changes", "execution_sidecar_copied",
        "read_replica_sidecar_copied",
    }
)
_M13_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m13/control.coordination.duckdb"
)
_M13_GENERATION = 14
_M13_TARGET_PLAN_REVISION = 14
_M13_TARGET_EVENT_WATERMARK = 191
_M13_PRIOR_CONTROL_STORE_SHA256 = (
    "9b09d7dcb2079963a20c6e22fa0a5ff1e0e6833a2a8ae027ac36a973adc037f2"
)
_M13_PRIOR_CONTROL_STORE_SIZE = 45_101_056
_M13_PRIOR_READ_REPLICA_STORE_SHA256 = _M13_PRIOR_CONTROL_STORE_SHA256
_M13_PRIOR_READ_REPLICA_STORE_SIZE = _M13_PRIOR_CONTROL_STORE_SIZE
_M13_PRIOR_COORDINATION_STORE_SHA256 = (
    "113f5b629317effeada5b8b81a0f8b8aaf4322ab36b6c111d830c5d9095eea55"
)
_M13_PRIOR_COORDINATION_STORE_SIZE = 12_857_344
_M13_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "authoritative",
        "control_database_is_authority",
        "coordination_database_is_authority",
        "receipt_is_final_pair_commit_marker",
        "migration_revision",
        "program_definition_cid",
        "plan_projection_cid",
        "migration_projection_cid",
        "migration_event_watermark",
        "projection_cid",
        "target_event_watermark",
        "target_generation",
        "target_quack_port",
        "validation_digest",
        "migration_digest",
        "migration_evidence_id",
        "plan_migration_event_id",
        "migration_evidence_event_id",
        "coordination_projection_digest",
        "coordination_event_count",
        "control_store_sha256",
        "control_store_size",
        "coordination_store_sha256",
        "coordination_store_size",
        "semantic_authority_digest",
        "frozen_base_authority_digest",
        "append_surface_digest",
        "prior_database_path",
        "prior_coordination_path",
        "database_path",
        "coordination_path",
        "prior_publication_control_store_sha256",
        "prior_control_store_sha256",
        "prior_coordination_store_sha256",
        "prior_control_wal_present",
        "prior_coordination_wal_present",
        "prior_owner_status_present",
        "prior_read_replica_store_sha256",
        "prior_read_replica_is_authority",
        "prior_read_replica_copied",
        "prior_event_prefix_sha256",
        "prior_source_binding_cid",
        "current_source_binding_cid",
        "prior_materialization_receipt_cid",
        "prior_materialization_receipt_precedes_failed_start_checkpoint",
        "declared_output_retry_successor_materialization_cid",
        "quack_refresh_successor_materialization_cid",
        "failed_quack_start_cid",
        "control_and_coordination_bases_copied",
        "prior_wals_absent",
        "prior_control_store_mutated",
        "prior_coordination_store_mutated",
        "prior_failed_start_artifacts_preserved",
        "plan_revision_changes",
        "evidence_node_changes",
        "coordination_semantic_changes",
        "task_revision_changes",
        "task_status_changes",
        "accepted_definition_changes",
        "accepted_completion_changes",
        "implementation_provider_invocations",
        "execution_sidecar_copied",
        "read_replica_sidecar_copied",
        "effect_claim_changes",
        "implementation_commit_changes",
        "merge_attempt_changes",
        "worker_self_approval",
        "receipt_cid",
    }
)
_M12_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m12/control.duckdb"
)
_M12_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m12/control.coordination.duckdb"
)
_M12_GENERATION = 14
_M12_TARGET_PLAN_REVISION = 13
_M12_TARGET_EVENT_WATERMARK = 189
_M12_TARGET_PROJECTION_CID = (
    "baguqeeragb5uufggmw6glss2ttbubyb2aixhfio6cmmit6w4njiuozj3csmq"
)
_M12_TARGET_COORDINATION_PROJECTION_DIGEST = (
    "sha256:358cd0667be10fb125476ac090db3cda9aec9b8d0876a2654de1ce5aa531c59f"
)
_M12_TARGET_COORDINATION_EVENT_COUNT = 224
_M12_TARGET_SEMANTIC_AUTHORITY_DIGEST = (
    "sha256:239db939a5e7af260f334b325c4ec075a2faf18f01b683448625647e0962d36c"
)
_M12_DECLARED_OUTPUT_RETRY_AUTHORITY_CID = (
    "sha256:786dde1f1728b907c3e28e5a09c746842c0a4a5ac0333f3ccb25f5290e8227a8"
)
_M12_PRIOR_CONTROL_STORE_SHA256 = (
    "d32a30bf320a07b2ebf9ebd1ee66012651346b51247a2fbec983a91772f4ee48"
)
_M12_PRIOR_CONTROL_STORE_SIZE = 45_101_056
_M12_PRIOR_COORDINATION_STORE_SHA256 = (
    "0c7335f3e9545859ad2da7ebc384eedee2302fd9a69b38e8e4df459b4930a0d5"
)
_M12_PRIOR_COORDINATION_STORE_SIZE = 12_857_344
_M12_LIVE_FAILURE_RECEIPT = MappingProxyType(
    {
        "attempt_id": "attempt:86d488ba8d7a4d3c9ed241c08d98eb36",
        "attempt_number": 3,
        "automatic_retry_admitted": False,
        "claim_id": "claim:a02a94800a9d4af89bc6e4724b6e01cf",
        "control_expected_revision": 14,
        "control_expected_status": "in_progress",
        "effect_claim_count": 0,
        "failure_kind": "terminal_portal_bridge_error",
        "failure_payload_digest": (
            "sha256:2606827e0e4880051b8c4b09a94605ffca73805a22b71f8ca00df5c97c71f879"
        ),
        "fence_epoch": 3,
        "fencing_token": 3,
        "lease_id": "lease:dbee38d120fc468b84e7260828dc103c",
        "operation": "database_task_claim_failure",
        "owner_session_id": "embedded-store:68915cfc68d681f24844b30951e7ae5c",
        "provider_invocation_count": 0,
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "task-claim-failure-settlement@1"
        ),
        "settlement_id": (
            "baguqeerafg6ci2g5mrq7ilfpa2qvinghhjqq2hdzmfprut4sbdiqosoxi5pq"
        ),
        "task_cid": (
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
    }
)
_M12_LIVE_CLAIM_RECEIPT = MappingProxyType(
    {
        "attempt_id": "attempt:86d488ba8d7a4d3c9ed241c08d98eb36",
        "attempt_number": 3,
        "claim_id": "claim:a02a94800a9d4af89bc6e4724b6e01cf",
        "fence_epoch": 3,
        "fencing_token": 3,
        "lease_id": "lease:dbee38d120fc468b84e7260828dc103c",
        "operation": "database_claim",
        "owner_session_id": "embedded-store:68915cfc68d681f24844b30951e7ae5c",
    }
)
_M12_REARM_RECEIPT = MappingProxyType(
    {
        "operation": "operator_control_plane_repair",
        "settlement_id": _M12_LIVE_FAILURE_RECEIPT["settlement_id"],
    }
)
_M11_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m11/control.duckdb"
)
_M11_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m11/control.coordination.duckdb"
)
_M11_GENERATION = 13
_M11_TARGET_PLAN_REVISION = 12
_M11_TARGET_EVENT_WATERMARK = 184
_M11_TARGET_PROJECTION_CID = (
    "baguqeeratdygaminyfax543hh5bik3kj5admm5filgtu37a3jfnyc6snwnxa"
)
_M11_TARGET_COORDINATION_PROJECTION_DIGEST = (
    "sha256:3b9ce361244387492da0156888cf6cb7377a020ca1f988b37509050975205ae7"
)
_M11_TARGET_COORDINATION_EVENT_COUNT = 56
_M11_TARGET_SEMANTIC_AUTHORITY_DIGEST = (
    "sha256:d9a1f5bd53884346a847cb9e440b5e8e37c7e4e7a87ea52e173d719f2933821c"
)
_M11_LIVE_PROVIDER_RETRY_AUTHORITY_CID = (
    "sha256:7f8404735adae0fb7ed890cda97dc674b78ae915efa193207e98abd95eb5193e"
)
_M11_PRIOR_COORDINATION_STORE_SHA256 = (
    "e69a4c22a1a1a8bd38ccf210803a02ecb040d0ee9d4cc776165a4570425d6862"
)
_M11_PRIOR_COORDINATION_STORE_SIZE = 9_711_616
_M11_PRIOR_COORDINATION_WAL_SHA256 = (
    "938dbc7028ec438889019c352b5a2529cc8c0f0fd18a11f0eb157a6772fc5321"
)
_M11_PRIOR_COORDINATION_WAL_SIZE = 24_746
_M10_FAILED_SEMANTIC_AUTHORITY_DIGEST = (
    "sha256:7dcdccefb3a54d4604716427bed164e36032429477612d9c92f3bf75fc4f9d46"
)
_M10_LIVE_FAILURE_CID = (
    "sha256:a1fae6ef892b02a19b1155a0a5ed7c0eefa8f8255f3949e25d5a8b8b9d721676"
)
_M10_FAILURE_RECEIPT = MappingProxyType(
    {
        "attempt_id": "attempt:5a657254c0b84063911e8292ef21a538",
        "attempt_number": 2,
        "automatic_retry_admitted": False,
        "claim_id": "claim:5fe583ac0e04406d8c975e8ddd2dcb3d",
        "control_expected_revision": 11,
        "control_expected_status": "in_progress",
        "effect_claim_count": 0,
        "failure_kind": "terminal_portal_bridge_error",
        "failure_payload_digest": (
            "sha256:ae39b0949ee96383a51692123359a59a787c355a6736db3ff332852d98144fcd"
        ),
        "fence_epoch": 2,
        "fencing_token": 2,
        "lease_id": "lease:6110decfef434aa39f16244d717c752d",
        "operation": "database_task_claim_failure",
        "owner_session_id": "embedded-store:5e8dddfdd778e8774949a4dea37110af",
        "provider_invocation_count": 0,
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "task-claim-failure-settlement@1"
        ),
        "settlement_id": (
            "baguqeera7cwinhpjgl2etuitwiuhs4ts6lix5txyb2pjtsbfz2npfyryznma"
        ),
        "task_cid": (
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
    }
)
_M10_CLAIM_RECEIPT = MappingProxyType(
    {
        "attempt_id": "attempt:5a657254c0b84063911e8292ef21a538",
        "attempt_number": 2,
        "claim_id": "claim:5fe583ac0e04406d8c975e8ddd2dcb3d",
        "fence_epoch": 2,
        "fencing_token": 2,
        "lease_id": "lease:6110decfef434aa39f16244d717c752d",
        "operation": "database_claim",
        "owner_session_id": "embedded-store:5e8dddfdd778e8774949a4dea37110af",
    }
)
_M11_REARM_RECEIPT = MappingProxyType(
    {
        "operation": "operator_control_plane_repair",
        "settlement_id": _M10_FAILURE_RECEIPT["settlement_id"],
    }
)
_M10_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m10/control.duckdb"
)
_M10_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m10/control.coordination.duckdb"
)
_M10_GENERATION = 12
_M9_HEAD_SEMANTIC_AUTHORITY_DIGEST = (
    "sha256:a4903791c91cc2e9c3337f2abdfd6af78389f7036d54cb7f8e43246cd4f0c023"
)
_M9_REARM_RECEIPT = MappingProxyType(
    {
        "operation": "operator_control_plane_repair",
        "settlement_id": (
            "baguqeeransrfearh5ojrnru6ls43mn7xsx6mlhndkhokrn4k7wlhowxjnshq"
        ),
    }
)
_M9_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m9/control.duckdb"
)
_M9_COORDINATION_STORE_ID = (
    "data/agent_supervisor/semantic_addressed_world_model/"
    "run-r2-m9/control.coordination.duckdb"
)
_M9_GENERATION = 11


class OperatorError(RuntimeError):
    pass


def _load_script(relative: str, name: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OperatorError(f"cannot load sealed operator script: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OperatorError("scheduler config must be an object")
    return value


def _emit(value: Mapping[str, Any]) -> int:
    secrets = {
        str(secret)
        for name, secret in os.environ.items()
        if name.endswith("_QUACK_TOKEN") and len(str(secret)) >= 8
    }

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            normalized: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise OperatorError("operator JSON output keys must be strings")
                safe_key = key
                for secret in sorted(secrets, key=len, reverse=True):
                    safe_key = safe_key.replace(secret, "<redacted-quack-token>")
                if safe_key in normalized:
                    raise OperatorError("operator JSON output keys collide after redaction")
                normalized[safe_key] = normalize(child)
            return normalized
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise OperatorError("operator JSON output contains a nonfinite number")
            return item
        if isinstance(item, str):
            rendered = item
            for secret in sorted(secrets, key=len, reverse=True):
                rendered = rendered.replace(secret, "<redacted-quack-token>")
            return rendered
        raise OperatorError(
            f"operator JSON output contains unsupported {type(item).__name__}"
        )

    normalized = normalize(value)
    print(json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False))
    return 0 if normalized.get("valid", True) is True else 2


def _credential_safe_error(exc: BaseException) -> str:
    """Render an operator error without exposing a live Quack credential."""

    message = f"{type(exc).__name__}: {exc}"
    secrets = {
        str(value)
        for name, value in os.environ.items()
        if name.endswith("_QUACK_TOKEN") and len(str(value)) >= 8
    }
    for secret in sorted(secrets, key=len, reverse=True):
        message = message.replace(secret, "<redacted-quack-token>")
    return message


def _validator(relative: str, function: str) -> dict[str, Any]:
    module = _load_script(relative, "_sawm_operator_" + function)
    return dict(getattr(module, function)(REPO_ROOT))


def _materializer():
    return _load_script("scripts/materialize_semantic_addressed_world_model_program.py", "_sawm_operator_materializer")


def _active_source_repair_materialization(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the newest sealed successor authority by key presence.

    M13 is the source-only initial-refresh lifecycle repair over the preserved
    stopped M12 failure.  Every newer key is selected by presence and a
    malformed value fails closed rather than silently selecting older evidence.
    """

    m34_key = "json_emission_normalization_successor_materialization"
    m33_key = "live_preflight_contract_successor_materialization"
    m32_key = "live_preflight_plan_anchor_successor_materialization"
    m31_key = "detached_coordinator_pid_recovery_successor_materialization"
    m30_key = "stopped_owner_restart_source_seal_successor_materialization"
    m29_key = "committed_evidence_verification_successor_materialization"
    m28_key = "live_claim_admission_recovery_successor_materialization"
    m27_key = "dead_owner_parallel_resume_successor_materialization"
    m26_key = "automatic_stall_recovery_successor_materialization"
    m25_key = "native_duckdb_preload_successor_materialization"
    m24_key = "multi_lane_sidecar_reopen_successor_materialization"
    m23_key = "multi_lane_successor_materialization"
    m22_key = "live_preflight_receipt_compatibility_successor_materialization"
    m21_key = "generation_realization_successor_materialization"
    m20_key = "test_isolation_successor_materialization"
    m19_key = "live_catalog_inventory_successor_materialization"
    m18_key = "portal_completion_persistence_successor_materialization"
    m17_key = "source_binding_successor_materialization"
    m16_key = "accepted_source_retry_successor_materialization"
    m15_key = "runtime_root_rebind_successor_materialization"
    m14_key = "stale_owner_restart_successor_materialization"
    quack_refresh_key = "quack_refresh_successor_materialization"
    declared_output_retry_key = "declared_output_retry_successor_materialization"
    provider_retry_key = "live_provider_retry_successor_materialization"
    projection_key = "live_projection_successor_materialization"
    recovery_key = "live_recovery_successor_materialization"
    successor_key = "source_repair_successor_materialization"
    historical_key = "source_repair_materialization"
    if m34_key in config:
        authority = config.get(m34_key)
        try:
            materializer = _materializer()
            expected = materializer._expected_m34_json_emission_normalization_authority()
            reference = materializer._m34_authority_reference()
            materializer._validated_m34_live_preflight_contract(expected)
        except Exception as exc:
            raise OperatorError("active M34 authority is unavailable") from exc
        target_root = str(Path(_M34_STORE_ID).parent)
        expected_runtime = {
            "root": target_root,
            "state": f"{target_root}/state",
            "worktrees": _M34_WORKTREE_ROOT,
            "merge_queue": f"{target_root}/merge-queue",
            "logs": f"{target_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }
        program = config.get("database_program")
        owner = config.get("quack_owner")
        binding = expected.get("runtime_binding")
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != reference
            or expected.get("migration_revision") != "SAWM-R2-M34"
            or not isinstance(binding, Mapping)
            or binding.get("store_id") != _M34_STORE_ID
            or binding.get("store_generation") != _M34_GENERATION
            or binding.get("target_event_watermark") != _M34_TARGET_EVENT_WATERMARK
            or binding.get("quack_port") != _M34_TARGET_QUACK_PORT
            or expected.get("target_projection_cid") != _M34_TARGET_PROJECTION_CID
            or not isinstance(program, Mapping)
            or program.get("store_id") != _M34_STORE_ID
            or program.get("store_generation") != str(_M34_GENERATION)
            or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
            or program.get("worktree_root") != _M34_WORKTREE_ROOT
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M34_STORE_ID
            or owner.get("store_id") != _M34_STORE_ID
            or owner.get("port") != _M34_TARGET_QUACK_PORT
            or owner.get("state_dir") != f"{target_root}/quack-owner"
            or config.get("runtime_paths") != expected_runtime
        ):
            raise OperatorError(
                "active M34 JSON-emission normalization authority is invalid"
            )
        return MappingProxyType(expected)
    if m33_key in config:
        authority = config.get(m33_key)
        try:
            expected = _materializer()._expected_m33_live_preflight_contract_authority()
            reference = _materializer()._m33_authority_reference()
            _materializer()._validated_m33_live_preflight_contract(expected)
        except Exception as exc:
            raise OperatorError("active M33 authority is unavailable") from exc
        target_root = str(Path(_M33_STORE_ID).parent)
        expected_runtime = {
            "root": target_root,
            "state": f"{target_root}/state",
            "worktrees": _M33_WORKTREE_ROOT,
            "merge_queue": f"{target_root}/merge-queue",
            "logs": f"{target_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }
        program = config.get("database_program")
        owner = config.get("quack_owner")
        binding = expected.get("runtime_binding")
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != reference
            or expected.get("migration_revision") != "SAWM-R2-M33"
            or not isinstance(binding, Mapping)
            or binding.get("store_id") != _M33_STORE_ID
            or binding.get("store_generation") != _M33_GENERATION
            or binding.get("target_event_watermark") != _M33_TARGET_EVENT_WATERMARK
            or binding.get("quack_port") != _M33_TARGET_QUACK_PORT
            or expected.get("target_projection_cid") != _M33_TARGET_PROJECTION_CID
            or not isinstance(program, Mapping)
            or program.get("store_id") != _M33_STORE_ID
            or program.get("store_generation") != str(_M33_GENERATION)
            or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
            or program.get("worktree_root") != _M33_WORKTREE_ROOT
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M33_STORE_ID
            or owner.get("store_id") != _M33_STORE_ID
            or owner.get("port") != _M33_TARGET_QUACK_PORT
            or owner.get("state_dir") != f"{target_root}/quack-owner"
            or config.get("runtime_paths") != expected_runtime
        ):
            raise OperatorError("active M33 live-preflight contract authority is invalid")
        return MappingProxyType(expected)
    if m32_key in config:
        authority = config.get(m32_key)
        try:
            expected = (
                _materializer()._expected_m32_live_preflight_plan_anchor_authority()
            )
            reference = _materializer()._m32_authority_reference()
        except Exception as exc:
            raise OperatorError("active M32 authority is unavailable") from exc
        target_root = str(Path(_M32_STORE_ID).parent)
        expected_runtime = {
            "root": target_root,
            "state": f"{target_root}/state",
            "worktrees": _M32_WORKTREE_ROOT,
            "merge_queue": f"{target_root}/merge-queue",
            "logs": f"{target_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }
        program = config.get("database_program")
        owner = config.get("quack_owner")
        binding = expected.get("runtime_binding")
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != reference
            or expected.get("migration_revision") != "SAWM-R2-M32"
            or not isinstance(binding, Mapping)
            or binding.get("store_id") != _M32_STORE_ID
            or binding.get("store_generation") != _M32_GENERATION
            or binding.get("target_event_watermark") != _M32_TARGET_EVENT_WATERMARK
            or binding.get("quack_port") != _M32_TARGET_QUACK_PORT
            or expected.get("target_projection_cid") != _M32_TARGET_PROJECTION_CID
            or not isinstance(program, Mapping)
            or program.get("store_id") != _M32_STORE_ID
            or program.get("store_generation") != str(_M32_GENERATION)
            or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
            or program.get("worktree_root") != _M32_WORKTREE_ROOT
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M32_STORE_ID
            or owner.get("store_id") != _M32_STORE_ID
            or owner.get("port") != _M32_TARGET_QUACK_PORT
            or owner.get("state_dir") != f"{target_root}/quack-owner"
            or config.get("runtime_paths") != expected_runtime
        ):
            raise OperatorError("active M32 live-preflight plan authority is invalid")
        return MappingProxyType(expected)
    if m31_key in config:
        authority = config.get(m31_key)
        try:
            expected = (
                _materializer()
                ._expected_m31_detached_coordinator_pid_recovery_authority()
            )
        except Exception as exc:
            raise OperatorError("active M31 authority is unavailable") from exc
        target_root = str(Path(_M31_STORE_ID).parent)
        expected_runtime = {
            "root": target_root,
            "state": f"{target_root}/state",
            "worktrees": _M31_WORKTREE_ROOT,
            "merge_queue": f"{target_root}/merge-queue",
            "logs": f"{target_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }
        program = config.get("database_program")
        owner = config.get("quack_owner")
        binding = authority.get("runtime_binding") if isinstance(authority, Mapping) else None
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != expected
            or authority.get("migration_revision") != "SAWM-R2-M31"
            or not isinstance(binding, Mapping)
            or binding.get("store_id") != _M31_STORE_ID
            or binding.get("store_generation") != _M31_GENERATION
            or binding.get("target_event_watermark") != _M31_TARGET_EVENT_WATERMARK
            or binding.get("quack_port") != _M31_TARGET_QUACK_PORT
            or authority.get("target_projection_cid") != _M31_TARGET_PROJECTION_CID
            or not isinstance(program, Mapping)
            or program.get("store_id") != _M31_STORE_ID
            or program.get("store_generation") != str(_M31_GENERATION)
            or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
            or program.get("worktree_root") != _M31_WORKTREE_ROOT
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M31_STORE_ID
            or owner.get("store_id") != _M31_STORE_ID
            or owner.get("port") != _M31_TARGET_QUACK_PORT
            or owner.get("state_dir") != f"{target_root}/quack-owner"
            or config.get("runtime_paths") != expected_runtime
        ):
            raise OperatorError("active M31 detached-coordinator authority is invalid")
        return authority
    if m30_key in config:
        authority = config.get(m30_key)
        try:
            expected = (
                _materializer()
                ._expected_m30_stopped_owner_restart_source_seal_authority()
            )
        except Exception as exc:
            raise OperatorError("active M30 authority is unavailable") from exc
        target_root = str(Path(_M30_STORE_ID).parent)
        expected_runtime = {
            "root": target_root,
            "state": f"{target_root}/state",
            "worktrees": _M30_WORKTREE_ROOT,
            "merge_queue": f"{target_root}/merge-queue",
            "logs": f"{target_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }
        program = config.get("database_program")
        owner = config.get("quack_owner")
        runtime = config.get("runtime_paths")
        binding = authority.get("runtime_binding") if isinstance(authority, Mapping) else None
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != expected
            or authority.get("migration_revision") != "SAWM-R2-M30"
            or not isinstance(binding, Mapping)
            or binding.get("store_id") != _M30_STORE_ID
            or binding.get("store_generation") != _M30_GENERATION
            or binding.get("target_event_watermark") != _M30_TARGET_EVENT_WATERMARK
            or binding.get("quack_port") != _M30_TARGET_QUACK_PORT
            or authority.get("target_projection_cid") != _M30_TARGET_PROJECTION_CID
            or not isinstance(program, Mapping)
            or program.get("store_id") != _M30_STORE_ID
            or program.get("store_generation") != str(_M30_GENERATION)
            or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
            or program.get("worktree_root") != _M30_WORKTREE_ROOT
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M30_STORE_ID
            or owner.get("store_id") != _M30_STORE_ID
            or owner.get("port") != _M30_TARGET_QUACK_PORT
            or owner.get("state_dir") != f"{target_root}/quack-owner"
            or runtime != expected_runtime
        ):
            raise OperatorError("active M30 stopped-owner restart authority is invalid")
        return authority
    if m29_key in config:
        authority = config.get(m29_key)
        try:
            expected = (
                _materializer()
                ._expected_m29_committed_evidence_verification_authority()
            )
        except Exception as exc:
            raise OperatorError(
                "active M29 committed-evidence authority is unavailable"
            ) from exc
        program = config.get("database_program")
        owner = config.get("quack_owner")
        runtime = config.get("runtime_paths")
        target_root = str(Path(_M29_STORE_ID).parent)
        expected_runtime = {
            "root": target_root,
            "state": f"{target_root}/state",
            "worktrees": _M29_WORKTREE_ROOT,
            "merge_queue": f"{target_root}/merge-queue",
            "logs": f"{target_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }
        binding = (
            authority.get("runtime_binding")
            if isinstance(authority, Mapping)
            else None
        )
        changes = (
            authority.get("exact_changes")
            if isinstance(authority, Mapping)
            else None
        )
        preservation = (
            authority.get("preservation")
            if isinstance(authority, Mapping)
            else None
        )
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != expected
            or authority.get("migration_revision") != "SAWM-R2-M29"
            or not isinstance(binding, Mapping)
            or binding.get("store_id") != _M29_STORE_ID
            or binding.get("coordination_store_id")
            != _M29_COORDINATION_STORE_ID
            or binding.get("store_generation") != _M29_GENERATION
            or binding.get("plan_revision") != _M29_TARGET_PLAN_REVISION
            or binding.get("target_event_watermark")
            != _M29_TARGET_EVENT_WATERMARK
            or binding.get("quack_port") != _M29_TARGET_QUACK_PORT
            or authority.get("target_authority", {}).get("projection_cid")
            != _M29_TARGET_PROJECTION_CID
            or not isinstance(changes, Mapping)
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or any(
                changes.get(name) != 0
                for name in (
                    "task_revision_changes", "task_status_changes",
                    "plan_revision_changes", "accepted_completion_changes",
                    "coordination_semantic_changes", "sidecar_changes",
                    "store_generation_row_changes", "state_server_row_changes",
                    "credential_row_changes",
                )
            )
            or not isinstance(preservation, Mapping)
            or preservation.get("failed_m28_post_append_attempt_preserved")
            is not True
            or preservation.get("m28_receipt_created_or_rewritten") is not False
            or preservation.get("sidecars_preserved") is not True
            or preservation.get("worker_self_approval") is not False
            or not isinstance(program, Mapping)
            or program.get("store_id") != _M29_STORE_ID
            or program.get("store_generation") != str(_M29_GENERATION)
            or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
            or program.get("worktree_root") != _M29_WORKTREE_ROOT
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M29_STORE_ID
            or owner.get("store_id") != _M29_STORE_ID
            or owner.get("port") != _M29_TARGET_QUACK_PORT
            or owner.get("state_dir") != f"{target_root}/quack-owner"
            or not isinstance(runtime, Mapping)
            or dict(runtime) != expected_runtime
        ):
            raise OperatorError(
                "active M29 committed-evidence authority is invalid"
            )
        return authority
    if m28_key in config:
        authority = config.get(m28_key)
        try:
            expected = (
                _materializer()
                ._expected_m28_live_claim_admission_recovery_authority()
            )
        except Exception as exc:
            raise OperatorError(
                "active M28 live-claim recovery authority is unavailable"
            ) from exc
        program = config.get("database_program")
        owner = config.get("quack_owner")
        runtime = config.get("runtime_paths")
        target_root = str(Path(_M28_STORE_ID).parent)
        expected_runtime = {
            "root": target_root,
            "state": f"{target_root}/state",
            "worktrees": _M28_WORKTREE_ROOT,
            "merge_queue": f"{target_root}/merge-queue",
            "logs": f"{target_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }
        binding = authority.get("runtime_binding") if isinstance(authority, Mapping) else None
        changes = authority.get("exact_changes") if isinstance(authority, Mapping) else None
        preservation = authority.get("preservation") if isinstance(authority, Mapping) else None
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != expected
            or authority.get("migration_revision") != "SAWM-R2-M28"
            or not isinstance(binding, Mapping)
            or binding.get("store_id") != _M28_STORE_ID
            or binding.get("coordination_store_id") != _M28_COORDINATION_STORE_ID
            or binding.get("store_generation") != _M28_GENERATION
            or binding.get("plan_revision") != _M28_TARGET_PLAN_REVISION
            or binding.get("target_event_watermark")
            != _M28_TARGET_EVENT_WATERMARK
            or binding.get("quack_port") != _M28_TARGET_QUACK_PORT
            or authority.get("target_authority", {}).get("projection_cid")
            != _M28_TARGET_PROJECTION_CID
            or not isinstance(changes, Mapping)
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or any(
                changes.get(name) != 0
                for name in (
                    "task_revision_changes", "task_status_changes",
                    "plan_revision_changes", "accepted_completion_changes",
                    "coordination_semantic_changes", "sidecar_changes",
                )
            )
            or not isinstance(preservation, Mapping)
            or preservation.get("sidecars_preserved") is not True
            or preservation.get("worker_self_approval") is not False
            or not isinstance(program, Mapping)
            or program.get("store_id") != _M28_STORE_ID
            or program.get("store_generation") != "27"
            or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
            or program.get("worktree_root") != _M28_WORKTREE_ROOT
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M28_STORE_ID
            or owner.get("store_id") != _M28_STORE_ID
            or owner.get("port") != _M28_TARGET_QUACK_PORT
            or owner.get("state_dir") != f"{target_root}/quack-owner"
            or not isinstance(runtime, Mapping)
            or dict(runtime) != expected_runtime
        ):
            raise OperatorError("active M28 live-claim recovery authority is invalid")
        return authority
    if m27_key in config:
        authority = config.get(m27_key)
        try:
            expected = (
                _materializer()._expected_m27_dead_owner_parallel_resume_authority()
            )
        except Exception as exc:
            raise OperatorError(
                "active M27 dead-owner parallel-resume authority is unavailable"
            ) from exc
        program = config.get("database_program")
        owner = config.get("quack_owner")
        runtime = config.get("runtime_paths")
        lanes = config.get("lanes")
        expected_lanes = [
            (0, "sawm-lane-0", 0, ["SAWM-008"]),
            (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
            (2, "sawm-lane-2", 2, ["SAWM-015"]),
            (3, "sawm-lane-3", 3, ["SAWM-012"]),
        ]
        observed_lanes = (
            [
                (
                    lane.get("index"),
                    lane.get("name"),
                    lane.get("strict_shard_remainder"),
                    lane.get("initial_task_ids"),
                )
                for lane in lanes
            ]
            if isinstance(lanes, list)
            and all(isinstance(lane, Mapping) for lane in lanes)
            else None
        )
        target_root = str(Path(_M27_STORE_ID).parent)
        expected_runtime = {
            "root": target_root,
            "state": f"{target_root}/state",
            "worktrees": _M27_WORKTREE_ROOT,
            "merge_queue": f"{target_root}/merge-queue",
            "logs": f"{target_root}/logs",
            "generated_runtime_artifacts_are_completion_authority": False,
        }
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != expected
            or authority.get("migration_revision") != "SAWM-R2-M27"
            or authority.get("target_store_id") != _M27_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M27_COORDINATION_STORE_ID
            or authority.get("target_generation") != _M27_GENERATION
            or authority.get("target_plan_revision")
            != _M27_TARGET_PLAN_REVISION
            or authority.get("target_event_watermark")
            != _M27_TARGET_EVENT_WATERMARK
            or authority.get("target_quack_port") != _M27_TARGET_QUACK_PORT
            or set(authority.get("interrupted_claims", {}))
            != {"SAWM-006", "SAWM-008", "SAWM-012", "SAWM-015"}
            or set(authority.get("task_rearms", {}))
            != {"SAWM-006", "SAWM-012"}
            or authority.get("accepted_completion_changes") != 0
            or authority.get("worker_self_approval") is not False
            or not isinstance(program, Mapping)
            or program.get("store_id") != _M27_STORE_ID
            or program.get("store_generation") != "26"
            or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
            or program.get("worktree_root") != _M27_WORKTREE_ROOT
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M27_STORE_ID
            or owner.get("store_id") != _M27_STORE_ID
            or owner.get("port") != _M27_TARGET_QUACK_PORT
            or owner.get("state_dir") != f"{target_root}/quack-owner"
            or not isinstance(runtime, Mapping)
            or dict(runtime) != expected_runtime
            or config.get("max_lanes") != 4
            or config.get("strict_task_sharding") is not True
            or config.get("idle_lane_work_stealing") != ""
            or observed_lanes != expected_lanes
        ):
            raise OperatorError(
                "active M27 dead-owner parallel-resume authority is invalid"
            )
        return authority
    if m26_key in config:
        authority = config.get(m26_key)
        try:
            materializer = _materializer()
            expected = (
                materializer._expected_m26_automatic_stall_recovery_authority()
            )
        except Exception as exc:
            raise OperatorError(
                "active M26 automatic-stall-recovery authority is unavailable"
            ) from exc
        program = config.get("database_program")
        owner = config.get("quack_owner")
        runtime = config.get("runtime_paths")
        lanes = config.get("lanes")
        expected_lanes = [
            (0, "sawm-lane-0", 0, ["SAWM-008"]),
            (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
            (2, "sawm-lane-2", 2, ["SAWM-015"]),
            (3, "sawm-lane-3", 3, ["SAWM-012"]),
        ]
        observed_lanes = (
            [
                (
                    lane.get("index"),
                    lane.get("name"),
                    lane.get("strict_shard_remainder"),
                    lane.get("initial_task_ids"),
                )
                for lane in lanes
            ]
            if isinstance(lanes, list)
            and all(isinstance(lane, Mapping) for lane in lanes)
            else None
        )
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != expected
            or authority.get("migration_revision") != "SAWM-R2-M26"
            or authority.get("prior_store_id") != _M25_STORE_ID
            or authority.get("prior_coordination_store_id")
            != _M25_COORDINATION_STORE_ID
            or authority.get("target_store_id") != _M26_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M26_COORDINATION_STORE_ID
            or authority.get("target_generation") != _M26_GENERATION
            or authority.get("target_plan_revision")
            != _M26_TARGET_PLAN_REVISION
            or authority.get("target_event_watermark")
            != _M26_TARGET_EVENT_WATERMARK
            or authority.get("target_quack_port") != _M26_TARGET_QUACK_PORT
            or set(authority.get("orphan_claims", {}))
            != {"SAWM-006", "SAWM-015"}
            or set(authority.get("failure_receipts", {}))
            != {"SAWM-008", "SAWM-012"}
            or set(authority.get("task_rearms", {}))
            != {"SAWM-008", "SAWM-012"}
            or authority.get("accepted_completion_changes") != 0
            or authority.get("implementation_provider_invocations") != 0
            or authority.get("worker_self_approval") is not False
            or not isinstance(program, Mapping)
            or program.get("store_id") != _M26_STORE_ID
            or program.get("store_generation") != "25"
            or program.get("quack_endpoint") != "quack:127.0.0.1:24069"
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M26_STORE_ID
            or owner.get("store_id") != _M26_STORE_ID
            or owner.get("port") != _M26_TARGET_QUACK_PORT
            or owner.get("state_dir")
            != f"{authority.get('target_runtime_root')}/quack-owner"
            or not isinstance(runtime, Mapping)
            or runtime.get("root") != authority.get("target_runtime_root")
            or config.get("max_lanes") != 4
            or config.get("strict_task_sharding") is not True
            or config.get("idle_lane_work_stealing") != ""
            or observed_lanes != expected_lanes
        ):
            raise OperatorError(
                "active M26 automatic-stall-recovery authority is invalid"
            )
        return authority
    if m25_key in config:
        authority = config.get(m25_key)
        try:
            materializer = _materializer()
            expected = (
                materializer._expected_m25_native_duckdb_preload_authority()
            )
        except Exception as exc:
            raise OperatorError(
                "active M25 exact native-DuckDB authority is unavailable"
            ) from exc
        lanes = config.get("lanes")
        database = config.get("database_program")
        owner = config.get("quack_owner")
        runtime = config.get("runtime_paths")
        provider = config.get("provider")
        accepted_repair = (
            authority.get("accepted_control_plane_repair")
            if isinstance(authority, Mapping)
            else None
        )
        failed_start = (
            authority.get("failed_quack_start")
            if isinstance(authority, Mapping)
            else None
        )
        native_repair = (
            authority.get("native_preload_repair")
            if isinstance(authority, Mapping)
            else None
        )
        expected_lanes = [
            (0, "sawm-lane-0", 0, ["SAWM-008"]),
            (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
            (2, "sawm-lane-2", 2, ["SAWM-015"]),
            (3, "sawm-lane-3", 3, ["SAWM-012"]),
        ]
        lanes_are_exact = bool(
            isinstance(lanes, list)
            and len(lanes) == len(expected_lanes)
            and all(
                isinstance(lane, Mapping)
                and type(lane.get("index")) is int
                and type(lane.get("strict_shard_remainder")) is int
                and (
                    lane.get("index"),
                    lane.get("name"),
                    lane.get("strict_shard_remainder"),
                    lane.get("initial_task_ids"),
                )
                == expected_lane
                for lane, expected_lane in zip(
                    lanes, expected_lanes, strict=True
                )
            )
        )
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != expected
            or authority.get("schema")
            != (
                "sawm/native-duckdb-preload-successor-"
                "materialization-authorization@1"
            )
            or authority.get("migration_revision") != "SAWM-R2-M25"
            or authority.get("prior_store_id") != _M24_STORE_ID
            or authority.get("prior_coordination_store_id")
            != _M24_COORDINATION_STORE_ID
            or authority.get("prior_source_head")
            != "bb02830699388df9b52c01830c9e970a56c56796"
            or authority.get("repair_source_commit")
            != "135b077c5ad9482bbb167fdfc81d8b7855ff5fab"
            or not isinstance(accepted_repair, Mapping)
            or accepted_repair.get("precursor_source_head")
            != "bb02830699388df9b52c01830c9e970a56c56796"
            or accepted_repair.get("repair_source_commit")
            != "135b077c5ad9482bbb167fdfc81d8b7855ff5fab"
            or not isinstance(failed_start, Mapping)
            or failed_start.get("phase")
            != "pre_identity_replica_extension_load"
            or failed_start.get("failed_operation") != "LOAD httpfs"
            or failed_start.get("load_quack_reached") is not False
            or failed_start.get("quack_serve_reached") is not False
            or failed_start.get("embedded_quack_involved") is not False
            or failed_start.get("identity_publication_reached") is not False
            or failed_start.get("state_server_row_created") is not False
            or failed_start.get("store_generation_row_created") is not False
            or authority.get("failed_quack_start_cid")
            != materializer._identity(dict(failed_start))
            or not isinstance(native_repair, Mapping)
            or native_repair.get("ambient_loader_environment_rejected")
            is not True
            or native_repair.get("sanitized_process_birth_required") is not True
            or native_repair.get("in_process_loader_environment_mutation")
            is not False
            or native_repair.get("preload_before_offline_validation")
            is not True
            or native_repair.get("preload_before_owner_start") is not True
            or authority.get(
                "prior_materialization_receipt_precedes_failed_start_checkpoint"
            )
            is not True
            or authority.get("target_store_id") != _M25_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M25_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M25_STORE_ID).parent)
            or type(authority.get("target_generation")) is not int
            or authority.get("target_generation") != _M25_GENERATION
            or type(authority.get("target_quack_port")) is not int
            or authority.get("target_quack_port") != _M25_TARGET_QUACK_PORT
            or type(authority.get("target_plan_revision")) is not int
            or authority.get("target_plan_revision")
            != _M25_TARGET_PLAN_REVISION
            or type(authority.get("target_event_watermark")) is not int
            or authority.get("target_event_watermark")
            != _M25_TARGET_EVENT_WATERMARK
            or authority.get("plan_revision_changes") != 1
            or authority.get("evidence_node_changes") != 1
            or authority.get("task_revision_changes") != 0
            or authority.get("task_status_changes") != 0
            or authority.get("coordination_semantic_changes") != 0
            or authority.get("accepted_definition_changes") != 0
            or authority.get("accepted_completion_changes") != 0
            or authority.get("implementation_provider_invocations") != 0
            or authority.get("control_base_copied") is not True
            or authority.get("coordination_base_copied") is not True
            or authority.get("read_replica_sidecar_copied") is not False
            or not isinstance(database, Mapping)
            or database.get("store_id") != _M25_STORE_ID
            or database.get("store_generation") != str(_M25_GENERATION)
            or database.get("quack_endpoint")
            != f"quack:127.0.0.1:{_M25_TARGET_QUACK_PORT}"
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M25_STORE_ID
            or owner.get("store_id") != _M25_STORE_ID
            or owner.get("port") != _M25_TARGET_QUACK_PORT
            or owner.get("state_dir")
            != f"{authority.get('target_runtime_root')}/quack-owner"
            or not isinstance(runtime, Mapping)
            or dict(runtime)
            != {
                "root": authority.get("target_runtime_root"),
                "state": f"{authority.get('target_runtime_root')}/state",
                "worktrees": (
                    f"{authority.get('target_runtime_root')}/worktrees"
                ),
                "merge_queue": (
                    f"{authority.get('target_runtime_root')}/merge-queue"
                ),
                "logs": f"{authority.get('target_runtime_root')}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
            or config.get("max_lanes") != 4
            or config.get("strict_task_sharding") is not True
            or config.get("idle_lane_work_stealing") != ""
            or not lanes_are_exact
            or not isinstance(provider, Mapping)
            or type(provider.get("max_concurrency")) is not int
            or provider.get("max_concurrency", 0) < 4
        ):
            raise OperatorError(
                "active M25 native-DuckDB preload successor authority is invalid"
            )
        return authority
    if m24_key in config:
        authority = config.get(m24_key)
        try:
            expected = _materializer()._expected_m24_sidecar_reopen_authority()
        except Exception as exc:
            raise OperatorError("active M24 exact authority is unavailable") from exc
        lanes = config.get("lanes")
        database = config.get("database_program")
        owner = config.get("quack_owner")
        runtime = config.get("runtime_paths")
        provider = config.get("provider")
        expected_lanes = [
            (0, "sawm-lane-0", 0, ["SAWM-008"]),
            (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
            (2, "sawm-lane-2", 2, ["SAWM-015"]),
            (3, "sawm-lane-3", 3, ["SAWM-012"]),
        ]
        lanes_are_exact = bool(
            isinstance(lanes, list)
            and len(lanes) == len(expected_lanes)
            and all(
                isinstance(lane, Mapping)
                and type(lane.get("index")) is int
                and type(lane.get("strict_shard_remainder")) is int
                and (
                    lane.get("index"),
                    lane.get("name"),
                    lane.get("strict_shard_remainder"),
                    lane.get("initial_task_ids"),
                )
                == expected_lane
                for lane, expected_lane in zip(lanes, expected_lanes, strict=True)
            )
        )
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != expected
            or authority.get("schema")
            != "sawm/multi-lane-sidecar-reopen-successor-materialization-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M24"
            or authority.get("target_store_id") != _M24_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M24_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M24_STORE_ID).parent)
            or type(authority.get("target_generation")) is not int
            or authority.get("target_generation") != _M24_GENERATION
            or type(authority.get("target_quack_port")) is not int
            or authority.get("target_quack_port") != _M24_TARGET_QUACK_PORT
            or type(authority.get("target_plan_revision")) is not int
            or authority.get("target_plan_revision") != _M24_TARGET_PLAN_REVISION
            or type(authority.get("target_event_watermark")) is not int
            or authority.get("target_event_watermark")
            != _M24_TARGET_EVENT_WATERMARK
            or authority.get("task_revision_changes") != 2
            or authority.get("task_status_changes") != 2
            or authority.get("coordination_semantic_changes") != 2
            or not isinstance(database, Mapping)
            or database.get("store_id") != _M24_STORE_ID
            or database.get("store_generation") != str(_M24_GENERATION)
            or database.get("quack_endpoint")
            != f"quack:127.0.0.1:{_M24_TARGET_QUACK_PORT}"
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M24_STORE_ID
            or owner.get("port") != _M24_TARGET_QUACK_PORT
            or not isinstance(runtime, Mapping)
            or runtime.get("root") != authority.get("target_runtime_root")
            or config.get("max_lanes") != 4
            or config.get("strict_task_sharding") is not True
            or config.get("idle_lane_work_stealing") != ""
            or not lanes_are_exact
            or not isinstance(provider, Mapping)
            or type(provider.get("max_concurrency")) is not int
            or provider.get("max_concurrency", 0) < 4
        ):
            raise OperatorError(
                "active M24 sidecar-reopen successor authority is invalid"
            )
        return authority
    if m23_key in config:
        authority = config.get(m23_key)
        try:
            expected = _materializer()._expected_m23_multi_lane_authority()
        except Exception as exc:
            raise OperatorError("active M23 exact authority is unavailable") from exc
        lanes = config.get("lanes")
        database = config.get("database_program")
        owner = config.get("quack_owner")
        runtime = config.get("runtime_paths")
        provider = config.get("provider")
        expected_lanes = [
            (0, "sawm-lane-0", 0, ["SAWM-008"]),
            (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
            (2, "sawm-lane-2", 2, ["SAWM-015"]),
            (3, "sawm-lane-3", 3, ["SAWM-012"]),
        ]
        lanes_are_exact = bool(
            isinstance(lanes, list)
            and len(lanes) == len(expected_lanes)
            and all(
                isinstance(lane, Mapping)
                and type(lane.get("index")) is int
                and type(lane.get("strict_shard_remainder")) is int
                and (
                    lane.get("index"),
                    lane.get("name"),
                    lane.get("strict_shard_remainder"),
                    lane.get("initial_task_ids"),
                )
                == expected_lane
                for lane, expected_lane in zip(lanes, expected_lanes, strict=True)
            )
        )
        if (
            not isinstance(authority, Mapping)
            or dict(authority) != expected
            or authority.get("schema")
            != "sawm/multi-lane-successor-materialization-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M23"
            or authority.get("target_store_id") != _M23_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M23_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M23_STORE_ID).parent)
            or type(authority.get("target_generation")) is not int
            or authority.get("target_generation") != _M23_GENERATION
            or type(authority.get("target_quack_port")) is not int
            or authority.get("target_quack_port") != _M23_TARGET_QUACK_PORT
            or type(authority.get("target_plan_revision")) is not int
            or authority.get("target_plan_revision") != _M23_TARGET_PLAN_REVISION
            or type(authority.get("target_event_watermark")) is not int
            or authority.get("target_event_watermark")
            != _M23_TARGET_EVENT_WATERMARK
            or authority.get("task_revision_changes") != 2
            or authority.get("task_status_changes") != 2
            or authority.get("coordination_semantic_changes") != 2
            or not isinstance(database, Mapping)
            or database.get("store_id") != _M23_STORE_ID
            or database.get("store_generation") != str(_M23_GENERATION)
            or database.get("quack_endpoint")
            != f"quack:127.0.0.1:{_M23_TARGET_QUACK_PORT}"
            or not isinstance(owner, Mapping)
            or owner.get("database_path") != _M23_STORE_ID
            or owner.get("port") != _M23_TARGET_QUACK_PORT
            or not isinstance(runtime, Mapping)
            or runtime.get("root") != authority.get("target_runtime_root")
            or config.get("max_lanes") != 4
            or config.get("strict_task_sharding") is not True
            or config.get("idle_lane_work_stealing") != ""
            or not lanes_are_exact
            or not isinstance(provider, Mapping)
            or type(provider.get("max_concurrency")) is not int
            or provider.get("max_concurrency", 0) < 4
        ):
            raise OperatorError("active M23 multi-lane successor authority is invalid")
        return authority
    if m22_key in config:
        authority = config.get(m22_key)
        required = {
            "schema",
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_runtime_root",
            "target_generation",
            "target_quack_port",
            "target_plan_revision",
            "target_event_watermark",
            "target_coordination_projection_digest",
            "target_coordination_event_count",
            "prior_control_store_sha256",
            "prior_coordination_store_sha256",
            "prior_migration_receipt_sha256",
            "legacy_completion_receipt_cids",
            "legacy_operational_validation_receipt_cids",
            "accepted_control_plane_repair",
        }
        try:
            expected = (
                _materializer()
                ._expected_m22_live_preflight_receipt_compatibility_authority()
            )
        except Exception as exc:
            raise OperatorError("active M22 exact authority is unavailable") from exc
        if (
            not isinstance(authority, Mapping)
            or any(authority.get(key) in (None, "") for key in required)
            or authority.get("schema")
            != "sawm/live-preflight-receipt-compatibility-successor-materialization-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M22"
            or authority.get("target_store_id") != _M22_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M22_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M22_STORE_ID).parent)
            or int(authority.get("target_generation") or 0) != _M22_GENERATION
            or int(authority.get("target_quack_port") or 0)
            != _M22_TARGET_QUACK_PORT
            or int(authority.get("target_plan_revision") or 0)
            != _M22_TARGET_PLAN_REVISION
            or int(authority.get("target_event_watermark") or 0)
            != _M22_TARGET_EVENT_WATERMARK
            or int(authority.get("task_revision_changes", -1)) != 0
            or int(authority.get("task_status_changes", -1)) != 0
            or dict(authority) != expected
        ):
            raise OperatorError(
                "active M22 live-preflight receipt compatibility successor "
                "authority is invalid"
            )
        return authority
    if m21_key in config:
        authority = config.get(m21_key)
        required = {
            "schema",
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_runtime_root",
            "target_generation",
            "target_quack_port",
            "target_plan_revision",
            "target_event_watermark",
            "target_coordination_projection_digest",
            "target_coordination_event_count",
            "prior_control_store_sha256",
            "prior_coordination_store_sha256",
            "prior_migration_receipt_sha256",
            "accepted_control_plane_repair",
        }
        try:
            expected = _materializer()._expected_m21_generation_realization_authority()
        except Exception as exc:
            raise OperatorError("active M21 exact authority is unavailable") from exc
        if (
            not isinstance(authority, Mapping)
            or any(authority.get(key) in (None, "") for key in required)
            or authority.get("schema")
            != "sawm/generation-realization-successor-materialization-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M21"
            or authority.get("target_store_id") != _M21_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M21_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M21_STORE_ID).parent)
            or int(authority.get("target_generation") or 0) != _M21_GENERATION
            or int(authority.get("target_quack_port") or 0)
            != _M21_TARGET_QUACK_PORT
            or int(authority.get("target_plan_revision") or 0)
            != _M21_TARGET_PLAN_REVISION
            or int(authority.get("target_event_watermark") or 0)
            != _M21_TARGET_EVENT_WATERMARK
            or int(authority.get("task_revision_changes", -1)) != 0
            or int(authority.get("task_status_changes", -1)) != 0
            or dict(authority) != expected
        ):
            raise OperatorError(
                "active M21 generation-realization successor authority is invalid"
            )
        return authority
    if m20_key in config:
        authority = config.get(m20_key)
        required = {
            "schema",
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_runtime_root",
            "target_generation",
            "target_quack_port",
            "target_plan_revision",
            "target_event_watermark",
            "target_coordination_projection_digest",
            "target_coordination_event_count",
            "prior_control_store_sha256",
            "prior_coordination_store_sha256",
            "prior_migration_receipt_sha256",
            "prior_validation_digest",
            "accepted_source_repair",
        }
        if (
            not isinstance(authority, Mapping)
            or any(authority.get(key) in (None, "") for key in required)
            or authority.get("schema")
            != "sawm/post-materialization-test-isolation-repair-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M20"
            or authority.get("target_store_id") != _M20_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M20_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M20_STORE_ID).parent)
            or int(authority.get("target_generation") or 0) != _M20_GENERATION
            or int(authority.get("target_quack_port") or 0)
            != _M20_TARGET_QUACK_PORT
            or int(authority.get("target_plan_revision") or 0)
            != _M20_TARGET_PLAN_REVISION
            or int(authority.get("target_event_watermark") or 0)
            != _M20_TARGET_EVENT_WATERMARK
            or int(authority.get("task_revision_changes", -1)) != 0
            or int(authority.get("task_status_changes", -1)) != 0
        ):
            raise OperatorError(
                "active M20 test-isolation successor authority is invalid"
            )
        return authority
    if m19_key in config:
        authority = config.get(m19_key)
        required = {
            "schema",
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_runtime_root",
            "target_generation",
            "target_quack_port",
            "target_plan_revision",
            "target_event_watermark",
            "target_coordination_projection_digest",
            "target_coordination_event_count",
            "prior_control_store_sha256",
            "prior_coordination_store_sha256",
            "prior_stopped_status_projection_sha256",
            "prior_migration_receipt_sha256",
            "accepted_source_repair",
        }
        if (
            not isinstance(authority, Mapping)
            or any(authority.get(key) in (None, "") for key in required)
            or authority.get("schema")
            != "sawm/live-quack-catalog-inventory-repair-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M19"
            or authority.get("target_store_id") != _M19_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M19_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M19_STORE_ID).parent)
            or int(authority.get("target_generation") or 0) != _M19_GENERATION
            or int(authority.get("target_quack_port") or 0)
            != _M19_TARGET_QUACK_PORT
            or int(authority.get("target_plan_revision") or 0)
            != _M19_TARGET_PLAN_REVISION
            or int(authority.get("target_event_watermark") or 0)
            != _M19_TARGET_EVENT_WATERMARK
            or int(authority.get("task_revision_changes", -1)) != 0
            or int(authority.get("task_status_changes", -1)) != 0
        ):
            raise OperatorError(
                "active M19 live-catalog-inventory successor authority is invalid"
            )
        return authority
    if m18_key in config:
        authority = config.get(m18_key)
        required = {
            "schema",
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_runtime_root",
            "target_generation",
            "target_quack_port",
            "target_plan_revision",
            "target_event_watermark",
            "target_projection_cid",
            "target_coordination_projection_digest",
            "target_coordination_event_count",
            "prior_control_store_sha256",
            "prior_coordination_store_sha256",
            "task_rearms",
            "failure_receipts",
            "accepted_source_repair",
        }
        if (
            not isinstance(authority, Mapping)
            or any(authority.get(key) in (None, "") for key in required)
            or authority.get("schema")
            != "sawm/portal-completion-persistence-repair-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M18"
            or authority.get("target_store_id") != _M18_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M18_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M18_STORE_ID).parent)
            or int(authority.get("target_generation") or 0) != _M18_GENERATION
            or int(authority.get("target_quack_port") or 0)
            != _M18_TARGET_QUACK_PORT
            or int(authority.get("target_plan_revision") or 0)
            != _M18_TARGET_PLAN_REVISION
            or int(authority.get("target_event_watermark") or 0)
            != _M18_TARGET_EVENT_WATERMARK
            or authority.get("target_projection_cid")
            != _M18_TARGET_PROJECTION_CID
            or set(authority.get("task_rearms", {})) != {"SAWM-007"}
            or set(authority.get("failure_receipts", {})) != {"SAWM-007"}
        ):
            raise OperatorError(
                "active M18 portal-completion successor authority is invalid"
            )
        return authority
    if m17_key in config:
        authority = config.get(m17_key)
        required = {
            "schema", "migration_revision", "target_store_id",
            "target_coordination_store_id", "target_runtime_root",
            "target_generation", "target_quack_port", "target_plan_revision",
            "target_event_watermark", "target_projection_cid",
            "prior_control_store_sha256", "prior_coordination_store_sha256",
            "prior_stopped_status_projection_sha256",
            "prior_migration_receipt_sha256",
        }
        if (
            not isinstance(authority, Mapping)
            or any(authority.get(key) in (None, "") for key in required)
            or authority.get("schema")
            != "sawm/post-commit-source-binding-successor-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M17"
            or authority.get("target_store_id") != _M17_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M17_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M17_STORE_ID).parent)
            or int(authority.get("target_generation") or 0) != _M17_GENERATION
            or int(authority.get("target_quack_port") or 0)
            != _M17_TARGET_QUACK_PORT
            or int(authority.get("target_plan_revision") or 0)
            != _M17_TARGET_PLAN_REVISION
            or int(authority.get("target_event_watermark") or 0)
            != _M17_TARGET_EVENT_WATERMARK
            or authority.get("target_projection_cid")
            != _M17_TARGET_PROJECTION_CID
        ):
            raise OperatorError(
                "active M17 source-binding successor authority is invalid"
            )
        return authority
    if m16_key in config:
        authority = config.get(m16_key)
        required = {
            "schema", "migration_revision", "target_store_id",
            "target_coordination_store_id", "target_runtime_root",
            "target_generation", "target_quack_port", "target_plan_revision",
            "target_event_watermark", "target_projection_cid",
            "target_coordination_projection_digest",
            "target_coordination_event_count", "prior_control_store_sha256",
            "prior_coordination_store_sha256", "task_rearms",
            "accepted_source_repair",
        }
        if (
            not isinstance(authority, Mapping)
            or any(authority.get(key) in (None, "") for key in required)
            or authority.get("schema")
            != "sawm/portal-accepted-source-repair-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M16"
            or authority.get("target_store_id") != _M16_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M16_COORDINATION_STORE_ID
            or authority.get("target_runtime_root")
            != str(Path(_M16_STORE_ID).parent)
            or int(authority.get("target_generation") or 0) != _M16_GENERATION
            or int(authority.get("target_quack_port") or 0)
            != _M16_TARGET_QUACK_PORT
            or int(authority.get("target_plan_revision") or 0)
            != _M16_TARGET_PLAN_REVISION
            or int(authority.get("target_event_watermark") or 0)
            != _M16_TARGET_EVENT_WATERMARK
            or authority.get("target_projection_cid")
            != _M16_TARGET_PROJECTION_CID
            or not isinstance(authority.get("task_rearms"), Mapping)
            or set(authority["task_rearms"]) != {"SAWM-003", "SAWM-004"}
            or not isinstance(authority.get("accepted_source_repair"), Mapping)
        ):
            raise OperatorError(
                "active M16 accepted-source retry successor authority is invalid"
            )
        return authority
    if m15_key in config:
        authority = config.get(m15_key)
        required = {
            "schema", "migration_revision", "target_store_id",
            "target_coordination_store_id", "target_runtime_root",
            "target_generation", "target_quack_port", "target_plan_revision",
            "target_event_watermark", "target_projection_cid",
            "prior_control_store_sha256", "prior_coordination_store_sha256",
            "prior_stopped_status_projection_present", "detached_launch_blocker",
        }
        blocker = authority.get("detached_launch_blocker") if isinstance(authority, Mapping) else None
        if (
            not isinstance(authority, Mapping)
            or any(authority.get(key) in (None, "") for key in required)
            or authority.get("schema")
            != "sawm/runtime-root-rebind-repair-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M15"
            or authority.get("target_store_id") != _M15_STORE_ID
            or authority.get("target_coordination_store_id")
            != _M15_COORDINATION_STORE_ID
            or int(authority.get("target_generation") or 0) != _M15_GENERATION
            or int(authority.get("target_quack_port") or 0) != 24_058
            or int(authority.get("target_plan_revision") or 0)
            != _M15_TARGET_PLAN_REVISION
            or int(authority.get("target_event_watermark") or 0)
            != _M15_TARGET_EVENT_WATERMARK
            or authority.get("prior_owner_marker_present") is not False
            or authority.get("prior_stopped_status_projection_present") is not True
            or not isinstance(blocker, Mapping)
            or blocker.get("failure_kind") != "historical_runtime_pid_collision"
            or blocker.get("worker_dispatched") is not False
            or blocker.get("provider_dispatched") is not False
        ):
            raise OperatorError("active M15 runtime-root successor authority is invalid")
        return authority
    if m14_key in config:
        authority = config.get(m14_key)
        required = {
            "schema", "migration_revision", "target_store_id",
            "target_coordination_store_id", "target_generation",
            "target_quack_port", "target_plan_revision",
            "target_event_watermark", "target_projection_cid",
            "prior_control_store_sha256", "prior_coordination_store_sha256",
            "prior_stopped_status_projection_present",
            "prior_stale_owner_recovery_receipt_present",
        }
        if (
            not isinstance(authority, Mapping)
            or any(authority.get(key) in (None, "") for key in required)
            or authority.get("schema") != "sawm/stale-owner-restart-repair-authorization@1"
            or authority.get("migration_revision") != "SAWM-R2-M14"
            or authority.get("target_store_id") != _M14_STORE_ID
            or authority.get("target_coordination_store_id") != _M14_COORDINATION_STORE_ID
            or int(authority.get("target_generation") or 0) != _M14_GENERATION
            or int(authority.get("target_quack_port") or 0) != 24_057
            or int(authority.get("target_plan_revision") or 0) != _M14_TARGET_PLAN_REVISION
            or int(authority.get("target_event_watermark") or 0) != _M14_TARGET_EVENT_WATERMARK
            or authority.get("prior_owner_marker_present") is not False
            or authority.get("prior_stopped_status_projection_present") is not True
            or authority.get("prior_stale_owner_recovery_receipt_present") is not True
        ):
            raise OperatorError("active M14 stale-owner restart successor authority is invalid")
        return authority
    if quack_refresh_key in config:
        quack_refresh = config.get(quack_refresh_key)
        required = {
            "schema",
            "migration_revision",
            "migration_kind",
            "target_store_id",
            "target_coordination_store_id",
            "target_generation",
            "target_quack_port",
            "target_plan_revision",
            "target_event_watermark",
            "target_projection_cid",
            "target_coordination_projection_digest",
            "target_semantic_authority_digest",
            "provider_strategy",
            "quack_refresh_repair",
            "failed_quack_start",
            "failed_quack_start_cid",
        }
        strategy = (
            quack_refresh.get("provider_strategy")
            if isinstance(quack_refresh, Mapping)
            else None
        )
        repair = (
            quack_refresh.get("quack_refresh_repair")
            if isinstance(quack_refresh, Mapping)
            else None
        )
        failure = (
            quack_refresh.get("failed_quack_start")
            if isinstance(quack_refresh, Mapping)
            else None
        )
        if (
            not isinstance(quack_refresh, Mapping)
            or any(quack_refresh.get(key) in (None, "") for key in required)
            or quack_refresh.get("schema")
            != "sawm/quack-initial-refresh-repair-authorization@1"
            or quack_refresh.get("migration_revision") != "SAWM-R2-M13"
            or not isinstance(strategy, Mapping)
            or set(strategy)
            != {
                "route_changed",
                "capability_probe_required",
                "provider_result_is_completion_authority",
            }
            or strategy.get("route_changed") is not False
            or strategy.get("capability_probe_required") is not True
            or strategy.get("provider_result_is_completion_authority") is not False
            or not isinstance(repair, Mapping)
            or set(repair)
            != {
                "operator_path",
                "refresh_probe_parameter",
                "initial_refresh_probe",
                "post_identity_refresh_probe",
                "mutation_refresh_probe",
                "unprobed_refresh_live",
                "target_port",
                "generation_changed",
                "transport_protocol_changed",
            }
            or repair.get("refresh_probe_parameter") is not True
            or repair.get("initial_refresh_probe") is not False
            or repair.get("post_identity_refresh_probe") is not True
            or repair.get("mutation_refresh_probe") is not True
            or repair.get("unprobed_refresh_live") is not False
            or not isinstance(failure, Mapping)
            or not str(failure.get("schema") or "")
            or not str(failure.get("failure_kind") or "")
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(quack_refresh.get("failed_quack_start_cid") or ""),
            )
            is None
        ):
            raise OperatorError(
                "active M13 Quack-refresh successor authority is invalid"
            )
        return quack_refresh
    if declared_output_retry_key in config:
        declared_output_retry = config.get(declared_output_retry_key)
        required = {
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_generation",
            "target_plan_revision",
            "target_event_watermark",
            "target_projection_cid",
            "target_coordination_projection_digest",
            "target_semantic_authority_digest",
            "provider_strategy",
            "task_rearm",
            "declared_output_adapter_repair",
            "live_implementation_failure",
            "live_implementation_failure_cid",
        }
        if (
            not isinstance(declared_output_retry, Mapping)
            or any(
                declared_output_retry.get(key) in (None, "")
                for key in required
            )
            or not isinstance(
                declared_output_retry.get("provider_strategy"), Mapping
            )
            or not isinstance(declared_output_retry.get("task_rearm"), Mapping)
            or not isinstance(
                declared_output_retry.get("declared_output_adapter_repair"),
                Mapping,
            )
            or not isinstance(
                declared_output_retry.get("live_implementation_failure"),
                Mapping,
            )
        ):
            raise OperatorError(
                "active M12 declared-output retry successor authority is invalid"
            )
        return declared_output_retry
    if provider_retry_key in config:
        provider_retry = config.get(provider_retry_key)
        required = {
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_generation",
            "target_plan_revision",
            "target_event_watermark",
            "target_projection_cid",
            "target_coordination_projection_digest",
            "target_semantic_authority_digest",
            "provider_strategy",
            "task_rearm",
            "runner_repair",
            "live_implementation_failure",
            "live_implementation_failure_cid",
        }
        if (
            not isinstance(provider_retry, Mapping)
            or any(provider_retry.get(key) in (None, "") for key in required)
            or not isinstance(provider_retry.get("provider_strategy"), Mapping)
            or not isinstance(provider_retry.get("task_rearm"), Mapping)
            or not isinstance(provider_retry.get("runner_repair"), Mapping)
            or not isinstance(
                provider_retry.get("live_implementation_failure"), Mapping
            )
        ):
            raise OperatorError(
                "active M11 provider-retry successor authority is invalid"
            )
        return provider_retry
    if projection_key in config:
        projection = config.get(projection_key)
        required = {
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_generation",
            "target_plan_revision",
            "target_event_watermark",
            "target_projection_cid",
            "target_coordination_projection_digest",
            "target_semantic_authority_digest",
            "provider_strategy",
        }
        if (
            not isinstance(projection, Mapping)
            or any(projection.get(key) in (None, "") for key in required)
            or not isinstance(projection.get("provider_strategy"), Mapping)
        ):
            raise OperatorError(
                "active M10 live-projection successor authority is invalid"
            )
        return projection
    if recovery_key in config:
        recovery = config.get(recovery_key)
        required = {
            "migration_revision",
            "target_store_id",
            "target_coordination_store_id",
            "target_generation",
            "target_plan_revision",
            "target_event_watermark",
            "target_projection_cid",
            "target_coordination_projection_digest",
            "provider_strategy",
            "task_rearm",
        }
        if (
            not isinstance(recovery, Mapping)
            or any(recovery.get(key) in (None, "") for key in required)
            or not isinstance(recovery.get("provider_strategy"), Mapping)
            or not isinstance(recovery.get("task_rearm"), Mapping)
        ):
            raise OperatorError("active M9 live-recovery successor authority is invalid")
        return recovery
    if successor_key in config:
        successor = config.get(successor_key)
        if not isinstance(successor, Mapping):
            raise OperatorError("active source-only successor authority is invalid")
        return successor
    historical = config.get(historical_key)
    if not isinstance(historical, Mapping):
        raise OperatorError("source-only successor authority is unavailable")
    return historical


def _successor_materialization_configured(config: Mapping[str, Any]) -> bool:
    """Return whether any append-only successor key is present.

    Key presence is intentional: malformed newer controls must reach the
    fail-closed selector instead of falling through to historical authority.
    """

    return any(
        key in config
        for key in (
            "json_emission_normalization_successor_materialization",
            "live_preflight_contract_successor_materialization",
            "live_preflight_plan_anchor_successor_materialization",
            "detached_coordinator_pid_recovery_successor_materialization",
            "stopped_owner_restart_source_seal_successor_materialization",
            "committed_evidence_verification_successor_materialization",
            "live_claim_admission_recovery_successor_materialization",
            "dead_owner_parallel_resume_successor_materialization",
            "automatic_stall_recovery_successor_materialization",
            "native_duckdb_preload_successor_materialization",
            "multi_lane_sidecar_reopen_successor_materialization",
            "multi_lane_successor_materialization",
            "live_preflight_receipt_compatibility_successor_materialization",
            "generation_realization_successor_materialization",
            "test_isolation_successor_materialization",
            "live_catalog_inventory_successor_materialization",
            "portal_completion_persistence_successor_materialization",
            "source_binding_successor_materialization",
            "accepted_source_retry_successor_materialization",
            "runtime_root_rebind_successor_materialization",
            "stale_owner_restart_successor_materialization",
            "quack_refresh_successor_materialization",
            "declared_output_retry_successor_materialization",
            "live_provider_retry_successor_materialization",
            "live_projection_successor_materialization",
            "live_recovery_successor_materialization",
            "source_repair_successor_materialization",
            "source_repair_materialization",
        )
    )


def _validate_m13_runtime_binding(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
) -> None:
    """Bind M13 to the exact stopped-M12 failure and lifecycle-only repair."""

    if "quack_refresh_successor_materialization" not in config:
        return
    program = config.get("database_program")
    owner = config.get("quack_owner")
    provider = config.get("provider")
    strategy = authority.get("provider_strategy")
    repair = authority.get("quack_refresh_repair")
    failure = authority.get("failed_quack_start")
    if not all(
        isinstance(value, Mapping)
        for value in (program, owner, provider, strategy, repair, failure)
    ):
        raise OperatorError("M13 runtime authority binding is incomplete")
    try:
        expected_authority = (
            materializer._expected_m13_quack_refresh_authority()
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise OperatorError(
            "M13 exact Quack-refresh authority is unavailable"
        ) from exc
    expected_provider = {
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_provider_id": "codex",
        "fallback_reasoning_effort": "medium",
        "fallback_trigger": "primary_quota_exhausted",
        "max_concurrency": 1,
        "primary_executable": "/home/barberb/.local/bin/grok",
        "primary_model_id": "grok-4.6",
        "primary_provider_id": "grok_cli",
        "probe_before_live_launch": True,
        "provider_results_are_completion_authority": False,
        "secrets_from_environment_only": True,
        "secrets_in_argv_prompts_logs_or_receipts": False,
    }
    if (
        dict(authority) != expected_authority
        or authority.get("schema")
        != "sawm/quack-initial-refresh-repair-authorization@1"
        or authority.get("migration_revision") != "SAWM-R2-M13"
        or authority.get("migration_kind")
        != "quack_initial_refresh_rebind_repair"
        or authority.get("target_store_id") != _M13_STORE_ID
        or authority.get("target_coordination_store_id")
        != _M13_COORDINATION_STORE_ID
        or int(authority.get("target_generation") or 0) != _M13_GENERATION
        or int(authority.get("target_quack_port") or 0) != 24_056
        or int(authority.get("target_plan_revision") or 0)
        != _M13_TARGET_PLAN_REVISION
        or int(authority.get("target_event_watermark") or 0)
        != _M13_TARGET_EVENT_WATERMARK
        or authority.get("target_coordination_projection_digest")
        != _M12_TARGET_COORDINATION_PROJECTION_DIGEST
        or int(authority.get("target_coordination_event_count") or 0)
        != _M12_TARGET_COORDINATION_EVENT_COUNT
        or authority.get("target_semantic_authority_digest")
        != _M12_TARGET_SEMANTIC_AUTHORITY_DIGEST
        or authority.get("prior_store_id") != _M12_STORE_ID
        or authority.get("prior_control_store_sha256")
        != _M13_PRIOR_CONTROL_STORE_SHA256
        or int(authority.get("prior_control_store_size") or 0)
        != _M13_PRIOR_CONTROL_STORE_SIZE
        or authority.get("prior_read_replica_store_sha256")
        != _M13_PRIOR_READ_REPLICA_STORE_SHA256
        or int(authority.get("prior_read_replica_store_size") or 0)
        != _M13_PRIOR_READ_REPLICA_STORE_SIZE
        or authority.get("prior_coordination_store_sha256")
        != _M13_PRIOR_COORDINATION_STORE_SHA256
        or int(authority.get("prior_coordination_store_size") or 0)
        != _M13_PRIOR_COORDINATION_STORE_SIZE
        or authority.get("prior_control_wal_present") is not False
        or authority.get("prior_coordination_wal_present") is not False
        or authority.get("prior_owner_status_present") is not False
        or authority.get("prior_owner_directory_empty") is not True
        or authority.get("control_base_copied") is not True
        or authority.get("coordination_base_copied") is not True
        or authority.get("prior_read_replica_copied") is not False
        or authority.get("prior_owner_status_copied") is not False
        or authority.get("prior_failed_start_artifacts_preserved") is not True
        or authority.get("coordination_semantic_changes") != 0
        or authority.get("plan_revision_changes") != 1
        or authority.get("evidence_node_changes") != 1
        or authority.get("task_revision_changes") != 0
        or authority.get("task_status_changes") != 0
        or any(
            authority.get(field) != 0
            for field in (
                "accepted_definition_changes",
                "accepted_completion_changes",
                "effect_claim_changes",
                "implementation_commit_changes",
                "merge_attempt_changes",
            )
        )
        or authority.get("worker_self_approval") is not False
        or program.get("store_id") != _M13_STORE_ID
        or int(program.get("store_generation") or 0) != _M13_GENERATION
        or program.get("quack_endpoint") != "quack:127.0.0.1:24056"
        or owner.get("database_path") != _M13_STORE_ID
        or owner.get("store_id") != _M13_STORE_ID
        or int(owner.get("port") or 0) != 24_056
        or strategy.get("route_changed") is not False
        or strategy.get("capability_probe_required") is not True
        or strategy.get("provider_result_is_completion_authority") is not False
        or dict(provider) != expected_provider
    ):
        raise OperatorError(
            "M13 runtime store, lifecycle repair, or provider binding differs"
        )


def _validate_m12_runtime_binding(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Bind M12 to the exact stopped-M11 bases and adapter-only rearm."""

    if "declared_output_retry_successor_materialization" not in config:
        return
    program = config.get("database_program")
    owner = config.get("quack_owner")
    provider = config.get("provider")
    strategy = authority.get("provider_strategy")
    rearm = authority.get("task_rearm")
    adapter = authority.get("declared_output_adapter_repair")
    failure = authority.get("live_implementation_failure")
    if not all(
        isinstance(value, Mapping)
        for value in (program, owner, provider, strategy, rearm, adapter, failure)
    ):
        raise OperatorError("M12 runtime authority binding is incomplete")
    authority_cid = "sha256:" + hashlib.sha256(
        json.dumps(
            dict(authority),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    failure_cid = "sha256:" + hashlib.sha256(
        json.dumps(
            dict(failure),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    prior_route = str(strategy.get("prior_route") or "")
    expected_provider = {
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_provider_id": "codex",
        "fallback_reasoning_effort": "medium",
        "fallback_trigger": "primary_quota_exhausted",
        "max_concurrency": 1,
        "primary_executable": "/home/barberb/.local/bin/grok",
        "primary_model_id": "grok-4.6",
        "primary_provider_id": "grok_cli",
        "probe_before_live_launch": True,
        "provider_results_are_completion_authority": False,
        "secrets_from_environment_only": True,
        "secrets_in_argv_prompts_logs_or_receipts": False,
    }
    if (
        authority_cid != _M12_DECLARED_OUTPUT_RETRY_AUTHORITY_CID
        or authority.get("schema")
        != "sawm/portal-declared-output-repair-authorization@1"
        or authority.get("migration_revision") != "SAWM-R2-M12"
        or authority.get("migration_kind")
        != "database_portal_declared_path_projection_repair"
        or authority.get("target_store_id") != _M12_STORE_ID
        or authority.get("target_coordination_store_id")
        != _M12_COORDINATION_STORE_ID
        or int(authority.get("target_generation") or 0) != _M12_GENERATION
        or int(authority.get("target_quack_port") or 0) != 45_256
        or int(authority.get("target_plan_revision") or 0)
        != _M12_TARGET_PLAN_REVISION
        or int(authority.get("target_event_watermark") or 0)
        != _M12_TARGET_EVENT_WATERMARK
        or authority.get("target_projection_cid")
        != _M12_TARGET_PROJECTION_CID
        or authority.get("target_coordination_projection_digest")
        != _M12_TARGET_COORDINATION_PROJECTION_DIGEST
        or int(authority.get("target_coordination_event_count") or 0)
        != _M12_TARGET_COORDINATION_EVENT_COUNT
        or authority.get("target_semantic_authority_digest")
        != _M12_TARGET_SEMANTIC_AUTHORITY_DIGEST
        or authority.get("prior_control_store_sha256")
        != _M12_PRIOR_CONTROL_STORE_SHA256
        or int(authority.get("prior_control_store_size") or 0)
        != _M12_PRIOR_CONTROL_STORE_SIZE
        or authority.get("prior_coordination_store_sha256")
        != _M12_PRIOR_COORDINATION_STORE_SHA256
        or int(authority.get("prior_coordination_store_size") or 0)
        != _M12_PRIOR_COORDINATION_STORE_SIZE
        or authority.get("prior_control_wal_present") is not False
        or authority.get("prior_coordination_wal_present") is not False
        or authority.get("prior_wals_absent") is not True
        or authority.get("control_base_copied") is not True
        or authority.get("coordination_base_copied") is not True
        or authority.get("prior_control_wal_copied") is not False
        or authority.get("prior_coordination_wal_copied") is not False
        or authority.get("execution_sidecar_copied") is not False
        or authority.get("read_replica_sidecar_copied") is not False
        or authority.get("coordination_semantic_changes") != 1
        or authority.get("task_revision_changes") != 1
        or authority.get("task_status_changes") != 1
        or any(
            authority.get(field) != 0
            for field in (
                "accepted_definition_changes",
                "accepted_completion_changes",
                "settlement_provider_invocation_count",
                "effect_claim_changes",
                "implementation_commit_changes",
                "merge_attempt_changes",
            )
        )
        or authority.get("implementation_provider_invocations_observed") != 2
        or authority.get("implementation_provider_model_calls_observed") != 92
        or authority.get("implementation_provider_tokens_observed") != 11_202_083
        or authority.get("implementation_provider_cost_usd_observed")
        != "1.23726850"
        or authority.get("provider_execution_accounting_mismatch") is not True
        or authority.get("worker_self_approval") is not False
        or failure_cid != authority.get("live_implementation_failure_cid")
        or failure.get("schema") != "sawm/live-control-implementation-failure@3"
        or failure.get("failure_kind")
        != "declared_output_paths_projected_as_effect_identities"
        or failure.get("terminal_status") != "blocked"
        or int(failure.get("terminal_revision") or 0) != 15
        or failure.get("provider_executions_observed") != 2
        or failure.get("provider_model_calls_observed") != 92
        or failure.get("provider_tokens_observed") != 11_202_083
        or failure.get("provider_cost_usd_observed") != "1.23726850"
        or failure.get("provider_accounting_distinct_from_settlement") is not True
        or failure.get("settlement_provider_invocation_count") != 0
        or failure.get("failure_receipt") != dict(_M12_LIVE_FAILURE_RECEIPT)
        or adapter.get("adapter_path")
        != (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "database_portal_bridge.py"
        )
        or adapter.get("focused_test_path")
        != "test/api/test_agent_supervisor_database_portal_bridge.py"
        or adapter.get("failure_root")
        != "effect_identity_selected_instead_of_declared_path"
        or adapter.get("canonical_field") != "effect.declared_path"
        or adapter.get("legacy_fallback_field") != "path"
        or adapter.get("provider_envelope_changed") is not False
        or adapter.get("proposal_gate_changed") is not False
        or rearm.get("settlement_id")
        != _M12_LIVE_FAILURE_RECEIPT["settlement_id"]
        or rearm.get("from_status") != "blocked"
        or int(rearm.get("from_revision") or 0) != 15
        or rearm.get("to_status") != "retrying"
        or int(rearm.get("to_revision") or 0) != 16
        or rearm.get("automatic_retry_admitted") is not False
        or rearm.get("coordination_rearm_required") is not True
        or rearm.get("historical_settlement_preserved") is not True
        or program.get("store_id") != _M12_STORE_ID
        or int(program.get("store_generation") or 0) != _M12_GENERATION
        or program.get("quack_endpoint") != "quack:127.0.0.1:45256"
        or owner.get("database_path") != _M12_STORE_ID
        or owner.get("store_id") != _M12_STORE_ID
        or int(owner.get("port") or 0) != 45_256
        or not prior_route
        or strategy.get("target_route") != prior_route
        or strategy.get("route_changed") is not False
        or strategy.get("capability_probe_required") is not True
        or strategy.get("provider_result_is_completion_authority") is not False
        or dict(provider) != expected_provider
    ):
        raise OperatorError(
            "M12 runtime store, rearm, adapter repair, or provider binding differs"
        )


def _validate_m11_runtime_binding(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Bind M11 to its exact pair, rearm, runner repair, and prior route."""

    if "live_provider_retry_successor_materialization" not in config:
        return
    program = config.get("database_program")
    owner = config.get("quack_owner")
    provider = config.get("provider")
    strategy = authority.get("provider_strategy")
    rearm = authority.get("task_rearm")
    runner_repair = authority.get("runner_repair")
    failure = authority.get("live_implementation_failure")
    if not all(
        isinstance(value, Mapping)
        for value in (
            program,
            owner,
            provider,
            strategy,
            rearm,
            runner_repair,
            failure,
        )
    ):
        raise OperatorError("M11 runtime authority binding is incomplete")
    authority_cid = "sha256:" + hashlib.sha256(
        json.dumps(
            dict(authority),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    prior_route = str(strategy.get("prior_route") or "")
    if (
        authority_cid != _M11_LIVE_PROVIDER_RETRY_AUTHORITY_CID
        or "prior_coordination_replayed_store_sha256" in authority
        or "prior_coordination_replayed_store_size" in authority
        or authority.get("prior_coordination_store_sha256")
        != _M11_PRIOR_COORDINATION_STORE_SHA256
        or int(authority.get("prior_coordination_store_size") or 0)
        != _M11_PRIOR_COORDINATION_STORE_SIZE
        or authority.get("prior_coordination_wal_sha256")
        != _M11_PRIOR_COORDINATION_WAL_SHA256
        or int(authority.get("prior_coordination_wal_size") or 0)
        != _M11_PRIOR_COORDINATION_WAL_SIZE
        or authority.get("schema")
        != "sawm/provider-launch-repair-authorization@1"
        or authority.get("migration_revision") != "SAWM-R2-M11"
        or authority.get("target_store_id") != _M11_STORE_ID
        or authority.get("target_coordination_store_id")
        != _M11_COORDINATION_STORE_ID
        or int(authority.get("target_generation") or 0) != _M11_GENERATION
        or int(authority.get("target_plan_revision") or 0)
        != _M11_TARGET_PLAN_REVISION
        or int(authority.get("target_event_watermark") or 0)
        != _M11_TARGET_EVENT_WATERMARK
        or authority.get("target_projection_cid")
        != _M11_TARGET_PROJECTION_CID
        or authority.get("target_coordination_projection_digest")
        != _M11_TARGET_COORDINATION_PROJECTION_DIGEST
        or int(authority.get("target_coordination_event_count") or 0)
        != _M11_TARGET_COORDINATION_EVENT_COUNT
        or authority.get("target_semantic_authority_digest")
        != _M11_TARGET_SEMANTIC_AUTHORITY_DIGEST
        or authority.get("prior_semantic_authority_digest")
        != _M10_FAILED_SEMANTIC_AUTHORITY_DIGEST
        or authority.get("live_implementation_failure_cid")
        != _M10_LIVE_FAILURE_CID
        or program.get("store_id") != _M11_STORE_ID
        or int(program.get("store_generation") or 0) != _M11_GENERATION
        or program.get("quack_endpoint") != "quack:127.0.0.1:45255"
        or owner.get("database_path") != _M11_STORE_ID
        or owner.get("store_id") != _M11_STORE_ID
        or int(owner.get("port") or 0) != 45255
        or not prior_route
        or strategy.get("target_route") != prior_route
        or strategy.get("route_changed") is not False
        or strategy.get("capability_probe_required") is not True
        or strategy.get("provider_result_is_completion_authority") is not False
        or not str(provider.get("primary_provider_id") or "")
        or not str(provider.get("primary_model_id") or "")
        or not str(provider.get("fallback_provider_id") or "")
        or not str(provider.get("fallback_model_id") or "")
        or provider.get("fallback_trigger") != "primary_quota_exhausted"
        or provider.get("provider_results_are_completion_authority") is not False
        or rearm.get("settlement_id")
        != _M10_FAILURE_RECEIPT["settlement_id"]
        or rearm.get("from_status") != "blocked"
        or int(rearm.get("from_revision") or 0) != 12
        or rearm.get("to_status") != "retrying"
        or int(rearm.get("to_revision") or 0) != 13
        or rearm.get("automatic_retry_admitted") is not False
        or rearm.get("coordination_rearm_required") is not True
        or rearm.get("historical_settlement_preserved") is not True
        or rearm.get("provider_invocation_count") != 0
        or authority.get("coordination_base_and_wal_copied") is not True
        or authority.get("coordination_wal_replayed_on_copy") is not True
        or authority.get("prior_coordination_store_mutated") is not False
        or authority.get("coordination_semantic_changes") != 1
        or authority.get("implementation_provider_invocations_observed") != 1
        or authority.get("settlement_provider_invocation_count") != 0
        or runner_repair.get("provider_route_changed") is not False
        or runner_repair.get("failure_root")
        != "attached_start_executed_in_typed_create_phase"
        or runner_repair.get("expected_lifecycle")
        != "typed_route_owns_create_identity_verification_and_attached_start"
        or runner_repair.get("provider_execution_observed") is not True
        or runner_repair.get("provider_execution_accounting_mismatch") is not True
        or runner_repair.get("settlement_provider_invocation_count") != 0
        or runner_repair.get("effect_claim_count") != 0
        or runner_repair.get("runner_path")
        != "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py"
        or runner_repair.get("focused_test_path")
        != "test/api/test_agent_supervisor_grok_quota_terra_gate.py"
        or failure.get("settlement_id")
        != _M10_FAILURE_RECEIPT["settlement_id"]
        or failure.get("schema")
        != "sawm/live-control-implementation-failure@2"
        or failure.get("failure_kind")
        != "attached_provider_execution_misclassified_as_container_create_timeout"
        or failure.get("terminal_status") != "blocked"
        or int(failure.get("terminal_revision") or 0) != 12
        or failure.get("provider_execution_observed") is not True
        or failure.get("provider_execution_accounting_mismatch") is not True
        or failure.get("settlement_provider_invocation_count") != 0
        or failure.get("failure_receipt") != dict(_M10_FAILURE_RECEIPT)
    ):
        raise OperatorError(
            "M11 runtime store, rearm, runner repair, or provider binding differs"
        )


def _validate_m10_runtime_binding(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Bind M10 to its exact pair, generation, and unchanged provider route."""

    if "live_projection_successor_materialization" not in config:
        return
    program = config.get("database_program")
    owner = config.get("quack_owner")
    provider = config.get("provider")
    strategy = authority.get("provider_strategy")
    if not all(
        isinstance(value, Mapping)
        for value in (program, owner, provider, strategy)
    ):
        raise OperatorError("M10 runtime authority binding is incomplete")
    prior_route = str(strategy.get("prior_route") or "")
    if (
        authority.get("migration_revision") != "SAWM-R2-M10"
        or authority.get("target_store_id") != _M10_STORE_ID
        or authority.get("target_coordination_store_id")
        != _M10_COORDINATION_STORE_ID
        or int(authority.get("target_generation") or 0) != _M10_GENERATION
        or authority.get("target_semantic_authority_digest")
        != _M9_HEAD_SEMANTIC_AUTHORITY_DIGEST
        or program.get("store_id") != _M10_STORE_ID
        or int(program.get("store_generation") or 0) != _M10_GENERATION
        or owner.get("database_path") != _M10_STORE_ID
        or owner.get("store_id") != _M10_STORE_ID
        or not prior_route
        or strategy.get("target_route") != prior_route
        or strategy.get("route_changed") is not False
        or strategy.get("capability_probe_required") is not True
        or strategy.get("provider_result_is_completion_authority") is not False
        or not str(provider.get("primary_provider_id") or "")
        or not str(provider.get("primary_model_id") or "")
        or not str(provider.get("fallback_provider_id") or "")
        or not str(provider.get("fallback_model_id") or "")
        or provider.get("fallback_trigger") != "primary_quota_exhausted"
        or provider.get("provider_results_are_completion_authority") is not False
    ):
        raise OperatorError(
            "M10 runtime store, generation, digest, or provider binding differs"
        )


def _validate_m9_runtime_binding(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Bind M9 to its closed store pair, generation, and unchanged route."""

    if "live_recovery_successor_materialization" not in config:
        return
    program = config.get("database_program")
    owner = config.get("quack_owner")
    provider = config.get("provider")
    strategy = authority.get("provider_strategy")
    rearm = authority.get("task_rearm")
    if not all(
        isinstance(value, Mapping)
        for value in (program, owner, provider, strategy, rearm)
    ):
        raise OperatorError("M9 runtime authority binding is incomplete")
    prior_route = str(strategy.get("prior_route") or "")
    if (
        authority.get("migration_revision") != "SAWM-R2-M9"
        or authority.get("target_store_id") != _M9_STORE_ID
        or authority.get("target_coordination_store_id")
        != _M9_COORDINATION_STORE_ID
        or int(authority.get("target_generation") or 0) != _M9_GENERATION
        or program.get("store_id") != _M9_STORE_ID
        or int(program.get("store_generation") or 0) != _M9_GENERATION
        or owner.get("database_path") != _M9_STORE_ID
        or owner.get("store_id") != _M9_STORE_ID
        or not prior_route
        or strategy.get("target_route") != prior_route
        or strategy.get("route_changed") is not False
        or strategy.get("capability_probe_required") is not True
        or strategy.get("provider_result_is_completion_authority") is not False
        or not str(provider.get("primary_provider_id") or "")
        or not str(provider.get("primary_model_id") or "")
        or not str(provider.get("fallback_provider_id") or "")
        or not str(provider.get("fallback_model_id") or "")
        or provider.get("fallback_trigger") != "primary_quota_exhausted"
        or provider.get("provider_results_are_completion_authority") is not False
        or rearm.get("coordination_rearm_required") is not True
        or int(rearm.get("target_coordination_event_count") or 0) != 36
        or rearm.get("historical_settlement_preserved") is not True
    ):
        raise OperatorError("M9 runtime store, generation, or provider binding differs")


def _require_m9_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require the materializer's final marker for the verified M9 pair.

    Offline startup supplies the complete ``check_materialized`` report.  Live
    preflight rechecks its immutable final marker without opening the Quack-
    owned control database directly.
    """

    if "live_recovery_successor_materialization" not in config:
        return MappingProxyType({})
    _validate_m9_runtime_binding(config, authority)
    receipt_path = REPO_ROOT / Path(_M9_STORE_ID).parent / "migration-receipt.json"
    try:
        observed, _receipt_sha256 = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M9 final pair commit marker",
        )
    except Exception as exc:
        raise OperatorError("M9 materialized final pair marker is unavailable") from exc
    if not isinstance(observed, Mapping):
        raise OperatorError("M9 materialized final pair marker is invalid")
    coordination_path = (REPO_ROOT / _M9_COORDINATION_STORE_ID).resolve()
    try:
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination_path,
                root=REPO_ROOT,
                noun="materialized M9 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M9 materialized coordination store is unavailable") from exc
    unhashed = dict(observed)
    claimed_cid = str(unhashed.pop("receipt_cid", ""))
    rearm = authority["task_rearm"]
    if (
        claimed_cid != materializer._identity(unhashed)
        or observed.get("schema") != "sawm/non-authoritative-migration-receipt@7"
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != authority["migration_revision"]
        or observed.get("database_path") != _M9_STORE_ID
        or observed.get("coordination_path") != _M9_COORDINATION_STORE_ID
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or observed.get("migration_projection_cid")
        != authority["target_projection_cid"]
        or observed.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or int(observed.get("migration_event_watermark") or 0)
        != int(authority["target_event_watermark"])
        or int(observed.get("coordination_event_count") or 0)
        != int(rearm["target_coordination_event_count"])
        or observed.get("coordination_sidecar_copied_before_rearm") is not True
        or observed.get("failed_attempt_history_preserved") is not True
        or observed.get("failed_completion_barrier_rearmed") is not True
        or observed.get("execution_sidecar_copied") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M9 materialized final pair marker differs")
    if checked is not None:
        expected_control = str((REPO_ROOT / _M9_STORE_ID).resolve())
        expected_coordination = str(coordination_path)
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != expected_control
            or checked.get("coordination_path") != expected_coordination
            or checked.get("projection_cid") != authority["target_projection_cid"]
            or checked.get("coordination_projection_digest")
            != authority["target_coordination_projection_digest"]
            or int(checked.get("event_watermark") or 0)
            != int(authority["target_event_watermark"])
            or checked.get("receipt") != observed
        ):
            raise OperatorError("M9 materializer check differs from its final pair marker")
    return MappingProxyType(dict(observed))


def _require_m10_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M10's receipt-last marker for the unchanged M9-head pair."""

    if "live_projection_successor_materialization" not in config:
        return MappingProxyType({})
    _validate_m10_runtime_binding(config, authority)
    receipt_path = (
        REPO_ROOT / Path(_M10_STORE_ID).parent / "migration-receipt.json"
    )
    try:
        observed, _receipt_sha256 = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M10 final pair commit marker",
        )
    except Exception as exc:
        raise OperatorError(
            "M10 materialized final pair marker is unavailable"
        ) from exc
    if not isinstance(observed, Mapping):
        raise OperatorError("M10 materialized final pair marker is invalid")
    coordination_path = (REPO_ROOT / _M10_COORDINATION_STORE_ID).resolve()
    try:
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination_path,
                root=REPO_ROOT,
                noun="materialized M10 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError(
            "M10 materialized coordination store is unavailable"
        ) from exc
    unhashed = dict(observed)
    claimed_cid = str(unhashed.pop("receipt_cid", ""))
    if (
        claimed_cid != materializer._identity(unhashed)
        or observed.get("schema") != "sawm/non-authoritative-migration-receipt@8"
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != authority["migration_revision"]
        or observed.get("database_path") != _M10_STORE_ID
        or observed.get("coordination_path") != _M10_COORDINATION_STORE_ID
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or observed.get("migration_projection_cid")
        != authority["target_projection_cid"]
        or observed.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or observed.get("semantic_authority_digest")
        != _M9_HEAD_SEMANTIC_AUTHORITY_DIGEST
        or observed.get("live_projection_successor_materialization_cid")
        != materializer._identity(dict(authority))
        or int(observed.get("migration_event_watermark") or 0)
        != int(authority["target_event_watermark"])
        or int(observed.get("coordination_event_count") or 0) != 36
        or observed.get("coordination_sidecar_copied_unchanged") is not True
        or observed.get("coordination_semantic_changes") != 0
        or observed.get("task_revision_changes") != 0
        or observed.get("task_status_changes") != 0
        or observed.get("execution_sidecar_copied") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M10 materialized final pair marker differs")
    if checked is not None:
        expected_control = str((REPO_ROOT / _M10_STORE_ID).resolve())
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != expected_control
            or checked.get("coordination_path") != str(coordination_path)
            or checked.get("projection_cid") != authority["target_projection_cid"]
            or checked.get("coordination_projection_digest")
            != authority["target_coordination_projection_digest"]
            or int(checked.get("event_watermark") or 0)
            != int(authority["target_event_watermark"])
            or checked.get("receipt") != observed
        ):
            raise OperatorError(
                "M10 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _require_m13_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M13's receipt-last marker for its lifecycle-only successor."""

    if "quack_refresh_successor_materialization" not in config:
        return MappingProxyType({})
    _validate_m13_runtime_binding(config, authority, materializer)
    receipt_path = (
        REPO_ROOT / Path(_M13_STORE_ID).parent / "migration-receipt.json"
    )
    try:
        observed, _receipt_sha256 = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M13 final pair commit marker",
        )
    except Exception as exc:
        raise OperatorError(
            "M13 materialized final pair marker is unavailable"
        ) from exc
    if not isinstance(observed, Mapping):
        raise OperatorError("M13 materialized final pair marker is invalid")
    coordination_path = (REPO_ROOT / _M13_COORDINATION_STORE_ID).resolve()
    try:
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination_path,
                root=REPO_ROOT,
                noun="materialized M13 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError(
            "M13 materialized coordination store is unavailable"
        ) from exc
    population = materializer.build_population(REPO_ROOT)
    dependency = materializer._validator_report(
        REPO_ROOT,
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
    )
    board = materializer._validator_report(
        REPO_ROOT,
        "scripts/validate_semantic_addressed_world_model_board.py",
    )
    validation_digest = materializer._identity(
        {
            "dependency": dependency,
            "board": board,
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    migration_body = materializer._m13_migration_body(
        population,
        config,
        validation_digest,
    )
    migration_digest = materializer._identity(migration_body)
    migration_evidence_id = content_identity(
        {
            "task_cid": population["migration_inventory"]["prior_task_cids"][
                "SAWM-000"
            ],
            "evidence_kind": "operator_control_plane_source_migration",
            "digest": migration_digest,
            "body": migration_body,
        }
    )
    unhashed = dict(observed)
    claimed_cid = str(unhashed.pop("receipt_cid", ""))
    authority_cid = materializer._identity(dict(authority))
    if (
        set(observed) != _M13_RECEIPT_KEYS
        or claimed_cid != materializer._identity(unhashed)
        or observed.get("schema")
        != "sawm/non-authoritative-migration-receipt@11"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != authority["migration_revision"]
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("plan_projection_cid") != authority["prior_projection_cid"]
        or observed.get("database_path") != _M13_STORE_ID
        or observed.get("coordination_path") != _M13_COORDINATION_STORE_ID
        or observed.get("prior_database_path") != authority["prior_store_id"]
        or observed.get("prior_coordination_path")
        != authority["prior_coordination_store_id"]
        or observed.get("prior_publication_control_store_sha256")
        != authority["prior_publication_control_store_sha256"]
        or observed.get("prior_control_store_sha256")
        != _M13_PRIOR_CONTROL_STORE_SHA256
        or observed.get("prior_coordination_store_sha256")
        != _M13_PRIOR_COORDINATION_STORE_SHA256
        or observed.get("prior_control_wal_present") is not False
        or observed.get("prior_coordination_wal_present") is not False
        or observed.get("prior_owner_status_present") is not False
        or observed.get("prior_read_replica_store_sha256")
        != authority["prior_read_replica_store_sha256"]
        or observed.get("prior_read_replica_is_authority") is not False
        or observed.get("prior_read_replica_copied") is not False
        or observed.get("prior_event_prefix_sha256")
        != authority["prior_event_prefix_sha256"]
        or observed.get("prior_source_binding_cid")
        != authority["prior_source_binding_cid"]
        or observed.get("prior_materialization_receipt_cid")
        != authority["prior_materialization_receipt_cid"]
        or observed.get(
            "prior_materialization_receipt_precedes_failed_start_checkpoint"
        )
        is not True
        or observed.get("declared_output_retry_successor_materialization_cid")
        != _M12_DECLARED_OUTPUT_RETRY_AUTHORITY_CID
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("validation_digest")
        == observed.get("prior_validation_digest")
        or observed.get("migration_digest") != migration_digest
        or observed.get("migration_evidence_id") != migration_evidence_id
        or not str(observed.get("plan_migration_event_id") or "")
        or not str(observed.get("migration_evidence_event_id") or "")
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(observed.get("control_store_sha256") or ""),
        )
        is None
        or int(observed.get("control_store_size") or 0) <= 0
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or observed.get("migration_projection_cid")
        != authority["target_projection_cid"]
        or observed.get("projection_cid") != authority["target_projection_cid"]
        or int(observed.get("migration_event_watermark") or 0)
        != _M13_TARGET_EVENT_WATERMARK
        or int(observed.get("target_event_watermark") or 0)
        != _M13_TARGET_EVENT_WATERMARK
        or int(observed.get("target_generation") or 0) != _M13_GENERATION
        or int(observed.get("target_quack_port") or 0) != 24_056
        or observed.get("coordination_projection_digest")
        != _M12_TARGET_COORDINATION_PROJECTION_DIGEST
        or int(observed.get("coordination_event_count") or 0)
        != _M12_TARGET_COORDINATION_EVENT_COUNT
        or observed.get("semantic_authority_digest")
        != _M12_TARGET_SEMANTIC_AUTHORITY_DIGEST
        or observed.get("frozen_base_authority_digest")
        != authority["target_frozen_base_authority_digest"]
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(observed.get("append_surface_digest") or ""),
        )
        is None
        or observed.get("quack_refresh_successor_materialization_cid")
        != authority_cid
        or observed.get("failed_quack_start_cid")
        != authority["failed_quack_start_cid"]
        or observed.get("control_and_coordination_bases_copied") is not True
        or observed.get("prior_wals_absent") is not True
        or observed.get("prior_control_store_mutated") is not False
        or observed.get("prior_coordination_store_mutated") is not False
        or observed.get("prior_failed_start_artifacts_preserved") is not True
        or observed.get("coordination_semantic_changes") != 0
        or observed.get("plan_revision_changes") != 1
        or observed.get("evidence_node_changes") != 1
        or observed.get("task_revision_changes") != 0
        or observed.get("task_status_changes") != 0
        or observed.get("accepted_definition_changes") != 0
        or observed.get("accepted_completion_changes") != 0
        or observed.get("implementation_provider_invocations") != 0
        or observed.get("effect_claim_changes") != 0
        or observed.get("implementation_commit_changes") != 0
        or observed.get("merge_attempt_changes") != 0
        or observed.get("execution_sidecar_copied") is not False
        or observed.get("read_replica_sidecar_copied") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M13 materialized final pair marker differs")
    if checked is not None:
        expected_control = str((REPO_ROOT / _M13_STORE_ID).resolve())
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != expected_control
            or checked.get("coordination_path") != str(coordination_path)
            or checked.get("projection_cid") != authority["target_projection_cid"]
            or checked.get("coordination_projection_digest")
            != authority["target_coordination_projection_digest"]
            or int(checked.get("event_watermark") or 0)
            != int(authority["target_event_watermark"])
            or checked.get("receipt") != observed
        ):
            raise OperatorError(
                "M13 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _require_m12_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M12's receipt-last marker for its repaired/rearmed pair."""

    if "declared_output_retry_successor_materialization" not in config:
        return MappingProxyType({})
    _validate_m12_runtime_binding(config, authority)
    receipt_path = (
        REPO_ROOT / Path(_M12_STORE_ID).parent / "migration-receipt.json"
    )
    try:
        observed, _receipt_sha256 = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M12 final pair commit marker",
        )
    except Exception as exc:
        raise OperatorError(
            "M12 materialized final pair marker is unavailable"
        ) from exc
    if not isinstance(observed, Mapping):
        raise OperatorError("M12 materialized final pair marker is invalid")
    coordination_path = (REPO_ROOT / _M12_COORDINATION_STORE_ID).resolve()
    try:
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination_path,
                root=REPO_ROOT,
                noun="materialized M12 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError(
            "M12 materialized coordination store is unavailable"
        ) from exc
    population = materializer.build_population(REPO_ROOT)
    dependency = materializer._validator_report(
        REPO_ROOT,
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
    )
    board = materializer._validator_report(
        REPO_ROOT,
        "scripts/validate_semantic_addressed_world_model_board.py",
    )
    validation_digest = materializer._identity(
        {
            "dependency": dependency,
            "board": board,
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    migration_body = materializer._m12_migration_body(
        population,
        config,
        validation_digest,
    )
    migration_digest = materializer._identity(migration_body)
    migration_evidence_id = content_identity(
        {
            "task_cid": population["migration_inventory"]["prior_task_cids"][
                "SAWM-000"
            ],
            "evidence_kind": "operator_control_plane_source_migration",
            "digest": migration_digest,
            "body": migration_body,
        }
    )
    task_rearm_receipt_cid = content_identity(dict(_M12_REARM_RECEIPT))
    expected_keys = {
        "schema",
        "authoritative",
        "control_database_is_authority",
        "coordination_database_is_authority",
        "receipt_is_final_pair_commit_marker",
        "migration_revision",
        "program_definition_cid",
        "plan_projection_cid",
        "migration_projection_cid",
        "migration_event_watermark",
        "projection_cid",
        "target_event_watermark",
        "validation_digest",
        "migration_digest",
        "migration_evidence_id",
        "plan_migration_event_id",
        "migration_evidence_event_id",
        "task_rearm_event_id",
        "task_rearm_receipt_cid",
        "coordination_rearm_event_id",
        "coordination_rearm_id",
        "coordination_projection_digest",
        "coordination_event_count",
        "control_store_sha256",
        "control_store_size",
        "coordination_store_sha256",
        "coordination_store_size",
        "semantic_authority_digest",
        "frozen_base_authority_digest",
        "append_surface_digest",
        "prior_database_path",
        "prior_control_wal_path",
        "prior_coordination_path",
        "prior_coordination_wal_path",
        "database_path",
        "coordination_path",
        "prior_publication_control_store_sha256",
        "prior_control_store_sha256",
        "prior_publication_coordination_store_sha256",
        "prior_coordination_store_sha256",
        "prior_control_wal_present",
        "prior_coordination_wal_present",
        "prior_event_prefix_sha256",
        "prior_source_binding_cid",
        "prior_materialization_receipt_cid",
        "prior_materialization_receipt_precedes_live_suffix",
        "current_source_binding_cid",
        "live_implementation_failure_cid",
        "declared_output_retry_successor_materialization_cid",
        "control_and_coordination_bases_copied",
        "prior_wals_absent",
        "prior_control_store_mutated",
        "prior_coordination_store_mutated",
        "failed_attempt_history_preserved",
        "failed_completion_barrier_rearmed",
        "coordination_semantic_changes",
        "execution_sidecar_copied",
        "read_replica_sidecar_copied",
        "task_revision_changes",
        "task_status_changes",
        "accepted_definition_changes",
        "accepted_completion_changes",
        "implementation_provider_invocations_observed",
        "implementation_provider_model_calls_observed",
        "implementation_provider_tokens_observed",
        "implementation_provider_cost_usd_observed",
        "settlement_provider_invocation_count",
        "provider_execution_accounting_mismatch",
        "effect_claim_changes",
        "implementation_commit_changes",
        "merge_attempt_changes",
        "worker_self_approval",
        "receipt_cid",
    }
    unhashed = dict(observed)
    claimed_cid = str(unhashed.pop("receipt_cid", ""))
    rearm = authority["task_rearm"]
    if (
        set(observed) != expected_keys
        or claimed_cid != materializer._identity(unhashed)
        or observed.get("schema") != "sawm/non-authoritative-migration-receipt@10"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != authority["migration_revision"]
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("plan_projection_cid") != authority["prior_projection_cid"]
        or observed.get("database_path") != _M12_STORE_ID
        or observed.get("coordination_path") != _M12_COORDINATION_STORE_ID
        or observed.get("prior_database_path") != authority["prior_store_id"]
        or observed.get("prior_control_wal_path")
        != authority["prior_control_wal_id"]
        or observed.get("prior_coordination_path")
        != authority["prior_coordination_store_id"]
        or observed.get("prior_coordination_wal_path")
        != authority["prior_coordination_wal_id"]
        or observed.get("prior_publication_control_store_sha256")
        != authority["prior_publication_control_store_sha256"]
        or observed.get("prior_control_store_sha256")
        != authority["prior_control_store_sha256"]
        or observed.get("prior_publication_coordination_store_sha256")
        != authority["prior_publication_coordination_store_sha256"]
        or observed.get("prior_coordination_store_sha256")
        != authority["prior_coordination_store_sha256"]
        or observed.get("prior_control_wal_present") is not False
        or observed.get("prior_coordination_wal_present") is not False
        or observed.get("prior_event_prefix_sha256")
        != authority["prior_event_prefix_sha256"]
        or observed.get("prior_source_binding_cid")
        != authority["prior_source_binding_cid"]
        or observed.get("prior_materialization_receipt_cid")
        != authority["prior_materialization_receipt_cid"]
        or observed.get("prior_materialization_receipt_precedes_live_suffix")
        is not True
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or observed.get("migration_evidence_id") != migration_evidence_id
        or observed.get("task_rearm_receipt_cid") != task_rearm_receipt_cid
        or any(
            not str(observed.get(field) or "")
            for field in (
                "plan_migration_event_id",
                "migration_evidence_event_id",
                "task_rearm_event_id",
                "coordination_rearm_event_id",
                "coordination_rearm_id",
            )
        )
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(observed.get("control_store_sha256") or ""),
        )
        is None
        or int(observed.get("control_store_size") or 0) <= 0
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or observed.get("migration_projection_cid")
        != authority["target_projection_cid"]
        or observed.get("projection_cid") != authority["target_projection_cid"]
        or int(observed.get("target_event_watermark") or 0)
        != int(authority["target_event_watermark"])
        or observed.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or observed.get("semantic_authority_digest")
        != authority["target_semantic_authority_digest"]
        or observed.get("frozen_base_authority_digest")
        != authority["target_frozen_base_authority_digest"]
        or observed.get("declared_output_retry_successor_materialization_cid")
        != materializer._identity(dict(authority))
        or observed.get("live_implementation_failure_cid")
        != authority["live_implementation_failure_cid"]
        or int(observed.get("migration_event_watermark") or 0)
        != int(authority["target_event_watermark"])
        or int(observed.get("coordination_event_count") or 0)
        != int(rearm["target_coordination_event_count"])
        or observed.get("control_and_coordination_bases_copied") is not True
        or observed.get("prior_wals_absent") is not True
        or observed.get("prior_control_store_mutated") is not False
        or observed.get("prior_coordination_store_mutated") is not False
        or observed.get("coordination_semantic_changes") != 1
        or observed.get("failed_attempt_history_preserved") is not True
        or observed.get("failed_completion_barrier_rearmed") is not True
        or observed.get("task_revision_changes") != 1
        or observed.get("task_status_changes") != 1
        or observed.get("accepted_definition_changes") != 0
        or observed.get("accepted_completion_changes") != 0
        or observed.get("implementation_provider_invocations_observed") != 2
        or observed.get("implementation_provider_model_calls_observed") != 92
        or observed.get("implementation_provider_tokens_observed") != 11_202_083
        or observed.get("implementation_provider_cost_usd_observed")
        != "1.23726850"
        or observed.get("settlement_provider_invocation_count") != 0
        or observed.get("provider_execution_accounting_mismatch") is not True
        or observed.get("effect_claim_changes") != 0
        or observed.get("implementation_commit_changes") != 0
        or observed.get("merge_attempt_changes") != 0
        or observed.get("execution_sidecar_copied") is not False
        or observed.get("read_replica_sidecar_copied") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M12 materialized final pair marker differs")
    if checked is not None:
        expected_control = str((REPO_ROOT / _M12_STORE_ID).resolve())
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != expected_control
            or checked.get("coordination_path") != str(coordination_path)
            or checked.get("projection_cid") != authority["target_projection_cid"]
            or checked.get("coordination_projection_digest")
            != authority["target_coordination_projection_digest"]
            or int(checked.get("event_watermark") or 0)
            != int(authority["target_event_watermark"])
            or checked.get("receipt") != observed
        ):
            raise OperatorError(
                "M12 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _require_m11_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M11's receipt-last marker for its repaired/rearmed pair."""

    if "live_provider_retry_successor_materialization" not in config:
        return MappingProxyType({})
    _validate_m11_runtime_binding(config, authority)
    receipt_path = (
        REPO_ROOT / Path(_M11_STORE_ID).parent / "migration-receipt.json"
    )
    try:
        observed, _receipt_sha256 = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M11 final pair commit marker",
        )
    except Exception as exc:
        raise OperatorError(
            "M11 materialized final pair marker is unavailable"
        ) from exc
    if not isinstance(observed, Mapping):
        raise OperatorError("M11 materialized final pair marker is invalid")
    coordination_path = (REPO_ROOT / _M11_COORDINATION_STORE_ID).resolve()
    try:
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination_path,
                root=REPO_ROOT,
                noun="materialized M11 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError(
            "M11 materialized coordination store is unavailable"
        ) from exc
    unhashed = dict(observed)
    claimed_cid = str(unhashed.pop("receipt_cid", ""))
    rearm = authority["task_rearm"]
    if (
        claimed_cid != materializer._identity(unhashed)
        or observed.get("schema") != "sawm/non-authoritative-migration-receipt@9"
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != authority["migration_revision"]
        or observed.get("database_path") != _M11_STORE_ID
        or observed.get("coordination_path") != _M11_COORDINATION_STORE_ID
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or observed.get("migration_projection_cid")
        != authority["target_projection_cid"]
        or observed.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or observed.get("semantic_authority_digest")
        != authority["target_semantic_authority_digest"]
        or observed.get("live_provider_retry_successor_materialization_cid")
        != materializer._identity(dict(authority))
        or observed.get("live_implementation_failure_cid")
        != authority["live_implementation_failure_cid"]
        or int(observed.get("migration_event_watermark") or 0)
        != int(authority["target_event_watermark"])
        or int(observed.get("coordination_event_count") or 0)
        != int(rearm["target_coordination_event_count"])
        or observed.get("coordination_base_and_wal_copied") is not True
        or observed.get("coordination_wal_replayed_on_copy") is not True
        or observed.get("prior_coordination_store_mutated") is not False
        or observed.get("coordination_semantic_changes") != 1
        or observed.get("failed_attempt_history_preserved") is not True
        or observed.get("failed_completion_barrier_rearmed") is not True
        or observed.get("task_revision_changes") != 1
        or observed.get("task_status_changes") != 1
        or observed.get("accepted_definition_changes") != 0
        or observed.get("accepted_completion_changes") != 0
        or observed.get("implementation_provider_invocations_observed") != 1
        or observed.get("settlement_provider_invocation_count") != 0
        or observed.get("provider_execution_accounting_mismatch") is not True
        or observed.get("effect_claim_changes") != 0
        or observed.get("implementation_commit_changes") != 0
        or observed.get("merge_attempt_changes") != 0
        or observed.get("execution_sidecar_copied") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M11 materialized final pair marker differs")
    if checked is not None:
        expected_control = str((REPO_ROOT / _M11_STORE_ID).resolve())
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != expected_control
            or checked.get("coordination_path") != str(coordination_path)
            or checked.get("projection_cid") != authority["target_projection_cid"]
            or checked.get("coordination_projection_digest")
            != authority["target_coordination_projection_digest"]
            or int(checked.get("event_watermark") or 0)
            != int(authority["target_event_watermark"])
            or checked.get("receipt") != observed
        ):
            raise OperatorError(
                "M11 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


_LIVE_APPEND_TABLES = frozenset(
    {"domain_events", "evidence_nodes", "plan_revisions", "plans"}
)


def _sql_table_names(rows: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(row[0]) for row in rows)


def _live_remote_frozen_table_names(connection: Any) -> tuple[str, ...]:
    """List frozen base tables from the remote-visible schema surface.

    Attached Quack can omit ``information_schema.tables`` ``BASE TABLE``
    entries while still exposing columns and rows. Prefer ``duckdb_tables()``,
    then reconstruct from columns minus views. Never open a filesystem store.
    """

    duck_names = _sql_table_names(
        connection.execute(
            "SELECT table_name FROM duckdb_tables() "
            "WHERE database_name = current_database() AND schema_name = 'main' "
            "ORDER BY table_name"
        ).fetchall()
    )
    info_names = _sql_table_names(
        connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE' "
            "ORDER BY table_name"
        ).fetchall()
    )
    if duck_names:
        if info_names and info_names != duck_names:
            raise OperatorError("M18 live frozen table inventory differs")
        names = duck_names
    elif info_names:
        names = info_names
    else:
        column_names = _sql_table_names(
            connection.execute(
                "SELECT DISTINCT table_name FROM information_schema.columns "
                "WHERE table_schema='main' ORDER BY table_name"
            ).fetchall()
        )
        view_names = set(
            _sql_table_names(
                connection.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='main' AND table_type='VIEW' "
                    "ORDER BY table_name"
                ).fetchall()
            )
        )
        names = tuple(name for name in column_names if name not in view_names)
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names):
        raise OperatorError("M18 live frozen table inventory differs")
    tables = tuple(name for name in names if name not in _LIVE_APPEND_TABLES)
    if not tables or _LIVE_APPEND_TABLES.intersection(tables):
        raise OperatorError("M18 live frozen table inventory differs")
    return tables


def _m18_live_frozen_authority_digest(
    connection: Any,
    identity: Mapping[str, Any],
    remote_identity: Mapping[str, Any],
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
) -> str:
    """Reconstruct M18's sealed pre-owner frozen projection from live rows."""

    import datetime

    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        PINNED_PROFILE_ID,
        PINNED_QUACK_EXTENSION,
    )

    generation = int(authority["target_generation"])
    server_id = str(identity.get("server_id") or "")
    process_birth_id = str(identity.get("process_birth_id") or "")
    started_at = str(identity.get("started_at") or "")
    credential_generation = int(identity.get("credential_generation") or 0)
    secret_handle = str(identity.get("secret_handle") or "")
    capability_id = f"cap:{server_id}:{generation}"
    credential_id = f"cred:{server_id}:{credential_generation}"
    owner = config.get("quack_owner")
    if (
        not isinstance(owner, Mapping)
        or remote_identity.get("canonical_rows_verified") is not True
        or remote_identity.get("live") is not True
        or any(
            remote_identity.get(key) != identity.get(key)
            for key in (
                "server_id",
                "store_id",
                "database_uuid",
                "process_birth_id",
                "listen_uri",
                "extension_fingerprint",
                "schema_revision",
                "generation",
                "credential_generation",
            )
        )
        or int(identity.get("generation") or 0) != generation
        or int(identity.get("fence_epoch") or 0) != generation
        or int(identity.get("revision", -1)) != 0
        or credential_generation != generation
        or identity.get("status") != "ready"
        or identity.get("store_id") != authority["target_store_id"]
        or identity.get("database_uuid") != authority["prior_database_uuid"]
        or identity.get("listen_uri")
        != config.get("database_program", {}).get("quack_endpoint")
        or secret_handle != owner.get("secret_handle")
        or not server_id
        or not process_birth_id
        or not started_at
        or int(identity.get("startup_epoch") or 0) <= 0
        or not str(identity.get("extension_fingerprint") or "")
    ):
        raise OperatorError("M18 live lifecycle identity differs")

    lifecycle_rows: dict[str, list[tuple[Any, ...]]] = {}
    for table, order in (
        ("state_servers", "1,8"),
        ("store_generations", "1"),
        ("server_epochs", "1,2"),
        ("capability_snapshots", "1"),
        ("credentials", "1"),
    ):
        lifecycle_rows[table] = [
            tuple(row[index] for index in range(len(row)))
            for row in connection.execute(
                f'SELECT * FROM "{table}" ORDER BY {order}'
            ).fetchall()
        ]

    extension_fingerprint = str(identity["extension_fingerprint"])
    capability_body = json.dumps(
        {
            "status": "compatible",
            "profile_id": PINNED_PROFILE_ID,
            "extension_fingerprint": extension_fingerprint,
        },
        sort_keys=True,
    )
    expected_lifecycle = {
        "state_servers": (
            server_id,
            authority["target_store_id"],
            authority["prior_database_uuid"],
            process_birth_id,
            identity["listen_uri"],
            extension_fingerprint,
            int(identity["schema_revision"]),
            generation,
            started_at,
            None,
            "ready",
            1,
            "",
            "{}",
        ),
        "store_generations": (
            generation,
            int(identity["schema_revision"]),
            generation,
            0,
            authority["prior_database_uuid"],
            process_birth_id,
            started_at,
            "",
            "{}",
        ),
        "server_epochs": (
            server_id,
            int(identity["startup_epoch"]),
            generation,
            started_at,
            None,
        ),
        "capability_snapshots": (
            capability_id,
            server_id,
            PINNED_PROFILE_ID,
            str(duckdb.__version__),
            PINNED_QUACK_EXTENSION,
            extension_fingerprint,
            "compatible",
            started_at,
            capability_body,
        ),
        "credentials": (
            credential_id,
            secret_handle,
            credential_generation,
            "quack-auth",
            started_at,
            None,
            None,
            0,
        ),
    }
    selectors = {
        "state_servers": lambda row: int(row[7]) == generation,
        "store_generations": lambda row: int(row[0]) == generation,
        "server_epochs": lambda row: str(row[0]) == server_id,
        "capability_snapshots": lambda row: str(row[1]) == server_id,
        "credentials": lambda row: int(row[2]) == credential_generation,
    }
    for table, expected in expected_lifecycle.items():
        selected = [row for row in lifecycle_rows[table] if selectors[table](row)]
        if selected != [expected]:
            raise OperatorError(f"M18 live {table} lifecycle row differs")

    def jsonable(value: Any) -> Any:
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
            return {"iso8601": value.isoformat()}
        return value

    tables = _live_remote_frozen_table_names(connection)
    projection: dict[str, list[list[Any]]] = {}
    for table in tables:
        columns = connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name=? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        if not columns:
            raise OperatorError(f"M18 live frozen table is missing: {table}")
        order = ", ".join(str(index) for index in range(1, len(columns) + 1))
        rows = (
            list(lifecycle_rows[table])
            if table in lifecycle_rows
            else [
                tuple(row[index] for index in range(len(columns)))
                for row in connection.execute(
                    f'SELECT * FROM "{table}" ORDER BY {order}'
                ).fetchall()
            ]
        )
        if table in selectors:
            rows = [row for row in rows if not selectors[table](row)]
        projection[table] = [
            [jsonable(value) for value in row]
            for row in rows
        ]
    return materializer._identity(projection)


def _verify_m18_live_final_pair_authority_from_verified_copies(
    source: Any,
    coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    observed: Mapping[str, Any],
    validation_digest: str,
    live_identity: Mapping[str, Any],
    remote_identity: Mapping[str, Any],
    *,
    prior_control: Path,
    prior_coordination: Path,
) -> Mapping[str, Any]:
    """Resolve M18 fields while querying sealed disposable M17 copies only."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        TASK_CLAIM_FAILURE_REARMED_EVENT,
        read_coordination_registry_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
        canonical_content_cid,
    )

    expected_body = materializer._m18_migration_body(
        population,
        config,
        validation_digest,
    )
    migration_digest = materializer._identity(expected_body)
    expected_delta = materializer._m18_migration_plan_delta(population, config)
    operator_task_cid = str(
        population["migration_inventory"]["prior_task_cids"]["SAWM-000"]
    )
    migration_evidence_id = content_identity(
        {
            "task_cid": operator_task_cid,
            "evidence_kind": "operator_control_plane_source_migration",
            "digest": migration_digest,
            "body": expected_body,
        }
    )
    recorded_at = str(materializer._M18_CONTROL_RECORDED_AT)
    owner_id = "sawm-r2-portal-completion-persistence-migrator"
    prior_control_before = (
        authority["prior_control_store_sha256"],
        int(authority["prior_control_store_size"]),
    )

    before_watermark = source.intent.event_watermark()
    live_events = tuple(
        dict(event)
        for event in source.intent.list_events(
            after_global_sequence=int(authority["prior_event_watermark"]),
            limit=4,
        )
    )
    import duckdb

    prior = duckdb.connect(str(prior_control), read_only=True)
    with source.intent._connection(write=False) as connection:
        try:
            prefix = materializer._event_prefix_digest(
                connection,
                int(authority["prior_event_watermark"]),
            )
            semantic_digest = materializer._semantic_authority_digest_on(connection)
            append_digest = materializer._authority_table_digest_on(
                connection,
                ("plans", "plan_revisions", "evidence_nodes", "domain_events"),
            )
            native_connection = getattr(connection, "_connection", connection)
            catalog_digest = materializer._main_catalog_digest_on(
                native_connection
            )
            frozen_digest = _m18_live_frozen_authority_digest(
                connection,
                live_identity,
                remote_identity,
                config,
                authority,
                materializer,
            )
            plan_rows = materializer._positional_rows(
                connection.execute("SELECT * FROM plans ORDER BY 1,2").fetchall(),
                8,
            )
            revision_rows = materializer._positional_rows(
                connection.execute(
                    "SELECT * FROM plan_revisions ORDER BY 1,2,3,4"
                ).fetchall(),
                4,
            )
            evidence_rows = materializer._positional_rows(
                connection.execute(
                    "SELECT * FROM evidence_nodes ORDER BY 1,2,3,4,5,6,7"
                ).fetchall(),
                7,
            )
            event_sql = (
                "SELECT event_id,stream_id,sequence,global_sequence,event_type,"
                "task_cid,attempt_id,session_id,recorded_at,body_json "
                "FROM domain_events ORDER BY global_sequence"
            )
            event_rows = materializer._positional_rows(
                connection.execute(event_sql).fetchall(),
                10,
            )

            prior_prefix = materializer._event_prefix_digest(
                prior,
                int(authority["prior_event_watermark"]),
            )
            prior_semantic_digest = materializer._semantic_authority_digest_on(prior)
            prior_frozen_digest = materializer._frozen_base_authority_digest_on(prior)
            prior_append_digest = materializer._authority_table_digest_on(
                prior,
                ("plans", "plan_revisions", "evidence_nodes", "domain_events"),
            )
            prior_catalog_digest = materializer._main_catalog_digest_on(prior)
            prior_plan_rows = materializer._positional_rows(
                prior.execute("SELECT * FROM plans ORDER BY 1,2").fetchall(),
                8,
            )
            prior_revision_rows = materializer._positional_rows(
                prior.execute(
                    "SELECT * FROM plan_revisions ORDER BY 1,2,3,4"
                ).fetchall(),
                4,
            )
            prior_evidence_rows = materializer._positional_rows(
                prior.execute(
                    "SELECT * FROM evidence_nodes ORDER BY 1,2,3,4,5,6,7"
                ).fetchall(),
                7,
            )
            prior_event_rows = materializer._positional_rows(
                prior.execute(event_sql).fetchall(),
                10,
            )
        finally:
            prior.close()
    after_watermark = source.intent.event_watermark()
    prior_control_after = materializer._stable_regular_sha256(
        prior_control,
        root=prior_control.parent,
        noun="verified stopped M17 control copy",
        required_link_count=1,
    )

    prior_revision = [
        row
        for row in revision_rows
        if int(row[1]) == int(authority["prior_plan_revision"])
    ]
    target_revision = [
        row
        for row in revision_rows
        if int(row[1]) == int(authority["target_plan_revision"])
    ]
    if len(plan_rows) != 1 or len(prior_revision) != 1 or len(target_revision) != 1:
        raise OperatorError("M18 live plan revision authority is incomplete")
    current_plan = plan_rows[0]
    prior_plan_body = json.loads(str(prior_plan_rows[0][7]))
    expected_plan_body = {
        **prior_plan_body,
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "source_migration_revision": "SAWM-R2-M18",
        "source_migration_digest": migration_digest,
        "supersession_mode": authority["supersession_mode"],
        "last_delta": expected_delta,
    }
    prior_evidence = {str(row[0]): row for row in prior_evidence_rows}
    current_evidence = {str(row[0]): row for row in evidence_rows}
    evidence = current_evidence.get(migration_evidence_id)
    if (
        before_watermark != _M18_TARGET_EVENT_WATERMARK
        or after_watermark != before_watermark
        or prior_control_after != prior_control_before
        or prefix
        != (
            authority["prior_event_prefix_sha256"],
            authority["prior_event_watermark"],
        )
        or prior_prefix != prefix
        or prior_semantic_digest != authority["prior_semantic_authority_digest"]
        or prior_frozen_digest != authority["prior_frozen_base_authority_digest"]
        or prior_append_digest != authority["prior_append_surface_digest"]
        or prior_catalog_digest != authority["prior_catalog_digest"]
        or semantic_digest != authority["target_semantic_authority_digest"]
        or frozen_digest != authority["target_frozen_base_authority_digest"]
        or catalog_digest != authority["prior_catalog_digest"]
        or len(prior_plan_rows) != 1
        or len(prior_revision_rows) != int(authority["prior_plan_revision"])
        or revision_rows[:-1] != prior_revision_rows
        or len(evidence_rows) != len(prior_evidence_rows) + 1
        or len(prior_event_rows) != int(authority["prior_event_watermark"])
        or event_rows[: int(authority["prior_event_watermark"])]
        != prior_event_rows
        or len(event_rows) != int(authority["target_event_watermark"])
        or current_plan[:5] != prior_plan_rows[0][:5]
        or len(revision_rows) != int(authority["target_plan_revision"])
        or int(current_plan[6]) != int(authority["target_plan_revision"])
        or str(current_plan[5]) != recorded_at
        or json.loads(str(current_plan[7])) != expected_plan_body
        or any(current_evidence.get(key) != row for key, row in prior_evidence.items())
        or set(current_evidence) - set(prior_evidence) != {migration_evidence_id}
        or target_revision[0]
        != (
            str(population["plan_root_cid"]),
            int(authority["target_plan_revision"]),
            materializer._canonical(expected_plan_body).decode("utf-8"),
            recorded_at,
        )
        or evidence is None
        or tuple(evidence[1:6])
        != (
            "",
            operator_task_cid,
            "operator_control_plane_source_migration",
            migration_digest,
            recorded_at,
        )
        or json.loads(str(evidence[6])) != expected_body
    ):
        raise OperatorError("M18 live plan/evidence authority differs")

    expected_plan_envelope = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": "intent.plan_revision_appended",
        "subject_id": str(population["plan_root_cid"]),
        "owner_id": owner_id,
        "recorded_at": recorded_at,
        "body": {
            "plan_cid": str(population["plan_root_cid"]),
            "goal_cid": str(
                population["migration_inventory"]["prior_goal_cids"]["SAWM-G000"]
            ),
            "plan_alias": "SAWM-PLAN-R2",
            "status": "active",
            "revision": int(authority["target_plan_revision"]),
            "body": expected_plan_body,
            "delta": expected_delta,
            "recorded_at": recorded_at,
        },
    }
    expected_evidence_envelope = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": "intent.evidence_recorded",
        "subject_id": migration_evidence_id,
        "owner_id": owner_id,
        "recorded_at": recorded_at,
        "body": {
            "evidence_id": migration_evidence_id,
            "parent_evidence_id": "",
            "task_cid": operator_task_cid,
            "evidence_kind": "operator_control_plane_source_migration",
            "digest": migration_digest,
            "body": expected_body,
            "created_at": recorded_at,
            "revision": 0,
        },
    }
    task = next(
        (
            item
            for item in population["taskboard"]
            if str(item.get("task_alias") or "") == "SAWM-007"
        ),
        None,
    )
    if task is None:
        raise OperatorError("M18 live SAWM-007 authority is unavailable")
    rearm_receipt = materializer._m18_task_rearm_receipt("SAWM-007")
    expected_task_envelope = {
        "schema": "ipfs_accelerate_py/agent-supervisor/intent-event@1",
        "event_type": "intent.task_status_changed",
        "subject_id": str(task["task_cid"]),
        "owner_id": owner_id,
        "recorded_at": recorded_at,
        "body": {
            "task_cid": str(task["task_cid"]),
            "task_alias": "SAWM-007",
            "goal_cid": str(task["goal_cid"]),
            "previous_status": "blocked",
            "status": "retrying",
            "revision": 5,
            "receipt": rearm_receipt,
            "recorded_at": recorded_at,
        },
    }
    envelopes = (
        ("intent.plan_revision_appended", "", expected_plan_envelope),
        ("intent.evidence_recorded", operator_task_cid, expected_evidence_envelope),
        ("intent.task_status_changed", str(task["task_cid"]), expected_task_envelope),
    )
    expected_events: list[dict[str, Any]] = []
    for offset, (event_type, task_cid, envelope) in enumerate(envelopes, start=1):
        sequence = int(authority["prior_event_watermark"]) + offset
        event_id = content_identity(
            {
                "stream_id": "stream:intent",
                "sequence": sequence,
                "global_sequence": sequence,
                "event_type": event_type,
                "body": envelope,
            }
        )
        expected_events.append(
            {
                "event_id": event_id,
                "stream_id": "stream:intent",
                "sequence": sequence,
                "global_sequence": sequence,
                "event_type": event_type,
                "task_cid": task_cid,
                "attempt_id": "",
                "session_id": "session:intent",
                "recorded_at": recorded_at,
                "body": envelope,
            }
        )
    if live_events != tuple(expected_events):
        raise OperatorError("M18 live intent-event tail differs")

    coordination_before = materializer._stable_regular_sha256(
        coordination,
        root=REPO_ROOT,
        noun="materialized M18 coordination store",
        required_link_count=1,
    )
    prior_anchor = materializer._stable_regular_sha256(
        prior_coordination,
        root=prior_coordination.parent,
        noun="verified stopped M17 coordination copy",
        required_link_count=1,
    )
    projection = read_coordination_registry_projection(coordination)
    prior_projection = read_coordination_registry_projection(prior_coordination)
    prior_db = duckdb.connect(str(prior_coordination), read_only=True)
    target_db = duckdb.connect(str(coordination), read_only=True)
    try:
        prior_coordination_events = materializer._positional_rows(
            prior_db.execute("SELECT * FROM lease_events ORDER BY event_id").fetchall(),
            8,
        )
        target_coordination_events = materializer._positional_rows(
            target_db.execute("SELECT * FROM lease_events ORDER BY event_id").fetchall(),
            8,
        )
    finally:
        target_db.close()
        prior_db.close()
    coordination_after = materializer._stable_regular_sha256(
        coordination,
        root=REPO_ROOT,
        noun="materialized M18 coordination store",
        required_link_count=1,
    )
    prior_map = {str(row[0]): row for row in prior_coordination_events}
    target_map = {str(row[0]): row for row in target_coordination_events}
    new_event_ids = set(target_map) - set(prior_map)
    prior_counts = prior_projection.get("counts")
    active_count_names = (
        "active_fenced_leases",
        "active_maintenance_leases",
        "active_resource_claims",
        "active_task_attempts",
        "active_task_claims",
    )
    prior_active_counts_are_zero = (
        isinstance(prior_counts, Mapping)
        and set(active_count_names).issubset(prior_counts)
        and all(
            type(prior_counts[name]) is int and prior_counts[name] == 0
            for name in active_count_names
        )
    )
    failure = authority["failure_receipts"]["SAWM-007"]
    expected_coordination_rearm = {
        "schema": "ipfs_accelerate_py/agent-supervisor/task-claim-failure-rearm@1",
        "operation": "operator_control_plane_repair",
        "task_cid": failure["task_cid"],
        "failure_settlement_id": failure["settlement_id"],
        "control_status": "retrying",
        "control_revision": 5,
        "control_receipt_id": canonical_content_cid(rearm_receipt),
    }
    expected_coordination_rearm["rearm_id"] = canonical_content_cid(
        expected_coordination_rearm
    )
    new_event = target_map[next(iter(new_event_ids))] if len(new_event_ids) == 1 else None
    if (
        coordination_before != coordination_after
        or coordination_after
        != (
            str(observed["coordination_store_sha256"]),
            int(observed["coordination_store_size"]),
        )
        or prior_anchor
        != (
            authority["prior_coordination_store_sha256"],
            int(authority["prior_coordination_store_size"]),
        )
        or prior_projection.get("projection_root")
        != authority["prior_coordination_projection_digest"]
        or int(authority["prior_active_count"]) != 0
        or not prior_active_counts_are_zero
        or projection.get("projection_root")
        != authority["target_coordination_projection_digest"]
        or any(target_map.get(key) != row for key, row in prior_map.items())
        or new_event is None
        or len(target_coordination_events)
        != int(authority["target_coordination_event_count"])
        or str(new_event[1]) != failure["lease_id"]
        or str(new_event[2]) != f"task:{failure['task_cid']}"
        or str(new_event[3]) != TASK_CLAIM_FAILURE_REARMED_EVENT
        or int(new_event[4]) != int(failure["fencing_token"])
        or int(new_event[5]) != int(failure["fence_epoch"])
        or int(new_event[6])
        != int(authority["task_rearms"]["SAWM-007"]["coordination_rearm_observed_at_ms"])
        or json.loads(str(new_event[7])) != expected_coordination_rearm
    ):
        raise OperatorError("M18 offline coordination rearm authority differs")

    event_ids = {
        "SAWM-007": expected_events[2]["event_id"],
    }
    coordination_event_ids = {"SAWM-007": str(new_event[0])}
    return MappingProxyType(
        {
            "validation_digest": validation_digest,
            "migration_digest": migration_digest,
            "migration_evidence_id": migration_evidence_id,
            "plan_migration_event_id": expected_events[0]["event_id"],
            "migration_evidence_event_id": expected_events[1]["event_id"],
            "task_rearm_event_ids": event_ids,
            "task_rearm_receipt_cids": {
                "SAWM-007": content_identity(rearm_receipt)
            },
            "coordination_rearm_event_ids": coordination_event_ids,
            "coordination_rearm_ids": {
                "SAWM-007": expected_coordination_rearm["rearm_id"]
            },
            "append_surface_digest": append_digest,
            "catalog_digest": catalog_digest,
            "coordination_projection_digest": projection["projection_root"],
            "coordination_event_count": len(target_coordination_events),
        }
    )


def _verify_m18_live_final_pair_authority(
    source: Any,
    coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    observed: Mapping[str, Any],
    validation_digest: str,
    live_identity: Mapping[str, Any],
    remote_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Verify M18 while preserving exact stopped M17 byte/lifecycle anchors."""

    try:
        prior_before = materializer._m18_prior_artifact_anchor_snapshot(
            REPO_ROOT,
            authority,
        )
    except Exception as exc:
        raise OperatorError("M18 stopped M17 artifact authority differs") from exc

    def require_unchanged_prior() -> None:
        try:
            prior_after = materializer._m18_prior_artifact_anchor_snapshot(
                REPO_ROOT,
                authority,
            )
        except Exception as exc:
            raise OperatorError(
                "M18 stopped M17 artifact authority changed during verification"
            ) from exc
        if prior_after != prior_before:
            raise OperatorError(
                "M18 stopped M17 artifact authority changed during verification"
            )

    try:
        with tempfile.TemporaryDirectory(
            prefix="sawm-r2-m18-live-prior-",
            dir="/tmp",
        ) as temporary:
            copy_root = Path(temporary)
            prior_control = copy_root / "control.duckdb"
            prior_coordination = copy_root / "control.coordination.duckdb"
            shutil.copyfile(Path(prior_before["control"]), prior_control)
            shutil.copyfile(
                Path(prior_before["coordination"]),
                prior_coordination,
            )
            expected_copies = {
                prior_control: prior_before["anchors"]["control"],
                prior_coordination: prior_before["anchors"]["coordination"],
            }
            for path, expected in expected_copies.items():
                observed_copy = materializer._stable_regular_sha256(
                    path,
                    root=copy_root,
                    noun=f"verified stopped M17 {path.name} copy",
                    required_link_count=1,
                )
                if observed_copy != expected:
                    raise OperatorError("M18 stopped M17 disposable copy differs")
            report = _verify_m18_live_final_pair_authority_from_verified_copies(
                source,
                coordination,
                population,
                config,
                authority,
                materializer,
                observed,
                validation_digest,
                live_identity,
                remote_identity,
                prior_control=prior_control,
                prior_coordination=prior_coordination,
            )
    except Exception:
        require_unchanged_prior()
        raise
    require_unchanged_prior()
    return report


def _require_m18_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
    live_source: Any | None = None,
    validation_digest: str = "",
    live_identity: Mapping[str, Any] | None = None,
    remote_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M18's receipt-last marker and exact sole-rearm bindings."""

    key = "portal_completion_persistence_successor_materialization"
    try:
        expected_authority = (
            materializer._expected_m18_portal_completion_persistence_authority()
        )
    except Exception as exc:
        raise OperatorError("M18 exact portal-completion authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M18 portal-completion authority differs")

    control = (REPO_ROOT / _M18_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M18_COORDINATION_STORE_ID).resolve()
    receipt_path = control.parent / "migration-receipt.json"
    try:
        materializer._assert_m18_no_pending_receipts(control)
        observed, _ = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M18 final pair marker",
        )
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M18 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M18 final pair marker is unavailable") from exc

    population = materializer.build_population(REPO_ROOT)
    unhashed = dict(observed)
    claimed = unhashed.pop("receipt_cid", "")
    if set(observed) != _M18_RECEIPT_KEYS or claimed != materializer._identity(
        unhashed
    ):
        raise OperatorError("M18 materialized final pair marker differs")
    effective_validation_digest = (
        str(checked.get("validation_digest") or "")
        if checked is not None
        else str(validation_digest or "")
    )
    live_report: Mapping[str, Any] | None = None
    if checked is None:
        if (
            live_source is None
            or not effective_validation_digest
            or not isinstance(live_identity, Mapping)
            or not isinstance(remote_identity, Mapping)
        ):
            raise OperatorError("M18 live final-pair authority is unavailable")
        try:
            live_report = _verify_m18_live_final_pair_authority(
                live_source,
                coordination,
                population,
                config,
                authority,
                materializer,
                observed,
                effective_validation_digest,
                live_identity,
                remote_identity,
            )
        except OperatorError:
            raise
        except Exception as exc:
            raise OperatorError("M18 live final-pair authority differs") from exc
    plural_names = (
        "task_rearm_event_ids",
        "task_rearm_receipt_cids",
        "coordination_rearm_event_ids",
        "coordination_rearm_ids",
    )
    plural_bindings_are_closed = all(
        isinstance(observed.get(name), Mapping)
        and set(observed[name]) == {"SAWM-007"}
        and all(str(value) for value in observed[name].values())
        for name in plural_names
    )
    authority_fields = {
        "prior_database_path": "prior_store_id",
        "prior_coordination_path": "prior_coordination_store_id",
        "prior_control_store_sha256": "prior_control_store_sha256",
        "prior_control_store_size": "prior_control_store_size",
        "prior_coordination_store_sha256": "prior_coordination_store_sha256",
        "prior_coordination_store_size": "prior_coordination_store_size",
        "prior_event_watermark": "prior_event_watermark",
        "prior_event_prefix_sha256": "prior_event_prefix_sha256",
        "prior_projection_cid": "prior_projection_cid",
        "prior_semantic_authority_digest": "prior_semantic_authority_digest",
        "prior_frozen_base_authority_digest": (
            "prior_frozen_base_authority_digest"
        ),
        "prior_append_surface_digest": "prior_append_surface_digest",
        "prior_catalog_digest": "prior_catalog_digest",
        "prior_coordination_projection_digest": (
            "prior_coordination_projection_digest"
        ),
        "prior_coordination_event_count": "prior_coordination_event_count",
        "prior_generation": "prior_generation",
        "prior_plan_revision": "prior_plan_revision",
        "prior_server_id": "prior_server_id",
        "prior_process_birth_id": "prior_process_birth_id",
        "prior_startup_epoch": "prior_startup_epoch",
        "prior_started_at": "prior_started_at",
        "prior_stopped_at": "prior_stopped_at",
        "prior_stopped_status_projection_path": (
            "prior_stopped_status_projection_path"
        ),
        "prior_stopped_status_projection_sha256": (
            "prior_stopped_status_projection_sha256"
        ),
        "prior_stopped_status_projection_size": (
            "prior_stopped_status_projection_size"
        ),
        "prior_migration_receipt_path": "prior_migration_receipt_path",
        "prior_migration_receipt_sha256": "prior_migration_receipt_sha256",
        "prior_migration_receipt_size": "prior_migration_receipt_size",
        "prior_migration_receipt_cid": "prior_migration_receipt_cid",
        "repair_source_commit": "repair_source_commit",
        "accepted_source_repair": "accepted_source_repair",
        "failure_receipts_preserved": "failure_receipts",
        "operator_task_rearms": "task_rearms",
        "semantic_authority_digest": "target_semantic_authority_digest",
        "frozen_base_authority_digest": "target_frozen_base_authority_digest",
    }
    exact_flags = {
        "runtime_root_changed": True,
        "control_and_coordination_bases_copied": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "prior_lifecycle_artifacts_preserved": True,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
        "coordination_semantic_changes": 1,
        "task_revision_changes": 1,
        "task_status_changes": 1,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    if (
        observed.get("schema") != "sawm/non-authoritative-migration-receipt@16"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != "SAWM-R2-M18"
        or observed.get("database_path") != _M18_STORE_ID
        or observed.get("coordination_path") != _M18_COORDINATION_STORE_ID
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != effective_validation_digest
        or int(observed.get("target_generation") or 0) != _M18_GENERATION
        or int(observed.get("target_quack_port") or 0)
        != _M18_TARGET_QUACK_PORT
        or int(observed.get("target_event_watermark") or 0)
        != _M18_TARGET_EVENT_WATERMARK
        or observed.get("target_runtime_root") != str(Path(_M18_STORE_ID).parent)
        or observed.get("projection_cid") != _M18_TARGET_PROJECTION_CID
        or observed.get("plan_projection_cid")
        != authority["prior_projection_cid"]
        or observed.get("migration_projection_cid") != _M18_TARGET_PROJECTION_CID
        or int(observed.get("migration_event_watermark") or 0)
        != _M18_TARGET_EVENT_WATERMARK
        or observed.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or int(observed.get("coordination_event_count") or 0)
        != int(authority["target_coordination_event_count"])
        or observed.get(f"{key}_cid") != materializer._identity(dict(authority))
        or any(
            observed.get(receipt_key) != authority[authority_key]
            for receipt_key, authority_key in authority_fields.items()
        )
        or any(observed.get(name) != value for name, value in exact_flags.items())
        or not plural_bindings_are_closed
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(observed.get("control_store_sha256") or ""),
        )
        is None
        or int(observed.get("control_store_size") or 0) <= 0
        or int(observed.get("control_store_size") or 0) > 2 * 1024 * 1024 * 1024
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0) != coordination_size
        or config.get("runtime_paths", {}).get("root")
        != authority["target_runtime_root"]
        or (
            live_report is not None
            and any(
                observed.get(name) != live_report.get(name)
                for name in (
                    "validation_digest",
                    "migration_digest",
                    "migration_evidence_id",
                    "plan_migration_event_id",
                    "migration_evidence_event_id",
                    "task_rearm_event_ids",
                    "task_rearm_receipt_cids",
                    "coordination_rearm_event_ids",
                    "coordination_rearm_ids",
                    "append_surface_digest",
                    "catalog_digest",
                    "coordination_projection_digest",
                    "coordination_event_count",
                )
            )
        )
    ):
        raise OperatorError("M18 materialized final pair marker differs")

    if checked is not None:
        validation_digest = str(checked.get("validation_digest") or "")
        try:
            expected = materializer._expected_m18_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError("M18 exact materializer report is unavailable") from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or checked.get("projection_cid") != _M18_TARGET_PROJECTION_CID
            or int(checked.get("event_watermark") or 0)
            != _M18_TARGET_EVENT_WATERMARK
            or checked.get("receipt") != observed
            or observed != expected
        ):
            raise OperatorError(
                "M18 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _require_m34_source_successor_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M34 event 285 and the immutable M33 receipt/prefix."""

    key = "json_emission_normalization_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    expected_authority = (
        materializer._expected_m34_json_emission_normalization_authority()
    )
    expected_reference = materializer._m34_authority_reference()
    source_chain = expected_authority.get("source_chain", {})
    if (
        dict(authority) != expected_authority
        or config.get(key) != expected_reference
        or source_chain.get("initial_control_commit") != _M34_INITIAL_CONTROL_COMMIT
        or source_chain.get("initial_control_tree") != _M34_INITIAL_CONTROL_TREE
    ):
        raise OperatorError("M34 JSON-emission normalization authority differs")
    materializer._validated_m34_live_preflight_contract(expected_authority)
    if checked is None:
        raise OperatorError("M34 marker requires exact live materializer verification")

    m33_authority = materializer._expected_m33_live_preflight_contract_authority()
    m33_path = (REPO_ROOT / _M34_STORE_ID).resolve().parent / (
        "m33-source-successor-receipt.json"
    )
    try:
        m33_observed, m33_sha256 = materializer._load_nofollow_json(
            m33_path, root=REPO_ROOT, noun="M34 preserved M33 source receipt"
        )
    except Exception as exc:
        raise OperatorError("M34 preserved M33 source receipt is unavailable") from exc
    m33_unhashed = dict(m33_observed)
    m33_claimed = str(m33_unhashed.pop("receipt_cid", ""))
    prior = expected_authority["prior_authority"]
    if (
        m33_sha256 != prior["m33_receipt_sha256"]
        or m33_claimed != prior["m33_receipt_cid"]
        or m33_claimed != materializer._identity(m33_unhashed)
        or m33_observed.get("target_event_watermark")
        != prior["event_watermark"]
        or m33_observed.get("projection_cid") != prior["projection_cid"]
        or m33_observed.get("target_event_prefix_sha256")
        != prior["event_prefix_sha256"]
    ):
        raise OperatorError("M34 preserved M33 source receipt differs")
    m33_marker = _require_m33_source_successor_marker(
        config,
        m33_authority,
        materializer,
        checked={
            "valid": True,
            "event_watermark": m33_observed.get("target_event_watermark"),
            "projection_cid": m33_observed.get("projection_cid"),
            "prior_event_prefix_verified": True,
            "full_event_and_evidence_body_verified": m33_observed.get(
                "full_event_and_evidence_body_verified"
            ),
            "receipt": m33_observed,
        },
    )

    path = (REPO_ROOT / _M34_STORE_ID).resolve().parent / (
        "m34-source-successor-receipt.json"
    )
    try:
        observed, _ = materializer._load_nofollow_json(
            path, root=REPO_ROOT, noun="M34 source successor receipt"
        )
    except Exception as exc:
        raise OperatorError("M34 exact source successor receipt is unavailable") from exc
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    if (
        claimed != materializer._identity(unhashed)
        or checked.get("valid") is not True
        or checked.get("event_watermark") != _M34_TARGET_EVENT_WATERMARK
        or checked.get("projection_cid") != _M34_TARGET_PROJECTION_CID
        or checked.get("prior_event_prefix_verified") is not True
        or checked.get("full_event_and_evidence_body_verified") is not True
        or checked.get("receipt") != observed
        or observed.get("migration_revision") != "SAWM-R2-M34"
        or observed.get(f"{key}_cid") != materializer._identity(expected_authority)
        or observed.get("target_generation") != _M34_GENERATION
        or observed.get("target_event_watermark") != _M34_TARGET_EVENT_WATERMARK
        or observed.get("projection_cid") != _M34_TARGET_PROJECTION_CID
        or observed.get("m33_receipt_cid") != m33_claimed
        or observed.get("m33_event_prefix_verified") is not True
        or observed.get("normalized_live_preflight_contract_verified") is not True
        or observed.get("json_emission_normalization_verified") is not True
        or observed.get("recursive_json_emission_normalization_verified") is not True
        or observed.get("same_live_generation_29_owner_verified") is not True
        or observed.get("queried_and_mutated_through_live_quack_only") is not True
        or observed.get("direct_authoritative_file_opened") is not False
        or observed.get("full_event_and_evidence_body_verified") is not True
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M34 exact source successor receipt differs")
    return MappingProxyType(
        {
            **dict(m33_marker),
            **dict(observed),
            "prior_final_pair_receipt_cid": m33_marker[
                "prior_final_pair_receipt_cid"
            ],
            "m29_source_successor_receipt_cid": m33_marker[
                "m29_source_successor_receipt_cid"
            ],
            "m30_source_successor_receipt_cid": m33_marker[
                "m30_source_successor_receipt_cid"
            ],
            "m31_source_successor_receipt_cid": m33_marker[
                "m31_source_successor_receipt_cid"
            ],
            "m32_source_successor_receipt_cid": m33_marker[
                "m32_source_successor_receipt_cid"
            ],
            "m33_source_successor_receipt_cid": m33_claimed,
            "m34_source_successor_receipt_cid": claimed,
            "source_successor_receipt_cid": claimed,
            "source_successor_receipt_verified": True,
            "source_successor_chain": {
                **dict(m33_marker["source_successor_chain"]),
                "m34_source_successor_receipt_cid": claimed,
            },
            "final_pair_commit_marker_verified": False,
        }
    )


def _require_m33_source_successor_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M33 event 284 and the immutable M32 receipt/prefix."""

    key = "live_preflight_contract_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    expected_authority = materializer._expected_m33_live_preflight_contract_authority()
    expected_reference = materializer._m33_authority_reference()
    source_chain = expected_authority.get("source_chain", {})
    if (
        dict(authority) != expected_authority
        or config.get(key) != expected_reference
        or source_chain.get("initial_control_commit") != _M33_INITIAL_CONTROL_COMMIT
        or source_chain.get("initial_control_tree") != _M33_INITIAL_CONTROL_TREE
    ):
        raise OperatorError("M33 live-preflight contract authority differs")
    materializer._validated_m33_live_preflight_contract(expected_authority)
    if checked is None:
        raise OperatorError("M33 marker requires exact live materializer verification")
    m32_authority = materializer._expected_m32_live_preflight_plan_anchor_authority()
    m32_path = (REPO_ROOT / _M33_STORE_ID).resolve().parent / (
        "m32-source-successor-receipt.json"
    )
    try:
        m32_observed, m32_sha256 = materializer._load_nofollow_json(
            m32_path, root=REPO_ROOT, noun="M33 preserved M32 source receipt"
        )
    except Exception as exc:
        raise OperatorError("M33 preserved M32 source receipt is unavailable") from exc
    m32_unhashed = dict(m32_observed)
    m32_claimed = str(m32_unhashed.pop("receipt_cid", ""))
    prior = expected_authority["prior_authority"]
    if (
        m32_sha256 != prior["m32_receipt_sha256"]
        or m32_claimed != prior["m32_receipt_cid"]
        or m32_claimed != materializer._identity(m32_unhashed)
    ):
        raise OperatorError("M33 preserved M32 source receipt differs")
    m32_marker = _require_m32_source_successor_marker(
        config,
        m32_authority,
        materializer,
        checked={
            "valid": True,
            "event_watermark": m32_observed.get("target_event_watermark"),
            "projection_cid": m32_observed.get("projection_cid"),
            "prior_event_prefix_verified": True,
            "full_event_and_evidence_body_verified": m32_observed.get(
                "full_event_and_evidence_body_verified"
            ),
            "receipt": m32_observed,
        },
    )
    path = (REPO_ROOT / _M33_STORE_ID).resolve().parent / (
        "m33-source-successor-receipt.json"
    )
    try:
        observed, _ = materializer._load_nofollow_json(
            path, root=REPO_ROOT, noun="M33 source successor receipt"
        )
    except Exception as exc:
        raise OperatorError("M33 exact source successor receipt is unavailable") from exc
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    if (
        claimed != materializer._identity(unhashed)
        or checked.get("valid") is not True
        or checked.get("event_watermark") != _M33_TARGET_EVENT_WATERMARK
        or checked.get("projection_cid") != _M33_TARGET_PROJECTION_CID
        or checked.get("prior_event_prefix_verified") is not True
        or checked.get("full_event_and_evidence_body_verified") is not True
        or checked.get("receipt") != observed
        or observed.get("migration_revision") != "SAWM-R2-M33"
        or observed.get(f"{key}_cid") != materializer._identity(expected_authority)
        or observed.get("target_generation") != _M33_GENERATION
        or observed.get("target_event_watermark") != _M33_TARGET_EVENT_WATERMARK
        or observed.get("projection_cid") != _M33_TARGET_PROJECTION_CID
        or observed.get("normalized_live_preflight_contract_verified") is not True
        or observed.get("same_live_generation_29_owner_verified") is not True
        or observed.get("queried_and_mutated_through_live_quack_only") is not True
        or observed.get("direct_authoritative_file_opened") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M33 exact source successor receipt differs")
    return MappingProxyType(
        {
            **dict(m32_marker),
            **dict(observed),
            "prior_final_pair_receipt_cid": m32_marker[
                "prior_final_pair_receipt_cid"
            ],
            "m29_source_successor_receipt_cid": m32_marker[
                "m29_source_successor_receipt_cid"
            ],
            "m30_source_successor_receipt_cid": m32_marker[
                "m30_source_successor_receipt_cid"
            ],
            "m31_source_successor_receipt_cid": m32_marker[
                "m31_source_successor_receipt_cid"
            ],
            "m32_source_successor_receipt_cid": m32_claimed,
            "m33_source_successor_receipt_cid": claimed,
            "source_successor_receipt_cid": claimed,
            "source_successor_receipt_verified": True,
            "source_successor_chain": {
                **dict(m32_marker["source_successor_chain"]),
                "m33_source_successor_receipt_cid": claimed,
            },
            "final_pair_commit_marker_verified": False,
        }
    )


def _require_m32_source_successor_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M32's exact event-283 receipt and immutable M31 prefix."""

    key = "live_preflight_plan_anchor_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    expected_authority = materializer._expected_m32_live_preflight_plan_anchor_authority()
    expected_reference = materializer._m32_authority_reference()
    source_chain = expected_authority.get("source_chain", {})
    if (
        dict(authority) != expected_authority
        or config.get(key) != expected_reference
        or source_chain.get("initial_control_commit") != _M32_INITIAL_CONTROL_COMMIT
        or source_chain.get("initial_control_tree") != _M32_INITIAL_CONTROL_TREE
    ):
        raise OperatorError("M32 live-preflight plan authority differs")
    if checked is None:
        raise OperatorError("M32 marker requires exact live materializer verification")
    m31_authority = (
        materializer._expected_m31_detached_coordinator_pid_recovery_authority()
    )
    m31_path = (REPO_ROOT / _M32_STORE_ID).resolve().parent / (
        "m31-source-successor-receipt.json"
    )
    try:
        m31_observed, m31_sha256 = materializer._load_nofollow_json(
            m31_path, root=REPO_ROOT, noun="M32 preserved M31 source receipt"
        )
    except Exception as exc:
        raise OperatorError("M32 preserved M31 source receipt is unavailable") from exc
    m31_unhashed = dict(m31_observed)
    m31_claimed = str(m31_unhashed.pop("receipt_cid", ""))
    prior = expected_authority["prior_authority"]
    if (
        m31_sha256 != prior["m31_receipt_sha256"]
        or m31_claimed != prior["m31_receipt_cid"]
        or m31_claimed != materializer._identity(m31_unhashed)
    ):
        raise OperatorError("M32 preserved M31 source receipt differs")
    m31_marker = _require_m31_source_successor_marker(
        config,
        m31_authority,
        materializer,
        checked={
            "valid": True,
            "event_watermark": m31_observed.get("target_event_watermark"),
            "projection_cid": m31_observed.get("projection_cid"),
            "generation_28_29_restart_rows_verified": m31_observed.get(
                "generation_28_29_restart_rows_verified"
            ),
            "prior_event_prefix_verified": checked.get("prior_event_prefix_verified"),
            "full_event_and_evidence_body_verified": m31_observed.get(
                "full_event_and_evidence_body_verified"
            ),
            "receipt": m31_observed,
        },
    )
    path = (REPO_ROOT / _M32_STORE_ID).resolve().parent / (
        "m32-source-successor-receipt.json"
    )
    try:
        observed, _ = materializer._load_nofollow_json(
            path, root=REPO_ROOT, noun="M32 source successor receipt"
        )
    except Exception as exc:
        raise OperatorError("M32 exact source successor receipt is unavailable") from exc
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    if (
        claimed != materializer._identity(unhashed)
        or checked.get("valid") is not True
        or checked.get("event_watermark") != _M32_TARGET_EVENT_WATERMARK
        or checked.get("projection_cid") != _M32_TARGET_PROJECTION_CID
        or checked.get("prior_event_prefix_verified") is not True
        or checked.get("full_event_and_evidence_body_verified") is not True
        or checked.get("receipt") != observed
        or observed.get("migration_revision") != "SAWM-R2-M32"
        or observed.get(f"{key}_cid") != materializer._identity(expected_authority)
        or observed.get("target_generation") != _M32_GENERATION
        or observed.get("target_event_watermark") != _M32_TARGET_EVENT_WATERMARK
        or observed.get("projection_cid") != _M32_TARGET_PROJECTION_CID
        or observed.get("same_live_generation_29_owner_verified") is not True
        or observed.get("queried_and_mutated_through_live_quack_only") is not True
        or observed.get("direct_authoritative_file_opened") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M32 exact source successor receipt differs")
    return MappingProxyType(
        {
            **dict(m31_marker),
            **dict(observed),
            "prior_final_pair_receipt_cid": m31_marker[
                "prior_final_pair_receipt_cid"
            ],
            "m29_source_successor_receipt_cid": m31_marker[
                "m29_source_successor_receipt_cid"
            ],
            "m30_source_successor_receipt_cid": m31_marker[
                "m30_source_successor_receipt_cid"
            ],
            "m31_source_successor_receipt_cid": m31_claimed,
            "m32_source_successor_receipt_cid": claimed,
            "source_successor_receipt_cid": claimed,
            "source_successor_receipt_verified": True,
            "source_successor_chain": {
                **dict(m31_marker["source_successor_chain"]),
                "m32_source_successor_receipt_cid": claimed,
            },
            "final_pair_commit_marker_verified": False,
        }
    )


def _require_m31_source_successor_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M31's exact live generation-29/event-282 receipt."""

    key = "detached_coordinator_pid_recovery_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    expected_authority = (
        materializer._expected_m31_detached_coordinator_pid_recovery_authority()
    )
    source_chain = expected_authority.get("source_chain", {})
    if (
        dict(authority) != expected_authority
        or config.get(key) != expected_authority
        or source_chain.get("initial_control_commit")
        != _M31_INITIAL_CONTROL_COMMIT
        or source_chain.get("initial_control_tree") != _M31_INITIAL_CONTROL_TREE
    ):
        raise OperatorError("M31 detached-coordinator authority differs")
    if checked is None:
        raise OperatorError("M31 marker requires exact live materializer verification")
    m30_authority = (
        materializer._expected_m30_stopped_owner_restart_source_seal_authority()
    )
    m30_path = (REPO_ROOT / _M31_STORE_ID).resolve().parent / (
        "m30-source-successor-receipt.json"
    )
    try:
        m30_observed, m30_sha256 = materializer._load_nofollow_json(
            m30_path, root=REPO_ROOT, noun="M31 preserved M30 source receipt"
        )
    except Exception as exc:
        raise OperatorError("M31 preserved M30 source receipt is unavailable") from exc
    m30_unhashed = dict(m30_observed)
    m30_claimed = str(m30_unhashed.pop("receipt_cid", ""))
    m31_prior = expected_authority["prior_authority"]
    if (
        m30_sha256 != m31_prior["m30_receipt_sha256"]
        or m30_claimed != m31_prior["m30_receipt_cid"]
        or m30_claimed != materializer._identity(m30_unhashed)
    ):
        raise OperatorError("M31 preserved M30 source receipt differs")
    m30_marker = _require_m30_source_successor_marker(
        config,
        m30_authority,
        materializer,
        checked={
            "valid": True,
            "event_watermark": m30_observed.get("target_event_watermark"),
            "projection_cid": m30_observed.get("projection_cid"),
            "generation_27_28_restart_rows_verified": m30_observed.get(
                "generation_27_28_restart_rows_verified"
            ),
            # The M31 verifier rehashed the complete immutable prefix through
            # event 281 before this historical receipt is admitted.
            "prior_event_prefix_verified": checked.get(
                "prior_event_prefix_verified"
            ),
            "full_event_and_evidence_body_verified": m30_observed.get(
                "full_event_and_evidence_body_verified"
            ),
            "receipt": m30_observed,
        },
    )
    path = (REPO_ROOT / _M31_STORE_ID).resolve().parent / (
        "m31-source-successor-receipt.json"
    )
    try:
        observed, _ = materializer._load_nofollow_json(
            path, root=REPO_ROOT, noun="M31 source successor receipt"
        )
    except Exception as exc:
        raise OperatorError("M31 exact source successor receipt is unavailable") from exc
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    if (
        claimed != materializer._identity(unhashed)
        or checked.get("valid") is not True
        or checked.get("event_watermark") != _M31_TARGET_EVENT_WATERMARK
        or checked.get("projection_cid") != _M31_TARGET_PROJECTION_CID
        or checked.get("generation_28_29_restart_rows_verified") is not True
        or checked.get("prior_event_prefix_verified") is not True
        or checked.get("full_event_and_evidence_body_verified") is not True
        or checked.get("receipt") != observed
        or observed.get("migration_revision") != "SAWM-R2-M31"
        or observed.get(f"{key}_cid") != materializer._identity(expected_authority)
        or observed.get("target_generation") != _M31_GENERATION
        or observed.get("target_event_watermark") != _M31_TARGET_EVENT_WATERMARK
        or observed.get("projection_cid") != _M31_TARGET_PROJECTION_CID
        or observed.get("queried_and_mutated_through_live_quack_only") is not True
        or observed.get("direct_authoritative_file_opened") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M31 exact source successor receipt differs")
    return MappingProxyType(
        {
            **dict(m30_marker),
            **dict(observed),
            "prior_final_pair_receipt_cid": m30_marker[
                "prior_final_pair_receipt_cid"
            ],
            "m29_source_successor_receipt_cid": m30_marker[
                "m29_source_successor_receipt_cid"
            ],
            "m30_source_successor_receipt_cid": m30_claimed,
            "m31_source_successor_receipt_cid": claimed,
            "source_successor_receipt_cid": claimed,
            "source_successor_receipt_verified": True,
            "source_successor_chain": {
                "m27_final_pair_receipt_cid": m30_marker[
                    "prior_final_pair_receipt_cid"
                ],
                "m29_source_successor_receipt_cid": m30_marker[
                    "m29_source_successor_receipt_cid"
                ],
                "m30_source_successor_receipt_cid": m30_claimed,
                "m31_source_successor_receipt_cid": claimed,
            },
            "final_pair_commit_marker_verified": False,
        }
    )


def _require_m30_source_successor_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M30's exact live generation-28 source-seal receipt."""

    key = "stopped_owner_restart_source_seal_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    expected_authority = (
        materializer._expected_m30_stopped_owner_restart_source_seal_authority()
    )
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M30 stopped-owner restart authority differs")
    if checked is None:
        raise OperatorError("M30 marker requires exact live materializer verification")
    m29_authority = (
        materializer._expected_m29_committed_evidence_verification_authority()
    )
    m29_path = (REPO_ROOT / _M30_STORE_ID).resolve().parent / (
        "m29-source-successor-receipt.json"
    )
    try:
        m29_observed, m29_sha256 = materializer._load_nofollow_json(
            m29_path, root=REPO_ROOT, noun="M30 preserved M29 source receipt"
        )
    except Exception as exc:
        raise OperatorError("M30 preserved M29 source receipt is unavailable") from exc
    if m29_sha256 != expected_authority["prior_authority"]["m29_receipt_sha256"]:
        raise OperatorError("M30 preserved M29 source receipt bytes differ")
    m29_marker = _require_m29_source_successor_marker(
        config,
        m29_authority,
        materializer,
        checked={
            "valid": True,
            "event_watermark": m29_observed.get("target_event_watermark"),
            "projection_cid": m29_observed.get("projection_cid"),
            "failed_m28_post_append_attempt_verified": m29_observed.get(
                "failed_m28_post_append_attempt_verified"
            ),
            "full_event_and_evidence_body_verified": m29_observed.get(
                "full_event_and_evidence_body_verified"
            ),
            "generation_26_27_restart_rows_verified": m29_observed.get(
                "generation_26_27_restart_rows_verified"
            ),
            "receipt": m29_observed,
        },
    )
    path = (REPO_ROOT / _M30_STORE_ID).resolve().parent / (
        "m30-source-successor-receipt.json"
    )
    try:
        observed, _ = materializer._load_nofollow_json(
            path, root=REPO_ROOT, noun="M30 source successor receipt"
        )
    except Exception as exc:
        raise OperatorError("M30 exact source successor receipt is unavailable") from exc
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    if (
        claimed != materializer._identity(unhashed)
        or checked.get("valid") is not True
        or checked.get("event_watermark") != _M30_TARGET_EVENT_WATERMARK
        or checked.get("projection_cid") != _M30_TARGET_PROJECTION_CID
        or checked.get("generation_27_28_restart_rows_verified") is not True
        or checked.get("prior_event_prefix_verified") is not True
        or checked.get("full_event_and_evidence_body_verified") is not True
        or checked.get("receipt") != observed
        or observed.get("migration_revision") != "SAWM-R2-M30"
        or observed.get(f"{key}_cid") != materializer._identity(expected_authority)
        or observed.get("target_generation") != _M30_GENERATION
        or observed.get("target_event_watermark") != _M30_TARGET_EVENT_WATERMARK
        or observed.get("projection_cid") != _M30_TARGET_PROJECTION_CID
        or observed.get("queried_and_mutated_through_live_quack_only") is not True
        or observed.get("direct_authoritative_file_opened") is not False
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M30 exact source successor receipt differs")
    return MappingProxyType(
        {
            **dict(m29_marker),
            **dict(observed),
            "prior_final_pair_receipt_cid": m29_marker[
                "prior_final_pair_receipt_cid"
            ],
            "m29_source_successor_receipt_cid": m29_marker[
                "source_successor_receipt_cid"
            ],
            "m30_source_successor_receipt_cid": claimed,
            "source_successor_receipt_cid": claimed,
            "source_successor_receipt_verified": True,
            "source_successor_chain": {
                "m27_final_pair_receipt_cid": m29_marker[
                    "prior_final_pair_receipt_cid"
                ],
                "m29_source_successor_receipt_cid": m29_marker[
                    "source_successor_receipt_cid"
                ],
                "m30_source_successor_receipt_cid": claimed,
            },
            "final_pair_commit_marker_verified": False,
        }
    )


def _require_m29_source_successor_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M27's final pair plus M29's exact verified source receipt."""

    key = "committed_evidence_verification_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    expected_authority = (
        materializer._expected_m29_committed_evidence_verification_authority()
    )
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M29 committed-evidence authority differs")
    m27_authority = materializer._expected_m27_dead_owner_parallel_resume_authority()
    m27_marker = _require_m27_final_pair_marker(
        config,
        m27_authority,
        materializer,
        require_current_coordination_store=False,
    )
    prior_final_pair_cid = str(m27_marker.get("receipt_cid") or "")
    if (
        prior_final_pair_cid
        != expected_authority["prior_authority"]["m27_migration_receipt_cid"]
    ):
        raise OperatorError("M29 prior M27 final-pair receipt differs")
    try:
        materializer._assert_m29_m28_receipt_absent(REPO_ROOT, expected_authority)
        path = (REPO_ROOT / _M29_STORE_ID).resolve().parent / (
            "m29-source-successor-receipt.json"
        )
        observed, _ = materializer._load_nofollow_json(
            path, root=REPO_ROOT, noun="M29 source successor receipt"
        )
    except Exception as exc:
        raise OperatorError("M29 exact source successor receipt is unavailable") from exc
    if checked is None:
        raise OperatorError("M29 marker requires exact live materializer verification")
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    if (
        claimed != materializer._identity(unhashed)
        or checked.get("valid") is not True
        or checked.get("event_watermark") != _M29_TARGET_EVENT_WATERMARK
        or checked.get("projection_cid") != _M29_TARGET_PROJECTION_CID
        or checked.get("failed_m28_post_append_attempt_verified") is not True
        or checked.get("full_event_and_evidence_body_verified") is not True
        or checked.get("generation_26_27_restart_rows_verified") is not True
        or checked.get("receipt") != observed
        or observed.get("migration_revision") != "SAWM-R2-M29"
        or observed.get(
            "committed_evidence_verification_successor_materialization_cid"
        )
        != materializer._identity(expected_authority)
        or observed.get("target_generation") != _M29_GENERATION
        or observed.get("target_event_watermark") != _M29_TARGET_EVENT_WATERMARK
        or observed.get("projection_cid") != _M29_TARGET_PROJECTION_CID
        or observed.get("failed_m28_post_append_attempt_verified") is not True
        or observed.get("m28_receipt_created_or_rewritten") is not False
        or observed.get("full_event_and_evidence_body_verified") is not True
        or observed.get("generation_26_27_restart_rows_verified") is not True
        or observed.get("worker_self_approval") is not False
    ):
        raise OperatorError("M29 exact source successor receipt differs")
    return MappingProxyType(
        {
            **dict(observed),
            "prior_final_pair_receipt_cid": prior_final_pair_cid,
            "prior_final_pair_commit_marker_verified": True,
            "source_successor_receipt_cid": claimed,
            "source_successor_receipt_verified": True,
            "final_pair_commit_marker_verified": False,
        }
    )


def _require_m28_source_successor_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M27's immutable pair marker plus M28's adjacent source seal."""

    key = "live_claim_admission_recovery_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    expected_authority = (
        materializer._expected_m28_live_claim_admission_recovery_authority()
    )
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M28 live-claim recovery authority differs")
    m27_authority = materializer._expected_m27_dead_owner_parallel_resume_authority()
    m27_marker = _require_m27_final_pair_marker(
        config,
        m27_authority,
        materializer,
        require_current_coordination_store=False,
    )
    if (
        m27_marker.get("receipt_cid")
        != expected_authority["prior_authority"]["migration_receipt_cid"]
    ):
        raise OperatorError("M28 prior M27 final-pair receipt differs")
    path = (REPO_ROOT / _M28_STORE_ID).resolve().parent / (
        "m28-source-successor-receipt.json"
    )
    try:
        observed, _ = materializer._load_nofollow_json(
            path, root=REPO_ROOT, noun="M28 source successor receipt"
        )
    except Exception as exc:
        raise OperatorError("M28 source successor receipt is unavailable") from exc
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    exact = {
        "schema": "sawm/non-authoritative-live-source-successor-receipt@1",
        "authoritative": False,
        "control_database_is_authority": True,
        "receipt_is_final_pair_commit_marker": False,
        "receipt_is_evidence_source_seal_marker": True,
        "migration_revision": "SAWM-R2-M28",
        f"{key}_cid": materializer._identity(expected_authority),
        "database_path": _M28_STORE_ID,
        "coordination_path": _M28_COORDINATION_STORE_ID,
        "target_generation": _M28_GENERATION,
        "target_plan_revision": _M28_TARGET_PLAN_REVISION,
        "target_event_watermark": _M28_TARGET_EVENT_WATERMARK,
        "projection_cid": _M28_TARGET_PROJECTION_CID,
        "coordination_projection_digest": expected_authority[
            "prior_authority"
        ]["coordination_projection_digest"],
        "coordination_event_count": expected_authority["prior_authority"][
            "coordination_event_count"
        ],
        "generation_bearing_owner_restart_verified": True,
        "prior_owner_generation": 26,
        "live_owner_generation": _M28_GENERATION,
        "queried_and_mutated_through_live_quack_only": True,
        "direct_authoritative_file_opened": False,
        "plan_revision_changes": 0,
        "evidence_node_changes": 1,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "coordination_semantic_changes": 0,
        "sidecars_preserved": True,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    if (
        claimed != materializer._identity(unhashed)
        or any(observed.get(name) != value for name, value in exact.items())
    ):
        raise OperatorError("M28 source successor receipt differs")
    if checked is not None and (
        checked.get("valid") is not True
        or checked.get("event_watermark") != _M28_TARGET_EVENT_WATERMARK
        or checked.get("projection_cid") != _M28_TARGET_PROJECTION_CID
        or checked.get("coordination_projection_digest")
        != expected_authority["prior_authority"][
            "coordination_projection_digest"
        ]
        or checked.get("coordination_event_count")
        != expected_authority["prior_authority"]["coordination_event_count"]
        or checked.get("generation_bearing_owner_restart_verified") is not True
        or checked.get("prior_owner_generation") != 26
        or checked.get("live_owner_generation") != _M28_GENERATION
        or checked.get("receipt") != observed
    ):
        raise OperatorError("M28 materializer check differs from source receipt")
    return MappingProxyType(
        {
            **dict(m27_marker),
            **dict(observed),
            "prior_final_pair_receipt_cid": m27_marker["receipt_cid"],
            "source_successor_receipt_cid": observed["receipt_cid"],
        }
    )


def _require_m27_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
    require_current_coordination_store: bool = True,
) -> Mapping[str, Any]:
    """Require M27's receipt-last dead-owner recovery marker."""

    key = "dead_owner_parallel_resume_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = (
            materializer._expected_m27_dead_owner_parallel_resume_authority()
        )
    except Exception as exc:
        raise OperatorError("M27 exact recovery authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M27 dead-owner parallel-resume authority differs")
    control = (REPO_ROOT / _M27_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M27_COORDINATION_STORE_ID).resolve()
    try:
        materializer._assert_m27_no_staging_or_pending(control)
        observed, _ = materializer._load_nofollow_json(
            control.parent / "migration-receipt.json",
            root=REPO_ROOT,
            noun="M27 final pair marker",
        )
        coordination_hash = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M27 coordination store",
                required_link_count=1,
            )
            if require_current_coordination_store
            else None
        )
    except Exception as exc:
        raise OperatorError("M27 final pair marker is unavailable") from exc
    if not isinstance(observed, Mapping):
        raise OperatorError("M27 final pair marker is malformed")
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    exact = {
        "schema": "sawm/non-authoritative-migration-receipt@25",
        "authoritative": False,
        "receipt_is_final_pair_commit_marker": True,
        "migration_revision": "SAWM-R2-M27",
        f"{key}_cid": materializer._identity(expected_authority),
        "database_path": _M27_STORE_ID,
        "coordination_path": _M27_COORDINATION_STORE_ID,
        "target_runtime_root": str(Path(_M27_STORE_ID).parent),
        "target_generation": _M27_GENERATION,
        "target_quack_port": _M27_TARGET_QUACK_PORT,
        "target_plan_revision": _M27_TARGET_PLAN_REVISION,
        "target_event_watermark": _M27_TARGET_EVENT_WATERMARK,
        "task_revision_changes": 2,
        "task_status_changes": 2,
        "orphan_claim_expirations": 4,
        "coordination_semantic_changes": 4,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "worker_self_approval": False,
    }
    if (
        claimed != materializer._identity(unhashed)
        or any(observed.get(name) != value for name, value in exact.items())
        # Starting the Quack owner appends its state-server identity to the
        # control authority.  The receipt still pins the pre-start pair, while
        # live identity/projection checks below qualify the post-start head.
        or re.fullmatch(
            r"[0-9a-f]{64}", str(observed.get("control_store_sha256") or "")
        )
        is None
        or int(observed.get("control_store_size") or 0) <= 0
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(observed.get("coordination_store_sha256") or ""),
        )
        is None
        or int(observed.get("coordination_store_size") or 0) <= 0
        or (
            coordination_hash is not None
            and (
                observed.get("coordination_store_sha256")
                != coordination_hash[0]
                or observed.get("coordination_store_size")
                != coordination_hash[1]
            )
        )
        or observed.get("coordination_projection_digest")
        != expected_authority["target_coordination_projection_digest"]
        or observed.get("coordination_event_count") != 2_586
        or set(observed.get("interrupted_claims", {}))
        != {"SAWM-006", "SAWM-008", "SAWM-012", "SAWM-015"}
        or set(observed.get("task_rearms", {}))
        != {"SAWM-006", "SAWM-012"}
    ):
        raise OperatorError("M27 materialized final pair marker differs")
    if checked is not None and (
        checked.get("valid") is not True
        or checked.get("database_path") != str(control)
        or checked.get("coordination_path") != str(coordination)
        or checked.get("event_watermark") != _M27_TARGET_EVENT_WATERMARK
        or checked.get("receipt") != observed
    ):
        raise OperatorError("M27 materializer check differs from its final marker")
    return MappingProxyType(dict(observed))


def _require_m26_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M26's exact receipt-last stopped-runtime successor marker."""

    key = "automatic_stall_recovery_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = (
            materializer._expected_m26_automatic_stall_recovery_authority()
        )
    except Exception as exc:
        raise OperatorError("M26 exact recovery authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M26 automatic-stall-recovery authority differs")
    control = (REPO_ROOT / _M26_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M26_COORDINATION_STORE_ID).resolve()
    try:
        materializer._assert_m26_no_staging_or_pending(control)
        observed, _ = materializer._load_nofollow_json(
            control.parent / "migration-receipt.json",
            root=REPO_ROOT,
            noun="M26 final pair marker",
        )
        coordination_hash = materializer._stable_regular_sha256(
            coordination,
            root=REPO_ROOT,
            noun="materialized M26 coordination store",
            required_link_count=1,
        )
    except Exception as exc:
        raise OperatorError("M26 final pair marker is unavailable") from exc
    if not isinstance(observed, Mapping):
        raise OperatorError("M26 final pair marker is malformed")
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    exact = {
        "schema": "sawm/non-authoritative-migration-receipt@24",
        "authoritative": False,
        "receipt_is_final_pair_commit_marker": True,
        "migration_revision": "SAWM-R2-M26",
        "automatic_stall_recovery_successor_materialization_cid": (
            materializer._identity(expected_authority)
        ),
        "database_path": _M26_STORE_ID,
        "coordination_path": _M26_COORDINATION_STORE_ID,
        "target_runtime_root": expected_authority["target_runtime_root"],
        "target_generation": 25,
        "target_quack_port": 24_069,
        "target_plan_revision": 27,
        "target_event_watermark": 262,
        "task_revision_changes": 2,
        "task_status_changes": 2,
        "orphan_claim_expirations": 2,
        "logical_completion_removals_for_rearm": 2,
        "coordination_semantic_changes": 4,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "worker_self_approval": False,
    }
    if (
        claimed != materializer._identity(unhashed)
        or any(observed.get(name) != value for name, value in exact.items())
        or re.fullmatch(
            r"[0-9a-f]{64}", str(observed.get("control_store_sha256") or "")
        )
        is None
        or int(observed.get("control_store_size") or 0) <= 0
        or observed.get("coordination_store_sha256") != coordination_hash[0]
        or observed.get("coordination_store_size") != coordination_hash[1]
        or observed.get("coordination_projection_digest")
        != expected_authority["target_coordination_projection_digest"]
        or observed.get("coordination_event_count") != 2_421
        or set(observed.get("orphan_claims", {})) != {"SAWM-006", "SAWM-015"}
        or set(observed.get("failure_receipts", {}))
        != {"SAWM-008", "SAWM-012"}
        or set(observed.get("task_rearms", {})) != {"SAWM-008", "SAWM-012"}
    ):
        raise OperatorError("M26 materialized final pair marker differs")
    if checked is not None:
        population = materializer.build_population(REPO_ROOT)
        validation_digest = materializer._identity(
            {
                "dependency": materializer._validator_report(
                    REPO_ROOT,
                    "scripts/validate_semantic_addressed_world_model_dependencies.py",
                ),
                "board": materializer._validator_report(
                    REPO_ROOT,
                    "scripts/validate_semantic_addressed_world_model_board.py",
                ),
                "program_definition_cid": population["program_definition_cid"],
            }
        )
        expected = materializer._expected_m26_migration_receipt(
            REPO_ROOT,
            control,
            coordination,
            population,
            checked,
            validation_digest,
        )
        if (
            checked.get("valid") is not True
            or checked.get("event_watermark") != 262
            or checked.get("receipt") != observed
            or expected != observed
        ):
            raise OperatorError(
                "M26 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _m25_receipt_authority_fields(
    authority: Mapping[str, Any],
    materializer: Any,
) -> Mapping[str, Any]:
    """Return M25 receipt fields fixed by its sealed successor authority."""

    fields: dict[str, Any] = {
        "schema": "sawm/non-authoritative-migration-receipt@23",
        "authoritative": False,
        "control_database_is_authority": True,
        "coordination_database_is_authority": True,
        "receipt_is_final_pair_commit_marker": True,
        "migration_revision": "SAWM-R2-M25",
        "native_duckdb_preload_successor_materialization_cid": (
            materializer._identity(dict(authority))
        ),
        "database_path": _M25_STORE_ID,
        "coordination_path": _M25_COORDINATION_STORE_ID,
        "prior_database_path": authority["prior_store_id"],
        "prior_coordination_path": authority["prior_coordination_store_id"],
        "target_runtime_root": authority["target_runtime_root"],
        "target_generation": authority["target_generation"],
        "target_quack_port": authority["target_quack_port"],
        "target_plan_revision": authority["target_plan_revision"],
        "target_event_watermark": authority["target_event_watermark"],
    }
    # The materializer's closed key set decides which authority facts are
    # receipt-bearing.  Exact name matches must retain the exact sealed value;
    # this keeps new failure-anchor fields fail-closed without duplicating a
    # second schema in the operator.
    receipt_keys = set(materializer._M25_RECEIPT_KEYS)
    for name in (receipt_keys & set(authority)) - set(fields):
        fields[name] = authority[name]
    mapped = {
        "semantic_authority_digest": "target_semantic_authority_digest",
        "frozen_base_authority_digest": "target_frozen_base_authority_digest",
        "coordination_projection_digest": (
            "target_coordination_projection_digest"
        ),
        "coordination_event_count": "target_coordination_event_count",
        "plan_projection_cid": "prior_projection_cid",
        "migration_projection_cid": "target_projection_cid",
        "projection_cid": "target_projection_cid",
    }
    for receipt_name, authority_name in mapped.items():
        if receipt_name in receipt_keys and authority_name in authority:
            fields[receipt_name] = authority[authority_name]
    return MappingProxyType(fields)


def _require_m25_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M25's receipt-last marker and preserved failed-start anchor."""

    key = "native_duckdb_preload_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = (
            materializer._expected_m25_native_duckdb_preload_authority()
        )
    except Exception as exc:
        raise OperatorError("M25 exact native-DuckDB authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M25 native-DuckDB preload authority differs")

    control = (REPO_ROOT / _M25_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M25_COORDINATION_STORE_ID).resolve()
    try:
        materializer._assert_m25_no_staging_or_pending(control)
        observed, _ = materializer._load_nofollow_json(
            control.parent / "migration-receipt.json",
            root=REPO_ROOT,
            noun="M25 final pair marker",
        )
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M25 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M25 final pair marker is unavailable") from exc

    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    try:
        migration_digest = materializer._identity(
            materializer._m25_migration_body(
                population,
                config,
                validation_digest,
            )
        )
    except Exception as exc:
        raise OperatorError("M25 exact migration body is unavailable") from exc

    if not isinstance(observed, Mapping):
        raise OperatorError("M25 final pair marker is malformed")
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    digest_names = (
        "control_store_sha256",
        "coordination_store_sha256",
        "semantic_authority_digest",
        "frozen_base_authority_digest",
        "append_surface_digest",
        "catalog_digest",
        "coordination_projection_digest",
    )
    exact_flags = {
        "runtime_root_changed": True,
        "source_binding_changed": True,
        "control_and_coordination_bases_copied": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "prior_failed_start_artifacts_preserved": True,
        "prior_materialization_receipt_precedes_failed_start_checkpoint": True,
        "prior_read_replica_copied": False,
        "prior_owner_status_copied": False,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
        "coordination_semantic_changes": 0,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "goal_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "runtime_state_copied": False,
        "worktrees_copied": False,
        "logs_copied": False,
        "worker_self_approval": False,
    }
    receipt_keys = set(materializer._M25_RECEIPT_KEYS)
    authority_fields = _m25_receipt_authority_fields(
        expected_authority,
        materializer,
    )
    if (
        set(observed) != receipt_keys
        or claimed != materializer._identity(unhashed)
        or any(
            observed.get(name) != value
            for name, value in authority_fields.items()
        )
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or observed.get("semantic_authority_digest")
        != expected_authority["prior_semantic_authority_digest"]
        or observed.get("frozen_base_authority_digest")
        != expected_authority["prior_frozen_base_authority_digest"]
        or observed.get("catalog_digest")
        != expected_authority["prior_catalog_digest"]
        or type(observed.get("event_watermark")) is not int
        or observed.get("event_watermark") != _M25_TARGET_EVENT_WATERMARK
        or type(observed.get("migration_event_watermark")) is not int
        or observed.get("migration_event_watermark")
        != _M25_TARGET_EVENT_WATERMARK
        or not all(
            str(observed.get(name) or "")
            for name in (
                "migration_evidence_id",
                "plan_migration_event_id",
                "migration_evidence_event_id",
                "projection_cid",
            )
        )
        or not all(
            re.fullmatch(
                r"(?:sha256:)?[0-9a-f]{64}",
                str(observed.get(name) or ""),
            )
            for name in digest_names
        )
        or type(observed.get("control_store_size")) is not int
        or observed.get("control_store_size", 0) <= 0
        or observed.get("coordination_store_sha256") != coordination_sha256
        or type(observed.get("coordination_store_size")) is not int
        or observed.get("coordination_store_size") != coordination_size
        or observed.get("failed_quack_start_cid")
        != expected_authority["failed_quack_start_cid"]
        or any(
            observed.get(name) != value
            for name, value in exact_flags.items()
            if name in receipt_keys
        )
        or config.get("runtime_paths", {}).get("root")
        != expected_authority["target_runtime_root"]
    ):
        raise OperatorError("M25 materialized final pair marker differs")

    if checked is not None:
        try:
            expected = materializer._expected_m25_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError(
                "M25 exact materializer report is unavailable"
            ) from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or type(checked.get("event_watermark")) is not int
            or checked.get("event_watermark")
            != _M25_TARGET_EVENT_WATERMARK
            or checked.get("receipt") != observed
            or expected != observed
        ):
            raise OperatorError(
                "M25 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _m24_receipt_authority_fields(
    authority: Mapping[str, Any],
    materializer: Any,
) -> Mapping[str, Any]:
    """Return every M24 receipt field fixed by the sealed operator authority."""

    fields: dict[str, Any] = {
        "schema": "sawm/non-authoritative-migration-receipt@22",
        "authoritative": False,
        "control_database_is_authority": True,
        "coordination_database_is_authority": True,
        "receipt_is_final_pair_commit_marker": True,
        "migration_revision": "SAWM-R2-M24",
        "multi_lane_sidecar_reopen_successor_materialization_cid": (
            materializer._identity(dict(authority))
        ),
        "database_path": _M24_STORE_ID,
        "coordination_path": _M24_COORDINATION_STORE_ID,
        "prior_database_path": authority["prior_store_id"],
        "prior_coordination_path": authority["prior_coordination_store_id"],
        "ordinary_source_changes": 0,
        "target_runtime_root": authority["target_runtime_root"],
        "target_generation": authority["target_generation"],
        "target_quack_port": authority["target_quack_port"],
        "target_plan_revision": authority["target_plan_revision"],
        "target_event_watermark": authority["target_event_watermark"],
        "runtime_root_changed": True,
        "source_binding_changed": True,
        "control_and_coordination_bases_copied": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "prior_runtime_artifacts_preserved": True,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
        "coordination_semantic_changes": 2,
        "task_revision_changes": 2,
        "task_status_changes": 2,
        "goal_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "prior_execution_sidecars_copied": False,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "runtime_state_copied": False,
        "worktrees_copied": False,
        "logs_copied": False,
        "worker_self_approval": False,
    }
    for name in (
        "prior_control_store_sha256",
        "prior_control_store_size",
        "prior_coordination_store_sha256",
        "prior_coordination_store_size",
        "prior_read_replica_path",
        "prior_read_replica_sha256",
        "prior_read_replica_size",
        "prior_stopped_status_projection_path",
        "prior_stopped_status_projection_sha256",
        "prior_stopped_status_projection_size",
        "prior_migration_receipt_path",
        "prior_migration_receipt_sha256",
        "prior_migration_receipt_size",
        "prior_migration_receipt_cid",
        "prior_event_watermark",
        "prior_event_prefix_sha256",
        "prior_projection_cid",
        "prior_semantic_authority_digest",
        "prior_frozen_base_authority_digest",
        "prior_append_surface_digest",
        "prior_catalog_digest",
        "prior_validation_digest",
        "prior_source_binding_cid",
        "prior_coordination_projection_digest",
        "prior_coordination_event_count",
        "prior_database_uuid",
        "prior_server_id",
        "prior_process_birth_id",
        "prior_startup_epoch",
        "prior_started_at",
        "prior_stopped_at",
        "prior_generation",
        "prior_active_claim_count",
        "prior_active_attempt_count",
        "prior_active_lease_count",
        "prior_lane_runtime_anchors",
        "interrupted_claim_evidence",
        "failure_receipt",
        "task_rearm",
        "lane_contract",
        "repair_source_first_commit",
        "repair_source_first_tree",
        "repair_source_commit",
        "repair_source_tree",
        "repair_source_blobs",
        "accepted_control_plane_repair",
    ):
        fields[name] = authority[name]
    return MappingProxyType(fields)


def _require_m24_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M24's receipt-last marker without admitting its own output."""

    key = "multi_lane_sidecar_reopen_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = materializer._expected_m24_sidecar_reopen_authority()
    except Exception as exc:
        raise OperatorError("M24 exact authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M24 sidecar-reopen authority differs")

    control = (REPO_ROOT / _M24_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M24_COORDINATION_STORE_ID).resolve()
    try:
        materializer._assert_m24_no_staging_or_pending(control)
        observed, _ = materializer._load_nofollow_json(
            control.parent / "migration-receipt.json",
            root=REPO_ROOT,
            noun="M24 final pair marker",
        )
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M24 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M24 final pair marker is unavailable") from exc

    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    try:
        migration_digest = materializer._identity(
            materializer._m24_migration_body(
                population,
                config,
                validation_digest,
            )
        )
    except Exception as exc:
        raise OperatorError("M24 exact migration body is unavailable") from exc
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    digest_names = (
        "control_store_sha256",
        "coordination_store_sha256",
        "semantic_authority_digest",
        "frozen_base_authority_digest",
        "append_surface_digest",
        "catalog_digest",
        "coordination_projection_digest",
    )
    if (
        set(observed) != materializer._M24_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or any(
            observed.get(name) != value
            for name, value in _m24_receipt_authority_fields(
                expected_authority,
                materializer,
            ).items()
        )
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or observed.get("event_watermark") != _M24_TARGET_EVENT_WATERMARK
        or not all(
            str(observed.get(name) or "")
            for name in (
                "migration_evidence_id",
                "plan_migration_event_id",
                "migration_evidence_event_id",
                "projection_cid",
                "coordination_settlement_event_id",
                "coordination_rearm_event_id",
                "coordination_rearm_id",
            )
        )
        or not all(
            re.fullmatch(
                r"(?:sha256:)?[0-9a-f]{64}",
                str(observed.get(name) or ""),
            )
            for name in digest_names
        )
        or not isinstance(observed.get("task_rearm_event_ids"), Mapping)
        or set(observed["task_rearm_event_ids"]) != {"failure", "retry"}
        or not isinstance(observed.get("task_rearm_receipt_cids"), Mapping)
        or set(observed["task_rearm_receipt_cids"]) != {"failure", "retry"}
        or observed.get("coordination_store_sha256") != coordination_sha256
        or observed.get("coordination_store_size") != coordination_size
        or observed.get("coordination_projection_digest")
        != expected_authority["target_coordination_projection_digest"]
        or observed.get("coordination_event_count")
        != expected_authority["target_coordination_event_count"]
        or config.get("runtime_paths", {}).get("root")
        != expected_authority["target_runtime_root"]
    ):
        raise OperatorError("M24 materialized final pair marker differs")
    if checked is not None:
        try:
            expected = materializer._expected_m24_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError(
                "M24 exact materializer report is unavailable"
            ) from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or checked.get("event_watermark") != _M24_TARGET_EVENT_WATERMARK
            or checked.get("receipt") != observed
            or expected != observed
        ):
            raise OperatorError(
                "M24 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _m23_receipt_authority_fields(
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return every M23 receipt field fixed directly by sealed authority."""

    fields: dict[str, Any] = {
        "schema": "sawm/non-authoritative-migration-receipt@21",
        "authoritative": False,
        "control_database_is_authority": True,
        "coordination_database_is_authority": True,
        "receipt_is_final_pair_commit_marker": True,
        "migration_revision": "SAWM-R2-M23",
        "multi_lane_successor_materialization_cid": _materializer()._identity(
            dict(authority)
        ),
        "database_path": _M23_STORE_ID,
        "coordination_path": _M23_COORDINATION_STORE_ID,
        "prior_database_path": authority["prior_store_id"],
        "prior_coordination_path": authority["prior_coordination_store_id"],
        "ordinary_source_changes": 0,
        "target_runtime_root": authority["target_runtime_root"],
        "target_generation": authority["target_generation"],
        "target_quack_port": authority["target_quack_port"],
        "target_plan_revision": authority["target_plan_revision"],
        "target_event_watermark": authority["target_event_watermark"],
        "event_watermark": authority["target_event_watermark"],
        "runtime_root_changed": True,
        "source_binding_changed": True,
        "control_and_coordination_bases_copied": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "prior_lifecycle_artifacts_preserved": True,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
        "coordination_semantic_changes": 2,
        "task_revision_changes": 2,
        "task_status_changes": 2,
        "goal_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "worker_self_approval": False,
    }
    for name in (
        "prior_control_store_sha256",
        "prior_control_store_size",
        "prior_coordination_store_sha256",
        "prior_coordination_store_size",
        "prior_read_replica_path",
        "prior_read_replica_sha256",
        "prior_read_replica_size",
        "prior_execution_sidecar_path",
        "prior_execution_sidecar_sha256",
        "prior_execution_sidecar_size",
        "prior_receipt_publication_control_store_sha256",
        "prior_receipt_publication_control_store_size",
        "prior_receipt_publication_coordination_store_sha256",
        "prior_receipt_publication_coordination_store_size",
        "prior_receipt_publication_projection_cid",
        "prior_receipt_publication_semantic_authority_digest",
        "prior_receipt_publication_frozen_base_authority_digest",
        "prior_receipt_publication_append_surface_digest",
        "prior_receipt_publication_coordination_projection_digest",
        "prior_receipt_publication_coordination_event_count",
        "prior_receipt_publication_event_watermark",
        "prior_receipt_publication_catalog_digest",
        "prior_stopped_status_projection_path",
        "prior_stopped_status_projection_sha256",
        "prior_stopped_status_projection_size",
        "prior_supervisor_status_path",
        "prior_supervisor_status_sha256",
        "prior_supervisor_status_size",
        "prior_event_watermark",
        "prior_event_prefix_sha256",
        "prior_projection_cid",
        "prior_semantic_authority_digest",
        "prior_frozen_base_authority_digest",
        "prior_append_surface_digest",
        "prior_catalog_digest",
        "prior_validation_digest",
        "prior_migration_receipt_path",
        "prior_migration_receipt_sha256",
        "prior_migration_receipt_size",
        "prior_migration_receipt_cid",
        "prior_source_binding_cid",
        "prior_database_uuid",
        "prior_server_id",
        "prior_process_birth_id",
        "prior_startup_epoch",
        "prior_started_at",
        "prior_stopped_at",
        "prior_generation",
        "prior_active_claim_count",
        "prior_active_attempt_count",
        "prior_active_lease_count",
        "interrupted_claim_evidence",
        "failure_receipt",
        "task_rearm",
        "lane_contract",
        "repair_source_commit",
        "repair_source_tree",
        "repair_source_blobs",
        "accepted_control_plane_repair",
    ):
        fields[name] = authority[name]
    return MappingProxyType(fields)


def _require_m23_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M23's receipt-last marker without admitting its own output."""

    key = "multi_lane_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = materializer._expected_m23_multi_lane_authority()
    except Exception as exc:
        raise OperatorError("M23 exact authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M23 multi-lane authority differs")

    control = (REPO_ROOT / _M23_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M23_COORDINATION_STORE_ID).resolve()
    try:
        materializer._assert_m23_no_staging_or_pending(control)
        observed, _ = materializer._load_nofollow_json(
            control.parent / "migration-receipt.json",
            root=REPO_ROOT,
            noun="M23 final pair marker",
        )
        coordination_sha256, coordination_size = materializer._stable_regular_sha256(
            coordination,
            root=REPO_ROOT,
            noun="materialized M23 coordination store",
            required_link_count=1,
        )
    except Exception as exc:
        raise OperatorError("M23 final pair marker is unavailable") from exc

    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    try:
        migration_digest = materializer._identity(
            materializer._m23_migration_body(population, config, validation_digest)
        )
    except Exception as exc:
        raise OperatorError("M23 exact migration body is unavailable") from exc

    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    digest_names = (
        "control_store_sha256",
        "coordination_store_sha256",
        "semantic_authority_digest",
        "frozen_base_authority_digest",
        "append_surface_digest",
        "catalog_digest",
        "coordination_projection_digest",
    )
    if (
        set(observed) != materializer._M23_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or any(
            observed.get(name) != value
            for name, value in _m23_receipt_authority_fields(authority).items()
        )
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or not all(
            str(observed.get(name) or "")
            for name in (
                "migration_evidence_id",
                "plan_migration_event_id",
                "migration_evidence_event_id",
                "projection_cid",
                "coordination_settlement_event_id",
                "coordination_rearm_event_id",
                "coordination_rearm_id",
            )
        )
        or not all(
            re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(observed.get(name) or ""))
            for name in digest_names
        )
        or not isinstance(observed.get("task_rearm_event_ids"), Mapping)
        or set(observed["task_rearm_event_ids"]) != {"failure", "retry"}
        or not isinstance(observed.get("task_rearm_receipt_cids"), Mapping)
        or set(observed["task_rearm_receipt_cids"]) != {"failure", "retry"}
        or observed.get("coordination_store_sha256") != coordination_sha256
        or observed.get("coordination_store_size") != coordination_size
        or config.get("runtime_paths", {}).get("root")
        != authority["target_runtime_root"]
    ):
        raise OperatorError("M23 materialized final pair marker differs")

    if checked is not None:
        try:
            expected = materializer._expected_m23_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError("M23 exact materializer report is unavailable") from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or checked.get("event_watermark") != _M23_TARGET_EVENT_WATERMARK
            or checked.get("receipt") != observed
            or expected != observed
        ):
            raise OperatorError(
                "M23 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _m22_receipt_authority_fields(
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return every static M22 receipt field bound by sealed authority."""

    key = "live_preflight_receipt_compatibility_successor_materialization"
    return MappingProxyType(
        {
            "schema": "sawm/non-authoritative-migration-receipt@20",
            "authoritative": False,
            "control_database_is_authority": True,
            "coordination_database_is_authority": True,
            "receipt_is_final_pair_commit_marker": True,
            "migration_revision": "SAWM-R2-M22",
            f"{key}_cid": _materializer()._identity(dict(authority)),
            "database_path": _M22_STORE_ID,
            "coordination_path": _M22_COORDINATION_STORE_ID,
            "semantic_authority_digest": authority[
                "target_semantic_authority_digest"
            ],
            "frozen_base_authority_digest": authority[
                "target_frozen_base_authority_digest"
            ],
            "catalog_digest": authority["prior_catalog_digest"],
            "coordination_projection_digest": authority[
                "target_coordination_projection_digest"
            ],
            "coordination_event_count": authority[
                "target_coordination_event_count"
            ],
            "prior_database_path": authority["prior_store_id"],
            "prior_coordination_path": authority[
                "prior_coordination_store_id"
            ],
            "prior_control_store_sha256": authority[
                "prior_control_store_sha256"
            ],
            "prior_control_store_size": authority["prior_control_store_size"],
            "prior_coordination_store_sha256": authority[
                "prior_coordination_store_sha256"
            ],
            "prior_coordination_store_size": authority[
                "prior_coordination_store_size"
            ],
            "prior_read_replica_path": authority["prior_read_replica_path"],
            "prior_read_replica_sha256": authority[
                "prior_read_replica_sha256"
            ],
            "prior_read_replica_size": authority["prior_read_replica_size"],
            "prior_execution_sidecar_path": authority[
                "prior_execution_sidecar_path"
            ],
            "prior_execution_sidecar_sha256": authority[
                "prior_execution_sidecar_sha256"
            ],
            "prior_execution_sidecar_size": authority[
                "prior_execution_sidecar_size"
            ],
            "prior_receipt_publication_control_store_sha256": authority[
                "prior_receipt_publication_control_store_sha256"
            ],
            "prior_stopped_status_projection_path": authority[
                "prior_stopped_status_projection_path"
            ],
            "prior_stopped_status_projection_sha256": authority[
                "prior_stopped_status_projection_sha256"
            ],
            "prior_stopped_status_projection_size": authority[
                "prior_stopped_status_projection_size"
            ],
            "prior_event_watermark": authority["prior_event_watermark"],
            "prior_event_prefix_sha256": authority[
                "prior_event_prefix_sha256"
            ],
            "prior_projection_cid": authority["prior_projection_cid"],
            "prior_semantic_authority_digest": authority[
                "prior_semantic_authority_digest"
            ],
            "prior_frozen_base_authority_digest": authority[
                "prior_frozen_base_authority_digest"
            ],
            "prior_append_surface_digest": authority[
                "prior_append_surface_digest"
            ],
            "prior_catalog_digest": authority["prior_catalog_digest"],
            "prior_validation_digest": authority["prior_validation_digest"],
            "prior_migration_receipt_path": authority[
                "prior_migration_receipt_path"
            ],
            "prior_migration_receipt_sha256": authority[
                "prior_migration_receipt_sha256"
            ],
            "prior_migration_receipt_size": authority[
                "prior_migration_receipt_size"
            ],
            "prior_migration_receipt_cid": authority[
                "prior_migration_receipt_cid"
            ],
            "prior_source_binding_cid": authority["prior_source_binding_cid"],
            "prior_database_uuid": authority["prior_database_uuid"],
            "prior_server_id": authority["prior_server_id"],
            "prior_process_birth_id": authority["prior_process_birth_id"],
            "prior_startup_epoch": authority["prior_startup_epoch"],
            "prior_started_at": authority["prior_started_at"],
            "prior_stopped_at": authority["prior_stopped_at"],
            "prior_generation": authority["prior_generation"],
            "legacy_completion_receipt_cids": authority[
                "legacy_completion_receipt_cids"
            ],
            "legacy_operational_validation_receipt_cids": authority[
                "legacy_operational_validation_receipt_cids"
            ],
            "legacy_completion_worker_field_required_absent": True,
            "observed_launch_blockers": authority[
                "observed_launch_blockers"
            ],
            "accepted_control_plane_repair": authority[
                "accepted_control_plane_repair"
            ],
            "ordinary_source_changes": 0,
            "target_runtime_root": authority["target_runtime_root"],
            "target_generation": authority["target_generation"],
            "target_quack_port": authority["target_quack_port"],
            "target_plan_revision": authority["target_plan_revision"],
            "target_event_watermark": authority["target_event_watermark"],
            "runtime_root_changed": True,
            "source_binding_changed": True,
            "control_and_coordination_bases_copied": True,
            "prior_control_store_mutated": False,
            "prior_coordination_store_mutated": False,
            "prior_lifecycle_artifacts_preserved": True,
            "plan_revision_changes": 1,
            "evidence_node_changes": 1,
            "coordination_semantic_changes": 0,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "goal_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
            "execution_sidecar_copied": False,
            "read_replica_sidecar_copied": False,
            "worker_self_approval": False,
        }
    )


def _m22_receipt_has_exact_authority_fields(
    receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> bool:
    """Reject a self-rehashed marker when any sealed M22 field changes."""

    return all(
        receipt.get(name) == value
        for name, value in _m22_receipt_authority_fields(authority).items()
    )


def _require_m22_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M22's receipt-last marker without opening its live control."""

    key = "live_preflight_receipt_compatibility_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = (
            materializer
            ._expected_m22_live_preflight_receipt_compatibility_authority()
        )
    except Exception as exc:
        raise OperatorError("M22 exact authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M22 receipt-compatibility authority differs")

    control = (REPO_ROOT / _M22_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M22_COORDINATION_STORE_ID).resolve()
    try:
        materializer._assert_m22_no_staging_or_pending(control)
        observed, _ = materializer._load_nofollow_json(
            control.parent / "migration-receipt.json",
            root=REPO_ROOT,
            noun="M22 final pair marker",
        )
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M22 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M22 final pair marker is unavailable") from exc

    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    try:
        migration_digest = materializer._identity(
            materializer._m22_migration_body(
                population, config, validation_digest
            )
        )
    except Exception as exc:
        raise OperatorError("M22 exact migration body is unavailable") from exc

    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    digest_names = (
        "control_store_sha256",
        "coordination_store_sha256",
        "semantic_authority_digest",
        "frozen_base_authority_digest",
        "append_surface_digest",
        "catalog_digest",
        "coordination_projection_digest",
    )
    if (
        set(observed) != materializer._M22_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or not _m22_receipt_has_exact_authority_fields(observed, authority)
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or not all(
            str(observed.get(name) or "")
            for name in (
                "migration_evidence_id",
                "plan_migration_event_id",
                "migration_evidence_event_id",
                "projection_cid",
            )
        )
        or not all(
            re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(observed.get(name) or ""))
            for name in digest_names
        )
        or int(observed.get("event_watermark") or 0)
        != _M22_TARGET_EVENT_WATERMARK
        or int(observed.get("control_store_size") or 0) <= 0
        or int(observed.get("control_store_size") or 0) > 2 * 1024 * 1024 * 1024
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or config.get("runtime_paths", {}).get("root")
        != authority["target_runtime_root"]
    ):
        raise OperatorError("M22 materialized final pair marker differs")

    if checked is not None:
        try:
            expected = materializer._expected_m22_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError("M22 exact materializer report is unavailable") from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or int(checked.get("event_watermark") or 0)
            != _M22_TARGET_EVENT_WATERMARK
            or checked.get("receipt") != observed
            or expected != observed
        ):
            raise OperatorError(
                "M22 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _m21_receipt_authority_fields(
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return every static M21 receipt field bound by sealed authority."""

    key = "generation_realization_successor_materialization"
    return MappingProxyType(
        {
            "schema": "sawm/non-authoritative-migration-receipt@19",
            "authoritative": False,
            "control_database_is_authority": True,
            "coordination_database_is_authority": True,
            "receipt_is_final_pair_commit_marker": True,
            "migration_revision": "SAWM-R2-M21",
            f"{key}_cid": _materializer()._identity(dict(authority)),
            "database_path": _M21_STORE_ID,
            "coordination_path": _M21_COORDINATION_STORE_ID,
            "semantic_authority_digest": authority[
                "target_semantic_authority_digest"
            ],
            "frozen_base_authority_digest": authority[
                "target_frozen_base_authority_digest"
            ],
            "catalog_digest": authority["prior_catalog_digest"],
            "coordination_projection_digest": authority[
                "target_coordination_projection_digest"
            ],
            "coordination_event_count": authority[
                "target_coordination_event_count"
            ],
            "prior_database_path": authority["prior_store_id"],
            "prior_coordination_path": authority[
                "prior_coordination_store_id"
            ],
            "prior_control_store_sha256": authority[
                "prior_control_store_sha256"
            ],
            "prior_control_store_size": authority["prior_control_store_size"],
            "prior_coordination_store_sha256": authority[
                "prior_coordination_store_sha256"
            ],
            "prior_coordination_store_size": authority[
                "prior_coordination_store_size"
            ],
            "prior_read_replica_path": authority["prior_read_replica_path"],
            "prior_read_replica_sha256": authority[
                "prior_read_replica_sha256"
            ],
            "prior_read_replica_size": authority["prior_read_replica_size"],
            "prior_stopped_status_projection_path": authority[
                "prior_stopped_status_projection_path"
            ],
            "prior_stopped_status_projection_sha256": authority[
                "prior_stopped_status_projection_sha256"
            ],
            "prior_stopped_status_projection_size": authority[
                "prior_stopped_status_projection_size"
            ],
            "prior_event_watermark": authority["prior_event_watermark"],
            "prior_event_prefix_sha256": authority[
                "prior_event_prefix_sha256"
            ],
            "prior_projection_cid": authority["prior_projection_cid"],
            "prior_semantic_authority_digest": authority[
                "prior_semantic_authority_digest"
            ],
            "prior_frozen_base_authority_digest": authority[
                "prior_frozen_base_authority_digest"
            ],
            "prior_append_surface_digest": authority[
                "prior_append_surface_digest"
            ],
            "prior_catalog_digest": authority["prior_catalog_digest"],
            "prior_validation_digest": authority["prior_validation_digest"],
            "prior_migration_receipt_path": authority[
                "prior_migration_receipt_path"
            ],
            "prior_migration_receipt_sha256": authority[
                "prior_migration_receipt_sha256"
            ],
            "prior_migration_receipt_size": authority[
                "prior_migration_receipt_size"
            ],
            "prior_migration_receipt_cid": authority[
                "prior_migration_receipt_cid"
            ],
            "prior_source_binding_cid": authority["prior_source_binding_cid"],
            "prior_database_uuid": authority["prior_database_uuid"],
            "prior_server_id": authority["prior_server_id"],
            "prior_process_birth_id": authority["prior_process_birth_id"],
            "prior_startup_epoch": authority["prior_startup_epoch"],
            "prior_started_at": authority["prior_started_at"],
            "prior_stopped_at": authority["prior_stopped_at"],
            "prior_generation": authority["prior_generation"],
            "prior_sealed_target_generation": authority[
                "prior_sealed_target_generation"
            ],
            "prior_realized_generation": authority["prior_realized_generation"],
            "generation_mismatch": authority["generation_mismatch"],
            "accepted_control_plane_repair": authority[
                "accepted_control_plane_repair"
            ],
            "ordinary_source_changes": 0,
            "target_runtime_root": authority["target_runtime_root"],
            "target_generation": authority["target_generation"],
            "target_quack_port": authority["target_quack_port"],
            "target_plan_revision": authority["target_plan_revision"],
            "target_event_watermark": authority["target_event_watermark"],
            "event_watermark": authority["target_event_watermark"],
            "runtime_root_changed": True,
            "source_binding_changed": True,
            "control_and_coordination_bases_copied": True,
            "prior_control_store_mutated": False,
            "prior_coordination_store_mutated": False,
            "prior_lifecycle_artifacts_preserved": True,
            "plan_revision_changes": 1,
            "evidence_node_changes": 1,
            "coordination_semantic_changes": 0,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "goal_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
            "execution_sidecar_copied": False,
            "read_replica_sidecar_copied": False,
            "worker_self_approval": False,
        }
    )


def _m21_receipt_has_exact_authority_fields(
    receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> bool:
    """Reject a self-rehashed marker when any sealed M21 field changes."""

    return all(
        receipt.get(name) == value
        for name, value in _m21_receipt_authority_fields(authority).items()
    )


def _require_m21_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M21's receipt-last marker without opening its live control."""

    key = "generation_realization_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = (
            materializer._expected_m21_generation_realization_authority()
        )
    except Exception as exc:
        raise OperatorError("M21 exact authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M21 generation-realization authority differs")

    control = (REPO_ROOT / _M21_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M21_COORDINATION_STORE_ID).resolve()
    try:
        observed, _ = materializer._load_nofollow_json(
            control.parent / "migration-receipt.json",
            root=REPO_ROOT,
            noun="M21 final pair marker",
        )
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M21 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M21 final pair marker is unavailable") from exc

    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    try:
        migration_digest = materializer._identity(
            materializer._m21_migration_body(
                population, config, validation_digest
            )
        )
    except Exception as exc:
        raise OperatorError("M21 exact migration body is unavailable") from exc

    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    if (
        set(observed) != materializer._M21_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or not _m21_receipt_has_exact_authority_fields(observed, authority)
        or observed.get("schema")
        != "sawm/non-authoritative-migration-receipt@19"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != "SAWM-R2-M21"
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or observed.get(f"{key}_cid")
        != materializer._identity(dict(authority))
        or observed.get("database_path") != _M21_STORE_ID
        or observed.get("coordination_path") != _M21_COORDINATION_STORE_ID
        or observed.get("target_runtime_root") != str(Path(_M21_STORE_ID).parent)
        or int(observed.get("target_generation") or 0) != _M21_GENERATION
        or int(observed.get("target_quack_port") or 0)
        != _M21_TARGET_QUACK_PORT
        or int(observed.get("target_plan_revision") or 0)
        != _M21_TARGET_PLAN_REVISION
        or int(observed.get("target_event_watermark") or 0)
        != _M21_TARGET_EVENT_WATERMARK
        or int(observed.get("event_watermark") or 0)
        != _M21_TARGET_EVENT_WATERMARK
        or observed.get("prior_control_store_sha256")
        != authority["prior_control_store_sha256"]
        or observed.get("prior_coordination_store_sha256")
        != authority["prior_coordination_store_sha256"]
        or observed.get("prior_migration_receipt_cid")
        != authority["prior_migration_receipt_cid"]
        or observed.get("prior_source_binding_cid")
        != authority["prior_source_binding_cid"]
        or observed.get("accepted_control_plane_repair")
        != authority["accepted_control_plane_repair"]
        or int(observed.get("ordinary_source_changes", -1)) != 0
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or observed.get("semantic_authority_digest")
        != authority["target_semantic_authority_digest"]
        or observed.get("frozen_base_authority_digest")
        != authority["target_frozen_base_authority_digest"]
        or observed.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or int(observed.get("coordination_event_count") or 0)
        != int(authority["target_coordination_event_count"])
        or any(
            observed.get(name) != value
            for name, value in {
                "runtime_root_changed": True,
                "source_binding_changed": True,
                "control_and_coordination_bases_copied": True,
                "prior_control_store_mutated": False,
                "prior_coordination_store_mutated": False,
                "plan_revision_changes": 1,
                "evidence_node_changes": 1,
                "coordination_semantic_changes": 0,
                "task_revision_changes": 0,
                "task_status_changes": 0,
                "goal_changes": 0,
                "accepted_definition_changes": 0,
                "accepted_completion_changes": 0,
                "implementation_provider_invocations": 0,
                "effect_claim_changes": 0,
                "implementation_commit_changes": 0,
                "merge_attempt_changes": 0,
                "execution_sidecar_copied": False,
                "read_replica_sidecar_copied": False,
                "worker_self_approval": False,
            }.items()
        )
    ):
        raise OperatorError("M21 materialized final pair marker differs")

    if checked is not None:
        try:
            expected = materializer._expected_m21_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError("M21 exact materializer report is unavailable") from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or int(checked.get("event_watermark") or 0)
            != _M21_TARGET_EVENT_WATERMARK
            or checked.get("receipt") != observed
            or expected != observed
        ):
            raise OperatorError(
                "M21 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _require_m20_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M20's receipt-last marker without opening its control file."""

    key = "test_isolation_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = materializer._expected_m20_test_isolation_authority()
    except Exception as exc:
        raise OperatorError("M20 exact authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M20 test-isolation authority differs")
    control = (REPO_ROOT / _M20_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M20_COORDINATION_STORE_ID).resolve()
    try:
        observed, _ = materializer._load_nofollow_json(
            control.parent / "migration-receipt.json",
            root=REPO_ROOT,
            noun="M20 final pair marker",
        )
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M20 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M20 final pair marker is unavailable") from exc
    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    try:
        migration_digest = materializer._identity(
            materializer._m20_migration_body(population, config, validation_digest)
        )
    except Exception as exc:
        raise OperatorError("M20 exact migration body is unavailable") from exc
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    if (
        set(observed) != materializer._M20_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or observed.get("schema")
        != "sawm/non-authoritative-migration-receipt@18"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != "SAWM-R2-M20"
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or observed.get(f"{key}_cid") != materializer._identity(dict(authority))
        or observed.get("database_path") != _M20_STORE_ID
        or observed.get("coordination_path") != _M20_COORDINATION_STORE_ID
        or observed.get("target_runtime_root") != str(Path(_M20_STORE_ID).parent)
        or int(observed.get("target_generation") or 0) != _M20_GENERATION
        or int(observed.get("target_quack_port") or 0)
        != _M20_TARGET_QUACK_PORT
        or int(observed.get("target_plan_revision") or 0)
        != _M20_TARGET_PLAN_REVISION
        or int(observed.get("target_event_watermark") or 0)
        != _M20_TARGET_EVENT_WATERMARK
        or int(observed.get("event_watermark") or 0)
        != _M20_TARGET_EVENT_WATERMARK
        or observed.get("prior_control_store_sha256")
        != authority["prior_control_store_sha256"]
        or observed.get("prior_coordination_store_sha256")
        != authority["prior_coordination_store_sha256"]
        or observed.get("prior_migration_receipt_cid")
        != authority["prior_migration_receipt_cid"]
        or observed.get("prior_source_binding_cid")
        != authority["prior_source_binding_cid"]
        or observed.get("repair_source_commit") != authority["repair_source_commit"]
        or observed.get("accepted_source_repair")
        != authority["accepted_source_repair"]
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or observed.get("semantic_authority_digest")
        != authority["target_semantic_authority_digest"]
        or observed.get("frozen_base_authority_digest")
        != authority["target_frozen_base_authority_digest"]
        or observed.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or int(observed.get("coordination_event_count") or 0)
        != int(authority["target_coordination_event_count"])
        or any(
            observed.get(name) != value
            for name, value in {
                "runtime_root_changed": True,
                "source_binding_changed": True,
                "control_and_coordination_bases_copied": True,
                "prior_control_store_mutated": False,
                "prior_coordination_store_mutated": False,
                "prior_runtime_artifacts_absent": True,
                "plan_revision_changes": 1,
                "evidence_node_changes": 1,
                "coordination_semantic_changes": 0,
                "task_revision_changes": 0,
                "task_status_changes": 0,
                "goal_changes": 0,
                "accepted_definition_changes": 0,
                "accepted_completion_changes": 0,
                "implementation_provider_invocations": 0,
                "effect_claim_changes": 0,
                "implementation_commit_changes": 0,
                "merge_attempt_changes": 0,
                "execution_sidecar_copied": False,
                "read_replica_sidecar_copied": False,
                "worker_self_approval": False,
            }.items()
        )
    ):
        raise OperatorError("M20 materialized final pair marker differs")
    if checked is not None:
        try:
            expected = materializer._expected_m20_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError("M20 exact materializer report is unavailable") from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or int(checked.get("event_watermark") or 0)
            != _M20_TARGET_EVENT_WATERMARK
            or checked.get("receipt") != observed
            or expected != observed
        ):
            raise OperatorError(
                "M20 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _require_m19_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M19's receipt-last marker without opening its live control file."""

    key = "live_catalog_inventory_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = materializer._expected_m19_live_catalog_inventory_authority()
    except Exception as exc:
        raise OperatorError("M19 exact source-binding authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M19 source-binding authority differs")

    control = (REPO_ROOT / _M19_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M19_COORDINATION_STORE_ID).resolve()
    receipt_path = control.parent / "migration-receipt.json"
    try:
        materializer._assert_m19_no_pending_receipts(control)
        observed, _ = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M19 final pair marker",
        )
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M19 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M19 final pair marker is unavailable") from exc

    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    try:
        migration_digest = materializer._identity(
            materializer._m19_migration_body(
                population,
                config,
                validation_digest,
            )
        )
    except Exception as exc:
        raise OperatorError("M19 exact migration body is unavailable") from exc

    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    authority_fields = {
        "prior_database_path": "prior_store_id",
        "prior_coordination_path": "prior_coordination_store_id",
        "prior_control_store_sha256": "prior_control_store_sha256",
        "prior_control_store_size": "prior_control_store_size",
        "prior_coordination_store_sha256": "prior_coordination_store_sha256",
        "prior_coordination_store_size": "prior_coordination_store_size",
        "prior_event_watermark": "prior_event_watermark",
        "prior_event_prefix_sha256": "prior_event_prefix_sha256",
        "prior_projection_cid": "prior_projection_cid",
        "prior_semantic_authority_digest": "prior_semantic_authority_digest",
        "prior_frozen_base_authority_digest": (
            "prior_frozen_base_authority_digest"
        ),
        "prior_append_surface_digest": "prior_append_surface_digest",
        "prior_catalog_digest": "prior_catalog_digest",
        "prior_coordination_projection_digest": (
            "prior_coordination_projection_digest"
        ),
        "prior_coordination_event_count": "prior_coordination_event_count",
        "prior_generation": "prior_generation",
        "prior_plan_revision": "prior_plan_revision",
        "prior_server_id": "prior_server_id",
        "prior_process_birth_id": "prior_process_birth_id",
        "prior_startup_epoch": "prior_startup_epoch",
        "prior_started_at": "prior_started_at",
        "prior_stopped_at": "prior_stopped_at",
        "prior_stopped_status_projection_path": (
            "prior_stopped_status_projection_path"
        ),
        "prior_stopped_status_projection_sha256": (
            "prior_stopped_status_projection_sha256"
        ),
        "prior_stopped_status_projection_size": (
            "prior_stopped_status_projection_size"
        ),
        "prior_migration_receipt_path": "prior_migration_receipt_path",
        "prior_migration_receipt_sha256": "prior_migration_receipt_sha256",
        "prior_migration_receipt_size": "prior_migration_receipt_size",
        "prior_migration_receipt_cid": "prior_migration_receipt_cid",
        "prior_receipt_publication_control_store_sha256": (
            "prior_receipt_publication_control_store_sha256"
        ),
        "prior_source_binding_cid": "prior_source_binding_cid",
        "repair_source_commit": "repair_source_commit",
        "accepted_source_repair": "accepted_source_repair",
        "semantic_authority_digest": "target_semantic_authority_digest",
        "frozen_base_authority_digest": "target_frozen_base_authority_digest",
        "coordination_projection_digest": (
            "target_coordination_projection_digest"
        ),
        "coordination_event_count": "target_coordination_event_count",
    }
    exact_flags = {
        "runtime_root_changed": True,
        "source_binding_changed": True,
        "control_and_coordination_bases_copied": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "prior_lifecycle_artifacts_preserved": True,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
        "coordination_semantic_changes": 0,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "goal_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    digest_names = (
        "control_store_sha256",
        "coordination_store_sha256",
        "semantic_authority_digest",
        "frozen_base_authority_digest",
        "append_surface_digest",
        "catalog_digest",
        "coordination_projection_digest",
    )
    if (
        set(observed) != materializer._M19_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or observed.get("schema")
        != "sawm/non-authoritative-migration-receipt@17"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != "SAWM-R2-M19"
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or not all(
            str(observed.get(name) or "")
            for name in (
                "migration_evidence_id",
                "plan_migration_event_id",
                "migration_evidence_event_id",
            )
        )
        or observed.get("database_path") != _M19_STORE_ID
        or observed.get("coordination_path") != _M19_COORDINATION_STORE_ID
        or observed.get("target_runtime_root") != str(Path(_M19_STORE_ID).parent)
        or int(observed.get("target_generation") or 0) != _M19_GENERATION
        or int(observed.get("target_quack_port") or 0)
        != _M19_TARGET_QUACK_PORT
        or int(observed.get("target_event_watermark") or 0)
        != _M19_TARGET_EVENT_WATERMARK
        or int(observed.get("migration_event_watermark") or 0)
        != _M19_TARGET_EVENT_WATERMARK
        or observed.get("plan_projection_cid")
        != authority["prior_projection_cid"]
        or (
            authority.get("target_projection_cid")
            and (
                observed.get("migration_projection_cid")
                != authority.get("target_projection_cid")
                or observed.get("projection_cid")
                != authority.get("target_projection_cid")
            )
        )
        or observed.get(f"{key}_cid") != materializer._identity(dict(authority))
        or any(
            observed.get(receipt_key) != authority[authority_key]
            for receipt_key, authority_key in authority_fields.items()
        )
        or any(observed.get(name) != value for name, value in exact_flags.items())
        or not all(
            re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(observed.get(name) or ""))
            for name in digest_names
        )
        or int(observed.get("control_store_size") or 0) <= 0
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or config.get("runtime_paths", {}).get("root")
        != authority["target_runtime_root"]
    ):
        raise OperatorError("M19 materialized final pair marker differs")

    if checked is not None:
        try:
            expected = materializer._expected_m19_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError("M19 exact materializer report is unavailable") from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or (
                authority.get("target_projection_cid")
                and checked.get("projection_cid")
                != authority.get("target_projection_cid")
            )
            or int(checked.get("event_watermark") or 0)
            != _M19_TARGET_EVENT_WATERMARK
            or checked.get("receipt") != observed
            or expected != observed
        ):
            raise OperatorError(
                "M19 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))

def _require_m17_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M17's receipt-last marker without opening its live control file."""

    key = "source_binding_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = materializer._expected_m17_source_binding_authority()
    except Exception as exc:
        raise OperatorError("M17 exact source-binding authority is unavailable") from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M17 source-binding authority differs")

    control = (REPO_ROOT / _M17_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M17_COORDINATION_STORE_ID).resolve()
    receipt_path = control.parent / "migration-receipt.json"
    try:
        materializer._assert_m17_no_pending_receipts(control)
        observed, _ = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M17 final pair marker",
        )
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M17 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M17 final pair marker is unavailable") from exc

    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    try:
        migration_digest = materializer._identity(
            materializer._m17_migration_body(
                population,
                config,
                validation_digest,
            )
        )
    except Exception as exc:
        raise OperatorError("M17 exact migration body is unavailable") from exc

    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    authority_fields = {
        "prior_database_path": "prior_store_id",
        "prior_coordination_path": "prior_coordination_store_id",
        "prior_control_store_sha256": "prior_control_store_sha256",
        "prior_control_store_size": "prior_control_store_size",
        "prior_coordination_store_sha256": "prior_coordination_store_sha256",
        "prior_coordination_store_size": "prior_coordination_store_size",
        "prior_event_watermark": "prior_event_watermark",
        "prior_event_prefix_sha256": "prior_event_prefix_sha256",
        "prior_projection_cid": "prior_projection_cid",
        "prior_semantic_authority_digest": "prior_semantic_authority_digest",
        "prior_frozen_base_authority_digest": (
            "prior_frozen_base_authority_digest"
        ),
        "prior_append_surface_digest": "prior_append_surface_digest",
        "prior_catalog_digest": "prior_catalog_digest",
        "prior_coordination_projection_digest": (
            "prior_coordination_projection_digest"
        ),
        "prior_coordination_event_count": "prior_coordination_event_count",
        "prior_generation": "prior_generation",
        "prior_plan_revision": "prior_plan_revision",
        "prior_server_id": "prior_server_id",
        "prior_process_birth_id": "prior_process_birth_id",
        "prior_startup_epoch": "prior_startup_epoch",
        "prior_started_at": "prior_started_at",
        "prior_stopped_at": "prior_stopped_at",
        "prior_stopped_status_projection_path": (
            "prior_stopped_status_projection_path"
        ),
        "prior_stopped_status_projection_sha256": (
            "prior_stopped_status_projection_sha256"
        ),
        "prior_stopped_status_projection_size": (
            "prior_stopped_status_projection_size"
        ),
        "prior_migration_receipt_path": "prior_migration_receipt_path",
        "prior_migration_receipt_sha256": "prior_migration_receipt_sha256",
        "prior_migration_receipt_size": "prior_migration_receipt_size",
        "prior_migration_receipt_cid": "prior_migration_receipt_cid",
        "prior_receipt_publication_control_store_sha256": (
            "prior_receipt_publication_control_store_sha256"
        ),
        "prior_source_binding_cid": "prior_source_binding_cid",
        "repair_source_commit": "repair_source_commit",
        "accepted_source_repair": "accepted_source_repair",
        "semantic_authority_digest": "target_semantic_authority_digest",
        "frozen_base_authority_digest": "target_frozen_base_authority_digest",
        "coordination_projection_digest": (
            "target_coordination_projection_digest"
        ),
        "coordination_event_count": "target_coordination_event_count",
    }
    exact_flags = {
        "runtime_root_changed": True,
        "source_binding_changed": True,
        "control_and_coordination_bases_copied": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "prior_lifecycle_artifacts_preserved": True,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
        "coordination_semantic_changes": 0,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "goal_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    digest_names = (
        "control_store_sha256",
        "coordination_store_sha256",
        "semantic_authority_digest",
        "frozen_base_authority_digest",
        "append_surface_digest",
        "catalog_digest",
        "coordination_projection_digest",
    )
    if (
        set(observed) != materializer._M17_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or observed.get("schema")
        != "sawm/non-authoritative-migration-receipt@15"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != "SAWM-R2-M17"
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or not all(
            str(observed.get(name) or "")
            for name in (
                "migration_evidence_id",
                "plan_migration_event_id",
                "migration_evidence_event_id",
            )
        )
        or observed.get("database_path") != _M17_STORE_ID
        or observed.get("coordination_path") != _M17_COORDINATION_STORE_ID
        or observed.get("target_runtime_root") != str(Path(_M17_STORE_ID).parent)
        or int(observed.get("target_generation") or 0) != _M17_GENERATION
        or int(observed.get("target_quack_port") or 0)
        != _M17_TARGET_QUACK_PORT
        or int(observed.get("target_event_watermark") or 0)
        != _M17_TARGET_EVENT_WATERMARK
        or int(observed.get("migration_event_watermark") or 0)
        != _M17_TARGET_EVENT_WATERMARK
        or observed.get("plan_projection_cid")
        != authority["prior_projection_cid"]
        or observed.get("migration_projection_cid")
        != _M17_TARGET_PROJECTION_CID
        or observed.get("projection_cid") != _M17_TARGET_PROJECTION_CID
        or observed.get(f"{key}_cid") != materializer._identity(dict(authority))
        or any(
            observed.get(receipt_key) != authority[authority_key]
            for receipt_key, authority_key in authority_fields.items()
        )
        or any(observed.get(name) != value for name, value in exact_flags.items())
        or not all(
            re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(observed.get(name) or ""))
            for name in digest_names
        )
        or int(observed.get("control_store_size") or 0) <= 0
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or config.get("runtime_paths", {}).get("root")
        != authority["target_runtime_root"]
    ):
        raise OperatorError("M17 materialized final pair marker differs")

    if checked is not None:
        try:
            expected = materializer._expected_m17_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError("M17 exact materializer report is unavailable") from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or checked.get("projection_cid") != _M17_TARGET_PROJECTION_CID
            or int(checked.get("event_watermark") or 0)
            != _M17_TARGET_EVENT_WATERMARK
            or checked.get("receipt") != observed
            or expected != observed
        ):
            raise OperatorError(
                "M17 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _require_m16_final_pair_marker(
    config: Mapping[str, Any], authority: Mapping[str, Any], materializer: Any, *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M16's receipt-last marker and exact plural rearm bindings."""

    key = "accepted_source_retry_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = (
            materializer._expected_m16_accepted_source_retry_authority()
        )
    except Exception as exc:
        raise OperatorError(
            "M16 exact accepted-source retry authority is unavailable"
        ) from exc
    if dict(authority) != expected_authority or config.get(key) != expected_authority:
        raise OperatorError("M16 accepted-source retry authority differs")

    control = (REPO_ROOT / _M16_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M16_COORDINATION_STORE_ID).resolve()
    receipt_path = control.parent / "migration-receipt.json"
    try:
        materializer._assert_m16_no_pending_receipts(control)
        observed, _ = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M16 final pair marker",
        )
    except Exception as exc:
        raise OperatorError("M16 final pair marker is unavailable") from exc
    try:
        coordination_sha256, coordination_size = (
            materializer._stable_regular_sha256(
                coordination,
                root=REPO_ROOT,
                noun="materialized M16 coordination store",
                required_link_count=1,
            )
        )
    except Exception as exc:
        raise OperatorError("M16 coordination store is unavailable") from exc

    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    try:
        migration_digest = materializer._identity(
            materializer._m16_migration_body(
                population,
                config,
                validation_digest,
            )
        )
    except Exception as exc:
        raise OperatorError("M16 exact migration body is unavailable") from exc

    unhashed = dict(observed)
    claimed = unhashed.pop("receipt_cid", "")
    plural_names = (
        "task_rearm_event_ids",
        "task_rearm_receipt_cids",
        "coordination_rearm_event_ids",
        "coordination_rearm_ids",
    )
    plural_bindings_are_closed = all(
        isinstance(observed.get(name), Mapping)
        and set(observed[name]) == {"SAWM-003", "SAWM-004"}
        and all(str(value) for value in observed[name].values())
        for name in plural_names
    )
    digest_names = (
        "control_store_sha256",
        "coordination_store_sha256",
        "semantic_authority_digest",
        "frozen_base_authority_digest",
        "append_surface_digest",
        "catalog_digest",
        "coordination_projection_digest",
    )
    receipt_digests_are_well_formed = all(
        re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(observed.get(name) or ""))
        is not None
        for name in digest_names
    )
    authority_fields = {
        "prior_database_path": "prior_store_id",
        "prior_coordination_path": "prior_coordination_store_id",
        "prior_control_store_sha256": "prior_control_store_sha256",
        "prior_control_store_size": "prior_control_store_size",
        "prior_coordination_store_sha256": "prior_coordination_store_sha256",
        "prior_coordination_store_size": "prior_coordination_store_size",
        "prior_event_watermark": "prior_event_watermark",
        "prior_event_prefix_sha256": "prior_event_prefix_sha256",
        "prior_projection_cid": "prior_projection_cid",
        "prior_semantic_authority_digest": "prior_semantic_authority_digest",
        "prior_frozen_base_authority_digest": (
            "prior_frozen_base_authority_digest"
        ),
        "prior_append_surface_digest": "prior_append_surface_digest",
        "prior_catalog_digest": "prior_catalog_digest",
        "prior_coordination_projection_digest": (
            "prior_coordination_projection_digest"
        ),
        "prior_coordination_event_count": "prior_coordination_event_count",
        "prior_generation": "prior_generation",
        "prior_plan_revision": "prior_plan_revision",
        "prior_server_id": "prior_server_id",
        "prior_process_birth_id": "prior_process_birth_id",
        "prior_startup_epoch": "prior_startup_epoch",
        "prior_started_at": "prior_started_at",
        "prior_stopped_at": "prior_stopped_at",
        "prior_stopped_status_projection_path": (
            "prior_stopped_status_projection_path"
        ),
        "prior_stopped_status_projection_sha256": (
            "prior_stopped_status_projection_sha256"
        ),
        "prior_stopped_status_projection_size": (
            "prior_stopped_status_projection_size"
        ),
        "prior_migration_receipt_path": "prior_migration_receipt_path",
        "prior_migration_receipt_sha256": "prior_migration_receipt_sha256",
        "prior_migration_receipt_size": "prior_migration_receipt_size",
        "prior_migration_receipt_cid": "prior_migration_receipt_cid",
        "repair_source_commit": "repair_source_commit",
        "accepted_source_repair": "accepted_source_repair",
        "failure_receipts_preserved": "failure_receipts",
        "operator_task_rearms": "task_rearms",
        "semantic_authority_digest": "target_semantic_authority_digest",
        "frozen_base_authority_digest": "target_frozen_base_authority_digest",
    }
    exact_flags = {
        "runtime_root_changed": True,
        "control_and_coordination_bases_copied": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "prior_lifecycle_artifacts_preserved": True,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
        "coordination_semantic_changes": 2,
        "task_revision_changes": 2,
        "task_status_changes": 2,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    if (
        set(observed) != _M16_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or observed.get("schema")
        != "sawm/non-authoritative-migration-receipt@14"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != "SAWM-R2-M16"
        or observed.get("database_path") != _M16_STORE_ID
        or observed.get("coordination_path") != _M16_COORDINATION_STORE_ID
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != migration_digest
        or not all(
            str(observed.get(name) or "")
            for name in (
                "migration_evidence_id",
                "plan_migration_event_id",
                "migration_evidence_event_id",
            )
        )
        or int(observed.get("target_generation") or 0) != _M16_GENERATION
        or int(observed.get("target_quack_port") or 0)
        != _M16_TARGET_QUACK_PORT
        or int(observed.get("target_event_watermark") or 0)
        != _M16_TARGET_EVENT_WATERMARK
        or int(observed.get("migration_event_watermark") or 0)
        != _M16_TARGET_EVENT_WATERMARK
        or observed.get("target_runtime_root")
        != str(Path(_M16_STORE_ID).parent)
        or observed.get("migration_projection_cid")
        != _M16_TARGET_PROJECTION_CID
        or observed.get("projection_cid") != _M16_TARGET_PROJECTION_CID
        or observed.get("plan_projection_cid")
        != authority["prior_projection_cid"]
        or observed.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or int(observed.get("coordination_event_count") or 0)
        != int(authority["target_coordination_event_count"])
        or observed.get(f"{key}_cid")
        != materializer._identity(dict(authority))
        or any(
            observed.get(receipt_key) != authority[authority_key]
            for receipt_key, authority_key in authority_fields.items()
        )
        or any(observed.get(name) != value for name, value in exact_flags.items())
        or not plural_bindings_are_closed
        or not receipt_digests_are_well_formed
        or int(observed.get("control_store_size") or 0) <= 0
        or int(observed.get("coordination_store_size") or 0) <= 0
        or observed.get("coordination_store_sha256") != coordination_sha256
        or int(observed.get("coordination_store_size") or 0)
        != coordination_size
        or config.get("runtime_paths", {}).get("root")
        != authority["target_runtime_root"]
    ):
        raise OperatorError("M16 materialized final pair marker differs")

    if checked is not None:
        try:
            expected = materializer._expected_m16_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError(
                "M16 exact materializer report is unavailable"
            ) from exc
        if (
            checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or checked.get("projection_cid") != _M16_TARGET_PROJECTION_CID
            or int(checked.get("event_watermark") or 0)
            != _M16_TARGET_EVENT_WATERMARK
            or checked.get("coordination_projection_digest")
            != authority["target_coordination_projection_digest"]
            or int(checked.get("coordination_event_count") or 0)
            != int(authority["target_coordination_event_count"])
            or checked.get("receipt") != observed
            or observed != expected
        ):
            raise OperatorError(
                "M16 materializer check differs from its final pair marker"
            )
    try:
        materializer._assert_m16_no_pending_receipts(control)
    except Exception as exc:
        raise OperatorError("M16 final pair marker has a pending temporary") from exc
    return MappingProxyType(dict(observed))


def _require_m15_final_pair_marker(
    config: Mapping[str, Any], authority: Mapping[str, Any], materializer: Any, *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require M15's receipt without deep-opening a potentially live store."""

    key = "runtime_root_rebind_successor_materialization"
    if key not in config:
        return MappingProxyType({})
    try:
        expected_authority = (
            materializer._expected_m15_runtime_root_rebind_authority()
        )
    except Exception as exc:
        raise OperatorError("M15 exact runtime-root authority is unavailable") from exc
    if dict(authority) != expected_authority:
        raise OperatorError("M15 runtime-root rebind authority differs")
    control = (REPO_ROOT / _M15_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M15_COORDINATION_STORE_ID).resolve()
    receipt_path = control.parent / "migration-receipt.json"
    try:
        materializer._assert_m15_no_pending_receipts(control)
        observed, _ = materializer._load_nofollow_json(
            receipt_path, root=REPO_ROOT, noun="M15 final pair marker"
        )
    except Exception as exc:
        raise OperatorError("M15 final pair marker is unavailable") from exc
    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity(
        {
            "dependency": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_dependencies.py",
            ),
            "board": materializer._validator_report(
                REPO_ROOT,
                "scripts/validate_semantic_addressed_world_model_board.py",
            ),
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    body = materializer._m15_migration_body(
        population, config, validation_digest
    )
    unhashed = dict(observed)
    claimed = unhashed.pop("receipt_cid", "")
    authority_fields = {
        "prior_database_path": "prior_store_id",
        "prior_coordination_path": "prior_coordination_store_id",
        "prior_control_store_sha256": "prior_control_store_sha256",
        "prior_control_store_size": "prior_control_store_size",
        "prior_coordination_store_sha256": (
            "prior_coordination_store_sha256"
        ),
        "prior_coordination_store_size": "prior_coordination_store_size",
        "prior_event_watermark": "prior_event_watermark",
        "prior_event_prefix_sha256": "prior_event_prefix_sha256",
        "prior_projection_cid": "prior_projection_cid",
        "prior_semantic_authority_digest": (
            "prior_semantic_authority_digest"
        ),
        "prior_frozen_base_authority_digest": (
            "prior_frozen_base_authority_digest"
        ),
        "prior_append_surface_digest": "prior_append_surface_digest",
        "prior_catalog_digest": "prior_catalog_digest",
        "prior_generation": "prior_generation",
        "prior_plan_revision": "prior_plan_revision",
        "prior_server_id": "prior_server_id",
        "prior_process_birth_id": "prior_process_birth_id",
        "prior_startup_epoch": "prior_startup_epoch",
        "prior_started_at": "prior_started_at",
        "prior_stopped_at": "prior_stopped_at",
        "prior_stopped_status_projection_path": (
            "prior_stopped_status_projection_path"
        ),
        "prior_stopped_status_projection_sha256": (
            "prior_stopped_status_projection_sha256"
        ),
        "prior_stopped_status_projection_size": (
            "prior_stopped_status_projection_size"
        ),
        "prior_migration_receipt_path": "prior_migration_receipt_path",
        "prior_migration_receipt_sha256": "prior_migration_receipt_sha256",
        "prior_migration_receipt_size": "prior_migration_receipt_size",
        "prior_migration_receipt_cid": "prior_migration_receipt_cid",
        "detached_launch_blocker": "detached_launch_blocker",
        "coordination_projection_digest": (
            "prior_coordination_projection_digest"
        ),
        "coordination_event_count": "prior_coordination_event_count",
    }
    exact_flags = {
        "runtime_root_changed": True,
        "runtime_root_collision_removed": True,
        "control_and_coordination_bases_copied": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "prior_lifecycle_artifacts_preserved": True,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
        "coordination_semantic_changes": 0,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    receipt_hashes_are_well_formed = all(
        re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(observed.get(name) or ""))
        is not None
        for name in (
            "control_store_sha256",
            "coordination_store_sha256",
            "semantic_authority_digest",
            "frozen_base_authority_digest",
            "append_surface_digest",
            "catalog_digest",
        )
    )
    if (
        set(observed) != _M15_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or observed.get("schema")
        != "sawm/non-authoritative-migration-receipt@13"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != "SAWM-R2-M15"
        or observed.get("database_path") != _M15_STORE_ID
        or observed.get("coordination_path") != _M15_COORDINATION_STORE_ID
        or observed.get("program_definition_cid")
        != population["program_definition_cid"]
        or observed.get("current_source_binding_cid")
        != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != materializer._identity(body)
        or not receipt_hashes_are_well_formed
        or int(observed.get("control_store_size") or 0) <= 0
        or int(observed.get("coordination_store_size") or 0) <= 0
        or any(
            observed.get(receipt_key) != authority[authority_key]
            for receipt_key, authority_key in authority_fields.items()
        )
        or any(observed.get(name) != value for name, value in exact_flags.items())
        or int(observed.get("target_generation") or 0) != _M15_GENERATION
        or int(observed.get("target_event_watermark") or 0)
        != _M15_TARGET_EVENT_WATERMARK
        or int(observed.get("migration_event_watermark") or 0)
        != _M15_TARGET_EVENT_WATERMARK
        or int(observed.get("target_quack_port") or 0) != 24_058
        or observed.get("target_runtime_root")
        != authority["target_runtime_root"]
        or observed.get("plan_projection_cid")
        != authority["prior_projection_cid"]
        or observed.get("migration_projection_cid")
        != authority["target_projection_cid"]
        or observed.get("projection_cid")
        != authority["target_projection_cid"]
        or observed.get("semantic_authority_digest")
        != authority["prior_semantic_authority_digest"]
        or observed.get("frozen_base_authority_digest")
        != authority["prior_frozen_base_authority_digest"]
        or observed.get("catalog_digest") != authority["prior_catalog_digest"]
        or observed.get("runtime_root_rebind_successor_materialization_cid")
        != materializer._identity(dict(authority))
        or config.get("runtime_paths", {}).get("root")
        != authority["target_runtime_root"]
    ):
        raise OperatorError("M15 materialized final pair marker differs")
    if checked is not None:
        try:
            expected = materializer._expected_m15_migration_receipt(
                REPO_ROOT,
                control,
                coordination,
                population,
                config,
                checked,
                validation_digest,
            )
        except Exception as exc:
            raise OperatorError("M15 exact materializer report is unavailable") from exc
        deep_bindings = {
            "projection_cid": "projection_cid",
            "event_watermark": "target_event_watermark",
            "coordination_projection_digest": (
                "coordination_projection_digest"
            ),
            "coordination_event_count": "coordination_event_count",
            "semantic_authority_digest": "semantic_authority_digest",
            "frozen_base_authority_digest": "frozen_base_authority_digest",
            "append_surface_digest": "append_surface_digest",
            "catalog_digest": "catalog_digest",
            "plan_migration_event_id": "plan_migration_event_id",
            "migration_evidence_event_id": "migration_evidence_event_id",
            "migration_evidence_id": "migration_evidence_id",
            "migration_digest": "migration_digest",
            "plan_revision_changes": "plan_revision_changes",
            "evidence_node_changes": "evidence_node_changes",
            "task_revision_changes": "task_revision_changes",
            "task_status_changes": "task_status_changes",
            "accepted_definition_changes": "accepted_definition_changes",
            "accepted_completion_changes": "accepted_completion_changes",
            "coordination_semantic_changes": "coordination_semantic_changes",
        }
        if (
            checked.get("valid") is not True
            or checked.get("validation_digest") != validation_digest
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or checked.get("receipt") != observed
            or observed != expected
            or any(
                checked.get(report_key) != observed.get(receipt_key)
                for report_key, receipt_key in deep_bindings.items()
            )
        ):
            raise OperatorError(
                "M15 materializer check differs from its final pair marker"
            )
    try:
        materializer._assert_m15_no_pending_receipts(control)
    except Exception as exc:
        raise OperatorError("M15 final pair marker has a pending temporary") from exc
    return MappingProxyType(dict(observed))


def _require_m14_final_pair_marker(
    config: Mapping[str, Any], authority: Mapping[str, Any], materializer: Any, *,
    checked: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Require the reconstructed, receipt-last M14 pair marker."""
    if "stale_owner_restart_successor_materialization" not in config:
        return MappingProxyType({})
    try:
        expected_authority = materializer._expected_m14_stale_owner_restart_authority()
    except Exception as exc:
        raise OperatorError("M14 exact stale-owner authority is unavailable") from exc
    if dict(authority) != expected_authority:
        raise OperatorError("M14 stale-owner authority differs")
    control = (REPO_ROOT / _M14_STORE_ID).resolve()
    coordination = (REPO_ROOT / _M14_COORDINATION_STORE_ID).resolve()
    receipt_path = control.parent / "migration-receipt.json"
    try:
        observed, _ = materializer._load_nofollow_json(
            receipt_path,
            root=REPO_ROOT,
            noun="M14 final pair marker",
        )
    except Exception as exc:
        raise OperatorError("M14 materialized final pair marker is unavailable") from exc
    population = materializer.build_population(REPO_ROOT)
    validation_digest = materializer._identity({
        "dependency": materializer._validator_report(REPO_ROOT, "scripts/validate_semantic_addressed_world_model_dependencies.py"),
        "board": materializer._validator_report(REPO_ROOT, "scripts/validate_semantic_addressed_world_model_board.py"),
        "program_definition_cid": population["program_definition_cid"],
    })
    body = materializer._m14_migration_body(population, config, validation_digest)
    unhashed = dict(observed)
    claimed = unhashed.pop("receipt_cid", "")
    authority_fields = {
        "prior_database_path": "prior_store_id",
        "prior_coordination_path": "prior_coordination_store_id",
        "prior_control_store_sha256": "prior_control_store_sha256",
        "prior_control_store_size": "prior_control_store_size",
        "prior_coordination_store_sha256": "prior_coordination_store_sha256",
        "prior_coordination_store_size": "prior_coordination_store_size",
        "prior_event_watermark": "prior_event_watermark",
        "prior_event_prefix_sha256": "prior_event_prefix_sha256",
        "prior_projection_cid": "prior_projection_cid",
        "prior_semantic_authority_digest": "prior_semantic_authority_digest",
        "prior_frozen_base_authority_digest": (
            "prior_frozen_base_authority_digest"
        ),
        "prior_append_surface_digest": "prior_append_surface_digest",
        "prior_catalog_digest": "prior_catalog_digest",
        "prior_database_uuid": "prior_database_uuid",
        "prior_generation": "prior_generation",
        "prior_server_id": "prior_server_id",
        "prior_process_birth_id": "prior_process_birth_id",
        "prior_startup_epoch": "prior_startup_epoch",
        "prior_state_server_revision": "prior_state_server_revision",
        "prior_stopped_at": "prior_stopped_at",
        "prior_source_head": "prior_source_head",
        "prior_source_tree": "prior_source_tree",
        "post_stop_source_head": "post_stop_source_head",
        "post_stop_source_tree": "post_stop_source_tree",
        "prior_stopped_status_projection_path": (
            "prior_stopped_status_projection_path"
        ),
        "prior_stopped_status_projection_present": (
            "prior_stopped_status_projection_present"
        ),
        "prior_stopped_status_projection_sha256": (
            "prior_stopped_status_projection_sha256"
        ),
        "prior_stopped_status_projection_size": (
            "prior_stopped_status_projection_size"
        ),
        "prior_stale_owner_recovery_receipt_path": (
            "prior_stale_owner_recovery_receipt_path"
        ),
        "prior_stale_owner_recovery_receipt_present": (
            "prior_stale_owner_recovery_receipt_present"
        ),
        "prior_stale_owner_recovery_receipt_sha256": (
            "prior_stale_owner_recovery_receipt_sha256"
        ),
        "prior_stale_owner_recovery_receipt_size": (
            "prior_stale_owner_recovery_receipt_size"
        ),
        "prior_stale_owner_recovery_cid": "prior_stale_owner_recovery_cid",
        "prior_migration_receipt_path": "prior_migration_receipt_path",
        "prior_migration_receipt_sha256": "prior_migration_receipt_sha256",
        "prior_migration_receipt_size": "prior_migration_receipt_size",
        "prior_migration_receipt_cid": "prior_migration_receipt_cid",
        "coordination_projection_digest": (
            "prior_coordination_projection_digest"
        ),
        "coordination_event_count": "prior_coordination_event_count",
    }
    exact_flags = {
        "control_and_coordination_bases_copied": True,
        "prior_wals_absent": True,
        "prior_owner_marker_absent": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "prior_lifecycle_artifacts_preserved": True,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
        "coordination_semantic_changes": 0,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    receipt_hashes_are_well_formed = all(
        re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(observed.get(key) or ""))
        is not None
        for key in (
            "control_store_sha256",
            "coordination_store_sha256",
            "semantic_authority_digest",
            "frozen_base_authority_digest",
            "append_surface_digest",
            "catalog_digest",
        )
    )
    if (
        set(observed) != _M14_RECEIPT_KEYS
        or claimed != materializer._identity(unhashed)
        or observed.get("schema") != "sawm/non-authoritative-migration-receipt@12"
        or observed.get("authoritative") is not False
        or observed.get("control_database_is_authority") is not True
        or observed.get("coordination_database_is_authority") is not True
        or observed.get("receipt_is_final_pair_commit_marker") is not True
        or observed.get("migration_revision") != "SAWM-R2-M14"
        or observed.get("database_path") != _M14_STORE_ID
        or observed.get("coordination_path") != _M14_COORDINATION_STORE_ID
        or observed.get("program_definition_cid") != population["program_definition_cid"]
        or observed.get("current_source_binding_cid") != population["source_binding"]["source_binding_cid"]
        or observed.get("validation_digest") != validation_digest
        or observed.get("migration_digest") != materializer._identity(body)
        or not receipt_hashes_are_well_formed
        or int(observed.get("control_store_size") or 0) <= 0
        or int(observed.get("coordination_store_size") or 0) <= 0
        or any(
            observed.get(receipt_key) != authority[authority_key]
            for receipt_key, authority_key in authority_fields.items()
        )
        or any(observed.get(key) != value for key, value in exact_flags.items())
        or int(observed.get("target_generation") or 0) != _M14_GENERATION
        or int(observed.get("target_event_watermark") or 0) != _M14_TARGET_EVENT_WATERMARK
        or int(observed.get("target_quack_port") or 0) != 24_057
        or observed.get("migration_projection_cid") != authority["target_projection_cid"]
        or observed.get("plan_projection_cid") != authority["prior_projection_cid"]
        or observed.get("projection_cid") != authority["target_projection_cid"]
        or int(observed.get("migration_event_watermark") or 0) != _M14_TARGET_EVENT_WATERMARK
        or observed.get("semantic_authority_digest") != authority["prior_semantic_authority_digest"]
        or observed.get("frozen_base_authority_digest") != authority["prior_frozen_base_authority_digest"]
        or observed.get("catalog_digest") != authority["prior_catalog_digest"]
        or observed.get("stale_owner_restart_successor_materialization_cid") != materializer._identity(dict(authority))
    ):
        raise OperatorError("M14 materialized final pair marker differs")
    if checked is not None:
        try:
            control_sha, control_size = materializer._stable_regular_sha256(
                control,
                root=REPO_ROOT,
                noun="M14 control store",
                required_link_count=1,
            )
            coordination_sha, coordination_size = (
                materializer._stable_regular_sha256(
                    coordination,
                    root=REPO_ROOT,
                    noun="M14 coordination store",
                    required_link_count=1,
                )
            )
        except Exception as exc:
            raise OperatorError("M14 materialized pair is unavailable") from exc
        if (
            observed.get("control_store_sha256") != control_sha
            or int(observed.get("control_store_size") or 0) != control_size
            or observed.get("coordination_store_sha256") != coordination_sha
            or int(observed.get("coordination_store_size") or 0)
            != coordination_size
            or checked.get("valid") is not True
            or checked.get("database_path") != str(control)
            or checked.get("coordination_path") != str(coordination)
            or checked.get("projection_cid") != authority["target_projection_cid"]
            or checked.get("coordination_projection_digest")
            != authority["prior_coordination_projection_digest"]
            or int(checked.get("event_watermark") or 0)
            != int(authority["target_event_watermark"])
            or checked.get("semantic_authority_digest")
            != authority["prior_semantic_authority_digest"]
            or checked.get("receipt") != observed
        ):
            raise OperatorError(
                "M14 materializer check differs from its final pair marker"
            )
    return MappingProxyType(dict(observed))


def _require_active_final_pair_marker(
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
    materializer: Any,
    *,
    checked: Mapping[str, Any] | None = None,
    live_source: Any | None = None,
    validation_digest: str = "",
    live_identity: Mapping[str, Any] | None = None,
    remote_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Dispatch to the newest key-present pair marker contract."""

    if "json_emission_normalization_successor_materialization" in config:
        return _require_m34_source_successor_marker(
            config, authority, materializer, checked=checked
        )
    if "live_preflight_contract_successor_materialization" in config:
        return _require_m33_source_successor_marker(
            config, authority, materializer, checked=checked
        )
    if "live_preflight_plan_anchor_successor_materialization" in config:
        return _require_m32_source_successor_marker(
            config, authority, materializer, checked=checked
        )
    if "detached_coordinator_pid_recovery_successor_materialization" in config:
        return _require_m31_source_successor_marker(
            config, authority, materializer, checked=checked
        )
    if "stopped_owner_restart_source_seal_successor_materialization" in config:
        return _require_m30_source_successor_marker(
            config, authority, materializer, checked=checked
        )
    if "committed_evidence_verification_successor_materialization" in config:
        return _require_m29_source_successor_marker(
            config, authority, materializer, checked=checked
        )
    if "live_claim_admission_recovery_successor_materialization" in config:
        return _require_m28_source_successor_marker(
            config, authority, materializer, checked=checked
        )
    if "dead_owner_parallel_resume_successor_materialization" in config:
        return _require_m27_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "automatic_stall_recovery_successor_materialization" in config:
        return _require_m26_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "native_duckdb_preload_successor_materialization" in config:
        return _require_m25_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "multi_lane_sidecar_reopen_successor_materialization" in config:
        return _require_m24_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "multi_lane_successor_materialization" in config:
        return _require_m23_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "live_preflight_receipt_compatibility_successor_materialization" in config:
        return _require_m22_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "generation_realization_successor_materialization" in config:
        return _require_m21_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "test_isolation_successor_materialization" in config:
        return _require_m20_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "live_catalog_inventory_successor_materialization" in config:
        return _require_m19_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "portal_completion_persistence_successor_materialization" in config:
        return _require_m18_final_pair_marker(
            config,
            authority,
            materializer,
            checked=checked,
            live_source=live_source,
            validation_digest=validation_digest,
            live_identity=live_identity,
            remote_identity=remote_identity,
        )
    if "source_binding_successor_materialization" in config:
        return _require_m17_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "accepted_source_retry_successor_materialization" in config:
        return _require_m16_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "runtime_root_rebind_successor_materialization" in config:
        return _require_m15_final_pair_marker(
            config, authority, materializer, checked=checked
        )
    if "stale_owner_restart_successor_materialization" in config:
        return _require_m14_final_pair_marker(config, authority, materializer, checked=checked)
    if "quack_refresh_successor_materialization" in config:
        return _require_m13_final_pair_marker(
            config,
            authority,
            materializer,
            checked=checked,
        )
    if "declared_output_retry_successor_materialization" in config:
        return _require_m12_final_pair_marker(
            config,
            authority,
            materializer,
            checked=checked,
        )
    if "live_provider_retry_successor_materialization" in config:
        return _require_m11_final_pair_marker(
            config,
            authority,
            materializer,
            checked=checked,
        )
    if "live_projection_successor_materialization" in config:
        return _require_m10_final_pair_marker(
            config,
            authority,
            materializer,
            checked=checked,
        )
    return _require_m9_final_pair_marker(
        config,
        authority,
        materializer,
        checked=checked,
    )


def _quack_args(config: Mapping[str, Any], command: str) -> list[str]:
    owner = config["quack_owner"]
    return [
        "--database", str(REPO_ROOT / owner["database_path"]),
        "--state-dir", str(REPO_ROOT / owner["state_dir"]),
        "--host", str(owner["host"]), "--port", str(owner["port"]),
        "--store-id", str(owner["store_id"]),
        "--repository-id", str(owner["repository_id"]),
        "--secret-handle", str(owner["secret_handle"]), "--json", command,
    ]


def _recover_stale_quack(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Settle one dead owner's canonical stop before successor sealing."""

    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        recover_stale_state_server,
    )

    owner = config.get("quack_owner")
    program = config.get("database_program")
    if not isinstance(owner, Mapping) or not isinstance(program, Mapping):
        raise OperatorError("stale Quack recovery lacks configured authority")
    store_id = str(program.get("store_id") or "")
    generation = int(program.get("store_generation") or 0)
    if (
        str(owner.get("database_path") or "") != store_id
        or str(owner.get("store_id") or "") != store_id
        or generation < 1
    ):
        raise OperatorError("stale Quack recovery store binding differs")
    active = _active_source_repair_materialization(config)
    contract = _normalized_live_preflight_contract(active, _materializer())
    database_uuid = str(contract.get("database_uuid") or "")
    if (
        not database_uuid
        or contract.get("target_store_id") != store_id
        or int(contract.get("target_generation") or 0) != generation
    ):
        raise OperatorError(
            "stale Quack recovery has no sealed database UUID authority"
        )
    return recover_stale_state_server(
        database_path=REPO_ROOT / str(owner["database_path"]),
        state_dir=REPO_ROOT / str(owner["state_dir"]),
        expected_store_id=store_id,
        expected_generation=generation,
        expected_database_uuid=database_uuid,
    )


def _owner_connection(path: Path, owner: Mapping[str, Any]):
    """Open the sealed canonical writer without loading or serving Quack."""

    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnection,
        connect_duckdb_with_policy,
        exclusive_file_lock,
    )

    database = Path(path).resolve()
    lock = exclusive_file_lock(database.with_name(f".{database.name}.lock"))
    lock.__enter__()
    connection = None
    try:
        connection = connect_duckdb_with_policy(
            duckdb, database,
            configuration={"threads": 1, "memory_limit": "256MB"},
        )
        wrapped = DuckDBConnection.wrap(connection)
        wrapped.path = database
        wrapped._lock_context = lock
        return wrapped
    except BaseException:
        if connection is not None:
            connection.close()
        lock.__exit__(*sys.exc_info())
        raise


def _remote_owner_identity(
    uri: str,
    token: str,
    identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Resolve the live owner from canonical rows on the Quack replica."""

    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        _schema_fingerprint_digest,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_quack_transport_connection,
    )

    expected = dict(identity)
    required = {
        "server_id",
        "store_id",
        "database_uuid",
        "process_birth_id",
        "listen_uri",
        "extension_fingerprint",
        "schema_revision",
        "schema_fingerprint",
        "generation",
        "fence_epoch",
        "revision",
        "credential_generation",
        "secret_handle",
    }
    missing = sorted(key for key in required if expected.get(key) in (None, ""))
    if missing:
        raise OperatorError(
            "expected Quack owner identity is incomplete: " + ", ".join(missing)
        )

    try:
        connection = open_quack_transport_connection(uri, token=token)
    except Exception:
        # The transport currently authenticates during ATTACH.  DuckDB may
        # include failed SQL in its exception, so replace that exception before
        # it can reach the operator's JSON/log surface.
        raise OperatorError("authenticated Quack owner connection failed") from None
    try:
        try:
            state_rows = connection.execute(
                "SELECT server_id, store_id, database_uuid, process_birth_id, "
                "listen_uri, extension_fingerprint, schema_revision, generation, "
                "status, revision FROM state_servers WHERE server_id = ? AND generation = ?",
                [expected["server_id"], int(expected["generation"])],
            ).fetchall()
            generation_rows = connection.execute(
                "SELECT generation, schema_revision, fence_epoch, revision, "
                "database_uuid, birth_id FROM store_generations WHERE generation = ?",
                [int(expected["generation"])],
            ).fetchall()
            credential_id = (
                f"cred:{expected['server_id']}:{int(expected['credential_generation'])}"
            )
            credential_rows = connection.execute(
                "SELECT credential_id, secret_handle, generation, purpose, revoked_at, "
                "revision FROM credentials WHERE credential_id = ?",
                [credential_id],
            ).fetchall()
            metadata_rows = connection.execute(
                "SELECT key, value FROM control_plane_metadata WHERE key IN "
                "('database_uuid', 'schema_version', 'schema_fingerprint') ORDER BY key"
            ).fetchall()
        except Exception:
            raise OperatorError("authenticated Quack owner query failed") from None
    finally:
        try:
            connection.close()
        except Exception:
            raise OperatorError("authenticated Quack owner close failed") from None

    if len(state_rows) != 1 or len(generation_rows) != 1 or len(credential_rows) != 1:
        raise OperatorError("live Quack owner identity rows are missing or ambiguous")
    state = state_rows[0]
    generation = generation_rows[0]
    credential = credential_rows[0]
    metadata = {str(row[0]): str(row[1]) for row in metadata_rows}
    normalized_metadata = {
        **metadata,
        "schema_fingerprint": _schema_fingerprint_digest(
            metadata.get("schema_fingerprint", "")
        ),
    }
    expected_state = (
        str(expected["server_id"]),
        str(expected["store_id"]),
        str(expected["database_uuid"]),
        str(expected["process_birth_id"]),
        str(expected["listen_uri"]),
        str(expected["extension_fingerprint"]),
        int(expected["schema_revision"]),
        int(expected["generation"]),
        "ready",
    )
    observed_state = (
        str(state[0]),
        str(state[1]),
        str(state[2]),
        str(state[3]),
        str(state[4]),
        str(state[5]),
        int(state[6]),
        int(state[7]),
        str(state[8]),
    )
    if observed_state != expected_state or int(state[9]) < int(expected["revision"]):
        raise OperatorError("live Quack state-server row differs from the owner identity")
    if (
        int(generation[0]),
        int(generation[1]),
        int(generation[2]),
        int(generation[3]),
        str(generation[4]),
        str(generation[5]),
    ) != (
        int(expected["generation"]),
        int(expected["schema_revision"]),
        int(expected["fence_epoch"]),
        int(expected["revision"]),
        str(expected["database_uuid"]),
        str(expected["process_birth_id"]),
    ):
        raise OperatorError("live Quack store-generation row differs from the owner identity")
    if (
        str(credential[0]),
        str(credential[1]),
        int(credential[2]),
        str(credential[3]),
        credential[4],
    ) != (
        credential_id,
        str(expected["secret_handle"]),
        int(expected["credential_generation"]),
        "quack-auth",
        None,
    ) or int(credential[5]) < int(expected["revision"]):
        raise OperatorError("live Quack credential row differs from the owner identity")
    if normalized_metadata != {
        "database_uuid": str(expected["database_uuid"]),
        "schema_fingerprint": str(expected["schema_fingerprint"]),
        "schema_version": str(int(expected["schema_revision"])),
    }:
        raise OperatorError("live Quack schema metadata differs from the owner identity")
    return MappingProxyType(
        {
            "server_id": str(state[0]),
            "store_id": str(state[1]),
            "database_uuid": str(state[2]),
            "process_birth_id": str(state[3]),
            "listen_uri": str(state[4]),
            "extension_fingerprint": str(state[5]),
            "schema_revision": int(state[6]),
            "generation": int(state[7]),
            "schema_fingerprint": normalized_metadata["schema_fingerprint"],
            "credential_generation": int(credential[2]),
            "live": True,
            "canonical_rows_verified": True,
        }
    )


class _SawmQuackTransport:
    """Serve only an atomically refreshed, read-only replica through Quack."""

    def __init__(self, owner: Mapping[str, Any]) -> None:
        self._owner = dict(owner)
        self._serve_uri = ""
        self._server_identity: dict[str, Any] = {}
        self._replica_connection = None
        self._replica_path: Path | None = None
        self._refresh_sequence = 0
        self._extension_projection_parent: Path | None = None
        self._sealed_extension_set: Any | None = None

    @staticmethod
    def _verify_extension_source(
        name: str,
        pin: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        """Verify accepted extension and metadata bytes before native LOAD."""

        if name not in {"httpfs", "quack"}:
            raise OperatorError("extension name is outside the reviewed set")
        expected_fields = {
            "path",
            "info_path",
            "version",
            "sha256",
            "size",
            "info_sha256",
            "info_size",
            "network_install_allowed",
            "unsigned_extension_allowed",
            *(
                {"service_external_access_limitation"}
                if name == "quack"
                else set()
            ),
        }
        if (
            type(pin) is not dict
            or set(pin) != expected_fields
            or pin.get("network_install_allowed") is not False
            or pin.get("unsigned_extension_allowed") is not False
            or type(pin.get("version")) is not str
            or re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}", pin["version"])
            is None
            or type(pin.get("sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) is None
            or type(pin.get("info_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", pin["info_sha256"]) is None
            or (
                name == "quack"
                and pin.get("service_external_access_limitation")
                != "canonical_writer_sealed; "
                "pinned_extension_preloaded_only_in_locked_read_only_loopback_replica"
            )
        ):
            raise OperatorError(f"{name} extension pin is noncanonical")

        def evidence(
            path_value: object,
            expected_size: object,
            *,
            maximum_size: int,
        ) -> tuple[Path, str]:
            if type(path_value) is not str or type(expected_size) is not int:
                raise OperatorError(f"{name} extension path or size is invalid")
            path = Path(path_value)
            size = expected_size
            if not path.is_absolute() or size <= 0 or size > maximum_size:
                raise OperatorError(f"{name} extension size is outside policy")
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise OperatorError(f"{name} extension source is unavailable") from exc
            if resolved != path:
                raise OperatorError(f"{name} extension source path is noncanonical")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise OperatorError(f"{name} extension source is unavailable") from exc
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_uid != os.geteuid()
                    or before.st_nlink != 1
                    or before.st_size != size
                ):
                    raise OperatorError(f"{name} extension source is not stable evidence")
                digest = hashlib.sha256()
                offset = 0
                while offset < size:
                    block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
                    if not block:
                        break
                    digest.update(block)
                    offset += len(block)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            try:
                current = os.stat(path, follow_symlinks=False)
            except OSError as exc:
                raise OperatorError(f"{name} extension source changed after read") from exc

            def identity(item: os.stat_result) -> tuple[int, ...]:
                return (
                    item.st_dev,
                    item.st_ino,
                    item.st_mode,
                    item.st_uid,
                    item.st_nlink,
                    item.st_size,
                    item.st_mtime_ns,
                    item.st_ctime_ns,
                )

            if (
                offset != size
                or identity(before) != identity(after)
                or identity(after) != identity(current)
            ):
                raise OperatorError(f"{name} extension source changed while read")
            return resolved, digest.hexdigest()

        payload_path, payload_sha = evidence(
            pin.get("path"), pin.get("size"), maximum_size=64 * 1024 * 1024
        )
        info_path, info_sha = evidence(
            pin.get("info_path"),
            pin.get("info_size"),
            maximum_size=64 * 1024,
        )
        if (
            payload_path.name != f"{name}.duckdb_extension"
            or info_path != payload_path.with_name(f"{payload_path.name}.info")
            or payload_sha != pin["sha256"]
            or info_sha != pin["info_sha256"]
        ):
            raise OperatorError(f"{name} extension bytes differ from the reviewed pin")
        return payload_path, info_path

    def _ensure_extension_projection(self) -> Any:
        """Publish reviewed bytes and return the shared sealed-set custodian."""

        if self._sealed_extension_set is not None:
            self._verify_sealed_extension_projection()
            return self._sealed_extension_set
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_extension_projection import (
            ConfiguredBoardExtensionProjectionError,
            build_configured_board_extension_set_pin,
            inspect_configured_board_extension_sources,
            project_configured_board_extension_set_home,
            seal_configured_board_extension_set_home,
        )

        source_pins = {
            "httpfs": self._owner.get("pinned_httpfs_extension") or {},
            "quack": self._owner.get("pinned_extension") or {},
        }
        sources = {
            name: self._verify_extension_source(name, pin)
            for name, pin in source_pins.items()
        }
        roots = {payload.parent for payload, _info in sources.values()}
        if len(roots) != 1:
            raise OperatorError("reviewed extensions do not share one exact root")
        source_root = next(iter(roots))
        engine_version = source_root.parent.name
        platform = source_root.name
        if (
            re.fullmatch(r"v[0-9][0-9A-Za-z.+_-]{0,62}", engine_version) is None
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", platform)
            is None
        ):
            raise OperatorError("reviewed extension engine/platform root is invalid")
        pins = {
            name: inspect_configured_board_extension_sources(
                payload,
                info,
                name=name,
                engine_version=engine_version,
                platform=platform,
            )
            for name, (payload, info) in sources.items()
        }
        for name, pin in pins.items():
            accepted = source_pins[name]
            if (
                pin.payload_sha256 != "sha256:" + str(accepted["sha256"])
                or pin.payload_size != accepted["size"]
                or pin.info_sha256 != "sha256:" + str(accepted["info_sha256"])
                or pin.info_size != accepted["info_size"]
            ):
                raise OperatorError(
                    f"{name} extension projection differs from its reviewed pin"
                )
        parent = Path(
            tempfile.mkdtemp(prefix="sawm-quack-extension-projection-", dir="/tmp")
        )
        os.chmod(parent, 0o700)
        sealed = None
        try:
            regular_home = project_configured_board_extension_set_home(
                pins,
                sources=sources,
                parent=parent,
            )
            set_pin = build_configured_board_extension_set_pin(
                pins,
                versions={
                    name: str(source_pins[name]["version"])
                    for name in sorted(source_pins)
                },
            )
            sealed = seal_configured_board_extension_set_home(
                set_pin,
                regular_home,
            )
            sealed.verify()
            self._extension_projection_parent = parent
            self._sealed_extension_set = sealed
            return sealed
        except BaseException as exc:
            if sealed is not None:
                sealed.close()
            self._remove_regular_extension_projection(parent)
            if isinstance(exc, ConfiguredBoardExtensionProjectionError):
                raise OperatorError(str(exc)) from exc
            raise

    def _verify_sealed_extension_projection(self) -> None:
        """Delegate custody and byte verification to the shared authority."""

        sealed = self._sealed_extension_set
        if sealed is None:
            raise OperatorError("sealed extension projection is incomplete")
        try:
            sealed.verify()
        except (OSError, ValueError) as exc:
            raise OperatorError(str(exc)) from exc

    def _load_reviewed_extensions(self, connection: Any) -> None:
        """Load exact sealed httpfs+Quack bytes and verify DuckDB's mapping."""

        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_extension_projection import (
            ConfiguredBoardExtensionProjectionError,
        )

        sealed = self._ensure_extension_projection()
        extension_pins = {
            "httpfs": self._owner.get("pinned_httpfs_extension") or {},
            "quack": self._owner.get("pinned_extension") or {},
        }
        try:
            with sealed.load_guard():
                connection.execute("LOAD httpfs")
                connection.execute("LOAD quack")
                observed_rows = connection.execute(
                    "SELECT extension_name, install_path, extension_version "
                    "FROM duckdb_extensions() "
                    "WHERE extension_name IN ('httpfs', 'quack') "
                    "AND installed AND loaded ORDER BY extension_name"
                ).fetchall()
        except ConfiguredBoardExtensionProjectionError as exc:
            raise OperatorError(str(exc)) from exc
        observed: dict[str, tuple[Path, str]] = {}
        if len(observed_rows) != len(extension_pins):
            raise OperatorError("loaded extension set is missing or ambiguous")
        for row in observed_rows:
            if not isinstance(row, (tuple, list)) or len(row) != 3:
                raise OperatorError("loaded extension observation is malformed")
            extension_name = str(row[0] or "")
            if extension_name not in extension_pins or extension_name in observed:
                raise OperatorError("loaded extension set is missing or ambiguous")
            install_path = Path(str(row[1] or ""))
            if not install_path.is_absolute():
                raise OperatorError(
                    f"loaded {extension_name} extension path is unavailable"
                )
            observed[extension_name] = (install_path, str(row[2] or ""))
        if set(observed) != set(extension_pins):
            raise OperatorError("loaded extension set is missing or ambiguous")
        for name, pin in extension_pins.items():
            expected_path = sealed.install_paths[name]
            if observed[name] != (expected_path, pin["version"]):
                raise OperatorError(
                    f"loaded {name} extension differs from the reviewed pin"
                )

    def _open_replica_connection(self, path: Path):
        import duckdb

        sealed = self._ensure_extension_projection()
        connection = duckdb.connect(
            str(path),
            read_only=True,
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
                "enable_external_access": "true",
                "allow_unsigned_extensions": "false",
                "extension_directory": str(sealed.extension_directory),
                "threads": "1",
                "memory_limit": "256MB",
            },
        )
        try:
            self._load_reviewed_extensions(connection)
            connection.execute("SET autoload_known_extensions = false")
            connection.execute("SET enable_external_access = false")
            connection.execute("SET lock_configuration = true")
            settings = connection.execute(
                "SELECT current_setting('autoinstall_known_extensions'), "
                "current_setting('autoload_known_extensions'), "
                "current_setting('enable_external_access'), "
                "current_setting('allow_unsigned_extensions'), "
                "current_setting('lock_configuration')"
            ).fetchone()
            if settings != (False, False, False, False, True):
                raise OperatorError("read-only Quack replica policy did not lock exactly")
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _copy_replica(source: Path, target: Path) -> Mapping[str, Any]:
        source = source.resolve()
        target = target.resolve()
        if source == target or source.parent != target.parent:
            raise OperatorError("Quack replica target is not a confined sibling")
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        target_fd = -1
        digest = hashlib.sha256()
        size = 0
        try:
            source_stat = os.fstat(source_fd)
            if not source_stat.st_size or source_stat.st_size > 8 * 1024**3:
                raise OperatorError("canonical store exceeds the replica copy bound")
            target_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_fd, view)
                    view = view[written:]
            if size != source_stat.st_size:
                raise OperatorError("canonical store changed during replica copy")
            os.fsync(target_fd)
            os.close(target_fd)
            target_fd = -1
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.chmod(target, 0o600)
            return {
                "authority": "non_authoritative_read_replica",
                "path": str(target),
                "source_database_path": str(source),
                "sha256": digest.hexdigest(),
                "size_bytes": size,
            }
        finally:
            os.close(source_fd)
            if target_fd >= 0:
                os.close(target_fd)
            temporary.unlink(missing_ok=True)

    def _stop_replica(self) -> None:
        connection = self._replica_connection
        if connection is None:
            return
        try:
            if self._serve_uri:
                connection.execute("SELECT * FROM quack_stop(?)", [self._serve_uri])
        finally:
            connection.close()
            self._replica_connection = None

    @staticmethod
    def _remove_regular_extension_projection(projection_parent: Path | None) -> None:
        if projection_parent is None:
            return
        try:
            if (
                not projection_parent.is_absolute()
                or projection_parent.parent != Path("/tmp")
                or not projection_parent.name.startswith(
                    "sawm-quack-extension-projection-"
                )
            ):
                return
            for current, directories, files in os.walk(projection_parent):
                current_path = Path(current)
                os.chmod(current_path, 0o700)
                for name in directories:
                    candidate = current_path / name
                    if not candidate.is_symlink():
                        os.chmod(candidate, 0o700)
                for name in files:
                    candidate = current_path / name
                    if not candidate.is_symlink():
                        os.chmod(candidate, 0o600)
            shutil.rmtree(projection_parent)
        except OSError:
            return

    def _remove_extension_projection(self) -> None:
        sealed = self._sealed_extension_set
        projection_parent = self._extension_projection_parent
        self._sealed_extension_set = None
        self._extension_projection_parent = None
        if sealed is not None:
            sealed.close()
        self._remove_regular_extension_projection(projection_parent)

    def refresh(
        self,
        writer,
        *,
        probe: bool = True,
    ) -> Mapping[str, Any]:
        """Refresh the served replica and optionally verify it over Quack.

        ``QuackStateServer.start`` publishes canonical owner identity rows only
        after ``transport.start`` returns.  Its initial refresh therefore must
        not create a client socket: the operator immediately replaces that
        pre-identity replica, and a connected socket can keep the loopback port
        unavailable during the required stop/rebind.  The post-identity refresh
        remains probed and is the only refresh allowed to report ``live``.
        """

        database = Path(writer.path).resolve()
        replica = database.with_name(
            f"{database.stem}.read-replica{database.suffix}"
        )
        self._stop_replica()
        writer.execute("CHECKPOINT")
        observation = dict(self._copy_replica(database, replica))
        connection = self._open_replica_connection(replica)
        try:
            connection.execute(
                "SELECT * FROM quack_serve(?, token := ?, "
                "allow_other_hostname := false, disable_ssl := true)",
                [self._serve_uri, self._owner_token],
            )
        except BaseException:
            connection.close()
            raise
        self._replica_connection = connection
        self._replica_path = replica
        self._refresh_sequence += 1
        observation.update(
            {
                "refresh_sequence": self._refresh_sequence,
                "serve_started": True,
                "probe_performed": probe,
                "live": False,
                **self._server_identity,
            }
        )
        if probe:
            self._probe()
            observation["live"] = True
        return MappingProxyType(observation)

    def start(self, connection, *, host: str, port: int, token: str, identity):
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import listen_uri

        uri = listen_uri(host, port)
        self._serve_uri = uri
        self._owner_token = token
        self._server_identity = {
            "server_id": identity.server_id,
            "store_id": identity.store_id,
            "database_uuid": identity.database_uuid,
            "schema_revision": identity.schema_revision,
            "schema_fingerprint": identity.schema_fingerprint,
            "generation": identity.generation,
            "process_birth_id": identity.process_birth_id,
            "listen_uri": uri,
        }
        return self.refresh(connection, probe=False)

    def _probe(self) -> None:
        if not self._serve_uri:
            raise OperatorError("Quack replica transport has not started")
        import duckdb

        last_error: Exception | None = None
        deadline = time.monotonic() + 3.0
        sealed = self._ensure_extension_projection()
        while True:
            client = duckdb.connect(
                ":memory:",
                config={
                    "autoinstall_known_extensions": "false",
                    "autoload_known_extensions": "false",
                    "enable_external_access": "true",
                    "allow_unsigned_extensions": "false",
                    "extension_directory": str(sealed.extension_directory),
                    "lock_configuration": "true",
                },
            )
            try:
                self._load_reviewed_extensions(client)
                try:
                    rows = client.execute(
                        "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := true)",
                        [self._serve_uri, "SELECT count(*) FROM tasks", self._owner_token],
                    ).fetchall()
                    if len(rows) != 1:
                        raise OperatorError("Quack replica probe returned an invalid row count")
                    return
                except Exception as exc:
                    last_error = exc
            finally:
                client.close()
            if time.monotonic() >= deadline:
                raise OperatorError(
                    "authenticated Quack replica probe failed: "
                    f"{type(last_error).__name__ if last_error else 'unknown'}"
                ) from last_error
            time.sleep(0.05)

    def live_query(self, connection, *, identity, token: str):
        del connection
        if token != self._owner_token:
            raise OperatorError("Quack readiness token differs from the owner token")
        self._probe()
        return _remote_owner_identity(
            self._serve_uri,
            token,
            identity.to_dict(),
        )

    def stop(self, connection=None) -> None:
        del connection
        try:
            self._stop_replica()
        finally:
            self._remove_extension_projection()
            self._serve_uri = ""
            self._server_identity = {}


def _process_mutation_inbox(server: Any, *, max_requests: int = 32) -> None:
    """Service closed, atomic protocol-2 bundles on the exclusive writer."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
        mutation_binding_from_identity,
        service_mutation_inbox,
    )

    identity = server.identity
    connection = getattr(server, "_connection", None)
    vault = getattr(server, "_vault", None)
    transport = getattr(server, "transport", None)
    if identity is None or connection is None or vault is None or transport is None:
        raise OperatorError("Quack mutation inbox requires the live exclusive owner")
    token = vault.resolve(identity.secret_handle)
    inbox = Path(server.config.state_dir) / "mutations"
    service_mutation_inbox(
        connection,
        inbox=inbox,
        binding=mutation_binding_from_identity(identity),
        token=token,
        refresh_replica=lambda: transport.refresh(connection),
        max_requests=max_requests,
    )


def _serve_sawm_owner(server: Any) -> dict[str, Any]:
    stop_requested = {"value": False}

    def handle_signal(_signum: int, _frame: Any) -> None:
        stop_requested["value"] = True

    previous_int = signal.signal(signal.SIGINT, handle_signal)
    previous_term = signal.signal(signal.SIGTERM, handle_signal)
    try:
        control = server.stop_control_path()
        while server.lifecycle.value == "ready" and not stop_requested["value"]:
            if control.is_file():
                break
            _process_mutation_inbox(server, max_requests=32)
            time.sleep(0.05)
        return server.stop()
    except BaseException:
        server.stop()
        raise
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


def _validate_offline_quack_start(
    config: Mapping[str, Any],
    config_path: Path = CONFIG_PATH,
) -> Mapping[str, Any]:
    """Revalidate committed controls and the exact store before native LOAD."""

    dependency = _validator(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "validate_dependencies",
    )
    board = _validator(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "validate_program",
    )
    if dependency.get("valid") is not True or board.get("valid") is not True:
        raise OperatorError("Quack start requires valid dependency and board seals")
    materializer = _materializer()
    population = materializer.build_population(REPO_ROOT)
    materializer._assert_committed_clean_source(REPO_ROOT, population)
    if "json_emission_normalization_successor_materialization" in config:
        raise OperatorError(
            "M34 requires the exact live generation-29 owner; restart is not authorized"
        )
    if "live_preflight_contract_successor_materialization" in config:
        raise OperatorError(
            "M33 requires the exact live generation-29 owner; restart is not authorized"
        )
    if "live_preflight_plan_anchor_successor_materialization" in config:
        raise OperatorError(
            "M32 requires the exact live generation-29 owner; restart is not authorized"
        )
    if "detached_coordinator_pid_recovery_successor_materialization" in config:
        active_materialization = _active_source_repair_materialization(config)
        try:
            admitted = materializer._check_m31_prestart_admission(REPO_ROOT, config)
        except Exception as exc:
            raise OperatorError(
                "M31 stopped generation-28 restart is not admissible"
            ) from exc
        if (
            admitted.get("valid") is not True
            or admitted.get("prior_generation") != 28
            or admitted.get("target_generation") != 29
            or admitted.get("prior_event_watermark") != 281
            or admitted.get("stale_pid_projection_verified") is not True
            or admitted.get("stale_pid_projection_quarantined") is not False
        ):
            raise OperatorError("M31 prestart admission report differs")
        return MappingProxyType(
            {
                "dependency_valid": True,
                "board_valid": True,
                "prior_authority": active_materialization,
                "store": admitted,
            }
        )
    if "stopped_owner_restart_source_seal_successor_materialization" in config:
        active_materialization = _active_source_repair_materialization(config)
        try:
            admitted = materializer._check_m30_prestart_admission(REPO_ROOT, config)
        except Exception as exc:
            raise OperatorError(
                "M30 stopped generation-27 restart is not admissible"
            ) from exc
        if (
            admitted.get("valid") is not True
            or admitted.get("prior_generation") != 27
            or admitted.get("target_generation") != 28
            or admitted.get("prior_event_watermark") != 280
        ):
            raise OperatorError("M30 prestart admission report differs")
        return MappingProxyType(
            {
                "dependency_valid": True,
                "board_valid": True,
                "prior_authority": active_materialization,
                "store": admitted,
            }
        )
    if _successor_materialization_configured(config):
        active_materialization = _active_source_repair_materialization(config)
        checked = materializer.check_materialized(REPO_ROOT, config_path)
        if checked.get("valid") is not True:
            raise OperatorError(
                "current append-only successor authority does not verify"
            )
        _require_active_final_pair_marker(
            config,
            active_materialization,
            materializer,
            checked=checked,
        )
        return MappingProxyType(
            {
                "dependency_valid": True,
                "board_valid": True,
                # M14's valid historical check report predates the explicit
                # prior_authority member.  The sealed active authority is the
                # fail-closed fallback; never index an optional report key.
                "prior_authority": checked.get(
                    "prior_authority", active_materialization
                ),
                "store": checked,
            }
        )
    validation_digest = materializer._identity(
        {
            "dependency": dependency,
            "board": board,
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    prior = materializer._verify_prior_store(REPO_ROOT, config, population)
    verified = materializer._verify_store(
        REPO_ROOT / str(config["database_program"]["store_id"]),
        population,
        require_operator_complete=True,
        require_migration=True,
        migration_config=config,
        expected_validation_digest=validation_digest,
    )
    return MappingProxyType(
        {
            "dependency_valid": True,
            "board_valid": True,
            "prior_authority": prior,
            "store": verified,
        }
    )


@contextmanager
def _sealed_quack_native_runtime(config_path: Path) -> Iterator[Any]:
    """Hold the protected native DuckDB dependency for one Quack lifetime.

    The operator's validation environment intentionally excludes ambient user
    site packages.  Consequently a plain ``import duckdb`` may resolve an
    older validation dependency even though the board admits newer, exact
    DuckDB and extension bytes.  Reuse the configured-board native authority
    before any offline validator can import DuckDB, and retain its sealed
    descriptor until the foreground owner has stopped.
    """

    aliases = ("_duckdb", "duckdb")
    if any(name in sys.modules for name in aliases):
        raise OperatorError(
            "Quack start refuses preloaded ambient DuckDB aliases"
        )
    if any(name.startswith("LD_") for name in os.environ):
        raise OperatorError(
            "Quack native DuckDB bootstrap failed closed: ambient loader "
            "environment requires sanitized process birth"
        )
    from ipfs_accelerate_py.agent_implementation_route import (
        preload_agent_supervisor_native_dependency,
        verify_agent_supervisor_native_dependency_sealed_fd,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        _configured_board_dependency_seal_snapshot,
        _seal_configured_board_native_dependency,
        load_configured_board,
    )

    if any(name in sys.modules for name in aliases):
        raise OperatorError(
            "Quack native bootstrap imported an ambient DuckDB alias"
        )
    launch = None
    descriptor = -1
    try:
        board = load_configured_board(config_path, repo_root=REPO_ROOT)
        dependency_snapshot = _configured_board_dependency_seal_snapshot(board)
        launch = _seal_configured_board_native_dependency(
            board,
            dependency_seal_snapshot=dependency_snapshot,
        )
        descriptor = int(launch.descriptor.descriptor)
        module = preload_agent_supervisor_native_dependency(launch)
        executable = verify_agent_supervisor_native_dependency_sealed_fd(launch)
        if (
            descriptor < 3
            or executable != f"/proc/self/fd/{descriptor}"
            or sys.modules.get("_duckdb") is not module
            or sys.modules.get("duckdb") is not module
            or getattr(module, "__file__", None) != executable
            or getattr(module, "__version__", None)
            != launch.pin.distribution_version
        ):
            raise OperatorError(
                "Quack native DuckDB bootstrap identity differs from the seal"
            )
    except OperatorError:
        if descriptor >= 3:
            os.close(descriptor)
        raise
    except Exception as exc:
        if descriptor >= 3:
            os.close(descriptor)
        raise OperatorError(
            "Quack native DuckDB bootstrap failed closed"
        ) from exc

    try:
        yield module
    finally:
        try:
            verify_agent_supervisor_native_dependency_sealed_fd(launch)
        finally:
            os.close(descriptor)


def _run_quack_start(
    config: Mapping[str, Any],
    config_path: Path = CONFIG_PATH,
) -> int:
    """Validate and serve Quack under the exact protected native runtime."""

    with _sealed_quack_native_runtime(config_path):
        _validate_offline_quack_start(config, config_path)
        return _start_quack(config)


def _start_quack(config: Mapping[str, Any]) -> int:
    owner = config["quack_owner"]
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        install_datasets_authoritative_operational_schema,
    )
    server = build_server(
        database_path=REPO_ROOT / owner["database_path"],
        state_dir=REPO_ROOT / owner["state_dir"], host=str(owner["host"]),
        port=int(owner["port"]), repository_id=str(owner["repository_id"]),
        store_id=str(owner["store_id"]), allow_experimental=True,
        secret_handle=str(owner["secret_handle"]),
        # Critical authority boundary: never install the generic full schema.
        migrate=install_datasets_authoritative_operational_schema,
        connection_factory=lambda path: _owner_connection(path, owner),
        transport=_SawmQuackTransport(owner),
    )
    identity = server.start()
    try:
        # State-server identity rows are published after transport.start();
        # refresh once more so readiness resolves those canonical rows through
        # the same read-only Quack replica used by schedulers.
        server.transport.refresh(server._connection, probe=True)
        readiness = server.ready()
    except BaseException:
        server.stop()
        raise
    print(json.dumps({"identity": identity.to_dict(), "readiness": readiness}, indent=2, sort_keys=True))
    sys.stdout.flush()
    result = _serve_sawm_owner(server)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _verify_m9_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify immutable M6 contracts plus M9's exact rearmed task head."""

    try:
        materializer._verify_m6_task_projection(source, population)
    except materializer.MigrationRequired as exc:
        if str(exc) != "frozen M6 task status/revision projection differs":
            raise

    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        expected_status = (
            "completed"
            if alias == "SAWM-000"
            else "retrying"
            if alias == "SAWM-001"
            else "todo"
        )
        expected_revision = (
            2 if alias == "SAWM-000" else 10 if alias == "SAWM-001" else 2
        )
        if (
            observed is None
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M9-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-001":
            if completion_receipt != dict(_M9_REARM_RECEIPT):
                raise materializer.MigrationRequired(
                    "M9-head SAWM-001 rearm receipt differs"
                )
        elif alias != "SAWM-000" and completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M9-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)

    with source.intent._connection(write=False) as connection:
        task_revision_rows = connection.execute(
            "SELECT task_cid, revision, status, body_json "
            "FROM task_revisions ORDER BY task_cid, revision"
        ).fetchall()
        completion_receipt_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM completion_receipts"
            ).fetchone()[0]
        )
    if (
        len(task_revision_rows) != 1
        or str(task_revision_rows[0][0])
        != "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        or int(task_revision_rows[0][1]) != 10
        or str(task_revision_rows[0][2]) != "retrying"
        or json.loads(str(task_revision_rows[0][3])).get("completion_receipt")
        != dict(_M9_REARM_RECEIPT)
        or completion_receipt_count != 1
    ):
        raise materializer.MigrationRequired(
            "M9-head task revision/completion projection differs"
        )
    return statuses, revisions, receipt_cids


def _verify_m11_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M11's rearmed head and the complete immutable M9/M10 history."""

    try:
        materializer._verify_m6_task_projection(source, population)
    except materializer.MigrationRequired as exc:
        if str(exc) != "frozen M6 task status/revision projection differs":
            raise

    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        expected_status = (
            "completed"
            if alias == "SAWM-000"
            else "retrying"
            if alias == "SAWM-001"
            else "todo"
        )
        expected_revision = (
            2 if alias == "SAWM-000" else 13 if alias == "SAWM-001" else 2
        )
        if (
            observed is None
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M11-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-001":
            if completion_receipt != dict(_M11_REARM_RECEIPT):
                raise materializer.MigrationRequired(
                    "M11-head SAWM-001 rearm receipt differs"
                )
        elif alias != "SAWM-000" and completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M11-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)

    with source.intent._connection(write=False) as connection:
        task_revision_rows = connection.execute(
            "SELECT task_cid, revision, status, body_json "
            "FROM task_revisions ORDER BY task_cid, revision"
        ).fetchall()
        completion_receipt_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM completion_receipts"
            ).fetchone()[0]
        )
    expected_task_cid = (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    )
    expected_history = (
        (10, "retrying", dict(_M9_REARM_RECEIPT)),
        (11, "in_progress", dict(_M10_CLAIM_RECEIPT)),
        (12, "blocked", dict(_M10_FAILURE_RECEIPT)),
        (13, "retrying", dict(_M11_REARM_RECEIPT)),
    )
    observed_history = tuple(
        (
            int(row[1]),
            str(row[2]),
            json.loads(str(row[3])).get("completion_receipt"),
        )
        for row in task_revision_rows
        if str(row[0]) == expected_task_cid
    )
    if (
        len(task_revision_rows) != len(expected_history)
        or any(str(row[0]) != expected_task_cid for row in task_revision_rows)
        or observed_history != expected_history
        or completion_receipt_count != 1
    ):
        raise materializer.MigrationRequired(
            "M11-head task revision/completion history differs"
        )
    return statuses, revisions, receipt_cids


def _verify_m12_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M12's rearmed head and immutable revisions 10 through 16."""

    try:
        materializer._verify_m6_task_projection(source, population)
    except materializer.MigrationRequired as exc:
        if str(exc) != "frozen M6 task status/revision projection differs":
            raise

    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        expected_status = (
            "completed"
            if alias == "SAWM-000"
            else "retrying"
            if alias == "SAWM-001"
            else "todo"
        )
        expected_revision = (
            2 if alias == "SAWM-000" else 16 if alias == "SAWM-001" else 2
        )
        if (
            observed is None
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M12-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-001":
            if completion_receipt != dict(_M12_REARM_RECEIPT):
                raise materializer.MigrationRequired(
                    "M12-head SAWM-001 rearm receipt differs"
                )
        elif alias != "SAWM-000" and completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M12-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)

    with source.intent._connection(write=False) as connection:
        task_revision_rows = connection.execute(
            "SELECT task_cid, revision, status, body_json "
            "FROM task_revisions ORDER BY task_cid, revision"
        ).fetchall()
        completion_receipt_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM completion_receipts"
            ).fetchone()[0]
        )
    expected_task_cid = str(_M12_LIVE_FAILURE_RECEIPT["task_cid"])
    expected_history = (
        (10, "retrying", dict(_M9_REARM_RECEIPT)),
        (11, "in_progress", dict(_M10_CLAIM_RECEIPT)),
        (12, "blocked", dict(_M10_FAILURE_RECEIPT)),
        (13, "retrying", dict(_M11_REARM_RECEIPT)),
        (14, "in_progress", dict(_M12_LIVE_CLAIM_RECEIPT)),
        (15, "blocked", dict(_M12_LIVE_FAILURE_RECEIPT)),
        (16, "retrying", dict(_M12_REARM_RECEIPT)),
    )
    observed_history = tuple(
        (
            int(row[1]),
            str(row[2]),
            json.loads(str(row[3])).get("completion_receipt"),
        )
        for row in task_revision_rows
        if str(row[0]) == expected_task_cid
    )
    if (
        len(task_revision_rows) != len(expected_history)
        or any(str(row[0]) != expected_task_cid for row in task_revision_rows)
        or observed_history != expected_history
        or completion_receipt_count != 1
    ):
        raise materializer.MigrationRequired(
            "M12-head task revision/completion history differs"
        )
    return statuses, revisions, receipt_cids


def _verify_m14_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify the exact accepted M13 task head before M14 dispatch."""

    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    task_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(
            alias,
            ("todo", 2),
        )
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M14-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M14-head operator completion receipt differs"
                )
        elif alias in {"SAWM-001", "SAWM-002"}:
            validation = (
                completion_receipt.get("validation")
                if isinstance(completion_receipt, Mapping)
                else None
            )
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("operation") != "database_complete"
                or not isinstance(validation, Mapping)
                or validation.get("task_cid") != task_cid
                or validation.get("outcome") != "passed"
            ):
                raise materializer.MigrationRequired(
                    f"M14-head accepted completion differs: {alias}"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M14-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
        task_cids[alias] = task_cid

    with source.intent._connection(write=False) as connection:
        task_revision_rows = connection.execute(
            "SELECT task_cid, revision, status FROM task_revisions "
            "ORDER BY task_cid, revision"
        ).fetchall()
        completion_task_cids = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT task_cid FROM completion_receipts ORDER BY task_cid"
            ).fetchall()
        )
        semantic_authority_digest = materializer._semantic_authority_digest_on(
            connection
        )
    expected_history = {
        task_cids["SAWM-001"]: (
            (10, "retrying"),
            (11, "in_progress"),
            (12, "blocked"),
            (13, "retrying"),
            (14, "in_progress"),
            (15, "blocked"),
            (16, "retrying"),
            (17, "in_progress"),
            (18, "completed"),
        ),
        task_cids["SAWM-002"]: (
            (3, "in_progress"),
            (4, "completed"),
        ),
    }
    observed_history = {
        task_cid: tuple(
            (int(row[1]), str(row[2]))
            for row in task_revision_rows
            if str(row[0]) == task_cid
        )
        for task_cid in expected_history
    }
    expected_completion_task_cids = tuple(
        sorted(
            (
                task_cids["SAWM-000"],
                task_cids["SAWM-001"],
                task_cids["SAWM-002"],
            )
        )
    )
    authority = materializer._expected_m14_stale_owner_restart_authority()
    if (
        len(task_revision_rows) != 11
        or set(str(row[0]) for row in task_revision_rows)
        != set(expected_history)
        or observed_history != expected_history
        or completion_task_cids != expected_completion_task_cids
        or semantic_authority_digest
        != authority["prior_semantic_authority_digest"]
    ):
        raise materializer.MigrationRequired(
            "M14-head task revision/completion authority differs"
        )
    return statuses, revisions, receipt_cids


def _m22_legacy_completion_is_compatible(
    materializer: Any,
    *,
    task_alias: str,
    task_cid: str,
    completion_receipt: Any,
    operational_validation_revision: Any,
) -> bool:
    """Admit only the five exact frozen pre-field completion receipts.

    The database completion receipts predate the top-level
    ``worker_self_approval`` field.  Absence is compatible only when the
    separately stored operator-control operational-validation receipt is
    present, rehashes exactly, and carries the fail-closed authority facts.
    This is deliberately not a general missing-field default.
    """

    try:
        authority = (
            materializer
            ._expected_m22_live_preflight_receipt_compatibility_authority()
        )
    except Exception:
        return False
    completion_cids = authority.get("legacy_completion_receipt_cids")
    operational_cids = authority.get(
        "legacy_operational_validation_receipt_cids"
    )
    if (
        task_alias not in {f"SAWM-{number:03d}" for number in range(1, 6)}
        or not isinstance(completion_cids, Mapping)
        or not isinstance(operational_cids, Mapping)
        or not isinstance(completion_receipt, Mapping)
        or not isinstance(operational_validation_revision, Mapping)
        or "worker_self_approval" in completion_receipt
        or completion_receipt.get("operation") != "database_complete"
        or materializer._identity(dict(completion_receipt))
        != completion_cids.get(task_alias)
    ):
        return False
    validation = completion_receipt.get("validation")
    if (
        not isinstance(validation, Mapping)
        or validation.get("validator") != "DatabasePortalExecutionBridge@1"
        or validation.get("outcome") != "passed"
        or validation.get("task_cid") != task_cid
    ):
        return False
    transition = validation.get("accepted_source_transition")
    if task_alias in {"SAWM-003", "SAWM-004", "SAWM-005"}:
        if (
            not isinstance(transition, Mapping)
            or transition.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@1"
            or transition.get("task_alias") != task_alias
            or transition.get("database_task_cid") != task_cid
            or transition.get("worker_self_approval") is not False
            or transition.get("task_completion_authority") is not False
        ):
            return False
    elif transition is not None:
        return False
    operational = dict(operational_validation_revision)
    claimed = str(operational.pop("receipt_cid", ""))
    return bool(
        claimed
        and claimed == operational_cids.get(task_alias)
        and claimed == materializer._identity(operational)
        and operational_validation_revision.get("schema")
        == "sawm/operational-validation-revision@1"
        and operational_validation_revision.get("authority_class")
        == "operator_control_plane"
        and operational_validation_revision.get("task_alias") == task_alias
        and operational_validation_revision.get("task_cid") == task_cid
        and operational_validation_revision.get("accepted_definition_changes") == 0
        and operational_validation_revision.get("accepted_completion_changes") == 0
        and operational_validation_revision.get("historical_definition_rewritten")
        is False
        and operational_validation_revision.get("worker_self_approval") is False
    )


def _m23_portal_completion_is_compatible(
    materializer: Any,
    *,
    task_alias: str,
    task_cid: str,
    completion_receipt: Any,
    operational_validation_revision: Any,
) -> bool:
    """Validate post-M22 portal completions without granting them authority."""

    if (
        not isinstance(completion_receipt, Mapping)
        or "worker_self_approval" in completion_receipt
        or completion_receipt.get("operation") != "database_complete"
        or not isinstance(operational_validation_revision, Mapping)
    ):
        return False
    validation = completion_receipt.get("validation")
    transition = validation.get("accepted_source_transition") if isinstance(validation, Mapping) else None
    if (
        not isinstance(validation, Mapping)
        or validation.get("validator") != "DatabasePortalExecutionBridge@1"
        or validation.get("outcome") != "passed"
        or validation.get("task_cid") != task_cid
        or not isinstance(transition, Mapping)
        or transition.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@1"
        or transition.get("task_alias") != task_alias
        or transition.get("database_task_cid") != task_cid
        or transition.get("worker_self_approval") is not False
        or transition.get("task_completion_authority") is not False
    ):
        return False
    operational = dict(operational_validation_revision)
    claimed = str(operational.pop("receipt_cid", ""))
    return bool(
        claimed
        and claimed == materializer._identity(operational)
        and operational_validation_revision.get("schema")
        == "sawm/operational-validation-revision@1"
        and operational_validation_revision.get("authority_class")
        == "operator_control_plane"
        and operational_validation_revision.get("task_alias") == task_alias
        and operational_validation_revision.get("task_cid") == task_cid
        and operational_validation_revision.get("accepted_definition_changes") == 0
        and operational_validation_revision.get("accepted_completion_changes") == 0
        and operational_validation_revision.get("historical_definition_rewritten")
        is False
        and operational_validation_revision.get("worker_self_approval") is False
    )


def _verify_m34_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    authority: Mapping[str, Any],
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M34's evidence-only JSON-emission normalization target."""

    materializer._validated_m34_live_preflight_contract(authority)
    head = materializer._inspect_m34_live_projection(
        source,
        population,
        authority,
        expected_event_watermark=_M34_TARGET_EVENT_WATERMARK,
        expected_projection_cid=expected_projection_cid,
    )
    if (
        head.get("event_watermark") != _M34_TARGET_EVENT_WATERMARK
        or expected_projection_cid != _M34_TARGET_PROJECTION_CID
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M34 live head projection differs")
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        if observed is None:
            raise materializer.MigrationRequired(f"M34 task is missing: {alias}")
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m33_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    authority: Mapping[str, Any],
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M33's evidence-only target and normalized preflight contract."""

    materializer._validated_m33_live_preflight_contract(authority)
    head = materializer._inspect_m33_live_projection(
        source,
        population,
        authority,
        expected_event_watermark=_M33_TARGET_EVENT_WATERMARK,
        expected_projection_cid=expected_projection_cid,
    )
    if (
        head.get("event_watermark") != _M33_TARGET_EVENT_WATERMARK
        or expected_projection_cid != _M33_TARGET_PROJECTION_CID
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M33 live head projection differs")
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        if observed is None:
            raise materializer.MigrationRequired(f"M33 task is missing: {alias}")
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m32_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    authority: Mapping[str, Any],
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M32's evidence-only target and unchanged task heads."""

    head = materializer._inspect_m32_live_projection(
        source,
        population,
        authority,
        expected_event_watermark=_M32_TARGET_EVENT_WATERMARK,
        expected_projection_cid=expected_projection_cid,
    )
    if (
        head.get("event_watermark") != _M32_TARGET_EVENT_WATERMARK
        or expected_projection_cid != _M32_TARGET_PROJECTION_CID
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M32 live head projection differs")
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        if observed is None:
            raise materializer.MigrationRequired(f"M32 task is missing: {alias}")
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m31_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    authority: Mapping[str, Any],
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M31's evidence-only target while preserving task heads."""

    head = materializer._inspect_m31_live_projection(
        source,
        population,
        authority,
        expected_event_watermark=_M31_TARGET_EVENT_WATERMARK,
        expected_projection_cid=expected_projection_cid,
    )
    if (
        head.get("event_watermark") != _M31_TARGET_EVENT_WATERMARK
        or expected_projection_cid != _M31_TARGET_PROJECTION_CID
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M31 live head projection differs")
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        if observed is None:
            raise materializer.MigrationRequired(f"M31 task is missing: {alias}")
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m30_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    authority: Mapping[str, Any],
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M30's evidence-only target while preserving event-280 heads."""

    head = materializer._inspect_m30_live_projection(
        source,
        population,
        authority,
        expected_event_watermark=_M30_TARGET_EVENT_WATERMARK,
        expected_projection_cid=expected_projection_cid,
    )
    if (
        head.get("event_watermark") != _M30_TARGET_EVENT_WATERMARK
        or expected_projection_cid != _M30_TARGET_PROJECTION_CID
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M30 live head projection differs")
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        if observed is None:
            raise materializer.MigrationRequired(f"M30 task is missing: {alias}")
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m29_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    authority: Mapping[str, Any],
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M29's evidence-only target while preserving exact task heads."""

    head = materializer._inspect_m28_live_projection(
        source,
        population,
        authority,
        expected_event_watermark=_M29_TARGET_EVENT_WATERMARK,
        expected_projection_cid=expected_projection_cid,
    )
    if (
        head.get("event_watermark") != _M29_TARGET_EVENT_WATERMARK
        or expected_projection_cid != _M29_TARGET_PROJECTION_CID
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M29 live head projection differs")
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        if observed is None:
            raise materializer.MigrationRequired(f"M29 task is missing: {alias}")
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m28_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    authority: Mapping[str, Any],
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify the evidence-only M28 target while preserving task heads."""

    head = materializer._inspect_m28_live_projection(
        source,
        population,
        authority,
        expected_event_watermark=_M28_TARGET_EVENT_WATERMARK,
        expected_projection_cid=expected_projection_cid,
    )
    if (
        head.get("event_watermark") != _M28_TARGET_EVENT_WATERMARK
        or expected_projection_cid != _M28_TARGET_PROJECTION_CID
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M28 live head projection differs")
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        observed = source.get_task(str(expected["task_cid"]))
        if observed is None:
            raise materializer.MigrationRequired(f"M28 task is missing: {alias}")
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m27_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M27 expired four dead claims and rearmed only two tasks."""

    head = materializer._inspect_m27_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or head.get("event_watermark") != _M27_TARGET_EVENT_WATERMARK
        or not expected_projection_cid
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M27 live head projection differs")
    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-006": ("retrying", 4),
        "SAWM-007": ("completed", 7),
        "SAWM-008": ("retrying", 8),
        "SAWM-010": ("completed", 4),
        "SAWM-011": ("completed", 4),
        "SAWM-012": ("retrying", 7),
        "SAWM-015": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(alias, ("todo", 2))
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M27-head task status/revision differs: {alias}"
            )
        completion = observed.body.get("completion_receipt")
        operational = observed.body.get("operational_validation_revision")
        if alias == "SAWM-000":
            if (
                not isinstance(completion, Mapping)
                or completion.get("schema") != "sawm/operator-bootstrap-completion@1"
                or completion.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M27-head operator completion differs"
                )
        elif alias in {"SAWM-001", "SAWM-002", "SAWM-003", "SAWM-004", "SAWM-005"}:
            if not _m22_legacy_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M27-head legacy completion differs: {alias}"
                )
        elif alias in {"SAWM-007", "SAWM-010", "SAWM-011"}:
            if not _m23_portal_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M27-head accepted portal completion differs: {alias}"
                )
        elif alias in {"SAWM-006", "SAWM-012"}:
            if completion != materializer._m27_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    f"M27-head operator interruption rearm differs: {alias}"
                )
        elif alias == "SAWM-008":
            if completion != materializer._m26_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    "M27-head historical SAWM-008 rearm differs"
                )
        elif alias == "SAWM-015":
            if completion != materializer._m23_task_rearm_receipt():
                raise materializer.MigrationRequired(
                    "M27-head historical SAWM-015 rearm differs"
                )
        elif completion is not None:
            raise materializer.MigrationRequired(
                f"M27-head unaccepted completion exists: {alias}"
            )
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = observed.status
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m26_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M26's exact two-expiry/two-rearm task head."""

    head = materializer._inspect_m26_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or head.get("event_watermark") != _M26_TARGET_EVENT_WATERMARK
        or not expected_projection_cid
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M26 live head projection differs")
    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-006": ("todo", 2),
        "SAWM-007": ("completed", 7),
        "SAWM-008": ("retrying", 8),
        "SAWM-010": ("completed", 4),
        "SAWM-011": ("completed", 4),
        "SAWM-012": ("retrying", 5),
        "SAWM-015": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(alias, ("todo", 2))
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M26-head task status/revision differs: {alias}"
            )
        completion = observed.body.get("completion_receipt")
        operational = observed.body.get("operational_validation_revision")
        if alias == "SAWM-000":
            if (
                not isinstance(completion, Mapping)
                or completion.get("schema") != "sawm/operator-bootstrap-completion@1"
                or completion.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M26-head operator completion differs"
                )
        elif alias in {"SAWM-001", "SAWM-002", "SAWM-003", "SAWM-004", "SAWM-005"}:
            if not _m22_legacy_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M26-head legacy completion differs: {alias}"
                )
        elif alias in {"SAWM-007", "SAWM-010", "SAWM-011"}:
            if not _m23_portal_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M26-head accepted portal completion differs: {alias}"
                )
        elif alias in {"SAWM-008", "SAWM-012"}:
            if completion != materializer._m26_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    f"M26-head operator rearm differs: {alias}"
                )
        elif alias == "SAWM-015":
            if completion != materializer._m23_task_rearm_receipt():
                raise materializer.MigrationRequired(
                    "M26-head historical SAWM-015 rearm differs"
                )
        elif completion is not None:
            raise materializer.MigrationRequired(
                f"M26-head unaccepted completion exists: {alias}"
            )
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = observed.status
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m25_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M25's source-only successor preserved M24's exact task head."""

    head = materializer._inspect_m25_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or head.get("event_watermark") != _M25_TARGET_EVENT_WATERMARK
        or head.get("plan_revision") != _M25_TARGET_PLAN_REVISION
        or not expected_projection_cid
        or head.get("projection_cid") != expected_projection_cid
        or set(head.get("tasks") or {})
    ):
        raise materializer.MigrationRequired("M25 live head projection differs")
    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-007": ("completed", 7),
        "SAWM-008": ("retrying", 5),
        "SAWM-011": ("completed", 4),
        "SAWM-015": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(
            alias, ("todo", 2)
        )
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M25-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        operational = observed.body.get("operational_validation_revision")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M25-head operator completion receipt differs"
                )
        elif alias in {
            "SAWM-001",
            "SAWM-002",
            "SAWM-003",
            "SAWM-004",
            "SAWM-005",
        }:
            if not _m22_legacy_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion_receipt,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M25-head legacy completion differs: {alias}"
                )
        elif alias in {"SAWM-007", "SAWM-011"}:
            if not _m23_portal_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion_receipt,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M25-head portal completion differs: {alias}"
                )
        elif alias == "SAWM-015":
            if completion_receipt != materializer._m23_task_rearm_receipt():
                raise materializer.MigrationRequired(
                    "M25-head historical SAWM-015 rearm differs"
                )
        elif alias == "SAWM-008":
            if completion_receipt != materializer._m24_task_rearm_receipt():
                raise materializer.MigrationRequired(
                    "M25-head historical SAWM-008 rearm differs"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M25-head unaccepted completion receipt exists: {alias}"
            )
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m24_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M24's exact settled/rearmed head through live Quack only."""

    head = materializer._inspect_m24_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or head.get("event_watermark") != _M24_TARGET_EVENT_WATERMARK
        or head.get("plan_revision") != _M24_TARGET_PLAN_REVISION
        or not expected_projection_cid
        or head.get("projection_cid") != expected_projection_cid
        or set(head.get("tasks") or {}) != {"SAWM-008"}
    ):
        raise materializer.MigrationRequired("M24 live head projection differs")
    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-007": ("completed", 7),
        "SAWM-008": ("retrying", 5),
        "SAWM-011": ("completed", 4),
        "SAWM-015": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(alias, ("todo", 2))
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M24-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        operational = observed.body.get("operational_validation_revision")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M24-head operator completion receipt differs"
                )
        elif alias in {"SAWM-001", "SAWM-002", "SAWM-003", "SAWM-004", "SAWM-005"}:
            if not _m22_legacy_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion_receipt,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M24-head legacy completion differs: {alias}"
                )
        elif alias in {"SAWM-007", "SAWM-011"}:
            if not _m23_portal_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion_receipt,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M24-head portal completion differs: {alias}"
                )
        elif alias == "SAWM-015":
            if completion_receipt != materializer._m23_task_rearm_receipt():
                raise materializer.MigrationRequired(
                    "M24-head historical SAWM-015 rearm differs"
                )
        elif alias == "SAWM-008":
            if completion_receipt != materializer._m24_task_rearm_receipt():
                raise materializer.MigrationRequired(
                    "M24-head SAWM-008 operator rearm differs"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M24-head unaccepted completion receipt exists: {alias}"
            )
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m23_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M23's exact settled/rearmed head through live Quack only."""

    head = materializer._inspect_m23_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or head.get("event_watermark") != _M23_TARGET_EVENT_WATERMARK
        or head.get("plan_revision") != _M23_TARGET_PLAN_REVISION
        or not expected_projection_cid
        or head.get("projection_cid") != expected_projection_cid
        or set(head.get("tasks") or {}) != {"SAWM-015"}
    ):
        raise materializer.MigrationRequired("M23 live head projection differs")
    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-007": ("completed", 7),
        "SAWM-011": ("completed", 4),
        "SAWM-015": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(alias, ("todo", 2))
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M23-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        operational = observed.body.get("operational_validation_revision")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M23-head operator completion receipt differs"
                )
        elif alias in {"SAWM-001", "SAWM-002", "SAWM-003", "SAWM-004", "SAWM-005"}:
            if not _m22_legacy_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion_receipt,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M23-head legacy completion differs: {alias}"
                )
        elif alias in {"SAWM-007", "SAWM-011"}:
            if not _m23_portal_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion_receipt,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M23-head portal completion differs: {alias}"
                )
        elif alias == "SAWM-015":
            if completion_receipt != materializer._m23_task_rearm_receipt():
                raise materializer.MigrationRequired(
                    "M23-head SAWM-015 operator rearm differs"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M23-head unaccepted completion receipt exists: {alias}"
            )
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m22_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M22's frozen task head through its bounded compatibility rule."""

    head = materializer._inspect_m22_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or int(head.get("event_watermark") or 0)
        != _M22_TARGET_EVENT_WATERMARK
        or int(head.get("plan_revision") or 0) != _M22_TARGET_PLAN_REVISION
        or not expected_projection_cid
        or head.get("projection_cid") != expected_projection_cid
        or set(head.get("tasks") or {}) != {"SAWM-007"}
    ):
        raise materializer.MigrationRequired("M22 live head projection differs")
    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-007": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(alias, ("todo", 2))
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M22-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        operational = observed.body.get("operational_validation_revision")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M22-head operator completion receipt differs"
                )
        elif alias in {
            "SAWM-001", "SAWM-002", "SAWM-003", "SAWM-004", "SAWM-005"
        }:
            if not _m22_legacy_completion_is_compatible(
                materializer,
                task_alias=alias,
                task_cid=task_cid,
                completion_receipt=completion_receipt,
                operational_validation_revision=operational,
            ):
                raise materializer.MigrationRequired(
                    f"M22-head accepted completion differs: {alias}"
                )
        elif alias == "SAWM-007":
            if completion_receipt != materializer._m18_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    "M22-head SAWM-007 operator rearm differs"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M22-head unaccepted completion receipt exists: {alias}"
            )
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m21_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M21's generation-realization head without inventing task state."""

    head = materializer._inspect_m21_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or int(head.get("event_watermark") or 0)
        != _M21_TARGET_EVENT_WATERMARK
        or int(head.get("plan_revision") or 0) != _M21_TARGET_PLAN_REVISION
        or not expected_projection_cid
        or head.get("projection_cid") != expected_projection_cid
        or set(head.get("tasks") or {}) != {"SAWM-007"}
    ):
        raise materializer.MigrationRequired("M21 live head projection differs")
    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-007": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(alias, ("todo", 2))
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M21-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M21-head operator completion receipt differs"
                )
        elif alias in {
            "SAWM-001",
            "SAWM-002",
            "SAWM-003",
            "SAWM-004",
            "SAWM-005",
        }:
            validation = (
                completion_receipt.get("validation")
                if isinstance(completion_receipt, Mapping)
                else None
            )
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("operation") != "database_complete"
                or not isinstance(validation, Mapping)
                or validation.get("task_cid") != task_cid
                or validation.get("outcome") != "passed"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    f"M21-head accepted completion differs: {alias}"
                )
        elif alias == "SAWM-007":
            if completion_receipt != materializer._m18_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    "M21-head SAWM-007 operator rearm differs"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M21-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m20_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
    *,
    expected_projection_cid: str,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M20's source-only head while preserving the M18 rearm."""

    head = materializer._inspect_m20_head_task_projection(source, population)
    if (
        int(head.get("event_watermark") or 0) != _M20_TARGET_EVENT_WATERMARK
        or int(head.get("plan_revision") or 0) != _M20_TARGET_PLAN_REVISION
        or not expected_projection_cid
        or head.get("projection_cid") != expected_projection_cid
    ):
        raise materializer.MigrationRequired("M20 live head projection differs")
    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-007": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(alias, ("todo", 2))
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M20-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M20-head operator completion receipt differs"
                )
        elif alias in {"SAWM-001", "SAWM-002", "SAWM-003", "SAWM-004", "SAWM-005"}:
            validation = (
                completion_receipt.get("validation")
                if isinstance(completion_receipt, Mapping)
                else None
            )
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("operation") != "database_complete"
                or not isinstance(validation, Mapping)
                or validation.get("task_cid") != task_cid
                or validation.get("outcome") != "passed"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    f"M20-head accepted completion differs: {alias}"
                )
        elif alias == "SAWM-007":
            if completion_receipt != materializer._m18_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    "M20-head SAWM-007 operator rearm differs"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M20-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _expected_live_projection_cid(
    active_source_repair: Mapping[str, Any],
    final_pair_marker: Mapping[str, Any],
) -> str:
    """Resolve the exact live projection from authority or its closed marker."""

    return str(
        active_source_repair.get("target_projection_cid")
        or final_pair_marker.get("projection_cid")
        or ""
    )


def _verify_m19_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M19's source-only head, preserving M18's SAWM-007 rearm."""

    head = materializer._inspect_m19_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or int(head.get("event_watermark") or 0)
        != _M19_TARGET_EVENT_WATERMARK
        or not str(head.get("projection_cid") or "")
        or int(head.get("plan_revision") or 0)
        != _M19_TARGET_PLAN_REVISION
        or set(head.get("tasks") or {}) != {"SAWM-007"}
    ):
        raise materializer.MigrationRequired("M19 live head projection differs")

    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-007": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(alias, ("todo", 2))
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M19-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M19-head operator completion receipt differs"
                )
        elif alias in {"SAWM-001", "SAWM-002", "SAWM-003", "SAWM-004", "SAWM-005"}:
            validation = (
                completion_receipt.get("validation")
                if isinstance(completion_receipt, Mapping)
                else None
            )
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("operation") != "database_complete"
                or not isinstance(validation, Mapping)
                or validation.get("task_cid") != task_cid
                or validation.get("outcome") != "passed"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    f"M19-head accepted completion differs: {alias}"
                )
        elif alias == "SAWM-007":
            if completion_receipt != materializer._m18_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    "M19-head SAWM-007 operator rearm differs"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M19-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids

def _verify_m18_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M18's accepted M17 history plus sole SAWM-007 rearm."""

    head = materializer._inspect_m18_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or int(head.get("event_watermark") or 0)
        != _M18_TARGET_EVENT_WATERMARK
        or head.get("projection_cid") != _M18_TARGET_PROJECTION_CID
        or int(head.get("plan_revision") or 0)
        != _M18_TARGET_PLAN_REVISION
        or set(head.get("tasks") or {}) != {"SAWM-007"}
    ):
        raise materializer.MigrationRequired("M18 live head projection differs")

    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("completed", 7),
        "SAWM-004": ("completed", 7),
        "SAWM-005": ("completed", 4),
        "SAWM-007": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(alias, ("todo", 2))
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M18-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M18-head operator completion receipt differs"
                )
        elif alias in {"SAWM-001", "SAWM-002", "SAWM-003", "SAWM-004", "SAWM-005"}:
            validation = (
                completion_receipt.get("validation")
                if isinstance(completion_receipt, Mapping)
                else None
            )
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("operation") != "database_complete"
                or not isinstance(validation, Mapping)
                or validation.get("task_cid") != task_cid
                or validation.get("outcome") != "passed"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    f"M18-head accepted completion differs: {alias}"
                )
        elif alias == "SAWM-007":
            if completion_receipt != materializer._m18_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    "M18-head SAWM-007 operator rearm differs"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M18-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m17_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M17's source-only head without replaying the live authority."""

    head = materializer._inspect_m17_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or int(head.get("event_watermark") or 0)
        != _M17_TARGET_EVENT_WATERMARK
        or head.get("projection_cid") != _M17_TARGET_PROJECTION_CID
        or int(head.get("plan_revision") or 0)
        != _M17_TARGET_PLAN_REVISION
    ):
        raise materializer.MigrationRequired("M17 live head projection differs")

    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("retrying", 5),
        "SAWM-004": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(
            alias,
            ("todo", 2),
        )
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M17-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M17-head operator completion receipt differs"
                )
        elif alias in {"SAWM-001", "SAWM-002"}:
            validation = (
                completion_receipt.get("validation")
                if isinstance(completion_receipt, Mapping)
                else None
            )
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("operation") != "database_complete"
                or not isinstance(validation, Mapping)
                or validation.get("task_cid") != task_cid
                or validation.get("outcome") != "passed"
            ):
                raise materializer.MigrationRequired(
                    f"M17-head accepted completion differs: {alias}"
                )
        elif alias in {"SAWM-003", "SAWM-004"}:
            if completion_receipt != materializer._m16_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    f"M17-head preserved operator rearm differs: {alias}"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M17-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _verify_m16_live_head_task_projection(
    source: Any,
    population: Mapping[str, Any],
    materializer: Any,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M16's exact all-task head after both operator rearms."""

    head = materializer._inspect_m16_head_task_projection(source, population)
    if (
        not isinstance(head, Mapping)
        or int(head.get("event_watermark") or 0)
        != _M16_TARGET_EVENT_WATERMARK
        or head.get("projection_cid") != _M16_TARGET_PROJECTION_CID
        or int(head.get("plan_revision") or 0)
        != _M16_TARGET_PLAN_REVISION
        or set(head.get("tasks") or {}) != {"SAWM-003", "SAWM-004"}
    ):
        raise materializer.MigrationRequired("M16 live head projection differs")

    expected_heads = {
        "SAWM-000": ("completed", 2),
        "SAWM-001": ("completed", 18),
        "SAWM-002": ("completed", 4),
        "SAWM-003": ("retrying", 5),
        "SAWM-004": ("retrying", 5),
    }
    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        task_cid = str(expected["task_cid"])
        observed = source.get_task(task_cid)
        expected_status, expected_revision = expected_heads.get(
            alias,
            ("todo", 2),
        )
        if (
            observed is None
            or observed.task_cid != task_cid
            or observed.status != expected_status
            or int(observed.revision) != expected_revision
        ):
            raise materializer.MigrationRequired(
                f"M16-head task status/revision differs: {alias}"
            )
        completion_receipt = observed.body.get("completion_receipt")
        if alias == "SAWM-000":
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("schema")
                != "sawm/operator-bootstrap-completion@1"
                or completion_receipt.get("worker_self_approval") is not False
            ):
                raise materializer.MigrationRequired(
                    "M16-head operator completion receipt differs"
                )
        elif alias in {"SAWM-001", "SAWM-002"}:
            validation = (
                completion_receipt.get("validation")
                if isinstance(completion_receipt, Mapping)
                else None
            )
            if (
                not isinstance(completion_receipt, Mapping)
                or completion_receipt.get("operation") != "database_complete"
                or not isinstance(validation, Mapping)
                or validation.get("task_cid") != task_cid
                or validation.get("outcome") != "passed"
            ):
                raise materializer.MigrationRequired(
                    f"M16-head accepted completion differs: {alias}"
                )
        elif alias in {"SAWM-003", "SAWM-004"}:
            if completion_receipt != materializer._m16_task_rearm_receipt(alias):
                raise materializer.MigrationRequired(
                    f"M16-head operator rearm receipt differs: {alias}"
                )
        elif completion_receipt is not None:
            raise materializer.MigrationRequired(
                f"M16-head unaccepted completion receipt exists: {alias}"
            )
        operational = observed.body.get("operational_validation_revision")
        if alias != "SAWM-000" and isinstance(operational, Mapping):
            receipt_cids[alias] = str(operational.get("receipt_cid") or "")
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    return statuses, revisions, receipt_cids


def _preserved_m27_plan_anchor(
    config: Mapping[str, Any],
    active_source_repair: Mapping[str, Any],
    materializer: Any,
) -> Mapping[str, str]:
    """Resolve the unchanged M27 plan anchor for evidence-only successors."""

    m29_key = "committed_evidence_verification_successor_materialization"
    if m29_key in config:
        observed = config.get(m29_key)
        expected = materializer._expected_m29_committed_evidence_verification_authority()
        if not isinstance(observed, Mapping) or dict(observed) != expected:
            raise OperatorError("preserved M29 plan authority differs")
        prior = expected.get("prior_authority")
    else:
        prior = active_source_repair.get("prior_authority")
    if not isinstance(prior, Mapping):
        raise OperatorError("preserved M27 plan authority is unavailable")
    source_binding = str(prior.get("plan_source_binding_cid") or "")
    migration_digest = str(prior.get("plan_migration_digest") or "")
    digest_pattern = re.compile(r"sha256:[0-9a-f]{64}\Z")
    if (
        digest_pattern.fullmatch(source_binding) is None
        or digest_pattern.fullmatch(migration_digest) is None
    ):
        raise OperatorError("preserved M27 plan authority is malformed")
    return MappingProxyType(
        {
            "plan_source_binding_cid": source_binding,
            "plan_migration_digest": migration_digest,
        }
    )


def _normalized_live_preflight_contract(
    active_source_repair: Mapping[str, Any],
    materializer: Any,
) -> Mapping[str, Any]:
    """Resolve one closed preflight view without shape-dependent aliases."""

    revision = str(active_source_repair.get("migration_revision") or "")
    if revision == "SAWM-R2-M34":
        try:
            return materializer._validated_m34_live_preflight_contract(
                active_source_repair
            )
        except (
            materializer.MigrationRequired,
            materializer.MaterializationError,
        ) as exc:
            raise OperatorError(
                f"M34 normalized preflight contract differs: {exc}"
            ) from exc
    if revision == "SAWM-R2-M33":
        try:
            return materializer._validated_m33_live_preflight_contract(
                active_source_repair
            )
        except (
            materializer.MigrationRequired,
            materializer.MaterializationError,
        ) as exc:
            raise OperatorError(
                f"M33 normalized preflight contract differs: {exc}"
            ) from exc
    runtime = active_source_repair.get("runtime_binding")
    owner = active_source_repair.get("live_owner")
    prior = active_source_repair.get("prior_authority")
    if not isinstance(runtime, Mapping):
        runtime = MappingProxyType({})
    if not isinstance(owner, Mapping):
        owner = MappingProxyType({})
    if not isinstance(prior, Mapping):
        prior = MappingProxyType({})
    database_uuid = str(
        active_source_repair.get("prior_database_uuid")
        or runtime.get("database_uuid")
        or owner.get("database_uuid")
        or ""
    )
    semantic_digest = str(
        prior.get("semantic_authority_digest")
        or active_source_repair.get("prior_semantic_authority_digest")
        or active_source_repair.get("target_semantic_authority_digest")
        or ""
    )
    contract = {
        "schema": "sawm/live-preflight-contract@compat",
        "migration_revision": revision,
        "successor_class": "historical",
        "target_store_id": str(active_source_repair.get("target_store_id") or ""),
        "database_uuid": database_uuid,
        "target_generation": int(active_source_repair.get("target_generation") or 0),
        "target_event_watermark": int(
            active_source_repair.get("target_event_watermark") or 0
        ),
        "target_plan_revision": int(
            active_source_repair.get("target_plan_revision") or 0
        ),
        "target_projection_cid": str(
            active_source_repair.get("target_projection_cid") or ""
        ),
        "semantic_authority_digest": semantic_digest,
        "preserved_plan_anchor": dict(
            active_source_repair.get("preserved_plan_anchor") or {}
        ),
        "expected_task_heads": dict(
            active_source_repair.get("expected_task_heads") or {}
        ),
    }
    if (
        not contract["migration_revision"]
        or not contract["target_store_id"]
        or not contract["database_uuid"]
        or contract["target_generation"] < 1
        or contract["target_event_watermark"] < 1
        or contract["target_plan_revision"] < 1
    ):
        raise OperatorError("historical live-preflight contract is incomplete")
    return MappingProxyType(contract)


def _live_preflight(
    config: Mapping[str, Any],
    *,
    probe_provider: bool = False,
    retire_provider_token_handoff: bool = False,
    before_token_handoff_retirement: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    if probe_provider and not retire_provider_token_handoff:
        raise OperatorError(
            "provider probe requires prior retirement of the token handoff"
        )
    if (
        before_token_handoff_retirement is not None
        and not retire_provider_token_handoff
    ):
        raise OperatorError(
            "pre-retirement coordinator reservation requires handoff retirement"
        )
    dependency = _validator("scripts/validate_semantic_addressed_world_model_dependencies.py", "validate_dependencies")
    board = _validator("scripts/validate_semantic_addressed_world_model_board.py", "validate_program")
    if dependency.get("valid") is not True or board.get("valid") is not True:
        raise OperatorError("sealed dependency or board validation failed")
    materializer = _materializer()
    population = materializer.build_population(REPO_ROOT)
    active_source_repair = _active_source_repair_materialization(config)
    preflight_contract = _normalized_live_preflight_contract(
        active_source_repair, materializer
    )
    validation_digest = materializer._identity(
        {
            "dependency": dependency,
            "board": board,
            "program_definition_cid": population["program_definition_cid"],
        }
    )
    active_revision = str(active_source_repair.get("migration_revision") or "")
    m34_active = active_revision == "SAWM-R2-M34"
    m33_active = active_revision == "SAWM-R2-M33"
    m32_active = active_revision == "SAWM-R2-M32"
    m31_active = active_revision == "SAWM-R2-M31"
    m30_active = active_revision == "SAWM-R2-M30"
    m29_active = active_revision == "SAWM-R2-M29"
    m28_active = active_revision == "SAWM-R2-M28"
    m27_active = active_revision == "SAWM-R2-M27"
    m26_active = active_revision == "SAWM-R2-M26"
    m25_active = active_revision == "SAWM-R2-M25"
    m24_active = active_revision == "SAWM-R2-M24"
    m23_active = active_revision == "SAWM-R2-M23"
    m22_active = active_revision == "SAWM-R2-M22"
    m21_active = active_revision == "SAWM-R2-M21"
    m20_active = active_revision == "SAWM-R2-M20"
    m19_active = active_revision == "SAWM-R2-M19"
    m18_active = active_revision == "SAWM-R2-M18"
    evidence_only_post_m27 = any(
        (
            m34_active,
            m33_active,
            m32_active,
            m31_active,
            m30_active,
            m29_active,
            m28_active,
        )
    )
    deferred_live_evidence_marker = any(
        (m34_active, m33_active, m32_active, m31_active, m30_active, m29_active)
    )
    final_pair_marker: Mapping[str, Any] = MappingProxyType({})
    if not m18_active and not deferred_live_evidence_marker:
        final_pair_marker = _require_active_final_pair_marker(
            config,
            active_source_repair,
            materializer,
        )
    expected_event_cursor = int(preflight_contract["target_event_watermark"])
    expected_projection_cid = str(preflight_contract["target_projection_cid"])
    expected_plan_revision = int(preflight_contract["target_plan_revision"])
    store = REPO_ROOT / config["database_program"]["store_id"]

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QUACK_ENDPOINT_ENV,
        QUACK_MUTATION_BINDING_ENV,
        QUACK_STORE_ID_ENV,
        QUACK_TOKEN_ENV,
        discover_live_quack_endpoint,
    )
    discovery = discover_live_quack_endpoint(store)
    expected_uri = str(config["database_program"]["quack_endpoint"])
    if not discovery.uri or discovery.uri != expected_uri or not discovery.token:
        raise OperatorError(f"exact live Quack owner unavailable: {discovery.reason}")
    try:
        owner_status = json.loads(Path(discovery.status_path).read_text(encoding="utf-8"))
        live_identity = owner_status["identity"]
        live_generation = int(live_identity["generation"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperatorError("live Quack generation is not bound by its status identity") from exc
    expected_generation = int(config["database_program"]["store_generation"])
    if live_generation != expected_generation:
        raise OperatorError(
            f"live Quack generation {live_generation} differs from the sealed "
            f"generation {expected_generation}"
        )
    remote_identity = _remote_owner_identity(
        discovery.uri,
        discovery.token,
        live_identity,
    )
    expected_store_id = str(preflight_contract["target_store_id"])
    expected_database_uuid = str(preflight_contract["database_uuid"])
    if (
        remote_identity.get("canonical_rows_verified") is not True
        or remote_identity.get("live") is not True
        or any(
            live_identity.get(key) != remote_identity.get(key)
            for key in (
                "server_id",
                "store_id",
                "database_uuid",
                "process_birth_id",
                "listen_uri",
                "extension_fingerprint",
                "schema_revision",
                "generation",
                "credential_generation",
            )
        )
        or live_identity.get("store_id") != expected_store_id
        or remote_identity.get("store_id") != expected_store_id
        or live_identity.get("database_uuid") != expected_database_uuid
        or remote_identity.get("database_uuid") != expected_database_uuid
        or int(remote_identity.get("generation") or 0) != expected_generation
        or live_identity.get("listen_uri") != expected_uri
        or remote_identity.get("listen_uri") != expected_uri
    ):
        raise OperatorError(
            "live Quack owner identity is not bound to the sealed successor store"
        )
    # Resolve the opaque secret handle only into this coordinator environment.
    # No raw token is printed, put in argv, receipts or provider inputs.
    os.environ[QUACK_ENDPOINT_ENV] = discovery.uri
    os.environ[QUACK_STORE_ID_ENV] = str(config["database_program"]["store_id"])
    os.environ[QUACK_TOKEN_ENV] = discovery.token
    os.environ["IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION"] = str(live_generation)
    mutation_binding = {
        "server_id": str(live_identity["server_id"]),
        "store_id": str(live_identity["store_id"]),
        "database_uuid": str(live_identity["database_uuid"]),
        "schema_revision": int(live_identity["schema_revision"]),
        "schema_fingerprint": str(live_identity["schema_fingerprint"]),
        "generation": live_generation,
        "process_birth_id": str(live_identity["process_birth_id"]),
        "listen_uri": str(live_identity["listen_uri"]),
        "extension_fingerprint": str(
            live_identity.get("extension_fingerprint") or "none"
        ),
    }
    os.environ[QUACK_MUTATION_BINDING_ENV] = json.dumps(
        mutation_binding, sort_keys=True, separators=(",", ":")
    )
    os.environ["IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"] = str(
        (REPO_ROOT / config["quack_owner"]["state_dir"] / "mutations").resolve()
    )
    os.environ["SAWM_QUACK_TOKEN"] = discovery.token
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    try:
        live = DatabaseTaskSource(
            discovery.uri,
            install_schema=False,
            repository_tree_id=population["repository_tree_id"],
            plan_root_cid=population["plan_root_cid"],
            owner_id="sawm-r2-live-preflight",
        )
    except Exception:
        raise OperatorError("authenticated live Quack preflight open failed") from None
    try:
        if m34_active:
            try:
                m34_verified = materializer._verify_m34_live_materialization(
                    live,
                    live_identity,
                    population,
                    config,
                    active_source_repair,
                    validation_digest,
                )
                expected_m34_receipt = (
                    materializer._expected_m34_source_successor_receipt(
                        population,
                        active_source_repair,
                        validation_digest,
                        m34_verified,
                    )
                )
                final_pair_marker = _require_active_final_pair_marker(
                    config,
                    active_source_repair,
                    materializer,
                    checked={
                        "valid": True,
                        "receipt": expected_m34_receipt,
                        **m34_verified,
                    },
                )
            except (
                materializer.MigrationRequired,
                materializer.MaterializationError,
            ) as exc:
                raise OperatorError(
                    f"M34 exact JSON-emission source seal failed: {exc}"
                ) from exc
        elif m33_active:
            try:
                m33_verified = materializer._verify_m33_live_materialization(
                    live,
                    live_identity,
                    population,
                    config,
                    active_source_repair,
                    validation_digest,
                )
                expected_m33_receipt = (
                    materializer._expected_m33_source_successor_receipt(
                        population,
                        active_source_repair,
                        validation_digest,
                        m33_verified,
                    )
                )
                final_pair_marker = _require_active_final_pair_marker(
                    config,
                    active_source_repair,
                    materializer,
                    checked={
                        "valid": True,
                        "receipt": expected_m33_receipt,
                        **m33_verified,
                    },
                )
            except (
                materializer.MigrationRequired,
                materializer.MaterializationError,
            ) as exc:
                raise OperatorError(
                    f"M33 exact live-preflight contract source seal failed: {exc}"
                ) from exc
        elif m32_active:
            try:
                m32_verified = materializer._verify_m32_live_materialization(
                    live,
                    live_identity,
                    population,
                    config,
                    active_source_repair,
                    validation_digest,
                )
                expected_m32_receipt = (
                    materializer._expected_m32_source_successor_receipt(
                        population,
                        active_source_repair,
                        validation_digest,
                        m32_verified,
                    )
                )
                final_pair_marker = _require_active_final_pair_marker(
                    config,
                    active_source_repair,
                    materializer,
                    checked={
                        "valid": True,
                        "receipt": expected_m32_receipt,
                        **m32_verified,
                    },
                )
            except (
                materializer.MigrationRequired,
                materializer.MaterializationError,
            ) as exc:
                raise OperatorError(
                    f"M32 exact live-preflight source seal failed: {exc}"
                ) from exc
        elif m31_active:
            try:
                m31_verified = materializer._verify_m31_live_materialization(
                    live,
                    live_identity,
                    population,
                    config,
                    active_source_repair,
                    validation_digest,
                )
                expected_m31_receipt = (
                    materializer._expected_m31_source_successor_receipt(
                        population,
                        active_source_repair,
                        validation_digest,
                        m31_verified,
                    )
                )
                final_pair_marker = _require_active_final_pair_marker(
                    config,
                    active_source_repair,
                    materializer,
                    checked={
                        "valid": True,
                        "receipt": expected_m31_receipt,
                        **m31_verified,
                    },
                )
            except (
                materializer.MigrationRequired,
                materializer.MaterializationError,
            ) as exc:
                raise OperatorError(
                    f"M31 exact detached-coordinator source seal failed: {exc}"
                ) from exc
        elif m30_active:
            try:
                m30_verified = materializer._verify_m30_live_materialization(
                    live,
                    live_identity,
                    population,
                    config,
                    active_source_repair,
                    validation_digest,
                )
                expected_m30_receipt = (
                    materializer._expected_m30_source_successor_receipt(
                        population,
                        active_source_repair,
                        validation_digest,
                        m30_verified,
                    )
                )
                final_pair_marker = _require_active_final_pair_marker(
                    config,
                    active_source_repair,
                    materializer,
                    checked={
                        "valid": True,
                        "receipt": expected_m30_receipt,
                        **m30_verified,
                    },
                )
            except (
                materializer.MigrationRequired,
                materializer.MaterializationError,
            ) as exc:
                raise OperatorError(
                    f"M30 exact stopped-owner source seal failed: {exc}"
                ) from exc
        elif m29_active:
            try:
                materializer._assert_m29_m28_receipt_absent(
                    REPO_ROOT, active_source_repair
                )
                m29_verified = materializer._verify_m29_live_materialization(
                    live,
                    live_identity,
                    population,
                    config,
                    active_source_repair,
                    validation_digest,
                )
                coordination = (
                    REPO_ROOT
                    / str(
                        active_source_repair["runtime_binding"][
                            "coordination_store_id"
                        ]
                    )
                ).resolve()
                m29_verified.update(
                    materializer._inspect_m28_coordination_projection(
                        coordination, active_source_repair
                    )
                )
                expected_m29_receipt = (
                    materializer._expected_m29_source_successor_receipt(
                        population,
                        active_source_repair,
                        validation_digest,
                        m29_verified,
                    )
                )
                final_pair_marker = _require_active_final_pair_marker(
                    config,
                    active_source_repair,
                    materializer,
                    checked={
                        "valid": True,
                        "receipt": expected_m29_receipt,
                        **m29_verified,
                    },
                )
            except (
                materializer.MigrationRequired,
                materializer.MaterializationError,
            ) as exc:
                raise OperatorError(
                    f"M29 exact committed-evidence verification failed: {exc}"
                ) from exc
        elif m18_active:
            final_pair_marker = _require_active_final_pair_marker(
                config,
                active_source_repair,
                materializer,
                live_source=live,
                validation_digest=validation_digest,
                live_identity=live_identity,
                remote_identity=remote_identity,
            )
        live_snapshot = live.snapshot().to_dict()
        if (
            live_snapshot["task_count"] != 45
            or live_snapshot["goal_count"] != 29
            or live_snapshot["dependency_count"] != 136
            or live_snapshot["plan_count"] != 1
            or live_snapshot["event_cursor"] != expected_event_cursor
            or (
                expected_projection_cid
                and live_snapshot["projection_cid"] != expected_projection_cid
            )
            or live_snapshot["plan_root_cid"] != population["plan_root_cid"]
        ):
            raise OperatorError("live Quack snapshot differs from the exact program root/counts")
        try:
            if "json_emission_normalization_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m34_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        authority=active_source_repair,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "live_preflight_contract_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m33_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        authority=active_source_repair,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "live_preflight_plan_anchor_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m32_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        authority=active_source_repair,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "detached_coordinator_pid_recovery_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m31_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        authority=active_source_repair,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "stopped_owner_restart_source_seal_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m30_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        authority=active_source_repair,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "committed_evidence_verification_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m29_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        authority=active_source_repair,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "live_claim_admission_recovery_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m28_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        authority=active_source_repair,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "dead_owner_parallel_resume_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m27_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "automatic_stall_recovery_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m26_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "native_duckdb_preload_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m25_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "multi_lane_sidecar_reopen_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m24_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "multi_lane_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m23_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif (
                "live_preflight_receipt_compatibility_successor_materialization"
                in config
            ):
                statuses, _revisions, _receipts = (
                    _verify_m22_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "generation_realization_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m21_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "test_isolation_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m20_live_head_task_projection(
                        live,
                        population,
                        materializer,
                        expected_projection_cid=expected_projection_cid,
                    )
                )
            elif "live_catalog_inventory_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m19_live_head_task_projection(
                        live,
                        population,
                        materializer,
                    )
                )
            elif "portal_completion_persistence_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m18_live_head_task_projection(
                        live,
                        population,
                        materializer,
                    )
                )
            elif "source_binding_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m17_live_head_task_projection(
                        live,
                        population,
                        materializer,
                    )
                )
            elif "accepted_source_retry_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m16_live_head_task_projection(
                        live,
                        population,
                        materializer,
                    )
                )
            elif "stale_owner_restart_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m14_head_task_projection(
                        live,
                        population,
                        materializer,
                    )
                )
            elif "quack_refresh_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m12_head_task_projection(
                        live,
                        population,
                        materializer,
                    )
                )
            elif "declared_output_retry_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m12_head_task_projection(
                        live,
                        population,
                        materializer,
                    )
                )
            elif "live_provider_retry_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m11_head_task_projection(
                        live,
                        population,
                        materializer,
                    )
                )
            elif "live_projection_successor_materialization" in config:
                statuses, _revisions, _receipts = (
                    _verify_m9_head_task_projection(
                        live,
                        population,
                        materializer,
                    )
                )
            else:
                statuses, _revisions, _receipts = (
                    materializer._verify_m6_task_projection(live, population)
                )
            with live.intent._connection(write=False) as connection:
                semantic_authority_digest = (
                    materializer._semantic_authority_digest_on(connection)
                )
        except (
            materializer.MigrationRequired,
            materializer.MaterializationError,
        ) as exc:
            raise OperatorError(
                f"live Quack task authority conflict: {exc}"
            ) from exc
        expected_semantic_authority_digest = (
            str(preflight_contract["semantic_authority_digest"])
            if evidence_only_post_m27
            else str(final_pair_marker.get("semantic_authority_digest") or "")
            if (m27_active or m26_active or m25_active or m24_active or m23_active)
            else
            active_source_repair["prior_semantic_authority_digest"]
            if "stale_owner_restart_successor_materialization" in config
            else active_source_repair["target_semantic_authority_digest"]
            if (
                "live_preflight_receipt_compatibility_successor_materialization"
                in config
                or "generation_realization_successor_materialization" in config
                or "test_isolation_successor_materialization" in config
                or "live_catalog_inventory_successor_materialization" in config
                or "portal_completion_persistence_successor_materialization" in config
                or "source_binding_successor_materialization" in config
                or "accepted_source_retry_successor_materialization" in config
                or "quack_refresh_successor_materialization" in config
                or "declared_output_retry_successor_materialization" in config
                or "live_provider_retry_successor_materialization" in config
                or "live_projection_successor_materialization" in config
            )
            else active_source_repair["prior_semantic_authority_digest"]
        )
        if (
            semantic_authority_digest != expected_semantic_authority_digest
            or (
                evidence_only_post_m27
                and final_pair_marker.get("semantic_authority_digest")
                != expected_semantic_authority_digest
            )
        ):
            raise OperatorError("live Quack semantic task authority differs")
        for expected in population["objectives"]:
            observed = live.get_goal(expected["goal_cid"])
            body = observed.get("body") if isinstance(observed, Mapping) else {}
            if (
                observed is None
                or observed.get("goal_cid") != expected["goal_cid"]
                or observed.get("goal_alias") != expected["goal_id"]
                or observed.get("objective_id")
                != str(expected.get("objective_id") or "")
                or observed.get("parent_goal_cid")
                != expected["parent_goal_cid"]
                or int(observed.get("ordinal") or 0) != int(expected["ordinal"])
                or observed.get("title") != expected["title"]
                or observed.get("status") != expected["status"]
                or int(observed.get("revision") or 0) != 1
                or body.get("definition_cid") != expected["definition_cid"]
                or body.get("definition") != expected["definition"]
                or materializer._identity(body.get("definition"))
                != expected["definition_cid"]
            ):
                raise OperatorError(f"live Quack goal definition conflict: {expected['goal_id']}")
        live_plan = live.plans.get(str(population["plan_root_cid"]))
        live_plan_body = (
            live_plan.get("body") if isinstance(live_plan, Mapping) else {}
        )
        if m34_active or m33_active:
            contract_anchor = preflight_contract["preserved_plan_anchor"]
            preserved_plan_anchor = MappingProxyType(
                {
                    "plan_source_binding_cid": str(
                        contract_anchor["current_source_binding_cid"]
                    ),
                    "plan_migration_digest": str(
                        contract_anchor["source_migration_digest"]
                    ),
                }
            )
        elif evidence_only_post_m27:
            preserved_plan_anchor = _preserved_m27_plan_anchor(
                config, active_source_repair, materializer
            )
        else:
            preserved_plan_anchor = MappingProxyType({})
        expected_source_migration_digest = (
            materializer._identity(
                materializer._m27_migration_body(
                    population,
                    config,
                    validation_digest,
                )
            )
            if m27_active
            else
            materializer._identity(
                materializer._m26_migration_body(
                    population,
                    config,
                    validation_digest,
                )
            )
            if m26_active
            else
            materializer._identity(
                materializer._m25_migration_body(
                    population,
                    config,
                    validation_digest,
                )
            )
            if m25_active
            else materializer._identity(
                materializer._m24_migration_body(
                    population,
                    config,
                    validation_digest,
                )
            )
            if m24_active
            else ""
        )
        if (
            live_plan is None
            or live_plan.get("plan_cid") != population["plan_root_cid"]
            or int(live_plan.get("revision") or 0) != expected_plan_revision
            or (
                evidence_only_post_m27
                and (
                    live_plan_body.get("current_source_binding_cid")
                    != preserved_plan_anchor["plan_source_binding_cid"]
                    or live_plan_body.get("source_migration_revision")
                    != "SAWM-R2-M27"
                    or live_plan_body.get("source_migration_digest")
                    != preserved_plan_anchor["plan_migration_digest"]
                    or live_plan_body.get("supersession_mode")
                    != "append_only_stopped_run_recovery_successor"
                )
            )
            or (
                not evidence_only_post_m27
                and (
                    live_plan_body.get("current_source_binding_cid")
                    != population["source_binding"]["source_binding_cid"]
                    or live_plan_body.get("source_migration_revision")
                    != active_source_repair["migration_revision"]
                )
            )
            or (
                (m27_active or m26_active or m25_active or m24_active)
                and (
                    live_plan_body.get("source_migration_digest")
                    != expected_source_migration_digest
                    or final_pair_marker.get("migration_digest")
                    != expected_source_migration_digest
                )
            )
        ):
            raise OperatorError(
                "live Quack plan head is not bound to the current successor source"
            )
        if statuses.get("SAWM-000") not in {"completed", "complete", "done"}:
            raise OperatorError("live Quack authority lacks the SAWM-000 completion CAS")
    except Exception as exc:
        if discovery.token and discovery.token in str(exc):
            raise OperatorError("authenticated live Quack preflight failed") from None
        raise
    finally:
        try:
            live.close()
        except Exception:
            raise OperatorError("authenticated live Quack preflight close failed") from None
    store_report = {
        "valid": True, "task_count": live_snapshot["task_count"],
        "goal_count": live_snapshot["goal_count"],
        "projection_cid": live_snapshot["projection_cid"],
        "event_cursor": live_snapshot["event_cursor"],
        "store_generation": live_generation,
        "queried_through_live_quack_only": True,
        "direct_authoritative_file_opened": False,
        "statuses": statuses,
    }
    if final_pair_marker:
        source_successor_receipt_cid = str(
            final_pair_marker.get("source_successor_receipt_cid") or ""
        )
        prior_final_pair_receipt_cid = str(
            final_pair_marker.get("prior_final_pair_receipt_cid") or ""
        )
        source_successor_chain = final_pair_marker.get(
            "source_successor_chain"
        )
        store_report.update(
            {
                "coordination_path": str(
                    (
                        REPO_ROOT
                        / str(final_pair_marker["coordination_path"])
                    ).resolve()
                ),
                "coordination_projection_digest": final_pair_marker[
                    "coordination_projection_digest"
                ],
                "coordination_event_count": final_pair_marker[
                    "coordination_event_count"
                ],
                "materialization_receipt_cid": (
                    prior_final_pair_receipt_cid
                    or str(final_pair_marker["receipt_cid"])
                ),
                "final_pair_commit_marker_verified": bool(
                    prior_final_pair_receipt_cid
                    or final_pair_marker.get(
                        "receipt_is_final_pair_commit_marker"
                    )
                    is True
                ),
                "source_successor_receipt_cid": (
                    source_successor_receipt_cid or None
                ),
                "source_successor_receipt_verified": bool(
                    source_successor_receipt_cid
                    and final_pair_marker.get(
                        "receipt_is_evidence_source_seal_marker"
                    )
                    is True
                ),
                "source_successor_chain": (
                    dict(source_successor_chain)
                    if isinstance(source_successor_chain, Mapping)
                    else None
                ),
            }
        )

    credential_report: dict[str, Any] = {
        "retired": False,
        "reason": "provider_launch_not_requested",
    }
    if retire_provider_token_handoff:
        from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
            retire_token_handoff,
        )

        status_path = Path(discovery.status_path).resolve()
        expected_state_dir = (
            REPO_ROOT / str(config["quack_owner"]["state_dir"])
        ).resolve()
        if status_path.parent != expected_state_dir:
            raise OperatorError(
                "live token handoff is outside the sealed owner state directory"
            )
        if before_token_handoff_retirement is not None:
            before_token_handoff_retirement()
        credential_report = retire_token_handoff(
            state_dir=expected_state_dir,
            secret_handle=str(live_identity["secret_handle"]),
            expected_token=discovery.token,
        )
        if credential_report.get("retired") is not True:
            raise OperatorError("live token handoff retirement failed closed")

    provider_report: dict[str, Any] = {
        "probed": False,
        "reason": "deferred_until_real_launch",
    }
    if probe_provider:
        provider = config["provider"]
        from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
            DatabaseProgramConfig,
            provider_subprocess_environment,
        )
        from ipfs_accelerate_py.llm_router import probe_grok_codex_agent_route_readiness

        database_program = DatabaseProgramConfig.from_mapping(
            config["database_program"]
        )
        provider_environment = provider_subprocess_environment(
            os.environ,
            program=database_program,
        )
        if any(
            discovery.token in str(value)
            for value in provider_environment.values()
        ):
            raise OperatorError("provider probe environment retained owner credential")
        readiness = probe_grok_codex_agent_route_readiness(
            grok_model=str(provider["primary_model_id"]),
            codex_model=str(provider["fallback_model_id"]),
            codex_reasoning_effort=str(provider["fallback_reasoning_effort"]),
            environment=provider_environment,
        )
        failure = readiness.failure_kind.value if readiness.failure_kind is not None else ""
        provider_report = {**dataclasses.asdict(readiness), "failure_kind": failure, "probed": True}
        if not readiness.effective_provider:
            raise OperatorError(f"ordered provider route unavailable: {readiness.reason_code}")
        if readiness.effective_provider == "codex" and (
            provider["fallback_trigger"] != "primary_quota_exhausted"
            or failure != "grok_quota_exhausted"
        ):
            raise OperatorError("Codex fallback is not admitted by the reviewed quota-only trigger")
    return {
        "schema": "sawm/live-control-preflight@1", "valid": True,
        "dependency_valid": True, "board_valid": True,
        "store": store_report,
        "quack": {"uri": discovery.uri, "source": discovery.source,
                  "reason": discovery.reason, "token_present": True,
                  "live_query": True, "task_count": live_snapshot["task_count"],
                  "canonical_owner_rows_verified": True,
                  "provider_token_handoff": credential_report},
        "provider": provider_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "validate-dependencies", "validate-board", "materialize", "render", "check",
        "quack-start", "quack-status", "quack-ready", "quack-stop",
        "quack-recover-stale", "preflight", "dry-run",
    ):
        sub.add_parser(command)
    launch = sub.add_parser("launch")
    launch.add_argument("--foreground", action="store_true")
    launch.add_argument("--duration-seconds", type=float, default=float("inf"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
        config = _config(config_path)
        if args.command == "validate-dependencies":
            return _emit(_validator("scripts/validate_semantic_addressed_world_model_dependencies.py", "validate_dependencies"))
        if args.command == "validate-board":
            return _emit(_validator("scripts/validate_semantic_addressed_world_model_board.py", "validate_program"))
        if args.command in {"materialize", "render", "check"}:
            materializer = _materializer()
            if args.command == "render":
                return _emit({"valid": True, **materializer.build_population(REPO_ROOT)})
            if args.command == "check":
                population = materializer.build_population(REPO_ROOT)
                store = REPO_ROOT / config["database_program"]["store_id"]
                from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
                    discover_live_quack_endpoint,
                )
                discovery = discover_live_quack_endpoint(store)
                if discovery.uri:
                    return _emit({"action": "checked_live", **_live_preflight(config, probe_provider=False)})
                if _successor_materialization_configured(config):
                    active_materialization = _active_source_repair_materialization(
                        config
                    )
                    checked = materializer.check_materialized(
                        REPO_ROOT, config_path
                    )
                    _require_active_final_pair_marker(
                        config,
                        active_materialization,
                        materializer,
                        checked=checked,
                    )
                    return _emit(checked)
                materializer._assert_committed_clean_source(REPO_ROOT, population)
                dependency = materializer._validator_report(
                    REPO_ROOT,
                    "scripts/validate_semantic_addressed_world_model_dependencies.py",
                )
                board = materializer._validator_report(
                    REPO_ROOT,
                    "scripts/validate_semantic_addressed_world_model_board.py",
                )
                validation_digest = materializer._identity(
                    {
                        "dependency": dependency,
                        "board": board,
                        "program_definition_cid": population["program_definition_cid"],
                    }
                )
                prior = materializer._verify_prior_store(
                    REPO_ROOT, config, population
                )
                verified = materializer._verify_store(
                    store,
                    population,
                    require_operator_complete=True,
                    require_migration=True,
                    migration_config=config,
                    expected_validation_digest=validation_digest,
                )
                receipt = materializer._ensure_migration_receipt(
                    REPO_ROOT,
                    store,
                    population,
                    verified,
                    validation_digest,
                )
                return _emit(
                    {
                        "valid": True,
                        "action": "checked",
                        "prior_authority": prior,
                        "receipt": receipt,
                        **verified,
                    }
                )
            return _emit(materializer.materialize(REPO_ROOT, config_path))
        if args.command == "quack-start":
            return _run_quack_start(config, config_path)
        if args.command == "quack-recover-stale":
            return _emit(_recover_stale_quack(config))
        if args.command in {"quack-status", "quack-ready", "quack-stop"}:
            ops = _load_script("scripts/ops/agent_supervisor/quack_state_server.py", "_sawm_landed_quack_ops")
            return int(ops.main(_quack_args(config, args.command.removeprefix("quack-"))))

        real_launch = args.command == "launch"
        real_detached_launch = real_launch and not args.foreground
        from ipfs_accelerate_py.agent_supervisor.runtime import (
            configured_board_scheduler as scheduler_runtime,
        )

        scheduler_board = scheduler_runtime.load_configured_board(
            config_path,
            repo_root=REPO_ROOT,
        )
        coordinator_pid_reservation: Any | None = None

        def reserve_coordinator_pid_before_retirement() -> Any:
            nonlocal coordinator_pid_reservation
            if coordinator_pid_reservation is not None:
                raise OperatorError(
                    "detached coordinator PID reservation callback repeated"
                )
            coordinator_pid_reservation = (
                scheduler_runtime._reserve_detached_coordinator_pid(
                    scheduler_board
                )
            )
            return coordinator_pid_reservation

        live: dict[str, Any]
        try:
            live = _live_preflight(
                config,
                probe_provider=real_launch,
                retire_provider_token_handoff=real_launch,
                before_token_handoff_retirement=(
                    reserve_coordinator_pid_before_retirement
                    if real_detached_launch
                    else None
                ),
            )
        except BaseException:
            if (
                coordinator_pid_reservation is not None
                and coordinator_pid_reservation.state == "reserved"
            ):
                scheduler_runtime._discard_coordinator_pid_reservation(
                    coordinator_pid_reservation
                )
            raise
        scheduler_args = ["--repo-root", str(REPO_ROOT), "--config", str(config_path)]
        try:
            if args.command == "preflight":
                result = int(
                    scheduler_runtime.main([*scheduler_args, "preflight"])
                )
            else:
                launch_args = [*scheduler_args, "launch", "--implement"]
                if args.command == "dry-run":
                    launch_args.append("--dry-run")
                else:
                    if args.foreground:
                        launch_args.append("--foreground")
                    if math.isfinite(args.duration_seconds):
                        launch_args.extend(
                            ["--duration-seconds", str(args.duration_seconds)]
                        )
                result = int(
                    scheduler_runtime.main(
                        launch_args,
                        coordinator_pid_reservation=(
                            coordinator_pid_reservation
                            if real_detached_launch
                            else None
                        ),
                    )
                )
        finally:
            # Once claimed, scheduler ownership is authoritative.  The facade
            # only discards a reservation that never crossed that boundary.
            if (
                coordinator_pid_reservation is not None
                and coordinator_pid_reservation.state == "reserved"
            ):
                scheduler_runtime._discard_coordinator_pid_reservation(
                    coordinator_pid_reservation
                )
        if result:
            return result
        # Only secret-free preflight facts are emitted by this facade.
        print(json.dumps({"schema": "sawm/operator-delegation@1", "valid": True,
                          "command": args.command, "live_preflight": live}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        return _emit({"schema": "sawm/operator-error@1", "valid": False,
                      "error": _credential_safe_error(exc)})


if __name__ == "__main__":
    raise SystemExit(main())
