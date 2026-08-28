#!/usr/bin/env python3
"""Render, migrate, and verify the append-only SAWM R2 supervisor program.

The default action verifies the sealed prior authority, copies only its exact
canonical control-store bytes to a confined stage, appends the bounded source
migration, successor operational-validation revisions, and operator-authorized
settled-task recovery through the landed intent repository, and atomically
publishes the successor store. It never rewrites accepted task/goal definitions
or completion evidence. Predecessor execution sidecars are never copied; the
M9 recovery alone copies the exact failed coordination sidecar and applies one
closed operator rearm before publishing the successor pair.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NAMESPACE = "semantic-addressed-world-model-v1"
REVISION = "SAWM-PLAN-R2"
ROOT_GOAL = "SAWM-G000"
SCHEMA = "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-materialization@1"
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"

_M3_PREWORKER_FAILURE_V2: dict[str, Any] = {
    "schema": "sawm/pre-worker-launch-failure@2",
    "command": "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
    "phase": "detached_coordinator_provider_entry_module_preflight",
    "exit_code": 2,
    "outer_launch_command": (
        "python scripts/ops/agent_supervisor/"
        "semantic_addressed_world_model.py launch"
    ),
    "outer_launch_exit_code": 0,
    "error_payload": {
        "schema": "sawm/coordinator-error@1",
        "valid": False,
        "error": (
            "ModuleNotFoundError: No module named "
            "'ipfs_accelerate_py.agent_supervisor.provider_command_environment'"
        ),
    },
    "error_payload_cid": (
        "sha256:7eccdc0993160ddf88afeaa8acd66f951c525d0758c21326a6075d850b996afb"
    ),
    "source_head": "29fb3cec38d682702c952dd20111b56d993345bb",
    "source_tree": "4e3e7532d4d850edabad1610d98bab82d4fe2410",
    "configuration_root": (
        "baguqeeraime2bkkng2cuttamu3qtrqxxlp3xim7hzpkpce7sunv7ptumfkeq"
    ),
    "control_plane_admission_cid": (
        "baguqeeraug4eqqturk3lnsf2xfmg2sqrrm3ij6v7u5pnlxlcfbivzhyi6enq"
    ),
    "control_plane_capsule_id": (
        "sha256:50b6acb20e822d5a841282fbafc7aa5d32dfce9af68bca99c2107fffeb50ef9b"
    ),
    "control_plane_archive_sha256": (
        "sha256:4ea5c26e631544abc5890e28f176685957a8fb367858c8653e9aee5289a00646"
    ),
    "coordinator_pid": 3669646,
    "coordinator_pid_projection_after_failure": "absent",
    "coordinator_log_path": (
        "data/agent_supervisor/semantic_addressed_world_model/run-r2-m3/"
        "logs/configured-board-20260828T060454Z.log"
    ),
    "coordinator_log_sha256": (
        "sha256:d9932d7bb2f5d906a09a9fe9f1f28886913aec4a40dbdd803d1c17d049a8bc8b"
    ),
    "store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m3/control.duckdb"
    ),
    "owner_generation": 5,
    "owner_server_id": "server:67f0d1bb-0289-45ae-b2bc-33c87f040124",
    "owner_process_birth_id": "birth:eaf37ba44ad162fc5231eb6e4dbbf35e",
    "credential_handoff_retired": True,
    "accepted_control_plane_admitted": True,
    "provider_capability_probed": True,
    "worker_started": False,
    "task_claimed": False,
    "task_state_changed": False,
    "implementation_provider_invoked": False,
    "failure_time_authority": "unavailable",
}

_M5_FROZEN_PRIOR_BINDING: dict[str, Any] = {
    "prior_store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m5/control.duckdb"
    ),
    "prior_control_store_sha256": (
        "e2b8e8a15abe9c2a5b53dd540e9621c11a85f63fcc3799da331bb55e3bfbc408"
    ),
    "prior_event_watermark": 121,
    "prior_event_prefix_sha256": (
        "f07bb22f380f30901d0ca16af4bf6bbc3b544f039fb1f43728828d92a95a3f9e"
    ),
    "prior_projection_cid": (
        "baguqeeram7gdldpraa6twhuvfbqbeou4gefgmj5szitgb6ij2nmwobx727wq"
    ),
    "prior_database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
    "prior_generation": 7,
    "prior_plan_revision": 6,
}

_M6_EXPECTED_PROJECTION_CID = (
    "baguqeeraztioh4pdrleiio2hzt2dtxh7o7ae2jk237fvcflktepffj6dpuya"
)
_M6_OPERATIONAL_TASK_COUNT = 44
_M6_OPERATIONAL_VALIDATION_COUNT = 46
_M6_EVENT_SUFFIX_LENGTH = 47
_M6_MIGRATION_EVENT_OFFSET = 2
_M6_OPERATIONAL_FIRST_OFFSET = 3
_M6_OPERATIONAL_LAST_OFFSET = 46
_M6_RECOVERY_EVENT_OFFSET = 47
_M6_VALIDATION_TOKEN_FROM = "/home/barberb/.local/bin/python"
_M6_VALIDATION_TOKEN_TO = "python"
_M6_TASK_IDENTITY_MANIFEST_CID = (
    "sha256:902037a87e711082e57c8dcf002eab669f0dc83c35e9f372ba197a18f8acf1d3"
)
_M6_SUPERSESSION_REASON = (
    "source_authority_revision_and_settled_preprovider_task_requeue"
)
_M6_SUPERSESSION_MODE = (
    "source_authority_revision_and_operational_validation_recovery"
)

_M6_FROZEN_SUCCESSOR_BINDING: dict[str, Any] = {
    "store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m6/control.duckdb"
    ),
    "control_store_sha256": (
        "17e4dd76258b0584df1106e0324654c380fcf4f36587463a17936193a260381f"
    ),
    "control_store_size": 39_071_744,
    "event_watermark": 168,
    "event_prefix_sha256": (
        "23093117ac6a8cf15c65668e4a455673800b4d1dcbc457dddec9b0528e7f0e9e"
    ),
    "projection_cid": _M6_EXPECTED_PROJECTION_CID,
    "source_head": "9c92490e245a806e3f860ce71c61104a23c2a1b4",
    "source_tree": "d86d0ca4fd820c12316aba7c56dcd788242b5186",
    "source_binding_cid": (
        "sha256:fdf0f6ea842e5e8326a5e5f488f9b19a72983dae296081756164756346ae53b4"
    ),
    "database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
    "generation": 8,
    "plan_revision": 7,
    "process_birth_id": "birth:8f543ddf21dd6ff4e98e73eafa9c88f3",
    "server_id": "server:a1bb77fa-e5c6-427a-b151-e971a2fe0b0b",
    "migration_receipt_path": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m6/migration-receipt.json"
    ),
    "migration_receipt_cid": (
        "sha256:6fa4147b4ec8badd2872da24c77c115afeb5f6fc576c8fcbda8e0273f54bc8b3"
    ),
    "migration_receipt_file_sha256": (
        "acb938ad0454a058ad8e5f8ae60b7ad7ae7f01128e9632794b1d5ee38404c93f"
    ),
    "owner_status_path": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m6/quack-owner/quack-state-server.status.json"
    ),
    "owner_status_sha256": (
        "6f82495e11614b7fcca1d95c78ef6300b0691b76f9ac63bba19c64eee7e5326c"
    ),
    "validator_digest": (
        "sha256:c280087b49e10e9797d1a8815ed7a085ce42cdb50b7941851da16bd76a67d80d"
    ),
    "semantic_authority_digest": (
        "sha256:e08b5d3695c3c9471f5738d0efed4165a1f87987851f7263296c0cff9259cf65"
    ),
    "frozen_base_authority_digest": (
        "sha256:3b5ca340647cc6820e8827040fc96526986bf1cfc6a6e2d243e945be7a70835c"
    ),
    "append_surface_digest": (
        "sha256:ab641eb7736abc257cd4aea145d90fb5a6c4ed56eec6e4776e92e33ec60ee842"
    ),
}

_M7_EXPECTED_PROJECTION_CID = (
    "baguqeeraypw3zwt7gud33pymtc2xsjgl5ezap73jcbh4tvbzs5psea7gdspq"
)
_M7_EVENT_SUFFIX_LENGTH = 2
_M7_TARGET_EVENT_WATERMARK = 170
_M7_TARGET_PLAN_REVISION = 8
_M7_TARGET_GENERATION = 9
_M7_MIGRATION_REVISION = "SAWM-R2-M7"
_M7_SUPERSESSION_MODE = "source_only_live_task_contract_comparator_repair"
_M7_SUPERSESSION_REASON = "authenticated_live_task_contract_comparator_repair"
_M7_SOURCE_REPAIR_AUTHORITY_CID = (
    "sha256:326b6d9185ff9974050333d6c8e011ab3dce8772c322fd1143a27f926b72e468"
)

_M6_LIVE_PREFLIGHT_FAILURE: dict[str, Any] = {
    "schema": "sawm/live-control-preflight-failure@1",
    "phase": "authenticated_live_task_definition_validation",
    "command": (
        "python scripts/ops/agent_supervisor/"
        "semantic_addressed_world_model.py preflight"
    ),
    "exit_code": 2,
    "error_payload": {
        "schema": "sawm/operator-error@1",
        "valid": False,
        "error": "OperatorError: live Quack task definition conflict: SAWM-001",
    },
    "failure_kind": "historical_operational_validation_view_mismatch",
    "source_head": _M6_FROZEN_SUCCESSOR_BINDING["source_head"],
    "source_tree": _M6_FROZEN_SUCCESSOR_BINDING["source_tree"],
    "source_binding_cid": _M6_FROZEN_SUCCESSOR_BINDING["source_binding_cid"],
    "store_id": _M6_FROZEN_SUCCESSOR_BINDING["store_id"],
    "owner_generation": 8,
    "owner_server_id": _M6_FROZEN_SUCCESSOR_BINDING["server_id"],
    "owner_process_birth_id": _M6_FROZEN_SUCCESSOR_BINDING[
        "process_birth_id"
    ],
    "canonical_owner_rows_verified": True,
    "task_count_verified": 45,
    "goal_count_verified": 29,
    "provider_probed": False,
    "task_claimed": False,
    "task_state_changed": False,
    "implementation_provider_invoked": False,
    "failure_time_authority": "unavailable",
}

_M7_PREPUBLICATION_MATERIALIZATION_FAILURE: dict[str, Any] = {
    "schema": "sawm/source-materialization-failure@1",
    "attempt": "SAWM-R2-M7-A1",
    "phase": "frozen_m6_authority_verification",
    "command": (
        "python scripts/materialize_semantic_addressed_world_model_program.py "
        "materialize"
    ),
    "exit_code": 1,
    "error_payload": {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "semantic-addressed-world-model-materialization@1"
        ),
        "valid": False,
        "action": "migration_required",
        "migration_required": True,
        "error": "frozen M6 event/owner/completion evidence differs",
    },
    "failure_kind": "quack_row_value_projection_not_normalized",
    "source_head": "434402ee0fd20d876bd6dd0e75ea122ec5bd1136",
    "source_tree": "e23f272a52b8ffe56f4aa2bc94579479ea9a5777",
    "prior_store_id": _M6_FROZEN_SUCCESSOR_BINDING["store_id"],
    "prior_control_store_sha256": _M6_FROZEN_SUCCESSOR_BINDING[
        "control_store_sha256"
    ],
    "prior_event_prefix_sha256": _M6_FROZEN_SUCCESSOR_BINDING[
        "event_prefix_sha256"
    ],
    "diagnosis": {
        "row_adapter_type": "DuckDBRow",
        "direct_tuple_equality": False,
        "positional_value_projection_matches": True,
        "event_prefix_matches": True,
        "event_count_matches": True,
        "owner_values_match": True,
        "completion_values_match": True,
    },
    "frozen_m6_bytes_preserved": True,
    "target_store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m7/control.duckdb"
    ),
    "target_published": False,
    "task_claimed": False,
    "task_state_changed": False,
    "implementation_provider_invoked": False,
    "failure_time_authority": "unavailable",
}

_M7_FROZEN_SUCCESSOR_BINDING: dict[str, Any] = {
    "store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m7/control.duckdb"
    ),
    "control_store_sha256": (
        "7f25cdab2a33ba9fb9433926ab7541ac85a77dd9c7747c66a1f8bd7867e1cd4b"
    ),
    "control_store_size": 39_071_744,
    "event_watermark": 170,
    "event_prefix_sha256": (
        "a2e17a1f65391e3786fb055e664483af2615db6279b5404cdde6a1ede5ec6ef6"
    ),
    "projection_cid": _M7_EXPECTED_PROJECTION_CID,
    "source_head": "bf50aea3e332ce3c520cd97b29050c8464342548",
    "source_tree": "b77ad846057e36236fde186296de41a3603cba64",
    "source_binding_cid": (
        "sha256:acf273e8787cca8044ee6ec63d55e3f46c8f81d1fd295c35d81a692dc32d7af3"
    ),
    "database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
    "generation": 9,
    "plan_revision": 8,
    "process_birth_id": "birth:8fd3d851535249b91d23d185547a0f75",
    "server_id": "server:76aea654-573d-4626-81ff-938f9bd04273",
    "migration_receipt_path": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m7/migration-receipt.json"
    ),
    "migration_receipt_cid": (
        "sha256:f18f81498120d8e1a8aa41b24a674e2d42aa0bc4168dbbe32ec820ad7291d1e3"
    ),
    "migration_receipt_file_sha256": (
        "daf19fac056872b3b66514c6b163d98a082f6755120fb9c6c403490c888c6fa7"
    ),
    "owner_status_path": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m7/quack-owner/quack-state-server.status.json"
    ),
    "owner_status_sha256": (
        "3036aaf439cdbdd956e9882791a0136da8dc11f73440d2a315b3d0bf523df0db"
    ),
    "validator_digest": (
        "sha256:ea2638a05a4e7437dd920622712618eddc5065b7fe5228813304a8ac66b4355a"
    ),
    "semantic_authority_digest": (
        "sha256:e08b5d3695c3c9471f5738d0efed4165a1f87987851f7263296c0cff9259cf65"
    ),
    "frozen_base_authority_digest": (
        "sha256:6e4428920ae70d1fac053e9fc607a276cd06085d65725a4313912b3e3ade49b0"
    ),
    "append_surface_digest": (
        "sha256:4f87ca81a4ba869bff77f3c530f62fa8e46f6f9787b662517dd4ddda7d661960"
    ),
}

_M8_EXPECTED_PROJECTION_CID = (
    "baguqeerale774wldjbmz4wfexfzcbfrc3i4rkpsxnvsdds5f243bjp7a3iea"
)
_M8_EVENT_SUFFIX_LENGTH = 2
_M8_TARGET_EVENT_WATERMARK = 172
_M8_TARGET_PLAN_REVISION = 9
_M8_TARGET_GENERATION = 10
_M8_MIGRATION_REVISION = "SAWM-R2-M8"
_M8_SUPERSESSION_MODE = "source_only_optional_goal_projection_repair"
_M8_SUPERSESSION_REASON = "optional_nonroot_goal_objective_projection_repair"
_M8_SOURCE_REPAIR_AUTHORITY_CID = (
    "sha256:01935058ca682411743904513524b3eab41369b057151db529af14d46c5c0963"
)

_M9_EXPECTED_PROJECTION_CID = (
    "baguqeeraebsdnrj7pvgd6yp26yr6ob7ocbrfzjwg57xfjnr6sn4xfkuvkq2q"
)
_M9_EVENT_SUFFIX_LENGTH = 3
_M9_TARGET_EVENT_WATERMARK = 177
_M9_TARGET_PLAN_REVISION = 10
_M9_TARGET_GENERATION = 11
_M9_MIGRATION_REVISION = "SAWM-R2-M9"
_M9_SUPERSESSION_MODE = "append_only_live_failure_rearm"
_M9_SUPERSESSION_REASON = (
    "board_scoped_checkout_lock_and_portal_deferral_recovery"
)
_M9_LIVE_RECOVERY_AUTHORITY_CID = (
    "sha256:e7834d12a6150a7d4dd1f90dcb5369d73a6d8620a38bf6baa0cbbb3da17bebb9"
)
_M9_COORDINATION_REARM_OBSERVED_AT_MS = 1_787_939_913_492
_M8_COORDINATION_PROJECTION_DIGEST = (
    "sha256:a31ff114882052cff614cfc4e134ed6cbe44de456fdd337bfcb296d34caa6a24"
)
_M9_COORDINATION_PROJECTION_DIGEST = (
    "sha256:7fb9bacb0f76fe832cc34dd5fb2ccdef13ddafeef4ec2a3d532cb902aa62011e"
)

_M10_EXPECTED_PROJECTION_CID = (
    "baguqeerareq2bngq3hffyk5vidym2ukeleg5gehpaxhqvvdjayn7ucxplcaq"
)
_M10_EVENT_SUFFIX_LENGTH = 2
_M10_TARGET_EVENT_WATERMARK = 179
_M10_TARGET_PLAN_REVISION = 11
_M10_TARGET_GENERATION = 12
_M10_MIGRATION_REVISION = "SAWM-R2-M10"
_M10_SUPERSESSION_MODE = "source_only_live_projection_comparator_repair"
_M10_SUPERSESSION_REASON = (
    "authenticated_live_task_and_semantic_projection_comparator_repair"
)
_M10_LIVE_PROJECTION_AUTHORITY_CID = (
    "sha256:f27878def0ee9b406d0dbaac669728278e4f375cb267ee7f76330faaf9b14f10"
)

_M9_FROZEN_LIVE_BINDING: dict[str, Any] = {
    "store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m9/control.duckdb"
    ),
    # M9's final pair marker records the bytes published before its only live
    # owner started.  The stopped live checkpoint is a distinct physical file
    # whose unchanged semantic projections are sealed independently below.
    "publication_control_store_sha256": (
        "2cf9b143ab59a528ca4031754637a5a27dcd7e3e7d52b1db6bbaa5fc560d2fce"
    ),
    "control_store_sha256": (
        "dedd6ce6107b556a87b716ac59839df7f714cd73d72dc2698d38d8f1dca92d2f"
    ),
    "control_store_size": 45_101_056,
    "coordination_store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m9/control.coordination.duckdb"
    ),
    "coordination_store_sha256": (
        "e69a4c22a1a1a8bd38ccf210803a02ecb040d0ee9d4cc776165a4570425d6862"
    ),
    "coordination_store_size": 9_711_616,
    "coordination_event_count": 36,
    "event_watermark": 177,
    "event_prefix_sha256": (
        "76377c47b01b15ab1c03889ec1c2e3b9ced25fa080ad39a48b6e0a4b3b1bb464"
    ),
    "projection_cid": _M9_EXPECTED_PROJECTION_CID,
    "source_head": "486c5b68477c0f2ecdaf7b04a01e24629fa4b5a0",
    "source_tree": "b8e0c0d928b407c71383c45ce1da2f1155996c9e",
    "source_binding_cid": (
        "sha256:2a631f1b465d3d5a39636f8b588e1dfad1ebcb1cc130defff4ee7d60b9c686c6"
    ),
    "database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
    "generation": 11,
    "plan_revision": 10,
    "process_birth_id": "birth:4338518933c02a3de9372e1492954473",
    "server_id": "server:2680629a-310c-4e7b-943d-4e6e977a4446",
    "migration_receipt_path": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m9/migration-receipt.json"
    ),
    "migration_receipt_cid": (
        "sha256:f6b0475d21111091246803de4ee664d6a2cd0d5486667890f0969424abb436a1"
    ),
    "migration_receipt_file_sha256": (
        "64c1ee94ad532eff39585938d5ceab29d13ebc1989fb6c81ad131b98c3da64e9"
    ),
    "owner_status_path": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m9/quack-owner/quack-state-server.status.json"
    ),
    "owner_status_sha256": (
        "b50fb27ad63ffaa3360b536e06ab9b1a0dcd726259f2dcf86c6b06ece75af87f"
    ),
    "semantic_authority_digest": (
        "sha256:a4903791c91cc2e9c3337f2abdfd6af78389f7036d54cb7f8e43246cd4f0c023"
    ),
    "frozen_base_authority_digest": (
        "sha256:e26c4db78b0e8c2914b6d9febc7dd0c9079b47a1f2c1be9b73d310fc98361032"
    ),
    "append_surface_digest": (
        "sha256:7082997db9232a17411402c86fc003085367ebbde9b99068231485c754d6007c"
    ),
    "catalog_digest": (
        "sha256:3cc2e066bd4495e6efc0a3410a72c5df29242dcacfa94cd75d30f61c658539a7"
    ),
}

_M9_LIVE_PREFLIGHT_FAILURE: dict[str, Any] = {
    "schema": "sawm/live-control-preflight-failure@1",
    "attempt": "SAWM-R2-M9-PREFLIGHT-A1",
    "phase": "authenticated_live_task_authority_validation",
    "command": (
        "python scripts/ops/agent_supervisor/"
        "semantic_addressed_world_model.py preflight"
    ),
    "exit_code": 2,
    "error_payload": {
        "schema": "sawm/operator-error@1",
        "valid": False,
        "error": (
            "OperatorError: live Quack task authority conflict: frozen M6 "
            "task status/revision projection differs"
        ),
    },
    "failure_kind": "historical_m6_task_status_revision_projection_mismatch",
    "source_head": _M9_FROZEN_LIVE_BINDING["source_head"],
    "source_tree": _M9_FROZEN_LIVE_BINDING["source_tree"],
    "source_binding_cid": _M9_FROZEN_LIVE_BINDING["source_binding_cid"],
    "store_id": _M9_FROZEN_LIVE_BINDING["store_id"],
    "owner_generation": _M9_FROZEN_LIVE_BINDING["generation"],
    "owner_server_id": _M9_FROZEN_LIVE_BINDING["server_id"],
    "owner_process_birth_id": _M9_FROZEN_LIVE_BINDING["process_birth_id"],
    "canonical_owner_rows_verified": True,
    "task_count_verified": 45,
    "goal_count_verified": 29,
    "dependency_count_verified": 136,
    "plan_count_verified": 1,
    "event_watermark_verified": _M9_TARGET_EVENT_WATERMARK,
    "projection_cid_verified": _M9_EXPECTED_PROJECTION_CID,
    "task_cid": (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    ),
    "observed_task_status": "retrying",
    "observed_task_revision": 10,
    "incorrect_comparator_status": "todo",
    "incorrect_comparator_revision": 7,
    "semantic_authority_digest": _M9_FROZEN_LIVE_BINDING[
        "semantic_authority_digest"
    ],
    "incorrect_semantic_comparator_digest": (
        "sha256:dd9150737feae6de30ce9b2b34968499bd24d7341d44567c43af794989dc7bbb"
    ),
    "provider_probed": False,
    "task_claimed": False,
    "task_state_changed": False,
    "coordination_state_changed": False,
    "implementation_provider_invoked": False,
    "effect_claim_recorded": False,
    "implementation_commit_created": False,
    "merge_attempted": False,
    "completion_changed": False,
    "failure_time_authority": "unavailable",
}

_M8_FROZEN_LIVE_BINDING: dict[str, Any] = {
    "store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m8/control.duckdb"
    ),
    "control_store_sha256": (
        "e8b0814314b38f4e73c77258403aacd495c341d1c249d8fd84c5b29b0d1a21ab"
    ),
    "pre_audit_control_store_sha256": (
        "b5ff24522329f1bc54c3b3ea28232f8f2317213eb626755e0bbf96d41da405ae"
    ),
    "control_store_size": 45_101_056,
    "coordination_store_id": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m8/control.coordination.duckdb"
    ),
    "coordination_store_sha256": (
        "9363b3763bdfea7ba3277f4b19853c093ca2ede36a8592832b1ca04d033576af"
    ),
    "coordination_store_size": 6_828_032,
    "coordination_event_count": 35,
    "event_watermark": 174,
    "event_prefix_sha256": (
        "ac685db15d9e2f47c904c618ce7b1573aedeceb5613db144c78a2f93a0a0fe1a"
    ),
    "projection_cid": (
        "baguqeerarcj7jcicssddipfwagqjy5zhikzcohotihhsycnsl5ctqyuydgra"
    ),
    "source_head": "308a3bff9a9648afcf657f1fa646c14162f13dbc",
    "source_tree": "d9a31cf53650307c6a802b505407660aaaf83de7",
    "source_binding_cid": (
        "sha256:04ba84c98ca25b6362052078e4e6642aa3e920935fe97312034ec0d53e542258"
    ),
    "database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
    "generation": 10,
    "plan_revision": 9,
    "process_birth_id": "birth:8aa7f2a215e07322a9bfc2b1bb543754",
    "server_id": "server:0ac3d685-1755-4a1a-ba41-1303f91c875f",
    "migration_receipt_path": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m8/migration-receipt.json"
    ),
    "migration_receipt_cid": (
        "sha256:ce8dfbf291ad146fef7931983407211c2d7fb25e0c5b1eaceb80819db156f454"
    ),
    "migration_receipt_file_sha256": (
        "fa0d37ffa40f3e69293ae93d8aa8b6c4199c8a8bdd85636a2f6573865645cc47"
    ),
    "owner_status_path": (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m8/quack-owner/quack-state-server.status.json"
    ),
    "owner_status_sha256": (
        "db730f03550d9e56dffcffc964db32de7d8ea1b4040f3372646a8da88a7c1473"
    ),
    "validator_digest": (
        "sha256:826b4bb2a30701fd430e8037352772cebbd561c59b7f0f0d55743bc8a2001d3b"
    ),
    "semantic_authority_digest": (
        "sha256:dd9150737feae6de30ce9b2b34968499bd24d7341d44567c43af794989dc7bbb"
    ),
    "frozen_base_authority_digest": (
        "sha256:22db1bbcab0ad1302a12f34d799e7e879ecfe1121a5500d976bfefd49ec165ee"
    ),
    "append_surface_digest": (
        "sha256:4f07770df3892c7039505d8970b9f8d568859d1e5cd4e7db82c586090879a5a1"
    ),
}

_M8_LIVE_FAILURE_RECEIPT: dict[str, Any] = {
    "schema": "ipfs_accelerate_py/agent-supervisor/task-claim-failure-settlement@1",
    "operation": "database_task_claim_failure",
    "failure_kind": "terminal_portal_bridge_error",
    "failure_payload_digest": (
        "sha256:9d8bb2daa636d6cea52318e8c2de14ec08a8bfc8e67f33705ebcccff918322b5"
    ),
    "task_cid": (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    ),
    "attempt_id": "attempt:0b7907789d77439a87c02dcd15306d90",
    "attempt_number": 1,
    "claim_id": "claim:bc46ab91ee7f49769e4a2c67115087a5",
    "lease_id": "lease:4980d9dc576f4e43b3a5205cac5355ec",
    "owner_session_id": "embedded-store:78d2afe662423ee4ec64e584e559dc82",
    "fencing_token": 1,
    "fence_epoch": 1,
    "provider_invocation_count": 0,
    "effect_claim_count": 0,
    "automatic_retry_admitted": False,
    "control_expected_status": "in_progress",
    "control_expected_revision": 8,
    "settlement_id": (
        "baguqeeransrfearh5ojrnru6ls43mn7xsx6mlhndkhokrn4k7wlhowxjnshq"
    ),
}

_M8_LIVE_IMPLEMENTATION_FAILURE: dict[str, Any] = {
    "schema": "sawm/live-control-implementation-failure@1",
    "attempt": "SAWM-R2-M8-LIVE-A1",
    "phase": "implementation_execution",
    "failure_kind": "external_protected_checkout_recovery_required",
    "task_alias": "SAWM-001",
    "task_cid": _M8_LIVE_FAILURE_RECEIPT["task_cid"],
    "attempt_id": _M8_LIVE_FAILURE_RECEIPT["attempt_id"],
    "claim_id": _M8_LIVE_FAILURE_RECEIPT["claim_id"],
    "lease_id": _M8_LIVE_FAILURE_RECEIPT["lease_id"],
    "settlement_id": _M8_LIVE_FAILURE_RECEIPT["settlement_id"],
    "failure_payload_digest": _M8_LIVE_FAILURE_RECEIPT[
        "failure_payload_digest"
    ],
    "claim_event_id": (
        "baguqeerao3xer2z3pgtstadkqzkiuilwwhp55b45eelviqkjfnhwczxicvca"
    ),
    "settlement_event_id": (
        "baguqeeratipemuuwggmymvouqtzwgrijivxrtgvqa4edvohxrrxne3ew2uqq"
    ),
    "from_status": "todo",
    "claim_status": "in_progress",
    "terminal_status": "blocked",
    "from_revision": 7,
    "claim_revision": 8,
    "terminal_revision": 9,
    "provider_probe_failure_class": "hard_quota_exhausted",
    "provider_probe_receipt_id": (
        "sha256:606ad014e49c67e838ba731374d6035fc274eb49a8f570810648a472e551bb36"
    ),
    "independent_quota_verifier_status": "not_run",
    "quota_evidence_id": "",
    "codex_fallback_dispatched": False,
    "provider_invocation_count": 0,
    "effect_claim_count": 0,
    "implementation_commit_count": 0,
    "merge_attempt_count": 0,
    "worker_self_approval": False,
    "failure_receipt": _M8_LIVE_FAILURE_RECEIPT,
}

_M7_LIVE_PREFLIGHT_FAILURE: dict[str, Any] = {
    "schema": "sawm/live-control-preflight-failure@2",
    "attempt": "SAWM-R2-M7-LIVE-A1",
    "phase": "authenticated_live_goal_definition_validation",
    "command": (
        "python scripts/ops/agent_supervisor/"
        "semantic_addressed_world_model.py preflight"
    ),
    "exit_code": 2,
    "error_payload": {
        "schema": "sawm/operator-error@1",
        "valid": False,
        "error": "KeyError: 'objective_id'",
    },
    "error_payload_cid": (
        "sha256:1d2afc9dd9b3c4b558c4e5504b79d0696646acdbae2febeccaa49144f9be022e"
    ),
    "failure_kind": "optional_goal_objective_field_read_as_required",
    "diagnosis": {
        "failing_goal_alias": "SAWM-G010",
        "population_key_present": False,
        "observed_objective_id": "",
        "required_expected_default": "",
    },
    "source_head": _M7_FROZEN_SUCCESSOR_BINDING["source_head"],
    "source_tree": _M7_FROZEN_SUCCESSOR_BINDING["source_tree"],
    "source_binding_cid": _M7_FROZEN_SUCCESSOR_BINDING[
        "source_binding_cid"
    ],
    "store_id": _M7_FROZEN_SUCCESSOR_BINDING["store_id"],
    "post_failure_frozen_control_store_sha256": _M7_FROZEN_SUCCESSOR_BINDING[
        "control_store_sha256"
    ],
    "event_watermark": _M7_FROZEN_SUCCESSOR_BINDING["event_watermark"],
    "event_prefix_sha256": _M7_FROZEN_SUCCESSOR_BINDING[
        "event_prefix_sha256"
    ],
    "projection_cid": _M7_FROZEN_SUCCESSOR_BINDING["projection_cid"],
    "semantic_authority_digest": _M7_FROZEN_SUCCESSOR_BINDING[
        "semantic_authority_digest"
    ],
    "owner_generation": _M7_FROZEN_SUCCESSOR_BINDING["generation"],
    "owner_server_id": _M7_FROZEN_SUCCESSOR_BINDING["server_id"],
    "owner_process_birth_id": _M7_FROZEN_SUCCESSOR_BINDING[
        "process_birth_id"
    ],
    "canonical_owner_rows_verified": True,
    "task_contract_count_verified": 45,
    "goal_count_verified": 29,
    "goal_contract_count_verified_before_failure": 1,
    "provider_probed": False,
    "task_claimed": False,
    "task_state_changed": False,
    "effect_claim_recorded": False,
    "implementation_commit_created": False,
    "merge_attempted": False,
    "implementation_provider_invoked": False,
    "worker_self_approval": False,
    "failure_time_authority": "unavailable",
}


class MaterializationError(RuntimeError):
    """Fail-closed SAWM bootstrap error."""


class MigrationRequired(MaterializationError):
    """An existing append-only authority contains a different population."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def closed(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=closed)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _stable_regular_sha256(
    path: Path,
    *,
    root: Path,
    noun: str,
    required_link_count: int | None = None,
) -> tuple[str, int]:
    """Hash one confined no-follow regular file without identity drift."""

    resolved_root = root.resolve()
    if not path.parent.resolve().is_relative_to(resolved_root):
        raise MigrationRequired(f"{noun} escapes the repository root")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MigrationRequired(f"{noun} is not a no-follow regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MigrationRequired(f"{noun} is not a regular file")
        if (
            required_link_count is not None
            and int(before.st_nlink) != int(required_link_count)
        ):
            raise MigrationRequired(
                f"{noun} does not have exactly {required_link_count} link(s)"
            )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        if identity_before != identity_after:
            raise MigrationRequired(f"{noun} changed while it was read")
        return digest.hexdigest(), int(before.st_size)
    finally:
        os.close(descriptor)


def _confined_leaf(root: Path, relative: str, *, noun: str) -> Path:
    """Resolve only a relative path's parent, retaining its no-follow leaf."""

    candidate = Path(str(relative or ""))
    if (
        candidate.is_absolute()
        or not candidate.name
        or ".." in candidate.parts
    ):
        raise MigrationRequired(f"{noun} path is not a confined relative leaf")
    parent = (root / candidate.parent).resolve()
    if not parent.is_relative_to(root.resolve()):
        raise MigrationRequired(f"{noun} parent escapes the repository root")
    return parent / candidate.name


def _load_nofollow_json(
    path: Path,
    *,
    root: Path,
    noun: str,
    maximum_bytes: int = 1024 * 1024,
    required_link_count: int = 1,
) -> tuple[dict[str, Any], str]:
    """Load one confined, single-link JSON object through a pinned descriptor."""

    resolved_root = root.resolve()
    if not path.parent.resolve().is_relative_to(resolved_root):
        raise MigrationRequired(f"{noun} escapes the repository root")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MigrationRequired(f"{noun} is not a no-follow regular file") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != int(required_link_count)
            or before.st_size > maximum_bytes
        ):
            raise MigrationRequired(
                f"{noun} is not a bounded {required_link_count}-link file"
            )
        payload = bytearray()
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            payload.extend(block)
            if len(payload) > maximum_bytes:
                raise MigrationRequired(f"{noun} exceeds its size bound")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise MigrationRequired(f"{noun} changed while it was read")
    finally:
        os.close(descriptor)

    def closed(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MigrationRequired(f"duplicate JSON key {key!r} in {noun}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise MigrationRequired(f"non-finite JSON constant {value!r} in {noun}")

    try:
        value = json.loads(
            bytes(payload).decode("utf-8"),
            object_pairs_hook=closed,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationRequired(f"{noun} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise MigrationRequired(f"{noun} must contain an object")
    return value, hashlib.sha256(payload).hexdigest()


def _board_module(root: Path):
    import importlib.util
    name = "_sawm_board_validator_for_materializer"
    path = root / "scripts/validate_semantic_addressed_world_model_board.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MaterializationError("unable to load the sealed board parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=20,
    )
    if result.returncode:
        raise MaterializationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _source_binding(root: Path) -> dict[str, Any]:
    controls = (
        ".gitignore",
        "requirements.txt",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        "docs/architecture/semantic_addressed_world_model.objectives.md",
        "docs/architecture/semantic_addressed_world_model.todo.md",
        "docs/architecture/semantic_addressed_world_model_inventory/repository_baseline.json",
        "docs/architecture/semantic_addressed_world_model_inventory/authority_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/overlap_gap_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/identity_inventory.json",
        "docs/architecture/semantic_addressed_world_model_inventory/interface_inventory.json",
        "docs/architecture/semantic_addressed_world_model_inventory/dependency_graph.json",
        "docs/architecture/semantic_addressed_world_model_inventory/capability_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/rollout_baseline.json",
        "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "config/semantic_addressed_world_model_native_dependency.authorization.json",
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "ipfs_accelerate_py/agent_implementation_route.py",
        "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_extension_projection.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_live_capsule.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/board_control_plane.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_attestation.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_quack_protocol.py",
        "test/api/test_agent_supervisor_configured_board_extension_projection.py",
        "test/api/test_agent_supervisor_configured_board_live_capsule.py",
        "test/api/test_agent_supervisor_configured_board_scheduler.py",
        "test/api/test_agent_supervisor_database_coordination.py",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "test/api/test_agent_supervisor_native_dependency_pin.py",
        "test/api/test_agent_supervisor_project_dependency_preflight.py",
        "test/api/test_agent_supervisor_provider_command_binding.py",
        "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
    )
    missing = [path for path in controls if not (root / path).is_file()]
    if missing:
        raise MaterializationError(
            "sealed source controls are missing: " + ", ".join(missing)
        )
    file_digests = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in controls
    }
    payload = {
        "schema": "sawm/current-source-binding@1",
        "head": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "branch": _git(root, "branch", "--show-current"),
        "datasets_gitlink": _git(root, "rev-parse", "HEAD:ipfs_datasets_py"),
        "kit_gitlink": _git(root, "rev-parse", "HEAD:ipfs_kit_py"),
        "control_sha256": file_digests,
    }
    return {**payload, "source_binding_cid": _identity(payload)}


def _metadata_payload(card: Any) -> dict[str, Any]:
    """Retain every closed board field without treating Markdown as state."""
    return {str(key).replace(" ", "_"): str(value) for key, value in sorted(card.metadata.items())}


def _operational_task_aliases() -> tuple[str, ...]:
    return tuple(f"SAWM-{ordinal:03d}" for ordinal in range(1, 45))


def _task_identity_manifest_cid(task_cids: Mapping[str, Any]) -> str:
    manifest = [
        {"task_alias": alias, "task_cid": str(task_cids[alias])}
        for alias in _operational_task_aliases()
    ]
    return _identity(manifest)


def _operational_validation_commands(
    commands: Sequence[Sequence[str]],
    *,
    task_alias: str,
) -> tuple[tuple[str, ...], int]:
    revised: list[tuple[str, ...]] = []
    replacements = 0
    for argv in commands:
        revised_argv: list[str] = []
        for part in argv:
            text = str(part)
            count = text.count(_M6_VALIDATION_TOKEN_FROM)
            replacements += count
            revised_argv.append(
                text.replace(_M6_VALIDATION_TOKEN_FROM, _M6_VALIDATION_TOKEN_TO)
            )
        revised.append(tuple(revised_argv))
    if not revised or replacements != len(revised):
        raise MigrationRequired(
            f"{task_alias} operational validation token cardinality differs"
        )
    return tuple(revised), replacements


def _operational_validation_receipt(
    population: Mapping[str, Any],
    *,
    task_alias: str,
    task_cid: str,
    expected_status: str,
    expected_revision: int,
    prior_validations: Sequence[Sequence[str]],
    operational_validations: Sequence[Sequence[str]],
    replacement_count: int,
) -> dict[str, Any]:
    migration = population["migration_inventory"]
    receipt = {
        "schema": "sawm/operational-validation-revision@1",
        "authority_class": "operator_control_plane",
        "migration_revision": migration["migration_revision"],
        "task_cid": task_cid,
        "task_alias": task_alias,
        "historical_definition_cid": migration["prior_task_cids"][task_alias],
        "historical_definition_rewritten": False,
        "expected_status": expected_status,
        "expected_revision": int(expected_revision),
        "status": expected_status,
        "target_revision": int(expected_revision) + 1,
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "validation_count": len(prior_validations),
        "validation_token_from": _M6_VALIDATION_TOKEN_FROM,
        "validation_token_to": _M6_VALIDATION_TOKEN_TO,
        "validation_token_replacement_count": int(replacement_count),
        "prior_validation_digest": _identity(
            {
                "task_cid": task_cid,
                "validations": [list(argv) for argv in prior_validations],
            }
        ),
        "operational_validation_digest": _identity(
            {
                "task_cid": task_cid,
                "validations": [list(argv) for argv in operational_validations],
            }
        ),
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def build_population(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    board = _board_module(root)
    tasks = board._parse_cards(root / "docs/architecture/semantic_addressed_world_model.todo.md", goal=False)
    goals = board._parse_cards(root / "docs/architecture/semantic_addressed_world_model.objectives.md", goal=True)
    if tuple(item.identifier for item in tasks) != board.TASK_IDS or tuple(item.identifier for item in goals) != board.GOAL_IDS:
        raise MaterializationError("board population differs from the sealed 45-task/29-goal identity")
    source = _source_binding(root)
    migration = _load_json(
        root
        / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json"
    )
    migration_history = migration.get("migration_history")
    expected_migration_revision = (
        f"SAWM-R2-M{len(migration_history) + 1}"
        if isinstance(migration_history, list)
        else ""
    )
    if (
        migration.get("schema")
        != "sawm/prior-materialization-migration-inventory@3"
        or migration.get("migration_revision") != expected_migration_revision
        or migration.get("migration_kind")
        != "bounded_validation_runtime_and_operational_board_command_recovery"
        or migration.get("supersession_reason") != _M6_SUPERSESSION_REASON
    ):
        raise MaterializationError("the sealed append-only migration revision is missing")
    launch_failure = migration.get("preworker_launch_failure")
    if (
        not isinstance(launch_failure, dict)
        or launch_failure != _M3_PREWORKER_FAILURE_V2
        or _identity(launch_failure.get("error_payload"))
        != launch_failure.get("error_payload_cid")
    ):
        raise MaterializationError("the typed pre-worker launch failure is invalid")
    preprovider_failure = migration.get("preprovider_task_failure")
    if (
        not isinstance(preprovider_failure, dict)
        or preprovider_failure.get("schema") != "sawm/pre-provider-task-failure@2"
        or preprovider_failure.get("store_id") != migration.get("prior_store_id")
        or preprovider_failure.get("control_store_sha256")
        != migration.get("prior_control_store_sha256")
        or preprovider_failure.get("canonical_event_watermark")
        != migration.get("prior_event_watermark")
        or preprovider_failure.get("canonical_event_prefix_sha256")
        != migration.get("prior_event_prefix_sha256")
        or preprovider_failure.get("canonical_projection_cid")
        != migration.get("prior_projection_cid")
        or preprovider_failure.get("source_binding_cid")
        != migration.get("prior_source_binding_cid")
        or preprovider_failure.get("successor_migration_revision")
        != migration.get("migration_revision")
        or preprovider_failure.get("successor_plan_revision")
        != migration.get("target_plan_revision")
        or preprovider_failure.get("successor_generation")
        != migration.get("target_generation")
        or preprovider_failure.get("successor_store_id")
        != migration.get("target_store_id")
        or preprovider_failure.get("implementation_provider_invoked") is not False
        or preprovider_failure.get("provider_dispatched") is not False
        or preprovider_failure.get("effect_claim_recorded") is not False
        or preprovider_failure.get("implementation_commit_created") is not False
        or preprovider_failure.get("merge_attempted") is not False
        or preprovider_failure.get("task_completed") is not False
        or preprovider_failure.get("canonical_task_status") != "blocked"
        or int(preprovider_failure.get("canonical_task_revision") or 0) != 5
        or preprovider_failure.get("settlement_closed") is not True
    ):
        raise MaterializationError("the sealed pre-provider task failure is invalid")
    bounded_paths = migration.get("bounded_control_plane_repair_paths")
    if (
        not isinstance(bounded_paths, list)
        or len(bounded_paths) != len(set(bounded_paths))
        or any(path not in source["control_sha256"] for path in bounded_paths)
    ):
        raise MaterializationError("bounded control-plane repairs are not source-bound")
    prior_goal_cids = migration.get("prior_goal_cids")
    prior_task_cids = migration.get("prior_task_cids")
    if not isinstance(prior_goal_cids, dict) or not isinstance(prior_task_cids, dict):
        raise MaterializationError("prior SAWM identity maps are incomplete")
    if (
        _task_identity_manifest_cid(prior_task_cids)
        != _M6_TASK_IDENTITY_MANIFEST_CID
    ):
        raise MaterializationError("operational task identity manifest differs")

    goal_cids: dict[str, str] = {}
    goal_definitions: dict[str, dict[str, Any]] = {}
    for ordinal, card in enumerate(goals, 1):
        definition = {
            "schema": "sawm/goal-definition@1", "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "goal_id": card.identifier,
            "title": card.title, "ordinal": ordinal,
            "metadata": _metadata_payload(card),
            "source_binding_cid": migration["definition_source_binding_cid"],
        }
        goal_definitions[card.identifier] = definition
        goal_cids[card.identifier] = str(prior_goal_cids[card.identifier])
        if _identity(definition) != goal_cids[card.identifier]:
            raise MaterializationError(
                f"preserved goal definition does not rehash: {card.identifier}"
            )

    objective_id = str(migration["prior_objective_id"])
    objective_rows: list[dict[str, Any]] = []
    goal_edges: list[dict[str, Any]] = []
    for ordinal, card in enumerate(goals, 1):
        parent_alias = board.GOAL_PARENT[card.identifier]
        definition = goal_definitions[card.identifier]
        row = {
            "goal_cid": goal_cids[card.identifier], "goal_id": card.identifier,
            "goal_alias": card.identifier, "title": card.title, "ordinal": ordinal,
            "status": str(card.metadata.get("status") or "open"),
            "parent_goal_cid": goal_cids[parent_alias] if parent_alias else "",
            "definition_cid": goal_cids[card.identifier], "definition": definition,
            "board_namespace": NAMESPACE, "plan_revision": REVISION,
        }
        if card.identifier == ROOT_GOAL:
            row.update({"objective_id": objective_id, "objective_alias": ROOT_GOAL,
                        "priority": str(card.metadata.get("priority") or "P0")})
        objective_rows.append(row)
        if parent_alias:
            goal_edges.append({"parent_goal_cid": goal_cids[parent_alias],
                               "child_goal_cid": goal_cids[card.identifier],
                               "edge_kind": "goal_refinement"})

    plan_definition = {
        "schema": "sawm/plan-definition@1", "board_namespace": NAMESPACE,
        "plan_revision": REVISION, "root_goal_cid": goal_cids[ROOT_GOAL],
        "goal_definition_cids": [goal_cids[item.identifier] for item in goals],
        "source_binding_cid": migration["definition_source_binding_cid"],
    }
    plan_cid = str(migration["prior_plan_root_cid"])
    if _identity(plan_definition) != plan_cid:
        raise MaterializationError("preserved plan definition does not rehash")

    task_definitions: dict[str, dict[str, Any]] = {}
    task_cids: dict[str, str] = {}
    for ordinal, card in enumerate(tasks, 1):
        definition = {
            "schema": "sawm/task-definition@1", "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "task_id": card.identifier,
            "title": card.title, "ordinal": ordinal,
            "metadata": _metadata_payload(card),
            "source_binding_cid": migration["definition_source_binding_cid"],
        }
        task_definitions[card.identifier] = definition
        task_cids[card.identifier] = str(prior_task_cids[card.identifier])
        if _identity(definition) != task_cids[card.identifier]:
            raise MaterializationError(
                f"preserved task definition does not rehash: {card.identifier}"
            )

    taskboard: list[dict[str, Any]] = []
    for ordinal, card in enumerate(tasks, 1):
        meta = card.metadata
        deps = json.loads(meta["dependencies json"])
        outputs = json.loads(meta["outputs json"])
        validations = json.loads(meta["validation commands json"])
        goal_alias = str(meta["goal id"])
        definition = task_definitions[card.identifier]
        taskboard.append(
            {
                "task_cid": task_cids[card.identifier], "task_id": card.identifier,
                "task_alias": card.identifier, "title": card.title, "ordinal": ordinal,
                # The operator card is deliberately born ready.  Its Markdown
                # completed marker cannot bypass current evidence and CAS.
                "status": "ready" if card.identifier == "SAWM-000" else "todo",
                "priority": str(meta.get("priority") or "P2"),
                "goal_cid": goal_cids[goal_alias], "goal_id": goal_alias,
                "plan_cid": plan_cid,
                "depends_on": [task_cids[str(dep)] for dep in deps],
                # IntentRepository output keys use the closed safe-ID grammar;
                # retain the exact file path inside the canonical effect body.
                "outputs": [{"effect_id": _identity({"task": card.identifier, "path": str(path)}),
                             "declared_path": str(path), "effect": "declared_output"}
                            for path in outputs],
                "acceptance_criteria": [{
                    "ordinal": 1, "criterion": str(meta.get("acceptance") or ""),
                    # IntentRepository stores this complete mapping as the
                    # evidence policy, so the required kind belongs here.
                    **({"evidence_kind": "operator_control_validation"}
                       if card.identifier == "SAWM-000" else {}),
                }],
                "validation_commands": validations,
                "definition_cid": task_cids[card.identifier], "definition": definition,
                "board_namespace": NAMESPACE, "plan_revision": REVISION,
                "completion_mode": str(meta.get("completion mode") or "automatic"),
                "protected_paths": str(meta.get("protected paths") or ""),
                "repository_owner": str(meta.get("owning repository") or ""),
                "provider_role": str(meta.get("provider role") or ""),
                "rollout_mode": str(meta.get("rollout mode") or ""),
            }
        )

    program_definition = {
        "schema": "sawm/program-definition@1", "board_namespace": NAMESPACE,
        "plan_revision": REVISION,
        "source_binding_cid": migration["definition_source_binding_cid"],
        "plan_cid": plan_cid, "goal_cids": [goal_cids[item.identifier] for item in goals],
        "task_cids": [task_cids[item.identifier] for item in tasks],
    }
    program_definition_cid = str(migration["prior_program_definition_cid"])
    if _identity(program_definition) != program_definition_cid:
        raise MaterializationError("preserved program definition does not rehash")
    for row in (*objective_rows, *taskboard):
        row["program_definition_cid"] = program_definition_cid
    return {
        "schema": "sawm/program-population@1", "board_namespace": NAMESPACE,
        "plan_revision": REVISION, "program_definition_cid": program_definition_cid,
        "program_definition": program_definition, "repository_tree_id": source["source_binding_cid"],
        "source_binding": source, "plan_root_cid": plan_cid,
        # DatabaseTaskSource consumes `objectives` before `goals`; all 29 goal
        # records are intentionally supplied in this authoritative key.
        "objectives": objective_rows, "goal_edges": goal_edges,
        "plans": [{"plan_cid": plan_cid, "plan_alias": REVISION,
                   "goal_cid": goal_cids[ROOT_GOAL], "status": "active", **plan_definition}],
        "taskboard": taskboard, "migration_inventory": migration,
    }


def _store_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positional_rows(rows: Sequence[Any], width: int) -> list[tuple[Any, ...]]:
    """Project local tuples and Quack DuckDBRow values identically."""

    if width < 1:
        raise ValueError("row projection width must be positive")
    return [tuple(row[index] for index in range(width)) for row in rows]


def _m7_source_repair_authority(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the closed source-only successor authority shared by both seals."""

    repair_paths = [
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        (
            "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ),
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
    ]
    expected = {
        "schema": "sawm/source-only-launch-repair-authorization@1",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": _M7_MIGRATION_REVISION,
        "migration_kind": _M7_SUPERSESSION_REASON,
        "supersession_mode": _M7_SUPERSESSION_MODE,
        "prior_store_id": _M6_FROZEN_SUCCESSOR_BINDING["store_id"],
        "prior_control_store_sha256": _M6_FROZEN_SUCCESSOR_BINDING[
            "control_store_sha256"
        ],
        "prior_control_store_size": _M6_FROZEN_SUCCESSOR_BINDING[
            "control_store_size"
        ],
        "prior_event_watermark": _M6_FROZEN_SUCCESSOR_BINDING[
            "event_watermark"
        ],
        "prior_event_prefix_sha256": _M6_FROZEN_SUCCESSOR_BINDING[
            "event_prefix_sha256"
        ],
        "prior_projection_cid": _M6_FROZEN_SUCCESSOR_BINDING[
            "projection_cid"
        ],
        "prior_source_head": _M6_FROZEN_SUCCESSOR_BINDING["source_head"],
        "prior_source_tree": _M6_FROZEN_SUCCESSOR_BINDING["source_tree"],
        "prior_source_binding_cid": _M6_FROZEN_SUCCESSOR_BINDING[
            "source_binding_cid"
        ],
        "prior_database_uuid": _M6_FROZEN_SUCCESSOR_BINDING["database_uuid"],
        "prior_generation": _M6_FROZEN_SUCCESSOR_BINDING["generation"],
        "prior_plan_revision": _M6_FROZEN_SUCCESSOR_BINDING["plan_revision"],
        "prior_process_birth_id": _M6_FROZEN_SUCCESSOR_BINDING[
            "process_birth_id"
        ],
        "prior_server_id": _M6_FROZEN_SUCCESSOR_BINDING["server_id"],
        "prior_materialization_receipt_path": _M6_FROZEN_SUCCESSOR_BINDING[
            "migration_receipt_path"
        ],
        "prior_materialization_receipt_cid": _M6_FROZEN_SUCCESSOR_BINDING[
            "migration_receipt_cid"
        ],
        "prior_materialization_receipt_file_sha256": (
            _M6_FROZEN_SUCCESSOR_BINDING["migration_receipt_file_sha256"]
        ),
        "prior_owner_status_path": _M6_FROZEN_SUCCESSOR_BINDING[
            "owner_status_path"
        ],
        "prior_owner_status_sha256": _M6_FROZEN_SUCCESSOR_BINDING[
            "owner_status_sha256"
        ],
        "prior_validator_digest": _M6_FROZEN_SUCCESSOR_BINDING[
            "validator_digest"
        ],
        "prior_semantic_authority_digest": _M6_FROZEN_SUCCESSOR_BINDING[
            "semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": _M6_FROZEN_SUCCESSOR_BINDING[
            "frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": _M6_FROZEN_SUCCESSOR_BINDING[
            "append_surface_digest"
        ],
        "target_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m7/control.duckdb"
        ),
        "target_generation": _M7_TARGET_GENERATION,
        "target_plan_revision": _M7_TARGET_PLAN_REVISION,
        "target_event_watermark": _M7_TARGET_EVENT_WATERMARK,
        "target_projection_cid": _M7_EXPECTED_PROJECTION_CID,
        "event_suffix_length": _M7_EVENT_SUFFIX_LENGTH,
        "bounded_control_plane_repair_paths": repair_paths,
        "live_preflight_failure": _M6_LIVE_PREFLIGHT_FAILURE,
        "live_preflight_failure_cid": _identity(_M6_LIVE_PREFLIGHT_FAILURE),
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    configured = config.get("source_repair_materialization")
    inventoried = population["migration_inventory"].get(
        "source_repair_materialization"
    )
    if (
        configured != expected
        or inventoried != expected
        or _identity(expected) != _M7_SOURCE_REPAIR_AUTHORITY_CID
    ):
        raise MaterializationError(
            "M7 source-only repair authority differs across scheduler and inventory"
        )
    return expected


def _assert_m7_source_delta(
    root: Path,
    population: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Prove the committed M6-to-M7 source delta is exactly authorized."""

    prior_head = str(authority["prior_source_head"])
    current_head = str(population["source_binding"]["head"])
    if current_head == prior_head:
        raise MaterializationError("M7 source repair has no successor commit")
    _git(root, "merge-base", "--is-ancestor", prior_head, current_head)
    changed: set[str] = set()
    output = _git(
        root,
        "diff",
        "--name-status",
        "--no-renames",
        prior_head,
        current_head,
        "--",
    )
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] != "M":
            raise MaterializationError(
                f"M7 source delta contains a non-modification entry: {line}"
            )
        path = fields[1]
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise MaterializationError("M7 source delta path is not confined")
        changed.add(path)
    authorized = set(authority["bounded_control_plane_repair_paths"])
    if changed != authorized:
        raise MaterializationError(
            "M7 committed source delta differs from the exact repair paths: "
            + json.dumps(
                {
                    "missing": sorted(authorized - changed),
                    "unexpected": sorted(changed - authorized),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def _m8_source_repair_authority(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the closed M8 successor authority shared by both source seals."""

    repair_paths = [
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        (
            "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ),
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
    ]
    expected = {
        "schema": "sawm/source-only-launch-repair-authorization@2",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": _M8_MIGRATION_REVISION,
        "migration_kind": _M8_SUPERSESSION_REASON,
        "supersession_mode": _M8_SUPERSESSION_MODE,
        "prior_store_id": _M7_FROZEN_SUCCESSOR_BINDING["store_id"],
        "prior_control_store_sha256": _M7_FROZEN_SUCCESSOR_BINDING[
            "control_store_sha256"
        ],
        "prior_control_store_size": _M7_FROZEN_SUCCESSOR_BINDING[
            "control_store_size"
        ],
        "prior_event_watermark": _M7_FROZEN_SUCCESSOR_BINDING[
            "event_watermark"
        ],
        "prior_event_prefix_sha256": _M7_FROZEN_SUCCESSOR_BINDING[
            "event_prefix_sha256"
        ],
        "prior_projection_cid": _M7_FROZEN_SUCCESSOR_BINDING[
            "projection_cid"
        ],
        "prior_source_head": _M7_FROZEN_SUCCESSOR_BINDING["source_head"],
        "prior_source_tree": _M7_FROZEN_SUCCESSOR_BINDING["source_tree"],
        "prior_source_binding_cid": _M7_FROZEN_SUCCESSOR_BINDING[
            "source_binding_cid"
        ],
        "prior_database_uuid": _M7_FROZEN_SUCCESSOR_BINDING[
            "database_uuid"
        ],
        "prior_generation": _M7_FROZEN_SUCCESSOR_BINDING["generation"],
        "prior_plan_revision": _M7_FROZEN_SUCCESSOR_BINDING[
            "plan_revision"
        ],
        "prior_process_birth_id": _M7_FROZEN_SUCCESSOR_BINDING[
            "process_birth_id"
        ],
        "prior_server_id": _M7_FROZEN_SUCCESSOR_BINDING["server_id"],
        "prior_materialization_receipt_path": _M7_FROZEN_SUCCESSOR_BINDING[
            "migration_receipt_path"
        ],
        "prior_materialization_receipt_cid": _M7_FROZEN_SUCCESSOR_BINDING[
            "migration_receipt_cid"
        ],
        "prior_materialization_receipt_file_sha256": (
            _M7_FROZEN_SUCCESSOR_BINDING["migration_receipt_file_sha256"]
        ),
        "prior_owner_status_path": _M7_FROZEN_SUCCESSOR_BINDING[
            "owner_status_path"
        ],
        "prior_owner_status_sha256": _M7_FROZEN_SUCCESSOR_BINDING[
            "owner_status_sha256"
        ],
        "prior_validator_digest": _M7_FROZEN_SUCCESSOR_BINDING[
            "validator_digest"
        ],
        "prior_semantic_authority_digest": _M7_FROZEN_SUCCESSOR_BINDING[
            "semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": _M7_FROZEN_SUCCESSOR_BINDING[
            "frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": _M7_FROZEN_SUCCESSOR_BINDING[
            "append_surface_digest"
        ],
        "target_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m8/control.duckdb"
        ),
        "target_generation": _M8_TARGET_GENERATION,
        "target_plan_revision": _M8_TARGET_PLAN_REVISION,
        "target_event_watermark": _M8_TARGET_EVENT_WATERMARK,
        "target_projection_cid": _M8_EXPECTED_PROJECTION_CID,
        "event_suffix_length": _M8_EVENT_SUFFIX_LENGTH,
        "bounded_control_plane_repair_paths": repair_paths,
        "live_preflight_failure": _M7_LIVE_PREFLIGHT_FAILURE,
        "live_preflight_failure_cid": _identity(_M7_LIVE_PREFLIGHT_FAILURE),
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    configured = config.get("source_repair_successor_materialization")
    inventoried = population["migration_inventory"].get(
        "source_repair_successor_materialization"
    )
    if (
        configured != expected
        or inventoried != expected
        or _identity(expected) != _M8_SOURCE_REPAIR_AUTHORITY_CID
    ):
        raise MaterializationError(
            "M8 source-only repair authority differs across scheduler and inventory"
        )
    return expected


def _m9_live_recovery_authority(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the closed M9 live-failure recovery authority."""

    repair_paths = [
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        (
            "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ),
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "implementation_daemon_runner.py"
        ),
        (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "implementation_supervisor.py"
        ),
        (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "database_portal_bridge.py"
        ),
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
    ]
    task_rearm = {
        "schema": "sawm/control-plane-task-rearm-authorization@1",
        "task_alias": "SAWM-001",
        "task_cid": _M8_LIVE_FAILURE_RECEIPT["task_cid"],
        "from_status": "blocked",
        "from_revision": 9,
        "to_status": "retrying",
        "to_revision": 10,
        "settlement_id": _M8_LIVE_FAILURE_RECEIPT["settlement_id"],
        "historical_settlement_preserved": True,
        "automatic_retry_admitted": False,
        "operation": "operator_control_plane_repair",
        "coordination_rearm_required": True,
        "coordination_rearm_observed_at_ms": (
            _M9_COORDINATION_REARM_OBSERVED_AT_MS
        ),
        "prior_coordination_event_count": 35,
        "target_coordination_event_count": 36,
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "implementation_commit_count": 0,
        "merge_attempt_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    expected = {
        "schema": "sawm/control-plane-runtime-recovery-authorization@1",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": _M9_MIGRATION_REVISION,
        "migration_kind": _M9_SUPERSESSION_REASON,
        "supersession_mode": _M9_SUPERSESSION_MODE,
        "prior_store_id": _M8_FROZEN_LIVE_BINDING["store_id"],
        "prior_control_store_sha256": _M8_FROZEN_LIVE_BINDING[
            "control_store_sha256"
        ],
        "prior_pre_audit_control_store_sha256": _M8_FROZEN_LIVE_BINDING[
            "pre_audit_control_store_sha256"
        ],
        "prior_offline_checkpoint_semantic_change": False,
        "prior_control_store_size": _M8_FROZEN_LIVE_BINDING[
            "control_store_size"
        ],
        "prior_coordination_store_id": _M8_FROZEN_LIVE_BINDING[
            "coordination_store_id"
        ],
        "prior_coordination_store_sha256": _M8_FROZEN_LIVE_BINDING[
            "coordination_store_sha256"
        ],
        "prior_coordination_store_size": _M8_FROZEN_LIVE_BINDING[
            "coordination_store_size"
        ],
        "prior_coordination_event_count": _M8_FROZEN_LIVE_BINDING[
            "coordination_event_count"
        ],
        "prior_coordination_projection_digest": (
            _M8_COORDINATION_PROJECTION_DIGEST
        ),
        "prior_event_watermark": _M8_FROZEN_LIVE_BINDING[
            "event_watermark"
        ],
        "prior_event_prefix_sha256": _M8_FROZEN_LIVE_BINDING[
            "event_prefix_sha256"
        ],
        "prior_projection_cid": _M8_FROZEN_LIVE_BINDING[
            "projection_cid"
        ],
        "prior_source_head": _M8_FROZEN_LIVE_BINDING["source_head"],
        "prior_source_tree": _M8_FROZEN_LIVE_BINDING["source_tree"],
        "prior_source_binding_cid": _M8_FROZEN_LIVE_BINDING[
            "source_binding_cid"
        ],
        "prior_database_uuid": _M8_FROZEN_LIVE_BINDING["database_uuid"],
        "prior_generation": _M8_FROZEN_LIVE_BINDING["generation"],
        "prior_plan_revision": _M8_FROZEN_LIVE_BINDING["plan_revision"],
        "prior_process_birth_id": _M8_FROZEN_LIVE_BINDING[
            "process_birth_id"
        ],
        "prior_server_id": _M8_FROZEN_LIVE_BINDING["server_id"],
        "prior_materialization_receipt_path": _M8_FROZEN_LIVE_BINDING[
            "migration_receipt_path"
        ],
        "prior_materialization_receipt_cid": _M8_FROZEN_LIVE_BINDING[
            "migration_receipt_cid"
        ],
        "prior_materialization_receipt_file_sha256": (
            _M8_FROZEN_LIVE_BINDING["migration_receipt_file_sha256"]
        ),
        "prior_owner_status_path": _M8_FROZEN_LIVE_BINDING[
            "owner_status_path"
        ],
        "prior_owner_status_sha256": _M8_FROZEN_LIVE_BINDING[
            "owner_status_sha256"
        ],
        "prior_validator_digest": _M8_FROZEN_LIVE_BINDING[
            "validator_digest"
        ],
        "prior_semantic_authority_digest": _M8_FROZEN_LIVE_BINDING[
            "semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": _M8_FROZEN_LIVE_BINDING[
            "frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": _M8_FROZEN_LIVE_BINDING[
            "append_surface_digest"
        ],
        "target_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m9/control.duckdb"
        ),
        "target_coordination_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m9/control.coordination.duckdb"
        ),
        "target_generation": _M9_TARGET_GENERATION,
        "target_plan_revision": _M9_TARGET_PLAN_REVISION,
        "target_event_watermark": _M9_TARGET_EVENT_WATERMARK,
        "target_projection_cid": _M9_EXPECTED_PROJECTION_CID,
        "target_coordination_projection_digest": (
            _M9_COORDINATION_PROJECTION_DIGEST
        ),
        "event_suffix_length": _M9_EVENT_SUFFIX_LENGTH,
        "bounded_control_plane_repair_paths": repair_paths,
        "live_implementation_failure": _M8_LIVE_IMPLEMENTATION_FAILURE,
        "task_rearm": task_rearm,
        "provider_strategy": {
            "prior_route": (
                "grok-4.6_then_codex_on_independently_verified_quota"
            ),
            "target_route": (
                "grok-4.6_then_codex_on_independently_verified_quota"
            ),
            "route_changed": False,
            "capability_probe_required": True,
            "provider_result_is_completion_authority": False,
        },
        "task_revision_changes": 1,
        "task_status_changes": 1,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    configured = config.get("live_recovery_successor_materialization")
    inventoried = population["migration_inventory"].get(
        "live_recovery_successor_materialization"
    )
    if (
        configured != expected
        or inventoried != expected
        or _identity(expected) != _M9_LIVE_RECOVERY_AUTHORITY_CID
    ):
        raise MaterializationError(
            "M9 live recovery authority differs across scheduler and inventory"
        )
    return expected


def _expected_m10_live_projection_authority() -> dict[str, Any]:
    """Return the closed M10 source-only comparator-repair authority."""

    repair_paths = [
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        (
            "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ),
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
    ]
    return {
        "schema": "sawm/live-control-projection-repair-authorization@1",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": _M10_MIGRATION_REVISION,
        "migration_kind": _M10_SUPERSESSION_REASON,
        "supersession_mode": _M10_SUPERSESSION_MODE,
        "prior_store_id": _M9_FROZEN_LIVE_BINDING["store_id"],
        "prior_publication_control_store_sha256": _M9_FROZEN_LIVE_BINDING[
            "publication_control_store_sha256"
        ],
        "prior_control_store_sha256": _M9_FROZEN_LIVE_BINDING[
            "control_store_sha256"
        ],
        "prior_live_checkpoint_semantic_change": False,
        "prior_control_store_size": _M9_FROZEN_LIVE_BINDING[
            "control_store_size"
        ],
        "prior_coordination_store_id": _M9_FROZEN_LIVE_BINDING[
            "coordination_store_id"
        ],
        "prior_coordination_store_sha256": _M9_FROZEN_LIVE_BINDING[
            "coordination_store_sha256"
        ],
        "prior_coordination_store_size": _M9_FROZEN_LIVE_BINDING[
            "coordination_store_size"
        ],
        "prior_coordination_event_count": _M9_FROZEN_LIVE_BINDING[
            "coordination_event_count"
        ],
        "prior_coordination_projection_digest": (
            _M9_COORDINATION_PROJECTION_DIGEST
        ),
        "prior_event_watermark": _M9_FROZEN_LIVE_BINDING["event_watermark"],
        "prior_event_prefix_sha256": _M9_FROZEN_LIVE_BINDING[
            "event_prefix_sha256"
        ],
        "prior_projection_cid": _M9_FROZEN_LIVE_BINDING["projection_cid"],
        "prior_source_head": _M9_FROZEN_LIVE_BINDING["source_head"],
        "prior_source_tree": _M9_FROZEN_LIVE_BINDING["source_tree"],
        "prior_source_binding_cid": _M9_FROZEN_LIVE_BINDING[
            "source_binding_cid"
        ],
        "prior_database_uuid": _M9_FROZEN_LIVE_BINDING["database_uuid"],
        "prior_generation": _M9_FROZEN_LIVE_BINDING["generation"],
        "prior_plan_revision": _M9_FROZEN_LIVE_BINDING["plan_revision"],
        "prior_process_birth_id": _M9_FROZEN_LIVE_BINDING[
            "process_birth_id"
        ],
        "prior_server_id": _M9_FROZEN_LIVE_BINDING["server_id"],
        "prior_materialization_receipt_path": _M9_FROZEN_LIVE_BINDING[
            "migration_receipt_path"
        ],
        "prior_materialization_receipt_cid": _M9_FROZEN_LIVE_BINDING[
            "migration_receipt_cid"
        ],
        "prior_materialization_receipt_file_sha256": _M9_FROZEN_LIVE_BINDING[
            "migration_receipt_file_sha256"
        ],
        "prior_owner_status_path": _M9_FROZEN_LIVE_BINDING[
            "owner_status_path"
        ],
        "prior_owner_status_sha256": _M9_FROZEN_LIVE_BINDING[
            "owner_status_sha256"
        ],
        "prior_semantic_authority_digest": _M9_FROZEN_LIVE_BINDING[
            "semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": _M9_FROZEN_LIVE_BINDING[
            "frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": _M9_FROZEN_LIVE_BINDING[
            "append_surface_digest"
        ],
        "prior_catalog_digest": _M9_FROZEN_LIVE_BINDING["catalog_digest"],
        "target_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m10/control.duckdb"
        ),
        "target_coordination_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m10/control.coordination.duckdb"
        ),
        "target_generation": _M10_TARGET_GENERATION,
        "target_plan_revision": _M10_TARGET_PLAN_REVISION,
        "target_event_watermark": _M10_TARGET_EVENT_WATERMARK,
        "target_projection_cid": _M10_EXPECTED_PROJECTION_CID,
        "target_coordination_projection_digest": (
            _M9_COORDINATION_PROJECTION_DIGEST
        ),
        "target_coordination_event_count": 36,
        "target_semantic_authority_digest": _M9_FROZEN_LIVE_BINDING[
            "semantic_authority_digest"
        ],
        "event_suffix_length": _M10_EVENT_SUFFIX_LENGTH,
        "bounded_control_plane_repair_paths": repair_paths,
        "live_preflight_failure": _M9_LIVE_PREFLIGHT_FAILURE,
        "live_preflight_failure_cid": _identity(_M9_LIVE_PREFLIGHT_FAILURE),
        "live_task_projection": {
            "schema": "sawm/live-task-projection-expectation@1",
            "task_count": 45,
            "task_revision_count": 1,
            "operator_task_alias": "SAWM-000",
            "operator_status": "completed",
            "operator_revision": 2,
            "candidate_task_alias": "SAWM-001",
            "candidate_task_cid": _M9_LIVE_PREFLIGHT_FAILURE["task_cid"],
            "candidate_status": "retrying",
            "candidate_revision": 10,
            "candidate_completion_receipt": {
                "operation": "operator_control_plane_repair",
                "settlement_id": (
                    "baguqeeransrfearh5ojrnru6ls43mn7xsx6mlhndkhokrn4k7wlhowxjnshq"
                ),
            },
            "remaining_task_status": "todo",
            "remaining_task_revision": 2,
        },
        "provider_strategy": {
            "prior_route": (
                "grok-4.6_then_codex_on_independently_verified_quota"
            ),
            "target_route": (
                "grok-4.6_then_codex_on_independently_verified_quota"
            ),
            "route_changed": False,
            "capability_probe_required": True,
            "provider_result_is_completion_authority": False,
        },
        "coordination_sidecar_copied_unchanged": True,
        "coordination_semantic_changes": 0,
        "execution_sidecar_copied": False,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }


def _m10_live_projection_authority(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile the exact M10 authority across scheduler and inventory."""

    expected = _expected_m10_live_projection_authority()
    configured = config.get("live_projection_successor_materialization")
    inventoried = population["migration_inventory"].get(
        "live_projection_successor_materialization"
    )
    if (
        configured != expected
        or inventoried != expected
        or _identity(expected) != _M10_LIVE_PROJECTION_AUTHORITY_CID
    ):
        raise MaterializationError(
            "M10 live projection authority differs across scheduler and inventory"
        )
    return expected


def _assert_m8_source_delta(
    root: Path,
    population: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Prove the committed M7-to-M8 source delta is exactly authorized."""

    prior_head = str(authority["prior_source_head"])
    current_head = str(population["source_binding"]["head"])
    if current_head == prior_head:
        raise MaterializationError("M8 source repair has no successor commit")
    _git(root, "merge-base", "--is-ancestor", prior_head, current_head)
    changed: set[str] = set()
    output = _git(
        root,
        "diff",
        "--name-status",
        "--no-renames",
        prior_head,
        current_head,
        "--",
    )
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] != "M":
            raise MaterializationError(
                f"M8 source delta contains a non-modification entry: {line}"
            )
        path = fields[1]
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise MaterializationError("M8 source delta path is not confined")
        changed.add(path)
    authorized = set(authority["bounded_control_plane_repair_paths"])
    if changed != authorized:
        raise MaterializationError(
            "M8 committed source delta differs from the exact repair paths: "
            + json.dumps(
                {
                    "missing": sorted(authorized - changed),
                    "unexpected": sorted(changed - authorized),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def _assert_m9_source_delta(
    root: Path,
    population: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Prove the committed M8-to-M9 blocker repair is exactly confined."""

    prior_head = str(authority["prior_source_head"])
    current_head = str(population["source_binding"]["head"])
    if current_head == prior_head:
        raise MaterializationError("M9 recovery has no successor commit")
    _git(root, "merge-base", "--is-ancestor", prior_head, current_head)
    changed: set[str] = set()
    output = _git(
        root,
        "diff",
        "--name-status",
        "--no-renames",
        prior_head,
        current_head,
        "--",
    )
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] != "M":
            raise MaterializationError(
                f"M9 source delta contains a non-modification entry: {line}"
            )
        path = fields[1]
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise MaterializationError("M9 source delta path is not confined")
        changed.add(path)
    authorized = set(authority["bounded_control_plane_repair_paths"])
    if changed != authorized:
        raise MaterializationError(
            "M9 committed source delta differs from the exact recovery paths: "
            + json.dumps(
                {
                    "missing": sorted(authorized - changed),
                    "unexpected": sorted(changed - authorized),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def _assert_m10_source_delta(
    root: Path,
    population: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    """Prove the committed M9-to-M10 comparator repair is exactly confined."""

    prior_head = str(authority["prior_source_head"])
    current_head = str(population["source_binding"]["head"])
    if current_head == prior_head:
        raise MaterializationError("M10 projection repair has no successor commit")
    _git(root, "merge-base", "--is-ancestor", prior_head, current_head)
    changed: set[str] = set()
    output = _git(
        root,
        "diff",
        "--name-status",
        "--no-renames",
        prior_head,
        current_head,
        "--",
    )
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] != "M":
            raise MaterializationError(
                f"M10 source delta contains a non-modification entry: {line}"
            )
        path = fields[1]
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise MaterializationError("M10 source delta path is not confined")
        changed.add(path)
    authorized = set(authority["bounded_control_plane_repair_paths"])
    if changed != authorized:
        raise MaterializationError(
            "M10 committed source delta differs from the exact projection repair paths: "
            + json.dumps(
                {
                    "missing": sorted(authorized - changed),
                    "unexpected": sorted(changed - authorized),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def _assert_offline(path: Path) -> None:
    # Direct DuckDB verification is an offline operation.  Once the Quack
    # state owner is live, every read and write must pass through that owner;
    # opening the file here would violate the single-owner authority boundary.
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        discover_live_quack_endpoint,
    )
    discovery = discover_live_quack_endpoint(path)
    owner_marker = path.with_name(f".{path.name}.state-owner.json")
    if discovery.uri or owner_marker.exists():
        detail = discovery.reason or "active_owner_marker"
        raise MaterializationError(
            f"offline store verification refused while Quack ownership may be active: {detail}"
        )


def _event_prefix_digest(connection: Any, watermark: int) -> tuple[str, int]:
    rows = connection.execute(
        """
        SELECT global_sequence, event_id, event_type, stream_id, sequence,
               task_cid, attempt_id, session_id, recorded_at, body_json
        FROM domain_events
        WHERE global_sequence <= ?
        ORDER BY global_sequence
        """,
        [int(watermark)],
    ).fetchall()
    normalized_rows = [
        [row[index] for index in range(10)]
        for row in rows
    ]
    payload = json.dumps(
        normalized_rows,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(rows)


def _verify_receipt_anchor(path: Path, expected_cid: str) -> None:
    if not path.is_file():
        raise MigrationRequired(f"sealed receipt is missing: {path}")
    receipt = _load_json(path)
    claimed = str(receipt.pop("receipt_cid", ""))
    if claimed != expected_cid or _identity(receipt) != expected_cid:
        raise MigrationRequired(f"sealed receipt identity differs: {path}")


def _verify_migration_history(
    root: Path,
    connection: Any,
    migration: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Verify the complete immutable predecessor chain before adding a suffix."""

    raw_history = migration.get("migration_history")
    if not isinstance(raw_history, list) or not raw_history:
        raise MigrationRequired("sealed migration history is missing")
    history: list[Mapping[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for ordinal, raw in enumerate(raw_history, 1):
        if not isinstance(raw, Mapping):
            raise MigrationRequired("sealed migration history entry is not an object")
        entry = dict(raw)
        expected_revision = f"SAWM-R2-M{ordinal}"
        schema = entry.get("schema")
        prior_watermark = int(entry.get("prior_event_watermark") or 0)
        migration_watermark = int(
            entry.get("migration_event_watermark")
            or entry.get("target_event_watermark")
            or 0
        )
        materialization_watermark = int(
            entry.get("materialization_event_watermark")
            or entry.get("target_event_watermark")
            or 0
        )
        target_watermark = int(entry.get("target_event_watermark") or 0)
        if entry.get("migration_revision") != expected_revision:
            raise MigrationRequired("sealed migration history sequence differs")
        if schema == "sawm/source-migration-history-entry@1":
            if (
                ordinal > 3
                or migration_watermark != prior_watermark + 2
                or target_watermark != migration_watermark
            ):
                raise MigrationRequired("sealed @1 migration history sequence differs")
        elif schema == "sawm/source-migration-history-entry@2":
            if (
                ordinal != 4
                or migration_watermark != prior_watermark + 2
                or target_watermark != migration_watermark + 1
                or entry.get("post_migration_event_type")
                != "intent.task_status_changed"
                or entry.get("post_migration_task_status") != "in_progress"
                or int(entry.get("post_migration_task_revision") or 0) != 2
            ):
                raise MigrationRequired("sealed @2 migration history sequence differs")
        elif schema == "sawm/source-migration-history-entry@3":
            runtime_events = entry.get("post_materialization_events")
            if (
                ordinal != 5
                or migration_watermark != prior_watermark + 2
                or materialization_watermark != migration_watermark + 1
                or target_watermark != materialization_watermark + 2
                or not isinstance(runtime_events, list)
                or len(runtime_events) != 2
                or [int(item.get("global_sequence") or 0) for item in runtime_events]
                != [materialization_watermark + 1, materialization_watermark + 2]
                or [item.get("event_type") for item in runtime_events]
                != ["intent.task_status_changed", "intent.task_status_changed"]
                or [item.get("task_status") for item in runtime_events]
                != ["in_progress", "blocked"]
                or [int(item.get("task_revision") or 0) for item in runtime_events]
                != [4, 5]
            ):
                raise MigrationRequired("sealed @3 migration history sequence differs")
        else:
            raise MigrationRequired("sealed migration history schema differs")
        if previous is not None and (
            entry.get("prior_store_id") != previous.get("target_store_id")
            or entry.get("prior_control_store_sha256")
            != previous.get("target_control_store_sha256")
            or entry.get("prior_event_watermark")
            != previous.get("target_event_watermark")
            or entry.get("prior_event_prefix_sha256")
            != previous.get("target_event_prefix_sha256")
            or entry.get("prior_source_binding_cid")
            != previous.get("current_source_binding_cid")
        ):
            raise MigrationRequired("sealed migration history continuity differs")
        for path_key, digest_key in (
            ("prior_store_id", "prior_control_store_sha256"),
            ("target_store_id", "target_control_store_sha256"),
        ):
            store_path = (root / str(entry.get(path_key) or "")).resolve()
            if not store_path.is_relative_to(root) or not store_path.is_file():
                raise MigrationRequired("sealed migration-history store is missing")
            if _store_sha256(store_path) != entry.get(digest_key):
                raise MigrationRequired("sealed migration-history store bytes differ")
        receipt_path = (root / str(entry.get("migration_receipt_path") or "")).resolve()
        if not receipt_path.is_relative_to(root):
            raise MigrationRequired("sealed migration-history receipt escapes the source root")
        _verify_receipt_anchor(
            receipt_path,
            str(entry.get("migration_receipt_cid") or ""),
        )
        receipt_file_sha256 = str(entry.get("migration_receipt_file_sha256") or "")
        if (
            receipt_file_sha256
            and _store_sha256(receipt_path) != receipt_file_sha256
        ):
            raise MigrationRequired("sealed migration-history receipt bytes differ")
        evidence = connection.execute(
            "SELECT digest FROM evidence_nodes WHERE evidence_id = ? "
            "AND evidence_kind = 'operator_control_plane_source_migration'",
            [entry.get("migration_evidence_id")],
        ).fetchall()
        if len(evidence) != 1 or str(evidence[0][0]) != entry.get("migration_digest"):
            raise MigrationRequired("sealed migration-history evidence differs")
        event_sequences = [migration_watermark - 1, migration_watermark]
        expected_event_ids = [
            entry.get("plan_migration_event_id"),
            entry.get("migration_evidence_event_id"),
        ]
        if schema == "sawm/source-migration-history-entry@2":
            event_sequences.append(target_watermark)
            expected_event_ids.append(entry.get("post_migration_event_id"))
        elif schema == "sawm/source-migration-history-entry@3":
            event_sequences.append(materialization_watermark)
            expected_event_ids.append(entry.get("materialization_event_id"))
            for runtime_event in entry["post_materialization_events"]:
                event_sequences.append(int(runtime_event["global_sequence"]))
                expected_event_ids.append(runtime_event.get("event_id"))
        placeholders = ",".join("?" for _ in event_sequences)
        event_ids = connection.execute(
            "SELECT global_sequence, event_id, event_type, task_cid, body_json "
            f"FROM domain_events WHERE global_sequence IN ({placeholders}) "
            "ORDER BY global_sequence",
            event_sequences,
        ).fetchall()
        if [str(row[1]) for row in event_ids] != expected_event_ids:
            raise MigrationRequired("sealed migration-history event identities differ")
        if schema == "sawm/source-migration-history-entry@2":
            post_event = event_ids[-1]
            post_payload = json.loads(str(post_event[4]))
            post_body = post_payload.get("body") or {}
            if (
                int(post_event[0]) != target_watermark
                or str(post_event[2]) != entry.get("post_migration_event_type")
                or str(post_event[3]) != entry.get("post_migration_task_cid")
                or post_body.get("task_cid") != entry.get("post_migration_task_cid")
                or post_body.get("status") != entry.get("post_migration_task_status")
                or int(post_body.get("revision") or 0)
                != int(entry.get("post_migration_task_revision") or 0)
            ):
                raise MigrationRequired("sealed post-migration runtime event differs")
            migration_prefix, migration_count = _event_prefix_digest(
                connection, migration_watermark
            )
            if (
                migration_count != migration_watermark
                or migration_prefix != entry.get("migration_event_prefix_sha256")
            ):
                raise MigrationRequired("sealed M4 migration-only event prefix differs")
        elif schema == "sawm/source-migration-history-entry@3":
            materialization_event = event_ids[2]
            materialization_payload = json.loads(str(materialization_event[4]))
            materialization_body = materialization_payload.get("body") or {}
            if (
                int(materialization_event[0]) != materialization_watermark
                or str(materialization_event[2]) != "intent.task_status_changed"
                or str(materialization_event[3])
                != entry.get("materialization_task_cid")
                or materialization_body.get("task_cid")
                != entry.get("materialization_task_cid")
                or materialization_body.get("status") != "todo"
                or int(materialization_body.get("revision") or 0) != 3
            ):
                raise MigrationRequired("sealed M5 materialization event differs")
            for observed, expected in zip(
                event_ids[3:], entry["post_materialization_events"], strict=True
            ):
                payload = json.loads(str(observed[4]))
                body = payload.get("body") or {}
                if (
                    int(observed[0]) != int(expected["global_sequence"])
                    or str(observed[1]) != expected["event_id"]
                    or str(observed[2]) != expected["event_type"]
                    or str(observed[3]) != expected["task_cid"]
                    or body.get("task_cid") != expected["task_cid"]
                    or body.get("status") != expected["task_status"]
                    or int(body.get("revision") or 0)
                    != int(expected["task_revision"])
                ):
                    raise MigrationRequired("sealed M5 runtime event differs")
            for watermark, digest_key, noun in (
                (
                    migration_watermark,
                    "migration_event_prefix_sha256",
                    "migration-only",
                ),
                (
                    materialization_watermark,
                    "materialization_event_prefix_sha256",
                    "materialization",
                ),
                (target_watermark, "target_event_prefix_sha256", "runtime"),
            ):
                prefix, count = _event_prefix_digest(connection, watermark)
                if count != watermark or prefix != entry.get(digest_key):
                    raise MigrationRequired(
                        f"sealed M5 {noun} event prefix differs"
                    )
        history.append(entry)
        previous = entry

    latest = history[-1]
    if (
        latest.get("target_store_id") != migration.get("prior_store_id")
        or latest.get("target_control_store_sha256")
        != migration.get("prior_control_store_sha256")
        or latest.get("target_event_watermark")
        != migration.get("prior_event_watermark")
        or latest.get("target_event_prefix_sha256")
        != migration.get("prior_event_prefix_sha256")
        or latest.get("current_source_binding_cid")
        != migration.get("prior_source_binding_cid")
        or latest.get("projection_cid") != migration.get("prior_projection_cid")
        or latest.get("migration_receipt_cid")
        != migration.get("prior_materialization_receipt_cid")
        or latest.get("migration_receipt_path")
        != migration.get("prior_materialization_receipt_path")
    ):
        raise MigrationRequired("latest migration-history entry does not bind the predecessor")
    if latest.get("schema") in {
        "sawm/source-migration-history-entry@2",
        "sawm/source-migration-history-entry@3",
    }:
        latest_materialization_watermark = latest.get(
            "materialization_event_watermark",
            latest.get("migration_event_watermark"),
        )
        latest_materialization_projection = latest.get(
            "materialization_projection_cid",
            latest.get("migration_projection_cid"),
        )
        if (
            latest_materialization_watermark
            != migration.get("prior_materialization_event_watermark")
            or latest_materialization_projection
            != migration.get("prior_materialization_projection_cid")
            or latest.get("migration_receipt_file_sha256")
            != migration.get("prior_materialization_receipt_file_sha256")
        ):
            raise MigrationRequired("latest materialization receipt binding differs")
    return tuple(history)


def _assert_source_binding_matches(
    root: Path,
    population: Mapping[str, Any],
) -> None:
    """Rebind the cached population to the exact current source identity."""

    expected_binding = population.get("source_binding")
    if not isinstance(expected_binding, Mapping):
        raise MaterializationError("cached source binding is unavailable")
    observed_binding = _source_binding(root)
    if dict(observed_binding) != dict(expected_binding):
        raise MaterializationError(
            "source binding changed after the program population was built: "
            + json.dumps(
                {
                    "expected_head": str(expected_binding.get("head") or ""),
                    "expected_tree": str(expected_binding.get("tree") or ""),
                    "expected_source_binding_cid": str(
                        expected_binding.get("source_binding_cid") or ""
                    ),
                    "observed_head": str(observed_binding.get("head") or ""),
                    "observed_tree": str(observed_binding.get("tree") or ""),
                    "observed_source_binding_cid": str(
                        observed_binding.get("source_binding_cid") or ""
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def _assert_committed_clean_source(
    root: Path,
    population: Mapping[str, Any],
) -> None:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise MaterializationError(
            "materialization requires a clean committed source tree"
        )
    controls = tuple(population["source_binding"]["control_sha256"])
    untracked: list[str] = []
    mismatched: list[str] = []
    for relative in controls:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if result.returncode:
            untracked.append(relative)
            continue
        if _git(root, "hash-object", relative) != _git(
            root, "rev-parse", f"HEAD:{relative}"
        ):
            mismatched.append(relative)
    if untracked or mismatched:
        raise MaterializationError(
            "source controls are not exact committed blobs: "
            + json.dumps(
                {"untracked": untracked, "mismatched": mismatched},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    _assert_source_binding_matches(root, population)


def _m8_successor_configured(config: Mapping[str, Any]) -> bool:
    """Select M8 by key presence and reject malformed successor authority."""

    key = "source_repair_successor_materialization"
    if key not in config:
        return False
    if not isinstance(config.get(key), Mapping):
        raise MaterializationError("M8 source-only successor authority is invalid")
    return True


def _m9_successor_configured(config: Mapping[str, Any]) -> bool:
    """Select M9 by key presence and reject malformed recovery authority."""

    key = "live_recovery_successor_materialization"
    if key not in config:
        return False
    if not isinstance(config.get(key), Mapping):
        raise MaterializationError("M9 live recovery authority is invalid")
    return True


def _m10_successor_configured(config: Mapping[str, Any]) -> bool:
    """Select M10 first by key presence and reject malformed authority."""

    key = "live_projection_successor_materialization"
    if key not in config:
        return False
    if not isinstance(config.get(key), Mapping):
        raise MaterializationError("M10 live projection authority is invalid")
    return True


def _m9_target_paths(
    root: Path,
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Resolve and validate the closed M9 store-pair binding."""

    target_id = str(authority["target_store_id"])
    coordination_id = str(authority["target_coordination_store_id"])
    program = config.get("database_program")
    quack_owner = config.get("quack_owner")
    if not isinstance(program, Mapping) or not isinstance(quack_owner, Mapping):
        raise MaterializationError("scheduler M9 database authority is missing")
    if (
        str(program.get("store_id") or "") != target_id
        or int(program.get("store_generation") or 0) != _M9_TARGET_GENERATION
        or str(quack_owner.get("database_path") or "") != target_id
        or str(quack_owner.get("store_id") or "") != target_id
    ):
        raise MaterializationError("scheduler M9 target/generation binding differs")
    control = _confined_leaf(root, target_id, noun="M9 control store")
    coordination = _confined_leaf(
        root,
        coordination_id,
        noun="M9 coordination store",
    )
    if (
        not control.is_relative_to(root)
        or not coordination.is_relative_to(root)
        or control.parent != coordination.parent
        or coordination != control.with_name("control.coordination.duckdb")
    ):
        raise MaterializationError("scheduler M9 store pair is not confined")
    return control, coordination


def _m10_target_paths(
    root: Path,
    config: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Resolve and validate the closed M10 store-pair binding."""

    target_id = str(authority["target_store_id"])
    coordination_id = str(authority["target_coordination_store_id"])
    program = config.get("database_program")
    quack_owner = config.get("quack_owner")
    if not isinstance(program, Mapping) or not isinstance(quack_owner, Mapping):
        raise MaterializationError("scheduler M10 database authority is missing")
    if (
        str(program.get("store_id") or "") != target_id
        or int(program.get("store_generation") or 0) != _M10_TARGET_GENERATION
        or str(quack_owner.get("database_path") or "") != target_id
        or str(quack_owner.get("store_id") or "") != target_id
    ):
        raise MaterializationError("scheduler M10 target/generation binding differs")
    control = _confined_leaf(root, target_id, noun="M10 control store")
    coordination = _confined_leaf(
        root,
        coordination_id,
        noun="M10 coordination store",
    )
    if (
        control == coordination
        or control.parent != coordination.parent
        or not control.is_relative_to(root)
        or not coordination.is_relative_to(root)
    ):
        raise MaterializationError("scheduler M10 store pair is not confined")
    return control, coordination


def _verify_sealed_file(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    *,
    noun: str,
) -> Path:
    path = (root / str(relative_path or "")).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise MigrationRequired(f"sealed {noun} is missing or escapes the source root")
    expected = str(expected_sha256 or "")
    if expected.startswith("sha256:"):
        expected = expected.removeprefix("sha256:")
    if len(expected) != 64 or _store_sha256(path) != expected:
        raise MigrationRequired(f"sealed {noun} bytes differ")
    return path


def _table_count(connection: Any, table_name: str) -> int:
    row = connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
    return int(row[0]) if row is not None else -1


def _verify_preprovider_failure_artifacts(
    root: Path,
    migration: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the frozen M5 sidecars without making them successor authority."""

    failure = migration["preprovider_task_failure"]
    if (
        failure.get("authority_class")
        != "operator_frozen_predecessor_observation"
        or failure.get("owner_status") != "stopped"
        or failure.get("retry_deferred") is not True
        or failure.get("attempt_consumed") is not False
        or failure.get("provider_call_allowed") is not False
        or failure.get("provider_dispatch_attempted") is not False
        or failure.get("provider_invocation_recorded") is not False
        or failure.get("authoritative_completion_evidence") is not False
        or failure.get("settlement_closed") is not True
    ):
        raise MigrationRequired("sealed M5 pre-provider authority classification differs")

    execution_path = _verify_sealed_file(
        root,
        str(failure["execution_store_id"]),
        str(failure["execution_store_sha256"]),
        noun="M5 execution sidecar",
    )
    _verify_sealed_file(
        root,
        str(failure["execution_wal_path"]),
        str(failure["execution_wal_sha256"]),
        noun="M5 execution WAL",
    )
    coordination_path = _verify_sealed_file(
        root,
        str(failure["coordination_store_id"]),
        str(failure["coordination_store_sha256"]),
        noun="M5 coordination sidecar",
    )
    for path_key, digest_key, noun in (
        ("configured_board_launch_log_path", "configured_board_launch_log_sha256", "M5 launch log"),
        ("implementation_daemon_log_path", "implementation_daemon_log_sha256", "M5 implementation log"),
        ("supervisor_event_log_path", "supervisor_event_log_sha256", "M5 supervisor event log"),
        ("supervisor_status_path", "supervisor_status_sha256", "M5 supervisor status"),
        ("merge_queue_store_path", "merge_queue_store_sha256", "M5 merge queue"),
        ("quack_owner_status_path", "quack_owner_status_sha256", "M5 stopped owner status"),
        ("portal_attempt_binding_path", "portal_attempt_binding_sha256", "M5 portal attempt binding"),
    ):
        _verify_sealed_file(
            root,
            str(failure[path_key]),
            str(failure[digest_key]),
            noun=noun,
        )

    import duckdb

    execution = duckdb.connect(str(execution_path), read_only=True)
    try:
        execution_counts = {
            "database_task_attempts": _table_count(execution, "database_task_attempts"),
            "attempt_phases": _table_count(execution, "attempt_phases"),
            "provider_invocations": _table_count(execution, "provider_invocations"),
            "effect_claims": _table_count(execution, "effect_claims"),
        }
        if execution_counts != failure["execution_authority_counts"]:
            raise MigrationRequired("sealed M5 execution-sidecar counts differ")
        attempt = execution.execute(
            "SELECT claim_id, task_cid, task_alias, attempt_number, owner_session_id, "
            "fencing_token, fence_epoch, lease_id, committed_phase, status, revision "
            "FROM database_task_attempts WHERE attempt_id = ?",
            [failure["attempt_id"]],
        ).fetchall()
        expected_attempt = [
            (
                failure["claim_id"],
                failure["task_cid"],
                failure["task_alias"],
                1,
                failure["owner_session_id"],
                int(failure["fencing_token"]),
                int(failure["fence_epoch"]),
                failure["lease_id"],
                failure["database_attempt_committed_phase"],
                failure["database_attempt_status"],
                int(failure["database_attempt_revision"]),
            )
        ]
        if attempt != expected_attempt:
            raise MigrationRequired("sealed M5 execution attempt differs")
        phases = execution.execute(
            "SELECT phase, fencing_token, fence_epoch, revision, body_json "
            "FROM attempt_phases WHERE attempt_id = ? ORDER BY revision",
            [failure["attempt_id"]],
        ).fetchall()
        failed_body = json.loads(str(phases[-1][4])) if phases else {}
        if (
            [(str(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in phases]
            != [("claimed", 1, 1, 1), ("context", 1, 1, 2), ("failed", 1, 1, 3)]
            or failed_body.get("settlement_id") != failure["settlement_id"]
            or failed_body.get("failure_kind")
            != failure["settlement_failure_kind"]
            or failed_body.get("failure_payload_digest")
            != failure["failure_payload_digest"]
            or int(failed_body.get("provider_invocation_count") or 0) != 0
            or int(failed_body.get("effect_claim_count") or 0) != 0
        ):
            raise MigrationRequired("sealed M5 execution settlement differs")
    finally:
        execution.close()

    coordination = duckdb.connect(str(coordination_path), read_only=True)
    try:
        coordination_counts = {
            "task_attempts": _table_count(coordination, "task_attempts"),
            "task_claims": _table_count(coordination, "task_claims"),
            "fenced_leases": _table_count(coordination, "fenced_leases"),
            "resource_claims": _table_count(coordination, "resource_claims"),
            "task_completions": _table_count(coordination, "task_completions"),
        }
        if coordination_counts != failure["coordination_authority_counts"]:
            raise MigrationRequired("sealed M5 coordination-sidecar counts differ")
        attempt = coordination.execute(
            "SELECT task_cid, attempt_number, owner_session_id, fencing_token, "
            "fence_epoch, status, revision FROM task_attempts WHERE attempt_id = ?",
            [failure["attempt_id"]],
        ).fetchall()
        if attempt != [
            (
                failure["task_cid"],
                1,
                failure["owner_session_id"],
                int(failure["fencing_token"]),
                int(failure["fence_epoch"]),
                failure["coordination_attempt_status"],
                int(failure["coordination_attempt_revision"]),
            )
        ]:
            raise MigrationRequired("sealed M5 coordination attempt differs")
        claim = coordination.execute(
            "SELECT task_cid, owner_session_id, fencing_token, fence_epoch, "
            "expires_at_ms, state, revision, attempt_id, attempt_number, lease_id "
            "FROM task_claims WHERE claim_id = ?",
            [failure["claim_id"]],
        ).fetchall()
        if claim != [
            (
                failure["task_cid"],
                failure["owner_session_id"],
                int(failure["fencing_token"]),
                int(failure["fence_epoch"]),
                int(failure["coordination_claim_expires_at_epoch_ms"]),
                failure["coordination_claim_status"],
                int(failure["coordination_claim_revision"]),
                failure["attempt_id"],
                1,
                failure["lease_id"],
            )
        ]:
            raise MigrationRequired("sealed M5 coordination claim differs")
        lease = coordination.execute(
            "SELECT task_cid, owner_session_id, fencing_token, fence_epoch, "
            "expires_at_ms, state, revision, claim_id, attempt_id, attempt_number "
            "FROM fenced_leases WHERE lease_id = ?",
            [failure["lease_id"]],
        ).fetchall()
        if claim and lease != [
            (
                failure["task_cid"],
                failure["owner_session_id"],
                int(failure["fencing_token"]),
                int(failure["fence_epoch"]),
                int(failure["coordination_claim_expires_at_epoch_ms"]),
                failure["coordination_lease_status"],
                int(failure["coordination_lease_revision"]),
                failure["claim_id"],
                failure["attempt_id"],
                1,
            )
        ]:
            raise MigrationRequired("sealed M5 coordination lease differs")
        completion = coordination.execute(
            "SELECT status, body_json FROM task_completions WHERE task_cid = ?",
            [failure["task_cid"]],
        ).fetchall()
        completion_body = json.loads(str(completion[0][1])) if completion else {}
        if (
            len(completion) != 1
            or str(completion[0][0]) != "failed"
            or completion_body.get("settlement_id") != failure["settlement_id"]
        ):
            raise MigrationRequired("sealed M5 coordination settlement differs")
        ready = coordination.execute(
            "SELECT ready FROM coordination_tasks WHERE task_cid = ?",
            [failure["task_cid"]],
        ).fetchall()
        if ready != [(bool(failure["coordination_task_ready"]),)]:
            raise MigrationRequired("sealed M5 coordination readiness differs")
    finally:
        coordination.close()

    binding_path = _verify_sealed_file(
        root,
        str(failure["portal_attempt_binding_path"]),
        str(failure["portal_attempt_binding_sha256"]),
        noun="M5 portal attempt binding",
    )
    binding = _load_json(binding_path)
    claimed_binding_id = str(binding.pop("binding_id", ""))
    if (
        claimed_binding_id != failure["portal_attempt_binding_id"]
        or _identity(binding) != claimed_binding_id
        or binding.get("task_cid") != failure["task_cid"]
        or binding.get("task_alias") != failure["task_alias"]
        or int(binding.get("task_revision") or 0)
        != int(failure["canonical_claim_revision"])
        or binding.get("attempt_id") != failure["attempt_id"]
        or binding.get("claim_id") != failure["claim_id"]
        or binding.get("lease_id") != failure["lease_id"]
        or binding.get("projection_seed_digest")
        != "sha256:" + failure["portal_task_projection_sha256"]
        or binding.get("projection_authority") is not False
    ):
        raise MigrationRequired("sealed M5 portal attempt binding differs")

    portal_dir = binding_path.parent
    event_log = _verify_sealed_file(
        root,
        str((portal_dir / "portal-events.jsonl").relative_to(root)),
        str(failure["portal_event_log_sha256"]),
        noun="M5 portal event log",
    )
    manifest_path = _verify_sealed_file(
        root,
        str((portal_dir / "portal-events.jsonl.manifest.json").relative_to(root)),
        str(failure["portal_event_manifest_sha256"]),
        noun="M5 portal event manifest",
    )
    _verify_sealed_file(
        root,
        str((portal_dir / "task-projection.md").relative_to(root)),
        str(failure["portal_task_projection_sha256"]),
        noun="M5 portal task projection",
    )
    event_rows = [
        json.loads(line)
        for line in event_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(event_rows) != int(failure["portal_event_count"])
        or [int(row.get("sequence") or 0) for row in event_rows]
        != list(
            range(
                int(failure["portal_event_first_sequence"]),
                int(failure["portal_event_last_sequence"]) + 1,
            )
        )
        or event_rows[-1].get("event_id") != failure["portal_event_tail_id"]
        or any(
            row.get("previous_event_id")
            != ("" if index == 0 else event_rows[index - 1].get("event_id"))
            for index, row in enumerate(event_rows)
        )
    ):
        raise MigrationRequired("sealed M5 portal event chain differs")
    events_by_id = {str(row.get("event_id") or ""): row for row in event_rows}
    probe_event = events_by_id.get(str(failure["failure_event_id"])) or {}
    preflight = probe_event.get("dependency_preflight") or {}
    probe = preflight.get("probe") or {}
    exception_event = events_by_id.get(
        str(failure["implementation_exception_event_id"])
    ) or {}
    finished_event = events_by_id.get(
        str(failure["implementation_finished_event_id"])
    ) or {}
    if (
        probe_event.get("type") != failure["failure_event_type"]
        or probe_event.get("task_id") != failure["task_alias"]
        or preflight.get("receipt_id") != failure["preflight_receipt_id"]
        or preflight.get("retry_fingerprint")
        != failure["preflight_retry_fingerprint"]
        or preflight.get("reason") != failure["preflight_reason"]
        or probe.get("reason") != failure["probe_reason"]
        or preflight.get("validation_roots") != failure["validation_roots"]
        or exception_event.get("type") != "implementation_exception"
        or exception_event.get("failure_kind") != failure["failure_kind"]
        or exception_event.get("phase") != failure["failure_phase"]
        or exception_event.get("reason") != failure["failure_reason"]
        or exception_event.get("provider_call_allowed") is not False
        or finished_event.get("type") != "implementation_finished"
        or finished_event.get("provider_dispatched") is not False
        or finished_event.get("implementation_commit")
        or (finished_event.get("merge_result") or {}).get("merged") is not False
        or finished_event.get("attempt_consumed") is not False
    ):
        raise MigrationRequired("sealed M5 pre-provider portal failure differs")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("snapshot_id") != failure["portal_event_snapshot_id"]
        or manifest.get("last_event_id") != failure["portal_event_tail_id"]
        or int(manifest.get("earliest_sequence") or 0)
        != int(failure["portal_event_first_sequence"])
        or int(manifest.get("latest_sequence") or 0)
        != int(failure["portal_event_last_sequence"])
    ):
        raise MigrationRequired("sealed M5 portal manifest binding differs")
    for mutation in failure["owner_mutation_receipts"]:
        path = _verify_sealed_file(
            root,
            mutation["path"],
            mutation["sha256"],
            noun="M5 owner mutation receipt",
        )
        result = _load_json(path)
        observed = result.get("observed") or {}
        if (
            result.get("request_id") != mutation["request_id"]
            or result.get("result_cid") != mutation["result_cid"]
            or observed.get("event_id") != mutation["event_id"]
            or observed.get("old_status") != mutation["old_status"]
            or int(observed.get("old_revision") or 0)
            != int(mutation["old_revision"])
            or observed.get("new_status") != mutation["new_status"]
            or int(observed.get("new_revision") or 0)
            != int(mutation["new_revision"])
        ):
            raise MigrationRequired("sealed M5 owner mutation result differs")
    return {
        "execution_store_sha256": failure["execution_store_sha256"],
        "coordination_store_sha256": failure["coordination_store_sha256"],
        "portal_event_log_sha256": failure["portal_event_log_sha256"],
        "portal_event_manifest_sha256": failure["portal_event_manifest_sha256"],
        "settlement_id": failure["settlement_id"],
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "implementation_commit_count": 0,
        "merge_attempt_count": 0,
    }


def _verify_prior_store(
    root: Path,
    config: Mapping[str, Any],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    migration = population["migration_inventory"]
    if any(
        migration.get(key) != expected
        for key, expected in _M5_FROZEN_PRIOR_BINDING.items()
    ):
        raise MigrationRequired("sealed M5 predecessor binding differs")
    if migration.get("preworker_launch_failure") != _M3_PREWORKER_FAILURE_V2:
        raise MigrationRequired("sealed M3 historical launch failure differs")
    preprovider_failure = migration["preprovider_task_failure"]
    configured = config.get("prior_materialization") or {}
    for inventory_key, config_key in (
        ("migration_revision", "migration_revision"),
        ("prior_store_id", "store_id"),
        ("prior_program_definition_cid", "program_definition_cid"),
        ("prior_plan_root_cid", "plan_root_cid"),
        ("prior_source_binding_cid", "source_binding_cid"),
        ("prior_projection_cid", "projection_cid"),
    ):
        if migration.get(inventory_key) != configured.get(config_key):
            raise MaterializationError(
                f"scheduler and migration inventory disagree on {inventory_key}"
            )
    if (
        configured.get("operator_task_cid")
        != migration["prior_task_cids"]["SAWM-000"]
        or configured.get("inventory_path")
        != "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json"
        or configured.get("reason") != migration["supersession_reason"]
        or configured.get("preserve_append_only") is not True
        or int(configured.get("migration_history_count") or 0)
        != len(migration["migration_history"])
        or int(configured.get("prior_plan_revision") or 0)
        != int(migration["prior_plan_revision"])
        or int(configured.get("target_plan_revision") or 0)
        != int(migration["target_plan_revision"])
        or int(configured.get("migration_event_watermark") or 0)
        != int(migration["prior_event_watermark"]) + _M6_MIGRATION_EVENT_OFFSET
        or int(configured.get("operational_validation_first_event_watermark") or 0)
        != int(migration["prior_event_watermark"]) + _M6_OPERATIONAL_FIRST_OFFSET
        or int(configured.get("operational_validation_last_event_watermark") or 0)
        != int(migration["prior_event_watermark"]) + _M6_OPERATIONAL_LAST_OFFSET
        or int(configured.get("target_event_watermark") or 0)
        != int(migration["prior_event_watermark"]) + _M6_EVENT_SUFFIX_LENGTH
        or configured.get("target_projection_cid") != _M6_EXPECTED_PROJECTION_CID
        or int(configured.get("prior_materialization_event_watermark") or 0)
        != int(migration["prior_materialization_event_watermark"])
        or configured.get("prior_materialization_projection_cid")
        != migration["prior_materialization_projection_cid"]
        or configured.get("prior_materialization_receipt_cid")
        != migration["prior_materialization_receipt_cid"]
        or configured.get("prior_materialization_receipt_path")
        != migration["prior_materialization_receipt_path"]
        or configured.get("prior_materialization_receipt_file_sha256")
        != migration["prior_materialization_receipt_file_sha256"]
    ):
        raise MaterializationError("scheduler prior-authority policy differs from its inventory")
    requeue = configured.get("operational_validation_requeue") or {}
    expected_requeue = {
        "schema": "sawm/operational-validation-requeue-authorization@1",
        "authorized": True,
        "authority": "operator_source_migration",
        "task_alias_first": "SAWM-001",
        "task_alias_last": "SAWM-044",
        "task_count": _M6_OPERATIONAL_TASK_COUNT,
        "task_identity_manifest_cid": _M6_TASK_IDENTITY_MANIFEST_CID,
        "validation_command_count": _M6_OPERATIONAL_VALIDATION_COUNT,
        "validation_token_from": _M6_VALIDATION_TOKEN_FROM,
        "validation_token_to": _M6_VALIDATION_TOKEN_TO,
        "recovered_task_alias": "SAWM-001",
        "recovered_task_cid": migration["prior_task_cids"]["SAWM-001"],
        "from_status": "blocked",
        "from_revision": 5,
        "operational_status": "blocked",
        "operational_revision": 6,
        "to_status": "todo",
        "to_revision": 7,
        "reason": migration["migration_kind"],
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "implementation_commit_count": 0,
        "merge_attempt_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "operational_validation_revision_changes": _M6_OPERATIONAL_TASK_COUNT,
        "expected_event_watermark": int(migration["prior_event_watermark"])
        + _M6_EVENT_SUFFIX_LENGTH,
        "worker_self_approval": False,
    }
    if requeue != expected_requeue:
        raise MaterializationError(
            "scheduler operational-validation requeue authority differs from its inventory"
        )
    if not isinstance(config.get("validation_runtime"), Mapping) or not config[
        "validation_runtime"
    ]:
        raise MaterializationError("scheduler validation runtime is not sealed")
    prior = (root / str(migration["prior_store_id"])).resolve()
    target = (root / str(migration["target_store_id"])).resolve()
    if not prior.is_relative_to(root) or not target.is_relative_to(root) or prior == target:
        raise MaterializationError("prior and target stores must be distinct confined paths")
    if not prior.is_file():
        raise MigrationRequired(f"sealed prior authority is missing: {prior}")
    _assert_offline(prior)
    initial_hash = _store_sha256(prior)
    if initial_hash != migration["prior_control_store_sha256"]:
        raise MigrationRequired("sealed prior control-store bytes differ")

    import duckdb
    connection = duckdb.connect(str(prior), read_only=True)
    try:
        task_rows = connection.execute(
            "SELECT task_alias, task_cid, status, revision, body_json "
            "FROM tasks ORDER BY ordinal"
        ).fetchall()
        goal_rows = connection.execute(
            "SELECT goal_alias, goal_cid, body_json FROM goals ORDER BY ordinal"
        ).fetchall()
        task_map = {str(row[0]): str(row[1]) for row in task_rows}
        goal_map = {str(row[0]): str(row[1]) for row in goal_rows}
        if len(task_rows) != int(migration["prior_task_count"]):
            raise MigrationRequired("sealed prior task count differs")
        if len(goal_rows) != int(migration["prior_goal_count"]):
            raise MigrationRequired("sealed prior goal count differs")
        if task_map != migration["prior_task_cids"]:
            raise MigrationRequired("sealed prior task identity map differs")
        if goal_map != migration["prior_goal_cids"]:
            raise MigrationRequired("sealed prior goal identity map differs")
        for row in task_rows:
            body = json.loads(str(row[4]))
            if body.get("program_definition_cid") != migration["prior_program_definition_cid"]:
                raise MigrationRequired(
                    f"sealed prior task program binding differs: {row[0]}"
                )
        for row in goal_rows:
            body = json.loads(str(row[2]))
            if body.get("program_definition_cid") != migration["prior_program_definition_cid"]:
                raise MigrationRequired(
                    f"sealed prior goal program binding differs: {row[0]}"
                )
        objective = connection.execute(
            "SELECT objective_id, objective_alias, body_json FROM objectives"
        ).fetchall()
        if len(objective) != 1 or (
            str(objective[0][0]) != migration["prior_objective_id"]
            or str(objective[0][1]) != ROOT_GOAL
            or json.loads(str(objective[0][2])).get("program_definition_cid")
            != migration["prior_program_definition_cid"]
        ):
            raise MigrationRequired("sealed prior objective binding differs")
        completed = [str(row[0]) for row in task_rows if str(row[2]) == "completed"]
        if completed != ["SAWM-000"]:
            raise MigrationRequired(f"sealed prior completion set differs: {completed}")
        statuses = {str(row[0]): str(row[2]) for row in task_rows}
        expected_statuses = {
            alias: (
                "completed"
                if alias == "SAWM-000"
                else "blocked"
                if alias == "SAWM-001"
                else "todo"
            )
            for alias in migration["prior_task_cids"]
        }
        if statuses != expected_statuses:
            raise MigrationRequired("sealed M5 canonical task states differ")
        sawm_001 = next(row for row in task_rows if row[0] == "SAWM-001")
        if (
            int(sawm_001[3])
            != int(preprovider_failure["canonical_task_revision"])
            or str(sawm_001[1]) != preprovider_failure["task_cid"]
        ):
            raise MigrationRequired("sealed M5 SAWM-001 revision binding differs")
        canonical_counts = {
            table_name: _table_count(connection, table_name)
            for table_name in preprovider_failure["canonical_authority_counts"]
        }
        if canonical_counts != preprovider_failure["canonical_authority_counts"]:
            raise MigrationRequired("sealed M5 canonical authority counts differ")
        owner = connection.execute(
            "SELECT server_id, process_birth_id, status "
            "FROM state_servers WHERE generation = ?",
            [int(preprovider_failure["owner_generation"])],
        ).fetchall()
        if owner != [
            (
                preprovider_failure["owner_server_id"],
                preprovider_failure["owner_process_birth_id"],
                "stopped",
            )
        ]:
            raise MigrationRequired("sealed M5 launch owner identity or stopped state differs")
        operator = next(row for row in task_rows if row[0] == "SAWM-000")
        if str(operator[2]) != migration["prior_operator_status"] or int(operator[3]) != 2:
            raise MigrationRequired("sealed prior operator status/revision differs")

        plan = connection.execute(
            "SELECT revision, body_json FROM plans WHERE plan_cid = ?",
            [migration["prior_plan_root_cid"]],
        ).fetchone()
        if plan is None or int(plan[0]) != int(migration["prior_plan_revision"]):
            raise MigrationRequired("sealed prior plan revision differs")
        plan_body = json.loads(str(plan[1]))
        history = migration["migration_history"]
        latest_history = history[-1]
        if (
            plan_body.get("source_binding_cid")
            != migration["definition_source_binding_cid"]
            or plan_body.get("current_source_binding_cid")
            != migration["prior_source_binding_cid"]
            or plan_body.get("source_migration_revision")
            != latest_history["migration_revision"]
            or plan_body.get("source_migration_digest")
            != latest_history["migration_digest"]
        ):
            raise MigrationRequired("sealed prior plan source binding differs")
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM control_plane_metadata"
            ).fetchall()
        )
        if metadata.get("database_uuid") != migration["prior_database_uuid"]:
            raise MigrationRequired("sealed prior database UUID differs")
        generation = connection.execute(
            "SELECT generation, database_uuid, birth_id FROM store_generations "
            "ORDER BY generation DESC LIMIT 1"
        ).fetchone()
        if generation is None or (
            int(generation[0]) != int(migration["prior_generation"])
            or str(generation[1]) != migration["prior_database_uuid"]
            or str(generation[2]) != migration["prior_process_birth_id"]
        ):
            raise MigrationRequired("sealed prior generation identity differs")
        evidence_ids = {
            str(row[0])
            for row in connection.execute(
                "SELECT evidence_id FROM evidence_nodes WHERE task_cid = ?",
                [migration["prior_task_cids"]["SAWM-000"]],
            ).fetchall()
        }
        if not {
            migration["prior_operator_evidence_id"],
            migration["prior_validation_evidence_id"],
        }.issubset(evidence_ids):
            raise MigrationRequired("sealed prior operator evidence chain differs")
        validation = connection.execute(
            "SELECT outcome FROM validation_results WHERE result_id = ?",
            [migration["prior_validation_result_id"]],
        ).fetchone()
        completion = connection.execute(
            "SELECT task_cid FROM completion_receipts WHERE receipt_cid = ?",
            [migration["prior_completion_receipt_cid"]],
        ).fetchone()
        if validation is None or str(validation[0]) != "passed":
            raise MigrationRequired("sealed prior validation result differs")
        if completion is None or str(completion[0]) != migration["prior_task_cids"]["SAWM-000"]:
            raise MigrationRequired("sealed prior completion receipt differs")
        digest, count = _event_prefix_digest(
            connection, int(migration["prior_event_watermark"])
        )
        if count != int(migration["prior_event_watermark"]):
            raise MigrationRequired("sealed prior event sequence is not contiguous")
        if digest != migration["prior_event_prefix_sha256"]:
            raise MigrationRequired("sealed prior event prefix differs")
        _verify_migration_history(root, connection, migration)
    finally:
        connection.close()

    _verify_receipt_anchor(
        root / str(migration["prior_materialization_receipt_path"]),
        str(migration["prior_materialization_receipt_cid"]),
    )
    preprovider_evidence = _verify_preprovider_failure_artifacts(root, migration)
    # Landed schema and replay verification may obtain a read/write adapter;
    # run those checks only on an isolated copy. The accepted prior authority
    # above is opened read-only and its bytes are rehashed below.
    with tempfile.TemporaryDirectory(prefix="sawm-r2-prior-", dir="/tmp") as temp_dir:
        prior_copy = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(prior, prior_copy)
        prior_verified = _verify_store(
            prior_copy,
            population,
            require_operator_complete=True,
            require_migration=False,
        )
    if prior_verified["projection_cid"] != migration["prior_projection_cid"]:
        raise MigrationRequired("sealed prior projection identity differs")
    if _store_sha256(prior) != initial_hash:
        raise MaterializationError("offline prior verification changed authority bytes")
    return {
        "valid": True,
        "database_path": str(prior),
        "database_sha256": initial_hash,
        "event_prefix_sha256": migration["prior_event_prefix_sha256"],
        "event_watermark": migration["prior_event_watermark"],
        "projection_cid": migration["prior_projection_cid"],
        "preprovider_evidence": preprovider_evidence,
    }


def _verify_store(
    path: Path,
    population: Mapping[str, Any],
    *,
    require_operator_complete: bool,
    require_migration: bool = True,
    migration_config: Mapping[str, Any] | None = None,
    expected_validation_digest: str = "",
) -> dict[str, Any]:
    _assert_offline(path)
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    schema = verify_datasets_authoritative_operational_schema(path)
    if schema.get("valid") is not True:
        raise MigrationRequired(
            "existing database does not verify as datasets-authoritative operational schema"
        )
    source = DatabaseTaskSource(
        path,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
    )
    try:
        migration_details: dict[str, Any] = {}
        snap = source.snapshot()
        expected_tasks = list(population["taskboard"])
        expected_goals = list(population["objectives"])
        migration = population["migration_inventory"]
        if (
            snap.task_count != 45
            or snap.goal_count != 29
            or snap.plan_root_cid != population["plan_root_cid"]
        ):
            raise MigrationRequired(
                "population counts/root conflict: "
                f"tasks={snap.task_count} goals={snap.goal_count} "
                f"root={snap.plan_root_cid}"
            )

        status_by_alias: dict[str, str] = {}
        revision_by_alias: dict[str, int] = {}
        operational_receipts: dict[str, dict[str, Any]] = {}
        for expected in expected_tasks:
            alias = str(expected["task_id"])
            raw = source.intent.get_task(str(expected["task_cid"]))
            observed = source.get_task(str(expected["task_cid"]))
            prior_validations = tuple(
                (str(command),) for command in expected["validation_commands"]
            )
            expected_validations: tuple[tuple[str, ...], ...] = prior_validations
            expected_body_receipt: Mapping[str, Any] | None = None
            if require_migration and alias in _operational_task_aliases():
                expected_validations, replacement_count = (
                    _operational_validation_commands(
                        prior_validations,
                        task_alias=alias,
                    )
                )
                prior_status = "blocked" if alias == "SAWM-001" else "todo"
                prior_revision = 5 if alias == "SAWM-001" else 1
                expected_body_receipt = _operational_validation_receipt(
                    population,
                    task_alias=alias,
                    task_cid=str(expected["task_cid"]),
                    expected_status=prior_status,
                    expected_revision=prior_revision,
                    prior_validations=prior_validations,
                    operational_validations=expected_validations,
                    replacement_count=replacement_count,
                )
                operational_receipts[alias] = dict(expected_body_receipt)
            expected_outputs = [dict(item) for item in expected["outputs"]]
            expected_acceptance = [
                dict(item) for item in expected["acceptance_criteria"]
            ]
            expected_identity = {
                "task_cid": str(expected["task_cid"]),
                "task_alias": alias,
                "repository_tree_id": migration["definition_source_binding_cid"],
            }
            if (
                observed is None
                or raw is None
                or observed.task_alias != alias
                or raw["identity"] != expected_identity
                or observed.body.get("definition_cid")
                != expected["definition_cid"]
                or observed.body.get("definition") != expected["definition"]
                or sorted(observed.dependencies) != sorted(expected["depends_on"])
                or [dict(item.get("effect") or {}) for item in observed.outputs]
                != expected_outputs
                or [dict(item.get("evidence_policy") or {}) for item in observed.acceptance]
                != expected_acceptance
                or tuple(
                    tuple(str(part) for part in item.get("argv") or ())
                    for item in observed.validations
                )
                != expected_validations
                or any(dict(item.get("policy") or {}) for item in observed.validations)
            ):
                raise MigrationRequired(f"task definition conflict: {alias}")
            if require_migration:
                if alias == "SAWM-000":
                    if "operational_validation_revision" in observed.body:
                        raise MigrationRequired("operator task acquired an operational revision")
                elif observed.body.get("operational_validation_revision") != expected_body_receipt:
                    raise MigrationRequired(
                        f"operational validation receipt differs: {alias}"
                    )
            status_by_alias[alias] = observed.status
            revision_by_alias[alias] = int(observed.revision)

        for expected in expected_goals:
            observed = source.get_goal(str(expected["goal_cid"]))
            if (
                observed is None
                or str(observed.get("goal_alias")) != expected["goal_id"]
                or (observed.get("body") or {}).get("definition_cid")
                != expected["definition_cid"]
                or (observed.get("body") or {}).get("definition")
                != expected["definition"]
            ):
                raise MigrationRequired(f"goal definition conflict: {expected['goal_id']}")
        if (
            require_operator_complete
            and status_by_alias.get("SAWM-000")
            not in {"completed", "complete", "done"}
        ):
            raise MigrationRequired("SAWM-000 lacks an admitted completion CAS")

        if require_migration:
            if migration_config is None or not expected_validation_digest:
                raise MaterializationError(
                    "migration verification requires exact config and validator binding"
                )
            expected_statuses = {
                alias: "completed" if alias == "SAWM-000" else "todo"
                for alias in migration["prior_task_cids"]
            }
            expected_revisions = {
                alias: (
                    2
                    if alias == "SAWM-000"
                    else 7
                    if alias == "SAWM-001"
                    else 2
                )
                for alias in migration["prior_task_cids"]
            }
            if (
                status_by_alias != expected_statuses
                or revision_by_alias != expected_revisions
            ):
                raise MigrationRequired("M6 task-state recovery projection differs")
            expected_target_watermark = (
                int(migration["prior_event_watermark"])
                + _M6_EVENT_SUFFIX_LENGTH
            )
            if int(snap.event_cursor) != expected_target_watermark:
                raise MigrationRequired("M6 materialization event watermark differs")
            plan = source.plans.get(str(population["plan_root_cid"]))
            if (
                plan is None
                or int(plan.get("revision") or 0)
                != int(migration["target_plan_revision"])
            ):
                raise MigrationRequired(
                    "append-only source migration plan revision is missing"
                )

            with source.intent._connection(write=False) as connection:
                migration_rows = connection.execute(
                    "SELECT evidence_id, digest, body_json, created_at FROM evidence_nodes "
                    "WHERE task_cid = ? AND evidence_kind = ? ORDER BY created_at",
                    [
                        migration["prior_task_cids"]["SAWM-000"],
                        "operator_control_plane_source_migration",
                    ],
                ).fetchall()
                history = tuple(migration["migration_history"])
                if len(migration_rows) != len(history) + 1:
                    raise MigrationRequired("source migration receipt chain length differs")
                parsed_rows = [
                    {
                        "evidence_id": str(row[0]),
                        "digest": str(row[1]),
                        "body": json.loads(str(row[2])),
                        "created_at": str(row[3]),
                    }
                    for row in migration_rows
                ]
                for entry in history:
                    matched = [
                        row
                        for row in parsed_rows
                        if row["evidence_id"] == entry["migration_evidence_id"]
                    ]
                    if (
                        len(matched) != 1
                        or matched[0]["digest"] != entry["migration_digest"]
                        or matched[0]["body"].get("migration_revision")
                        != entry["migration_revision"]
                    ):
                        raise MigrationRequired("prior source migration evidence differs")
                current_rows = [
                    row
                    for row in parsed_rows
                    if row["body"].get("migration_revision")
                    == migration["migration_revision"]
                ]
                if len(current_rows) != 1:
                    raise MigrationRequired(
                        "current source migration receipt is ambiguous"
                    )
                current_row = current_rows[0]
                migration_evidence_id = current_row["evidence_id"]
                migration_digest = current_row["digest"]
                migration_body = current_row["body"]
                migration_created_at = current_row["created_at"]
                expected_migration_body = _migration_body(
                    population,
                    migration_config,
                    expected_validation_digest,
                )
                if migration_body != expected_migration_body:
                    raise MigrationRequired("source migration receipt binding differs")
                if migration_digest != _identity(migration_body):
                    raise MigrationRequired("source migration evidence digest differs")
                expected_evidence_id = content_identity(
                    {
                        "task_cid": migration["prior_task_cids"]["SAWM-000"],
                        "evidence_kind": "operator_control_plane_source_migration",
                        "digest": migration_digest,
                        "body": migration_body,
                    }
                )
                if migration_evidence_id != expected_evidence_id:
                    raise MigrationRequired("source migration evidence identity differs")

                revision_rows = connection.execute(
                    "SELECT revision, body_json, recorded_at FROM plan_revisions "
                    "WHERE plan_cid = ? AND revision <= ? ORDER BY revision",
                    [population["plan_root_cid"], migration["target_plan_revision"]],
                ).fetchall()
                if [int(row[0]) for row in revision_rows] != list(
                    range(1, int(migration["target_plan_revision"]) + 1)
                ):
                    raise MigrationRequired(
                        "exact source migration plan revisions are missing"
                    )
                prior_plan_body = json.loads(
                    str(revision_rows[int(migration["prior_plan_revision"]) - 1][1])
                )
                plan_delta = _migration_plan_delta(population)
                expected_plan_body = {
                    **prior_plan_body,
                    "current_source_binding_cid": population["source_binding"][
                        "source_binding_cid"
                    ],
                    "source_migration_revision": migration["migration_revision"],
                    "source_migration_digest": migration_digest,
                    "supersession_mode": _M6_SUPERSESSION_MODE,
                    "last_delta": plan_delta,
                }
                observed_plan_body = json.loads(str(revision_rows[-1][1]))
                plan_recorded_at = str(revision_rows[-1][2])
                if observed_plan_body != expected_plan_body:
                    raise MigrationRequired("source migration plan revision body differs")

                prefix_digest, prefix_count = _event_prefix_digest(
                    connection,
                    int(migration["prior_event_watermark"]),
                )
                if (
                    prefix_count != int(migration["prior_event_watermark"])
                    or prefix_digest != migration["prior_event_prefix_sha256"]
                ):
                    raise MigrationRequired(
                        "accepted event prefix changed during migration"
                    )
                suffix_sequences = list(
                    range(
                        int(migration["prior_event_watermark"]) + 1,
                        expected_target_watermark + 1,
                    )
                )
                placeholders = ",".join("?" for _ in suffix_sequences)
                suffix_rows = connection.execute(
                    "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
                    "task_cid, attempt_id, session_id, recorded_at, body_json "
                    f"FROM domain_events WHERE global_sequence IN ({placeholders}) "
                    "ORDER BY global_sequence",
                    suffix_sequences,
                ).fetchall()
                if len(suffix_rows) != _M6_EVENT_SUFFIX_LENGTH:
                    raise MigrationRequired("migration event suffix differs")
                events: list[dict[str, Any]] = []
                for row in suffix_rows:
                    event = {
                        "event_id": str(row[0]),
                        "stream_id": str(row[1]),
                        "sequence": int(row[2]),
                        "global_sequence": int(row[3]),
                        "event_type": str(row[4]),
                        "task_cid": str(row[5]),
                        "attempt_id": str(row[6]),
                        "session_id": str(row[7]),
                        "recorded_at": str(row[8]),
                        "body": json.loads(str(row[9])),
                    }
                    payload = event["body"]
                    if (
                        content_identity(
                            {
                                "stream_id": event["stream_id"],
                                "sequence": event["sequence"],
                                "global_sequence": event["global_sequence"],
                                "event_type": event["event_type"],
                                "body": payload,
                            }
                        )
                        != event["event_id"]
                        or set(payload)
                        != {
                            "schema",
                            "event_type",
                            "subject_id",
                            "body",
                            "recorded_at",
                            "owner_id",
                        }
                        or payload.get("schema")
                        != "ipfs_accelerate_py/agent-supervisor/intent-event@1"
                        or payload.get("event_type") != event["event_type"]
                        or payload.get("recorded_at") != event["recorded_at"]
                        or payload.get("owner_id") != "sawm-r2-source-migrator"
                        or event["stream_id"] != "stream:intent"
                        or event["session_id"] != "session:intent"
                        or event["attempt_id"]
                        or event["sequence"] != event["global_sequence"]
                    ):
                        raise MigrationRequired("migration event identity differs")
                    events.append(event)

                plan_event = events[0]
                evidence_event = events[1]
                operational_events = events[2:-1]
                recovery_event = events[-1]
                expected_plan_event_body = {
                    "plan_cid": population["plan_root_cid"],
                    "goal_cid": migration["prior_goal_cids"][ROOT_GOAL],
                    "plan_alias": REVISION,
                    "status": "active",
                    "revision": int(migration["target_plan_revision"]),
                    "body": expected_plan_body,
                    "delta": plan_delta,
                    "recorded_at": plan_recorded_at,
                }
                if (
                    plan_event["event_type"]
                    != "intent.plan_revision_appended"
                    or plan_event["task_cid"]
                    or plan_event["body"].get("subject_id")
                    != population["plan_root_cid"]
                    or plan_event["body"].get("body")
                    != expected_plan_event_body
                ):
                    raise MigrationRequired("source migration plan event differs")
                expected_evidence_event_body = {
                    "evidence_id": migration_evidence_id,
                    "parent_evidence_id": "",
                    "task_cid": migration["prior_task_cids"]["SAWM-000"],
                    "evidence_kind": "operator_control_plane_source_migration",
                    "digest": migration_digest,
                    "body": migration_body,
                    "created_at": migration_created_at,
                    "revision": 0,
                }
                if (
                    evidence_event["event_type"] != "intent.evidence_recorded"
                    or evidence_event["task_cid"]
                    != migration["prior_task_cids"]["SAWM-000"]
                    or evidence_event["body"].get("subject_id")
                    != migration_evidence_id
                    or evidence_event["body"].get("body")
                    != expected_evidence_event_body
                ):
                    raise MigrationRequired("source migration evidence event differs")

                expected_by_alias = {
                    str(item["task_id"]): item for item in expected_tasks
                }
                operational_event_ids: dict[str, str] = {}
                if len(operational_events) != _M6_OPERATIONAL_TASK_COUNT:
                    raise MigrationRequired(
                        "operational validation event count differs"
                    )
                for event, alias in zip(
                    operational_events,
                    _operational_task_aliases(),
                    strict=True,
                ):
                    expected = expected_by_alias[alias]
                    task_cid = str(expected["task_cid"])
                    prior_status = "blocked" if alias == "SAWM-001" else "todo"
                    prior_revision = 5 if alias == "SAWM-001" else 1
                    target_revision = prior_revision + 1
                    prior_body_row = connection.execute(
                        "SELECT body_json FROM task_revisions "
                        "WHERE task_cid = ? AND revision = ?",
                        [task_cid, prior_revision],
                    ).fetchall()
                    if len(prior_body_row) == 1:
                        prior_body = json.loads(str(prior_body_row[0][0]))
                    elif not prior_body_row and prior_revision == 1:
                        # The original M1 projection predates per-task
                        # revision rows for untouched todo tasks. Reconstruct
                        # their immutable body from the sealed @1 population;
                        # the definition CID and all relation tables were
                        # independently checked above.
                        prior_body = {
                            key: value
                            for key, value in expected.items()
                            if key
                            not in {
                                "task_cid",
                                "task_id",
                                "task_alias",
                                "cid",
                                "goal_cid",
                                "goal_id",
                                "depends_on",
                                "dependencies",
                                "effects",
                                "outputs",
                                "acceptance_criteria",
                                "acceptance",
                                "validation_commands",
                                "validations",
                                "status",
                                "priority",
                                "ordinal",
                                "plan_cid",
                                "objective_id",
                            }
                        }
                    else:
                        raise MigrationRequired(
                            f"historical task revision is ambiguous: {alias}"
                        )
                    expected_body = {
                        **prior_body,
                        "operational_validation_revision": operational_receipts[
                            alias
                        ],
                    }
                    target_revision_rows = connection.execute(
                        "SELECT status, body_json, recorded_at FROM task_revisions "
                        "WHERE task_cid = ? AND revision = ?",
                        [task_cid, target_revision],
                    ).fetchall()
                    if len(target_revision_rows) != 1:
                        raise MigrationRequired(
                            f"operational task revision is ambiguous: {alias}"
                        )
                    target_revision_status = str(target_revision_rows[0][0])
                    target_revision_body = json.loads(
                        str(target_revision_rows[0][1])
                    )
                    target_recorded_at = str(target_revision_rows[0][2])
                    if (
                        target_revision_status != prior_status
                        or target_revision_body != expected_body
                        or not target_recorded_at
                    ):
                        raise MigrationRequired(
                            f"operational task revision differs: {alias}"
                        )
                    operational_validations, _ = _operational_validation_commands(
                        tuple(
                            (str(command),)
                            for command in expected["validation_commands"]
                        ),
                        task_alias=alias,
                    )
                    inner = event["body"].get("body") or {}
                    expected_inner = {
                        "task_cid": task_cid,
                        "task_alias": alias,
                        "goal_cid": str(expected["goal_cid"]),
                        "plan_cid": str(expected["plan_cid"]),
                        "objective_id": "",
                        "ordinal": int(expected["ordinal"]),
                        "status": prior_status,
                        "priority": str(expected["priority"]),
                        "revision": target_revision,
                        "identity": {
                            "task_cid": task_cid,
                            "task_alias": alias,
                            "repository_tree_id": migration[
                                "definition_source_binding_cid"
                            ],
                        },
                        "body": expected_body,
                        # Task revision and event envelope timestamps are made
                        # by two consecutive landed-authority calls.  Their
                        # values may straddle a clock tick; bind the task-body
                        # timestamp to the exact revision row rather than
                        # incorrectly requiring equality with the envelope.
                        "recorded_at": target_recorded_at,
                        "dependencies": sorted(expected["depends_on"]),
                        "outputs": [dict(item) for item in expected["outputs"]],
                        "acceptance": [
                            dict(item) for item in expected["acceptance_criteria"]
                        ],
                        "validations": [
                            {"argv": list(argv)}
                            for argv in operational_validations
                        ],
                    }
                    if (
                        event["event_type"] != "intent.task_upserted"
                        or event["task_cid"] != task_cid
                        or event["body"].get("subject_id") != task_cid
                        or inner != expected_inner
                    ):
                        differing_keys = sorted(
                            key
                            for key in set(inner) | set(expected_inner)
                            if inner.get(key) != expected_inner.get(key)
                        )
                        raise MigrationRequired(
                            "operational validation event differs: "
                            f"{alias}; fields={differing_keys}"
                        )
                    operational_event_ids[alias] = event["event_id"]

                recovery_receipt = _task_recovery_receipt(population)
                expected_recovery_body = {
                    "task_cid": migration["prior_task_cids"]["SAWM-001"],
                    "task_alias": "SAWM-001",
                    "goal_cid": migration["prior_goal_cids"]["SAWM-G011"],
                    "previous_status": "blocked",
                    "status": "todo",
                    "revision": 7,
                    "receipt": recovery_receipt,
                    "recorded_at": recovery_event["recorded_at"],
                }
                if (
                    recovery_event["event_type"]
                    != "intent.task_status_changed"
                    or recovery_event["task_cid"]
                    != migration["prior_task_cids"]["SAWM-001"]
                    or recovery_event["body"].get("subject_id")
                    != migration["prior_task_cids"]["SAWM-001"]
                    or (recovery_event["body"].get("body") or {})
                    != expected_recovery_body
                ):
                    raise MigrationRequired("operator task-recovery event differs")
                completion_rows = [
                    (str(row[0]), str(row[1]))
                    for row in connection.execute(
                        "SELECT receipt_cid, task_cid FROM completion_receipts "
                        "ORDER BY receipt_cid"
                    ).fetchall()
                ]
                if completion_rows != [
                    (
                        migration["prior_completion_receipt_cid"],
                        migration["prior_task_cids"]["SAWM-000"],
                    )
                ]:
                    raise MigrationRequired(
                        "operational recovery changed completion evidence"
                    )
                migration_details = {
                    "migration_digest": migration_digest,
                    "migration_evidence_id": migration_evidence_id,
                    "plan_migration_event_id": plan_event["event_id"],
                    "migration_evidence_event_id": evidence_event["event_id"],
                    "migration_event_watermark": int(
                        migration["prior_event_watermark"]
                    )
                    + _M6_MIGRATION_EVENT_OFFSET,
                    "operational_validation_receipts": operational_receipts,
                    "operational_validation_event_ids": operational_event_ids,
                    "operational_validation_first_event_watermark": int(
                        migration["prior_event_watermark"]
                    )
                    + _M6_OPERATIONAL_FIRST_OFFSET,
                    "operational_validation_last_event_watermark": int(
                        migration["prior_event_watermark"]
                    )
                    + _M6_OPERATIONAL_LAST_OFFSET,
                    "task_recovery_receipt": recovery_receipt,
                    "task_recovery_event_id": recovery_event["event_id"],
                    "target_event_watermark": expected_target_watermark,
                    "accepted_definition_changes": 0,
                    "accepted_completion_changes": 0,
                    "operational_validation_revision_changes": (
                        _M6_OPERATIONAL_TASK_COUNT
                    ),
                    "nonterminal_task_status_recovery_changes": 1,
                }

        # Replay in a short private directory, never beside either accepted
        # authority. Rebuild itself is the landed deterministic verifier.
        with tempfile.TemporaryDirectory(
            prefix="sawm-r2-replay-", dir="/tmp"
        ) as temp_dir:
            replay_copy = Path(temp_dir) / "control.duckdb"
            shutil.copyfile(path, replay_copy)
            replay = DatabaseTaskSource(
                replay_copy,
                install_schema=False,
                repository_tree_id=str(population["repository_tree_id"]),
                plan_root_cid=str(population["plan_root_cid"]),
            )
            try:
                projection_matches = replay.projection_matches_events()
            finally:
                replay.close()
        if not projection_matches:
            raise MigrationRequired(
                "event replay projection differs from the accepted projection"
            )
        if (
            require_migration
            and str(snap.projection_cid) != _M6_EXPECTED_PROJECTION_CID
        ):
            raise MigrationRequired("M6 deterministic projection identity differs")
        return {
            "valid": True,
            "task_count": snap.task_count,
            "goal_count": snap.goal_count,
            "projection_cid": snap.projection_cid,
            "event_watermark": snap.event_cursor,
            "projection_matches_events": True,
            "statuses": status_by_alias,
            "revisions": revision_by_alias,
            **migration_details,
        }
    finally:
        source.close()


def _validator_report(root: Path, script: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(root / script), "--check-all", "--repo-root", str(root)],
        cwd=root, check=False, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "ipfs_datasets_py"), str(root / "ipfs_kit_py"), str(root)))},
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MaterializationError(f"{script} did not emit deterministic JSON: {exc}; stderr={result.stderr[-1000:]}") from exc
    if result.returncode or report.get("valid") is not True:
        raise MaterializationError(f"{script} failed current-tree validation: {report.get('errors')}")
    return report


def _task_recovery_receipt(population: Mapping[str, Any]) -> dict[str, Any]:
    migration = population["migration_inventory"]
    failure = migration["preprovider_task_failure"]
    task = next(
        item for item in population["taskboard"] if item["task_id"] == "SAWM-001"
    )
    prior_validations = tuple(
        (str(command),) for command in task["validation_commands"]
    )
    operational_validations, replacement_count = _operational_validation_commands(
        prior_validations,
        task_alias="SAWM-001",
    )
    operational_receipt = _operational_validation_receipt(
        population,
        task_alias="SAWM-001",
        task_cid=str(failure["task_cid"]),
        expected_status="blocked",
        expected_revision=5,
        prior_validations=prior_validations,
        operational_validations=operational_validations,
        replacement_count=replacement_count,
    )
    return {
        "schema": "sawm/operator-settled-preprovider-task-recovery@1",
        "operation": "operator_settled_preprovider_task_requeue",
        "authority_class": "operator_control_plane",
        "migration_revision": migration["migration_revision"],
        "task_cid": failure["task_cid"],
        "task_alias": failure["task_alias"],
        "task_definition_cid": migration["prior_task_cids"]["SAWM-001"],
        "expected_status": "blocked",
        "expected_revision": 6,
        "status": "todo",
        "target_revision": 7,
        "frozen_status": failure["canonical_task_status"],
        "frozen_revision": int(failure["canonical_task_revision"]),
        "prior_event_id": failure["canonical_settlement_event_id"],
        "prior_event_watermark": int(failure["canonical_event_watermark"]),
        "prior_projection_cid": failure["canonical_projection_cid"],
        "preprovider_failure_cid": _identity(failure),
        "failure_kind": failure["failure_kind"],
        "failure_reason": failure["failure_reason"],
        "settlement_id": failure["settlement_id"],
        "settlement_failure_kind": failure["settlement_failure_kind"],
        "failure_payload_digest": failure["failure_payload_digest"],
        "attempt_id": failure["attempt_id"],
        "claim_id": failure["claim_id"],
        "lease_id": failure["lease_id"],
        "owner_session_id": failure["owner_session_id"],
        "fencing_token": int(failure["fencing_token"]),
        "fence_epoch": int(failure["fence_epoch"]),
        "operational_validation_revision_receipt_cid": operational_receipt[
            "receipt_cid"
        ],
        "settlement_verified": True,
        "historical_execution_sidecar_copied": False,
        "historical_coordination_sidecar_copied": False,
        "automatic_retry_admitted": False,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "nonterminal_task_status_recovery_changes": 1,
        "implementation_provider_invoked": False,
        "effect_claim_recorded": False,
        "implementation_commit_created": False,
        "merge_attempted": False,
        "task_completed": False,
        "worker_self_approval": False,
    }


def _migration_body(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    migration = population["migration_inventory"]
    operational_receipts: dict[str, str] = {}
    for task in population["taskboard"]:
        alias = str(task["task_id"])
        if alias not in _operational_task_aliases():
            continue
        prior_status = "blocked" if alias == "SAWM-001" else "todo"
        prior_revision = 5 if alias == "SAWM-001" else 1
        prior_validations = tuple(
            (str(command),) for command in task["validation_commands"]
        )
        operational_validations, replacement_count = (
            _operational_validation_commands(
                prior_validations,
                task_alias=alias,
            )
        )
        operational_receipts[alias] = _operational_validation_receipt(
            population,
            task_alias=alias,
            task_cid=str(task["task_cid"]),
            expected_status=prior_status,
            expected_revision=prior_revision,
            prior_validations=prior_validations,
            operational_validations=operational_validations,
            replacement_count=replacement_count,
        )["receipt_cid"]
    return {
        "schema": "sawm/operator-control-plane-source-migration@3",
        "migration_revision": migration["migration_revision"],
        "board_namespace": NAMESPACE,
        "plan_revision": REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "operator_task_cid": migration["prior_task_cids"]["SAWM-000"],
        "prior_store_id": migration["prior_store_id"],
        "target_store_id": migration["target_store_id"],
        "prior_control_store_sha256": migration["prior_control_store_sha256"],
        "prior_event_prefix_sha256": migration["prior_event_prefix_sha256"],
        "prior_event_watermark": migration["prior_event_watermark"],
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "prior_plan_revision": migration["prior_plan_revision"],
        "target_plan_revision": migration["target_plan_revision"],
        "migration_history_receipt_cids": [
            entry["migration_receipt_cid"]
            for entry in migration["migration_history"]
        ],
        "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
        "prior_head": migration["prior_source_head"],
        "prior_tree": migration["prior_source_tree"],
        "current_head": population["source_binding"]["head"],
        "current_tree": population["source_binding"]["tree"],
        "bounded_control_plane_repair_paths": migration["bounded_control_plane_repair_paths"],
        "bounded_control_plane_repair_sha256": {
            path: population["source_binding"]["control_sha256"][path]
            for path in migration["bounded_control_plane_repair_paths"]
        },
        "quack_extension_pin": dict(config["quack_owner"]["pinned_extension"]),
        "preworker_launch_failure": dict(migration["preworker_launch_failure"]),
        "preprovider_task_failure": dict(migration["preprovider_task_failure"]),
        "validation_runtime": dict(config["validation_runtime"]),
        "operational_validation_requeue": dict(
            config["prior_materialization"]["operational_validation_requeue"]
        ),
        "operational_validation_revision_receipt_cids": operational_receipts,
        "nonterminal_task_recovery_receipt": _task_recovery_receipt(population),
        "validator_digest": validation_digest,
        "supersession_reason": migration["supersession_reason"],
        "supersession_mode": _M6_SUPERSESSION_MODE,
        "prior_authority_preserved": True,
        "accepted_task_definitions_rewritten": False,
        "accepted_goal_definitions_rewritten": False,
        "accepted_completion_changes": 0,
        "operational_validation_revision_changes": _M6_OPERATIONAL_TASK_COUNT,
        "nonterminal_task_status_recovery_changes": 1,
        "operator_completion_replayed": False,
        "worker_self_approval": False,
    }


def _migration_plan_delta(population: Mapping[str, Any]) -> dict[str, Any]:
    migration = population["migration_inventory"]
    return {
        "kind": str(migration["migration_kind"]),
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "prior_migration_receipt_cids": [
            entry["migration_receipt_cid"]
            for entry in migration["migration_history"]
        ],
        "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "operational_validation_revision_changes": _M6_OPERATIONAL_TASK_COUNT,
        "operational_validation_task_identity_manifest_cid": (
            _M6_TASK_IDENTITY_MANIFEST_CID
        ),
        "operational_validation_token_from": _M6_VALIDATION_TOKEN_FROM,
        "operational_validation_token_to": _M6_VALIDATION_TOKEN_TO,
        "nonterminal_task_status_recovery_changes": 1,
        "recovered_task_cid": migration["prior_task_cids"]["SAWM-001"],
        "recovered_task_frozen_status": "blocked",
        "recovered_task_frozen_revision": 5,
        "recovered_task_expected_status": "blocked",
        "recovered_task_expected_revision": 6,
        "recovered_task_status": "todo",
        "recovered_task_revision": 7,
    }


def _projection_at_watermark(
    path: Path,
    population: Mapping[str, Any],
    watermark: int,
) -> str:
    """Rebuild a bounded historical projection on a disposable copy."""

    with tempfile.TemporaryDirectory(prefix="sawm-r2-prefix-", dir="/tmp") as temp_dir:
        copy = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(path, copy)
        import duckdb
        connection = duckdb.connect(str(copy))
        try:
            connection.execute(
                "DELETE FROM domain_events WHERE global_sequence > ?",
                [int(watermark)],
            )
        finally:
            connection.close()
        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
            DatabaseTaskSource,
        )
        source = DatabaseTaskSource(
            copy,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        try:
            snapshot = source.rebuild_from_events()
            if int(snapshot.event_cursor) != int(watermark):
                raise MigrationRequired("historical migration projection watermark differs")
            return str(snapshot.projection_cid)
        finally:
            source.close()


def _expected_migration_receipt(
    root: Path,
    target: Path,
    population: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    migration = population["migration_inventory"]
    migration_watermark = (
        int(migration["prior_event_watermark"]) + _M6_MIGRATION_EVENT_OFFSET
    )
    operational_first_watermark = (
        int(migration["prior_event_watermark"]) + _M6_OPERATIONAL_FIRST_OFFSET
    )
    operational_last_watermark = (
        int(migration["prior_event_watermark"]) + _M6_OPERATIONAL_LAST_OFFSET
    )
    target_watermark = (
        int(migration["prior_event_watermark"]) + _M6_EVENT_SUFFIX_LENGTH
    )
    recovery_receipt = _task_recovery_receipt(population)
    receipt = {
        "schema": "sawm/non-authoritative-migration-receipt@3",
        "authoritative": False,
        "database_is_authority": True,
        "migration_revision": migration["migration_revision"],
        "program_definition_cid": population["program_definition_cid"],
        "migration_projection_cid": _projection_at_watermark(
            target, population, migration_watermark
        ),
        "migration_event_watermark": migration_watermark,
        "operational_validation_projection_cid": _projection_at_watermark(
            target, population, operational_last_watermark
        ),
        "operational_validation_first_event_watermark": (
            operational_first_watermark
        ),
        "operational_validation_last_event_watermark": operational_last_watermark,
        "operational_validation_revision_changes": _M6_OPERATIONAL_TASK_COUNT,
        "operational_validation_receipt_cids": {
            alias: receipt["receipt_cid"]
            for alias, receipt in sorted(
                verified["operational_validation_receipts"].items()
            )
        },
        "operational_validation_event_ids": dict(
            sorted(verified["operational_validation_event_ids"].items())
        ),
        "projection_cid": verified["projection_cid"],
        "target_event_watermark": target_watermark,
        "validation_digest": validation_digest,
        "migration_digest": verified["migration_digest"],
        "migration_evidence_id": verified["migration_evidence_id"],
        "plan_migration_event_id": verified["plan_migration_event_id"],
        "migration_evidence_event_id": verified["migration_evidence_event_id"],
        "task_recovery_event_id": verified["task_recovery_event_id"],
        "task_recovery_receipt_cid": _identity(recovery_receipt),
        "task_recovery_receipt": recovery_receipt,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "nonterminal_task_status_recovery_changes": 1,
        "prior_control_store_sha256": migration["prior_control_store_sha256"],
        "prior_event_prefix_sha256": migration["prior_event_prefix_sha256"],
        "prior_source_binding_cid": migration["prior_source_binding_cid"],
        "prior_migration_receipt_cids": [
            entry["migration_receipt_cid"]
            for entry in migration["migration_history"]
        ],
        "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
        "preprovider_failure_cid": _identity(migration["preprovider_task_failure"]),
        "prior_database_path": migration["prior_store_id"],
        "database_path": str(target.relative_to(root)),
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def _ensure_migration_receipt(
    root: Path,
    target: Path,
    population: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    expected = _expected_migration_receipt(
        root, target, population, verified, validation_digest
    )
    receipt_path = target.parent / "migration-receipt.json"
    if receipt_path.exists():
        observed = _load_json(receipt_path)
        if observed != expected:
            raise MigrationRequired("external migration receipt differs from authority")
        claimed = str(observed.get("receipt_cid") or "")
        unhashed = dict(observed)
        unhashed.pop("receipt_cid", None)
        if claimed != _identity(unhashed):
            raise MigrationRequired("external migration receipt CID does not rehash")
        return observed

    # A persistent advisory lock survives as an inert inode, while its kernel
    # ownership never survives a crashed process.  This lets a later verifier
    # reconstruct a missing non-authoritative receipt without deleting a lock
    # that might belong to a concurrent live materializer.
    lock = receipt_path.with_name(f".{receipt_path.name}.publish.lock")
    fd = os.open(
        lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MaterializationError("migration receipt lock is not a regular file")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise MaterializationError(
                        "timed out acquiring the migration receipt publication lock"
                    ) from exc
                time.sleep(0.02)
        if receipt_path.exists():
            observed = _load_json(receipt_path)
            if observed != expected:
                raise MigrationRequired("concurrent migration receipt differs")
            return observed
        temporary = receipt_path.with_name(
            f".{receipt_path.name}.{os.getpid()}.tmp"
        )
        out = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            payload = _canonical(expected) + b"\n"
            view = memoryview(payload)
            while view:
                written = os.write(out, view)
                view = view[written:]
            os.fsync(out)
        finally:
            os.close(out)
        os.replace(temporary, receipt_path)
        directory = os.open(
            receipt_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return expected
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _expected_m7_migration_receipt(
    root: Path,
    target: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    authority = _m7_source_repair_authority(population, config)
    _assert_m7_source_delta(root, population, authority)
    receipt = {
        "schema": "sawm/non-authoritative-migration-receipt@5",
        "authoritative": False,
        "database_is_authority": True,
        "migration_revision": _M7_MIGRATION_REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_projection_cid": _projection_at_watermark(
            target,
            population,
            _M7_TARGET_EVENT_WATERMARK - 1,
        ),
        "migration_projection_cid": verified["projection_cid"],
        "migration_event_watermark": _M7_TARGET_EVENT_WATERMARK,
        "projection_cid": verified["projection_cid"],
        "target_event_watermark": _M7_TARGET_EVENT_WATERMARK,
        "validation_digest": validation_digest,
        "migration_digest": verified["migration_digest"],
        "migration_evidence_id": verified["migration_evidence_id"],
        "plan_migration_event_id": verified["plan_migration_event_id"],
        "migration_evidence_event_id": verified[
            "migration_evidence_event_id"
        ],
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "prior_control_store_sha256": authority[
            "prior_control_store_sha256"
        ],
        "prior_event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "prior_semantic_authority_digest": authority[
            "prior_semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": authority[
            "prior_frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": authority[
            "prior_append_surface_digest"
        ],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "live_preflight_failure_cid": authority["live_preflight_failure_cid"],
        "prepublication_materialization_failure_cid": _identity(
            _M7_PREPUBLICATION_MATERIALIZATION_FAILURE
        ),
        "prior_database_path": authority["prior_store_id"],
        "database_path": str(target.relative_to(root)),
        "worker_self_approval": False,
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def _ensure_m7_migration_receipt(
    root: Path,
    target: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    expected = _expected_m7_migration_receipt(
        root,
        target,
        population,
        config,
        verified,
        validation_digest,
    )
    receipt_path = target.parent / "migration-receipt.json"
    if receipt_path.exists():
        observed = _load_json(receipt_path)
        if observed != expected:
            raise MigrationRequired("external M7 migration receipt differs")
        unhashed = dict(observed)
        claimed = str(unhashed.pop("receipt_cid", ""))
        if claimed != _identity(unhashed):
            raise MigrationRequired("external M7 receipt CID does not rehash")
        return observed
    lock = receipt_path.with_name(f".{receipt_path.name}.publish.lock")
    fd = os.open(
        lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MaterializationError("M7 receipt lock is not a regular file")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise MaterializationError(
                        "timed out acquiring the M7 receipt publication lock"
                    ) from exc
                time.sleep(0.02)
        if receipt_path.exists():
            observed = _load_json(receipt_path)
            if observed != expected:
                raise MigrationRequired("concurrent M7 migration receipt differs")
            return observed
        temporary = receipt_path.with_name(
            f".{receipt_path.name}.{os.getpid()}.tmp"
        )
        out = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            payload = _canonical(expected) + b"\n"
            view = memoryview(payload)
            while view:
                view = view[os.write(out, view) :]
            os.fsync(out)
        finally:
            os.close(out)
        os.replace(temporary, receipt_path)
        directory = os.open(
            receipt_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return expected
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _expected_m8_migration_receipt(
    root: Path,
    target: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    authority = _m8_source_repair_authority(population, config)
    _assert_m8_source_delta(root, population, authority)
    prior_repair = population["migration_inventory"].get(
        "source_repair_materialization"
    )
    if not isinstance(prior_repair, Mapping):
        raise MaterializationError("M8 prior source-repair authority is missing")
    receipt = {
        "schema": "sawm/non-authoritative-migration-receipt@6",
        "authoritative": False,
        "database_is_authority": True,
        "migration_revision": _M8_MIGRATION_REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_projection_cid": _projection_at_watermark(
            target,
            population,
            _M8_TARGET_EVENT_WATERMARK - 1,
        ),
        "migration_projection_cid": verified["projection_cid"],
        "migration_event_watermark": _M8_TARGET_EVENT_WATERMARK,
        "projection_cid": verified["projection_cid"],
        "target_event_watermark": _M8_TARGET_EVENT_WATERMARK,
        "validation_digest": validation_digest,
        "migration_digest": verified["migration_digest"],
        "migration_evidence_id": verified["migration_evidence_id"],
        "plan_migration_event_id": verified["plan_migration_event_id"],
        "migration_evidence_event_id": verified[
            "migration_evidence_event_id"
        ],
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "prior_control_store_sha256": authority[
            "prior_control_store_sha256"
        ],
        "prior_event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "prior_source_repair_materialization_cid": _identity(prior_repair),
        "source_repair_successor_materialization_cid": (
            _M8_SOURCE_REPAIR_AUTHORITY_CID
        ),
        "prior_semantic_authority_digest": authority[
            "prior_semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": authority[
            "prior_frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": authority[
            "prior_append_surface_digest"
        ],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "live_preflight_failure_cid": authority["live_preflight_failure_cid"],
        "prior_database_path": authority["prior_store_id"],
        "database_path": str(target.relative_to(root)),
        "worker_self_approval": False,
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def _ensure_m8_migration_receipt(
    root: Path,
    target: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    expected = _expected_m8_migration_receipt(
        root,
        target,
        population,
        config,
        verified,
        validation_digest,
    )
    receipt_path = target.parent / "migration-receipt.json"
    if os.path.lexists(receipt_path):
        observed, _ = _load_nofollow_json(
            receipt_path,
            root=root,
            noun="external M8 migration receipt",
        )
        if observed != expected:
            raise MigrationRequired("external M8 migration receipt differs")
        unhashed = dict(observed)
        claimed = str(unhashed.pop("receipt_cid", ""))
        if claimed != _identity(unhashed):
            raise MigrationRequired("external M8 receipt CID does not rehash")
        return observed
    lock = receipt_path.with_name(f".{receipt_path.name}.publish.lock")
    fd = os.open(
        lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MaterializationError("M8 receipt lock is not a regular file")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise MaterializationError(
                        "timed out acquiring the M8 receipt publication lock"
                    ) from exc
                time.sleep(0.02)
        if os.path.lexists(receipt_path):
            observed, _ = _load_nofollow_json(
                receipt_path,
                root=root,
                noun="concurrent M8 migration receipt",
            )
            if observed != expected:
                raise MigrationRequired("concurrent M8 migration receipt differs")
            return observed
        temporary = receipt_path.with_name(
            f".{receipt_path.name}.{os.getpid()}.tmp"
        )
        out = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            payload = _canonical(expected) + b"\n"
            view = memoryview(payload)
            while view:
                view = view[os.write(out, view) :]
            os.fsync(out)
        finally:
            os.close(out)
        os.replace(temporary, receipt_path)
        directory = os.open(
            receipt_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return expected
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _expected_m9_migration_receipt(
    root: Path,
    control: Path,
    coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Bind the independently verified M9 control/coordinator pair."""

    authority = _m9_live_recovery_authority(population, config)
    _assert_m9_source_delta(root, population, authority)
    if (
        control.parent != coordination.parent
        or not control.is_relative_to(root)
        or not coordination.is_relative_to(root)
    ):
        raise MaterializationError("M9 receipt paths are not a confined pair")
    control_sha256, control_size = _stable_regular_sha256(
        control,
        root=root,
        noun="published M9 control store",
        required_link_count=1,
    )
    coordination_sha256, coordination_size = _stable_regular_sha256(
        coordination,
        root=root,
        noun="published M9 coordination store",
        required_link_count=1,
    )
    if (
        control_sha256 != verified.get("control_store_sha256")
        or coordination_sha256 != verified.get("coordination_store_sha256")
        or verified.get("projection_cid") != _M9_EXPECTED_PROJECTION_CID
        or verified.get("coordination_projection_digest")
        != _M9_COORDINATION_PROJECTION_DIGEST
        or int(verified.get("event_watermark") or 0)
        != _M9_TARGET_EVENT_WATERMARK
        or int(verified.get("coordination_event_count") or 0) != 36
    ):
        raise MigrationRequired("M9 verified pair changed before receipt publication")
    receipt = {
        "schema": "sawm/non-authoritative-migration-receipt@7",
        "authoritative": False,
        "control_database_is_authority": True,
        "coordination_database_is_authority": True,
        "receipt_is_final_pair_commit_marker": True,
        "migration_revision": _M9_MIGRATION_REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_projection_cid": _projection_at_watermark(
            control,
            population,
            int(authority["prior_event_watermark"]),
        ),
        "migration_projection_cid": verified["projection_cid"],
        "coordination_projection_digest": verified[
            "coordination_projection_digest"
        ],
        "migration_event_watermark": _M9_TARGET_EVENT_WATERMARK,
        "coordination_event_count": verified["coordination_event_count"],
        "validation_digest": validation_digest,
        "migration_digest": verified["migration_digest"],
        "migration_evidence_id": verified["migration_evidence_id"],
        "plan_migration_event_id": verified["plan_migration_event_id"],
        "migration_evidence_event_id": verified[
            "migration_evidence_event_id"
        ],
        "task_rearm_event_id": verified["task_rearm_event_id"],
        "task_rearm_receipt_cid": verified["task_rearm_receipt_cid"],
        "coordination_rearm_event_id": verified[
            "coordination_rearm_event_id"
        ],
        "coordination_rearm_id": verified["coordination_rearm_id"],
        "task_revision_changes": 1,
        "task_status_changes": 1,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "prior_control_store_sha256": authority[
            "prior_control_store_sha256"
        ],
        "prior_coordination_store_sha256": authority[
            "prior_coordination_store_sha256"
        ],
        "prior_event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "source_repair_successor_materialization_cid": (
            _M8_SOURCE_REPAIR_AUTHORITY_CID
        ),
        "live_recovery_successor_materialization_cid": (
            _M9_LIVE_RECOVERY_AUTHORITY_CID
        ),
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "live_implementation_failure_cid": _identity(
            authority["live_implementation_failure"]
        ),
        "failure_settlement_id": _M8_LIVE_FAILURE_RECEIPT[
            "settlement_id"
        ],
        "control_store_sha256": control_sha256,
        "control_store_size": control_size,
        "coordination_store_sha256": coordination_sha256,
        "coordination_store_size": coordination_size,
        "semantic_authority_digest": verified["semantic_authority_digest"],
        "frozen_base_authority_digest": verified[
            "frozen_base_authority_digest"
        ],
        "append_surface_digest": verified["append_surface_digest"],
        "prior_database_path": authority["prior_store_id"],
        "prior_coordination_path": authority["prior_coordination_store_id"],
        "database_path": str(control.relative_to(root)),
        "coordination_path": str(coordination.relative_to(root)),
        "coordination_sidecar_copied_before_rearm": True,
        "execution_sidecar_copied": False,
        "failed_attempt_history_preserved": True,
        "failed_completion_barrier_rearmed": True,
        "worker_self_approval": False,
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def _assert_m9_receipt_commit_inputs(
    root: Path,
    control: Path,
    coordination: Path,
    expected: Mapping[str, Any],
) -> None:
    """Rebind the exact offline pair at a receipt commit point."""

    execution = control.with_name("control.execution.duckdb")
    if os.path.lexists(execution):
        raise MigrationRequired("M9 execution sidecar exists at receipt commit")

    def exact_pair() -> tuple[tuple[str, int], tuple[str, int]]:
        return (
            _stable_regular_sha256(
                control,
                root=root,
                noun="M9 control store at receipt commit",
                required_link_count=1,
            ),
            _stable_regular_sha256(
                coordination,
                root=root,
                noun="M9 coordination store at receipt commit",
                required_link_count=1,
            ),
        )

    before = exact_pair()
    _assert_offline(control)
    if os.path.lexists(execution):
        raise MigrationRequired("M9 execution sidecar exists at receipt commit")
    after = exact_pair()
    expected_pair = (
        (
            str(expected.get("control_store_sha256") or ""),
            int(expected.get("control_store_size") or 0),
        ),
        (
            str(expected.get("coordination_store_sha256") or ""),
            int(expected.get("coordination_store_size") or 0),
        ),
    )
    if before != after or after != expected_pair:
        raise MigrationRequired("M9 store pair changed at receipt commit")


def _ensure_m9_migration_receipt(
    root: Path,
    control: Path,
    coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
    *,
    pair_lock_held: bool = False,
) -> dict[str, Any]:
    """Publish the final M9 pair marker without following or replacing links."""

    if not pair_lock_held:
        pair_lock = control.parent / ".m9-store-pair.publish.lock"
        pair_descriptor = os.open(
            pair_lock,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            pair_stat = os.fstat(pair_descriptor)
            if not stat.S_ISREG(pair_stat.st_mode) or pair_stat.st_nlink != 1:
                raise MaterializationError("M9 bundle lock is not a regular file")
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    fcntl.flock(
                        pair_descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise MaterializationError(
                            "timed out acquiring the M9 pair publication lock"
                        ) from exc
                    time.sleep(0.02)
            return _ensure_m9_migration_receipt(
                root,
                control,
                coordination,
                population,
                config,
                verified,
                validation_digest,
                pair_lock_held=True,
            )
        finally:
            try:
                fcntl.flock(pair_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(pair_descriptor)

    authority = _m9_live_recovery_authority(population, config)
    if os.path.lexists(control.with_name("control.execution.duckdb")):
        raise MigrationRequired("M9 execution sidecar exists before pair admission")
    _assert_committed_clean_source(root, population)
    _assert_m9_source_delta(root, population, authority)
    prior_control, prior_coordination = _assert_m9_prior_publication_anchor(
        root,
        authority,
    )
    current_verified = _verify_m9_store_pair(
        root,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    if dict(current_verified) != dict(verified):
        raise MigrationRequired("M9 pair changed before receipt publication")
    expected = _expected_m9_migration_receipt(
        root,
        control,
        coordination,
        population,
        config,
        current_verified,
        validation_digest,
    )
    receipt_path = control.parent / "migration-receipt.json"
    lock = receipt_path.with_name(f".{receipt_path.name}.publish.lock")
    descriptor = os.open(
        lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MaterializationError("M9 receipt lock is not a regular file")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise MaterializationError(
                        "timed out acquiring the M9 receipt publication lock"
                    ) from exc
                time.sleep(0.02)
        directory = os.open(
            receipt_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            _assert_m9_receipt_commit_inputs(
                root,
                control,
                coordination,
                expected,
            )
            pending = sorted(
                receipt_path.parent.glob(f".{receipt_path.name}.*.tmp")
            )
            if os.path.lexists(receipt_path):
                receipt_stat = os.lstat(receipt_path)
                if not stat.S_ISREG(receipt_stat.st_mode):
                    raise MigrationRequired("M9 receipt leaf is not regular")
                if receipt_stat.st_nlink == 2:
                    aliases = []
                    for candidate in pending:
                        candidate_stat = os.lstat(candidate)
                        if (
                            stat.S_ISREG(candidate_stat.st_mode)
                            and candidate_stat.st_dev == receipt_stat.st_dev
                            and candidate_stat.st_ino == receipt_stat.st_ino
                        ):
                            aliases.append(candidate)
                    if len(aliases) != 1 or len(pending) != 1:
                        raise MigrationRequired(
                            "M9 receipt has an ambiguous pending hardlink"
                        )
                    observed, _ = _load_nofollow_json(
                        receipt_path,
                        root=root,
                        noun="pending M9 migration receipt",
                        required_link_count=2,
                    )
                    if observed != expected:
                        raise MigrationRequired("pending M9 receipt differs")
                    _assert_m9_receipt_commit_inputs(
                        root,
                        control,
                        coordination,
                        expected,
                    )
                    os.unlink(aliases[0])
                    os.fsync(directory)
                    pending = []
                elif receipt_stat.st_nlink != 1 or pending:
                    raise MigrationRequired(
                        "M9 receipt or pending alias population differs"
                    )
                observed, _ = _load_nofollow_json(
                    receipt_path,
                    root=root,
                    noun="external M9 migration receipt",
                )
                if observed != expected:
                    raise MigrationRequired("external M9 migration receipt differs")
                unhashed = dict(observed)
                claimed = str(unhashed.pop("receipt_cid", ""))
                if claimed != _identity(unhashed):
                    raise MigrationRequired("external M9 receipt CID does not rehash")
                _assert_m9_receipt_commit_inputs(
                    root,
                    control,
                    coordination,
                    expected,
                )
                return observed

            if len(pending) > 1:
                raise MigrationRequired("M9 has ambiguous pending receipt files")
            if pending:
                temporary = pending[0]
                observed, _ = _load_nofollow_json(
                    temporary,
                    root=root,
                    noun="pending M9 receipt temporary",
                )
                if observed != expected:
                    raise MigrationRequired("pending M9 receipt temporary differs")
            else:
                temporary = receipt_path.with_name(
                    f".{receipt_path.name}.{os.getpid()}.tmp"
                )
                out = os.open(
                    temporary,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    payload = _canonical(expected) + b"\n"
                    view = memoryview(payload)
                    while view:
                        view = view[os.write(out, view) :]
                    os.fsync(out)
                finally:
                    os.close(out)
            _assert_m9_receipt_commit_inputs(
                root,
                control,
                coordination,
                expected,
            )
            try:
                os.link(temporary, receipt_path, follow_symlinks=False)
            except FileExistsError as exc:
                raise MigrationRequired(
                    "M9 receipt target appeared during publication"
                ) from exc
            linked, _ = _load_nofollow_json(
                receipt_path,
                root=root,
                noun="new M9 migration receipt",
                required_link_count=2,
            )
            if linked != expected:
                raise MigrationRequired("new M9 receipt differs before commit")
            linked_stat = os.lstat(receipt_path)
            try:
                _assert_m9_receipt_commit_inputs(
                    root,
                    control,
                    coordination,
                    expected,
                )
            except Exception:
                try:
                    current_stat = os.lstat(receipt_path)
                except FileNotFoundError:
                    current_stat = None
                if (
                    current_stat is not None
                    and current_stat.st_dev == linked_stat.st_dev
                    and current_stat.st_ino == linked_stat.st_ino
                ):
                    os.unlink(receipt_path)
                    os.fsync(directory)
                raise
            os.unlink(temporary)
            os.fsync(directory)
            observed, _ = _load_nofollow_json(
                receipt_path,
                root=root,
                noun="committed M9 migration receipt",
            )
            if observed != expected:
                raise MigrationRequired("committed M9 receipt differs")
            try:
                _assert_m9_receipt_commit_inputs(
                    root,
                    control,
                    coordination,
                    expected,
                )
            except Exception:
                try:
                    current_stat = os.lstat(receipt_path)
                except FileNotFoundError:
                    current_stat = None
                if (
                    current_stat is not None
                    and current_stat.st_dev == linked_stat.st_dev
                    and current_stat.st_ino == linked_stat.st_ino
                ):
                    os.unlink(receipt_path)
                    os.fsync(directory)
                raise
        finally:
            os.close(directory)
        return observed
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _expected_m10_migration_receipt(
    root: Path,
    control: Path,
    coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Bind the independently verified source-only M10 store pair."""

    authority = _m10_live_projection_authority(population, config)
    _assert_m10_source_delta(root, population, authority)
    control_sha256, control_size = _stable_regular_sha256(
        control,
        root=root,
        noun="published M10 control store",
        required_link_count=1,
    )
    coordination_sha256, coordination_size = _stable_regular_sha256(
        coordination,
        root=root,
        noun="published M10 coordination store",
        required_link_count=1,
    )
    if (
        control_sha256 != verified.get("control_store_sha256")
        or coordination_sha256 != verified.get("coordination_store_sha256")
        or verified.get("projection_cid") != _M10_EXPECTED_PROJECTION_CID
        or int(verified.get("event_watermark") or 0)
        != _M10_TARGET_EVENT_WATERMARK
        or verified.get("semantic_authority_digest")
        != authority["target_semantic_authority_digest"]
        or verified.get("coordination_projection_digest")
        != authority["target_coordination_projection_digest"]
        or int(verified.get("coordination_event_count") or 0)
        != int(authority["target_coordination_event_count"])
        or int(verified.get("task_revision_changes") or 0) != 0
        or int(verified.get("task_status_changes") or 0) != 0
    ):
        raise MigrationRequired("M10 verified pair changed before receipt publication")
    receipt = {
        "schema": "sawm/non-authoritative-migration-receipt@8",
        "authoritative": False,
        "control_database_is_authority": True,
        "coordination_database_is_authority": True,
        "receipt_is_final_pair_commit_marker": True,
        "migration_revision": _M10_MIGRATION_REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_projection_cid": _projection_at_watermark(
            control,
            population,
            int(authority["prior_event_watermark"]),
        ),
        "migration_projection_cid": verified["projection_cid"],
        "migration_event_watermark": _M10_TARGET_EVENT_WATERMARK,
        "coordination_projection_digest": verified[
            "coordination_projection_digest"
        ],
        "coordination_event_count": verified["coordination_event_count"],
        "validation_digest": validation_digest,
        "migration_digest": verified["migration_digest"],
        "migration_evidence_id": verified["migration_evidence_id"],
        "plan_migration_event_id": verified["plan_migration_event_id"],
        "migration_evidence_event_id": verified[
            "migration_evidence_event_id"
        ],
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "coordination_semantic_changes": 0,
        "prior_publication_control_store_sha256": authority[
            "prior_publication_control_store_sha256"
        ],
        "prior_control_store_sha256": authority["prior_control_store_sha256"],
        "prior_coordination_store_sha256": authority[
            "prior_coordination_store_sha256"
        ],
        "prior_event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "live_recovery_successor_materialization_cid": (
            _M9_LIVE_RECOVERY_AUTHORITY_CID
        ),
        "live_projection_successor_materialization_cid": (
            _M10_LIVE_PROJECTION_AUTHORITY_CID
        ),
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "live_preflight_failure_cid": authority["live_preflight_failure_cid"],
        "control_store_sha256": control_sha256,
        "control_store_size": control_size,
        "coordination_store_sha256": coordination_sha256,
        "coordination_store_size": coordination_size,
        "semantic_authority_digest": verified["semantic_authority_digest"],
        "frozen_base_authority_digest": verified[
            "frozen_base_authority_digest"
        ],
        "append_surface_digest": verified["append_surface_digest"],
        "prior_database_path": authority["prior_store_id"],
        "prior_coordination_path": authority["prior_coordination_store_id"],
        "database_path": str(control.relative_to(root)),
        "coordination_path": str(coordination.relative_to(root)),
        "coordination_sidecar_copied_unchanged": True,
        "execution_sidecar_copied": False,
        "worker_self_approval": False,
    }
    return {**receipt, "receipt_cid": _identity(receipt)}


def _assert_m10_receipt_commit_inputs(
    root: Path,
    control: Path,
    coordination: Path,
    expected: Mapping[str, Any],
) -> None:
    """Rebind the exact offline M10 pair at a receipt commit point."""

    execution = control.with_name("control.execution.duckdb")
    if os.path.lexists(execution):
        raise MigrationRequired("M10 execution sidecar exists at receipt commit")

    def exact_pair() -> tuple[tuple[str, int], tuple[str, int]]:
        return (
            _stable_regular_sha256(
                control,
                root=root,
                noun="M10 control store at receipt commit",
                required_link_count=1,
            ),
            _stable_regular_sha256(
                coordination,
                root=root,
                noun="M10 coordination store at receipt commit",
                required_link_count=1,
            ),
        )

    before = exact_pair()
    _assert_offline(control)
    if os.path.lexists(execution):
        raise MigrationRequired("M10 execution sidecar exists at receipt commit")
    after = exact_pair()
    expected_pair = (
        (
            str(expected.get("control_store_sha256") or ""),
            int(expected.get("control_store_size") or 0),
        ),
        (
            str(expected.get("coordination_store_sha256") or ""),
            int(expected.get("coordination_store_size") or 0),
        ),
    )
    if before != after or after != expected_pair:
        raise MigrationRequired("M10 store pair changed at receipt commit")


def _ensure_m10_migration_receipt(
    root: Path,
    control: Path,
    coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
    *,
    pair_lock_held: bool = False,
) -> dict[str, Any]:
    """Publish the M10 final marker last, with crash-state recovery."""

    if not pair_lock_held:
        pair_lock = control.parent / ".m10-store-pair.publish.lock"
        pair_descriptor = os.open(
            pair_lock,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            lock_stat = os.fstat(pair_descriptor)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
                raise MaterializationError("M10 bundle lock is not a regular file")
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    fcntl.flock(
                        pair_descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise MaterializationError(
                            "timed out acquiring the M10 pair publication lock"
                        ) from exc
                    time.sleep(0.02)
            return _ensure_m10_migration_receipt(
                root,
                control,
                coordination,
                population,
                config,
                verified,
                validation_digest,
                pair_lock_held=True,
            )
        finally:
            try:
                fcntl.flock(pair_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(pair_descriptor)

    authority = _m10_live_projection_authority(population, config)
    _assert_committed_clean_source(root, population)
    _assert_m10_source_delta(root, population, authority)
    prior_control, prior_coordination = _assert_m10_prior_publication_anchor(
        root,
        authority,
    )
    current_verified = _verify_m10_store_pair(
        root,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    if dict(current_verified) != dict(verified):
        raise MigrationRequired("M10 pair changed before receipt publication")
    expected = _expected_m10_migration_receipt(
        root,
        control,
        coordination,
        population,
        config,
        current_verified,
        validation_digest,
    )
    receipt_path = control.parent / "migration-receipt.json"
    lock = receipt_path.with_name(f".{receipt_path.name}.publish.lock")
    descriptor = os.open(
        lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MaterializationError("M10 receipt lock is not a regular file")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise MaterializationError(
                        "timed out acquiring the M10 receipt publication lock"
                    ) from exc
                time.sleep(0.02)
        directory = os.open(
            receipt_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            _assert_m10_receipt_commit_inputs(
                root,
                control,
                coordination,
                expected,
            )
            pending = sorted(
                receipt_path.parent.glob(f".{receipt_path.name}.*.tmp")
            )
            if os.path.lexists(receipt_path):
                receipt_stat = os.lstat(receipt_path)
                if not stat.S_ISREG(receipt_stat.st_mode):
                    raise MigrationRequired("M10 receipt leaf is not regular")
                if receipt_stat.st_nlink == 2:
                    aliases = []
                    for candidate in pending:
                        candidate_stat = os.lstat(candidate)
                        if (
                            stat.S_ISREG(candidate_stat.st_mode)
                            and candidate_stat.st_dev == receipt_stat.st_dev
                            and candidate_stat.st_ino == receipt_stat.st_ino
                        ):
                            aliases.append(candidate)
                    if len(aliases) != 1 or len(pending) != 1:
                        raise MigrationRequired(
                            "M10 receipt has an ambiguous pending hardlink"
                        )
                    observed, _ = _load_nofollow_json(
                        receipt_path,
                        root=root,
                        noun="pending M10 migration receipt",
                        required_link_count=2,
                    )
                    if observed != expected:
                        raise MigrationRequired("pending M10 receipt differs")
                    _assert_m10_receipt_commit_inputs(
                        root,
                        control,
                        coordination,
                        expected,
                    )
                    os.unlink(aliases[0])
                    os.fsync(directory)
                    pending = []
                elif receipt_stat.st_nlink != 1 or pending:
                    raise MigrationRequired(
                        "M10 receipt or pending alias population differs"
                    )
                observed, _ = _load_nofollow_json(
                    receipt_path,
                    root=root,
                    noun="external M10 migration receipt",
                )
                unhashed = dict(observed)
                claimed = str(unhashed.pop("receipt_cid", ""))
                if observed != expected or claimed != _identity(unhashed):
                    raise MigrationRequired("external M10 migration receipt differs")
                _assert_m10_receipt_commit_inputs(
                    root,
                    control,
                    coordination,
                    expected,
                )
                return observed

            if len(pending) > 1:
                raise MigrationRequired("M10 has ambiguous pending receipt files")
            if pending:
                temporary = pending[0]
                observed, _ = _load_nofollow_json(
                    temporary,
                    root=root,
                    noun="pending M10 receipt temporary",
                )
                if observed != expected:
                    raise MigrationRequired("pending M10 receipt temporary differs")
            else:
                temporary = receipt_path.with_name(
                    f".{receipt_path.name}.{os.getpid()}.tmp"
                )
                out = os.open(
                    temporary,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    payload = _canonical(expected) + b"\n"
                    view = memoryview(payload)
                    while view:
                        view = view[os.write(out, view) :]
                    os.fsync(out)
                finally:
                    os.close(out)
            _assert_m10_receipt_commit_inputs(
                root,
                control,
                coordination,
                expected,
            )
            try:
                os.link(temporary, receipt_path, follow_symlinks=False)
            except FileExistsError as exc:
                raise MigrationRequired(
                    "M10 receipt target appeared during publication"
                ) from exc
            linked, _ = _load_nofollow_json(
                receipt_path,
                root=root,
                noun="new M10 migration receipt",
                required_link_count=2,
            )
            if linked != expected:
                raise MigrationRequired("new M10 receipt differs before commit")
            linked_stat = os.lstat(receipt_path)
            try:
                _assert_m10_receipt_commit_inputs(
                    root,
                    control,
                    coordination,
                    expected,
                )
            except Exception:
                current_stat = os.lstat(receipt_path)
                if (
                    current_stat.st_dev == linked_stat.st_dev
                    and current_stat.st_ino == linked_stat.st_ino
                ):
                    os.unlink(receipt_path)
                    os.fsync(directory)
                raise
            os.unlink(temporary)
            os.fsync(directory)
            observed, _ = _load_nofollow_json(
                receipt_path,
                root=root,
                noun="committed M10 migration receipt",
            )
            if observed != expected:
                raise MigrationRequired("committed M10 receipt differs")
            try:
                _assert_m10_receipt_commit_inputs(
                    root,
                    control,
                    coordination,
                    expected,
                )
            except Exception:
                try:
                    current_stat = os.lstat(receipt_path)
                except FileNotFoundError:
                    current_stat = None
                if (
                    current_stat is not None
                    and current_stat.st_dev == linked_stat.st_dev
                    and current_stat.st_ino == linked_stat.st_ino
                ):
                    os.unlink(receipt_path)
                    os.fsync(directory)
                raise
            return observed
        finally:
            os.close(directory)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _verify_existing_m10_migration_receipt(
    root: Path,
    control: Path,
    coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    verified: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Verify M10's final marker without creating locks or changing files."""

    expected = _expected_m10_migration_receipt(
        root,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    receipt_path = control.parent / "migration-receipt.json"
    pending = sorted(receipt_path.parent.glob(f".{receipt_path.name}.*.tmp"))
    if pending:
        raise MigrationRequired("M10 has pending receipt publication state")
    _assert_m10_receipt_commit_inputs(root, control, coordination, expected)
    observed, _ = _load_nofollow_json(
        receipt_path,
        root=root,
        noun="committed M10 migration receipt",
    )
    unhashed = dict(observed)
    claimed = str(unhashed.pop("receipt_cid", ""))
    if observed != expected or claimed != _identity(unhashed):
        raise MigrationRequired("committed M10 migration receipt differs")
    _assert_m10_receipt_commit_inputs(root, control, coordination, expected)
    return observed


def _ducklake_projection(
    root: Path,
    config: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project advisory history idempotently and publish a durable side receipt."""

    policy = config.get("ducklake_history_projection") or {}
    receipt_relative = Path(str(policy.get("receipt_path")))
    if receipt_relative.is_absolute() or ".." in receipt_relative.parts:
        raise MaterializationError("DuckLake receipt path escapes the repository root")
    receipt_path = root / receipt_relative
    if not receipt_path.parent.resolve().is_relative_to(root.resolve()):
        raise MaterializationError("DuckLake receipt parent escapes the repository root")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    lock = receipt_path.with_name(f".{receipt_path.name}.publish.lock")
    descriptor = os.open(
        lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MaterializationError(
                "DuckLake receipt lock is not a regular single-link file"
            )
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise MaterializationError(
                        "timed out acquiring the DuckLake receipt lock"
                    ) from exc
                time.sleep(0.02)

        receipt: dict[str, Any] = {
            "schema": "sawm/ducklake-history-projection-receipt@1",
            "authority": False,
            "scheduling_prerequisite": False,
            "completion_prerequisite": False,
            "install_attempted": False,
            "network_used": False,
        }
        connection = None
        try:
            import duckdb
            from ipfs_accelerate_py.agent_supervisor.integrations.ducklake_history_projection import (
                project_history,
            )

            connection = duckdb.connect(":memory:")
            connection.execute("SET autoinstall_known_extensions = false")
            connection.execute("SET autoload_known_extensions = false")
            row = connection.execute(
                "SELECT installed, install_path FROM duckdb_extensions() "
                "WHERE extension_name = 'ducklake'"
            ).fetchone()
            if not row or not bool(row[0]) or not str(row[1] or ""):
                raise RuntimeError("ducklake_extension_not_locally_installed")
            connection.execute("LOAD ducklake")  # local LOAD only; INSTALL is forbidden
            catalog = (root / str(policy["catalog_path"])).resolve()
            data = (root / str(policy["data_path"])).resolve()
            if not catalog.is_relative_to(root) or not data.is_relative_to(root):
                raise RuntimeError("ducklake_projection_path_escapes_repository")
            catalog.parent.mkdir(parents=True, exist_ok=True)
            data.mkdir(parents=True, exist_ok=True)

            def literal(value: Path) -> str:
                return "'" + str(value).replace("'", "''") + "'"

            connection.execute(
                "ATTACH "
                + literal(Path("ducklake:" + str(catalog)))
                + " AS sawm_history (DATA_PATH "
                + literal(data)
                + ")"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS sawm_history.control_history "
                "(program_definition_cid VARCHAR, projection_cid VARCHAR, "
                "authoritative BOOLEAN)"
            )
            existing = connection.execute(
                "SELECT authoritative FROM sawm_history.control_history "
                "WHERE program_definition_cid = ? AND projection_cid = ?",
                [record["program_definition_cid"], record["projection_cid"]],
            ).fetchall()
            if not existing:
                connection.execute(
                    "INSERT INTO sawm_history.control_history VALUES (?, ?, false)",
                    [record["program_definition_cid"], record["projection_cid"]],
                )
            elif len(existing) != 1 or bool(existing[0][0]):
                raise RuntimeError("ducklake_history_identity_is_not_unique_advisory")
            projection = dict(project_history({"receipt": record}))
            receipt.update(
                {
                    "status": "available",
                    "typed_unavailability": None,
                    "projection": projection,
                    "catalog_path": str(catalog),
                    "data_path": str(data),
                }
            )
        except Exception as exc:
            receipt.update(
                {
                    "status": "typed_unavailability",
                    "typed_unavailability": type(exc).__name__ + ": " + str(exc),
                    "projection": None,
                }
            )
        finally:
            if connection is not None:
                connection.close()
        receipt["receipt_cid"] = _identity(receipt)
        payload = _canonical(receipt) + b"\n"
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        if os.path.lexists(receipt_path):
            observed, observed_sha256 = _load_nofollow_json(
                receipt_path,
                root=root,
                noun="DuckLake history receipt",
            )
            if observed == receipt and observed_sha256 == payload_sha256:
                return observed
            raise MigrationRequired("DuckLake history receipt differs")

        temporary = receipt_path.with_name(f".{receipt_path.name}.pending")
        if os.path.lexists(temporary):
            staged, staged_sha256 = _load_nofollow_json(
                temporary,
                root=root,
                noun="pending DuckLake history receipt",
            )
            if staged != receipt or staged_sha256 != payload_sha256:
                raise MigrationRequired("pending DuckLake history receipt differs")
            os.replace(temporary, receipt_path)
            directory = os.open(
                receipt_path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return receipt
        out = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(out, view) :]
            os.fsync(out)
        finally:
            os.close(out)
        os.replace(temporary, receipt_path)
        directory = os.open(
            receipt_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return receipt
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _assert_fresh_successor_operational_state(target: Path) -> None:
    """Refuse to adopt execution/coordination state from an earlier launch."""

    prohibited = (
        target.with_name("control.execution.duckdb"),
        target.with_name("control.coordination.duckdb"),
        target.with_name(f".{target.name}.state-owner.json"),
        target.parent / "state",
        target.parent / "events",
        target.parent / "registry",
        target.parent / "worktrees",
        target.parent / "merge-queue",
        target.parent / "quack-owner",
    )
    present = [str(path) for path in prohibited if path.exists()]
    if present:
        raise MaterializationError(
            "successor execution/coordination authority is not fresh: "
            + json.dumps(present, sort_keys=True, separators=(",", ":"))
        )


def _m6_receipt_population(
    population: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the immutable M6 source binding for receipt reconstruction."""

    source_binding = dict(population["source_binding"])
    source_binding.update(
        {
            "head": _M6_FROZEN_SUCCESSOR_BINDING["source_head"],
            "tree": _M6_FROZEN_SUCCESSOR_BINDING["source_tree"],
            "source_binding_cid": _M6_FROZEN_SUCCESSOR_BINDING[
                "source_binding_cid"
            ],
        }
    )
    return {**dict(population), "source_binding": source_binding}


def _expected_m6_operational_contract(
    population: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> tuple[tuple[tuple[str, ...], ...], dict[str, Any] | None]:
    """Return the exact validations and historical receipt admitted by M6."""

    alias = str(expected["task_id"])
    historical = tuple(
        (str(command),) for command in expected["validation_commands"]
    )
    if alias == "SAWM-000":
        return historical, None
    if alias not in _operational_task_aliases():
        raise MigrationRequired(f"task is outside the M6 operational set: {alias}")
    operational, replacement_count = _operational_validation_commands(
        historical,
        task_alias=alias,
    )
    prior_status = "blocked" if alias == "SAWM-001" else "todo"
    prior_revision = 5 if alias == "SAWM-001" else 1
    receipt = _operational_validation_receipt(
        _m6_receipt_population(population),
        task_alias=alias,
        task_cid=str(expected["task_cid"]),
        expected_status=prior_status,
        expected_revision=prior_revision,
        prior_validations=historical,
        operational_validations=operational,
        replacement_count=replacement_count,
    )
    return operational, receipt


def _verify_m6_task_projection(
    source: Any,
    population: Mapping[str, Any],
    *,
    expected_status_projection: Mapping[str, str] | None = None,
    expected_revision_projection: Mapping[str, int] | None = None,
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify immutable definitions plus M6's exact operational revisions."""

    statuses: dict[str, str] = {}
    revisions: dict[str, int] = {}
    receipt_cids: dict[str, str] = {}
    migration = population["migration_inventory"]
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        raw = source.intent.get_task(str(expected["task_cid"]))
        observed = source.get_task(str(expected["task_cid"]))
        expected_validations, expected_receipt = _expected_m6_operational_contract(
            population,
            expected,
        )
        expected_identity = {
            "task_cid": str(expected["task_cid"]),
            "task_alias": alias,
            "repository_tree_id": migration["definition_source_binding_cid"],
        }
        expected_outputs = [
            {
                "ordinal": index,
                "path": str(item["effect_id"]),
                "effect": dict(item),
            }
            for index, item in enumerate(expected["outputs"])
        ]
        expected_acceptance = [
            {
                "ordinal": index,
                "criterion": str(item["criterion"]),
                "evidence_policy": dict(item),
            }
            for index, item in enumerate(expected["acceptance_criteria"])
        ]
        expected_validation_rows = [
            {"ordinal": index, "argv": list(argv), "policy": {}}
            for index, argv in enumerate(expected_validations)
        ]
        if (
            observed is None
            or raw is None
            or observed.task_cid != expected["task_cid"]
            or observed.task_alias != alias
            or observed.goal_cid != expected["goal_cid"]
            or observed.plan_cid != expected["plan_cid"]
            or observed.objective_id != ""
            or int(observed.ordinal) != int(expected["ordinal"])
            or observed.priority != expected["priority"]
            or raw["identity"] != expected_identity
            or observed.body.get("definition_cid") != expected["definition_cid"]
            or observed.body.get("definition") != expected["definition"]
            or _identity(observed.body.get("definition"))
            != expected["definition_cid"]
            or sorted(observed.dependencies) != sorted(expected["depends_on"])
            or [dict(item) for item in observed.outputs] != expected_outputs
            or [dict(item) for item in observed.acceptance]
            != expected_acceptance
            or [dict(item) for item in observed.validations]
            != expected_validation_rows
        ):
            raise MigrationRequired(f"frozen M6 task contract differs: {alias}")
        observed_receipt = observed.body.get("operational_validation_revision")
        if alias == "SAWM-000":
            if observed_receipt is not None:
                raise MigrationRequired(
                    "frozen M6 operator task acquired an operational receipt"
                )
        else:
            if observed_receipt != expected_receipt:
                raise MigrationRequired(
                    f"frozen M6 operational receipt differs: {alias}"
                )
            receipt_cids[alias] = str(expected_receipt["receipt_cid"])
        statuses[alias] = str(observed.status)
        revisions[alias] = int(observed.revision)
    expected_statuses = dict(expected_status_projection or {})
    expected_revisions = {
        alias: int(value)
        for alias, value in dict(expected_revision_projection or {}).items()
    }
    if expected_status_projection is None:
        expected_statuses = {
            alias: "completed" if alias == "SAWM-000" else "todo"
            for alias in migration["prior_task_cids"]
        }
    if expected_revision_projection is None:
        expected_revisions = {
            alias: 2 if alias == "SAWM-000" else 7 if alias == "SAWM-001" else 2
            for alias in migration["prior_task_cids"]
        }
    expected_aliases = set(migration["prior_task_cids"])
    if set(expected_statuses) != expected_aliases or set(expected_revisions) != expected_aliases:
        raise MigrationRequired("task status/revision expectation set differs")
    if statuses != expected_statuses or revisions != expected_revisions:
        raise MigrationRequired("frozen M6 task status/revision projection differs")
    return statuses, revisions, receipt_cids


def _verify_m9_live_task_projection(
    source: Any,
    population: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Verify M9's exact rearmed task view without weakening M6 contracts."""

    aliases = tuple(population["migration_inventory"]["prior_task_cids"])
    statuses = {
        alias: (
            "completed"
            if alias == "SAWM-000"
            else "retrying"
            if alias == "SAWM-001"
            else "todo"
        )
        for alias in aliases
    }
    revisions = {
        alias: 2 if alias == "SAWM-000" else 10 if alias == "SAWM-001" else 2
        for alias in aliases
    }
    observed = _verify_m6_task_projection(
        source,
        population,
        expected_status_projection=statuses,
        expected_revision_projection=revisions,
    )
    candidate = source.get_task("SAWM-001")
    if (
        candidate is None
        or candidate.task_cid != _M9_LIVE_PREFLIGHT_FAILURE["task_cid"]
        or candidate.status != "retrying"
        or int(candidate.revision) != 10
        or dict(candidate.body).get("completion_receipt")
        != _m9_task_rearm_receipt()
    ):
        raise MigrationRequired("frozen M9 task rearm projection differs")
    return observed


def _verify_frozen_m6_authority(
    root: Path,
    config: Mapping[str, Any],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the stopped M6 bytes before copying them into M7."""

    authority = _m7_source_repair_authority(population, config)
    _assert_m7_source_delta(root, population, authority)
    prior = (root / str(authority["prior_store_id"])).resolve()
    target = (root / str(authority["target_store_id"])).resolve()
    if (
        not prior.is_relative_to(root)
        or not target.is_relative_to(root)
        or prior == target
        or not prior.is_file()
    ):
        raise MigrationRequired("M7 prior/target authority paths are not confined")
    _assert_offline(prior)
    initial_hash = _store_sha256(prior)
    if (
        initial_hash != authority["prior_control_store_sha256"]
        or prior.stat().st_size != int(authority["prior_control_store_size"])
    ):
        raise MigrationRequired("frozen M6 control-store bytes differ")
    receipt_path = root / str(authority["prior_materialization_receipt_path"])
    if (
        _store_sha256(receipt_path)
        != authority["prior_materialization_receipt_file_sha256"]
    ):
        raise MigrationRequired("frozen M6 migration receipt bytes differ")
    _verify_receipt_anchor(
        receipt_path,
        str(authority["prior_materialization_receipt_cid"]),
    )
    status_path = root / str(authority["prior_owner_status_path"])
    if _store_sha256(status_path) != authority["prior_owner_status_sha256"]:
        raise MigrationRequired("frozen M6 owner status bytes differ")
    owner_status = _load_json(status_path)
    identity = owner_status.get("identity") or {}
    if (
        owner_status.get("lifecycle") != "stopped"
        or identity.get("status") != "stopped"
        or identity.get("server_id") != authority["prior_server_id"]
        or identity.get("process_birth_id")
        != authority["prior_process_birth_id"]
        or int(identity.get("generation") or 0)
        != int(authority["prior_generation"])
        or identity.get("database_uuid") != authority["prior_database_uuid"]
    ):
        raise MigrationRequired("frozen M6 owner is not exactly stopped")
    marker = prior.with_name(f".{prior.name}.state-owner.json")
    if marker.exists():
        raise MigrationRequired("frozen M6 still has a live owner marker")

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    with tempfile.TemporaryDirectory(prefix="sawm-r2-m6-replay-", dir="/tmp") as temp_dir:
        replay_path = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(prior, replay_path)
        if _store_sha256(replay_path) != initial_hash:
            raise MigrationRequired("frozen M6 verification copy differs")
        schema = verify_datasets_authoritative_operational_schema(replay_path)
        if schema.get("valid") is not True:
            raise MigrationRequired("frozen M6 operational schema does not verify")
        semantic_authority_digest = _semantic_authority_digest(replay_path)
        if (
            semantic_authority_digest
            != authority["prior_semantic_authority_digest"]
        ):
            raise MigrationRequired("frozen M6 semantic authority differs")
        frozen_base_authority_digest = _frozen_base_authority_digest(
            replay_path
        )
        if (
            frozen_base_authority_digest
            != authority["prior_frozen_base_authority_digest"]
        ):
            raise MigrationRequired("frozen M6 base authority differs")
        append_surface_digest = _append_surface_digest(replay_path)
        if append_surface_digest != authority["prior_append_surface_digest"]:
            raise MigrationRequired("frozen M6 append surfaces differ")
        source = DatabaseTaskSource(
            replay_path,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        try:
            snap = source.snapshot()
            statuses, revisions, receipt_cids = _verify_m6_task_projection(
                source,
                population,
            )
            plan = source.plans.get(str(population["plan_root_cid"]))
            if (
                snap.task_count != 45
                or snap.goal_count != 29
                or int(snap.event_cursor)
                != int(authority["prior_event_watermark"])
                or str(snap.projection_cid) != authority["prior_projection_cid"]
                or plan is None
                or int(plan.get("revision") or 0)
                != int(authority["prior_plan_revision"])
            ):
                raise MigrationRequired("frozen M6 snapshot identity differs")
            with source.intent._connection(write=False) as connection:
                prefix, count = _event_prefix_digest(
                    connection,
                    int(authority["prior_event_watermark"]),
                )
                owner = _positional_rows(
                    connection.execute(
                        "SELECT server_id, process_birth_id, status "
                        "FROM state_servers WHERE generation = ?",
                        [int(authority["prior_generation"])],
                    ).fetchall(),
                    3,
                )
                completion = _positional_rows(
                    connection.execute(
                        "SELECT receipt_cid, task_cid FROM completion_receipts "
                        "ORDER BY receipt_cid"
                    ).fetchall(),
                    2,
                )
                if (
                    prefix != authority["prior_event_prefix_sha256"]
                    or count != int(authority["prior_event_watermark"])
                    or owner
                    != [
                        (
                            authority["prior_server_id"],
                            authority["prior_process_birth_id"],
                            "stopped",
                        )
                    ]
                    or completion
                    != [
                        (
                            population["migration_inventory"][
                                "prior_completion_receipt_cid"
                            ],
                            population["migration_inventory"][
                                "prior_task_cids"
                            ]["SAWM-000"],
                        )
                    ]
                ):
                    raise MigrationRequired(
                        "frozen M6 event/owner/completion evidence differs"
                    )
            if not source.projection_matches_events():
                raise MigrationRequired("frozen M6 replay projection differs")
        finally:
            source.close()
    if _store_sha256(prior) != initial_hash:
        raise MaterializationError("frozen M6 verification changed authority bytes")
    return {
        "valid": True,
        "database_path": str(prior),
        "database_sha256": initial_hash,
        "event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "event_watermark": authority["prior_event_watermark"],
        "projection_cid": authority["prior_projection_cid"],
        "statuses": statuses,
        "revisions": revisions,
        "operational_validation_receipt_cids": receipt_cids,
        "owner_status_sha256": authority["prior_owner_status_sha256"],
        "semantic_authority_digest": semantic_authority_digest,
        "frozen_base_authority_digest": frozen_base_authority_digest,
        "append_surface_digest": append_surface_digest,
    }


def _verify_frozen_m7_authority(
    root: Path,
    config: Mapping[str, Any],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify stopped M7 only through an exact disposable byte copy."""

    authority = _m8_source_repair_authority(population, config)
    _assert_m8_source_delta(root, population, authority)
    prior = (root / str(authority["prior_store_id"])).resolve()
    target = (root / str(authority["target_store_id"])).resolve()
    if (
        not prior.is_relative_to(root)
        or not target.is_relative_to(root)
        or prior == target
        or not prior.is_file()
    ):
        raise MigrationRequired("M8 prior/target authority paths are not confined")
    _assert_offline(prior)
    initial_hash = _store_sha256(prior)
    if (
        initial_hash != authority["prior_control_store_sha256"]
        or prior.stat().st_size != int(authority["prior_control_store_size"])
    ):
        raise MigrationRequired("frozen M7 control-store bytes differ")

    receipt_path = root / str(authority["prior_materialization_receipt_path"])
    if (
        not receipt_path.is_file()
        or _store_sha256(receipt_path)
        != authority["prior_materialization_receipt_file_sha256"]
    ):
        raise MigrationRequired("frozen M7 migration receipt bytes differ")
    _verify_receipt_anchor(
        receipt_path,
        str(authority["prior_materialization_receipt_cid"]),
    )
    status_path = root / str(authority["prior_owner_status_path"])
    if (
        not status_path.is_file()
        or _store_sha256(status_path) != authority["prior_owner_status_sha256"]
    ):
        raise MigrationRequired("frozen M7 owner status bytes differ")
    owner_status = _load_json(status_path)
    owner_identity = owner_status.get("identity") or {}
    if (
        owner_status.get("lifecycle") != "stopped"
        or owner_identity.get("status") != "stopped"
        or owner_identity.get("server_id") != authority["prior_server_id"]
        or owner_identity.get("process_birth_id")
        != authority["prior_process_birth_id"]
        or int(owner_identity.get("generation") or 0)
        != int(authority["prior_generation"])
        or owner_identity.get("database_uuid")
        != authority["prior_database_uuid"]
    ):
        raise MigrationRequired("frozen M7 owner is not exactly stopped")
    marker = prior.with_name(f".{prior.name}.state-owner.json")
    if marker.exists():
        raise MigrationRequired("frozen M7 still has a live owner marker")

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m7-frozen-verify-",
        dir="/tmp",
    ) as temp_dir:
        replay_path = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(prior, replay_path)
        if _store_sha256(replay_path) != initial_hash:
            raise MigrationRequired("frozen M7 verification copy differs")
        schema = verify_datasets_authoritative_operational_schema(replay_path)
        if schema.get("valid") is not True:
            raise MigrationRequired("frozen M7 operational schema does not verify")
        semantic_authority_digest = _semantic_authority_digest(replay_path)
        frozen_base_authority_digest = _frozen_base_authority_digest(
            replay_path
        )
        append_surface_digest = _append_surface_digest(replay_path)
        if (
            semantic_authority_digest
            != authority["prior_semantic_authority_digest"]
            or frozen_base_authority_digest
            != authority["prior_frozen_base_authority_digest"]
            or append_surface_digest != authority["prior_append_surface_digest"]
        ):
            raise MigrationRequired("frozen M7 authority digest differs")

        source = DatabaseTaskSource(
            replay_path,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        try:
            snap = source.snapshot()
            statuses, revisions, receipt_cids = _verify_m6_task_projection(
                source,
                population,
            )
            plan = source.plans.get(str(population["plan_root_cid"]))
            if (
                snap.task_count != 45
                or snap.goal_count != 29
                or int(snap.event_cursor)
                != int(authority["prior_event_watermark"])
                or str(snap.projection_cid)
                != authority["prior_projection_cid"]
                or plan is None
                or int(plan.get("revision") or 0)
                != int(authority["prior_plan_revision"])
            ):
                raise MigrationRequired("frozen M7 snapshot identity differs")
            for expected in population["objectives"]:
                observed = source.get_goal(str(expected["goal_cid"]))
                expected_body = {
                    key: value
                    for key, value in expected.items()
                    if key
                    not in {
                        "goal_cid",
                        "goal_id",
                        "goal_alias",
                        "title",
                        "status",
                        "ordinal",
                        "objective_id",
                    }
                }
                if (
                    observed is None
                    or str(observed.get("goal_cid")) != expected["goal_cid"]
                    or str(observed.get("goal_alias")) != expected["goal_id"]
                    or str(observed.get("objective_id") or "")
                    != str(expected.get("objective_id") or "")
                    or str(observed.get("parent_goal_cid") or "")
                    != str(expected.get("parent_goal_cid") or "")
                    or int(observed.get("ordinal") or 0)
                    != int(expected["ordinal"])
                    or str(observed.get("title")) != expected["title"]
                    or str(observed.get("status")) != expected["status"]
                    or int(observed.get("revision") or 0) != 1
                    or dict(observed.get("body") or {}) != expected_body
                ):
                    raise MigrationRequired(
                        f"frozen M7 goal contract differs: {expected['goal_id']}"
                    )
            migration = population["migration_inventory"]
            with source.intent._connection(write=False) as connection:
                prefix, count = _event_prefix_digest(
                    connection,
                    int(authority["prior_event_watermark"]),
                )
                owner = _positional_rows(
                    connection.execute(
                        "SELECT server_id, process_birth_id, status "
                        "FROM state_servers WHERE generation = ?",
                        [int(authority["prior_generation"])],
                    ).fetchall(),
                    3,
                )
                completion = _positional_rows(
                    connection.execute(
                        "SELECT receipt_cid, task_cid FROM completion_receipts "
                        "ORDER BY receipt_cid"
                    ).fetchall(),
                    2,
                )
                counts = {
                    table: int(_table_count(connection, table))
                    for table in (
                        "tasks",
                        "task_revisions",
                        "goals",
                        "plans",
                        "plan_revisions",
                        "evidence_nodes",
                        "domain_events",
                        "completion_receipts",
                    )
                }
                if (
                    prefix != authority["prior_event_prefix_sha256"]
                    or count != int(authority["prior_event_watermark"])
                    or owner
                    != [
                        (
                            authority["prior_server_id"],
                            authority["prior_process_birth_id"],
                            "stopped",
                        )
                    ]
                    or completion
                    != [
                        (
                            migration["prior_completion_receipt_cid"],
                            migration["prior_task_cids"]["SAWM-000"],
                        )
                    ]
                    or counts
                    != {
                        "tasks": 45,
                        "task_revisions": 49,
                        "goals": 29,
                        "plans": 1,
                        "plan_revisions": 8,
                        "evidence_nodes": 9,
                        "domain_events": 170,
                        "completion_receipts": 1,
                    }
                ):
                    raise MigrationRequired(
                        "frozen M7 event/owner/completion evidence differs"
                    )
            if not source.projection_matches_events():
                raise MigrationRequired("frozen M7 replay projection differs")
        finally:
            source.close()
    if _store_sha256(prior) != initial_hash:
        raise MaterializationError("frozen M7 verification changed authority bytes")
    return {
        "valid": True,
        "database_path": str(prior),
        "database_sha256": initial_hash,
        "event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "event_watermark": authority["prior_event_watermark"],
        "projection_cid": authority["prior_projection_cid"],
        "statuses": statuses,
        "revisions": revisions,
        "operational_validation_receipt_cids": receipt_cids,
        "owner_status_sha256": authority["prior_owner_status_sha256"],
        "semantic_authority_digest": semantic_authority_digest,
        "frozen_base_authority_digest": frozen_base_authority_digest,
        "append_surface_digest": append_surface_digest,
    }


def _assert_m8_prior_publication_anchor(
    root: Path,
    authority: Mapping[str, Any],
    prior: Path,
) -> None:
    """Recheck stopped M7 immediately around M8 publication."""

    if not prior.is_relative_to(root) or not prior.is_file():
        raise MigrationRequired("M8 prior authority path is not confined")
    _assert_offline(prior)
    prior_sha256, prior_size = _stable_regular_sha256(
        prior,
        root=root,
        noun="frozen M7 control store",
    )
    if (
        prior_sha256 != authority["prior_control_store_sha256"]
        or prior_size != int(authority["prior_control_store_size"])
    ):
        raise MigrationRequired("frozen M7 bytes changed before M8 publication")

    marker = prior.with_name(f".{prior.name}.state-owner.json")
    if os.path.lexists(marker):
        raise MigrationRequired("frozen M7 regained a live owner marker")

    receipt_path = root / str(authority["prior_materialization_receipt_path"])
    _, receipt_sha256 = _load_nofollow_json(
        receipt_path,
        root=root,
        noun="frozen M7 migration receipt",
    )
    if receipt_sha256 != authority["prior_materialization_receipt_file_sha256"]:
        raise MigrationRequired("frozen M7 migration receipt bytes changed")

    status_path = root / str(authority["prior_owner_status_path"])
    owner_status, status_sha256 = _load_nofollow_json(
        status_path,
        root=root,
        noun="frozen M7 owner status",
    )
    identity = owner_status.get("identity") or {}
    if (
        status_sha256 != authority["prior_owner_status_sha256"]
        or owner_status.get("lifecycle") != "stopped"
        or identity.get("status") != "stopped"
        or identity.get("server_id") != authority["prior_server_id"]
        or identity.get("process_birth_id")
        != authority["prior_process_birth_id"]
        or int(identity.get("generation") or 0)
        != int(authority["prior_generation"])
        or identity.get("database_uuid") != authority["prior_database_uuid"]
    ):
        raise MigrationRequired("frozen M7 owner status changed before publication")


def _m7_migration_body(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    authority = _m7_source_repair_authority(population, config)
    repair_paths = authority["bounded_control_plane_repair_paths"]
    return {
        "schema": "sawm/operator-control-plane-source-migration@5",
        "migration_revision": _M7_MIGRATION_REVISION,
        "board_namespace": NAMESPACE,
        "plan_revision": REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "operator_task_cid": population["migration_inventory"][
            "prior_task_cids"
        ]["SAWM-000"],
        "prior_store_id": authority["prior_store_id"],
        "target_store_id": authority["target_store_id"],
        "prior_control_store_sha256": authority[
            "prior_control_store_sha256"
        ],
        "prior_event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "prior_event_watermark": authority["prior_event_watermark"],
        "prior_projection_cid": authority["prior_projection_cid"],
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_head": authority["prior_source_head"],
        "prior_tree": authority["prior_source_tree"],
        "prior_plan_revision": authority["prior_plan_revision"],
        "target_plan_revision": authority["target_plan_revision"],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "current_head": population["source_binding"]["head"],
        "current_tree": population["source_binding"]["tree"],
        "bounded_control_plane_repair_paths": list(repair_paths),
        "bounded_control_plane_repair_sha256": {
            path: population["source_binding"]["control_sha256"][path]
            for path in repair_paths
        },
        "live_preflight_failure": authority["live_preflight_failure"],
        "live_preflight_failure_cid": authority["live_preflight_failure_cid"],
        "prepublication_materialization_failure": dict(
            _M7_PREPUBLICATION_MATERIALIZATION_FAILURE
        ),
        "prepublication_materialization_failure_cid": _identity(
            _M7_PREPUBLICATION_MATERIALIZATION_FAILURE
        ),
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "prior_materialization_receipt_file_sha256": authority[
            "prior_materialization_receipt_file_sha256"
        ],
        "prior_owner_status_sha256": authority["prior_owner_status_sha256"],
        "prior_semantic_authority_digest": authority[
            "prior_semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": authority[
            "prior_frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": authority[
            "prior_append_surface_digest"
        ],
        "validation_runtime": dict(config["validation_runtime"]),
        "validator_digest": validation_digest,
        "supersession_reason": _M7_SUPERSESSION_REASON,
        "supersession_mode": _M7_SUPERSESSION_MODE,
        "prior_authority_preserved": True,
        "accepted_task_definitions_rewritten": False,
        "accepted_goal_definitions_rewritten": False,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "operator_completion_replayed": False,
        "worker_self_approval": False,
    }


def _m7_migration_plan_delta(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _m7_source_repair_authority(population, config)
    return {
        "kind": _M7_SUPERSESSION_REASON,
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_migration_receipt_cids": [
            *[
                entry["migration_receipt_cid"]
                for entry in population["migration_inventory"][
                    "migration_history"
                ]
            ],
            authority["prior_materialization_receipt_cid"],
        ],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "live_preflight_failure_cid": authority["live_preflight_failure_cid"],
        "prepublication_materialization_failure_cid": _identity(
            _M7_PREPUBLICATION_MATERIALIZATION_FAILURE
        ),
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }


def _m8_migration_body(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    authority = _m8_source_repair_authority(population, config)
    repair_paths = authority["bounded_control_plane_repair_paths"]
    prior_repair = population["migration_inventory"].get(
        "source_repair_materialization"
    )
    if not isinstance(prior_repair, Mapping):
        raise MaterializationError("M8 prior source-repair authority is missing")
    return {
        "schema": "sawm/operator-control-plane-source-migration@6",
        "migration_revision": _M8_MIGRATION_REVISION,
        "board_namespace": NAMESPACE,
        "plan_revision": REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "operator_task_cid": population["migration_inventory"][
            "prior_task_cids"
        ]["SAWM-000"],
        "prior_store_id": authority["prior_store_id"],
        "target_store_id": authority["target_store_id"],
        "prior_control_store_sha256": authority[
            "prior_control_store_sha256"
        ],
        "prior_event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "prior_event_watermark": authority["prior_event_watermark"],
        "prior_projection_cid": authority["prior_projection_cid"],
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_head": authority["prior_source_head"],
        "prior_tree": authority["prior_source_tree"],
        "prior_plan_revision": authority["prior_plan_revision"],
        "target_plan_revision": authority["target_plan_revision"],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "current_head": population["source_binding"]["head"],
        "current_tree": population["source_binding"]["tree"],
        "bounded_control_plane_repair_paths": list(repair_paths),
        "bounded_control_plane_repair_sha256": {
            path: population["source_binding"]["control_sha256"][path]
            for path in repair_paths
        },
        "live_preflight_failure": authority["live_preflight_failure"],
        "live_preflight_failure_cid": authority["live_preflight_failure_cid"],
        "prior_source_repair_materialization_cid": _identity(prior_repair),
        "source_repair_successor_materialization_cid": (
            _M8_SOURCE_REPAIR_AUTHORITY_CID
        ),
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "prior_materialization_receipt_file_sha256": authority[
            "prior_materialization_receipt_file_sha256"
        ],
        "prior_owner_status_sha256": authority["prior_owner_status_sha256"],
        "prior_semantic_authority_digest": authority[
            "prior_semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": authority[
            "prior_frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": authority[
            "prior_append_surface_digest"
        ],
        "validation_runtime": dict(config["validation_runtime"]),
        "validator_digest": validation_digest,
        "supersession_reason": _M8_SUPERSESSION_REASON,
        "supersession_mode": _M8_SUPERSESSION_MODE,
        "prior_authority_preserved": True,
        "accepted_task_definitions_rewritten": False,
        "accepted_goal_definitions_rewritten": False,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "operator_completion_replayed": False,
        "worker_self_approval": False,
    }


def _m8_migration_plan_delta(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _m8_source_repair_authority(population, config)
    prior_repair = population["migration_inventory"].get(
        "source_repair_materialization"
    )
    if not isinstance(prior_repair, Mapping):
        raise MaterializationError("M8 prior source-repair authority is missing")
    return {
        "kind": _M8_SUPERSESSION_REASON,
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_migration_receipt_cids": [
            *[
                entry["migration_receipt_cid"]
                for entry in population["migration_inventory"][
                    "migration_history"
                ]
            ],
            prior_repair["prior_materialization_receipt_cid"],
            authority["prior_materialization_receipt_cid"],
        ],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "live_preflight_failure_cid": authority["live_preflight_failure_cid"],
        "prior_source_repair_materialization_cid": _identity(prior_repair),
        "source_repair_successor_materialization_cid": (
            _M8_SOURCE_REPAIR_AUTHORITY_CID
        ),
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }


def _m9_task_rearm_receipt() -> dict[str, Any]:
    """Return the exact closed receipt accepted by coordinator rearm."""

    return {
        "operation": "operator_control_plane_repair",
        "settlement_id": _M8_LIVE_FAILURE_RECEIPT["settlement_id"],
    }


def _m9_migration_body(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    authority = _m9_live_recovery_authority(population, config)
    repair_paths = authority["bounded_control_plane_repair_paths"]
    return {
        "schema": "sawm/operator-control-plane-runtime-recovery@1",
        "migration_revision": _M9_MIGRATION_REVISION,
        "board_namespace": NAMESPACE,
        "plan_revision": REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "operator_task_cid": population["migration_inventory"][
            "prior_task_cids"
        ]["SAWM-000"],
        "rearmed_task_cid": _M8_LIVE_FAILURE_RECEIPT["task_cid"],
        "prior_store_id": authority["prior_store_id"],
        "prior_coordination_store_id": authority[
            "prior_coordination_store_id"
        ],
        "target_store_id": authority["target_store_id"],
        "target_coordination_store_id": authority[
            "target_coordination_store_id"
        ],
        "prior_control_store_sha256": authority[
            "prior_control_store_sha256"
        ],
        "prior_pre_audit_control_store_sha256": authority[
            "prior_pre_audit_control_store_sha256"
        ],
        "prior_offline_checkpoint_semantic_change": False,
        "prior_coordination_store_sha256": authority[
            "prior_coordination_store_sha256"
        ],
        "prior_coordination_projection_digest": authority[
            "prior_coordination_projection_digest"
        ],
        "target_coordination_projection_digest": authority[
            "target_coordination_projection_digest"
        ],
        "prior_event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "prior_event_watermark": authority["prior_event_watermark"],
        "prior_projection_cid": authority["prior_projection_cid"],
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_head": authority["prior_source_head"],
        "prior_tree": authority["prior_source_tree"],
        "prior_plan_revision": authority["prior_plan_revision"],
        "target_plan_revision": authority["target_plan_revision"],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "current_head": population["source_binding"]["head"],
        "current_tree": population["source_binding"]["tree"],
        "bounded_control_plane_repair_paths": list(repair_paths),
        "bounded_control_plane_repair_sha256": {
            path: population["source_binding"]["control_sha256"][path]
            for path in repair_paths
        },
        "live_implementation_failure": authority[
            "live_implementation_failure"
        ],
        "live_implementation_failure_cid": _identity(
            authority["live_implementation_failure"]
        ),
        "task_rearm": authority["task_rearm"],
        "task_rearm_receipt": _m9_task_rearm_receipt(),
        "provider_strategy": authority["provider_strategy"],
        "source_repair_successor_materialization_cid": (
            _M8_SOURCE_REPAIR_AUTHORITY_CID
        ),
        "live_recovery_successor_materialization_cid": (
            _M9_LIVE_RECOVERY_AUTHORITY_CID
        ),
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "prior_materialization_receipt_file_sha256": authority[
            "prior_materialization_receipt_file_sha256"
        ],
        "prior_owner_status_sha256": authority["prior_owner_status_sha256"],
        "prior_semantic_authority_digest": authority[
            "prior_semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": authority[
            "prior_frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": authority[
            "prior_append_surface_digest"
        ],
        "validation_runtime": dict(config["validation_runtime"]),
        "validator_digest": validation_digest,
        "supersession_reason": _M9_SUPERSESSION_REASON,
        "supersession_mode": _M9_SUPERSESSION_MODE,
        "prior_authority_preserved": True,
        "coordination_sidecar_copied_before_rearm": True,
        "execution_sidecar_copied": False,
        "failed_attempt_history_preserved": True,
        "failed_completion_barrier_rearmed": True,
        "accepted_task_definitions_rewritten": False,
        "accepted_goal_definitions_rewritten": False,
        "task_revision_changes": 1,
        "task_status_changes": 1,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "operator_completion_replayed": False,
        "worker_self_approval": False,
    }


def _m9_migration_plan_delta(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _m9_live_recovery_authority(population, config)
    return {
        "kind": _M9_SUPERSESSION_REASON,
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "live_implementation_failure_cid": _identity(
            authority["live_implementation_failure"]
        ),
        "failure_settlement_id": _M8_LIVE_FAILURE_RECEIPT["settlement_id"],
        "source_repair_successor_materialization_cid": (
            _M8_SOURCE_REPAIR_AUTHORITY_CID
        ),
        "live_recovery_successor_materialization_cid": (
            _M9_LIVE_RECOVERY_AUTHORITY_CID
        ),
        "prior_coordination_projection_digest": authority[
            "prior_coordination_projection_digest"
        ],
        "target_coordination_projection_digest": authority[
            "target_coordination_projection_digest"
        ],
        "rearmed_task_cid": _M8_LIVE_FAILURE_RECEIPT["task_cid"],
        "rearmed_task_status": "retrying",
        "rearmed_task_revision": 10,
        "task_revision_changes": 1,
        "task_status_changes": 1,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }


def _m10_migration_body(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Build the exact source-binding evidence appended by M10."""

    authority = _m10_live_projection_authority(population, config)
    repair_paths = authority["bounded_control_plane_repair_paths"]
    return {
        "schema": "sawm/operator-control-plane-source-migration@7",
        "migration_revision": _M10_MIGRATION_REVISION,
        "board_namespace": NAMESPACE,
        "plan_revision": REVISION,
        "program_definition_cid": population["program_definition_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "operator_task_cid": population["migration_inventory"][
            "prior_task_cids"
        ]["SAWM-000"],
        "prior_store_id": authority["prior_store_id"],
        "prior_coordination_store_id": authority[
            "prior_coordination_store_id"
        ],
        "target_store_id": authority["target_store_id"],
        "target_coordination_store_id": authority[
            "target_coordination_store_id"
        ],
        "prior_publication_control_store_sha256": authority[
            "prior_publication_control_store_sha256"
        ],
        "prior_control_store_sha256": authority["prior_control_store_sha256"],
        "prior_live_checkpoint_semantic_change": False,
        "prior_coordination_store_sha256": authority[
            "prior_coordination_store_sha256"
        ],
        "prior_coordination_projection_digest": authority[
            "prior_coordination_projection_digest"
        ],
        "target_coordination_projection_digest": authority[
            "target_coordination_projection_digest"
        ],
        "prior_event_prefix_sha256": authority["prior_event_prefix_sha256"],
        "prior_event_watermark": authority["prior_event_watermark"],
        "prior_projection_cid": authority["prior_projection_cid"],
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_head": authority["prior_source_head"],
        "prior_tree": authority["prior_source_tree"],
        "prior_plan_revision": authority["prior_plan_revision"],
        "target_plan_revision": authority["target_plan_revision"],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "current_head": population["source_binding"]["head"],
        "current_tree": population["source_binding"]["tree"],
        "bounded_control_plane_repair_paths": list(repair_paths),
        "bounded_control_plane_repair_sha256": {
            path: population["source_binding"]["control_sha256"][path]
            for path in repair_paths
        },
        "live_preflight_failure": authority["live_preflight_failure"],
        "live_preflight_failure_cid": authority["live_preflight_failure_cid"],
        "live_task_projection": authority["live_task_projection"],
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "prior_materialization_receipt_file_sha256": authority[
            "prior_materialization_receipt_file_sha256"
        ],
        "prior_owner_status_sha256": authority["prior_owner_status_sha256"],
        "prior_semantic_authority_digest": authority[
            "prior_semantic_authority_digest"
        ],
        "target_semantic_authority_digest": authority[
            "target_semantic_authority_digest"
        ],
        "prior_frozen_base_authority_digest": authority[
            "prior_frozen_base_authority_digest"
        ],
        "prior_append_surface_digest": authority[
            "prior_append_surface_digest"
        ],
        "prior_catalog_digest": authority["prior_catalog_digest"],
        "live_recovery_successor_materialization_cid": (
            _M9_LIVE_RECOVERY_AUTHORITY_CID
        ),
        "live_projection_successor_materialization_cid": (
            _M10_LIVE_PROJECTION_AUTHORITY_CID
        ),
        "validation_runtime": dict(config["validation_runtime"]),
        "validator_digest": validation_digest,
        "supersession_reason": _M10_SUPERSESSION_REASON,
        "supersession_mode": _M10_SUPERSESSION_MODE,
        "prior_authority_preserved": True,
        "coordination_sidecar_copied_unchanged": True,
        "coordination_semantic_changes": 0,
        "execution_sidecar_copied": False,
        "accepted_task_definitions_rewritten": False,
        "accepted_goal_definitions_rewritten": False,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "operator_completion_replayed": False,
        "worker_self_approval": False,
    }


def _m10_migration_plan_delta(
    population: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _m10_live_projection_authority(population, config)
    return {
        "kind": _M10_SUPERSESSION_REASON,
        "prior_source_binding_cid": authority["prior_source_binding_cid"],
        "prior_materialization_receipt_cid": authority[
            "prior_materialization_receipt_cid"
        ],
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "live_preflight_failure_cid": authority["live_preflight_failure_cid"],
        "live_recovery_successor_materialization_cid": (
            _M9_LIVE_RECOVERY_AUTHORITY_CID
        ),
        "live_projection_successor_materialization_cid": (
            _M10_LIVE_PROJECTION_AUTHORITY_CID
        ),
        "prior_coordination_projection_digest": authority[
            "prior_coordination_projection_digest"
        ],
        "target_coordination_projection_digest": authority[
            "target_coordination_projection_digest"
        ],
        "target_semantic_authority_digest": authority[
            "target_semantic_authority_digest"
        ],
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "coordination_semantic_changes": 0,
        "worker_self_approval": False,
    }


def _m9_expected_coordination_projection(
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the only admitted coordinator delta from the frozen M8 view."""

    if prior.get("projection_root") != _M8_COORDINATION_PROJECTION_DIGEST:
        raise MigrationRequired("M8 coordination projection root differs")
    expected = json.loads(json.dumps(dict(prior), sort_keys=True))
    expected.pop("projection_root", None)
    tasks = [
        task
        for task in expected.get("tasks", ())
        if task.get("task_cid") == _M8_LIVE_FAILURE_RECEIPT["task_cid"]
    ]
    if len(tasks) != 1 or tasks[0].get("ready") is not False:
        raise MigrationRequired("M8 coordination task readiness differs")
    tasks[0]["ready"] = True
    expected_completion = {
        "task_cid": _M8_LIVE_FAILURE_RECEIPT["task_cid"],
        "status": "failed",
        "body": _M8_LIVE_FAILURE_RECEIPT,
    }
    if expected.get("logical_completions") != [expected_completion]:
        raise MigrationRequired("M8 failed coordination barrier differs")
    expected["logical_completions"] = []
    counts = expected.get("counts")
    if not isinstance(counts, dict) or counts.get("logical_completions") != 1:
        raise MigrationRequired("M8 coordination completion count differs")
    counts["logical_completions"] = 0
    expected["projection_root"] = _identity(expected)
    if expected["projection_root"] != _M9_COORDINATION_PROJECTION_DIGEST:
        raise MaterializationError("derived M9 coordination projection differs")
    return expected


def _verify_frozen_m8_live_authority(
    root: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify stopped M8 control and coordination authority without mutation."""

    control = _confined_leaf(
        root,
        str(authority["prior_store_id"]),
        noun="frozen M8 control store",
    )
    coordination = _confined_leaf(
        root,
        str(authority["prior_coordination_store_id"]),
        noun="frozen M8 coordination store",
    )
    if control == coordination:
        raise MigrationRequired("M8 live authority pair is missing or unconfined")
    control_sha256, control_size = _stable_regular_sha256(
        control,
        root=root,
        noun="frozen M8 control store",
        required_link_count=1,
    )
    coordination_sha256, coordination_size = _stable_regular_sha256(
        coordination,
        root=root,
        noun="frozen M8 coordination store",
        required_link_count=1,
    )
    if (
        control_size != int(authority["prior_control_store_size"])
        or control_sha256 != authority["prior_control_store_sha256"]
        or coordination_size != int(authority["prior_coordination_store_size"])
        or coordination_sha256 != authority["prior_coordination_store_sha256"]
    ):
        raise MigrationRequired("M8 live authority bytes differ")
    _assert_offline(control)

    receipt_path = _confined_leaf(
        root,
        str(authority["prior_materialization_receipt_path"]),
        noun="M8 migration receipt",
    )
    receipt, receipt_sha = _load_nofollow_json(
        receipt_path,
        root=root,
        noun="M8 migration receipt",
    )
    unhashed_receipt = dict(receipt)
    receipt_cid = str(unhashed_receipt.pop("receipt_cid", ""))
    if (
        receipt_sha != authority["prior_materialization_receipt_file_sha256"]
        or receipt_cid != authority["prior_materialization_receipt_cid"]
        or _identity(unhashed_receipt) != receipt_cid
        or receipt.get("current_source_binding_cid")
        != authority["prior_source_binding_cid"]
        or receipt.get("projection_cid") != _M8_EXPECTED_PROJECTION_CID
    ):
        raise MigrationRequired("M8 migration receipt differs")

    owner_path = _confined_leaf(
        root,
        str(authority["prior_owner_status_path"]),
        noun="stopped M8 Quack owner status",
    )
    owner, owner_sha = _load_nofollow_json(
        owner_path,
        root=root,
        noun="stopped M8 Quack owner status",
    )
    identity = owner.get("identity")
    if (
        owner_sha != authority["prior_owner_status_sha256"]
        or owner.get("lifecycle") != "stopped"
        or not isinstance(identity, Mapping)
        or identity.get("status") != "stopped"
        or int(identity.get("generation") or 0) != int(authority["prior_generation"])
        or identity.get("database_uuid") != authority["prior_database_uuid"]
        or identity.get("process_birth_id") != authority["prior_process_birth_id"]
        or identity.get("server_id") != authority["prior_server_id"]
        or identity.get("store_id") != authority["prior_store_id"]
    ):
        raise MigrationRequired("M8 Quack owner stop evidence differs")
    owner_marker = control.with_name(f".{control.name}.state-owner.json")
    if os.path.lexists(owner_marker):
        raise MigrationRequired("M8 Quack owner marker remains present")

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with tempfile.TemporaryDirectory(prefix="sawm-r2-m8-live-", dir="/tmp") as temp_dir:
        control_copy = Path(temp_dir) / "control.duckdb"
        coordination_copy = Path(temp_dir) / "control.coordination.duckdb"
        shutil.copyfile(control, control_copy)
        shutil.copyfile(coordination, coordination_copy)
        copied_control_sha256, copied_control_size = _stable_regular_sha256(
            control_copy,
            root=Path(temp_dir),
            noun="disposable frozen M8 control copy",
            required_link_count=1,
        )
        copied_coordination_sha256, copied_coordination_size = (
            _stable_regular_sha256(
                coordination_copy,
                root=Path(temp_dir),
                noun="disposable frozen M8 coordination copy",
                required_link_count=1,
            )
        )
        if (
            copied_control_sha256 != control_sha256
            or copied_control_size != control_size
            or copied_coordination_sha256 != coordination_sha256
            or copied_coordination_size != coordination_size
        ):
            raise MigrationRequired("disposable M8 authority copy differs")

        # These digests bind the raw frozen bytes.  DatabaseTaskSource may
        # reconstruct derived projection timestamps when it replays the
        # disposable copy, so compute the append-surface receipt before that
        # replay and never mistake the replayed copy for the raw authority.
        semantic_digest = _semantic_authority_digest(control_copy)
        frozen_digest = _frozen_base_authority_digest(control_copy)
        append_digest = _append_surface_digest(control_copy)
        if (
            semantic_digest != authority["prior_semantic_authority_digest"]
            or frozen_digest != authority["prior_frozen_base_authority_digest"]
            or append_digest != authority["prior_append_surface_digest"]
        ):
            raise MigrationRequired("M8 live authority digests differ")

        schema = verify_datasets_authoritative_operational_schema(control_copy)
        if schema.get("valid") is not True:
            raise MigrationRequired("M8 operational schema does not verify")
        source = DatabaseTaskSource(control_copy, install_schema=False)
        try:
            snapshot = source.snapshot()
            candidate = source.get_task("SAWM-001")
            operator = source.get_task("SAWM-000")
            if (
                int(snapshot.event_cursor) != int(authority["prior_event_watermark"])
                or snapshot.projection_cid != authority["prior_projection_cid"]
                or int(snapshot.task_count) != 45
                or candidate is None
                or candidate.status != "blocked"
                or candidate.revision != 9
                or dict(candidate.body).get("completion_receipt")
                != _M8_LIVE_FAILURE_RECEIPT
                or operator is None
                or operator.status != "completed"
                or operator.revision != 2
                or not source.projection_matches_events()
            ):
                raise MigrationRequired("M8 live control projection differs")
        finally:
            source.close()

        import duckdb

        connection = duckdb.connect(str(control_copy), read_only=True)
        try:
            prefix_digest, prefix_count = _event_prefix_digest(
                connection,
                int(authority["prior_event_watermark"]),
            )
            counts = {
                table: int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM "{table}"'
                    ).fetchone()[0]
                )
                for table in (
                    "tasks",
                    "goals",
                    "plans",
                    "plan_revisions",
                    "evidence_nodes",
                    "domain_events",
                    "task_revisions",
                    "completion_receipts",
                )
            }
            tail = connection.execute(
                "SELECT global_sequence, event_id, event_type, task_cid, "
                "body_json FROM domain_events WHERE global_sequence IN (173, 174) "
                "ORDER BY global_sequence"
            ).fetchall()
            plan_rows = connection.execute(
                "SELECT plan_cid, revision FROM plans"
            ).fetchall()
        finally:
            connection.close()
        if (
            prefix_digest != authority["prior_event_prefix_sha256"]
            or prefix_count != int(authority["prior_event_watermark"])
            or counts
            != {
                "tasks": 45,
                "goals": 29,
                "plans": 1,
                "plan_revisions": 9,
                "evidence_nodes": 10,
                "domain_events": 174,
                "task_revisions": 0,
                "completion_receipts": 1,
            }
            or len(plan_rows) != 1
            or int(plan_rows[0][1]) != 9
            or len(tail) != 2
            or str(tail[0][1])
            != _M8_LIVE_IMPLEMENTATION_FAILURE["claim_event_id"]
            or str(tail[1][1])
            != _M8_LIVE_IMPLEMENTATION_FAILURE["settlement_event_id"]
        ):
            raise MigrationRequired("M8 live event authority differs")
        first_body = json.loads(str(tail[0][4]))["body"]
        second_body = json.loads(str(tail[1][4]))["body"]
        if (
            first_body.get("previous_status") != "todo"
            or first_body.get("status") != "in_progress"
            or int(first_body.get("revision") or 0) != 8
            or second_body.get("previous_status") != "in_progress"
            or second_body.get("status") != "blocked"
            or int(second_body.get("revision") or 0) != 9
            or second_body.get("receipt") != _M8_LIVE_FAILURE_RECEIPT
        ):
            raise MigrationRequired("M8 live status transition evidence differs")

        prior_coordination = read_coordination_registry_projection(
            coordination_copy
        )
        expected_coordination = _m9_expected_coordination_projection(
            prior_coordination
        )
        connection = duckdb.connect(str(coordination_copy), read_only=True)
        try:
            coordination_event_count = int(
                connection.execute("SELECT COUNT(*) FROM lease_events").fetchone()[0]
            )
            failure_rows = connection.execute(
                "SELECT status, body_json FROM task_completions WHERE task_cid = ?",
                [_M8_LIVE_FAILURE_RECEIPT["task_cid"]],
            ).fetchall()
        finally:
            connection.close()
        if (
            coordination_event_count
            != int(authority["prior_coordination_event_count"])
            or failure_rows
            != [("failed", _canonical(_M8_LIVE_FAILURE_RECEIPT).decode("utf-8"))]
        ):
            raise MigrationRequired("M8 coordination settlement differs")

    final_control_sha256, final_control_size = _stable_regular_sha256(
        control,
        root=root,
        noun="frozen M8 control store after verification",
        required_link_count=1,
    )
    final_coordination_sha256, final_coordination_size = _stable_regular_sha256(
        coordination,
        root=root,
        noun="frozen M8 coordination store after verification",
        required_link_count=1,
    )
    if (
        (final_control_sha256, final_control_size)
        != (control_sha256, control_size)
        or (final_coordination_sha256, final_coordination_size)
        != (coordination_sha256, coordination_size)
    ):
        raise MigrationRequired("M8 live authority changed during verification")

    return {
        "database_path": str(control),
        "coordination_path": str(coordination),
        "projection_cid": authority["prior_projection_cid"],
        "event_watermark": authority["prior_event_watermark"],
        "control_store_sha256": authority["prior_control_store_sha256"],
        "coordination_store_sha256": authority[
            "prior_coordination_store_sha256"
        ],
        "coordination_projection": prior_coordination,
        "expected_coordination_projection": expected_coordination,
        "semantic_authority_digest": semantic_digest,
        "frozen_base_authority_digest": frozen_digest,
        "append_surface_digest": append_digest,
    }


def _verify_frozen_m9_live_authority(
    root: Path,
    authority: Mapping[str, Any],
    population: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the stopped post-preflight M9 pair without mutating it."""

    control = _confined_leaf(
        root,
        str(authority["prior_store_id"]),
        noun="frozen M9 control store",
    )
    coordination = _confined_leaf(
        root,
        str(authority["prior_coordination_store_id"]),
        noun="frozen M9 coordination store",
    )
    if control == coordination:
        raise MigrationRequired("M9 live authority pair is missing or unconfined")
    control_sha256, control_size = _stable_regular_sha256(
        control,
        root=root,
        noun="frozen M9 control store",
        required_link_count=1,
    )
    coordination_sha256, coordination_size = _stable_regular_sha256(
        coordination,
        root=root,
        noun="frozen M9 coordination store",
        required_link_count=1,
    )
    if (
        control_sha256 != authority["prior_control_store_sha256"]
        or control_size != int(authority["prior_control_store_size"])
        or coordination_sha256 != authority["prior_coordination_store_sha256"]
        or coordination_size != int(authority["prior_coordination_store_size"])
    ):
        raise MigrationRequired("M9 stopped live authority bytes differ")
    _assert_offline(control)
    if os.path.lexists(control.with_name("control.execution.duckdb")):
        raise MigrationRequired("frozen M9 acquired an execution sidecar")

    receipt_path = _confined_leaf(
        root,
        str(authority["prior_materialization_receipt_path"]),
        noun="M9 final pair commit marker",
    )
    receipt, receipt_sha256 = _load_nofollow_json(
        receipt_path,
        root=root,
        noun="M9 final pair commit marker",
    )
    unhashed_receipt = dict(receipt)
    receipt_cid = str(unhashed_receipt.pop("receipt_cid", ""))
    if (
        receipt_sha256
        != authority["prior_materialization_receipt_file_sha256"]
        or receipt_cid != authority["prior_materialization_receipt_cid"]
        or receipt_cid != _identity(unhashed_receipt)
        or receipt.get("schema") != "sawm/non-authoritative-migration-receipt@7"
        or receipt.get("receipt_is_final_pair_commit_marker") is not True
        or receipt.get("database_path") != authority["prior_store_id"]
        or receipt.get("coordination_path")
        != authority["prior_coordination_store_id"]
        or receipt.get("control_store_sha256")
        != authority["prior_publication_control_store_sha256"]
        or int(receipt.get("control_store_size") or 0) != control_size
        or receipt.get("coordination_store_sha256") != coordination_sha256
        or int(receipt.get("coordination_store_size") or 0)
        != coordination_size
        or receipt.get("migration_projection_cid")
        != authority["prior_projection_cid"]
        or int(receipt.get("migration_event_watermark") or 0)
        != int(authority["prior_event_watermark"])
        or receipt.get("coordination_projection_digest")
        != authority["prior_coordination_projection_digest"]
        or int(receipt.get("coordination_event_count") or 0)
        != int(authority["prior_coordination_event_count"])
        or receipt.get("semantic_authority_digest")
        != authority["prior_semantic_authority_digest"]
        or receipt.get("current_source_binding_cid")
        != authority["prior_source_binding_cid"]
        or receipt.get("execution_sidecar_copied") is not False
        or receipt.get("worker_self_approval") is not False
    ):
        raise MigrationRequired("M9 final pair commit marker differs")

    owner_path = _confined_leaf(
        root,
        str(authority["prior_owner_status_path"]),
        noun="stopped M9 Quack owner status",
    )
    owner, owner_sha256 = _load_nofollow_json(
        owner_path,
        root=root,
        noun="stopped M9 Quack owner status",
    )
    identity = owner.get("identity")
    if (
        owner_sha256 != authority["prior_owner_status_sha256"]
        or owner.get("lifecycle") != "stopped"
        or int(owner.get("port") or 0) != 45_251
        or not isinstance(identity, Mapping)
        or identity.get("status") != "stopped"
        or int(identity.get("generation") or 0)
        != int(authority["prior_generation"])
        or identity.get("database_uuid") != authority["prior_database_uuid"]
        or identity.get("process_birth_id")
        != authority["prior_process_birth_id"]
        or identity.get("server_id") != authority["prior_server_id"]
        or identity.get("store_id") != authority["prior_store_id"]
    ):
        raise MigrationRequired("M9 Quack owner stop evidence differs")
    if os.path.lexists(control.with_name(f".{control.name}.state-owner.json")):
        raise MigrationRequired("M9 Quack owner marker remains present")

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m9-live-",
        dir="/tmp",
    ) as temp_dir:
        control_copy = Path(temp_dir) / "control.duckdb"
        coordination_copy = Path(temp_dir) / "control.coordination.duckdb"
        replay_copy = Path(temp_dir) / "control.replay.duckdb"
        shutil.copyfile(control, control_copy)
        shutil.copyfile(coordination, coordination_copy)
        copied_control = _stable_regular_sha256(
            control_copy,
            root=Path(temp_dir),
            noun="disposable frozen M9 control copy",
            required_link_count=1,
        )
        copied_coordination = _stable_regular_sha256(
            coordination_copy,
            root=Path(temp_dir),
            noun="disposable frozen M9 coordination copy",
            required_link_count=1,
        )
        if copied_control != (control_sha256, control_size) or copied_coordination != (
            coordination_sha256,
            coordination_size,
        ):
            raise MigrationRequired("disposable M9 authority copy differs")

        semantic_digest = _semantic_authority_digest(control_copy)
        frozen_digest = _frozen_base_authority_digest(control_copy)
        append_digest = _append_surface_digest(control_copy)
        import duckdb

        connection = duckdb.connect(str(control_copy), read_only=True)
        try:
            catalog_digest = _main_catalog_digest_on(connection)
            prefix_digest, prefix_count = _event_prefix_digest(
                connection,
                int(authority["prior_event_watermark"]),
            )
            counts = {
                table: int(_table_count(connection, table))
                for table in (
                    "tasks",
                    "goals",
                    "plans",
                    "plan_revisions",
                    "evidence_nodes",
                    "domain_events",
                    "task_revisions",
                    "completion_receipts",
                    "provider_calls",
                    "provider_invocations",
                    "effect_claims",
                    "merge_attempts",
                )
            }
            state_rows = _positional_rows(
                connection.execute(
                    "SELECT generation, status, server_id, process_birth_id, "
                    "database_uuid FROM state_servers WHERE generation = ?",
                    [int(authority["prior_generation"])],
                ).fetchall(),
                5,
            )
        finally:
            connection.close()
        if (
            semantic_digest != authority["prior_semantic_authority_digest"]
            or frozen_digest != authority["prior_frozen_base_authority_digest"]
            or append_digest != authority["prior_append_surface_digest"]
            or catalog_digest != authority["prior_catalog_digest"]
            or prefix_digest != authority["prior_event_prefix_sha256"]
            or prefix_count != int(authority["prior_event_watermark"])
            or counts
            != {
                "tasks": 45,
                "goals": 29,
                "plans": 1,
                "plan_revisions": 10,
                "evidence_nodes": 11,
                "domain_events": 177,
                "task_revisions": 1,
                "completion_receipts": 1,
                "provider_calls": 0,
                "provider_invocations": 0,
                "effect_claims": 0,
                "merge_attempts": 0,
            }
            or state_rows
            != [
                (
                    int(authority["prior_generation"]),
                    "stopped",
                    authority["prior_server_id"],
                    authority["prior_process_birth_id"],
                    authority["prior_database_uuid"],
                )
            ]
        ):
            raise MigrationRequired("M9 stopped live authority projection differs")

        schema = verify_datasets_authoritative_operational_schema(control_copy)
        if schema.get("valid") is not True:
            raise MigrationRequired("M9 operational schema does not verify")
        shutil.copyfile(control_copy, replay_copy)
        source = DatabaseTaskSource(
            replay_copy,
            install_schema=False,
        )
        try:
            snapshot = source.snapshot()
            statuses, revisions, receipt_cids = _verify_m9_live_task_projection(
                source,
                population,
            )
            plan = source.get_plan(str(receipt["program_definition_cid"]))
            if plan is None:
                plan = source.get_plan(
                    "sha256:d9481937430405ff6a512e779b14b7ce676de45d277c65d3763ebe49445ba914"
                )
            if (
                int(snapshot.event_cursor) != int(authority["prior_event_watermark"])
                or snapshot.projection_cid != authority["prior_projection_cid"]
                or int(snapshot.task_count) != 45
                or int(snapshot.goal_count) != 29
                or plan is None
                or int(plan.get("revision") or 0)
                != int(authority["prior_plan_revision"])
                or not source.projection_matches_events()
            ):
                raise MigrationRequired("M9 stopped live snapshot differs")
        finally:
            source.close()

        coordination_projection = read_coordination_registry_projection(
            coordination_copy
        )
        coordination_connection = duckdb.connect(
            str(coordination_copy),
            read_only=True,
        )
        try:
            coordination_event_count = int(
                coordination_connection.execute(
                    "SELECT COUNT(*) FROM lease_events"
                ).fetchone()[0]
            )
        finally:
            coordination_connection.close()
        if (
            coordination_projection.get("projection_root")
            != authority["prior_coordination_projection_digest"]
            or coordination_event_count
            != int(authority["prior_coordination_event_count"])
        ):
            raise MigrationRequired("M9 stopped coordination authority differs")

    final_control = _stable_regular_sha256(
        control,
        root=root,
        noun="frozen M9 control store after verification",
        required_link_count=1,
    )
    final_coordination = _stable_regular_sha256(
        coordination,
        root=root,
        noun="frozen M9 coordination store after verification",
        required_link_count=1,
    )
    if final_control != (control_sha256, control_size) or final_coordination != (
        coordination_sha256,
        coordination_size,
    ):
        raise MigrationRequired("M9 live authority changed during verification")
    return {
        "valid": True,
        "database_path": str(control),
        "coordination_path": str(coordination),
        "control_store_sha256": control_sha256,
        "coordination_store_sha256": coordination_sha256,
        "event_watermark": authority["prior_event_watermark"],
        "projection_cid": authority["prior_projection_cid"],
        "coordination_event_count": coordination_event_count,
        "coordination_projection_digest": coordination_projection[
            "projection_root"
        ],
        "semantic_authority_digest": semantic_digest,
        "frozen_base_authority_digest": frozen_digest,
        "append_surface_digest": append_digest,
        "catalog_digest": catalog_digest,
        "statuses": statuses,
        "revisions": revisions,
        "operational_validation_receipt_cids": receipt_cids,
        "receipt": receipt,
    }


def _assert_m10_prior_publication_anchor(
    root: Path,
    authority: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Rebind the exact stopped M9 pair and evidence at a commit point."""

    control = _confined_leaf(
        root,
        str(authority["prior_store_id"]),
        noun="frozen M9 control store",
    )
    coordination = _confined_leaf(
        root,
        str(authority["prior_coordination_store_id"]),
        noun="frozen M9 coordination store",
    )
    if (
        _stable_regular_sha256(
            control,
            root=root,
            noun="frozen M9 control store",
            required_link_count=1,
        )
        != (
            authority["prior_control_store_sha256"],
            int(authority["prior_control_store_size"]),
        )
        or _stable_regular_sha256(
            coordination,
            root=root,
            noun="frozen M9 coordination store",
            required_link_count=1,
        )
        != (
            authority["prior_coordination_store_sha256"],
            int(authority["prior_coordination_store_size"]),
        )
    ):
        raise MigrationRequired("frozen M9 pair changed before M10 publication")
    _assert_offline(control)
    if (
        os.path.lexists(control.with_name(f".{control.name}.state-owner.json"))
        or os.path.lexists(control.with_name("control.execution.duckdb"))
    ):
        raise MigrationRequired("frozen M9 regained mutable live authority")

    receipt_path = _confined_leaf(
        root,
        str(authority["prior_materialization_receipt_path"]),
        noun="frozen M9 final pair marker",
    )
    receipt, receipt_sha256 = _load_nofollow_json(
        receipt_path,
        root=root,
        noun="frozen M9 final pair marker",
    )
    unhashed = dict(receipt)
    receipt_cid = str(unhashed.pop("receipt_cid", ""))
    if (
        receipt_sha256
        != authority["prior_materialization_receipt_file_sha256"]
        or receipt_cid != authority["prior_materialization_receipt_cid"]
        or receipt_cid != _identity(unhashed)
        or receipt.get("control_store_sha256")
        != authority["prior_publication_control_store_sha256"]
    ):
        raise MigrationRequired("frozen M9 final pair marker changed")

    owner_path = _confined_leaf(
        root,
        str(authority["prior_owner_status_path"]),
        noun="stopped M9 Quack owner status",
    )
    owner, owner_sha256 = _load_nofollow_json(
        owner_path,
        root=root,
        noun="stopped M9 Quack owner status",
    )
    identity = owner.get("identity")
    if (
        owner_sha256 != authority["prior_owner_status_sha256"]
        or owner.get("lifecycle") != "stopped"
        or not isinstance(identity, Mapping)
        or identity.get("status") != "stopped"
        or int(identity.get("generation") or 0)
        != int(authority["prior_generation"])
        or identity.get("server_id") != authority["prior_server_id"]
        or identity.get("process_birth_id")
        != authority["prior_process_birth_id"]
    ):
        raise MigrationRequired("stopped M9 owner evidence changed")
    return control, coordination


def _assert_m9_prior_publication_anchor(
    root: Path,
    authority: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Rebind the stopped M8 pair immediately around M9 publication."""

    control = _confined_leaf(
        root,
        str(authority["prior_store_id"]),
        noun="frozen M8 control store",
    )
    coordination = _confined_leaf(
        root,
        str(authority["prior_coordination_store_id"]),
        noun="frozen M8 coordination store",
    )
    if control == coordination:
        raise MigrationRequired("M8 publication anchor paths are not confined")
    control_sha256, control_size = _stable_regular_sha256(
        control,
        root=root,
        noun="frozen M8 control store",
        required_link_count=1,
    )
    coordination_sha256, coordination_size = _stable_regular_sha256(
        coordination,
        root=root,
        noun="frozen M8 coordination store",
        required_link_count=1,
    )
    _assert_offline(control)
    if (
        control_sha256 != authority["prior_control_store_sha256"]
        or control_size != int(authority["prior_control_store_size"])
        or coordination_sha256
        != authority["prior_coordination_store_sha256"]
        or coordination_size
        != int(authority["prior_coordination_store_size"])
    ):
        raise MigrationRequired("frozen M8 pair changed before M9 publication")
    if os.path.lexists(control.with_name(f".{control.name}.state-owner.json")):
        raise MigrationRequired("frozen M8 control store regained a live owner")

    receipt_path = _confined_leaf(
        root,
        str(authority["prior_materialization_receipt_path"]),
        noun="frozen M8 migration receipt",
    )
    receipt, receipt_sha256 = _load_nofollow_json(
        receipt_path,
        root=root,
        noun="frozen M8 migration receipt",
    )
    unhashed = dict(receipt)
    receipt_cid = str(unhashed.pop("receipt_cid", ""))
    if (
        receipt_sha256 != authority["prior_materialization_receipt_file_sha256"]
        or receipt_cid != authority["prior_materialization_receipt_cid"]
        or receipt_cid != _identity(unhashed)
    ):
        raise MigrationRequired("frozen M8 migration receipt changed")

    owner_path = _confined_leaf(
        root,
        str(authority["prior_owner_status_path"]),
        noun="stopped M8 Quack owner status",
    )
    owner, owner_sha256 = _load_nofollow_json(
        owner_path,
        root=root,
        noun="stopped M8 Quack owner status",
    )
    identity = owner.get("identity")
    if (
        owner_sha256 != authority["prior_owner_status_sha256"]
        or owner.get("lifecycle") != "stopped"
        or not isinstance(identity, Mapping)
        or identity.get("status") != "stopped"
        or int(identity.get("generation") or 0)
        != int(authority["prior_generation"])
        or identity.get("database_uuid") != authority["prior_database_uuid"]
        or identity.get("process_birth_id")
        != authority["prior_process_birth_id"]
        or identity.get("server_id") != authority["prior_server_id"]
        or identity.get("store_id") != authority["prior_store_id"]
    ):
        raise MigrationRequired("stopped M8 Quack owner status changed")
    return control, coordination


def _verify_m9_control_copy(
    control: Path,
    prior_control: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Verify the exact three-event M9 control delta on disposable bytes."""

    authority = _m9_live_recovery_authority(population, config)
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    expected_body = _m9_migration_body(population, config, validation_digest)
    expected_digest = _identity(expected_body)
    expected_delta = _m9_migration_plan_delta(population, config)
    # DatabaseTaskSource reconstructs a few projection timestamps on open.
    # Exercise that replay only on a nested disposable copy; the raw M9 bytes
    # below are compared row-for-row before they can ever be published.
    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m9-projection-verify-",
        dir="/tmp",
    ) as temp_dir:
        projection_copy = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(control, projection_copy)
        schema = verify_datasets_authoritative_operational_schema(projection_copy)
        if schema.get("valid") is not True:
            raise MigrationRequired("M9 operational schema does not verify")
        source = DatabaseTaskSource(projection_copy, install_schema=False)
        try:
            snapshot = source.snapshot()
            candidate = source.get_task("SAWM-001")
            operator = source.get_task("SAWM-000")
            plan = source.get_plan(str(population["plan_root_cid"]))
            if (
                int(snapshot.event_cursor) != _M9_TARGET_EVENT_WATERMARK
                or snapshot.projection_cid != _M9_EXPECTED_PROJECTION_CID
                or int(snapshot.task_count) != 45
                or candidate is None
                or candidate.status != "retrying"
                or candidate.revision != 10
                or dict(candidate.body).get("completion_receipt")
                != _m9_task_rearm_receipt()
                or operator is None
                or operator.status != "completed"
                or operator.revision != 2
                or plan is None
                or int(plan.get("revision") or 0) != _M9_TARGET_PLAN_REVISION
                or not source.projection_matches_events()
            ):
                raise MigrationRequired("M9 control projection differs")
        finally:
            source.close()

    import duckdb

    prior_connection = duckdb.connect(str(prior_control), read_only=True)
    connection = duckdb.connect(str(control), read_only=True)
    try:
        prefix_digest, prefix_count = _event_prefix_digest(
            connection,
            int(authority["prior_event_watermark"]),
        )
        counts = {
            table: int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
            )
            for table in (
                "tasks",
                "goals",
                "plans",
                "plan_revisions",
                "evidence_nodes",
                "domain_events",
                "task_revisions",
                "completion_receipts",
            )
        }
        changed_tables = frozenset(
            {
                "tasks",
                "task_revisions",
                "plans",
                "plan_revisions",
                "evidence_nodes",
                "domain_events",
            }
        )
        prior_tables = _main_table_names(prior_connection)
        target_tables = _main_table_names(connection)
        if (
            prior_tables != target_tables
            or not changed_tables.issubset(prior_tables)
            or _main_catalog_digest_on(prior_connection)
            != _main_catalog_digest_on(connection)
        ):
            raise MigrationRequired("M9 changed the frozen control catalog")
        stable_tables = tuple(
            table for table in prior_tables if table not in changed_tables
        )
        if _authority_table_digest_on(
            prior_connection,
            stable_tables,
        ) != _authority_table_digest_on(connection, stable_tables):
            raise MigrationRequired("M9 changed a non-authorized control table")

        prior_tasks = prior_connection.execute(
            "SELECT * FROM tasks ORDER BY task_cid"
        ).fetchall()
        target_tasks = connection.execute(
            "SELECT * FROM tasks ORDER BY task_cid"
        ).fetchall()
        if len(prior_tasks) != 45 or len(target_tasks) != 45:
            raise MigrationRequired("M9 task table population differs")
        changed_tasks = 0
        for prior_row, target_row in zip(prior_tasks, target_tasks, strict=True):
            if prior_row == target_row:
                continue
            if (
                str(prior_row[0]) != _M8_LIVE_FAILURE_RECEIPT["task_cid"]
                or str(target_row[0]) != str(prior_row[0])
            ):
                raise MigrationRequired("M9 changed an unauthorized task row")
            changed_tasks += 1
            stable_indexes = (0, 1, 2, 3, 4, 5, 8, 9, 11, 13, 14)
            if any(prior_row[index] != target_row[index] for index in stable_indexes):
                raise MigrationRequired("M9 changed SAWM-001 immutable task fields")
            if (
                str(prior_row[6]) != "blocked"
                or int(prior_row[7]) != 9
                or str(target_row[6]) != "retrying"
                or int(target_row[7]) != 10
            ):
                raise MigrationRequired("M9 SAWM-001 status/revision differs")
            prior_body = json.loads(str(prior_row[12]))
            target_body = json.loads(str(target_row[12]))
            if prior_body.pop("completion_receipt", None) != _M8_LIVE_FAILURE_RECEIPT:
                raise MigrationRequired("M9 prior failure receipt differs")
            if target_body.pop("completion_receipt", None) != _m9_task_rearm_receipt():
                raise MigrationRequired("M9 rearm receipt differs")
            if prior_body != target_body:
                raise MigrationRequired("M9 changed SAWM-001 contract body")
        if changed_tasks != 1:
            raise MigrationRequired("M9 did not change exactly one task row")

        prior_plans = prior_connection.execute(
            "SELECT * FROM plans ORDER BY plan_cid"
        ).fetchall()
        target_plans = connection.execute(
            "SELECT * FROM plans ORDER BY plan_cid"
        ).fetchall()
        prior_plan_revisions = prior_connection.execute(
            "SELECT * FROM plan_revisions ORDER BY plan_cid, revision"
        ).fetchall()
        target_plan_revisions = connection.execute(
            "SELECT * FROM plan_revisions ORDER BY plan_cid, revision"
        ).fetchall()
        prior_evidence = prior_connection.execute(
            "SELECT * FROM evidence_nodes ORDER BY evidence_id"
        ).fetchall()
        target_evidence = connection.execute(
            "SELECT * FROM evidence_nodes ORDER BY evidence_id"
        ).fetchall()
        prior_events = prior_connection.execute(
            "SELECT * FROM domain_events ORDER BY global_sequence"
        ).fetchall()
        target_events = connection.execute(
            "SELECT * FROM domain_events ORDER BY global_sequence"
        ).fetchall()
        revision_rows = connection.execute(
            "SELECT task_cid, revision, status, body_json, recorded_at "
            "FROM task_revisions ORDER BY task_cid, revision"
        ).fetchall()
        suffix = target_events[-_M9_EVENT_SUFFIX_LENGTH:]
        evidence_rows = connection.execute(
            "SELECT evidence_id, task_cid, evidence_kind, digest, body_json, "
            "created_at FROM evidence_nodes WHERE digest = ?",
            [expected_digest],
        ).fetchall()
        plan_rows = connection.execute(
            "SELECT revision, body_json, recorded_at "
            "FROM plan_revisions WHERE plan_cid = ? AND revision = 10",
            [str(population["plan_root_cid"])],
        ).fetchall()
    finally:
        connection.close()
        prior_connection.close()

    if (
        prefix_digest != authority["prior_event_prefix_sha256"]
        or prefix_count != int(authority["prior_event_watermark"])
        or counts
        != {
            "tasks": 45,
            "goals": 29,
            "plans": 1,
            "plan_revisions": 10,
            "evidence_nodes": 11,
            "domain_events": 177,
            "task_revisions": 1,
            "completion_receipts": 1,
        }
        or len(revision_rows) != 1
        or str(revision_rows[0][0])
        != _M8_LIVE_FAILURE_RECEIPT["task_cid"]
        or int(revision_rows[0][1]) != 10
        or str(revision_rows[0][2]) != "retrying"
        or json.loads(str(revision_rows[0][3])).get("completion_receipt")
        != _m9_task_rearm_receipt()
        or len(prior_plans) != 1
        or len(target_plans) != 1
        or len(prior_plan_revisions) != 9
        or target_plan_revisions[:-1] != prior_plan_revisions
        or len(target_plan_revisions) != 10
        or len(prior_evidence) != 10
        or len(target_evidence) != 11
        or {
            str(row[0]): row for row in target_evidence
        }.get(str(prior_evidence[0][0])) is None
        or any(
            {str(row[0]): row for row in target_evidence}.get(str(row[0]))
            != row
            for row in prior_evidence
        )
        or len(prior_events) != int(authority["prior_event_watermark"])
        or target_events[: len(prior_events)] != prior_events
        or len(target_events) != _M9_TARGET_EVENT_WATERMARK
        or len(evidence_rows) != 1
        or len(plan_rows) != 1
        or json.loads(str(plan_rows[0][1])).get("last_delta") != expected_delta
        or len(suffix) != _M9_EVENT_SUFFIX_LENGTH
    ):
        raise MigrationRequired("M9 append surfaces differ")

    evidence_id = str(evidence_rows[0][0])
    prior_evidence_by_id = {str(row[0]): row for row in prior_evidence}
    target_evidence_by_id = {str(row[0]): row for row in target_evidence}
    extra_evidence_ids = set(target_evidence_by_id) - set(prior_evidence_by_id)
    if (
        extra_evidence_ids != {evidence_id}
        or
        str(evidence_rows[0][1])
        != population["migration_inventory"]["prior_task_cids"]["SAWM-000"]
        or str(evidence_rows[0][2])
        != "operator_control_plane_runtime_recovery"
        or json.loads(str(evidence_rows[0][4])) != expected_body
    ):
        raise MigrationRequired("M9 recovery evidence differs")

    events: list[dict[str, Any]] = []
    for row in suffix:
        envelope = json.loads(str(row[9]))
        event = {
            "event_id": str(row[0]),
            "stream_id": str(row[1]),
            "sequence": int(row[2]),
            "global_sequence": int(row[3]),
            "event_type": str(row[4]),
            "task_cid": str(row[5]),
            "attempt_id": str(row[6]),
            "session_id": str(row[7]),
            "recorded_at": str(row[8]),
            "body": envelope,
        }
        if (
            content_identity(
                {
                    "stream_id": event["stream_id"],
                    "sequence": event["sequence"],
                    "global_sequence": event["global_sequence"],
                    "event_type": event["event_type"],
                    "body": envelope,
                }
            )
            != event["event_id"]
            or event["stream_id"] != "stream:intent"
            or event["session_id"] != "session:intent"
            or event["attempt_id"]
            or envelope.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/intent-event@1"
            or envelope.get("event_type") != event["event_type"]
            or envelope.get("owner_id") != "sawm-r2-runtime-recovery-migrator"
        ):
            raise MigrationRequired("M9 event identity differs")
        events.append(event)
    if [event["event_type"] for event in events] != [
        "intent.plan_revision_appended",
        "intent.evidence_recorded",
        "intent.task_status_changed",
    ]:
        raise MigrationRequired("M9 event ordering differs")
    plan_inner = events[0]["body"].get("body")
    evidence_inner = events[1]["body"].get("body")
    status_inner = events[2]["body"].get("body")
    prior_plan = prior_plans[0]
    target_plan = target_plans[0]
    prior_task = next(
        row
        for row in prior_tasks
        if str(row[0]) == _M8_LIVE_FAILURE_RECEIPT["task_cid"]
    )
    target_task = next(
        row
        for row in target_tasks
        if str(row[0]) == _M8_LIVE_FAILURE_RECEIPT["task_cid"]
    )
    evidence_row = target_evidence_by_id[evidence_id]
    if (
        not isinstance(plan_inner, Mapping)
        or int(plan_inner.get("revision") or 0) != 10
        or plan_inner.get("delta") != expected_delta
        or prior_plan[:5] != target_plan[:5]
        or int(prior_plan[6]) != 9
        or int(target_plan[6]) != 10
        or str(target_plan[5]) != events[0]["recorded_at"]
        or json.loads(str(target_plan[7])) != plan_inner.get("body")
        or target_plan_revisions[-1]
        != (
            str(population["plan_root_cid"]),
            10,
            str(target_plan[7]),
            events[0]["recorded_at"],
        )
        or not isinstance(evidence_inner, Mapping)
        or evidence_inner.get("evidence_id") != evidence_id
        or evidence_inner.get("body") != expected_body
        or evidence_row
        != (
            evidence_id,
            "",
            population["migration_inventory"]["prior_task_cids"]["SAWM-000"],
            "operator_control_plane_runtime_recovery",
            expected_digest,
            events[1]["recorded_at"],
            _canonical(expected_body).decode("utf-8"),
        )
        or not isinstance(status_inner, Mapping)
        or status_inner.get("task_cid") != _M8_LIVE_FAILURE_RECEIPT["task_cid"]
        or status_inner.get("previous_status") != "blocked"
        or status_inner.get("status") != "retrying"
        or int(status_inner.get("revision") or 0) != 10
        or status_inner.get("receipt") != _m9_task_rearm_receipt()
        or str(target_task[10]) != events[2]["recorded_at"]
        or str(prior_task[9]) != str(target_task[9])
    ):
        raise MigrationRequired("M9 event bodies differ")

    return {
        "projection_cid": _M9_EXPECTED_PROJECTION_CID,
        "event_watermark": _M9_TARGET_EVENT_WATERMARK,
        "projection_matches_events": True,
        "migration_digest": expected_digest,
        "migration_evidence_id": evidence_id,
        "plan_migration_event_id": events[0]["event_id"],
        "migration_evidence_event_id": events[1]["event_id"],
        "task_rearm_event_id": events[2]["event_id"],
        "task_rearm_receipt_cid": content_identity(_m9_task_rearm_receipt()),
        "semantic_authority_digest": _semantic_authority_digest(control),
        "frozen_base_authority_digest": _frozen_base_authority_digest(control),
        "append_surface_digest": _append_surface_digest(control),
        "task_revision_changes": 1,
        "task_status_changes": 1,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
    }


def _verify_m9_coordination_copy(
    coordination: Path,
    prior_coordination: Path,
) -> dict[str, Any]:
    """Verify the exact sanctioned rearm on a copied M8 coordinator."""

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        TASK_CLAIM_FAILURE_REARMED_EVENT,
        read_coordination_registry_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
        canonical_content_cid,
    )

    prior_projection = read_coordination_registry_projection(prior_coordination)
    expected_projection = _m9_expected_coordination_projection(prior_projection)
    observed_projection = read_coordination_registry_projection(coordination)
    if observed_projection != expected_projection:
        raise MigrationRequired("M9 coordination projection differs")

    import duckdb

    prior = duckdb.connect(str(prior_coordination), read_only=True)
    target = duckdb.connect(str(coordination), read_only=True)
    try:
        changed_tables = frozenset(
            {"coordination_tasks", "task_completions", "lease_events"}
        )
        prior_tables = _main_table_names(prior)
        target_tables = _main_table_names(target)
        if (
            prior_tables != target_tables
            or not changed_tables.issubset(prior_tables)
            or _main_catalog_digest_on(prior) != _main_catalog_digest_on(target)
        ):
            raise MigrationRequired("M9 changed the frozen coordination catalog")
        stable_tables = tuple(
            table for table in prior_tables if table not in changed_tables
        )
        if _authority_table_digest_on(
            prior,
            stable_tables,
        ) != _authority_table_digest_on(target, stable_tables):
            raise MigrationRequired("M9 changed a non-authorized coordination table")
        prior_count = int(prior.execute("SELECT COUNT(*) FROM lease_events").fetchone()[0])
        target_count = int(target.execute("SELECT COUNT(*) FROM lease_events").fetchone()[0])
        prior_events = prior.execute(
            "SELECT * FROM lease_events ORDER BY event_id"
        ).fetchall()
        retained_events = target.execute(
            "SELECT * FROM lease_events "
            "WHERE event_type <> ? ORDER BY event_id",
            [TASK_CLAIM_FAILURE_REARMED_EVENT],
        ).fetchall()
        rearm_rows = target.execute(
            "SELECT event_id, lease_id, scope_key, event_type, fencing_token, "
            "fence_epoch, observed_at_ms, body_json FROM lease_events "
            "WHERE event_type = ?",
            [TASK_CLAIM_FAILURE_REARMED_EVENT],
        ).fetchall()
        prior_tasks = prior.execute(
            "SELECT * FROM coordination_tasks ORDER BY task_cid"
        ).fetchall()
        target_tasks = target.execute(
            "SELECT * FROM coordination_tasks ORDER BY task_cid"
        ).fetchall()
        prior_completions = prior.execute(
            "SELECT * FROM task_completions ORDER BY task_cid"
        ).fetchall()
        target_completions = target.execute(
            "SELECT * FROM task_completions ORDER BY task_cid"
        ).fetchall()
    finally:
        target.close()
        prior.close()
    if (
        prior_count != 35
        or target_count != 36
        or retained_events != prior_events
        or len(rearm_rows) != 1
        or len(prior_tasks) != 1
        or len(target_tasks) != 1
        or prior_tasks[0][:4] != target_tasks[0][:4]
        or prior_tasks[0][5:] != target_tasks[0][5:]
        or prior_tasks[0][4] is not False
        or target_tasks[0][4] is not True
        or str(prior_tasks[0][0]) != _M8_LIVE_FAILURE_RECEIPT["task_cid"]
        or prior_completions
        != [
            (
                _M8_LIVE_FAILURE_RECEIPT["task_cid"],
                _M9_COORDINATION_REARM_OBSERVED_AT_MS - 1,
                "failed",
                _canonical(_M8_LIVE_FAILURE_RECEIPT).decode("utf-8"),
            )
        ]
        or target_completions
    ):
        raise MigrationRequired("M9 coordination event history differs")
    row = rearm_rows[0]
    body = json.loads(str(row[7]))
    expected_body = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "task-claim-failure-rearm@1"
        ),
        "operation": "operator_control_plane_repair",
        "task_cid": _M8_LIVE_FAILURE_RECEIPT["task_cid"],
        "failure_settlement_id": _M8_LIVE_FAILURE_RECEIPT["settlement_id"],
        "control_status": "retrying",
        "control_revision": 10,
        "control_receipt_id": canonical_content_cid(
            _m9_task_rearm_receipt()
        ),
    }
    expected_body["rearm_id"] = canonical_content_cid(expected_body)
    if (
        body != expected_body
        or not re.fullmatch(r"lease-event:[0-9a-f]{32}", str(row[0]))
        or str(row[1]) != _M8_LIVE_FAILURE_RECEIPT["lease_id"]
        or str(row[2])
        != "task:" + _M8_LIVE_FAILURE_RECEIPT["task_cid"]
        or int(row[4]) != 1
        or int(row[5]) != 1
        or int(row[6]) != _M9_COORDINATION_REARM_OBSERVED_AT_MS
    ):
        raise MigrationRequired("M9 coordination rearm event differs")
    return {
        "coordination_projection_digest": observed_projection["projection_root"],
        "coordination_event_count": target_count,
        "coordination_rearm_event_id": str(row[0]),
        "coordination_rearm_id": body["rearm_id"],
        "coordination_store_sha256": _store_sha256(coordination),
    }


def _verify_m9_store_pair_copy(
    control: Path,
    coordination: Path,
    prior_control: Path,
    prior_coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    initial_control_sha256 = _store_sha256(control)
    initial_coordination_sha256 = _store_sha256(coordination)
    initial_prior_control_sha256 = _store_sha256(prior_control)
    initial_prior_coordination_sha256 = _store_sha256(prior_coordination)
    control_report = _verify_m9_control_copy(
        control,
        prior_control,
        population,
        config,
        validation_digest,
    )
    coordination_report = _verify_m9_coordination_copy(
        coordination,
        prior_coordination,
    )
    if (
        _store_sha256(control) != initial_control_sha256
        or _store_sha256(coordination) != initial_coordination_sha256
        or _store_sha256(prior_control) != initial_prior_control_sha256
        or _store_sha256(prior_coordination)
        != initial_prior_coordination_sha256
    ):
        raise MaterializationError("M9 verification changed store-pair bytes")
    return {
        "valid": True,
        "task_count": 45,
        "goal_count": 29,
        **control_report,
        **coordination_report,
        "control_store_sha256": initial_control_sha256,
        "coordination_store_sha256": initial_coordination_sha256,
    }


def _verify_m9_store_pair(
    root: Path,
    control: Path,
    coordination: Path,
    prior_control: Path,
    prior_coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Verify both published files only through exact disposable copies."""

    _assert_offline(control)
    initial_control_hash, _ = _stable_regular_sha256(
        control,
        root=root,
        noun="published M9 control store",
        required_link_count=1,
    )
    initial_coordination_hash, _ = _stable_regular_sha256(
        coordination,
        root=root,
        noun="published M9 coordination store",
        required_link_count=1,
    )
    with tempfile.TemporaryDirectory(prefix="sawm-r2-m9-verify-", dir="/tmp") as temp_dir:
        control_copy = Path(temp_dir) / "control.duckdb"
        coordination_copy = Path(temp_dir) / "control.coordination.duckdb"
        prior_control_copy = Path(temp_dir) / "prior-control.duckdb"
        prior_coordination_copy = Path(temp_dir) / "prior-control.coordination.duckdb"
        shutil.copyfile(control, control_copy)
        shutil.copyfile(coordination, coordination_copy)
        shutil.copyfile(prior_control, prior_control_copy)
        shutil.copyfile(prior_coordination, prior_coordination_copy)
        report = _verify_m9_store_pair_copy(
            control_copy,
            coordination_copy,
            prior_control_copy,
            prior_coordination_copy,
            population,
            config,
            validation_digest,
        )
    if (
        _stable_regular_sha256(
            control,
            root=root,
            noun="published M9 control store",
            required_link_count=1,
        )[0]
        != initial_control_hash
        or _stable_regular_sha256(
            coordination,
            root=root,
            noun="published M9 coordination store",
            required_link_count=1,
        )[0]
        != initial_coordination_hash
    ):
        raise MaterializationError("M9 verification changed published bytes")
    return {
        **report,
        "control_store_sha256": initial_control_hash,
        "coordination_store_sha256": initial_coordination_hash,
    }


def _verify_m10_store_pair_copy(
    control: Path,
    coordination: Path,
    prior_control: Path,
    prior_coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Verify M10's two-event control append and byte-identical coordinator."""

    initial_control_sha256 = _store_sha256(control)
    initial_coordination_sha256 = _store_sha256(coordination)
    initial_prior_control_sha256 = _store_sha256(prior_control)
    initial_prior_coordination_sha256 = _store_sha256(prior_coordination)
    authority = _m10_live_projection_authority(population, config)
    expected_body = _m10_migration_body(population, config, validation_digest)
    expected_digest = _identity(expected_body)
    expected_delta = _m10_migration_plan_delta(population, config)

    if (
        initial_prior_control_sha256 != authority["prior_control_store_sha256"]
        or initial_prior_coordination_sha256
        != authority["prior_coordination_store_sha256"]
        or initial_coordination_sha256 != initial_prior_coordination_sha256
    ):
        raise MigrationRequired("M10 staged predecessor or coordinator bytes differ")

    import duckdb
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    prior = duckdb.connect(str(prior_control), read_only=True)
    target = duckdb.connect(str(control), read_only=True)
    try:
        append_tables = frozenset(
            {"plans", "plan_revisions", "evidence_nodes", "domain_events"}
        )
        prior_tables = _main_table_names(prior)
        target_tables = _main_table_names(target)
        if (
            prior_tables != target_tables
            or not append_tables.issubset(prior_tables)
            or _main_catalog_digest_on(prior) != _main_catalog_digest_on(target)
            or _main_catalog_digest_on(prior) != authority["prior_catalog_digest"]
        ):
            raise MigrationRequired("M10 changed the frozen control catalog")
        stable_tables = tuple(
            table for table in prior_tables if table not in append_tables
        )
        if _authority_table_digest_on(
            prior,
            stable_tables,
        ) != _authority_table_digest_on(target, stable_tables):
            raise MigrationRequired("M10 changed a non-authorized control table")

        prior_plans = _positional_rows(
            prior.execute("SELECT * FROM plans ORDER BY 1, 2").fetchall(),
            8,
        )
        target_plans = _positional_rows(
            target.execute("SELECT * FROM plans ORDER BY 1, 2").fetchall(),
            8,
        )
        prior_revisions = _positional_rows(
            prior.execute(
                "SELECT * FROM plan_revisions ORDER BY 1, 2, 3, 4"
            ).fetchall(),
            4,
        )
        target_revisions = _positional_rows(
            target.execute(
                "SELECT * FROM plan_revisions ORDER BY 1, 2, 3, 4"
            ).fetchall(),
            4,
        )
        prior_evidence = _positional_rows(
            prior.execute(
                "SELECT * FROM evidence_nodes ORDER BY 1, 2, 3, 4, 5, 6, 7"
            ).fetchall(),
            7,
        )
        target_evidence = _positional_rows(
            target.execute(
                "SELECT * FROM evidence_nodes ORDER BY 1, 2, 3, 4, 5, 6, 7"
            ).fetchall(),
            7,
        )
        prior_events = _positional_rows(
            prior.execute(
                "SELECT event_id, stream_id, sequence, global_sequence, "
                "event_type, task_cid, attempt_id, session_id, recorded_at, "
                "body_json FROM domain_events ORDER BY global_sequence"
            ).fetchall(),
            10,
        )
        target_events = _positional_rows(
            target.execute(
                "SELECT event_id, stream_id, sequence, global_sequence, "
                "event_type, task_cid, attempt_id, session_id, recorded_at, "
                "body_json FROM domain_events ORDER BY global_sequence"
            ).fetchall(),
            10,
        )
    finally:
        target.close()
        prior.close()

    if (
        len(prior_plans) != 1
        or len(target_plans) != 1
        or len(prior_revisions) != 10
        or len(target_revisions) != 11
        or target_revisions[:-1] != prior_revisions
        or len(prior_evidence) != 11
        or len(target_evidence) != 12
        or len(prior_events) != _M9_TARGET_EVENT_WATERMARK
        or len(target_events) != _M10_TARGET_EVENT_WATERMARK
        or target_events[: len(prior_events)] != prior_events
    ):
        raise MigrationRequired("M10 append-only control surfaces differ")

    prior_plan = prior_plans[0]
    target_plan = target_plans[0]
    new_revision = target_revisions[-1]
    prior_plan_body = json.loads(str(prior_plan[7]))
    expected_plan_body = {
        **prior_plan_body,
        "current_source_binding_cid": population["source_binding"][
            "source_binding_cid"
        ],
        "source_migration_revision": _M10_MIGRATION_REVISION,
        "source_migration_digest": expected_digest,
        "supersession_mode": _M10_SUPERSESSION_MODE,
        "last_delta": expected_delta,
    }
    expected_evidence_id = content_identity(
        {
            "task_cid": population["migration_inventory"]["prior_task_cids"][
                "SAWM-000"
            ],
            "evidence_kind": "operator_control_plane_source_migration",
            "digest": expected_digest,
            "body": expected_body,
        }
    )
    prior_evidence_by_id = {str(row[0]): row for row in prior_evidence}
    target_evidence_by_id = {str(row[0]): row for row in target_evidence}
    extra_evidence_ids = set(target_evidence_by_id) - set(prior_evidence_by_id)
    new_evidence = target_evidence_by_id.get(expected_evidence_id)
    if (
        any(
            target_evidence_by_id.get(evidence_id) != row
            for evidence_id, row in prior_evidence_by_id.items()
        )
        or extra_evidence_ids != {expected_evidence_id}
        or new_evidence is None
        or prior_plan[:5] != target_plan[:5]
        or int(prior_plan[6]) != _M9_TARGET_PLAN_REVISION
        or int(target_plan[6]) != _M10_TARGET_PLAN_REVISION
        or json.loads(str(target_plan[7])) != expected_plan_body
        or str(target_plan[5]) != str(new_revision[3])
        or new_revision
        != (
            str(population["plan_root_cid"]),
            _M10_TARGET_PLAN_REVISION,
            _canonical(expected_plan_body).decode("utf-8"),
            str(target_plan[5]),
        )
        or str(new_evidence[1])
        or str(new_evidence[2])
        != population["migration_inventory"]["prior_task_cids"]["SAWM-000"]
        or str(new_evidence[3]) != "operator_control_plane_source_migration"
        or str(new_evidence[4]) != expected_digest
        or json.loads(str(new_evidence[6])) != expected_body
    ):
        raise MigrationRequired("M10 plan or migration evidence differs")

    suffix = target_events[-_M10_EVENT_SUFFIX_LENGTH:]
    events: list[dict[str, Any]] = []
    for row in suffix:
        envelope = json.loads(str(row[9]))
        event = {
            "event_id": str(row[0]),
            "stream_id": str(row[1]),
            "sequence": int(row[2]),
            "global_sequence": int(row[3]),
            "event_type": str(row[4]),
            "task_cid": str(row[5]),
            "attempt_id": str(row[6]),
            "session_id": str(row[7]),
            "recorded_at": str(row[8]),
            "body": envelope,
        }
        if (
            content_identity(
                {
                    "stream_id": event["stream_id"],
                    "sequence": event["sequence"],
                    "global_sequence": event["global_sequence"],
                    "event_type": event["event_type"],
                    "body": envelope,
                }
            )
            != event["event_id"]
            or event["stream_id"] != "stream:intent"
            or event["sequence"] != event["global_sequence"]
            or event["session_id"] != "session:intent"
            or event["attempt_id"]
            or envelope.get("schema")
            != "ipfs_accelerate_py/agent-supervisor/intent-event@1"
            or envelope.get("event_type") != event["event_type"]
            or envelope.get("recorded_at") != event["recorded_at"]
            or envelope.get("owner_id") != "sawm-r2-live-projection-migrator"
        ):
            raise MigrationRequired("M10 event identity differs")
        events.append(event)
    if [event["event_type"] for event in events] != [
        "intent.plan_revision_appended",
        "intent.evidence_recorded",
    ]:
        raise MigrationRequired("M10 event ordering differs")
    plan_event, evidence_event = events
    if (
        plan_event["global_sequence"] != 178
        or evidence_event["global_sequence"] != 179
        or plan_event["body"].get("subject_id")
        != population["plan_root_cid"]
        or plan_event["body"].get("body", {}).get("body")
        != expected_plan_body
        or evidence_event["task_cid"]
        != population["migration_inventory"]["prior_task_cids"]["SAWM-000"]
        or evidence_event["body"].get("subject_id") != expected_evidence_id
        or evidence_event["body"].get("body", {}).get("body") != expected_body
    ):
        raise MigrationRequired("M10 event bodies differ")

    semantic_digest = _semantic_authority_digest(control)
    frozen_digest = _frozen_base_authority_digest(control)
    append_digest = _append_surface_digest(control)
    if (
        semantic_digest != authority["target_semantic_authority_digest"]
        or frozen_digest != authority["prior_frozen_base_authority_digest"]
    ):
        raise MigrationRequired("M10 changed frozen semantic authority")

    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m10-projection-",
        dir="/tmp",
    ) as temp_dir:
        replay_path = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(control, replay_path)
        schema = verify_datasets_authoritative_operational_schema(replay_path)
        if schema.get("valid") is not True:
            raise MigrationRequired("M10 operational schema does not verify")
        source = DatabaseTaskSource(
            replay_path,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        try:
            snapshot = source.snapshot()
            statuses, revisions, receipt_cids = _verify_m9_live_task_projection(
                source,
                population,
            )
            plan = source.get_plan(str(population["plan_root_cid"]))
            if (
                int(snapshot.event_cursor) != _M10_TARGET_EVENT_WATERMARK
                or snapshot.projection_cid != _M10_EXPECTED_PROJECTION_CID
                or int(snapshot.task_count) != 45
                or int(snapshot.goal_count) != 29
                or plan is None
                or int(plan.get("revision") or 0) != _M10_TARGET_PLAN_REVISION
                or plan.get("body") != expected_plan_body
                or not source.projection_matches_events()
            ):
                raise MigrationRequired("M10 control projection differs")
        finally:
            source.close()

    prior_coordination_projection = read_coordination_registry_projection(
        prior_coordination
    )
    coordination_projection = read_coordination_registry_projection(coordination)
    if (
        coordination_projection != prior_coordination_projection
        or coordination_projection.get("projection_root")
        != authority["target_coordination_projection_digest"]
    ):
        raise MigrationRequired("M10 changed coordination semantics")
    coordination_connection = duckdb.connect(str(coordination), read_only=True)
    try:
        coordination_event_count = int(
            coordination_connection.execute(
                "SELECT COUNT(*) FROM lease_events"
            ).fetchone()[0]
        )
    finally:
        coordination_connection.close()
    if coordination_event_count != int(authority["target_coordination_event_count"]):
        raise MigrationRequired("M10 coordination event count differs")

    if (
        _store_sha256(control) != initial_control_sha256
        or _store_sha256(coordination) != initial_coordination_sha256
        or _store_sha256(prior_control) != initial_prior_control_sha256
        or _store_sha256(prior_coordination)
        != initial_prior_coordination_sha256
    ):
        raise MaterializationError("M10 verification changed store-pair bytes")
    return {
        "valid": True,
        "task_count": 45,
        "goal_count": 29,
        "projection_cid": _M10_EXPECTED_PROJECTION_CID,
        "event_watermark": _M10_TARGET_EVENT_WATERMARK,
        "projection_matches_events": True,
        "statuses": statuses,
        "revisions": revisions,
        "operational_validation_receipt_cids": receipt_cids,
        "migration_digest": expected_digest,
        "migration_evidence_id": expected_evidence_id,
        "plan_migration_event_id": plan_event["event_id"],
        "migration_evidence_event_id": evidence_event["event_id"],
        "semantic_authority_digest": semantic_digest,
        "frozen_base_authority_digest": frozen_digest,
        "append_surface_digest": append_digest,
        "coordination_projection_digest": coordination_projection[
            "projection_root"
        ],
        "coordination_event_count": coordination_event_count,
        "control_store_sha256": initial_control_sha256,
        "coordination_store_sha256": initial_coordination_sha256,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "coordination_semantic_changes": 0,
    }


def _verify_m10_store_pair(
    root: Path,
    control: Path,
    coordination: Path,
    prior_control: Path,
    prior_coordination: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Verify M10 only through exact disposable copies of both stores."""

    _assert_offline(control)
    initial_control = _stable_regular_sha256(
        control,
        root=root,
        noun="published M10 control store",
        required_link_count=1,
    )
    initial_coordination = _stable_regular_sha256(
        coordination,
        root=root,
        noun="published M10 coordination store",
        required_link_count=1,
    )
    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m10-verify-",
        dir="/tmp",
    ) as temp_dir:
        control_copy = Path(temp_dir) / "control.duckdb"
        coordination_copy = Path(temp_dir) / "control.coordination.duckdb"
        prior_control_copy = Path(temp_dir) / "prior-control.duckdb"
        prior_coordination_copy = Path(temp_dir) / "prior-control.coordination.duckdb"
        shutil.copyfile(control, control_copy)
        shutil.copyfile(coordination, coordination_copy)
        shutil.copyfile(prior_control, prior_control_copy)
        shutil.copyfile(prior_coordination, prior_coordination_copy)
        report = _verify_m10_store_pair_copy(
            control_copy,
            coordination_copy,
            prior_control_copy,
            prior_coordination_copy,
            population,
            config,
            validation_digest,
        )
    if (
        _stable_regular_sha256(
            control,
            root=root,
            noun="published M10 control store",
            required_link_count=1,
        )
        != initial_control
        or _stable_regular_sha256(
            coordination,
            root=root,
            noun="published M10 coordination store",
            required_link_count=1,
        )
        != initial_coordination
    ):
        raise MaterializationError("M10 verification changed published bytes")
    return {
        **report,
        "control_store_sha256": initial_control[0],
        "coordination_store_sha256": initial_coordination[0],
    }


def _verify_m7_store_copy(
    path: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Verify the two-event M7 source migration and unchanged task authority."""

    _assert_offline(path)
    authority = _m7_source_repair_authority(population, config)
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    schema = verify_datasets_authoritative_operational_schema(path)
    if schema.get("valid") is not True:
        raise MigrationRequired("M7 operational schema does not verify")
    frozen_base_authority_digest = _frozen_base_authority_digest(path)
    if (
        frozen_base_authority_digest
        != authority["prior_frozen_base_authority_digest"]
    ):
        raise MigrationRequired("M7 changed a frozen base authority table")
    source = DatabaseTaskSource(
        path,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
    )
    try:
        snap = source.snapshot()
        statuses, revisions, receipt_cids = _verify_m6_task_projection(
            source,
            population,
        )
        if (
            snap.task_count != 45
            or snap.goal_count != 29
            or snap.plan_root_cid != population["plan_root_cid"]
            or int(snap.event_cursor) != _M7_TARGET_EVENT_WATERMARK
            or str(snap.projection_cid) != _M7_EXPECTED_PROJECTION_CID
        ):
            raise MigrationRequired("M7 snapshot identity differs")
        for expected in population["objectives"]:
            goal = source.get_goal(str(expected["goal_cid"]))
            if (
                goal is None
                or str(goal.get("goal_alias")) != expected["goal_id"]
                or (goal.get("body") or {}).get("definition_cid")
                != expected["definition_cid"]
                or (goal.get("body") or {}).get("definition")
                != expected["definition"]
            ):
                raise MigrationRequired(
                    f"M7 goal definition differs: {expected['goal_id']}"
                )
        plan = source.plans.get(str(population["plan_root_cid"]))
        if plan is None or int(plan.get("revision") or 0) != _M7_TARGET_PLAN_REVISION:
            raise MigrationRequired("M7 plan revision is missing")

        expected_migration_body = _m7_migration_body(
            population,
            config,
            validation_digest,
        )
        expected_migration_digest = _identity(expected_migration_body)
        plan_delta = _m7_migration_plan_delta(population, config)
        migration = population["migration_inventory"]
        with source.intent._connection(write=False) as connection:
            prefix, prefix_count = _event_prefix_digest(
                connection,
                int(authority["prior_event_watermark"]),
            )
            if (
                prefix != authority["prior_event_prefix_sha256"]
                or prefix_count != int(authority["prior_event_watermark"])
            ):
                raise MigrationRequired("M7 changed the frozen M6 event prefix")
            if int(_table_count(connection, "task_revisions")) != 49:
                raise MigrationRequired("M7 changed the frozen task revision set")
            completion = [
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    "SELECT receipt_cid, task_cid FROM completion_receipts "
                    "ORDER BY receipt_cid"
                ).fetchall()
            ]
            if completion != [
                (
                    migration["prior_completion_receipt_cid"],
                    migration["prior_task_cids"]["SAWM-000"],
                )
            ]:
                raise MigrationRequired("M7 changed completion evidence")
            evidence_rows = connection.execute(
                "SELECT evidence_id, parent_evidence_id, task_cid, "
                "evidence_kind, digest, created_at, body_json "
                "FROM evidence_nodes WHERE task_cid = ? AND evidence_kind = ? "
                "ORDER BY created_at",
                [
                    migration["prior_task_cids"]["SAWM-000"],
                    "operator_control_plane_source_migration",
                ],
            ).fetchall()
            current = []
            for row in evidence_rows:
                body = json.loads(str(row[6]))
                if body.get("migration_revision") == _M7_MIGRATION_REVISION:
                    current.append(
                        {
                            "evidence_id": str(row[0]),
                            "parent_evidence_id": str(row[1]),
                            "task_cid": str(row[2]),
                            "evidence_kind": str(row[3]),
                            "digest": str(row[4]),
                            "body": body,
                            "created_at": str(row[5]),
                        }
                    )
            if len(current) != 1:
                raise MigrationRequired("M7 source migration evidence is ambiguous")
            evidence_row = current[0]
            if (
                evidence_row["body"] != expected_migration_body
                or evidence_row["digest"] != expected_migration_digest
                or evidence_row["parent_evidence_id"]
                or evidence_row["task_cid"]
                != migration["prior_task_cids"]["SAWM-000"]
                or evidence_row["evidence_kind"]
                != "operator_control_plane_source_migration"
            ):
                raise MigrationRequired("M7 source migration evidence differs")
            expected_evidence_id = content_identity(
                {
                    "task_cid": migration["prior_task_cids"]["SAWM-000"],
                    "evidence_kind": "operator_control_plane_source_migration",
                    "digest": expected_migration_digest,
                    "body": expected_migration_body,
                }
            )
            if evidence_row["evidence_id"] != expected_evidence_id:
                raise MigrationRequired("M7 migration evidence identity differs")
            append_counts = {
                table: int(_table_count(connection, table))
                for table in (
                    "plans",
                    "plan_revisions",
                    "evidence_nodes",
                    "domain_events",
                )
            }
            if append_counts != {
                "plans": 1,
                "plan_revisions": 8,
                "evidence_nodes": 9,
                "domain_events": 170,
            }:
                raise MigrationRequired("M7 append-surface counts differ")
            if (
                _m7_prior_append_surface_digest_on(
                    connection,
                    expected_evidence_id=expected_evidence_id,
                )
                != authority["prior_append_surface_digest"]
            ):
                raise MigrationRequired("M7 changed an append-surface prefix")

            plan_rows = connection.execute(
                "SELECT revision, body_json, recorded_at FROM plan_revisions "
                "WHERE plan_cid = ? AND revision IN (?, ?) ORDER BY revision",
                [
                    population["plan_root_cid"],
                    authority["prior_plan_revision"],
                    authority["target_plan_revision"],
                ],
            ).fetchall()
            if [int(row[0]) for row in plan_rows] != [7, 8]:
                raise MigrationRequired("M7 plan revision pair differs")
            prior_plan_body = json.loads(str(plan_rows[0][1]))
            expected_plan_body = {
                **prior_plan_body,
                "current_source_binding_cid": population["source_binding"][
                    "source_binding_cid"
                ],
                "source_migration_revision": _M7_MIGRATION_REVISION,
                "source_migration_digest": expected_migration_digest,
                "supersession_mode": _M7_SUPERSESSION_MODE,
                "last_delta": plan_delta,
            }
            observed_plan_body = json.loads(str(plan_rows[1][1]))
            plan_recorded_at = str(plan_rows[1][2])
            if (
                observed_plan_body != expected_plan_body
                or plan.get("body") != expected_plan_body
                or str(plan.get("updated_at") or "") != plan_recorded_at
            ):
                raise MigrationRequired("M7 plan body differs")

            suffix = connection.execute(
                "SELECT event_id, stream_id, sequence, global_sequence, "
                "event_type, task_cid, attempt_id, session_id, recorded_at, "
                "body_json FROM domain_events WHERE global_sequence IN (169, 170) "
                "ORDER BY global_sequence"
            ).fetchall()
            if len(suffix) != _M7_EVENT_SUFFIX_LENGTH:
                raise MigrationRequired("M7 event suffix length differs")
            events: list[dict[str, Any]] = []
            for row in suffix:
                event = {
                    "event_id": str(row[0]),
                    "stream_id": str(row[1]),
                    "sequence": int(row[2]),
                    "global_sequence": int(row[3]),
                    "event_type": str(row[4]),
                    "task_cid": str(row[5]),
                    "attempt_id": str(row[6]),
                    "session_id": str(row[7]),
                    "recorded_at": str(row[8]),
                    "body": json.loads(str(row[9])),
                }
                envelope = event["body"]
                if (
                    content_identity(
                        {
                            "stream_id": event["stream_id"],
                            "sequence": event["sequence"],
                            "global_sequence": event["global_sequence"],
                            "event_type": event["event_type"],
                            "body": envelope,
                        }
                    )
                    != event["event_id"]
                    or event["stream_id"] != "stream:intent"
                    or event["session_id"] != "session:intent"
                    or event["attempt_id"]
                    or event["sequence"] != event["global_sequence"]
                    or envelope.get("schema")
                    != "ipfs_accelerate_py/agent-supervisor/intent-event@1"
                    or envelope.get("event_type") != event["event_type"]
                    or envelope.get("recorded_at") != event["recorded_at"]
                    or envelope.get("owner_id") != "sawm-r2-source-migrator"
                ):
                    raise MigrationRequired("M7 event identity differs")
                events.append(event)
            plan_event, evidence_event = events
            expected_plan_inner = {
                "plan_cid": population["plan_root_cid"],
                "goal_cid": migration["prior_goal_cids"][ROOT_GOAL],
                "plan_alias": REVISION,
                "status": "active",
                "revision": _M7_TARGET_PLAN_REVISION,
                "body": expected_plan_body,
                "delta": plan_delta,
                "recorded_at": plan_recorded_at,
            }
            if (
                plan_event["event_type"] != "intent.plan_revision_appended"
                or plan_event["task_cid"]
                or plan_event["body"].get("subject_id")
                != population["plan_root_cid"]
                or plan_event["body"].get("body") != expected_plan_inner
            ):
                raise MigrationRequired("M7 plan event differs")
            expected_evidence_inner = {
                "evidence_id": evidence_row["evidence_id"],
                "parent_evidence_id": "",
                "task_cid": migration["prior_task_cids"]["SAWM-000"],
                "evidence_kind": "operator_control_plane_source_migration",
                "digest": expected_migration_digest,
                "body": expected_migration_body,
                "created_at": evidence_row["created_at"],
                "revision": 0,
            }
            if (
                evidence_event["event_type"] != "intent.evidence_recorded"
                or evidence_event["task_cid"]
                != migration["prior_task_cids"]["SAWM-000"]
                or evidence_event["body"].get("subject_id")
                != evidence_row["evidence_id"]
                or evidence_event["body"].get("body")
                != expected_evidence_inner
            ):
                raise MigrationRequired("M7 evidence event differs")
    finally:
        source.close()

    with tempfile.TemporaryDirectory(prefix="sawm-r2-m7-replay-", dir="/tmp") as temp_dir:
        replay_path = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(path, replay_path)
        replay = DatabaseTaskSource(
            replay_path,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        try:
            if not replay.projection_matches_events():
                raise MigrationRequired("M7 replay projection differs")
        finally:
            replay.close()
    return {
        "valid": True,
        "task_count": 45,
        "goal_count": 29,
        "projection_cid": _M7_EXPECTED_PROJECTION_CID,
        "event_watermark": _M7_TARGET_EVENT_WATERMARK,
        "projection_matches_events": True,
        "statuses": statuses,
        "revisions": revisions,
        "operational_validation_receipt_cids": receipt_cids,
        "migration_digest": expected_migration_digest,
        "migration_evidence_id": evidence_row["evidence_id"],
        "plan_migration_event_id": plan_event["event_id"],
        "migration_evidence_event_id": evidence_event["event_id"],
        "migration_event_watermark": _M7_TARGET_EVENT_WATERMARK,
        "target_event_watermark": _M7_TARGET_EVENT_WATERMARK,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "frozen_base_authority_digest": frozen_base_authority_digest,
    }


def _verify_m7_store(
    path: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Verify M7 only through an exact disposable copy of accepted bytes."""

    _assert_offline(path)
    initial_hash = _store_sha256(path)
    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m7-verify-",
        dir="/tmp",
    ) as temp_dir:
        verification_copy = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(path, verification_copy)
        if _store_sha256(verification_copy) != initial_hash:
            raise MigrationRequired("M7 verification copy differs")
        report = _verify_m7_store_copy(
            verification_copy,
            population,
            config,
            validation_digest,
        )
    if _store_sha256(path) != initial_hash:
        raise MaterializationError("M7 verification changed accepted bytes")
    return report


def _verify_m8_store_copy(
    path: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Verify M8's two-event append and unchanged task/goal authority."""

    _assert_offline(path)
    authority = _m8_source_repair_authority(population, config)
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        verify_datasets_authoritative_operational_schema,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    schema = verify_datasets_authoritative_operational_schema(path)
    if schema.get("valid") is not True:
        raise MigrationRequired("M8 operational schema does not verify")
    semantic_authority_digest = _semantic_authority_digest(path)
    frozen_base_authority_digest = _frozen_base_authority_digest(path)
    if (
        semantic_authority_digest
        != authority["prior_semantic_authority_digest"]
        or frozen_base_authority_digest
        != authority["prior_frozen_base_authority_digest"]
    ):
        raise MigrationRequired("M8 changed a frozen authority table")

    source = DatabaseTaskSource(
        path,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
    )
    try:
        snap = source.snapshot()
        statuses, revisions, receipt_cids = _verify_m6_task_projection(
            source,
            population,
        )
        if (
            snap.task_count != 45
            or snap.goal_count != 29
            or snap.plan_root_cid != population["plan_root_cid"]
            or int(snap.event_cursor) != _M8_TARGET_EVENT_WATERMARK
            or str(snap.projection_cid) != _M8_EXPECTED_PROJECTION_CID
        ):
            raise MigrationRequired("M8 snapshot identity differs")
        for expected in population["objectives"]:
            observed = source.get_goal(str(expected["goal_cid"]))
            expected_body = {
                key: value
                for key, value in expected.items()
                if key
                not in {
                    "goal_cid",
                    "goal_id",
                    "goal_alias",
                    "title",
                    "status",
                    "ordinal",
                    "objective_id",
                }
            }
            if (
                observed is None
                or str(observed.get("goal_cid")) != expected["goal_cid"]
                or str(observed.get("goal_alias")) != expected["goal_id"]
                or str(observed.get("objective_id") or "")
                != str(expected.get("objective_id") or "")
                or str(observed.get("parent_goal_cid") or "")
                != str(expected.get("parent_goal_cid") or "")
                or int(observed.get("ordinal") or 0)
                != int(expected["ordinal"])
                or str(observed.get("title")) != expected["title"]
                or str(observed.get("status")) != expected["status"]
                or int(observed.get("revision") or 0) != 1
                or dict(observed.get("body") or {}) != expected_body
            ):
                raise MigrationRequired(
                    f"M8 goal contract differs: {expected['goal_id']}"
                )
        plan = source.plans.get(str(population["plan_root_cid"]))
        if plan is None or int(plan.get("revision") or 0) != _M8_TARGET_PLAN_REVISION:
            raise MigrationRequired("M8 plan revision is missing")

        expected_migration_body = _m8_migration_body(
            population,
            config,
            validation_digest,
        )
        expected_migration_digest = _identity(expected_migration_body)
        plan_delta = _m8_migration_plan_delta(population, config)
        migration = population["migration_inventory"]
        with source.intent._connection(write=False) as connection:
            prefix, prefix_count = _event_prefix_digest(
                connection,
                int(authority["prior_event_watermark"]),
            )
            if (
                prefix != authority["prior_event_prefix_sha256"]
                or prefix_count != int(authority["prior_event_watermark"])
            ):
                raise MigrationRequired("M8 changed the frozen M7 event prefix")
            if int(_table_count(connection, "task_revisions")) != 49:
                raise MigrationRequired("M8 changed the frozen task revision set")
            completion = _positional_rows(
                connection.execute(
                    "SELECT receipt_cid, task_cid FROM completion_receipts "
                    "ORDER BY receipt_cid"
                ).fetchall(),
                2,
            )
            if completion != [
                (
                    migration["prior_completion_receipt_cid"],
                    migration["prior_task_cids"]["SAWM-000"],
                )
            ]:
                raise MigrationRequired("M8 changed completion evidence")
            evidence_rows = connection.execute(
                "SELECT evidence_id, parent_evidence_id, task_cid, "
                "evidence_kind, digest, created_at, body_json "
                "FROM evidence_nodes WHERE task_cid = ? AND evidence_kind = ? "
                "ORDER BY created_at",
                [
                    migration["prior_task_cids"]["SAWM-000"],
                    "operator_control_plane_source_migration",
                ],
            ).fetchall()
            current = []
            for row in evidence_rows:
                body = json.loads(str(row[6]))
                if body.get("migration_revision") == _M8_MIGRATION_REVISION:
                    current.append(
                        {
                            "evidence_id": str(row[0]),
                            "parent_evidence_id": str(row[1]),
                            "task_cid": str(row[2]),
                            "evidence_kind": str(row[3]),
                            "digest": str(row[4]),
                            "body": body,
                            "created_at": str(row[5]),
                        }
                    )
            if len(current) != 1:
                raise MigrationRequired("M8 source migration evidence is ambiguous")
            evidence_row = current[0]
            if (
                evidence_row["body"] != expected_migration_body
                or evidence_row["digest"] != expected_migration_digest
                or evidence_row["parent_evidence_id"]
                or evidence_row["task_cid"]
                != migration["prior_task_cids"]["SAWM-000"]
                or evidence_row["evidence_kind"]
                != "operator_control_plane_source_migration"
            ):
                raise MigrationRequired("M8 source migration evidence differs")
            expected_evidence_id = content_identity(
                {
                    "task_cid": migration["prior_task_cids"]["SAWM-000"],
                    "evidence_kind": "operator_control_plane_source_migration",
                    "digest": expected_migration_digest,
                    "body": expected_migration_body,
                }
            )
            if evidence_row["evidence_id"] != expected_evidence_id:
                raise MigrationRequired("M8 migration evidence identity differs")
            append_counts = {
                table: int(_table_count(connection, table))
                for table in (
                    "plans",
                    "plan_revisions",
                    "evidence_nodes",
                    "domain_events",
                )
            }
            if append_counts != {
                "plans": 1,
                "plan_revisions": 9,
                "evidence_nodes": 10,
                "domain_events": 172,
            }:
                raise MigrationRequired("M8 append-surface counts differ")
            if (
                _m8_prior_append_surface_digest_on(
                    connection,
                    expected_evidence_id=expected_evidence_id,
                )
                != authority["prior_append_surface_digest"]
            ):
                raise MigrationRequired("M8 changed an append-surface prefix")

            plan_rows = connection.execute(
                "SELECT revision, body_json, recorded_at FROM plan_revisions "
                "WHERE plan_cid = ? AND revision IN (?, ?) ORDER BY revision",
                [
                    population["plan_root_cid"],
                    authority["prior_plan_revision"],
                    authority["target_plan_revision"],
                ],
            ).fetchall()
            if [int(row[0]) for row in plan_rows] != [8, 9]:
                raise MigrationRequired("M8 plan revision pair differs")
            prior_plan_body = json.loads(str(plan_rows[0][1]))
            expected_plan_body = {
                **prior_plan_body,
                "current_source_binding_cid": population["source_binding"][
                    "source_binding_cid"
                ],
                "source_migration_revision": _M8_MIGRATION_REVISION,
                "source_migration_digest": expected_migration_digest,
                "supersession_mode": _M8_SUPERSESSION_MODE,
                "last_delta": plan_delta,
            }
            observed_plan_body = json.loads(str(plan_rows[1][1]))
            plan_recorded_at = str(plan_rows[1][2])
            if (
                observed_plan_body != expected_plan_body
                or plan.get("body") != expected_plan_body
                or str(plan.get("updated_at") or "") != plan_recorded_at
            ):
                raise MigrationRequired("M8 plan body differs")

            suffix = connection.execute(
                "SELECT event_id, stream_id, sequence, global_sequence, "
                "event_type, task_cid, attempt_id, session_id, recorded_at, "
                "body_json FROM domain_events WHERE global_sequence IN (171, 172) "
                "ORDER BY global_sequence"
            ).fetchall()
            if len(suffix) != _M8_EVENT_SUFFIX_LENGTH:
                raise MigrationRequired("M8 event suffix length differs")
            events: list[dict[str, Any]] = []
            for row in suffix:
                event = {
                    "event_id": str(row[0]),
                    "stream_id": str(row[1]),
                    "sequence": int(row[2]),
                    "global_sequence": int(row[3]),
                    "event_type": str(row[4]),
                    "task_cid": str(row[5]),
                    "attempt_id": str(row[6]),
                    "session_id": str(row[7]),
                    "recorded_at": str(row[8]),
                    "body": json.loads(str(row[9])),
                }
                envelope = event["body"]
                if (
                    content_identity(
                        {
                            "stream_id": event["stream_id"],
                            "sequence": event["sequence"],
                            "global_sequence": event["global_sequence"],
                            "event_type": event["event_type"],
                            "body": envelope,
                        }
                    )
                    != event["event_id"]
                    or event["stream_id"] != "stream:intent"
                    or event["session_id"] != "session:intent"
                    or event["attempt_id"]
                    or event["sequence"] != event["global_sequence"]
                    or envelope.get("schema")
                    != "ipfs_accelerate_py/agent-supervisor/intent-event@1"
                    or envelope.get("event_type") != event["event_type"]
                    or envelope.get("recorded_at") != event["recorded_at"]
                    or envelope.get("owner_id") != "sawm-r2-source-migrator"
                ):
                    raise MigrationRequired("M8 event identity differs")
                events.append(event)
            plan_event, evidence_event = events
            expected_plan_inner = {
                "plan_cid": population["plan_root_cid"],
                "goal_cid": migration["prior_goal_cids"][ROOT_GOAL],
                "plan_alias": REVISION,
                "status": "active",
                "revision": _M8_TARGET_PLAN_REVISION,
                "body": expected_plan_body,
                "delta": plan_delta,
                "recorded_at": plan_recorded_at,
            }
            if (
                plan_event["event_type"] != "intent.plan_revision_appended"
                or plan_event["task_cid"]
                or plan_event["body"].get("subject_id")
                != population["plan_root_cid"]
                or plan_event["body"].get("body") != expected_plan_inner
            ):
                raise MigrationRequired("M8 plan event differs")
            expected_evidence_inner = {
                "evidence_id": evidence_row["evidence_id"],
                "parent_evidence_id": "",
                "task_cid": migration["prior_task_cids"]["SAWM-000"],
                "evidence_kind": "operator_control_plane_source_migration",
                "digest": expected_migration_digest,
                "body": expected_migration_body,
                "created_at": evidence_row["created_at"],
                "revision": 0,
            }
            if (
                evidence_event["event_type"] != "intent.evidence_recorded"
                or evidence_event["task_cid"]
                != migration["prior_task_cids"]["SAWM-000"]
                or evidence_event["body"].get("subject_id")
                != evidence_row["evidence_id"]
                or evidence_event["body"].get("body")
                != expected_evidence_inner
            ):
                raise MigrationRequired("M8 evidence event differs")
    finally:
        source.close()

    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m8-replay-",
        dir="/tmp",
    ) as temp_dir:
        replay_path = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(path, replay_path)
        replay = DatabaseTaskSource(
            replay_path,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        try:
            if not replay.projection_matches_events():
                raise MigrationRequired("M8 replay projection differs")
        finally:
            replay.close()
    return {
        "valid": True,
        "task_count": 45,
        "goal_count": 29,
        "projection_cid": _M8_EXPECTED_PROJECTION_CID,
        "event_watermark": _M8_TARGET_EVENT_WATERMARK,
        "projection_matches_events": True,
        "statuses": statuses,
        "revisions": revisions,
        "operational_validation_receipt_cids": receipt_cids,
        "migration_digest": expected_migration_digest,
        "migration_evidence_id": evidence_row["evidence_id"],
        "plan_migration_event_id": plan_event["event_id"],
        "migration_evidence_event_id": evidence_event["event_id"],
        "migration_event_watermark": _M8_TARGET_EVENT_WATERMARK,
        "target_event_watermark": _M8_TARGET_EVENT_WATERMARK,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "semantic_authority_digest": semantic_authority_digest,
        "frozen_base_authority_digest": frozen_base_authority_digest,
    }


def _verify_m8_store(
    path: Path,
    population: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_digest: str,
) -> dict[str, Any]:
    """Verify M8 only through an exact disposable copy of accepted bytes."""

    _assert_offline(path)
    initial_hash = _store_sha256(path)
    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m8-verify-",
        dir="/tmp",
    ) as temp_dir:
        verification_copy = Path(temp_dir) / "control.duckdb"
        shutil.copyfile(path, verification_copy)
        if _store_sha256(verification_copy) != initial_hash:
            raise MigrationRequired("M8 verification copy differs")
        report = _verify_m8_store_copy(
            verification_copy,
            population,
            config,
            validation_digest,
        )
    if _store_sha256(path) != initial_hash:
        raise MaterializationError("M8 verification changed accepted bytes")
    return report


def _authority_table_digest_on(
    connection: Any,
    tables: Sequence[str],
) -> str:
    """Hash complete rows for a closed, ordered base-table set."""

    import datetime

    def jsonable(value: Any) -> Any:
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
            return {"iso8601": value.isoformat()}
        return value

    projection: dict[str, Any] = {}
    for table in tables:
        columns = connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? "
            "ORDER BY ordinal_position",
            [table],
        ).fetchall()
        if not columns:
            raise MigrationRequired(
                f"source-only authority table is missing: {table}"
            )
        order = ", ".join(str(index) for index in range(1, len(columns) + 1))
        rows = connection.execute(
            f'SELECT * FROM "{table}" ORDER BY {order}'
        ).fetchall()
        projection[table] = [
            [jsonable(row[index]) for index in range(len(columns))]
            for row in rows
        ]
    return _identity(projection)


def _main_catalog_digest_on(connection: Any) -> str:
    """Hash main-schema table, view, column, and index definitions."""

    queries = {
        "tables": (
            "SELECT schema_name, table_name, column_count, sql "
            "FROM duckdb_tables() WHERE database_name = current_database() "
            "ORDER BY schema_name, table_name"
        ),
        "views": (
            "SELECT schema_name, view_name, column_count, sql "
            "FROM duckdb_views() WHERE database_name = current_database() "
            "ORDER BY schema_name, view_name"
        ),
        "indexes": (
            "SELECT schema_name, table_name, index_name, is_unique, "
            "is_primary, expressions, sql FROM duckdb_indexes() "
            "WHERE database_name = current_database() "
            "ORDER BY schema_name, table_name, index_name"
        ),
        "columns": (
            "SELECT table_schema, table_name, column_name, ordinal_position, "
            "column_default, is_nullable, data_type "
            "FROM information_schema.columns "
            "WHERE table_catalog = current_database() "
            "ORDER BY table_schema, table_name, ordinal_position"
        ),
    }
    return _identity(
        {
            name: [list(row) for row in connection.execute(sql).fetchall()]
            for name, sql in queries.items()
        }
    )


def _main_table_names(connection: Any) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT table_name FROM duckdb_tables() "
        "WHERE database_name = current_database() AND schema_name = 'main' "
        "ORDER BY table_name"
    ).fetchall()
    names = tuple(str(row[0]) for row in rows)
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names):
        raise MigrationRequired("control database has a non-closed table name")
    return names


def _semantic_authority_digest_on(connection: Any) -> str:
    """Hash semantic authority rows on a local or live read-only connection."""

    return _authority_table_digest_on(
        connection,
        (
            "objectives",
            "objective_revisions",
            "goals",
            "goal_edges",
            "tasks",
            "task_revisions",
            "task_dependencies",
            "task_outputs",
            "task_acceptance",
            "task_validations",
            "validation_results",
            "completion_receipts",
        ),
    )


def _frozen_base_authority_digest_on(connection: Any) -> str:
    """Hash every base table except the four exact M7 append surfaces."""

    allowed_append_tables = {
        "domain_events",
        "evidence_nodes",
        "plan_revisions",
        "plans",
    }
    tables = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        ).fetchall()
        if str(row[0]) not in allowed_append_tables
    )
    if not tables or allowed_append_tables.intersection(tables):
        raise MigrationRequired("M7 frozen base-table inventory is invalid")
    return _authority_table_digest_on(connection, tables)


def _m7_prior_append_surface_digest_on(
    connection: Any,
    *,
    expected_evidence_id: str,
) -> str:
    """Reconstruct and hash the exact M6 prefix of M7's append surfaces."""

    plan_rows = [
        tuple(row[index] for index in range(8))
        for row in connection.execute(
            "SELECT * FROM plans ORDER BY 1, 2, 3, 4, 5, 6, 7, 8"
        ).fetchall()
    ]
    revision_rows = [
        tuple(row[index] for index in range(4))
        for row in connection.execute(
            "SELECT * FROM plan_revisions WHERE revision <= 7 "
            "ORDER BY 1, 2, 3, 4"
        ).fetchall()
    ]
    revision_seven = [row for row in revision_rows if int(row[1]) == 7]
    if len(plan_rows) != 1 or len(revision_rows) != 7 or len(revision_seven) != 1:
        raise MigrationRequired("M7 cannot reconstruct the M6 plan prefix")
    current_plan = plan_rows[0]
    prior_revision = revision_seven[0]
    prior_plan = (
        current_plan[0],
        current_plan[1],
        current_plan[2],
        current_plan[3],
        current_plan[4],
        prior_revision[3],
        7,
        prior_revision[2],
    )
    evidence_rows = [
        tuple(row[index] for index in range(7))
        for row in connection.execute(
            "SELECT * FROM evidence_nodes ORDER BY 1, 2, 3, 4, 5, 6, 7"
        ).fetchall()
        if str(row[0]) != expected_evidence_id
    ]
    event_rows = [
        tuple(row[index] for index in range(10))
        for row in connection.execute(
            "SELECT * FROM domain_events WHERE global_sequence <= 168 "
            "ORDER BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
        ).fetchall()
    ]
    if len(evidence_rows) != 8 or len(event_rows) != 168:
        raise MigrationRequired("M7 append-surface prefix counts differ")
    projection = {
        "plans": [list(prior_plan)],
        "plan_revisions": [list(row) for row in revision_rows],
        "evidence_nodes": [list(row) for row in evidence_rows],
        "domain_events": [list(row) for row in event_rows],
    }
    return _identity(projection)


def _m8_prior_append_surface_digest_on(
    connection: Any,
    *,
    expected_evidence_id: str,
) -> str:
    """Reconstruct and hash the exact M7 prefix of M8's append surfaces."""

    plan_rows = [
        tuple(row[index] for index in range(8))
        for row in connection.execute(
            "SELECT * FROM plans ORDER BY 1, 2, 3, 4, 5, 6, 7, 8"
        ).fetchall()
    ]
    revision_rows = [
        tuple(row[index] for index in range(4))
        for row in connection.execute(
            "SELECT * FROM plan_revisions WHERE revision <= 8 "
            "ORDER BY 1, 2, 3, 4"
        ).fetchall()
    ]
    revision_eight = [row for row in revision_rows if int(row[1]) == 8]
    if (
        len(plan_rows) != 1
        or len(revision_rows) != 8
        or len(revision_eight) != 1
    ):
        raise MigrationRequired("M8 cannot reconstruct the M7 plan prefix")
    current_plan = plan_rows[0]
    prior_revision = revision_eight[0]
    prior_plan = (
        current_plan[0],
        current_plan[1],
        current_plan[2],
        current_plan[3],
        current_plan[4],
        prior_revision[3],
        8,
        prior_revision[2],
    )
    evidence_rows = [
        tuple(row[index] for index in range(7))
        for row in connection.execute(
            "SELECT * FROM evidence_nodes ORDER BY 1, 2, 3, 4, 5, 6, 7"
        ).fetchall()
        if str(row[0]) != expected_evidence_id
    ]
    event_rows = [
        tuple(row[index] for index in range(10))
        for row in connection.execute(
            "SELECT * FROM domain_events WHERE global_sequence <= 170 "
            "ORDER BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
        ).fetchall()
    ]
    if len(evidence_rows) != 9 or len(event_rows) != 170:
        raise MigrationRequired("M8 append-surface prefix counts differ")
    projection = {
        "plans": [list(prior_plan)],
        "plan_revisions": [list(row) for row in revision_rows],
        "evidence_nodes": [list(row) for row in evidence_rows],
        "domain_events": [list(row) for row in event_rows],
    }
    return _identity(projection)


def _semantic_authority_digest(path: Path) -> str:
    """Hash rows that an M7 source-only migration is forbidden to change."""

    import duckdb

    connection = duckdb.connect(str(path), read_only=True)
    try:
        return _semantic_authority_digest_on(connection)
    finally:
        connection.close()


def _frozen_base_authority_digest(path: Path) -> str:
    """Hash every M7-frozen base table through a read-only local handle."""

    import duckdb

    connection = duckdb.connect(str(path), read_only=True)
    try:
        return _frozen_base_authority_digest_on(connection)
    finally:
        connection.close()


def _append_surface_digest(path: Path) -> str:
    """Hash the exact four M6 surfaces authorized for the M7 append."""

    import duckdb

    connection = duckdb.connect(str(path), read_only=True)
    try:
        return _authority_table_digest_on(
            connection,
            ("plans", "plan_revisions", "evidence_nodes", "domain_events"),
        )
    finally:
        connection.close()


def _m7_validation_digest(root: Path, population: Mapping[str, Any]) -> str:
    dependency = _validator_report(
        root,
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
    )
    board = _validator_report(
        root,
        "scripts/validate_semantic_addressed_world_model_board.py",
    )
    return _identity(
        {
            "dependency": dependency,
            "board": board,
            "program_definition_cid": population["program_definition_cid"],
        }
    )


def _check_m7_materialized(
    repo_root: Path | str = REPO_ROOT,
    config_path: Path | str = CONFIG_PATH,
) -> dict[str, Any]:
    """Verify the historical M7 source-only successor and its receipt."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_json(config_file)
    population = build_population(root)
    _assert_committed_clean_source(root, population)
    authority = _m7_source_repair_authority(population, config)
    validation_digest = _m7_validation_digest(root, population)
    m5 = _verify_prior_store(root, config, population)
    m6 = _verify_frozen_m6_authority(root, config, population)
    target = (root / str(authority["target_store_id"])).resolve()
    if not target.is_file():
        raise MigrationRequired("M7 materialized authority is missing")
    verified = _verify_m7_store(
        target,
        population,
        config,
        validation_digest,
    )
    receipt = _ensure_m7_migration_receipt(
        root,
        target,
        population,
        config,
        verified,
        validation_digest,
    )
    return {
        "schema": SCHEMA,
        "valid": True,
        "action": "checked",
        "database_path": str(target),
        "program_definition_cid": population["program_definition_cid"],
        "validation_digest": validation_digest,
        "m5_authority": m5,
        "prior_authority": m6,
        "receipt": receipt,
        **verified,
    }


def _check_m8_materialized(
    repo_root: Path | str = REPO_ROOT,
    config_path: Path | str = CONFIG_PATH,
) -> dict[str, Any]:
    """Verify the current M8 source-only successor and its receipt."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_json(config_file)
    population = build_population(root)
    _assert_committed_clean_source(root, population)
    authority = _m8_source_repair_authority(population, config)
    validation_digest = _m7_validation_digest(root, population)
    m5 = _verify_prior_store(root, config, population)
    m6 = _verify_frozen_m6_authority(root, config, population)
    m7 = _verify_frozen_m7_authority(root, config, population)
    target = (root / str(authority["target_store_id"])).resolve()
    if not target.is_file():
        raise MigrationRequired("M8 materialized authority is missing")
    verified = _verify_m8_store(
        target,
        population,
        config,
        validation_digest,
    )
    receipt = _ensure_m8_migration_receipt(
        root,
        target,
        population,
        config,
        verified,
        validation_digest,
    )
    return {
        "schema": SCHEMA,
        "valid": True,
        "action": "checked",
        "database_path": str(target),
        "program_definition_cid": population["program_definition_cid"],
        "validation_digest": validation_digest,
        "m5_authority": m5,
        "m6_authority": m6,
        "prior_authority": m7,
        "receipt": receipt,
        **verified,
    }


def _check_m9_materialized(
    repo_root: Path | str = REPO_ROOT,
    config_path: Path | str = CONFIG_PATH,
) -> dict[str, Any]:
    """Verify the M9 control/coordinator successor and final pair marker."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_json(config_file)
    population = build_population(root)
    _assert_committed_clean_source(root, population)
    authority = _m9_live_recovery_authority(population, config)
    _assert_m9_source_delta(root, population, authority)
    control, coordination = _m9_target_paths(root, config, authority)
    if not os.path.lexists(control) or not os.path.lexists(coordination):
        raise MigrationRequired("M9 materialized store pair is missing")
    validation_digest = _m7_validation_digest(root, population)
    prior_control, prior_coordination = _assert_m9_prior_publication_anchor(
        root,
        authority,
    )
    prior = _verify_frozen_m8_live_authority(root, authority)
    verified = _verify_m9_store_pair(
        root,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    receipt = _ensure_m9_migration_receipt(
        root,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    return {
        "schema": SCHEMA,
        "valid": True,
        "action": "checked",
        "database_path": str(control),
        "coordination_path": str(coordination),
        "program_definition_cid": population["program_definition_cid"],
        "validation_digest": validation_digest,
        "prior_authority": prior,
        "receipt": receipt,
        **verified,
    }


def _check_m10_materialized(
    repo_root: Path | str = REPO_ROOT,
    config_path: Path | str = CONFIG_PATH,
) -> dict[str, Any]:
    """Verify M10 and its final pair marker without mutating authority."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_json(config_file)
    population = build_population(root)
    _assert_committed_clean_source(root, population)
    authority = _m10_live_projection_authority(population, config)
    _assert_m10_source_delta(root, population, authority)
    control, coordination = _m10_target_paths(root, config, authority)
    receipt_path = control.parent / "migration-receipt.json"
    if (
        not os.path.lexists(control)
        or not os.path.lexists(coordination)
        or not os.path.lexists(receipt_path)
    ):
        raise MigrationRequired("M10 materialized pair or final marker is missing")
    validation_digest = _m7_validation_digest(root, population)
    prior_control, prior_coordination = _assert_m10_prior_publication_anchor(
        root,
        authority,
    )
    prior = _verify_frozen_m9_live_authority(root, authority, population)
    verified = _verify_m10_store_pair(
        root,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    receipt = _verify_existing_m10_migration_receipt(
        root,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    return {
        "schema": SCHEMA,
        "valid": True,
        "action": "checked",
        "database_path": str(control),
        "coordination_path": str(coordination),
        "program_definition_cid": population["program_definition_cid"],
        "validation_digest": validation_digest,
        "prior_authority": prior,
        "receipt": receipt,
        **verified,
    }


def check_materialized(
    repo_root: Path | str = REPO_ROOT,
    config_path: Path | str = CONFIG_PATH,
) -> dict[str, Any]:
    """Verify the newest configured append-only materialization authority."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_json(config_file)
    if _m10_successor_configured(config):
        return _check_m10_materialized(root, config_file)
    if _m9_successor_configured(config):
        return _check_m9_materialized(root, config_file)
    if _m8_successor_configured(config):
        return _check_m8_materialized(root, config_file)
    return _check_m7_materialized(root, config_file)


def _materialize_m7(
    root: Path,
    config_file: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    population = build_population(root)
    _assert_committed_clean_source(root, population)
    authority = _m7_source_repair_authority(population, config)
    target = (root / str(config["database_program"]["store_id"])).resolve()
    if (
        str(config["database_program"]["store_id"])
        != authority["target_store_id"]
        or int(config["database_program"].get("store_generation") or 0)
        != _M7_TARGET_GENERATION
        or str(config["quack_owner"].get("database_path") or "")
        != authority["target_store_id"]
        or str(config["quack_owner"].get("store_id") or "")
        != authority["target_store_id"]
    ):
        raise MaterializationError("scheduler M7 target/generation binding differs")
    if not target.is_relative_to(root):
        raise MaterializationError("M7 target escapes the repository root")
    validation_digest = _m7_validation_digest(root, population)
    m5 = _verify_prior_store(root, config, population)
    m6 = _verify_frozen_m6_authority(root, config, population)
    prior_path = Path(m6["database_path"])
    if target.exists():
        verified = _verify_m7_store(
            target,
            population,
            config,
            validation_digest,
        )
        receipt = _ensure_m7_migration_receipt(
            root,
            target,
            population,
            config,
            verified,
            validation_digest,
        )
        return {
            "schema": SCHEMA,
            "valid": True,
            "action": "verified_existing_noop",
            "migration_required": False,
            "database_path": str(target),
            "program_definition_cid": population["program_definition_cid"],
            "validation_digest": validation_digest,
            "m5_authority": m5,
            "prior_authority": m6,
            "receipt": receipt,
            **verified,
        }

    _assert_fresh_successor_operational_state(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    preserved_stages = sorted(target.parent.glob(target.name + ".installing.*"))
    if preserved_stages:
        raise MaterializationError(
            "preserved M7 staging attempt requires inspection: "
            + ", ".join(str(item) for item in preserved_stages)
        )
    stage = target.with_name(
        target.name
        + f".installing.{os.getpid()}.{validation_digest[-12:]}"
    )
    shutil.copy2(prior_path, stage)
    if _store_sha256(stage) != authority["prior_control_store_sha256"]:
        raise MaterializationError("staged M6 copy differs before M7 migration")
    before_semantic_authority = _semantic_authority_digest(stage)
    before_frozen_base_authority = _frozen_base_authority_digest(stage)
    if (
        before_semantic_authority
        != authority["prior_semantic_authority_digest"]
        or before_frozen_base_authority
        != authority["prior_frozen_base_authority_digest"]
    ):
        raise MaterializationError("staged M6 authority digest differs")

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    source = DatabaseTaskSource(
        stage,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        owner_id="sawm-r2-source-migrator",
    )
    try:
        operator = source.get_task("SAWM-000")
        if operator is None or operator.status != "completed" or operator.revision != 2:
            raise MaterializationError("M7 preserved operator completion differs")
        migration_body = _m7_migration_body(
            population,
            config,
            validation_digest,
        )
        migration_digest = _identity(migration_body)
        plan_delta = _m7_migration_plan_delta(population, config)
        plan_receipt = source.plans.append_revision(
            plan_cid=str(population["plan_root_cid"]),
            expected_revision=int(authority["prior_plan_revision"]),
            body={
                "current_source_binding_cid": population["source_binding"][
                    "source_binding_cid"
                ],
                "source_migration_revision": _M7_MIGRATION_REVISION,
                "source_migration_digest": migration_digest,
                "supersession_mode": _M7_SUPERSESSION_MODE,
            },
            delta=plan_delta,
        )
        evidence = source.record_evidence(
            task_cid=operator.task_cid,
            evidence_kind="operator_control_plane_source_migration",
            digest=migration_digest,
            body=migration_body,
        )
        unchanged = source.get_task(operator.task_cid)
        if unchanged is None or unchanged.status != "completed" or unchanged.revision != 2:
            raise MaterializationError("M7 migration changed operator task state")
    finally:
        source.close()
    after_semantic_authority = _semantic_authority_digest(stage)
    after_frozen_base_authority = _frozen_base_authority_digest(stage)
    if (
        after_semantic_authority != before_semantic_authority
        or after_frozen_base_authority != before_frozen_base_authority
    ):
        raise MaterializationError(
            "M7 source-only migration changed frozen authority"
        )
    verified = _verify_m7_store(
        stage,
        population,
        config,
        validation_digest,
    )
    if (
        int(verified["event_watermark"]) != _M7_TARGET_EVENT_WATERMARK
        or verified["projection_cid"] != _M7_EXPECTED_PROJECTION_CID
    ):
        raise MaterializationError("M7 emitted an unexpected event/projection suffix")
    if _store_sha256(prior_path) != authority["prior_control_store_sha256"]:
        raise MaterializationError("M7 migration changed frozen M6 authority")

    # Close the source-tree TOCTOU window before making the staged authority
    # visible.  The staged database is intentionally retained for inspection
    # if this check fails.
    _assert_committed_clean_source(root, population)
    _assert_m7_source_delta(root, population, authority)
    with stage.open("rb") as staged_file:
        os.fsync(staged_file.fileno())

    lock = target.with_name(target.name + ".publish.lock")
    fd = None
    directory_fd = None
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        if target.exists():
            raise MigrationRequired(
                "another writer published M7; inspect it append-only"
            )
        os.link(stage, target)
        os.fsync(directory_fd)
        os.unlink(stage)
        os.fsync(directory_fd)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        if fd is not None:
            os.close(fd)
            lock.unlink(missing_ok=True)
    receipt = _ensure_m7_migration_receipt(
        root,
        target,
        population,
        config,
        verified,
        validation_digest,
    )
    history = _ducklake_projection(
        root,
        config,
        {
            "program_definition_cid": population["program_definition_cid"],
            "projection_cid": verified["projection_cid"],
        },
    )
    report = {
        "schema": SCHEMA,
        "valid": True,
        "action": "migrated_append_only",
        "migration_required": False,
        "database_path": str(target),
        "program_definition_cid": population["program_definition_cid"],
        "validation_digest": validation_digest,
        "m5_authority": m5,
        "prior_authority": m6,
        "plan_migration_event_id": plan_receipt.event_id,
        "migration_evidence_event_id": evidence.event_id,
        "migration_digest": migration_digest,
        "semantic_authority_digest_before": before_semantic_authority,
        "semantic_authority_digest_after": after_semantic_authority,
        "frozen_base_authority_digest_before": before_frozen_base_authority,
        "frozen_base_authority_digest_after": after_frozen_base_authority,
        "ducklake_history": history,
        "receipt": receipt,
        **verified,
    }
    return report


def _materialize_m10(
    root: Path,
    config_file: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Append source evidence and publish M10's unchanged-coordinator pair."""

    population = build_population(root)
    _assert_committed_clean_source(root, population)
    authority = _m10_live_projection_authority(population, config)
    _assert_m10_source_delta(root, population, authority)
    control, coordination = _m10_target_paths(root, config, authority)
    receipt_path = control.parent / "migration-receipt.json"
    validation_digest = _m7_validation_digest(root, population)
    prior_control, prior_coordination = _assert_m10_prior_publication_anchor(
        root,
        authority,
    )
    prior = _verify_frozen_m9_live_authority(root, authority, population)

    control_present = os.path.lexists(control)
    coordination_present = os.path.lexists(coordination)
    receipt_present = os.path.lexists(receipt_path)
    if control_present != coordination_present:
        raise MigrationRequired(
            "M10 has a partial published store pair without a final marker"
        )
    if receipt_present and not control_present:
        raise MigrationRequired("M10 has an orphaned pair commit marker")
    if control_present:
        verified = _verify_m10_store_pair(
            root,
            control,
            coordination,
            prior_control,
            prior_coordination,
            population,
            config,
            validation_digest,
        )
        receipt = _ensure_m10_migration_receipt(
            root,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
        history = _ducklake_projection(
            root,
            config,
            {
                "program_definition_cid": population["program_definition_cid"],
                "projection_cid": verified["projection_cid"],
            },
        )
        return {
            "schema": SCHEMA,
            "valid": True,
            "action": "verified_existing_noop",
            "migration_required": False,
            "database_path": str(control),
            "coordination_path": str(coordination),
            "program_definition_cid": population["program_definition_cid"],
            "validation_digest": validation_digest,
            "prior_authority": prior,
            "receipt": receipt,
            "ducklake_history": history,
            **verified,
        }

    prohibited = (
        control,
        coordination,
        receipt_path,
        control.with_name("control.execution.duckdb"),
        control.with_name("control.read-replica.duckdb"),
        control.with_name(f".{control.name}.state-owner.json"),
        control.parent / "state",
        control.parent / "events",
        control.parent / "registry",
        control.parent / "worktrees",
        control.parent / "merge-queue",
        control.parent / "quack-owner",
    )
    present = [str(path) for path in prohibited if os.path.lexists(path)]
    if present:
        raise MaterializationError(
            "M10 successor operational authority is not fresh: "
            + json.dumps(present, sort_keys=True, separators=(",", ":"))
        )
    control.parent.mkdir(parents=True, exist_ok=True)
    preserved_stages = sorted(control.parent.glob(".m10-installing.*"))
    if preserved_stages:
        raise MaterializationError(
            "preserved M10 staging attempt requires inspection: "
            + ", ".join(str(item) for item in preserved_stages)
        )
    stage_dir = control.parent / (
        f".m10-installing.{os.getpid()}.{validation_digest[-12:]}"
    )
    os.mkdir(stage_dir, 0o700)
    stage_control = stage_dir / "control.duckdb"
    stage_coordination = stage_dir / "control.coordination.duckdb"
    shutil.copy2(prior_control, stage_control)
    shutil.copy2(prior_coordination, stage_coordination)
    if (
        _store_sha256(stage_control) != authority["prior_control_store_sha256"]
        or stage_control.stat().st_size
        != int(authority["prior_control_store_size"])
        or _store_sha256(stage_coordination)
        != authority["prior_coordination_store_sha256"]
        or stage_coordination.stat().st_size
        != int(authority["prior_coordination_store_size"])
    ):
        raise MaterializationError("staged M9 pair differs before M10 migration")
    before_semantic = _semantic_authority_digest(stage_control)
    before_frozen = _frozen_base_authority_digest(stage_control)
    before_coordination_sha256 = _store_sha256(stage_coordination)
    if (
        before_semantic != authority["prior_semantic_authority_digest"]
        or before_frozen != authority["prior_frozen_base_authority_digest"]
    ):
        raise MaterializationError("staged M9 frozen authority digest differs")

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(
        stage_control,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        owner_id="sawm-r2-live-projection-migrator",
    )
    try:
        operator = source.get_task("SAWM-000")
        candidate = source.get_task("SAWM-001")
        if (
            operator is None
            or operator.status != "completed"
            or operator.revision != 2
            or candidate is None
            or candidate.status != "retrying"
            or candidate.revision != 10
            or dict(candidate.body).get("completion_receipt")
            != _m9_task_rearm_receipt()
        ):
            raise MaterializationError("staged M9 task authority differs")
        migration_body = _m10_migration_body(
            population,
            config,
            validation_digest,
        )
        migration_digest = _identity(migration_body)
        plan_receipt = source.plans.append_revision(
            plan_cid=str(population["plan_root_cid"]),
            expected_revision=int(authority["prior_plan_revision"]),
            body={
                "current_source_binding_cid": population["source_binding"][
                    "source_binding_cid"
                ],
                "source_migration_revision": _M10_MIGRATION_REVISION,
                "source_migration_digest": migration_digest,
                "supersession_mode": _M10_SUPERSESSION_MODE,
            },
            delta=_m10_migration_plan_delta(population, config),
        )
        evidence = source.record_evidence(
            task_cid=operator.task_cid,
            evidence_kind="operator_control_plane_source_migration",
            digest=migration_digest,
            body=migration_body,
        )
        unchanged_operator = source.get_task("SAWM-000")
        unchanged_candidate = source.get_task("SAWM-001")
        if (
            unchanged_operator is None
            or unchanged_operator.status != "completed"
            or unchanged_operator.revision != 2
            or unchanged_candidate is None
            or unchanged_candidate.status != "retrying"
            or unchanged_candidate.revision != 10
            or dict(unchanged_candidate.body).get("completion_receipt")
            != _m9_task_rearm_receipt()
        ):
            raise MaterializationError("M10 source append changed task authority")
    finally:
        source.close()

    after_semantic = _semantic_authority_digest(stage_control)
    after_frozen = _frozen_base_authority_digest(stage_control)
    if (
        after_semantic != before_semantic
        or after_frozen != before_frozen
        or _store_sha256(stage_coordination) != before_coordination_sha256
    ):
        raise MaterializationError("M10 source append changed frozen authority")
    verified = _verify_m10_store_pair_copy(
        stage_control,
        stage_coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    if (
        verified["projection_cid"] != _M10_EXPECTED_PROJECTION_CID
        or verified["coordination_projection_digest"]
        != _M9_COORDINATION_PROJECTION_DIGEST
        or verified["coordination_store_sha256"]
        != authority["prior_coordination_store_sha256"]
    ):
        raise MaterializationError("M10 staged pair has unexpected projections")
    _assert_m10_prior_publication_anchor(root, authority)
    _assert_committed_clean_source(root, population)
    _assert_m10_source_delta(root, population, authority)
    for stage in (stage_control, stage_coordination):
        with stage.open("rb") as staged_file:
            os.fsync(staged_file.fileno())

    bundle_lock = control.parent / ".m10-store-pair.publish.lock"
    descriptor = os.open(
        bundle_lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    created: list[tuple[Path, int, int]] = []
    stage_aliases_removed = False
    committed = False
    receipt: dict[str, Any] | None = None
    published_verified: dict[str, Any] | None = None
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MaterializationError("M10 bundle lock is not a regular file")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise MaterializationError(
                        "timed out acquiring the M10 pair publication lock"
                    ) from exc
                time.sleep(0.02)
        if any(
            os.path.lexists(path)
            for path in (control, coordination, receipt_path)
        ):
            raise MigrationRequired(
                "another writer published part of the M10 store pair"
            )
        _assert_committed_clean_source(root, population)
        _assert_m10_source_delta(root, population, authority)
        _assert_m10_prior_publication_anchor(root, authority)
        directory = os.open(
            control.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            stage_control_stat = stage_control.stat(follow_symlinks=False)
            os.link(stage_control, control, follow_symlinks=False)
            created.append(
                (control, stage_control_stat.st_dev, stage_control_stat.st_ino)
            )
            os.fsync(directory)
            stage_coordination_stat = stage_coordination.stat(
                follow_symlinks=False
            )
            os.link(stage_coordination, coordination, follow_symlinks=False)
            created.append(
                (
                    coordination,
                    stage_coordination_stat.st_dev,
                    stage_coordination_stat.st_ino,
                )
            )
            os.fsync(directory)
            _assert_committed_clean_source(root, population)
            _assert_m10_source_delta(root, population, authority)
            _assert_m10_prior_publication_anchor(root, authority)
            for path in (
                stage_control,
                stage_coordination,
                stage_dir / ".control.duckdb.intent.lock",
                stage_dir / ".control.duckdb.lock",
                stage_dir / ".control.coordination.duckdb.lock",
            ):
                if not os.path.lexists(path):
                    continue
                leaf_stat = os.lstat(path)
                if not stat.S_ISREG(leaf_stat.st_mode):
                    raise MaterializationError(
                        "M10 staging leaf changed before alias removal"
                    )
                os.unlink(path)
            stage_dir.rmdir()
            os.fsync(directory)
            stage_aliases_removed = True
            if (
                os.path.lexists(control.with_name("control.execution.duckdb"))
                or os.path.lexists(
                    control.with_name("control.read-replica.duckdb")
                )
            ):
                raise MigrationRequired(
                    "M10 mutable sidecar appeared before pair admission"
                )
            published_verified = _verify_m10_store_pair(
                root,
                control,
                coordination,
                prior_control,
                prior_coordination,
                population,
                config,
                validation_digest,
            )
            if (
                published_verified["control_store_sha256"]
                != verified["control_store_sha256"]
                or published_verified["coordination_store_sha256"]
                != verified["coordination_store_sha256"]
                or published_verified["migration_evidence_id"]
                != verified["migration_evidence_id"]
            ):
                raise MaterializationError("published M10 pair differs from staging")
            receipt = _ensure_m10_migration_receipt(
                root,
                control,
                coordination,
                population,
                config,
                published_verified,
                validation_digest,
                pair_lock_held=True,
            )
            committed = True
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if (
            not committed
            and not stage_aliases_removed
            and not os.path.lexists(receipt_path)
        ):
            rollback_directory = os.open(
                control.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                for published, expected_device, expected_inode in reversed(created):
                    if not os.path.lexists(published):
                        continue
                    published_stat = published.stat(follow_symlinks=False)
                    if (
                        published_stat.st_dev == expected_device
                        and published_stat.st_ino == expected_inode
                    ):
                        os.unlink(published)
                os.fsync(rollback_directory)
            finally:
                os.close(rollback_directory)
        raise
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)

    if not committed or receipt is None or published_verified is None:
        raise MaterializationError("M10 pair publication lacked its final marker")
    if not stage_aliases_removed:
        raise MaterializationError("M10 receipt preceded staging-alias removal")
    history = _ducklake_projection(
        root,
        config,
        {
            "program_definition_cid": population["program_definition_cid"],
            "projection_cid": published_verified["projection_cid"],
        },
    )
    return {
        "schema": SCHEMA,
        "valid": True,
        "action": "migrated_append_only_control_and_unchanged_coordination",
        "migration_required": False,
        "database_path": str(control),
        "coordination_path": str(coordination),
        "program_definition_cid": population["program_definition_cid"],
        "validation_digest": validation_digest,
        "prior_authority": prior,
        "plan_migration_event_id": plan_receipt.event_id,
        "migration_evidence_event_id": evidence.event_id,
        "migration_digest": migration_digest,
        "stage_cleanup_complete": True,
        "ducklake_history": history,
        "receipt": receipt,
        **published_verified,
    }


def _materialize_m9(
    root: Path,
    config_file: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Append, rearm, and publish the M9 control/coordinator pair."""

    population = build_population(root)
    _assert_committed_clean_source(root, population)
    authority = _m9_live_recovery_authority(population, config)
    _assert_m9_source_delta(root, population, authority)
    control, coordination = _m9_target_paths(root, config, authority)
    receipt_path = control.parent / "migration-receipt.json"
    validation_digest = _m7_validation_digest(root, population)
    prior_control, prior_coordination = _assert_m9_prior_publication_anchor(
        root,
        authority,
    )
    prior = _verify_frozen_m8_live_authority(root, authority)

    control_present = os.path.lexists(control)
    coordination_present = os.path.lexists(coordination)
    receipt_present = os.path.lexists(receipt_path)
    if control_present != coordination_present:
        raise MigrationRequired(
            "M9 has a partial published store pair without a commit marker"
        )
    if receipt_present and not control_present:
        raise MigrationRequired("M9 has an orphaned pair commit marker")
    if control_present:
        verified = _verify_m9_store_pair(
            root,
            control,
            coordination,
            prior_control,
            prior_coordination,
            population,
            config,
            validation_digest,
        )
        receipt = _ensure_m9_migration_receipt(
            root,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
        history = _ducklake_projection(
            root,
            config,
            {
                "program_definition_cid": population["program_definition_cid"],
                "projection_cid": verified["projection_cid"],
            },
        )
        return {
            "schema": SCHEMA,
            "valid": True,
            "action": "verified_existing_noop",
            "migration_required": False,
            "database_path": str(control),
            "coordination_path": str(coordination),
            "program_definition_cid": population["program_definition_cid"],
            "validation_digest": validation_digest,
            "prior_authority": prior,
            "receipt": receipt,
            "ducklake_history": history,
            **verified,
        }

    prohibited = (
        control,
        coordination,
        receipt_path,
        control.with_name("control.execution.duckdb"),
        control.with_name(f".{control.name}.state-owner.json"),
        control.parent / "state",
        control.parent / "events",
        control.parent / "registry",
        control.parent / "worktrees",
        control.parent / "merge-queue",
        control.parent / "quack-owner",
    )
    present = [str(path) for path in prohibited if os.path.lexists(path)]
    if present:
        raise MaterializationError(
            "M9 successor operational authority is not fresh: "
            + json.dumps(present, sort_keys=True, separators=(",", ":"))
        )
    control.parent.mkdir(parents=True, exist_ok=True)
    preserved_stages = sorted(control.parent.glob(".m9-installing.*"))
    if preserved_stages:
        raise MaterializationError(
            "preserved M9 staging attempt requires inspection: "
            + ", ".join(str(item) for item in preserved_stages)
        )
    stage_dir = control.parent / (
        f".m9-installing.{os.getpid()}.{validation_digest[-12:]}"
    )
    os.mkdir(stage_dir, 0o700)
    stage_control = stage_dir / "control.duckdb"
    stage_coordination = stage_dir / "control.coordination.duckdb"
    shutil.copy2(prior_control, stage_control)
    shutil.copy2(prior_coordination, stage_coordination)
    if (
        _store_sha256(stage_control) != authority["prior_control_store_sha256"]
        or stage_control.stat().st_size
        != int(authority["prior_control_store_size"])
        or _store_sha256(stage_coordination)
        != authority["prior_coordination_store_sha256"]
        or stage_coordination.stat().st_size
        != int(authority["prior_coordination_store_size"])
    ):
        raise MaterializationError("staged M8 store pair differs before M9 migration")

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(
        stage_control,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        owner_id="sawm-r2-runtime-recovery-migrator",
    )
    try:
        operator = source.get_task("SAWM-000")
        candidate = source.get_task("SAWM-001")
        if (
            operator is None
            or operator.status != "completed"
            or operator.revision != 2
            or candidate is None
            or candidate.status != "blocked"
            or candidate.revision != 9
            or dict(candidate.body).get("completion_receipt")
            != _M8_LIVE_FAILURE_RECEIPT
        ):
            raise MaterializationError("staged M8 task authority differs")
        migration_body = _m9_migration_body(
            population,
            config,
            validation_digest,
        )
        migration_digest = _identity(migration_body)
        plan_receipt = source.plans.append_revision(
            plan_cid=str(population["plan_root_cid"]),
            expected_revision=int(authority["prior_plan_revision"]),
            body={
                "current_source_binding_cid": population["source_binding"][
                    "source_binding_cid"
                ],
                "source_migration_revision": _M9_MIGRATION_REVISION,
                "source_migration_digest": migration_digest,
                "supersession_mode": _M9_SUPERSESSION_MODE,
            },
            delta=_m9_migration_plan_delta(population, config),
        )
        evidence = source.record_evidence(
            task_cid=operator.task_cid,
            evidence_kind="operator_control_plane_runtime_recovery",
            digest=migration_digest,
            body=migration_body,
        )
        task_cas = source.compare_and_set_status(
            _M8_LIVE_FAILURE_RECEIPT["task_cid"],
            9,
            "retrying",
            _m9_task_rearm_receipt(),
        )
        if (
            task_cas.changed is not True
            or task_cas.previous_status != "blocked"
            or task_cas.revision != 10
            or task_cas.event_cursor != _M9_TARGET_EVENT_WATERMARK
        ):
            raise MaterializationError("M9 operator task rearm CAS differs")
    finally:
        source.close()

    coordinator = DatabaseCoordinator(stage_coordination).open()
    try:
        coordination_rearm = coordinator.rearm_failed_task(
            failure_receipt=_M8_LIVE_FAILURE_RECEIPT,
            control_task_observation=task_cas.to_dict(),
            now_ms=_M9_COORDINATION_REARM_OBSERVED_AT_MS,
        )
    finally:
        coordinator.close()
    if (
        coordination_rearm.get("ready") is not True
        or coordination_rearm.get("replayed") is not False
        or coordination_rearm.get("rearm_id")
        != "baguqeerayovxeby66ijejx2upicgqg2kyil4bcf4larci52bau6av6qc5dga"
    ):
        raise MaterializationError("M9 coordinator rearm result differs")

    verified = _verify_m9_store_pair_copy(
        stage_control,
        stage_coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    if (
        verified["projection_cid"] != _M9_EXPECTED_PROJECTION_CID
        or verified["coordination_projection_digest"]
        != _M9_COORDINATION_PROJECTION_DIGEST
    ):
        raise MaterializationError("M9 staged pair has unexpected projections")
    _assert_m9_prior_publication_anchor(root, authority)
    _assert_committed_clean_source(root, population)
    _assert_m9_source_delta(root, population, authority)
    for stage in (stage_control, stage_coordination):
        with stage.open("rb") as staged_file:
            os.fsync(staged_file.fileno())

    bundle_lock = control.parent / ".m9-store-pair.publish.lock"
    descriptor = os.open(
        bundle_lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    created: list[tuple[Path, int, int]] = []
    stage_aliases_removed = False
    committed = False
    receipt: dict[str, Any] | None = None
    published_verified: dict[str, Any] | None = None
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise MaterializationError("M9 bundle lock is not a regular file")
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise MaterializationError(
                        "timed out acquiring the M9 pair publication lock"
                    ) from exc
                time.sleep(0.02)
        if any(
            os.path.lexists(path)
            for path in (control, coordination, receipt_path)
        ):
            raise MigrationRequired(
                "another writer published part of the M9 store pair"
            )
        _assert_committed_clean_source(root, population)
        _assert_m9_source_delta(root, population, authority)
        _assert_m9_prior_publication_anchor(root, authority)
        directory = os.open(
            control.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            stage_control_stat = stage_control.stat(follow_symlinks=False)
            os.link(stage_control, control, follow_symlinks=False)
            created.append(
                (control, stage_control_stat.st_dev, stage_control_stat.st_ino)
            )
            os.fsync(directory)
            stage_coordination_stat = stage_coordination.stat(
                follow_symlinks=False
            )
            os.link(stage_coordination, coordination, follow_symlinks=False)
            created.append(
                (
                    coordination,
                    stage_coordination_stat.st_dev,
                    stage_coordination_stat.st_ino,
                )
            )
            os.fsync(directory)
            _assert_committed_clean_source(root, population)
            _assert_m9_source_delta(root, population, authority)
            _assert_m9_prior_publication_anchor(root, authority)
            # Remove every mutable staging alias before final verification and
            # before the receipt can become the durable pair commit marker.
            for path in (
                stage_control,
                stage_coordination,
                stage_dir / ".control.duckdb.intent.lock",
                stage_dir / ".control.duckdb.lock",
                stage_dir / ".control.coordination.duckdb.lock",
            ):
                if not os.path.lexists(path):
                    continue
                leaf_stat = os.lstat(path)
                if not stat.S_ISREG(leaf_stat.st_mode):
                    raise MaterializationError(
                        "M9 staging leaf changed before alias removal"
                    )
                os.unlink(path)
            stage_dir.rmdir()
            os.fsync(directory)
            stage_aliases_removed = True
            if os.path.lexists(control.with_name("control.execution.duckdb")):
                raise MigrationRequired(
                    "M9 execution sidecar appeared before pair admission"
                )
            published_verified = _verify_m9_store_pair(
                root,
                control,
                coordination,
                prior_control,
                prior_coordination,
                population,
                config,
                validation_digest,
            )
            if (
                published_verified["control_store_sha256"]
                != verified["control_store_sha256"]
                or published_verified["coordination_store_sha256"]
                != verified["coordination_store_sha256"]
                or published_verified["coordination_rearm_event_id"]
                != verified["coordination_rearm_event_id"]
            ):
                raise MaterializationError("published M9 pair differs from staging")
            receipt = _ensure_m9_migration_receipt(
                root,
                control,
                coordination,
                population,
                config,
                published_verified,
                validation_digest,
                pair_lock_held=True,
            )
            committed = True
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if (
            not committed
            and not stage_aliases_removed
            and not os.path.lexists(receipt_path)
        ):
            rollback_directory = os.open(
                control.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                for published, expected_device, expected_inode in reversed(created):
                    if not os.path.lexists(published):
                        continue
                    published_stat = published.stat(follow_symlinks=False)
                    if (
                        published_stat.st_dev == expected_device
                        and published_stat.st_ino == expected_inode
                    ):
                        os.unlink(published)
                os.fsync(rollback_directory)
            finally:
                os.close(rollback_directory)
        raise
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)

    if not committed or receipt is None or published_verified is None:
        raise MaterializationError("M9 pair publication lacked its final marker")
    if not stage_aliases_removed:
        raise MaterializationError("M9 receipt preceded staging-alias removal")

    history = _ducklake_projection(
        root,
        config,
        {
            "program_definition_cid": population["program_definition_cid"],
            "projection_cid": published_verified["projection_cid"],
        },
    )
    return {
        "schema": SCHEMA,
        "valid": True,
        "action": "migrated_append_only_control_and_coordination",
        "migration_required": False,
        "database_path": str(control),
        "coordination_path": str(coordination),
        "program_definition_cid": population["program_definition_cid"],
        "validation_digest": validation_digest,
        "prior_authority": prior,
        "plan_migration_event_id": plan_receipt.event_id,
        "migration_evidence_event_id": evidence.event_id,
        "task_rearm_event_id": published_verified["task_rearm_event_id"],
        "coordination_rearm_event_id": published_verified[
            "coordination_rearm_event_id"
        ],
        "migration_digest": migration_digest,
        "stage_cleanup_complete": True,
        "ducklake_history": history,
        "receipt": receipt,
        **published_verified,
    }


def _materialize_m8(
    root: Path,
    config_file: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Append and durably publish the source-only M8 successor."""

    population = build_population(root)
    _assert_committed_clean_source(root, population)
    authority = _m8_source_repair_authority(population, config)
    target = (root / str(config["database_program"]["store_id"])).resolve()
    if (
        str(config["database_program"]["store_id"])
        != authority["target_store_id"]
        or int(config["database_program"].get("store_generation") or 0)
        != _M8_TARGET_GENERATION
        or str(config["quack_owner"].get("database_path") or "")
        != authority["target_store_id"]
        or str(config["quack_owner"].get("store_id") or "")
        != authority["target_store_id"]
    ):
        raise MaterializationError("scheduler M8 target/generation binding differs")
    if not target.is_relative_to(root):
        raise MaterializationError("M8 target escapes the repository root")
    validation_digest = _m7_validation_digest(root, population)
    m5 = _verify_prior_store(root, config, population)
    m6 = _verify_frozen_m6_authority(root, config, population)
    m7 = _verify_frozen_m7_authority(root, config, population)
    prior_path = Path(m7["database_path"])
    if target.exists():
        verified = _verify_m8_store(
            target,
            population,
            config,
            validation_digest,
        )
        receipt = _ensure_m8_migration_receipt(
            root,
            target,
            population,
            config,
            verified,
            validation_digest,
        )
        history = _ducklake_projection(
            root,
            config,
            {
                "program_definition_cid": population["program_definition_cid"],
                "projection_cid": verified["projection_cid"],
            },
        )
        return {
            "schema": SCHEMA,
            "valid": True,
            "action": "verified_existing_noop",
            "migration_required": False,
            "database_path": str(target),
            "program_definition_cid": population["program_definition_cid"],
            "validation_digest": validation_digest,
            "m5_authority": m5,
            "m6_authority": m6,
            "prior_authority": m7,
            "receipt": receipt,
            "ducklake_history": history,
            **verified,
        }

    _assert_fresh_successor_operational_state(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    preserved_stages = sorted(target.parent.glob(target.name + ".installing.*"))
    if preserved_stages:
        raise MaterializationError(
            "preserved M8 staging attempt requires inspection: "
            + ", ".join(str(item) for item in preserved_stages)
        )
    stage = target.with_name(
        target.name + f".installing.{os.getpid()}.{validation_digest[-12:]}"
    )
    shutil.copy2(prior_path, stage)
    if _store_sha256(stage) != authority["prior_control_store_sha256"]:
        raise MaterializationError("staged M7 copy differs before M8 migration")
    before_semantic_authority = _semantic_authority_digest(stage)
    before_frozen_base_authority = _frozen_base_authority_digest(stage)
    before_append_surface = _append_surface_digest(stage)
    if (
        before_semantic_authority
        != authority["prior_semantic_authority_digest"]
        or before_frozen_base_authority
        != authority["prior_frozen_base_authority_digest"]
        or before_append_surface != authority["prior_append_surface_digest"]
    ):
        raise MaterializationError("staged M7 authority digest differs")

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    source = DatabaseTaskSource(
        stage,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        owner_id="sawm-r2-source-migrator",
    )
    try:
        operator = source.get_task("SAWM-000")
        if operator is None or operator.status != "completed" or operator.revision != 2:
            raise MaterializationError("M8 preserved operator completion differs")
        migration_body = _m8_migration_body(
            population,
            config,
            validation_digest,
        )
        migration_digest = _identity(migration_body)
        plan_delta = _m8_migration_plan_delta(population, config)
        plan_receipt = source.plans.append_revision(
            plan_cid=str(population["plan_root_cid"]),
            expected_revision=int(authority["prior_plan_revision"]),
            body={
                "current_source_binding_cid": population["source_binding"][
                    "source_binding_cid"
                ],
                "source_migration_revision": _M8_MIGRATION_REVISION,
                "source_migration_digest": migration_digest,
                "supersession_mode": _M8_SUPERSESSION_MODE,
            },
            delta=plan_delta,
        )
        evidence = source.record_evidence(
            task_cid=operator.task_cid,
            evidence_kind="operator_control_plane_source_migration",
            digest=migration_digest,
            body=migration_body,
        )
        unchanged = source.get_task(operator.task_cid)
        if (
            unchanged is None
            or unchanged.status != "completed"
            or unchanged.revision != 2
        ):
            raise MaterializationError("M8 migration changed operator task state")
    finally:
        source.close()
    after_semantic_authority = _semantic_authority_digest(stage)
    after_frozen_base_authority = _frozen_base_authority_digest(stage)
    if (
        after_semantic_authority != before_semantic_authority
        or after_frozen_base_authority != before_frozen_base_authority
    ):
        raise MaterializationError(
            "M8 source-only migration changed frozen authority"
        )
    verified = _verify_m8_store(
        stage,
        population,
        config,
        validation_digest,
    )
    if (
        int(verified["event_watermark"]) != _M8_TARGET_EVENT_WATERMARK
        or verified["projection_cid"] != _M8_EXPECTED_PROJECTION_CID
    ):
        raise MaterializationError("M8 emitted an unexpected event/projection suffix")
    if _store_sha256(prior_path) != authority["prior_control_store_sha256"]:
        raise MaterializationError("M8 migration changed frozen M7 authority")

    # Rebind the exact committed source immediately before publication.  A
    # failed final check intentionally leaves the staged bytes for audit.
    _assert_committed_clean_source(root, population)
    _assert_m8_source_delta(root, population, authority)
    with stage.open("rb") as staged_file:
        os.fsync(staged_file.fileno())

    lock = target.with_name(target.name + ".publish.lock")
    fd = None
    directory_fd = None
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        if target.exists():
            raise MigrationRequired(
                "another writer published M8; inspect it append-only"
            )
        # The staged events bind the source snapshot captured in `population`.
        # Recompute that binding while holding the publication lock so a clean
        # checkout to another commit cannot publish stale source evidence.
        _assert_committed_clean_source(root, population)
        _assert_m8_source_delta(root, population, authority)
        _assert_m8_prior_publication_anchor(root, authority, prior_path)
        os.link(stage, target)
        try:
            # Keep the staging link until the same binding has been observed
            # after publication. On drift, remove only the hard link created
            # above and retain the staged bytes for inspection.
            _assert_committed_clean_source(root, population)
            _assert_m8_source_delta(root, population, authority)
            _assert_m8_prior_publication_anchor(root, authority, prior_path)
        except Exception:
            stage_stat = stage.stat(follow_symlinks=False)
            target_stat = target.stat(follow_symlinks=False)
            if (
                stage_stat.st_dev != target_stat.st_dev
                or stage_stat.st_ino != target_stat.st_ino
            ):
                raise MaterializationError(
                    "M8 publication target changed during source revalidation"
                )
            os.unlink(target)
            os.fsync(directory_fd)
            raise
        os.fsync(directory_fd)
        os.unlink(stage)
        os.fsync(directory_fd)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        if fd is not None:
            os.close(fd)
            lock.unlink(missing_ok=True)
    receipt = _ensure_m8_migration_receipt(
        root,
        target,
        population,
        config,
        verified,
        validation_digest,
    )
    history = _ducklake_projection(
        root,
        config,
        {
            "program_definition_cid": population["program_definition_cid"],
            "projection_cid": verified["projection_cid"],
        },
    )
    return {
        "schema": SCHEMA,
        "valid": True,
        "action": "migrated_append_only",
        "migration_required": False,
        "database_path": str(target),
        "program_definition_cid": population["program_definition_cid"],
        "validation_digest": validation_digest,
        "m5_authority": m5,
        "m6_authority": m6,
        "prior_authority": m7,
        "plan_migration_event_id": plan_receipt.event_id,
        "migration_evidence_event_id": evidence.event_id,
        "migration_digest": migration_digest,
        "semantic_authority_digest_before": before_semantic_authority,
        "semantic_authority_digest_after": after_semantic_authority,
        "frozen_base_authority_digest_before": before_frozen_base_authority,
        "frozen_base_authority_digest_after": after_frozen_base_authority,
        "prior_append_surface_digest": before_append_surface,
        "ducklake_history": history,
        "receipt": receipt,
        **verified,
    }


def materialize(repo_root: Path | str = REPO_ROOT, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_json(config_file)
    if _m10_successor_configured(config):
        return _materialize_m10(root, config_file, config)
    if _m9_successor_configured(config):
        return _materialize_m9(root, config_file, config)
    if _m8_successor_configured(config):
        return _materialize_m8(root, config_file, config)
    if isinstance(config.get("source_repair_materialization"), Mapping):
        return _materialize_m7(root, config_file, config)
    population = build_population(root)
    _assert_committed_clean_source(root, population)
    migration = population["migration_inventory"]
    target = (root / str(config["database_program"]["store_id"])).resolve()
    if str(config["database_program"]["store_id"]) != str(migration["target_store_id"]):
        raise MaterializationError("scheduler target differs from the sealed migration target")
    if (
        int(config["database_program"].get("store_generation") or 0)
        != int(migration["target_generation"])
        or str(config["quack_owner"].get("database_path") or "")
        != str(migration["target_store_id"])
        or str(config["quack_owner"].get("store_id") or "")
        != str(migration["target_store_id"])
    ):
        raise MaterializationError("scheduler successor generation/owner binding differs")
    if not target.is_relative_to(root):
        raise MaterializationError("migration target escapes the repository root")
    dependency = _validator_report(root, "scripts/validate_semantic_addressed_world_model_dependencies.py")
    board = _validator_report(root, "scripts/validate_semantic_addressed_world_model_board.py")
    validation_digest = _identity({"dependency": dependency, "board": board,
                                   "program_definition_cid": population["program_definition_cid"]})
    prior = _verify_prior_store(root, config, population)
    prior_path = Path(prior["database_path"])
    if target.exists():
        verified = _verify_store(
            target,
            population,
            require_operator_complete=True,
            require_migration=True,
            migration_config=config,
            expected_validation_digest=validation_digest,
        )
        receipt = _ensure_migration_receipt(
            root, target, population, verified, validation_digest
        )
        return {"schema": SCHEMA, "valid": True, "action": "verified_existing_noop",
                "migration_required": False, "database_path": str(target),
                "program_definition_cid": population["program_definition_cid"],
                "receipt": receipt, **verified}

    _assert_fresh_successor_operational_state(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    preserved_stages = sorted(target.parent.glob(target.name + ".installing.*"))
    if preserved_stages:
        raise MaterializationError(
            "preserved prior staging attempt requires inspection: "
            + ", ".join(str(item) for item in preserved_stages)
        )
    stage = target.with_name(target.name + f".installing.{os.getpid()}.{validation_digest[-12:]}")
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    # Preserve only the accepted canonical control store. M5 execution and
    # coordination sidecars remain immutable historical evidence and are
    # deliberately never copied into the successor authority.
    shutil.copy2(prior_path, stage)
    if _store_sha256(stage) != migration["prior_control_store_sha256"]:
        raise MaterializationError("staged prior authority copy differs before migration")
    source = DatabaseTaskSource(stage, install_schema=False,
                                repository_tree_id=str(population["repository_tree_id"]),
                                plan_root_cid=str(population["plan_root_cid"]),
                                owner_id="sawm-r2-source-migrator")
    try:
        operator = source.get_task("SAWM-000")
        if operator is None or operator.status != "completed" or operator.revision != 2:
            raise MaterializationError("preserved SAWM-000 completion authority differs")
        migration_body = _migration_body(population, config, validation_digest)
        migration_digest = _identity(migration_body)
        plan_delta = _migration_plan_delta(population)
        plan_receipt = source.plans.append_revision(
            plan_cid=str(population["plan_root_cid"]),
            expected_revision=int(migration["prior_plan_revision"]),
            body={
                "current_source_binding_cid": population["source_binding"]["source_binding_cid"],
                "source_migration_revision": migration["migration_revision"],
                "source_migration_digest": migration_digest,
                "supersession_mode": _M6_SUPERSESSION_MODE,
            },
            delta=plan_delta,
        )
        evidence = source.record_evidence(
            task_cid=operator.task_cid,
            evidence_kind="operator_control_plane_source_migration",
            digest=migration_digest,
            body=migration_body,
        )
        unchanged = source.get_task(operator.task_cid)
        if unchanged is None or unchanged.status != "completed" or unchanged.revision != 2:
            raise MaterializationError("source migration changed accepted SAWM-000 state")
        operational_event_ids: dict[str, str] = {}
        operational_receipts: dict[str, dict[str, Any]] = {}
        for alias in _operational_task_aliases():
            prior_task = source.intent.get_task(alias)
            if prior_task is None:
                raise MaterializationError(
                    f"preserved operational task is missing: {alias}"
                )
            expected_status = "blocked" if alias == "SAWM-001" else "todo"
            expected_revision = 5 if alias == "SAWM-001" else 1
            if (
                prior_task["task_cid"] != migration["prior_task_cids"][alias]
                or prior_task["status"] != expected_status
                or int(prior_task["revision"]) != expected_revision
            ):
                raise MaterializationError(
                    f"preserved operational task precondition differs: {alias}"
                )
            prior_validations = tuple(
                tuple(str(part) for part in item.get("argv") or ())
                for item in prior_task["validations"]
            )
            operational_validations, replacement_count = (
                _operational_validation_commands(
                    prior_validations,
                    task_alias=alias,
                )
            )
            operational_receipt = _operational_validation_receipt(
                population,
                task_alias=alias,
                task_cid=str(prior_task["task_cid"]),
                expected_status=expected_status,
                expected_revision=expected_revision,
                prior_validations=prior_validations,
                operational_validations=operational_validations,
                replacement_count=replacement_count,
            )
            upsert = source.intent.upsert_task(
                task_cid=str(prior_task["task_cid"]),
                task_alias=str(prior_task["task_alias"]),
                goal_cid=str(prior_task["goal_cid"]),
                ordinal=int(prior_task["ordinal"]),
                status=str(prior_task["status"]),
                priority=str(prior_task["priority"]),
                plan_cid=str(prior_task["plan_cid"]),
                objective_id=str(prior_task["objective_id"]),
                body={
                    **dict(prior_task["body"]),
                    "operational_validation_revision": operational_receipt,
                },
                identity=dict(prior_task["identity"]),
                expected_revision=expected_revision,
                dependencies=list(prior_task["dependencies"]),
                outputs=[dict(item["effect"]) for item in prior_task["outputs"]],
                acceptance=[
                    dict(item["evidence_policy"])
                    for item in prior_task["acceptance"]
                ],
                validations=[
                    {
                        **dict(prior["policy"]),
                        "argv": list(argv),
                    }
                    for prior, argv in zip(
                        prior_task["validations"],
                        operational_validations,
                        strict=True,
                    )
                ],
            )
            if (
                not upsert.changed
                or upsert.event_type != "intent.task_upserted"
                or int(upsert.revision) != expected_revision + 1
            ):
                raise MaterializationError(
                    f"operational validation revision was not admitted: {alias}"
                )
            operational_event_ids[alias] = upsert.event_id
            operational_receipts[alias] = operational_receipt
        candidate = source.get_task("SAWM-001")
        if (
            candidate is None
            or candidate.status != "blocked"
            or candidate.revision != 6
            or candidate.task_cid != migration["prior_task_cids"]["SAWM-001"]
        ):
            raise MaterializationError(
                "operationally revised SAWM-001 recovery precondition differs"
            )
        recovery_receipt = _task_recovery_receipt(population)
        recovery = source.compare_and_set_status(
            candidate.task_cid,
            expected_revision=6,
            status="todo",
            receipt=recovery_receipt,
        )
        if (
            not recovery.changed
            or recovery.previous_status != "blocked"
            or recovery.task.status != "todo"
            or recovery.task.revision != 7
        ):
            raise MaterializationError("operator SAWM-001 recovery CAS was not admitted")
    finally:
        source.close()
    verified = _verify_store(
        stage,
        population,
        require_operator_complete=True,
        require_migration=True,
        migration_config=config,
        expected_validation_digest=validation_digest,
    )
    expected_watermark = (
        int(migration["prior_event_watermark"]) + _M6_EVENT_SUFFIX_LENGTH
    )
    if int(verified["event_watermark"]) != expected_watermark:
        raise MaterializationError(
            f"source migration emitted an unexpected event suffix: {verified['event_watermark']}"
        )
    if _store_sha256(prior_path) != migration["prior_control_store_sha256"]:
        raise MaterializationError("source migration changed the preserved prior authority")
    lock = target.with_name(target.name + ".publish.lock")
    fd = None
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        if target.exists():
            raise MigrationRequired("another writer published a control store; inspect it append-only")
        os.link(stage, target)
        os.unlink(stage)
    finally:
        if fd is not None:
            os.close(fd)
            lock.unlink(missing_ok=True)
    receipt = _ensure_migration_receipt(
        root, target, population, verified, validation_digest
    )
    history = _ducklake_projection(root, config, {"program_definition_cid": population["program_definition_cid"],
                                                   "projection_cid": verified["projection_cid"]})
    report = {"schema": SCHEMA, "valid": True, "action": "migrated_append_only",
            "migration_required": False, "database_path": str(target),
            "program_definition_cid": population["program_definition_cid"],
            "validation_digest": validation_digest, "prior_authority": prior,
            "plan_migration_event_id": plan_receipt.event_id,
            "migration_evidence_event_id": evidence.event_id,
            "operational_validation_event_ids": operational_event_ids,
            "operational_validation_receipts": operational_receipts,
            "task_recovery_event_id": recovery.receipt_cid,
            "task_recovery_receipt": recovery_receipt,
            "migration_digest": migration_digest,
            "ducklake_history": history, **verified}
    return {**report, "receipt": receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("materialize", "render", "check"), default="materialize")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        population = build_population(args.repo_root)
        if args.action == "render":
            report: dict[str, Any] = {"valid": True, **population}
        elif args.action == "check":
            root = Path(args.repo_root).resolve()
            config = _load_json(args.config if args.config.is_absolute() else root / args.config)
            if (
                "source_repair_successor_materialization" in config
                or isinstance(config.get("source_repair_materialization"), Mapping)
            ):
                report = check_materialized(root, args.config)
                print(json.dumps(report, indent=2, sort_keys=True))
                return 0
            _assert_committed_clean_source(root, population)
            dependency = _validator_report(
                root, "scripts/validate_semantic_addressed_world_model_dependencies.py"
            )
            board = _validator_report(
                root, "scripts/validate_semantic_addressed_world_model_board.py"
            )
            validation_digest = _identity(
                {"dependency": dependency, "board": board,
                 "program_definition_cid": population["program_definition_cid"]}
            )
            prior = _verify_prior_store(root, config, population)
            path = root / str(config["database_program"]["store_id"])
            verified = _verify_store(
                path,
                population,
                require_operator_complete=True,
                require_migration=True,
                migration_config=config,
                expected_validation_digest=validation_digest,
            )
            receipt = _ensure_migration_receipt(
                root, path, population, verified, validation_digest
            )
            report = {"schema": SCHEMA, "valid": True, "action": "checked",
                      "database_path": str(path), "program_definition_cid": population["program_definition_cid"],
                      "prior_authority": prior, "receipt": receipt, **verified}
        else:
            report = materialize(args.repo_root, args.config)
    except MigrationRequired as exc:
        report = {"schema": SCHEMA, "valid": False, "action": "migration_required",
                  "migration_required": True, "error": str(exc)}
    except Exception as exc:
        report = {"schema": SCHEMA, "valid": False, "action": "failed",
                  "migration_required": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
