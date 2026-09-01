#!/usr/bin/env python3
"""Verify the exact, offline SAWM R2 source and dependency seal.

This gate is deliberately read-only.  It does not import optional providers in
the validator process.  A separate, hermetic child recomputes the reviewed
M16 control dependency binding without starting an owner.
native DuckDB pin, consumes those bytes through a sealed anonymous descriptor,
loads the exact local HTTPFS and Quack projections, and opens only an in-memory
database for ``SELECT 42``.  Network and extension installation remain denied.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SEAL_PATH = REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json"
NAMESPACE = "semantic-addressed-world-model-v1"
REVISION = "SAWM-PLAN-R2"
SCHEMA = "semantic-addressed-world-model/dependency-seal-validation@1"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_M42_AUTHORITY_CID = (
    "sha256:1e1df3ea3d6b1805dc32f6dd43c61bb95d99e2d08da4c96872441dbc9fdbf587"
)
_M43_AUTHORITY_CID = (
    "sha256:6ddc11cb9e37da82023e5a89124298f532cfc943cc347fa5ea67ff13bcd1eb43"
)
_M43_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M43_FINAL_CONTROL_AUTHORITY_CID"
_M44_AUTHORITY_CID = (
    "sha256:36cba6a8006b50d36d8de2ed6cf76d4adf8d688669a8c6414621173409535845"
)
_M44_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M44_FINAL_CONTROL_AUTHORITY_CID"
_M44_SUCCESSOR_KEY = (
    "post_m43_hardened_procfs_user_manager_restart_successor_materialization"
)
_M44_MIGRATION_REVISION = "SAWM-R2-M44"
_M44_M43_RECEIPT_CID = (
    "sha256:eea44e0f2aae970c1580c59e7b3310904fad480f249b3b1f5df5cd887fc1f050"
)
_M44_M43_EVENT_ID = (
    "baguqeerazsdqgqrzx5xyny5mh4dwpu5olor6rf4oj2p5c65onpykcy35hrvq"
)
_M44_M43_EVIDENCE_ID = (
    "baguqeera2britcaqnxj7ulksuxeg3v5zvdf6uh6xmq2oklgmho3tepaw6ljq"
)
_M44_PRIOR_PROJECTION_CID = (
    "baguqeeraelrmqrsph27tld2bk6ydvq336uff5hihbl42oelhkhuwsg3sqrpa"
)
_M44_TARGET_PROJECTION_CID = (
    "baguqeerageftwrpvliedl3nkrbqlchwo2tzogehnjeyu5iqn6p7jfes4tmea"
)
_M44_M43_FINAL_CONTROL_COMMIT = "de2864eac1f7e49ea44b1dd5f8ad689167374c18"
_M44_M43_FINAL_CONTROL_TREE = "6908632ea9a0cd42a357cae70fb42f48c90a321a"
_M44_REPAIR_COMMIT = "b2136e3eb88600df829afa563b74ab0957ed4061"
_M44_REPAIR_TREE = "99acd200a8c7cd65087fe6df316c0428b731d492"
_M44_REPAIR_DIFF_SHA256 = (
    "66ce916093e2fc6b091ca1963c8eef45c7d4419b1b480c870241117dd92f73bb"
)
_M44_REPAIR_BLOBS = {
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py": (
        "35f423045d8093f8a165f0ed01fdd938b14bafc2"
    ),
    "test/api/test_agent_supervisor_database_portal_bridge.py": (
        "bfccdd879a6c0b2cace3a69abc0d5b428654199c"
    ),
}
_M44_REPAIR_MODES = {path: "100644" for path in _M44_REPAIR_BLOBS}
_M45_AUTHORITY_CID = (
    "sha256:1fece222571aff1238d6e32465cae69f3f766d3e4da303b8775a5df892b95ef7"
)
_M45_AUTHORITY_SIZE = 14_585
_M45_UNSEALED_AUTHORITY_CID = (
    "sha256:PENDING_M45_FINAL_CONTROL_AUTHORITY_CID"
)
_M45_SUCCESSOR_KEY = (
    "failed_pre_authoritative_m44_validation_successor_materialization"
)
_M45_MIGRATION_REVISION = "SAWM-R2-M45"
_M46_AUTHORITY_CID = (
    "sha256:47ed315018541ef5c4c759c1c486cae2a411821ed4e9902877b08439b79b4455"
)
_M46_AUTHORITY_SIZE = 19_365
_M46_UNSEALED_AUTHORITY_CID = "sha256:PENDING_M46_FINAL_CONTROL_AUTHORITY_CID"
_M46_SUCCESSOR_KEY = "legacy_no_delta_rescue_recovery_successor_materialization"
_M46_MIGRATION_REVISION = "SAWM-R2-M46"
_M46_M45_RECEIPT_CID = (
    "sha256:46508d2540469d9cbc3ab0cc5220db0bf0cf0cb3cfdce6eed2ff8c8f4da71a97"
)
_M46_M45_RECEIPT_SHA256 = (
    "61a1b31a062c798b6ef3c94808f32755b263261e799ae20740c65eb96be747d0"
)
_M46_M45_RECEIPT_SIZE = 9_308
_M46_M45_FINAL_CONTROL_COMMIT = "5408e29ba1067c1282589818d175d930aeade1de"
_M46_REPAIR_COMMIT = "e666d239ba37738da9830497be50c8b0737271fc"
_M46_REPAIR_TREE = "6bc26fda6ab6a0b1b27a17b561e6fdea04dce928"
_M46_REPAIR_DIFF_SHA256 = (
    "4610655e8bacc63c38c00e79475e3c715e78c5a74c02f06d47ef2dec99a56669"
)
_M46_REPAIR_BLOBS = {
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py": (
        "274b5a6f0dbcab4ae58e806ceec28fea168d4f87"
    ),
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py": (
        "df4c2679e87ba0b6aa33143e62d2892c07282929"
    ),
    "test/api/test_agent_supervisor_database_portal_bridge.py": (
        "13c454af282400a3c70dc415abb6aacfef03c620"
    ),
    "test/api/test_agent_supervisor_reconciliation_auto_unblock.py": (
        "bb9c70e43c77352c58607f0f8e116637a64e0014"
    ),
}
_M46_REPAIR_MODES = {path: "100644" for path in _M46_REPAIR_BLOBS}
_M46_TARGET_PROJECTION_CID = (
    "baguqeera3gwgex35vb2k2c2u2vq5qq6d65immvntj3nl2fqfzjpts6jvrfta"
)
_HISTORICAL_VALIDATION_PYTHON = "/home/barberb/.local/bin/python"
_OPERATIONAL_VALIDATION_PYTHON = "python"
_VALIDATION_RUNTIME_ROOTS = {
    "delta_deployment": Path(
        "/opt/ipfs-accelerate-aseh-validation-9b3ba6caebcf"
    ),
    "base_deployment": Path(
        "/opt/ipfs-accelerate-legal-validation-7ffe92439767"
    ),
}
_VALIDATION_PYTHONPATH = (
    "/opt/ipfs-accelerate-aseh-validation-9b3ba6caebcf/"
    "site-packages-py-multihash",
    "/opt/ipfs-accelerate-aseh-validation-9b3ba6caebcf/site-packages",
    "/opt/ipfs-accelerate-legal-validation-7ffe92439767/site-packages",
)
_VALIDATION_RUNTIME_ARTIFACTS = {
    "delta_deployment": (
        ("BUILD_CONTRACT.json", "17e20a218f3934bf7b6dd6e80bcba8b76a2f42e85362c49a226547927f5b311e"),
        ("CLOSURE.json", "3c26150daa5697e78405e2da0d9bffb1a91d030aff906811b4f93b4b133ef8e3"),
        ("COLLISIONS.json", "8ad82936c0f6aa14be1066483b92500930ae5520acea9a6ea9652e5cac885fe5"),
        ("DEPLOYMENT.json", "0d557e24f3254e958f8d37ec8d10299b9fdd34e7685621b09dddb84245932cf2"),
        ("DIRECT_REQUIREMENTS.txt", "0842be6780f2cafb759087e506ca11a34ef20df68bbe2dd504a89522ce020633"),
        ("IMPORT_CONTRACT.json", "de9ac3b168124f212df644f391626a6f16808e6aa7f5ab4652fdf7f86bef17ff"),
        ("IMPORT_SMOKE.json", "92315ed765b6740af6baa84d319658c27f678287083e14c508b7edb47d8fb460"),
        ("LOCK.txt", "d3db803570fc0d590187971dcbb6e22514c8b42be93c880993e6dd4c4e3deb86"),
        ("PAYLOAD_MANIFEST.jsonl", "9b3ba6caebcff215c3de2ef00d5b588d059aff92ddc15cd4ffde8eb2dcd766b8"),
        ("TREE_AUDIT.json", "22cf8eaad767eb65f74ddc2ea371086031afc11f47e25bc7cf8d9a3320cfd3e8"),
        ("VERIFICATION_CONTRACT.json", "bf4e60384438f22c0eb43b6146d341667a23154ef6472a610249f81daf36c295"),
        ("WHEELS.json", "d020ce204068f34678a1d5c5fc65860bc4b759370c3016ebf3c9463da8316d67"),
        ("WHEEL_AUDIT.json", "c1928ca420025ce510d87db9a5cbadb6e97c7050c76b5f568189c0715ccc53c8"),
    ),
    "base_deployment": (
        ("ASYNC_SMOKE.json", "ca2f2289f808cfbc66fbc7cdb5f9af6fddb9d6880417a136da6e3ec4f7e737c7"),
        ("CLOSURE.json", "cf2cf2e6dc17dc5906c834aad3d42d4d123653b458ed290f73c3d9a11c43710a"),
        ("COLLECTION.json", "eda31c90a7a7be79b38f257e926426de86733cf46028bb6ebe6c7e7c5b9d64f8"),
        ("DEPLOYMENT.json", "654d64e130c9b8e748ea76c3947eb47cc52bea64adb40f2592f7204dfe503ad0"),
        ("DIRECT_REQUIREMENTS.txt", "d84de7ee9fa44796973e3656680583d1e8a69142018d9b724b666a7d561ffa38"),
        ("IMPORT_SMOKE.json", "45e856c88fbdf3d33b51bca2756284ddf5c0b00fda6f3b6dc737671501cbdfda"),
        ("LOCK.txt", "cc6678e4a724136ae866392b36bf98aa274c28cd3a37ff1a29c2d917449f73a9"),
        ("PAYLOAD_MANIFEST.jsonl", "7ffe92439767e99c849a4f7aad0ee5d64e19ab9f754b5f0915f00571ac51f85a"),
        ("WHEELS.json", "771062b6a93d2e4feb41acf90982cf32009522cb69fb373deb4044b6f3ea6eda"),
        ("WHEEL_AUDIT.json", "01e56d36c78b43f280d70dce8f8364536f448279d6be224df792e23ec64f6fcc"),
    ),
}
_NATIVE_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "board_namespace",
        "plan_revision",
        "status",
        "scope",
        "dependency_id",
        "payload_sha256",
        "python_executable_sha256",
        "authority_basis",
        "inspection_is_authority",
        "authorization_may_claim_task_completion",
        "authorization_id",
    }
)

# These are the only source differences admitted above the sealed base before
# implementation workers begin.  The two README files explain controls but are
# intentionally not completion authority or worker-owned outputs.
CONTROL_PATHS = frozenset(
    {
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
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
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
        "test/api/test_agent_supervisor_grok_quota_terra_gate.py",
        "test/api/test_agent_supervisor_database_coordination.py",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "test/api/test_agent_supervisor_native_dependency_pin.py",
        "test/api/test_agent_supervisor_project_dependency_preflight.py",
        "test/api/test_agent_supervisor_provider_command_binding.py",
        "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
        "test/api/semantic_world/README.md",
        "benchmarks/agent_supervisor/semantic_addressed_world_model/README.md",
    }
)

INTERFACES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py", ("DatabaseTaskSource", "materialize", "snapshot", "get_task", "list_tasks", "ready_tasks", "compare_and_set_status", "record_evidence", "record_validation_result", "projection_matches_events")),
    ("ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py", ("install_datasets_authoritative_operational_schema", "verify_datasets_authoritative_operational_schema")),
    ("ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py", ("load_configured_board", "preflight_configured_board", "configured_board_launch_plan", "main")),
    ("ipfs_accelerate_py/agent_supervisor/runtime/configured_board_live_capsule.py", ("ConfiguredBoardLiveCapsuleAdmission", "parse_configured_board_live_capsule_policy", "parse_configured_board_live_capsule_admission", "build_configured_board_live_capsule_admission", "verify_configured_board_live_capsule")),
    ("ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py", ("DatabaseProgramConfig",)),
    ("ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py", ("preflight_provider_entry_module",)),
    ("ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py", ("QuackStateServer", "build_server")),
    ("ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py", ("discover_live_quack_endpoint", "DuckDBConnection")),
    ("ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py", ("build_mutation_request", "validate_mutation_request", "execute_mutation_bundle", "execute_owner_mutation", "service_mutation_inbox")),
    ("ipfs_accelerate_py/llm_router.py", ("probe_grok_codex_agent_route_readiness",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py", ("WorktreeLifecycleStore",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py", ("LeaseCoordinator",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py", ("CheckoutMutationLease",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/merge_queue.py", ("MergeQueue",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/merge_train.py", ("MergeTrain",)),
    ("ipfs_accelerate_py/agent_supervisor/integrations/ducklake_history_projection.py", ("project_history",)),
    ("ipfs_accelerate_py/agent_supervisor/context/context_compiler.py", ("ContextCompiler",)),
    ("ipfs_accelerate_py/agent_supervisor/context/decision_runtime.py", ("DecisionRuntime",)),
    ("ipfs_accelerate_py/agent_supervisor/semantic_state/harness.py", ("SemanticCompressionHarness",)),
    ("ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py", ("SemanticCompressionGovernor",)),
    ("ipfs_accelerate_py/agent_supervisor/verification/planner.py", ("IncrementalVerificationPlanner",)),
    ("ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/sealer.py", ("IncrementalProofSealer",)),
)


def _duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _run(root: Path, *args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(args), cwd=cwd or root, check=False, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=20,
        env={**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
    )
    if completed.returncode:
        raise RuntimeError(f"{' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _git(root: Path, *args: str, cwd: Path | None = None) -> str:
    return _run(root, "git", *args, cwd=cwd)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _has_presence_based_key_selection(source: str, key: str) -> bool:
    """Return whether source selects *key* by membership, not truthiness.

    The selector may use the literal directly or a local constant.  This keeps
    the gate independent of the operator helper's name while still forbidding
    a malformed newest authority from silently falling back to history.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.If)
            or not isinstance(node.test, ast.Compare)
            or len(node.test.ops) != 1
            or not isinstance(node.test.ops[0], ast.In)
            or len(node.test.comparators) != 1
            or not isinstance(node.test.comparators[0], ast.Name)
            or node.test.comparators[0].id != "config"
        ):
            continue
        left = node.test.left
        selected = (
            left.value
            if isinstance(left, ast.Constant) and isinstance(left.value, str)
            else constants.get(left.id) if isinstance(left, ast.Name) else None
        )
        if selected == key:
            return True
    return False


def _m5_source_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify the mixed-schema history and bounded M5 nonterminal recovery."""

    errors: list[str] = []
    prior_store = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m4/control.duckdb"
    )
    target_store = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m5/control.duckdb"
    )
    m4_hash = "d0c531d2ea30c512beb3587152527c4f605b5a5bf89e6311bf7aaa1974bff670"
    m4_prefix = "77b1a6b834658038c0f0cc44870a28f9fb480750e34dda408ab5718e83ee8911"
    m4_projection = (
        "baguqeera65d24eqqbusuk6vznpmhm4nr65fas3i75dytebq5bev72bs6jaha"
    )
    m4_source = (
        "sha256:72ae538afc063f98a4e7b0a799a619d6a8c499ceb51c9ceb4bffe95de5f9d323"
    )
    task_cid = (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    )
    target_projection = (
        "baguqeerafx22x24mx7qrjjkfmyfdamtjkjd3ikrmhfqrp2gesqhe33l5467q"
    )

    def differs(value: Any, expected: Mapping[str, Any]) -> bool:
        return not isinstance(value, Mapping) or any(
            value.get(key) != item for key, item in expected.items()
        )

    expected_requeue = {
        "schema": "sawm/nonterminal-task-requeue-authorization@1",
        "authorized": True,
        "authority": "operator_source_migration",
        "task_alias": "SAWM-001",
        "task_cid": task_cid,
        "from_status": "in_progress",
        "from_revision": 2,
        "to_status": "todo",
        "to_revision": 3,
        "reason": "bounded_preprovider_capsule_loader_and_attempt_settlement_recovery",
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "expected_event_watermark": 119,
        "worker_self_approval": False,
    }
    expected_inventory = {
        "schema": "sawm/prior-materialization-migration-inventory@3",
        "migration_revision": "SAWM-R2-M5",
        "migration_kind": "bounded_preprovider_capsule_loader_and_attempt_settlement_recovery",
        "supersession_reason": "source_authority_revision_and_preprovider_task_requeue",
        "prior_store_id": prior_store,
        "target_store_id": target_store,
        "prior_plan_revision": 5,
        "target_plan_revision": 6,
        "target_generation": 7,
        "prior_event_watermark": 116,
        "prior_event_prefix_sha256": m4_prefix,
        "prior_control_store_sha256": m4_hash,
        "prior_projection_cid": m4_projection,
        "prior_source_binding_cid": m4_source,
        "prior_source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
        "prior_source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
        "prior_materialization_receipt_cid": (
            "sha256:b24b3a2a0d325540761aaee03ea79bc88836b58577dab8583fec43f0de4c2a21"
        ),
        "prior_authority_preserved": True,
    }
    if differs(migration, expected_inventory):
        errors.append("M5 inventory does not bind the exact frozen M4 authority")

    prior = scheduler.get("prior_materialization")
    if differs(
        prior,
        {
            "migration_revision": "SAWM-R2-M5",
            "reason": "source_authority_revision_and_preprovider_task_requeue",
            "store_id": prior_store,
            "prior_plan_revision": 5,
            "target_plan_revision": 6,
            "migration_history_count": 4,
            "event_watermark": 116,
            "event_prefix_sha256": m4_prefix,
            "control_store_sha256": m4_hash,
            "projection_cid": m4_projection,
            "source_binding_cid": m4_source,
            "source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
            "source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
            "migration_event_watermark": 118,
            "target_event_watermark": 119,
            "target_projection_cid": target_projection,
            "preserve_append_only": True,
        },
    ):
        errors.append("scheduler M5 predecessor binding is not exact")
    if not isinstance(prior, Mapping) or dict(prior.get("nonterminal_task_requeue") or {}) != expected_requeue:
        errors.append("scheduler does not authorize exactly one SAWM-001 recovery CAS")
    program = scheduler.get("database_program")
    if differs(program, {"store_id": target_store, "store_generation": "7"}):
        errors.append("scheduler target must be run-r2-m5 generation 7")

    sealed = seal.get("source_migration")
    if differs(
        sealed,
        {
            "migration_revision": "SAWM-R2-M5",
            "migration_kind": "bounded_preprovider_capsule_loader_and_attempt_settlement_recovery",
            "supersession_reason": "source_authority_revision_and_preprovider_task_requeue",
            "mode": "append_only_source_authority_revision",
            "prior_store_id": prior_store,
            "target_store_id": target_store,
            "prior_plan_revision": 5,
            "target_plan_revision": 6,
            "prior_migration_count": 4,
            "prior_event_watermark": 116,
            "prior_event_prefix_sha256": m4_prefix,
            "prior_control_store_sha256": m4_hash,
            "prior_projection_cid": m4_projection,
            "prior_source_binding_cid": m4_source,
            "prior_source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
            "prior_source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
            "prior_migration_receipt_cid": (
                "sha256:b24b3a2a0d325540761aaee03ea79bc88836b58577dab8583fec43f0de4c2a21"
            ),
            "migration_event_watermark": 118,
            "target_event_watermark": 119,
            "target_projection_cid": target_projection,
            "accepted_definition_rewrite_allowed": False,
            "accepted_completion_replay_allowed": False,
            "prior_authority_preserved": True,
        },
    ):
        errors.append("dependency seal does not preserve the exact M4 authority")
    if not isinstance(sealed, Mapping) or dict(sealed.get("nonterminal_task_requeue") or {}) != expected_requeue:
        errors.append("dependency seal recovery authorization is not exact")

    history = migration.get("migration_history")
    if (
        not isinstance(history, list)
        or len(history) != 4
        or any(not isinstance(entry, Mapping) for entry in history)
    ):
        errors.append("M1-through-M4 history must contain four typed entries")
        history = []
    if history:
        if [entry.get("migration_revision") for entry in history] != [
            "SAWM-R2-M1",
            "SAWM-R2-M2",
            "SAWM-R2-M3",
            "SAWM-R2-M4",
        ]:
            errors.append("M1-through-M4 history revisions are not contiguous")
        for index, entry in enumerate(history[:3], start=1):
            if (
                entry.get("schema") != "sawm/source-migration-history-entry@1"
                or not isinstance(entry.get("prior_event_watermark"), int)
                or entry.get("target_event_watermark")
                != entry.get("prior_event_watermark") + 2
            ):
                errors.append(f"M{index} @1 history must add exactly two events")
        for previous, current in zip(history, history[1:], strict=False):
            if any(
                current.get(prior_key) != previous.get(target_key)
                for prior_key, target_key in (
                    ("prior_store_id", "target_store_id"),
                    ("prior_control_store_sha256", "target_control_store_sha256"),
                    ("prior_event_watermark", "target_event_watermark"),
                    ("prior_event_prefix_sha256", "target_event_prefix_sha256"),
                    ("prior_source_binding_cid", "current_source_binding_cid"),
                )
            ):
                errors.append("migration history continuity is broken")
                break
        m4 = history[3]
        if differs(
            m4,
            {
                "schema": "sawm/source-migration-history-entry@2",
                "migration_revision": "SAWM-R2-M4",
                "prior_event_watermark": 113,
                "migration_event_watermark": 115,
                "target_event_watermark": 116,
                "migration_event_prefix_sha256": "724c47eaf1c70d6ddcf61059f133c752f2787a619aba040e4f2334ecbd287731",
                "target_event_prefix_sha256": m4_prefix,
                "migration_projection_cid": "baguqeeramvvb5ij2hs4eniw735kyvfv3t253vp3tn7atk6vmw2qrcfuqf3sq",
                "projection_cid": m4_projection,
                "target_control_store_sha256": m4_hash,
                "target_store_id": prior_store,
                "current_source_binding_cid": m4_source,
                "post_migration_event_id": "baguqeera3qvtzkbfquvcwhk3weocoesvqlajklm736surlkl25jn2ljkllha",
                "post_migration_event_type": "intent.task_status_changed",
                "post_migration_task_cid": task_cid,
                "post_migration_task_revision": 2,
                "post_migration_task_status": "in_progress",
            },
        ):
            errors.append("M4 @2 migration/frozen-target watermarks are not exact")
        if (
            m4.get("migration_receipt_cid")
            != migration.get("prior_materialization_receipt_cid")
            or m4.get("migration_receipt_path")
            != migration.get("prior_materialization_receipt_path")
        ):
            errors.append("M4 history receipt does not bind the frozen predecessor")

    failure = migration.get("preprovider_task_failure")
    if differs(
        failure,
        {
            "schema": "sawm/pre-provider-task-failure@1",
            "authority_class": "operator_frozen_predecessor_observation",
            "authoritative_completion_evidence": False,
            "store_id": prior_store,
            "control_store_sha256": m4_hash,
            "canonical_event_watermark": 116,
            "canonical_event_prefix_sha256": m4_prefix,
            "canonical_projection_cid": m4_projection,
            "canonical_claim_event_id": "baguqeera3qvtzkbfquvcwhk3weocoesvqlajklm736surlkl25jn2ljkllha",
            "task_alias": "SAWM-001",
            "task_cid": task_cid,
            "task_status": "in_progress",
            "task_revision": 2,
            "execution_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m4/control.execution.duckdb"
            ),
            "execution_store_sha256": "84e010b517a791039a6db16c2860a3292f6fb123d9884521ea52dc31f4223989",
            "coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m4/control.coordination.duckdb"
            ),
            "coordination_store_sha256": "6e4211ed41f02e84e95877a20f94d56660f94d00e2bb80769d4fa4550032a3a2",
            "portal_attempt_path": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m4/"
                "state/sawm_database_portal_attempts/72474c065ac95c1e878159f3"
            ),
            "portal_attempt_binding_path": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m4/"
                "state/sawm_database_portal_attempts/72474c065ac95c1e878159f3/"
                "database-attempt-binding.json"
            ),
            "portal_attempt_binding_id": (
                "sha256:7c9ef3b030749b39ee01b772a3b54b43a77035dbb16f7ea1ec440f4646009f83"
            ),
            "portal_attempt_binding_sha256": "db00bde3459f4eb4973532110647baabe2b2ca335a6704b3dbd22f706ef0e92f",
            "portal_task_projection_sha256": "5710f2c084a967201a9c0ae1fd19d4aaaf7cb5257799462fba2eb768ac31dd01",
            "portal_event_log_sha256": "aca7c522a6868965405696bfcadcc88ca47236eebfca6ca50171e20c9c9a035e",
            "portal_event_manifest_sha256": "48fd39f4e4fe4a4c0d744111589d21eb84cc524fbc8a8c3ae67e458bec0bc67f",
            "portal_event_snapshot_id": "event-log-snapshot:sha256:806ba18bf8230033aadf344c166a193b85c8cf77fd4a21792e4e61cd8ad02171",
            "portal_event_tail_id": "sha256:46e15c941d784c8e7e986b2668bd2a457a13777c3c08297ec8f936c31732590d",
            "portal_event_count": 13,
            "portal_event_first_sequence": 1,
            "portal_event_last_sequence": 13,
            "attempt_consumed": False,
            "retry_deferred": True,
            "successor_migration_revision": "SAWM-R2-M5",
            "successor_plan_revision": 6,
            "successor_generation": 7,
            "successor_store_id": target_store,
        },
    ):
        errors.append("frozen M4 control/companion-store/portal evidence is not exact")
    if not isinstance(failure, Mapping):
        failure = {}
    if failure.get("canonical_authority_counts") != {
        "completion_receipts": 1,
        "effect_claims": 0,
        "merge_attempts": 0,
        "provider_calls": 0,
        "provider_invocations": 0,
        "provider_responses": 0,
        "task_assignments": 0,
        "task_attempts": 0,
        "task_claims": 0,
    }:
        errors.append("frozen M4 canonical zero-authority counts are not exact")
    for field in (
        "provider_call_allowed",
        "provider_dispatch_attempted",
        "provider_dispatched",
        "provider_invocation_recorded",
        "implementation_provider_invoked",
        "effect_claim_recorded",
        "implementation_commit_created",
        "merge_attempted",
        "task_completed",
    ):
        if failure.get(field) is not False:
            errors.append(f"frozen M4 evidence must retain {field}=false")
    if failure.get("merge_queue_request_count") != 0:
        errors.append("frozen M4 evidence must retain zero merge requests")

    required_paths = {
        "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
        "test/api/test_agent_supervisor_database_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "test/api/test_agent_supervisor_project_dependency_preflight.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
        "test/api/test_agent_supervisor_provider_command_binding.py",
    }
    repair_paths = tuple(migration.get("bounded_control_plane_repair_paths") or ())
    protected_paths = set(scheduler.get("protected_paths") or ())
    if (
        not repair_paths
        or len(repair_paths) != len(set(repair_paths))
        or any(path not in CONTROL_PATHS for path in repair_paths)
        or not required_paths.issubset(repair_paths)
        or not required_paths.issubset(protected_paths)
    ):
        errors.append("M5 bounded repair sources/tests are not protected controls")
    return errors


def _m6_operational_validation_authorization() -> dict[str, Any]:
    return {
        "schema": "sawm/operational-validation-requeue-authorization@1",
        "authorized": True,
        "authority": "operator_source_migration",
        "task_alias_first": "SAWM-001",
        "task_alias_last": "SAWM-044",
        "task_count": 44,
        "task_identity_manifest_cid": (
            "sha256:902037a87e711082e57c8dcf002eab669f0dc83c35e9f372ba197a18f8acf1d3"
        ),
        "validation_command_count": 46,
        "validation_token_from": _HISTORICAL_VALIDATION_PYTHON,
        "validation_token_to": _OPERATIONAL_VALIDATION_PYTHON,
        "recovered_task_alias": "SAWM-001",
        "recovered_task_cid": (
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
        "from_status": "blocked",
        "from_revision": 5,
        "operational_status": "blocked",
        "operational_revision": 6,
        "to_status": "todo",
        "to_revision": 7,
        "reason": "bounded_validation_runtime_and_operational_board_command_recovery",
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "implementation_commit_count": 0,
        "merge_attempt_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "operational_validation_revision_changes": 44,
        "expected_event_watermark": 168,
        "worker_self_approval": False,
    }


def _m6_source_migration_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify append-only M6 operational validation and settled requeue."""

    errors: list[str] = []
    prior_store = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m5/control.duckdb"
    )
    target_store = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m6/control.duckdb"
    )
    task_cid = (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    )
    prior_projection = (
        "baguqeeram7gdldpraa6twhuvfbqbeou4gefgmj5szitgb6ij2nmwobx727wq"
    )
    target_projection = (
        "baguqeeraztioh4pdrleiio2hzt2dtxh7o7ae2jk237fvcflktepffj6dpuya"
    )
    expected_authorization = _m6_operational_validation_authorization()
    expected_common = {
        "migration_revision": "SAWM-R2-M6",
        "migration_kind": "bounded_validation_runtime_and_operational_board_command_recovery",
        "supersession_reason": "source_authority_revision_and_settled_preprovider_task_requeue",
        "prior_store_id": prior_store,
        "prior_control_store_sha256": (
            "e2b8e8a15abe9c2a5b53dd540e9621c11a85f63fcc3799da331bb55e3bfbc408"
        ),
        "prior_event_watermark": 121,
        "prior_event_prefix_sha256": (
            "f07bb22f380f30901d0ca16af4bf6bbc3b544f039fb1f43728828d92a95a3f9e"
        ),
        "prior_projection_cid": prior_projection,
        "prior_source_binding_cid": (
            "sha256:81d0fbd74e48c214b5334432cfa0f55e5d6b51a6f5096fa85e323512cf11fb77"
        ),
        "prior_source_head": "89fc7dcac140431f45d117ecf56bdbbd17f16be6",
        "prior_source_tree": "ec160726a9c92436222071e8119730df81eba8a6",
        "prior_generation": 7,
        "prior_plan_revision": 6,
        "target_store_id": target_store,
        "target_generation": 8,
        "target_plan_revision": 7,
        "migration_event_watermark": 123,
        "operational_validation_first_event_watermark": 124,
        "operational_validation_last_event_watermark": 167,
        "target_event_watermark": 168,
        "target_projection_cid": target_projection,
        "operational_validation_revision_changes": 44,
    }

    prior = scheduler.get("prior_materialization")
    expected_prior = {
        "program_definition_cid": "sha256:f581af1f2234c127231b47bb1bb8d42910b984dcd1bece9a6303949ac1ba0b72",
        "plan_root_cid": "sha256:d9481937430405ff6a512e779b14b7ce676de45d277c65d3763ebe49445ba914",
        "operator_task_cid": "sha256:8b8f43dd51ea4d8467af0e5cae4100478f16666d36c6f4fad49c23fd8e43a3d6",
        "store_id": prior_store,
        "control_store_sha256": expected_common["prior_control_store_sha256"],
        "event_watermark": 121,
        "event_prefix_sha256": expected_common["prior_event_prefix_sha256"],
        "projection_cid": prior_projection,
        "source_binding_cid": expected_common["prior_source_binding_cid"],
        "source_head": expected_common["prior_source_head"],
        "source_tree": expected_common["prior_source_tree"],
        "prior_generation": 7,
        "prior_plan_revision": 6,
        "prior_materialization_event_watermark": 119,
        "prior_materialization_projection_cid": (
            "baguqeerafx22x24mx7qrjjkfmyfdamtjkjd3ikrmhfqrp2gesqhe33l5467q"
        ),
        "prior_materialization_receipt_cid": (
            "sha256:7a3800b6ba8cdf722d846bb2fd0bf505b78114cd659f806f7eafe7df81ec71b8"
        ),
        "prior_materialization_receipt_file_sha256": (
            "37838a3153c6fa59d6a9a31d8c17fea3bf902f50fed517112ceac23932ad14ea"
        ),
        "prior_materialization_receipt_path": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m5/migration-receipt.json"
        ),
        "migration_revision": "SAWM-R2-M6",
        "reason": expected_common["supersession_reason"],
        "migration_history_count": 5,
        "target_generation": 8,
        "target_plan_revision": 7,
        "migration_event_watermark": 123,
        "operational_validation_first_event_watermark": 124,
        "operational_validation_last_event_watermark": 167,
        "target_event_watermark": 168,
        "target_projection_cid": target_projection,
        "operational_validation_revision_changes": 44,
        "operational_validation_requeue": expected_authorization,
        "preserve_append_only": True,
        "inventory_path": (
            "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ),
    }
    if type(prior) is not dict or prior != expected_prior:
        errors.append("scheduler M6 operational migration authorization is not exact")

    sealed = seal.get("source_migration")
    if not isinstance(sealed, Mapping):
        errors.append("dependency seal M6 source migration is absent")
        sealed = {}
    expected_sealed_fields = {
        "migration_revision",
        "migration_kind",
        "supersession_reason",
        "supersession_mode",
        "prior_store_id",
        "target_store_id",
        "prior_control_store_sha256",
        "prior_event_watermark",
        "prior_event_prefix_sha256",
        "prior_projection_cid",
        "prior_source_binding_cid",
        "prior_source_head",
        "prior_source_tree",
        "prior_generation",
        "prior_plan_revision",
        "prior_materialization_event_watermark",
        "prior_materialization_projection_cid",
        "prior_materialization_receipt_cid",
        "prior_materialization_receipt_path",
        "prior_materialization_receipt_file_sha256",
        "prior_migration_count",
        "target_migration_count",
        "target_generation",
        "target_plan_revision",
        "migration_event_watermark",
        "operational_validation_first_event_watermark",
        "operational_validation_last_event_watermark",
        "target_event_watermark",
        "target_projection_cid",
        "operational_validation_revision_changes",
        "operational_validation_requeue",
        "program_definition_cid",
        "plan_root_cid",
        "operator_task_cid",
        "bounded_control_plane_repair_paths",
        "accepted_definition_rewrite_allowed",
        "accepted_completion_replay_allowed",
        "prior_authority_preserved",
    }
    if set(sealed) != expected_sealed_fields:
        errors.append("dependency seal M6 source-migration fields are noncanonical")
    for field, expected in expected_common.items():
        if sealed.get(field) != expected:
            errors.append(f"dependency seal M6 {field} is not exact")
    expected_sealed_bindings = {
        "program_definition_cid": expected_prior["program_definition_cid"],
        "plan_root_cid": expected_prior["plan_root_cid"],
        "operator_task_cid": expected_prior["operator_task_cid"],
        "prior_materialization_event_watermark": expected_prior[
            "prior_materialization_event_watermark"
        ],
        "prior_materialization_projection_cid": expected_prior[
            "prior_materialization_projection_cid"
        ],
        "prior_materialization_receipt_cid": expected_prior[
            "prior_materialization_receipt_cid"
        ],
        "prior_materialization_receipt_path": expected_prior[
            "prior_materialization_receipt_path"
        ],
        "prior_materialization_receipt_file_sha256": expected_prior[
            "prior_materialization_receipt_file_sha256"
        ],
    }
    for field, expected in expected_sealed_bindings.items():
        if sealed.get(field) != expected:
            errors.append(f"dependency seal M6 {field} is not exact")
    if (
        sealed.get("operational_validation_requeue") != expected_authorization
        or sealed.get("supersession_mode")
        != "source_authority_revision_and_operational_validation_recovery"
        or sealed.get("prior_migration_count") != 5
        or sealed.get("target_migration_count") != 6
        or sealed.get("accepted_definition_rewrite_allowed") is not False
        or sealed.get("accepted_completion_replay_allowed") is not False
        or sealed.get("prior_authority_preserved") is not True
    ):
        errors.append("dependency seal M6 closed authority policy is not exact")

    inventory_expected = {
        "schema": "sawm/prior-materialization-migration-inventory@3",
        "migration_revision": expected_common["migration_revision"],
        "migration_kind": expected_common["migration_kind"],
        "supersession_reason": expected_common["supersession_reason"],
        "prior_store_id": prior_store,
        "prior_control_store_sha256": expected_common["prior_control_store_sha256"],
        "prior_event_watermark": 121,
        "prior_event_prefix_sha256": expected_common["prior_event_prefix_sha256"],
        "prior_projection_cid": prior_projection,
        "prior_source_binding_cid": expected_common["prior_source_binding_cid"],
        "prior_source_head": expected_common["prior_source_head"],
        "prior_source_tree": expected_common["prior_source_tree"],
        "prior_generation": 7,
        "prior_plan_revision": 6,
        "target_store_id": target_store,
        "target_generation": 8,
        "target_plan_revision": 7,
        "prior_authority_preserved": True,
    }
    for field, expected in inventory_expected.items():
        if migration.get(field) != expected:
            errors.append(f"M6 inventory {field} is not exact")
    task_cids = migration.get("prior_task_cids")
    if isinstance(task_cids, Mapping):
        operational_manifest = [
            {
                "task_alias": f"SAWM-{index:03d}",
                "task_cid": str(task_cids.get(f"SAWM-{index:03d}") or ""),
            }
            for index in range(1, 45)
        ]
        manifest_cid = "sha256:" + hashlib.sha256(
            json.dumps(
                operational_manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if manifest_cid != expected_authorization["task_identity_manifest_cid"]:
            errors.append("M6 44-task operational identity manifest does not rehash")
    else:
        errors.append("M6 prior task identity map is absent")
    if (
        expected_authorization["task_alias_first"] != "SAWM-001"
        or "SAWM-000" in {
            f"SAWM-{index:03d}" for index in range(1, 45)
        }
        or expected_authorization["accepted_completion_changes"] != 0
    ):
        errors.append("M6 operational revision set could alter a completed task")

    history = migration.get("migration_history")
    if (
        not isinstance(history, list)
        or len(history) != 5
        or any(not isinstance(entry, Mapping) for entry in history)
    ):
        errors.append("M1-through-M5 history must contain five typed entries")
        history = []
    if history:
        if [entry.get("migration_revision") for entry in history] != [
            f"SAWM-R2-M{index}" for index in range(1, 6)
        ]:
            errors.append("M1-through-M5 migration history is not contiguous")
        if [entry.get("schema") for entry in history[:4]] != [
            "sawm/source-migration-history-entry@1",
            "sawm/source-migration-history-entry@1",
            "sawm/source-migration-history-entry@1",
            "sawm/source-migration-history-entry@2",
        ]:
            errors.append("M1-through-M4 history schemas changed")
        for previous, current in zip(history, history[1:], strict=False):
            if any(
                current.get(prior_key) != previous.get(target_key)
                for prior_key, target_key in (
                    ("prior_store_id", "target_store_id"),
                    ("prior_control_store_sha256", "target_control_store_sha256"),
                    ("prior_event_watermark", "target_event_watermark"),
                    ("prior_event_prefix_sha256", "target_event_prefix_sha256"),
                    ("prior_source_binding_cid", "current_source_binding_cid"),
                )
            ):
                errors.append("M1-through-M5 migration-history continuity is broken")
                break
        m5 = history[4]
        expected_post_materialization = [
            {
                "global_sequence": 120,
                "event_id": "baguqeeram2bg7rwq3hfl75gd7irlklgmcb365m4d6kdvvwm4icirtp6m5jna",
                "event_type": "intent.task_status_changed",
                "task_cid": task_cid,
                "task_status": "in_progress",
                "task_revision": 4,
            },
            {
                "global_sequence": 121,
                "event_id": "baguqeerahtwmflkih723eeb63svkcogqagmgyblkhrd4sycn7pvjcya7dgqq",
                "event_type": "intent.task_status_changed",
                "task_cid": task_cid,
                "task_status": "blocked",
                "task_revision": 5,
            },
        ]
        if (
            m5.get("schema") != "sawm/source-migration-history-entry@3"
            or m5.get("prior_event_watermark") != 116
            or m5.get("migration_event_watermark") != 118
            or m5.get("materialization_event_watermark") != 119
            or m5.get("target_event_watermark") != 121
            or m5.get("migration_event_prefix_sha256")
            != "f026f7b7180b652774fb22d318039f6ab6720caa51b8c89d5959e3b42541cb41"
            or m5.get("materialization_event_prefix_sha256")
            != "2a8e967ebfa966b8409b285133057788fc947727feda859ea190361b8d46b053"
            or m5.get("target_control_store_sha256")
            != expected_common["prior_control_store_sha256"]
            or m5.get("target_event_prefix_sha256")
            != expected_common["prior_event_prefix_sha256"]
            or m5.get("projection_cid") != prior_projection
            or m5.get("target_store_id") != prior_store
            or m5.get("current_source_binding_cid")
            != expected_common["prior_source_binding_cid"]
            or m5.get("migration_projection_cid")
            != "baguqeeralq7ohxikjp3ngml4ffevwniisq5uelo4gsuiclad2zdha2rwv5ta"
            or m5.get("materialization_projection_cid")
            != "baguqeerafx22x24mx7qrjjkfmyfdamtjkjd3ikrmhfqrp2gesqhe33l5467q"
            or m5.get("post_materialization_events")
            != expected_post_materialization
        ):
            errors.append("M5 @3 materialization/runtime history is not exact")

    failure = migration.get("preprovider_task_failure")
    if not isinstance(failure, Mapping):
        errors.append("frozen M5 pre-provider failure is absent")
        failure = {}
    expected_failure_subset = {
        "schema": "sawm/pre-provider-task-failure@2",
        "authority_class": "operator_frozen_predecessor_observation",
        "authoritative_completion_evidence": False,
        "store_id": prior_store,
        "control_store_sha256": expected_common["prior_control_store_sha256"],
        "canonical_event_watermark": 121,
        "canonical_event_prefix_sha256": expected_common["prior_event_prefix_sha256"],
        "canonical_projection_cid": prior_projection,
        "canonical_claim_event_id": (
            "baguqeeram2bg7rwq3hfl75gd7irlklgmcb365m4d6kdvvwm4icirtp6m5jna"
        ),
        "canonical_claim_revision": 4,
        "canonical_settlement_event_id": (
            "baguqeerahtwmflkih723eeb63svkcogqagmgyblkhrd4sycn7pvjcya7dgqq"
        ),
        "canonical_task_status": "blocked",
        "canonical_task_revision": 5,
        "task_alias": "SAWM-001",
        "task_cid": task_cid,
        "task_status": "in_progress",
        "task_revision": 4,
        "failure_event_type": "validation_project_dependency_preflight_failed",
        "failure_reason": "validation_project_dependency_preflight_failed",
        "settlement_closed": True,
        "settlement_failure_kind": "terminal_portal_bridge_error",
        "retry_deferred": True,
        "attempt_consumed": False,
        "provider_call_allowed": False,
        "provider_dispatch_attempted": False,
        "provider_dispatched": False,
        "provider_invocation_recorded": False,
        "implementation_provider_invoked": False,
        "effect_claim_recorded": False,
        "implementation_commit_created": False,
        "merge_attempted": False,
        "task_completed": False,
    }
    for field, expected in expected_failure_subset.items():
        if failure.get(field) != expected:
            errors.append(f"frozen M5 pre-provider failure {field} is not exact")
    if failure.get("canonical_authority_counts") != {
        "completion_receipts": 1,
        "effect_claims": 0,
        "merge_attempts": 0,
        "provider_calls": 0,
        "provider_invocations": 0,
        "provider_responses": 0,
        "task_assignments": 0,
        "task_attempts": 0,
        "task_claims": 0,
    }:
        errors.append("frozen M5 canonical authority counts are not exact")

    repair_paths = tuple(sealed.get("bounded_control_plane_repair_paths") or ())
    required_paths = {
        "requirements.txt",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
    }
    protected_paths = set(scheduler.get("protected_paths") or ())
    capsule = scheduler.get("configured_board_live_capsule")
    capsule_paths = set(capsule.get("control_paths") or ()) if isinstance(capsule, Mapping) else set()
    if (
        set(repair_paths) != required_paths
        or not required_paths.issubset(CONTROL_PATHS)
        or not required_paths.issubset(protected_paths)
        or not required_paths.issubset(capsule_paths)
    ):
        errors.append("M6 bounded validation-runtime repairs are not exact protected controls")
    return errors


def _m7_source_repair_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify the exact source-only successor that repairs M6 live comparison."""

    errors: list[str] = []
    expected_cid = (
        "sha256:326b6d9185ff9974050333d6c8e011ab3dce8772c322fd1143a27f926b72e468"
    )
    configured = scheduler.get("source_repair_materialization")
    inventoried = migration.get("source_repair_materialization")
    if type(configured) is not dict or configured != inventoried:
        errors.append("M7 source-only repair authority differs across controls")
        configured = {}
    observed_cid = "sha256:" + hashlib.sha256(
        _canonical_json(configured)
    ).hexdigest()
    if (
        observed_cid != expected_cid
        or seal.get("source_repair_materialization_cid") != expected_cid
    ):
        errors.append("M7 source-only repair authority CID is not exact")

    expected_subset = {
        "schema": "sawm/source-only-launch-repair-authorization@1",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": "SAWM-R2-M7",
        "migration_kind": "authenticated_live_task_contract_comparator_repair",
        "supersession_mode": "source_only_live_task_contract_comparator_repair",
        "prior_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m6/control.duckdb"
        ),
        "prior_control_store_sha256": (
            "17e4dd76258b0584df1106e0324654c380fcf4f36587463a17936193a260381f"
        ),
        "prior_control_store_size": 39_071_744,
        "prior_event_watermark": 168,
        "prior_event_prefix_sha256": (
            "23093117ac6a8cf15c65668e4a455673800b4d1dcbc457dddec9b0528e7f0e9e"
        ),
        "prior_projection_cid": (
            "baguqeeraztioh4pdrleiio2hzt2dtxh7o7ae2jk237fvcflktepffj6dpuya"
        ),
        "prior_source_head": "9c92490e245a806e3f860ce71c61104a23c2a1b4",
        "prior_source_tree": "d86d0ca4fd820c12316aba7c56dcd788242b5186",
        "prior_source_binding_cid": (
            "sha256:fdf0f6ea842e5e8326a5e5f488f9b19a72983dae296081756164756346ae53b4"
        ),
        "prior_database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
        "prior_generation": 8,
        "prior_plan_revision": 7,
        "prior_process_birth_id": "birth:8f543ddf21dd6ff4e98e73eafa9c88f3",
        "prior_server_id": "server:a1bb77fa-e5c6-427a-b151-e971a2fe0b0b",
        "prior_materialization_receipt_cid": (
            "sha256:6fa4147b4ec8badd2872da24c77c115afeb5f6fc576c8fcbda8e0273f54bc8b3"
        ),
        "prior_materialization_receipt_file_sha256": (
            "acb938ad0454a058ad8e5f8ae60b7ad7ae7f01128e9632794b1d5ee38404c93f"
        ),
        "prior_owner_status_sha256": (
            "6f82495e11614b7fcca1d95c78ef6300b0691b76f9ac63bba19c64eee7e5326c"
        ),
        "prior_validator_digest": (
            "sha256:c280087b49e10e9797d1a8815ed7a085ce42cdb50b7941851da16bd76a67d80d"
        ),
        "prior_semantic_authority_digest": (
            "sha256:e08b5d3695c3c9471f5738d0efed4165a1f87987851f7263296c0cff9259cf65"
        ),
        "prior_frozen_base_authority_digest": (
            "sha256:3b5ca340647cc6820e8827040fc96526986bf1cfc6a6e2d243e945be7a70835c"
        ),
        "prior_append_surface_digest": (
            "sha256:ab641eb7736abc257cd4aea145d90fb5a6c4ed56eec6e4776e92e33ec60ee842"
        ),
        "target_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m7/control.duckdb"
        ),
        "target_generation": 9,
        "target_plan_revision": 8,
        "target_event_watermark": 170,
        "target_projection_cid": (
            "baguqeeraypw3zwt7gud33pymtc2xsjgl5ezap73jcbh4tvbzs5psea7gdspq"
        ),
        "event_suffix_length": 2,
        "live_preflight_failure_cid": (
            "sha256:2587c8e77f72011245b0be7d9f1bbb28c640c20c41bce811a5501dd03d770cf4"
        ),
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
    for field, expected in expected_subset.items():
        if configured.get(field) != expected:
            errors.append(f"M7 source-only repair {field} is not exact")

    failure = configured.get("live_preflight_failure")
    if not isinstance(failure, Mapping) or (
        "sha256:" + hashlib.sha256(_canonical_json(failure)).hexdigest()
        != expected_subset["live_preflight_failure_cid"]
        or failure.get("error_payload")
        != {
            "schema": "sawm/operator-error@1",
            "valid": False,
            "error": "OperatorError: live Quack task definition conflict: SAWM-001",
        }
        or failure.get("provider_probed") is not False
        or failure.get("task_claimed") is not False
        or failure.get("task_state_changed") is not False
        or failure.get("implementation_provider_invoked") is not False
    ):
        errors.append("M7 frozen live-preflight failure is not exact")

    required_repair_paths = {
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
    }
    if (
        set(configured.get("bounded_control_plane_repair_paths") or ())
        != required_repair_paths
        or not required_repair_paths.issubset(CONTROL_PATHS)
        or not required_repair_paths.issubset(
            set(scheduler.get("protected_paths") or ())
        )
        or not required_repair_paths.issubset(
            set(
                (scheduler.get("configured_board_live_capsule") or {}).get(
                    "control_paths"
                )
                or ()
            )
        )
    ):
        errors.append("M7 bounded source-repair paths are not exact protected controls")

    # M7 is immutable historical authority after M8.  Active execution paths
    # belong to the adjacent M8 successor and are checked below; requiring the
    # scheduler to keep pointing at M7 would silently make the successor
    # impossible to activate.
    return errors


def _m8_source_repair_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify the exact M8 optional child-goal projection repair successor."""

    errors: list[str] = []
    expected_cid = (
        "sha256:01935058ca682411743904513524b3eab41369b057151db529af14d46c5c0963"
    )
    configured = scheduler.get("source_repair_successor_materialization")
    inventoried = migration.get("source_repair_successor_materialization")
    if type(configured) is not dict or configured != inventoried:
        errors.append("M8 source-only repair authority differs across controls")
        configured = {}
    observed_cid = "sha256:" + hashlib.sha256(
        _canonical_json(configured)
    ).hexdigest()
    if (
        observed_cid != expected_cid
        or seal.get("source_repair_successor_materialization_cid")
        != expected_cid
    ):
        errors.append("M8 source-only repair authority CID is not exact")

    expected_subset = {
        "schema": "sawm/source-only-launch-repair-authorization@2",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": "SAWM-R2-M8",
        "migration_kind": "optional_nonroot_goal_objective_projection_repair",
        "supersession_mode": "source_only_optional_goal_projection_repair",
        "prior_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m7/control.duckdb"
        ),
        "prior_control_store_sha256": (
            "7f25cdab2a33ba9fb9433926ab7541ac85a77dd9c7747c66a1f8bd7867e1cd4b"
        ),
        "prior_control_store_size": 39_071_744,
        "prior_event_watermark": 170,
        "prior_event_prefix_sha256": (
            "a2e17a1f65391e3786fb055e664483af2615db6279b5404cdde6a1ede5ec6ef6"
        ),
        "prior_projection_cid": (
            "baguqeeraypw3zwt7gud33pymtc2xsjgl5ezap73jcbh4tvbzs5psea7gdspq"
        ),
        "prior_source_head": "bf50aea3e332ce3c520cd97b29050c8464342548",
        "prior_source_tree": "b77ad846057e36236fde186296de41a3603cba64",
        "prior_source_binding_cid": (
            "sha256:acf273e8787cca8044ee6ec63d55e3f46c8f81d1fd295c35d81a692dc32d7af3"
        ),
        "prior_database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
        "prior_generation": 9,
        "prior_plan_revision": 8,
        "prior_process_birth_id": "birth:8fd3d851535249b91d23d185547a0f75",
        "prior_server_id": "server:76aea654-573d-4626-81ff-938f9bd04273",
        "prior_materialization_receipt_path": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m7/migration-receipt.json"
        ),
        "prior_materialization_receipt_cid": (
            "sha256:f18f81498120d8e1a8aa41b24a674e2d42aa0bc4168dbbe32ec820ad7291d1e3"
        ),
        "prior_materialization_receipt_file_sha256": (
            "daf19fac056872b3b66514c6b163d98a082f6755120fb9c6c403490c888c6fa7"
        ),
        "prior_owner_status_path": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m7/quack-owner/quack-state-server.status.json"
        ),
        "prior_owner_status_sha256": (
            "3036aaf439cdbdd956e9882791a0136da8dc11f73440d2a315b3d0bf523df0db"
        ),
        "prior_validator_digest": (
            "sha256:ea2638a05a4e7437dd920622712618eddc5065b7fe5228813304a8ac66b4355a"
        ),
        "prior_semantic_authority_digest": (
            "sha256:e08b5d3695c3c9471f5738d0efed4165a1f87987851f7263296c0cff9259cf65"
        ),
        "prior_frozen_base_authority_digest": (
            "sha256:6e4428920ae70d1fac053e9fc607a276cd06085d65725a4313912b3e3ade49b0"
        ),
        "prior_append_surface_digest": (
            "sha256:4f87ca81a4ba869bff77f3c530f62fa8e46f6f9787b662517dd4ddda7d661960"
        ),
        "target_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m8/control.duckdb"
        ),
        "target_generation": 10,
        "target_plan_revision": 9,
        "target_event_watermark": 172,
        "target_projection_cid": (
            "baguqeerale774wldjbmz4wfexfzcbfrc3i4rkpsxnvsdds5f243bjp7a3iea"
        ),
        "event_suffix_length": 2,
        "live_preflight_failure_cid": (
            "sha256:78943b17ce5a613c10b08e90a4a012cf2c10eafe49b50f5602513374b320ee82"
        ),
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
    for field, expected in expected_subset.items():
        if configured.get(field) != expected:
            errors.append(f"M8 source-only repair {field} is not exact")

    failure = configured.get("live_preflight_failure")
    expected_error_payload = {
        "schema": "sawm/operator-error@1",
        "valid": False,
        "error": "KeyError: 'objective_id'",
    }
    expected_failure_subset = {
        "schema": "sawm/live-control-preflight-failure@2",
        "attempt": "SAWM-R2-M7-LIVE-A1",
        "phase": "authenticated_live_goal_definition_validation",
        "command": (
            "python scripts/ops/agent_supervisor/"
            "semantic_addressed_world_model.py preflight"
        ),
        "exit_code": 2,
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
        "store_id": expected_subset["prior_store_id"],
        "post_failure_frozen_control_store_sha256": expected_subset[
            "prior_control_store_sha256"
        ],
        "event_watermark": 170,
        "event_prefix_sha256": expected_subset["prior_event_prefix_sha256"],
        "projection_cid": expected_subset["prior_projection_cid"],
        "semantic_authority_digest": expected_subset[
            "prior_semantic_authority_digest"
        ],
        "owner_generation": 9,
        "owner_server_id": expected_subset["prior_server_id"],
        "owner_process_birth_id": expected_subset["prior_process_birth_id"],
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
    if not isinstance(failure, Mapping):
        errors.append("M8 frozen live-preflight failure is not exact")
    else:
        error_payload = failure.get("error_payload")
        if (
            error_payload != expected_error_payload
            or "sha256:"
            + hashlib.sha256(_canonical_json(error_payload)).hexdigest()
            != expected_failure_subset["error_payload_cid"]
            or "sha256:" + hashlib.sha256(_canonical_json(failure)).hexdigest()
            != expected_subset["live_preflight_failure_cid"]
            or any(
                failure.get(field) != expected
                for field, expected in expected_failure_subset.items()
            )
        ):
            errors.append("M8 frozen live-preflight failure is not exact")

    required_repair_paths = {
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
    }
    if (
        set(configured.get("bounded_control_plane_repair_paths") or ())
        != required_repair_paths
        or not required_repair_paths.issubset(CONTROL_PATHS)
        or not required_repair_paths.issubset(
            set(scheduler.get("protected_paths") or ())
        )
        or not required_repair_paths.issubset(
            set(
                (scheduler.get("configured_board_live_capsule") or {}).get(
                    "control_paths"
                )
                or ()
            )
        )
    ):
        errors.append("M8 bounded source-repair paths are not exact protected controls")

    # Preserve the complete M8 execution-path gate for an historical control
    # set.  Presence (not truthiness) of the M9 key transfers active-path
    # authority to M9; a present malformed M9 record is rejected below rather
    # than silently falling back here.
    if "live_recovery_successor_materialization" not in scheduler:
        target_store = expected_subset["target_store_id"]
        program = scheduler.get("database_program")
        owner = scheduler.get("quack_owner")
        history = scheduler.get("ducklake_history_projection")
        if not isinstance(program, Mapping) or (
            program.get("authority_mode") != "quack"
            or program.get("task_source_kind") != "duckdb"
            or program.get("endpoint_secret_handle")
            != "env://SAWM_QUACK_TOKEN"
            or program.get("store_id") != target_store
            or program.get("store_generation") != "10"
            or program.get("quack_endpoint") != "quack:127.0.0.1:45250"
            or program.get("schema_revision")
            != "datasets-authoritative-operational-v1"
            or program.get("failover_policy") != "fail_closed"
            or program.get("explicit_legacy") is not False
            or program.get("event_store_path")
            != (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/events"
            )
            or program.get("runtime_registry_path")
            != (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/registry"
            )
            or program.get("worktree_root")
            != (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/worktrees"
            )
        ):
            errors.append("scheduler M8 execution authority is not exact")
        if not isinstance(owner, Mapping) or (
            owner.get("database_path") != target_store
            or owner.get("store_id") != target_store
            or owner.get("state_dir")
            != (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/quack-owner"
            )
            or owner.get("host") != "127.0.0.1"
            or owner.get("port") != 45250
            or owner.get("secret_handle") != "env://SAWM_QUACK_TOKEN"
            or owner.get("migration_profile")
            != "datasets-authoritative-operational-v1"
            or owner.get("start_once_before_scheduler") is not True
            or owner.get("live_transport_query_required") is not True
            or owner.get("status_file_alone_is_ready") is not False
            or owner.get("automatic_direct_file_fallback") is not False
        ):
            errors.append("scheduler M8 Quack owner authority is not exact")
        if not isinstance(history, Mapping) or (
            history.get("catalog_path")
            != (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/ducklake/history.ducklake"
            )
            or history.get("data_path")
            != (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/ducklake/data"
            )
            or history.get("receipt_path")
            != (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/ducklake-history-receipt.json"
            )
        ):
            errors.append("scheduler M8 DuckLake projection paths are not exact")
        expected_runtime = {
            "root": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m8"
            ),
            "state": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/state"
            ),
            "worktrees": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/worktrees"
            ),
            "merge_queue": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/merge-queue"
            ),
            "logs": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m8/logs"
            ),
            "generated_runtime_artifacts_are_completion_authority": False,
        }
        if scheduler.get("runtime_paths") != expected_runtime:
            errors.append("scheduler M8 runtime paths are not exact")
    return errors


def _m9_live_recovery_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify the exact M9 live-failure rearm and active runtime authority."""

    errors: list[str] = []
    authority_key = "live_recovery_successor_materialization"
    expected_cid = (
        "sha256:e7834d12a6150a7d4dd1f90dcb5369d73a6d8620a38bf6baa0cbbb3da17bebb9"
    )
    if authority_key not in scheduler:
        errors.append("M9 live recovery authority key is absent from scheduler")
    if authority_key not in migration:
        errors.append("M9 live recovery authority key is absent from inventory")
    configured = scheduler[authority_key] if authority_key in scheduler else None
    inventoried = migration[authority_key] if authority_key in migration else None
    if type(configured) is not dict or configured != inventoried:
        errors.append("M9 live recovery authority differs across controls")
        configured = {}
    observed_cid = "sha256:" + hashlib.sha256(
        _canonical_json(configured)
    ).hexdigest()
    if "live_recovery_successor_materialization_cid" not in seal:
        errors.append("M9 live recovery authority CID key is absent from seal")
    if (
        observed_cid != expected_cid
        or seal.get("live_recovery_successor_materialization_cid")
        != expected_cid
    ):
        errors.append("M9 live recovery authority CID is not exact")

    expected_subset = {
        "schema": "sawm/control-plane-runtime-recovery-authorization@1",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": "SAWM-R2-M9",
        "migration_kind": (
            "board_scoped_checkout_lock_and_portal_deferral_recovery"
        ),
        "supersession_mode": "append_only_live_failure_rearm",
        "prior_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m8/control.duckdb"
        ),
        "prior_control_store_sha256": (
            "e8b0814314b38f4e73c77258403aacd495c341d1c249d8fd84c5b29b0d1a21ab"
        ),
        "prior_control_store_size": 45_101_056,
        "prior_event_watermark": 174,
        "prior_event_prefix_sha256": (
            "ac685db15d9e2f47c904c618ce7b1573aedeceb5613db144c78a2f93a0a0fe1a"
        ),
        "prior_projection_cid": (
            "baguqeerarcj7jcicssddipfwagqjy5zhikzcohotihhsycnsl5ctqyuydgra"
        ),
        "prior_generation": 10,
        "prior_plan_revision": 9,
        "prior_coordination_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m8/control.coordination.duckdb"
        ),
        "prior_coordination_store_sha256": (
            "9363b3763bdfea7ba3277f4b19853c093ca2ede36a8592832b1ca04d033576af"
        ),
        "prior_coordination_store_size": 6_828_032,
        "prior_coordination_event_count": 35,
        "prior_coordination_projection_digest": (
            "sha256:a31ff114882052cff614cfc4e134ed6cbe44de456fdd337bfcb296d34caa6a24"
        ),
        "target_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m9/control.duckdb"
        ),
        "target_coordination_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m9/control.coordination.duckdb"
        ),
        "target_generation": 11,
        "target_plan_revision": 10,
        "target_event_watermark": 177,
        "target_projection_cid": (
            "baguqeeraebsdnrj7pvgd6yp26yr6ob7ocbrfzjwg57xfjnr6sn4xfkuvkq2q"
        ),
        "target_coordination_projection_digest": (
            "sha256:7fb9bacb0f76fe832cc34dd5fb2ccdef13ddafeef4ec2a3d532cb902aa62011e"
        ),
        "event_suffix_length": 3,
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
    for field, expected in expected_subset.items():
        if configured.get(field) != expected:
            errors.append(f"M9 live recovery {field} is not exact")

    expected_provider_strategy = {
        "capability_probe_required": True,
        "prior_route": "grok-4.6_then_codex_on_independently_verified_quota",
        "provider_result_is_completion_authority": False,
        "route_changed": False,
        "target_route": "grok-4.6_then_codex_on_independently_verified_quota",
    }
    if configured.get("provider_strategy") != expected_provider_strategy:
        errors.append("M9 provider strategy does not preserve the sealed route")

    expected_task_rearm = {
        "schema": "sawm/control-plane-task-rearm-authorization@1",
        "operation": "operator_control_plane_repair",
        "task_alias": "SAWM-001",
        "task_cid": (
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
        "settlement_id": (
            "baguqeeransrfearh5ojrnru6ls43mn7xsx6mlhndkhokrn4k7wlhowxjnshq"
        ),
        "from_status": "blocked",
        "from_revision": 9,
        "to_status": "retrying",
        "to_revision": 10,
        "automatic_retry_admitted": False,
        "coordination_rearm_required": True,
        "coordination_rearm_observed_at_ms": 1_787_939_913_492,
        "prior_coordination_event_count": 35,
        "target_coordination_event_count": 36,
        "historical_settlement_preserved": True,
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "implementation_commit_count": 0,
        "merge_attempt_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    if configured.get("task_rearm") != expected_task_rearm:
        errors.append("M9 task rearm authority is not exact")

    failure = configured.get("live_implementation_failure")
    if not isinstance(failure, Mapping) or (
        failure.get("schema") != "sawm/live-control-implementation-failure@1"
        or failure.get("attempt") != "SAWM-R2-M8-LIVE-A1"
        or failure.get("phase") != "implementation_execution"
        or failure.get("failure_kind")
        != "external_protected_checkout_recovery_required"
        or failure.get("task_alias") != "SAWM-001"
        or failure.get("from_status") != "todo"
        or failure.get("from_revision") != 7
        or failure.get("claim_status") != "in_progress"
        or failure.get("claim_revision") != 8
        or failure.get("terminal_status") != "blocked"
        or failure.get("terminal_revision") != 9
        or failure.get("provider_invocation_count") != 0
        or failure.get("effect_claim_count") != 0
        or failure.get("implementation_commit_count") != 0
        or failure.get("merge_attempt_count") != 0
        or failure.get("worker_self_approval") is not False
    ):
        errors.append("M9 frozen live implementation failure is not exact")

    required_repair_paths = {
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
    }
    protected = set(scheduler.get("protected_paths") or ())
    capsule = scheduler.get("configured_board_live_capsule")
    capsule_controls = set(
        capsule.get("control_paths") or ()
        if isinstance(capsule, Mapping)
        else ()
    )
    if (
        set(configured.get("bounded_control_plane_repair_paths") or ())
        != required_repair_paths
        or len(configured.get("bounded_control_plane_repair_paths") or ()) != 14
        or not required_repair_paths.issubset(CONTROL_PATHS)
        or not required_repair_paths.issubset(protected)
        or not required_repair_paths.issubset(capsule_controls)
    ):
        errors.append("M9 bounded repair paths are not exact protected controls")

    # M9 remains immutable historical authority once M10 is declared.  Key
    # presence, including a malformed declaration in any one control, moves
    # active runtime validation to the fail-closed M10 gate below; it must not
    # make this historical helper demand the superseded generation-11 paths.
    m10_key = "live_projection_successor_materialization"
    m10_selected = any(
        (
            m10_key in scheduler,
            m10_key in migration,
            "live_projection_successor_materialization_cid" in seal,
        )
    )
    if m10_selected:
        return errors

    root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m9"
    target_store = f"{root}/control.duckdb"
    expected_program = {
        "authority_mode": "quack",
        "endpoint_secret_handle": "env://SAWM_QUACK_TOKEN",
        "event_store_path": f"{root}/events",
        "explicit_legacy": False,
        "export_profile": "semantic-addressed-world-model-r2",
        "failover_policy": "fail_closed",
        "quack_endpoint": "quack:127.0.0.1:45251",
        "runtime_registry_path": f"{root}/registry",
        "schema_revision": "datasets-authoritative-operational-v1",
        "store_generation": "11",
        "store_id": target_store,
        "task_source_kind": "duckdb",
        "worktree_root": f"{root}/worktrees",
    }
    if scheduler.get("database_program") != expected_program:
        errors.append("scheduler M9 execution authority is not exact")

    owner = scheduler.get("quack_owner")
    expected_owner_fields = {
        "automatic_direct_file_fallback": False,
        "database_path": target_store,
        "host": "127.0.0.1",
        "live_transport_query_required": True,
        "migration_profile": "datasets-authoritative-operational-v1",
        "port": 45_251,
        "repository_id": "ipfs_accelerate_py:semantic-addressed-world-model-v1",
        "secret_handle": "env://SAWM_QUACK_TOKEN",
        "start_once_before_scheduler": True,
        "state_dir": f"{root}/quack-owner",
        "status_file_alone_is_ready": False,
        "store_id": target_store,
    }
    if not isinstance(owner, Mapping) or any(
        owner.get(field) != expected
        for field, expected in expected_owner_fields.items()
    ):
        errors.append("scheduler M9 Quack owner authority is not exact")

    expected_history = {
        "authority": False,
        "catalog_path": f"{root}/ducklake/history.ducklake",
        "completion_prerequisite": False,
        "data_path": f"{root}/ducklake/data",
        "enabled_when_locally_available": True,
        "install_or_network_forbidden": True,
        "load_local_only": True,
        "receipt_path": f"{root}/ducklake-history-receipt.json",
        "scheduling_prerequisite": False,
        "typed_unavailability_is_permitted": True,
    }
    if scheduler.get("ducklake_history_projection") != expected_history:
        errors.append("scheduler M9 DuckLake projection is not exact")

    expected_runtime = {
        "root": root,
        "state": f"{root}/state",
        "worktrees": f"{root}/worktrees",
        "merge_queue": f"{root}/merge-queue",
        "logs": f"{root}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    if scheduler.get("runtime_paths") != expected_runtime:
        errors.append("scheduler M9 runtime paths are not exact")

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
    if scheduler.get("provider") != expected_provider:
        errors.append("scheduler M9 provider route is not exact")
    return errors


def _m10_live_projection_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify M10's exact successor-aware live projection repair."""

    errors: list[str] = []
    authority_key = "live_projection_successor_materialization"
    cid_key = "live_projection_successor_materialization_cid"
    if authority_key not in scheduler:
        errors.append("M10 live projection authority key is absent from scheduler")
    if authority_key not in migration:
        errors.append("M10 live projection authority key is absent from inventory")
    if cid_key not in seal:
        errors.append("M10 live projection authority CID key is absent from seal")
    configured = scheduler[authority_key] if authority_key in scheduler else None
    inventoried = migration[authority_key] if authority_key in migration else None
    if type(configured) is not dict or configured != inventoried:
        errors.append("M10 live projection authority differs across controls")
        configured = {}
    expected_cid = (
        "sha256:f27878def0ee9b406d0dbaac669728278e4f375cb267ee7f76330faaf9b14f10"
    )
    sealed_cid = seal.get(cid_key)
    observed_cid = "sha256:" + hashlib.sha256(
        _canonical_json(configured)
    ).hexdigest()
    if (
        not isinstance(sealed_cid, str)
        or _SHA256.fullmatch(sealed_cid) is None
        or observed_cid != expected_cid
        or sealed_cid != expected_cid
    ):
        errors.append("M10 live projection authority CID is not exact")

    root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m10"
    prior_root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m9"
    m9_projection = (
        "baguqeeraebsdnrj7pvgd6yp26yr6ob7ocbrfzjwg57xfjnr6sn4xfkuvkq2q"
    )
    m10_projection = (
        "baguqeerareq2bngq3hffyk5vidym2ukeleg5gehpaxhqvvdjayn7ucxplcaq"
    )
    semantic_digest = (
        "sha256:a4903791c91cc2e9c3337f2abdfd6af78389f7036d54cb7f8e43246cd4f0c023"
    )
    coordination_digest = (
        "sha256:7fb9bacb0f76fe832cc34dd5fb2ccdef13ddafeef4ec2a3d532cb902aa62011e"
    )
    expected_subset = {
        "schema": "sawm/live-control-projection-repair-authorization@1",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": "SAWM-R2-M10",
        "migration_kind": (
            "authenticated_live_task_and_semantic_projection_comparator_repair"
        ),
        "supersession_mode": "source_only_live_projection_comparator_repair",
        "prior_store_id": f"{prior_root}/control.duckdb",
        "prior_control_store_sha256": (
            "dedd6ce6107b556a87b716ac59839df7f714cd73d72dc2698d38d8f1dca92d2f"
        ),
        "prior_live_checkpoint_semantic_change": False,
        "prior_publication_control_store_sha256": (
            "2cf9b143ab59a528ca4031754637a5a27dcd7e3e7d52b1db6bbaa5fc560d2fce"
        ),
        "prior_control_store_size": 45_101_056,
        "prior_coordination_store_id": f"{prior_root}/control.coordination.duckdb",
        "prior_coordination_store_sha256": (
            "e69a4c22a1a1a8bd38ccf210803a02ecb040d0ee9d4cc776165a4570425d6862"
        ),
        "prior_coordination_store_size": 9_711_616,
        "prior_coordination_event_count": 36,
        "prior_coordination_projection_digest": coordination_digest,
        "prior_event_watermark": 177,
        "prior_event_prefix_sha256": (
            "76377c47b01b15ab1c03889ec1c2e3b9ced25fa080ad39a48b6e0a4b3b1bb464"
        ),
        "prior_projection_cid": m9_projection,
        "prior_source_head": "486c5b68477c0f2ecdaf7b04a01e24629fa4b5a0",
        "prior_source_tree": "b8e0c0d928b407c71383c45ce1da2f1155996c9e",
        "prior_source_binding_cid": (
            "sha256:2a631f1b465d3d5a39636f8b588e1dfad1ebcb1cc130defff4ee7d60b9c686c6"
        ),
        "prior_database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
        "prior_generation": 11,
        "prior_plan_revision": 10,
        "prior_process_birth_id": "birth:4338518933c02a3de9372e1492954473",
        "prior_server_id": "server:2680629a-310c-4e7b-943d-4e6e977a4446",
        "prior_materialization_receipt_path": f"{prior_root}/migration-receipt.json",
        "prior_materialization_receipt_cid": (
            "sha256:f6b0475d21111091246803de4ee664d6a2cd0d5486667890f0969424abb436a1"
        ),
        "prior_materialization_receipt_file_sha256": (
            "64c1ee94ad532eff39585938d5ceab29d13ebc1989fb6c81ad131b98c3da64e9"
        ),
        "prior_owner_status_path": f"{prior_root}/quack-owner/quack-state-server.status.json",
        "prior_owner_status_sha256": (
            "b50fb27ad63ffaa3360b536e06ab9b1a0dcd726259f2dcf86c6b06ece75af87f"
        ),
        "prior_semantic_authority_digest": semantic_digest,
        "prior_frozen_base_authority_digest": (
            "sha256:e26c4db78b0e8c2914b6d9febc7dd0c9079b47a1f2c1be9b73d310fc98361032"
        ),
        "prior_append_surface_digest": (
            "sha256:7082997db9232a17411402c86fc003085367ebbde9b99068231485c754d6007c"
        ),
        "prior_catalog_digest": (
            "sha256:3cc2e066bd4495e6efc0a3410a72c5df29242dcacfa94cd75d30f61c658539a7"
        ),
        "target_store_id": f"{root}/control.duckdb",
        "target_coordination_store_id": f"{root}/control.coordination.duckdb",
        "target_generation": 12,
        "target_plan_revision": 11,
        "target_event_watermark": 179,
        "target_projection_cid": m10_projection,
        "target_semantic_authority_digest": semantic_digest,
        "target_coordination_projection_digest": coordination_digest,
        "target_coordination_event_count": 36,
        "event_suffix_length": 2,
        "task_revision_changes": 0,
        "task_status_changes": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "coordination_sidecar_copied_unchanged": True,
        "coordination_semantic_changes": 0,
        "execution_sidecar_copied": False,
        "worker_self_approval": False,
    }
    for field, expected in expected_subset.items():
        if configured.get(field) != expected:
            errors.append(f"M10 live projection {field} is not exact")

    expected_provider_strategy = {
        "capability_probe_required": True,
        "prior_route": "grok-4.6_then_codex_on_independently_verified_quota",
        "provider_result_is_completion_authority": False,
        "route_changed": False,
        "target_route": "grok-4.6_then_codex_on_independently_verified_quota",
    }
    if configured.get("provider_strategy") != expected_provider_strategy:
        errors.append("M10 provider strategy does not preserve the sealed route")

    expected_live_projection = {
        "schema": "sawm/live-task-projection-expectation@1",
        "task_count": 45,
        "task_revision_count": 1,
        "operator_task_alias": "SAWM-000",
        "operator_status": "completed",
        "operator_revision": 2,
        "candidate_task_alias": "SAWM-001",
        "candidate_task_cid": (
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
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
    }
    if configured.get("live_task_projection") != expected_live_projection:
        errors.append("M10 live task projection expectation is not exact")

    failure = configured.get("live_preflight_failure")
    expected_failure_cid = (
        "sha256:e25ac816a31b2cb239f847dae92d8017598f90190c1b551d63029ca0dc434407"
    )
    expected_error = {
        "schema": "sawm/operator-error@1",
        "valid": False,
        "error": (
            "OperatorError: live Quack task authority conflict: frozen M6 task "
            "status/revision projection differs"
        ),
    }
    if (
        configured.get("live_preflight_failure_cid") != expected_failure_cid
        or not isinstance(failure, Mapping)
        or "sha256:"
        + hashlib.sha256(_canonical_json(failure)).hexdigest()
        != expected_failure_cid
        or failure.get("schema") != "sawm/live-control-preflight-failure@1"
        or failure.get("attempt") != "SAWM-R2-M9-PREFLIGHT-A1"
        or failure.get("phase")
        != "authenticated_live_task_authority_validation"
        or failure.get("command")
        != "python scripts/ops/agent_supervisor/semantic_addressed_world_model.py preflight"
        or failure.get("exit_code") != 2
        or failure.get("error_payload") != expected_error
        or failure.get("failure_kind")
        != "historical_m6_task_status_revision_projection_mismatch"
        or failure.get("store_id") != f"{prior_root}/control.duckdb"
        or failure.get("owner_generation") != 11
        or failure.get("task_count_verified") != 45
        or failure.get("goal_count_verified") != 29
        or failure.get("dependency_count_verified") != 136
        or failure.get("plan_count_verified") != 1
        or failure.get("event_watermark_verified") != 177
        or failure.get("projection_cid_verified") != m9_projection
        or failure.get("task_cid")
        != "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        or failure.get("observed_task_status") != "retrying"
        or failure.get("observed_task_revision") != 10
        or failure.get("incorrect_comparator_status") != "todo"
        or failure.get("incorrect_comparator_revision") != 7
        or failure.get("semantic_authority_digest") != semantic_digest
        or failure.get("incorrect_semantic_comparator_digest")
        != "sha256:dd9150737feae6de30ce9b2b34968499bd24d7341d44567c43af794989dc7bbb"
        or any(
            failure.get(field) is not False
            for field in (
                "provider_probed",
                "task_claimed",
                "task_state_changed",
                "coordination_state_changed",
                "implementation_provider_invoked",
                "effect_claim_recorded",
                "implementation_commit_created",
                "merge_attempted",
                "completion_changed",
            )
        )
        or failure.get("failure_time_authority") != "unavailable"
    ):
        errors.append("M10 frozen live projection failure is not exact")

    required_repair_paths = {
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
    }
    protected = set(scheduler.get("protected_paths") or ())
    capsule = scheduler.get("configured_board_live_capsule")
    capsule_controls = set(
        capsule.get("control_paths") or ()
        if isinstance(capsule, Mapping)
        else ()
    )
    if (
        set(configured.get("bounded_control_plane_repair_paths") or ())
        != required_repair_paths
        or len(configured.get("bounded_control_plane_repair_paths") or ()) != 9
        or not required_repair_paths.issubset(CONTROL_PATHS)
        or not required_repair_paths.issubset(protected)
        or not required_repair_paths.issubset(capsule_controls)
    ):
        errors.append("M10 bounded repair paths are not exact protected controls")

    # M10 remains immutable historical authority once M11 is declared.  Key
    # presence transfers only the active runtime assertions to M11; malformed
    # or partial M11 controls are rejected by the newer fail-closed gate.
    m11_key = "live_provider_retry_successor_materialization"
    if any(
        (
            m11_key in scheduler,
            m11_key in migration,
            "live_provider_retry_successor_materialization_cid" in seal,
        )
    ):
        return errors

    expected_program = {
        "authority_mode": "quack",
        "endpoint_secret_handle": "env://SAWM_QUACK_TOKEN",
        "event_store_path": f"{root}/events",
        "explicit_legacy": False,
        "export_profile": "semantic-addressed-world-model-r2",
        "failover_policy": "fail_closed",
        "quack_endpoint": "quack:127.0.0.1:45253",
        "runtime_registry_path": f"{root}/registry",
        "schema_revision": "datasets-authoritative-operational-v1",
        "store_generation": "12",
        "store_id": f"{root}/control.duckdb",
        "task_source_kind": "duckdb",
        "worktree_root": f"{root}/worktrees",
    }
    if scheduler.get("database_program") != expected_program:
        errors.append("scheduler M10 execution authority is not exact")

    owner = scheduler.get("quack_owner")
    expected_owner_fields = {
        "automatic_direct_file_fallback": False,
        "database_path": f"{root}/control.duckdb",
        "host": "127.0.0.1",
        "live_transport_query_required": True,
        "migration_profile": "datasets-authoritative-operational-v1",
        "port": 45_253,
        "repository_id": "ipfs_accelerate_py:semantic-addressed-world-model-v1",
        "secret_handle": "env://SAWM_QUACK_TOKEN",
        "start_once_before_scheduler": True,
        "state_dir": f"{root}/quack-owner",
        "status_file_alone_is_ready": False,
        "store_id": f"{root}/control.duckdb",
    }
    if not isinstance(owner, Mapping) or any(
        owner.get(field) != expected
        for field, expected in expected_owner_fields.items()
    ):
        errors.append("scheduler M10 Quack owner authority is not exact")

    expected_history = {
        "authority": False,
        "catalog_path": f"{root}/ducklake/history.ducklake",
        "completion_prerequisite": False,
        "data_path": f"{root}/ducklake/data",
        "enabled_when_locally_available": True,
        "install_or_network_forbidden": True,
        "load_local_only": True,
        "receipt_path": f"{root}/ducklake-history-receipt.json",
        "scheduling_prerequisite": False,
        "typed_unavailability_is_permitted": True,
    }
    if scheduler.get("ducklake_history_projection") != expected_history:
        errors.append("scheduler M10 DuckLake projection is not exact")

    expected_runtime = {
        "root": root,
        "state": f"{root}/state",
        "worktrees": f"{root}/worktrees",
        "merge_queue": f"{root}/merge-queue",
        "logs": f"{root}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    if scheduler.get("runtime_paths") != expected_runtime:
        errors.append("scheduler M10 runtime paths are not exact")

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
    if scheduler.get("provider") != expected_provider:
        errors.append("scheduler M10 provider route is not exact")
    return errors


def _m11_provider_retry_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify M11's exact provider lifecycle repair and failed-task rearm."""

    errors: list[str] = []
    authority_key = "live_provider_retry_successor_materialization"
    cid_key = "live_provider_retry_successor_materialization_cid"
    if authority_key not in scheduler:
        errors.append("M11 provider retry authority key is absent from scheduler")
    if authority_key not in migration:
        errors.append("M11 provider retry authority key is absent from inventory")
    if cid_key not in seal:
        errors.append("M11 provider retry authority CID key is absent from seal")
    configured = scheduler[authority_key] if authority_key in scheduler else None
    inventoried = migration[authority_key] if authority_key in migration else None
    if type(configured) is not dict or configured != inventoried:
        errors.append("M11 provider retry authority differs across controls")
        configured = {}
    expected_cid = (
        "sha256:7f8404735adae0fb7ed890cda97dc674b78ae915efa193207e98abd95eb5193e"
    )
    observed_cid = "sha256:" + hashlib.sha256(
        _canonical_json(configured)
    ).hexdigest()
    if seal.get(cid_key) != expected_cid or observed_cid != expected_cid:
        errors.append("M11 provider retry authority CID is not exact")

    prior_root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m10"
    root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m11"
    settlement_id = (
        "baguqeera7cwinhpjgl2etuitwiuhs4ts6lix5txyb2pjtsbfz2npfyryznma"
    )
    task_cid = (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    )
    expected_subset = {
        "schema": "sawm/provider-launch-repair-authorization@1",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": "SAWM-R2-M11",
        "migration_kind": "authenticated_grok_container_lifecycle_repair",
        "supersession_mode": (
            "source_only_grok_container_lifecycle_and_retry_rearm"
        ),
        "prior_store_id": f"{prior_root}/control.duckdb",
        "prior_publication_control_store_sha256": (
            "f0106a98b390e7933ff55f3f0f5326dbb8a6951adaa84721c13beff65e0a01ec"
        ),
        "prior_control_store_sha256": (
            "c38a6477e4e87aecea1c50f68d75e7c5c7cbe1ac0f50f543a9a8c2ace55c2467"
        ),
        "prior_control_store_size": 45_101_056,
        "prior_coordination_store_id": f"{prior_root}/control.coordination.duckdb",
        "prior_coordination_store_sha256": (
            "e69a4c22a1a1a8bd38ccf210803a02ecb040d0ee9d4cc776165a4570425d6862"
        ),
        "prior_coordination_store_size": 9_711_616,
        "prior_coordination_wal_id": (
            f"{prior_root}/control.coordination.duckdb.wal"
        ),
        "prior_coordination_wal_sha256": (
            "938dbc7028ec438889019c352b5a2529cc8c0f0fd18a11f0eb157a6772fc5321"
        ),
        "prior_coordination_wal_size": 24_746,
        "prior_coordination_event_count": 55,
        "prior_coordination_projection_digest": (
            "sha256:25b8bd03cfe9a684a2626a6c5e6c955b295ba6073248c7a1cf2cd83d7550576d"
        ),
        "prior_event_watermark": 181,
        "prior_event_prefix_sha256": (
            "279fab4757fbc5bcd2966af48ee26172591ef18f32a36b184d7d981521722f11"
        ),
        "prior_projection_cid": (
            "baguqeeramqb56wqvzegrmmuchnldydvhtu3o3takpwgzh2lyd55k4lxtzboa"
        ),
        "prior_source_head": "64a7a2a8ea535f6c0adcd99fc2a3717241c9b81e",
        "prior_source_tree": "7203a523d32fc1661c70caa4b4e7329fd676c34b",
        "prior_source_binding_cid": (
            "sha256:815f0ee9029f340875aa2c605f262ba6475dfe19329d9c436837b4a53ec3b855"
        ),
        "prior_database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
        "prior_generation": 12,
        "prior_plan_revision": 11,
        "prior_process_birth_id": "birth:c461fb92e92eb447b54242c92bd92bc6",
        "prior_server_id": "server:bfc6757f-4243-4558-87d0-9eab167a72d8",
        "prior_materialization_receipt_path": f"{prior_root}/migration-receipt.json",
        "prior_materialization_receipt_cid": (
            "sha256:e83d457e93ba922a66ed586f84556e4c282f295db4a24188fc2378a18791a93d"
        ),
        "prior_materialization_receipt_file_sha256": (
            "781a48edd7d4ff18e48965be68f483c04c2bffea6c7fc01dac86fcd95d8c5774"
        ),
        "prior_owner_status_path": (
            f"{prior_root}/quack-owner/quack-state-server.status.json"
        ),
        "prior_owner_status_sha256": (
            "82aa9dcff5f3645200ef2ad6580655f79f7176b0dad1859d18fad1d839861d51"
        ),
        "prior_semantic_authority_digest": (
            "sha256:7dcdccefb3a54d4604716427bed164e36032429477612d9c92f3bf75fc4f9d46"
        ),
        "prior_frozen_base_authority_digest": (
            "sha256:86830295dfa9b17c0c15053c1caf9fa9c993137d0f198e5c61c64f8a23e6a1f6"
        ),
        "prior_append_surface_digest": (
            "sha256:342b08f9687b3511eced01e9febcb81517dccc57f3d41114eabe5819c9c7072e"
        ),
        "prior_catalog_digest": (
            "sha256:3cc2e066bd4495e6efc0a3410a72c5df29242dcacfa94cd75d30f61c658539a7"
        ),
        "target_store_id": f"{root}/control.duckdb",
        "target_coordination_store_id": f"{root}/control.coordination.duckdb",
        "target_generation": 13,
        "target_plan_revision": 12,
        "target_event_watermark": 184,
        "target_projection_cid": (
            "baguqeeratdygaminyfax543hh5bik3kj5admm5filgtu37a3jfnyc6snwnxa"
        ),
        "target_coordination_projection_digest": (
            "sha256:3b9ce361244387492da0156888cf6cb7377a020ca1f988b37509050975205ae7"
        ),
        "target_coordination_event_count": 56,
        "target_semantic_authority_digest": (
            "sha256:d9a1f5bd53884346a847cb9e440b5e8e37c7e4e7a87ea52e173d719f2933821c"
        ),
        "target_frozen_base_authority_digest": (
            "sha256:f07f9836f6442ce4b8ba0db2a8ee2142d2f33141cbd65ff47ff1b07e412baa1a"
        ),
        "event_suffix_length": 3,
        "control_recorded_at": "2026-08-28T20:35:23Z",
        "coordination_rearm_observed_at_ms": 1_787_949_321_963,
        "coordination_base_and_wal_copied": True,
        "coordination_wal_replayed_on_copy": True,
        "prior_coordination_store_mutated": False,
        "coordination_semantic_changes": 1,
        "execution_sidecar_copied": False,
        "task_revision_changes": 1,
        "task_status_changes": 1,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations_observed": 1,
        "settlement_provider_invocation_count": 0,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    for field, expected in expected_subset.items():
        if configured.get(field) != expected:
            errors.append(f"M11 provider retry {field} is not exact")

    expected_provider_strategy = {
        "capability_probe_required": True,
        "prior_route": "grok-4.6_then_codex_on_independently_verified_quota",
        "provider_result_is_completion_authority": False,
        "route_changed": False,
        "target_route": "grok-4.6_then_codex_on_independently_verified_quota",
    }
    if configured.get("provider_strategy") != expected_provider_strategy:
        errors.append("M11 provider strategy does not preserve the sealed route")

    expected_runner_repair = {
        "runner_path": (
            "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py"
        ),
        "focused_test_path": (
            "test/api/test_agent_supervisor_grok_quota_terra_gate.py"
        ),
        "failure_root": "attached_start_executed_in_typed_create_phase",
        "expected_lifecycle": (
            "typed_route_owns_create_identity_verification_and_attached_start"
        ),
        "provider_route_changed": False,
        "provider_execution_observed": True,
        "provider_execution_accounting_mismatch": True,
        "settlement_provider_invocation_count": 0,
        "effect_claim_count": 0,
    }
    if configured.get("runner_repair") != expected_runner_repair:
        errors.append("M11 runner repair contract is not exact")

    expected_rearm = {
        "schema": "sawm/control-plane-task-rearm-authorization@1",
        "operation": "operator_control_plane_repair",
        "task_alias": "SAWM-001",
        "task_cid": task_cid,
        "settlement_id": settlement_id,
        "from_status": "blocked",
        "from_revision": 12,
        "to_status": "retrying",
        "to_revision": 13,
        "automatic_retry_admitted": False,
        "coordination_rearm_required": True,
        "coordination_rearm_observed_at_ms": 1_787_949_321_963,
        "prior_coordination_event_count": 55,
        "target_coordination_event_count": 56,
        "historical_settlement_preserved": True,
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "implementation_commit_count": 0,
        "merge_attempt_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    if configured.get("task_rearm") != expected_rearm:
        errors.append("M11 blocked-head rearm authority is not exact")

    expected_live_projection = {
        "schema": "sawm/live-task-projection-expectation@1",
        "task_count": 45,
        "task_revision_count": 1,
        "operator_task_alias": "SAWM-000",
        "operator_status": "completed",
        "operator_revision": 2,
        "candidate_task_alias": "SAWM-001",
        "candidate_task_cid": task_cid,
        "prior_candidate_status": "blocked",
        "prior_candidate_revision": 12,
        "target_candidate_status": "retrying",
        "target_candidate_revision": 13,
        "target_candidate_completion_receipt": {
            "operation": "operator_control_plane_repair",
            "settlement_id": settlement_id,
        },
        "remaining_task_status": "todo",
        "remaining_task_revision": 2,
    }
    if configured.get("live_task_projection") != expected_live_projection:
        errors.append("M11 live task projection expectation is not exact")

    failure = configured.get("live_implementation_failure")
    expected_failure_cid = (
        "sha256:a1fae6ef892b02a19b1155a0a5ed7c0eefa8f8255f3949e25d5a8b8b9d721676"
    )
    failure_receipt = failure.get("failure_receipt") if isinstance(failure, Mapping) else {}
    if (
        configured.get("live_implementation_failure_cid") != expected_failure_cid
        or not isinstance(failure, Mapping)
        or "sha256:" + hashlib.sha256(_canonical_json(failure)).hexdigest()
        != expected_failure_cid
        or failure.get("schema") != "sawm/live-control-implementation-failure@2"
        or failure.get("attempt") != "SAWM-R2-M10-LIVE-A1"
        or failure.get("phase") != "implementation_execution"
        or failure.get("failure_kind")
        != "attached_provider_execution_misclassified_as_container_create_timeout"
        or failure.get("task_alias") != "SAWM-001"
        or failure.get("task_cid") != task_cid
        or failure.get("settlement_id") != settlement_id
        or failure.get("from_status") != "retrying"
        or failure.get("claim_status") != "in_progress"
        or failure.get("terminal_status") != "blocked"
        or failure.get("from_revision") != 10
        or failure.get("claim_revision") != 11
        or failure.get("terminal_revision") != 12
        or failure.get("portal_return_code") != 127
        or failure.get("provider_execution_observed") is not True
        or failure.get("provider_execution_accounting_mismatch") is not True
        or failure.get("settlement_provider_invocation_count") != 0
        or any(
            failure.get(field) != 0
            for field in (
                "effect_claim_count",
                "implementation_commit_count",
                "merge_attempt_count",
                "validation_run_count",
            )
        )
        or failure.get("worker_self_approval") is not False
        or not isinstance(failure_receipt, Mapping)
        or failure_receipt.get("settlement_id") != settlement_id
        or failure_receipt.get("provider_invocation_count") != 0
        or failure_receipt.get("effect_claim_count") != 0
        or failure_receipt.get("automatic_retry_admitted") is not False
        or failure_receipt.get("control_expected_status") != "in_progress"
        or failure_receipt.get("control_expected_revision") != 11
    ):
        errors.append("M11 frozen provider-launch failure is not exact")

    required_repair_paths = {
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        (
            "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ),
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
        "test/api/test_agent_supervisor_grok_quota_terra_gate.py",
    }
    protected = set(scheduler.get("protected_paths") or ())
    capsule = scheduler.get("configured_board_live_capsule")
    capsule_controls = set(
        capsule.get("control_paths") or () if isinstance(capsule, Mapping) else ()
    )
    if (
        set(configured.get("bounded_control_plane_repair_paths") or ())
        != required_repair_paths
        or len(configured.get("bounded_control_plane_repair_paths") or ()) != 11
        or not required_repair_paths.issubset(CONTROL_PATHS)
        or not required_repair_paths.issubset(protected)
        or not required_repair_paths.issubset(capsule_controls)
    ):
        errors.append("M11 bounded repair paths are not exact protected controls")

    # M11 remains immutable historical authority once M12 is declared.  Only
    # the adjacent runtime pair moves forward; malformed M12 controls are
    # rejected by the newer fail-closed gate.
    m12_key = "declared_output_retry_successor_materialization"
    if any(
        (
            m12_key in scheduler,
            m12_key in migration,
            "declared_output_retry_successor_materialization_cid" in seal,
        )
    ):
        return errors

    expected_program = {
        "authority_mode": "quack",
        "endpoint_secret_handle": "env://SAWM_QUACK_TOKEN",
        "event_store_path": f"{root}/events",
        "explicit_legacy": False,
        "export_profile": "semantic-addressed-world-model-r2",
        "failover_policy": "fail_closed",
        "quack_endpoint": "quack:127.0.0.1:45255",
        "runtime_registry_path": f"{root}/registry",
        "schema_revision": "datasets-authoritative-operational-v1",
        "store_generation": "13",
        "store_id": f"{root}/control.duckdb",
        "task_source_kind": "duckdb",
        "worktree_root": f"{root}/worktrees",
    }
    if scheduler.get("database_program") != expected_program:
        errors.append("scheduler M11 execution authority is not exact")

    owner = scheduler.get("quack_owner")
    expected_owner_fields = {
        "automatic_direct_file_fallback": False,
        "database_path": f"{root}/control.duckdb",
        "host": "127.0.0.1",
        "live_transport_query_required": True,
        "migration_profile": "datasets-authoritative-operational-v1",
        "port": 45_255,
        "repository_id": "ipfs_accelerate_py:semantic-addressed-world-model-v1",
        "secret_handle": "env://SAWM_QUACK_TOKEN",
        "start_once_before_scheduler": True,
        "state_dir": f"{root}/quack-owner",
        "status_file_alone_is_ready": False,
        "store_id": f"{root}/control.duckdb",
    }
    if not isinstance(owner, Mapping) or any(
        owner.get(field) != expected
        for field, expected in expected_owner_fields.items()
    ):
        errors.append("scheduler M11 Quack owner authority is not exact")

    expected_history = {
        "authority": False,
        "catalog_path": f"{root}/ducklake/history.ducklake",
        "completion_prerequisite": False,
        "data_path": f"{root}/ducklake/data",
        "enabled_when_locally_available": True,
        "install_or_network_forbidden": True,
        "load_local_only": True,
        "receipt_path": f"{root}/ducklake-history-receipt.json",
        "scheduling_prerequisite": False,
        "typed_unavailability_is_permitted": True,
    }
    if scheduler.get("ducklake_history_projection") != expected_history:
        errors.append("scheduler M11 DuckLake projection is not exact")

    expected_runtime = {
        "root": root,
        "state": f"{root}/state",
        "worktrees": f"{root}/worktrees",
        "merge_queue": f"{root}/merge-queue",
        "logs": f"{root}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    if scheduler.get("runtime_paths") != expected_runtime:
        errors.append("scheduler M11 runtime paths are not exact")

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
    if scheduler.get("provider") != expected_provider:
        errors.append("scheduler M11 provider route is not exact")
    return errors


def _m12_declared_output_retry_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify M12's exact Portal declared-output repair and failed-task rearm."""

    errors: list[str] = []
    authority_key = "declared_output_retry_successor_materialization"
    cid_key = "declared_output_retry_successor_materialization_cid"
    if authority_key not in scheduler:
        errors.append("M12 declared-output retry authority key is absent from scheduler")
    if authority_key not in migration:
        errors.append("M12 declared-output retry authority key is absent from inventory")
    if cid_key not in seal:
        errors.append("M12 declared-output retry authority CID key is absent from seal")
    configured = scheduler[authority_key] if authority_key in scheduler else None
    inventoried = migration[authority_key] if authority_key in migration else None
    if type(configured) is not dict or configured != inventoried:
        errors.append("M12 declared-output retry authority differs across controls")
        configured = {}
    expected_cid = (
        "sha256:786dde1f1728b907c3e28e5a09c746842c0a4a5ac0333f3ccb25f5290e8227a8"
    )
    observed_cid = "sha256:" + hashlib.sha256(
        _canonical_json(configured)
    ).hexdigest()
    if seal.get(cid_key) != expected_cid or observed_cid != expected_cid:
        errors.append("M12 declared-output retry authority CID is not exact")

    prior_root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m11"
    root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m12"
    expected_subset = {
        "schema": "sawm/portal-declared-output-repair-authorization@1",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": "SAWM-R2-M12",
        "migration_kind": "database_portal_declared_path_projection_repair",
        "supersession_mode": (
            "source_only_database_portal_declared_path_adapter_and_retry_rearm"
        ),
        "prior_store_id": f"{prior_root}/control.duckdb",
        "prior_publication_control_store_sha256": (
            "5accfe02488f9216b3f110a35f672cce98b24762cfeca1d8918cc1c182d7a3d7"
        ),
        "prior_control_store_sha256": (
            "d32a30bf320a07b2ebf9ebd1ee66012651346b51247a2fbec983a91772f4ee48"
        ),
        "prior_control_store_size": 45_101_056,
        "prior_control_wal_present": False,
        "prior_coordination_store_id": f"{prior_root}/control.coordination.duckdb",
        "prior_publication_coordination_store_sha256": (
            "c7d2fdce85eb9f7003feff0205a7dd3fbccfeffb5c3da7c821c0de658dc06af0"
        ),
        "prior_coordination_store_sha256": (
            "0c7335f3e9545859ad2da7ebc384eedee2302fd9a69b38e8e4df459b4930a0d5"
        ),
        "prior_coordination_store_size": 12_857_344,
        "prior_coordination_wal_present": False,
        "prior_coordination_event_count": 223,
        "prior_coordination_projection_digest": (
            "sha256:b27edbdc376ccab7a167a65e13967738a30f79b42affebeb1c94710db869d963"
        ),
        "prior_event_watermark": 186,
        "prior_event_prefix_sha256": (
            "0ef7e38a73e68b5c393151dafa5b5c9a6b793fcbf5eea3b610bbd65b81e97fac"
        ),
        "prior_projection_cid": (
            "baguqeeraj42oauw2k2dqekni2dsdgh6sfgbewkqpyintvry4sfzxmiu3pkfa"
        ),
        "prior_source_head": "6e5fcd42b2c5e4d1ebbc883918a4c8cbe42a12ef",
        "prior_source_tree": "22c02cd10b6340e85ce517db7590bd9cce9e122b",
        "prior_source_binding_cid": (
            "sha256:8772a16513abfb8462c2bb91b79f2e9dee2fbdd7d130b4cbd3a9155deb16ab80"
        ),
        "prior_generation": 13,
        "prior_plan_revision": 12,
        "prior_materialization_receipt_path": f"{prior_root}/migration-receipt.json",
        "prior_materialization_receipt_cid": (
            "sha256:f3035739011ef03260d8c24671920629f29bbf321b2c18a16f05e6e65e315d5b"
        ),
        "prior_materialization_receipt_file_sha256": (
            "e781baaea85357ab1daf40f65139d20dd6bcfd31a609a9999815e93940af0f84"
        ),
        "prior_materialization_receipt_is_publication_marker": True,
        "prior_materialization_receipt_precedes_live_suffix": True,
        "prior_semantic_authority_digest": (
            "sha256:44bf091895a4bc1fb2c5f0ecb95949dd01a4c5bcd1d5b4c909b49a3806978086"
        ),
        "prior_frozen_base_authority_digest": (
            "sha256:23e1e957dea4674abbfd0b6278872aef3c7be51f4d2bfcd6bc06c758b900c732"
        ),
        "prior_append_surface_digest": (
            "sha256:9f8670c078b905833a5f0cb0cc8fba00732ca60989f23862b12d1770da135660"
        ),
        "target_store_id": f"{root}/control.duckdb",
        "target_coordination_store_id": f"{root}/control.coordination.duckdb",
        "target_generation": 14,
        "target_quack_port": 45_256,
        "target_plan_revision": 13,
        "target_event_watermark": 189,
        "target_projection_cid": (
            "baguqeeragb5uufggmw6glss2ttbubyb2aixhfio6cmmit6w4njiuozj3csmq"
        ),
        "target_coordination_projection_digest": (
            "sha256:358cd0667be10fb125476ac090db3cda9aec9b8d0876a2654de1ce5aa531c59f"
        ),
        "target_coordination_event_count": 224,
        "target_semantic_authority_digest": (
            "sha256:239db939a5e7af260f334b325c4ec075a2faf18f01b683448625647e0962d36c"
        ),
        "target_frozen_base_authority_digest": (
            "sha256:12b25f50a7c5d3b1020b7c1f86412fa863a6c00027ee780ecd1f78c88dfa7a95"
        ),
        "event_suffix_length": 3,
        "control_recorded_at": "2026-08-28T23:01:48Z",
        "coordination_rearm_observed_at_ms": 1_787_958_108_000,
        "control_base_copied": True,
        "coordination_base_copied": True,
        "prior_control_wal_copied": False,
        "prior_coordination_wal_copied": False,
        "prior_wals_absent": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "coordination_semantic_changes": 1,
        "execution_sidecar_copied": False,
        "read_replica_sidecar_copied": False,
        "task_revision_changes": 1,
        "task_status_changes": 1,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "implementation_provider_invocations_observed": 2,
        "implementation_provider_model_calls_observed": 92,
        "implementation_provider_tokens_observed": 11_202_083,
        "implementation_provider_cost_usd_observed": "1.23726850",
        "settlement_provider_invocation_count": 0,
        "provider_execution_accounting_mismatch": True,
        "effect_claim_changes": 0,
        "implementation_commit_changes": 0,
        "merge_attempt_changes": 0,
        "worker_self_approval": False,
    }
    for field, expected in expected_subset.items():
        if configured.get(field) != expected:
            errors.append(f"M12 declared-output retry {field} is not exact")

    settlement_id = (
        "baguqeerafg6ci2g5mrq7ilfpa2qvinghhjqq2hdzmfprut4sbdiqosoxi5pq"
    )
    task_cid = (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    )
    rearm = configured.get("task_rearm")
    expected_rearm = {
        "schema": "sawm/control-plane-task-rearm-authorization@1",
        "operation": "operator_control_plane_repair",
        "task_alias": "SAWM-001",
        "task_cid": task_cid,
        "settlement_id": settlement_id,
        "from_status": "blocked",
        "from_revision": 15,
        "to_status": "retrying",
        "to_revision": 16,
        "automatic_retry_admitted": False,
        "coordination_rearm_required": True,
        "coordination_rearm_observed_at_ms": 1_787_958_108_000,
        "prior_coordination_event_count": 223,
        "target_coordination_event_count": 224,
        "historical_settlement_preserved": True,
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "implementation_commit_count": 0,
        "merge_attempt_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "worker_self_approval": False,
    }
    if rearm != expected_rearm:
        errors.append("M12 blocked-head rearm authority is not exact")

    adapter = configured.get("declared_output_adapter_repair")
    expected_adapter = {
        "adapter_path": (
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "database_portal_bridge.py"
        ),
        "focused_test_path": "test/api/test_agent_supervisor_database_portal_bridge.py",
        "failure_root": "effect_identity_selected_instead_of_declared_path",
        "canonical_field": "effect.declared_path",
        "legacy_fallback_field": "path",
        "provider_envelope_changed": False,
        "proposal_gate_changed": False,
        "declared_output_paths_restored": True,
    }
    if adapter != expected_adapter:
        errors.append("M12 declared-output adapter repair contract is not exact")

    failure = configured.get("live_implementation_failure")
    if not isinstance(failure, Mapping):
        failure = {}
        errors.append("M12 frozen live implementation failure is absent")
    failure_receipt = failure.get("failure_receipt")
    if (
        configured.get("live_implementation_failure_cid")
        != "sha256:" + hashlib.sha256(_canonical_json(failure)).hexdigest()
        or failure.get("schema") != "sawm/live-control-implementation-failure@3"
        or failure.get("task_cid") != task_cid
        or failure.get("settlement_id") != settlement_id
        or failure.get("failure_kind")
        != "declared_output_paths_projected_as_effect_identities"
        or failure.get("terminal_status") != "blocked"
        or failure.get("terminal_revision") != 15
        or failure.get("provider_executions_observed") != 2
        or failure.get("provider_model_calls_observed") != 92
        or failure.get("provider_tokens_observed") != 11_202_083
        or failure.get("provider_cost_usd_observed") != "1.23726850"
        or failure.get("provider_accounting_distinct_from_settlement") is not True
        or failure.get("settlement_provider_invocation_count") != 0
        or not isinstance(failure_receipt, Mapping)
        or failure_receipt.get("provider_invocation_count") != 0
        or failure_receipt.get("effect_claim_count") != 0
        or failure_receipt.get("settlement_id") != settlement_id
    ):
        errors.append("M12 frozen declared-output failure is not exact")

    required_repair_paths = {
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        (
            "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
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
    }
    protected = set(scheduler.get("protected_paths") or ())
    capsule = scheduler.get("configured_board_live_capsule")
    capsule_controls = set(
        capsule.get("control_paths") or () if isinstance(capsule, Mapping) else ()
    )
    if (
        set(configured.get("bounded_control_plane_repair_paths") or ())
        != required_repair_paths
        or len(configured.get("bounded_control_plane_repair_paths") or ()) != 11
        or not required_repair_paths.issubset(CONTROL_PATHS)
        or not required_repair_paths.issubset(protected)
        or not required_repair_paths.issubset(capsule_controls)
    ):
        errors.append("M12 bounded repair paths are not exact protected controls")

    bridge_source = (
        REPO_ROOT
        / "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py"
    ).read_text(encoding="utf-8")
    for required_source in (
        "_canonical_declared_output_path",
        "_nested_declared_output_path",
        'effect.get("effect") != "declared_output"',
        "value.splitlines() != [value]",
        '{"none", "n/a"}',
    ):
        if required_source not in bridge_source:
            errors.append(
                f"M12 Portal adapter source omits {required_source}"
            )

    # M12 remains immutable historical authority once M13 is declared.  The
    # M13 gate validates the adjacent active runtime binding; malformed M13
    # controls cannot reactivate this predecessor.
    m13_key = "quack_refresh_successor_materialization"
    if any(
        (
            m13_key in scheduler,
            m13_key in migration,
            "quack_refresh_successor_materialization_cid" in seal,
        )
    ):
        return errors

    expected_program = {
        "authority_mode": "quack",
        "endpoint_secret_handle": "env://SAWM_QUACK_TOKEN",
        "event_store_path": f"{root}/events",
        "explicit_legacy": False,
        "export_profile": "semantic-addressed-world-model-r2",
        "failover_policy": "fail_closed",
        "quack_endpoint": "quack:127.0.0.1:45256",
        "runtime_registry_path": f"{root}/registry",
        "schema_revision": "datasets-authoritative-operational-v1",
        "store_generation": "14",
        "store_id": f"{root}/control.duckdb",
        "task_source_kind": "duckdb",
        "worktree_root": f"{root}/worktrees",
    }
    if scheduler.get("database_program") != expected_program:
        errors.append("scheduler M12 execution authority is not exact")
    owner = scheduler.get("quack_owner")
    expected_owner_fields = {
        "automatic_direct_file_fallback": False,
        "database_path": f"{root}/control.duckdb",
        "host": "127.0.0.1",
        "live_transport_query_required": True,
        "migration_profile": "datasets-authoritative-operational-v1",
        "port": 45_256,
        "repository_id": "ipfs_accelerate_py:semantic-addressed-world-model-v1",
        "secret_handle": "env://SAWM_QUACK_TOKEN",
        "start_once_before_scheduler": True,
        "state_dir": f"{root}/quack-owner",
        "status_file_alone_is_ready": False,
        "store_id": f"{root}/control.duckdb",
    }
    if not isinstance(owner, Mapping) or any(
        owner.get(field) != expected
        for field, expected in expected_owner_fields.items()
    ):
        errors.append("scheduler M12 Quack owner authority is not exact")
    expected_history = {
        "authority": False,
        "catalog_path": f"{root}/ducklake/history.ducklake",
        "completion_prerequisite": False,
        "data_path": f"{root}/ducklake/data",
        "enabled_when_locally_available": True,
        "install_or_network_forbidden": True,
        "load_local_only": True,
        "receipt_path": f"{root}/ducklake-history-receipt.json",
        "scheduling_prerequisite": False,
        "typed_unavailability_is_permitted": True,
    }
    if scheduler.get("ducklake_history_projection") != expected_history:
        errors.append("scheduler M12 DuckLake projection is not exact")
    expected_runtime = {
        "root": root,
        "state": f"{root}/state",
        "worktrees": f"{root}/worktrees",
        "merge_queue": f"{root}/merge-queue",
        "logs": f"{root}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    if scheduler.get("runtime_paths") != expected_runtime:
        errors.append("scheduler M12 runtime paths are not exact")
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
    if scheduler.get("provider") != expected_provider:
        errors.append("scheduler M12 provider route is not exact")

    return errors


def _m18_portal_completion_persistence_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M18's stopped-M17, portal repair, and one-task rearm controls."""

    key = "portal_completion_persistence_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m18_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M18 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m18_portal_completion_persistence_authority()
        )
        errors: list[str] = []
        if (
            type(scheduler.get(key)) is not dict
            or materializer._identity(scheduler.get(key))
            != materializer._identity(expected)
            or type(migration.get(key)) is not dict
            or materializer._identity(migration.get(key))
            != materializer._identity(expected)
        ):
            errors.append("M18 portal-completion authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M18 portal-completion authority CID is not exact")

        required = {
            "schema": "sawm/portal-completion-persistence-repair-authorization@1",
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M18",
            "migration_kind": "portal_completion_persistence_repair",
            "supersession_mode": (
                "append_only_portal_completion_persistence_repair_and_exact_rearm"
            ),
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m17/control.duckdb"
            ),
            "prior_control_store_sha256": (
                "6ae3d7ba8196c84c9dc9b50b60f10888fbcfdd07a6985010d4257975a283f814"
            ),
            "prior_coordination_store_sha256": (
                "bfb52452ae7f3bc4390a519a782ee02cb496c463f989a7d2255f02e6aa25e630"
            ),
            "prior_event_watermark": 222,
            "prior_plan_revision": 18,
            "prior_generation": 18,
            "prior_owner_marker_present": False,
            "prior_stop_control_present": False,
            "prior_token_handoff_present": False,
            "prior_wals_absent": True,
            "prior_listener_present": False,
            "prior_pid_present": False,
            "prior_active_count": 0,
            "repair_precursor_source_commit": (
                "d24d6dd2ae955d115da1c34150c2a84f83d1d799"
            ),
            "repair_source_commit": (
                "37f45a54ed5b93dec356cfe3e1f81f18deb567a9"
            ),
            "repair_source_tree": "875340accca120b6f2382735ede7c9ca7096d99b",
            "target_runtime_root": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m18"
            ),
            "target_generation": 19,
            "target_quack_port": 24_061,
            "target_plan_revision": 19,
            "target_event_watermark": 225,
            "target_projection_cid": (
                "baguqeeraqsmnfs6rzwc6bvjjdszmryjtdrtg5aosnk2wdsppaxymsrrpqyxa"
            ),
            "target_coordination_event_count": 1_125,
            "event_suffix_length": 3,
            "coordination_semantic_changes": 1,
            "task_revision_changes": 1,
            "task_status_changes": 1,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if any(expected.get(name) != value for name, value in required.items()):
            errors.append("M18 portal-completion authority is not exact")
        repair = expected.get("accepted_source_repair")
        wanted_blobs = {
            (
                "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
                "database_portal_bridge.py"
            ): "5ff1def4410f184c3f41f1b468e230df3a5e4794",
            "test/api/test_agent_supervisor_database_portal_bridge.py": (
                "91459021212f4bc4366950def56962c2ce623a89"
            ),
        }
        if (
            not isinstance(repair, Mapping)
            or repair.get("blob_oids") != wanted_blobs
            or set(repair.get("changed_paths", ())) != set(wanted_blobs)
            or repair.get("repository_authority_weakened") is not False
        ):
            errors.append("M18 exact fail-closed portal repair blobs differ")
        rearms = expected.get("task_rearms")
        failure = expected.get("failure_receipts")
        if (
            not isinstance(rearms, Mapping)
            or set(rearms) != {"SAWM-007"}
            or not isinstance(failure, Mapping)
            or set(failure) != {"SAWM-007"}
            or rearms["SAWM-007"].get("settlement_id")
            != failure["SAWM-007"].get("settlement_id")
            or rearms["SAWM-007"].get("from_revision") != 4
            or rearms["SAWM-007"].get("to_revision") != 5
            or rearms["SAWM-007"].get("to_status") != "retrying"
            or rearms["SAWM-007"].get("worker_self_approval") is not False
        ):
            errors.append("M18 exact SAWM-007 rearm authority differs")
        try:
            if require_active_runtime:
                current_head = _git(root, "rev-parse", "HEAD")
                if "live_catalog_inventory_successor_materialization" in scheduler:
                    current_head = str(
                        materializer._expected_m19_live_catalog_inventory_authority()[
                            "prior_source_head"
                        ]
                    )
                materializer._assert_m18_source_delta(
                    root,
                    {
                        "source_binding": {
                            "head": current_head,
                            "tree": _git(
                                root, "rev-parse", f"{current_head}^{{tree}}"
                            ),
                            "datasets_gitlink": _git(
                                root,
                                "rev-parse",
                                f"{current_head}:ipfs_datasets_py",
                            ),
                            "kit_gitlink": _git(
                                root, "rev-parse", f"{current_head}:ipfs_kit_py"
                            ),
                        }
                    },
                    expected,
                )
            else:
                _assert_m23_historical_source_handoff(
                    materializer, scheduler, seal, migration, root=root
                )
        except Exception as exc:
            errors.append(
                "M18 exact repair/source seal differs: "
                f"{type(exc).__name__}: {exc}"
            )

        target_root = str(expected["target_runtime_root"])
        target_store = str(expected["target_store_id"])
        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "19",
                "quack:127.0.0.1:24061",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_061,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M18 target/runtime binding is not exact")
        return errors
    except Exception as exc:
        return [
            "M18 portal-completion authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m46_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Return true when any protected surface declares the M46 successor."""

    key = _M46_SUCCESSOR_KEY
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m46_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    """Require M46's exact repair child and protected-control reseal."""

    try:
        chain = authority.get("source_chain")
        if (
            not isinstance(chain, Mapping)
            or chain.get("m45_final_control_commit")
            != _M46_M45_FINAL_CONTROL_COMMIT
            or chain.get("repair_parent") != _M46_M45_FINAL_CONTROL_COMMIT
            or chain.get("repair_commit") != _M46_REPAIR_COMMIT
            or chain.get("repair_tree") != _M46_REPAIR_TREE
            or chain.get("repair_diff_sha256") != _M46_REPAIR_DIFF_SHA256
            or chain.get("repair_blobs") != _M46_REPAIR_BLOBS
            or chain.get("repair_modes") != _M46_REPAIR_MODES
            or chain.get("final_control_parent") != _M46_REPAIR_COMMIT
            or int(chain.get("repair_commit_count") or 0) != 1
            or int(chain.get("final_control_commit_count") or 0) != 1
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M46_OPERATOR_CONTROL_PATHS)
        ):
            raise RuntimeError("M46 sealed source-chain identity fields differ")
        population = materializer.build_population(root)
        materializer._assert_m46_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M46 exact legacy-rescue repair/control source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m46_legacy_no_delta_rescue_recovery_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Validate M46 while retaining M45 as immutable received history."""

    key = _M46_SUCCESSOR_KEY
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m46_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M46 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer
            ._expected_m46_legacy_no_delta_rescue_recovery_successor_authority()
        )
        contract = materializer._validated_m46_live_preflight_contract(expected)
        reference = dict(materializer._m46_authority_reference())
        configured_cid = str(materializer._M46_AUTHORITY_CID)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M46 successor authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M46 successor reference differs")
        if seal.get(f"{key}_cid") != configured_cid:
            errors.append("M46 successor CID differs")
        if configured_cid == _M46_UNSEALED_AUTHORITY_CID:
            errors.append("M46 final control identities are not resealed")
        elif configured_cid != _M46_AUTHORITY_CID:
            errors.append("M46 successor authority CID is not the sealed identity")
        elif materializer._identity(expected) != configured_cid:
            errors.append("M46 successor authority CID does not bind its body")
        elif len(materializer._canonical(expected)) != _M46_AUTHORITY_SIZE:
            errors.append("M46 successor authority canonical size differs")

        try:
            materializer._assert_m46_historical_m45_controls(
                scheduler, migration, seal
            )
        except Exception as exc:
            errors.append(
                "M46 historical M45 control verification differs: "
                f"{type(exc).__name__}: {exc}"
            )

        prior = expected.get("prior_authority", {})
        stopped = expected.get("stopped_owner", {})
        artifacts = expected.get("stopped_prestart_artifacts", {})
        receipt = expected.get("preserved_m45_receipt", {})
        preserved = expected.get("preserved_m45_materialization", {})
        repair = expected.get("accepted_legacy_no_delta_rescue_repair", {})
        target = expected.get("target_authority", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        required_functions = (
            "_assert_m46_historical_m45_controls",
            "_assert_m46_source_delta",
            "_validated_m46_live_preflight_contract",
            "_check_m46_prestart_admission",
            "_verify_m46_live_materialization",
            "_expected_m46_source_successor_receipt",
            "_materialize_m46",
            "_check_m46_materialized",
        )
        if (
            expected.get("schema")
            != "sawm/legacy-no-delta-rescue-recovery-authorization@1"
            or expected.get("migration_revision") != _M46_MIGRATION_REVISION
            or expected.get("migration_kind") != key
            or expected.get("authorized") is not True
            or expected.get("authority") != "operator_control_plane"
            or expected.get("supersession_mode")
            != materializer._M46_SUPERSESSION_MODE
            or int(expected.get("target_generation") or 0) != 33
            or int(expected.get("target_event_watermark") or 0) != 303
            or int(expected.get("target_plan_revision") or 0) != 28
            or expected.get("target_projection_cid")
            != _M46_TARGET_PROJECTION_CID
            or int(prior.get("event_watermark") or 0) != 302
            or prior.get("projection_cid") != _M44_TARGET_PROJECTION_CID
            or stopped.get("status") != "stopped"
            or int(stopped.get("generation") or 0) != 32
            or int(stopped.get("target_generation") or 0) != 33
            or stopped.get("server_id") != materializer._M46_PRIOR_SERVER_ID
            or stopped.get("process_birth_id")
            != materializer._M46_PRIOR_PROCESS_BIRTH_ID
            or stopped.get("started_at") != materializer._M46_PRIOR_STARTED_AT
            or stopped.get("stopped_at") != materializer._M46_PRIOR_STOPPED_AT
            or int(stopped.get("startup_epoch") or 0)
            != materializer._M46_PRIOR_STARTUP_EPOCH
            or artifacts.get("control_sha256")
            != materializer._M46_PRIOR_CONTROL_SHA256
            or artifacts.get("control_size") != materializer._M46_PRIOR_CONTROL_SIZE
            or artifacts.get("coordination_sha256")
            != materializer._M46_PRIOR_COORDINATION_SHA256
            or artifacts.get("coordination_size")
            != materializer._M46_PRIOR_COORDINATION_SIZE
            or artifacts.get("status_sha256")
            != materializer._M46_STOPPED_STATUS_SHA256
            or artifacts.get("status_size") != materializer._M46_STOPPED_STATUS_SIZE
            or artifacts.get("owner_marker_absent") is not True
            or artifacts.get("stop_control_absent") is not True
            or artifacts.get("token_handoff_absent") is not True
            or artifacts.get("control_wal_absent") is not True
            or artifacts.get("coordination_wal_absent") is not True
            or artifacts.get("m46_receipt_absent") is not True
            or artifacts.get("m44_receipt_absent") is not True
            or receipt.get("receipt_cid") != _M46_M45_RECEIPT_CID
            or receipt.get("sha256") != _M46_M45_RECEIPT_SHA256
            or receipt.get("size") != _M46_M45_RECEIPT_SIZE
            or receipt.get("mode") != "0600"
            or receipt.get("created_or_rewritten") is not False
            or preserved.get("authority_cid") != _M45_AUTHORITY_CID
            or preserved.get("receipt_cid") != _M46_M45_RECEIPT_CID
            or int(preserved.get("event_watermark") or 0) != 302
            or preserved.get("m44_receipt_absent") is not True
            or repair.get("repair_parent") != _M46_M45_FINAL_CONTROL_COMMIT
            or repair.get("repair_commit") != _M46_REPAIR_COMMIT
            or repair.get("repair_tree") != _M46_REPAIR_TREE
            or repair.get("binary_diff_sha256") != _M46_REPAIR_DIFF_SHA256
            or repair.get("blob_oids") != _M46_REPAIR_BLOBS
            or repair.get("path_modes") != _M46_REPAIR_MODES
            or repair.get("status_empty_or_nested_gitlink_only_required")
            is not True
            or repair.get("attestation_metadata_validated_exactly") is not True
            or repair.get("mutation_authority") is not False
            or repair.get("merge_authority") is not False
            or repair.get("task_completion_authority") is not False
            or repair.get("validation_weakened") is not False
            or repair.get("authority_weakened") is not False
            or int(target.get("event_watermark") or 0) != 303
            or target.get("projection_cid") != _M46_TARGET_PROJECTION_CID
            or contract.get("migration_revision") != _M46_MIGRATION_REVISION
            or int(contract.get("prior_generation") or 0) != 32
            or int(contract.get("target_generation") or 0) != 33
            or int(contract.get("prior_event_watermark") or 0) != 302
            or int(contract.get("target_event_watermark") or 0) != 303
            or contract.get("prior_projection_cid") != _M44_TARGET_PROJECTION_CID
            or contract.get("target_projection_cid") != _M46_TARGET_PROJECTION_CID
            or contract.get("m45_receipt_must_be_preserved") is not True
            or contract.get("m44_receipt_must_remain_absent") is not True
            or contract.get("event_302_must_be_preserved") is not True
            or contract.get("event_303_must_be_absent_before_append") is not True
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or changes.get("evidence_event_changes") != 1
            or changes.get("store_generation_row_changes") != 1
            or changes.get("state_server_row_changes") != 1
            or changes.get("task_revision_changes") != 0
            or changes.get("task_status_changes") != 0
            or changes.get("goal_revision_changes") != 0
            or changes.get("plan_revision_changes") != 0
            or changes.get("coordination_semantic_changes") != 0
            or changes.get("accepted_completion_changes") != 0
            or changes.get("worker_self_approval") is not False
            or preservation.get("generation_32_preserved_stopped") is not True
            or preservation.get("m45_authority_preserved_exactly") is not True
            or preservation.get("m45_receipt_preserved_exactly") is not True
            or preservation.get("m45_receipt_created_or_rewritten") is not False
            or preservation.get("m44_receipt_remains_absent") is not True
            or preservation.get("event_302_preserved_exactly") is not True
            or preservation.get("coordination_store_bytes_preserved_exactly")
            is not True
            or preservation.get("worker_self_approval") is not False
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M46 legacy no-delta rescue recovery delta is not exact")
        if require_active_runtime:
            runtime = scheduler.get("runtime_paths")
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            target_root = str(expected["target_runtime_root"])
            if (
                not isinstance(program, Mapping)
                or program.get("store_generation") != "33"
                or program.get("store_id") != expected["target_store_id"]
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M46 target/runtime binding is not exact")
        errors.extend(_m46_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M46 legacy-no-delta rescue recovery authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m45_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Return true when any protected surface declares the M45 successor."""

    key = _M45_SUCCESSOR_KEY
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m45_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    """Require M45's exact test repair and protected-control reseal."""

    try:
        required = (
            "_assert_m45_source_delta",
            "_M45_OPERATOR_CONTROL_PATHS",
        )
        if any(not hasattr(materializer, name) for name in required):
            raise RuntimeError("M45 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        materializer._assert_m45_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M45 exact test-repair/control source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m45_failed_pre_authoritative_m44_validation_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Validate M45 while treating M44 as immutable historical authority."""

    key = _M45_SUCCESSOR_KEY
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m45_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M45 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer
            ._expected_m45_failed_pre_authoritative_m44_validation_successor_authority()
        )
        contract = materializer._validated_m45_live_preflight_contract(expected)
        reference = dict(materializer._m45_authority_reference())
        configured_cid = str(materializer._M45_AUTHORITY_CID)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M45 successor authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M45 successor reference differs")
        if seal.get(f"{key}_cid") != configured_cid:
            errors.append("M45 successor CID differs")
        if configured_cid == _M45_UNSEALED_AUTHORITY_CID:
            errors.append("M45 final control identities are not resealed")
        elif configured_cid != _M45_AUTHORITY_CID:
            errors.append("M45 successor authority CID is not the sealed identity")
        elif materializer._identity(expected) != configured_cid:
            errors.append("M45 successor authority CID does not bind its body")
        elif len(materializer._canonical(expected)) != _M45_AUTHORITY_SIZE:
            errors.append("M45 successor authority canonical size differs")

        # M44 remains exact history under M45.  Its active current-HEAD source
        # gate is intentionally not entered after the accepted test repair.
        try:
            materializer._assert_m45_historical_m44_controls(
                scheduler, migration, seal
            )
        except Exception as exc:
            errors.append(
                "M45 historical M44 control verification differs: "
                f"{type(exc).__name__}: {exc}"
            )

        preserved = expected.get("preserved_m44_authority", {})
        failed = expected.get("failed_m44_prepublication_validation", {})
        correction = expected.get("accepted_test_expectation_correction", {})
        delegated = expected.get("delegated_m44_materialization", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        required_functions = (
            "_expected_m45_failed_pre_authoritative_m44_validation_successor_authority",
            "_m45_authority_reference",
            "_validated_m45_live_preflight_contract",
            "_assert_m45_historical_m44_controls",
            "_assert_m45_source_delta",
        )
        if (
            expected.get("schema")
            != (
                "sawm/failed-pre-authoritative-m44-validation-successor-"
                "authorization@1"
            )
            or expected.get("migration_revision") != _M45_MIGRATION_REVISION
            or expected.get("migration_kind") != key
            or expected.get("authorized") is not True
            or expected.get("authority") != "operator_control_plane"
            or int(expected.get("target_generation") or 0) != 32
            or int(expected.get("target_event_watermark") or 0) != 302
            or int(expected.get("target_plan_revision") or 0) != 28
            or preserved.get("authority_cid") != _M44_AUTHORITY_CID
            or preserved.get("authority_amended_or_rewritten") is not False
            or preserved.get("receipt_created") is not False
            or failed.get("runtime_started") is not False
            or int(failed.get("database_mutations") or 0) != 0
            or failed.get("m44_receipt_created") is not False
            or correction.get("exact_replacement_count") != 1
            or correction.get("operator_behavior_changed") is not False
            or correction.get("validation_weakened") is not False
            or delegated.get("authority_cid") != _M44_AUTHORITY_CID
            or delegated.get("m44_transition_semantics_preserved_exactly")
            is not True
            or delegated.get("m44_receipt_created") is not False
            or contract.get("migration_revision") != _M45_MIGRATION_REVISION
            or int(contract.get("prior_generation") or 0) != 31
            or int(contract.get("target_generation") or 0) != 32
            or int(contract.get("prior_event_watermark") or 0) != 301
            or int(contract.get("target_event_watermark") or 0) != 302
            or contract.get("m44_authority_must_be_preserved") is not True
            or changes.get("test_expectation_changes") != 1
            or changes.get("event_suffix_length") != 1
            or changes.get("task_revision_changes") != 0
            or changes.get("accepted_completion_changes") != 0
            or preservation.get("m44_authority_preserved_exactly") is not True
            or preservation.get("m44_receipt_absent_before_publication")
            is not True
            or preservation.get("coordination_store_bytes_preserved_exactly")
            is not True
            or preservation.get("worker_self_approval") is not False
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M45 prepublication validation successor delta is not exact")
        if require_active_runtime:
            runtime = scheduler.get("runtime_paths")
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            target_root = str(expected["target_runtime_root"])
            if (
                not isinstance(program, Mapping)
                or program.get("store_generation") != "32"
                or program.get("store_id") != expected["target_store_id"]
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M45 target/runtime binding is not exact")
        errors.extend(_m45_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M45 failed-pre-authoritative-M44-validation authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m44_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Return true when any protected surface declares the M44 successor."""

    key = _M44_SUCCESSOR_KEY
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m44_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    """Require M44's one repair commit and one protected-control reseal."""

    try:
        chain = authority.get("source_chain")
        if (
            not isinstance(chain, Mapping)
            or chain.get("m43_final_control_commit")
            != _M44_M43_FINAL_CONTROL_COMMIT
            or chain.get("m43_final_control_tree") != _M44_M43_FINAL_CONTROL_TREE
            or chain.get("repair_parent") != _M44_M43_FINAL_CONTROL_COMMIT
            or chain.get("repair_commit") != _M44_REPAIR_COMMIT
            or chain.get("repair_tree") != _M44_REPAIR_TREE
            or chain.get("repair_diff_sha256") != _M44_REPAIR_DIFF_SHA256
            or chain.get("repair_blobs") != _M44_REPAIR_BLOBS
            or chain.get("repair_modes") != _M44_REPAIR_MODES
            or chain.get("final_control_parent") != _M44_REPAIR_COMMIT
            or int(chain.get("repair_commit_count") or 0) != 1
            or int(chain.get("final_control_commit_count") or 0) != 1
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M44_OPERATOR_CONTROL_PATHS)
        ):
            raise RuntimeError("M44 sealed source-chain identity fields differ")
        if not hasattr(materializer, "_assert_m44_source_delta"):
            raise RuntimeError("M44 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        materializer._assert_m44_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M44 exact repair/control source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m44_post_m43_hardened_procfs_user_manager_restart_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Validate M44 and keep M43 immutable without rerunning M43's HEAD gate."""

    key = _M44_SUCCESSOR_KEY
    m43_key = "dead_attempt_lifecycle_recovery_restart_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m44_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M44 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer
            ._expected_m44_post_m43_hardened_procfs_user_manager_restart_successor_authority()
        )
        contract = materializer._validated_m44_live_preflight_contract(expected)
        reference = dict(materializer._m44_authority_reference())
        configured_cid = str(materializer._M44_AUTHORITY_CID)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M44 successor authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M44 successor reference differs")
        if seal.get(f"{key}_cid") != configured_cid:
            errors.append("M44 successor CID differs")
        if configured_cid == _M44_UNSEALED_AUTHORITY_CID:
            errors.append("M44 final control identities are not resealed")
        elif configured_cid != _M44_AUTHORITY_CID:
            errors.append("M44 successor authority CID is not the sealed identity")
        elif materializer._identity(expected) != configured_cid:
            errors.append("M44 successor authority CID does not bind its body")

        # M43 is immutable historical authority under M44.  Deliberately do
        # not call _m43_source_chain_errors: its active source gate requires
        # M43's old final control to be the current repository root.
        m43_expected = (
            materializer
            ._expected_m43_dead_attempt_lifecycle_recovery_restart_authority()
        )
        m43_reference = dict(materializer._m43_authority_reference())
        try:
            materializer._assert_m44_historical_m43_controls(
                scheduler, migration, seal
            )
        except Exception as exc:
            errors.append(
                "M44 historical M43 control verification differs: "
                f"{type(exc).__name__}: {exc}"
            )
        preserved_receipt = expected.get("preserved_m43_receipt", {})
        preserved_materialization = expected.get(
            "preserved_m43_materialization", {}
        )
        suffix = expected.get("post_m43_operational_suffix", {})
        prior = expected.get("prior_authority", {})
        stopped = expected.get("stopped_owner", {})
        repair = expected.get("accepted_user_manager_procfs_repair", {})
        target = expected.get("target_authority", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        required_functions = (
            "_assert_m44_historical_m43_controls",
            "_assert_m44_source_delta",
            "_validated_m44_live_preflight_contract",
            "_check_m44_prestart_admission",
            "_verify_m44_live_materialization",
            "_materialize_m44",
            "_check_m44_materialized",
        )
        if (
            scheduler.get(m43_key) != m43_reference
            or migration.get(m43_key) != m43_reference
            or seal.get(f"{m43_key}_cid") != _M43_AUTHORITY_CID
            or materializer._identity(m43_expected) != _M43_AUTHORITY_CID
            or expected.get("schema")
            != (
                "sawm/post-m43-hardened-procfs-user-manager-restart-"
                "authorization@1"
            )
            or expected.get("migration_revision") != _M44_MIGRATION_REVISION
            or expected.get("migration_kind") != key
            or expected.get("authorized") is not True
            or expected.get("authority") != "operator_control_plane"
            or expected.get("supersession_mode")
            != materializer._M44_SUPERSESSION_MODE
            or int(expected.get("target_generation") or 0) != 32
            or int(expected.get("target_event_watermark") or 0) != 302
            or int(expected.get("target_plan_revision") or 0) != 28
            or expected.get("target_projection_cid")
            != _M44_TARGET_PROJECTION_CID
            or int(prior.get("event_watermark") or 0) != 301
            or prior.get("projection_cid") != _M44_PRIOR_PROJECTION_CID
            or stopped.get("status") != "stopped"
            or int(stopped.get("generation") or 0) != 31
            or int(stopped.get("target_generation") or 0) != 32
            or preserved_receipt.get("receipt_cid") != _M44_M43_RECEIPT_CID
            or preserved_receipt.get("created_or_rewritten") is not False
            or preserved_materialization.get("event_id") != _M44_M43_EVENT_ID
            or preserved_materialization.get("evidence_id")
            != _M44_M43_EVIDENCE_ID
            or int(suffix.get("first_event_watermark") or 0) != 298
            or int(suffix.get("last_event_watermark") or 0) != 301
            or int(suffix.get("event_count") or 0) != 4
            or repair.get("repair_parent") != _M44_M43_FINAL_CONTROL_COMMIT
            or repair.get("repair_commit") != _M44_REPAIR_COMMIT
            or repair.get("repair_tree") != _M44_REPAIR_TREE
            or repair.get("blob_oids") != _M44_REPAIR_BLOBS
            or repair.get("path_modes") != _M44_REPAIR_MODES
            or repair.get("validation_weakened") is not False
            or repair.get("authority_weakened") is not False
            or int(target.get("event_watermark") or 0) != 302
            or target.get("projection_cid") != _M44_TARGET_PROJECTION_CID
            or contract.get("migration_revision") != _M44_MIGRATION_REVISION
            or int(contract.get("prior_generation") or 0) != 31
            or int(contract.get("target_generation") or 0) != 32
            or int(contract.get("prior_event_watermark") or 0) != 301
            or int(contract.get("target_event_watermark") or 0) != 302
            or contract.get("prior_projection_cid")
            != _M44_PRIOR_PROJECTION_CID
            or contract.get("target_projection_cid")
            != _M44_TARGET_PROJECTION_CID
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or changes.get("evidence_event_changes") != 1
            or changes.get("store_generation_row_changes") != 1
            or changes.get("state_server_row_changes") != 1
            or changes.get("task_revision_changes") != 0
            or changes.get("task_status_changes") != 0
            or changes.get("goal_revision_changes") != 0
            or changes.get("plan_revision_changes") != 0
            or changes.get("coordination_semantic_changes") != 0
            or changes.get("accepted_completion_changes") != 0
            or changes.get("worker_self_approval") is not False
            or preservation.get("m43_receipt_preserved_exactly") is not True
            or preservation.get("m43_event_297_preserved_exactly") is not True
            or preservation.get("post_m43_operational_suffix_preserved_exactly")
            is not True
            or preservation.get("coordination_store_bytes_preserved_exactly")
            is not True
            or preservation.get("worker_self_approval") is not False
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M44 procfs user-manager restart delta is not exact")
        if require_active_runtime:
            runtime = scheduler.get("runtime_paths")
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            target_root = str(expected["target_runtime_root"])
            if (
                not isinstance(program, Mapping)
                or program.get("store_generation") != "32"
                or program.get("store_id") != expected["target_store_id"]
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M44 target/runtime binding is not exact")
        errors.extend(_m44_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M44 hardened procfs user-manager restart authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m43_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Return true when any protected surface declares the M43 successor."""

    key = "dead_attempt_lifecycle_recovery_restart_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m43_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    """Require the sealed M43 repair, failed validation, and final reseal chain."""

    try:
        chain = authority.get("source_chain", {})
        repair_blobs = chain.get("lifecycle_repair_blobs", {})
        control_blobs = chain.get("initial_control_blobs", {})
        failed_control_blobs = chain.get(
            "failed_pre_authoritative_control_blobs", {}
        )
        fixture_repair_blobs = chain.get("historical_fixture_repair_blobs", {})
        prior_final_blobs = chain.get("prior_final_control_blobs", {})
        verifier_repair_blobs = chain.get(
            "bounded_materializer_verifier_repair_blobs", {}
        )
        revision3_final_blobs = chain.get("revision3_final_control_blobs", {})
        successor_verifier_repair_blobs = chain.get(
            "successor_evidence_verifier_repair_blobs", {}
        )
        expected_repair_paths = {
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "database_portal_bridge.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
            "implementation_daemon_runner.py",
            "test/api/test_agent_supervisor_database_portal_bridge.py",
            "test/api/test_agent_supervisor_configured_board_live_capsule.py",
        }
        if (
            not isinstance(chain, Mapping)
            or not isinstance(repair_blobs, Mapping)
            or not isinstance(control_blobs, Mapping)
            or not isinstance(failed_control_blobs, Mapping)
            or not isinstance(fixture_repair_blobs, Mapping)
            or not isinstance(prior_final_blobs, Mapping)
            or not isinstance(verifier_repair_blobs, Mapping)
            or not isinstance(revision3_final_blobs, Mapping)
            or not isinstance(successor_verifier_repair_blobs, Mapping)
            or chain.get("m42_final_control_commit")
            != materializer._M43_BASE_CONTROL_COMMIT
            or chain.get("m42_final_control_tree")
            != materializer._M43_BASE_CONTROL_TREE
            or chain.get("lifecycle_repair_parent")
            != materializer._M43_BASE_CONTROL_COMMIT
            or chain.get("lifecycle_repair_commit")
            != materializer._M43_LIFECYCLE_REPAIR_COMMIT
            or chain.get("lifecycle_repair_tree")
            != materializer._M43_LIFECYCLE_REPAIR_TREE
            or dict(repair_blobs)
            != dict(materializer._M43_LIFECYCLE_REPAIR_BLOBS)
            or chain.get("lifecycle_repair_modes")
            != dict(materializer._M43_LIFECYCLE_REPAIR_MODES)
            or set(repair_blobs) != expected_repair_paths
            or chain.get("lifecycle_repair_path_set_finalized")
            is not materializer._M43_LIFECYCLE_REPAIR_PATHS_FINALIZED
            or chain.get("initial_control_parent")
            != materializer._M43_LIFECYCLE_REPAIR_COMMIT
            or chain.get("initial_control_commit")
            != materializer._M43_INITIAL_CONTROL_COMMIT
            or chain.get("initial_control_tree")
            != materializer._M43_INITIAL_CONTROL_TREE
            or dict(control_blobs) != dict(materializer._M43_INITIAL_CONTROL_BLOBS)
            or chain.get("initial_control_modes")
            != dict(materializer._M43_INITIAL_CONTROL_MODES)
            or chain.get("failed_pre_authoritative_control_parent")
            != materializer._M43_INITIAL_CONTROL_COMMIT
            or chain.get("failed_pre_authoritative_control_commit")
            != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_COMMIT
            or chain.get("failed_pre_authoritative_control_tree")
            != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_TREE
            or dict(failed_control_blobs)
            != dict(materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_BLOBS)
            or chain.get("failed_pre_authoritative_control_modes")
            != dict(materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_MODES)
            or chain.get("historical_fixture_repair_parent")
            != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_COMMIT
            or chain.get("historical_fixture_repair_commit")
            != materializer._M43_HISTORICAL_FIXTURE_REPAIR_COMMIT
            or chain.get("historical_fixture_repair_tree")
            != materializer._M43_HISTORICAL_FIXTURE_REPAIR_TREE
            or dict(fixture_repair_blobs)
            != dict(materializer._M43_HISTORICAL_FIXTURE_REPAIR_BLOBS)
            or chain.get("historical_fixture_repair_modes")
            != dict(materializer._M43_HISTORICAL_FIXTURE_REPAIR_MODES)
            or chain.get("prior_final_control_parent")
            != materializer._M43_HISTORICAL_FIXTURE_REPAIR_COMMIT
            or chain.get("prior_final_control_commit")
            != materializer._M43_PRIOR_FINAL_CONTROL_COMMIT
            or chain.get("prior_final_control_tree")
            != materializer._M43_PRIOR_FINAL_CONTROL_TREE
            or dict(prior_final_blobs)
            != dict(materializer._M43_PRIOR_FINAL_CONTROL_BLOBS)
            or chain.get("prior_final_control_modes")
            != dict(materializer._M43_PRIOR_FINAL_CONTROL_MODES)
            or chain.get("bounded_materializer_verifier_repair_parent")
            != materializer._M43_PRIOR_FINAL_CONTROL_COMMIT
            or chain.get("bounded_materializer_verifier_repair_commit")
            != materializer._M43_VERIFIER_REPAIR_COMMIT
            or chain.get("bounded_materializer_verifier_repair_tree")
            != materializer._M43_VERIFIER_REPAIR_TREE
            or dict(verifier_repair_blobs)
            != dict(materializer._M43_VERIFIER_REPAIR_BLOBS)
            or chain.get("bounded_materializer_verifier_repair_modes")
            != dict(materializer._M43_VERIFIER_REPAIR_MODES)
            or chain.get("revision3_final_control_parent")
            != materializer._M43_VERIFIER_REPAIR_COMMIT
            or chain.get("revision3_final_control_commit")
            != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
            or chain.get("revision3_final_control_tree")
            != materializer._M43_REVISION3_FINAL_CONTROL_TREE
            or dict(revision3_final_blobs)
            != dict(materializer._M43_REVISION3_FINAL_CONTROL_BLOBS)
            or chain.get("revision3_final_control_modes")
            != dict(materializer._M43_REVISION3_FINAL_CONTROL_MODES)
            or chain.get("successor_evidence_verifier_repair_parent")
            != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
            or chain.get("successor_evidence_verifier_repair_commit")
            != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_COMMIT
            or chain.get("successor_evidence_verifier_repair_tree")
            != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_TREE
            or dict(successor_verifier_repair_blobs)
            != dict(materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_BLOBS)
            or chain.get("successor_evidence_verifier_repair_modes")
            != dict(materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_MODES)
            or chain.get("final_reseal_parent")
            != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_COMMIT
            or int(
                chain.get("failed_pre_authoritative_control_commit_count") or 0
            )
            != 2
            or int(chain.get("historical_fixture_repair_commit_count") or 0)
            != 1
            or int(
                chain.get("bounded_materializer_verifier_repair_commit_count")
                or 0
            )
            != 1
            or int(chain.get("revision3_final_control_commit_count") or 0) != 1
            or int(
                chain.get("successor_evidence_verifier_repair_commit_count") or 0
            )
            != 1
            or int(chain.get("failed_disposable_partial_append_count") or 0) != 1
            or int(chain.get("final_reseal_commit_count") or 0) != 3
            or int(chain.get("final_control_commit_count") or 0) != 4
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M43_OPERATOR_CONTROL_PATHS)
            or set(control_blobs) != set(materializer._M43_OPERATOR_CONTROL_PATHS)
            or set(failed_control_blobs)
            != set(materializer._M43_OPERATOR_CONTROL_PATHS)
            or set(fixture_repair_blobs)
            != set(materializer._M43_HISTORICAL_FIXTURE_REPAIR_BLOBS)
            or set(prior_final_blobs)
            != set(materializer._M43_OPERATOR_CONTROL_PATHS)
            or set(verifier_repair_blobs)
            != set(materializer._M43_VERIFIER_REPAIR_BLOBS)
            or set(revision3_final_blobs)
            != set(materializer._M43_OPERATOR_CONTROL_PATHS)
            or set(successor_verifier_repair_blobs)
            != set(materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_BLOBS)
        ):
            raise RuntimeError("M43 sealed source-chain identity fields differ")
        if not hasattr(materializer, "_assert_m43_source_delta"):
            raise RuntimeError("M43 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        materializer._assert_m43_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M43 exact repair/control source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m43_dead_attempt_lifecycle_recovery_restart_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Validate M43 before it masks M42's stopped-generation authority."""

    key = "dead_attempt_lifecycle_recovery_restart_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m43_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M43 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer
            ._expected_m43_dead_attempt_lifecycle_recovery_restart_authority()
        )
        contract = materializer._validated_m43_live_preflight_contract(expected)
        reference = dict(materializer._m43_authority_reference())
        configured_cid = materializer._M43_AUTHORITY_CID
        computed_cid = materializer._identity(expected)
        pending = configured_cid == _M43_UNSEALED_AUTHORITY_CID
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M43 successor authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M43 successor reference differs")
        if seal.get(f"{key}_cid") != configured_cid:
            errors.append("M43 successor CID differs")
        if pending:
            errors.append("M43 final control identities are not resealed")
        elif configured_cid != _M43_AUTHORITY_CID:
            errors.append("M43 successor authority CID is not the sealed identity")
        elif configured_cid != computed_cid:
            errors.append("M43 successor authority CID does not bind its body")

        m42_key = (
            "failed_pre_authoritative_m41_evidence_projection_"
            "successor_materialization"
        )
        m42_expected = (
            materializer
            ._expected_m42_failed_pre_authoritative_m41_evidence_projection_successor_authority()
        )
        m42_reference = dict(materializer._m42_authority_reference())
        try:
            materializer._assert_m43_historical_m42_controls(
                scheduler, migration, seal
            )
        except Exception as exc:
            errors.append(
                "M43 historical M42 control verification differs: "
                f"{type(exc).__name__}: {exc}"
            )
        if (
            scheduler.get(m42_key) != m42_reference
            or migration.get(m42_key) != m42_reference
            or seal.get(f"{m42_key}_cid") != _M42_AUTHORITY_CID
            or materializer._identity(m42_expected) != _M42_AUTHORITY_CID
            or expected.get("prior_m42_authority") != m42_expected
            or expected.get("prior_m42_reference") != m42_reference
        ):
            errors.append("M43 embedded historical M42 controls differ")

        prior = expected.get("prior_authority", {})
        stopped = expected.get("stopped_owner", {})
        artifacts = expected.get("stopped_prestart_artifacts", {})
        receipt = expected.get("preserved_m42_receipt", {})
        repair = expected.get("accepted_lifecycle_repair", {})
        failed_validation = expected.get(
            "failed_pre_authoritative_control_validation", {}
        )
        fixture_repair = expected.get("accepted_historical_fixture_repair", {})
        failed_reseal = expected.get(
            "failed_pre_authoritative_reseal_validation", {}
        )
        verifier_repair = expected.get(
            "accepted_bounded_materializer_verifier_repair", {}
        )
        final_hardening = expected.get("final_control_hardening", {})
        prior_control = expected.get("prior_control_authorization", {})
        prior_revision3 = expected.get("prior_revision3_authority", {})
        revision4_validation = expected.get("revision4_validation_attempts", {})
        revision4_attempts = revision4_validation.get("attempts", ())
        revision3_suite = (
            revision4_attempts[0]
            if isinstance(revision4_attempts, Sequence)
            and len(revision4_attempts) == 2
            else {}
        )
        disposable_rehearsal = (
            revision4_attempts[1]
            if isinstance(revision4_attempts, Sequence)
            and len(revision4_attempts) == 2
            else {}
        )
        authoritative_anchors = revision4_validation.get(
            "authoritative_anchors_after_both_attempts", {}
        )
        successor_verifier_repair = expected.get(
            "accepted_successor_evidence_verifier_repair", {}
        )
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        required_functions = (
            "_assert_m43_historical_m42_controls",
            "_assert_m43_source_delta",
            "_validated_m43_live_preflight_contract",
            "_m43_source_binding_authority",
            "_verify_m43_preserved_m42_receipt",
            "_check_m43_prestart_admission",
            "_verify_m43_live_materialization",
            "_materialize_m43",
            "_check_m43_materialized",
        )
        if (
            expected.get("schema")
            != "sawm/dead-attempt-lifecycle-recovery-restart-authorization@4"
            or expected.get("authorization_revision") != 4
            or expected.get("authorization_amended_at")
            != materializer._M43_AUTHORIZATION_AMENDED_AT
            or expected.get("prior_authorization_cid")
            != materializer._M43_REVISION3_AUTHORITY_CID
            or expected.get("control_recorded_at") != "2026-09-01T11:00:00Z"
            or expected.get("authorization_amendment_paths")
            != sorted(materializer._M43_OPERATOR_CONTROL_PATHS)
            or prior_control.get("authorization_cid")
            != materializer._M43_REVISION3_AUTHORITY_CID
            or prior_control.get("authorization_revision") != 3
            or prior_control.get("prior_authorization_cid")
            != materializer._M43_PRIOR_FINAL_AUTHORITY_CID
            or prior_control.get("control_recorded_at")
            != "2026-09-01T10:00:00Z"
            or prior_control.get("control_commit")
            != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
            or prior_control.get("control_tree")
            != materializer._M43_REVISION3_FINAL_CONTROL_TREE
            or prior_control.get("superseded_before_authoritative_event_297")
            is not True
            or prior_control.get("event_297_appended") is not False
            or prior_control.get("receipt_published") is not False
            or prior_revision3
            != materializer._expected_m43_revision3_authority()
            or materializer._identity(prior_revision3)
            != materializer._M43_REVISION3_AUTHORITY_CID
            or prior_revision3.get("schema")
            != "sawm/dead-attempt-lifecycle-recovery-restart-authorization@3"
            or prior_revision3.get("authorization_revision") != 3
            or prior_revision3.get("control_recorded_at")
            != "2026-09-01T10:00:00Z"
            or prior_revision3.get("prior_authorization_cid")
            != materializer._M43_PRIOR_FINAL_AUTHORITY_CID
            or expected.get("migration_revision") != "SAWM-R2-M43"
            or expected.get("migration_kind") != key
            or expected.get("target_generation") != 31
            or expected.get("target_event_watermark") != 297
            or expected.get("target_plan_revision") != 28
            or expected.get("target_projection_cid")
            != "baguqeerazspjonqzwhd5e2jmnpl4lacaasfmtkziur4awfrihibnh6mxkpoa"
            or prior.get("event_watermark") != 296
            or prior.get("event_prefix_sha256")
            != "89c64a4f018c2a3cfdce675eb8fb27913674e76995d64d89cabec42dd2967b70"
            or prior.get("projection_cid")
            != "baguqeerasguaepwupk3d5vme3cqsbicujihwvxnemnoenvdtnu3uwrscbt6q"
            or prior.get("semantic_authority_digest")
            != "sha256:a9f7e45d543cd983b36d475f3145344c956c547524bf33adfdc630de2bda7ae0"
            or prior.get("control_store_sha256")
            != "6798563648545b3fc05f1b7638ad2d0448c743d3a788bf78208a5d28a76a95f7"
            or prior.get("control_store_size") != 43_528_192
            or prior.get("coordination_store_sha256")
            != "ddbdf352e6a41452c6584cfa06fc760b90a94f1ff6473ff2c5eeb93de7551785"
            or prior.get("coordination_store_size") != 16_789_504
            or stopped.get("database_uuid")
            != "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4"
            or stopped.get("generation") != 30
            or stopped.get("target_generation") != 31
            or stopped.get("status") != "stopped"
            or stopped.get("revision") != 2
            or stopped.get("server_id")
            != "server:b9be27fb-cb7f-4cf3-919f-8e9807cd9e9d"
            or stopped.get("process_birth_id")
            != "birth:35978a75d075a1770e4da1f973a6f78b"
            or stopped.get("started_at") != "2026-09-01T01:58:03Z"
            or stopped.get("stopped_at") != "2026-09-01T06:56:59Z"
            or stopped.get("startup_epoch") != 1_788_227_883
            or artifacts.get("status_sha256")
            != "3f8c1227e7bc29c3057d238e550880cdfb6144a3687a73dea12e3ca063148a4c"
            or artifacts.get("status_size") != 2_408
            or any(
                artifacts.get(name) is not True
                for name in (
                    "owner_marker_absent",
                    "stop_control_absent",
                    "token_handoff_absent",
                    "control_wal_absent",
                    "coordination_wal_absent",
                )
            )
            or receipt.get("sha256")
            != "ff9a24d339cf06eacb3573cd2825e0648a558efe5ec9539c0c4f489002ca609d"
            or receipt.get("size") != 14_112
            or receipt.get("receipt_cid")
            != "sha256:31565b6bfc8e071f4278acc88fd3500ca5c4d25eee63d0131b16ceca3e7a9169"
            or receipt.get("created_or_rewritten") is not False
            or repair.get("changed_paths")
            != sorted(materializer._M43_LIFECYCLE_REPAIR_BLOBS)
            or repair.get("repair_parent")
            != "a8bce148b793dcd15ac742df3a29e5773a178f28"
            or repair.get("repair_commit")
            != "7e3fa1170edac23149e0d1f38f5ff6b5f5ddb571"
            or repair.get("repair_tree")
            != "df40d6f879753c8c2ca00f35fd28054a29fd600a"
            or repair.get("blob_oids")
            != {
                "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
                "database_portal_bridge.py": (
                    "97f44d032063a8a98cfca277dd123c998244f076"
                ),
                "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
                "implementation_daemon_runner.py": (
                    "51e577170d975f056e3f97a9ded55d210bee8792"
                ),
                "test/api/test_agent_supervisor_database_portal_bridge.py": (
                    "07fb7dd738256126c96e1370dd5a60501a13647b"
                ),
                "test/api/test_agent_supervisor_configured_board_live_capsule.py": (
                    "4638112c6c2da24cb5912914192f332f2903bd6b"
                ),
            }
            or repair.get("path_modes")
            != {path: "100644" for path in repair.get("changed_paths", ())}
            or repair.get("repair_path_set_finalized")
            is not materializer._M43_LIFECYCLE_REPAIR_PATHS_FINALIZED
            or repair.get("newest_first_attempt_selection") is not True
            or repair.get("active_marker_snapshot_proof_required") is not True
            or repair.get("declared_output_identity_preserved") is not True
            or repair.get("prepared_receipt_precedes_lifecycle_cas") is not True
            or repair.get("marker_retirement_is_no_replace") is not True
            or repair.get("post_rename_crash_recovery_is_idempotent") is not True
            or repair.get("authority_weakened") is not False
            or failed_validation.get("authority_cid")
            != materializer._M43_PRE_AUTHORITATIVE_AUTHORITY_CID
            or failed_validation.get("control_commit")
            != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_COMMIT
            or failed_validation.get("control_tree")
            != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_TREE
            or failed_validation.get("control_blobs")
            != dict(materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_BLOBS)
            or failed_validation.get("control_modes")
            != dict(materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_MODES)
            or failed_validation.get("collected_tests") != 280
            or failed_validation.get("passed_tests") != 271
            or failed_validation.get("failed_tests") != 9
            or failed_validation.get("failed_test_ids")
            != list(materializer._M43_FAILED_TEST_IDS)
            or failed_validation.get("production_runtime_defect") is not False
            or failed_validation.get("materializer_invoked") is not False
            or failed_validation.get("quack_started") is not False
            or failed_validation.get("authenticated_mutation_request_created")
            is not False
            or failed_validation.get("event_297_rows_created") != 0
            or failed_validation.get("m43_receipt_created") is not False
            or any(
                failed_validation.get(name) != 0
                for name in (
                    "evidence_node_changes",
                    "evidence_event_changes",
                    "validation_event_changes",
                    "store_generation_row_changes",
                    "state_server_row_changes",
                    "credential_row_changes",
                    "server_epoch_row_changes",
                    "capability_snapshot_row_changes",
                    "task_revision_changes",
                    "task_status_changes",
                    "goal_revision_changes",
                    "plan_revision_changes",
                    "effect_claim_changes",
                    "merge_attempt_changes",
                    "implementation_provider_invocations",
                    "coordination_semantic_changes",
                    "accepted_completion_changes",
                )
            )
            or failed_validation.get("worker_self_approval") is not False
            or fixture_repair.get("repair_parent")
            != materializer._M43_FAILED_PRE_AUTHORITATIVE_CONTROL_COMMIT
            or fixture_repair.get("repair_commit")
            != materializer._M43_HISTORICAL_FIXTURE_REPAIR_COMMIT
            or fixture_repair.get("repair_tree")
            != materializer._M43_HISTORICAL_FIXTURE_REPAIR_TREE
            or fixture_repair.get("blob_oids")
            != dict(materializer._M43_HISTORICAL_FIXTURE_REPAIR_BLOBS)
            or fixture_repair.get("path_modes")
            != dict(materializer._M43_HISTORICAL_FIXTURE_REPAIR_MODES)
            or fixture_repair.get("focused_former_failures_replayed") != 9
            or fixture_repair.get("focused_former_failures_passed") != 9
            or fixture_repair.get("focused_replay_is_full_suite") is not False
            or fixture_repair.get("production_code_changes") != 0
            or fixture_repair.get(
                "fixture_collection_or_selection_logic_changed"
            )
            is not False
            or fixture_repair.get("validation_weakened") is not False
            or fixture_repair.get("authority_weakened") is not False
            or failed_reseal.get("authority_cid")
            != materializer._M43_PRIOR_FINAL_AUTHORITY_CID
            or failed_reseal.get("control_commit")
            != materializer._M43_PRIOR_FINAL_CONTROL_COMMIT
            or failed_reseal.get("control_tree")
            != materializer._M43_PRIOR_FINAL_CONTROL_TREE
            or failed_reseal.get("observed_failure_count") != 2
            or failed_reseal.get("isolated_live_rehearsal", {}).get(
                "disposable_generation_31_ready"
            )
            is not True
            or failed_reseal.get("full_suite_validation", {}).get("error")
            != "MigrationRequired: operator task-recovery event differs"
            or failed_reseal.get("authenticated_mutation_request_created")
            is not False
            or failed_reseal.get("event_297_rows_created") != 0
            or failed_reseal.get("accepted_completion_changes") != 0
            or verifier_repair.get("repair_parent")
            != materializer._M43_PRIOR_FINAL_CONTROL_COMMIT
            or verifier_repair.get("repair_commit")
            != materializer._M43_VERIFIER_REPAIR_COMMIT
            or verifier_repair.get("repair_tree")
            != materializer._M43_VERIFIER_REPAIR_TREE
            or verifier_repair.get("blob_oids")
            != dict(materializer._M43_VERIFIER_REPAIR_BLOBS)
            or verifier_repair.get("path_modes")
            != dict(materializer._M43_VERIFIER_REPAIR_MODES)
            or verifier_repair.get("repair_class_count") != 2
            or verifier_repair.get("repair_classes")
            != [
                "duckdb_positional_row_normalization",
                "task_revision_timestamp_source_binding",
            ]
            or verifier_repair.get("repair_site_count") != 5
            or verifier_repair.get("repair_sites")
            != [
                "m43_operational_suffix_rows",
                "m43_prestart_event_type_counts",
                "m43_live_preappend_event_type_counts",
                "m43_live_postappend_event_type_counts",
                "m6_recovery_revision_recorded_at",
            ]
            or verifier_repair.get("validation_weakened") is not False
            or verifier_repair.get("authority_weakened") is not False
            or revision4_validation
            != materializer._m43_revision4_validation_attempts()
            or revision4_validation.get("schema")
            != "sawm/m43-revision-4-validation-attempts@1"
            or not isinstance(revision4_attempts, Sequence)
            or len(revision4_attempts) != 2
            or revision3_suite.get("kind")
            != "authoritative_current_tree_full_suite"
            or revision3_suite.get("source_commit")
            != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
            or revision3_suite.get("source_tree")
            != materializer._M43_REVISION3_FINAL_CONTROL_TREE
            or revision3_suite.get("result") != "passed"
            or revision3_suite.get("collected") != 280
            or revision3_suite.get("passed") != 280
            or revision3_suite.get("failed") != 0
            or revision3_suite.get("retained_log", {}).get("sha256")
            != materializer._M43_REVISION3_FULL_SUITE_LOG_SHA256
            or revision3_suite.get("retained_log", {}).get("size_bytes")
            != materializer._M43_REVISION3_FULL_SUITE_LOG_SIZE
            or disposable_rehearsal.get("kind")
            != "fresh_disposable_clone_partial_append_rehearsal"
            or disposable_rehearsal.get("disposable") is not True
            or disposable_rehearsal.get("authoritative") is not False
            or disposable_rehearsal.get("source_commit")
            != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
            or disposable_rehearsal.get("source_tree")
            != materializer._M43_REVISION3_FINAL_CONTROL_TREE
            or disposable_rehearsal.get("authorization_cid")
            != materializer._M43_REVISION3_AUTHORITY_CID
            or disposable_rehearsal.get("result")
            != "failed_after_authenticated_append_before_receipt"
            or disposable_rehearsal.get("typed_error")
            != "MigrationRequired: M42 exact evidence projection membership differs"
            or disposable_rehearsal.get("post_stop_generation") != 31
            or disposable_rehearsal.get("partial_append", {}).get(
                "event_watermark_after"
            )
            != 297
            or disposable_rehearsal.get("partial_append", {}).get(
                "evidence_count_after"
            )
            != 50
            or disposable_rehearsal.get("partial_append", {}).get(
                "m43_receipt_present"
            )
            is not False
            or disposable_rehearsal.get("task_status_changes") != 0
            or disposable_rehearsal.get("accepted_completion_changes") != 0
            or authoritative_anchors.get("unchanged") is not True
            or authoritative_anchors.get("authoritative_event_watermark") != 296
            or authoritative_anchors.get("authoritative_evidence_count") != 49
            or authoritative_anchors.get("authoritative_generation") != 30
            or revision4_validation.get("worker_self_approval") is not False
            or successor_verifier_repair
            != materializer._m43_successor_evidence_verifier_repair()
            or successor_verifier_repair.get("schema")
            != "sawm/bounded-successor-evidence-verifier-repair@1"
            or successor_verifier_repair.get("repair_parent")
            != materializer._M43_REVISION3_FINAL_CONTROL_COMMIT
            or successor_verifier_repair.get("repair_commit")
            != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_COMMIT
            or successor_verifier_repair.get("repair_tree")
            != materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_TREE
            or successor_verifier_repair.get("blob_oids")
            != dict(materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_BLOBS)
            or successor_verifier_repair.get("path_modes")
            != dict(materializer._M43_SUCCESSOR_EVIDENCE_VERIFIER_REPAIR_MODES)
            or successor_verifier_repair.get(
                "permitted_successor_evidence_row_count"
            )
            != 1
            or successor_verifier_repair.get("successor_row_is_caller_bound")
            is not True
            or successor_verifier_repair.get(
                "successor_row_content_identity_rehashed"
            )
            is not True
            or successor_verifier_repair.get("unlisted_extra_evidence_rejected")
            is not True
            or successor_verifier_repair.get("wrong_successor_evidence_rejected")
            is not True
            or successor_verifier_repair.get("preappend_default_behavior_changed")
            is not False
            or successor_verifier_repair.get("m42_return_fields_changed")
            is not False
            or successor_verifier_repair.get("m42_return_counts_changed")
            is not False
            or successor_verifier_repair.get("authority_weakened") is not False
            or successor_verifier_repair.get("worker_self_approval") is not False
            or final_hardening.get(
                "operational_suffix_malformed_rows_are_typed_conflicts"
            )
            is not True
            or final_hardening.get(
                "pending_authority_rejected_by_operator_facade"
            )
            is not True
            or final_hardening.get(
                "exact_successor_evidence_is_the_only_postappend_allowance"
            )
            is not True
            or final_hardening.get("m42_return_contract_preserved") is not True
            or final_hardening.get("unlisted_physical_evidence_still_fails_closed")
            is not True
            or final_hardening.get("authority_weakened") is not False
            or expected.get("ordinary_source_changes")
            != len(materializer._M43_LIFECYCLE_REPAIR_BLOBS)
            or contract.get("prior_generation") != 30
            or contract.get("target_generation") != 31
            or contract.get("prior_event_watermark") != 296
            or contract.get("target_event_watermark") != 297
            or contract.get("generation_restart_authorized") is not True
            or contract.get("m42_receipt_must_be_preserved") is not True
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or changes.get("evidence_event_changes") != 1
            or changes.get("store_generation_row_changes") != 1
            or changes.get("state_server_row_changes") != 1
            or changes.get("task_revision_changes") != 0
            or changes.get("task_status_changes") != 0
            or changes.get("goal_revision_changes") != 0
            or changes.get("plan_revision_changes") != 0
            or changes.get("coordination_semantic_changes") != 0
            or changes.get("accepted_completion_changes") != 0
            or preservation.get("m42_receipt_preserved_exactly") is not True
            or preservation.get("m42_receipt_created_or_rewritten") is not False
            or preservation.get("events_293_through_296_preserved_exactly")
            is not True
            or preservation.get("worker_self_approval") is not False
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M43 dead-attempt recovery restart delta is not exact")
        if require_active_runtime:
            runtime = scheduler.get("runtime_paths")
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            target_root = str(expected["target_runtime_root"])
            if (
                not isinstance(program, Mapping)
                or program.get("store_generation") != "31"
                or program.get("store_id") != expected["target_store_id"]
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M43 target/runtime binding is not exact")
        errors.extend(_m43_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M43 dead-attempt lifecycle recovery authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m42_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Return true when any protected surface declares the M42 successor."""

    key = "failed_pre_authoritative_m41_evidence_projection_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m42_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    """Require M41 final -> exact two-file repair -> nine-path M42 child."""

    try:
        chain = authority.get("source_chain", {})
        repair_blobs = chain.get("projection_repair_blobs", {})
        if (
            not isinstance(chain, Mapping)
            or not isinstance(repair_blobs, Mapping)
            or chain.get("m41_final_control_commit")
            != materializer._M42_M41_FINAL_CONTROL_COMMIT
            or chain.get("m41_final_control_tree")
            != materializer._M42_M41_FINAL_CONTROL_TREE
            or chain.get("projection_repair_parent")
            != materializer._M42_M41_FINAL_CONTROL_COMMIT
            or chain.get("projection_repair_commit")
            != materializer._M42_PROJECTION_REPAIR_COMMIT
            or chain.get("projection_repair_tree")
            != materializer._M42_PROJECTION_REPAIR_TREE
            or dict(repair_blobs)
            != dict(materializer._M42_PROJECTION_REPAIR_BLOBS)
            or int(chain.get("bounded_repair_commit_count") or 0) != 1
            or chain.get("final_control_parent")
            != materializer._M42_PROJECTION_REPAIR_COMMIT
            or chain.get("final_control_commit_is_current_head") is not True
            or int(chain.get("final_control_commit_count") or 0) != 1
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M42_OPERATOR_CONTROL_PATHS)
            or set(repair_blobs)
            != {
                "scripts/materialize_semantic_addressed_world_model_program.py",
                "test/api/semantic_world/"
                "test_semantic_addressed_world_model_board.py",
            }
        ):
            raise RuntimeError("M42 sealed source-chain identity fields differ")
        if not hasattr(materializer, "_assert_m42_source_delta"):
            raise RuntimeError("M42 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        materializer._assert_m42_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M42 exact repair/control source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m42_failed_pre_authoritative_m41_evidence_projection_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Validate M42 before masking M41's pre-authoritative projection failure."""

    key = "failed_pre_authoritative_m41_evidence_projection_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m42_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M42 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer
            ._expected_m42_failed_pre_authoritative_m41_evidence_projection_successor_authority()
        )
        contract = materializer._validated_m42_live_preflight_contract(expected)
        reference = dict(materializer._m42_authority_reference())
        expected_cid = materializer._identity(expected)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M42 successor authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M42 successor reference differs")
        if seal.get(f"{key}_cid") != expected_cid:
            errors.append("M42 successor CID differs")

        m41_key = "failed_pre_authoritative_m40_validation_successor_materialization"
        m41_expected = (
            materializer
            ._expected_m41_failed_pre_authoritative_m40_validation_successor_authority()
        )
        m41_reference = dict(materializer._m41_authority_reference())
        m41_cid = materializer._identity(m41_expected)
        if (
            scheduler.get(m41_key) != m41_reference
            or migration.get(m41_key) != m41_reference
            or seal.get(f"{m41_key}_cid") != m41_cid
            or m41_cid
            != "sha256:25ad5550b59024c8da9b4821fba2d7b1b49d2a17a781e5d4bd2cbe58cb7b0233"
            or expected.get("prior_m41_authority") != m41_expected
            or expected.get("prior_m41_reference") != m41_reference
        ):
            errors.append("M42 embedded historical M41 controls differ")

        failed = expected.get("failed_m41_pre_authoritative_materialization", {})
        split_clock = expected.get("split_timestamp_writer_evidence", {})
        legacy = expected.get("exact_legacy_projection_authority", {})
        repair = expected.get("accepted_projection_repair", {})
        derivation = expected.get("target_projection_derivation", {})
        preservation = expected.get("preservation", {})
        changes = expected.get("exact_changes", {})
        required_functions = (
            "_assert_m42_historical_m41_controls",
            "_assert_m42_source_delta",
            "_validated_m42_live_preflight_contract",
            "_m42_source_binding_authority",
            "_verify_m42_exact_legacy_projection",
            "_verify_m42_live_materialization",
            "_materialize_m42",
            "_check_m42_materialized",
        )
        if (
            expected.get("schema")
            != (
                "sawm/failed-pre-authoritative-m41-evidence-projection-"
                "successor-authorization@1"
            )
            or expected_cid != _M42_AUTHORITY_CID
            or expected_cid != materializer._M42_AUTHORITY_CID
            or expected.get("migration_revision") != "SAWM-R2-M42"
            or expected.get("migration_kind") != key
            or expected.get("control_recorded_at") != "2026-09-01T06:00:00Z"
            or expected.get("target_generation") != 30
            or expected.get("target_event_watermark") != 292
            or expected.get("target_projection_cid")
            != "baguqeera6t2s6prg5atpg4gkgrlmqclu34firbn4z2o3wsgv6btp7p6q66tq"
            or failed.get("schema")
            != "sawm/pre-authoritative-m41-materialization-failure@1"
            or failed.get("attempt") != "SAWM-R2-M41-MATERIALIZE-A1"
            or failed.get("phase")
            != "exact_historical_evidence_projection_verification"
            or failed.get("failure_kind")
            != "landed_projection_differs_from_event_only_replay"
            or failed.get("source_head")
            != materializer._M42_M41_FINAL_CONTROL_COMMIT
            or failed.get("source_tree") != materializer._M42_M41_FINAL_CONTROL_TREE
            or failed.get("sealed_suite_passed") != 314
            or failed.get("sealed_suite_failed") != 0
            or failed.get("materializer_invoked") is not True
            or failed.get("authenticated_live_quack_read_opened") is not True
            or failed.get("direct_duckdb_opened") is not False
            or failed.get("quack_mutation_request_created") is not False
            or failed.get("record_evidence_reached") is not False
            or failed.get("event_292_rows_created") != 0
            or failed.get("m41_receipt_created") is not False
            or failed.get("accepted_completion_changes") != 0
            or failed.get("worker_self_approval") is not False
            or split_clock.get("writer_inner_time_sampled_before_event_append")
            is not True
            or split_clock.get("legacy_mismatch_count") != 4
            or split_clock.get("event_identity_validation_preserved") is not True
            or split_clock.get("run_and_result_identity_validation_preserved")
            is not True
            or legacy.get("schema") != "sawm/m42-legacy-projection-authority@1"
            or legacy.get("manifest_cid")
            != materializer._M42_LEGACY_PROJECTION_MANIFEST_CID
            or legacy.get("evidence_projection_digest")
            != materializer._M42_LEGACY_EVIDENCE_PROJECTION_DIGEST
            or legacy.get("validation_runs_digest")
            != materializer._M42_LEGACY_VALIDATION_RUNS_DIGEST
            or legacy.get("validation_results_digest")
            != materializer._M42_LEGACY_VALIDATION_RESULTS_DIGEST
            or legacy.get("event_watermark") != 291
            or legacy.get("evidence_node_count") != 48
            or legacy.get("evidence_event_count") != 37
            or legacy.get("validation_event_count") != 11
            or legacy.get("validation_run_count") != 11
            or legacy.get("validation_result_count") != 11
            or legacy.get("evidence_refresh_overlay_count") != 9
            or legacy.get("compact_validation_evidence_overlay_count") != 1
            or legacy.get("validation_attempt_overlay_count") != 1
            or legacy.get("overlay_is_closed") is not True
            or legacy.get("event_only_replay_remains_strict_by_default") is not True
            or repair.get("schema") != "sawm/m41-evidence-projection-repair@1"
            or repair.get("repair_parent")
            != materializer._M42_M41_FINAL_CONTROL_COMMIT
            or repair.get("repair_commit")
            != materializer._M42_PROJECTION_REPAIR_COMMIT
            or repair.get("repair_tree") != materializer._M42_PROJECTION_REPAIR_TREE
            or repair.get("changed_paths")
            != sorted(materializer._M42_PROJECTION_REPAIR_BLOBS)
            or repair.get("blob_oids")
            != dict(materializer._M42_PROJECTION_REPAIR_BLOBS)
            or repair.get("strict_default_m38_verifier_preserved") is not True
            or repair.get("m42_exact_overlay_verifier_added") is not True
            or repair.get("production_authority_selection_changed") is not False
            or repair.get("historical_rows_rewritten") is not False
            or repair.get("validation_weakened") is not False
            or repair.get("focused_tests_passed") != 21
            or contract.get("generation_restart_authorized") is not False
            or contract.get("prior_event_watermark") != 291
            or contract.get("target_event_watermark") != 292
            or contract.get("m37_receipt_must_be_absent") is not True
            or contract.get("m38_receipt_must_be_absent") is not True
            or contract.get("m39_receipt_must_be_absent") is not True
            or contract.get("m40_receipt_must_be_absent") is not True
            or contract.get("m41_receipt_must_be_absent") is not True
            or contract.get("legacy_projection_manifest_cid")
            != legacy.get("manifest_cid")
            or contract.get("legacy_evidence_projection_digest")
            != legacy.get("evidence_projection_digest")
            or contract.get("legacy_validation_runs_digest")
            != legacy.get("validation_runs_digest")
            or contract.get("legacy_validation_results_digest")
            != legacy.get("validation_results_digest")
            or derivation.get("recomputed_not_inherited") is not True
            or derivation.get("event_body_bound_by_event_prefix_not_projection")
            is not True
            or preservation.get("failed_m41_materialization_attempt_preserved")
            is not True
            or preservation.get("m37_receipt_created_or_rewritten") is not False
            or preservation.get("m38_receipt_created_or_rewritten") is not False
            or preservation.get("m39_receipt_created_or_rewritten") is not False
            or preservation.get("m40_receipt_created_or_rewritten") is not False
            or preservation.get("m41_receipt_created_or_rewritten") is not False
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or changes.get("evidence_event_changes") != 1
            or changes.get("validation_event_changes") != 0
            or changes.get("store_generation_row_changes") != 0
            or changes.get("state_server_row_changes") != 0
            or changes.get("accepted_completion_changes") != 0
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M42 failed-M41-projection successor delta is not exact")
        if require_active_runtime:
            runtime = scheduler.get("runtime_paths")
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            target_root = str(expected["target_runtime_root"])
            if (
                not isinstance(program, Mapping)
                or program.get("store_generation") != "30"
                or program.get("store_id") != expected["target_store_id"]
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M42 target/runtime binding is not exact")
        errors.extend(_m42_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M42 failed-pre-authoritative-M41-projection authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m41_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Return true when any protected surface declares the M41 successor."""

    key = "failed_pre_authoritative_m40_validation_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m41_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    """Require M40 final -> exact one-file repair -> nine-path M41 child."""

    try:
        chain = authority.get("source_chain", {})
        repair_blobs = chain.get("historical_test_helper_repair_blobs", {})
        if (
            not isinstance(chain, Mapping)
            or not isinstance(repair_blobs, Mapping)
            or chain.get("m40_final_control_commit")
            != "7564064e191c88679f3a8b540fd5db847b0b492c"
            or chain.get("m40_final_control_tree")
            != "aa7035cc4a3bd0f0c1073b2081bcbf9928db4e4e"
            or chain.get("historical_test_helper_repair_parent")
            != "7564064e191c88679f3a8b540fd5db847b0b492c"
            or chain.get("historical_test_helper_repair_commit")
            != "599baee49905108a4361d37dcc6bd01d2829ee79"
            or chain.get("historical_test_helper_repair_tree")
            != "a96d83f1928c7482cd1d49d673744e8f4084c732"
            or dict(repair_blobs)
            != dict(materializer._M41_HISTORICAL_TEST_HELPER_REPAIR_BLOBS)
            or int(chain.get("bounded_repair_commit_count") or 0) != 1
            or chain.get("final_control_parent")
            != "599baee49905108a4361d37dcc6bd01d2829ee79"
            or chain.get("final_control_commit_is_current_head") is not True
            or int(chain.get("final_control_commit_count") or 0) != 1
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M41_OPERATOR_CONTROL_PATHS)
            or set(repair_blobs)
            != {
                "test/api/semantic_world/"
                "test_semantic_addressed_world_model_board.py",
            }
        ):
            raise RuntimeError("M41 sealed source-chain identity fields differ")
        if not hasattr(materializer, "_assert_m41_source_delta"):
            raise RuntimeError("M41 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        materializer._assert_m41_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M41 exact repair/control source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m41_failed_pre_authoritative_m40_validation_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Validate M41 before masking M40's pre-materialization suite failure."""

    key = "failed_pre_authoritative_m40_validation_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m41_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M41 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer
            ._expected_m41_failed_pre_authoritative_m40_validation_successor_authority()
        )
        contract = materializer._validated_m41_live_preflight_contract(expected)
        reference = dict(materializer._m41_authority_reference())
        expected_cid = materializer._identity(expected)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M41 successor authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M41 successor reference differs")
        if seal.get(f"{key}_cid") != expected_cid:
            errors.append("M41 successor CID differs")

        m40_key = "failed_pre_authoritative_m39_successor_materialization"
        m40_expected = (
            materializer._expected_m40_failed_pre_authoritative_m39_successor_authority()
        )
        m40_reference = dict(materializer._m40_authority_reference())
        m40_cid = materializer._identity(m40_expected)
        if (
            scheduler.get(m40_key) != m40_reference
            or migration.get(m40_key) != m40_reference
            or seal.get(f"{m40_key}_cid") != m40_cid
            or m40_cid
            != "sha256:377b6e7269a7025f236642f12aaf92264582f42a1efa4899eb4d6e64b0e41db2"
        ):
            errors.append("M41 historical M40 controls differ")

        failed = expected.get("failed_m40_pre_authoritative_validation", {})
        repair = expected.get("accepted_historical_test_helper_repair", {})
        prior = expected.get("prior_m40_authority", {})
        prior_reference = expected.get("prior_m40_reference", {})
        derivation = expected.get("target_projection_derivation", {})
        preservation = expected.get("preservation", {})
        changes = expected.get("exact_changes", {})
        source_chain = expected.get("source_chain", {})
        required_functions = (
            "_assert_m41_source_delta",
            "_validated_m41_live_preflight_contract",
            "_m41_source_binding_authority",
            "_materialize_m41",
            "_check_m41_materialized",
        )
        if (
            expected.get("schema")
            != (
                "sawm/failed-pre-authoritative-m40-validation-successor-"
                "authorization@1"
            )
            or expected_cid
            != "sha256:25ad5550b59024c8da9b4821fba2d7b1b49d2a17a781e5d4bd2cbe58cb7b0233"
            or expected.get("migration_revision") != "SAWM-R2-M41"
            or expected.get("migration_kind") != key
            or expected.get("control_recorded_at") != "2026-09-01T04:00:00Z"
            or expected.get("target_generation") != 30
            or expected.get("target_event_watermark") != 292
            or expected.get("target_projection_cid")
            != "baguqeera6t2s6prg5atpg4gkgrlmqclu34firbn4z2o3wsgv6btp7p6q66tq"
            or not isinstance(prior, Mapping)
            or materializer._identity(prior) != m40_cid
            or prior_reference != m40_reference
            or failed.get("schema")
            != "sawm/pre-authoritative-m40-validation-failure@1"
            or failed.get("attempt") != "SAWM-R2-M40-SEALED-SUITE-A1"
            or failed.get("phase") != "sealed_semantic_world_test_suite"
            or failed.get("failure_kind")
            != "historical_fixture_omitted_m40_successor_control_key"
            or failed.get("source_head")
            != "7564064e191c88679f3a8b540fd5db847b0b492c"
            or failed.get("source_tree")
            != "aa7035cc4a3bd0f0c1073b2081bcbf9928db4e4e"
            or failed.get("tests_passed") != 285
            or failed.get("tests_failed") != 22
            or len(failed.get("failed_test_node_ids") or ()) != 22
            or failed.get("materializer_invoked") is not False
            or failed.get("quack_mutation_request_created") is not False
            or failed.get("event_292_rows_created") != 0
            or failed.get("evidence_node_changes") != 0
            or failed.get("m40_receipt_created") is not False
            or failed.get("task_revision_changes") != 0
            or failed.get("task_status_changes") != 0
            or failed.get("goal_revision_changes") != 0
            or failed.get("plan_revision_changes") != 0
            or failed.get("coordination_semantic_changes") != 0
            or failed.get("provider_invocations") != 0
            or failed.get("implementation_commit_changes") != 0
            or failed.get("merge_attempt_changes") != 0
            or failed.get("accepted_completion_changes") != 0
            or failed.get("worker_self_approval") is not False
            or repair.get("schema")
            != "sawm/m40-historical-test-helper-repair@1"
            or repair.get("repair_commit")
            != "599baee49905108a4361d37dcc6bd01d2829ee79"
            or repair.get("changed_paths")
            != [
                "test/api/semantic_world/"
                "test_semantic_addressed_world_model_board.py"
            ]
            or repair.get("adds_m40_to_successor_control_keys_newest_first")
            is not True
            or repair.get("m40_key_index") != 0
            or repair.get("focused_former_failures_passed") != 22
            or repair.get("last_failed_rerun_passed") != 22
            or repair.get("direct_regression_passed") is not True
            or repair.get("production_selection_changed") is not False
            or repair.get("validation_weakened") is not False
            or contract.get("generation_restart_authorized") is not False
            or contract.get("prior_event_watermark") != 291
            or contract.get("target_event_watermark") != 292
            or contract.get("m39_receipt_must_be_absent") is not True
            or contract.get("m40_receipt_must_be_absent") is not True
            or derivation.get("recomputed_not_inherited") is not True
            or source_chain.get("historical_test_helper_repair_commit")
            != "599baee49905108a4361d37dcc6bd01d2829ee79"
            or preservation.get("failed_m40_validation_attempt_preserved")
            is not True
            or preservation.get("m39_receipt_created_or_rewritten") is not False
            or preservation.get("m40_receipt_created_or_rewritten") is not False
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or changes.get("evidence_event_changes") != 1
            or changes.get("store_generation_row_changes") != 0
            or changes.get("state_server_row_changes") != 0
            or changes.get("accepted_completion_changes") != 0
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M41 failed-M40-validation successor delta is not exact")
        if require_active_runtime:
            runtime = scheduler.get("runtime_paths")
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            target_root = str(expected["target_runtime_root"])
            if (
                not isinstance(program, Mapping)
                or program.get("store_generation") != "30"
                or program.get("store_id") != expected["target_store_id"]
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M41 target/runtime binding is not exact")
        errors.extend(_m41_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M41 failed-pre-authoritative-M40-validation authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m40_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Return true when any protected surface declares the M40 successor."""

    key = "failed_pre_authoritative_m39_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m40_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    """Require repair C and its direct nine-path M40 control child."""

    try:
        chain = authority.get("source_chain", {})
        repair_blobs = chain.get("restart_helper_repair_blobs", {})
        if (
            not isinstance(chain, Mapping)
            or not isinstance(repair_blobs, Mapping)
            or chain.get("m39_final_control_commit")
            != "64651e11f9d390a98a9daecc70c672e329487a1b"
            or chain.get("m39_final_control_tree")
            != "75b32dda51c7475a0a54de28d6e6a3c693c2d7f3"
            or chain.get("restart_helper_repair_parent")
            != "64651e11f9d390a98a9daecc70c672e329487a1b"
            or chain.get("restart_helper_repair_commit")
            != "00d15b870f7fdaa8e7165a94719eb2c7d3df7eca"
            or chain.get("restart_helper_repair_tree")
            != "83c17ce2ed68d10ef54e53c0a5e312f2eae8060b"
            or dict(repair_blobs)
            != dict(materializer._M40_RESTART_HELPER_REPAIR_BLOBS)
            or int(chain.get("bounded_repair_commit_count") or 0) != 1
            or chain.get("final_control_parent")
            != "00d15b870f7fdaa8e7165a94719eb2c7d3df7eca"
            or chain.get("final_control_commit_is_current_head") is not True
            or int(chain.get("final_control_commit_count") or 0) != 1
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M40_OPERATOR_CONTROL_PATHS)
            or set(repair_blobs)
            != {
                "scripts/materialize_semantic_addressed_world_model_program.py",
                "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
            }
        ):
            raise RuntimeError("M40 sealed source-chain identity fields differ")
        if not hasattr(materializer, "_assert_m40_source_delta"):
            raise RuntimeError("M40 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        materializer._assert_m40_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M40 exact repair/control source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m40_failed_pre_authoritative_m39_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Validate M40 before permitting it to mask M39's no-write attempt."""

    key = "failed_pre_authoritative_m39_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m40_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M40 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m40_failed_pre_authoritative_m39_successor_authority()
        )
        contract = materializer._validated_m40_live_preflight_contract(expected)
        reference = dict(materializer._m40_authority_reference())
        expected_cid = materializer._identity(expected)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M40 successor authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M40 successor reference differs")
        if seal.get(f"{key}_cid") != expected_cid:
            errors.append("M40 successor CID differs")

        m39_key = "committed_m38_evidence_reconciliation_successor_materialization"
        m39_expected = (
            materializer._expected_m39_committed_m38_evidence_reconciliation_authority()
        )
        m39_reference = dict(materializer._m39_authority_reference())
        m39_cid = materializer._identity(m39_expected)
        if (
            scheduler.get(m39_key) != m39_reference
            or migration.get(m39_key) != m39_reference
            or seal.get(f"{m39_key}_cid") != m39_cid
        ):
            errors.append("M40 historical M39 controls differ")

        failed = expected.get("failed_m39_pre_authoritative_attempt", {})
        prior = expected.get("prior_m39_authority", {})
        prior_reference = expected.get("prior_m39_reference", {})
        derivation = expected.get("target_projection_derivation", {})
        preservation = expected.get("preservation", {})
        changes = expected.get("exact_changes", {})
        source_chain = expected.get("source_chain", {})
        required_functions = (
            "_assert_m40_source_delta",
            "_validated_m40_live_preflight_contract",
            "_m40_source_binding_authority",
            "_materialize_m40",
            "_check_m40_materialized",
        )
        if (
            expected.get("schema")
            != "sawm/failed-pre-authoritative-m39-successor-authorization@1"
            or expected_cid
            != "sha256:377b6e7269a7025f236642f12aaf92264582f42a1efa4899eb4d6e64b0e41db2"
            or expected.get("migration_revision") != "SAWM-R2-M40"
            or expected.get("migration_kind") != key
            or expected.get("control_recorded_at") != "2026-09-01T03:00:00Z"
            or expected.get("target_generation") != 30
            or expected.get("target_event_watermark") != 292
            or expected.get("target_projection_cid")
            != "baguqeera6t2s6prg5atpg4gkgrlmqclu34firbn4z2o3wsgv6btp7p6q66tq"
            or not isinstance(prior, Mapping)
            or materializer._identity(prior) != m39_cid
            or prior_reference != m39_reference
            or failed.get("schema")
            != "sawm/pre-authoritative-m39-materialization-failure@1"
            or failed.get("attempt") != "SAWM-R2-M39-LIVE-A1"
            or failed.get("phase")
            != "generation_restart_row_verification"
            or failed.get("failure_kind")
            != "m39_top_level_authority_missing_stopped_owner"
            or failed.get("error") != "KeyError: 'stopped_owner'"
            or failed.get("source_head")
            != "64651e11f9d390a98a9daecc70c672e329487a1b"
            or failed.get("source_tree")
            != "75b32dda51c7475a0a54de28d6e6a3c693c2d7f3"
            or failed.get("quack_mutation_request_created") is not False
            or failed.get("event_292_rows_created") != 0
            or failed.get("evidence_node_changes") != 0
            or failed.get("m39_receipt_created") is not False
            or failed.get("task_revision_changes") != 0
            or failed.get("task_status_changes") != 0
            or failed.get("goal_revision_changes") != 0
            or failed.get("plan_revision_changes") != 0
            or failed.get("coordination_semantic_changes") != 0
            or failed.get("provider_invocations") != 0
            or failed.get("implementation_commit_changes") != 0
            or failed.get("merge_attempt_changes") != 0
            or failed.get("accepted_completion_changes") != 0
            or failed.get("worker_self_approval") is not False
            or contract.get("generation_restart_authorized") is not False
            or contract.get("prior_event_watermark") != 291
            or contract.get("target_event_watermark") != 292
            or derivation.get("recomputed_not_inherited") is not True
            or source_chain.get("restart_helper_repair_commit")
            != "00d15b870f7fdaa8e7165a94719eb2c7d3df7eca"
            or preservation.get("m39_receipt_created_or_rewritten") is not False
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or changes.get("evidence_event_changes") != 1
            or changes.get("store_generation_row_changes") != 0
            or changes.get("state_server_row_changes") != 0
            or changes.get("accepted_completion_changes") != 0
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M40 failed-M39/no-write successor delta is not exact")
        if require_active_runtime:
            runtime = scheduler.get("runtime_paths")
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            target_root = str(expected["target_runtime_root"])
            if (
                not isinstance(program, Mapping)
                or program.get("store_generation") != "30"
                or program.get("store_id") != expected["target_store_id"]
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M40 target/runtime binding is not exact")
        errors.extend(_m40_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M40 failed-pre-authoritative-M39 authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m39_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Return true when any protected surface declares the M39 successor."""

    key = "committed_m38_evidence_reconciliation_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m39_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    """Require the exact bounded repairs and direct nine-path M39 control child."""

    try:
        chain = authority.get("source_chain", {})
        json_blobs = chain.get("json_comparison_repair_blobs", {})
        envelope_blobs = chain.get("canonical_envelope_repair_blobs", {})
        materializer_blobs = chain.get("materializer_repair_blobs", {})
        if (
            not isinstance(chain, Mapping)
            or not isinstance(json_blobs, Mapping)
            or not isinstance(envelope_blobs, Mapping)
            or not isinstance(materializer_blobs, Mapping)
            or chain.get("base_control_commit")
            != "1ef0e89b5c4dd4418485adce4c9e6a0d66d18f94"
            or chain.get("base_control_tree")
            != "cd02f2ea823e1369bc9d453c473a503929bb8d6a"
            or chain.get("json_comparison_repair_parent")
            != "1ef0e89b5c4dd4418485adce4c9e6a0d66d18f94"
            or chain.get("json_comparison_repair_commit")
            != "146653af91fe3846cb98e49a54ae1173e3a3dc66"
            or chain.get("json_comparison_repair_tree")
            != "08c7ee04a5119ab091e70f4225303b83d2d8a5b0"
            or dict(json_blobs)
            != dict(materializer._M39_JSON_COMPARISON_REPAIR_BLOBS)
            or chain.get("canonical_envelope_repair_parent")
            != "146653af91fe3846cb98e49a54ae1173e3a3dc66"
            or chain.get("canonical_envelope_repair_commit")
            != "32d2966c4944157d664748c536cfa167f7ae38f5"
            or chain.get("canonical_envelope_repair_tree")
            != "eb62a255c6ebb8f15ebc9a69a9b850cb73e7983b"
            or dict(envelope_blobs)
            != dict(materializer._M39_CANONICAL_ENVELOPE_REPAIR_BLOBS)
            or chain.get("materializer_repair_parent")
            != "32d2966c4944157d664748c536cfa167f7ae38f5"
            or chain.get("materializer_repair_commit")
            != "b581305f42ad4eda6b3d749680e79107c1c150b3"
            or chain.get("materializer_repair_tree")
            != "c870812938d25731d11a694a2365c1650c03204c"
            or dict(materializer_blobs)
            != dict(materializer._M39_MATERIALIZER_REPAIR_BLOBS)
            or int(chain.get("bounded_repair_commit_count") or 0) != 3
            or chain.get("final_control_parent")
            != "b581305f42ad4eda6b3d749680e79107c1c150b3"
            or chain.get("final_control_commit_is_current_head") is not True
            or int(chain.get("final_control_commit_count") or 0) != 1
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M39_OPERATOR_CONTROL_PATHS)
            or set(materializer_blobs)
            != {
                "scripts/materialize_semantic_addressed_world_model_program.py",
                "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
            }
            or set(envelope_blobs)
            != {"scripts/materialize_semantic_addressed_world_model_program.py"}
            or len(json_blobs) != 8
        ):
            raise RuntimeError("M39 sealed source-chain identity fields differ")
        if not hasattr(materializer, "_assert_m39_source_delta"):
            raise RuntimeError("M39 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        materializer._assert_m39_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M39 exact bounded-repair/control source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m39_committed_m38_evidence_reconciliation_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Validate M39 before permitting it to mask M38's partial live attempt."""

    key = "committed_m38_evidence_reconciliation_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m39_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M39 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m39_committed_m38_evidence_reconciliation_authority()
        )
        contract = materializer._validated_m39_live_preflight_contract(expected)
        reference = dict(materializer._m39_authority_reference())
        expected_cid = materializer._identity(expected)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M39 reconciliation authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M39 reconciliation reference differs")
        if seal.get(f"{key}_cid") != expected_cid:
            errors.append("M39 reconciliation CID differs")
        materializer._assert_m39_historical_m38_controls(
            scheduler, migration, seal
        )
        prior = expected.get("prior_authority", {})
        derivation = expected.get("target_projection_derivation", {})
        preservation = expected.get("preservation", {})
        changes = expected.get("exact_changes", {})
        required_functions = (
            "_assert_m39_source_delta",
            "_m39_expected_historical_m38_rows",
            "_verify_m39_committed_m38_event",
            "_inspect_m39_live_projection",
            "_verify_m39_live_materialization",
            "_expected_m39_source_successor_receipt",
            "_materialize_m39",
            "_check_m39_materialized",
        )
        if (
            expected.get("schema")
            != "sawm/committed-m38-evidence-reconciliation-successor-authorization@1"
            or expected.get("migration_revision") != "SAWM-R2-M39"
            or expected.get("migration_kind") != key
            or expected.get("supersession_mode")
            != "append_only_committed_m38_evidence_reconciliation_source_seal"
            or expected_cid
            != "sha256:7b986174605b4f2739abf69473d0ce27dfe880d551c55144f076b8f981f633c2"
            or expected.get("target_generation") != 30
            or expected.get("target_event_watermark") != 292
            or expected.get("target_projection_cid")
            != "baguqeera6t2s6prg5atpg4gkgrlmqclu34firbn4z2o3wsgv6btp7p6q66tq"
            or prior.get("event_watermark") != 291
            or prior.get("event_id")
            != "baguqeeraixekx362a3e3ip42kxxpfkw23fldkpf74iqcnwkv3ihmibb3ibaq"
            or prior.get("evidence_id")
            != "baguqeeraycvlnf2opjttijhwva5ve4shweerffcgl4jq2sjyxwal5gojmoka"
            or prior.get("authorization_cid")
            != "sha256:c6d6c0b6951301d6b8bda94efade51d3e6ceb25dac3a82cdbc100e189c9cac19"
            or prior.get("receipt_present") is not False
            or prior.get("failure_stage")
            != "after_event_commit_before_receipt_publication"
            or contract.get("generation_restart_authorized") is not False
            or contract.get("m38_receipt_must_be_absent") is not True
            or contract.get("prior_event_watermark") != 291
            or contract.get("target_event_watermark") != 292
            or derivation.get("recomputed_not_inherited") is not True
            or preservation.get("committed_m38_event_291_preserved_exactly")
            is not True
            or preservation.get("m38_receipt_created_or_rewritten") is not False
            or preservation.get("same_live_generation_30_owner") is not True
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or changes.get("evidence_event_changes") != 1
            or changes.get("store_generation_row_changes") != 0
            or changes.get("state_server_row_changes") != 0
            or changes.get("accepted_completion_changes") != 0
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M39 committed-M38 reconciliation delta is not exact")
        if require_active_runtime:
            runtime = scheduler.get("runtime_paths")
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            target_root = str(expected["target_runtime_root"])
            if (
                not isinstance(program, Mapping)
                or program.get("store_generation") != "30"
                or program.get("store_id") != expected["target_store_id"]
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M39 target/runtime binding is not exact")
        errors.extend(_m39_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M39 committed-M38 reconciliation authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


# Live M38 event bodies are Quack envelopes; verification compares canonical JSON.
def _m38_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    key = "pre_authoritative_custody_restart_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m38_source_chain_identity_state(
    materializer: Any,
    authority: Mapping[str, Any],
) -> str:
    chain = authority.get("source_chain", {})
    runtime_blobs = chain.get("runtime_repair_blobs", {})
    control_blobs = chain.get("initial_control_blobs", {})
    if (
        not isinstance(chain, Mapping)
        or not isinstance(runtime_blobs, Mapping)
        or not isinstance(control_blobs, Mapping)
        or chain.get("runtime_repair_commit")
        != materializer._M38_RUNTIME_REPAIR_COMMIT
        or chain.get("runtime_repair_tree") != materializer._M38_RUNTIME_REPAIR_TREE
        or dict(runtime_blobs) != dict(materializer._M38_RUNTIME_REPAIR_BLOBS)
        or set(runtime_blobs) != set(materializer._M38_RUNTIME_REPAIR_PATHS)
        or chain.get("initial_control_commit")
        != materializer._M38_INITIAL_CONTROL_COMMIT
        or chain.get("initial_control_tree")
        != materializer._M38_INITIAL_CONTROL_TREE
        or chain.get("first_reseal_commit")
        != materializer._M38_FIRST_RESEAL_COMMIT
        or chain.get("first_reseal_tree") != materializer._M38_FIRST_RESEAL_TREE
        or chain.get("second_reseal_commit")
        != materializer._M38_SECOND_RESEAL_COMMIT
        or chain.get("third_reseal_commit")
        != materializer._M38_THIRD_RESEAL_COMMIT
        or chain.get("fourth_reseal_commit")
        != materializer._M38_FOURTH_RESEAL_COMMIT
        or chain.get("fifth_reseal_commit")
        != materializer._M38_FIFTH_RESEAL_COMMIT
        or chain.get("final_reseal_parent")
        != materializer._M38_FIFTH_RESEAL_COMMIT
        or dict(control_blobs) != dict(materializer._M38_INITIAL_CONTROL_BLOBS)
        or set(control_blobs) != set(materializer._M38_OPERATOR_CONTROL_PATHS)
        or len(runtime_blobs) != 4
        or len(control_blobs) != 9
    ):
        raise RuntimeError("M38 source-chain identity fields differ")
    runtime_identities = (
        chain.get("runtime_repair_commit"),
        chain.get("runtime_repair_tree"),
        *runtime_blobs.values(),
    )
    if not all(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{40}", value) is not None
        and value != "0" * 40
        for value in runtime_identities
    ):
        raise RuntimeError("M38 runtime repair identities are not sealed")
    control_identities = (
        chain.get("initial_control_commit"),
        chain.get("initial_control_tree"),
        chain.get("final_reseal_parent"),
        *control_blobs.values(),
    )
    expected_placeholder_blobs = {
        path: f"PENDING_M38_INITIAL_CONTROL_BLOB_{index}"
        for index, path in enumerate(
            sorted(materializer._M38_OPERATOR_CONTROL_PATHS), start=1
        )
    }
    if (
        chain.get("initial_control_commit")
        == "PENDING_M38_INITIAL_CONTROL_COMMIT"
        and chain.get("initial_control_tree")
        == "PENDING_M38_INITIAL_CONTROL_TREE"
        and dict(control_blobs) == expected_placeholder_blobs
    ):
        return "placeholder"
    if all(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{40}", value) is not None
        and value != "0" * 40
        for value in control_identities
    ):
        return "sealed"
    raise RuntimeError("M38 initial-control identities mix placeholder/sealed values")


def _m38_authority_reference_for_source_state(
    materializer: Any,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    if _m38_source_chain_identity_state(materializer, authority) == "placeholder":
        return {
            "schema": "sawm/operator-control-authority-reference@1",
            "migration_revision": "SAWM-R2-M38",
            "authority_cid": "sha256:PENDING_M38_AUTHORITY_CID",
        }
    return dict(materializer._m38_authority_reference())


def _m38_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    try:
        chain = authority.get("source_chain", {})
        if (
            chain.get("base_control_commit")
            != "02a16d6c76eb2f1f165b544c8c72d62c533d7d3b"
            or chain.get("base_control_tree")
            != "739df0426b5af860b461e68a4fbd9f1cc541c753"
            or chain.get("runtime_repair_commit")
            != "ad30bfa90cd0309a77a1a9936815f739e072a8a7"
            or chain.get("runtime_repair_tree")
            != "71a35a3ce148b203b4b2ee75919d51b5a4420657"
            or set(authority.get("runtime_repair_paths", ()))
            != set(materializer._M38_RUNTIME_REPAIR_PATHS)
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M38_OPERATOR_CONTROL_PATHS)
            or _git(root, "rev-parse", "02a16d6c76eb2f1f165b544c8c72d62c533d7d3b^{tree}")
            != "739df0426b5af860b461e68a4fbd9f1cc541c753"
            or _git(root, "rev-parse", "ad30bfa90cd0309a77a1a9936815f739e072a8a7^{tree}")
            != "71a35a3ce148b203b4b2ee75919d51b5a4420657"
            or _git(
                root,
                "diff",
                "--name-only",
                "02a16d6c76eb2f1f165b544c8c72d62c533d7d3b",
                "ad30bfa90cd0309a77a1a9936815f739e072a8a7",
            ).splitlines()
            != sorted(materializer._M38_RUNTIME_REPAIR_PATHS)
        ):
            raise RuntimeError("M38 base/runtime-repair identity differs")
        state = _m38_source_chain_identity_state(materializer, authority)
        if not hasattr(materializer, "_assert_m38_source_delta"):
            raise RuntimeError("M38 final source-delta verifier is unavailable")
        if state == "placeholder":
            return []
        population = materializer.build_population(root)
        materializer._assert_m38_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M38 exact custody repair/control/reseal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m38_pre_authoritative_custody_restart_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    key = "pre_authoritative_custody_restart_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m38_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M38 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m38_pre_authoritative_custody_restart_authority()
        )
        materializer._assert_m38_historical_m37_triplet(
            scheduler, migration, seal
        )
        contract = materializer._validated_m38_live_preflight_contract(expected)
        reference = _m38_authority_reference_for_source_state(materializer, expected)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M38 custody restart authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M38 custody restart reference differs")
        state = _m38_source_chain_identity_state(materializer, expected)
        expected_cid = (
            "sha256:PENDING_M38_AUTHORITY_CID"
            if state == "placeholder"
            else materializer._identity(expected)
        )
        if seal.get(f"{key}_cid") != expected_cid:
            errors.append("M38 custody restart CID differs")
        if (
            state == "sealed"
            and expected_cid
            != "sha256:664af21f470ada7e4d4bf02313df473ed345c539b122f30036db8c8ee171a8eb"
        ):
            errors.append("M38 sealed custody restart CID differs")
        failed = expected.get("failed_m37_pre_authority_attempt", {})
        custody = expected.get("pre_authoritative_custody_repair", {})
        lane = expected.get("stale_lane_pid_repair", {})
        derivation = expected.get("target_projection_derivation", {})
        zero_fields = (
            "generation_30_rows_created",
            "event_291_rows_created",
            "evidence_nodes_created",
            "task_status_changes",
            "task_revision_changes",
            "goal_revision_changes",
            "plan_revision_changes",
            "effect_claim_changes",
            "implementation_provider_invocations",
            "accepted_completion_changes",
        )
        required_functions = (
            "_assert_m38_source_delta",
            "_check_m38_prestart_admission",
            "_m38_projection_cid_at_watermark",
            "_verify_m38_live_materialization",
            "_expected_m38_source_successor_receipt",
            "_materialize_m38",
            "_check_m38_materialized",
        )
        if (
            expected.get("schema")
            != "sawm/pre-authoritative-custody-restart-successor-authorization@1"
            or expected.get("migration_revision") != "SAWM-R2-M38"
            or expected.get("migration_kind") != key
            or expected.get("control_recorded_at") != "2026-09-01T01:10:00Z"
            or expected.get("target_generation") != 30
            or expected.get("target_event_watermark") != 291
            or expected.get("target_projection_cid")
            != "baguqeeravycbuo73fyu5mpad55qi5duk3la53lubqeu7nu6kjtnahehjtnsq"
            or expected.get("prior_authority", {}).get("control_store_sha256")
            != "ea5b66208455f398502e8ad939566a5957f5bc65f35ed5afe8cc3998be66eb41"
            or expected.get("prior_authority", {}).get("read_replica_sha256")
            != "ea5b66208455f398502e8ad939566a5957f5bc65f35ed5afe8cc3998be66eb41"
            or expected.get("prior_authority", {}).get("event_watermark") != 290
            or expected.get("prior_authority", {}).get("projection_cid")
            != "baguqeerahwerrrfx6cx6ukpljlp2r4i32lkac3bnhq5ej2f3hozg7cnt6shq"
            or expected.get("superseded_m37_authority", {}).get("authority_cid")
            != "sha256:c776180b7e65de98d5de235765db60148f7693148512b335260ddb772563a795"
            or failed.get("failure_phase")
            != "after_database_open_and_checkpoint_before_identity_publication"
            or failed.get("inotify_add_watch_errno") != 28
            or any(failed.get(field) != 0 for field in zero_fields)
            or custody.get("database_open_before_custody_forbidden") is not True
            or lane.get("live_or_changed_pid_fails_closed") is not True
            or len(lane.get("evidence", {})) != 4
            or derivation.get("recomputed_not_inherited") is not True
            or contract.get("pre_authoritative_custody_required") is not True
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M38 pre-authoritative custody restart delta is not exact")
        if require_active_runtime:
            runtime = scheduler.get("runtime_paths")
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            target_root = str(expected["target_runtime_root"])
            if (
                not isinstance(program, Mapping)
                or program.get("store_generation") != "30"
                or program.get("store_id") != expected["target_store_id"]
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M38 target/runtime binding is not exact")
        errors.extend(_m38_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M38 custody restart authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m37_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Return true when any protected surface declares the M37 successor."""

    key = "post_reboot_generation_restart_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m37_source_chain_identity_state(
    materializer: Any,
    authority: Mapping[str, Any],
) -> str:
    """Return ``placeholder`` or ``sealed`` for one complete M37 chain."""

    chain = authority.get("source_chain", {})
    blobs = chain.get("initial_control_blobs", {})
    if (
        not isinstance(chain, Mapping)
        or not isinstance(blobs, Mapping)
        or chain.get("initial_control_commit")
        != materializer._M37_INITIAL_CONTROL_COMMIT
        or chain.get("initial_control_tree")
        != materializer._M37_INITIAL_CONTROL_TREE
        or chain.get("final_reseal_parent")
        != materializer._M37_INITIAL_CONTROL_COMMIT
        or dict(blobs) != dict(materializer._M37_INITIAL_CONTROL_BLOBS)
        or set(blobs) != set(authority.get("operator_control_paths", ()))
        or len(blobs) != 9
    ):
        raise RuntimeError("M37 source-chain identity fields differ")
    placeholder_blobs = {
        path: f"PENDING_M37_INITIAL_BLOB_{index:02d}"
        for index, path in enumerate(sorted(blobs), start=1)
    }
    if (
        chain.get("initial_control_commit")
        == "PENDING_M37_INITIAL_CONTROL_COMMIT"
        and chain.get("initial_control_tree")
        == "PENDING_M37_INITIAL_CONTROL_TREE"
        and chain.get("final_reseal_parent")
        == "PENDING_M37_INITIAL_CONTROL_COMMIT"
        and dict(blobs) == placeholder_blobs
    ):
        return "placeholder"
    identities = (
        chain.get("initial_control_commit"),
        chain.get("initial_control_tree"),
        chain.get("final_reseal_parent"),
        *blobs.values(),
    )
    if all(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{40}", value) is not None
        and value != "0" * 40
        for value in identities
    ):
        return "sealed"
    raise RuntimeError(
        "M37 source identities mix placeholder and sealed values or use "
        "noncanonical placeholder sentinels"
    )


def _m37_authority_reference_for_source_state(
    materializer: Any,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the only admissible compact reference for the chain state."""

    if _m37_source_chain_identity_state(materializer, authority) == "placeholder":
        return {
            "schema": "sawm/operator-control-authority-reference@1",
            "migration_revision": "SAWM-R2-M37",
            "authority_cid": "sha256:PENDING_M37_AUTHORITY_CID",
        }
    return dict(materializer._m37_authority_reference())


def _m37_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
) -> list[str]:
    """Validate the exact M36 base and complete M37 control/reseal chain."""

    try:
        chain = authority.get("source_chain", {})
        if (
            not isinstance(chain, Mapping)
            or chain.get("base_control_commit")
            != "a3db1cde328c5aeba86896d4f6813821251ceb7e"
            or chain.get("base_control_tree")
            != "fba8c205656afda738ac5f14c2841fb1da452ea0"
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M37_OPERATOR_CONTROL_PATHS)
            or set(materializer._M37_OPERATOR_CONTROL_PATHS)
            != set(materializer._M36_OPERATOR_CONTROL_PATHS)
            or _git(
                root,
                "rev-parse",
                "a3db1cde328c5aeba86896d4f6813821251ceb7e^{tree}",
            )
            != "fba8c205656afda738ac5f14c2841fb1da452ea0"
        ):
            raise RuntimeError("M37 accepted base/control path identity differs")
        state = _m37_source_chain_identity_state(materializer, authority)
        if not hasattr(materializer, "_assert_m37_source_delta"):
            raise RuntimeError("M37 final source-delta verifier is unavailable")
        if state == "placeholder":
            return []
        population = materializer.build_population(root)
        materializer._assert_m37_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M37 exact post-reboot restart control/reseal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m37_post_reboot_generation_restart_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Validate M37's one-shot generation-29 to generation-30 restart."""

    key = "post_reboot_generation_restart_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m37_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M37 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m37_post_reboot_generation_restart_authority()
        contract = materializer._validated_m37_live_preflight_contract(expected)
        reference = _m37_authority_reference_for_source_state(
            materializer, expected
        )
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M37 post-reboot restart authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M37 post-reboot restart reference differs")
        expected_cid = materializer._identity(expected)
        sealed_cid = seal.get(f"{key}_cid")
        identity_state = _m37_source_chain_identity_state(materializer, expected)
        expected_seal_cid = (
            "sha256:PENDING_M37_AUTHORITY_CID"
            if identity_state == "placeholder"
            else expected_cid
        )
        if sealed_cid != expected_seal_cid:
            errors.append("M37 post-reboot restart CID differs")

        binding = expected.get("runtime_binding", {})
        prior = expected.get("prior_authority", {})
        m36_anchor = expected.get("m36_historical_anchor", {})
        operational_suffix = expected.get("post_m36_operational_suffix", {})
        recovery = expected.get("stale_owner_recovery", {})
        stopped = expected.get("stopped_owner", {})
        target = expected.get("target_authority", {})
        repair = expected.get("accepted_control_plane_repair", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        chain = expected.get("source_chain", {})
        receipt_cid = str(m36_anchor.get("receipt_cid") or "")
        recovery_cid = str(recovery.get("receipt_cid") or "")
        required_functions = (
            "_assert_m37_source_delta",
            "_check_m37_prestart_admission",
            "_inspect_m37_live_projection",
            "_verify_m37_live_materialization",
            "_expected_m37_source_successor_receipt",
        )
        if (
            expected.get("schema")
            != "sawm/post-reboot-generation-restart-successor-authorization@1"
            or expected.get("authorized") is not True
            or expected.get("authority") != "operator_control_plane"
            or expected.get("migration_revision") != "SAWM-R2-M37"
            or expected.get("migration_kind") != key
            or expected.get("supersession_mode")
            != "generation_bearing_post_reboot_restart_source_seal"
            or expected.get("control_recorded_at") != "2026-09-01T00:10:00Z"
            or expected_cid
            != "sha256:c776180b7e65de98d5de235765db60148f7693148512b335260ddb772563a795"
            or materializer._M37_INITIAL_CONTROL_COMMIT
            != "fb6672403850bc4b473db8e5e759176e3028adc0"
            or materializer._M37_INITIAL_CONTROL_TREE
            != "bb9132b3dd333ad11c60efac45186e1fc6cdb11a"
            or expected.get("target_generation") != 30
            or expected.get("target_event_watermark") != 291
            or expected.get("target_projection_cid")
            != "baguqeeravycbuo73fyu5mpad55qi5duk3la53lubqeu7nu6kjtnahehjtnsq"
            or expected.get("prior_database_uuid") != materializer._M37_DATABASE_UUID
            or binding.get("run_id") != "run-r2-m27"
            or binding.get("store_generation") != 30
            or binding.get("prior_event_watermark") != 290
            or binding.get("target_event_watermark") != 291
            or binding.get("quack_port") != 24_070
            or binding.get("quack_endpoint") != "quack:127.0.0.1:24070"
            or binding.get("database_uuid") != expected.get("prior_database_uuid")
            or prior.get("schema") != "sawm/current-operational-head@1"
            or prior.get("event_watermark") != 290
            or prior.get("event_prefix_sha256")
            != "27ec7ecda1536d1bf7ed4b7e709b3b7284efcc84fbfb804b63db437ed31bd22c"
            or prior.get("projection_cid")
            != "baguqeerahwerrrfx6cx6ukpljlp2r4i32lkac3bnhq5ej2f3hozg7cnt6shq"
            or prior.get("semantic_authority_digest")
            != "sha256:f51d9cb949538441218254297e279fa2bf5884e1bdc9d20693cf8213db841dde"
            or prior.get("source_head")
            != "a3db1cde328c5aeba86896d4f6813821251ceb7e"
            or prior.get("source_tree")
            != "fba8c205656afda738ac5f14c2841fb1da452ea0"
            or prior.get("expected_task_heads")
            != materializer._m37_expected_task_heads()
            or m36_anchor.get("migration_revision") != "SAWM-R2-M36"
            or m36_anchor.get("authorization_cid")
            != "sha256:d366f997c8972a3fb9a76ecd386c80f3a33d3001e215ca0f8cb60c0c653f760c"
            or m36_anchor.get("event_watermark") != 286
            or m36_anchor.get("event_prefix_sha256")
            != "d415613fd55f337c0142ce09ce770e73b4956006fab77cab50fbf19ea5bd9b00"
            or m36_anchor.get("projection_cid")
            != "baguqeeravzrhagxizn7o45ukuevzkhreb7dzpabd4g4kmyci2if5rxr32dda"
            or m36_anchor.get("semantic_authority_digest")
            != "sha256:395168339f24de03f6d6f91cc0ca0365df2ffa8d71a80ac9fecab6e282817163"
            or m36_anchor.get("receipt_sha256")
            != "92dcb41ab624b7327a7c56b3725fe84f7114e76b5e69c1ab7081e5f7ab62d764"
            or m36_anchor.get("receipt_size") != 2_793
            or receipt_cid
            != "sha256:b37bb7a32eefdb431e6bad9faec723d58b7cf1dc69fdc52bac43621da6047c1e"
            or operational_suffix != materializer._m37_post_m36_operational_suffix()
            or operational_suffix.get("from_event_exclusive") != 286
            or operational_suffix.get("to_event_inclusive") != 290
            or len(operational_suffix.get("events", ())) != 4
            or operational_suffix.get("task_head_changes_from_m36")
            != {
                "SAWM-006": {
                    "prior_status": "in_progress",
                    "prior_revision": 7,
                    "status": "in_progress",
                    "revision": 9,
                },
                "SAWM-008": {
                    "prior_status": "in_progress",
                    "prior_revision": 9,
                    "status": "in_progress",
                    "revision": 11,
                },
            }
            or operational_suffix.get("expired_attempt_provider_invocation_count")
            != 0
            or operational_suffix.get("expired_attempt_effect_claim_count") != 0
            or operational_suffix.get("accepted_completion_changes") != 0
            or recovery.get("receipt_sha256")
            != "6a5375adc871d60a8e85c7ee86bd48af33fcaf5b653c822026075dba642ac011"
            or recovery.get("receipt_size") != 727
            or recovery_cid
            != "baguqeerayoxp2turydlpaytskth6jevi23wdp3iiheubcsu2q5hs3fk55dma"
            or recovery.get("status_sha256")
            != "c2675d2efb2fff8672cb926d7930fd79ea2c2cba20e341eccbc0e4abd181e21a"
            or recovery.get("status_size") != 2_491
            or recovery.get("owner_liveness") != "dead"
            or recovery.get("resulting_status") != "stopped"
            or recovery.get("database_bookkeeping_settled") is not True
            or recovery.get("owner_marker_removed") is not True
            or recovery.get("replay_safe") is not True
            or recovery.get("task_completion_authority") is not False
            or stopped.get("generation") != 29
            or stopped.get("target_generation") != 30
            or stopped.get("status") != "stopped"
            or stopped.get("server_id")
            != "server:1205bb8e-2f09-440c-829a-50458e9f9e7f"
            or stopped.get("process_birth_id")
            != "birth:2458ebe71d9a348272706db40cfd3fec"
            or stopped.get("stopped_at") != "2026-09-01T00:00:51Z"
            or stopped.get("target_identity_is_runtime_generated") is not True
            or target.get("event_watermark") != 291
            or target.get("projection_cid")
            != "baguqeeravycbuo73fyu5mpad55qi5duk3la53lubqeu7nu6kjtnahehjtnsq"
            or target.get("plan_revision") != 28
            or target.get("operator_task_alias") != "SAWM-000"
            or target.get("operator_task_cid")
            != "sha256:8b8f43dd51ea4d8467af0e5cae4100478f16666d36c6f4fad49c23fd8e43a3d6"
            or target.get("operator_task_status") != "completed"
            or target.get("operator_task_revision") != 2
            or contract.get("migration_revision") != "SAWM-R2-M37"
            or contract.get("target_generation") != 30
            or contract.get("prior_event_watermark") != 290
            or contract.get("target_event_watermark") != 291
            or contract.get("expected_task_heads")
            != materializer._m37_expected_task_heads()
            or repair
            != {
                "defect": "host_reboot_terminated_generation_29_after_m36",
                "resolution": (
                    "authorize_one_exact_recovered_generation_29_to_30_restart"
                ),
                "changed_paths": sorted(materializer._M37_OPERATOR_CONTROL_PATHS),
                "ordinary_program_implementation": False,
                "authority_weakened": False,
                "receipts_rewritten": False,
                "worker_self_approval": False,
            }
            or changes
            != {
                "event_suffix_length": 1,
                "evidence_node_changes": 1,
                "store_generation_row_changes": 1,
                "state_server_row_changes": 1,
                "server_epoch_row_changes": 1,
                "capability_snapshot_row_changes": 1,
                "credential_row_changes": 1,
                "task_revision_changes": 0,
                "task_status_changes": 0,
                "goal_revision_changes": 0,
                "plan_revision_changes": 0,
                "accepted_definition_changes": 0,
                "accepted_completion_changes": 0,
                "coordination_semantic_changes": 0,
                "sidecar_semantic_changes": 0,
                "effect_claim_changes": 0,
                "implementation_commit_changes": 0,
                "implementation_provider_invocations": 0,
                "merge_attempt_changes": 0,
            }
            or preservation
            != {
                "m36_historical_anchor_preserved": True,
                "post_m36_operational_suffix_preserved": True,
                "stale_owner_recovery_receipt_preserved": True,
                "generation_29_preserved_stopped": True,
                "same_database_uuid": True,
                "same_runtime_root": True,
                "same_store_path": True,
                "generation_bearing_owner_restart": True,
                "control_base_copied": False,
                "coordination_base_copied": False,
                "task_heads_preserved": True,
                "plan_head_preserved": True,
                "sidecars_preserved": True,
                "worktrees_copied": False,
                "worker_self_approval": False,
            }
            or expected.get("ordinary_source_changes") != 0
            or chain.get("base_control_commit")
            != "a3db1cde328c5aeba86896d4f6813821251ceb7e"
            or chain.get("base_control_tree")
            != "fba8c205656afda738ac5f14c2841fb1da452ea0"
            or identity_state not in {"placeholder", "sealed"}
            or any(not hasattr(materializer, name) for name in required_functions)
        ):
            errors.append("M37 post-reboot generation restart delta is not exact")

        if require_active_runtime:
            target_root = str(expected["target_runtime_root"])
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            runtime = scheduler.get("runtime_paths")
            if (
                not isinstance(program, Mapping)
                or program.get("store_id") != expected["target_store_id"]
                or program.get("store_generation") != "30"
                or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("store_id") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M37 target/runtime binding is not exact")
        errors.extend(_m37_source_chain_errors(root, materializer, expected))
        return errors
    except Exception as exc:
        return [
            "M37 post-reboot restart authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m36_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    key = "operator_task_binding_correction_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m36_source_chain_identity_state(
    materializer: Any,
    authority: Mapping[str, Any],
) -> str:
    """Return ``placeholder`` or ``sealed`` for one complete M36 chain."""

    chain = authority.get("source_chain", {})
    blobs = chain.get("initial_control_blobs", {})
    if (
        not isinstance(chain, Mapping)
        or not isinstance(blobs, Mapping)
        or chain.get("initial_control_commit")
        != materializer._M36_INITIAL_CONTROL_COMMIT
        or chain.get("initial_control_tree") != materializer._M36_INITIAL_CONTROL_TREE
        or chain.get("final_reseal_parent")
        != materializer._M36_INITIAL_CONTROL_COMMIT
        or dict(blobs) != dict(materializer._M36_INITIAL_CONTROL_BLOBS)
        or set(blobs) != set(authority.get("operator_control_paths", ()))
        or len(blobs) != 9
    ):
        raise RuntimeError("M36 source-chain identity fields differ")
    zero = "0" * 40
    identities = (
        chain.get("initial_control_commit"),
        chain.get("initial_control_tree"),
        chain.get("final_reseal_parent"),
        *blobs.values(),
    )
    if all(value == zero for value in identities):
        return "placeholder"
    if all(
        isinstance(value, str)
        and value != zero
        and re.fullmatch(r"[0-9a-f]{40}", value) is not None
        for value in identities
    ):
        return "sealed"
    raise RuntimeError("M36 source identities mix placeholder and sealed values")


def _m36_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    """Validate M36's exact M35 base and complete repair/reseal chain."""

    try:
        chain = authority.get("source_chain", {})
        if (
            not isinstance(chain, Mapping)
            or chain.get("base_control_commit")
            != "5bbf2dec97458585ee95034c986057a1552b805d"
            or chain.get("base_control_tree")
            != "954736c103dd644d7896e032c9676ed38d989831"
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M36_OPERATOR_CONTROL_PATHS)
            or set(materializer._M36_OPERATOR_CONTROL_PATHS)
            != set(materializer._M35_OPERATOR_CONTROL_PATHS)
            or _git(
                root,
                "rev-parse",
                "5bbf2dec97458585ee95034c986057a1552b805d^{tree}",
            )
            != "954736c103dd644d7896e032c9676ed38d989831"
        ):
            raise RuntimeError("M36 accepted base/control path identity differs")
        state = _m36_source_chain_identity_state(materializer, authority)
        if state == "placeholder":
            return []
        if not hasattr(materializer, "_assert_m36_source_delta"):
            raise RuntimeError("M36 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m36_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M36 exact operator-task binding repair/reseal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m36_operator_task_binding_correction_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    key = "operator_task_binding_correction_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m36_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M36 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m36_operator_task_binding_correction_authority()
        )
        reference = materializer._m36_authority_reference()
        contract = materializer._validated_m36_live_preflight_contract(expected)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append(
                "M36 operator-task binding authority is only partially declared"
            )
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M36 operator-task binding reference differs")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M36 operator-task binding CID differs")
        binding = expected.get("runtime_binding", {})
        prior = expected.get("prior_authority", {})
        superseded = expected.get("superseded_source_authority", {})
        failed = expected.get("failed_m35_append", {})
        target = expected.get("target_authority", {})
        repair = expected.get("accepted_control_plane_repair", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        source_chain = expected.get("source_chain", {})
        identity_state = _m36_source_chain_identity_state(materializer, expected)
        if (
            expected.get("schema")
            != "sawm/operator-task-binding-correction-successor-authorization@1"
            or expected.get("migration_revision") != "SAWM-R2-M36"
            or expected.get("migration_kind") != key
            or expected.get("supersession_mode")
            != "append_only_operator_task_binding_correction"
            or expected.get("control_recorded_at") != "2026-08-31T20:50:00Z"
            or expected.get("target_generation") != 29
            or expected.get("target_event_watermark") != 286
            or expected.get("target_projection_cid")
            != "baguqeeravzrhagxizn7o45ukuevzkhreb7dzpabd4g4kmyci2if5rxr32dda"
            or binding.get("run_id") != "run-r2-m27"
            or binding.get("prior_event_watermark") != 285
            or binding.get("store_generation") != 29
            or binding.get("quack_port") != 24_070
            or prior.get("migration_revision") != "SAWM-R2-M34"
            or prior.get("event_prefix_sha256")
            != "79797c1e1593fa880a4ac088796e1ca713dc276df8ad118c12aa0c1ab49f1c7f"
            or prior.get("projection_cid")
            != "baguqeeragsizyo6v4izu7qfvjbj5l5bjkuycw2xyaf3vhlzx2nai7xyrvd4q"
            or prior.get("m34_receipt_cid")
            != "sha256:759b71e0d0fa73a1ac81bb98fb1b96c93f3150b09667cd19fc05330b34b50b01"
            or superseded.get("migration_revision") != "SAWM-R2-M35"
            or superseded.get("authority_cid")
            != "sha256:be555c52e0ddf0737b1f3796ab21618e11226c7fae14d176c50d0cb4edfaf5d6"
            or superseded.get("final_control_commit")
            != "5bbf2dec97458585ee95034c986057a1552b805d"
            or superseded.get("final_control_tree")
            != "954736c103dd644d7896e032c9676ed38d989831"
            or superseded.get("materialized") is not False
            or superseded.get("receipt_created") is not False
            or failed.get("failure_kind") != "operator_task_cid_not_found"
            or failed.get("sealed_target_task_cid")
            != "sha256:308a38585461080c06bf51f36a5b9cff75c4bf5a6e88ddcbccbca73198a51d1d"
            or failed.get("sealed_target_task_present") is not False
            or failed.get("authoritative_operator_task_alias") != "SAWM-000"
            or failed.get("authoritative_operator_task_cid")
            != "sha256:8b8f43dd51ea4d8467af0e5cae4100478f16666d36c6f4fad49c23fd8e43a3d6"
            or failed.get("authoritative_operator_task_present") is not True
            or failed.get("event_watermark_before") != 285
            or failed.get("event_watermark_after") != 285
            or failed.get("m35_event_appended") is not False
            or failed.get("m35_receipt_created") is not False
            or target.get("operator_task_alias") != "SAWM-000"
            or target.get("operator_task_cid")
            != "sha256:8b8f43dd51ea4d8467af0e5cae4100478f16666d36c6f4fad49c23fd8e43a3d6"
            or target.get("operator_task_status") != "completed"
            or target.get("operator_task_revision") != 2
            or contract.get("migration_revision") != "SAWM-R2-M36"
            or contract.get("expected_task_heads")
            != materializer._m30_expected_task_heads()
            or repair.get("defect")
            != "sealed_operator_evidence_target_task_cid_was_absent"
            or repair.get("resolution")
            != "bind_operator_evidence_to_exact_authoritative_sawm_000_task"
            or repair.get("authority_weakened") is not False
            or repair.get("receipts_rewritten") is not False
            or repair.get("worker_self_approval") is not False
            or changes.get("event_suffix_length") != 1
            or changes.get("operator_task_binding_correction_changes") != 1
            or any(
                changes.get(field) != 0
                for field in (
                    "task_revision_changes",
                    "task_status_changes",
                    "plan_revision_changes",
                    "goal_revision_changes",
                    "owner_generation_changes",
                    "coordination_semantic_changes",
                    "accepted_completion_changes",
                )
            )
            or preservation.get("same_live_owner") is not True
            or preservation.get("generation_restart") is not False
            or preservation.get("m34_receipt_preserved") is not True
            or preservation.get("m35_source_controls_preserved") is not True
            or preservation.get("failed_m35_append_preserved") is not True
            or preservation.get("m35_receipt_absent") is not True
            or source_chain.get("base_control_commit")
            != "5bbf2dec97458585ee95034c986057a1552b805d"
            or source_chain.get("base_control_tree")
            != "954736c103dd644d7896e032c9676ed38d989831"
            or source_chain.get("initial_control_commit")
            != "e208bc490b8d12f6d86b286d98d1ac63bb4e62be"
            or source_chain.get("initial_control_tree")
            != "53fb95b9fd8e4f6f776c0380b0f1f959ddd58326"
            or identity_state != "sealed"
        ):
            errors.append("M36 operator-task binding correction delta is not exact")
        if require_active_runtime:
            target_root = str(expected["target_runtime_root"])
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            runtime = scheduler.get("runtime_paths")
            if (
                not isinstance(program, Mapping)
                or program.get("store_id") != expected["target_store_id"]
                or program.get("store_generation") != "29"
                or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("store_id") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M36 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m37_successor_declared(
            scheduler, seal, migration
        ):
            m37 = (
                materializer._expected_m37_post_reboot_generation_restart_authority()
            )
            m37_source_chain = m37.get("source_chain", {})
            if isinstance(m37_source_chain, Mapping):
                historical_control_head = str(
                    m37_source_chain.get("base_control_commit") or ""
                ) or None
            if historical_control_head is None:
                errors.append("M37 prior M36 control source head is absent")
        errors.extend(
            _m36_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M36 operator-task binding authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m35_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    key = "immutable_authority_identity_normalization_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m35_source_chain_identity_state(
    materializer: Any,
    authority: Mapping[str, Any],
) -> str:
    """Return ``placeholder`` or ``sealed`` for one complete M35 chain."""

    chain = authority.get("source_chain", {})
    blobs = chain.get("initial_control_blobs", {})
    if (
        not isinstance(chain, Mapping)
        or not isinstance(blobs, Mapping)
        or chain.get("initial_control_commit")
        != materializer._M35_INITIAL_CONTROL_COMMIT
        or chain.get("initial_control_tree") != materializer._M35_INITIAL_CONTROL_TREE
        or chain.get("final_reseal_parent")
        != materializer._M35_INITIAL_CONTROL_COMMIT
        or dict(blobs) != dict(materializer._M35_INITIAL_CONTROL_BLOBS)
        or set(blobs) != set(authority.get("operator_control_paths", ()))
        or len(blobs) != 9
    ):
        raise RuntimeError("M35 source-chain identity fields differ")
    zero = "0" * 40
    identities = (
        chain.get("initial_control_commit"),
        chain.get("initial_control_tree"),
        chain.get("final_reseal_parent"),
        *blobs.values(),
    )
    if all(value == zero for value in identities):
        return "placeholder"
    if all(
        isinstance(value, str)
        and value != zero
        and re.fullmatch(r"[0-9a-f]{40}", value) is not None
        for value in identities
    ):
        return "sealed"
    raise RuntimeError("M35 source identities mix placeholder and sealed values")


def _m35_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    """Validate the accepted base plus a complete placeholder or final chain."""

    try:
        chain = authority.get("source_chain", {})
        if (
            not isinstance(chain, Mapping)
            or chain.get("base_control_commit")
            != "4dfe1c4c81ffd65f6a2d5c5cdc38b1cd33f1f443"
            or chain.get("base_control_tree")
            != "894d9a4d206faf4e59328111023ec44fcf26f96e"
            or set(authority.get("operator_control_paths", ()))
            != set(materializer._M35_OPERATOR_CONTROL_PATHS)
            or set(materializer._M35_OPERATOR_CONTROL_PATHS)
            != set(materializer._M34_OPERATOR_CONTROL_PATHS)
            or _git(
                root,
                "rev-parse",
                "4dfe1c4c81ffd65f6a2d5c5cdc38b1cd33f1f443^{tree}",
            )
            != "894d9a4d206faf4e59328111023ec44fcf26f96e"
        ):
            raise RuntimeError("M35 accepted base/control path identity differs")
        state = _m35_source_chain_identity_state(materializer, authority)
        if state == "placeholder":
            return []
        if not hasattr(materializer, "_assert_m35_source_delta"):
            raise RuntimeError("M35 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m35_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M35 exact immutable-authority repair/reseal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m35_immutable_authority_identity_normalization_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    key = "immutable_authority_identity_normalization_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m35_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M35 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m35_immutable_authority_identity_normalization_authority()
        )
        reference = materializer._m35_authority_reference()
        contract = materializer._validated_m35_live_preflight_contract(expected)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append(
                "M35 immutable-authority identity authority is only partially declared"
            )
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M35 immutable-authority identity reference differs")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M35 immutable-authority identity CID differs")
        binding = expected.get("runtime_binding", {})
        prior = expected.get("prior_authority", {})
        repair = expected.get("accepted_control_plane_repair", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        source_chain = expected.get("source_chain", {})
        identity_state = _m35_source_chain_identity_state(materializer, expected)
        sealed_initial_blobs = {
            "config/agent_supervisor_semantic_addressed_world_model_scheduler.json": "21ab73ae195f9514a713d83e8442d29db9c27547",
            "config/semantic_addressed_world_model_dependencies.seal.json": "f30dee923eeeba60b391c1af0982b525055fa441",
            "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md": "da11d8b4d5f5eed7e5d955b51deded59abc34059",
            "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json": "560f6f039b8934ef6b25d32758db5f7374b5d872",
            "scripts/materialize_semantic_addressed_world_model_program.py": "159db3cdedb34b0d406b8c1d9b7bd8b71eb3b83f",
            "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": "99129c72bb1ffc63e5624649e64982dfcb46c5f9",
            "scripts/validate_semantic_addressed_world_model_board.py": "41ad96cb9a5cb9ae412f3124c6f4a3e6aeaf19d6",
            "scripts/validate_semantic_addressed_world_model_dependencies.py": "b33b9a61c7f930154d65f7b97cf6b73e0855bc4c",
            "test/api/semantic_world/test_semantic_addressed_world_model_board.py": "345f964cdb11e80e9840a0d779aea345c3746fd9",
        }
        if (
            expected.get("schema")
            != "sawm/immutable-authority-identity-normalization-successor-authorization@1"
            or expected.get("migration_revision") != "SAWM-R2-M35"
            or expected.get("migration_kind") != key
            or expected.get("supersession_mode")
            != "append_only_immutable_authority_identity_normalization"
            or expected.get("control_recorded_at") != "2026-08-31T20:20:00Z"
            or expected.get("target_generation") != 29
            or expected.get("target_event_watermark") != 286
            or expected.get("target_projection_cid")
            != "baguqeeravzrhagxizn7o45ukuevzkhreb7dzpabd4g4kmyci2if5rxr32dda"
            or binding.get("run_id") != "run-r2-m27"
            or binding.get("prior_event_watermark") != 285
            or binding.get("store_generation") != 29
            or binding.get("quack_port") != 24_070
            or prior.get("migration_revision") != "SAWM-R2-M34"
            or prior.get("event_prefix_sha256")
            != "79797c1e1593fa880a4ac088796e1ca713dc276df8ad118c12aa0c1ab49f1c7f"
            or prior.get("projection_cid")
            != "baguqeeragsizyo6v4izu7qfvjbj5l5bjkuycw2xyaf3vhlzx2nai7xyrvd4q"
            or prior.get("m34_receipt_sha256")
            != "3ec2d6100998f430a388e4c32cbcdda2feb53f070b683b10bfef28ccfa85a872"
            or prior.get("m34_receipt_cid")
            != "sha256:759b71e0d0fa73a1ac81bb98fb1b96c93f3150b09667cd19fc05330b34b50b01"
            or contract.get("migration_revision") != "SAWM-R2-M35"
            or contract.get("expected_task_heads")
            != materializer._m30_expected_task_heads()
            or repair.get("defect")
            != "mappingproxy_authority_hashed_without_normalization"
            or repair.get("resolution")
            != "normalize_closed_authority_mapping_before_identity_calculation_at_operator_call_boundary"
            or repair.get("authority_weakened") is not False
            or changes.get("event_suffix_length") != 1
            or changes.get("immutable_authority_identity_normalization_changes") != 1
            or any(
                changes.get(field) != 0
                for field in (
                    "task_revision_changes",
                    "task_status_changes",
                    "plan_revision_changes",
                    "goal_revision_changes",
                    "owner_generation_changes",
                    "coordination_semantic_changes",
                    "accepted_completion_changes",
                )
            )
            or preservation.get("same_live_owner") is not True
            or preservation.get("generation_restart") is not False
            or preservation.get("m34_receipt_preserved") is not True
            or source_chain.get("base_control_commit")
            != "4dfe1c4c81ffd65f6a2d5c5cdc38b1cd33f1f443"
            or source_chain.get("base_control_tree")
            != "894d9a4d206faf4e59328111023ec44fcf26f96e"
            or source_chain.get("initial_control_commit")
            != "27c5e1e5925228757fc02378eb9d3b2a8addc60e"
            or source_chain.get("initial_control_tree")
            != "75f9c888b22911d47e941fb03b771c8d8f92dd99"
            or source_chain.get("final_reseal_parent")
            != "27c5e1e5925228757fc02378eb9d3b2a8addc60e"
            or source_chain.get("initial_control_blobs") != sealed_initial_blobs
            or identity_state != "sealed"
        ):
            errors.append("M35 immutable-authority identity delta is not exact")
        if require_active_runtime:
            target_root = str(expected["target_runtime_root"])
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            runtime = scheduler.get("runtime_paths")
            if (
                not isinstance(program, Mapping)
                or program.get("store_id") != expected["target_store_id"]
                or program.get("store_generation") != "29"
                or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("store_id") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M35 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m36_successor_declared(
            scheduler, seal, migration
        ):
            m36 = (
                materializer._expected_m36_operator_task_binding_correction_authority()
            )
            m36_source_chain = m36.get("source_chain", {})
            if isinstance(m36_source_chain, Mapping):
                historical_control_head = str(
                    m36_source_chain.get("base_control_commit") or ""
                ) or None
            if historical_control_head is None:
                errors.append("M36 prior M35 control source head is absent")
        errors.extend(
            _m35_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M35 immutable-authority identity authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m34_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    key = "json_emission_normalization_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m34_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    """Validate the M34 base and either its placeholders or final reseal."""

    try:
        chain = authority.get("source_chain", {})
        initial_blobs = chain.get("initial_control_blobs", {})
        zero = "0" * 40
        if (
            not isinstance(chain, Mapping)
            or not isinstance(initial_blobs, Mapping)
            or chain.get("base_control_commit")
            != "a5aa77fe58cd706ed8a1a2ae9d3f1e652f28b7f9"
            or chain.get("base_control_tree")
            != "8fe38992e84c291a0cab6d2afc6cda5bb4f08ab8"
            or set(initial_blobs) != set(authority.get("operator_control_paths", ()))
            or len(initial_blobs) != 9
            or _git(
                root,
                "rev-parse",
                "a5aa77fe58cd706ed8a1a2ae9d3f1e652f28b7f9^{tree}",
            )
            != "8fe38992e84c291a0cab6d2afc6cda5bb4f08ab8"
        ):
            raise RuntimeError("M34 accepted base/control path identity differs")
        identities = (
            chain.get("initial_control_commit"),
            chain.get("initial_control_tree"),
            chain.get("final_reseal_parent"),
            *initial_blobs.values(),
        )
        if all(value == zero for value in identities):
            return []
        if not hasattr(materializer, "_assert_m34_source_delta"):
            raise RuntimeError("M34 final source-delta verifier is unavailable")
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m34_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M34 exact JSON-emission repair/reseal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m34_json_emission_normalization_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    key = "json_emission_normalization_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m34_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M34 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m34_json_emission_normalization_authority()
        )
        reference = materializer._m34_authority_reference()
        contract = materializer._validated_m34_live_preflight_contract(expected)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append(
                "M34 JSON-emission normalization authority is only partially declared"
            )
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M34 JSON-emission authority reference differs")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M34 JSON-emission authority CID differs")
        binding = expected.get("runtime_binding", {})
        prior = expected.get("prior_authority", {})
        repair = expected.get("accepted_control_plane_repair", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        source_chain = expected.get("source_chain", {})
        initial_blobs = source_chain.get("initial_control_blobs", {})
        sealed_initial_commit = "e342b63f3f143bb85ed4744e5391c4f8e7c961cd"
        sealed_initial_tree = "30f0f07fa41947647461ccbd44d6dc4008848259"
        sealed_initial_blobs = {
            "config/agent_supervisor_semantic_addressed_world_model_scheduler.json": "71de45675fe8f1d134aa42af3f7fc782fbe9388c",
            "config/semantic_addressed_world_model_dependencies.seal.json": "68dc298dccd48352273bd3d7bf96368a56e4b561",
            "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md": "b3f998a018f2928df7b80b584e8ea88072f5046a",
            "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json": "73831143dd7d320dfe6a8a63c3331eedd387ad72",
            "scripts/materialize_semantic_addressed_world_model_program.py": "fa41e431afb1b8e26220c9715f8242c06096cc3a",
            "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": "1a9bc64217aa43fec874dd18754773931d7b8c9d",
            "scripts/validate_semantic_addressed_world_model_board.py": "91f7630233c696c180e7ab237aa5716274a884b2",
            "scripts/validate_semantic_addressed_world_model_dependencies.py": "0b38fd8d3a996139e0eb728e8437f5cbc820b871",
            "test/api/semantic_world/test_semantic_addressed_world_model_board.py": "d7c58109e86878017dcb629dc1104c1d5d6af428",
        }
        if (
            expected.get("schema")
            != "sawm/json-emission-normalization-successor-authorization@1"
            or expected.get("migration_revision") != "SAWM-R2-M34"
            or expected.get("migration_kind") != key
            or expected.get("target_generation") != 29
            or expected.get("target_event_watermark") != 285
            or expected.get("target_projection_cid")
            != "baguqeeragsizyo6v4izu7qfvjbj5l5bjkuycw2xyaf3vhlzx2nai7xyrvd4q"
            or binding.get("run_id") != "run-r2-m27"
            or binding.get("prior_event_watermark") != 284
            or binding.get("store_generation") != 29
            or binding.get("quack_port") != 24_070
            or prior.get("migration_revision") != "SAWM-R2-M33"
            or prior.get("event_prefix_sha256")
            != "22687fc6b6f5c082b1c30fcc4a1669d661bc8a67fe39e52e453b56d8eb3ee236"
            or prior.get("projection_cid")
            != "baguqeera5wkenkpg5zpndh5whgwrqkvpq2e7qz6xv6rflrajynrpqf7dtmla"
            or prior.get("m33_receipt_sha256")
            != "cb5040ce01d739240d0e29ce89f0874f9dd56302a4d0836acf4c076f56f4a682"
            or prior.get("m33_receipt_cid")
            != "sha256:ae8270d95f6b5d199a6dc20ba63b6fe5cb7f0fe4f8a30044b03af796a72dcf64"
            or contract.get("migration_revision") != "SAWM-R2-M34"
            or contract.get("expected_task_heads")
            != materializer._m30_expected_task_heads()
            or repair.get("defect") != "non_recursive_operator_json_serialization"
            or repair.get("resolution")
            != "recursively_normalize_closed_operator_output_before_json_encoding"
            or repair.get("authority_weakened") is not False
            or changes.get("event_suffix_length") != 1
            or changes.get("operator_json_serialization_changes") != 1
            or any(
                changes.get(field) != 0
                for field in (
                    "task_revision_changes",
                    "task_status_changes",
                    "plan_revision_changes",
                    "goal_revision_changes",
                    "owner_generation_changes",
                    "coordination_semantic_changes",
                    "accepted_completion_changes",
                )
            )
            or preservation.get("same_live_owner") is not True
            or preservation.get("generation_restart") is not False
            or preservation.get("m33_receipt_preserved") is not True
            or source_chain.get("base_control_commit")
            != "a5aa77fe58cd706ed8a1a2ae9d3f1e652f28b7f9"
            or source_chain.get("base_control_tree")
            != "8fe38992e84c291a0cab6d2afc6cda5bb4f08ab8"
            or source_chain.get("initial_control_commit")
            != materializer._M34_INITIAL_CONTROL_COMMIT
            or source_chain.get("initial_control_commit") != sealed_initial_commit
            or source_chain.get("initial_control_tree")
            != materializer._M34_INITIAL_CONTROL_TREE
            or source_chain.get("initial_control_tree") != sealed_initial_tree
            or source_chain.get("final_reseal_parent")
            != materializer._M34_INITIAL_CONTROL_COMMIT
            or source_chain.get("final_reseal_parent") != sealed_initial_commit
            or not isinstance(initial_blobs, Mapping)
            or len(initial_blobs) != 9
            or dict(initial_blobs) != dict(materializer._M34_INITIAL_CONTROL_BLOBS)
            or dict(initial_blobs) != sealed_initial_blobs
        ):
            errors.append("M34 JSON-emission normalization delta is not exact")
        if require_active_runtime:
            target_root = str(expected["target_runtime_root"])
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            runtime = scheduler.get("runtime_paths")
            if (
                not isinstance(program, Mapping)
                or program.get("store_id") != expected["target_store_id"]
                or program.get("store_generation") != "29"
                or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("store_id") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M34 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m35_successor_declared(
            scheduler, seal, migration
        ):
            m35 = (
                materializer._expected_m35_immutable_authority_identity_normalization_authority()
            )
            m35_source_chain = m35.get("source_chain", {})
            if isinstance(m35_source_chain, Mapping):
                historical_control_head = str(
                    m35_source_chain.get("base_control_commit") or ""
                ) or None
            if historical_control_head is None:
                errors.append("M35 prior M34 control source head is absent")
        errors.extend(
            _m34_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M34 JSON-emission normalization authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m33_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    key = "live_preflight_contract_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m33_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    try:
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m33_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M33 exact contract-repair/reseal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m33_live_preflight_contract_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    key = "live_preflight_contract_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m33_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M33 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m33_live_preflight_contract_authority()
        reference = materializer._m33_authority_reference()
        contract = materializer._validated_m33_live_preflight_contract(expected)
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M33 live-preflight contract authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M33 live-preflight contract authority reference differs")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M33 live-preflight contract authority CID differs")
        binding = expected.get("runtime_binding", {})
        prior = expected.get("prior_authority", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        source_chain = expected.get("source_chain", {})
        if (
            expected.get("migration_revision") != "SAWM-R2-M33"
            or expected.get("target_generation") != 29
            or expected.get("target_event_watermark") != 284
            or expected.get("target_projection_cid")
            != "baguqeera5wkenkpg5zpndh5whgwrqkvpq2e7qz6xv6rflrajynrpqf7dtmla"
            or binding.get("prior_event_watermark") != 283
            or binding.get("store_generation") != 29
            or prior.get("m32_receipt_cid")
            != "sha256:e4981d0d623454ac9c55820b972fe818c78c7d5d7cb2208b9e05eba7461ad744"
            or contract.get("database_uuid")
            != "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4"
            or contract.get("successor_class") != "evidence_only_post_m27"
            or changes.get("event_suffix_length") != 1
            or any(
                changes.get(field) != 0
                for field in (
                    "task_revision_changes",
                    "task_status_changes",
                    "plan_revision_changes",
                    "goal_revision_changes",
                    "owner_generation_changes",
                    "coordination_semantic_changes",
                    "accepted_completion_changes",
                )
            )
            or preservation.get("same_live_owner") is not True
            or preservation.get("generation_restart") is not False
            or preservation.get("m32_receipt_preserved") is not True
            or source_chain.get("initial_control_commit")
            != "b0526d4085b231f1eca0cf638743e8d455debc08"
            or source_chain.get("initial_control_tree")
            != "61054b8cdf793a6c9ed4f326215e60b9799a4a6d"
        ):
            errors.append("M33 live-preflight contract repair delta is not exact")
        if require_active_runtime:
            target_root = str(expected["target_runtime_root"])
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            runtime = scheduler.get("runtime_paths")
            if (
                not isinstance(program, Mapping)
                or program.get("store_id") != expected["target_store_id"]
                or program.get("store_generation") != "29"
                or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("store_id") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M33 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m34_successor_declared(
            scheduler, seal, migration
        ):
            m34 = (
                materializer._expected_m34_json_emission_normalization_authority()
            )
            m34_source_chain = m34.get("source_chain", {})
            if isinstance(m34_source_chain, Mapping):
                historical_control_head = str(
                    m34_source_chain.get("base_control_commit") or ""
                ) or None
            if historical_control_head is None:
                errors.append("M34 prior M33 control source head is absent")
        errors.extend(
            _m33_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M33 live-preflight contract authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m32_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    key = "live_preflight_plan_anchor_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m32_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    try:
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m32_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M32 exact control-repair/reseal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m32_live_preflight_plan_anchor_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    key = "live_preflight_plan_anchor_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m32_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M32 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m32_live_preflight_plan_anchor_authority()
        reference = materializer._m32_authority_reference()
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M32 live-preflight authority is only partially declared")
        if scheduler.get(key) != reference or migration.get(key) != reference:
            errors.append("M32 live-preflight authority reference differs")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M32 live-preflight authority CID differs")
        binding = expected.get("runtime_binding", {})
        prior = expected.get("prior_authority", {})
        anchor = expected.get("preserved_plan_anchor", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        source_chain = expected.get("source_chain", {})
        expected_task_heads = expected.get("expected_task_heads", {})
        if (
            expected.get("migration_revision") != "SAWM-R2-M32"
            or expected.get("target_generation") != 29
            or expected.get("target_event_watermark") != 283
            or expected.get("target_projection_cid")
            != "baguqeerab3m6k3ea4ulaouojsdazccvepipcfryyblps7676ymqrtbyc5tiq"
            or binding.get("prior_event_watermark") != 282
            or binding.get("store_generation") != 29
            or prior.get("m31_receipt_cid")
            != "sha256:e1baa264e6c62eaf723208fe41e0501d087876e7999204626942810978a3edde"
            or anchor.get("current_source_binding_cid")
            != "sha256:83e28e01de41699d5b2312ead03e7f33d9989d809d97924ea0a230af2c038856"
            or anchor.get("source_migration_revision") != "SAWM-R2-M27"
            or changes.get("event_suffix_length") != 1
            or any(
                changes.get(field) != 0
                for field in (
                    "task_revision_changes",
                    "task_status_changes",
                    "plan_revision_changes",
                    "goal_revision_changes",
                    "owner_generation_changes",
                    "coordination_semantic_changes",
                    "accepted_completion_changes",
                )
            )
            or preservation.get("same_live_owner") is not True
            or preservation.get("generation_restart") is not False
            or preservation.get("m31_receipt_preserved") is not True
            or source_chain.get("initial_control_commit")
            != "547485ffacd636c6046b00ed23dc0f4c53de4315"
            or source_chain.get("initial_control_tree")
            != "10e153a78f5709e8b841567ab4475f60eff05db3"
            or expected_task_heads != materializer._m30_expected_task_heads()
        ):
            errors.append("M32 live-preflight plan-anchor repair delta is not exact")
        if require_active_runtime:
            target_root = str(expected["target_runtime_root"])
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            runtime = scheduler.get("runtime_paths")
            if (
                not isinstance(program, Mapping)
                or program.get("store_id") != expected["target_store_id"]
                or program.get("store_generation") != "29"
                or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("store_id") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M32 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m33_successor_declared(
            scheduler, seal, migration
        ):
            m33 = materializer._expected_m33_live_preflight_contract_authority()
            source_chain = m33.get("source_chain", {})
            if isinstance(source_chain, Mapping):
                historical_control_head = str(
                    source_chain.get("base_control_commit") or ""
                ) or None
            if historical_control_head is None:
                errors.append("M33 prior M32 control source head is absent")
        errors.extend(
            _m32_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M32 live-preflight authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m31_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M31 on any declaration surface, including partial state."""

    key = "detached_coordinator_pid_recovery_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m31_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    try:
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m31_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M31 exact runtime-repair/control/reseal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m31_detached_coordinator_pid_recovery_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M31's exact stopped-owner/PID-recovery successor contract."""

    key = "detached_coordinator_pid_recovery_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m31_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M31 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m31_detached_coordinator_pid_recovery_authority()
        )
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M31 detached-coordinator authority is only partially declared")
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("M31 detached-coordinator authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M31 detached-coordinator authority CID differs")
        binding = expected.get("runtime_binding", {})
        stopped = expected.get("stopped_owner", {})
        failure = expected.get("failed_launch_evidence", {})
        stale = failure.get("stale_pid_projection", {})
        quarantine = expected.get("pid_quarantine_authorization", {})
        changes = expected.get("exact_changes", {})
        preservation = expected.get("preservation", {})
        source_chain = expected.get("source_chain", {})
        if (
            expected.get("schema")
            != (
                "sawm/detached-coordinator-pid-recovery-successor-"
                "materialization-authorization@1"
            )
            or expected.get("migration_revision") != "SAWM-R2-M31"
            or expected.get("target_generation") != 29
            or expected.get("target_event_watermark") != 282
            or expected.get("target_projection_cid")
            != "baguqeerakjradc5sa5dmflygtfh2birrd5onygnt6q2pkvoomsxrspi22jaa"
            or binding.get("prior_event_watermark") != 281
            or binding.get("store_generation") != 29
            or stopped.get("generation") != 28
            or stopped.get("status") != "stopped"
            or stopped.get("target_generation") != 29
            or failure.get("coordinator_process_spawned") is not False
            or failure.get("provider_token_handoff_retired") is not True
            or stale.get("pid") != 3_554_888
            or stale.get("content_sha256")
            != "592c926b10dfc688ab07af087bb761228c61a1f8f2829c7f577464322eacca46"
            or stale.get("mode") != 0o664
            or stale.get("link_count") != 1
            or stale.get("inode") != 97_255_434
            or stale.get("pid_observed_dead") is not True
            or quarantine.get("dead_pid_must_be_reverified_under_lock") is not True
            or quarantine.get("stable_inode_must_be_reverified") is not True
            or quarantine.get("new_projection_requires_exclusive_0600_create") is not True
            or changes.get("event_suffix_length") != 1
            or changes.get("task_revision_changes") != 0
            or changes.get("task_status_changes") != 0
            or changes.get("coordination_semantic_changes") != 0
            or preservation.get("event_281_and_m30_receipt_preserved") is not True
            or preservation.get("failed_detached_launch_preserved") is not True
            or source_chain.get("initial_control_commit")
            != "07aed87e3ebc4ef5667541435fd04f2d62a39b25"
            or source_chain.get("initial_control_tree")
            != "c9fe9a657d2d0e1e05172688dbc44dea6f699265"
            or source_chain.get("final_reseal_parent")
            != "07aed87e3ebc4ef5667541435fd04f2d62a39b25"
        ):
            errors.append("M31 detached-coordinator recovery delta is not exact")
        if require_active_runtime:
            target_root = str(expected["target_runtime_root"])
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            runtime = scheduler.get("runtime_paths")
            if (
                not isinstance(program, Mapping)
                or program.get("store_id") != expected["target_store_id"]
                or program.get("store_generation") != "29"
                or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("store_id") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M31 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m32_successor_declared(
            scheduler, seal, migration
        ):
            m32 = materializer._expected_m32_live_preflight_plan_anchor_authority()
            source_chain = m32.get("source_chain", {})
            if isinstance(source_chain, Mapping):
                historical_control_head = str(
                    source_chain.get("base_control_commit") or ""
                ) or None
            if historical_control_head is None:
                errors.append("M32 prior M31 control source head is absent")
        errors.extend(
            _m31_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M31 detached-coordinator authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m30_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M30 on any declaration surface, including partial state."""

    key = "stopped_owner_restart_source_seal_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m30_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    try:
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m30_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M30 exact recovery/initial-seal/cursor-repair/reseal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m30_stopped_owner_restart_source_seal_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M30's one-generation restart and one-event live source seal."""

    key = "stopped_owner_restart_source_seal_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m30_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M30 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m30_stopped_owner_restart_source_seal_authority()
        )
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append("M30 stopped-owner restart authority is only partially declared")
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("M30 stopped-owner restart authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M30 stopped-owner restart authority CID differs")
        binding = expected.get("runtime_binding", {})
        stopped = expected.get("stopped_owner", {})
        changes = expected.get("exact_changes", {})
        cursor_repair = expected.get("bounded_materializer_repair", {})
        prior_control = expected.get("prior_control_authorization", {})
        if (
            expected.get("schema")
            != (
                "sawm/stopped-owner-restart-source-seal-successor-"
                "materialization-authorization@2"
            )
            or expected.get("authorization_revision") != 2
            or expected.get("prior_authorization_cid")
            != "sha256:ef37e79ce07cbb18d16259feeb85c3176661ebc64ec3cb108c3d7b5624e294fe"
            or expected.get("migration_revision") != "SAWM-R2-M30"
            or expected.get("target_generation") != 28
            or expected.get("target_event_watermark") != 281
            or binding.get("prior_event_watermark") != 280
            or binding.get("store_generation") != 28
            or stopped.get("generation") != 27
            or stopped.get("status") != "stopped"
            or stopped.get("target_generation") != 28
            or changes.get("event_suffix_length") != 1
            or changes.get("evidence_node_changes") != 1
            or changes.get("state_server_row_changes") != 1
            or changes.get("store_generation_row_changes") != 1
            or changes.get("credential_row_changes") != 1
            or changes.get("task_revision_changes") != 0
            or changes.get("task_status_changes") != 0
            or changes.get("coordination_semantic_changes") != 0
            or cursor_repair.get("schema")
            != "sawm/bounded-live-row-normalization-repair@1"
            or cursor_repair.get("repair_commit")
            != "95505a7eec81a5eedd859e7efb97539d759c918f"
            or cursor_repair.get("database_mutations") != 0
            or cursor_repair.get("event_append_changes") != 0
            or cursor_repair.get("lifecycle_checks_weakened") is not False
            or prior_control.get("control_commit")
            != "8233b47ba4c05470235ec832e95a70fdce13316d"
            or prior_control.get("superseded_before_event_281") is not True
            or prior_control.get("event_281_appended") is not False
            or prior_control.get("receipt_published") is not False
        ):
            errors.append("M30 stopped-owner restart delta is not exact")
        if require_active_runtime:
            target_root = str(expected["target_runtime_root"])
            program = scheduler.get("database_program")
            owner = scheduler.get("quack_owner")
            runtime = scheduler.get("runtime_paths")
            if (
                not isinstance(program, Mapping)
                or program.get("store_id") != expected["target_store_id"]
                or program.get("store_generation") != "28"
                or program.get("quack_endpoint") != "quack:127.0.0.1:24070"
                or not isinstance(owner, Mapping)
                or owner.get("database_path") != expected["target_store_id"]
                or owner.get("store_id") != expected["target_store_id"]
                or owner.get("port") != 24_070
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M30 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m31_successor_declared(
            scheduler, seal, migration
        ):
            m31 = (
                materializer
                ._expected_m31_detached_coordinator_pid_recovery_authority()
            )
            source_chain = m31.get("source_chain", {})
            if isinstance(source_chain, Mapping):
                historical_control_head = str(
                    source_chain.get("m30_control_commit") or ""
                ) or None
            if historical_control_head is None:
                errors.append("M31 prior M30 control source head is absent")
        errors.extend(
            _m30_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M30 stopped-owner restart authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m29_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M29 on any declaration surface, including partial state."""

    key = "committed_evidence_verification_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m29_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    """Bind M28's control seal, the verifier repair, and M29's seal."""

    try:
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m29_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M29 exact M28-control, verifier-repair, and nine-control seal "
            f"chain differs: {type(exc).__name__}: {exc}"
        ]


def _m29_committed_evidence_verification_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M29's append-only verification of M28's committed evidence."""

    key = "committed_evidence_verification_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m29_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M29 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer
            ._expected_m29_committed_evidence_verification_authority()
        )
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append(
                "M29 committed-evidence-verification authority is only "
                "partially declared"
            )
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append(
                "M29 committed-evidence-verification authority differs "
                "across controls"
            )
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append(
                "M29 committed-evidence-verification authority CID is not exact"
            )

        if expected.get("schema") != (
            "sawm/committed-evidence-verification-successor-materialization-"
            "authorization@1"
        ):
            errors.append("M29 authority schema differs")
        if expected.get("migration_revision") != "SAWM-R2-M29":
            errors.append("M29 migration revision differs")
        if (
            expected.get("migration_kind") != key
            or expected.get("supersession_mode")
            != "append_only_committed_evidence_verification_source_seal"
        ):
            errors.append("M29 successor kind/mode differs")
        if expected.get("authorized") is not True:
            errors.append("M29 operator authorization is absent")
        if expected.get("authority") != "operator_control_plane":
            errors.append("M29 operator authority class differs")

        runtime = expected.get("runtime_binding")
        target_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        if not isinstance(runtime, Mapping) or (
            runtime.get("run_id"),
            runtime.get("runtime_root"),
            runtime.get("store_id"),
            runtime.get("coordination_store_id"),
            runtime.get("worktree_root"),
            runtime.get("store_generation"),
            runtime.get("quack_port"),
            runtime.get("quack_endpoint"),
            runtime.get("database_uuid"),
            runtime.get("server_id"),
            runtime.get("process_birth_id"),
            runtime.get("plan_revision"),
            runtime.get("prior_event_watermark"),
            runtime.get("target_event_watermark"),
        ) != (
            "run-r2-m27",
            target_root,
            f"{target_root}/control.duckdb",
            f"{target_root}/control.coordination.duckdb",
            f"{target_root}/worktrees",
            27,
            24_070,
            "quack:127.0.0.1:24070",
            "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
            "server:ff5834df-4af2-4cb0-a4a4-7dbc4f258457",
            "birth:7157839e6e5bc6ce351f41c7d9cd6c94",
            28,
            273,
            274,
        ):
            errors.append("M29 runtime/event binding differs")

        prior = expected.get("prior_authority")
        if not isinstance(prior, Mapping) or (
            prior.get("migration_revision"),
            prior.get("authorization_cid"),
            prior.get("receipt_absent"),
            prior.get("failure_stage"),
            prior.get("event_watermark"),
            prior.get("event_prefix_sha256"),
            prior.get("projection_cid"),
            prior.get("semantic_authority_digest"),
            prior.get("event_id"),
            prior.get("event_body_sha256"),
            prior.get("evidence_id"),
            prior.get("evidence_digest"),
            prior.get("source_head"),
            prior.get("source_tree"),
        ) != (
            "SAWM-R2-M28",
            "sha256:f766e6f16a88ad50b5810312bacf8fb804474fb5a4e69de8973e728e98c93079",
            True,
            "post_append_verification_before_receipt_publication",
            273,
            "519bc8635dfebaf3f6b7fa843440aee9025f077e5b83a27d48f37816bb17e4c5",
            "baguqeerazyzybkjnlihpazozv7xjiign23bfius47mpb3iol6fciuonvzd7a",
            "sha256:9de5fc91f217a511ffdb108cbfca5715804354db9d1493a4bd11d5e49a345092",
            "baguqeeragd2blhdw2gdjwwqovgwuoxlaulnto35lxtlycappubgqz4fm3lia",
            "sha256:3cb367e5179dd16975e3504c5c51156ddc8e7995f7c43d525f8163fb099d991e",
            "baguqeeraumvois7bdkb7dk27zfbqecpudpc5htrxskz4x2jgau56ivalqvla",
            "sha256:09cf02e35e461801b012c91afbd4f52b1304dad1ab56dd254471cdf348b92ee1",
            "d7e2a4ba9bc7eef32ffad131ffd092ff11f934c4",
            "56b012539e6ca76af819f64dfdaa1f3d7e3df66c",
        ):
            errors.append("M29 failed-M28 predecessor authority differs")
        elif (
            prior.get("receipt_path")
            != f"{target_root}/m28-source-successor-receipt.json"
            or prior.get("program_definition_cid")
            != "sha256:f581af1f2234c127231b47bb1bb8d42910b984dcd1bece9a6303949ac1ba0b72"
            or prior.get("source_binding_cid")
            != "sha256:543c314eba87de4cff2af07296e87985dfc34fbda8b6436dad5cebc50652590e"
            or prior.get("validation_digest")
            != "sha256:2599a864dcec5441a709ca2425af0954f937a39928646c73ad3941e0d7b84a8e"
            or prior.get("plan_source_binding_cid")
            != "sha256:83e28e01de41699d5b2312ead03e7f33d9989d809d97924ea0a230af2c038856"
            or prior.get("plan_migration_digest")
            != "sha256:43eb5eb9b3f05c6ffe00c918901524e61c4a94a8d7334a3a3261ef95870cc73b"
        ):
            errors.append("M29 exact failed-M28 evidence binding differs")

        target = expected.get("target_authority")
        if not isinstance(target, Mapping) or target != {
            "event_watermark": 274,
            "projection_cid": (
                "baguqeera4z7aafxgr5xb7tfnb4mdhih4d2ynzkugodxrmrwpisi427hxnz7a"
            ),
            "plan_revision": 28,
            "evidence_kind": (
                "operator_control_plane_committed_evidence_verification_successor"
            ),
            "operator_task_cid": (
                "sha256:8b8f43dd51ea4d8467af0e5cae4100478f16666d36c6f4fad49c23fd8e43a3d6"
            ),
        }:
            errors.append("M29 target evidence authority differs")

        exact_changes = expected.get("exact_changes")
        if exact_changes != {
            "event_suffix_length": 1,
            "evidence_node_changes": 1,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "plan_revision_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "coordination_semantic_changes": 0,
            "sidecar_changes": 0,
            "store_generation_row_changes": 0,
            "state_server_row_changes": 0,
            "credential_row_changes": 0,
            "implementation_provider_invocations": 0,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
        } or expected.get("ordinary_source_changes") != 0:
            errors.append("M29 evidence-only change inventory differs")

        preservation = expected.get("preservation")
        if preservation != {
            "failed_m28_post_append_attempt_preserved": True,
            "m28_receipt_created_or_rewritten": False,
            "same_live_owner": True,
            "same_runtime_root": True,
            "same_store_generation": True,
            "control_base_copied": False,
            "coordination_base_copied": False,
            "sidecars_preserved": True,
            "lane_sidecars_preserved_in_place": True,
            "execution_sidecar_copied": False,
            "read_replica_sidecar_copied": False,
            "worktrees_copied": False,
            "active_sawm_006_attempt_preserved": True,
            "worker_self_approval": False,
        }:
            errors.append("M29 failed-attempt/sidecar preservation differs")

        source_chain = expected.get("source_chain")
        if source_chain != {
            "m28_control_commit": "d7e2a4ba9bc7eef32ffad131ffd092ff11f934c4",
            "verifier_repair_commit": "bb79ffc69b199a735e672072a8cf534417918393",
            "verifier_repair_parent": "d7e2a4ba9bc7eef32ffad131ffd092ff11f934c4",
            "verifier_repair_tree": "c1e0ebea9b82b635d70eaf0b61044118dc05c052",
            "verifier_repair_blobs": {
                "scripts/materialize_semantic_addressed_world_model_program.py": (
                    "d88e511bbee55a577d046760de4311c3aabdb188"
                ),
                "scripts/ops/agent_supervisor/semantic_addressed_world_model.py": (
                    "d7356bdf08f75e97494f2fc721cf8fb34894bd35"
                ),
                "test/api/semantic_world/"
                "test_semantic_addressed_world_model_board.py": (
                    "a1f2167768f6289ae6feff336ccd30a50821d13f"
                ),
            },
            "final_control_commit_count": 1,
        }:
            errors.append("M29 verifier-repair source chain differs")

        expected_heads = expected.get("expected_task_heads")
        if (
            not isinstance(expected_heads, Mapping)
            or len(expected_heads) != 45
            or expected_heads.get("SAWM-006")
            != {"status": "in_progress", "revision": 5}
            or expected_heads.get("SAWM-008")
            != {"status": "retrying", "revision": 8}
            or expected_heads.get("SAWM-012")
            != {"status": "completed", "revision": 9}
            or expected_heads.get("SAWM-015")
            != {"status": "retrying", "revision": 5}
        ):
            errors.append("M29 unchanged task-head inventory differs")

        restart = expected.get("generation_26_27_restart_rows")
        if (
            not isinstance(restart, Mapping)
            or restart.get("global_counts")
            != {"state_servers": 27, "store_generations": 27, "credentials": 27}
            or [row[:8] for row in restart.get("state_servers", [])]
            != [
                [
                    "server:f0e56096-6fe5-4fdb-ba58-d1cb16cb7f2e",
                    f"{target_root}/control.duckdb",
                    "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
                    "birth:e5067605f91c1d1dee62d2e7a0626d9f",
                    "quack:127.0.0.1:24070",
                    "sha256:b77954ae50ecc06e10c6e20fc6fd421d73b5c31cf72bb60ae3f29b1f8a85f20b",
                    1,
                    26,
                ],
                [
                    "server:ff5834df-4af2-4cb0-a4a4-7dbc4f258457",
                    f"{target_root}/control.duckdb",
                    "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
                    "birth:7157839e6e5bc6ce351f41c7d9cd6c94",
                    "quack:127.0.0.1:24070",
                    "sha256:b77954ae50ecc06e10c6e20fc6fd421d73b5c31cf72bb60ae3f29b1f8a85f20b",
                    1,
                    27,
                ],
            ]
            or [row[:4] for row in restart.get("store_generations", [])]
            != [[26, 1, 26, 0], [27, 1, 27, 0]]
            or [row[1:4] for row in restart.get("credentials", [])]
            != [
                ["env://SAWM_QUACK_TOKEN", 26, "quack-auth"],
                ["env://SAWM_QUACK_TOKEN", 27, "quack-auth"],
            ]
        ):
            errors.append("M29 unchanged generation-26/27 restart rows differ")

        repair = expected.get("historical_transition_repair")
        if (
            not isinstance(repair, Mapping)
            or repair.get("schema")
            != "sawm/historical-transition-admission-mismatch@1"
            or repair.get("task_alias") != "SAWM-012"
            or repair.get("actual_configured_board_admission_cid") != ""
            or repair.get("expected_configured_board_admission_cid")
            != "baguqeeraelsagoqf62zk3etgymookxuzxrn763rwbo7i6iqpqte6ehj7giuq"
            or repair.get("accepted_completion_changed") is not False
            or repair.get("historical_completion_rewritten") is not False
            or repair.get("task_completion_authority") is not False
            or repair.get("worker_self_approval") is not False
        ):
            errors.append("M29 historical SAWM-012 mismatch receipt differs")

        required_control_paths = {
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
        }
        control_paths = expected.get("operator_control_paths")
        if (
            not isinstance(control_paths, list)
            or set(control_paths) != required_control_paths
            or control_paths != sorted(control_paths)
        ):
            errors.append("M29 protected nine-control path inventory differs")
        if (
            expected.get("current_datasets_gitlink")
            != "b9f5b86199c03e427fd51fcea302479880421ff8"
            or expected.get("current_datasets_tree")
            != "52c0c7be05a51956ba5aa2b6f85d38e03588f3b1"
            or expected.get("current_kit_gitlink")
            != "fc9248073e9f67ac59ca607c7736746907b08037"
            or expected.get("current_kit_tree")
            != "b26e05db1b199e7e491b45b686a0845fabbabadb"
        ):
            errors.append("M29 nested source identities differ")

        if require_active_runtime:
            target_store = f"{target_root}/control.duckdb"
            program = scheduler.get("database_program", {})
            owner = scheduler.get("quack_owner", {})
            runtime_paths = scheduler.get("runtime_paths", {})
            observed = (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            wanted = (
                target_store,
                "27",
                "quack:127.0.0.1:24070",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_070,
            )
            expected_runtime_paths = {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
            if observed != wanted or runtime_paths != expected_runtime_paths:
                errors.append("scheduler M29 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m30_successor_declared(
            scheduler, seal, migration
        ):
            m30 = materializer._expected_m30_stopped_owner_restart_source_seal_authority()
            source_chain = m30.get("source_chain", {})
            if isinstance(source_chain, Mapping):
                historical_control_head = str(
                    source_chain.get("m29_control_commit") or ""
                ) or None
            if historical_control_head is None:
                errors.append("M30 prior M29 control source head is absent")
        errors.extend(
            _m29_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M29 committed-evidence-verification authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m28_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M28 on any declaration surface, including partial state."""

    key = "live_claim_admission_recovery_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m28_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    """Bind the bounded supervisor repair and protected control seal."""

    try:
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m28_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M28 exact supervisor-repair and nine-control seal chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m28_live_claim_admission_recovery_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M28's evidence-only repair of live-claim/admission plumbing."""

    key = "live_claim_admission_recovery_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m28_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M28 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m28_live_claim_admission_recovery_authority()
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append(
                "M28 live-claim/admission-recovery authority is only partially "
                "declared"
            )
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append(
                "M28 live-claim/admission-recovery authority differs across controls"
            )
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append(
                "M28 live-claim/admission-recovery authority CID is not exact"
            )

        if expected.get("schema") != (
            "sawm/live-claim-admission-recovery-successor-materialization-"
            "authorization@1"
        ):
            errors.append("M28 authority schema differs")
        if expected.get("migration_revision") != "SAWM-R2-M28":
            errors.append("M28 migration revision differs")
        if expected.get("migration_kind") != key or expected.get(
            "supersession_mode"
        ) != "append_only_live_owner_evidence_source_seal":
            errors.append("M28 successor kind/mode differs")
        if expected.get("authorized") is not True:
            errors.append("M28 operator authorization is absent")
        if expected.get("authority") != "operator_control_plane":
            errors.append("M28 operator authority class differs")

        runtime = expected.get("runtime_binding")
        target_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
        )
        if not isinstance(runtime, Mapping) or (
            runtime.get("run_id"),
            runtime.get("runtime_root"),
            runtime.get("store_id"),
            runtime.get("coordination_store_id"),
            runtime.get("worktree_root"),
            runtime.get("store_generation"),
            runtime.get("quack_port"),
            runtime.get("quack_endpoint"),
            runtime.get("database_uuid"),
            runtime.get("server_id"),
            runtime.get("process_birth_id"),
            runtime.get("plan_revision"),
            runtime.get("prior_event_watermark"),
            runtime.get("target_event_watermark"),
        ) != (
            "run-r2-m27",
            target_root,
            f"{target_root}/control.duckdb",
            f"{target_root}/control.coordination.duckdb",
            f"{target_root}/worktrees",
            27,
            24_070,
            "quack:127.0.0.1:24070",
            "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
            "server:ff5834df-4af2-4cb0-a4a4-7dbc4f258457",
            "birth:7157839e6e5bc6ce351f41c7d9cd6c94",
            28,
            272,
            273,
        ):
            errors.append("M28 evidence-only runtime/event binding differs")

        prior = expected.get("prior_authority")
        target = expected.get("target_authority")
        if (
            not isinstance(prior, Mapping)
            or prior.get("migration_revision") != "SAWM-R2-M27"
            or prior.get("event_watermark") != 272
            or prior.get("coordination_event_count") != 3_019
            or prior.get("coordination_projection_digest")
            != "sha256:7abbd22e48ed99b31fb02190c6406b631f3341de22c9904af3d2b5b10d9509c0"
            or not isinstance(target, Mapping)
            or target.get("event_watermark") != 273
            or target.get("plan_revision") != 28
            or target.get("projection_cid")
            != "baguqeerazyzybkjnlihpazozv7xjiign23bfius47mpb3iol6fciuonvzd7a"
        ):
            errors.append("M28 prior/target authority binding differs")

        restart = expected.get("credential_handoff_recovery")
        if not isinstance(restart, Mapping) or (
            restart.get("schema"),
            restart.get("prior_server_id"),
            restart.get("prior_generation"),
            restart.get("prior_resulting_status"),
            restart.get("target_server_id"),
            restart.get("target_generation"),
            restart.get("target_status"),
            restart.get("store_generation_row_changes"),
            restart.get("state_server_row_changes"),
            restart.get("credential_row_changes"),
            restart.get("domain_event_changes"),
            restart.get("worker_self_approval"),
        ) != (
            "sawm/generation-bearing-owner-restart@1",
            "server:f0e56096-6fe5-4fdb-ba58-d1cb16cb7f2e",
            26,
            "stopped",
            "server:ff5834df-4af2-4cb0-a4a4-7dbc4f258457",
            27,
            "ready",
            1,
            2,
            1,
            0,
            False,
        ):
            errors.append("M28 generation-bearing owner restart differs")

        exact_changes = expected.get("exact_changes")
        zero_change_fields = (
            "task_revision_changes",
            "task_status_changes",
            "plan_revision_changes",
            "accepted_definition_changes",
            "accepted_completion_changes",
            "coordination_semantic_changes",
            "sidecar_changes",
            "implementation_provider_invocations",
            "effect_claim_changes",
            "implementation_commit_changes",
            "merge_attempt_changes",
        )
        if (
            not isinstance(exact_changes, Mapping)
            or any(exact_changes.get(field) != 0 for field in zero_change_fields)
            or exact_changes.get("evidence_node_changes") != 1
            or exact_changes.get("event_suffix_length") != 1
            or exact_changes.get("store_generation_row_changes") != 1
            or exact_changes.get("state_server_row_changes") != 2
            or exact_changes.get("credential_row_changes") != 1
            or expected.get("ordinary_source_changes") != 0
        ):
            errors.append("M28 evidence-only change inventory differs")

        preservation = expected.get("preservation")
        if (
            not isinstance(preservation, Mapping)
            or preservation.get("sidecars_preserved") is not True
            or preservation.get("same_live_owner") is not False
            or preservation.get("same_store_generation") is not False
            or preservation.get("generation_bearing_owner_restart") is not True
            or preservation.get("worker_self_approval") is not False
        ):
            errors.append("M28 sidecar/self-approval preservation differs")

        repair = expected.get("historical_transition_repair")
        if not isinstance(repair, Mapping):
            errors.append("M28 historical SAWM-012 transition repair is absent")
        else:
            required_repair = {
                "schema": "sawm/historical-transition-admission-mismatch@1",
                "task_alias": "SAWM-012",
                "task_cid": (
                    "sha256:7b3443e7a80b5b81aed08f89733253d7145d3e2553e2c466"
                    "77566075fe08ac99"
                ),
                "task_revision": 9,
                "accepted_transition_cid": (
                    "sha256:97b5896f0e2f32fbdddc0a904a57ff0d5ddc11abbe1ecf3a9"
                    "aad73178d77cd1f"
                ),
                "actual_configured_board_admission_cid": "",
                "expected_configured_board_admission_cid": (
                    "baguqeeraelsagoqf62zk3etgymookxuzxrn763rwbo7i6iqpqte6ehj7giuq"
                ),
                "implementation_commit": (
                    "e9d03bff8b527e2b6c1a216cffb60a52a037c581"
                ),
                "merge_commit": "8d2acae71f4d8caaceaf2677d4f0ca87d15e606e",
                "validation_event_id": (
                    "baguqeerabqgo366z7qgn2e2c6uzunbgdkllptgk6fg2pbdya57apck6tr4fq"
                ),
                "completion_event_id": (
                    "baguqeeraemvqjbjurp7qa6szkuyhvwuy3jzwjaqhsfvhq3iwyf67aruatj3a"
                ),
                "portal_event_log_sha256": (
                    "sha256:3721dc40482fa85c4abc3a5496c64f24d7c4c37a73762a8f8774"
                    "fe46fb77d777"
                ),
                "accepted_completion_changed": False,
                "historical_completion_rewritten": False,
                "task_completion_authority": False,
                "worker_self_approval": False,
            }
            if any(repair.get(field) != value for field, value in required_repair.items()):
                errors.append(
                    "M28 historical SAWM-012 admission mismatch receipt differs"
                )

        control_paths = expected.get("operator_control_paths")
        required_control_paths = {
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
        }
        if (
            not isinstance(control_paths, list)
            or set(control_paths) != required_control_paths
            or control_paths != sorted(control_paths)
        ):
            errors.append("M28 protected nine-control path inventory differs")

        if require_active_runtime:
            target_store = f"{target_root}/control.duckdb"
            program = scheduler.get("database_program", {})
            owner = scheduler.get("quack_owner", {})
            runtime_paths = scheduler.get("runtime_paths", {})
            observed = (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            wanted = (
                target_store,
                "27",
                "quack:127.0.0.1:24070",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_070,
            )
            expected_runtime_paths = {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
            if observed != wanted or runtime_paths != expected_runtime_paths:
                errors.append("scheduler M28 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m29_successor_declared(
            scheduler, seal, migration
        ):
            m29 = (
                materializer
                ._expected_m29_committed_evidence_verification_authority()
            )
            source_chain = m29.get("source_chain", {})
            if isinstance(source_chain, Mapping):
                historical_control_head = str(
                    source_chain.get("m28_control_commit") or ""
                ) or None
            if historical_control_head is None:
                errors.append("M29 prior M28 control source head is absent")
        errors.extend(
            _m28_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M28 live-claim/admission-recovery authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m27_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M27 on any declaration surface, including partial state."""

    key = "dead_owner_parallel_resume_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m27_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    """Bind the landed worker merge, watchdog repair, and control seal."""

    try:
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m27_source_delta(root, population, authority)
        return []
    except Exception as exc:
        return [
            "M27 exact worker-merge, watchdog-repair, and control-seal chain "
            f"differs: {type(exc).__name__}: {exc}"
        ]


def _m27_dead_owner_parallel_resume_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M27's stopped M26 anchor and exact dead-owner recovery."""

    key = "dead_owner_parallel_resume_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m27_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M27 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m27_dead_owner_parallel_resume_authority()
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append(
                "M27 dead-owner parallel-resume authority is only partially declared"
            )
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append(
                "M27 dead-owner parallel-resume authority differs across controls"
            )
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M27 dead-owner parallel-resume authority CID is not exact")

        required = {
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M27",
            "migration_kind": "stopped_run_orphan_recovery_successor_materialization",
            "prior_event_watermark": 264,
            "prior_plan_revision": 27,
            "prior_generation": 25,
            "prior_active_claim_count": 4,
            "prior_active_attempt_count": 4,
            "prior_active_lease_count": 4,
            "target_generation": 26,
            "target_quack_port": 24_070,
            "target_plan_revision": 28,
            "target_event_watermark": 268,
            "target_coordination_event_count": 2_586,
            "orphan_claim_expirations": 4,
            "task_revision_changes": 2,
            "task_status_changes": 2,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        for field, value in required.items():
            if expected.get(field) != value:
                errors.append(f"M27 authority field differs: {field}")
        dead_claims = expected.get("interrupted_claims", {})
        rearms = expected.get("task_rearms", {})
        if (
            not isinstance(dead_claims, Mapping)
            or set(dead_claims) != {"SAWM-006", "SAWM-008", "SAWM-012", "SAWM-015"}
            or not isinstance(rearms, Mapping)
            or set(rearms) != {"SAWM-006", "SAWM-012"}
            or expected.get("observed_unaccepted_source", {})
            .get("completion_authority")
            is not False
        ):
            errors.append("M27 exact dead-claim, rearm, or source authority differs")
        if require_active_runtime:
            target_root = (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27"
            )
            active_worktrees = (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m27/worktrees"
            )
            target_store = f"{target_root}/control.duckdb"
            program = scheduler.get("database_program", {})
            owner = scheduler.get("quack_owner", {})
            runtime = scheduler.get("runtime_paths", {})
            observed = (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            wanted = (
                target_store,
                "26",
                "quack:127.0.0.1:24070",
                f"{target_root}/events",
                f"{target_root}/registry",
                active_worktrees,
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_070,
            )
            expected_runtime = {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": active_worktrees,
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
            if observed != wanted or runtime != expected_runtime:
                errors.append("scheduler M27 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m28_successor_declared(
            scheduler, seal, migration
        ):
            m28 = (
                materializer
                ._expected_m28_live_claim_admission_recovery_authority()
            )
            source_chain = m28.get("source_chain", {})
            if isinstance(source_chain, Mapping):
                historical_control_head = str(
                    source_chain.get("prior_control_source_head")
                    or source_chain.get("m27_control_source_head")
                    or source_chain.get("m27_control_commit")
                    or ""
                ) or None
            if historical_control_head is None:
                source_authority = m28.get("source_authority", {})
                if isinstance(source_authority, Mapping):
                    historical_control_head = str(
                        source_authority.get("prior_control_source_head")
                        or source_authority.get("prior_source_head")
                        or ""
                    ) or None
            if historical_control_head is None:
                errors.append("M28 prior M27 control source head is absent")
        errors.extend(
            _m27_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        return errors
    except Exception as exc:
        return [
            "M27 dead-owner parallel-resume authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m26_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    key = "automatic_stall_recovery_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m26_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    try:
        population = materializer.build_population(root)
        if current_head:
            binding = dict(population["source_binding"])
            binding.update(
                {
                    "head": current_head,
                    "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                    "datasets_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                    ),
                    "kit_gitlink": _git(
                        root, "rev-parse", f"{current_head}:ipfs_kit_py"
                    ),
                }
            )
            population = {**population, "source_binding": binding}
        materializer._assert_m26_source_delta(root, population, authority)
        repair = str(authority["repair_source_commit"])
        expected = {path: "M" for path in authority["source_repair_paths"]}
        expected["test/api/test_agent_supervisor_database_dispatch_watchdog.py"] = "A"
        if materializer._m26_name_status(
            root, str(authority["prior_source_head"]), repair
        ) != expected:
            return ["M26 repair path/status inventory differs"]
        return []
    except Exception as exc:
        return [
            "M26 exact repair plus nine-control source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m26_automatic_stall_recovery_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    key = "automatic_stall_recovery_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m26_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M26 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m26_automatic_stall_recovery_authority()
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append(
                "M26 automatic-stall-recovery authority is only partially declared"
            )
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append(
                "M26 automatic-stall-recovery authority differs across controls"
            )
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M26 automatic-stall-recovery authority CID is not exact")
        required = {
            "schema": (
                "sawm/automatic-stall-recovery-successor-materialization-"
                "authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M26",
            "migration_kind": key,
            "prior_event_watermark": 258,
            "prior_plan_revision": 26,
            "prior_generation": 24,
            "prior_active_claim_count": 2,
            "prior_active_attempt_count": 2,
            "prior_active_lease_count": 2,
            "target_generation": 25,
            "target_quack_port": 24_069,
            "target_plan_revision": 27,
            "target_event_watermark": 262,
            "target_coordination_event_count": 2_421,
            "target_coordination_projection_digest": (
                "sha256:96b801b636e67c62ec35bc6338ba4b39494b0d6c8ca7439e8d88f4beab60c085"
            ),
            "orphan_claim_expirations": 2,
            "logical_completion_removals_for_rearm": 2,
            "task_revision_changes": 2,
            "task_status_changes": 2,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        for field, value in required.items():
            if expected.get(field) != value:
                errors.append(f"M26 authority field differs: {field}")
        if (
            set(expected.get("orphan_claims", {})) != {"SAWM-006", "SAWM-015"}
            or set(expected.get("failure_receipts", {}))
            != {"SAWM-008", "SAWM-012"}
            or set(expected.get("task_rearms", {}))
            != {"SAWM-008", "SAWM-012"}
            or expected.get("observed_source_chain", {})
            .get("SAWM-012", {})
            .get("completion_authority")
            is not False
            or expected.get("observed_source_chain", {})
            .get("SAWM-010", {})
            .get("completion_authority")
            is not True
        ):
            errors.append("M26 exact orphan/failure/source authority differs")
        later = expected.get("later_source_admission", {})
        if (
            later.get("bare_descendant_allowed") is not False
            or later.get("two_parent_supervisor_merge_required") is not True
            or later.get("canonical_quack_completion_required") is not True
            or later.get("accepted_source_transition_schema")
            != "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@3"
        ):
            errors.append("M26 receipt-backed later-source policy differs")
        if require_active_runtime:
            target_root = (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m26"
            )
            target_store = f"{target_root}/control.duckdb"
            program = scheduler.get("database_program", {})
            owner = scheduler.get("quack_owner", {})
            runtime = scheduler.get("runtime_paths", {})
            if (
                (
                    program.get("store_id"),
                    program.get("store_generation"),
                    program.get("quack_endpoint"),
                    program.get("event_store_path"),
                    program.get("runtime_registry_path"),
                    program.get("worktree_root"),
                    owner.get("database_path"),
                    owner.get("store_id"),
                    owner.get("state_dir"),
                    owner.get("port"),
                )
                != (
                    target_store,
                    "25",
                    "quack:127.0.0.1:24069",
                    f"{target_root}/events",
                    f"{target_root}/registry",
                    f"{target_root}/worktrees",
                    target_store,
                    target_store,
                    f"{target_root}/quack-owner",
                    24_069,
                )
                or runtime
                != {
                    "root": target_root,
                    "state": f"{target_root}/state",
                    "worktrees": f"{target_root}/worktrees",
                    "merge_queue": f"{target_root}/merge-queue",
                    "logs": f"{target_root}/logs",
                    "generated_runtime_artifacts_are_completion_authority": False,
                }
            ):
                errors.append("scheduler M26 target/runtime binding is not exact")
        historical_control_head = None
        if not require_active_runtime and _m27_successor_declared(
            scheduler, seal, migration
        ):
            m27 = materializer._expected_m27_dead_owner_parallel_resume_authority()
            historical_control_head = str(m27["prior_control_source_head"])
        errors.extend(
            _m26_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )
        capsule_source = (
            root
            / "ipfs_accelerate_py/agent_supervisor/runtime/"
            "configured_board_live_capsule.py"
        ).read_text(encoding="utf-8")
        if (
            "accepted-source-transition@3" not in capsule_source
            or "canonical Quack completion state" not in capsule_source
            or "contains a non-supervisor merge" not in capsule_source
        ):
            errors.append("M26 @3 receipt-backed source admission repair is absent")
        return errors
    except Exception as exc:
        return [
            "M26 automatic-stall-recovery authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m25_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M25 on any declaration surface, including partial state."""

    key = "native_duckdb_preload_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m25_name_status(root: Path, before: str, after: str) -> dict[str, str]:
    """Return an exact modifications-only source delta for M25."""

    changed: dict[str, str] = {}
    for line in _git(
        root, "diff", "--name-status", "--no-renames", before, after, "--"
    ).splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] != "M" or fields[1] in changed:
            raise RuntimeError(
                f"M25 source delta contains a non-modification entry: {line}"
            )
        changed[fields[1]] = fields[0]
    return changed


def _m25_source_chain_errors(
    root: Path,
    materializer: Any,
    authority: Mapping[str, Any],
    *,
    current_head: str | None = None,
) -> list[str]:
    """Bind the repair commit followed by one exact nine-control commit."""

    try:
        prior_head = str(authority["prior_source_head"])
        repair_head = str(authority["repair_source_commit"])
        current_head = current_head or _git(root, "rev-parse", "HEAD")
        repair_paths = set(str(path) for path in authority["source_repair_paths"])
        operator_paths = set(
            str(path) for path in authority["operator_control_paths"]
        )
        expected_repair_blobs = dict(authority["repair_source_blobs"])
        accepted = authority.get("accepted_control_plane_repair")
        expected_operator_paths = set(materializer._M18_OPERATOR_CONTROL_PATHS)
        if (
            current_head in {prior_head, repair_head}
            or _git(root, "rev-parse", f"{repair_head}^") != prior_head
            or _git(root, "rev-parse", f"{current_head}^") != repair_head
            or _git(root, "rev-parse", f"{prior_head}^{{tree}}")
            != authority["prior_source_tree"]
            or _git(root, "rev-parse", f"{repair_head}^{{tree}}")
            != authority["repair_source_tree"]
            or repair_paths != set(materializer._M25_NATIVE_PRELOAD_SOURCE_BLOBS)
            or expected_repair_blobs
            != dict(materializer._M25_NATIVE_PRELOAD_SOURCE_BLOBS)
            or operator_paths != expected_operator_paths
            or not isinstance(accepted, Mapping)
            or accepted.get("precursor_source_head") != prior_head
            or accepted.get("repair_source_commit") != repair_head
            or set(accepted.get("changed_paths", ())) != repair_paths
            or dict(accepted.get("blob_oids", {})) != expected_repair_blobs
            or int(accepted.get("ordinary_source_changes", -1)) != 0
            or int(authority.get("ordinary_source_changes", -1)) != 0
            or _m25_name_status(root, prior_head, repair_head)
            != {path: "M" for path in repair_paths}
            or _m25_name_status(root, repair_head, current_head)
            != {path: "M" for path in operator_paths}
        ):
            return [
                "M25 source transition is not the exact native preload repair "
                "followed by one nine-control commit"
            ]
        _git(root, "merge-base", "--is-ancestor", prior_head, repair_head)
        _git(root, "merge-base", "--is-ancestor", repair_head, current_head)
        for path, blob_oid in expected_repair_blobs.items():
            if _git(root, "rev-parse", f"{repair_head}:{path}") != blob_oid:
                return [f"M25 native preload repair blob differs: {path}"]
        for dependency, gitlink_key, tree_key in (
            ("ipfs_datasets_py", "prior_datasets_gitlink", "prior_datasets_tree"),
            ("ipfs_kit_py", "prior_kit_gitlink", "prior_kit_tree"),
        ):
            gitlink = str(authority[gitlink_key])
            if (
                any(
                    _git(root, "rev-parse", f"{head}:{dependency}") != gitlink
                    for head in (prior_head, repair_head, current_head)
                )
                or _git(root / dependency, "rev-parse", f"{gitlink}^{{tree}}")
                != authority[tree_key]
            ):
                return [f"M25 {dependency} source authority differs"]
        return []
    except Exception as exc:
        return [
            "M25 exact two-commit source chain differs: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m25_native_duckdb_preload_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M25's failed-start anchor, source repair and fresh successor."""

    key = "native_duckdb_preload_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m25_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M25 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m25_native_duckdb_preload_authority()
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append(
                "M25 native-DuckDB preload successor authority is only "
                "partially declared"
            )
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append(
                "M25 native-DuckDB preload authority differs across controls"
            )
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M25 native-DuckDB preload authority CID is not exact")

        target_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m25"
        )
        target_store = f"{target_root}/control.duckdb"
        required = {
            "schema": (
                "sawm/native-duckdb-preload-successor-materialization-"
                "authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M25",
            "migration_kind": key,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m24/control.duckdb"
            ),
            "prior_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m24/control.coordination.duckdb"
            ),
            "prior_failed_start_control_store_sha256": (
                "27c7bf7f923005ec66eec6f4b75b68cbe84fd3c8bf6273ebbb2b6d954ac64f35"
            ),
            "prior_failed_start_control_store_size": 43_528_192,
            "prior_coordination_store_sha256": (
                "0671fbc77fa65bb30bf2c7227ccf963b9833223a9ed178461c278b21413e1957"
            ),
            "prior_coordination_store_size": 13_905_920,
            "prior_read_replica_sha256": (
                "27c7bf7f923005ec66eec6f4b75b68cbe84fd3c8bf6273ebbb2b6d954ac64f35"
            ),
            "prior_owner_status_present": False,
            "prior_listener_present": False,
            "prior_owner_marker_present": False,
            "prior_pid_present": False,
            "prior_stop_control_present": False,
            "prior_token_handoff_present": False,
            "prior_wals_absent": True,
            "prior_runtime_state_absent": True,
            "maximum_persisted_generation": 23,
            "failed_attempted_generation": 24,
            "failed_attempt_allocated_generation": False,
            "prior_source_head": "bb02830699388df9b52c01830c9e970a56c56796",
            "repair_source_commit": "135b077c5ad9482bbb167fdfc81d8b7855ff5fab",
            "target_store_id": target_store,
            "target_coordination_store_id": (
                f"{target_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": target_root,
            "target_generation": 24,
            "target_quack_port": 24_068,
            "target_plan_revision": 26,
            "target_event_watermark": 251,
            "target_coordination_event_count": 1_407,
            "event_suffix_length": 2,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "coordination_semantic_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if any(expected.get(name) != value for name, value in required.items()):
            errors.append(
                "M25 native-DuckDB preload successor authority is not exact"
            )
        failure = expected.get("failed_quack_start")
        diagnosis = (
            failure.get("diagnosis") if isinstance(failure, Mapping) else None
        )
        native_repair = expected.get("native_preload_repair")
        if (
            not isinstance(failure, Mapping)
            or not isinstance(diagnosis, Mapping)
            or expected.get("failed_quack_start_cid")
            != materializer._identity(dict(failure))
            or failure.get("attempt") != "SAWM-R2-M24-QUACK-START-A1"
            or failure.get("phase") != "pre_identity_replica_extension_load"
            or failure.get("failed_operation") != "LOAD httpfs"
            or failure.get("load_quack_reached") is not False
            or failure.get("quack_serve_reached") is not False
            or failure.get("embedded_quack_involved") is not False
            or failure.get("identity_publication_reached") is not False
            or failure.get("listener_created") is not False
            or failure.get("owner_status_created") is not False
            or failure.get("state_server_row_created") is not False
            or failure.get("store_generation_row_created") is not False
            or failure.get("maximum_persisted_generation") != 23
            or failure.get("task_revision_changes") != 0
            or failure.get("task_status_changes") != 0
            or failure.get("coordination_semantic_changes") != 0
            or failure.get("provider_invocations") != 0
            or failure.get("worker_self_approval") is not False
            or diagnosis.get("accepted_native_distribution_version") != "1.5.5"
            or diagnosis.get("failed_process_native_distribution_version")
            != "1.5.2"
            or "v1.5.2/linux_arm64/httpfs.duckdb_extension"
            not in str(failure.get("normalized_error") or "")
            or diagnosis.get("compatibility_extension_required") is not False
            or diagnosis.get("network_or_install_required") is not False
        ):
            errors.append("M25 failed M24 start evidence is not exact")
        if (
            not isinstance(native_repair, Mapping)
            or native_repair.get("accepted_native_distribution_version")
            != "1.5.5"
            or native_repair.get("preload_before_offline_validation") is not True
            or native_repair.get("preload_before_owner_start") is not True
            or native_repair.get("sealed_descriptor_retained_for_owner_lifetime")
            is not True
            or native_repair.get("ambient_duckdb_alias_rejected") is not True
            or native_repair.get("ambient_loader_environment_rejected") is not True
            or native_repair.get("sanitized_process_birth_required") is not True
            or native_repair.get("in_process_loader_environment_mutation")
            is not False
            or native_repair.get("network_install_allowed") is not False
            or native_repair.get("compatibility_extension_added") is not False
            or native_repair.get("native_distribution_changed") is not False
            or "ambient_loader_paths_removed" in native_repair
        ):
            errors.append(
                "M25 native preload repair does not require external sanitized "
                "process birth"
            )
        historical_control_head: str | None = None
        if not require_active_runtime and _m26_successor_declared(
            scheduler, seal, migration
        ):
            successor_prior = str(
                materializer._expected_m26_automatic_stall_recovery_authority()[
                    "prior_source_head"
                ]
            )
            first_parent_lineage = _git(
                root,
                "rev-list",
                "--first-parent",
                "--reverse",
                f"{expected['repair_source_commit']}..{successor_prior}",
            ).splitlines()
            if not first_parent_lineage:
                errors.append("M25 historical control commit is unavailable")
            else:
                historical_control_head = first_parent_lineage[0]
        errors.extend(
            _m25_source_chain_errors(
                root,
                materializer,
                expected,
                current_head=historical_control_head,
            )
        )

        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        lanes = scheduler.get("lanes")
        provider = scheduler.get("provider", {})
        lane_identity = (
            [
                (
                    lane.get("index"),
                    lane.get("name"),
                    lane.get("strict_shard_remainder"),
                    lane.get("initial_task_ids"),
                )
                for lane in lanes
            ]
            if type(lanes) is list
            and all(type(lane) is dict for lane in lanes)
            else None
        )
        provider_cap = (
            provider.get("max_concurrency")
            if isinstance(provider, Mapping)
            else None
        )
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "24",
                "quack:127.0.0.1:24068",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_068,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M25 target/runtime binding is not exact")
        if require_active_runtime and (
            type(scheduler.get("max_lanes")) is not int
            or scheduler.get("max_lanes") != 4
            or scheduler.get("strict_task_sharding") is not True
            or scheduler.get("idle_lane_work_stealing") != ""
            or lane_identity
            != [
                (0, "sawm-lane-0", 0, ["SAWM-008"]),
                (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
                (2, "sawm-lane-2", 2, ["SAWM-015"]),
                (3, "sawm-lane-3", 3, ["SAWM-012"]),
            ]
            or any(
                type(value) is not int
                for lane in (lanes if type(lanes) is list else ())
                if type(lane) is dict
                for value in (
                    lane.get("index"),
                    lane.get("strict_shard_remainder"),
                )
            )
            or type(provider_cap) is not int
            or provider_cap < 4
        ):
            errors.append("scheduler M25 strict four-lane contract is not exact")
        return errors
    except Exception as exc:
        return [
            "M25 native-DuckDB preload successor authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m24_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M24 on any declaration surface, including partial state."""

    key = "multi_lane_sidecar_reopen_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m24_sidecar_reopen_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M24's exact sidecar-reopen repair and four-lane successor."""

    key = "multi_lane_sidecar_reopen_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m24_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M24 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m24_sidecar_reopen_authority()
        errors: list[str] = []
        presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
        if not all(presence):
            errors.append(
                "M24 sidecar-reopen successor authority is only partially declared"
            )
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("M24 sidecar-reopen authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M24 sidecar-reopen authority CID is not exact")

        target_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m24"
        )
        target_store = f"{target_root}/control.duckdb"
        required = {
            "schema": (
                "sawm/multi-lane-sidecar-reopen-successor-"
                "materialization-authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M24",
            "migration_kind": key,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m23/control.duckdb"
            ),
            "target_store_id": target_store,
            "target_coordination_store_id": (
                f"{target_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": target_root,
            "target_generation": 24,
            "target_quack_port": 24_067,
            "target_plan_revision": 25,
            "target_event_watermark": 249,
            "target_coordination_event_count": 1_407,
            "task_revision_changes": 2,
            "task_status_changes": 2,
            "coordination_semantic_changes": 2,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if any(expected.get(name) != value for name, value in required.items()):
            errors.append("M24 sidecar-reopen successor authority is not exact")
        try:
            current_head = _git(root, "rev-parse", "HEAD")
            if not require_active_runtime and _m25_successor_declared(
                scheduler, seal, migration
            ):
                current_head = str(
                    materializer._expected_m25_native_duckdb_preload_authority()[
                        "prior_source_head"
                    ]
                )
            materializer._assert_m24_source_delta(
                root,
                {
                    "source_binding": {
                        "head": current_head,
                        "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                        "datasets_gitlink": _git(
                            root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                        ),
                        "kit_gitlink": _git(
                            root, "rev-parse", f"{current_head}:ipfs_kit_py"
                        ),
                    }
                },
                expected,
            )
        except Exception as exc:
            errors.append(
                "M24 exact control/source seal differs: "
                f"{type(exc).__name__}: {exc}"
            )

        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        lanes = scheduler.get("lanes")
        provider = scheduler.get("provider", {})
        lane_identity = (
            [
                (
                    lane.get("index"),
                    lane.get("name"),
                    lane.get("strict_shard_remainder"),
                    lane.get("initial_task_ids"),
                )
                for lane in lanes
            ]
            if type(lanes) is list
            and all(type(lane) is dict for lane in lanes)
            else None
        )
        provider_cap = (
            provider.get("max_concurrency")
            if isinstance(provider, Mapping)
            else None
        )
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "24",
                "quack:127.0.0.1:24067",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_067,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M24 target/runtime binding is not exact")
        if require_active_runtime and (
            type(scheduler.get("max_lanes")) is not int
            or scheduler.get("max_lanes") != 4
            or scheduler.get("strict_task_sharding") is not True
            or scheduler.get("idle_lane_work_stealing") != ""
            or lane_identity
            != [
                (0, "sawm-lane-0", 0, ["SAWM-008"]),
                (1, "sawm-lane-1", 1, ["SAWM-006", "SAWM-010"]),
                (2, "sawm-lane-2", 2, ["SAWM-015"]),
                (3, "sawm-lane-3", 3, ["SAWM-012"]),
            ]
            or any(
                type(value) is not int
                for lane in (lanes if type(lanes) is list else ())
                if type(lane) is dict
                for value in (
                    lane.get("index"),
                    lane.get("strict_shard_remainder"),
                )
            )
            or type(provider_cap) is not int
            or provider_cap < 4
        ):
            errors.append("scheduler M24 strict four-lane contract is not exact")
        return errors
    except Exception as exc:
        return [
            "M24 sidecar-reopen successor authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m23_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M23 on any declaration surface, including partial state."""

    key = "multi_lane_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m23_multi_lane_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M23's exact sealed four-lane successor authority."""

    key = "multi_lane_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m23_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M23 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m23_multi_lane_authority()
        errors: list[str] = []
        presence = (
            key in scheduler,
            key in migration,
            f"{key}_cid" in seal,
        )
        if not all(presence):
            errors.append("M23 multi-lane successor authority is only partially declared")
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("M23 multi-lane authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M23 multi-lane authority CID is not exact")

        target_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m23"
        )
        target_store = f"{target_root}/control.duckdb"
        required = {
            "schema": "sawm/multi-lane-successor-materialization-authorization@1",
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M23",
            "migration_kind": key,
            "supersession_mode": key,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m22/control.duckdb"
            ),
            "target_store_id": target_store,
            "target_coordination_store_id": (
                f"{target_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": target_root,
            "target_generation": 23,
            "target_quack_port": 24_066,
            "target_plan_revision": 24,
            "target_event_watermark": 244,
            "lane_contract": {
                "max_lanes": 4,
                "lane_indices": [0, 1, 2, 3],
                "strict_shard_remainders": [0, 1, 2, 3],
                "strict_task_sharding": True,
                "idle_lane_work_stealing": "",
                "provider_minimum_concurrency": 4,
                "coordination_authority_count": 1,
                "coordination_transport": "quack_proxy_only",
                "lane_local_execution_sidecars": True,
            },
        }
        if any(expected.get(name) != value for name, value in required.items()):
            errors.append("M23 multi-lane successor authority is not exact")

        try:
            current_head = _git(root, "rev-parse", "HEAD")
            if not require_active_runtime and _m24_successor_declared(
                scheduler, seal, migration
            ):
                current_head = str(
                    materializer._expected_m24_sidecar_reopen_authority()[
                        "prior_source_head"
                    ]
                )
            materializer._assert_m23_source_delta(
                root,
                {
                    "source_binding": {
                        "head": current_head,
                        "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                        "datasets_gitlink": _git(
                            root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                        ),
                        "kit_gitlink": _git(
                            root, "rev-parse", f"{current_head}:ipfs_kit_py"
                        ),
                    }
                },
                expected,
            )
        except Exception as exc:
            errors.append(
                "M23 exact control/source seal differs: "
                f"{type(exc).__name__}: {exc}"
            )

        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        lanes = scheduler.get("lanes")
        provider = scheduler.get("provider", {})
        lane_identity = (
            [
                (
                    lane.get("index"),
                    lane.get("name"),
                    lane.get("strict_shard_remainder"),
                )
                for lane in lanes
            ]
            if type(lanes) is list
            and all(type(lane) is dict for lane in lanes)
            else None
        )
        provider_cap = (
            provider.get("max_concurrency")
            if isinstance(provider, Mapping)
            else None
        )
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "23",
                "quack:127.0.0.1:24066",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_066,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M23 target/runtime binding is not exact")
        if require_active_runtime and (
            type(scheduler.get("max_lanes")) is not int
            or scheduler.get("max_lanes") != 4
            or scheduler.get("strict_task_sharding") is not True
            or scheduler.get("idle_lane_work_stealing") != ""
            or lane_identity
            != [
                (0, "sawm-lane-0", 0),
                (1, "sawm-lane-1", 1),
                (2, "sawm-lane-2", 2),
                (3, "sawm-lane-3", 3),
            ]
            or any(
                type(value) is not int
                for lane in (lanes if type(lanes) is list else ())
                if type(lane) is dict
                for value in (
                    lane.get("index"),
                    lane.get("strict_shard_remainder"),
                )
            )
            or type(provider_cap) is not int
            or provider_cap < 4
        ):
            errors.append("scheduler M23 strict four-lane contract is not exact")
        return errors
    except Exception as exc:
        return [
            "M23 multi-lane successor authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m22_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M22 on any declaration surface, including partial state."""

    key = "live_preflight_receipt_compatibility_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _assert_m23_historical_source_handoff(
    materializer: Any,
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    """Seal inactive predecessors at M23's exact precursor/repair boundary."""

    key = "multi_lane_successor_materialization"
    expected = materializer._expected_m23_multi_lane_authority()
    repair_head = str(expected["repair_source_commit"])
    repair_blobs = expected["repair_source_blobs"]
    if (
        type(scheduler.get(key)) is not dict
        or materializer._identity(scheduler.get(key))
        != materializer._identity(expected)
        or type(migration.get(key)) is not dict
        or materializer._identity(migration.get(key))
        != materializer._identity(expected)
        or seal.get(f"{key}_cid") != materializer._identity(expected)
        or _git(root, "rev-parse", f"{repair_head}^{{tree}}")
        != expected["repair_source_tree"]
        or any(
            _git(root, "rev-parse", f"{repair_head}:{path}") != blob_oid
            for path, blob_oid in repair_blobs.items()
        )
    ):
        raise RuntimeError("M23 historical source handoff is not exact")
    _git(
        root,
        "merge-base",
        "--is-ancestor",
        str(expected["prior_source_head"]),
        repair_head,
    )


def _m22_live_preflight_receipt_compatibility_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M22's exact live-preflight receipt compatibility authority."""

    key = "live_preflight_receipt_compatibility_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m22_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M22 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = (
            materializer._expected_m22_live_preflight_receipt_compatibility_authority()
        )
        errors: list[str] = []
        presence = (
            key in scheduler,
            key in migration,
            f"{key}_cid" in seal,
        )
        if not all(presence):
            errors.append(
                "M22 live-preflight receipt compatibility successor authority "
                "is only partially declared"
            )
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append(
                "M22 live-preflight receipt compatibility authority differs "
                "across controls"
            )
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append(
                "M22 live-preflight receipt compatibility authority CID is not exact"
            )

        target_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m22"
        )
        target_store = f"{target_root}/control.duckdb"
        required = {
            "schema": (
                "sawm/live-preflight-receipt-compatibility-successor-"
                "materialization-authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M22",
            "migration_kind": key,
            "supersession_mode": key,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m21/control.duckdb"
            ),
            "target_store_id": target_store,
            "target_coordination_store_id": (
                f"{target_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": target_root,
            "target_generation": 22,
            "target_quack_port": 24_065,
            "target_plan_revision": 23,
            "target_event_watermark": 233,
            "event_suffix_length": 2,
            "ordinary_source_changes": 0,
            "coordination_semantic_changes": 0,
            "plan_revision_changes": 1,
            "evidence_node_changes": 1,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "goal_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
            "worker_self_approval": False,
        }
        if any(expected.get(name) != value for name, value in required.items()):
            errors.append(
                "M22 live-preflight receipt compatibility authority is not exact"
            )

        try:
            if require_active_runtime:
                current_head = _git(root, "rev-parse", "HEAD")
                materializer._assert_m22_source_delta(
                    root,
                    {
                        "source_binding": {
                            "head": current_head,
                            "tree": _git(
                                root, "rev-parse", f"{current_head}^{{tree}}"
                            ),
                            "datasets_gitlink": _git(
                                root,
                                "rev-parse",
                                f"{current_head}:ipfs_datasets_py",
                            ),
                            "kit_gitlink": _git(
                                root, "rev-parse", f"{current_head}:ipfs_kit_py"
                            ),
                        }
                    },
                    expected,
                )
            else:
                # M22 is historical only after M23 independently seals the
                # exact precursor, repair commit/tree, and every repair blob.
                m23_key = "multi_lane_successor_materialization"
                expected_m23 = materializer._expected_m23_multi_lane_authority()
                repair_head = str(expected_m23["repair_source_commit"])
                repair_blobs = expected_m23["repair_source_blobs"]
                if (
                    type(scheduler.get(m23_key)) is not dict
                    or materializer._identity(scheduler.get(m23_key))
                    != materializer._identity(expected_m23)
                    or type(migration.get(m23_key)) is not dict
                    or materializer._identity(migration.get(m23_key))
                    != materializer._identity(expected_m23)
                    or seal.get(f"{m23_key}_cid")
                    != materializer._identity(expected_m23)
                    or _git(root, "rev-parse", f"{repair_head}^{{tree}}")
                    != expected_m23["repair_source_tree"]
                    or any(
                        _git(root, "rev-parse", f"{repair_head}:{path}")
                        != blob_oid
                        for path, blob_oid in repair_blobs.items()
                    )
                ):
                    raise RuntimeError(
                        "M23 historical handoff does not seal the exact repair"
                    )
                _git(
                    root,
                    "merge-base",
                    "--is-ancestor",
                    str(expected_m23["prior_source_head"]),
                    repair_head,
                )
        except Exception as exc:
            errors.append(
                "M22 exact control/source seal differs: "
                f"{type(exc).__name__}: {exc}"
            )

        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "22",
                "quack:127.0.0.1:24065",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_065,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M22 target/runtime binding is not exact")
        return errors
    except Exception as exc:
        return [
            "M22 live-preflight receipt compatibility authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m21_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M21 on any declaration surface, including partial state."""

    key = "generation_realization_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m21_generation_realization_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M21's exact generation-realization successor authority."""

    key = "generation_realization_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m21_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M21 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m21_generation_realization_authority()
        errors: list[str] = []
        presence = (
            key in scheduler,
            key in migration,
            f"{key}_cid" in seal,
        )
        if not all(presence):
            errors.append(
                "M21 generation-realization successor authority is only "
                "partially declared"
            )
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append(
                "M21 generation-realization authority differs across controls"
            )
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M21 generation-realization authority CID is not exact")

        target_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m21"
        )
        target_store = f"{target_root}/control.duckdb"
        required = {
            "schema": (
                "sawm/generation-realization-successor-materialization-"
                "authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M21",
            "migration_kind": "generation_realization_successor_materialization",
            "supersession_mode": (
                "generation_realization_successor_materialization"
            ),
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m20/control.duckdb"
            ),
            "target_store_id": target_store,
            "target_coordination_store_id": (
                f"{target_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": target_root,
            "target_generation": 21,
            "target_quack_port": 24_064,
            "target_plan_revision": 22,
            "target_event_watermark": 231,
            "event_suffix_length": 2,
            "ordinary_source_changes": 0,
            "coordination_semantic_changes": 0,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "goal_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if any(expected.get(name) != value for name, value in required.items()):
            errors.append("M21 generation-realization authority is not exact")

        try:
            if require_active_runtime:
                current_head = _git(root, "rev-parse", "HEAD")
                materializer._assert_m21_source_delta(
                    root,
                    {
                        "source_binding": {
                            "head": current_head,
                            "tree": _git(
                                root, "rev-parse", f"{current_head}^{{tree}}"
                            ),
                            "datasets_gitlink": _git(
                                root,
                                "rev-parse",
                                f"{current_head}:ipfs_datasets_py",
                            ),
                            "kit_gitlink": _git(
                                root, "rev-parse", f"{current_head}:ipfs_kit_py"
                            ),
                        }
                    },
                    expected,
                )
            else:
                _assert_m23_historical_source_handoff(
                    materializer, scheduler, seal, migration, root=root
                )
        except Exception as exc:
            errors.append(
                "M21 exact control/source seal differs: "
                f"{type(exc).__name__}: {exc}"
            )

        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "21",
                "quack:127.0.0.1:24064",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_064,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M21 target/runtime binding is not exact")
        return errors
    except Exception as exc:
        return [
            "M21 generation-realization authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m20_successor_declared(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> bool:
    """Select M20 on any declaration surface, including partial/tampered state."""

    key = "test_isolation_successor_materialization"
    return any((key in scheduler, key in migration, f"{key}_cid" in seal))


def _m20_test_isolation_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M20's exact never-launched-M19 source-only successor."""

    key = "test_isolation_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m20_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M20 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m20_test_isolation_authority()
        errors: list[str] = []
        presence = (
            key in scheduler,
            key in migration,
            f"{key}_cid" in seal,
        )
        if not all(presence):
            errors.append(
                "M20 test-isolation successor authority is only partially declared"
            )
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("M20 test-isolation authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M20 test-isolation authority CID is not exact")
        required = {
            "schema": "sawm/post-materialization-test-isolation-repair-authorization@1",
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M20",
            "migration_kind": "post_materialization_test_isolation_repair",
            "supersession_mode": (
                "source_only_post_materialization_test_isolation_repair"
            ),
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m19/control.duckdb"
            ),
            "prior_control_store_sha256": (
                "0a505aae2923b21c445e019da803a97ae3bf168ec6c26cb45b419bed49d0df5d"
            ),
            "prior_control_store_size": 43_528_192,
            "prior_coordination_store_sha256": (
                "4ffd71f5ccbb1953a84e430d2ffbc114fffd3d787abdf7cc39a42a5d3ef13e41"
            ),
            "prior_coordination_store_size": 14_168_064,
            "prior_event_watermark": 227,
            "prior_event_prefix_sha256": (
                "45cccf8ea81087e168110d3e2abfd85c79656863519f632fcf0795a1d0208c80"
            ),
            "prior_projection_cid": (
                "baguqeeravlrp2wmy7cbtuhoe2h6hs6tbphsunilvaufkktev6vv3ubl2nbia"
            ),
            "prior_migration_receipt_sha256": (
                "cd62f5e9b597147633b4caef6157ddeccd648a6f5326bedc5ac7a92889a391d6"
            ),
            "prior_migration_receipt_size": 6_180,
            "prior_migration_receipt_cid": (
                "sha256:c89f0ba8fc663ac57d39d162fe7e283f606f556f37b38cb5ee71c83b274318c0"
            ),
            "prior_validation_digest": (
                "sha256:60cdc6646b46d52d5102ff7c82595fdaa95c0c4e632221e28fac9d126adc2873"
            ),
            "prior_source_head": "d5275b900cd223643658afad19c643506d32a748",
            "prior_source_tree": "9b6bf64f1bbca282c977ac9acfd9e204b4385da2",
            "prior_source_binding_cid": (
                "sha256:79b9fcb269ad29dd30d77084afeca33eab8b9cf8164a548748bf69a9908b298f"
            ),
            "repair_source_commit": "3e926a247acd5654887b276afa3d00b896c08a2d",
            "repair_source_tree": "f3d89a58c8df8dc2dfd29a2b8422e4fd8ce683d2",
            "target_runtime_root": (
                "data/agent_supervisor/semantic_addressed_world_model/run-r2-m20"
            ),
            "target_generation": 21,
            "target_quack_port": 24_063,
            "target_plan_revision": 21,
            "target_event_watermark": 229,
            "event_suffix_length": 2,
            "coordination_semantic_changes": 0,
            "task_revision_changes": 0,
            "task_status_changes": 0,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "worker_self_approval": False,
        }
        if any(expected.get(name) != value for name, value in required.items()):
            errors.append("M20 test-isolation authority is not exact")
        repair = expected.get("accepted_source_repair")
        wanted = {
            "test/api/semantic_world/test_semantic_addressed_world_model_board.py": (
                "fea8d6784b697ce4430460446b42262577d092a5"
            )
        }
        if (
            not isinstance(repair, Mapping)
            or repair.get("blob_oids") != wanted
            or set(repair.get("changed_paths", ())) != set(wanted)
            or repair.get("source_only") is not True
        ):
            errors.append("M20 exact test-isolation repair blob differs")
        expected_paths = {
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
        }
        if set(expected.get("bounded_control_plane_repair_paths", ())) != expected_paths:
            errors.append("M20 bounded source repair paths are not exact")
        try:
            if require_active_runtime:
                current_head = _git(root, "rev-parse", "HEAD")
                materializer._assert_m20_source_delta(
                    root,
                    {
                        "source_binding": {
                            "head": current_head,
                            "tree": _git(
                                root, "rev-parse", f"{current_head}^{{tree}}"
                            ),
                            "datasets_gitlink": _git(
                                root,
                                "rev-parse",
                                f"{current_head}:ipfs_datasets_py",
                            ),
                            "kit_gitlink": _git(
                                root, "rev-parse", f"{current_head}:ipfs_kit_py"
                            ),
                        }
                    },
                    expected,
                )
            else:
                _assert_m23_historical_source_handoff(
                    materializer, scheduler, seal, migration, root=root
                )
        except Exception as exc:
            errors.append(
                "M20 exact repair/source seal differs: "
                f"{type(exc).__name__}: {exc}"
            )
        target_root = str(expected["target_runtime_root"])
        target_store = str(expected["target_store_id"])
        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("store_id"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "21",
                "quack:127.0.0.1:24063",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                target_store,
                f"{target_root}/quack-owner",
                24_063,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M20 target/runtime binding is not exact")
        return errors
    except Exception as exc:
        return [
            "M20 test-isolation authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m19_live_catalog_inventory_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M19's exact stopped-M18, source-only successor controls."""

    key = "live_catalog_inventory_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m19_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M19 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m19_live_catalog_inventory_authority()
        errors: list[str] = []
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("M19 source-binding authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M19 source-binding authority CID is not exact")

        target_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m19"
        )
        target_store = f"{target_root}/control.duckdb"
        required = {
            "schema": (
                "sawm/live-quack-catalog-inventory-repair-authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M19",
            "migration_kind": "live_quack_catalog_inventory_repair",
            "supersession_mode": "source_only_live_quack_catalog_inventory_repair",
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m18/control.duckdb"
            ),
            "prior_control_store_sha256": (
                "2050e7a0869590c7744b42e08fa2333326690f5176f9414a429ac9ec44b272bc"
            ),
            "prior_control_store_size": 43_528_192,
            "prior_coordination_store_sha256": (
                "4ffd71f5ccbb1953a84e430d2ffbc114fffd3d787abdf7cc39a42a5d3ef13e41"
            ),
            "prior_coordination_store_size": 14_168_064,
            "prior_event_watermark": 225,
            "prior_event_prefix_sha256": (
                "6d392f1b9e3fee27ba2a8472f0c82fcf4d306a79acc65275051263675db1cd50"
            ),
            "prior_projection_cid": (
                "baguqeeraqsmnfs6rzwc6bvjjdszmryjtdrtg5aosnk2wdsppaxymsrrpqyxa"
            ),
            "prior_semantic_authority_digest": (
                "sha256:5e8d0afb732eaa5912512086a3d1ad288fa2de2b41b2b47462bf9006a6b7da5f"
            ),
            "prior_frozen_base_authority_digest": (
                "sha256:b7e832646a9e8014f64f61c86530476d219ef10c25a467086cd73638cbbd034b"
            ),
            "prior_append_surface_digest": (
                "sha256:d3eef2cbdce0cdb31a100ea8c6f6bac7929e7430fae26b49ded0b671cec2265d"
            ),
            "prior_catalog_digest": (
                "sha256:3cc2e066bd4495e6efc0a3410a72c5df29242dcacfa94cd75d30f61c658539a7"
            ),
            "prior_coordination_projection_digest": (
                "sha256:659b67b3f48e632337609d2c872c1d342650628921c6be52415a3fa9d73db1a1"
            ),
            "prior_coordination_event_count": 1_125,
            "prior_generation": 19,
            "prior_plan_revision": 19,
            "prior_server_id": "server:012fddc4-b201-4fad-a1c8-a8e930bb1a6b",
            "prior_process_birth_id": "birth:4197a4736b74b84f0e7713dbdcfd7c7e",
            "prior_startup_epoch": 1_788_025_991,
            "prior_started_at": "2026-08-29T17:53:11Z",
            "prior_stopped_at": "2026-08-29T18:00:30Z",
            "prior_stopped_status_projection_sha256": (
                "404184bae88f332675f24e6956b8c2ac79b326e87ce4e8e6591d03ea8d70419e"
            ),
            "prior_stopped_status_projection_size": 2_410,
            "prior_migration_receipt_sha256": (
                "a01d0341c0806ad591a151f0a6a47481347968b094d23083aa80d9a0b4ffeb4d"
            ),
            "prior_migration_receipt_size": 8_208,
            "prior_migration_receipt_cid": (
                "sha256:47fd0b5bee77c27d8f523c2cfe8ed8aec49377e8033b0f62c579d844b4efc225"
            ),
            "prior_source_head": "68ee411a072e8abe673fcc3df36c71698830a5aa",
            "prior_source_tree": "31c0325b7fe8b9c6497a232ee86a057102dbf1b5",
            "target_store_id": target_store,
            "target_coordination_store_id": (
                f"{target_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": target_root,
            "target_generation": 20,
            "target_quack_port": 24_062,
            "target_plan_revision": 20,
            "target_event_watermark": 227,
            "target_coordination_projection_digest": (
                "sha256:659b67b3f48e632337609d2c872c1d342650628921c6be52415a3fa9d73db1a1"
            ),
            "target_coordination_event_count": 1_125,
            "target_semantic_authority_digest": (
                "sha256:5e8d0afb732eaa5912512086a3d1ad288fa2de2b41b2b47462bf9006a6b7da5f"
            ),
            "target_frozen_base_authority_digest": (
                "sha256:b7e832646a9e8014f64f61c86530476d219ef10c25a467086cd73638cbbd034b"
            ),
            "event_suffix_length": 2,
            "coordination_semantic_changes": 0,
            "plan_revision_changes": 1,
            "evidence_node_changes": 1,
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
        if any(expected.get(name) != value for name, value in required.items()):
            errors.append("M19 live-catalog-inventory authority is not exact")
        repair = expected.get("accepted_source_repair")
        if (
            not isinstance(repair, Mapping)
            or set(repair.get("changed_paths", ()))
            != {
                "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
                "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
            }
            or repair.get("source_only") is not True
        ):
            errors.append("M19 catalog-inventory repair paths are not exact")
        blob_oids = repair.get("blob_oids") if isinstance(repair, Mapping) else None
        if not isinstance(blob_oids, Mapping) or set(blob_oids) != set(
            repair.get("changed_paths", ())
        ):
            errors.append("M19 catalog-inventory repair blobs are not exact")
        elif any(
            blob != "UNSEALED_REPAIR_BLOB" and len(str(blob)) != 40
            for blob in blob_oids.values()
        ):
            errors.append("M19 catalog-inventory repair blobs are not exact")
        expected_paths = {
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
        }
        if set(expected.get("bounded_control_plane_repair_paths", ())) != expected_paths:
            errors.append("M19 bounded source repair paths are not exact")
        try:
            if require_active_runtime:
                current_head = _git(root, "rev-parse", "HEAD")
                materializer._assert_m19_source_delta(
                    root,
                    {
                        "source_binding": {
                            "head": current_head,
                            "tree": _git(
                                root, "rev-parse", f"{current_head}^{{tree}}"
                            ),
                            "datasets_gitlink": _git(
                                root,
                                "rev-parse",
                                f"{current_head}:ipfs_datasets_py",
                            ),
                            "kit_gitlink": _git(
                                root, "rev-parse", f"{current_head}:ipfs_kit_py"
                            ),
                        }
                    },
                    expected,
                )
            else:
                _assert_m23_historical_source_handoff(
                    materializer, scheduler, seal, migration, root=root
                )
        except Exception as exc:
            errors.append(
                "M19 exact repair/source seal differs: "
                f"{type(exc).__name__}: {exc}"
            )
        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "20",
                "quack:127.0.0.1:24062",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                f"{target_root}/quack-owner",
                24_062,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M19 target/runtime binding is not exact")
        return errors
    except Exception as exc:
        return [
            "M19 source-binding authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]

def _m17_source_binding_successor_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M17's exact stopped-M16, source-only successor controls."""

    key = "source_binding_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m17_dependency_materializer",
            root / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M17 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m17_source_binding_authority()
        errors: list[str] = []
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("M17 source-binding authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M17 source-binding authority CID is not exact")

        target_root = (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m17"
        )
        target_store = f"{target_root}/control.duckdb"
        required = {
            "schema": (
                "sawm/post-commit-source-binding-successor-authorization@1"
            ),
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M17",
            "migration_kind": "post_commit_source_binding_repair",
            "supersession_mode": "post_commit_exact_source_binding_repair",
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m16/control.duckdb"
            ),
            "prior_control_store_sha256": (
                "11b837c173263c18f24e3d382ae1234771edc5e9ce03f56c8747be272fb8ad06"
            ),
            "prior_control_store_size": 43_528_192,
            "prior_coordination_store_sha256": (
                "ace915d5076a5d082a3377f669c9aef5f71216fe162d3e2e1c4f0aea7406fd83"
            ),
            "prior_coordination_store_size": 12_857_344,
            "prior_event_watermark": 209,
            "prior_event_prefix_sha256": (
                "4857d793597ca99b0df311f8c9f12733189b26488009052c438946935740e422"
            ),
            "prior_projection_cid": (
                "baguqeerakzd5xe55z5l6nifvwumea7unzobnhokoigbfkkdzg2chmrl4xa6q"
            ),
            "prior_semantic_authority_digest": (
                "sha256:c02c201a89884cfc990cae9e98f211bdfff7adbc18124281edac1d863aaba81b"
            ),
            "prior_frozen_base_authority_digest": (
                "sha256:24ace3d6006244240b71822a27a85a25ffe44dbb32d700d427479289d31127b2"
            ),
            "prior_append_surface_digest": (
                "sha256:c4bc499deb6f80ded467b105110a0c10849dfd62c821c5cec9270691bb631616"
            ),
            "prior_catalog_digest": (
                "sha256:3cc2e066bd4495e6efc0a3410a72c5df29242dcacfa94cd75d30f61c658539a7"
            ),
            "prior_coordination_projection_digest": (
                "sha256:3a1871c7bd682348897fd10da9a5515f671e1986beedff889c26d7dfe5a91773"
            ),
            "prior_coordination_event_count": 674,
            "prior_generation": 17,
            "prior_plan_revision": 17,
            "prior_server_id": "server:2703e3d2-1b55-4ffe-bdf2-ae26b269f41d",
            "prior_process_birth_id": "birth:fe721186f63c894b7aae4caea734c84d",
            "prior_startup_epoch": 1_788_006_651,
            "prior_started_at": "2026-08-29T12:30:51Z",
            "prior_stopped_at": "2026-08-29T12:42:59Z",
            "prior_stopped_status_projection_sha256": (
                "0ea511b7ed3e57c87fb0df72c9400daf01ab6172994f14d4681db7c642aa8908"
            ),
            "prior_stopped_status_projection_size": 2_410,
            "prior_migration_receipt_sha256": (
                "d72a8ff3cc2659a848b053915a6ca28aca57a2db9e8f885b63d691462af80031"
            ),
            "prior_migration_receipt_size": 9_956,
            "prior_migration_receipt_cid": (
                "sha256:d88eb8e31229fa295fb40785524a95896978c6c765a011e271f92c9bb3929619"
            ),
            "prior_source_binding_cid": (
                "sha256:97fab1ba85374b2bfd6763ed321c1425249263059c3529576420ee9a26d84d15"
            ),
            "prior_source_head": "f0150b9a064146b5b63d96efa1cb931995df6d3e",
            "prior_source_tree": "1e3890ab8696c03cbe53deec2ed3d3cfdb5b145d",
            "repair_source_commit": "5e113a8db7a2074d53c6525bcef5d8c20f1019f7",
            "repair_source_tree": "53cf375ea7047dd019d8996be86040a5a34c7f59",
            "target_store_id": target_store,
            "target_coordination_store_id": (
                f"{target_root}/control.coordination.duckdb"
            ),
            "target_runtime_root": target_root,
            "target_generation": 18,
            "target_quack_port": 24_060,
            "target_plan_revision": 18,
            "target_event_watermark": 211,
            "target_projection_cid": (
                "baguqeerat5ph3demwcyfmvtxjsq4dxdei5lf6xva2jhylwvgh2s4ewttscza"
            ),
            "target_coordination_projection_digest": (
                "sha256:3a1871c7bd682348897fd10da9a5515f671e1986beedff889c26d7dfe5a91773"
            ),
            "target_coordination_event_count": 674,
            "target_semantic_authority_digest": (
                "sha256:c02c201a89884cfc990cae9e98f211bdfff7adbc18124281edac1d863aaba81b"
            ),
            "target_frozen_base_authority_digest": (
                "sha256:24ace3d6006244240b71822a27a85a25ffe44dbb32d700d427479289d31127b2"
            ),
            "event_suffix_length": 2,
            "coordination_semantic_changes": 0,
            "plan_revision_changes": 1,
            "evidence_node_changes": 1,
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
        if any(expected.get(name) != value for name, value in required.items()):
            errors.append("M17 source-binding authority is not exact")
        expected_paths = {
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
        }
        if set(expected.get("bounded_control_plane_repair_paths", ())) != expected_paths:
            errors.append("M17 bounded source repair paths are not exact")
        try:
            current_head = _git(root, "rev-parse", "HEAD")
            if (
                "portal_completion_persistence_successor_materialization"
                in scheduler
            ):
                # M18 preserves M17's committed control endpoint as immutable
                # history.  Re-run the full M17 source-delta verifier against
                # that sealed endpoint rather than weakening the check or
                # incorrectly treating the later M18 source as M17 output.
                current_head = str(
                    materializer
                    ._expected_m18_portal_completion_persistence_authority()[
                        "prior_source_head"
                    ]
                )
            materializer._assert_m17_source_delta(
                root,
                {
                    "source_binding": {
                        "head": current_head,
                        "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                        "datasets_gitlink": _git(
                            root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                        ),
                        "kit_gitlink": _git(
                            root, "rev-parse", f"{current_head}:ipfs_kit_py"
                        ),
                    }
                },
                expected,
            )
        except Exception as exc:
            errors.append(
                "M17 exact repair/source seal differs: "
                f"{type(exc).__name__}: {exc}"
            )
        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "18",
                "quack:127.0.0.1:24060",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                f"{target_root}/quack-owner",
                24_060,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M17 target/runtime binding is not exact")
        return errors
    except Exception as exc:
        return [
            "M17 source-binding authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m16_accepted_source_retry_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
    *,
    root: Path = REPO_ROOT,
    require_active_runtime: bool = True,
) -> list[str]:
    """Check M16's exact accepted-source repair and two-task rearm controls.

    ``require_active_runtime`` is false only while a sealed successor is the
    active scheduler.  All immutable M16 authority, source, failure, and rearm
    checks remain mandatory; only the superseded M16 scheduler path/port tuple
    is then historical rather than active.
    """

    key = "accepted_source_retry_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m16_dependency_materializer",
            REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M16 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m16_accepted_source_retry_authority()
        errors: list[str] = []
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("M16 accepted-source retry authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M16 accepted-source retry authority CID is not exact")

        target_root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m16"
        target_store = f"{target_root}/control.duckdb"
        target_coordination = f"{target_root}/control.coordination.duckdb"
        required_scalars = {
            "schema": "sawm/portal-accepted-source-repair-authorization@1",
            "authorized": True,
            "authority": "operator_control_plane",
            "migration_revision": "SAWM-R2-M16",
            "migration_kind": "accepted_source_transition_repair",
            "supersession_mode": (
                "append_only_accepted_source_repair_and_exact_rearm"
            ),
            "target_runtime_root": target_root,
            "target_store_id": target_store,
            "target_coordination_store_id": target_coordination,
            "target_generation": 17,
            "target_quack_port": 24059,
            "target_plan_revision": 17,
            "target_event_watermark": 209,
            "target_projection_cid": (
                "baguqeerakzd5xe55z5l6nifvwumea7unzobnhokoigbfkkdzg2chmrl4xa6q"
            ),
            "target_coordination_projection_digest": (
                "sha256:3a1871c7bd682348897fd10da9a5515f671e1986beedff889c26d7dfe5a91773"
            ),
            "target_coordination_event_count": 674,
            "prior_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m15/control.duckdb"
            ),
            "prior_coordination_store_id": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m15/control.coordination.duckdb"
            ),
            "prior_control_store_sha256": (
                "d0254ec3a6c19ec7d28b6ce60abbaf409b9a20256b75686fd4527e34a521fffe"
            ),
            "prior_control_store_size": 43_528_192,
            "prior_coordination_store_sha256": (
                "00312cd51a65a73ac67bfca3ee93250316d119c21f469e12482245013f7b8b09"
            ),
            "prior_coordination_store_size": 12_857_344,
            "prior_event_watermark": 205,
            "prior_event_prefix_sha256": (
                "e8b1034cf3ae4f3ce409a66b500804436796460a605b9c11065ca5108544291c"
            ),
            "prior_projection_cid": (
                "baguqeerah65cdwuzzjaxjhq6d6tzrifa3xdygb2pgqxbdq62x7uv2qnayuda"
            ),
            "prior_semantic_authority_digest": (
                "sha256:6c74f6f168bb6c995081eaa6db8ad8b18591f27fd81b9a996eed7fb87c1499dc"
            ),
            "prior_frozen_base_authority_digest": (
                "sha256:b2583e5132a439e66455ef38c87c3e82deba05781162ad9b90732444c60f2234"
            ),
            "prior_append_surface_digest": (
                "sha256:623d5d3d5b9b0fb54d8c21552e5615e02639d697d42222028fe67b97c8505f89"
            ),
            "prior_catalog_digest": (
                "sha256:3cc2e066bd4495e6efc0a3410a72c5df29242dcacfa94cd75d30f61c658539a7"
            ),
            "prior_coordination_projection_digest": (
                "sha256:353c1c2bf8279204e70cc7d973f1095475af805ac55f1aaa53f96269c9d1999c"
            ),
            "prior_coordination_event_count": 672,
            "prior_generation": 16,
            "prior_plan_revision": 16,
            "prior_server_id": "server:8e8e84e9-9108-4c41-9a32-5019a71d0083",
            "prior_process_birth_id": "birth:cf7b2f64ef112e42ab6782ce22dcba3c",
            "prior_startup_epoch": 1_787_998_397,
            "prior_started_at": "2026-08-29T10:13:17Z",
            "prior_stopped_at": "2026-08-29T11:17:34Z",
            "prior_stopped_status_projection_sha256": (
                "4ea9e84bb571bbb3097e6c78a47444478455fb5ad5271549198548a01f3c6e5f"
            ),
            "prior_stopped_status_projection_size": 2_410,
            "prior_stopped_status_projection_path": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m15/quack-owner/quack-state-server.status.json"
            ),
            "prior_migration_receipt_path": (
                "data/agent_supervisor/semantic_addressed_world_model/"
                "run-r2-m15/migration-receipt.json"
            ),
            "prior_migration_receipt_sha256": (
                "77b688aa6ba7756f8050b21c8c77590ece6d3f3af19535f799976e00cdeee2c4"
            ),
            "prior_migration_receipt_size": 5_925,
            "prior_migration_receipt_cid": (
                "sha256:4d1fd0345a73ba84d9f471e139598190080998f99525f2bc9488a315a6b14f50"
            ),
            "prior_owner_marker_present": False,
            "prior_wals_absent": True,
            "prior_listener_present": False,
            "prior_pid_present": False,
            "prior_active_count": 0,
            "prior_source_tree": "f0c81b0c797fbf21aed94607eca3992c0873e2d1",
            "repair_source_commit": "5250fb379de06cf424dd06984e8d87154c107906",
            "repair_source_tree": "db8aa933218edb9bb9f361e588338dd98cfbbc61",
            "target_semantic_authority_digest": (
                "sha256:c02c201a89884cfc990cae9e98f211bdfff7adbc18124281edac1d863aaba81b"
            ),
            "target_frozen_base_authority_digest": (
                "sha256:09081077be61535c70e7d71d8921b16a7553b0c186822306015e1e12ce89acb7"
            ),
            "event_suffix_length": 4,
            "control_base_copied": True,
            "coordination_base_copied": True,
            "coordination_semantic_changes": 2,
            "plan_revision_changes": 1,
            "evidence_node_changes": 1,
            "task_revision_changes": 2,
            "task_status_changes": 2,
            "accepted_definition_changes": 0,
            "accepted_completion_changes": 0,
            "implementation_provider_invocations": 0,
            "effect_claim_changes": 0,
            "implementation_commit_changes": 0,
            "merge_attempt_changes": 0,
            "worker_self_approval": False,
        }
        if any(expected.get(name) != value for name, value in required_scalars.items()):
            errors.append("M16 accepted-source retry authority is not exact")

        expected_failures = {
            "SAWM-003": {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "task-claim-failure-settlement@1"
                ),
                "operation": "database_task_claim_failure",
                "failure_kind": "terminal_portal_bridge_error",
                "failure_payload_digest": (
                    "sha256:7123b86b2ce9030317660efa1bc0333d08013aac2b9217da96527bde6a9d8bd7"
                ),
                "task_cid": (
                    "sha256:9014d7a4538b8fac7caaf6d36904636fa2277f1ce530d8def46c20d138a24917"
                ),
                "attempt_id": "attempt:07e99e5cff954eaebe417f8c8074f0f1",
                "attempt_number": 1,
                "claim_id": "claim:16b5a4c742944050a76dc2f0ac10a015",
                "lease_id": "lease:df49803e97ca4565b9d04cd25cc69ae2",
                "owner_session_id": (
                    "embedded-store:ef0b99d66f5e7642811eb9bb9b60644b"
                ),
                "fencing_token": 1,
                "fence_epoch": 1,
                "provider_invocation_count": 0,
                "effect_claim_count": 0,
                "automatic_retry_admitted": False,
                "control_expected_status": "in_progress",
                "control_expected_revision": 3,
                "settlement_id": (
                    "baguqeeraafn4xwuix2i7ie3miynwpgtr45gvoxazzxyjvi2eyaxr456zbx7a"
                ),
            },
            "SAWM-004": {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "task-claim-failure-settlement@1"
                ),
                "operation": "database_task_claim_failure",
                "failure_kind": "terminal_portal_bridge_error",
                "failure_payload_digest": (
                    "sha256:5009ad7be944356f9beb9ad6a5f2554cbeb3bbee823f5d3c80e890a2a57f5e22"
                ),
                "task_cid": (
                    "sha256:7bac74f07d3755e90f51ce940ff591e69b8ac6dbe4c72a71f89dd22f76757463"
                ),
                "attempt_id": "attempt:8711f69f06204c6db7725ef1908d6d62",
                "attempt_number": 1,
                "claim_id": "claim:f64a9920c64b478db3a74b34fe8b46bd",
                "lease_id": "lease:9adca4a38a694b08bac9cdb4a176e75d",
                "owner_session_id": (
                    "embedded-store:ef0b99d66f5e7642811eb9bb9b60644b"
                ),
                "fencing_token": 1,
                "fence_epoch": 1,
                "provider_invocation_count": 0,
                "effect_claim_count": 0,
                "automatic_retry_admitted": False,
                "control_expected_status": "in_progress",
                "control_expected_revision": 3,
                "settlement_id": (
                    "baguqeerajv2zeh2d23fhrfdh7s2cthjbo3gb6djv5fs5pt6tm7et45lelgyq"
                ),
            },
        }
        if expected.get("failure_receipts") != expected_failures:
            errors.append("M16 frozen terminal failure receipts are not exact")
        task_rearms = expected.get("task_rearms")
        if (
            not isinstance(task_rearms, Mapping)
            or set(task_rearms) != set(expected_failures)
            or any(
                not isinstance(task_rearms.get(alias), Mapping)
                or task_rearms[alias].get("task_cid")
                != expected_failures[alias]["task_cid"]
                or task_rearms[alias].get("settlement_id")
                != expected_failures[alias]["settlement_id"]
                or task_rearms[alias].get("from_status") != "blocked"
                or task_rearms[alias].get("from_revision") != 4
                or task_rearms[alias].get("to_status") != "retrying"
                or task_rearms[alias].get("to_revision") != 5
                or task_rearms[alias].get("automatic_retry_admitted") is not False
                or task_rearms[alias].get("worker_self_approval") is not False
                for alias in expected_failures
            )
        ):
            errors.append("M16 exact two-task rearm authority is invalid")
        if expected.get("accepted_source_repair") != {
            "producer_path": (
                "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
                "implementation_daemon.py"
            ),
            "consumer_path": (
                "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
                "database_portal_bridge.py"
            ),
            "focused_test_path": (
                "test/api/test_agent_supervisor_database_portal_bridge.py"
            ),
            "repair_commit": "5250fb379de06cf424dd06984e8d87154c107906",
            "forward_top_level_binding": True,
            "nested_compatibility_fallback": True,
            "dual_binding_mismatch_rejected": True,
            "repository_authority_weakened": False,
        }:
            errors.append("M16 accepted-source adapter repair contract is not exact")

        expected_paths = {
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
            "test/api/test_agent_supervisor_database_portal_bridge.py",
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
        }
        if (
            expected.get("prior_source_head")
            != "8aafdcdfcd5a903695910f7f464fd6654943ccf2"
            or set(expected.get("bounded_control_plane_repair_paths", ()))
            != expected_paths
        ):
            errors.append("M16 bounded accepted-source repair paths are not exact")

        # Reuse the materializer's closed two-boundary source seal in this
        # read-only gate.  It binds the three accepted repair leaves to their
        # exact reviewed blobs, repair->successor to the nine operator files,
        # and prior->successor to the aggregate twelve-file delta.
        try:
            current_head = _git(root, "rev-parse", "HEAD")
            if "source_binding_successor_materialization" in scheduler:
                # M17 records the exact committed M16 control endpoint.  Use
                # that endpoint for the unchanged M16 source-seal verifier
                # when a later successor owns the active runtime namespace.
                current_head = str(
                    materializer._expected_m17_source_binding_authority()[
                        "prior_source_head"
                    ]
                )
            source_binding = {
                "head": current_head,
                "tree": _git(root, "rev-parse", f"{current_head}^{{tree}}"),
                "datasets_gitlink": _git(
                    root, "rev-parse", f"{current_head}:ipfs_datasets_py"
                ),
            }
            materializer._assert_m16_source_delta(
                root,
                {"source_binding": source_binding},
                expected,
            )
        except Exception as exc:
            errors.append(
                "M16 exact repair/source seal differs: "
                f"{type(exc).__name__}: {exc}"
            )

        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        if require_active_runtime and (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                target_store,
                "17",
                "quack:127.0.0.1:24059",
                f"{target_root}/events",
                f"{target_root}/registry",
                f"{target_root}/worktrees",
                target_store,
                f"{target_root}/quack-owner",
                24059,
            )
            or runtime
            != {
                "root": target_root,
                "state": f"{target_root}/state",
                "worktrees": f"{target_root}/worktrees",
                "merge_queue": f"{target_root}/merge-queue",
                "logs": f"{target_root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M16 target/runtime binding is not exact")
        return errors
    except Exception as exc:
        return [
            f"M16 accepted-source retry authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m15_runtime_root_rebind_errors(
    scheduler: Mapping[str, Any], seal: Mapping[str, Any], migration: Mapping[str, Any]
) -> list[str]:
    """Check M15 controls without opening or creating its runtime store."""

    key = "runtime_root_rebind_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m15_dependency_materializer",
            REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M15 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m15_runtime_root_rebind_authority()
        errors: list[str] = []
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("M15 runtime-root authority differs across controls")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("M15 runtime-root authority CID is not exact")
        root = expected["target_runtime_root"]
        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        runtime = scheduler.get("runtime_paths")
        if (
            (
                program.get("store_id"),
                program.get("store_generation"),
                program.get("quack_endpoint"),
                program.get("event_store_path"),
                program.get("runtime_registry_path"),
                program.get("worktree_root"),
                owner.get("database_path"),
                owner.get("state_dir"),
                owner.get("port"),
            )
            != (
                expected["target_store_id"],
                "16",
                "quack:127.0.0.1:24058",
                f"{root}/events",
                f"{root}/registry",
                f"{root}/worktrees",
                expected["target_store_id"],
                f"{root}/quack-owner",
                24058,
            )
            or runtime
            != {
                "root": root,
                "state": f"{root}/state",
                "worktrees": f"{root}/worktrees",
                "merge_queue": f"{root}/merge-queue",
                "logs": f"{root}/logs",
                "generated_runtime_artifacts_are_completion_authority": False,
            }
        ):
            errors.append("scheduler M15 target/runtime binding is not exact")
        blocker = expected.get("detached_launch_blocker", {})
        if (
            blocker.get("failure_kind") != "historical_runtime_pid_collision"
            or blocker.get("worker_dispatched") is not False
            or blocker.get("provider_dispatched") is not False
            or blocker.get("persisted_authoritative_failure_receipt_present")
            is not False
        ):
            errors.append("M15 detached-launch blocker evidence is invalid")
        return errors
    except Exception as exc:
        return [
            f"M15 runtime-root authority is unavailable: {type(exc).__name__}: {exc}"
        ]


def _m15_historical_authority_errors(
    scheduler: Mapping[str, Any], seal: Mapping[str, Any], migration: Mapping[str, Any]
) -> list[str]:
    """Verify immutable M15 controls without requiring its superseded runtime."""

    key = "runtime_root_rebind_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m15_historical_dependency_materializer",
            REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M15 historical materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m15_runtime_root_rebind_authority()
        errors: list[str] = []
        if scheduler.get(key) != expected or migration.get(key) != expected:
            errors.append("historical M15 runtime-root authority differs")
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            errors.append("historical M15 runtime-root authority CID is not exact")
        return errors
    except Exception as exc:
        return [
            f"historical M15 runtime-root authority is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ]


def _m14_stale_owner_restart_errors(
    scheduler: Mapping[str, Any], seal: Mapping[str, Any], migration: Mapping[str, Any]
) -> list[str]:
    """Check the newest M14 control without opening a runtime store."""
    key = "stale_owner_restart_successor_materialization"
    try:
        spec = importlib.util.spec_from_file_location(
            "sawm_m14_dependency_materializer",
            REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("M14 materializer cannot be loaded")
        materializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(materializer)
        expected = materializer._expected_m14_stale_owner_restart_authority()
        if scheduler.get(key) != expected or migration.get(key) != expected:
            return ["M14 stale-owner authority differs across controls"]
        if seal.get(f"{key}_cid") != materializer._identity(expected):
            return ["M14 stale-owner authority CID is not exact"]
        program = scheduler.get("database_program", {})
        owner = scheduler.get("quack_owner", {})
        if (program.get("store_id"), program.get("store_generation"), program.get("quack_endpoint"), owner.get("database_path"), owner.get("port")) != (expected["target_store_id"], "15", "quack:127.0.0.1:24057", expected["target_store_id"], 24057):
            return ["scheduler M14 target binding is not exact"]
        return []
    except Exception as exc:
        return [f"M14 stale-owner authority is unavailable: {type(exc).__name__}: {exc}"]


def _m13_quack_refresh_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify M13's source-only initial-refresh lifecycle repair."""

    errors: list[str] = []
    authority_key = "quack_refresh_successor_materialization"
    cid_key = "quack_refresh_successor_materialization_cid"
    if authority_key not in scheduler:
        errors.append("M13 Quack-refresh authority key is absent from scheduler")
    if authority_key not in migration:
        errors.append("M13 Quack-refresh authority key is absent from inventory")
    if cid_key not in seal:
        errors.append("M13 Quack-refresh authority CID key is absent from seal")
    configured = scheduler[authority_key] if authority_key in scheduler else None
    inventoried = migration[authority_key] if authority_key in migration else None
    if type(configured) is not dict or configured != inventoried:
        errors.append("M13 Quack-refresh authority differs across controls")
        configured = {}
    expected_cid = (
        "sha256:2a6be82b1d160f07ee192b6093eb8fae3518ab2df713155bce98f7fb280e9920"
    )
    observed_cid = "sha256:" + hashlib.sha256(
        _canonical_json(configured)
    ).hexdigest()
    if seal.get(cid_key) != expected_cid or observed_cid != expected_cid:
        errors.append("M13 Quack-refresh authority CID is not exact")

    prior_root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m12"
    root = "data/agent_supervisor/semantic_addressed_world_model/run-r2-m13"
    expected_subset = {
        "schema": "sawm/quack-initial-refresh-repair-authorization@1",
        "authorized": True,
        "authority": "operator_control_plane",
        "migration_revision": "SAWM-R2-M13",
        "migration_kind": "quack_initial_refresh_rebind_repair",
        "supersession_mode": "source_only_quack_initial_refresh_lifecycle_repair",
        "prior_store_id": f"{prior_root}/control.duckdb",
        "prior_publication_control_store_sha256": (
            "41874cc79b6e7056f32401f5a6cbf372a34f7b19f7626e8ef031d401c3a57c5e"
        ),
        "prior_control_store_sha256": (
            "9b09d7dcb2079963a20c6e22fa0a5ff1e0e6833a2a8ae027ac36a973adc037f2"
        ),
        "prior_control_store_size": 45_101_056,
        "prior_control_wal_id": f"{prior_root}/control.duckdb.wal",
        "prior_control_wal_present": False,
        "prior_coordination_store_id": f"{prior_root}/control.coordination.duckdb",
        "prior_coordination_store_sha256": (
            "113f5b629317effeada5b8b81a0f8b8aaf4322ab36b6c111d830c5d9095eea55"
        ),
        "prior_coordination_store_size": 12_857_344,
        "prior_coordination_wal_id": f"{prior_root}/control.coordination.duckdb.wal",
        "prior_coordination_wal_present": False,
        "prior_coordination_event_count": 224,
        "prior_coordination_projection_digest": (
            "sha256:358cd0667be10fb125476ac090db3cda9aec9b8d0876a2654de1ce5aa531c59f"
        ),
        "prior_event_watermark": 189,
        "prior_event_prefix_sha256": (
            "c9f83218a7a6bfdf145c1733c4d51e96fb0ba97c421604602e541f4ab04c84e0"
        ),
        "prior_projection_cid": (
            "baguqeeragb5uufggmw6glss2ttbubyb2aixhfio6cmmit6w4njiuozj3csmq"
        ),
        "prior_source_head": "609ce9ad4d2934d4d2b5400cb8c8bad56a277a6e",
        "prior_source_tree": "19522ae2517b2427959e44a396c311d285d319cb",
        "prior_source_binding_cid": (
            "sha256:1fa02a9d8044fa9a5acd0738d43574b32e0da5295eeae422e1b1c86a7f603379"
        ),
        "prior_database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
        "prior_generation": 13,
        "prior_plan_revision": 13,
        "prior_materialization_receipt_path": f"{prior_root}/migration-receipt.json",
        "prior_materialization_receipt_cid": (
            "sha256:d17b7cea163f6834ccd7b42b30b1e19109550802d83eb9806ebab11e9d55e196"
        ),
        "prior_materialization_receipt_file_sha256": (
            "fd736f7cf349efee968133fd970e33dfc1ff7f8ac8dbf22c98618ce2a0ac0185"
        ),
        "prior_materialization_receipt_precedes_failed_start_checkpoint": True,
        "prior_owner_status_path": (
            f"{prior_root}/quack-owner/quack-state-server.status.json"
        ),
        "prior_owner_status_present": False,
        "prior_owner_directory_empty": True,
        "prior_read_replica_store_id": f"{prior_root}/control.read-replica.duckdb",
        "prior_read_replica_store_sha256": (
            "9b09d7dcb2079963a20c6e22fa0a5ff1e0e6833a2a8ae027ac36a973adc037f2"
        ),
        "prior_read_replica_store_size": 45_101_056,
        "prior_read_replica_is_distinct_file": True,
        "prior_read_replica_is_authority": False,
        "prior_semantic_authority_digest": (
            "sha256:239db939a5e7af260f334b325c4ec075a2faf18f01b683448625647e0962d36c"
        ),
        "prior_frozen_base_authority_digest": (
            "sha256:12b25f50a7c5d3b1020b7c1f86412fa863a6c00027ee780ecd1f78c88dfa7a95"
        ),
        "prior_append_surface_digest": (
            "sha256:c5eebae5486bf281ebfaa03f369eb114ceef707d446afe45a929a27d250363aa"
        ),
        "prior_catalog_digest": (
            "sha256:3cc2e066bd4495e6efc0a3410a72c5df29242dcacfa94cd75d30f61c658539a7"
        ),
        "target_store_id": f"{root}/control.duckdb",
        "target_coordination_store_id": f"{root}/control.coordination.duckdb",
        "target_generation": 14,
        "target_quack_port": 24_056,
        "target_plan_revision": 14,
        "target_event_watermark": 191,
        "target_coordination_event_count": 224,
        "target_coordination_projection_digest": (
            "sha256:358cd0667be10fb125476ac090db3cda9aec9b8d0876a2654de1ce5aa531c59f"
        ),
        "target_semantic_authority_digest": (
            "sha256:239db939a5e7af260f334b325c4ec075a2faf18f01b683448625647e0962d36c"
        ),
        "target_frozen_base_authority_digest": (
            "sha256:12b25f50a7c5d3b1020b7c1f86412fa863a6c00027ee780ecd1f78c88dfa7a95"
        ),
        "event_suffix_length": 2,
        "control_recorded_at": "2026-08-29T00:00:00Z",
        "control_base_copied": True,
        "coordination_base_copied": True,
        "prior_control_wal_copied": False,
        "prior_coordination_wal_copied": False,
        "prior_read_replica_copied": False,
        "prior_owner_status_copied": False,
        "prior_failed_start_artifacts_preserved": True,
        "prior_control_store_mutated": False,
        "prior_coordination_store_mutated": False,
        "coordination_semantic_changes": 0,
        "plan_revision_changes": 1,
        "evidence_node_changes": 1,
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
    for field, expected in expected_subset.items():
        if configured.get(field) != expected:
            errors.append(f"M13 Quack-refresh {field} is not exact")
    if configured.get("target_projection_cid") != (
        "baguqeeraqpkofyd3pnjnaqiwwefz7pkubim7kaklbp65puqu47ckb2vdmtxa"
    ):
        errors.append("M13 target projection CID is not exact")

    repair = configured.get("quack_refresh_repair")
    expected_repair = {
        "operator_path": "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "refresh_probe_parameter": True,
        "initial_refresh_probe": False,
        "post_identity_refresh_probe": True,
        "mutation_refresh_probe": True,
        "unprobed_refresh_live": False,
        "target_port": 24_056,
        "generation_changed": False,
        "transport_protocol_changed": False,
    }
    if repair != expected_repair:
        errors.append("M13 Quack refresh lifecycle repair is not exact")

    failure = configured.get("failed_quack_start")
    if not isinstance(failure, Mapping):
        failure = {}
        errors.append("M13 frozen Quack-start failure is absent")
    if (
        configured.get("failed_quack_start_cid")
        != "sha256:" + hashlib.sha256(_canonical_json(failure)).hexdigest()
        or failure.get("schema") != "sawm/quack-initial-refresh-failure@1"
        or failure.get("attempt") != "SAWM-R2-M12-QUACK-START-A1"
        or failure.get("phase") != "pre_identity_initial_replica_bind"
        or failure.get("failure_kind")
        != "initial_quack_replica_bind_conflict"
        or failure.get("owner_status_created") is not False
        or failure.get("control_wal_created") is not False
        or failure.get("coordination_wal_created") is not False
        or failure.get("task_revision_changes") != 0
        or failure.get("task_status_changes") != 0
        or failure.get("provider_invocations") != 0
        or failure.get("worker_self_approval") is not False
    ):
        errors.append("M13 frozen Quack-start failure is not exact")

    live = configured.get("live_task_projection")
    expected_live = {
        "schema": "sawm/live-task-projection-expectation@1",
        "task_count": 45,
        "task_revision_count": 1,
        "operator_task_alias": "SAWM-000",
        "operator_status": "completed",
        "operator_revision": 2,
        "candidate_task_alias": "SAWM-001",
        "candidate_task_cid": (
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
        "prior_candidate_status": "retrying",
        "prior_candidate_revision": 16,
        "target_candidate_status": "retrying",
        "target_candidate_revision": 16,
        "target_candidate_completion_receipt": {
            "operation": "operator_control_plane_repair",
            "settlement_id": (
                "baguqeerafg6ci2g5mrq7ilfpa2qvinghhjqq2hdzmfprut4sbdiqosoxi5pq"
            ),
        },
        "remaining_task_status": "todo",
        "remaining_task_revision": 2,
    }
    if live != expected_live:
        errors.append("M13 unchanged live task projection is not exact")
    if configured.get("provider_strategy") != {
        "route_changed": False,
        "capability_probe_required": True,
        "provider_result_is_completion_authority": False,
    }:
        errors.append("M13 provider strategy is not exact")

    required_repair_paths = {
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
    }
    protected = set(scheduler.get("protected_paths") or ())
    capsule = scheduler.get("configured_board_live_capsule")
    capsule_controls = set(
        capsule.get("control_paths") or () if isinstance(capsule, Mapping) else ()
    )
    if (
        set(configured.get("bounded_control_plane_repair_paths") or ())
        != required_repair_paths
        or len(configured.get("bounded_control_plane_repair_paths") or ()) != 9
        or not required_repair_paths.issubset(CONTROL_PATHS)
        or not required_repair_paths.issubset(protected)
        or not required_repair_paths.issubset(capsule_controls)
    ):
        errors.append("M13 bounded repair paths are not exact protected controls")

    expected_program = {
        "authority_mode": "quack",
        "endpoint_secret_handle": "env://SAWM_QUACK_TOKEN",
        "event_store_path": f"{root}/events",
        "explicit_legacy": False,
        "export_profile": "semantic-addressed-world-model-r2",
        "failover_policy": "fail_closed",
        "quack_endpoint": "quack:127.0.0.1:24056",
        "runtime_registry_path": f"{root}/registry",
        "schema_revision": "datasets-authoritative-operational-v1",
        "store_generation": "14",
        "store_id": f"{root}/control.duckdb",
        "task_source_kind": "duckdb",
        "worktree_root": f"{root}/worktrees",
    }
    if scheduler.get("database_program") != expected_program:
        errors.append("scheduler M13 execution authority is not exact")
    owner = scheduler.get("quack_owner")
    expected_owner_fields = {
        "automatic_direct_file_fallback": False,
        "database_path": f"{root}/control.duckdb",
        "host": "127.0.0.1",
        "live_transport_query_required": True,
        "migration_profile": "datasets-authoritative-operational-v1",
        "port": 24_056,
        "repository_id": "ipfs_accelerate_py:semantic-addressed-world-model-v1",
        "secret_handle": "env://SAWM_QUACK_TOKEN",
        "start_once_before_scheduler": True,
        "state_dir": f"{root}/quack-owner",
        "status_file_alone_is_ready": False,
        "store_id": f"{root}/control.duckdb",
    }
    if not isinstance(owner, Mapping) or any(
        owner.get(field) != expected
        for field, expected in expected_owner_fields.items()
    ):
        errors.append("scheduler M13 Quack owner authority is not exact")
    expected_history = {
        "authority": False,
        "catalog_path": f"{root}/ducklake/history.ducklake",
        "completion_prerequisite": False,
        "data_path": f"{root}/ducklake/data",
        "enabled_when_locally_available": True,
        "install_or_network_forbidden": True,
        "load_local_only": True,
        "receipt_path": f"{root}/ducklake-history-receipt.json",
        "scheduling_prerequisite": False,
        "typed_unavailability_is_permitted": True,
    }
    if scheduler.get("ducklake_history_projection") != expected_history:
        errors.append("scheduler M13 DuckLake projection is not exact")
    expected_runtime = {
        "root": root,
        "state": f"{root}/state",
        "worktrees": f"{root}/worktrees",
        "merge_queue": f"{root}/merge-queue",
        "logs": f"{root}/logs",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    if scheduler.get("runtime_paths") != expected_runtime:
        errors.append("scheduler M13 runtime paths are not exact")

    operator_source = (
        REPO_ROOT / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"
    ).read_text(encoding="utf-8")
    for required_source in (
        "def refresh(",
        "probe: bool = True",
        "return self.refresh(connection, probe=False)",
        "server.transport.refresh(server._connection, probe=True)",
        "refresh_replica=lambda: transport.refresh(connection)",
    ):
        if required_source not in operator_source:
            errors.append(f"M13 operator source omits {required_source}")
    return errors


def _stable_regular_evidence(path: Path, *, maximum: int) -> dict[str, Any]:
    """Hash one no-follow regular file and reject identity changes mid-read."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise ValueError(f"{path} is not stable regular evidence")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_gid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if offset != before.st_size or identity(before) != identity(after):
        raise ValueError(f"{path} changed while it was hashed")
    return {
        "sha256": "sha256:" + digest.hexdigest(),
        "size": before.st_size,
    }


def _expected_validation_runtime() -> dict[str, Any]:
    """Return the closed M6 validation image declaration.

    The two scheduler/seal copies are intentionally insufficient on their own:
    an operator editing both must not be able to redirect the launch gate to a
    different root.  These exact external bytes were independently audited and
    are immutable, root-owned bootstrap inputs rather than implementation
    outputs or task-completion authority.
    """

    deployments: dict[str, dict[str, Any]] = {}
    deployment_properties = {
        "delta_deployment": (
            "ipfs-accelerate-aseh-wheel-delta-deployment@1",
            "sha256:c7d197d7cce26a2cd28a461220b7b7b848f77650265b84447cafb8e66505b18d",
            4526,
        ),
        "base_deployment": (
            "ipfs-accelerate-legal-validation-deployment@1",
            "sha256:566c3806a52b07aa72ee3b6bb4e604e00cde502ac771d3b87fd1ce7fd84d949a",
            28058,
        ),
    }
    for name, root in _VALIDATION_RUNTIME_ROOTS.items():
        schema, body_sha256, entry_count = deployment_properties[name]
        deployments[name] = {
            "root": str(root),
            "schema": schema,
            "deployment_body_sha256": body_sha256,
            "manifest_entry_count": entry_count,
            "artifacts": [
                {
                    "path": str(root / relative),
                    "sha256": "sha256:" + digest,
                }
                for relative, digest in _VALIDATION_RUNTIME_ARTIFACTS[name]
            ],
        }
    return {
        "schema": "semantic-addressed-world-model/validation-runtime@1",
        "python_executable": "/usr/bin/python3.12",
        "python_executable_sha256": (
            "sha256:1a301bb1763139d48ae638d97b11edf56de6cd185e1b054eae6dc28c271c0c5f"
        ),
        "python_version": "3.12.3",
        "pythonpath_entries": list(_VALIDATION_PYTHONPATH),
        "ordered_pythonpath_sha256": (
            "sha256:b33b316a356490f3f80be252ccc22ae94163c0c398a1f77548c4f0d10c5bc707"
        ),
        "required_modules": ["packaging", "pytest"],
        **deployments,
        "project_dependency_preflight_qualification": {
            "scope": "all_nonoperator_operational_validations",
            "task_alias_first": "SAWM-001",
            "task_alias_last": "SAWM-044",
            "validation_command_count": 46,
            "requirements_count": 43,
            "passed": True,
            "reason": "approved_validation_environment_satisfies_project_dependencies",
            "missing_count": 0,
            "incompatible_count": 0,
            "invalid_count": 0,
            "source_receipt_id": (
                "9f5beaff8aa5fefb28576d29d5cbc1b5332e8160f03f9cc7ac698edf9872ae10"
            ),
            "path_sensitive": True,
            "runtime_authority": False,
            "recompute_required": True,
        },
        "policy": {
            "required_owner_uid": 0,
            "writable_entries_allowed": False,
            "symlinks_allowed": False,
            "pyc_allowed": False,
            "network_allowed": False,
            "installer_allowed": False,
            "ambient_user_site_allowed": False,
            "automatic_install_allowed": False,
        },
    }


def _validation_runtime_contract_errors(
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Validate the two closed declarations against independent constants."""

    errors: list[str] = []
    configured = scheduler.get("validation_runtime")
    sealed = seal.get("validation_runtime")
    expected = _expected_validation_runtime()
    if type(configured) is not dict:
        errors.append("scheduler validation_runtime is not an exact object")
        configured = {}
    if type(sealed) is not dict:
        errors.append("dependency seal validation_runtime is not an exact object")
        sealed = {}
    if configured != sealed:
        errors.append("scheduler validation_runtime differs from the dependency seal")
    if configured != expected:
        errors.append("validation_runtime differs from the exact M6 immutable closure")
    return dict(configured), errors


def _root_owned_regular_digest(
    path: Path,
    *,
    expected_mode: int,
    expected_size: int | None = None,
) -> tuple[str, int]:
    """Rehash one exact root-owned, no-follow immutable regular file."""

    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError(f"{path} is not an exact absolute no-symlink path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != 0
            or before.st_gid != 0
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != expected_mode
            or before.st_mode & 0o022
            or (expected_size is not None and before.st_size != expected_size)
        ):
            raise ValueError(f"{path} ownership, mode, link, or size differs")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_gid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
    if offset != before.st_size or identity(before) != identity(after):
        raise ValueError(f"{path} changed while it was rehashed")
    return digest.hexdigest(), before.st_size


def _immutable_payload_errors(
    name: str,
    deployment: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Rehash one complete deployment manifest and its exact payload tree."""

    errors: list[str] = []
    root = _VALIDATION_RUNTIME_ROOTS[name]
    try:
        root_stat = root.lstat()
        if (
            root.resolve(strict=True) != root
            or not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != 0
            or root_stat.st_gid != 0
            or stat.S_IMODE(root_stat.st_mode) != 0o555
            or root_stat.st_mode & 0o022
        ):
            raise ValueError("deployment root is not exact root-owned mode 0555")
    except (OSError, ValueError) as exc:
        return {}, [f"{name}: {type(exc).__name__}: {exc}"]

    artifact_rows = deployment.get("artifacts")
    if not isinstance(artifact_rows, list):
        return {}, [f"{name}: artifact inventory is absent"]
    artifact_digests: dict[str, str] = {}
    for item in artifact_rows:
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            errors.append(f"{name}: artifact fields are noncanonical")
            continue
        path = Path(str(item.get("path") or ""))
        if path.parent != root:
            errors.append(f"{name}: artifact escapes the exact deployment root")
            continue
        try:
            digest, _size = _root_owned_regular_digest(path, expected_mode=0o444)
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {path.name}: {type(exc).__name__}: {exc}")
            continue
        artifact_digests[path.name] = digest
        if item.get("sha256") != "sha256:" + digest:
            errors.append(f"{name}: {path.name} differs from its artifact pin")

    manifest_path = root / "PAYLOAD_MANIFEST.jsonl"
    try:
        manifest_raw = manifest_path.read_bytes()
    except OSError as exc:
        return {}, errors + [f"{name}: manifest unreadable: {type(exc).__name__}: {exc}"]
    manifest_rows: list[dict[str, Any]] = []
    manifest_paths: set[str] = set()
    previous = ""
    for ordinal, raw_line in enumerate(manifest_raw.splitlines(), 1):
        try:
            row = json.loads(raw_line, object_pairs_hook=_duplicates)
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{name}: manifest row {ordinal} is invalid: {exc}")
            continue
        if type(row) is not dict or row.get("type") not in {"file", "directory"}:
            errors.append(f"{name}: manifest row {ordinal} has invalid type")
            continue
        expected_fields = (
            {"bytes", "mode", "path", "sha256", "type"}
            if row["type"] == "file"
            else {"bytes", "mode", "path", "type"}
        )
        relative = str(row.get("path") or "")
        pure = Path(relative)
        canonical_line = _canonical_json(row)
        if (
            set(row) != expected_fields
            or canonical_line != raw_line
            or not relative
            or pure.is_absolute()
            or relative != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or relative <= previous
            or relative in manifest_paths
        ):
            errors.append(f"{name}: manifest row {ordinal} is noncanonical")
            continue
        previous = relative
        manifest_paths.add(relative)
        manifest_rows.append(row)

    expected_count = deployment.get("manifest_entry_count")
    if len(manifest_rows) != expected_count:
        errors.append(
            f"{name}: manifest entry count {len(manifest_rows)} != {expected_count}"
        )

    file_count = 0
    total_bytes = 0
    for row in manifest_rows:
        relative = str(row["path"])
        path = root / relative
        mode_text = row.get("mode")
        if mode_text not in {"0444", "0555"}:
            errors.append(f"{name}: {relative} has a non-immutable manifest mode")
            continue
        expected_mode = int(str(mode_text), 8)
        if row["type"] == "directory":
            try:
                observed = path.lstat()
                if (
                    path.resolve(strict=True) != path
                    or not stat.S_ISDIR(observed.st_mode)
                    or observed.st_uid != 0
                    or observed.st_gid != 0
                    or stat.S_IMODE(observed.st_mode) != expected_mode
                    or observed.st_mode & 0o022
                    or row.get("bytes") != 0
                ):
                    raise ValueError("directory identity differs")
            except (OSError, ValueError) as exc:
                errors.append(f"{name}: {relative}: {type(exc).__name__}: {exc}")
            continue
        if relative.endswith((".pyc", ".pyo")):
            errors.append(f"{name}: compiled Python payload is forbidden: {relative}")
        try:
            expected_size = int(row.get("bytes"))
            digest, observed_size = _root_owned_regular_digest(
                path,
                expected_mode=expected_mode,
                expected_size=expected_size,
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {relative}: {type(exc).__name__}: {exc}")
            continue
        if row.get("sha256") != digest:
            errors.append(f"{name}: payload digest differs: {relative}")
        file_count += 1
        total_bytes += observed_size

    actual_paths: set[str] = set()
    try:
        for directory, child_directories, filenames in os.walk(root, followlinks=False):
            base = Path(directory)
            for child in (*child_directories, *filenames):
                actual_paths.add((base / child).relative_to(root).as_posix())
    except OSError as exc:
        errors.append(f"{name}: payload walk failed: {type(exc).__name__}: {exc}")
    allowed_extras = {"DEPLOYMENT.json", "PAYLOAD_MANIFEST.jsonl"}
    unexpected = sorted(actual_paths - manifest_paths - allowed_extras)
    absent = sorted(manifest_paths - actual_paths)
    if unexpected or absent:
        errors.append(
            f"{name}: payload inventory differs: unexpected={unexpected[:8]} "
            f"absent={absent[:8]}"
        )

    return {
        "root": str(root),
        "manifest_sha256": "sha256:" + hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_entry_count": len(manifest_rows),
        "file_count": file_count,
        "bytes": total_bytes,
        "artifact_count": len(artifact_digests),
    }, errors


def _validation_runtime_import_probe(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Import the exact reviewed module contract under ``-I -S -B``."""

    delta = Path(str(runtime["delta_deployment"]["root"]))
    import_contract = _load(delta / "IMPORT_CONTRACT.json")
    modules = import_contract.get("modules")
    if (
        import_contract.get("schema")
        != "ipfs-accelerate-aseh-wheel-delta-import-contract@1"
        or not isinstance(modules, list)
        or len(modules) != 39
        or len(set(modules)) != len(modules)
        or any(type(item) is not str or not item for item in modules)
    ):
        return {"valid": False, "error": "sealed import contract is noncanonical"}
    source = r'''
import importlib, json, os, sys
paths = json.loads(sys.argv.pop(1))
modules = json.loads(sys.argv.pop(1))
sys.path[:0] = paths
observed = []
for name in modules:
    module = importlib.import_module(name)
    origin = str(getattr(module, "__file__", "") or "")
    if not origin:
        spec = getattr(module, "__spec__", None)
        origin = str(getattr(spec, "origin", "") or "")
    if not origin or origin in {"built-in", "frozen"}:
        raise RuntimeError("external module has no exact origin: " + name)
    resolved = os.path.realpath(origin)
    if not any(
        resolved == root or resolved.startswith(root + os.sep)
        for root in paths
    ):
        raise RuntimeError("module escaped ordered closure: " + name)
    observed.append({"module": name, "origin_root": next(
        index for index, root in enumerate(paths)
        if resolved == root or resolved.startswith(root + os.sep)
    )})
print(json.dumps({"valid": True, "imports": observed}, sort_keys=True,
                 separators=(",", ":")))
'''
    paths = list(runtime["pythonpath_entries"])
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent/ipfs-accelerate-sawm-validation",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    try:
        completed = subprocess.run(
            [
                str(runtime["python_executable"]),
                "-I",
                "-S",
                "-B",
                "-c",
                source,
                json.dumps(paths, separators=(",", ":")),
                json.dumps(modules, separators=(",", ":")),
            ],
            cwd="/",
            env=environment,
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {
            "valid": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-1000:],
            "stderr": completed.stderr[-2000:],
        }
    try:
        result = json.loads(completed.stdout, object_pairs_hook=_duplicates)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"valid": False, "error": f"invalid import probe output: {exc}"}
    if type(result) is not dict:
        return {"valid": False, "error": "import probe output is not an object"}
    result["stderr"] = completed.stderr[-1000:]
    result["module_count"] = len(modules)
    return result


def _historical_and_operational_validation_commands(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Read immutable Markdown and derive the authorized M6 command view."""

    errors: list[str] = []
    text = (
        root / "docs/architecture/semantic_addressed_world_model.todo.md"
    ).read_text(encoding="utf-8")
    raw_lists = re.findall(
        r"^- Validation commands JSON:\s*(\[[^\n]*\])\s*$",
        text,
        flags=re.MULTILINE,
    )
    historical: list[list[str]] = []
    for ordinal, raw in enumerate(raw_lists):
        try:
            value = json.loads(raw, object_pairs_hook=_duplicates)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"historical validation {ordinal} is invalid JSON: {exc}")
            continue
        if not isinstance(value, list) or any(type(item) is not str for item in value):
            errors.append(f"historical validation {ordinal} is not a string list")
            continue
        historical.append(value)
    if len(historical) != 45:
        errors.append(f"historical validation card count {len(historical)} != 45")
        return [], [], errors
    historical_flat = [command for commands in historical for command in commands]
    historical_prefix = (
        "PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. "
        f"{_HISTORICAL_VALIDATION_PYTHON} "
    )
    if (
        len(historical_flat) != 48
        or any(not command.startswith(historical_prefix) for command in historical_flat)
    ):
        errors.append("immutable Markdown validation commands differ from M1 definitions")

    nonoperator = [command for commands in historical[1:] for command in commands]
    replacements = sum(command.count(_HISTORICAL_VALIDATION_PYTHON) for command in nonoperator)
    operational = [
        command.replace(
            _HISTORICAL_VALIDATION_PYTHON,
            _OPERATIONAL_VALIDATION_PYTHON,
        )
        for command in nonoperator
    ]
    operational_prefix = (
        "PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. "
        f"{_OPERATIONAL_VALIDATION_PYTHON} "
    )
    if (
        len(operational) != 46
        or replacements != 46
        or any(not command.startswith(operational_prefix) for command in operational)
        or any(_HISTORICAL_VALIDATION_PYTHON in command for command in operational)
    ):
        errors.append("M6 operational validation replacement is not exact")
    return historical_flat, operational, errors


def _validation_project_dependency_probe(
    root: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute dependency admission for all 44 operational tasks."""

    _historical, commands, command_errors = (
        _historical_and_operational_validation_commands(root)
    )
    if command_errors:
        return {"valid": False, "errors": command_errors}
    environment = {
        "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON": runtime["python_executable"],
        "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHONPATH": os.pathsep.join(
            runtime["pythonpath_entries"]
        ),
        "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON_MODULES": ",".join(
            runtime["required_modules"]
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PIP_NO_INDEX": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    try:
        from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
            preflight_validation_project_dependencies,
        )

        receipt = preflight_validation_project_dependencies(
            root,
            commands,
            environment=environment,
        )
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    projects = receipt.get("projects") if isinstance(receipt, Mapping) else None
    project = projects[0] if isinstance(projects, list) and len(projects) == 1 else {}
    valid = (
        receipt.get("passed") is True
        and receipt.get("reason")
        == "approved_validation_environment_satisfies_project_dependencies"
        and receipt.get("validation_command_count") == 46
        and receipt.get("validation_roots") == [""]
        and receipt.get("missing_requirements") == []
        and receipt.get("incompatible_requirements") == []
        and receipt.get("invalid_requirements") == []
        and receipt.get("invalid_commands") == []
        and isinstance(project, Mapping)
        and project.get("requirement_count") == 43
    )
    return {
        "valid": valid,
        "passed": receipt.get("passed"),
        "reason": receipt.get("reason"),
        "receipt_id": receipt.get("receipt_id"),
        "path_sensitive": True,
        "validation_command_count": receipt.get("validation_command_count"),
        "requirements_count": project.get("requirement_count")
        if isinstance(project, Mapping)
        else None,
        "missing_count": len(receipt.get("missing_requirements") or ()),
        "incompatible_count": len(receipt.get("incompatible_requirements") or ()),
        "invalid_count": len(receipt.get("invalid_requirements") or ())
        + len(receipt.get("invalid_commands") or ()),
        "failure": None if valid else receipt,
    }


def _validation_runtime_closure(
    root: Path,
    scheduler: Mapping[str, Any],
    seal: Mapping[str, Any],
    *,
    rehash_payloads: bool = True,
    probe_imports: bool = True,
    probe_dependencies: bool = True,
) -> dict[str, Any]:
    """Return a bounded fail-closed qualification of the exact M6 runtime."""

    runtime, errors = _validation_runtime_contract_errors(scheduler, seal)
    payloads: dict[str, Any] = {}
    if not errors:
        try:
            python_digest, _python_size = _root_owned_regular_digest(
                Path(runtime["python_executable"]),
                expected_mode=0o755,
            )
            if runtime.get("python_executable_sha256") != "sha256:" + python_digest:
                errors.append("validation interpreter differs from its exact pin")
        except (OSError, ValueError) as exc:
            errors.append(f"validation interpreter is invalid: {type(exc).__name__}: {exc}")
    if not errors and rehash_payloads:
        for name in ("delta_deployment", "base_deployment"):
            summary, deployment_errors = _immutable_payload_errors(
                name,
                runtime[name],
            )
            payloads[name] = summary
            errors.extend(deployment_errors)
    import_probe: dict[str, Any] = {"valid": None, "skipped": True}
    if not errors and probe_imports:
        import_probe = _validation_runtime_import_probe(runtime)
        if import_probe.get("valid") is not True:
            errors.append("validation runtime import probe failed")
    dependency_probe: dict[str, Any] = {"valid": None, "skipped": True}
    if not errors and probe_dependencies:
        dependency_probe = _validation_project_dependency_probe(root, runtime)
        if dependency_probe.get("valid") is not True:
            errors.append("validation project dependency preflight failed")
    return {
        "valid": not errors,
        "errors": errors,
        "runtime_schema": runtime.get("schema"),
        "python_executable": runtime.get("python_executable"),
        "pythonpath_entries": runtime.get("pythonpath_entries"),
        "payloads": payloads,
        "import_probe": import_probe,
        "project_dependency_preflight": dependency_probe,
    }


def _copy_pinned_regular(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    """Copy one exact no-follow source into an exclusive qualification path."""

    if _SHA256.fullmatch(expected_sha256) is None or expected_size <= 0:
        raise ValueError("extension projection pin is invalid")
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    target_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd = os.open(source, source_flags)
    target_fd = -1
    try:
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
        ):
            raise ValueError(f"{source} is not the pinned regular source")
        target_fd = os.open(destination, target_flags, 0o600)
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                source_fd,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise ValueError("extension projection write failed")
                view = view[written:]
            offset += len(block)
        after = os.fstat(source_fd)
        if (
            offset != expected_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
                before.st_gid,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_gid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or "sha256:" + digest.hexdigest() != expected_sha256
        ):
            raise ValueError(f"{source} differs from its pinned bytes")
        os.fsync(target_fd)
        os.fchmod(target_fd, 0o400)
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        os.close(source_fd)


def _restore_and_remove_private_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, files in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in files:
            candidate = current_path / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o600)
        for name in directories:
            candidate = current_path / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o700)
        os.chmod(current_path, 0o700)
    shutil.rmtree(root)


def _positive_launch_environment(
    fixed: Mapping[str, Any],
    *,
    home: Path,
) -> dict[str, str]:
    env = {str(key): str(value) for key, value in fixed.items()}
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "PYTHONUSERBASE": str(home / ".python-user-base"),
            "XDG_CACHE_HOME": str(home / ".cache"),
        }
    )
    if any(name.startswith("LD_") for name in env) or "PYTHONPATH" in env:
        raise ValueError("isolated launch environment contains an ambient loader path")
    return env


def _native_extension_identity(
    seal: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    """Bind DuckDB extension layout to the native ABI and sealed toolchain."""

    errors: list[str] = []
    native = seal.get("configured_board_native_dependency")
    native_pin = native.get("pin") if type(native) is dict else None
    toolchain = seal.get("toolchain")
    if type(native_pin) is not dict or type(toolchain) is not dict:
        return "", "", ["native DuckDB/toolchain extension identity is absent"]

    native_platforms = {
        ("linux", "aarch64"): "linux_arm64",
        ("linux", "x86_64"): "linux_amd64",
        ("darwin", "arm64"): "osx_arm64",
        ("darwin", "x86_64"): "osx_amd64",
        ("win32", "AMD64"): "windows_amd64",
    }
    toolchain_platforms = {
        ("Linux", "aarch64"): "linux_arm64",
        ("Linux", "x86_64"): "linux_amd64",
        ("Darwin", "arm64"): "osx_arm64",
        ("Darwin", "x86_64"): "osx_amd64",
        ("Windows", "AMD64"): "windows_amd64",
    }
    native_platform = native_platforms.get(
        (
            str(native_pin.get("platform_name") or ""),
            str(native_pin.get("platform_machine") or ""),
        )
    )
    toolchain_platform = toolchain_platforms.get(
        (
            str(toolchain.get("operating_system") or ""),
            str(toolchain.get("machine") or ""),
        )
    )
    native_engine = str(native_pin.get("engine_version") or "")
    distribution_version = str(native_pin.get("distribution_version") or "")
    toolchain_version = str(toolchain.get("duckdb_distribution_version") or "")
    if (
        not native_engine
        or native_engine != f"v{distribution_version}"
        or distribution_version != toolchain_version
    ):
        errors.append("native DuckDB engine identity differs from the sealed toolchain")
    if (
        native_platform is None
        or toolchain_platform is None
        or native_platform != toolchain_platform
    ):
        errors.append("native DuckDB platform identity differs from the sealed toolchain")
    return native_engine, native_platform or "", errors


def _project_version(path: Path) -> str:
    with path.open("rb") as handle:
        project = tomllib.load(handle).get("project") or {}
    return str(project.get("version") or "")


def _status_paths(root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
    )
    if completed.returncode:
        raise RuntimeError(f"git status failed: {completed.stderr.strip()}")
    lines = completed.stdout.splitlines()
    result: list[str] = []
    for line in lines:
        text = line[3:] if len(line) >= 4 else ""
        if " -> " in text:
            text = text.split(" -> ", 1)[1]
        result.append(text.strip('"'))
    return tuple(result)


def _validate_native_authorization(
    root: Path,
    seal: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    toolchain: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Authenticate authority independently of inspecting the native bytes."""

    errors: list[str] = []
    native_value = seal.get("configured_board_native_dependency")
    if type(native_value) is not dict or set(native_value) != {
        "schema",
        "source_path",
        "acceptance",
        "pin",
        "sealed_memfd_required",
        "ambient_site_import_allowed",
        "ambient_loader_environment_allowed",
    }:
        return {}, {}, ["configured native dependency seal is noncanonical"]
    native = dict(native_value)
    if (
        native.get("schema")
        != "semantic-addressed-world-model/configured-board-native-dependency@1"
        or native.get("sealed_memfd_required") is not True
        or native.get("ambient_site_import_allowed") is not False
        or native.get("ambient_loader_environment_allowed") is not False
    ):
        errors.append("configured native dependency policy is invalid")
    source = Path(str(native.get("source_path") or ""))
    if not source.is_absolute():
        errors.append("configured native dependency source is not absolute")

    pin = native.get("pin")
    reference = native.get("acceptance")
    if type(pin) is not dict:
        errors.append("configured native dependency pin is not an object")
        pin = {}
    if type(reference) is not dict or set(reference) != {
        "schema",
        "path",
        "sha256",
        "size",
        "authorization_id",
    }:
        errors.append("native authorization reference is noncanonical")
        return native, {}, errors
    if reference.get("schema") != (
        "semantic-addressed-world-model/"
        "native-dependency-authorization-reference@1"
    ):
        errors.append("native authorization reference schema is invalid")

    relative = Path(str(reference.get("path") or ""))
    protected = set(scheduler.get("protected_paths") or ())
    capsule = scheduler.get("configured_board_live_capsule")
    capsule_paths = set(capsule.get("control_paths") or ()) if type(capsule) is dict else set()
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() not in CONTROL_PATHS
        or relative.as_posix() not in protected
        or relative.as_posix() not in capsule_paths
    ):
        errors.append("native authorization artifact is not a protected control path")
        return native, {}, errors

    authorization_path = root / relative
    try:
        evidence = _stable_regular_evidence(authorization_path, maximum=65_536)
        authorization = _load(authorization_path)
    except Exception as exc:
        errors.append(f"native authorization artifact is unavailable: {type(exc).__name__}: {exc}")
        return native, {}, errors
    if (
        evidence.get("sha256") != reference.get("sha256")
        or evidence.get("size") != reference.get("size")
    ):
        errors.append("native authorization artifact bytes differ from the accepted reference")
    if set(authorization) != _NATIVE_AUTHORIZATION_FIELDS:
        errors.append("native authorization artifact fields are noncanonical")
        return native, authorization, errors

    unsigned = dict(authorization)
    authorization_id = unsigned.pop("authorization_id", None)
    expected_id = "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if (
        authorization_id != expected_id
        or authorization_id != reference.get("authorization_id")
        or authorization.get("schema") != (
            "semantic-addressed-world-model/"
            "native-dependency-launch-authorization@1"
        )
        or authorization.get("board_namespace") != NAMESPACE
        or authorization.get("plan_revision") != REVISION
        or authorization.get("status") != "accepted"
        or authorization.get("scope") != "configured-board-live-control-plane"
        or authorization.get("dependency_id") != pin.get("dependency_id")
        or authorization.get("payload_sha256") != pin.get("payload_sha256")
        or authorization.get("python_executable_sha256")
        != pin.get("python_executable_sha256")
        or authorization.get("authority_basis")
        != (
            "operator-owned protected control inside the accepted immutable "
            "source capsule"
        )
        or authorization.get("inspection_is_authority") is not False
        or authorization.get("authorization_may_claim_task_completion") is not False
    ):
        errors.append("native dependency authorization was not independently admitted")
    expected_python_digest = "sha256:" + str(toolchain.get("launch_python_sha256") or "")
    if (
        pin.get("python_executable_sha256") != expected_python_digest
        or pin.get("distribution_version")
        != toolchain.get("duckdb_distribution_version")
    ):
        errors.append("native dependency pin differs from the launch toolchain declaration")
    return native, authorization, errors


def _validate_extension_projection_pins(
    seal: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Rehash Quack and HTTPFS payload/metadata and bind scheduler projections."""

    errors: list[str] = []
    projections: dict[str, dict[str, Any]] = {}
    owner = scheduler.get("quack_owner")
    if type(owner) is not dict:
        return {}, ["scheduler quack_owner is absent"]
    expected_engine, expected_platform, identity_errors = (
        _native_extension_identity(seal)
    )
    errors.extend(identity_errors)

    configured = seal.get("configured_board_quack_projection")
    configured_pin: dict[str, Any] = {}
    if type(configured) is not dict or set(configured) != {
        "schema",
        "source_path",
        "info_path",
        "pin",
        "load_policy",
        "network_install_allowed",
        "unsigned_extension_allowed",
    }:
        errors.append("configured-board Quack projection is noncanonical")
    else:
        pin_value = configured.get("pin")
        if type(pin_value) is not dict or set(pin_value) != {
            "schema",
            "name",
            "engine_version",
            "platform",
            "payload_sha256",
            "payload_size",
            "info_sha256",
            "info_size",
            "projection_id",
        }:
            errors.append("configured-board Quack projection pin is noncanonical")
        else:
            configured_pin = dict(pin_value)
            identity_body = dict(configured_pin)
            observed_id = identity_body.pop("projection_id", None)
            expected_id = "sha256:" + hashlib.sha256(
                _canonical_json(identity_body)
            ).hexdigest()
            if (
                configured.get("schema")
                != "semantic-addressed-world-model/configured-board-quack-projection@1"
                or configured.get("load_policy") != "local_load_only"
                or configured.get("network_install_allowed") is not False
                or configured.get("unsigned_extension_allowed") is not False
                or configured_pin.get("schema") != (
                    "ipfs_accelerate_py.agent_supervisor."
                    "configured-board-duckdb-extension-pin@1"
                )
                or configured_pin.get("name") != "quack"
                or observed_id != expected_id
            ):
                errors.append("configured-board Quack projection identity is invalid")

    engine_version = str(configured_pin.get("engine_version") or "")
    platform_name = str(configured_pin.get("platform") or "")
    if (
        not engine_version.startswith("v")
        or not platform_name
        or engine_version != expected_engine
        or platform_name != expected_platform
    ):
        errors.append(
            "configured-board extension engine/platform differs from "
            "native DuckDB/toolchain"
        )

    for name, seal_key, scheduler_key in (
        ("quack", "quack_extension_pin", "pinned_extension"),
        ("httpfs", "httpfs_extension_pin", "pinned_httpfs_extension"),
    ):
        value = seal.get(seal_key)
        scheduled = owner.get(scheduler_key)
        required = {
            "path",
            "info_path",
            "version",
            "sha256",
            "size",
            "info_sha256",
            "info_size",
            "network_install_allowed",
            "unsigned_extension_allowed",
        }
        allowed = required | ({"service_external_access_limitation"} if name == "quack" else set())
        if type(value) is not dict or set(value) != allowed:
            errors.append(f"{name} extension source pin is noncanonical")
            continue
        if scheduled != value:
            errors.append(f"scheduler {name} extension pin differs from the dependency seal")
        path = Path(str(value.get("path") or ""))
        info_path = Path(str(value.get("info_path") or ""))
        size = value.get("size")
        info_size = value.get("info_size")
        if (
            not path.is_absolute()
            or not info_path.is_absolute()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or isinstance(info_size, bool)
            or not isinstance(info_size, int)
            or value.get("network_install_allowed") is not False
            or value.get("unsigned_extension_allowed") is not False
        ):
            errors.append(f"{name} extension source pin policy is invalid")
            continue
        if (
            path.parent.name != expected_platform
            or path.parent.parent.name != expected_engine
            or info_path.parent != path.parent
        ):
            errors.append(
                f"{name} extension path engine/platform differs from "
                "native DuckDB/toolchain"
            )
        try:
            payload_evidence = _stable_regular_evidence(path, maximum=64 * 1024 * 1024)
            info_evidence = _stable_regular_evidence(info_path, maximum=64 * 1024)
        except Exception as exc:
            errors.append(f"{name} extension source is unavailable: {type(exc).__name__}: {exc}")
            continue
        payload_sha256 = "sha256:" + str(value.get("sha256") or "")
        metadata_sha256 = "sha256:" + str(value.get("info_sha256") or "")
        if (
            payload_evidence != {"sha256": payload_sha256, "size": size}
            or info_evidence != {"sha256": metadata_sha256, "size": info_size}
        ):
            errors.append(f"{name} extension payload or metadata bytes differ")
        body: dict[str, Any] = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "configured-board-duckdb-extension-pin@1"
            ),
            "name": name,
            "engine_version": engine_version,
            "platform": platform_name,
            "payload_sha256": payload_sha256,
            "payload_size": size,
            "info_sha256": metadata_sha256,
            "info_size": info_size,
        }
        body["projection_id"] = "sha256:" + hashlib.sha256(
            _canonical_json(body)
        ).hexdigest()
        projections[name] = {
            **body,
            "source_path": str(path),
            "info_path": str(info_path),
            "version": str(value.get("version") or ""),
        }

    quack = projections.get("quack")
    if quack and configured_pin:
        quack_pin = {key: item for key, item in quack.items() if key not in {"source_path", "info_path", "version"}}
        if (
            configured.get("source_path") != quack.get("source_path")
            or configured.get("info_path") != quack.get("info_path")
            or configured_pin != quack_pin
        ):
            errors.append("configured-board Quack projection differs from exact source pins")
    return projections, errors


def _isolated_native_extension_probe(
    root: Path,
    *,
    launch_python: Path,
    expected_python_sha256: str,
    fixed: Mapping[str, Any],
    native: Mapping[str, Any],
    authorization: Mapping[str, Any],
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the launch stack under exact CPython flags and a positive env."""

    probe_root = Path(tempfile.mkdtemp(prefix="sawm-dependency-probe-", dir="/tmp"))
    try:
        os.chmod(probe_root, 0o700)
        home = probe_root / "home"
        home.mkdir(mode=0o700)
        (home / ".python-user-base").mkdir(mode=0o700)
        (home / ".cache").mkdir(mode=0o700)
        quack = projections["quack"]
        extension_directory = (
            home
            / ".duckdb/extensions"
            / str(quack["engine_version"])
            / str(quack["platform"])
        )
        extension_directory.mkdir(parents=True, mode=0o700)
        expected_install_paths: dict[str, str] = {}
        for name in ("httpfs", "quack"):
            projection = projections[name]
            extension = extension_directory / f"{name}.duckdb_extension"
            metadata = extension_directory / f"{name}.duckdb_extension.info"
            _copy_pinned_regular(
                Path(str(projection["source_path"])),
                extension,
                expected_sha256=str(projection["payload_sha256"]),
                expected_size=int(projection["payload_size"]),
            )
            _copy_pinned_regular(
                Path(str(projection["info_path"])),
                metadata,
                expected_sha256=str(projection["info_sha256"]),
                expected_size=int(projection["info_size"]),
            )
            expected_install_paths[name] = str(extension)
        for directory in sorted(
            (item for item in home.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if directory != home / ".cache":
                os.chmod(directory, 0o500)
        os.chmod(home, 0o500)

        payload = {
            "native": native,
            "authorization_id": authorization.get("authorization_id"),
            "launch_python_sha256": expected_python_sha256,
            "expected_install_paths": expected_install_paths,
            "extension_versions": {
                name: str(projections[name]["version"])
                for name in ("httpfs", "quack")
            },
            "fixed_environment": {str(key): str(value) for key, value in fixed.items()},
            "home": str(home),
        }
        probe = r'''
import hashlib, importlib.util, json, os, pathlib, platform, sys

payload = json.load(sys.stdin)
cmdline = pathlib.Path("/proc/self/cmdline").read_bytes().split(b"\0")
ambient_specs = {
    name: importlib.util.find_spec(name) is not None
    for name in ("duckdb", "_duckdb", "pytest")
}
stdlib_path = list(sys.path)
if (
    sys.flags.isolated != 1
    or sys.flags.no_site != 1
    or sys.flags.dont_write_bytecode != 1
    or sys.flags.no_user_site != 1
    or not all(flag in cmdline for flag in (b"-I", b"-S", b"-B"))
    or any(ambient_specs.values())
    or any("site-packages" in item or "dist-packages" in item for item in stdlib_path)
    or "site" in sys.modules
    or "PYTHONPATH" in os.environ
    or any(name.startswith("LD_") for name in os.environ)
):
    raise RuntimeError("isolated launch interpreter admitted ambient packages")
for name, value in payload["fixed_environment"].items():
    if os.environ.get(name) != value:
        raise RuntimeError("isolated launch fixed environment drifted")
if os.environ.get("HOME") != payload["home"]:
    raise RuntimeError("isolated launch HOME differs from qualification projection")

executable_sha256 = hashlib.sha256(pathlib.Path("/proc/self/exe").read_bytes()).hexdigest()
if (
    sys.executable != "/usr/bin/python3.12"
    or executable_sha256 != payload["launch_python_sha256"]
):
    raise RuntimeError("isolated launch interpreter bytes differ from the seal")

root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from ipfs_accelerate_py.agent_implementation_route import (
    inspect_agent_supervisor_native_dependency_source,
    parse_agent_supervisor_native_dependency_pin,
    preload_agent_supervisor_native_dependency_from_bootstrap,
    seal_agent_supervisor_native_dependency,
)

native = payload["native"]
expected_pin = parse_agent_supervisor_native_dependency_pin(native["pin"])
observed_pin = inspect_agent_supervisor_native_dependency_source(
    native["source_path"],
    distribution_version=expected_pin.distribution_version,
    engine_version=expected_pin.engine_version,
)
if observed_pin != expected_pin:
    raise RuntimeError("native dependency source differs from its accepted pin")
launch = seal_agent_supervisor_native_dependency(
    native["source_path"],
    expected_pin=expected_pin,
    accepted_authorization_id=payload["authorization_id"],
)
try:
    duckdb = preload_agent_supervisor_native_dependency_from_bootstrap(
        *launch.bootstrap_arguments
    )
    expected_origin = f"/proc/self/fd/{launch.descriptor.descriptor}"
    if getattr(duckdb, "__file__", None) != expected_origin:
        raise RuntimeError("DuckDB was not loaded from the sealed descriptor")
    connection = duckdb.connect(
        ":memory:",
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_unsigned_extensions": "false",
            "enable_external_access": "true",
            "extension_directory": str(
                pathlib.Path(payload["home"]) / ".duckdb/extensions"
            ),
        },
    )
    try:
        load_settings = connection.execute(
            "SELECT current_setting('enable_external_access'), "
            "current_setting('autoinstall_known_extensions'), "
            "current_setting('autoload_known_extensions'), "
            "current_setting('allow_unsigned_extensions')"
        ).fetchone()
        if list(load_settings or ()) != [True, False, False, False]:
            raise RuntimeError("local extension load window policy drifted")
        connection.execute("LOAD httpfs")
        connection.execute("LOAD quack")
        rows = connection.execute(
            "SELECT extension_name, extension_version, loaded, installed, install_path "
            "FROM duckdb_extensions() "
            "WHERE extension_name IN ('httpfs', 'quack') ORDER BY extension_name"
        ).fetchall()
        connection.execute("SET enable_external_access=false")
        answer = connection.execute("SELECT 42").fetchone()
        settings = connection.execute(
            "SELECT current_setting('enable_external_access'), "
            "current_setting('autoinstall_known_extensions'), "
            "current_setting('autoload_known_extensions'), "
            "current_setting('allow_unsigned_extensions')"
        ).fetchone()
    finally:
        connection.close()
finally:
    os.close(launch.descriptor.descriptor)

expected_rows = [
    [
        name,
        payload["extension_versions"][name],
        True,
        True,
        payload["expected_install_paths"][name],
    ]
    for name in ("httpfs", "quack")
]
if list(answer or ()) != [42] or [list(row) for row in rows] != expected_rows:
    raise RuntimeError("isolated DuckDB extension probe returned unexpected evidence")
if list(settings or ()) != [False, False, False, False]:
    raise RuntimeError("isolated DuckDB extension/network policy drifted")
print(json.dumps({
    "valid": True,
    "argv_flags": ["-I", "-S", "-B"],
    "ambient_specs": ambient_specs,
    "stdlib_path": stdlib_path,
    "python_executable": sys.executable,
    "python_executable_sha256": executable_sha256,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "operating_system": platform.system(),
    "machine": platform.machine(),
    "native_dependency_id": observed_pin.dependency_id,
    "native_payload_sha256": observed_pin.payload_sha256,
    "native_module_origin": expected_origin,
    "duckdb_distribution_version": str(duckdb.__version__),
    "extension_rows": [list(row) for row in rows],
    "settings": list(settings),
    "select_42": 42,
    "database_opened": True,
    "network_required": False,
}, sort_keys=True))
'''
        completed = subprocess.run(
            [str(launch_python), "-I", "-S", "-B", "-c", probe, str(root)],
            cwd=root,
            env=_positive_launch_environment(fixed, home=home),
            check=False,
            input=json.dumps(payload, sort_keys=True),
            capture_output=True,
            text=True,
            timeout=90,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return {
                "valid": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-4000:],
            }
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {
                "valid": False,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        if result.get("valid") is True:
            rows = result.get("extension_rows")
            normalized_rows: list[list[Any]] = []
            if not isinstance(rows, list) or len(rows) != 2:
                return {
                    "valid": False,
                    "error": "isolated extension rows are malformed",
                }
            for row in rows:
                if (
                    not isinstance(row, list)
                    or len(row) != 5
                    or row[0] not in expected_install_paths
                    or row[4] != expected_install_paths[row[0]]
                ):
                    return {
                        "valid": False,
                        "error": "isolated extension path evidence drifted",
                    }
                relative = Path(row[4]).relative_to(home).as_posix()
                normalized_rows.append(
                    [*row[:4], f"$ISOLATED_HOME/{relative}"]
                )
            result["extension_rows"] = normalized_rows
            result["extension_path_reporting"] = (
                "normalized_after_exact_runtime_path_verification"
            )
        return result
    finally:
        _restore_and_remove_private_tree(probe_root)


def _cold_import(root: Path, python: str, fixed: Mapping[str, Any]) -> dict[str, Any]:
    probe = r'''
import importlib.util, json, os, socket, subprocess, sys, threading
ambient_specs={name:importlib.util.find_spec(name) is not None for name in ("duckdb","_duckdb","pytest")}
stdlib_path=list(sys.path)
if (sys.flags.isolated != 1 or sys.flags.no_site != 1
    or sys.flags.dont_write_bytecode != 1 or sys.flags.no_user_site != 1
    or any(ambient_specs.values())
    or any("site-packages" in item or "dist-packages" in item for item in stdlib_path)
    or "site" in sys.modules):
    raise RuntimeError("cold_import_ambient_package_path")
sys.path[:0] = json.loads(sys.stdin.read())
before_env = dict(os.environ); before_threads = {t.ident for t in threading.enumerate()}
def denied(*a, **k): raise RuntimeError("cold_import_side_effect_denied")
class DeniedPopen(subprocess.Popen):
    def __init__(self, *a, **k): denied(*a, **k)
class DeniedSocket(socket.socket):
    def __init__(self, *a, **k): denied(*a, **k)
subprocess.Popen = DeniedPopen
for name in ("run", "call", "check_call", "check_output"):
    setattr(subprocess, name, denied)
socket.socket = DeniedSocket
mods = [
 "ipfs_accelerate_py", "ipfs_datasets_py", "ipfs_kit_py",
 "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source",
 "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema",
 "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler",
 "ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server",
]
errors=[]
for mod in mods:
    try: __import__(mod)
    except Exception as exc: errors.append({"module":mod,"error":type(exc).__name__+": "+str(exc)})
after_threads={t.ident for t in threading.enumerate()}
changed={k:[before_env.get(k),v] for k,v in os.environ.items() if before_env.get(k)!=v}
removed=sorted(set(before_env)-set(os.environ))
print(json.dumps({"valid":not errors and not changed and not removed and after_threads==before_threads,
 "errors":errors,"environment_changes":changed,"environment_removed":removed,
 "new_thread_count":len(after_threads-before_threads),"ambient_specs":ambient_specs,
 "stdlib_path":stdlib_path},sort_keys=True))
'''
    with tempfile.TemporaryDirectory(prefix="sawm-cold-import-", dir="/tmp") as directory:
        home = Path(directory)
        (home / ".python-user-base").mkdir(mode=0o700)
        (home / ".cache").mkdir(mode=0o700)
        completed = subprocess.run(
            [python, "-I", "-S", "-B", "-c", probe],
            cwd=root,
            env=_positive_launch_environment(fixed, home=home),
            check=False,
            input=json.dumps(
                [str(root / "ipfs_datasets_py"), str(root / "ipfs_kit_py"), str(root)]
            ),
            capture_output=True,
            text=True,
            timeout=45,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return {
                "valid": False,
                "returncode": completed.returncode,
                "stderr": completed.stderr[-2000:],
            }
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {
                "valid": False,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }


def _control_plane_mode_closure(
    root: Path,
    python: str,
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise the capsule's own closed file discovery and stable-read gate."""

    probe = r'''
import importlib.util, json, os, pathlib, stat, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
source = root / "ipfs_accelerate_py" / "agent_implementation_route.py"
spec = importlib.util.spec_from_file_location("_sawm_control_plane_mode_probe", source)
if spec is None or spec.loader is None:
    raise RuntimeError("control_plane_probe_loader_unavailable")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
files = module._agent_control_plane_source_files(
    root,
    verify_loaded_origins=False,
)
errors = []
mode_counts = {}
for path in files:
    try:
        module._agent_read_stable_file(path)
        mode = oct(stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    except Exception as exc:
        errors.append({
            "path": str(path.relative_to(root)),
            "error": type(exc).__name__ + ": " + str(exc),
        })
print(json.dumps({
    "valid": bool(files) and not errors,
    "file_count": len(files),
    "mode_counts": mode_counts,
    "errors": errors,
    "argv_flags": ["-I", "-S", "-B"],
}, sort_keys=True))
'''
    try:
        with tempfile.TemporaryDirectory(
            prefix="sawm-control-plane-mode-",
            dir="/tmp",
        ) as directory:
            home = Path(directory)
            (home / ".python-user-base").mkdir(mode=0o700)
            (home / ".cache").mkdir(mode=0o700)
            completed = subprocess.run(
                [python, "-I", "-S", "-B", "-c", probe, str(root)],
                cwd=root,
                env=_positive_launch_environment(fixed, home=home),
                check=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=45,
            )
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {
            "valid": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }


def _effective_nested_source_authorities(
    authorities: Sequence[Any],
    scheduler: Mapping[str, Any],
    migration: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Overlay the newest exact accepted nested-source transition."""

    effective = {
        str(item.get("package")): dict(item)
        for item in authorities
        if isinstance(item, Mapping) and str(item.get("package") or "")
    }
    m46_key = _M46_SUCCESSOR_KEY
    m46_presence = (
        m46_key in scheduler,
        m46_key in migration,
        f"{m46_key}_cid" in seal,
    )
    if any(m46_presence):
        if not all(m46_presence):
            return effective, ["active M46 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m46_nested_source_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M46 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer
                ._expected_m46_legacy_no_delta_rescue_recovery_successor_authority()
            )
            materializer._validated_m46_live_preflight_contract(authority)
            materializer._assert_m46_historical_m45_controls(
                scheduler, migration, seal
            )
            reference = dict(materializer._m46_authority_reference())
            source_errors = _m46_source_chain_errors(
                REPO_ROOT, materializer, authority
            )
            if source_errors:
                raise RuntimeError("; ".join(source_errors))
        except Exception as exc:
            return effective, [f"active M46 nested-source authority unavailable: {exc}"]
        if (
            scheduler.get(m46_key) != reference
            or migration.get(m46_key) != reference
            or seal.get(f"{m46_key}_cid") != materializer._M46_AUTHORITY_CID
            or materializer._M46_AUTHORITY_CID == _M46_UNSEALED_AUTHORITY_CID
            or materializer._M46_AUTHORITY_CID != _M46_AUTHORITY_CID
            or materializer._identity(authority) != materializer._M46_AUTHORITY_CID
            or len(materializer._canonical(authority)) != _M46_AUTHORITY_SIZE
        ):
            return effective, ["active M46 nested-source authority differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M46 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []
    m45_key = _M45_SUCCESSOR_KEY
    m45_presence = (
        m45_key in scheduler,
        m45_key in migration,
        f"{m45_key}_cid" in seal,
    )
    if any(m45_presence):
        if not all(m45_presence):
            return effective, ["active M45 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m45_nested_source_materializer",
                REPO_ROOT
                / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M45 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer
                ._expected_m45_failed_pre_authoritative_m44_validation_successor_authority()
            )
            materializer._validated_m45_live_preflight_contract(authority)
            materializer._assert_m45_historical_m44_controls(
                scheduler, migration, seal
            )
            reference = dict(materializer._m45_authority_reference())
            source_errors = _m45_source_chain_errors(
                REPO_ROOT, materializer, authority
            )
            if source_errors:
                raise RuntimeError("; ".join(source_errors))
        except Exception as exc:
            return effective, [f"active M45 nested-source authority unavailable: {exc}"]
        if (
            scheduler.get(m45_key) != reference
            or migration.get(m45_key) != reference
            or seal.get(f"{m45_key}_cid") != materializer._M45_AUTHORITY_CID
            or materializer._M45_AUTHORITY_CID == _M45_UNSEALED_AUTHORITY_CID
            or materializer._M45_AUTHORITY_CID != _M45_AUTHORITY_CID
            or materializer._identity(authority) != materializer._M45_AUTHORITY_CID
            or len(materializer._canonical(authority)) != _M45_AUTHORITY_SIZE
        ):
            return effective, ["active M45 nested-source authority differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M45 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []
    m44_key = _M44_SUCCESSOR_KEY
    m44_presence = (
        m44_key in scheduler,
        m44_key in migration,
        f"{m44_key}_cid" in seal,
    )
    if any(m44_presence):
        if not all(m44_presence):
            return effective, ["active M44 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m44_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M44 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer
                ._expected_m44_post_m43_hardened_procfs_user_manager_restart_successor_authority()
            )
            materializer._validated_m44_live_preflight_contract(authority)
            reference = dict(materializer._m44_authority_reference())
            source_errors = _m44_source_chain_errors(
                REPO_ROOT, materializer, authority
            )
            if source_errors:
                raise RuntimeError("; ".join(source_errors))
        except Exception as exc:
            return effective, [f"active M44 nested-source authority unavailable: {exc}"]
        if (
            scheduler.get(m44_key) != reference
            or migration.get(m44_key) != reference
            or seal.get(f"{m44_key}_cid") != materializer._M44_AUTHORITY_CID
            or materializer._M44_AUTHORITY_CID == _M44_UNSEALED_AUTHORITY_CID
            or materializer._M44_AUTHORITY_CID != _M44_AUTHORITY_CID
            or materializer._identity(authority) != materializer._M44_AUTHORITY_CID
        ):
            return effective, ["active M44 nested-source authority differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M44 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []
    m43_key = "dead_attempt_lifecycle_recovery_restart_successor_materialization"
    m43_presence = (
        m43_key in scheduler,
        m43_key in migration,
        f"{m43_key}_cid" in seal,
    )
    if any(m43_presence):
        if not all(m43_presence):
            return effective, ["active M43 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m43_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M43 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer
                ._expected_m43_dead_attempt_lifecycle_recovery_restart_authority()
            )
            materializer._validated_m43_live_preflight_contract(authority)
            reference = dict(materializer._m43_authority_reference())
            source_errors = _m43_source_chain_errors(
                REPO_ROOT, materializer, authority
            )
            if source_errors:
                raise RuntimeError("; ".join(source_errors))
        except Exception as exc:
            return effective, [f"active M43 nested-source authority unavailable: {exc}"]
        if (
            scheduler.get(m43_key) != reference
            or migration.get(m43_key) != reference
            or seal.get(f"{m43_key}_cid") != materializer._M43_AUTHORITY_CID
            or materializer._M43_AUTHORITY_CID == _M43_UNSEALED_AUTHORITY_CID
            or materializer._M43_AUTHORITY_CID != _M43_AUTHORITY_CID
            or materializer._identity(authority) != materializer._M43_AUTHORITY_CID
        ):
            return effective, ["active M43 nested-source authority differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M43 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []
    m42_key = (
        "failed_pre_authoritative_m41_evidence_projection_"
        "successor_materialization"
    )
    m42_presence = (
        m42_key in scheduler,
        m42_key in migration,
        f"{m42_key}_cid" in seal,
    )
    if any(m42_presence):
        if not all(m42_presence):
            return effective, ["active M42 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m42_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M42 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer
                ._expected_m42_failed_pre_authoritative_m41_evidence_projection_successor_authority()
            )
            materializer._validated_m42_live_preflight_contract(authority)
            reference = dict(materializer._m42_authority_reference())
            expected_cid = materializer._identity(authority)
            source_errors = _m42_source_chain_errors(
                REPO_ROOT, materializer, authority
            )
            if source_errors:
                raise RuntimeError("; ".join(source_errors))
        except Exception as exc:
            return effective, [f"active M42 nested-source authority unavailable: {exc}"]
        if (
            scheduler.get(m42_key) != reference
            or migration.get(m42_key) != reference
            or seal.get(f"{m42_key}_cid") != expected_cid
            or expected_cid != _M42_AUTHORITY_CID
        ):
            return effective, ["active M42 nested-source authority differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M42 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m41_key = "failed_pre_authoritative_m40_validation_successor_materialization"
    m41_presence = (
        m41_key in scheduler,
        m41_key in migration,
        f"{m41_key}_cid" in seal,
    )
    if any(m41_presence):
        if not all(m41_presence):
            return effective, ["active M41 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m41_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M41 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer._expected_m41_failed_pre_authoritative_m40_validation_successor_authority()
            )
            materializer._validated_m41_live_preflight_contract(authority)
            reference = dict(materializer._m41_authority_reference())
            expected_cid = materializer._identity(authority)
            source_errors = _m41_source_chain_errors(
                REPO_ROOT, materializer, authority
            )
            if source_errors:
                raise RuntimeError("; ".join(source_errors))
        except Exception as exc:
            return effective, [f"active M41 nested-source authority unavailable: {exc}"]
        if (
            scheduler.get(m41_key) != reference
            or migration.get(m41_key) != reference
            or seal.get(f"{m41_key}_cid") != expected_cid
            or expected_cid
            != "sha256:25ad5550b59024c8da9b4821fba2d7b1b49d2a17a781e5d4bd2cbe58cb7b0233"
        ):
            return effective, ["active M41 nested-source authority differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M41 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m40_key = "failed_pre_authoritative_m39_successor_materialization"
    m40_presence = (
        m40_key in scheduler,
        m40_key in migration,
        f"{m40_key}_cid" in seal,
    )
    if any(m40_presence):
        if not all(m40_presence):
            return effective, ["active M40 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m40_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M40 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer._expected_m40_failed_pre_authoritative_m39_successor_authority()
            )
            materializer._validated_m40_live_preflight_contract(authority)
            reference = dict(materializer._m40_authority_reference())
            expected_cid = materializer._identity(authority)
            source_errors = _m40_source_chain_errors(
                REPO_ROOT, materializer, authority
            )
            if source_errors:
                raise RuntimeError("; ".join(source_errors))
        except Exception as exc:
            return effective, [f"active M40 nested-source authority unavailable: {exc}"]
        if (
            scheduler.get(m40_key) != reference
            or migration.get(m40_key) != reference
            or seal.get(f"{m40_key}_cid") != expected_cid
            or expected_cid
            != "sha256:377b6e7269a7025f236642f12aaf92264582f42a1efa4899eb4d6e64b0e41db2"
        ):
            return effective, ["active M40 nested-source authority differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M40 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m39_key = "committed_m38_evidence_reconciliation_successor_materialization"
    m39_presence = (
        m39_key in scheduler,
        m39_key in migration,
        f"{m39_key}_cid" in seal,
    )
    if any(m39_presence):
        if not all(m39_presence):
            return effective, ["active M39 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m39_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M39 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer._expected_m39_committed_m38_evidence_reconciliation_authority()
            )
            materializer._validated_m39_live_preflight_contract(authority)
            reference = dict(materializer._m39_authority_reference())
            expected_cid = materializer._identity(authority)
            materializer._assert_m39_historical_m38_controls(
                scheduler, migration, seal
            )
        except Exception as exc:
            return effective, [f"active M39 nested-source authority unavailable: {exc}"]
        if (
            scheduler.get(m39_key) != reference
            or migration.get(m39_key) != reference
            or seal.get(f"{m39_key}_cid") != expected_cid
            or expected_cid
            != "sha256:7b986174605b4f2739abf69473d0ce27dfe880d551c55144f076b8f981f633c2"
        ):
            return effective, ["active M39 nested-source authority differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M39 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m38_key = "pre_authoritative_custody_restart_successor_materialization"
    m38_presence = (
        m38_key in scheduler,
        m38_key in migration,
        f"{m38_key}_cid" in seal,
    )
    if any(m38_presence):
        if not all(m38_presence):
            return effective, ["active M38 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m38_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M38 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer._expected_m38_pre_authoritative_custody_restart_authority()
            )
            materializer._validated_m38_live_preflight_contract(authority)
            reference = _m38_authority_reference_for_source_state(
                materializer, authority
            )
            identity_state = _m38_source_chain_identity_state(materializer, authority)
            expected_cid = (
                "sha256:PENDING_M38_AUTHORITY_CID"
                if identity_state == "placeholder"
                else materializer._identity(authority)
            )
        except Exception as exc:
            return effective, [f"active M38 nested-source authority unavailable: {exc}"]
        if (
            scheduler.get(m38_key) != reference
            or migration.get(m38_key) != reference
            or seal.get(f"{m38_key}_cid") != expected_cid
        ):
            return effective, ["active M38 nested-source authority differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M38 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m37_key = "post_reboot_generation_restart_successor_materialization"
    m37_presence = (
        m37_key in scheduler,
        m37_key in migration,
        f"{m37_key}_cid" in seal,
    )
    if any(m37_presence):
        if not all(m37_presence):
            return effective, ["active M37 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m37_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M37 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            authority = (
                materializer._expected_m37_post_reboot_generation_restart_authority()
            )
            materializer._validated_m37_live_preflight_contract(authority)
            reference = _m37_authority_reference_for_source_state(
                materializer, authority
            )
        except Exception as exc:
            return effective, [f"active M37 nested-source authority unavailable: {exc}"]
        if scheduler.get(m37_key) != reference or migration.get(m37_key) != reference:
            return effective, ["active M37 nested-source authority differs"]
        identity_state = _m37_source_chain_identity_state(materializer, authority)
        expected_seal_cid = (
            "sha256:PENDING_M37_AUTHORITY_CID"
            if identity_state == "placeholder"
            else materializer._identity(authority)
        )
        if seal.get(f"{m37_key}_cid") != expected_seal_cid:
            return effective, ["active M37 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M37 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m36_key = "operator_task_binding_correction_successor_materialization"
    m36_presence = (
        m36_key in scheduler,
        m36_key in migration,
        f"{m36_key}_cid" in seal,
    )
    if any(m36_presence):
        if not all(m36_presence):
            return effective, ["active M36 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m36_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M36 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            reference = materializer._m36_authority_reference()
            authority = (
                materializer._expected_m36_operator_task_binding_correction_authority()
            )
            materializer._validated_m36_live_preflight_contract(authority)
        except Exception as exc:
            return effective, [f"active M36 nested-source authority unavailable: {exc}"]
        if scheduler.get(m36_key) != reference or migration.get(m36_key) != reference:
            return effective, ["active M36 nested-source authority differs"]
        if seal.get(f"{m36_key}_cid") != materializer._identity(authority):
            return effective, ["active M36 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M36 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m35_key = "immutable_authority_identity_normalization_successor_materialization"
    m35_presence = (
        m35_key in scheduler,
        m35_key in migration,
        f"{m35_key}_cid" in seal,
    )
    if any(m35_presence):
        if not all(m35_presence):
            return effective, ["active M35 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m35_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M35 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            reference = materializer._m35_authority_reference()
            authority = (
                materializer._expected_m35_immutable_authority_identity_normalization_authority()
            )
            materializer._validated_m35_live_preflight_contract(authority)
        except Exception as exc:
            return effective, [f"active M35 nested-source authority unavailable: {exc}"]
        if scheduler.get(m35_key) != reference or migration.get(m35_key) != reference:
            return effective, ["active M35 nested-source authority differs"]
        if seal.get(f"{m35_key}_cid") != materializer._identity(authority):
            return effective, ["active M35 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M35 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m34_key = "json_emission_normalization_successor_materialization"
    m34_presence = (
        m34_key in scheduler,
        m34_key in migration,
        f"{m34_key}_cid" in seal,
    )
    if any(m34_presence):
        if not all(m34_presence):
            return effective, ["active M34 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m34_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M34 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            reference = materializer._m34_authority_reference()
            authority = (
                materializer._expected_m34_json_emission_normalization_authority()
            )
            materializer._validated_m34_live_preflight_contract(authority)
        except Exception as exc:
            return effective, [f"active M34 nested-source authority unavailable: {exc}"]
        if scheduler.get(m34_key) != reference or migration.get(m34_key) != reference:
            return effective, ["active M34 nested-source authority differs"]
        if seal.get(f"{m34_key}_cid") != materializer._identity(authority):
            return effective, ["active M34 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M34 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m33_key = "live_preflight_contract_successor_materialization"
    m33_presence = (
        m33_key in scheduler,
        m33_key in migration,
        f"{m33_key}_cid" in seal,
    )
    if any(m33_presence):
        if not all(m33_presence):
            return effective, ["active M33 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m33_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M33 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            reference = materializer._m33_authority_reference()
            authority = materializer._expected_m33_live_preflight_contract_authority()
            materializer._validated_m33_live_preflight_contract(authority)
        except Exception as exc:
            return effective, [f"active M33 nested-source authority unavailable: {exc}"]
        if scheduler.get(m33_key) != reference or migration.get(m33_key) != reference:
            return effective, ["active M33 nested-source authority differs"]
        if seal.get(f"{m33_key}_cid") != materializer._identity(authority):
            return effective, ["active M33 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M33 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m32_key = "live_preflight_plan_anchor_successor_materialization"
    m32_presence = (
        m32_key in scheduler,
        m32_key in migration,
        f"{m32_key}_cid" in seal,
    )
    if any(m32_presence):
        if not all(m32_presence):
            return effective, ["active M32 nested-source authority is partial"]
        try:
            spec = importlib.util.spec_from_file_location(
                "sawm_m32_nested_source_materializer",
                REPO_ROOT / "scripts/materialize_semantic_addressed_world_model_program.py",
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("M32 materializer unavailable")
            materializer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(materializer)
            reference = materializer._m32_authority_reference()
            authority = materializer._expected_m32_live_preflight_plan_anchor_authority()
        except Exception as exc:
            return effective, [f"active M32 nested-source authority unavailable: {exc}"]
        if scheduler.get(m32_key) != reference or migration.get(m32_key) != reference:
            return effective, ["active M32 nested-source authority differs"]
        if seal.get(f"{m32_key}_cid") != materializer._identity(authority):
            return effective, ["active M32 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(authority.get("current_datasets_gitlink") or ""),
                str(authority.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(authority.get("current_kit_gitlink") or ""),
                str(authority.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M32 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m31_key = "detached_coordinator_pid_recovery_successor_materialization"
    m31_presence = (
        m31_key in scheduler,
        m31_key in migration,
        f"{m31_key}_cid" in seal,
    )
    if any(m31_presence):
        if not all(m31_presence):
            return effective, ["active M31 nested-source authority is partial"]
        scheduled = scheduler.get(m31_key)
        migrated = migration.get(m31_key)
        if (
            type(scheduled) is not dict
            or type(migrated) is not dict
            or _canonical_json(scheduled) != _canonical_json(migrated)
        ):
            return effective, ["active M31 nested-source authority differs"]
        claimed_cid = "sha256:" + hashlib.sha256(
            _canonical_json(scheduled)
        ).hexdigest()
        if seal.get(f"{m31_key}_cid") != claimed_cid:
            return effective, ["active M31 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(scheduled.get("current_datasets_gitlink") or ""),
                str(scheduled.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(scheduled.get("current_kit_gitlink") or ""),
                str(scheduled.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M31 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m30_key = "stopped_owner_restart_source_seal_successor_materialization"
    m30_presence = (
        m30_key in scheduler,
        m30_key in migration,
        f"{m30_key}_cid" in seal,
    )
    if any(m30_presence):
        if not all(m30_presence):
            return effective, ["active M30 nested-source authority is partial"]
        scheduled = scheduler.get(m30_key)
        migrated = migration.get(m30_key)
        if (
            type(scheduled) is not dict
            or type(migrated) is not dict
            or _canonical_json(scheduled) != _canonical_json(migrated)
        ):
            return effective, ["active M30 nested-source authority differs"]
        claimed_cid = "sha256:" + hashlib.sha256(
            _canonical_json(scheduled)
        ).hexdigest()
        if seal.get(f"{m30_key}_cid") != claimed_cid:
            return effective, ["active M30 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(scheduled.get("current_datasets_gitlink") or ""),
                str(scheduled.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(scheduled.get("current_kit_gitlink") or ""),
                str(scheduled.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M30 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m29_key = "committed_evidence_verification_successor_materialization"
    m29_presence = (
        m29_key in scheduler,
        m29_key in migration,
        f"{m29_key}_cid" in seal,
    )
    if any(m29_presence):
        if not all(m29_presence):
            return effective, ["active M29 nested-source authority is partial"]
        scheduled = scheduler.get(m29_key)
        migrated = migration.get(m29_key)
        if (
            type(scheduled) is not dict
            or type(migrated) is not dict
            or _canonical_json(scheduled) != _canonical_json(migrated)
        ):
            return effective, ["active M29 nested-source authority differs"]
        claimed_cid = "sha256:" + hashlib.sha256(
            _canonical_json(scheduled)
        ).hexdigest()
        if seal.get(f"{m29_key}_cid") != claimed_cid:
            return effective, ["active M29 nested-source authority CID differs"]

        identities = {
            "ipfs_datasets_py": (
                str(scheduled.get("current_datasets_gitlink") or ""),
                str(scheduled.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(scheduled.get("current_kit_gitlink") or ""),
                str(scheduled.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M29 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m28_key = "live_claim_admission_recovery_successor_materialization"
    m28_presence = (
        m28_key in scheduler,
        m28_key in migration,
        f"{m28_key}_cid" in seal,
    )
    if any(m28_presence):
        if not all(m28_presence):
            return effective, ["active M28 nested-source authority is partial"]
        scheduled = scheduler.get(m28_key)
        migrated = migration.get(m28_key)
        if (
            type(scheduled) is not dict
            or type(migrated) is not dict
            or _canonical_json(scheduled) != _canonical_json(migrated)
        ):
            return effective, ["active M28 nested-source authority differs"]
        claimed_cid = "sha256:" + hashlib.sha256(
            _canonical_json(scheduled)
        ).hexdigest()
        if seal.get(f"{m28_key}_cid") != claimed_cid:
            return effective, ["active M28 nested-source authority CID differs"]

        identities = {
            "ipfs_datasets_py": (
                str(scheduled.get("current_datasets_gitlink") or ""),
                str(scheduled.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(scheduled.get("current_kit_gitlink") or ""),
                str(scheduled.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M28 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m27_key = "dead_owner_parallel_resume_successor_materialization"
    m27_presence = (
        m27_key in scheduler,
        m27_key in migration,
        f"{m27_key}_cid" in seal,
    )
    if any(m27_presence):
        if not all(m27_presence):
            return effective, ["active M27 nested-source authority is partial"]
        scheduled = scheduler.get(m27_key)
        migrated = migration.get(m27_key)
        if (
            type(scheduled) is not dict
            or type(migrated) is not dict
            or _canonical_json(scheduled) != _canonical_json(migrated)
        ):
            return effective, ["active M27 nested-source authority differs"]
        claimed_cid = "sha256:" + hashlib.sha256(
            _canonical_json(scheduled)
        ).hexdigest()
        if seal.get(f"{m27_key}_cid") != claimed_cid:
            return effective, ["active M27 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(scheduled.get("current_datasets_gitlink") or ""),
                str(scheduled.get("current_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(scheduled.get("current_kit_gitlink") or ""),
                str(scheduled.get("current_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M27 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m26_key = "automatic_stall_recovery_successor_materialization"
    m26_presence = (
        m26_key in scheduler,
        m26_key in migration,
        f"{m26_key}_cid" in seal,
    )
    if any(m26_presence):
        if not all(m26_presence):
            return effective, ["active M26 nested-source authority is partial"]
        scheduled = scheduler.get(m26_key)
        migrated = migration.get(m26_key)
        if (
            type(scheduled) is not dict
            or type(migrated) is not dict
            or _canonical_json(scheduled) != _canonical_json(migrated)
        ):
            return effective, ["active M26 nested-source authority differs"]
        claimed_cid = "sha256:" + hashlib.sha256(
            _canonical_json(scheduled)
        ).hexdigest()
        if seal.get(f"{m26_key}_cid") != claimed_cid:
            return effective, ["active M26 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(scheduled.get("prior_datasets_gitlink") or ""),
                str(scheduled.get("prior_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(scheduled.get("prior_kit_gitlink") or ""),
                str(scheduled.get("prior_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M26 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m25_key = "native_duckdb_preload_successor_materialization"
    m25_presence = (
        m25_key in scheduler,
        m25_key in migration,
        f"{m25_key}_cid" in seal,
    )
    if any(m25_presence):
        if not all(m25_presence):
            return effective, ["active M25 nested-source authority is partial"]
        scheduled = scheduler.get(m25_key)
        migrated = migration.get(m25_key)
        if (
            type(scheduled) is not dict
            or type(migrated) is not dict
            or _canonical_json(scheduled) != _canonical_json(migrated)
        ):
            return effective, ["active M25 nested-source authority differs"]
        claimed_cid = "sha256:" + hashlib.sha256(
            _canonical_json(scheduled)
        ).hexdigest()
        if seal.get(f"{m25_key}_cid") != claimed_cid:
            return effective, ["active M25 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(scheduled.get("prior_datasets_gitlink") or ""),
                str(scheduled.get("prior_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(scheduled.get("prior_kit_gitlink") or ""),
                str(scheduled.get("prior_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M25 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m24_key = "multi_lane_sidecar_reopen_successor_materialization"
    m24_presence = (
        m24_key in scheduler,
        m24_key in migration,
        f"{m24_key}_cid" in seal,
    )
    if any(m24_presence):
        if not all(m24_presence):
            return effective, ["active M24 nested-source authority is partial"]
        scheduled = scheduler.get(m24_key)
        migrated = migration.get(m24_key)
        if (
            type(scheduled) is not dict
            or type(migrated) is not dict
            or _canonical_json(scheduled) != _canonical_json(migrated)
        ):
            return effective, ["active M24 nested-source authority differs"]
        claimed_cid = "sha256:" + hashlib.sha256(
            _canonical_json(scheduled)
        ).hexdigest()
        if seal.get(f"{m24_key}_cid") != claimed_cid:
            return effective, ["active M24 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(scheduled.get("prior_datasets_gitlink") or ""),
                str(scheduled.get("prior_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(scheduled.get("prior_kit_gitlink") or ""),
                str(scheduled.get("prior_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M24 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    m23_key = "multi_lane_successor_materialization"
    m23_presence = (
        m23_key in scheduler,
        m23_key in migration,
        f"{m23_key}_cid" in seal,
    )
    if any(m23_presence):
        if not all(m23_presence):
            return effective, ["active M23 nested-source authority is partial"]
        scheduled = scheduler.get(m23_key)
        migrated = migration.get(m23_key)
        if (
            type(scheduled) is not dict
            or type(migrated) is not dict
            or _canonical_json(scheduled) != _canonical_json(migrated)
        ):
            return effective, ["active M23 nested-source authority differs"]
        claimed_cid = "sha256:" + hashlib.sha256(
            _canonical_json(scheduled)
        ).hexdigest()
        if seal.get(f"{m23_key}_cid") != claimed_cid:
            return effective, ["active M23 nested-source authority CID differs"]
        identities = {
            "ipfs_datasets_py": (
                str(scheduled.get("prior_datasets_gitlink") or ""),
                str(scheduled.get("prior_datasets_tree") or ""),
            ),
            "ipfs_kit_py": (
                str(scheduled.get("prior_kit_gitlink") or ""),
                str(scheduled.get("prior_kit_tree") or ""),
            ),
        }
        if any(
            package not in effective
            or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            or re.fullmatch(r"[0-9a-f]{40}", tree) is None
            for package, (gitlink, tree) in identities.items()
        ):
            return effective, ["active M23 nested-source identity is invalid"]
        for package, (gitlink, tree) in identities.items():
            effective[package] = {
                **effective[package],
                "head": gitlink,
                "gitlink_commit": gitlink,
                "tree": tree,
            }
        return effective, []

    key = "portal_completion_persistence_successor_materialization"
    presence = (key in scheduler, key in migration, f"{key}_cid" in seal)
    if not any(presence):
        return effective, []
    if not all(presence):
        return effective, ["active M18 nested-source authority is partial"]
    scheduled = scheduler.get(key)
    migrated = migration.get(key)
    if (
        type(scheduled) is not dict
        or type(migrated) is not dict
        or scheduled != migrated
    ):
        return effective, ["active M18 nested-source authority differs"]
    claimed_cid = "sha256:" + hashlib.sha256(
        _canonical_json(scheduled)
    ).hexdigest()
    if seal.get(f"{key}_cid") != claimed_cid:
        return effective, ["active M18 nested-source authority CID differs"]

    datasets = effective.get("ipfs_datasets_py")
    kit = effective.get("ipfs_kit_py")
    if not isinstance(datasets, dict) or not isinstance(kit, dict):
        return effective, ["historical nested-source authority is incomplete"]
    if (
        datasets.get("gitlink_commit") != scheduled.get("prior_datasets_gitlink")
        or datasets.get("tree") != scheduled.get("prior_datasets_tree")
        or kit.get("gitlink_commit") != scheduled.get("prior_kit_gitlink")
        or kit.get("tree") != scheduled.get("prior_kit_tree")
    ):
        return effective, ["active M18 nested-source predecessor differs"]
    runtime_gitlink = str(scheduled.get("runtime_datasets_gitlink") or "")
    runtime_tree = str(scheduled.get("runtime_datasets_tree") or "")
    if (
        re.fullmatch(r"[0-9a-f]{40}", runtime_gitlink) is None
        or re.fullmatch(r"[0-9a-f]{40}", runtime_tree) is None
    ):
        return effective, ["active M18 datasets successor identity is invalid"]
    effective["ipfs_datasets_py"] = {
        **datasets,
        "head": runtime_gitlink,
        "gitlink_commit": runtime_gitlink,
        "tree": runtime_tree,
    }
    return effective, []


def validate_dependencies(repo_root: Path | str = REPO_ROOT, *, cold_import: bool = True) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    try:
        seal = _load(root / SEAL_PATH.relative_to(REPO_ROOT))
    except Exception as exc:
        return {"schema": SCHEMA, "valid": False, "board_namespace": NAMESPACE,
                "plan_revision": REVISION, "errors": [f"seal_load: {type(exc).__name__}: {exc}"], "checks": []}

    check("seal_identity", seal.get("schema") == "semantic-addressed-world-model/dependency-seal@1" and seal.get("board_namespace") == NAMESPACE and seal.get("plan_revision") == REVISION and seal.get("status") == "sealed", seal.get("schema"))
    authorities = seal.get("source_authorities") if isinstance(seal.get("source_authorities"), list) else []
    by_package = {str(item.get("package")): item for item in authorities if isinstance(item, Mapping)}
    accel = by_package.get("ipfs_accelerate_py", {})
    scheduler_probe: Mapping[str, Any] = {}
    migration_probe: Mapping[str, Any] = {}
    try:
        head = _git(root, "rev-parse", "HEAD")
        branch = _git(root, "branch", "--show-current")
        ancestor_ok = subprocess.run(["git", "merge-base", "--is-ancestor", str(accel.get("head")), "HEAD"], cwd=root, check=False).returncode == 0
        origin = _git(root, "remote", "get-url", "origin")
        scheduler_probe = _load(root / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        migration_probe = _load(root / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json")
        m46_key = _M46_SUCCESSOR_KEY
        m46_presence = (
            m46_key in scheduler_probe,
            m46_key in migration_probe,
            f"{m46_key}_cid" in seal,
        )
        m45_key = _M45_SUCCESSOR_KEY
        m45_presence = (
            m45_key in scheduler_probe,
            m45_key in migration_probe,
            f"{m45_key}_cid" in seal,
        )
        m44_key = _M44_SUCCESSOR_KEY
        m44_presence = (
            m44_key in scheduler_probe,
            m44_key in migration_probe,
            f"{m44_key}_cid" in seal,
        )
        m43_key = (
            "dead_attempt_lifecycle_recovery_restart_"
            "successor_materialization"
        )
        m43_presence = (
            m43_key in scheduler_probe,
            m43_key in migration_probe,
            f"{m43_key}_cid" in seal,
        )
        m42_key = (
            "failed_pre_authoritative_m41_evidence_projection_"
            "successor_materialization"
        )
        m42_presence = (
            m42_key in scheduler_probe,
            m42_key in migration_probe,
            f"{m42_key}_cid" in seal,
        )
        m41_key = "failed_pre_authoritative_m40_validation_successor_materialization"
        m41_presence = (
            m41_key in scheduler_probe,
            m41_key in migration_probe,
            f"{m41_key}_cid" in seal,
        )
        m40_key = "failed_pre_authoritative_m39_successor_materialization"
        m40_presence = (
            m40_key in scheduler_probe,
            m40_key in migration_probe,
            f"{m40_key}_cid" in seal,
        )
        m39_key = "committed_m38_evidence_reconciliation_successor_materialization"
        m39_presence = (
            m39_key in scheduler_probe,
            m39_key in migration_probe,
            f"{m39_key}_cid" in seal,
        )
        m38_key = "pre_authoritative_custody_restart_successor_materialization"
        m38_presence = (
            m38_key in scheduler_probe,
            m38_key in migration_probe,
            f"{m38_key}_cid" in seal,
        )
        m37_key = "post_reboot_generation_restart_successor_materialization"
        m37_presence = (
            m37_key in scheduler_probe,
            m37_key in migration_probe,
            f"{m37_key}_cid" in seal,
        )
        m36_key = "operator_task_binding_correction_successor_materialization"
        m36_presence = (
            m36_key in scheduler_probe,
            m36_key in migration_probe,
            f"{m36_key}_cid" in seal,
        )
        m35_key = "immutable_authority_identity_normalization_successor_materialization"
        m35_presence = (
            m35_key in scheduler_probe,
            m35_key in migration_probe,
            f"{m35_key}_cid" in seal,
        )
        m34_key = "json_emission_normalization_successor_materialization"
        m34_presence = (
            m34_key in scheduler_probe,
            m34_key in migration_probe,
            f"{m34_key}_cid" in seal,
        )
        m33_key = "live_preflight_contract_successor_materialization"
        m33_presence = (
            m33_key in scheduler_probe,
            m33_key in migration_probe,
            f"{m33_key}_cid" in seal,
        )
        m32_key = "live_preflight_plan_anchor_successor_materialization"
        m32_presence = (
            m32_key in scheduler_probe,
            m32_key in migration_probe,
            f"{m32_key}_cid" in seal,
        )
        m31_key = "detached_coordinator_pid_recovery_successor_materialization"
        m31_presence = (
            m31_key in scheduler_probe,
            m31_key in migration_probe,
            f"{m31_key}_cid" in seal,
        )
        m30_key = "stopped_owner_restart_source_seal_successor_materialization"
        m30_presence = (
            m30_key in scheduler_probe,
            m30_key in migration_probe,
            f"{m30_key}_cid" in seal,
        )
        m29_key = "committed_evidence_verification_successor_materialization"
        m29_presence = (
            m29_key in scheduler_probe,
            m29_key in migration_probe,
            f"{m29_key}_cid" in seal,
        )
        m28_key = "live_claim_admission_recovery_successor_materialization"
        m28_presence = (
            m28_key in scheduler_probe,
            m28_key in migration_probe,
            f"{m28_key}_cid" in seal,
        )
        m27_key = "dead_owner_parallel_resume_successor_materialization"
        m27_presence = (
            m27_key in scheduler_probe,
            m27_key in migration_probe,
            f"{m27_key}_cid" in seal,
        )
        m26_key = "automatic_stall_recovery_successor_materialization"
        m26_presence = (
            m26_key in scheduler_probe,
            m26_key in migration_probe,
            f"{m26_key}_cid" in seal,
        )
        m25_key = "native_duckdb_preload_successor_materialization"
        m25_presence = (
            m25_key in scheduler_probe,
            m25_key in migration_probe,
            f"{m25_key}_cid" in seal,
        )
        m24_key = "multi_lane_sidecar_reopen_successor_materialization"
        m24_presence = (
            m24_key in scheduler_probe,
            m24_key in migration_probe,
            f"{m24_key}_cid" in seal,
        )
        m23_key = "multi_lane_successor_materialization"
        m23_presence = (
            m23_key in scheduler_probe,
            m23_key in migration_probe,
            f"{m23_key}_cid" in seal,
        )
        m18_key = "portal_completion_persistence_successor_materialization"
        m18_presence = (
            m18_key in scheduler_probe,
            m18_key in migration_probe,
            f"{m18_key}_cid" in seal,
        )
        m17_key = "source_binding_successor_materialization"
        m17_presence = (
            m17_key in scheduler_probe,
            m17_key in migration_probe,
            f"{m17_key}_cid" in seal,
        )
        m16_key = "accepted_source_retry_successor_materialization"
        m16_presence = (
            m16_key in scheduler_probe,
            m16_key in migration_probe,
            f"{m16_key}_cid" in seal,
        )
        m15_key = "runtime_root_rebind_successor_materialization"
        m15_presence = (
            m15_key in scheduler_probe,
            m15_key in migration_probe,
            f"{m15_key}_cid" in seal,
        )
        m14_key = "stale_owner_restart_successor_materialization"
        m14_presence = (m14_key in scheduler_probe, m14_key in migration_probe, f"{m14_key}_cid" in seal)
        if any(m46_presence):
            scheduled = scheduler_probe.get(m46_key)
            migrated = migration_probe.get(m46_key)
            if not all(m46_presence) or scheduled != migrated:
                unexpected = ["M46 authority is partial or differs across controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m46_source_status_materializer",
                    root
                    / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M46 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer
                        ._expected_m46_legacy_no_delta_rescue_recovery_successor_authority()
                    )
                    materializer._validated_m46_live_preflight_contract(expected)
                    reference = dict(materializer._m46_authority_reference())
                    configured_cid = materializer._M46_AUTHORITY_CID
                    try:
                        materializer._assert_m46_historical_m45_controls(
                            scheduler_probe, migration_probe, seal
                        )
                    except Exception as exc:
                        unexpected = [
                            "M46 historical M45 controls differ: "
                            f"{type(exc).__name__}: {exc}"
                        ]
                    else:
                        if (
                            scheduled != reference
                            or seal.get(f"{m46_key}_cid") != configured_cid
                            or configured_cid == _M46_UNSEALED_AUTHORITY_CID
                            or configured_cid != _M46_AUTHORITY_CID
                            or materializer._identity(expected) != configured_cid
                            or len(materializer._canonical(expected))
                            != _M46_AUTHORITY_SIZE
                        ):
                            unexpected = ["M46 authority/CID differs across controls"]
                        else:
                            unexpected = _m46_source_chain_errors(
                                root, materializer, expected
                            )
        elif any(m45_presence):
            scheduled = scheduler_probe.get(m45_key)
            migrated = migration_probe.get(m45_key)
            if not all(m45_presence) or scheduled != migrated:
                unexpected = ["M45 authority is partial or differs across controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m45_source_status_materializer",
                    root
                    / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M45 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer
                        ._expected_m45_failed_pre_authoritative_m44_validation_successor_authority()
                    )
                    materializer._validated_m45_live_preflight_contract(expected)
                    reference = dict(materializer._m45_authority_reference())
                    configured_cid = materializer._M45_AUTHORITY_CID
                    try:
                        materializer._assert_m45_historical_m44_controls(
                            scheduler_probe, migration_probe, seal
                        )
                    except Exception as exc:
                        unexpected = [
                            "M45 historical M44 controls differ: "
                            f"{type(exc).__name__}: {exc}"
                        ]
                    else:
                        if (
                            scheduled != reference
                            or seal.get(f"{m45_key}_cid") != configured_cid
                            or configured_cid == _M45_UNSEALED_AUTHORITY_CID
                            or configured_cid != _M45_AUTHORITY_CID
                            or materializer._identity(expected) != configured_cid
                            or len(materializer._canonical(expected))
                            != _M45_AUTHORITY_SIZE
                        ):
                            unexpected = ["M45 authority/CID differs across controls"]
                        else:
                            unexpected = _m45_source_chain_errors(
                                root, materializer, expected
                            )
        elif any(m44_presence):
            scheduled = scheduler_probe.get(m44_key)
            migrated = migration_probe.get(m44_key)
            if not all(m44_presence) or scheduled != migrated:
                unexpected = ["M44 authority is partial or differs across controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m44_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M44 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer
                        ._expected_m44_post_m43_hardened_procfs_user_manager_restart_successor_authority()
                    )
                    reference = dict(materializer._m44_authority_reference())
                    configured_cid = materializer._M44_AUTHORITY_CID
                    if (
                        scheduled != reference
                        or seal.get(f"{m44_key}_cid") != configured_cid
                        or configured_cid == _M44_UNSEALED_AUTHORITY_CID
                        or configured_cid != _M44_AUTHORITY_CID
                        or materializer._identity(expected) != configured_cid
                    ):
                        unexpected = ["M44 authority/CID differs across controls"]
                    else:
                        unexpected = _m44_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m43_presence):
            scheduled = scheduler_probe.get(m43_key)
            migrated = migration_probe.get(m43_key)
            if not all(m43_presence) or scheduled != migrated:
                unexpected = ["M43 authority is partial or differs across controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m43_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M43 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer
                        ._expected_m43_dead_attempt_lifecycle_recovery_restart_authority()
                    )
                    reference = dict(materializer._m43_authority_reference())
                    configured_cid = materializer._M43_AUTHORITY_CID
                    if (
                        scheduled != reference
                        or seal.get(f"{m43_key}_cid") != configured_cid
                        or configured_cid == _M43_UNSEALED_AUTHORITY_CID
                        or configured_cid != _M43_AUTHORITY_CID
                        or materializer._identity(expected) != configured_cid
                    ):
                        unexpected = ["M43 authority/CID differs across controls"]
                    else:
                        unexpected = _m43_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m42_presence):
            scheduled = scheduler_probe.get(m42_key)
            migrated = migration_probe.get(m42_key)
            if not all(m42_presence) or scheduled != migrated:
                unexpected = ["M42 authority is partial or differs across controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m42_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M42 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer
                        ._expected_m42_failed_pre_authoritative_m41_evidence_projection_successor_authority()
                    )
                    reference = dict(materializer._m42_authority_reference())
                    expected_seal_cid = materializer._identity(expected)
                    if (
                        scheduled != reference
                        or seal.get(f"{m42_key}_cid") != expected_seal_cid
                        or expected_seal_cid != _M42_AUTHORITY_CID
                    ):
                        unexpected = ["M42 authority/CID differs across controls"]
                    else:
                        unexpected = _m42_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m41_presence):
            scheduled = scheduler_probe.get(m41_key)
            migrated = migration_probe.get(m41_key)
            if not all(m41_presence) or scheduled != migrated:
                unexpected = ["M41 authority is partial or differs across controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m41_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M41 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer._expected_m41_failed_pre_authoritative_m40_validation_successor_authority()
                    )
                    reference = dict(materializer._m41_authority_reference())
                    expected_seal_cid = materializer._identity(expected)
                    if (
                        scheduled != reference
                        or seal.get(f"{m41_key}_cid") != expected_seal_cid
                        or expected_seal_cid
                        != "sha256:25ad5550b59024c8da9b4821fba2d7b1b49d2a17a781e5d4bd2cbe58cb7b0233"
                    ):
                        unexpected = ["M41 authority/CID differs across controls"]
                    else:
                        unexpected = _m41_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m40_presence):
            scheduled = scheduler_probe.get(m40_key)
            migrated = migration_probe.get(m40_key)
            if not all(m40_presence) or scheduled != migrated:
                unexpected = ["M40 authority is partial or differs across controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m40_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M40 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer._expected_m40_failed_pre_authoritative_m39_successor_authority()
                    )
                    reference = dict(materializer._m40_authority_reference())
                    expected_seal_cid = materializer._identity(expected)
                    if (
                        scheduled != reference
                        or seal.get(f"{m40_key}_cid") != expected_seal_cid
                        or expected_seal_cid
                        != "sha256:377b6e7269a7025f236642f12aaf92264582f42a1efa4899eb4d6e64b0e41db2"
                    ):
                        unexpected = ["M40 authority/CID differs across controls"]
                    else:
                        unexpected = _m40_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m39_presence):
            scheduled = scheduler_probe.get(m39_key)
            migrated = migration_probe.get(m39_key)
            if not all(m39_presence) or scheduled != migrated:
                unexpected = ["M39 authority is partial or differs across controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m39_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M39 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer._expected_m39_committed_m38_evidence_reconciliation_authority()
                    )
                    reference = dict(materializer._m39_authority_reference())
                    expected_seal_cid = materializer._identity(expected)
                    if (
                        scheduled != reference
                        or seal.get(f"{m39_key}_cid") != expected_seal_cid
                        or expected_seal_cid
                        != "sha256:7b986174605b4f2739abf69473d0ce27dfe880d551c55144f076b8f981f633c2"
                    ):
                        unexpected = ["M39 authority/CID differs across controls"]
                    else:
                        unexpected = _m39_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m38_presence):
            scheduled = scheduler_probe.get(m38_key)
            migrated = migration_probe.get(m38_key)
            if not all(m38_presence) or scheduled != migrated:
                unexpected = ["M38 authority is partial or differs across controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m38_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M38 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer._expected_m38_pre_authoritative_custody_restart_authority()
                    )
                    identity_state = _m38_source_chain_identity_state(
                        materializer, expected
                    )
                    expected_seal_cid = (
                        "sha256:PENDING_M38_AUTHORITY_CID"
                        if identity_state == "placeholder"
                        else materializer._identity(expected)
                    )
                    if (
                        scheduled
                        != _m38_authority_reference_for_source_state(
                            materializer, expected
                        )
                        or seal.get(f"{m38_key}_cid") != expected_seal_cid
                    ):
                        unexpected = ["M38 authority/CID differs across controls"]
                    else:
                        unexpected = _m38_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m37_presence):
            scheduled = scheduler_probe.get(m37_key)
            migrated = migration_probe.get(m37_key)
            if not all(m37_presence) or scheduled != migrated:
                unexpected = ["M37 authority is partial or differs across source controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m37_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M37 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer._expected_m37_post_reboot_generation_restart_authority()
                    )
                    identity_state = _m37_source_chain_identity_state(
                        materializer, expected
                    )
                    expected_seal_cid = (
                        "sha256:PENDING_M37_AUTHORITY_CID"
                        if identity_state == "placeholder"
                        else materializer._identity(expected)
                    )
                    if (
                        scheduled
                        != _m37_authority_reference_for_source_state(
                            materializer, expected
                        )
                        or seal.get(f"{m37_key}_cid") != expected_seal_cid
                    ):
                        unexpected = ["M37 authority/CID differs across source controls"]
                    else:
                        unexpected = _m37_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m36_presence):
            scheduled = scheduler_probe.get(m36_key)
            migrated = migration_probe.get(m36_key)
            if not all(m36_presence) or scheduled != migrated:
                unexpected = ["M36 authority is partial or differs across source controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m36_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M36 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer._expected_m36_operator_task_binding_correction_authority()
                    )
                    if (
                        scheduled != materializer._m36_authority_reference()
                        or seal.get(f"{m36_key}_cid")
                        != materializer._identity(expected)
                    ):
                        unexpected = ["M36 authority/CID differs across source controls"]
                    else:
                        unexpected = _m36_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m35_presence):
            scheduled = scheduler_probe.get(m35_key)
            migrated = migration_probe.get(m35_key)
            if not all(m35_presence) or scheduled != migrated:
                unexpected = ["M35 authority is partial or differs across source controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m35_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M35 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer._expected_m35_immutable_authority_identity_normalization_authority()
                    )
                    if (
                        scheduled != materializer._m35_authority_reference()
                        or seal.get(f"{m35_key}_cid")
                        != materializer._identity(expected)
                    ):
                        unexpected = ["M35 authority/CID differs across source controls"]
                    else:
                        unexpected = _m35_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m34_presence):
            scheduled = scheduler_probe.get(m34_key)
            migrated = migration_probe.get(m34_key)
            if not all(m34_presence) or scheduled != migrated:
                unexpected = ["M34 authority is partial or differs across source controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m34_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M34 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = (
                        materializer._expected_m34_json_emission_normalization_authority()
                    )
                    if (
                        scheduled != materializer._m34_authority_reference()
                        or seal.get(f"{m34_key}_cid")
                        != materializer._identity(expected)
                    ):
                        unexpected = ["M34 authority/CID differs across source controls"]
                    else:
                        unexpected = _m34_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m33_presence):
            scheduled = scheduler_probe.get(m33_key)
            migrated = migration_probe.get(m33_key)
            if not all(m33_presence) or scheduled != migrated:
                unexpected = ["M33 authority is partial or differs across source controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m33_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M33 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = materializer._expected_m33_live_preflight_contract_authority()
                    if (
                        scheduled != materializer._m33_authority_reference()
                        or seal.get(f"{m33_key}_cid")
                        != materializer._identity(expected)
                    ):
                        unexpected = ["M33 authority/CID differs across source controls"]
                    else:
                        unexpected = _m33_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m32_presence):
            scheduled = scheduler_probe.get(m32_key)
            migrated = migration_probe.get(m32_key)
            if not all(m32_presence) or scheduled != migrated:
                unexpected = ["M32 authority is partial or differs across source controls"]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m32_source_status_materializer",
                    root / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M32 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    expected = materializer._expected_m32_live_preflight_plan_anchor_authority()
                    if (
                        scheduled != materializer._m32_authority_reference()
                        or seal.get(f"{m32_key}_cid")
                        != materializer._identity(expected)
                    ):
                        unexpected = ["M32 authority/CID differs across source controls"]
                    else:
                        unexpected = _m32_source_chain_errors(
                            root, materializer, expected
                        )
        elif any(m31_presence):
            scheduled = scheduler_probe.get(m31_key)
            migrated = migration_probe.get(m31_key)
            if (
                not all(m31_presence)
                or type(scheduled) is not dict
                or type(migrated) is not dict
                or _canonical_json(scheduled) != _canonical_json(migrated)
                or seal.get(f"{m31_key}_cid")
                != "sha256:"
                + hashlib.sha256(_canonical_json(scheduled)).hexdigest()
            ):
                unexpected = [
                    "M31 authority is partial or differs across source controls"
                ]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m31_source_status_materializer",
                    root
                    / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M31 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    unexpected = _m31_source_chain_errors(
                        root, materializer, scheduled
                    )
        elif any(m30_presence):
            scheduled = scheduler_probe.get(m30_key)
            migrated = migration_probe.get(m30_key)
            if (
                not all(m30_presence)
                or type(scheduled) is not dict
                or type(migrated) is not dict
                or _canonical_json(scheduled) != _canonical_json(migrated)
                or seal.get(f"{m30_key}_cid")
                != "sha256:"
                + hashlib.sha256(_canonical_json(scheduled)).hexdigest()
            ):
                unexpected = [
                    "M30 authority is partial or differs across source controls"
                ]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m30_source_status_materializer",
                    root
                    / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M30 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    unexpected = _m30_source_chain_errors(
                        root, materializer, scheduled
                    )
        elif any(m29_presence):
            scheduled = scheduler_probe.get(m29_key)
            migrated = migration_probe.get(m29_key)
            if (
                not all(m29_presence)
                or type(scheduled) is not dict
                or type(migrated) is not dict
                or _canonical_json(scheduled) != _canonical_json(migrated)
                or seal.get(f"{m29_key}_cid")
                != "sha256:"
                + hashlib.sha256(_canonical_json(scheduled)).hexdigest()
            ):
                unexpected = [
                    "M29 authority is partial or differs across source controls"
                ]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m29_source_status_materializer",
                    root
                    / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M29 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    unexpected = _m29_source_chain_errors(
                        root,
                        materializer,
                        scheduled,
                    )
        elif any(m28_presence):
            scheduled = scheduler_probe.get(m28_key)
            migrated = migration_probe.get(m28_key)
            if (
                not all(m28_presence)
                or type(scheduled) is not dict
                or type(migrated) is not dict
                or _canonical_json(scheduled) != _canonical_json(migrated)
                or seal.get(f"{m28_key}_cid")
                != "sha256:"
                + hashlib.sha256(_canonical_json(scheduled)).hexdigest()
            ):
                unexpected = [
                    "M28 authority is partial or differs across source controls"
                ]
            else:
                spec = importlib.util.spec_from_file_location(
                    "sawm_m28_source_status_materializer",
                    root
                    / "scripts/materialize_semantic_addressed_world_model_program.py",
                )
                if spec is None or spec.loader is None:
                    unexpected = ["M28 source materializer cannot be loaded"]
                else:
                    materializer = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(materializer)
                    unexpected = _m28_source_chain_errors(
                        root,
                        materializer,
                        scheduled,
                    )
        elif any(m27_presence):
            scheduled = scheduler_probe.get(m27_key)
            migrated = migration_probe.get(m27_key)
            if (
                not all(m27_presence)
                or type(scheduled) is not dict
                or type(migrated) is not dict
                or _canonical_json(scheduled) != _canonical_json(migrated)
                or seal.get(f"{m27_key}_cid")
                != "sha256:"
                + hashlib.sha256(_canonical_json(scheduled)).hexdigest()
            ):
                unexpected = [
                    "M27 authority is partial or differs across source controls"
                ]
            else:
                expected_paths = set(
                    str(path)
                    for path in scheduled.get("operator_control_paths", ())
                )
                observed: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(scheduled.get("precursor_source_head")),
                    "HEAD",
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed[path] = status
                working = set(_status_paths(root))
                expected = {path: "M" for path in expected_paths}
                unexpected = (
                    []
                    if (
                        observed == expected
                        and working.issubset(expected_paths)
                        and _git(root, "rev-parse", "HEAD^")
                        == scheduled.get("precursor_source_head")
                    )
                    else ["M27 status-qualified control source delta differs"]
                )
        elif any(m26_presence):
            scheduled = scheduler_probe.get(m26_key)
            migrated = migration_probe.get(m26_key)
            if (
                not all(m26_presence)
                or type(scheduled) is not dict
                or type(migrated) is not dict
                or _canonical_json(scheduled) != _canonical_json(migrated)
                or seal.get(f"{m26_key}_cid")
                != "sha256:"
                + hashlib.sha256(_canonical_json(scheduled)).hexdigest()
            ):
                unexpected = [
                    "M26 authority is partial or differs across source controls"
                ]
            else:
                expected_control_paths = set(
                    str(path)
                    for path in scheduled.get("operator_control_paths", ())
                )
                expected_control = {
                    path: "M" for path in expected_control_paths
                }
                observed_control: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(scheduled.get("repair_source_commit")),
                    "HEAD",
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed_control[path] = status
                expected_repair = {
                    str(path): "M"
                    for path in scheduled.get("source_repair_paths", ())
                }
                expected_repair[
                    "test/api/test_agent_supervisor_database_dispatch_watchdog.py"
                ] = "A"
                observed_repair: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(scheduled.get("prior_source_head")),
                    str(scheduled.get("repair_source_commit")),
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed_repair[path] = status
                working = set(_status_paths(root))
                parent_is_repair = (
                    _git(root, "rev-parse", "HEAD^")
                    == scheduled.get("repair_source_commit")
                )
                repair_parent_is_prior = (
                    _git(
                        root,
                        "rev-parse",
                        f"{scheduled.get('repair_source_commit')}^",
                    )
                    == scheduled.get("prior_source_head")
                )
                unexpected = (
                    []
                    if (
                        observed_control == expected_control
                        and observed_repair == expected_repair
                        and working.issubset(expected_control_paths)
                        and parent_is_repair
                        and repair_parent_is_prior
                    )
                    else ["M26 status-qualified two-commit source delta differs"]
                )
        elif any(m25_presence):
            scheduled = scheduler_probe.get(m25_key)
            migrated = migration_probe.get(m25_key)
            if (
                not all(m25_presence)
                or type(scheduled) is not dict
                or type(migrated) is not dict
                or _canonical_json(scheduled) != _canonical_json(migrated)
                or seal.get(f"{m25_key}_cid")
                != "sha256:"
                + hashlib.sha256(_canonical_json(scheduled)).hexdigest()
            ):
                unexpected = [
                    "M25 authority is partial or differs across source controls"
                ]
            else:
                expected_paths = set(
                    str(path)
                    for path in scheduled.get("operator_control_paths", ())
                )
                expected = {path: "M" for path in expected_paths}
                observed: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(scheduled.get("repair_source_commit")),
                    "HEAD",
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed[path] = status
                working = set(_status_paths(root))
                parent_is_repair = (
                    _git(root, "rev-parse", "HEAD^")
                    == scheduled.get("repair_source_commit")
                )
                repair_parent_is_m24 = (
                    _git(
                        root,
                        "rev-parse",
                        f"{scheduled.get('repair_source_commit')}^",
                    )
                    == scheduled.get("prior_source_head")
                )
                unexpected = (
                    []
                    if (
                        observed == expected
                        and working.issubset(expected_paths)
                        and parent_is_repair
                        and repair_parent_is_m24
                    )
                    else ["M25 status-qualified two-commit source delta differs"]
                )
        elif any(m24_presence):
            scheduled = scheduler_probe.get(m24_key)
            migrated = migration_probe.get(m24_key)
            if (
                not all(m24_presence)
                or type(scheduled) is not dict
                or type(migrated) is not dict
                or _canonical_json(scheduled) != _canonical_json(migrated)
                or seal.get(f"{m24_key}_cid")
                != "sha256:"
                + hashlib.sha256(_canonical_json(scheduled)).hexdigest()
            ):
                unexpected = [
                    "M24 authority is partial or differs across source controls"
                ]
            else:
                expected_paths = set(
                    str(path)
                    for path in scheduled.get("operator_control_paths", ())
                )
                expected = {path: "M" for path in expected_paths}
                observed: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(scheduled.get("repair_source_commit")),
                    "HEAD",
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed[path] = status
                working = set(_status_paths(root))
                unexpected = (
                    []
                    if observed == expected and working.issubset(expected_paths)
                    else ["M24 status-qualified bounded source delta differs"]
                )
        elif any(m23_presence):
            scheduled = scheduler_probe.get(m23_key)
            migrated = migration_probe.get(m23_key)
            if (
                not all(m23_presence)
                or type(scheduled) is not dict
                or type(migrated) is not dict
                or _canonical_json(scheduled) != _canonical_json(migrated)
                or seal.get(f"{m23_key}_cid")
                != "sha256:"
                + hashlib.sha256(_canonical_json(scheduled)).hexdigest()
            ):
                unexpected = [
                    "M23 authority is partial or differs across source controls"
                ]
            else:
                expected_paths = set(
                    str(path) for path in scheduled.get("operator_control_paths", ())
                )
                expected = {path: "M" for path in expected_paths}
                observed: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(scheduled.get("repair_source_commit")),
                    "HEAD",
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed[path] = status
                working = set(_status_paths(root))
                unexpected = (
                    []
                    if observed == expected and working.issubset(expected_paths)
                    else ["M23 status-qualified bounded source delta differs"]
                )
        elif any(m18_presence):
            if (
                not all(m18_presence)
                or not isinstance(scheduler_probe.get(m18_key), Mapping)
                or scheduler_probe.get(m18_key) != migration_probe.get(m18_key)
            ):
                unexpected = [
                    "M18 authority is partial or differs across source controls"
                ]
            else:
                authority = scheduler_probe[m18_key]
                expected_paths = set(
                    str(path)
                    for path in authority.get(
                        "bounded_control_plane_repair_paths", ()
                    )
                )
                expected = {path: "M" for path in expected_paths}
                observed: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(authority.get("runtime_source_head")),
                    "HEAD",
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed[path] = status
                working = set(_status_paths(root))
                unexpected = (
                    []
                    if observed == expected and working.issubset(expected_paths)
                    else ["M18 status-qualified bounded source delta differs"]
                )
        elif any(m17_presence):
            if (
                not all(m17_presence)
                or not isinstance(scheduler_probe.get(m17_key), Mapping)
                or scheduler_probe.get(m17_key) != migration_probe.get(m17_key)
            ):
                unexpected = [
                    "M17 authority is partial or differs across source controls"
                ]
            else:
                authority = scheduler_probe[m17_key]
                expected_paths = set(
                    str(path)
                    for path in authority.get(
                        "bounded_control_plane_repair_paths", ()
                    )
                )
                expected = {path: "M" for path in expected_paths}
                observed: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(authority.get("prior_source_head")),
                    "HEAD",
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed[path] = status
                working = set(_status_paths(root))
                unexpected = (
                    []
                    if observed == expected and working.issubset(expected_paths)
                    else ["M17 status-qualified bounded source delta differs"]
                )
        elif any(m16_presence):
            if (
                not all(m16_presence)
                or not isinstance(scheduler_probe.get(m16_key), Mapping)
                or scheduler_probe.get(m16_key) != migration_probe.get(m16_key)
            ):
                unexpected = [
                    "M16 authority is partial or differs across source controls"
                ]
            else:
                authority = scheduler_probe[m16_key]
                expected_paths = set(
                    str(path)
                    for path in authority.get(
                        "bounded_control_plane_repair_paths", ()
                    )
                )
                expected = {path: "M" for path in expected_paths}
                observed: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(authority.get("prior_source_head")),
                    "HEAD",
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed[path] = status
                working = set(_status_paths(root))
                unexpected = (
                    []
                    if observed == expected and working.issubset(expected_paths)
                    else ["M16 status-qualified bounded source delta differs"]
                )
        elif any(m15_presence):
            if (
                not all(m15_presence)
                or not isinstance(scheduler_probe.get(m15_key), Mapping)
                or scheduler_probe.get(m15_key) != migration_probe.get(m15_key)
            ):
                unexpected = ["M15 authority is partial or differs across source controls"]
            else:
                authority = scheduler_probe[m15_key]
                expected = {
                    str(path): "M"
                    for path in authority.get(
                        "bounded_control_plane_repair_paths", ()
                    )
                }
                observed: dict[str, str] = {}
                for line in _git(
                    root,
                    "diff",
                    "--name-status",
                    "--no-renames",
                    str(authority.get("prior_source_head")),
                    "HEAD",
                    "--",
                ).splitlines():
                    status, path = line.split("\t", 1)
                    observed[path] = status
                working = set(_status_paths(root))
                unexpected = (
                    []
                    if observed == expected and working.issubset(set(expected))
                    else ["M15 status-qualified bounded source delta differs"]
                )
        elif any(m14_presence):
            if not all(m14_presence) or not isinstance(scheduler_probe.get(m14_key), Mapping) or scheduler_probe.get(m14_key) != migration_probe.get(m14_key):
                unexpected = ["M14 authority is partial or differs across source controls"]
            else:
                authority = scheduler_probe[m14_key]
                expected = {
                    str(path): ("A" if str(path) in {
                        "docs/architecture/semantic_addressed_world_model_evidence/SAWM-001-authority-overlap-receipt.json",
                        "test/api/semantic_world/test_authority_overlap_runtime.py",
                    } else "M")
                    for path in authority.get("bounded_control_plane_repair_paths", ())
                }
                observed: dict[str, str] = {}
                for line in _git(root, "diff", "--name-status", "--no-renames", str(authority.get("prior_source_head")), "HEAD", "--").splitlines():
                    status, path = line.split("\t", 1)
                    observed[path] = status
                working = set(_status_paths(root))
                unexpected = [] if observed == expected and working.issubset(set(expected)) else [
                    "M14 status-qualified bounded source delta differs",
                ]
        else:
            changed = set(_git(root, "diff", "--name-only", str(accel.get("head")), "--").splitlines())
            changed.update(_status_paths(root))
            unexpected = sorted(path for path in changed if path and path not in CONTROL_PATHS)
        check("accelerator_source", branch == accel.get("branch") and origin == accel.get("origin") and ancestor_ok and not unexpected,
              {"head": head, "branch": branch, "origin": origin, "unexpected_changes": unexpected})
    except Exception as exc:
        check("accelerator_source", False, f"{type(exc).__name__}: {exc}")

    effective_authorities, gitlink_errors = _effective_nested_source_authorities(
        authorities,
        scheduler_probe,
        migration_probe,
        seal,
    )
    for package in ("ipfs_datasets_py", "ipfs_kit_py"):
        authority = effective_authorities.get(package, {})
        nested = root / package
        try:
            index_oid = _git(root, "rev-parse", f"HEAD:{package}")
            nested_head = _git(root, "rev-parse", "HEAD", cwd=nested)
            nested_tree = _git(root, "rev-parse", "HEAD^{tree}", cwd=nested)
            nested_status = _git(root, "status", "--porcelain=v1", "--untracked-files=all", cwd=nested)
            version = _project_version(nested / "pyproject.toml")
            expected_origin = str(authority.get("origin") or "")
            actual_origin = _git(root, "remote", "get-url", "origin", cwd=nested)
            if not (
                index_oid == authority.get("gitlink_commit") == nested_head
                and nested_tree == authority.get("tree")
                and version == authority.get("package_version")
                and not nested_status
                and actual_origin == expected_origin
            ):
                gitlink_errors.append(f"{package}: index={index_oid} head={nested_head} tree={nested_tree} version={version} status={bool(nested_status)} origin={actual_origin}")
        except Exception as exc:
            gitlink_errors.append(f"{package}: {type(exc).__name__}: {exc}")
    check("gitlinks_and_nested_authorities", not gitlink_errors, gitlink_errors)

    dependency_errors: list[str] = []
    for item in seal.get("dependency_files") or ():
        if not isinstance(item, Mapping):
            dependency_errors.append("non-object dependency entry")
            continue
        path = root / str(item.get("path") or "")
        try:
            observed_sha = _sha(path)
            observed_blob = _git(root, "hash-object", str(path))
            if observed_sha != item.get("sha256") or observed_blob != item.get("git_blob_oid"):
                dependency_errors.append(f"{item.get('path')}: sha256={observed_sha} blob={observed_blob}")
        except Exception as exc:
            dependency_errors.append(f"{item.get('path')}: {type(exc).__name__}: {exc}")
    check("dependency_file_hashes", not dependency_errors, dependency_errors)

    policy = seal.get("environment_policy") if isinstance(seal.get("environment_policy"), Mapping) else {}
    fixed = policy.get("fixed") if isinstance(policy.get("fixed"), Mapping) else {}
    toolchain = seal.get("toolchain") if isinstance(seal.get("toolchain"), Mapping) else {}
    launch_python = Path(str(toolchain.get("launch_python_executable") or ""))
    legacy_python = Path(str(toolchain.get("python_executable") or ""))
    launch_toolchain_errors: list[str] = []
    try:
        launch_evidence = _stable_regular_evidence(launch_python, maximum=64 * 1024 * 1024)
    except Exception as exc:
        launch_evidence = {}
        launch_toolchain_errors.append(
            f"launch interpreter unavailable: {type(exc).__name__}: {exc}"
        )
    try:
        legacy_python_sha256 = _sha(legacy_python)
    except Exception as exc:
        legacy_python_sha256 = ""
        launch_toolchain_errors.append(
            f"reviewed interpreter entry point unavailable: {type(exc).__name__}: {exc}"
        )
    if (
        str(launch_python) != "/usr/bin/python3.12"
        or not launch_python.is_absolute()
        or launch_evidence.get("sha256")
        != "sha256:" + str(toolchain.get("launch_python_sha256") or "")
        or legacy_python_sha256 != toolchain.get("python_sha256")
        or toolchain.get("launch_python_sha256") != toolchain.get("python_sha256")
        or policy.get("python_invocation_flags") != ["-I", "-S", "-B"]
        or toolchain.get("implicit_install_allowed") is not False
        or toolchain.get("implicit_model_download_allowed") is not False
    ):
        launch_toolchain_errors.append("sealed launch interpreter declaration is invalid")
    check(
        "sealed_launch_toolchain_declaration",
        not launch_toolchain_errors,
        {
            "errors": launch_toolchain_errors,
            "launch_python_executable": str(launch_python),
            "launch_python_evidence": launch_evidence,
            "legacy_entry_point": str(legacy_python),
            "legacy_entry_point_sha256": legacy_python_sha256,
            "python_invocation_flags": policy.get("python_invocation_flags"),
            "pytest_distribution_version": {
                "declared": toolchain.get("pytest_distribution_version"),
                "authority": "test-only; excluded from launch admission",
            },
        },
    )

    mode_closure = (
        _control_plane_mode_closure(root, str(launch_python), fixed)
        if not launch_toolchain_errors
        else {"valid": False, "error": "launch toolchain is unavailable"}
    )
    check(
        "accepted_control_plane_mode_closure",
        mode_closure.get("valid") is True,
        mode_closure,
    )

    try:
        scheduler = _load(
            root / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        )
        scheduler_error = ""
    except Exception as exc:
        scheduler = {}
        scheduler_error = f"{type(exc).__name__}: {exc}"

    validation_runtime = (
        _validation_runtime_closure(root, scheduler, seal)
        if not scheduler_error
        else {
            "valid": False,
            "errors": [f"scheduler unavailable: {scheduler_error}"],
        }
    )
    check(
        "exact_immutable_validation_runtime_closure",
        validation_runtime.get("valid") is True,
        validation_runtime,
    )

    native, authorization, native_errors = _validate_native_authorization(
        root,
        seal,
        scheduler,
        toolchain,
    )
    if scheduler_error:
        native_errors.append(f"scheduler unavailable: {scheduler_error}")
    check(
        "independent_native_dependency_authorization",
        not native_errors,
        {
            "errors": native_errors,
            "authorization_id": authorization.get("authorization_id"),
            "inspection_is_authority": authorization.get("inspection_is_authority"),
        },
    )

    projections, projection_errors = _validate_extension_projection_pins(
        seal,
        scheduler,
    )
    if scheduler_error:
        projection_errors.append(f"scheduler unavailable: {scheduler_error}")
    check(
        "quack_httpfs_projection_pins",
        not projection_errors,
        {
            "errors": projection_errors,
            "projection_ids": {
                name: item.get("projection_id")
                for name, item in sorted(projections.items())
            },
        },
    )

    probe_ready = not launch_toolchain_errors and not native_errors and not projection_errors
    if probe_ready:
        try:
            dependency_probe = _isolated_native_extension_probe(
                root,
                launch_python=launch_python,
                expected_python_sha256=str(toolchain.get("launch_python_sha256") or ""),
                fixed=fixed,
                native=native,
                authorization=authorization,
                projections=projections,
            )
        except Exception as exc:
            dependency_probe = {
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        dependency_probe = {
            "valid": False,
            "skipped": "invalid toolchain, authorization, or projection prerequisite",
        }
    expected_isolated_toolchain = {
        "python_executable": str(launch_python),
        "python_executable_sha256": toolchain.get("launch_python_sha256"),
        "python_implementation": toolchain.get("python_implementation"),
        "python_version": toolchain.get("python_version"),
        "operating_system": toolchain.get("operating_system"),
        "machine": toolchain.get("machine"),
        "duckdb_distribution_version": toolchain.get("duckdb_distribution_version"),
    }
    observed_isolated_toolchain = {
        key: dependency_probe.get(key) for key in expected_isolated_toolchain
    }
    check(
        "isolated_launch_toolchain",
        dependency_probe.get("valid") is True
        and observed_isolated_toolchain == expected_isolated_toolchain
        and dependency_probe.get("argv_flags") == ["-I", "-S", "-B"]
        and dependency_probe.get("ambient_specs")
        == {"duckdb": False, "_duckdb": False, "pytest": False},
        {
            "expected": expected_isolated_toolchain,
            "observed": observed_isolated_toolchain,
            "argv_flags": dependency_probe.get("argv_flags"),
            "ambient_specs": dependency_probe.get("ambient_specs"),
            "failure": dependency_probe if dependency_probe.get("valid") is not True else None,
        },
    )
    native_pin = native.get("pin") if type(native.get("pin")) is dict else {}
    check(
        "recomputed_native_dependency_pin",
        dependency_probe.get("valid") is True
        and dependency_probe.get("native_dependency_id") == native_pin.get("dependency_id")
        and dependency_probe.get("native_payload_sha256") == native_pin.get("payload_sha256")
        and str(dependency_probe.get("native_module_origin") or "").startswith(
            "/proc/self/fd/"
        ),
        {
            "expected_dependency_id": native_pin.get("dependency_id"),
            "observed_dependency_id": dependency_probe.get("native_dependency_id"),
            "expected_payload_sha256": native_pin.get("payload_sha256"),
            "observed_payload_sha256": dependency_probe.get("native_payload_sha256"),
            "module_origin": dependency_probe.get("native_module_origin"),
        },
    )
    check(
        "isolated_duckdb_quack_httpfs_load",
        dependency_probe.get("valid") is True
        and dependency_probe.get("database_opened") is True
        and dependency_probe.get("network_required") is False
        and dependency_probe.get("select_42") == 42
        and dependency_probe.get("settings") == [False, False, False, False]
        and len(dependency_probe.get("extension_rows") or ()) == 2,
        dependency_probe,
    )

    interface_errors: list[str] = []
    for relative, names in INTERFACES:
        try:
            tree = ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
            observed = {node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
            missing = sorted(set(names) - observed)
            if missing:
                interface_errors.append(f"{relative}: missing {missing}")
        except Exception as exc:
            interface_errors.append(f"{relative}: {type(exc).__name__}: {exc}")
    check("landed_interfaces", not interface_errors, interface_errors)

    policy_ok = (
        policy.get("secrets_source") == "environment_only"
        and policy.get("secrets_in_prompt_argv_log_receipt_board_corpus_model") is False
        and policy.get("ordinary_test_network") is False
        and policy.get("cold_import_side_effects_allowed") is False
        and policy.get("arbitrary_code_deserialization_allowed") is False
        and policy.get("python_invocation_flags") == ["-I", "-S", "-B"]
        and policy.get("duckdb_extension_home")
        == "exact_private_read_only_projection"
        and all(
            str(fixed.get(key)) == value
            for key, value in {
                "IPFS_DATASETS_AUTO_INSTALL": "0",
                "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
                "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY": "disabled",
                "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
                "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }.items()
        )
    )
    check("offline_environment_policy", policy_ok, policy)

    direction = seal.get("package_dependency_direction") if isinstance(seal.get("package_dependency_direction"), Mapping) else {}
    check("authority_dependency_direction", direction.get("accelerate_consumes_datasets_semantics") is True and direction.get("accelerate_consumes_kit_storage") is True and direction.get("datasets_may_depend_on_accelerate_operations") is False and direction.get("kit_may_decide_datasets_semantics") is False and direction.get("parallel_authority_implementation_allowed") is False, direction)
    plane = seal.get("control_plane") if isinstance(seal.get("control_plane"), Mapping) else {}
    check(
        "duckdb_quack_ducklake_boundaries",
        plane.get("authoritative_store") == "DuckDB"
        and plane.get("exclusive_mutation_owner") == "canonical DuckDB writer"
        and plane.get("multi_process_query_transport") == "Quack read-only replica"
        and plane.get("authenticated_mutation_protocol") == "closed atomic owner-inbox protocol@2"
        and plane.get("canonical_writer_served_through_quack") is False
        and plane.get("quack_replica_authoritative") is False
        and plane.get("owner_protocol_allows_arbitrary_sql") is False
        and plane.get("ducklake_authoritative") is False
        and plane.get("markdown_authoritative") is False
        and plane.get("worker_self_completion_allowed") is False,
        plane,
    )

    protocol_errors: list[str] = []
    try:
        scheduler = _load(
            root / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        )
        migration = _load(
            root
            / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json"
        )
        configured_pin = scheduler.get("quack_owner", {}).get("pinned_extension", {})
        sealed_pin = seal.get("quack_extension_pin", {})
        pin_path = Path(str(configured_pin.get("path") or ""))
        if configured_pin != sealed_pin:
            protocol_errors.append("scheduler Quack pin differs from the dependency seal")
        if not pin_path.is_file() or _sha(pin_path) != configured_pin.get("sha256"):
            protocol_errors.append("reviewed local Quack extension bytes are unavailable or mismatched")
        if configured_pin.get("network_install_allowed") is not False:
            protocol_errors.append("Quack network installation must remain forbidden")
        if configured_pin.get("unsigned_extension_allowed") is not False:
            protocol_errors.append("unsigned Quack extensions must remain forbidden")

        protocol_errors.extend(_m6_source_migration_errors(scheduler, seal, migration))
        protocol_errors.extend(_m7_source_repair_errors(scheduler, seal, migration))
        protocol_errors.extend(_m8_source_repair_errors(scheduler, seal, migration))
        protocol_errors.extend(_m9_live_recovery_errors(scheduler, seal, migration))
        protocol_errors.extend(_m10_live_projection_errors(scheduler, seal, migration))
        protocol_errors.extend(_m11_provider_retry_errors(scheduler, seal, migration))
        protocol_errors.extend(
            _m12_declared_output_retry_errors(scheduler, seal, migration)
        )
        m46_declared = _m46_successor_declared(scheduler, seal, migration)
        m45_declared = _m45_successor_declared(scheduler, seal, migration)
        m44_declared = _m44_successor_declared(scheduler, seal, migration)
        m43_declared = _m43_successor_declared(scheduler, seal, migration)
        m42_declared = _m42_successor_declared(scheduler, seal, migration)
        m41_declared = _m41_successor_declared(scheduler, seal, migration)
        m40_declared = _m40_successor_declared(scheduler, seal, migration)
        m39_declared = _m39_successor_declared(scheduler, seal, migration)
        m38_declared = _m38_successor_declared(scheduler, seal, migration)
        m37_declared = _m37_successor_declared(scheduler, seal, migration)
        m36_declared = _m36_successor_declared(scheduler, seal, migration)
        m35_declared = _m35_successor_declared(scheduler, seal, migration)
        m34_declared = _m34_successor_declared(scheduler, seal, migration)
        m33_declared = _m33_successor_declared(scheduler, seal, migration)
        m32_declared = _m32_successor_declared(scheduler, seal, migration)
        m31_declared = _m31_successor_declared(scheduler, seal, migration)
        m30_declared = _m30_successor_declared(scheduler, seal, migration)
        m29_declared = _m29_successor_declared(scheduler, seal, migration)
        m28_declared = _m28_successor_declared(scheduler, seal, migration)
        m27_declared = _m27_successor_declared(scheduler, seal, migration)
        m26_declared = _m26_successor_declared(scheduler, seal, migration)
        if (
            m46_declared
            or m45_declared
            or m44_declared
            or m43_declared
            or m42_declared
            or m41_declared
            or m40_declared
            or m39_declared
            or m38_declared
            or m37_declared
            or m36_declared
            or m35_declared
            or m34_declared
            or m33_declared
            or m32_declared
            or m31_declared
            or m30_declared
            or m29_declared
            or m28_declared
            or m27_declared
            or m26_declared
            or _m25_successor_declared(scheduler, seal, migration)
        ):
            if m46_declared:
                protocol_errors.extend(
                    _m46_legacy_no_delta_rescue_recovery_successor_errors(
                        scheduler, seal, migration, root=root
                    )
                )
            if m45_declared and not m46_declared:
                protocol_errors.extend(
                    _m45_failed_pre_authoritative_m44_validation_successor_errors(
                        scheduler, seal, migration, root=root
                    )
                )
            if m44_declared and not m45_declared and not m46_declared:
                protocol_errors.extend(
                    _m44_post_m43_hardened_procfs_user_manager_restart_successor_errors(
                        scheduler, seal, migration, root=root
                    )
                )
            if (
                m43_declared
                and not m44_declared
                and not m45_declared
                and not m46_declared
            ):
                protocol_errors.extend(
                    _m43_dead_attempt_lifecycle_recovery_restart_successor_errors(
                        scheduler, seal, migration, root=root
                    )
                )
            if (
                m42_declared
                and not m43_declared
                and not m44_declared
                and not m45_declared
                and not m46_declared
            ):
                protocol_errors.extend(
                    _m42_failed_pre_authoritative_m41_evidence_projection_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                    )
                )
            if m41_declared and not m42_declared:
                protocol_errors.extend(
                    _m41_failed_pre_authoritative_m40_validation_successor_errors(
                        scheduler, seal, migration, root=root
                    )
                )
            if m40_declared and not m41_declared and not m42_declared:
                protocol_errors.extend(
                    _m40_failed_pre_authoritative_m39_successor_errors(
                        scheduler, seal, migration, root=root
                    )
                )
            if (
                m39_declared
                and not m40_declared
                and not m41_declared
                and not m42_declared
            ):
                protocol_errors.extend(
                    _m39_committed_m38_evidence_reconciliation_successor_errors(
                        scheduler, seal, migration, root=root
                    )
                )
            if (
                m38_declared
                and not m39_declared
                and not m40_declared
                and not m41_declared
                and not m42_declared
            ):
                protocol_errors.extend(
                    _m38_pre_authoritative_custody_restart_successor_errors(
                        scheduler, seal, migration, root=root
                    )
                )
            if (
                m37_declared
                and not m38_declared
                and not m39_declared
                and not m40_declared
                and not m41_declared
                and not m42_declared
            ):
                protocol_errors.extend(
                    _m37_post_reboot_generation_restart_successor_errors(
                        scheduler, seal, migration, root=root
                    )
                )
            if m36_declared:
                protocol_errors.extend(
                    _m36_operator_task_binding_correction_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m42_declared
                            or m41_declared
                            or m40_declared
                            or m39_declared
                            or m38_declared
                            or m37_declared
                        ),
                    )
                )
            if m35_declared:
                protocol_errors.extend(
                    _m35_immutable_authority_identity_normalization_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m38_declared or m37_declared or m36_declared
                        ),
                    )
                )
            if m34_declared:
                protocol_errors.extend(
                    _m34_json_emission_normalization_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m37_declared or m36_declared or m35_declared
                        ),
                    )
                )
            if m33_declared:
                protocol_errors.extend(
                    _m33_live_preflight_contract_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m37_declared
                            or m36_declared or m35_declared or m34_declared
                        ),
                    )
                )
            if m32_declared:
                protocol_errors.extend(
                    _m32_live_preflight_plan_anchor_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m37_declared
                            or m36_declared
                            or m35_declared or m34_declared or m33_declared
                        ),
                    )
                )
            if m31_declared:
                protocol_errors.extend(
                    _m31_detached_coordinator_pid_recovery_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m37_declared
                            or m36_declared
                            or m35_declared
                            or m34_declared or m33_declared or m32_declared
                        ),
                    )
                )
            if m30_declared:
                protocol_errors.extend(
                    _m30_stopped_owner_restart_source_seal_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m37_declared
                            or m36_declared
                            or m35_declared
                            or m34_declared
                            or m33_declared or m32_declared or m31_declared
                        ),
                    )
                )
            if m29_declared:
                protocol_errors.extend(
                    _m29_committed_evidence_verification_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m37_declared
                            or m36_declared
                            or m35_declared
                            or m34_declared
                            or m33_declared
                            or m32_declared
                            or m31_declared
                            or m30_declared
                        ),
                    )
                )
            if m28_declared:
                protocol_errors.extend(
                    _m28_live_claim_admission_recovery_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m37_declared
                            or m36_declared
                            or m35_declared
                            or m34_declared
                            or m33_declared
                            or m32_declared
                            or m31_declared
                            or m30_declared
                            or m29_declared
                        ),
                    )
                )
            if m27_declared:
                protocol_errors.extend(
                    _m27_dead_owner_parallel_resume_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m37_declared
                            or m36_declared
                            or m35_declared
                            or m34_declared
                            or m33_declared
                            or m32_declared
                            or m31_declared
                            or m30_declared
                            or m29_declared
                            or m28_declared
                        ),
                    )
                )
            if m26_declared:
                protocol_errors.extend(
                    _m26_automatic_stall_recovery_successor_errors(
                        scheduler,
                        seal,
                        migration,
                        root=root,
                        require_active_runtime=not (
                            m37_declared
                            or m36_declared
                            or m35_declared
                            or m34_declared
                            or m33_declared
                            or m32_declared
                            or m31_declared
                            or m30_declared
                            or m29_declared
                            or m28_declared
                            or m27_declared
                        ),
                    )
                )
            protocol_errors.extend(
                _m25_native_duckdb_preload_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=not (
                        m37_declared
                        or m36_declared
                        or m35_declared
                        or m34_declared
                        or m33_declared
                        or m32_declared
                        or m31_declared
                        or m30_declared
                        or m29_declared
                        or m28_declared
                        or m27_declared
                        or m26_declared
                    ),
                )
            )
            protocol_errors.extend(
                _m24_sidecar_reopen_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m23_multi_lane_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m22_live_preflight_receipt_compatibility_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m21_generation_realization_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m20_test_isolation_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m19_live_catalog_inventory_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m18_portal_completion_persistence_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m17_source_binding_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif _m24_successor_declared(scheduler, seal, migration):
            protocol_errors.extend(
                _m24_sidecar_reopen_successor_errors(
                    scheduler, seal, migration, root=root
                )
            )
            protocol_errors.extend(
                _m23_multi_lane_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m22_live_preflight_receipt_compatibility_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m21_generation_realization_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m20_test_isolation_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m19_live_catalog_inventory_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m18_portal_completion_persistence_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m17_source_binding_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif _m23_successor_declared(scheduler, seal, migration):
            protocol_errors.extend(
                _m23_multi_lane_successor_errors(
                    scheduler, seal, migration, root=root
                )
            )
            protocol_errors.extend(
                _m22_live_preflight_receipt_compatibility_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m21_generation_realization_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m20_test_isolation_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m19_live_catalog_inventory_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m18_portal_completion_persistence_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m17_source_binding_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif _m22_successor_declared(scheduler, seal, migration):
            protocol_errors.extend(
                _m22_live_preflight_receipt_compatibility_successor_errors(
                    scheduler, seal, migration, root=root
                )
            )
            protocol_errors.extend(
                _m21_generation_realization_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m20_test_isolation_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m19_live_catalog_inventory_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m18_portal_completion_persistence_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m17_source_binding_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif _m21_successor_declared(scheduler, seal, migration):
            protocol_errors.extend(
                _m21_generation_realization_successor_errors(
                    scheduler, seal, migration, root=root
                )
            )
            protocol_errors.extend(
                _m20_test_isolation_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m19_live_catalog_inventory_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m18_portal_completion_persistence_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m17_source_binding_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif _m20_successor_declared(scheduler, seal, migration):
            protocol_errors.extend(
                _m20_test_isolation_successor_errors(
                    scheduler, seal, migration, root=root
                )
            )
            protocol_errors.extend(
                _m19_live_catalog_inventory_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m18_portal_completion_persistence_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m17_source_binding_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif "live_catalog_inventory_successor_materialization" in scheduler:
            protocol_errors.extend(
                _m19_live_catalog_inventory_successor_errors(
                    scheduler, seal, migration, root=root
                )
            )
            protocol_errors.extend(
                _m18_portal_completion_persistence_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m17_source_binding_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif "portal_completion_persistence_successor_materialization" in scheduler:
            protocol_errors.extend(
                _m18_portal_completion_persistence_errors(
                    scheduler, seal, migration, root=root
                )
            )
            protocol_errors.extend(
                _m17_source_binding_successor_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif "source_binding_successor_materialization" in scheduler:
            protocol_errors.extend(
                _m17_source_binding_successor_errors(
                    scheduler, seal, migration, root=root
                )
            )
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler,
                    seal,
                    migration,
                    root=root,
                    require_active_runtime=False,
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif "accepted_source_retry_successor_materialization" in scheduler:
            protocol_errors.extend(
                _m16_accepted_source_retry_errors(
                    scheduler, seal, migration, root=root
                )
            )
            protocol_errors.extend(
                _m15_historical_authority_errors(scheduler, seal, migration)
            )
        elif "runtime_root_rebind_successor_materialization" in scheduler:
            protocol_errors.extend(
                _m15_runtime_root_rebind_errors(scheduler, seal, migration)
            )
        elif "stale_owner_restart_successor_materialization" in scheduler:
            protocol_errors.extend(_m14_stale_owner_restart_errors(scheduler, seal, migration))
        else:
            protocol_errors.extend(_m13_quack_refresh_errors(scheduler, seal, migration))

        protocol_source = (
            root / "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py"
        ).read_text(encoding="utf-8")
        operator_source = (
            root / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"
        ).read_text(encoding="utf-8")
        runner_source = (
            root / "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py"
        ).read_text(encoding="utf-8")
        daemon_source = (
            root / "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py"
        ).read_text(encoding="utf-8")
        if "quack-owner-mutation-request@2" not in protocol_source:
            protocol_errors.append("closed mutation protocol revision 2 is absent")
        if "MUTATION_SQL_TEMPLATES" not in protocol_source or "execute_owner_mutation" not in protocol_source:
            protocol_errors.append("closed atomic mutation catalog is absent")
        if "read_only=True" not in operator_source or "canonical writer without loading or serving Quack" not in operator_source:
            protocol_errors.append("read-only Quack replica / sealed writer boundary is absent")
        if not _has_presence_based_key_selection(
            operator_source,
            "detached_coordinator_pid_recovery_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M31 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "stopped_owner_restart_source_seal_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M30 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "committed_evidence_verification_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M29 authority by fail-closed key "
                "presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "live_claim_admission_recovery_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M28 authority by fail-closed key "
                "presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "native_duckdb_preload_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M25 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "multi_lane_sidecar_reopen_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M24 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "multi_lane_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M23 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "live_preflight_receipt_compatibility_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M22 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "generation_realization_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M21 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "portal_completion_persistence_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M18 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "live_recovery_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M9 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "live_projection_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M10 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "live_provider_retry_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M11 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "declared_output_retry_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M12 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "source_binding_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M17 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "accepted_source_retry_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M16 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "runtime_root_rebind_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M15 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "stale_owner_restart_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M14 authority by fail-closed key presence"
            )
        if not _has_presence_based_key_selection(
            operator_source,
            "quack_refresh_successor_materialization",
        ):
            protocol_errors.append(
                "operator does not select the M13 authority by fail-closed key presence"
            )
        for name in (
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_BINDING",
        ):
            if name not in runner_source:
                protocol_errors.append(f"state credential scrub list omits {name}")
        if daemon_source.count("inherit_environment=False") < 3:
            protocol_errors.append("provider subprocesses do not all use exact scrubbed environments")
    except Exception as exc:
        protocol_errors.append(f"{type(exc).__name__}: {exc}")
    check("append_only_migration_and_closed_quack_protocol", not protocol_errors, protocol_errors)

    if cold_import and not errors:
        # Use the sealed interpreter, not whichever Python happened to invoke a
        # caller.  The probe prepends explicit source roots because -I ignores
        # PYTHONPATH by design.
        python = str(toolchain.get("launch_python_executable") or "")
        result = _cold_import(root, python, fixed)
        check("cold_import_side_effects", result.get("valid") is True, result)
    else:
        checks.append({"name": "cold_import_side_effects", "passed": None, "detail": "skipped after prior failure or by request"})

    return {"schema": SCHEMA, "valid": not errors, "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "errors": errors, "checks": checks,
            "database_opened": dependency_probe.get("database_opened") is True,
            "network_required": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--skip-cold-import", action="store_true", help="diagnostic only; not a launch gate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_dependencies(args.repo_root, cold_import=not args.skip_cold_import)
    except Exception as exc:
        report = {"schema": SCHEMA, "valid": False, "board_namespace": NAMESPACE,
                  "plan_revision": REVISION, "errors": [f"unhandled_validator_error: {type(exc).__name__}: {exc}"], "checks": []}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
