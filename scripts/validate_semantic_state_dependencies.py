#!/usr/bin/env python3
"""Fail-closed validator for the semantic-compression harness dependency seal.

This is an operator control-plane drift gate, not a content-identity
implementation. Git object IDs and SHA-256 policy fingerprints identify the
exact reviewed dependency surfaces. Domain CIDs remain owned by
``ipfs_datasets_py``; generic MCP++ wire identity remains owned by the pinned
Profile A/B/F implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SEAL_SCHEMA = "ipfs-accelerate.agent-supervisor.semantic-state-dependency-seal@2"
EXPECTED_ROLES = (
    "accelerate_harness",
    "incremental_semantic_index",
    "semantic_state_contracts",
    "kit_state_roots",
    "mcp_plus_plus",
)
EXPECTED_REPOSITORIES = {
    "accelerate_harness": "endomorphosis/ipfs_accelerate_py",
    "incremental_semantic_index": "endomorphosis/ipfs_datasets_py",
    "semantic_state_contracts": "endomorphosis/ipfs_datasets_py",
    "kit_state_roots": "endomorphosis/ipfs_kit_py",
    "mcp_plus_plus": "endomorphosis/Mcp-Plus-Plus",
}
EXPECTED_ORIGINS = {
    role: f"https://github.com/{repository}" for role, repository in EXPECTED_REPOSITORIES.items()
}
EXPECTED_COMMITS = {
    "accelerate_harness": "UNRESOLVED_REPAIRED_ACCELERATE_COMMIT",
    "incremental_semantic_index": "UNRESOLVED_FINAL_ISI_COMMIT",
    "semantic_state_contracts": "UNRESOLVED_FINAL_DSS_COMMIT",
    "kit_state_roots": "05ba9375923cd5fb52e2c9c18b98b530d57d077f",
    "mcp_plus_plus": "dc3164653a48d059ae9812078359daeafb451c07",
}
EXPECTED_TREES = {
    "accelerate_harness": "UNRESOLVED_REPAIRED_ACCELERATE_TREE",
    "incremental_semantic_index": "UNRESOLVED_FINAL_ISI_TREE",
    "semantic_state_contracts": "UNRESOLVED_FINAL_DSS_TREE",
    "kit_state_roots": "a770206fe9e11852a9a230b9ce64d0cce254dd50",
    "mcp_plus_plus": "6560c3d0c926be12df860afb7d7c82043a1769ba",
}
REACHABILITY_POLICY = "exact_clean_head"

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "status",
        "unresolved_authority_reasons",
        "target",
        "toolchain",
        "wire_contract",
        "authorities",
    }
)
TARGET_FIELDS = frozenset({"language", "python_minor", "test_framework"})
TOOLCHAIN_FIELDS = frozenset(
    {
        "python_executable",
        "python_sha256",
        "python_implementation",
        "python_version",
        "pytest_version",
        "pytest_sha256",
        "environment_policy",
    }
)
ENVIRONMENT_POLICY_FIELDS = frozenset(
    {"inherit", "fixed", "private_home", "materialization_pythonpath"}
)
WIRE_FIELDS = frozenset(
    {
        "authority_role",
        "profiles",
        "payload_role",
        "generic_envelope_types_owned_externally",
        "local_envelope_hasher_forbidden",
    }
)
AUTHORITY_FIELDS = frozenset(
    {
        "role",
        "repository",
        "origin",
        "reachability_policy",
        "commit",
        "tree",
        "interface_fingerprint",
        "interface_contract",
        "required_blobs",
        "required_test_commands",
        "test_timeout_seconds",
        "closure_policy",
        "producer_receipt_schema",
    }
)
BLOB_FIELDS = frozenset({"path", "oid"})
INTERFACE_FIELDS = frozenset(
    {
        "contract_name",
        "consumer_schema_requirements",
        "consumer_api_requirements",
        "source_extractions",
    }
)
EXTRACTION_FIELDS = frozenset({"kind", "path", "selector", "value"})
CLOSURE_FIELDS = frozenset({"blob_scope", "import_scope", "test_scope", "materialization"})
HEX40 = re.compile(r"^[0-9a-f]{40}$")
FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")
PLACEHOLDER = re.compile(r"(?:UNRESOLVED|PLACEHOLDER|\\bTODO\\b)", re.IGNORECASE)

EXPECTED_TARGET = {
    "language": "python",
    "python_minor": "3.12",
    "test_framework": "pytest",
}
EXPECTED_UNRESOLVED_AUTHORITY_REASONS = {
    "accelerate_harness": [
        "live_owner_without_heartbeat_can_split_brain_and_swallow_lost_fence",
        "stale_owner_can_overwrite_newer_active_task_index",
        "empty_or_unavailable_process_snapshot_fails_open",
        "whitespace_validation_omits_untracked_and_submodule_outputs",
        "fast_zombie_birth_capture_can_leak_lease",
    ],
    "incremental_semantic_index": ["final_repaired_authority_not_supplied"],
    "semantic_state_contracts": ["final_repaired_authority_not_supplied"],
}
EXPECTED_ENVIRONMENT_POLICY = {
    "inherit": [],
    "fixed": {
        "IPFS_DATASETS_AUTO_INSTALL": "0",
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
        "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
        "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    },
    "private_home": True,
    "materialization_pythonpath": True,
}
EXPECTED_CLOSURE_POLICY = {
    "blob_scope": "entire_commit_tree",
    "import_scope": "entire_commit_tree_fail_closed",
    "test_scope": "sealed_argv_and_repository_configuration_with_tree_fallback",
    "materialization": "fresh_private_git_archive",
}
PRODUCER_RECEIPT_SCHEMA = "ipfs-accelerate.agent-supervisor.semantic-state-producer-test-receipt@1"
EXPECTED_WIRE_CONTRACT = {
    "authority_role": "mcp_plus_plus",
    "profiles": ["A", "B", "F"],
    "payload_role": "accelerate_application_payload_only",
    "generic_envelope_types_owned_externally": True,
    "local_envelope_hasher_forbidden": True,
}
EXPECTED_TEST_TIMEOUT_SECONDS = {
    "accelerate_harness": 1800,
    "incremental_semantic_index": 1800,
    "semantic_state_contracts": 2400,
    "kit_state_roots": 1800,
    "mcp_plus_plus": 900,
}
EXPECTED_REQUIRED_BLOB_PATHS = {
    "accelerate_harness": (
        "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
        "ipfs_accelerate_py/agent_supervisor/context/context_contracts.py",
        "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/merge/leased_lane.py",
        "ipfs_accelerate_py/agent_supervisor/multiformats_identity.py",
        "ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/provider_execution.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/event_log.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/production_context_slice.py",
        "ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_commands.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/worktree_lifecycle.py",
        "ipfs_accelerate_py/mcp_server/mcplusplus/artifacts.py",
        "ipfs_accelerate_py/mcp_server/mcplusplus/kubo_cid.py",
        "test/api/test_agent_supervisor_context_compiler.py",
        "test/api/test_agent_supervisor_lease_coordination.py",
        "test/api/test_agent_supervisor_production_context_slice.py",
        "test/api/test_agent_supervisor_proof_scheduler.py",
        "test/api/test_agent_supervisor_proposal_validation.py",
        "test/api/test_agent_supervisor_provider_execution.py",
        "test/api/test_agent_supervisor_resource_scheduler.py",
        "test/api/test_agent_supervisor_runtime_authority_vectors.py",
        "test/api/test_agent_supervisor_validation_scheduler.py",
        "test/api/test_agent_supervisor_worktree_lifecycle.py",
    ),
    "incremental_semantic_index": (
        "docs/software_contracts/INCREMENTAL_SEMANTIC_INDEX.md",
        "ipfs_datasets_py/logic/software_contracts/content.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_index/__init__.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_index/delta.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_index/index.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_index/invalidation.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_index/models.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_index/scanner.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_index/snapshot.py",
        "tests/cli/test_semantic_index_cli.py",
        "tests/unit/logic/software_contracts/semantic_index/test_acceptance.py",
        "tests/unit/logic/software_contracts/semantic_index/test_import_safety.py",
    ),
    "semantic_state_contracts": (
        "docs/software_contracts/SEMANTIC_STATE_CONTRACT.md",
        "ipfs_datasets_py/logic/software_contracts/content.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_state/__init__.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_state/api.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_state/capsules.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_state/freshness.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_state/models.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_state/schemas/semantic-state.payload.schema.json",
        "ipfs_datasets_py/logic/software_contracts/semantic_state/source.py",
        "ipfs_datasets_py/logic/software_contracts/semantic_state/test_selection.py",
        "tests/unit/logic/software_contracts/semantic_state/test_api.py",
        "tests/unit/logic/software_contracts/semantic_state/test_import_safety.py",
        "tests/unit/logic/software_contracts/semantic_state/test_public_semantic_state_pipeline.py",
    ),
    "kit_state_roots": (
        "ipfs_kit_py/mcp_server/mcplusplus/artifacts.py",
        "ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py",
        "ipfs_kit_py/mcp_server/mcplusplus/state_root_adapter.py",
        "ipfs_kit_py/mcp_server/mcplusplus/state_root_contracts.py",
        "tests/test_coordination_storage.py",
        "tests/test_semantic_state_root_acceptance.py",
        "tests/test_semantic_state_root_adapter.py",
        "tests/test_semantic_state_root_cas.py",
        "tests/test_semantic_state_root_contracts.py",
        "tests/test_semantic_state_root_recovery.py",
    ),
    "mcp_plus_plus": (
        "conformance/vectors/dag_event_epoch.json",
        "conformance/vectors/execution_receipt.json",
        "docs/spec/cid-native-artifacts.md",
        "docs/spec/event-dag-ordering.md",
        "docs/spec/mcp++-profiles-draft.md",
        "docs/spec/mcp-idl.md",
        "tests-py/integration/test_cid_envelopes.py",
        "tests-py/integration/test_conformance_vectors.py",
        "tests-py/integration/test_event_dag.py",
        "tests-py/integration/test_mcp_idl.py",
        "tests-py/validators/__init__.py",
        "tests-py/validators/base_mcp.py",
        "tests-py/validators/cid_artifacts.py",
        "tests-py/validators/event_dag.py",
        "tests-py/validators/mcp_idl.py",
        "tests-py/validators/models.py",
    ),
}
EXPECTED_REQUIRED_TEST_COMMANDS = {
    "accelerate_harness": (
        (
            "/home/barberb/lift_coding/.venv/bin/python",
            "-m",
            "pytest",
            "-q",
            "test/api/test_agent_supervisor_context_compiler.py",
            "test/api/test_agent_supervisor_lease_coordination.py",
            "test/api/test_agent_supervisor_production_context_slice.py",
            "test/api/test_agent_supervisor_proof_scheduler.py",
            "test/api/test_agent_supervisor_proposal_validation.py",
            "test/api/test_agent_supervisor_provider_execution.py",
            "test/api/test_agent_supervisor_resource_scheduler.py",
            "test/api/test_agent_supervisor_runtime_authority_vectors.py",
            "test/api/test_agent_supervisor_validation_scheduler.py",
            "test/api/test_agent_supervisor_worktree_lifecycle.py",
        ),
    ),
    "incremental_semantic_index": (
        (
            "/home/barberb/lift_coding/.venv/bin/python",
            "-m",
            "pytest",
            "-q",
            "tests/unit/logic/software_contracts/semantic_index",
            "tests/cli/test_semantic_index_cli.py",
        ),
    ),
    "semantic_state_contracts": (
        (
            "/home/barberb/lift_coding/.venv/bin/python",
            "-m",
            "pytest",
            "-q",
            "tests/unit/logic/software_contracts/semantic_state",
            "tests/unit/logic/software_contracts/semantic_index",
            "tests/unit/logic/software_contracts/test_content_identity.py",
            "tests/unit/logic/software_contracts/test_python_frontend.py",
            "tests/unit/logic/software_contracts/test_repository_manifest.py",
            "tests/unit/logic/software_contracts/test_resolver.py",
            "tests/cli/test_semantic_index_cli.py",
        ),
    ),
    "kit_state_roots": (
        (
            "/home/barberb/lift_coding/.venv/bin/python",
            "-m",
            "pytest",
            "-q",
            "tests/test_coordination_storage.py",
            "tests/test_semantic_state_root_contracts.py",
            "tests/test_semantic_state_root_adapter.py",
            "tests/test_semantic_state_root_cas.py",
            "tests/test_semantic_state_root_recovery.py",
            "tests/test_semantic_state_root_acceptance.py",
        ),
    ),
    "mcp_plus_plus": (
        (
            "/home/barberb/lift_coding/.venv/bin/python",
            "-m",
            "pytest",
            "-q",
            "tests-py/integration/test_mcp_idl.py",
            "tests-py/integration/test_cid_envelopes.py",
            "tests-py/integration/test_conformance_vectors.py",
            "tests-py/integration/test_event_dag.py",
        ),
    ),
}
EXPECTED_INTERFACE_CONTRACTS = {
    "accelerate_harness": {
        "contract_name": "SemanticCompressionHarnessConsumer@1",
        "consumer_schema_requirements": [
            ["board_namespace", "semantic-compression-harness-v1"],
            ["harness_contracts", "semantic-state-harness@1"],
            ["wire_boundary", "mcp-plus-plus-profiles-a-b-f"],
        ],
        "consumer_api_requirements": [
            "SemanticCapsuleRef(capsule_cid,semantic_state_root_cid,stable_symbol_id,version_cid,source_cid,confidence,validity_bindings,raw_source_required)",
            "SemanticStateProvider.open_semantic_state(root_cid:str,get_block:Callable[[str],bytes])->SemanticStateView",
            "SemanticStateRootManifest(repository_id,base_tree_cid,candidate_tree_cid,datasets_state_cid,datasets_semantic_state_root_cid,capsule_index_cid,delta_cid,invalidation_cid,obligation_set_cid,test_selection_cid,receipt_index_cid,environment_binding_cids,event_head_cid,versions,acceptance_disposition)",
            "TestSelectionRef(selection_cid,previous_semantic_state_root_cid_or_null,current_semantic_state_root_cid)",
        ],
    },
    "incremental_semantic_index": {
        "contract_name": "SemanticCapsuleIndexConsumer@2",
        "consumer_schema_requirements": [
            ["extractor_name", "UNRESOLVED_FINAL_ISI_EXTRACTOR_NAME"],
            ["extractor_version", "UNRESOLVED_FINAL_ISI_EXTRACTOR_VERSION"],
            ["semantic_index_schema", "UNRESOLVED_FINAL_ISI_SCHEMA"],
        ],
        "consumer_api_requirements": [
            "SemanticIndexForCapsules.incoming_edges(stable_symbol_id:str)->tuple[DependencyEdge,...]",
            "SemanticIndexForCapsules.outgoing_edges(stable_symbol_id:str)->tuple[DependencyEdge,...]",
            "SemanticIndexForCapsules.read_source_blob(source_cid:str)->bytes",
            "SemanticIndexForCapsules.read_source_span(stable_symbol_id:str)->bytes",
            "SemanticIndexForCapsules.source_slice(stable_symbol_id:str)->SourceSliceRef",
            "SemanticIndexForCapsules.state_root_cid:str",
            "SemanticIndexForCapsules.symbol(stable_symbol_id:str)->SymbolRecord",
            "calculate_invalidation(previous_state,current_state,delta)->InvalidationPlan",
            "diff_repository_states(previous_state,current_state)->RepositoryStateDelta",
            "scan_repository(repo_path,previous_state=None)->RepositoryState",
        ],
    },
    "semantic_state_contracts": {
        "contract_name": "SemanticStateProvider@1",
        "consumer_schema_requirements": [
            ["capsule_compiler_version", "UNRESOLVED_FINAL_DSS_CAPSULE_COMPILER_VERSION"],
            ["capsule_schema", "UNRESOLVED_FINAL_DSS_CAPSULE_SCHEMA"],
            ["merkle_compiler_version", "UNRESOLVED_FINAL_DSS_MERKLE_COMPILER_VERSION"],
            ["selection_schema", "UNRESOLVED_FINAL_DSS_SELECTION_SCHEMA"],
            ["semantic_state_schema", "UNRESOLVED_FINAL_DSS_SEMANTIC_STATE_SCHEMA"],
        ],
        "consumer_api_requirements": [
            "SemanticStateView.capsule(stable_symbol_id:str)->SemanticCapsule",
            "SemanticStateView.get_block(cid:str)->bytes",
            "SemanticStateView.root:SemanticStateRoot",
            "SemanticStateView.symbol_node(stable_symbol_id:str)->SymbolMerkleNode",
            "build_semantic_state(semantic_index:SemanticIndexForCapsules,*,environment_bindings:Sequence[EnvironmentBinding]=(),previous_bundle:SemanticStateBundle|None=None)->SemanticStateBundle",
            "open_semantic_state(root_cid:str,get_block:Callable[[str],bytes])->SemanticStateView",
            "select_tests_and_proofs(previous_state:SemanticStateView|None,current_state:SemanticStateView,invalidation:SemanticInvalidationPlan,*,policy:SelectionPolicy,explicit_rules:Sequence[SelectionRule]=())->TestSelection",
            "verify_semantic_state_bundle(bundle:SemanticStateBundle)->SemanticStateRoot",
        ],
    },
    "kit_state_roots": {
        "contract_name": "DurableStateRoots@1",
        "consumer_schema_requirements": [
            ["state_root_transition_schema", "mcp++/coordination/state-root-transition@1"],
            ["transport_cid_profile", "cidv1-dag-json-sha2-256"],
        ],
        "consumer_api_requirements": [
            "DurableStateRoots.compare_and_swap_root(namespace:str,expected_revision:int,expected_root_cid:str|None,new_root_cid:str,operation_id:str)->StateRootCASResult",
            "DurableStateRoots.current_root(namespace:str)->StateRootSnapshot",
            "DurableStateRoots.get_verified(cid:str)->Mapping[str,Any]",
            "DurableStateRoots.put_verified(payload:Mapping[str,Any],expected_cid:str,replicate:bool=True)->ArtifactWriteResult",
            "DurableStateRoots.recover_roots()->StateRootRecoveryReport",
        ],
    },
    "mcp_plus_plus": {
        "contract_name": "McpPlusPlusProfilesABF@dc316465",
        "consumer_schema_requirements": [
            ["profile_a", "interface-description"],
            ["profile_b", "cid-native-artifacts"],
            ["profile_f", "event-dag-ordering"],
        ],
        "consumer_api_requirements": [
            "ProfileA.InterfaceDescriptor(application_schema_cid)",
            "ProfileB.ExecutionEnvelope(payload_or_payload_cid)",
            "ProfileB.ExecutionReceipt(content_addressed_result)",
            "ProfileF.DAGEvent(parent_event_cids,payload_cid)",
        ],
    },
}

# The validator owns every selector.  The seal records the extracted value so
# that review sees the actual source/schema contract, but changing a selector
# cannot weaken the gate.  Values are recomputed from the pinned Git objects.
EXPECTED_EXTRACTION_SPECS = {
    "accelerate_harness": (
        (
            "python_signature",
            "ipfs_accelerate_py/mcp_server/mcplusplus/artifacts.py",
            "canonicalize_artifact",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/mcp_server/mcplusplus/kubo_cid.py",
            "cid_for_bytes",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
            "ContextCompiler.compile",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/production_context_slice.py",
            "assert_proposal_covered_by_context",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py",
            "ResourceScheduler.acquire",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/provider_execution.py",
            "ProviderExecutionGateway.execute",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/worktree_lifecycle.py",
            "WorktreeLifecycleStore.begin_preparing",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
            "LeaseCoordinator.validate",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/validation/validation_scheduler.py",
            "ValidationScheduler.run_staged",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py",
            "ProofScheduler.run_stages",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
            "profile_g_cid",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/multiformats_identity.py",
            "cid_for_dag_json",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py",
            "terminate_pid_tree",
        ),
        (
            "python_signature",
            "ipfs_accelerate_py/agent_supervisor/worktree_lifecycle.py",
            "read_process_birth",
        ),
    ),
    "incremental_semantic_index": (
        (
            "python_signature",
            "ipfs_datasets_py/logic/software_contracts/semantic_index/index.py",
            "scan_repository",
        ),
        (
            "python_signature",
            "ipfs_datasets_py/logic/software_contracts/semantic_index/index.py",
            "diff_repository_states",
        ),
        (
            "python_signature",
            "ipfs_datasets_py/logic/software_contracts/semantic_index/index.py",
            "calculate_invalidation",
        ),
        (
            "python_literal",
            "ipfs_datasets_py/logic/software_contracts/semantic_index/models.py",
            "SEMANTIC_INDEX_SCHEMA",
        ),
        (
            "python_literal",
            "ipfs_datasets_py/logic/software_contracts/semantic_index/__init__.py",
            "__all__",
        ),
    ),
    "semantic_state_contracts": (
        (
            "python_signature",
            "ipfs_datasets_py/logic/software_contracts/semantic_state/api.py",
            "build_semantic_state",
        ),
        (
            "python_signature",
            "ipfs_datasets_py/logic/software_contracts/semantic_state/api.py",
            "open_semantic_state",
        ),
        (
            "python_signature",
            "ipfs_datasets_py/logic/software_contracts/semantic_state/test_selection.py",
            "select_tests_and_proofs",
        ),
        (
            "json_keys",
            "ipfs_datasets_py/logic/software_contracts/semantic_state/schemas/semantic-state.payload.schema.json",
            "",
        ),
        (
            "python_literal",
            "ipfs_datasets_py/logic/software_contracts/semantic_state/__init__.py",
            "__all__",
        ),
    ),
    "kit_state_roots": (
        (
            "python_signature",
            "ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py",
            "DurableCoordinationStore.put",
        ),
        (
            "python_signature",
            "ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py",
            "DurableCoordinationStore.get",
        ),
        (
            "python_signature",
            "ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py",
            "DurableCoordinationStore.get_bytes",
        ),
        (
            "python_signature",
            "ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py",
            "DurableCoordinationStore.has",
        ),
        (
            "python_signature",
            "ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py",
            "DurableCoordinationStore.current_state_root",
        ),
        (
            "python_signature",
            "ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py",
            "DurableCoordinationStore.compare_and_swap_state_root",
        ),
        (
            "python_signature",
            "ipfs_kit_py/mcp_server/mcplusplus/coordination_storage.py",
            "DurableCoordinationStore.recover",
        ),
        (
            "python_return_keys",
            "ipfs_kit_py/mcp_server/mcplusplus/artifacts.py",
            "envelope_from_payloads",
        ),
    ),
    "mcp_plus_plus": (
        (
            "python_literal",
            "tests-py/validators/mcp_idl.py",
            "MCPIDLValidator.REQUIRED_DESCRIPTOR_FIELDS",
        ),
        (
            "python_literal",
            "tests-py/validators/cid_artifacts.py",
            "CIDExecutionValidator.REQUIRED_ENVELOPE_FIELDS",
        ),
        (
            "python_literal",
            "tests-py/validators/event_dag.py",
            "EventDAGValidator.validate_event.required_fields",
        ),
        ("json_keys", "conformance/vectors/execution_receipt.json", "/payload"),
        ("json_keys", "conformance/vectors/dag_event_epoch.json", "/payload"),
    ),
}


class DuplicateKeyError(ValueError):
    """Raised when JSON contains a repeated member name."""


def _closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _validate_json_value(value: Any, label: str = "seal", depth: int = 0) -> list[str]:
    """Require the exact recursive JSON type algebra used by the seal."""

    if depth > 64:
        return [f"{label}: JSON nesting exceeds 64 levels"]
    if value is None or isinstance(value, str | bool | int):
        return []
    if isinstance(value, float):
        return [f"{label}: floating-point values are forbidden"]
    if isinstance(value, list):
        errors: list[str] = []
        for index, item in enumerate(value):
            errors.extend(_validate_json_value(item, f"{label}[{index}]", depth + 1))
        return errors
    if isinstance(value, Mapping):
        errors = []
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append(f"{label}: object member names must be strings")
                continue
            errors.extend(_validate_json_value(item, f"{label}.{key}", depth + 1))
        return errors
    return [f"{label}: unsupported JSON value type {type(value).__name__}"]


def load_seal(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_closed_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, Mapping):
        raise ValueError("seal must be a JSON object")
    type_errors = _validate_json_value(value)
    if type_errors:
        raise ValueError(type_errors[0])
    return value


def _normal_origin(value: str) -> str:
    normalized = value.strip().rstrip("/")
    return normalized[:-4] if normalized.endswith(".git") else normalized


def authority_fingerprint(authority: Mapping[str, Any]) -> str:
    projection = {
        "role": authority["role"],
        "repository": authority["repository"],
        "origin": _normal_origin(authority["origin"]),
        "reachability_policy": authority["reachability_policy"],
        "commit": authority["commit"],
        "tree": authority["tree"],
        "interface_contract": authority["interface_contract"],
        "required_blobs": [[item["path"], item["oid"]] for item in authority["required_blobs"]],
        "required_test_commands": authority["required_test_commands"],
        "test_timeout_seconds": authority["test_timeout_seconds"],
        "closure_policy": authority["closure_policy"],
        "producer_receipt_schema": authority["producer_receipt_schema"],
    }
    encoded = json.dumps(
        projection,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _unknown_fields(value: Mapping[str, Any], allowed: frozenset[str], label: str) -> list[str]:
    errors: list[str] = []
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        errors.append(f"{label}: unknown fields: {', '.join(unknown)}")
    if missing:
        errors.append(f"{label}: missing fields: {', '.join(missing)}")
    return errors


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(PLACEHOLDER.search(value))
    if isinstance(value, Mapping):
        return any(
            _contains_placeholder(key) or _contains_placeholder(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def _safe_repo_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\x00" not in value


def validate_document(seal: Mapping[str, Any]) -> list[str]:
    """Validate the closed policy projection without touching a checkout."""

    errors = _validate_json_value(seal)
    errors.extend(_unknown_fields(seal, TOP_LEVEL_FIELDS, "seal"))
    if seal.get("schema") != SEAL_SCHEMA:
        errors.append(f"seal: schema must equal {SEAL_SCHEMA!r}")
    if seal.get("status") != "sealed":
        errors.append("seal: status must be 'sealed'")
    if _contains_placeholder(seal):
        errors.append("seal: unresolved placeholder present")
    unresolved_reasons = seal.get("unresolved_authority_reasons")
    if unresolved_reasons != EXPECTED_UNRESOLVED_AUTHORITY_REASONS:
        errors.append("seal: unresolved authority reasons do not equal the operator audit")
    if seal.get("status") == "sealed" and unresolved_reasons:
        errors.append("seal: sealed status cannot retain unresolved authority reasons")
    target = seal.get("target")
    if not isinstance(target, Mapping):
        errors.append("target: must be an object")
    else:
        errors.extend(_unknown_fields(target, TARGET_FIELDS, "target"))
        if target != EXPECTED_TARGET:
            errors.append("target: must be exactly Python 3.12 with pytest")
    toolchain = seal.get("toolchain")
    if not isinstance(toolchain, Mapping):
        errors.append("toolchain: must be an object")
    else:
        errors.extend(_unknown_fields(toolchain, TOOLCHAIN_FIELDS, "toolchain"))
        executable = toolchain.get("python_executable")
        if not isinstance(executable, str) or not Path(executable).is_absolute():
            errors.append("toolchain: python_executable must be an absolute path")
        if not FINGERPRINT.fullmatch(str(toolchain.get("python_sha256", ""))):
            errors.append("toolchain: python_sha256 must be a sha256 fingerprint")
        if toolchain.get("python_implementation") != "CPython":
            errors.append("toolchain: python_implementation must equal 'CPython'")
        version = toolchain.get("python_version")
        if not isinstance(version, str) or not version.startswith("3.12."):
            errors.append("toolchain: python_version must be an exact Python 3.12 patch")
        pytest_version = toolchain.get("pytest_version")
        if not isinstance(pytest_version, str) or not re.fullmatch(
            r"[0-9]+(?:\.[0-9]+){1,3}", pytest_version
        ):
            errors.append("toolchain: pytest_version must be an exact numeric version")
        if not FINGERPRINT.fullmatch(str(toolchain.get("pytest_sha256", ""))):
            errors.append("toolchain: pytest_sha256 must bind the pytest distribution")
        environment_policy = toolchain.get("environment_policy")
        if not isinstance(environment_policy, Mapping):
            errors.append("toolchain.environment_policy: must be an object")
        else:
            errors.extend(
                _unknown_fields(
                    environment_policy,
                    ENVIRONMENT_POLICY_FIELDS,
                    "toolchain.environment_policy",
                )
            )
            if environment_policy != EXPECTED_ENVIRONMENT_POLICY:
                errors.append(
                    "toolchain.environment_policy: does not equal the closed test environment"
                )
    wire = seal.get("wire_contract")
    if not isinstance(wire, Mapping):
        errors.append("wire_contract: must be an object")
    else:
        errors.extend(_unknown_fields(wire, WIRE_FIELDS, "wire_contract"))
        if wire != EXPECTED_WIRE_CONTRACT:
            errors.append("wire_contract: must preserve the exact generic Profile A/B/F boundary")

    authorities = seal.get("authorities")
    if not isinstance(authorities, list):
        errors.append("authorities: must be a list")
        return errors
    roles = [item.get("role") for item in authorities if isinstance(item, Mapping)]
    if roles != list(EXPECTED_ROLES):
        errors.append(
            "authorities: roles must be unique and ordered as " + ", ".join(EXPECTED_ROLES)
        )

    for index, authority in enumerate(authorities):
        label = f"authorities[{index}]"
        if not isinstance(authority, Mapping):
            errors.append(f"{label}: must be an object")
            continue
        errors.extend(_unknown_fields(authority, AUTHORITY_FIELDS, label))
        role = authority.get("role")
        if role not in EXPECTED_ROLES:
            errors.append(f"{label}: unknown role {role!r}")
            continue
        if authority.get("repository") != EXPECTED_REPOSITORIES[role]:
            errors.append(f"{label}: repository does not match role {role!r}")
        if _normal_origin(str(authority.get("origin", ""))) != EXPECTED_ORIGINS[role]:
            errors.append(f"{label}: origin does not match policy")
        if authority.get("reachability_policy") != REACHABILITY_POLICY:
            errors.append(f"{label}: reachability_policy must equal {REACHABILITY_POLICY!r}")
        if authority.get("commit") != EXPECTED_COMMITS[role]:
            errors.append(f"{label}: commit does not equal the operator-owned pin")
        if authority.get("tree") != EXPECTED_TREES[role]:
            errors.append(f"{label}: tree does not equal the operator-owned pin")
        for field in ("commit", "tree"):
            if not HEX40.fullmatch(str(authority.get(field, ""))):
                errors.append(f"{label}: {field} must be a lowercase 40-hex Git object ID")

        interface = authority.get("interface_contract")
        if not isinstance(interface, Mapping):
            errors.append(f"{label}: interface_contract must be an object")
        else:
            errors.extend(
                _unknown_fields(interface, INTERFACE_FIELDS, f"{label}.interface_contract")
            )
            reviewed_projection = {
                field: interface.get(field)
                for field in (
                    "contract_name",
                    "consumer_schema_requirements",
                    "consumer_api_requirements",
                )
            }
            if reviewed_projection != EXPECTED_INTERFACE_CONTRACTS[role]:
                errors.append(f"{label}: interface_contract must equal the reviewed role contract")
            extractions = interface.get("source_extractions")
            if not isinstance(extractions, list):
                errors.append(f"{label}.interface_contract.source_extractions: must be a list")
            else:
                actual_specs: list[tuple[str, str, str]] = []
                for extraction_index, extraction in enumerate(extractions):
                    extraction_label = (
                        f"{label}.interface_contract.source_extractions[{extraction_index}]"
                    )
                    if not isinstance(extraction, Mapping):
                        errors.append(f"{extraction_label}: must be an object")
                        continue
                    errors.extend(_unknown_fields(extraction, EXTRACTION_FIELDS, extraction_label))
                    actual_specs.append(
                        (
                            str(extraction.get("kind", "")),
                            str(extraction.get("path", "")),
                            str(extraction.get("selector", "")),
                        )
                    )
                    if not _safe_repo_path(str(extraction.get("path", ""))):
                        errors.append(f"{extraction_label}: path must be repository-relative")
                if tuple(actual_specs) != EXPECTED_EXTRACTION_SPECS[role]:
                    errors.append(f"{label}: source extraction selectors do not equal policy")

        blobs = authority.get("required_blobs")
        if not isinstance(blobs, list) or not blobs:
            errors.append(f"{label}: required_blobs must be a non-empty list")
            blobs = []
        paths: list[str] = []
        valid_blob_count = 0
        for blob_index, blob in enumerate(blobs):
            blob_label = f"{label}.required_blobs[{blob_index}]"
            if not isinstance(blob, Mapping):
                errors.append(f"{blob_label}: must be an object")
                continue
            errors.extend(_unknown_fields(blob, BLOB_FIELDS, blob_label))
            path = str(blob.get("path", ""))
            oid = str(blob.get("oid", ""))
            paths.append(path)
            if not _safe_repo_path(path):
                errors.append(f"{blob_label}: path must be safe and repository-relative")
            if not HEX40.fullmatch(oid):
                errors.append(f"{blob_label}: oid must be a lowercase 40-hex Git blob ID")
            if _safe_repo_path(path) and HEX40.fullmatch(oid):
                valid_blob_count += 1
        if paths != sorted(set(paths)):
            errors.append(f"{label}: required_blobs must be sorted by unique path")
        if tuple(paths) != EXPECTED_REQUIRED_BLOB_PATHS[role]:
            errors.append(f"{label}: required_blobs paths do not equal the reviewed role manifest")

        commands = authority.get("required_test_commands")
        expected_commands = [list(command) for command in EXPECTED_REQUIRED_TEST_COMMANDS[role]]
        if not isinstance(commands, list) or not commands:
            errors.append(f"{label}: required_test_commands must be non-empty")
        elif commands != expected_commands:
            errors.append(f"{label}: required_test_commands do not equal the reviewed argv tuples")
        else:
            for command_index, command in enumerate(commands):
                if not all(
                    isinstance(part, str) and part and "\x00" not in part for part in command
                ):
                    errors.append(
                        f"{label}.required_test_commands[{command_index}]: "
                        "must contain non-empty argv strings"
                    )

        timeout = authority.get("test_timeout_seconds")
        if isinstance(timeout, bool) or timeout != EXPECTED_TEST_TIMEOUT_SECONDS[role]:
            errors.append(
                f"{label}: test_timeout_seconds must equal {EXPECTED_TEST_TIMEOUT_SECONDS[role]}"
            )
        closure_policy = authority.get("closure_policy")
        if not isinstance(closure_policy, Mapping):
            errors.append(f"{label}: closure_policy must be an object")
        else:
            errors.extend(
                _unknown_fields(closure_policy, CLOSURE_FIELDS, f"{label}.closure_policy")
            )
            if closure_policy != EXPECTED_CLOSURE_POLICY:
                errors.append(f"{label}: closure_policy does not bind the entire tree")
        if authority.get("producer_receipt_schema") != PRODUCER_RECEIPT_SCHEMA:
            errors.append(f"{label}: producer_receipt_schema does not equal policy")
        fingerprint = str(authority.get("interface_fingerprint", ""))
        if not FINGERPRINT.fullmatch(fingerprint):
            errors.append(
                f"{label}: interface_fingerprint must be sha256 plus 64 lowercase hex digits"
            )
        elif valid_blob_count == len(blobs) and isinstance(interface, Mapping):
            try:
                expected_fingerprint = authority_fingerprint(authority)
            except (KeyError, TypeError, ValueError):
                expected_fingerprint = ""
            if fingerprint != expected_fingerprint:
                errors.append(
                    f"{label}: interface_fingerprint does not bind the complete authority contract"
                )
    return errors


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", os.fspath(path), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_bytes(path: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", os.fspath(path), *args],
        check=False,
        capture_output=True,
    )


def _git_blob_bytes(checkout: Path, commit: str, path: str) -> bytes:
    completed = _git_bytes(checkout, "show", f"{commit}:{path}")
    if completed.returncode:
        raise ValueError(f"cannot read pinned blob {path}")
    return completed.stdout


def _git_blob_oid(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git SHA-1 object ID


def _find_python_node(tree: ast.AST, selector: str) -> ast.AST:
    parts = selector.split(".")
    body = getattr(tree, "body", ())
    current: ast.AST | None = None
    for part in parts:
        current = next(
            (
                node
                for node in body
                if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name == part
            ),
            None,
        )
        if current is None:
            raise ValueError(f"Python selector not found: {selector}")
        body = getattr(current, "body", ())
    return current


def _signature_value(node: ast.AST) -> str:
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        raise ValueError("python_signature selector does not name a function")
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def _literal_node(tree: ast.AST, selector: str) -> ast.AST:
    parts = selector.split(".")
    body = getattr(tree, "body", ())
    for index, part in enumerate(parts):
        for node in body:
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name == part:
                    body = node.body
                    break
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if index == len(parts) - 1 and any(
                    isinstance(target, ast.Name) and target.id == part for target in targets
                ):
                    return node.value
        else:
            raise ValueError(f"Python literal selector not found: {selector}")
    raise ValueError(f"Python literal selector does not name an assignment: {selector}")


def _return_keys(node: ast.AST) -> list[str]:
    keys: set[str] = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Return) or not isinstance(candidate.value, ast.Dict):
            continue
        for key in candidate.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError("return mapping contains a non-literal key")
            keys.add(key.value)
    if not keys:
        raise ValueError("no literal return mapping found")
    return sorted(keys)


def _json_pointer(value: Any, pointer: str) -> Any:
    current = value
    if not pointer:
        return current
    if not pointer.startswith("/"):
        raise ValueError("JSON selector must be an RFC 6901 pointer")
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValueError(f"JSON pointer not found: {pointer}")
    return current


def extract_contract_value(checkout: Path, commit: str, extraction: Mapping[str, Any]) -> Any:
    """Extract one exact API/schema fact from the pinned Git object bytes."""

    kind = str(extraction["kind"])
    path = str(extraction["path"])
    selector = str(extraction["selector"])
    data = _git_blob_bytes(checkout, commit, path)
    if kind.startswith("python_"):
        tree = ast.parse(data.decode("utf-8"), filename=path)
        if kind == "python_signature":
            return _signature_value(_find_python_node(tree, selector))
        if kind == "python_literal":
            return ast.literal_eval(_literal_node(tree, selector))
        if kind == "python_return_keys":
            return _return_keys(_find_python_node(tree, selector))
    if kind == "json_keys":
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_closed_object,
            parse_constant=_reject_json_constant,
        )
        selected = _json_pointer(value, selector)
        if not isinstance(selected, Mapping):
            raise ValueError("json_keys selector does not name an object")
        return sorted(selected)
    raise ValueError(f"unsupported source extraction kind: {kind}")


def _verify_source_extractions(authority: Mapping[str, Any], checkout: Path) -> list[str]:
    role = str(authority.get("role", "unknown"))
    commit = str(authority.get("commit", ""))
    interface = authority.get("interface_contract", {})
    extractions = interface.get("source_extractions", []) if isinstance(interface, Mapping) else []
    errors: list[str] = []
    for index, extraction in enumerate(extractions):
        if not isinstance(extraction, Mapping):
            continue
        try:
            actual = extract_contract_value(checkout, commit, extraction)
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            errors.append(f"checkout[{role}]: source extraction {index} failed: {exc}")
            continue
        if actual != extraction.get("value"):
            errors.append(
                f"checkout[{role}]: source extraction mismatch: "
                f"{extraction.get('path')}#{extraction.get('selector')}"
            )
    return errors


def _tracked_entries(checkout: Path, commit: str) -> list[tuple[str, str, str]]:
    completed = _git_bytes(checkout, "ls-tree", "-rz", commit)
    if completed.returncode:
        raise ValueError("cannot enumerate pinned commit tree")
    entries: list[tuple[str, str, str]] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        header, separator, raw_path = raw.partition(b"\t")
        parts = header.split()
        if not separator or len(parts) != 3:
            raise ValueError("malformed Git tree record")
        entries.append(
            (
                parts[0].decode("ascii"),
                parts[2].decode("ascii"),
                raw_path.decode("utf-8", "surrogateescape"),
            )
        )
    return entries


def _working_bytes(path: Path, mode: str) -> bytes:
    info = path.lstat()
    if mode == "120000":
        if not stat.S_ISLNK(info.st_mode):
            raise ValueError("expected a symbolic link")
        return os.readlink(path).encode("utf-8", "surrogateescape")
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("expected a regular file")
    return path.read_bytes()


def validate_checkout(authority: Mapping[str, Any], checkout: Path) -> list[str]:
    """Verify an exact clean HEAD without claiming remote-ref advertisement."""

    role = str(authority.get("role", "unknown"))
    label = f"checkout[{role}]"
    if not checkout.is_dir():
        return [f"{label}: repository path does not exist: {checkout}"]
    inside = _git(checkout, "rev-parse", "--is-inside-work-tree")
    if inside.returncode or inside.stdout.strip() != "true":
        return [f"{label}: path is not a Git worktree"]
    top = _git(checkout, "rev-parse", "--show-toplevel")
    if top.returncode:
        return [f"{label}: cannot resolve the Git worktree root"]
    try:
        canonical_top = Path(top.stdout.strip()).resolve(strict=True)
        canonical_checkout = checkout.resolve(strict=True)
    except OSError:
        return [f"{label}: cannot resolve the canonical checkout path"]
    if canonical_top != canonical_checkout:
        return [f"{label}: path must be the canonical Git worktree root"]

    errors: list[str] = []
    status = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode:
        errors.append(f"{label}: cannot inspect cleanliness")
    elif status.stdout:
        errors.append(f"{label}: checkout is dirty")
    flags = _git_bytes(checkout, "ls-files", "-v", "-z")
    if flags.returncode:
        errors.append(f"{label}: cannot inspect tracked-file flags")
    else:
        for record in flags.stdout.split(b"\0"):
            if not record:
                continue
            tag = chr(record[0])
            path = record[2:].decode("utf-8", "surrogateescape")
            if tag == "S" or tag.islower():
                errors.append(f"{label}: tracked file uses skip-worktree/assume-unchanged: {path}")
    expected_commit = str(authority.get("commit", ""))
    head = _git(checkout, "rev-parse", "HEAD")
    if head.returncode or head.stdout.strip() != expected_commit:
        errors.append(f"{label}: HEAD does not equal sealed commit")
    commit = _git(checkout, "cat-file", "-e", f"{expected_commit}^{{commit}}")
    if commit.returncode:
        errors.append(f"{label}: sealed commit is not a commit object")
    tree = _git(checkout, "rev-parse", f"{expected_commit}^{{tree}}")
    if tree.returncode or tree.stdout.strip() != str(authority.get("tree", "")):
        errors.append(f"{label}: commit tree does not equal sealed tree")
    origin = _git(checkout, "remote", "get-url", "origin")
    if origin.returncode or _normal_origin(origin.stdout) != _normal_origin(
        str(authority.get("origin", ""))
    ):
        errors.append(f"{label}: origin does not equal sealed origin")

    try:
        entries = _tracked_entries(checkout, expected_commit)
    except ValueError as exc:
        errors.append(f"{label}: {exc}")
        entries = []
    for mode, expected_oid, tracked_path in entries:
        if mode == "160000":  # gitlink content is bound by the commit tree itself
            continue
        try:
            working = _working_bytes(checkout / tracked_path, mode)
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: working bytes unavailable: {tracked_path}: {exc}")
            continue
        if _git_blob_oid(working) != expected_oid:
            errors.append(f"{label}: working bytes differ from HEAD: {tracked_path}")

    for blob in authority.get("required_blobs", []):
        if not isinstance(blob, Mapping):
            continue
        blob_path = str(blob.get("path", ""))
        expected_oid = str(blob.get("oid", ""))
        entry = _git(checkout, "ls-tree", expected_commit, "--", blob_path)
        parts = entry.stdout.strip().split(None, 3)
        actual_oid = (
            parts[2] if not entry.returncode and len(parts) == 4 and parts[1] == "blob" else ""
        )
        if actual_oid != expected_oid:
            errors.append(f"{label}: required blob mismatch or missing: {blob_path}")
    errors.extend(_verify_source_extractions(authority, checkout))
    return errors


def _forbidden_duplicate_authorities(repo: Path) -> list[str]:
    """AST-audit the harness package for locally reimplemented authorities."""

    package = repo / "ipfs_accelerate_py/agent_supervisor/semantic_state"
    if not package.is_dir():
        return []
    forbidden_definitions = {
        "ArtifactFactNode",
        "DAGEvent",
        "DependencyEdge",
        "DurableCoordinationStore",
        "DurableStateRoots",
        "EnvironmentBinding",
        "ExecutionEnvelope",
        "ExecutionReceipt",
        "InterfaceDescriptor",
        "InvalidationPlan",
        "RepositoryScanner",
        "RepositoryState",
        "RepositoryStateDelta",
        "SemanticCapsule",
        "SemanticStateView",
        "SymbolGraph",
        "SymbolMerkleNode",
        "SymbolRecord",
        "TestSelection",
        "TestSelector",
        "build_semantic_state",
        "calculate_invalidation",
        "canonical_json_bytes",
        "canonicalize_artifact",
        "cid_for_bytes",
        "compare_test_selection_oracle",
        "compile_semantic_capsule",
        "compute_artifact_cid",
        "content_cid",
        "diff_repository_states",
        "open_semantic_state",
        "scan_repository",
        "select_tests_and_proofs",
        "verify_semantic_state_bundle",
    }
    forbidden_import_roots = {"ast", "hashlib", "multiformats"}
    cid_authority_terms = (
        "canonical",
        "hash",
        "bytes",
        "envelope",
        "receipt",
        "event",
        "capsule",
        "symbol",
        "state",
    )
    allowed_adapter_methods = {
        "IpfsDatasetsSemanticStateProvider": {
            "open_semantic_state",
            "scan_repository",
        },
        "SemanticStateProvider": {
            "open_semantic_state",
            "scan_repository",
        },
    }
    approved_identity_imports = {
        (
            "ipfs_accelerate_py.mcp_server.mcplusplus.artifacts",
            "canonicalize_artifact",
        ),
        ("ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid", "cid_for_bytes"),
    }
    approved_authority_import_prefixes = (
        "ipfs_datasets_py.logic.software_contracts.semantic_index",
        "ipfs_datasets_py.logic.software_contracts.semantic_state",
        "ipfs_kit_py.mcp_server.mcplusplus",
    )

    def semantic_authority_name(name: str) -> bool:
        lowered = name.lower()
        tokens = {
            token.lower()
            for token in re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+", name)
        }
        cid_builder = "cid" in tokens and bool(
            tokens.intersection(
                {"build", "canonical", "compute", "create", "derive", "encode", "make"}
            )
        )
        canonical_identity = bool(
            tokens.intersection({"canonical", "canonicalize", "cid", "hash"})
            and tokens.intersection(
                {"artifact", "bytes", "content", "envelope", "event", "receipt"}
            )
        )
        generic_wire = bool(
            tokens.intersection({"envelope", "receipt", "event"})
            and tokens.intersection({"execution", "generic", "wire", "dag"})
        )
        reversed_cid_authority = (
            "cid" in lowered
            and not lowered.startswith(("decode", "parse", "require", "validate", "verify"))
            and any(term in lowered for term in cid_authority_terms)
        )
        return canonical_identity or generic_wire or reversed_cid_authority or cid_builder

    def dotted_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    def is_direct_provider_delegation(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        body = list(node.body)
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            return False
        call = body[0].value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            return False
        if call.func.attr != node.name or not isinstance(call.func.value, ast.Attribute):
            return False
        owner = call.func.value
        if not isinstance(owner.value, ast.Name) or owner.value.id != "self":
            return False
        if owner.attr not in {"_api", "_provider", "_datasets"}:
            return False
        if node.decorator_list or node.args.vararg is not None or node.args.kwarg is not None:
            return False
        declared_positional = [
            argument.arg
            for argument in (*node.args.posonlyargs, *node.args.args)
            if argument.arg != "self"
        ]
        declared_keyword_only = [argument.arg for argument in node.args.kwonlyargs]
        if any(not isinstance(argument, ast.Name) for argument in call.args):
            return False
        positional = [argument.id for argument in call.args]
        if positional != declared_positional[: len(positional)]:
            return False
        keywords: list[str] = []
        for keyword in call.keywords:
            if (
                keyword.arg is None
                or not isinstance(keyword.value, ast.Name)
                or keyword.arg != keyword.value.id
            ):
                return False
            keywords.append(keyword.arg)
        forwarded = positional + keywords
        declared = declared_positional + declared_keyword_only
        return len(forwarded) == len(set(forwarded)) and set(forwarded) == set(declared)

    violations: list[str] = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(repo)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=os.fspath(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            violations.append(f"authority boundary: cannot AST-audit {relative}: {exc}")
            continue
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        imported_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                name = node.name
                parent = parents.get(node)
                approved_adapter_method = isinstance(
                    parent, ast.ClassDef
                ) and name in allowed_adapter_methods.get(parent.name, set())
                forbidden_named_definition = (
                    name in forbidden_definitions and not approved_adapter_method
                )
                if forbidden_named_definition or semantic_authority_name(name):
                    violations.append(
                        f"authority boundary: forbidden local authority at "
                        f"{relative}:{node.lineno}: {name}"
                    )
                if approved_adapter_method and not is_direct_provider_delegation(node):
                    violations.append(
                        f"authority boundary: provider method is not direct delegation at "
                        f"{relative}:{node.lineno}: {parent.name}.{name}"
                    )
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and (
                        target.id in forbidden_definitions
                        or (target.id[:1].isupper() and semantic_authority_name(target.id))
                    ):
                        violations.append(
                            f"authority boundary: forbidden local authority alias at "
                            f"{relative}:{node.lineno}: {target.id}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
                    if alias.name.split(".", 1)[0] in forbidden_import_roots:
                        violations.append(
                            f"authority boundary: forbidden analysis/identity import at "
                            f"{relative}:{node.lineno}: {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".", 1)[0] in forbidden_import_roots:
                    violations.append(
                        f"authority boundary: forbidden analysis/identity import at "
                        f"{relative}:{node.lineno}: {module}"
                    )
                for alias in node.names:
                    imported_aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
                    local_name = alias.asname or alias.name
                    if (
                        alias.name in forbidden_definitions or local_name in forbidden_definitions
                    ) and not module.startswith(approved_authority_import_prefixes):
                        violations.append(
                            f"authority boundary: authority imported from unsealed module at "
                            f"{relative}:{node.lineno}: {module}.{alias.name}"
                        )
                    if (
                        alias.name in {"canonicalize_artifact", "cid_for_bytes"}
                        and (module, alias.name) not in approved_identity_imports
                    ):
                        violations.append(
                            f"authority boundary: identity helper imported from unsealed "
                            f"module at {relative}:{node.lineno}: {module}.{alias.name}"
                        )
                    if alias.asname in forbidden_definitions:
                        violations.append(
                            f"authority boundary: forbidden imported authority alias at "
                            f"{relative}:{node.lineno}: {alias.asname}"
                        )
            elif isinstance(node, ast.Call):
                called = dotted_name(node.func)
                resolved = imported_aliases.get(called.split(".", 1)[0], called)
                if "." in called and called.split(".", 1)[0] in imported_aliases:
                    resolved += "." + called.split(".", 1)[1]
                dynamic_import = called == "__import__" or resolved in {
                    "importlib.import_module",
                    "builtins.__import__",
                }
                if dynamic_import:
                    violations.append(
                        f"authority boundary: dynamic import forbidden at "
                        f"{relative}:{node.lineno}: {called}"
                    )
                if called in {"eval", "exec"} or (
                    called == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in {"__import__", "import_module"}
                ):
                    violations.append(
                        f"authority boundary: dynamic code/import indirection forbidden at "
                        f"{relative}:{node.lineno}: {called}"
                    )
                if resolved.startswith(("hashlib.", "multiformats.")):
                    violations.append(
                        f"authority boundary: local CID/hash implementation at "
                        f"{relative}:{node.lineno}: {resolved}"
                    )
                if resolved == "json.dumps" and any(
                    keyword.arg in {"sort_keys", "separators"} for keyword in node.keywords
                ):
                    violations.append(
                        f"authority boundary: local canonicalizer at "
                        f"{relative}:{node.lineno}: json.dumps"
                    )
                if called in {"setattr", "globals", "locals"}:
                    violations.append(
                        f"authority boundary: reflective authority mutation at "
                        f"{relative}:{node.lineno}: {called}"
                    )
            elif isinstance(node, ast.Subscript):
                parent = parents.get(node)
                dynamic_namespace = isinstance(node.value, ast.Call) and dotted_name(
                    node.value.func
                ) in {"globals", "locals"}
                if isinstance(parent, ast.Assign | ast.AnnAssign) and dynamic_namespace:
                    violations.append(
                        f"authority boundary: dynamic authority alias at {relative}:{node.lineno}"
                    )
    return violations


def _parse_repo_bindings(
    values: Iterable[str],
) -> tuple[dict[str, Path], list[str]]:
    bindings: dict[str, Path] = {}
    errors: list[str] = []
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or role not in EXPECTED_ROLES or not raw_path:
            errors.append(
                f"--repo must be ROLE=PATH for one of {', '.join(EXPECTED_ROLES)}: {value!r}"
            )
            continue
        if role in bindings:
            errors.append(f"duplicate --repo binding for {role}")
            continue
        bindings[role] = Path(raw_path).expanduser().resolve()
    return bindings, errors


def _distinct_checkout_errors(repositories: Mapping[str, Path]) -> list[str]:
    by_path: dict[Path, list[str]] = {}
    for role, path in repositories.items():
        by_path.setdefault(path, []).append(role)
    return [
        "checkout bindings must be separate; shared path for roles: " + ", ".join(sorted(roles))
        for roles in by_path.values()
        if len(roles) > 1
    ]


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _test_environment(materialization: Path) -> dict[str, str]:
    """Return the complete closed environment; no ambient variable survives."""

    private_home = materialization / ".sch-private-home"
    private_home.mkdir(mode=0o700)
    environment = dict(EXPECTED_ENVIRONMENT_POLICY["fixed"])
    environment["HOME"] = os.fspath(private_home)
    environment["PYTHONPATH"] = os.fspath(materialization)
    return environment


def _pytest_distribution_probe(python: Path) -> tuple[dict[str, str] | None, str]:
    probe = (
        "import hashlib,importlib.metadata,json,pathlib,platform,sys;"
        "d=importlib.metadata.distribution('pytest');h=hashlib.sha256();"
        "files=sorted(str(x) for x in (d.files or ()) if '__pycache__' not in str(x) "
        "and not str(x).endswith('.pyc'));"
        "[(h.update(x.encode()+b'\\0'),h.update((pathlib.Path(d.locate_file(x)).read_bytes() "
        "if pathlib.Path(d.locate_file(x)).is_file() else b''))) for x in files];"
        "print(json.dumps({'implementation':platform.python_implementation(),"
        "'python_version':platform.python_version(),'pytest_version':d.version,"
        "'pytest_sha256':'sha256:'+h.hexdigest(),'sys_executable':sys.executable},sort_keys=True))"
    )
    completed = subprocess.run(
        [os.fspath(python), "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env={},
        timeout=30,
    )
    if completed.returncode:
        return None, completed.stderr.strip() or "toolchain probe failed"
    try:
        value = json.loads(completed.stdout, object_pairs_hook=_closed_object)
    except (json.JSONDecodeError, DuplicateKeyError) as exc:
        return None, f"malformed toolchain probe: {exc}"
    return value, ""


def validate_toolchain(toolchain: Mapping[str, Any], supplied_python: Path | None) -> list[str]:
    errors: list[str] = []
    declared = Path(str(toolchain.get("python_executable", "")))
    if supplied_python is None:
        return ["toolchain: --python is required for sealed validation"]
    if not supplied_python.is_absolute():
        return ["toolchain: --python must be an absolute path"]
    if os.fspath(supplied_python) != os.fspath(declared):
        errors.append("toolchain: --python does not equal the sealed executable path")
    try:
        binary = supplied_python.resolve(strict=True).read_bytes()
    except OSError as exc:
        return errors + [f"toolchain: cannot read Python executable: {exc}"]
    if _sha256_bytes(binary) != toolchain.get("python_sha256"):
        errors.append("toolchain: Python executable digest mismatch")
    probe, probe_error = _pytest_distribution_probe(supplied_python)
    if probe is None:
        errors.append(f"toolchain: {probe_error}")
        return errors
    expected = {
        "implementation": toolchain.get("python_implementation"),
        "python_version": toolchain.get("python_version"),
        "pytest_version": toolchain.get("pytest_version"),
        "pytest_sha256": toolchain.get("pytest_sha256"),
        "sys_executable": os.fspath(supplied_python),
    }
    if probe != expected:
        errors.append("toolchain: live Python/pytest projection differs from seal")
    return errors


def _safe_materialization_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\x00" in name:
        raise ValueError(f"unsafe archive member: {name!r}")
    return path


def _materialize_commit(checkout: Path, commit: str, destination: Path) -> None:
    archived = _git_bytes(checkout, "archive", "--format=tar", commit)
    if archived.returncode:
        raise ValueError("git archive failed")
    with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            relative = _safe_materialization_member(member.name)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if member.isdir():
                target.mkdir(mode=0o700, exist_ok=True)
                continue
            if member.issym():
                link = PurePosixPath(member.linkname)
                if link.is_absolute() or ".." in link.parts:
                    raise ValueError(f"unsafe archive symlink: {member.name!r}")
                os.symlink(member.linkname, target)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported archive member: {member.name!r}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"archive member has no bytes: {member.name!r}")
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o700 if member.mode & 0o111 else 0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(source.read())


def _verify_materialization(checkout: Path, commit: str, materialization: Path) -> list[str]:
    errors: list[str] = []
    try:
        entries = _tracked_entries(checkout, commit)
    except ValueError as exc:
        return [str(exc)]
    for mode, oid, relative in entries:
        if mode == "160000":
            continue
        try:
            data = _working_bytes(materialization / relative, mode)
        except (OSError, ValueError) as exc:
            errors.append(f"materialized bytes unavailable: {relative}: {exc}")
            continue
        if _git_blob_oid(data) != oid:
            errors.append(f"materialized bytes mismatch: {relative}")
    return errors


def _root_projection(
    authorities: Mapping[str, Mapping[str, Any]], repositories: Mapping[str, Path]
) -> dict[str, dict[str, str]]:
    return {
        role: {
            "commit": str(authorities[role]["commit"]),
            "tree": str(authorities[role]["tree"]),
        }
        for role in EXPECTED_ROLES
        if role in authorities and role in repositories
    }


def _revalidate_all_roots(
    authorities: Mapping[str, Mapping[str, Any]], repositories: Mapping[str, Path]
) -> list[str]:
    errors: list[str] = []
    for role in EXPECTED_ROLES:
        if role in authorities and role in repositories:
            errors.extend(validate_checkout(authorities[role], repositories[role]))
    return errors


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_gone(process: subprocess.Popen[bytes], *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        process.poll()  # Reap the group leader when it has exited.
        if not _process_group_exists(process.pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> bool:
    """Fence the complete owned process group, even after its leader exits."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return True
    if not _wait_process_group_gone(process, timeout_seconds=1.0):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    gone = _wait_process_group_gone(process, timeout_seconds=2.0)
    try:
        process.wait(timeout=0.1)
    except subprocess.TimeoutExpired:
        gone = False
    return gone


def _run_fenced_command(
    command: Sequence[str], cwd: Path, environment: Mapping[str, str], timeout: int
) -> tuple[int, bytes, bytes, bool, bool]:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired as exc:
            # A descendant that escaped the owned group must not stall the
            # control plane by retaining an inherited pipe.
            stdout = exc.output or b""
            stderr = exc.stderr or b""
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
    descendant_leak = False
    if not timed_out:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            descendant_leak = True
            _terminate_process_group(process)
    exit_code = process.returncode if isinstance(process.returncode, int) else -1
    return exit_code, stdout, stderr, timed_out, descendant_leak


RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "role",
        "repository",
        "commit",
        "tree",
        "command_index",
        "argv",
        "python_executable",
        "python_sha256",
        "pytest_sha256",
        "environment_policy_sha256",
        "materialization_tree",
        "closure_tree",
        "stdout_sha256",
        "stderr_sha256",
        "exit_code",
        "timed_out",
        "descendant_leak_detected",
        "pre_roots",
        "post_roots",
        "receipt_sha256",
    }
)


def seal_producer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("receipt_sha256", None)
    body["receipt_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return body


def _receipt_structure_errors(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    role = receipt.get("role")
    if role not in EXPECTED_ROLES:
        errors.append("receipt role is not sealed")
    elif receipt.get("repository") != EXPECTED_REPOSITORIES[role]:
        errors.append("receipt repository does not match its role")
    for field in (
        "commit",
        "tree",
        "materialization_tree",
        "closure_tree",
    ):
        if not HEX40.fullmatch(str(receipt.get(field, ""))):
            errors.append(f"receipt {field} is not a Git object ID")
    if receipt.get("materialization_tree") != receipt.get("tree"):
        errors.append("receipt materialization tree differs from authority tree")
    if receipt.get("closure_tree") != receipt.get("tree"):
        errors.append("receipt closure tree differs from authority tree")
    command_index = receipt.get("command_index")
    if isinstance(command_index, bool) or not isinstance(command_index, int) or command_index < 0:
        errors.append("receipt command index is not a non-negative integer")
    argv = receipt.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(part, str) and part and "\x00" not in part for part in argv)
    ):
        errors.append("receipt argv is not a non-empty string vector")
    python_executable = receipt.get("python_executable")
    if not isinstance(python_executable, str) or not Path(python_executable).is_absolute():
        errors.append("receipt Python executable is not absolute")
    elif isinstance(argv, list) and argv and argv[0] != python_executable:
        errors.append("receipt argv does not use its Python executable")
    for field in (
        "python_sha256",
        "pytest_sha256",
        "environment_policy_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "receipt_sha256",
    ):
        if not FINGERPRINT.fullmatch(str(receipt.get(field, ""))):
            errors.append(f"receipt {field} is not a SHA-256 fingerprint")
    exit_code = receipt.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        errors.append("receipt exit code is not an integer")
    for field in ("timed_out", "descendant_leak_detected"):
        if not isinstance(receipt.get(field), bool):
            errors.append(f"receipt {field} is not Boolean")
    roots: dict[str, Mapping[str, Any]] = {}
    for field in ("pre_roots", "post_roots"):
        projection = receipt.get(field)
        if not isinstance(projection, Mapping) or set(projection) != set(EXPECTED_ROLES):
            errors.append(f"receipt {field} does not bind all five roots")
            continue
        for root_role, root in projection.items():
            if not isinstance(root, Mapping) or set(root) != {"commit", "tree"}:
                errors.append(f"receipt {field}.{root_role} is not a closed root")
                continue
            if not all(HEX40.fullmatch(str(root.get(name, ""))) for name in ("commit", "tree")):
                errors.append(f"receipt {field}.{root_role} has an invalid Git object ID")
        roots[field] = projection
    if roots.get("pre_roots") != roots.get("post_roots"):
        errors.append("receipt roots changed across producer execution")
    if role in EXPECTED_ROLES and "pre_roots" in roots:
        own_root = roots["pre_roots"].get(role)
        if own_root != {"commit": receipt.get("commit"), "tree": receipt.get("tree")}:
            errors.append("receipt authority root differs from the all-root projection")
    return errors


def verify_producer_receipt(receipt: Mapping[str, Any]) -> bool:
    if set(receipt) != RECEIPT_FIELDS:
        return False
    if receipt.get("schema") != PRODUCER_RECEIPT_SCHEMA:
        return False
    if _validate_json_value(receipt):
        return False
    if _receipt_structure_errors(receipt):
        return False
    return seal_producer_receipt(receipt) == receipt


def _write_receipt(receipt_dir: Path, receipt: Mapping[str, Any]) -> None:
    if not verify_producer_receipt(receipt):
        raise ValueError("refusing to write invalid producer receipt")
    receipt_address = str(receipt["receipt_sha256"]).removeprefix("sha256:")
    destination = receipt_dir / f"{receipt_address}.json"
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_bytes(receipt) + b"\n")


def _receipt_population_errors(
    receipt_dir: Path, receipts: Sequence[Mapping[str, Any]]
) -> list[str]:
    expected = {
        f"{str(receipt['receipt_sha256']).removeprefix('sha256:')}.json": receipt
        for receipt in receipts
    }
    try:
        entries = {entry.name: entry for entry in receipt_dir.iterdir()}
    except OSError as exc:
        return [f"receipts: cannot enumerate receipt directory: {exc}"]
    if set(entries) != set(expected):
        return ["receipts: content-addressed population is not closed"]
    errors: list[str] = []
    for name, expected_receipt in expected.items():
        path = entries[name]
        try:
            metadata = path.lstat()
            data = path.read_bytes()
        except OSError as exc:
            errors.append(f"receipts: cannot read {name}: {exc}")
            continue
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            errors.append(f"receipts: {name} is not a regular file")
            continue
        if metadata.st_mode & 0o077:
            errors.append(f"receipts: {name} is not private")
        if data != _canonical_bytes(expected_receipt) + b"\n":
            errors.append(f"receipts: {name} bytes do not match their content address")
        if not verify_producer_receipt(expected_receipt):
            errors.append(f"receipts: {name} has an invalid closed receipt")
    return errors


def _run_all_required_tests(
    authorities: Mapping[str, Mapping[str, Any]],
    repositories: Mapping[str, Path],
    toolchain: Mapping[str, Any],
    receipt_dir: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    receipts: list[dict[str, Any]] = []
    pre_roots = _root_projection(authorities, repositories)
    executable = str(toolchain["python_executable"])
    for role in EXPECTED_ROLES:
        authority = authorities[role]
        checkout = repositories[role]
        for command_index, sealed_command in enumerate(authority["required_test_commands"]):
            pre_errors = _revalidate_all_roots(authorities, repositories)
            if pre_errors:
                return pre_errors, receipts
            pre_roots = _root_projection(authorities, repositories)
            command = list(sealed_command)
            if not command or command[0] != executable:
                errors.append(f"checkout[{role}]: argv does not use sealed Python executable")
                return errors, receipts
            with tempfile.TemporaryDirectory(prefix=f"sch-{role}-") as raw_directory:
                materialization = Path(raw_directory)
                os.chmod(materialization, 0o700)
                try:
                    _materialize_commit(checkout, str(authority["commit"]), materialization)
                except (OSError, ValueError, tarfile.TarError) as exc:
                    errors.append(f"checkout[{role}]: private materialization failed: {exc}")
                    return errors, receipts
                materialization_errors = _verify_materialization(
                    checkout, str(authority["commit"]), materialization
                )
                if materialization_errors:
                    errors.extend(f"checkout[{role}]: {error}" for error in materialization_errors)
                    return errors, receipts
                environment = _test_environment(materialization)
                try:
                    exit_code, stdout, stderr, timed_out, leaked = _run_fenced_command(
                        command,
                        materialization,
                        environment,
                        int(authority["test_timeout_seconds"]),
                    )
                except OSError as exc:
                    errors.append(f"checkout[{role}]: required test could not start: {exc}")
                    return errors, receipts
                post_errors = _revalidate_all_roots(authorities, repositories)
                post_roots = _root_projection(authorities, repositories)
                receipt = seal_producer_receipt(
                    {
                        "schema": PRODUCER_RECEIPT_SCHEMA,
                        "role": role,
                        "repository": authority["repository"],
                        "commit": authority["commit"],
                        "tree": authority["tree"],
                        "command_index": command_index,
                        "argv": command,
                        "python_executable": executable,
                        "python_sha256": toolchain["python_sha256"],
                        "pytest_sha256": toolchain["pytest_sha256"],
                        "environment_policy_sha256": _sha256_bytes(
                            _canonical_bytes(toolchain["environment_policy"])
                        ),
                        "materialization_tree": authority["tree"],
                        "closure_tree": authority["tree"],
                        "stdout_sha256": _sha256_bytes(stdout),
                        "stderr_sha256": _sha256_bytes(stderr),
                        "exit_code": exit_code,
                        "timed_out": timed_out,
                        "descendant_leak_detected": leaked,
                        "pre_roots": pre_roots,
                        "post_roots": post_roots,
                    }
                )
                receipts.append(receipt)
                try:
                    _write_receipt(receipt_dir, receipt)
                except (OSError, ValueError) as exc:
                    errors.append(f"checkout[{role}]: producer receipt write failed: {exc}")
                    return errors, receipts
                if timed_out:
                    errors.append(
                        f"checkout[{role}]: required test timed out after "
                        f"{authority['test_timeout_seconds']}s: {' '.join(command)}"
                    )
                if leaked:
                    errors.append(f"checkout[{role}]: required test leaked a descendant process")
                if exit_code:
                    errors.append(
                        f"checkout[{role}]: required test failed ({exit_code}): {' '.join(command)}"
                    )
                errors.extend(post_errors)
                if errors:
                    return errors, receipts
    errors.extend(_receipt_population_errors(receipt_dir, receipts))
    return errors, receipts


def _run_required_tests(authority: Mapping[str, Any], checkout: Path) -> list[str]:
    """Compatibility helper used by focused unit tests for one synthetic role."""

    errors = validate_checkout(authority, checkout)
    if errors:
        return errors
    role = str(authority["role"])
    for command in authority["required_test_commands"]:
        with tempfile.TemporaryDirectory(prefix="sch-unit-") as raw_directory:
            materialization = Path(raw_directory)
            _materialize_commit(checkout, str(authority["commit"]), materialization)
            environment = _test_environment(materialization)
            result = _run_fenced_command(
                command,
                materialization,
                environment,
                int(authority["test_timeout_seconds"]),
            )
            exit_code, _stdout, _stderr, timed_out, leaked = result
            if timed_out:
                errors.append(
                    f"checkout[{role}]: required test timed out after "
                    f"{authority['test_timeout_seconds']}s: {' '.join(command)}"
                )
            elif leaked:
                errors.append(f"checkout[{role}]: required test leaked a descendant process")
            elif exit_code:
                errors.append(
                    f"checkout[{role}]: required test failed ({exit_code}): {' '.join(command)}"
                )
            errors.extend(validate_checkout(authority, checkout))
            if errors:
                break
    return errors


def validate_seal(
    seal_path: Path,
    *,
    repositories: Mapping[str, Path] | None = None,
    run_tests: bool = False,
    python_executable: Path | None = None,
    receipt_dir: Path | None = None,
) -> list[str]:
    """Return every bounded validation error; an empty list is the only pass."""

    try:
        seal = load_seal(seal_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"seal: cannot load: {exc}"]
    errors = validate_document(seal)
    if seal.get("status") == "sealed" and not run_tests:
        errors.append("seal: sealed validation requires --run-tests")
    repositories = dict(repositories or {})
    missing = sorted(set(EXPECTED_ROLES) - set(repositories))
    unexpected = sorted(set(repositories) - set(EXPECTED_ROLES))
    if missing:
        errors.append("checkout bindings missing: " + ", ".join(missing))
    if unexpected:
        errors.append("checkout bindings unknown: " + ", ".join(unexpected))
    errors.extend(_distinct_checkout_errors(repositories))
    toolchain = seal.get("toolchain")
    if isinstance(toolchain, Mapping):
        errors.extend(validate_toolchain(toolchain, python_executable))

    by_role = {
        str(item.get("role")): item
        for item in seal.get("authorities", [])
        if isinstance(item, Mapping) and item.get("role") in EXPECTED_ROLES
    }
    for role in EXPECTED_ROLES:
        if role in repositories and role in by_role:
            errors.extend(validate_checkout(by_role[role], repositories[role]))
    harness_repo = Path(__file__).resolve().parents[1]
    errors.extend(_forbidden_duplicate_authorities(harness_repo))
    if run_tests and not errors:
        if receipt_dir is None:
            errors.append("receipts: --receipt-dir is required with --run-tests")
        elif not receipt_dir.is_absolute():
            errors.append("receipts: --receipt-dir must be absolute")
        elif receipt_dir.exists():
            errors.append("receipts: --receipt-dir must be a fresh nonexistent path")
        else:
            receipt_dir.mkdir(mode=0o700, parents=False)
            test_errors, receipts = _run_all_required_tests(
                by_role, repositories, toolchain, receipt_dir
            )
            errors.extend(test_errors)
            if not errors and len(receipts) != sum(
                len(by_role[role]["required_test_commands"]) for role in EXPECTED_ROLES
            ):
                errors.append("receipts: producer receipt population is incomplete")
    if python_executable is not None and os.fspath(Path(sys.executable)) != os.fspath(
        python_executable
    ):
        errors.append("runtime: validator itself must run with --python")
    if sys.version_info[:2] != (3, 12):
        errors.append(
            "runtime: validator must run under Python 3.12, got "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True, help="dependency seal JSON")
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="bind one authority role to its separate exact clean checkout",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="run every sealed argv command after source validation",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="absolute operator-supplied Python 3.12 executable bound by the seal",
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        help="fresh absolute private directory for producer-test receipts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repositories, errors = _parse_repo_bindings(args.repo)
    errors.extend(
        validate_seal(
            args.check.resolve(),
            repositories=repositories,
            run_tests=args.run_tests,
            python_executable=args.python,
            receipt_dir=args.receipt_dir,
        )
    )
    if errors:
        for error in dict.fromkeys(errors):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"semantic-state dependency seal verified: {args.check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
