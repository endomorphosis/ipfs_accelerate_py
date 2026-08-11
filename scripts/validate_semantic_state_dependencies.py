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
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SEAL_SCHEMA = "ipfs-accelerate.agent-supervisor.semantic-state-dependency-seal@1"
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
    role: f"https://github.com/{repository}"
    for role, repository in EXPECTED_REPOSITORIES.items()
}
EXPECTED_COMMITS = {
    "accelerate_harness": "ea11293bb996f052d620eae989f5377a956764b1",
    "incremental_semantic_index": "UNRESOLVED_FINAL_ISI_COMMIT",
    "semantic_state_contracts": "UNRESOLVED_FINAL_DSS_COMMIT",
    "kit_state_roots": "05ba9375923cd5fb52e2c9c18b98b530d57d077f",
    "mcp_plus_plus": "dc3164653a48d059ae9812078359daeafb451c07",
}
EXPECTED_TREES = {
    "accelerate_harness": "ea6869d70e25c7bc8b80e6458c1a46b8c03f945f",
    "incremental_semantic_index": "UNRESOLVED_FINAL_ISI_TREE",
    "semantic_state_contracts": "UNRESOLVED_FINAL_DSS_TREE",
    "kit_state_roots": "a770206fe9e11852a9a230b9ce64d0cce254dd50",
    "mcp_plus_plus": "6560c3d0c926be12df860afb7d7c82043a1769ba",
}
REACHABILITY_POLICY = "exact_clean_head"

TOP_LEVEL_FIELDS = frozenset(
    {"schema", "status", "target", "wire_contract", "authorities"}
)
TARGET_FIELDS = frozenset({"language", "python_minor", "test_framework"})
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
    }
)
BLOB_FIELDS = frozenset({"path", "oid"})
INTERFACE_FIELDS = frozenset({"contract_name", "schema_versions", "public_api"})
HEX40 = re.compile(r"^[0-9a-f]{40}$")
FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")
PLACEHOLDER = re.compile(r"(?:UNRESOLVED|PLACEHOLDER|\\bTODO\\b)", re.IGNORECASE)

EXPECTED_TARGET = {
    "language": "python",
    "python_minor": "3.12",
    "test_framework": "pytest",
}
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
        "ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py",
        "ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/event_log.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_execution.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/resource_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/production_context_slice.py",
        "ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_commands.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_scheduler.py",
        "test/api/test_agent_supervisor_context_compiler.py",
        "test/api/test_agent_supervisor_lease_coordination.py",
        "test/api/test_agent_supervisor_production_context_slice.py",
        "test/api/test_agent_supervisor_proof_scheduler.py",
        "test/api/test_agent_supervisor_proposal_validation.py",
        "test/api/test_agent_supervisor_provider_execution.py",
        "test/api/test_agent_supervisor_resource_scheduler.py",
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
        "tests-py/validators/cid_artifacts.py",
        "tests-py/validators/event_dag.py",
        "tests-py/validators/mcp_idl.py",
    ),
}
EXPECTED_REQUIRED_TEST_COMMANDS = {
    "accelerate_harness": (
        (
            "python3.12", "-m", "pytest", "-q",
            "test/api/test_agent_supervisor_context_compiler.py",
            "test/api/test_agent_supervisor_lease_coordination.py",
            "test/api/test_agent_supervisor_production_context_slice.py",
            "test/api/test_agent_supervisor_proof_scheduler.py",
            "test/api/test_agent_supervisor_proposal_validation.py",
            "test/api/test_agent_supervisor_provider_execution.py",
            "test/api/test_agent_supervisor_resource_scheduler.py",
            "test/api/test_agent_supervisor_validation_scheduler.py",
            "test/api/test_agent_supervisor_worktree_lifecycle.py",
        ),
    ),
    "incremental_semantic_index": (
        (
            "python3.12", "-m", "pytest", "-q",
            "tests/unit/logic/software_contracts/semantic_index",
            "tests/cli/test_semantic_index_cli.py",
        ),
    ),
    "semantic_state_contracts": (
        (
            "python3.12", "-m", "pytest", "-q",
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
            "python3.12", "-m", "pytest", "-q",
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
            "python3.12", "-m", "pytest", "-q",
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
        "schema_versions": [
            ["board_namespace", "semantic-compression-harness-v1"],
            ["harness_contracts", "semantic-state-harness@1"],
            ["wire_boundary", "mcp-plus-plus-profiles-a-b-f"],
        ],
        "public_api": [
            "SemanticCapsuleRef(capsule_cid,semantic_state_root_cid,stable_symbol_id,version_cid,source_cid,confidence,validity_bindings,raw_source_required)",
            "SemanticStateProvider.open_semantic_state(root_cid:str,get_block:Callable[[str],bytes])->SemanticStateView",
            "SemanticStateRootManifest(repository_id,base_tree_cid,candidate_tree_cid,datasets_state_cid,datasets_semantic_state_root_cid,capsule_index_cid,delta_cid,invalidation_cid,obligation_set_cid,test_selection_cid,receipt_index_cid,environment_binding_cids,event_head_cid,versions,acceptance_disposition)",
            "TestSelectionRef(selection_cid,previous_semantic_state_root_cid_or_null,current_semantic_state_root_cid)",
        ],
    },
    "incremental_semantic_index": {
        "contract_name": "SemanticCapsuleIndexConsumer@2",
        "schema_versions": [
            ["extractor_name", "UNRESOLVED_FINAL_ISI_EXTRACTOR_NAME"],
            ["extractor_version", "UNRESOLVED_FINAL_ISI_EXTRACTOR_VERSION"],
            ["semantic_index_schema", "UNRESOLVED_FINAL_ISI_SCHEMA"],
        ],
        "public_api": [
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
        "schema_versions": [
            ["capsule_compiler_version", "UNRESOLVED_FINAL_DSS_CAPSULE_COMPILER_VERSION"],
            ["capsule_schema", "UNRESOLVED_FINAL_DSS_CAPSULE_SCHEMA"],
            ["merkle_compiler_version", "UNRESOLVED_FINAL_DSS_MERKLE_COMPILER_VERSION"],
            ["selection_schema", "UNRESOLVED_FINAL_DSS_SELECTION_SCHEMA"],
            ["semantic_state_schema", "UNRESOLVED_FINAL_DSS_SEMANTIC_STATE_SCHEMA"],
        ],
        "public_api": [
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
        "schema_versions": [
            ["state_root_transition_schema", "mcp++/coordination/state-root-transition@1"],
            ["transport_cid_profile", "cidv1-dag-json-sha2-256"],
        ],
        "public_api": [
            "DurableStateRoots.compare_and_swap_root(namespace:str,expected_revision:int,expected_root_cid:str|None,new_root_cid:str,operation_id:str)->StateRootCASResult",
            "DurableStateRoots.current_root(namespace:str)->StateRootSnapshot",
            "DurableStateRoots.get_verified(cid:str)->Mapping[str,Any]",
            "DurableStateRoots.put_verified(payload:Mapping[str,Any],expected_cid:str,replicate:bool=True)->ArtifactWriteResult",
            "DurableStateRoots.recover_roots()->StateRootRecoveryReport",
        ],
    },
    "mcp_plus_plus": {
        "contract_name": "McpPlusPlusProfilesABF@dc316465",
        "schema_versions": [
            ["profile_a", "interface-description"],
            ["profile_b", "cid-native-artifacts"],
            ["profile_f", "event-dag-ordering"],
        ],
        "public_api": [
            "ProfileA.InterfaceDescriptor(application_schema_cid)",
            "ProfileB.ExecutionEnvelope(payload_or_payload_cid)",
            "ProfileB.ExecutionReceipt(content_addressed_result)",
            "ProfileF.DAGEvent(parent_event_cids,payload_cid)",
        ],
    },
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


def load_seal(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_closed_object)
    if not isinstance(value, Mapping):
        raise ValueError("seal must be a JSON object")
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
        "required_blobs": [
            [item["path"], item["oid"]] for item in authority["required_blobs"]
        ],
        "required_test_commands": authority["required_test_commands"],
        "test_timeout_seconds": authority["test_timeout_seconds"],
    }
    encoded = json.dumps(
        projection,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _unknown_fields(
    value: Mapping[str, Any], allowed: frozenset[str], label: str
) -> list[str]:
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
            _contains_placeholder(key) or _contains_placeholder(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def _safe_repo_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\x00" not in value
    )


def validate_document(seal: Mapping[str, Any]) -> list[str]:
    """Validate the closed policy projection without touching a checkout."""

    errors = _unknown_fields(seal, TOP_LEVEL_FIELDS, "seal")
    if seal.get("schema") != SEAL_SCHEMA:
        errors.append(f"seal: schema must equal {SEAL_SCHEMA!r}")
    if seal.get("status") != "sealed":
        errors.append("seal: status must be 'sealed'")
    if _contains_placeholder(seal):
        errors.append("seal: unresolved placeholder present")
    target = seal.get("target")
    if not isinstance(target, Mapping):
        errors.append("target: must be an object")
    else:
        errors.extend(_unknown_fields(target, TARGET_FIELDS, "target"))
        if target != EXPECTED_TARGET:
            errors.append("target: must be exactly Python 3.12 with pytest")
    wire = seal.get("wire_contract")
    if not isinstance(wire, Mapping):
        errors.append("wire_contract: must be an object")
    else:
        errors.extend(_unknown_fields(wire, WIRE_FIELDS, "wire_contract"))
        if wire != EXPECTED_WIRE_CONTRACT:
            errors.append(
                "wire_contract: must preserve the exact generic Profile A/B/F boundary"
            )

    authorities = seal.get("authorities")
    if not isinstance(authorities, list):
        errors.append("authorities: must be a list")
        return errors
    roles = [
        item.get("role") for item in authorities if isinstance(item, Mapping)
    ]
    if roles != list(EXPECTED_ROLES):
        errors.append(
            "authorities: roles must be unique and ordered as "
            + ", ".join(EXPECTED_ROLES)
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
            errors.append(
                f"{label}: reachability_policy must equal {REACHABILITY_POLICY!r}"
            )
        if authority.get("commit") != EXPECTED_COMMITS[role]:
            errors.append(f"{label}: commit does not equal the operator-owned pin")
        if authority.get("tree") != EXPECTED_TREES[role]:
            errors.append(f"{label}: tree does not equal the operator-owned pin")
        for field in ("commit", "tree"):
            if not HEX40.fullmatch(str(authority.get(field, ""))):
                errors.append(
                    f"{label}: {field} must be a lowercase 40-hex Git object ID"
                )

        interface = authority.get("interface_contract")
        if not isinstance(interface, Mapping):
            errors.append(f"{label}: interface_contract must be an object")
        else:
            errors.extend(
                _unknown_fields(
                    interface, INTERFACE_FIELDS, f"{label}.interface_contract"
                )
            )
            if interface != EXPECTED_INTERFACE_CONTRACTS[role]:
                errors.append(
                    f"{label}: interface_contract must equal the reviewed role contract"
                )

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
            errors.append(
                f"{label}: required_blobs paths do not equal the reviewed role manifest"
            )

        commands = authority.get("required_test_commands")
        expected_commands = [
            list(command) for command in EXPECTED_REQUIRED_TEST_COMMANDS[role]
        ]
        if not isinstance(commands, list) or not commands:
            errors.append(f"{label}: required_test_commands must be non-empty")
        elif commands != expected_commands:
            errors.append(
                f"{label}: required_test_commands do not equal the reviewed argv tuples"
            )
        else:
            for command_index, command in enumerate(commands):
                if not all(
                    isinstance(part, str) and part and "\x00" not in part
                    for part in command
                ):
                    errors.append(
                        f"{label}.required_test_commands[{command_index}]: "
                        "must contain non-empty argv strings"
                    )

        timeout = authority.get("test_timeout_seconds")
        if isinstance(timeout, bool) or timeout != EXPECTED_TEST_TIMEOUT_SECONDS[role]:
            errors.append(
                f"{label}: test_timeout_seconds must equal "
                f"{EXPECTED_TEST_TIMEOUT_SECONDS[role]}"
            )
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
    if (
        origin.returncode
        or _normal_origin(origin.stdout)
        != _normal_origin(str(authority.get("origin", "")))
    ):
        errors.append(f"{label}: origin does not equal sealed origin")

    for blob in authority.get("required_blobs", []):
        if not isinstance(blob, Mapping):
            continue
        blob_path = str(blob.get("path", ""))
        expected_oid = str(blob.get("oid", ""))
        entry = _git(checkout, "ls-tree", expected_commit, "--", blob_path)
        parts = entry.stdout.strip().split(None, 3)
        actual_oid = (
            parts[2]
            if not entry.returncode and len(parts) == 4 and parts[1] == "blob"
            else ""
        )
        if actual_oid != expected_oid:
            errors.append(f"{label}: required blob mismatch or missing: {blob_path}")
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
    forbidden_import_roots = {"ast", "hashlib"}
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
    violations: list[str] = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(repo)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=os.fspath(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            violations.append(f"authority boundary: cannot AST-audit {relative}: {exc}")
            continue
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                lowered = name.lower()
                parent = parents.get(node)
                approved_adapter_method = (
                    isinstance(parent, ast.ClassDef)
                    and name in allowed_adapter_methods.get(parent.name, set())
                )
                reversed_cid_authority = (
                    "cid" in lowered
                    and not lowered.startswith(
                        ("decode", "parse", "require", "validate", "verify")
                    )
                    and any(
                    term in lowered for term in cid_authority_terms
                    )
                )
                forbidden_named_definition = (
                    name in forbidden_definitions and not approved_adapter_method
                )
                if forbidden_named_definition or reversed_cid_authority:
                    violations.append(
                        f"authority boundary: forbidden local authority at "
                        f"{relative}:{node.lineno}: {name}"
                    )
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id in forbidden_definitions
                    ):
                        violations.append(
                            f"authority boundary: forbidden local authority alias at "
                            f"{relative}:{node.lineno}: {target.id}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
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
                    if alias.asname in forbidden_definitions:
                        violations.append(
                            f"authority boundary: forbidden imported authority alias at "
                            f"{relative}:{node.lineno}: {alias.asname}"
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
                f"--repo must be ROLE=PATH for one of "
                f"{', '.join(EXPECTED_ROLES)}: {value!r}"
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
        "checkout bindings must be separate; shared path for roles: "
        + ", ".join(sorted(roles))
        for roles in by_path.values()
        if len(roles) > 1
    ]


def _test_environment(checkout: Path) -> dict[str, str]:
    """Return a new role-local environment with no ambient import override."""

    environment = {
        key: os.environ[key]
        for key in (
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "SSL_CERT_FILE",
            "TEMP",
            "TMP",
            "TMPDIR",
        )
        if key in os.environ
    }
    environment.update(
        {
            "IPFS_DATASETS_AUTO_INSTALL": "0",
            "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
            "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
            "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.fspath(checkout),
        }
    )
    return environment


def _run_required_tests(
    authority: Mapping[str, Any], checkout: Path
) -> list[str]:
    """Run every sealed command and detect pre/post-test checkout mutation."""

    errors = validate_checkout(authority, checkout)
    if errors:
        return errors
    role = str(authority["role"])
    timeout_seconds = int(authority["test_timeout_seconds"])
    environment = _test_environment(checkout)
    for sealed_command in authority["required_test_commands"]:
        command = [sys.executable, *sealed_command[1:]]
        try:
            completed = subprocess.run(
                command,
                cwd=checkout,
                env=environment,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            errors.append(
                f"checkout[{role}]: required test timed out after "
                f"{timeout_seconds}s: {' '.join(sealed_command)}"
            )
        except OSError as exc:
            errors.append(
                f"checkout[{role}]: required test could not start: "
                f"{' '.join(sealed_command)}: {exc}"
            )
        else:
            if completed.returncode:
                errors.append(
                    f"checkout[{role}]: required test failed "
                    f"({completed.returncode}): {' '.join(sealed_command)}"
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

    by_role = {
        str(item.get("role")): item
        for item in seal.get("authorities", [])
        if isinstance(item, Mapping) and item.get("role") in EXPECTED_ROLES
    }
    for role in EXPECTED_ROLES:
        if role in repositories and role in by_role:
            errors.extend(validate_checkout(by_role[role], repositories[role]))
    if "accelerate_harness" in repositories:
        errors.extend(
            _forbidden_duplicate_authorities(repositories["accelerate_harness"])
        )
    if run_tests and not errors:
        for role in EXPECTED_ROLES:
            errors.extend(_run_required_tests(by_role[role], repositories[role]))
            if errors:
                break
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repositories, errors = _parse_repo_bindings(args.repo)
    errors.extend(
        validate_seal(
            args.check.resolve(),
            repositories=repositories,
            run_tests=args.run_tests,
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
