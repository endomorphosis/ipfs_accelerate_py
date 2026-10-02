"""Opt-in semantic manifest preparation over the datasets source owner.

This adapter binds the complete manifest into planning configuration. The native
finite observer still determines eligible facts; manifest declarations alone
cannot satisfy a requirement.
"""
from __future__ import annotations

from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy import prepare_policy_current
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy_live import verify_policy_current
from ipfs_datasets_py.logic.software_contracts.codebase_semantic_manifest import (
    build_codebase_semantic_manifest, load_codebase_semantic_manifest,
)
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes

SCHEMA = "terminal-codebase-semantic-index@1"


def prepare_semantic_index(*, index, repository, repository_id, operation_id,
        expected_head, contract, scheduler=None, parent_lease=None, cancel_event=None):
    """Capture all admitted source while selecting one explicit proof candidate."""
    if type(contract) is not IntegerOffsetContract:
        raise ValueError("native explicit source contract required")
    observed = prepare_policy_current(index, repository, repository_id=repository_id,
        operation_id=operation_id, expected_head=expected_head,
        training_paths=(), proof_paths=(contract.path,), scheduler=scheduler,
        parent_lease=parent_lease, cancel_event=cancel_event)
    prepared = build_codebase_semantic_manifest(index,
        policy_receipt_cid=observed["receipt_cid"], contracts=(contract,))
    descriptor = dict(schema=SCHEMA, manifest_cid=prepared["manifest_cid"],
        policy_receipt_cid=observed["receipt_cid"], head=observed["head"],
        contract=contract.to_dict(), coverage=prepared["coverage"],
        proof_authority=False, training_executed=False)
    head = CodebaseHead.from_dict(observed["head"])
    verify_semantic_index(index=index, repository=repository, expected_head=head,
        descriptor=descriptor, scheduler=scheduler, parent_lease=parent_lease,
        cancel_event=cancel_event)
    return head, descriptor


def verify_semantic_index(*, index, repository, expected_head, descriptor,
        scheduler=None, parent_lease=None, cancel_event=None):
    """Reconstruct immutable semantics and fence the current source twice."""
    fields = {"schema", "manifest_cid", "policy_receipt_cid", "head", "contract",
              "coverage", "proof_authority", "training_executed"}
    if (type(descriptor) is not dict or set(descriptor) != fields
            or descriptor["schema"] != SCHEMA
            or descriptor["proof_authority"] is not False
            or descriptor["training_executed"] is not False
            or canonical_dag_json_bytes(descriptor["head"]) != canonical_dag_json_bytes(expected_head.to_dict())):
        raise ValueError("exact semantic index descriptor and source head required")
    resources = dict(scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event)
    verify_policy_current(index, repository, expected_head=expected_head,
        receipt_cid=descriptor["policy_receipt_cid"], **resources)
    manifest = load_codebase_semantic_manifest(index, descriptor["manifest_cid"])
    if (manifest["policy_receipt_cid"] != descriptor["policy_receipt_cid"]
            or manifest["source_head"] != descriptor["head"]
            or manifest["coverage"] != descriptor["coverage"]
            or manifest["declarations"] != [descriptor["contract"]]):
        raise ValueError("semantic manifest does not bind the complete selected configuration")
    contract = IntegerOffsetContract.from_dict(descriptor["contract"])
    units = [row for row in manifest["units"] if row["path"] == contract.path]
    if (len(units) != 1 or units[0]["model_status"] != "source_bound_model"
            or units[0]["declared_contract_cid"] != contract.cid):
        raise ValueError("selected finite obligation lacks its exact source-bound semantic model")
    verify_policy_current(index, repository, expected_head=expected_head,
        receipt_cid=descriptor["policy_receipt_cid"], **resources)
    return manifest
