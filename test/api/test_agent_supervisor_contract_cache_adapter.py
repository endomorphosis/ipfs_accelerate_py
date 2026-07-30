"""Supervisor integration tests for the authoritative contract cache."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.contract_analysis.cache_adapter import (
    ContractAnalysisCacheAdapter,
    ContractAnalysisCacheBinding,
)
from ipfs_datasets_py.logic.software_contracts.cache import OUTCOME_UNKNOWN
from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_bytes,
    cid_for_structured,
)


RESULT_SCHEMA = "tests/supervisor-contract-analysis-result@1"


def _identity(name: str) -> str:
    return cid_for_structured({"identity": name})


def _binding(**changes: str) -> ContractAnalysisCacheBinding:
    values = {
        "analyzer_cid": _identity("analyzer-v1"),
        "configuration_cid": _identity("configuration-v1"),
        "semantics_cid": _identity("semantics-v1"),
        "policy_cid": _identity("policy-v1"),
        "solver_cid": _identity("solver-v1"),
        "toolchain_cid": _identity("toolchain-v1"),
        "result_schema": RESULT_SCHEMA,
    }
    values.update(changes)
    return ContractAnalysisCacheBinding(**values)


def _result(label: str) -> dict[str, object]:
    return {"schema": RESULT_SCHEMA, "label": label}


def test_adapter_delegates_exact_reusable_keys_and_bounded_outcomes(
    tmp_path: Path,
) -> None:
    now = [100]
    source_cid = cid_for_bytes(b"source")
    dependency_cid = cid_for_bytes(b"dependency")
    adapter = ContractAnalysisCacheAdapter(
        tmp_path,
        _binding(),
        clock=lambda: now[0],
        max_lease_seconds=10,
    )

    receipt = adapter.store(
        source_cid,
        _result("proved"),
        dependency_cids=(dependency_cid,),
    )
    key_payload = receipt.key.to_dict()
    assert "repository_tree_cid" not in key_payload
    assert adapter.lookup(
        source_cid,
        dependency_cids=(dependency_cid,),
    ).satisfies_completion

    # Every supervisor execution binding is part of reusable identity.
    changed_policy = ContractAnalysisCacheAdapter(
        tmp_path,
        _binding(policy_cid=_identity("policy-v2")),
        clock=lambda: now[0],
    )
    changed_toolchain = ContractAnalysisCacheAdapter(
        tmp_path,
        _binding(toolchain_cid=_identity("toolchain-v2")),
        clock=lambda: now[0],
    )
    assert not changed_policy.lookup(
        source_cid,
        dependency_cids=(dependency_cid,),
    ).hit
    assert not changed_toolchain.lookup(
        source_cid,
        dependency_cids=(dependency_cid,),
    ).hit
    assert not adapter.lookup(
        source_cid,
        dependency_cids=(cid_for_bytes(b"changed dependency"),),
    ).hit

    unknown = adapter.store(
        cid_for_bytes(b"unknown"),
        _result("unknown"),
        outcome=OUTCOME_UNKNOWN,
        lease_seconds=5,
    )
    unknown_lookup = adapter.lookup(cid_for_bytes(b"unknown"))
    assert unknown_lookup.hit
    assert not unknown_lookup.satisfies_completion
    assert not unknown.satisfies_completion(now[0])
    now[0] = 105
    assert not adapter.lookup(cid_for_bytes(b"unknown")).hit


def test_adapter_keeps_snapshot_identity_separate_and_rebuilds_indexes(
    tmp_path: Path,
) -> None:
    adapter = ContractAnalysisCacheAdapter(
        tmp_path,
        _binding(),
        clock=lambda: 100,
    )
    source_cid = cid_for_bytes(b"source")
    dependency_cid = cid_for_bytes(b"dependency")
    receipt = adapter.store(
        source_cid,
        _result("proved"),
        dependency_cids=(dependency_cid,),
    )
    tree_cid = cid_for_structured({"repository_tree": "snapshot-a"})
    snapshot = adapter.create_snapshot_receipt(tree_cid, (receipt,))

    assert snapshot.repository_tree_cid == tree_cid
    assert "repository_tree_cid" not in receipt.key.to_dict()
    assert adapter.read_snapshot_receipt(
        snapshot.cid,
        expected_repository_tree_cid=tree_cid,
        expected_key_cids=(receipt.key_cid,),
    ) == snapshot

    adapter.cache._index_path(receipt.key_cid).unlink()
    assert not adapter.lookup(
        source_cid,
        dependency_cids=(dependency_cid,),
    ).hit
    assert adapter.rebuild_indexes() == (receipt.key_cid,)
    assert adapter.lookup(
        source_cid,
        dependency_cids=(dependency_cid,),
    ).result == _result("proved")

    assert adapter.invalidate_source_closure(dependency_cid) == (
        receipt.key_cid,
    )
    assert not adapter.lookup(
        source_cid,
        dependency_cids=(dependency_cid,),
    ).hit
