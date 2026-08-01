"""Focused security tests for MCP++ Event-DAG compaction certificates."""

from dataclasses import replace
import json
import sys
import types

import pytest

from ipfs_accelerate_py.mcplusplus_module import dag_compaction


def _events():
    return [
        {"cid": "event-a", "parent_cids": [], "timestamp": 1.0},
        {"cid": "event-b", "parent_cids": ["event-a"], "timestamp": 2.0},
    ]


def _hash_commitment_proof():
    events = _events()
    merkle_root, _ = dag_compaction.build_merkle_tree(
        [event["cid"] for event in events]
    )
    return dag_compaction.CompactionProof(
        merkle_root=merkle_root,
        epoch_id=7,
        event_count=len(events),
        proof=dag_compaction.generate_compaction_proof(
            events,
            merkle_root,
            7,
        ),
        validation_digest=dag_compaction._compute_validation_digest(events),
    )


def _install_fake_zk_provider(monkeypatch, *, verifier_accepts):
    certificate = {
        "proof_system": "groth16-bn254-event-dag-test",
        "zero_knowledge": True,
        "event_count": 2,
        "verification_key_cid": "bafkreitestverificationkey",
        "verification_key_sha256": "a" * 64,
        "proof": {"pi_a": ["1", "2"]},
    }
    provider = types.ModuleType(
        "ipfs_datasets_py.mcp_server.event_dag_zkp"
    )
    provider.availability = lambda: {
        "available": True,
        "proof_system": certificate["proof_system"],
        "verification_key_cid": certificate["verification_key_cid"],
        "verification_key_sha256": certificate["verification_key_sha256"],
    }
    provider.prove_event_dag_compaction = lambda event_cids: dict(certificate)
    provider.verify_event_dag_compaction = (
        lambda proof, event_cids=None: {"valid": verifier_accepts}
    )

    datasets_package = types.ModuleType("ipfs_datasets_py")
    datasets_package.__path__ = []
    mcp_package = types.ModuleType("ipfs_datasets_py.mcp_server")
    mcp_package.__path__ = []
    datasets_package.mcp_server = mcp_package
    mcp_package.event_dag_zkp = provider
    monkeypatch.setitem(sys.modules, "ipfs_datasets_py", datasets_package)
    monkeypatch.setitem(
        sys.modules,
        "ipfs_datasets_py.mcp_server",
        mcp_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "ipfs_datasets_py.mcp_server.event_dag_zkp",
        provider,
    )
    return certificate


def test_hash_commitment_is_explicit_deterministic_and_reverifiable():
    proof = _hash_commitment_proof()
    events = _events()

    assert proof.proof_system == "hash-commitment-v1"
    assert proof.zero_knowledge is False
    assert dag_compaction.verify_compaction_proof(proof) is True
    assert proof.proof == dag_compaction.generate_compaction_proof(
        events,
        proof.merkle_root,
        proof.epoch_id,
    )
    assert proof.to_dict()["proof_system"] == "hash-commitment-v1"
    assert proof.to_dict()["zero_knowledge"] is False


@pytest.mark.parametrize("garbage", ["g" * 64, "0" * 64, "A" * 64])
def test_hash_commitment_rejects_arbitrary_64_character_values(garbage):
    proof = replace(_hash_commitment_proof(), proof=garbage)

    assert dag_compaction.verify_compaction_proof(proof) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("merkle_root", "f" * 64),
        ("epoch_id", 8),
        ("event_count", 3),
        ("validation_digest", "e" * 64),
        ("proof_system", "simulated-groth16"),
        ("zero_knowledge", True),
    ],
)
def test_hash_commitment_rejects_metadata_tampering(field, value):
    proof = replace(_hash_commitment_proof(), **{field: value})

    assert dag_compaction.verify_compaction_proof(proof) is False


def test_zk_label_requires_the_canonical_verifier_to_accept(monkeypatch):
    certificate = _install_fake_zk_provider(
        monkeypatch,
        verifier_accepts=False,
    )
    monkeypatch.setenv("MCPPP_PROFILE_F_ZK", "1")

    assert dag_compaction._profile_f_zk_certificate(
        ["event-a", "event-b"]
    ) is None

    monkeypatch.delenv("MCPPP_PROFILE_F_ZK")
    monkeypatch.setenv("IPFS_DATASETS_ENABLE_GROTH16", "1")
    assert dag_compaction._profile_f_zk_certificate(
        ["event-a", "event-b"]
    ) is None

    monkeypatch.setenv("MCPPP_PROFILE_F_ZK", "required")
    with pytest.raises(RuntimeError, match="required but unavailable"):
        dag_compaction._profile_f_zk_certificate(["event-a", "event-b"])

    certificate = _install_fake_zk_provider(
        monkeypatch,
        verifier_accepts=True,
    )
    accepted = dag_compaction._profile_f_zk_certificate(
        ["event-a", "event-b"]
    )
    assert accepted == certificate

    events = _events()
    merkle_root, _ = dag_compaction.build_merkle_tree(
        [event["cid"] for event in events]
    )
    proof = dag_compaction.CompactionProof(
        merkle_root=merkle_root,
        epoch_id=1,
        event_count=2,
        proof=json.dumps(certificate, sort_keys=True, separators=(",", ":")),
        proof_system=certificate["proof_system"],
        zero_knowledge=True,
        validation_digest=dag_compaction._compute_validation_digest(events),
        verification_key_cid=certificate["verification_key_cid"],
        verification_key_sha256=certificate["verification_key_sha256"],
    )
    assert dag_compaction.verify_compaction_proof(proof) is False
    assert dag_compaction.verify_compaction_proof(
        proof,
        ["event-a", "event-b"],
    ) is True


def test_compactor_persists_and_freshly_verifies_hash_commitment(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        dag_compaction,
        "_profile_f_zk_certificate",
        lambda event_cids: None,
    )
    events = {
        "event-a": {"parent_cids": [], "timestamp": 1.0},
        "event-b": {"parent_cids": ["event-a"], "timestamp": 2.0},
    }
    children = {"event-a": ["event-b"], "event-b": []}

    compactor = dag_compaction.DAGCompactor(
        storage_dir=str(tmp_path),
        epoch_size=2,
    )
    result = compactor.compact_epoch(events, children)

    assert result is not None
    assert result.proof.proof_system == "hash-commitment-v1"
    assert result.proof.zero_knowledge is False
    assert result.proof.verified is True
    assert compactor.verify_cold_epoch(result.proof.epoch_id) is True

    reloaded = dag_compaction.DAGCompactor(
        storage_dir=str(tmp_path),
        epoch_size=2,
    )
    assert reloaded.compaction_proofs[0].verified is True
    assert reloaded.verify_cold_epoch(result.proof.epoch_id) is True


def test_persisted_verified_flag_does_not_override_a_bad_commitment(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        dag_compaction,
        "_profile_f_zk_certificate",
        lambda event_cids: None,
    )
    events = {
        "event-a": {"parent_cids": [], "timestamp": 1.0},
        "event-b": {"parent_cids": ["event-a"], "timestamp": 2.0},
    }
    compactor = dag_compaction.DAGCompactor(
        storage_dir=str(tmp_path),
        epoch_size=2,
    )
    result = compactor.compact_epoch(
        events,
        {"event-a": ["event-b"], "event-b": []},
    )
    assert result is not None

    index_path = tmp_path / "compaction_index.json"
    index = json.loads(index_path.read_text())
    index["proofs"][0]["proof"] = "0" * 64
    index["proofs"][0]["verified"] = True
    index_path.write_text(json.dumps(index))

    reloaded = dag_compaction.DAGCompactor(
        storage_dir=str(tmp_path),
        epoch_size=2,
    )
    assert reloaded.compaction_proofs[0].verified is False
    assert reloaded.verify_cold_epoch(result.proof.epoch_id) is False


def test_cold_epoch_rejects_archive_reference_tampering(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        dag_compaction,
        "_profile_f_zk_certificate",
        lambda event_cids: None,
    )
    compactor = dag_compaction.DAGCompactor(
        storage_dir=str(tmp_path),
        epoch_size=2,
    )
    result = compactor.compact_epoch(
        {
            "event-a": {"parent_cids": [], "timestamp": 1.0},
            "event-b": {"parent_cids": ["event-a"], "timestamp": 2.0},
        },
        {"event-a": ["event-b"], "event-b": []},
    )
    assert result is not None

    epoch_path = tmp_path / "epoch_000000.json"
    epoch = json.loads(epoch_path.read_text())
    epoch["events"][1]["parent_cids"] = ["uncommitted-parent"]
    epoch_path.write_text(json.dumps(epoch))

    assert compactor.verify_cold_epoch(result.proof.epoch_id) is False
    assert result.proof.verified is False
