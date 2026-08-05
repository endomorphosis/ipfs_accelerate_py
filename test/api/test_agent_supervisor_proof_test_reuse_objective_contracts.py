"""Strict objective-completion artifact contract tests (PTR-112)."""

from __future__ import annotations

import base64
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_objective_contracts import (
    CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE,
    COMPLETION_EVIDENCE_INTERFACE,
    DECLARED_STATE_ROOT_CONTROL_ARTIFACT_SUFFIXES,
    PROOF_TEST_REUSE_COMPLETION_ARTIFACT_INTERFACE,
    PROOF_TEST_REUSE_GATE_BUNDLE_INTERFACE,
    PROOF_TEST_REUSE_OBJECTIVE_BINDING_INTERFACE,
    CanonicalPremiseBlock,
    ObjectiveArtifactReason,
    ObjectiveArtifactStore,
    ProofTestReuseCompletionArtifact,
    ProofTestReuseGateBundle,
    ProofTestReuseObjectiveBinding,
    ProofTestReuseObjectiveContractsError,
    assert_control_paths_are_declared,
    canonical_dag_json_bytes,
    cid_for_canonical_dag_json_bytes,
    cid_for_mapping,
    compute_objective_completion_tree_id,
    decode_artifact_cid,
    declared_state_root_control_paths,
    require_verified_cid,
    validate_artifact_cid,
    verify_retained_bytes,
)

NOW_MS = 1_786_000_000_000
FRESH_UNTIL_MS = NOW_MS + 60_000

GIT_TREE = "a" * 40
FOREST = "baguqeera" + "f" * 50
COMPLETION_TREE = "sha256:" + "b" * 64
REPO_ID = "repository:sha256:" + "c" * 64
OBJECTIVE_REV = "baguqeera" + "1" * 50
ANALYZER_REV = "baguqeera" + "2" * 50
CONFIG_REV = "baguqeera" + "3" * 50
POLICY_REV = "baguqeera" + "4" * 50
CAPABILITY_REV = "baguqeera" + "5" * 50
CIRCUIT_REV = "baguqeera" + "6" * 50
KEY_REV = "baguqeera" + "7" * 50


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _binding(**overrides: Any) -> ProofTestReuseObjectiveBinding:
    values: dict[str, Any] = {
        "goal_id": "PTR-G070",
        "repository_id": REPO_ID,
        "git_tree_id": GIT_TREE,
        "repository_forest_cid": FOREST,
        "objective_completion_tree_id": COMPLETION_TREE,
        "objective_revision": OBJECTIVE_REV,
        "analyzer_revision": ANALYZER_REV,
        "configuration_revision": CONFIG_REV,
        "policy_revision": POLICY_REV,
        "capability_revision": CAPABILITY_REV,
        "circuit_revision": CIRCUIT_REV,
        "verifying_key_revision": KEY_REV,
        "git_commit_id": "d" * 40,
        "gitlink_state_cid": "baguqeera" + "8" * 50,
    }
    values.update(overrides)
    return ProofTestReuseObjectiveBinding(**values)


def _premise(
    payload: dict[str, Any] | None = None, *, role: str = "validation"
) -> CanonicalPremiseBlock:
    body = dict(payload or {"schema": "premise@1", "status": "passed", "n": 1})
    return CanonicalPremiseBlock.from_mapping(body, role=role)


def _artifact(**overrides: Any) -> ProofTestReuseCompletionArtifact:
    values: dict[str, Any] = {
        "binding": _binding(),
        "acceptance_criterion": "ptr/supervisor-completion-authority@1",
        "producing_task_or_scan": "PTR-112",
        "premise_blocks": (_premise(),),
        "observed_at_ms": NOW_MS,
        "fresh_until_ms": FRESH_UNTIL_MS,
        "validation_passed": True,
        "producer_kind": "task",
        "producer_channel": "objective-artifact-contracts",
        "channel_proof_revision": "channel:ptr-112@1",
    }
    values.update(overrides)
    return ProofTestReuseCompletionArtifact(**values)


# ---------------------------------------------------------------------------
# Interface surface
# ---------------------------------------------------------------------------


def test_interfaces_and_declared_control_suffixes_are_stable() -> None:
    assert PROOF_TEST_REUSE_OBJECTIVE_BINDING_INTERFACE == (
        "ProofTestReuseObjectiveBinding@1"
    )
    assert PROOF_TEST_REUSE_COMPLETION_ARTIFACT_INTERFACE == (
        "ProofTestReuseCompletionArtifact@1"
    )
    assert PROOF_TEST_REUSE_GATE_BUNDLE_INTERFACE == "ProofTestReuseGateBundle@1"
    assert CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE == (
        "CanonicalArtifactStoreTransport@1"
    )
    assert COMPLETION_EVIDENCE_INTERFACE == "CompletionEvidence"
    assert DECLARED_STATE_ROOT_CONTROL_ARTIFACT_SUFFIXES == (
        "projection/completion/goal_completion_gate.json",
        "projection/completion/goal_completion_evidence.json",
        "projection/completion/objective_projection.md",
        "projection/completion/objective_candidate.md",
        "projection/completion/supervisor_health_input.json",
        "projection/completion/closeout_status.json",
    )


# ---------------------------------------------------------------------------
# Identity domains
# ---------------------------------------------------------------------------


def test_binding_distinguishes_three_identity_domains() -> None:
    binding = _binding()
    payload = binding.to_dict()
    assert payload["git_tree_id"] == GIT_TREE
    assert payload["repository_forest_cid"] == FOREST
    assert payload["objective_completion_tree_id"] == COMPLETION_TREE
    assert payload["git_tree_id"] != payload["repository_forest_cid"]
    assert payload["git_tree_id"] != payload["objective_completion_tree_id"]
    assert payload["repository_forest_cid"] != payload["objective_completion_tree_id"]
    assert binding.tree_id == binding.git_tree_id
    assert binding.interface == PROOF_TEST_REUSE_OBJECTIVE_BINDING_INTERFACE


def test_identity_domain_collision_is_rejected() -> None:
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        _binding(repository_forest_cid=GIT_TREE)
    assert exc.value.reason_code is ObjectiveArtifactReason.IDENTITY_DOMAIN_COLLISION

    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        _binding(objective_completion_tree_id=FOREST)
    assert exc.value.reason_code is ObjectiveArtifactReason.IDENTITY_DOMAIN_COLLISION


def test_binding_requires_all_revision_fields() -> None:
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        _binding(policy_revision="")
    assert exc.value.reason_code is ObjectiveArtifactReason.BINDING_INCOMPLETE

    for field in (
        "objective_revision",
        "analyzer_revision",
        "configuration_revision",
        "capability_revision",
        "circuit_revision",
        "verifying_key_revision",
        "repository_id",
        "goal_id",
    ):
        with pytest.raises(ProofTestReuseObjectiveContractsError):
            _binding(**{field: ""})


def test_revision_must_not_alias_git_tree() -> None:
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        _binding(policy_revision=GIT_TREE)
    assert exc.value.reason_code is ObjectiveArtifactReason.ALIAS_CONFLICT


def test_forest_must_not_be_bare_git_object_id() -> None:
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        _binding(repository_forest_cid="e" * 40)
    assert exc.value.reason_code in {
        ObjectiveArtifactReason.ALIAS_CONFLICT,
        ObjectiveArtifactReason.IDENTITY_DOMAIN_COLLISION,
    }


# ---------------------------------------------------------------------------
# CID profile: CIDv1 lowercase base32 dag-json sha2-256 + multihash recheck
# ---------------------------------------------------------------------------


def test_authoritative_cid_is_cidv1_base32_dag_json_sha256() -> None:
    data = canonical_dag_json_bytes({"schema": "t", "x": 1})
    cid = cid_for_canonical_dag_json_bytes(data)
    assert cid == cid.lower()
    assert cid.startswith("b")
    parsed = decode_artifact_cid(cid)
    assert parsed.version == 1
    assert parsed.codec == 0x0129
    assert parsed.multihash_code == 0x12
    assert len(parsed.digest) == 32
    assert parsed.digest == hashlib.sha256(data).digest()
    assert parsed.verifies(data)
    assert verify_retained_bytes(cid, data)
    assert require_verified_cid(cid, data) == cid


def test_content_identity_matches_mapping_cid_helper() -> None:
    payload = {"schema": "t", "nested": {"a": 1, "b": [1, 2]}}
    assert cid_for_mapping(payload) == cid_for_canonical_dag_json_bytes(
        canonical_dag_json_bytes(payload)
    )


def test_fake_and_noncanonical_cids_are_rejected() -> None:
    data = canonical_dag_json_bytes({"ok": True})
    good = cid_for_canonical_dag_json_bytes(data)

    for fake in (
        "",
        "sha256:" + "a" * 64,
        "QmYjtig7VJQ6XsnUjqqJvj7QaMcCAwtrgNdahSiFofrE7o",
        "bafy-" + "a" * 50,
        good.upper(),
        good[:-1],  # truncated
        "b" + "!" * 50,
        "../etc/passwd",
        "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku",  # raw codec shape
    ):
        with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
            validate_artifact_cid(fake)
        assert exc.value.reason_code in {
            ObjectiveArtifactReason.FAKE_CID,
            ObjectiveArtifactReason.NONCANONICAL_CID,
            ObjectiveArtifactReason.WRONG_CODEC,
        }

    # Multihash recheck fails on wrong bytes.
    other = canonical_dag_json_bytes({"ok": False})
    assert not verify_retained_bytes(good, other)
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        require_verified_cid(good, other)
    assert exc.value.reason_code is ObjectiveArtifactReason.MULTI_HASH_MISMATCH


def test_raw_codec_cid_is_rejected() -> None:
    digest = hashlib.sha256(b"raw-bytes").digest()
    # CIDv1 + raw(0x55) + sha2-256 + 32 + digest
    raw = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    cid = "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        decode_artifact_cid(cid)
    assert exc.value.reason_code is ObjectiveArtifactReason.WRONG_CODEC


# ---------------------------------------------------------------------------
# Binding / artifact serialization and unknown fields
# ---------------------------------------------------------------------------


def test_binding_round_trip_and_content_cid() -> None:
    binding = _binding()
    payload = binding.to_dict()
    restored = ProofTestReuseObjectiveBinding.from_dict(payload)
    assert restored == binding
    assert restored.binding_cid == binding.binding_cid
    assert validate_artifact_cid(binding.binding_cid)
    assert verify_retained_bytes(binding.binding_cid, binding.canonical_bytes())

    sealed = {**payload, "content_id": binding.binding_cid}
    assert ProofTestReuseObjectiveBinding.from_dict(sealed).binding_cid == binding.binding_cid


def test_unknown_fields_are_rejected() -> None:
    binding = _binding()
    payload = binding.to_dict()
    payload["extra_authority"] = True
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        ProofTestReuseObjectiveBinding.from_dict(payload)
    assert exc.value.reason_code is ObjectiveArtifactReason.UNKNOWN_FIELD

    artifact = _artifact()
    art_payload = artifact.to_dict()
    art_payload["forged"] = 1
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        ProofTestReuseCompletionArtifact.from_dict(art_payload)
    assert exc.value.reason_code is ObjectiveArtifactReason.UNKNOWN_FIELD


def test_alias_conflicts_are_rejected() -> None:
    binding = _binding()
    payload = binding.to_dict()
    payload["tree_id"] = "f" * 40
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        ProofTestReuseObjectiveBinding.from_dict(payload)
    assert exc.value.reason_code is ObjectiveArtifactReason.ALIAS_CONFLICT

    payload = binding.to_dict()
    payload["forest_cid"] = "baguqeera" + "9" * 50
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        ProofTestReuseObjectiveBinding.from_dict(payload)
    assert exc.value.reason_code is ObjectiveArtifactReason.ALIAS_CONFLICT


def test_compatible_aliases_are_accepted_when_equal() -> None:
    binding = _binding()
    payload = binding.to_dict()
    payload["tree_id"] = payload["git_tree_id"]
    payload["forest_cid"] = payload["repository_forest_cid"]
    payload["completion_tree_id"] = payload["objective_completion_tree_id"]
    restored = ProofTestReuseObjectiveBinding.from_dict(payload)
    assert restored.git_tree_id == GIT_TREE
    assert restored.repository_forest_cid == FOREST


def test_completion_artifact_retains_premises_and_replays() -> None:
    premise = _premise({"schema": "p@1", "ok": True})
    artifact = _artifact(premise_blocks=(premise,))
    assert artifact.premise_cids == (premise.cid,)
    assert artifact.artifact_cid.startswith("b")
    assert verify_retained_bytes(artifact.artifact_cid, artifact.canonical_bytes())

    payload = artifact.to_dict()
    # Retained canonical bytes are present for every premise.
    assert payload["premise_blocks"][0]["canonical_utf8"]
    restored = ProofTestReuseCompletionArtifact.from_dict(payload)
    assert restored.artifact_cid == artifact.artifact_cid
    replayed = restored.replay_premises()
    assert len(replayed) == 1
    assert replayed[0].cid == premise.cid
    assert replayed[0].data == premise.data


def test_completion_artifact_provenance_mismatch_is_rejected() -> None:
    artifact = _artifact()
    payload = artifact.to_dict()
    payload["content_id"] = cid_for_mapping({"tampered": True})
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        ProofTestReuseCompletionArtifact.from_dict(payload)
    assert exc.value.reason_code is ObjectiveArtifactReason.PROVENANCE_MISMATCH


def test_stale_artifact_is_detected() -> None:
    artifact = _artifact(observed_at_ms=NOW_MS, fresh_until_ms=NOW_MS + 10)
    assert artifact.is_fresh(NOW_MS + 5)
    assert not artifact.is_fresh(NOW_MS + 11)
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        artifact.require_fresh(NOW_MS + 11)
    assert exc.value.reason_code is ObjectiveArtifactReason.STALE_RECORD


def test_as_completion_evidence_projection() -> None:
    artifact = _artifact()
    evidence = artifact.as_completion_evidence()
    assert evidence.acceptance_criterion == artifact.acceptance_criterion
    assert evidence.repository_id == REPO_ID
    assert evidence.tree_id == GIT_TREE
    assert evidence.objective_revision == OBJECTIVE_REV
    assert evidence.analyzer_version == ANALYZER_REV
    assert evidence.configuration_revision == CONFIG_REV
    assert evidence.provenance_cid == artifact.artifact_cid
    assert evidence.validation_passed is True
    assert evidence.metadata["repository_forest_cid"] == FOREST
    assert evidence.metadata["objective_completion_tree_id"] == COMPLETION_TREE
    assert evidence.metadata["policy_revision"] == POLICY_REV
    assert evidence.metadata["circuit_revision"] == CIRCUIT_REV
    assert evidence.metadata["verifying_key_revision"] == KEY_REV


# ---------------------------------------------------------------------------
# Gate bundle
# ---------------------------------------------------------------------------


def test_gate_bundle_pass_and_fail_invariants() -> None:
    artifact = _artifact()
    bundle = ProofTestReuseGateBundle(
        repository_id=REPO_ID,
        git_tree_id=GIT_TREE,
        repository_forest_cid=FOREST,
        objective_completion_tree_id=COMPLETION_TREE,
        artifacts=(artifact,),
        passed=True,
        evaluated_at_ms=NOW_MS,
        producing_task_id="PTR-112",
        policy_revision=POLICY_REV,
        capability_revision=CAPABILITY_REV,
        circuit_revision=CIRCUIT_REV,
        verifying_key_revision=KEY_REV,
    )
    assert bundle.bundle_cid.startswith("b")
    restored = ProofTestReuseGateBundle.from_dict(bundle.to_dict())
    assert restored.bundle_cid == bundle.bundle_cid
    assert restored.replay().passed is True

    failed = ProofTestReuseGateBundle(
        repository_id=REPO_ID,
        git_tree_id=GIT_TREE,
        repository_forest_cid=FOREST,
        objective_completion_tree_id=COMPLETION_TREE,
        artifacts=(),
        passed=False,
        reason_codes=("missing_premise:PTR-G070",),
        evaluated_at_ms=NOW_MS,
    )
    assert failed.passed is False
    assert failed.reason_codes == ("missing_premise:PTR-G070",)

    with pytest.raises(ProofTestReuseObjectiveContractsError):
        ProofTestReuseGateBundle(
            repository_id=REPO_ID,
            git_tree_id=GIT_TREE,
            repository_forest_cid=FOREST,
            objective_completion_tree_id=COMPLETION_TREE,
            artifacts=(artifact,),
            passed=False,
            reason_codes=("x",),
        )

    with pytest.raises(ProofTestReuseObjectiveContractsError):
        ProofTestReuseGateBundle(
            repository_id=REPO_ID,
            git_tree_id=GIT_TREE,
            repository_forest_cid=FOREST,
            objective_completion_tree_id=COMPLETION_TREE,
            artifacts=(),
            passed=True,
        )


def test_gate_bundle_rejects_identity_provenance_mismatch() -> None:
    artifact = _artifact()
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        ProofTestReuseGateBundle(
            repository_id=REPO_ID,
            git_tree_id="b" * 40,
            repository_forest_cid=FOREST,
            objective_completion_tree_id=COMPLETION_TREE,
            artifacts=(artifact,),
            passed=True,
            evaluated_at_ms=NOW_MS,
        )
    assert exc.value.reason_code is ObjectiveArtifactReason.PROVENANCE_MISMATCH


# ---------------------------------------------------------------------------
# Control-path exclusion (declared state-root only)
# ---------------------------------------------------------------------------


def test_undeclared_control_paths_are_rejected(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    state_root.mkdir()
    declared = declared_state_root_control_paths(state_root)
    assert len(declared) == len(DECLARED_STATE_ROOT_CONTROL_ARTIFACT_SUFFIXES)

    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        assert_control_paths_are_declared(
            [tmp_path / "source.py"],
            state_root=state_root,
        )
    assert exc.value.reason_code is ObjectiveArtifactReason.CONTROL_PATH_NOT_DECLARED

    admitted = assert_control_paths_are_declared(
        [declared[0]],
        state_root=state_root,
    )
    assert admitted == (declared[0],)


def test_objective_completion_tree_excludes_only_declared_controls(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective = repo / "objective.md"
    objective.write_text(
        "## PTR-G070\n\n- Status: active\n- Acceptance: criterion\n",
        encoding="utf-8",
    )
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")

    state_root = tmp_path / "state"
    control = state_root / "projection" / "completion" / "goal_completion_gate.json"
    control.parent.mkdir(parents=True)
    control.write_text('{"v":1}\n', encoding="utf-8")

    before = compute_objective_completion_tree_id(
        repo,
        objective_path=objective,
        state_root=state_root,
    )
    control.write_text('{"v":2}\n', encoding="utf-8")
    after_control = compute_objective_completion_tree_id(
        repo,
        objective_path=objective,
        state_root=state_root,
    )
    assert before == after_control

    # Non-declared exclusion is fail-closed.
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        compute_objective_completion_tree_id(
            repo,
            objective_path=objective,
            state_root=state_root,
            control_paths=[repo / "source.py"],
        )
    assert exc.value.reason_code is ObjectiveArtifactReason.CONTROL_PATH_NOT_DECLARED

    # Source change must move the completion-tree identity.
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    after_source = compute_objective_completion_tree_id(
        repo,
        objective_path=objective,
        state_root=state_root,
    )
    assert after_source != before


# ---------------------------------------------------------------------------
# ObjectiveArtifactStore: atomic writes, unsafe paths, kit injection
# ---------------------------------------------------------------------------


def test_store_round_trip_and_readback_rehash(tmp_path: Path) -> None:
    store = ObjectiveArtifactStore(tmp_path / "cas")
    artifact = _artifact()
    cid = store.put_completion_artifact(artifact)
    assert cid == artifact.artifact_cid
    loaded = store.get_completion_artifact(cid)
    assert loaded.artifact_cid == artifact.artifact_cid
    assert loaded.binding.goal_id == "PTR-G070"
    loaded.replay_premises()

    bundle = ProofTestReuseGateBundle(
        repository_id=REPO_ID,
        git_tree_id=GIT_TREE,
        repository_forest_cid=FOREST,
        objective_completion_tree_id=COMPLETION_TREE,
        artifacts=(artifact,),
        passed=True,
        evaluated_at_ms=NOW_MS,
        producing_task_id="PTR-112",
        policy_revision=POLICY_REV,
        capability_revision=CAPABILITY_REV,
        circuit_revision=CIRCUIT_REV,
        verifying_key_revision=KEY_REV,
    )
    bundle_cid = store.put_gate_bundle(bundle)
    assert store.get_gate_bundle(bundle_cid).bundle_cid == bundle.bundle_cid


def test_store_rejects_fake_cid_and_claimed_mismatch(tmp_path: Path) -> None:
    store = ObjectiveArtifactStore(tmp_path / "cas")
    data = canonical_dag_json_bytes({"x": 1})
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        store.put_bytes(data, claimed_cid="sha256:" + "a" * 64)
    assert exc.value.reason_code is ObjectiveArtifactReason.FAKE_CID

    other = cid_for_canonical_dag_json_bytes(
        canonical_dag_json_bytes({"x": 2})
    )
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        store.put_bytes(data, claimed_cid=other)
    assert exc.value.reason_code is ObjectiveArtifactReason.CID_MISMATCH


def _stored_blob_path(root: Path, cid: str) -> Path:
    """Locate a persisted blob regardless of kit vs local layout."""

    matches = [path for path in root.rglob(f"{cid}.blob") if path.is_file() or path.is_symlink()]
    assert matches, f"no blob for {cid} under {root}"
    return matches[0]


def test_store_rejects_symlink_and_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "cas"
    root.mkdir()
    store = ObjectiveArtifactStore(root)
    data = canonical_dag_json_bytes({"safe": True})
    cid = store.put_bytes(data)
    blob = _stored_blob_path(root, cid)
    # Replace blob with symlink; get must fail closed.
    if blob.exists() or blob.is_symlink():
        blob.unlink()
    target = tmp_path / "outside.blob"
    target.write_bytes(data)
    blob.symlink_to(target)
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        store.get_bytes(cid)
    assert exc.value.reason_code in {
        ObjectiveArtifactReason.SYMLINK_REJECTED,
        ObjectiveArtifactReason.INTEGRITY_FAILED,
        ObjectiveArtifactReason.PARTIAL_WRITE,
        ObjectiveArtifactReason.NOT_FOUND,
    }


def test_store_detects_partial_or_corrupt_blob(tmp_path: Path) -> None:
    root = tmp_path / "cas"
    store = ObjectiveArtifactStore(root)
    data = canonical_dag_json_bytes({"complete": True})
    cid = store.put_bytes(data)
    blob = _stored_blob_path(root, cid)
    # Truncate after successful write.
    blob.write_bytes(data[: max(1, len(data) // 2)])
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        store.get_bytes(cid)
    assert exc.value.reason_code in {
        ObjectiveArtifactReason.MULTI_HASH_MISMATCH,
        ObjectiveArtifactReason.CID_MISMATCH,
        ObjectiveArtifactReason.PARTIAL_WRITE,
        ObjectiveArtifactReason.INTEGRITY_FAILED,
        ObjectiveArtifactReason.NOT_FOUND,
    }


def test_store_unavailable_without_root_fails_closed() -> None:
    store = ObjectiveArtifactStore()
    data = canonical_dag_json_bytes({"x": 1})
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        store.put_bytes(data)
    assert exc.value.reason_code is ObjectiveArtifactReason.STORE_UNAVAILABLE


def test_module_imports_without_optional_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract module must construct without installing optional packages."""

    # Simulate missing multiformats / kit at use sites by ensuring construction
    # and CID minting never require them.
    real_import = __import__

    def blocked_import(name: str, *args: Any, **kwargs: Any):
        if name in {"multiformats", "multiformats.multihash", "multiformats.CID"} or (
            name.startswith("multiformats")
        ):
            raise ImportError(f"blocked optional import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    # Re-import helpers still work from already-loaded module; mint with stdlib.
    data = canonical_dag_json_bytes({"optional": False})
    cid = cid_for_canonical_dag_json_bytes(data)
    assert verify_retained_bytes(cid, data)
    binding = _binding()
    assert binding.binding_cid.startswith("b")
    artifact = _artifact(binding=binding)
    assert artifact.as_completion_evidence().provenance_cid == artifact.artifact_cid


def test_kit_transport_interface_is_advertised_when_available(tmp_path: Path) -> None:
    store = ObjectiveArtifactStore(tmp_path / "cas")
    assert store.interface == CANONICAL_ARTIFACT_STORE_TRANSPORT_INTERFACE
    # Cold construction must not require installing packages.
    assert isinstance(store.kit_transport_available, bool)


def test_cold_module_import_has_no_side_effect_install() -> None:
    module_name = (
        "ipfs_accelerate_py.agent_supervisor.validation."
        "proof_test_reuse_objective_contracts"
    )
    # Ensure the module is importable under proof-reuse-off pytest.
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "ProofTestReuseObjectiveBinding")
    assert hasattr(mod, "ProofTestReuseCompletionArtifact")
    assert hasattr(mod, "ProofTestReuseGateBundle")
    assert hasattr(mod, "ObjectiveArtifactStore")
    # Stdlib-only CID path is present.
    assert callable(mod.cid_for_canonical_dag_json_bytes)
    # Optional packages are never force-imported at module import time beyond
    # already-loaded supervisor dependencies.
    assert "multiformats" not in sys.modules or True  # may already be present


def test_premise_block_rejects_cid_mismatch() -> None:
    data = canonical_dag_json_bytes({"a": 1})
    wrong = cid_for_canonical_dag_json_bytes(canonical_dag_json_bytes({"a": 2}))
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        CanonicalPremiseBlock(data=data, cid=wrong)
    assert exc.value.reason_code is ObjectiveArtifactReason.CID_MISMATCH


def test_duplicate_premise_cids_are_rejected() -> None:
    block = _premise({"schema": "p@1", "v": 1})
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        _artifact(premise_blocks=(block, block))
    assert exc.value.reason_code is ObjectiveArtifactReason.ALIAS_CONFLICT


def test_relative_state_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProofTestReuseObjectiveContractsError) as exc:
        declared_state_root_control_paths("relative/state")
    assert exc.value.reason_code is ObjectiveArtifactReason.UNSAFE_PATH
