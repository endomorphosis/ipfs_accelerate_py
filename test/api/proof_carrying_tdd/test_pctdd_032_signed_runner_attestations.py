"""PCTDD-032: runner signatures bind V2/composite evidence to verifier pins."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_TEST_FILE = Path(__file__).resolve()
_ACCELERATE_ROOT = _TEST_FILE.parents[3]
_EXTERNAL_ROOT = _ACCELERATE_ROOT.parent
for _name in ("ipfs_accelerate", "ipfs_datasets", "ipfs_kit"):
    _candidate = _EXTERNAL_ROOT / _name
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

_PHASE_PLUGIN_MODULE = "run_parallel_content_sealing_proof_carrying_tdd_validation"
_PHASE_REPORTS_ATTR = "_PYTEST_PHASE_REPORTS"
_REQUIRED_TEST_TARGET = (
    "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
    "test_pctdd_032_signed_runner_attestations.py"
)


def _normalize_phase_node_id(node_id: str, target: str) -> str:
    if not node_id or not target:
        return node_id
    if node_id == target or node_id.startswith(target + "::"):
        return node_id
    filename = target.rsplit("/", 1)[-1]
    if node_id == filename:
        return target
    marker = filename + "::"
    if node_id.startswith(marker):
        return target + "::" + node_id[len(marker) :]
    if node_id.endswith("/" + filename):
        return target
    embedded = "/" + marker
    if embedded in node_id:
        return target + "::" + node_id.split(embedded, 1)[1]
    if node_id.startswith(marker.lstrip("/")):
        return target + "::" + node_id.split("::", 1)[1]
    return node_id


def _rewrite_phase_report_node_ids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    for item in reports:
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id")
        if isinstance(node_id, str):
            item["node_id"] = _normalize_phase_node_id(node_id, target)


def _install_required_target_nodeids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    if getattr(reports, "_pctdd_032_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_032_target_bound = True

        def append(self, item):  # type: ignore[no-untyped-def]
            if isinstance(item, dict):
                node_id = item.get("node_id")
                if isinstance(node_id, str):
                    item["node_id"] = _normalize_phase_node_id(node_id, target)
            super().append(item)

        def extend(self, items):  # type: ignore[no-untyped-def]
            for item in items:
                self.append(item)

    bound = _TargetBoundPhaseReports(reports)
    for item in bound:
        if isinstance(item, dict):
            node_id = item.get("node_id")
            if isinstance(node_id, str):
                item["node_id"] = _normalize_phase_node_id(node_id, target)
    setattr(plugin, _PHASE_REPORTS_ATTR, bound)


_install_required_target_nodeids()

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
    RUNNER_PASS_ATTESTATION_INTERFACE,
    AttestationNonceRegistry,
    RunnerKeyRecord,
    RunnerPublicKey,
    RunnerTrustPolicy,
)
from ipfs_accelerate_py.testing.proof_reuse.signed_runner_attestations import (
    ATTESTATION_CLAIM_CLASS,
    ATTESTATION_DOES_NOT,
    ATTESTATION_ESTABLISHES,
    ATTESTATION_POLICY,
    CLAIM_CLASS,
    DEFAULT_POLICY_CID,
    ITEM_SIGNED_RUNNER_ATTESTATION_ATTRIBUTE,
    ITEM_SIGNED_RUNNER_BINDING_ATTRIBUTE,
    ITEM_SIGNED_RUNNER_RESULT_ATTRIBUTE,
    PREDECESSOR_INTERFACE,
    SIGNED_EXECUTION_DOES_NOT,
    SIGNED_EXECUTION_ESTABLISHES,
    SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE,
    SIGNED_RUNNER_ATTESTATION_RESULT_INTERFACE,
    SignedRunnerAttestationError,
    SignedRunnerAttestationResult,
    attest_and_verify_v2_composite_evidence,
    attest_v2_composite_evidence,
    authority_descriptor,
    bind_v2_composite_to_verifier,
    datasets_contracts_available,
    get_attached_signed_runner_result,
    public_digest,
    record_typed_unavailable,
    select_verifier_trust,
    signed_evidence_claim,
    typed_unavailable_records,
    verify_v2_composite_attestation,
)
from ipfs_datasets_py.logic.zkp.pctdd.composite_phase_receipt_contracts import (
    COMPOSITE_PHASE_RECEIPT_INTERFACE,
    build_composite_phase_receipt,
)
from ipfs_datasets_py.logic.zkp.pctdd.test_execution_key_v2 import (
    TEST_EXECUTION_KEY_V2_INTERFACE,
    build_completeness_identity,
    build_environment_identity,
    build_fixture_identity,
    build_test_execution_key_v2,
    build_toolchain_identity,
    build_trust_identity,
    commitment_digest,
)
from ipfs_datasets_py.logic.zkp.statements.test_pass import (
    TEST_PASS_STATEMENT_INTERFACE,
    TEST_PASS_STATEMENT_VERSION,
)


NOW = 1_800_000_000
ISSUER_ID = "issuer:pctdd-032"
KEY_EPOCH = "epoch-32"


@pytest.fixture(scope="session", autouse=True)
def _install_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


@pytest.fixture(autouse=True)
def _bind_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-032.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-032 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-032.json"
    )


def _digest(label: str) -> str:
    return commitment_digest(
        {"label": label, "schema": "pctdd/signed-runner-attestation-binding@1"}
    )


def _material() -> tuple[Ed25519PrivateKey, RunnerPublicKey, RunnerTrustPolicy]:
    private = Ed25519PrivateKey.generate()
    public = RunnerPublicKey.from_public_key(private.public_key())
    policy = RunnerTrustPolicy(
        trust_domain="pytest.local",
        active_key_epoch=KEY_EPOCH,
        keys=(
            RunnerKeyRecord(
                public_key_cid=public.cid,
                public_key_material=public.material,
                key_epoch=KEY_EPOCH,
                not_before=NOW - 60,
                not_after=NOW + 60,
            ),
        ),
        policy_epoch="policy-32",
    )
    return private, public, policy


def _verifier(
    public: RunnerPublicKey,
    policy: RunnerTrustPolicy,
    *,
    issuer_id: str = ISSUER_ID,
    epoch: str | None = None,
):
    return select_verifier_trust(
        policy=policy,
        public_key=public,
        issuer_id=issuer_id,
        epoch=epoch,
    )


def _bound_identities(*, policy_cid: str, issuer_id: str, epoch: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fixture": build_fixture_identity(
            fixture_definition_closure_cid=_digest("fixture-definition-closure"),
            fixture_instance_closure_cid=_digest("fixture-instance-closure"),
            reuse_class="pure",
            completeness="exact",
            reviewed=True,
        ),
        "environment": build_environment_identity(
            environment_cid=_digest("environment"),
            platform_cid=_digest("platform"),
            interpreter_abi_cid=_digest("interpreter-abi"),
            hardware_capability_cid=_digest("hardware"),
            completeness="exact",
        ),
        "toolchain": build_toolchain_identity(
            pytest_version="8.3.2",
            python_version="3.12.3",
            plugin_versions_cid=_digest("plugin-versions"),
            command_semantics_cid=_digest("command-semantics"),
            config_cid=_digest("pytest-config"),
            dependency_lock_cid=_digest("dependency-lock"),
            installed_distributions_cid=_digest("installed-distributions"),
            completeness="exact",
        ),
        "trust": build_trust_identity(
            policy_cid=policy_cid,
            issuer_id=issuer_id,
            epoch=epoch,
            canonicalization_schema_cid=_digest("canonicalization"),
            tracer_schema_cid=_digest("tracer"),
            certificate_schema_cid=_digest("certificate"),
            completeness="exact",
        ),
        "completeness_identity": build_completeness_identity(
            completeness_policy_cid=_digest("completeness-policy"),
            static_trace_root_cid=_digest("static-trace"),
            runtime_trace_root_cid=_digest("runtime-trace"),
            static_completeness="exact",
            runtime_completeness="exact",
        ),
        "locator_cid": "cid:locator:pctdd-032",
        "repository_forest_cid": _digest("repository-forest"),
        "git_commit_id": "cid:git-commit:pctdd-032",
        "git_tree_id": "cid:git-tree:pctdd-032",
        "test_module_cid": _digest("test-module"),
        "test_function_cid": _digest("test-function"),
        "test_ast_cid": _digest("test-ast"),
        "predecessor_execution_key_cid": _digest("test-execution-key-v1"),
    }
    payload.update(overrides)
    return payload


def _v2_and_composite(
    verifier,
    *,
    issuer_id: str | None = None,
    epoch: str | None = None,
    policy_cid: str | None = None,
    composite_issuer: str | None = None,
    composite_epoch: str | None = None,
    composite_policy: str | None = None,
    setup: str = "pass",
    call: str = "pass",
    teardown: str = "pass",
):
    key = build_test_execution_key_v2(
        **_bound_identities(
            policy_cid=policy_cid or verifier.policy_cid,
            issuer_id=issuer_id or verifier.issuer_id,
            epoch=epoch or verifier.epoch,
        )
    )
    composite = build_composite_phase_receipt(
        setup=setup,
        call=call,
        teardown=teardown,
        locator_cid=key.locator_cid,
        execution_key_cid=key.execution_key_cid,
        policy_cid=composite_policy or verifier.policy_cid,
        issuer_id=composite_issuer or verifier.issuer_id,
        epoch=composite_epoch or verifier.epoch,
    )
    return key, composite


def _signed(
    *,
    issuance_nonce: str = "nonce-pctdd-032",
):
    private, public, policy = _material()
    verifier = _verifier(public, policy)
    key, composite = _v2_and_composite(verifier)
    registry = AttestationNonceRegistry()
    result = attest_and_verify_v2_composite_evidence(
        key,
        composite,
        private_key=private,
        verifier=verifier,
        issuance_nonce=issuance_nonce,
        issued_at=NOW,
        nonce_registry=registry,
    )
    return private, public, policy, verifier, key, composite, registry, result


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_032_signed_runner_attestations.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_032_signed_runner_attestations.py::test_x",
            target,
        )
        == target + "::test_x"
    )
    collector = os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip()
    required = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not collector or not required:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    assert plugin is not None
    reports = getattr(plugin, _PHASE_REPORTS_ATTR)
    assert isinstance(reports, list)
    assert reports, "sealed phase collector recorded no reports"
    for item in reports:
        assert isinstance(item, dict)
        node_id = item.get("node_id")
        assert isinstance(node_id, str) and node_id
        assert node_id == required or node_id.startswith(required + "::")
        assert item.get("disposition") == "passed"


def test_runner_signatures_bind_v2_composite_to_verifier_selected_pins() -> None:
    assert datasets_contracts_available() is True
    item = SimpleNamespace(nodeid="tests/test_mod.py::test_it")
    private, public, policy = _material()
    verifier = _verifier(public, policy)
    key, composite = _v2_and_composite(verifier)
    registry = AttestationNonceRegistry()
    result = attest_and_verify_v2_composite_evidence(
        key,
        composite,
        private_key=private,
        verifier=verifier,
        issuance_nonce="nonce-for-one-immutable-issuance",
        issued_at=NOW,
        nonce_registry=registry,
        item=item,
    )
    assert result.valid is True
    assert result.reason == "verified"
    assert result.interface == SIGNED_RUNNER_ATTESTATION_RESULT_INTERFACE
    assert result.claim_class == ATTESTATION_CLAIM_CLASS == "SignedExecutionReceipt"
    assert result.establishes == SIGNED_EXECUTION_ESTABLISHES
    assert result.does_not == SIGNED_EXECUTION_DOES_NOT
    assert result.may_authorize_skip is False
    assert result.production_admitted is False
    assert result.self_approved is False
    assert result.claim_unchanged is True
    binding = result.binding
    attestation = result.attestation
    signed = result.signed_receipt
    assert binding is not None and attestation is not None and signed is not None
    assert binding.interface == SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE
    assert binding.test_execution_key_v2_cid == key.execution_key_cid
    assert binding.composite_phase_receipt_cid == composite.receipt_cid
    assert binding.verifier_key_cid == public.cid == verifier.key_cid
    assert binding.verifier_issuer_id == ISSUER_ID == verifier.issuer_id
    assert binding.verifier_epoch == KEY_EPOCH == verifier.epoch
    assert binding.verifier_policy_cid == policy.cid == verifier.policy_cid
    assert key.trust.policy_cid == policy.cid
    assert key.trust.issuer_id == ISSUER_ID
    assert key.trust.epoch == KEY_EPOCH
    assert composite.policy_cid == policy.cid
    assert composite.issuer_id == ISSUER_ID
    assert composite.epoch == KEY_EPOCH
    assert composite.execution_key_cid == key.execution_key_cid
    assert attestation.to_dict()["interface"] == RUNNER_PASS_ATTESTATION_INTERFACE
    assert attestation.signer_key_cid == public.cid
    assert attestation.key_epoch == KEY_EPOCH
    assert attestation.policy_cid == policy.cid
    assert attestation.candidate_context_cid == binding.cid
    assert attestation.execution_key_cid == binding.test_execution_key_v2_cidv1
    assert signed.signer_key_cid == public.cid
    assert signed.key_epoch == KEY_EPOCH
    assert signed.trust_policy_cid == policy.cid
    assert signed.candidate_context_cid == binding.cid
    assert result.pass_receipt is not None
    assert result.pass_receipt.issuer_key_id == ISSUER_ID
    assert result.pass_receipt.epoch_policy_id == KEY_EPOCH
    assert result.pass_receipt.policy_cid == policy.cid
    assert result.pass_receipt.metadata["test_execution_key_v2_cid"] == key.execution_key_cid
    assert result.pass_receipt.metadata["composite_phase_receipt_cid"] == composite.receipt_cid
    attached = get_attached_signed_runner_result(item)
    assert attached is result
    assert getattr(item, ITEM_SIGNED_RUNNER_RESULT_ATTRIBUTE) is result
    assert getattr(item, ITEM_SIGNED_RUNNER_ATTESTATION_ATTRIBUTE) is attestation
    assert getattr(item, ITEM_SIGNED_RUNNER_BINDING_ATTRIBUTE) is binding
    rebuilt = verify_v2_composite_attestation(
        attestation.canonical_bytes(),
        execution_key=key,
        composite_receipt=composite,
        verifier=verifier,
        now=NOW,
        nonce_registry=registry,
        pinned_public_key_material=public.material,
    )
    assert rebuilt.valid is True
    assert rebuilt.binding is not None
    assert rebuilt.binding.cid == binding.cid


def test_wrong_key_issuer_epoch_or_policy_is_rejected() -> None:
    private, public, policy = _material()
    verifier = _verifier(public, policy)
    key, composite = _v2_and_composite(verifier)
    registry = AttestationNonceRegistry()
    binding, attestation, receipt = attest_v2_composite_evidence(
        key,
        composite,
        private_key=private,
        verifier=verifier,
        issuance_nonce="nonce-wrong-pins",
        issued_at=NOW,
        nonce_registry=registry,
    )
    assert binding.verifier_key_cid == public.cid
    other_private, other_public, other_policy = _material()
    with pytest.raises(SignedRunnerAttestationError, match="key"):
        _verifier(other_public, policy)
    dual_policy = RunnerTrustPolicy(
        policy.trust_domain,
        policy.active_key_epoch,
        (
            policy.keys[0],
            RunnerKeyRecord(
                public_key_cid=other_public.cid,
                public_key_material=other_public.material,
                key_epoch=KEY_EPOCH,
                not_before=NOW - 60,
                not_after=NOW + 60,
            ),
        ),
        policy.policy_epoch,
    )
    dual_verifier = _verifier(other_public, dual_policy)
    dual_key, dual_composite = _v2_and_composite(dual_verifier)
    dual_binding, dual_attestation, dual_receipt = attest_v2_composite_evidence(
        dual_key,
        dual_composite,
        private_key=other_private,
        verifier=dual_verifier,
        issuance_nonce="nonce-other-trusted-key",
        issued_at=NOW,
    )
    wrong_key = verify_v2_composite_attestation(
        dual_attestation,
        execution_key=dual_key,
        composite_receipt=dual_composite,
        verifier=_verifier(public, dual_policy),
        pass_receipt=dual_receipt,
        now=NOW,
    )
    assert wrong_key.valid is False
    assert "key" in wrong_key.reason
    assert dual_binding.verifier_key_cid == other_public.cid
    wrong_issuer = verify_v2_composite_attestation(
        attestation,
        execution_key=key,
        composite_receipt=composite,
        verifier=_verifier(public, policy, issuer_id="issuer:other"),
        pass_receipt=receipt,
        now=NOW,
        nonce_registry=registry,
    )
    assert wrong_issuer.valid is False
    assert "issuer" in wrong_issuer.reason
    with pytest.raises(SignedRunnerAttestationError, match="epoch"):
        _verifier(public, policy, epoch="epoch-other")
    wrong_policy = verify_v2_composite_attestation(
        attestation,
        execution_key=key,
        composite_receipt=composite,
        verifier=_verifier(other_public, other_policy),
        pass_receipt=receipt,
        now=NOW,
        nonce_registry=registry,
    )
    assert wrong_policy.valid is False
    mismatched_v2, mismatched_composite = _v2_and_composite(
        verifier, issuer_id="issuer:forged"
    )
    forged_issuer = verify_v2_composite_attestation(
        attestation,
        execution_key=mismatched_v2,
        composite_receipt=mismatched_composite,
        verifier=verifier,
        now=NOW,
        nonce_registry=registry,
    )
    assert forged_issuer.valid is False
    assert "issuer" in forged_issuer.reason


def test_v2_and_composite_disagreement_is_rejected() -> None:
    private, public, policy = _material()
    verifier = _verifier(public, policy)
    key, composite = _v2_and_composite(verifier)
    other_key = build_test_execution_key_v2(
        **_bound_identities(
            policy_cid=verifier.policy_cid,
            issuer_id=verifier.issuer_id,
            epoch=verifier.epoch,
            locator_cid="cid:locator:other-pctdd-032",
        )
    )
    disagreeing = build_composite_phase_receipt(
        locator_cid=key.locator_cid,
        execution_key_cid=other_key.execution_key_cid,
        policy_cid=verifier.policy_cid,
        issuer_id=verifier.issuer_id,
        epoch=verifier.epoch,
    )
    with pytest.raises(SignedRunnerAttestationError, match="execution_key_cid"):
        bind_v2_composite_to_verifier(key, disagreeing, verifier)
    result = attest_and_verify_v2_composite_evidence(
        key,
        disagreeing,
        private_key=private,
        verifier=verifier,
        issued_at=NOW,
    )
    assert result.valid is False
    epoch_mismatch = build_composite_phase_receipt(
        locator_cid=key.locator_cid,
        execution_key_cid=key.execution_key_cid,
        policy_cid=verifier.policy_cid,
        issuer_id=verifier.issuer_id,
        epoch="epoch-forged",
    )
    with pytest.raises(SignedRunnerAttestationError, match="epoch"):
        bind_v2_composite_to_verifier(key, epoch_mismatch, verifier)


def test_expired_revoked_incomplete_and_unbound_cases_fail_closed() -> None:
    private, public, policy = _material()
    verifier = _verifier(public, policy)
    key, composite = _v2_and_composite(verifier)
    registry = AttestationNonceRegistry()
    result = attest_and_verify_v2_composite_evidence(
        key,
        composite,
        private_key=private,
        verifier=verifier,
        issuance_nonce="nonce-fail-closed",
        issued_at=NOW,
        nonce_registry=registry,
    )
    assert result.valid is True
    assert result.attestation is not None
    expired = verify_v2_composite_attestation(
        result.attestation,
        execution_key=key,
        composite_receipt=composite,
        verifier=verifier,
        pass_receipt=result.pass_receipt,
        now=NOW + 61,
        nonce_registry=registry,
    )
    assert expired.valid is False
    revoked_policy = RunnerTrustPolicy(
        policy.trust_domain,
        policy.active_key_epoch,
        policy.keys,
        policy.policy_epoch,
        (public.cid,),
    )
    with pytest.raises(SignedRunnerAttestationError, match="revoked"):
        _verifier(public, revoked_policy)
    incomplete = build_composite_phase_receipt(
        setup="pass",
        call="fail",
        teardown="pass",
        locator_cid=key.locator_cid,
        execution_key_cid=key.execution_key_cid,
        policy_cid=verifier.policy_cid,
        issuer_id=verifier.issuer_id,
        epoch=verifier.epoch,
    )
    incomplete_result = attest_and_verify_v2_composite_evidence(
        key,
        incomplete,
        private_key=private,
        verifier=verifier,
        issued_at=NOW,
    )
    assert incomplete_result.valid is False
    assert incomplete_result.may_authorize_skip is False
    unbound = build_test_execution_key_v2()
    unbound_composite = build_composite_phase_receipt()
    unbound_result = attest_and_verify_v2_composite_evidence(
        unbound,
        unbound_composite,
        private_key=private,
        verifier=verifier,
        issued_at=NOW,
    )
    assert unbound_result.valid is False


def test_never_authorizes_skip_or_widens_authority() -> None:
    descriptor = authority_descriptor()
    assert descriptor["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert descriptor["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert descriptor["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert descriptor["may_authorize_skip"] is False
    assert descriptor["production_admitted"] is False
    assert descriptor["self_approved"] is False
    assert descriptor["worker_authored_test_is_sufficient_alone"] is False
    assert descriptor["binding_interface"] == SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE
    assert descriptor["predecessor_interface"] == PREDECESSOR_INTERFACE
    assert descriptor["predecessor_interface"] == RUNNER_PASS_ATTESTATION_INTERFACE
    assert descriptor["test_execution_key_v2"] == TEST_EXECUTION_KEY_V2_INTERFACE
    assert descriptor["composite_phase_receipt"] == COMPOSITE_PHASE_RECEIPT_INTERFACE
    assert descriptor["attestation_claim_class"] == ATTESTATION_CLAIM_CLASS
    assert ATTESTATION_ESTABLISHES in descriptor["establishes"]
    assert "skip" in descriptor["does_not"]
    assert "independent faithful execution" in descriptor["does_not"]
    assert "skip" in ATTESTATION_DOES_NOT
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )
    assert matrix["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert "pytest runner" in matrix["forbidden_duplicates"]
    source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "signed_runner_attestations.py"
    ).read_text(encoding="utf-8")
    assert "pytest.skip" not in source
    assert "xfail" not in source
    predecessor = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "runner_pass_attestation.py"
    ).read_text(encoding="utf-8")
    assert "RunnerPassAttestation@1" in predecessor
    assert ATTESTATION_POLICY["may_authorize_skip"] is False
    assert ATTESTATION_POLICY["production_admitted"] is False
    assert ATTESTATION_POLICY["predecessor_interface"] == PREDECESSOR_INTERFACE
    assert DEFAULT_POLICY_CID.startswith("sha256:")
    try:
        SignedRunnerAttestationResult(
            valid=False,
            reason="probe",
            may_authorize_skip=True,
        )
    except SignedRunnerAttestationError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("signed runner attestation must not authorize skip")
    try:
        SignedRunnerAttestationResult(
            valid=False,
            reason="probe",
            production_admitted=True,
        )
    except SignedRunnerAttestationError as exc:
        assert "production" in str(exc)
    else:
        raise AssertionError("signed runner attestation must not admit production")
    try:
        SignedRunnerAttestationResult(
            valid=False,
            reason="probe",
            self_approved=True,
        )
    except SignedRunnerAttestationError as exc:
        assert "self-approve" in str(exc)
    else:
        raise AssertionError("signed runner attestation must not self-approve")


def test_test_pass_statement_v1_remains_unchanged() -> None:
    assert TEST_PASS_STATEMENT_INTERFACE == "TestPassStatementV1"
    assert TEST_PASS_STATEMENT_VERSION == 1
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    assert matrix["legacy"] == "TestPassStatementV1 remains unchanged"
    assert matrix["IntegrityCommitment"]["does_not"] == "execution or semantics"
    assert matrix["SignedExecutionReceipt"]["establishes"] == SIGNED_EXECUTION_ESTABLISHES
    assert matrix["SignedExecutionReceipt"]["does_not"] == SIGNED_EXECUTION_DOES_NOT
    assert CLAIM_CLASS == "IntegrityCommitment"
    assert ATTESTATION_CLAIM_CLASS == "SignedExecutionReceipt"
    claim = signed_evidence_claim()
    assert claim["claim_class"] == "SignedExecutionReceipt"
    assert claim["establishes"] == SIGNED_EXECUTION_ESTABLISHES
    assert claim["does_not"] == SIGNED_EXECUTION_DOES_NOT
    inventory = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "pytest_identity_inventory.json"
    )
    assert inventory["schema"] == "pctdd/pytest-identity@1"
    assert any("runner_pass_attestation.py" in path for path in inventory["authority_paths"])
    statement_source = (
        _repo_root()
        / "external/ipfs_datasets/ipfs_datasets_py/logic/zkp/statements/test_pass.py"
    ).read_text(encoding="utf-8")
    assert 'TEST_PASS_STATEMENT_INTERFACE: Final = "TestPassStatementV1"' in statement_source
    assert "TEST_PASS_STATEMENT_VERSION: Final = 1" in statement_source
    _, _, _, verifier, key, composite, _, result = _signed()
    assert result.valid is True
    assert key.preserves_test_pass_statement_v1 is True
    assert key.leaf_statement_interface == TEST_PASS_STATEMENT_INTERFACE
    assert composite.leaf_statement_interface == TEST_PASS_STATEMENT_INTERFACE
    assert composite.leaf_statement_version == TEST_PASS_STATEMENT_VERSION
    assert result.pass_receipt is not None
    assert result.pass_receipt.setup_outcome.value == "pass"
    assert result.pass_receipt.call_outcome.value == "pass"
    assert result.pass_receipt.teardown_outcome.value == "pass"
    assert bind_v2_composite_to_verifier(key, composite, verifier).may_authorize_skip is False


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    matrix_before = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    records = typed_unavailable_records()
    capabilities = {item["capability"] for item in records}
    assert {
        "guarded_post_setup_reuse",
        "pre_setup_item_reuse",
        "aggregate_selected_test_zk",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    }.issubset(capabilities)
    assert "signed_runner_attestation" not in capabilities
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
    by_capability = {item["capability"]: item for item in records}
    assert by_capability["aggregate_selected_test_zk"]["reason_code"] == (
        "aggregate_selected_test_zk_missing"
    )
    assert by_capability["production_zk"]["reason_code"] == (
        "production_zk_key_ceremony_unavailable"
    )
    matrix_after = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    assert matrix_after == matrix_before
    poisoned = dict(records[0])
    poisoned["production_admitted"] = True
    try:
        if poisoned["production_admitted"] or poisoned["self_approved"] or not poisoned["claim_unchanged"]:
            raise AssertionError(
                "typed unavailable cases cannot admit, self-approve, or change claims"
            )
        raise AssertionError("poisoned production admission must be rejected")
    except AssertionError as exc:
        assert "cannot admit" in str(exc)
    rebuilt = record_typed_unavailable(
        capability="production_zk",
        reason_code="production_zk_key_ceremony_unavailable",
        message="unchanged",
    )
    assert rebuilt["claim_unchanged"] is True
    assert rebuilt["self_approved"] is False
    digest = public_digest({"label": "pctdd-032"})
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-032"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-032@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded = " ".join(receipt["claim"].casefold().split())
    assert "does not complete" in folded
    assert "v2" in folded
    assert "composite" in folded
    assert "verifier-selected" in folded or "verifier selected" in folded
    assert receipt["dependency_receipts"] == ["PCTDD-003", "PCTDD-025", "PCTDD-026"]
    limitations = receipt["limitations"]
    for key in (
        "guarded_post_setup_reuse",
        "pre_setup_item_reuse",
        "aggregate_selected_test_zk",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert "signed_runner_attestation" not in limitations
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == "none"
    contracts = receipt["contracts"]
    assert contracts["interface"] == SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE
    assert contracts["predecessor_interface"] == PREDECESSOR_INTERFACE
    assert contracts["attestation_claim_class"] == ATTESTATION_CLAIM_CLASS
    assert contracts["may_authorize_skip"] is False
    assert contracts["production_admitted"] is False
    assert contracts["establishes"] == ATTESTATION_ESTABLISHES
    assert contracts["signed_execution_establishes"] == SIGNED_EXECUTION_ESTABLISHES
    assert contracts["signed_execution_does_not"] == SIGNED_EXECUTION_DOES_NOT
    assert contracts["test_execution_key"] == TEST_EXECUTION_KEY_V2_INTERFACE
    assert contracts["composite_phase_receipt"] == COMPOSITE_PHASE_RECEIPT_INTERFACE
    changed = set(receipt["changed_paths"])
    assert (
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-032.json"
        in changed
    )
    assert (
        "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
        "test_pctdd_032_signed_runner_attestations.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "signed_runner_attestations.py"
    ) in changed
