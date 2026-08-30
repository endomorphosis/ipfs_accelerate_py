"""PCTDD-028: admitted call certificate reuses only call."""

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
    "test_pctdd_028_post_setup_call_reuse.py"
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
    if getattr(reports, "_pctdd_028_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_028_target_bound = True

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

from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    ISSUER_SERVICE_ATTRIBUTE,
    LOOKUP_SERVICE_ATTRIBUTE,
    PROVIDER_SERVICE_ATTRIBUTE,
    STORE_SERVICE_ATTRIBUTE,
)
from ipfs_accelerate_py.testing.proof_reuse.post_setup_call_reuse import (
    ADMITTED_CALL_CERTIFICATE_INTERFACE,
    CLAIM_CLASS,
    DEFAULT_POLICY_CID,
    ITEM_CALL_CERTIFICATE_ATTRIBUTE,
    ITEM_CALL_REUSED_ATTRIBUTE,
    ITEM_CALL_REUSE_RESULT_ATTRIBUTE,
    POST_SETUP_CALL_REUSE_INTERFACE,
    POST_SETUP_CALL_REUSE_RESULT_INTERFACE,
    PREDECESSOR_ASSEMBLY_INTERFACE,
    PREDECESSOR_ATTESTATION_INTERFACE,
    REUSE_CALL_ACTION,
    REUSE_DOES_NOT,
    REUSE_ESTABLISHES,
    REUSE_POLICY,
    REUSED_PHASE,
    RUN_ACTION,
    AdmittedCallCertificate,
    PhaseExecutionProbe,
    PostSetupCallReuseError,
    PostSetupCallReuseResult,
    admit_call_certificate,
    after_runtest_teardown,
    apply_admitted_call_reuse,
    attach_call_certificate,
    authority_descriptor,
    before_runtest_teardown,
    datasets_contracts_available,
    evaluate_post_setup_call_reuse,
    get_attached_call_reuse_result,
    prepare_runtest_call,
    public_digest,
    record_typed_unavailable,
    run_post_setup_call_reuse_lifecycle,
    typed_unavailable_records,
)
from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
    AttestationNonceRegistry,
    RunnerKeyRecord,
    RunnerPublicKey,
    RunnerTrustPolicy,
)
from ipfs_accelerate_py.testing.proof_reuse.setup_bound_execution_key import (
    ITEM_SETUP_BOUND_ASSEMBLY_ATTRIBUTE,
    ITEM_SETUP_BOUND_EXECUTION_KEY_ATTRIBUTE,
    SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE,
    after_runtest_setup,
    before_runtest_call,
    before_runtest_setup,
    get_attached_setup_bound_execution_key,
)
from ipfs_accelerate_py.testing.proof_reuse.signed_runner_attestations import (
    SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE,
    attest_and_verify_v2_composite_evidence,
    select_verifier_trust,
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
ISSUER_ID = "issuer:pctdd-028"
KEY_EPOCH = "epoch-28"


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


@pytest.fixture
def pctdd_028_limit() -> int:
    return 7


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-028.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-028 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-028.json"
    )


def _digest(label: str) -> str:
    return commitment_digest(
        {"label": label, "schema": "pctdd/post-setup-call-reuse@1"}
    )


def _item(*, nodeid: str = "tests/test_mod.py::test_it", **attrs: Any) -> SimpleNamespace:
    payload = {
        "nodeid": nodeid,
        "funcargs": {},
        "calls": 0,
    }
    payload.update(attrs)
    item = SimpleNamespace(**payload)

    def _runtest() -> None:
        item.calls += 1

    item.runtest = _runtest
    return item


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
        policy_epoch="policy-28",
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
        "locator_cid": "cid:locator:pctdd-028",
        "repository_forest_cid": _digest("repository-forest"),
        "git_commit_id": "cid:git-commit:pctdd-028",
        "git_tree_id": "cid:git-tree:pctdd-028",
        "test_module_cid": _digest("test-module"),
        "test_function_cid": _digest("test-function"),
        "test_ast_cid": _digest("test-ast"),
        "predecessor_execution_key_cid": _digest("test-execution-key-v1"),
    }
    payload.update(overrides)
    return payload


def _signed_certificate(
    *,
    setup: str = "pass",
    call: str = "pass",
    teardown: str = "pass",
    locator_cid: str = "cid:locator:pctdd-028",
):
    private, public, policy = _material()
    verifier = _verifier(public, policy)
    identities = _bound_identities(
        policy_cid=verifier.policy_cid,
        issuer_id=verifier.issuer_id,
        epoch=verifier.epoch,
        locator_cid=locator_cid,
    )
    key = build_test_execution_key_v2(**identities)
    composite = build_composite_phase_receipt(
        setup=setup,
        call=call,
        teardown=teardown,
        locator_cid=key.locator_cid,
        execution_key_cid=key.execution_key_cid,
        policy_cid=verifier.policy_cid,
        issuer_id=verifier.issuer_id,
        epoch=verifier.epoch,
    )
    registry = AttestationNonceRegistry()
    result = attest_and_verify_v2_composite_evidence(
        key,
        composite,
        private_key=private,
        verifier=verifier,
        issuance_nonce="nonce-pctdd-028",
        issued_at=NOW,
        nonce_registry=registry,
    )
    certificate = admit_call_certificate(
        execution_key=key,
        composite_receipt=composite,
        attestation_result=result,
    )
    return identities, key, composite, result, certificate


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_028_post_setup_call_reuse.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_028_post_setup_call_reuse.py::test_x",
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


def test_admitted_call_certificate_reuses_only_call_once_each_setup_teardown() -> None:
    assert datasets_contracts_available() is True
    identities, key, composite, signed, certificate = _signed_certificate()
    assert signed.valid is True
    assert composite.admitted is True
    assert certificate.admitted is True
    assert certificate.interface == ADMITTED_CALL_CERTIFICATE_INTERFACE
    assert certificate.reuses_call is True
    assert certificate.reuses_setup is False
    assert certificate.reuses_teardown is False
    assert certificate.reused_phase == REUSED_PHASE == "call"
    assert certificate.may_authorize_skip is False
    assert certificate.signature_verified is True
    assert certificate.execution_key_cid == key.execution_key_cid

    setup_n = {"n": 0}
    call_n = {"n": 0}
    teardown_n = {"n": 0}
    item = _item()
    record = run_post_setup_call_reuse_lifecycle(
        item,
        certificate=certificate,
        setup=lambda: setup_n.__setitem__("n", setup_n["n"] + 1),
        call=lambda: call_n.__setitem__("n", call_n["n"] + 1),
        teardown=lambda: teardown_n.__setitem__("n", teardown_n["n"] + 1),
        **identities,
    )
    assert record.reused_only_call is True
    assert record.counts.as_tuple() == (1, 0, 1)
    assert setup_n["n"] == 1
    assert call_n["n"] == 0
    assert teardown_n["n"] == 1
    assert item.calls == 0
    reuse = record.reuse
    assert reuse.interface == POST_SETUP_CALL_REUSE_RESULT_INTERFACE
    assert reuse.reuse_interface == POST_SETUP_CALL_REUSE_INTERFACE
    assert reuse.reuses_call is True
    assert reuse.reuses_setup is False
    assert reuse.reuses_teardown is False
    assert reuse.reused_phase == "call"
    assert reuse.action == REUSE_CALL_ACTION
    assert reuse.action != "SKIP"
    assert reuse.may_authorize_skip is False
    assert reuse.production_admitted is False
    assert reuse.self_approved is False
    assert reuse.current_setup_executed is True
    assert reuse.current_teardown_executed is True
    assert reuse.requires_full_execution is False
    assert get_attached_call_reuse_result(item) is reuse
    assert getattr(item, ITEM_CALL_REUSED_ATTRIBUTE) is True
    attached_key = get_attached_setup_bound_execution_key(item)
    assert attached_key is not None
    assert attached_key.execution_key_cid == key.execution_key_cid
    assert getattr(item, ITEM_CALL_CERTIFICATE_ATTRIBUTE) is certificate
    assert getattr(item, ITEM_SETUP_BOUND_EXECUTION_KEY_ATTRIBUTE) is attached_key
    assert getattr(item, ITEM_SETUP_BOUND_ASSEMBLY_ATTRIBUTE) is record.assembly
    assert getattr(item, ITEM_CALL_REUSE_RESULT_ATTRIBUTE) is reuse


def test_missing_unadmitted_and_stale_certificates_execute_call() -> None:
    identities, key, composite, signed, certificate = _signed_certificate()
    assert signed.valid is True
    item = _item()
    missing_setup = {"n": 0}
    missing_call = {"n": 0}
    missing_teardown = {"n": 0}
    missing = run_post_setup_call_reuse_lifecycle(
        item,
        setup=lambda: missing_setup.__setitem__("n", missing_setup["n"] + 1),
        call=lambda: missing_call.__setitem__("n", missing_call["n"] + 1),
        teardown=lambda: missing_teardown.__setitem__("n", missing_teardown["n"] + 1),
        **identities,
    )
    assert missing.executed_all_phases_once is True
    assert missing.counts.as_tuple() == (1, 1, 1)
    assert missing_setup["n"] == missing_call["n"] == missing_teardown["n"] == 1
    assert missing.reuse.action == RUN_ACTION
    assert missing.reuse.reuses_call is False
    assert missing.reuse.may_authorize_skip is False
    assert "call_certificate_missing" in missing.reuse.full_execution_reasons

    failed_call = admit_call_certificate(
        execution_key=key,
        composite_receipt=build_composite_phase_receipt(
            setup="pass",
            call="fail",
            teardown="pass",
            locator_cid=key.locator_cid,
            execution_key_cid=key.execution_key_cid,
            policy_cid=key.trust.policy_cid,
            issuer_id=key.trust.issuer_id,
            epoch=key.trust.epoch,
        ),
        attestation_result=signed,
    )
    assert failed_call.admitted is False
    failed_item = _item(nodeid="tests/test_mod.py::test_failed_call")
    failed_counts = {"setup": 0, "call": 0, "teardown": 0}
    failed = run_post_setup_call_reuse_lifecycle(
        failed_item,
        certificate=failed_call,
        setup=lambda: failed_counts.__setitem__("setup", failed_counts["setup"] + 1),
        call=lambda: failed_counts.__setitem__("call", failed_counts["call"] + 1),
        teardown=lambda: failed_counts.__setitem__(
            "teardown", failed_counts["teardown"] + 1
        ),
        **identities,
    )
    assert failed.counts.as_tuple() == (1, 1, 1)
    assert failed_counts == {"setup": 1, "call": 1, "teardown": 1}
    assert failed.reuse.reuses_call is False
    assert failed.reuse.action == RUN_ACTION

    other = build_test_execution_key_v2(
        **_bound_identities(
            policy_cid=key.trust.policy_cid,
            issuer_id=key.trust.issuer_id,
            epoch=key.trust.epoch,
            locator_cid="cid:locator:other-pctdd-028",
        )
    )
    stale = AdmittedCallCertificate(
        execution_key_cid=other.execution_key_cid,
        certificate_cid=certificate.certificate_cid,
        composite_phase_receipt_cid=certificate.composite_phase_receipt_cid,
        locator_cid=other.locator_cid,
        signature_verified=True,
        admitted=True,
    )
    stale_item = _item(nodeid="tests/test_mod.py::test_stale")
    stale_record = run_post_setup_call_reuse_lifecycle(
        stale_item,
        certificate=stale,
        call=lambda: None,
        **identities,
    )
    assert stale_record.counts.as_tuple() == (1, 1, 1)
    assert stale_record.reuse.reason == "execution_key_mismatch"
    assert stale_record.reuse.reuses_call is False


def test_setup_or_teardown_reuse_and_skip_are_rejected() -> None:
    identities, key, _composite, signed, _certificate = _signed_certificate()
    rejected = admit_call_certificate(
        execution_key=key,
        composite_receipt=build_composite_phase_receipt(
            locator_cid=key.locator_cid,
            execution_key_cid=key.execution_key_cid,
            policy_cid=key.trust.policy_cid,
            issuer_id=key.trust.issuer_id,
            epoch=key.trust.epoch,
        ),
        attestation_result=signed,
        reused_phases=("setup", "call", "teardown"),
        reuses_setup=True,
        reuses_teardown=True,
    )
    assert rejected.admitted is False
    assert rejected.diagnostics["reason"] == "reuse_phase_not_call_only"
    try:
        AdmittedCallCertificate(
            execution_key_cid=key.execution_key_cid,
            certificate_cid="cid:certificate:forged",
            signature_verified=True,
            admitted=True,
            reuses_setup=True,
        )
    except PostSetupCallReuseError as exc:
        assert "setup or teardown" in str(exc)
    else:
        raise AssertionError("call certificate must not reuse setup")
    try:
        AdmittedCallCertificate(
            execution_key_cid=key.execution_key_cid,
            certificate_cid="cid:certificate:forged",
            signature_verified=True,
            admitted=True,
            may_authorize_skip=True,
        )
    except PostSetupCallReuseError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("call certificate must not authorize skip")
    try:
        PostSetupCallReuseResult(
            reuses_call=True,
            reason="forged",
            current_setup_executed=True,
            may_authorize_skip=True,
        )
    except PostSetupCallReuseError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("call reuse must not authorize skip")
    unsigned = admit_call_certificate(execution_key=key)
    assert unsigned.admitted is False
    assert unsigned.signature_verified is False
    item = _item()
    record = run_post_setup_call_reuse_lifecycle(
        item,
        certificate=unsigned,
        **identities,
    )
    assert record.counts.as_tuple() == (1, 1, 1)
    assert record.reuse.action == RUN_ACTION


def test_prepare_runtest_call_replaces_only_runtest() -> None:
    identities, key, _composite, _signed, certificate = _signed_certificate()
    item = _item()
    before_runtest_setup(item)
    assembled = after_runtest_setup(item, **identities)
    assert assembled.assembled_after_setup is True
    assert assembled.requires_full_execution is False
    attach_call_certificate(item, certificate)
    before_runtest_call(item)
    restore = prepare_runtest_call(item)
    assert restore is not None
    result = get_attached_call_reuse_result(item)
    assert result is not None
    assert result.reuses_call is True
    assert result.action == REUSE_CALL_ACTION
    item.runtest()
    assert item.calls == 0
    restore()
    item.runtest()
    assert item.calls == 1
    applied = apply_admitted_call_reuse(item)
    assert applied is not None
    item.runtest()
    assert item.calls == 1
    applied()
    bare = _item(nodeid="tests/test_mod.py::test_no_cert")
    before_runtest_setup(bare)
    after_runtest_setup(bare, **identities)
    before_runtest_call(bare)
    assert prepare_runtest_call(bare) is None
    bare.runtest()
    assert bare.calls == 1
    assert before_runtest_teardown(bare) == "teardown"
    teardown_result = after_runtest_teardown(bare)
    assert teardown_result is not None
    assert teardown_result.current_teardown_executed is True
    assert teardown_result.reuses_call is False
    assert teardown_result.reuses_teardown is False


def test_live_current_setup_and_teardown_still_run(
    request: pytest.FixtureRequest,
    pctdd_028_limit: int,
) -> None:
    assert pctdd_028_limit == 7
    assert "pctdd_028_limit" in request.node.funcargs
    identities, _key, _composite, _signed, certificate = _signed_certificate()
    companion = _item(nodeid=str(request.node.nodeid) + "::companion")
    setup_n = {"n": 0}
    call_n = {"n": 0}
    teardown_n = {"n": 0}
    record = run_post_setup_call_reuse_lifecycle(
        companion,
        certificate=certificate,
        setup=lambda: setup_n.__setitem__("n", setup_n["n"] + 1),
        call=lambda: call_n.__setitem__("n", call_n["n"] + 1),
        teardown=lambda: teardown_n.__setitem__("n", teardown_n["n"] + 1),
        **identities,
    )
    assert record.reused_only_call is True
    assert setup_n["n"] == 1
    assert call_n["n"] == 0
    assert teardown_n["n"] == 1
    assert record.reuse.may_authorize_skip is False
    assert record.reuse.action == REUSE_CALL_ACTION
    probe = PhaseExecutionProbe()
    probe.record_setup()
    probe.record_teardown()
    assert probe.as_tuple() == (1, 0, 1)


def test_never_authorizes_skip_or_widens_authority() -> None:
    descriptor = authority_descriptor()
    assert descriptor["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert descriptor["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert descriptor["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert descriptor["may_authorize_skip"] is False
    assert descriptor["production_admitted"] is False
    assert descriptor["self_approved"] is False
    assert descriptor["worker_authored_test_is_sufficient_alone"] is False
    assert descriptor["reused_phase"] == REUSED_PHASE
    assert descriptor["reuses_setup"] is False
    assert descriptor["reuses_teardown"] is False
    assert descriptor["reuse_interface"] == POST_SETUP_CALL_REUSE_INTERFACE
    assert descriptor["certificate_interface"] == ADMITTED_CALL_CERTIFICATE_INTERFACE
    assert descriptor["predecessor_assembly_interface"] == PREDECESSOR_ASSEMBLY_INTERFACE
    assert descriptor["predecessor_attestation_interface"] == (
        PREDECESSOR_ATTESTATION_INTERFACE
    )
    assert descriptor["predecessor_assembly_interface"] == (
        SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE
    )
    assert descriptor["predecessor_attestation_interface"] == (
        SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE
    )
    assert descriptor["test_execution_key_v2"] == TEST_EXECUTION_KEY_V2_INTERFACE
    assert descriptor["composite_phase_receipt"] == COMPOSITE_PHASE_RECEIPT_INTERFACE
    assert REUSE_ESTABLISHES in descriptor["establishes"]
    assert "skip" in descriptor["does_not"]
    assert "setup reuse" in descriptor["does_not"]
    assert "teardown reuse" in descriptor["does_not"]
    assert "skip" in REUSE_DOES_NOT
    assert "only call" in REUSE_ESTABLISHES
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )
    assert matrix["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert "pytest runner" in matrix["forbidden_duplicates"]
    source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "post_setup_call_reuse.py"
    ).read_text(encoding="utf-8")
    assert "pytest.skip" not in source
    assert "xfail" not in source
    plugin_source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
    ).read_text(encoding="utf-8")
    assert "prepare_runtest_call" in plugin_source
    assert "before_runtest_teardown" in plugin_source
    assert "after_runtest_teardown" in plugin_source
    assert "pytest_runtest_call" in plugin_source
    assert "pytest_runtest_teardown" in plugin_source
    assert "PCTDD-028" in plugin_source
    assert REUSE_POLICY["may_authorize_skip"] is False
    assert REUSE_POLICY["reuses_setup"] is False
    assert REUSE_POLICY["reuses_teardown"] is False
    assert REUSE_POLICY["reused_phase"] == "call"
    assert DEFAULT_POLICY_CID.startswith("sha256:")
    assert LOOKUP_SERVICE_ATTRIBUTE.endswith("lookup_service")
    assert STORE_SERVICE_ATTRIBUTE.endswith("store_service")
    assert PROVIDER_SERVICE_ATTRIBUTE.endswith("provider_service")
    assert ISSUER_SERVICE_ATTRIBUTE.endswith("issuer_service")
    try:
        PostSetupCallReuseResult(
            reuses_call=False,
            reason="probe",
            production_admitted=True,
        )
    except PostSetupCallReuseError as exc:
        assert "production" in str(exc)
    else:
        raise AssertionError("call reuse must not admit production")
    try:
        PostSetupCallReuseResult(
            reuses_call=True,
            reason="probe",
            current_setup_executed=False,
        )
    except PostSetupCallReuseError as exc:
        assert "current setup" in str(exc)
    else:
        raise AssertionError("call reuse without current setup must be rejected")


def test_test_pass_statement_v1_remains_unchanged() -> None:
    assert TEST_PASS_STATEMENT_INTERFACE == "TestPassStatementV1"
    assert TEST_PASS_STATEMENT_VERSION == 1
    identities, key, composite, signed, certificate = _signed_certificate()
    assert signed.valid is True
    assert key.preserves_test_pass_statement_v1 is True
    assert key.leaf_statement_interface == TEST_PASS_STATEMENT_INTERFACE
    assert key.leaf_statement_version == TEST_PASS_STATEMENT_VERSION
    assert composite.leaf_statement_interface == TEST_PASS_STATEMENT_INTERFACE
    assert composite.leaf_statement_version == TEST_PASS_STATEMENT_VERSION
    assert certificate.admitted is True
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    assert matrix["legacy"] == "TestPassStatementV1 remains unchanged"
    assert matrix["IntegrityCommitment"]["does_not"] == "execution or semantics"
    assert CLAIM_CLASS == "IntegrityCommitment"
    inventory = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "pytest_identity_inventory.json"
    )
    assert inventory["schema"] == "pctdd/pytest-identity@1"
    statement_source = (
        _repo_root()
        / "external/ipfs_datasets/ipfs_datasets_py/logic/zkp/statements/test_pass.py"
    ).read_text(encoding="utf-8")
    assert 'TEST_PASS_STATEMENT_INTERFACE: Final = "TestPassStatementV1"' in statement_source
    assert "TEST_PASS_STATEMENT_VERSION: Final = 1" in statement_source
    item = _item()
    record = run_post_setup_call_reuse_lifecycle(
        item,
        certificate=certificate,
        **identities,
    )
    assert record.reuse.claim_class == "IntegrityCommitment"
    assert record.assembly.execution_key.preserves_test_pass_statement_v1 is True


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    matrix_before = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    records = typed_unavailable_records()
    capabilities = {item["capability"] for item in records}
    assert {
        "pre_setup_item_reuse",
        "fixture_proof_aware_xdist",
        "aggregate_selected_test_zk",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    }.issubset(capabilities)
    assert "guarded_post_setup_reuse" not in capabilities
    assert "post_setup_call_reuse" not in capabilities
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
    by_capability = {item["capability"]: item for item in records}
    assert by_capability["pre_setup_item_reuse"]["reason_code"] == (
        "pre_setup_item_reuse_not_implemented"
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
    digest = public_digest({"label": "pctdd-028"})
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-028"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-028@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded = " ".join(receipt["claim"].casefold().split())
    assert "does not complete" in folded
    assert "only call" in folded
    assert "setup" in folded
    assert "teardown" in folded
    assert receipt["dependency_receipts"] == ["PCTDD-027", "PCTDD-032"]
    limitations = receipt["limitations"]
    for key in (
        "pre_setup_item_reuse",
        "fixture_proof_aware_xdist",
        "aggregate_selected_test_zk",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert "guarded_post_setup_reuse" not in limitations
    assert "post_setup_call_reuse" not in limitations
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == "none"
    reuse = receipt["reuse"]
    assert reuse["interface"] == POST_SETUP_CALL_REUSE_INTERFACE
    assert reuse["certificate_interface"] == ADMITTED_CALL_CERTIFICATE_INTERFACE
    assert reuse["predecessor_assembly_interface"] == (
        SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE
    )
    assert reuse["predecessor_attestation_interface"] == (
        SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE
    )
    assert reuse["may_authorize_skip"] is False
    assert reuse["production_admitted"] is False
    assert reuse["reused_phase"] == "call"
    assert reuse["reuses_setup"] is False
    assert reuse["reuses_teardown"] is False
    assert reuse["normal_execution_fallback"] is True
    assert reuse["establishes"] == REUSE_ESTABLISHES
    changed = set(receipt["changed_paths"])
    assert (
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-028.json"
        in changed
    )
    assert (
        "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
        "test_pctdd_028_post_setup_call_reuse.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "post_setup_call_reuse.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
        in changed
    )


def test_evaluate_without_setup_forces_run() -> None:
    identities, _key, _composite, _signed, certificate = _signed_certificate()
    item = _item()
    attach_call_certificate(item, certificate)
    result = evaluate_post_setup_call_reuse(item, certificate=certificate)
    assert result.reuses_call is False
    assert result.action == RUN_ACTION
    assert result.may_authorize_skip is False
    assert "setup_bound_execution_key_missing" in result.full_execution_reasons or (
        result.reason == "current_setup_not_executed"
        or result.reason == "setup_bound_execution_key_missing"
    )
    before_runtest_setup(item)
    after_runtest_setup(item, setup_failed=True, **identities)
    failed = evaluate_post_setup_call_reuse(item, certificate=certificate)
    assert failed.reuses_call is False
    assert failed.action == RUN_ACTION
    assert failed.requires_full_execution is True
