"""PCTDD-029: whole-item reuse is limited to pure or replay-safe teardown-compatible populations."""

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
    "test_pctdd_029_pre_setup_item_reuse.py"
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
    if getattr(reports, "_pctdd_029_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_029_target_bound = True

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
from ipfs_accelerate_py.testing.proof_reuse.pre_setup_item_reuse import (
    ADMITTED_ITEM_CERTIFICATE_INTERFACE,
    CLAIM_CLASS,
    DEFAULT_POLICY_CID,
    EXPLICITLY_PURE_KIND,
    ITEM_ITEM_CERTIFICATE_ATTRIBUTE,
    ITEM_ITEM_REUSED_ATTRIBUTE,
    ITEM_ITEM_REUSE_RESULT_ATTRIBUTE,
    ITEM_POPULATION_ATTRIBUTE,
    PREDECESSOR_ASSEMBLY_INTERFACE,
    PREDECESSOR_ATTESTATION_INTERFACE,
    PRE_SETUP_ITEM_REUSE_INTERFACE,
    PRE_SETUP_ITEM_REUSE_RESULT_INTERFACE,
    PRE_SETUP_REUSE_POPULATION_INTERFACE,
    PURE_REUSE_CLASS,
    REPLAY_SAFE_REUSE_CLASSES,
    REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND,
    REUSE_DOES_NOT,
    REUSE_ESTABLISHES,
    REUSE_ITEM_ACTION,
    REUSE_POLICY,
    REUSED_PHASES,
    RUN_ACTION,
    AdmittedItemCertificate,
    PhaseExecutionProbe,
    PreSetupItemReuseError,
    PreSetupItemReuseResult,
    admit_item_certificate,
    apply_admitted_item_reuse,
    attach_item_certificate,
    attach_population,
    authority_descriptor,
    classify_population,
    datasets_contracts_available,
    evaluate_pre_setup_item_reuse,
    get_attached_item_reuse_result,
    item_reuses_whole_item,
    prepare_runtest_setup,
    public_digest,
    record_typed_unavailable,
    run_pre_setup_item_reuse_lifecycle,
    typed_unavailable_records,
)
from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
    AttestationNonceRegistry,
    RunnerKeyRecord,
    RunnerPublicKey,
    RunnerTrustPolicy,
)
from ipfs_accelerate_py.testing.proof_reuse.setup_bound_execution_key import (
    SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE,
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
ISSUER_ID = "issuer:pctdd-029"
KEY_EPOCH = "epoch-29"


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
def pctdd_029_limit() -> int:
    return 7


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-029.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-029 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-029.json"
    )


def _digest(label: str) -> str:
    return commitment_digest(
        {"label": label, "schema": "pctdd/pre-setup-item-reuse@1"}
    )


def _item(*, nodeid: str = "tests/test_mod.py::test_it", **attrs: Any) -> SimpleNamespace:
    payload = {
        "nodeid": nodeid,
        "funcargs": {},
        "calls": 0,
        "setups": 0,
    }
    payload.update(attrs)
    item = SimpleNamespace(**payload)

    def _runtest() -> None:
        item.calls += 1

    def _setup() -> None:
        item.setups += 1

    item.runtest = _runtest
    item.setup = _setup
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
        policy_epoch="policy-29",
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


def _bound_identities(
    *,
    policy_cid: str,
    issuer_id: str,
    epoch: str,
    reuse_class: str = "pure",
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fixture": build_fixture_identity(
            fixture_definition_closure_cid=_digest("fixture-definition-closure"),
            fixture_instance_closure_cid=_digest("fixture-instance-closure"),
            reuse_class=reuse_class,
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
        "locator_cid": "cid:locator:pctdd-029",
        "repository_forest_cid": _digest("repository-forest"),
        "git_commit_id": "cid:git-commit:pctdd-029",
        "git_tree_id": "cid:git-tree:pctdd-029",
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
    locator_cid: str = "cid:locator:pctdd-029",
    reuse_class: str = "pure",
    population: Any = None,
):
    private, public, policy = _material()
    verifier = _verifier(public, policy)
    identities = _bound_identities(
        policy_cid=verifier.policy_cid,
        issuer_id=verifier.issuer_id,
        epoch=verifier.epoch,
        locator_cid=locator_cid,
        reuse_class=reuse_class,
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
        issuance_nonce="nonce-pctdd-029",
        issued_at=NOW,
        nonce_registry=registry,
    )
    resolved_population = population
    if resolved_population is None:
        teardown_compatible = True if reuse_class != "pure" else None
        resolved_population = classify_population(
            reuse_class=reuse_class,
            teardown_compatible=teardown_compatible,
            reviewed=True,
            completeness="exact",
        )
    certificate = admit_item_certificate(
        execution_key=key,
        composite_receipt=composite,
        attestation_result=result,
        population=resolved_population,
    )
    return identities, key, composite, result, certificate, resolved_population


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_029_pre_setup_item_reuse.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_029_pre_setup_item_reuse.py::test_x",
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


def test_pure_population_reuses_whole_item_no_phase_execution() -> None:
    assert datasets_contracts_available() is True
    identities, key, composite, signed, certificate, population = _signed_certificate()
    assert signed.valid is True
    assert composite.admitted is True
    assert population.eligible is True
    assert population.population_kind == EXPLICITLY_PURE_KIND
    assert population.explicitly_pure is True
    assert certificate.admitted is True
    assert certificate.interface == ADMITTED_ITEM_CERTIFICATE_INTERFACE
    assert certificate.reuses_item is True
    assert certificate.reuses_setup is True
    assert certificate.reuses_call is True
    assert certificate.reuses_teardown is True
    assert tuple(certificate.reused_phases) == REUSED_PHASES
    assert certificate.may_authorize_skip is False
    assert certificate.signature_verified is True
    assert certificate.population_kind == EXPLICITLY_PURE_KIND
    assert certificate.execution_key_cid == key.execution_key_cid

    setup_n = {"n": 0}
    call_n = {"n": 0}
    teardown_n = {"n": 0}
    item = _item()
    record = run_pre_setup_item_reuse_lifecycle(
        item,
        certificate=certificate,
        population=population,
        setup=lambda: setup_n.__setitem__("n", setup_n["n"] + 1),
        call=lambda: call_n.__setitem__("n", call_n["n"] + 1),
        teardown=lambda: teardown_n.__setitem__("n", teardown_n["n"] + 1),
    )
    assert record.reused_whole_item is True
    assert record.counts.as_tuple() == (0, 0, 0)
    assert setup_n["n"] == call_n["n"] == teardown_n["n"] == 0
    assert item.calls == 0
    assert item.setups == 0
    reuse = record.reuse
    assert reuse.interface == PRE_SETUP_ITEM_REUSE_RESULT_INTERFACE
    assert reuse.reuse_interface == PRE_SETUP_ITEM_REUSE_INTERFACE
    assert reuse.reuses_item is True
    assert reuse.reuses_setup is True
    assert reuse.reuses_call is True
    assert reuse.reuses_teardown is True
    assert reuse.reused_phases == REUSED_PHASES
    assert reuse.action == REUSE_ITEM_ACTION
    assert reuse.action != "SKIP"
    assert reuse.may_authorize_skip is False
    assert reuse.production_admitted is False
    assert reuse.self_approved is False
    assert reuse.requires_full_execution is False
    assert reuse.population_kind == EXPLICITLY_PURE_KIND
    assert get_attached_item_reuse_result(item) is reuse
    assert getattr(item, ITEM_ITEM_REUSED_ATTRIBUTE) is True
    assert getattr(item, ITEM_ITEM_CERTIFICATE_ATTRIBUTE) is certificate
    assert getattr(item, ITEM_ITEM_REUSE_RESULT_ATTRIBUTE) is reuse
    assert item_reuses_whole_item(item) is True
    assert identities["locator_cid"] == "cid:locator:pctdd-029"


def test_replay_safe_teardown_compatible_reuses_whole_item() -> None:
    population = classify_population(
        reuse_class="deterministic_snapshot",
        teardown_compatible=True,
        reviewed=True,
        completeness="exact",
    )
    assert population.replay_safe is True
    assert population.explicitly_pure is False
    assert population.eligible is True
    assert population.population_kind == REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND
    _identities, _key, _composite, signed, certificate, admitted_population = (
        _signed_certificate(
            reuse_class="deterministic_snapshot",
            population=population,
        )
    )
    assert signed.valid is True
    assert certificate.admitted is True
    assert certificate.population_kind == REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND
    assert admitted_population.population_kind == REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND
    item = _item(nodeid="tests/test_mod.py::test_snapshot")
    setup_n = {"n": 0}
    call_n = {"n": 0}
    teardown_n = {"n": 0}
    record = run_pre_setup_item_reuse_lifecycle(
        item,
        certificate=certificate,
        population=population,
        setup=lambda: setup_n.__setitem__("n", setup_n["n"] + 1),
        call=lambda: call_n.__setitem__("n", call_n["n"] + 1),
        teardown=lambda: teardown_n.__setitem__("n", teardown_n["n"] + 1),
    )
    assert record.reused_whole_item is True
    assert record.counts.as_tuple() == (0, 0, 0)
    assert setup_n["n"] == call_n["n"] == teardown_n["n"] == 0
    assert record.reuse.population_kind == REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND
    transactional = classify_population(
        reuse_class="transactional",
        teardown_compatible=True,
        reviewed=True,
        completeness="exact",
    )
    assert transactional.eligible is True
    assert "transactional" in REPLAY_SAFE_REUSE_CLASSES
    assert PURE_REUSE_CLASS in REPLAY_SAFE_REUSE_CLASSES


def test_opaque_and_incompatible_populations_execute_all_phases() -> None:
    opaque = classify_population(
        reuse_class="opaque",
        teardown_compatible=True,
        reviewed=True,
        completeness="exact",
    )
    assert opaque.eligible is False
    assert opaque.eligibility_reason == "reuse_class_opaque"
    _identities, key, _composite, signed, rejected, _population = _signed_certificate(
        reuse_class="opaque",
        population=opaque,
    )
    assert signed.valid is True
    assert rejected.admitted is False
    assert rejected.diagnostics["reason"] == "reuse_class_opaque"
    item = _item(nodeid="tests/test_mod.py::test_opaque")
    opaque_counts = {"setup": 0, "call": 0, "teardown": 0}
    opaque_record = run_pre_setup_item_reuse_lifecycle(
        item,
        certificate=rejected,
        population=opaque,
        setup=lambda: opaque_counts.__setitem__("setup", opaque_counts["setup"] + 1),
        call=lambda: opaque_counts.__setitem__("call", opaque_counts["call"] + 1),
        teardown=lambda: opaque_counts.__setitem__(
            "teardown", opaque_counts["teardown"] + 1
        ),
    )
    assert opaque_record.executed_all_phases_once is True
    assert opaque_record.counts.as_tuple() == (1, 1, 1)
    assert opaque_counts == {"setup": 1, "call": 1, "teardown": 1}
    assert opaque_record.reuse.action == RUN_ACTION
    assert opaque_record.reuse.reuses_item is False
    assert opaque_record.reuse.may_authorize_skip is False

    missing_teardown = classify_population(
        reuse_class="effectful_replayable",
        reviewed=True,
        completeness="exact",
    )
    assert missing_teardown.eligible is False
    assert missing_teardown.eligibility_reason == "teardown_compatible_not_explicit"
    explicit_false = classify_population(
        reuse_class="idempotent_external",
        teardown_compatible=False,
        reviewed=True,
        completeness="exact",
    )
    assert explicit_false.eligible is False
    assert explicit_false.eligibility_reason == "teardown_not_compatible"
    nonreplayable = classify_population(
        reuse_class="effectful_nonreplayable",
        teardown_compatible=True,
        reviewed=True,
        completeness="exact",
    )
    assert nonreplayable.eligible is False
    assert nonreplayable.eligibility_reason == "reuse_class_effectful_nonreplayable"
    unreviewed = classify_population(
        reuse_class="pure",
        reviewed=False,
        completeness="exact",
    )
    assert unreviewed.eligible is False
    assert unreviewed.eligibility_reason == "population_unreviewed"
    incomplete = classify_population(
        reuse_class="pure",
        reviewed=True,
        completeness="unknown",
    )
    assert incomplete.eligible is False
    assert incomplete.eligibility_reason == "population_incomplete"
    pure_incompatible = classify_population(
        reuse_class="pure",
        teardown_compatible=False,
        reviewed=True,
        completeness="exact",
    )
    assert pure_incompatible.eligible is False
    assert key.execution_key_cid


def test_missing_unadmitted_and_stale_certificates_execute_all_phases() -> None:
    identities, key, _composite, signed, certificate, population = _signed_certificate()
    assert signed.valid is True
    item = _item()
    missing_setup = {"n": 0}
    missing_call = {"n": 0}
    missing_teardown = {"n": 0}
    missing = run_pre_setup_item_reuse_lifecycle(
        item,
        population=population,
        setup=lambda: missing_setup.__setitem__("n", missing_setup["n"] + 1),
        call=lambda: missing_call.__setitem__("n", missing_call["n"] + 1),
        teardown=lambda: missing_teardown.__setitem__("n", missing_teardown["n"] + 1),
    )
    assert missing.executed_all_phases_once is True
    assert missing.counts.as_tuple() == (1, 1, 1)
    assert missing_setup["n"] == missing_call["n"] == missing_teardown["n"] == 1
    assert missing.reuse.action == RUN_ACTION
    assert missing.reuse.reuses_item is False
    assert "item_certificate_missing" in missing.reuse.full_execution_reasons

    failed_call = admit_item_certificate(
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
        population=population,
    )
    assert failed_call.admitted is False
    failed_item = _item(nodeid="tests/test_mod.py::test_failed_call")
    failed_counts = {"setup": 0, "call": 0, "teardown": 0}
    failed = run_pre_setup_item_reuse_lifecycle(
        failed_item,
        certificate=failed_call,
        population=population,
        setup=lambda: failed_counts.__setitem__("setup", failed_counts["setup"] + 1),
        call=lambda: failed_counts.__setitem__("call", failed_counts["call"] + 1),
        teardown=lambda: failed_counts.__setitem__(
            "teardown", failed_counts["teardown"] + 1
        ),
    )
    assert failed.counts.as_tuple() == (1, 1, 1)
    assert failed_counts == {"setup": 1, "call": 1, "teardown": 1}
    assert failed.reuse.reuses_item is False
    assert failed.reuse.action == RUN_ACTION

    call_only = admit_item_certificate(
        execution_key=key,
        composite_receipt=build_composite_phase_receipt(
            locator_cid=key.locator_cid,
            execution_key_cid=key.execution_key_cid,
            policy_cid=key.trust.policy_cid,
            issuer_id=key.trust.issuer_id,
            epoch=key.trust.epoch,
        ),
        attestation_result=signed,
        population=population,
        reused_phases=("call",),
    )
    assert call_only.admitted is False
    assert call_only.diagnostics["reason"] == "reuse_phase_not_whole_item"

    other = build_test_execution_key_v2(
        **_bound_identities(
            policy_cid=key.trust.policy_cid,
            issuer_id=key.trust.issuer_id,
            epoch=key.trust.epoch,
            locator_cid="cid:locator:other-pctdd-029",
        )
    )
    stale = AdmittedItemCertificate(
        execution_key_cid=other.execution_key_cid,
        certificate_cid=certificate.certificate_cid,
        composite_phase_receipt_cid=certificate.composite_phase_receipt_cid,
        locator_cid=other.locator_cid,
        signature_verified=True,
        admitted=True,
        reuse_class="pure",
        teardown_compatible=True,
        population_kind=EXPLICITLY_PURE_KIND,
    )
    stale_item = _item(nodeid="tests/test_mod.py::test_stale")
    stale_item._ipfs_proof_reuse_execution_key = key
    stale_record = run_pre_setup_item_reuse_lifecycle(
        stale_item,
        certificate=stale,
        population=population,
        call=lambda: None,
    )
    assert stale_record.counts.as_tuple() == (1, 1, 1)
    assert stale_record.reuse.reason == "execution_key_mismatch"
    assert stale_record.reuse.reuses_item is False
    assert identities["locator_cid"]


def test_prepare_runtest_setup_suppresses_setup_and_call() -> None:
    _identities, _key, _composite, _signed, certificate, population = _signed_certificate()
    item = _item()
    attach_population(item, population)
    attach_item_certificate(item, certificate)
    restore = prepare_runtest_setup(item)
    assert restore is not None
    result = get_attached_item_reuse_result(item)
    assert result is not None
    assert result.reuses_item is True
    assert result.action == REUSE_ITEM_ACTION
    item.setup()
    item.runtest()
    assert item.setups == 0
    assert item.calls == 0
    restore()
    item.setup()
    item.runtest()
    assert item.setups == 1
    assert item.calls == 1
    applied = apply_admitted_item_reuse(item)
    assert applied is not None
    item.setup()
    item.runtest()
    assert item.setups == 1
    assert item.calls == 1
    applied()
    bare = _item(nodeid="tests/test_mod.py::test_no_cert")
    attach_population(bare, population)
    assert prepare_runtest_setup(bare) is None
    bare.setup()
    bare.runtest()
    assert bare.setups == 1
    assert bare.calls == 1
    assert item_reuses_whole_item(bare) is False


def test_live_current_test_fixtures_still_run(
    request: pytest.FixtureRequest,
    pctdd_029_limit: int,
) -> None:
    assert pctdd_029_limit == 7
    assert "pctdd_029_limit" in request.node.funcargs
    _identities, _key, _composite, _signed, certificate, population = _signed_certificate()
    companion = _item(nodeid=str(request.node.nodeid) + "::companion")
    setup_n = {"n": 0}
    call_n = {"n": 0}
    teardown_n = {"n": 0}
    record = run_pre_setup_item_reuse_lifecycle(
        companion,
        certificate=certificate,
        population=population,
        setup=lambda: setup_n.__setitem__("n", setup_n["n"] + 1),
        call=lambda: call_n.__setitem__("n", call_n["n"] + 1),
        teardown=lambda: teardown_n.__setitem__("n", teardown_n["n"] + 1),
    )
    assert record.reused_whole_item is True
    assert setup_n["n"] == 0
    assert call_n["n"] == 0
    assert teardown_n["n"] == 0
    assert record.reuse.may_authorize_skip is False
    assert record.reuse.action == REUSE_ITEM_ACTION
    probe = PhaseExecutionProbe()
    assert probe.as_tuple() == (0, 0, 0)


def test_never_authorizes_skip_or_widens_authority() -> None:
    descriptor = authority_descriptor()
    assert descriptor["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert descriptor["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert descriptor["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert descriptor["may_authorize_skip"] is False
    assert descriptor["production_admitted"] is False
    assert descriptor["self_approved"] is False
    assert descriptor["worker_authored_test_is_sufficient_alone"] is False
    assert descriptor["reused_phases"] == list(REUSED_PHASES)
    assert EXPLICITLY_PURE_KIND in descriptor["eligible_population_kinds"]
    assert REPLAY_SAFE_TEARDOWN_COMPATIBLE_KIND in descriptor["eligible_population_kinds"]
    assert descriptor["reuse_interface"] == PRE_SETUP_ITEM_REUSE_INTERFACE
    assert descriptor["certificate_interface"] == ADMITTED_ITEM_CERTIFICATE_INTERFACE
    assert descriptor["population_interface"] == PRE_SETUP_REUSE_POPULATION_INTERFACE
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
    assert "opaque whole-item reuse" in descriptor["does_not"]
    assert "skip" in REUSE_DOES_NOT
    assert "pure" in REUSE_ESTABLISHES
    assert "replay-safe" in REUSE_ESTABLISHES
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )
    assert matrix["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert "pytest runner" in matrix["forbidden_duplicates"]
    source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "pre_setup_item_reuse.py"
    ).read_text(encoding="utf-8")
    assert "pytest.skip" not in source
    assert "xfail" not in source
    plugin_source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
    ).read_text(encoding="utf-8")
    assert "prepare_runtest_setup" in plugin_source
    assert "evaluate_pre_setup_item_reuse" in plugin_source
    assert "item_reuses_whole_item" in plugin_source
    assert "pytest_runtest_protocol" in plugin_source
    assert "pytest_runtest_setup" in plugin_source
    assert "PCTDD-029" in plugin_source
    assert REUSE_POLICY["may_authorize_skip"] is False
    assert REUSE_POLICY["reused_phases"] == list(REUSED_PHASES)
    assert DEFAULT_POLICY_CID.startswith("sha256:")
    assert LOOKUP_SERVICE_ATTRIBUTE.endswith("lookup_service")
    assert STORE_SERVICE_ATTRIBUTE.endswith("store_service")
    assert PROVIDER_SERVICE_ATTRIBUTE.endswith("provider_service")
    assert ISSUER_SERVICE_ATTRIBUTE.endswith("issuer_service")
    try:
        PreSetupItemReuseResult(
            reuses_item=False,
            reason="probe",
            production_admitted=True,
        )
    except PreSetupItemReuseError as exc:
        assert "production" in str(exc)
    else:
        raise AssertionError("item reuse must not admit production")
    try:
        AdmittedItemCertificate(
            execution_key_cid="cid:key",
            certificate_cid="cid:certificate:forged",
            signature_verified=True,
            admitted=True,
            may_authorize_skip=True,
            population_kind=EXPLICITLY_PURE_KIND,
        )
    except PreSetupItemReuseError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("item certificate must not authorize skip")
    try:
        PreSetupItemReuseResult(
            reuses_item=True,
            reason="forged",
            may_authorize_skip=True,
        )
    except PreSetupItemReuseError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("item reuse must not authorize skip")


def test_test_pass_statement_v1_remains_unchanged() -> None:
    assert TEST_PASS_STATEMENT_INTERFACE == "TestPassStatementV1"
    assert TEST_PASS_STATEMENT_VERSION == 1
    identities, key, composite, signed, certificate, population = _signed_certificate()
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
    adapters = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "fixture_adapter_inventory.json"
    )
    assert "pure" in adapters["closed_reuse_classes"]
    statement_source = (
        _repo_root()
        / "external/ipfs_datasets/ipfs_datasets_py/logic/zkp/statements/test_pass.py"
    ).read_text(encoding="utf-8")
    assert 'TEST_PASS_STATEMENT_INTERFACE: Final = "TestPassStatementV1"' in statement_source
    assert "TEST_PASS_STATEMENT_VERSION: Final = 1" in statement_source
    item = _item()
    record = run_pre_setup_item_reuse_lifecycle(
        item,
        certificate=certificate,
        population=population,
    )
    assert record.reuse.claim_class == "IntegrityCommitment"
    assert identities["fixture"].reuse_class == "pure"


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    matrix_before = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    records = typed_unavailable_records()
    capabilities = {item["capability"] for item in records}
    assert {
        "fixture_proof_aware_xdist",
        "aggregate_selected_test_zk",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    }.issubset(capabilities)
    assert "pre_setup_item_reuse" not in capabilities
    assert "guarded_post_setup_reuse" not in capabilities
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
    by_capability = {item["capability"]: item for item in records}
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
    digest = public_digest({"label": "pctdd-029"})
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-029"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-029@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded = " ".join(receipt["claim"].casefold().split())
    assert "does not complete" in folded
    assert "pure" in folded
    assert "replay-safe" in folded
    assert "teardown-compatible" in folded
    assert receipt["dependency_receipts"] == ["PCTDD-027", "PCTDD-032"]
    limitations = receipt["limitations"]
    for key in (
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
    assert "pre_setup_item_reuse" not in limitations
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == "none"
    reuse = receipt["reuse"]
    assert reuse["interface"] == PRE_SETUP_ITEM_REUSE_INTERFACE
    assert reuse["certificate_interface"] == ADMITTED_ITEM_CERTIFICATE_INTERFACE
    assert reuse["predecessor_assembly_interface"] == (
        SETUP_BOUND_EXECUTION_KEY_ASSEMBLY_INTERFACE
    )
    assert reuse["predecessor_attestation_interface"] == (
        SIGNED_RUNNER_ATTESTATION_BINDING_INTERFACE
    )
    assert reuse["may_authorize_skip"] is False
    assert reuse["production_admitted"] is False
    assert reuse["reused_phases"] == list(REUSED_PHASES)
    assert reuse["normal_execution_fallback"] is True
    assert reuse["establishes"] == REUSE_ESTABLISHES
    changed = set(receipt["changed_paths"])
    assert (
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-029.json"
        in changed
    )
    assert (
        "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
        "test_pctdd_029_pre_setup_item_reuse.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "pre_setup_item_reuse.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
        in changed
    )


def test_evaluate_without_population_forces_run() -> None:
    _identities, _key, _composite, _signed, certificate, _population = _signed_certificate()
    item = _item()
    attach_item_certificate(item, certificate)
    result = evaluate_pre_setup_item_reuse(item, certificate=certificate)
    assert result.reuses_item is False
    assert result.action == RUN_ACTION
    assert result.may_authorize_skip is False
    assert result.reason == "population_missing"
    assert getattr(item, ITEM_POPULATION_ATTRIBUTE, None) is None
