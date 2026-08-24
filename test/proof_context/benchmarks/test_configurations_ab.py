from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.proof_context.benchmarks.configurations_ab import (
    CONFIGURATION_A_CID,
    CONFIGURATION_B_CID,
    METRIC_NAMES,
    RUNNER_DESCRIPTOR_CID,
    BenchmarkInvocation,
    ConfigurationABError,
    ContextChunk,
    ExecutionPermit,
    FullVerificationObservation,
    HiddenDataDenied,
    OrdinaryRetrieval,
    PairIdentity,
    ProviderObservation,
    ProviderUnavailable,
    SemanticContextPack,
    TaskAgentView,
    VerificationRequest,
    configuration_cid,
    configuration_descriptor,
    estimate_context_tokens,
    run_arm,
    run_paired_ab,
    runner_descriptor,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_obj


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode(), codec="raw")


def _identity(**changes: Any) -> PairIdentity:
    values: dict[str, Any] = {
        "corpus_manifest_cid": _cid("corpus"),
        "task_record_cid": _cid("task"),
        "visible_projection_cid": _cid("visible"),
        "repository_state_cid": _cid("repository"),
        "environment_cid": _cid("environment"),
        "task_id": "typed-001",
        "provider_id": "provider/frontier",
        "model_id": "frontier-model",
        "model_revision": "frontier-model@immutable-2026-08-24",
        "seed": 60060,
        "attempt": 1,
    }
    values.update(changes)
    return PairIdentity(**values)


def _task() -> TaskAgentView:
    return TaskAgentView(
        objective="Correct the visible converter without broadening its public API.",
        owned_paths=("src/converter.py", "tests/test_converter.py"),
        routine_localized=True,
        risk_class="routine",
    )


def _ordinary() -> OrdinaryRetrieval:
    chunks = tuple(
        sorted(
            (
                ContextChunk(
                    path="src/converter.py",
                    text="def convert(value: int) -> str:\n    return str(value)\n" * 3,
                    content_cid=cid_for_bytes(
                        ("def convert(value: int) -> str:\n    return str(value)\n" * 3).encode(),
                        codec="raw",
                    ),
                ),
                ContextChunk(
                    path="tests/test_converter.py",
                    text="def test_convert():\n    assert convert(1) == '1'\n" * 2,
                    content_cid=cid_for_bytes(
                        ("def test_convert():\n    assert convert(1) == '1'\n" * 2).encode(),
                        codec="raw",
                    ),
                ),
            ),
            key=lambda item: (item.path, item.content_cid),
        )
    )
    return OrdinaryRetrieval(chunks)


def _pack() -> SemanticContextPack:
    rendered = "converter signature and integer-to-string contract"
    return SemanticContextPack(
        pack_cid=_cid("context-pack"),
        visible_projection_cid=_cid("visible"),
        rendered_context=rendered,
        declared_tokens=estimate_context_tokens(rendered),
        exact_source_tokens=4,
        capsule_tokens=3,
        fallback_count=1,
    )


def _permit(**changes: Any) -> ExecutionPermit:
    values: dict[str, Any] = {
        "permit_cid": _cid("permit"),
        "provider_id": "provider/frontier",
        "model_id": "frontier-model",
        "model_revision": "frontier-model@immutable-2026-08-24",
        "environment_cid": _cid("environment"),
        "provenance": "replayed",
        "available": True,
        "live_execution_eligible": False,
        "reason": "deterministic reviewed replay fixture",
    }
    values.update(changes)
    return ExecutionPermit(**values)


def _provider_observation(**changes: Any) -> ProviderObservation:
    values: dict[str, Any] = {
        "status": "succeeded",
        "provenance": "replayed",
        "evidence_cid": _cid("provider-evidence"),
        "proposal_cid": _cid("proposal"),
        "input_tokens": 120,
        "output_tokens": 21,
        "cached_input_tokens": 0,
        "inference_cost_micros": 310,
        "failure_cost_micros": 0,
    }
    values.update(changes)
    return ProviderObservation(**values)


def _verification_observation(**changes: Any) -> FullVerificationObservation:
    values: dict[str, Any] = {
        "proposal_cid": _cid("proposal"),
        "evidence_cid": _cid("verification-evidence"),
        "full_verification": True,
        "hidden_scoring_after_proposal": True,
        "full_test_count": 12,
        "full_test_pass_count": 12,
        "hidden_test_total_count": 4,
        "hidden_test_pass_count": 4,
        "regression_count": 0,
        "critical_regression_count": 0,
        "out_of_scope_edit_count": 0,
        "semantic_outcome_match": True,
        "verification_cost_micros": 90,
    }
    values.update(changes)
    return FullVerificationObservation(**values)


class RecordingProvider:
    def __init__(
        self,
        observation: ProviderObservation | None = None,
        *,
        unavailable: bool = False,
    ) -> None:
        self.observation = observation or _provider_observation()
        self.unavailable = unavailable
        self.invocations: list[BenchmarkInvocation] = []

    def propose(self, invocation: BenchmarkInvocation) -> ProviderObservation:
        self.invocations.append(invocation)
        if self.unavailable:
            raise ProviderUnavailable("fixture provider unavailable")
        return self.observation


class RecordingVerifier:
    def __init__(self, observation: FullVerificationObservation | None = None) -> None:
        self.observation = observation or _verification_observation()
        self.requests: list[VerificationRequest] = []

    def verify(self, request: VerificationRequest) -> FullVerificationObservation:
        self.requests.append(request)
        return self.observation


def _run_success(configuration_id: str = "A"):
    provider = RecordingProvider()
    verifier = RecordingVerifier()
    context = _ordinary() if configuration_id == "A" else _pack()
    run = run_arm(
        configuration_id=configuration_id,
        identity=_identity(),
        task=_task(),
        context=context,
        permit=_permit(),
        provider=provider,
        verifier=verifier,
    )
    return run, provider, verifier


def test_frozen_descriptors_and_cids_match_pcce060() -> None:
    assert CONFIGURATION_A_CID == "baguqeerab5ltrfd6dasxes2r76svhxbbcbj7hcjqgxpwsgi2rfbn53x2lmha"
    assert CONFIGURATION_B_CID == "baguqeeravan66pqbuayxtc6wzsqcn5ocbzwflqbdbeoowlvtore26hqmwzrq"
    assert configuration_cid("A") == cid_for_obj(configuration_descriptor("A"), codec="dag-json")
    assert configuration_cid("B") == cid_for_obj(configuration_descriptor("B"), codec="dag-json")


def test_only_a_to_b_treatment_difference_is_context_method() -> None:
    arm_a = configuration_descriptor("A")
    arm_b = configuration_descriptor("B")
    changed = sorted(
        key
        for key in set(arm_a) | set(arm_b)
        if key != "configuration_id" and arm_a.get(key) != arm_b.get(key)
    )
    assert changed == ["context_method"]
    assert arm_a["model_policy"] == arm_b["model_policy"]
    assert arm_a["verification_policy"] == arm_b["verification_policy"]
    for forbidden in (
        "routing_enabled",
        "incremental_verification_enabled",
        "proof_reuse_enabled",
        "sufficiency_enabled",
        "context_expansion_enabled",
        "assurance_enabled",
        "incremental_seal_enabled",
        "human_escalation_enabled",
    ):
        assert arm_a[forbidden] is False
        assert arm_b[forbidden] is False


def test_descriptor_copies_cannot_mutate_freeze() -> None:
    candidate = configuration_descriptor("A")
    candidate["model_policy"] = "changed"
    assert configuration_descriptor("A")["model_policy"] == (
        "execution-permit-exact-frontier-pair@1"
    )
    with pytest.raises(ConfigurationABError):
        configuration_descriptor("C")


def test_runner_module_is_provider_neutral_and_import_time_io_free() -> None:
    module_path = (
        Path(__file__).parents[3]
        / "ipfs_accelerate_py/proof_context/benchmarks/configurations_ab.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    denied_imports = {"requests", "httpx", "socket", "subprocess", "urllib"}
    imported = {
        node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not imported & denied_imports
    assert "open(" not in module_path.read_text(encoding="utf-8")


def test_runner_descriptor_and_metric_vocabulary_are_content_addressed() -> None:
    assert len(METRIC_NAMES) == 78
    assert len(METRIC_NAMES) == len(set(METRIC_NAMES))
    assert RUNNER_DESCRIPTOR_CID == cid_for_obj(runner_descriptor(), codec="dag-json")
    assert runner_descriptor()["only_a_to_b_difference"] == ["context_method"]


def test_ordinary_retrieval_is_exact_and_canonical() -> None:
    context = _ordinary()
    assert context.token_count == estimate_context_tokens(context.rendered_context)
    assert "<<<VISIBLE" in context.rendered_context
    with pytest.raises(ConfigurationABError, match="canonical-ordered"):
        OrdinaryRetrieval(tuple(reversed(context.chunks)))
    with pytest.raises(ConfigurationABError, match="content_cid"):
        ContextChunk("src/value.py", "x = 1\n", _cid("wrong"))


@pytest.mark.parametrize(
    "path",
    (
        "../answer.py",
        "/tmp/source.py",
        "sealed_evaluator/test_secret.py",
        "answers/patch.diff",
        ".git/config",
        "src\\escape.py",
    ),
)
def test_provider_projection_denies_hidden_or_escaping_paths(path: str) -> None:
    with pytest.raises(HiddenDataDenied):
        TaskAgentView("objective", (path,), True, "routine")


def test_task_mapping_rejects_hidden_answer_fields() -> None:
    visible = {
        "objective": "fix visible source",
        "owned_paths": ["src/value.py"],
        "routine_localized": True,
        "risk_class": "routine",
        "expected_patch": "hidden bytes",
    }
    with pytest.raises(HiddenDataDenied):
        TaskAgentView.from_mapping(visible)


def test_semantic_pack_rechecks_tokens_and_disables_expansion() -> None:
    with pytest.raises(ConfigurationABError, match="token count"):
        replace(_pack(), declared_tokens=_pack().declared_tokens + 1)
    with pytest.raises(ConfigurationABError, match="does not enable context expansion"):
        replace(_pack(), expansion_count=1, expansion_tokens=2)


def test_semantic_pack_must_bind_the_exact_visible_projection() -> None:
    with pytest.raises(HiddenDataDenied, match="different visible projection"):
        run_arm(
            configuration_id="B",
            identity=_identity(),
            task=_task(),
            context=replace(_pack(), visible_projection_cid=_cid("other-visible")),
            permit=_permit(),
            provider=RecordingProvider(),
            verifier=RecordingVerifier(),
        )


@pytest.mark.parametrize(
    "permit",
    (
        _permit(available=False, reason="provider absent"),
        _permit(
            provenance="live",
            live_execution_eligible=False,
            reason="PCCE-056 release gate is no-go",
        ),
        _permit(model_revision="different-revision", reason="revision mismatch"),
    ),
)
def test_unavailable_or_inexact_permit_never_dispatches(permit: ExecutionPermit) -> None:
    provider = RecordingProvider()
    verifier = RecordingVerifier()
    run = run_arm(
        configuration_id="A",
        identity=_identity(),
        task=_task(),
        context=_ordinary(),
        permit=permit,
        provider=provider,
        verifier=verifier,
    )
    assert run.raw_result["terminal_status"] == "unavailable"
    assert run.raw_result["metrics"]["provider_call_count"] == 0
    assert run.raw_result["metrics"]["inference_cost_micros"] is None
    assert "inference_cost_micros" in run.raw_result["missingness"]
    assert not provider.invocations
    assert not verifier.requests
    assert run.audit["provider_dispatched"] is False


@pytest.mark.parametrize("configuration_id", ("A", "B"))
def test_successful_replay_records_full_verification(configuration_id: str) -> None:
    run, provider, verifier = _run_success(configuration_id)
    raw = run.raw_result
    assert raw["schema"] == "ipfs-datasets.proof-context.benchmark-raw-result@1"
    assert raw["terminal_status"] == "succeeded"
    assert raw["provenance"] == "replayed"
    assert raw["configuration_cid"] == configuration_cid(configuration_id)
    assert raw["metrics"]["full_test_count"] == 12
    assert raw["metrics"]["full_test_pass_count"] == 12
    assert raw["metrics"]["accepted_patch_count"] == 1
    assert raw["metrics"]["provider_call_count"] == 1
    assert raw["metrics"]["route_frontier_count"] == 1
    assert raw["metrics"]["total_cost_micros"] == 400
    assert set(raw["missingness"]) == {
        key for key, value in raw["metrics"].items() if value is None
    }
    assert isinstance(run.as_dict()["evidence_cids"], list)
    assert isinstance(run.as_dict()["metrics"], dict)
    assert len(provider.invocations) == 1
    assert len(verifier.requests) == 1
    assert not hasattr(provider.invocations[0], "hidden_tests")
    assert verifier.requests[0].hidden_mount_phase == "after-patch-proposal"
    assert run.audit["incremental_verification_used"] is False
    assert run.audit["hidden_data_shared_with_provider"] is False
    with pytest.raises(TypeError):
        run.raw_result["terminal_status"] = "invalid"
    with pytest.raises(TypeError):
        run.raw_result["metrics"]["accepted_patch_count"] = 0


def test_a_and_b_record_their_distinct_context_measurements() -> None:
    arm_a, _, _ = _run_success("A")
    arm_b, _, _ = _run_success("B")
    assert arm_a.raw_result["metrics"]["ordinary_retrieval_tokens"] == _ordinary().token_count
    assert arm_a.raw_result["metrics"]["context_pack_tokens"] is None
    assert arm_b.raw_result["metrics"]["context_pack_tokens"] == _pack().token_count
    assert arm_b.raw_result["metrics"]["ordinary_retrieval_tokens"] is None
    assert arm_b.raw_result["metrics"]["context_fallback_count"] == 1
    assert arm_b.raw_result["metrics"]["context_expansion_count"] == 0


def test_simulated_success_is_forced_to_rejected_simulation() -> None:
    provider = RecordingProvider(
        _provider_observation(provenance="simulated", evidence_cid=_cid("sim-provider"))
    )
    verifier = RecordingVerifier()
    run = run_arm(
        configuration_id="A",
        identity=_identity(),
        task=_task(),
        context=_ordinary(),
        permit=_permit(provenance="simulated"),
        provider=provider,
        verifier=verifier,
    )
    assert run.raw_result["terminal_status"] == "simulated"
    assert run.raw_result["metrics"]["simulated_success_accepted_count"] == 0
    assert run.raw_result["metrics"]["accepted_patch_count"] is None
    assert not verifier.requests


def test_provider_cannot_relabel_replay_as_live() -> None:
    provider = RecordingProvider(_provider_observation(provenance="live"))
    with pytest.raises(ConfigurationABError, match="provenance differs"):
        run_arm(
            configuration_id="A",
            identity=_identity(),
            task=_task(),
            context=_ordinary(),
            permit=_permit(provenance="replayed"),
            provider=provider,
            verifier=RecordingVerifier(),
        )


def test_provider_failure_preserves_observed_cost() -> None:
    provider = RecordingProvider(
        _provider_observation(
            status="timeout",
            proposal_cid=None,
            input_tokens=80,
            output_tokens=0,
            inference_cost_micros=200,
            failure_cost_micros=35,
            reason="bounded provider timeout",
        )
    )
    run = run_arm(
        configuration_id="A",
        identity=_identity(),
        task=_task(),
        context=_ordinary(),
        permit=_permit(),
        provider=provider,
        verifier=RecordingVerifier(),
    )
    assert run.raw_result["terminal_status"] == "timeout"
    assert run.raw_result["metrics"]["total_cost_micros"] == 235
    assert run.raw_result["metrics"]["failed_attempt_cost_micros"] == 235


def test_provider_unavailable_exception_is_not_imputed() -> None:
    provider = RecordingProvider(unavailable=True)
    verifier = RecordingVerifier()
    run = run_arm(
        configuration_id="B",
        identity=_identity(),
        task=_task(),
        context=_pack(),
        permit=_permit(),
        provider=provider,
        verifier=verifier,
    )
    assert run.raw_result["terminal_status"] == "unavailable"
    assert run.raw_result["metrics"]["provider_call_count"] == 1
    assert run.raw_result["metrics"]["inference_cost_micros"] is None
    assert not verifier.requests


@pytest.mark.parametrize(
    ("observation", "message"),
    (
        (_verification_observation(full_verification=False), "incremental-only"),
        (
            _verification_observation(hidden_scoring_after_proposal=False),
            "before the proposal",
        ),
        (
            _verification_observation(proposal_cid=_cid("wrong-proposal")),
            "different proposal",
        ),
    ),
)
def test_verification_must_be_full_late_and_proposal_bound(
    observation: FullVerificationObservation, message: str
) -> None:
    with pytest.raises(ConfigurationABError, match=message):
        run_arm(
            configuration_id="A",
            identity=_identity(),
            task=_task(),
            context=_ordinary(),
            permit=_permit(),
            provider=RecordingProvider(),
            verifier=RecordingVerifier(observation),
        )


def test_full_verification_cannot_claim_an_empty_suite() -> None:
    with pytest.raises(ConfigurationABError, match="full_test_count"):
        _verification_observation(full_test_count=0, full_test_pass_count=0)
    with pytest.raises(ConfigurationABError, match="hidden_test_total_count"):
        _verification_observation(hidden_test_total_count=0, hidden_test_pass_count=0)


def test_failed_full_verification_is_visible_and_not_accepted() -> None:
    run = run_arm(
        configuration_id="B",
        identity=_identity(),
        task=_task(),
        context=_pack(),
        permit=_permit(),
        provider=RecordingProvider(),
        verifier=RecordingVerifier(
            _verification_observation(
                full_test_pass_count=11,
                hidden_test_pass_count=3,
                regression_count=1,
                semantic_outcome_match=False,
            )
        ),
    )
    assert run.raw_result["terminal_status"] == "verification_failed"
    assert run.raw_result["metrics"]["accepted_patch_count"] == 0
    assert run.raw_result["metrics"]["failed_attempt_cost_micros"] == 400


def test_pairing_rejects_identity_or_permit_drift_before_dispatch() -> None:
    provider_a = RecordingProvider()
    provider_b = RecordingProvider()
    with pytest.raises(ConfigurationABError, match="paired identity mismatch"):
        run_paired_ab(
            identity_a=_identity(),
            identity_b=_identity(seed=60061),
            task_a=_task(),
            task_b=_task(),
            ordinary_context=_ordinary(),
            semantic_context=_pack(),
            permit_a=_permit(),
            permit_b=_permit(),
            provider_a=provider_a,
            provider_b=provider_b,
            verifier_a=RecordingVerifier(),
            verifier_b=RecordingVerifier(),
        )
    assert not provider_a.invocations
    assert not provider_b.invocations
    with pytest.raises(ConfigurationABError, match="paired permits"):
        run_paired_ab(
            identity_a=_identity(),
            identity_b=_identity(),
            task_a=_task(),
            task_b=_task(),
            ordinary_context=_ordinary(),
            semantic_context=_pack(),
            permit_a=_permit(),
            permit_b=_permit(provenance="simulated"),
            provider_a=provider_a,
            provider_b=provider_b,
            verifier_a=RecordingVerifier(),
            verifier_b=RecordingVerifier(),
        )
    with pytest.raises(ConfigurationABError, match="paired permits"):
        run_paired_ab(
            identity_a=_identity(),
            identity_b=_identity(),
            task_a=_task(),
            task_b=_task(),
            ordinary_context=_ordinary(),
            semantic_context=_pack(),
            permit_a=_permit(),
            permit_b=_permit(available=False, reason="arm-specific outage"),
            provider_a=provider_a,
            provider_b=provider_b,
            verifier_a=RecordingVerifier(),
            verifier_b=RecordingVerifier(),
        )


def _paired_run():
    return run_paired_ab(
        identity_a=_identity(),
        identity_b=_identity(),
        task_a=_task(),
        task_b=_task(),
        ordinary_context=_ordinary(),
        semantic_context=_pack(),
        permit_a=_permit(),
        permit_b=_permit(),
        provider_a=RecordingProvider(),
        provider_b=RecordingProvider(),
        verifier_a=RecordingVerifier(),
        verifier_b=RecordingVerifier(),
    )


def test_paired_run_records_reduction_and_held_constants() -> None:
    pair = _paired_run()
    expected = (_ordinary().token_count - _pack().token_count) * 10000 // _ordinary().token_count
    assert pair.arm_b.raw_result["metrics"]["context_reduction_bp"] == expected
    assert pair.arm_a.raw_result["configuration_id"] == "A"
    assert pair.arm_b.raw_result["configuration_id"] == "B"
    assert pair.pairing_record["verification_policy_both_arms"] == ("full-runtime-verification@1")
    assert pair.pairing_record["full_verification_executed_both_arms"] is True
    for field in (
        "corpus_manifest_cid",
        "task_record_cid",
        "visible_projection_cid",
        "repository_state_cid",
        "environment_cid",
        "provider_id",
        "model_id",
        "model_revision",
        "seed",
        "attempt",
    ):
        assert pair.arm_a.raw_result[field] == pair.arm_b.raw_result[field]


def test_unavailable_pair_does_not_claim_verification_execution() -> None:
    unavailable = _permit(available=False, reason="frontier provider unavailable")
    provider_a = RecordingProvider()
    provider_b = RecordingProvider()
    pair = run_paired_ab(
        identity_a=_identity(),
        identity_b=_identity(),
        task_a=_task(),
        task_b=_task(),
        ordinary_context=_ordinary(),
        semantic_context=_pack(),
        permit_a=unavailable,
        permit_b=unavailable,
        provider_a=provider_a,
        provider_b=provider_b,
        verifier_a=RecordingVerifier(),
        verifier_b=RecordingVerifier(),
    )
    assert pair.arm_a.raw_result["terminal_status"] == "unavailable"
    assert pair.arm_b.raw_result["terminal_status"] == "unavailable"
    assert pair.pairing_record["full_verification_executed_both_arms"] is False
    assert not provider_a.invocations
    assert not provider_b.invocations


def test_replay_is_deterministic_and_content_addressed() -> None:
    first = _paired_run()
    second = _paired_run()
    assert first.arm_a.raw_result == second.arm_a.raw_result
    assert first.arm_b.raw_result == second.arm_b.raw_result
    assert first.arm_a.result_cid == second.arm_a.result_cid
    assert first.arm_b.result_cid == second.arm_b.result_cid
    assert first.pairing_cid == second.pairing_cid


def test_context_kind_cannot_be_swapped_between_arms() -> None:
    with pytest.raises(ConfigurationABError, match="configuration A"):
        run_arm(
            configuration_id="A",
            identity=_identity(),
            task=_task(),
            context=_pack(),
            permit=_permit(),
            provider=RecordingProvider(),
            verifier=RecordingVerifier(),
        )
    with pytest.raises(ConfigurationABError, match="configuration B"):
        run_arm(
            configuration_id="B",
            identity=_identity(),
            task=_task(),
            context=_ordinary(),
            permit=_permit(),
            provider=RecordingProvider(),
            verifier=RecordingVerifier(),
        )
