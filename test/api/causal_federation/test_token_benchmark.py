"""Adversarial unavailable and dormant evidence-boundary tests for CASF-041."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/run_token.py"
MANIFEST_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/token_manifest.json"
TEST_RELATIVE_PATH = "test/api/causal_federation/test_token_benchmark.py"
MATRIX_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
SCHEDULER_RELATIVE_PATH = "config/agent_supervisor_causal_event_federation_scheduler.json"
RUNNER_PATH = ROOT / RUNNER_RELATIVE_PATH
MANIFEST_PATH = ROOT / MANIFEST_RELATIVE_PATH
MATRIX_PATH = ROOT / MATRIX_RELATIVE_PATH
SCHEDULER_PATH = ROOT / SCHEDULER_RELATIVE_PATH
SPEC = importlib.util.spec_from_file_location("casf_token_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
token = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(token)

OWNED_PATHS = (MANIFEST_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH)
PreparedResult = tuple[dict[str, Any], Any, Path, dict[str, str]]


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        list(args), cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _prepared_clone(path: Path) -> tuple[Path, Any]:
    repository = path / "repository"
    _run("git", "clone", "-q", "--shared", str(ROOT), str(repository), cwd=path)
    for relative_path in OWNED_PATHS:
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, destination)
    _run("git", "add", "-f", *OWNED_PATHS, cwd=repository)
    _run(
        "git",
        "-c",
        "user.name=CASF Test",
        "-c",
        "user.email=casf-test@example.invalid",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "CASF-041 fixture",
        cwd=repository,
    )
    spec = importlib.util.spec_from_file_location(
        f"casf_token_{path.name}", repository / RUNNER_RELATIVE_PATH
    )
    assert spec is not None and spec.loader is not None
    cloned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloned)
    return repository, cloned


@pytest.fixture(scope="module")
def prepared_result(tmp_path_factory: pytest.TempPathFactory) -> PreparedResult:
    repository, cloned = _prepared_clone(tmp_path_factory.mktemp("token-unavailable"))
    result = cloned.run_benchmark(repository=repository, identities=None)
    return result, cloned, repository, cloned.repository_identity(repository)


def _ref(number: int) -> str:
    return f"sha256:{number:064x}"


def _valid_identity(module: Any) -> Any:
    return module.TokenIdentity(
        repository_commit="a" * 40,
        repository_tree="b" * 40,
        control_plane_generation=7,
        schema_fingerprint=_ref(1),
        policy_ref=_ref(2),
        policy_revision=3,
        capability_ref=_ref(4),
        federation_id="federation-1",
        task_id="CASF-041",
        attempt_id="attempt-7",
        worktree_id="worktree-7",
        assignment_revision=5,
        fencing_epoch=11,
    )


def _valid_admission(module: Any) -> Any:
    return module.TokenAdmission(
        schema=module.LIVE_ATTESTATION_SCHEMA,
        identity=_valid_identity(module),
        comparison_plan_ref=_ref(10),
        quack_admission_receipt_ref=_ref(11),
        exclusive_state_owner_birth_ref=_ref(12),
        provider_usage_attestation_ref=_ref(13),
        baseline_supervisor_process_birth_refs=(_ref(20),),
        candidate_supervisor_process_birth_refs=tuple(_ref(index) for index in range(30, 42)),
        expected_observation_sequence=9,
        replay_guard_ref=_ref(50),
        provider_ref="provider:grok",
        model_ref="model:grok-build",
        tokenizer_ref="tokenizer:grok-build-v1",
        host_ref=_ref(61),
        workload_ref=_ref(51),
        task_population_ref=_ref(52),
        criteria_ref=_ref(53),
        budget_ref=_ref(54),
        tests_ref=_ref(62),
        proofs_ref=_ref(63),
        assurance_ref=_ref(64),
        retry_policy_ref=_ref(55),
        replay_policy_ref=_ref(56),
        fallback_policy_ref=_ref(57),
        cancellation_policy_ref=_ref(58),
        capsule_policy_ref=_ref(59),
        board_scan_policy_ref=_ref(60),
        arms_execute_sequentially=True,
        cross_arm_concurrency_permitted=False,
        state_authority="authenticated_typed_quack",
        direct_database_access_permitted=False,
        ducklake_scheduling_authority_permitted=False,
        ducklake_projection_authoritative=False,
    )


def _arm(module: Any, arm_id: str, births: tuple[str, ...], offset: int) -> dict[str, Any]:
    baseline = arm_id == "baseline"
    populations = {
        "task_count": 10,
        "criterion_opportunities": 100,
        "accepted_criteria": 80,
        "repeated_context_input_tokens": 1000 if baseline else 500,
        "model_input_tokens": 10000 if baseline else 6000,
        "primary_model_calls": 60 if baseline else 50,
        "duplicate_model_calls": 20 if baseline else 8,
        "retry_model_calls": 10 if baseline else 5,
        "replay_model_calls": 5 if baseline else 2,
        "fallback_model_calls": 3 if baseline else 2,
        "cancelled_model_calls": 2 if baseline else 1,
        "total_model_calls": 100 if baseline else 68,
        "eligible_semantic_capsules": 100,
        "reused_semantic_capsules": 0 if baseline else 70,
        "recomputed_semantic_capsules": 100 if baseline else 30,
        "rejected_stale_semantic_capsules": 5 if baseline else 7,
        "stale_semantic_capsules_reused": 0,
        "board_scan_opportunities": 100,
        "complete_board_scans": 100 if baseline else 20,
        "incremental_board_scans": 0 if baseline else 80,
    }
    return {
        "schema": module.LIVE_ARM_SCHEMA,
        "arm_id": arm_id,
        "arm_execution_ref": _ref(offset),
        "supervisor_process_birth_refs": list(births),
        "population_receipts": {
            key: _ref(offset + index + 1) for index, key in enumerate(module._ARM_RECEIPT_KEYS)
        },
        "populations": populations,
    }


def _valid_observation(module: Any, admission: Any | None = None) -> dict[str, Any]:
    admitted = _valid_admission(module) if admission is None else admission
    return {
        "schema": module.LIVE_OBSERVATION_SCHEMA,
        "identity_ref": admitted.identity.content_ref(),
        "observation_sequence": admitted.expected_observation_sequence,
        "comparison_plan_ref": admitted.comparison_plan_ref,
        "quack_admission_receipt_ref": admitted.quack_admission_receipt_ref,
        "exclusive_state_owner_birth_ref": admitted.exclusive_state_owner_birth_ref,
        "provider_usage_attestation_ref": admitted.provider_usage_attestation_ref,
        "replay_guard_ref": admitted.replay_guard_ref,
        "host_ref": admitted.host_ref,
        "workload_ref": admitted.workload_ref,
        "task_population_ref": admitted.task_population_ref,
        "criteria_ref": admitted.criteria_ref,
        "provider_ref": admitted.provider_ref,
        "model_ref": admitted.model_ref,
        "tokenizer_ref": admitted.tokenizer_ref,
        "budget_ref": admitted.budget_ref,
        "tests_ref": admitted.tests_ref,
        "proofs_ref": admitted.proofs_ref,
        "assurance_ref": admitted.assurance_ref,
        "retry_policy_ref": admitted.retry_policy_ref,
        "replay_policy_ref": admitted.replay_policy_ref,
        "fallback_policy_ref": admitted.fallback_policy_ref,
        "cancellation_policy_ref": admitted.cancellation_policy_ref,
        "capsule_policy_ref": admitted.capsule_policy_ref,
        "board_scan_policy_ref": admitted.board_scan_policy_ref,
        "state_transport": "authenticated_typed_quack",
        "arms_executed_sequentially": True,
        "cross_arm_concurrency_observed": False,
        "direct_database_access_used": False,
        "ducklake_scheduling_authority_used": False,
        "ducklake_projection_authoritative": False,
        "attempt_population_complete": True,
        "retries_included": True,
        "replays_included": True,
        "fallbacks_included": True,
        "cancellations_included": True,
        "evidence_coverage_preserved": True,
        "evidence_coverage_receipt_ref": _ref(74),
        "zero_tolerance_gate_failures": {gate: 0 for gate in module.ZERO_TOLERANCE_GATES},
        "zero_tolerance_receipt_ref": _ref(75),
        "baseline_arm": _arm(
            module,
            "baseline",
            admitted.baseline_supervisor_process_birth_refs,
            100,
        ),
        "candidate_arm": _arm(
            module,
            "candidate",
            admitted.candidate_supervisor_process_birth_refs,
            200,
        ),
    }


def _rehash(module: Any, result: dict[str, Any]) -> dict[str, Any]:
    result["content_sha256"] = module.result_content_sha256(result)
    return result


def test_manifest_freezes_scheduler_capacity_comparison_and_unavailability() -> None:
    manifest = token.load_manifest(MANIFEST_PATH)
    assert token.validate_matrix_binding(manifest, MATRIX_PATH) == manifest["matrix_binding"]
    token._validate_scheduler(SCHEDULER_PATH.read_bytes())
    assert manifest["program_id"] == token.PROGRAM_ID
    assert manifest["measurement_scope"] == token.MEASUREMENT_SCOPE
    assert manifest["capacity_preflight"] == token.capacity_preflight()
    assert manifest["comparison"] == token._COMPARISON
    assert manifest["scheduler_binding"] == token._SCHEDULER_BINDING
    assert manifest["authoritative"] is False
    assert manifest["promotion_eligible"] is False
    assert tuple(manifest["future_identity_requirements"]) == token.REQUIRED_IDENTITIES


def test_unavailable_result_is_truthful_tree_scheduler_and_source_bound(
    prepared_result: PreparedResult,
) -> None:
    result, cloned, repository, identity = prepared_result
    assert (
        cloned.validate_result(result, repository=repository, current_identity=identity) == result
    )
    assert result["program_id"] == cloned.PROGRAM_ID
    assert result["measurement_scope"] == cloned.MEASUREMENT_SCOPE
    assert result["availability"] == "unavailable"
    assert result["execution_status"] == "not_run"
    for key in ("ran", "qualified", "authoritative", "promotion_eligible"):
        assert result[key] is False
    assert result["metrics_omitted"] is True
    assert result["repository_binding"] == {
        **identity,
        "git_status_porcelain_empty": True,
        "measured_paths_match_tracked_head_blobs": True,
        "observed_before_and_after": True,
    }
    expected_hashes = {
        path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
        for path in cloned.MEASURED_RELATIVE_PATHS
    }
    assert result["source_binding"]["source_sha256"] == expected_hashes
    expected_oids = {
        path: _run("git", "rev-parse", f"HEAD:{path}", cwd=repository)
        for path in cloned.MEASURED_RELATIVE_PATHS
    }
    assert result["source_binding"]["tracked_head_blob_oid"] == expected_oids
    assert result["source_binding"]["exact_tracked_head_bytes"] is True
    assert result["scheduler_binding"]["raw_sha256"] == expected_hashes[SCHEDULER_RELATIVE_PATH]
    assert result["content_sha256"] == cloned.result_content_sha256(result)
    assert not {"metric", "metrics", "observations", "values"} & set(result)


def test_preflight_occurs_before_paths_git_files_or_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stop(RuntimeError):
        pass

    def stop() -> dict[str, Any]:
        raise Stop

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("operation occurred before capacity preflight")

    monkeypatch.setattr(token, "capacity_preflight", stop)
    monkeypatch.setattr(token, "_bound_repository_and_recipe", forbidden)
    monkeypatch.setattr(token.subprocess, "run", forbidden)
    monkeypatch.setattr(token.os, "open", forbidden)
    with pytest.raises(Stop):
        token.run_benchmark(repository=object(), identities=object())


def test_unavailable_artifact_never_reads_caller_identities(
    prepared_result: PreparedResult,
) -> None:
    expected, cloned, repository, _identity = prepared_result

    class Exploding:
        def __iter__(self) -> Any:
            raise AssertionError("identities read")

        def __str__(self) -> str:
            raise AssertionError("identities rendered")

        def __fspath__(self) -> str:
            raise AssertionError("identities used as a path")

    assert cloned.run_benchmark(repository=repository, identities=Exploding()) == expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"metrics": {}}),
        lambda value: value.update({"ran": True}),
        lambda value: value.update({"program_id": "other-program"}),
        lambda value: value.update({"measurement_scope": "other-scope"}),
        lambda value: value["capacity_preflight"].update({"current_provider_concurrency": 2}),
        lambda value: value["scheduler_binding"].update({"raw_sha256": "f" * 64}),
        lambda value: value["source_binding"]["source_sha256"].update(
            {RUNNER_RELATIVE_PATH: "f" * 64}
        ),
        lambda value: value["source_binding"]["tracked_head_blob_oid"].update(
            {RUNNER_RELATIVE_PATH: "f" * 40}
        ),
        lambda value: value["future_required_comparison"].update(
            {"arms_execute_sequentially": False}
        ),
    ],
)
def test_fabricated_measurements_and_stale_bindings_fail_closed(
    prepared_result: PreparedResult, mutation: Any
) -> None:
    result, cloned, repository, _identity = prepared_result
    candidate = deepcopy(result)
    mutation(candidate)
    _rehash(cloned, candidate)
    with pytest.raises(cloned.TokenBenchmarkError):
        cloned.validate_result(candidate, repository=repository)


def test_unavailable_live_boundaries_touch_no_caller_object_or_property() -> None:
    class Untouchable:
        def __getattribute__(self, _name: str) -> Any:
            raise AssertionError("caller property was touched")

        def __iter__(self) -> Any:
            raise AssertionError("caller was iterated")

        def __str__(self) -> str:
            raise AssertionError("caller was rendered")

    caller = Untouchable()
    calls = (
        lambda: token.validate_admitted_token_observation(caller, caller),
        lambda: token.execute_admitted_token(caller, caller),
        lambda: token.require_live_capability(caller),
    )
    for call in calls:
        with pytest.raises(
            token.TokenCapabilityUnavailable,
            match="registered state-owner receipt verifier",
        ):
            call()


def test_replay_owner_substitution_and_cross_domain_aliases_cannot_validate() -> None:
    hostile_cases = (
        ({"observation_sequence": 7}, {"observation_sequence": 7}),
        (
            {"exclusive_state_owner_birth_ref": _ref(900)},
            {"exclusive_state_owner_birth_ref": _ref(901)},
        ),
        (
            {"quack_admission_receipt_ref": _ref(902)},
            {
                "provider_usage_population_ref": _ref(902),
                "capsule_population_ref": _ref(902),
            },
        ),
    )
    for admission, observation in hostile_cases:
        with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
            token.validate_admitted_token_observation(admission, observation)


def test_raw_population_boundary_is_not_validated_without_registered_verifier() -> None:
    admission = _valid_admission(token)
    with pytest.raises(
        token.TokenCapabilityUnavailable,
        match="registered state-owner receipt verifier",
    ):
        token.validate_admitted_token_observation(admission, _valid_observation(token, admission))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeated_context_input_tokens", 501),
        ("model_input_tokens", 6001),
        ("duplicate_model_calls", 9),
        ("reused_semantic_capsules", 69),
        ("complete_board_scans", 21),
    ],
)
def test_exact_integer_cross_multiplication_rejects_one_past_each_boundary(
    field: str, value: int
) -> None:
    admission = _valid_admission(token)
    observation = _valid_observation(token, admission)
    population = observation["candidate_arm"]["populations"]
    population[field] = value
    if field == "duplicate_model_calls":
        population["total_model_calls"] += 1
    elif field == "reused_semantic_capsules":
        population["recomputed_semantic_capsules"] += 1
    elif field == "complete_board_scans":
        population["incremental_board_scans"] -= 1
    with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
        token.validate_admitted_token_observation(admission, observation)


def test_caller_authored_percentage_claims_are_not_in_the_schema() -> None:
    admission = _valid_admission(token)
    observation = _valid_observation(token, admission)
    observation["eligible_semantic_capsule_reuse_percent"] = 101
    with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
        token.validate_admitted_token_observation(admission, observation)


@pytest.mark.parametrize("bad_value", [True, 1.0, "1", -1, 2**63])
def test_raw_populations_reject_nonexact_or_out_of_range_integers(bad_value: Any) -> None:
    admission = _valid_admission(token)
    observation = _valid_observation(token, admission)
    observation["candidate_arm"]["populations"]["task_count"] = bad_value
    with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
        token.validate_admitted_token_observation(admission, observation)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda arm: arm["populations"].update({"accepted_criteria": 101}),
        lambda arm: arm["populations"].update({"repeated_context_input_tokens": 6001}),
        lambda arm: arm["populations"].update({"total_model_calls": 69}),
        lambda arm: arm["populations"].update(
            {
                "model_input_tokens": 67,
                "repeated_context_input_tokens": 50,
            }
        ),
        lambda arm: arm["populations"].update({"recomputed_semantic_capsules": 31}),
        lambda arm: arm["populations"].update({"stale_semantic_capsules_reused": 1}),
        lambda arm: arm["populations"].update({"incremental_board_scans": 79}),
    ],
)
def test_impossible_raw_populations_fail_closed(mutation: Any) -> None:
    admission = _valid_admission(token)
    observation = _valid_observation(token, admission)
    mutation(observation["candidate_arm"])
    with pytest.raises(token.TokenBenchmarkError):
        token.validate_admitted_token_observation(admission, observation)


@pytest.mark.parametrize(
    "field",
    [
        "task_count",
        "criterion_opportunities",
        "accepted_criteria",
        "eligible_semantic_capsules",
        "board_scan_opportunities",
    ],
)
def test_arms_must_use_the_same_raw_population(field: str) -> None:
    admission = _valid_admission(token)
    observation = _valid_observation(token, admission)
    observation["candidate_arm"]["populations"][field] += 1
    if field == "criterion_opportunities":
        pass
    elif field == "eligible_semantic_capsules":
        observation["candidate_arm"]["populations"]["recomputed_semantic_capsules"] += 1
    elif field == "board_scan_opportunities":
        observation["candidate_arm"]["populations"]["incremental_board_scans"] += 1
    with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
        token.validate_admitted_token_observation(admission, observation)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("provider_ref", "provider:other"),
        ("model_ref", "model:other"),
        ("tokenizer_ref", "tokenizer:other"),
        ("host_ref", _ref(799)),
        ("workload_ref", _ref(800)),
        ("task_population_ref", _ref(801)),
        ("criteria_ref", _ref(802)),
        ("budget_ref", _ref(803)),
        ("tests_ref", _ref(810)),
        ("proofs_ref", _ref(811)),
        ("assurance_ref", _ref(812)),
        ("retry_policy_ref", _ref(804)),
        ("replay_policy_ref", _ref(805)),
        ("fallback_policy_ref", _ref(806)),
        ("cancellation_policy_ref", _ref(807)),
        ("capsule_policy_ref", _ref(808)),
        ("board_scan_policy_ref", _ref(809)),
    ],
)
def test_observation_revalidates_every_admitted_route_and_policy(field: str, bad: str) -> None:
    admission = _valid_admission(token)
    observation = _valid_observation(token, admission)
    observation[field] = bad
    with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
        token.validate_admitted_token_observation(admission, observation)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("arms_executed_sequentially", False),
        ("cross_arm_concurrency_observed", True),
        ("direct_database_access_used", True),
        ("ducklake_scheduling_authority_used", True),
        ("ducklake_projection_authoritative", True),
        ("attempt_population_complete", False),
        ("retries_included", False),
        ("replays_included", False),
        ("fallbacks_included", False),
        ("cancellations_included", False),
        ("evidence_coverage_preserved", False),
    ],
)
def test_observation_rejects_concurrency_authority_or_coverage_weakening(
    field: str, bad: bool
) -> None:
    admission = _valid_admission(token)
    observation = _valid_observation(token, admission)
    observation[field] = bad
    with pytest.raises(token.TokenBenchmarkError):
        token.validate_admitted_token_observation(admission, observation)


def test_replay_sequence_process_birth_and_receipt_reuse_fail_closed() -> None:
    admission = _valid_admission(token)
    stale = _valid_observation(token, admission)
    stale["observation_sequence"] -= 1
    with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
        token.validate_admitted_token_observation(admission, stale)

    wrong_birth = _valid_observation(token, admission)
    wrong_birth["candidate_arm"]["supervisor_process_birth_refs"][0] = _ref(999)
    wrong_birth["candidate_arm"]["supervisor_process_birth_refs"].sort()
    with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
        token.validate_admitted_token_observation(admission, wrong_birth)

    reused_receipt = _valid_observation(token, admission)
    key = token._ARM_RECEIPT_KEYS[0]
    reused_receipt["candidate_arm"]["population_receipts"][key] = reused_receipt["baseline_arm"][
        "population_receipts"
    ][key]
    with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
        token.validate_admitted_token_observation(admission, reused_receipt)


@pytest.mark.parametrize(
    "identity",
    [
        lambda value: value._replace(control_plane_generation=True),
        lambda value: value._replace(policy_revision=0),
        lambda value: value._replace(task_id="CASF-042"),
        lambda value: value._replace(schema_fingerprint="token=secret"),
        lambda value: value._replace(repository_commit="A" * 40),
    ],
)
def test_typed_identity_rejects_type_fence_task_secret_and_hash_forgery(
    identity: Any,
) -> None:
    admission = _valid_admission(token)
    forged = admission._replace(identity=identity(admission.identity))
    with pytest.raises(token.TokenBenchmarkError):
        token.validate_admitted_token_observation(forged, _valid_observation(token, forged))


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "casf/token-live-capacity-attestation@1"},
        {"arms_execute_sequentially": False},
        {"cross_arm_concurrency_permitted": True},
        {"state_authority": "ducklake"},
        {"quack_admission_receipt_ref": "token=secret"},
        {"direct_database_access_permitted": True},
        {"ducklake_scheduling_authority_permitted": True},
        {"ducklake_projection_authoritative": True},
        {"expected_observation_sequence": 1.0},
    ],
)
def test_typed_admission_rejects_forged_or_weakened_fields(changes: dict[str, Any]) -> None:
    admission = _valid_admission(token)._replace(**changes)
    with pytest.raises(token.TokenBenchmarkError):
        token.validate_admitted_token_observation(admission, _valid_observation(token, admission))


def test_zero_tolerance_failures_and_wrong_types_fail_closed() -> None:
    admission = _valid_admission(token)
    for bad in (1, True, 1.0):
        observation = _valid_observation(token, admission)
        observation["zero_tolerance_gate_failures"]["tenant_leakage"] = bad
        with pytest.raises(token.TokenBenchmarkError):
            token.validate_admitted_token_observation(admission, observation)


def test_dormant_execution_boundary_cannot_call_executor() -> None:
    class ForbiddenExecutor:
        interface = token.ADMITTED_EXECUTION_INTERFACE

        def execute_token_comparison(self, **_kwargs: Any) -> Any:
            raise AssertionError("executor must be unreachable")

    with pytest.raises(token.TokenCapabilityUnavailable, match="receipt verifier"):
        token._admitted_execution_boundary(ForbiddenExecutor(), admission=_valid_admission(token))


def test_git_identity_ignores_all_ambient_redirection_config_and_alternates(
    prepared_result: PreparedResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _result, cloned, repository, expected = prepared_result
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    _run("git", "init", "-q", cwd=unrelated)
    (unrelated / "file").write_text("unrelated", encoding="utf-8")
    _run("git", "add", "file", cwd=unrelated)
    _run(
        "git",
        "-c",
        "user.name=Other",
        "-c",
        "user.email=other@example.invalid",
        "commit",
        "-q",
        "-m",
        "other",
        cwd=unrelated,
    )
    malicious = {
        "GIT_DIR": str(unrelated / ".git"),
        "GIT_WORK_TREE": str(unrelated),
        "GIT_INDEX_FILE": str(unrelated / ".git" / "index"),
        "GIT_OBJECT_DIRECTORY": str(unrelated / ".git" / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(unrelated / ".git" / "objects"),
        "GIT_REPLACE_REF_BASE": "refs/evil/",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.bare",
        "GIT_CONFIG_VALUE_0": "true",
        "GIT_CONFIG_GLOBAL": str(unrelated / "malicious-config"),
    }
    for key, value in malicious.items():
        monkeypatch.setenv(key, value)
    assert cloned.repository_identity(repository) == expected
    assert cloned.run_benchmark(repository) == _result
    sanitized = cloned._git_environment()
    assert not (set(malicious) & set(sanitized)) - {
        "GIT_CONFIG_GLOBAL",
    }
    assert sanitized["GIT_CONFIG_GLOBAL"] == cloned.os.devnull


def test_git_uses_trusted_absolute_binary_and_drops_path_loader_and_shell_injection(
    prepared_result: PreparedResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_result, cloned, repository, _identity = prepared_result
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "fake-git-was-executed"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/python3\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('touched', encoding='utf-8')\n"
        "raise SystemExit(97)\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    fake_library = tmp_path / "fake-loader.so"
    fake_library.write_bytes(b"must never reach the dynamic loader")
    poisoned = {
        "PATH": str(fake_bin),
        "GIT_EXEC_PATH": str(fake_bin),
        "LD_PRELOAD": str(fake_library),
        "LD_LIBRARY_PATH": str(fake_bin),
        "LD_AUDIT": str(fake_library),
        "LD_DEBUG": "all",
        "DYLD_INSERT_LIBRARIES": str(fake_library),
        "DYLD_LIBRARY_PATH": str(fake_bin),
        "LIBPATH": str(fake_bin),
        "SHLIB_PATH": str(fake_bin),
        "BASH_ENV": str(fake_git),
        "ENV": str(fake_git),
        "CDPATH": str(fake_bin),
    }
    for key, value in poisoned.items():
        monkeypatch.setenv(key, value)
    real_run = cloned.subprocess.run
    observed_environments: list[dict[str, str]] = []

    def inspected_run(command: list[str], **kwargs: Any) -> Any:
        environment = kwargs["env"]
        observed_environments.append(environment)
        assert command[0] == str(cloned._TRUSTED_GIT_EXECUTABLE)
        assert environment["PATH"] == cloned._TRUSTED_PROCESS_PATH
        assert environment["GIT_EXEC_PATH"] == str(cloned._TRUSTED_GIT_EXEC_PATH)
        assert not (set(poisoned) - {"PATH", "GIT_EXEC_PATH"}) & set(environment)
        return real_run(command, **kwargs)

    monkeypatch.setattr(cloned.subprocess, "run", inspected_run)
    assert cloned.run_benchmark(repository) == expected_result
    assert observed_environments
    assert not marker.exists()


@pytest.mark.parametrize("candidate_kind", ["missing", "directory", "non_executable", "symlink"])
def test_git_fails_closed_when_trusted_executable_is_not_protected_regular_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_kind: str,
) -> None:
    candidate = tmp_path / candidate_kind
    if candidate_kind == "directory":
        candidate.mkdir()
    elif candidate_kind == "non_executable":
        candidate.write_bytes(b"not executable")
        candidate.chmod(0o644)
    elif candidate_kind == "symlink":
        candidate.symlink_to("/usr/bin/git")
    monkeypatch.setattr(token, "_TRUSTED_GIT_EXECUTABLE", candidate)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an untrusted executable was invoked")

    monkeypatch.setattr(token.subprocess, "run", forbidden)
    with pytest.raises(token.TokenBenchmarkError) as raised:
        token._git_bytes(ROOT, "--version")
    assert raised.value.reason_code == "trusted_git_unavailable"


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_exact_head_blob_check_defeats_hidden_index_flags(tmp_path: Path, index_flag: str) -> None:
    repository, cloned = _prepared_clone(tmp_path)
    manifest = repository / MANIFEST_RELATIVE_PATH
    _run("git", "update-index", index_flag, MANIFEST_RELATIVE_PATH, cwd=repository)
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    assert _run("git", "status", "--porcelain=v1", cwd=repository) == ""
    with pytest.raises(cloned.TokenBenchmarkError, match="tracked HEAD blobs"):
        cloned.run_benchmark(repository)


def test_ignored_untracked_recipe_cannot_substitute_for_missing_head_blob(
    tmp_path: Path,
) -> None:
    repository, cloned = _prepared_clone(tmp_path)
    _run("git", "rm", "-q", "--cached", MANIFEST_RELATIVE_PATH, cwd=repository)
    exclude = repository / ".git" / "info" / "exclude"
    exclude.write_text(
        exclude.read_text(encoding="utf-8") + f"\n/{MANIFEST_RELATIVE_PATH}\n",
        encoding="utf-8",
    )
    _run(
        "git",
        "-c",
        "user.name=CASF Test",
        "-c",
        "user.email=casf-test@example.invalid",
        "commit",
        "-q",
        "-m",
        "remove tracked manifest",
        cwd=repository,
    )
    assert (repository / MANIFEST_RELATIVE_PATH).is_file()
    assert _run("git", "status", "--porcelain=v1", "--untracked-files=all", cwd=repository) == ""
    with pytest.raises(cloned.TokenBenchmarkError, match="not exactly tracked"):
        cloned.run_benchmark(repository)


def test_internal_replace_ref_cannot_redirect_head_or_measured_blobs(
    tmp_path: Path,
) -> None:
    repository, cloned = _prepared_clone(tmp_path)
    expected_identity = cloned.repository_identity(repository)
    expected_result = cloned.run_benchmark(repository)
    unrelated = tmp_path / "replacement"
    unrelated.mkdir()
    _run("git", "init", "-q", cwd=unrelated)
    (unrelated / "different").write_text("replacement tree", encoding="utf-8")
    _run("git", "add", "different", cwd=unrelated)
    _run(
        "git",
        "-c",
        "user.name=Other",
        "-c",
        "user.email=other@example.invalid",
        "commit",
        "-q",
        "-m",
        "replacement",
        cwd=unrelated,
    )
    _run("git", "fetch", "-q", str(unrelated), "HEAD", cwd=repository)
    replacement = _run("git", "rev-parse", "FETCH_HEAD", cwd=repository)
    _run(
        "git",
        "replace",
        expected_identity["repository_commit"],
        replacement,
        cwd=repository,
    )
    assert cloned.repository_identity(repository) == expected_identity
    assert cloned.run_benchmark(repository) == expected_result


def test_scheduler_source_hash_and_capacity_drift_fail_closed(
    tmp_path: Path,
) -> None:
    repository, cloned = _prepared_clone(tmp_path)
    scheduler = repository / SCHEDULER_RELATIVE_PATH
    payload = json.loads(scheduler.read_text(encoding="utf-8"))
    payload["bootstrap_capacity"]["provider_concurrency"] = 12
    scheduler.write_text(json.dumps(payload), encoding="utf-8")
    _run("git", "add", SCHEDULER_RELATIVE_PATH, cwd=repository)
    _run(
        "git",
        "-c",
        "user.name=CASF Test",
        "-c",
        "user.email=casf-test@example.invalid",
        "commit",
        "-q",
        "-m",
        "scheduler drift",
        cwd=repository,
    )
    with pytest.raises(cloned.TokenBenchmarkError, match="scheduler source"):
        cloned.run_benchmark(repository)


def test_snapshot_detects_source_toctou(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository, cloned = _prepared_clone(tmp_path)
    original = cloned._read_bounded_regular_bytes
    changed = False

    def racing_read(path: Any, *, name: str, maximum_bytes: int) -> bytes:
        nonlocal changed
        payload = original(path, name=name, maximum_bytes=maximum_bytes)
        if not changed and Path(path) == repository / SCHEDULER_RELATIVE_PATH:
            changed = True
            (repository / SCHEDULER_RELATIVE_PATH).write_bytes(payload + b"\n")
        return payload

    monkeypatch.setattr(cloned, "_read_bounded_regular_bytes", racing_read)
    with pytest.raises(cloned.TokenBenchmarkError):
        cloned.run_benchmark(repository)


def test_bounded_no_follow_and_closed_json_inputs(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(token.TokenBenchmarkError, match="duplicate JSON key"):
        token._read_object(duplicate)
    linked = tmp_path / "linked.json"
    linked.symlink_to(duplicate)
    with pytest.raises(token.TokenBenchmarkError, match="non-symlink"):
        token._read_object(linked)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (token._MAX_MANIFEST_BYTES + 1))
    with pytest.raises(token.TokenBenchmarkError, match="byte limit"):
        token._read_object(oversized)


def test_cli_never_reads_identity_file_and_sanitizes_invalid_path(
    prepared_result: PreparedResult,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _result, cloned, repository, _identity = prepared_result
    identity_path = tmp_path / "sk-secret-identities.json"
    identity_path.write_text('{"token":"must-not-be-read"}', encoding="utf-8")
    assert cloned.main(["--repository", str(repository), "--identities", str(identity_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["execution_status"] == "not_run"
    assert output["metrics_omitted"] is True
    assert cloned.main(["--repository", "/tmp/sk-do-not-echo/repository"]) == 2
    assert "sk-do-not-echo" not in capsys.readouterr().out
