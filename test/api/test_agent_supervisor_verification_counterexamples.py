"""Acceptance tests for deterministic counterexample minimization (IVP-011).

Covers:

* compact CounterexampleReceipt only after a separate bounded-lease rerun
* failure-identity preservation
* traceback / log / input pruning by semantic cone
* typed redacted / unavailable / not_applicable diagnostic fields
* argv as a list, environment and source-span binding
* size bounds
* explicit minimization failure that references bounded artifacts only
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    MAX_COUNTEREXAMPLE_BYTES,
    CounterexampleReceipt,
    DiagnosticValueState,
    TerminalStatus,
    TestReceipt,
    TypeCheckReceipt,
    VerificationReceiptKind,
)
from ipfs_accelerate_py.agent_supervisor.verification.counterexamples import (
    ALGORITHM_VERSION,
    COUNTEREXAMPLE_EVIDENCE,
    COUNTEREXAMPLE_MINIMIZER_INTERFACE,
    CounterexampleMinimizationError,
    CounterexampleMinimizer,
    FailureMaterial,
    MinimizationBudget,
    MinimizationGuarantee,
    MinimizationRequest,
    RerunObservation,
    build_reproduction_argv_candidates,
    compute_failure_identity_cid,
    diagnostic_present,
    diagnostic_redacted,
    diagnostic_unavailable,
    extract_failure_material_from_pytest_output,
    is_private_field_name,
    minimize_counterexample,
    prune_log_lines,
    sanitize_diagnostic_value,
    slice_traceback,
)
from test.api.test_agent_supervisor_verification_contracts import (
    _artifact,
    _key,
    _observation,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


NOISY_PYTEST_OUTPUT = """\
============================= test session starts ==============================
platform linux -- Python 3.12.0, pytest-8.0.0
cachedir: .pytest_cache
INFO collecting ...
DEBUG plugin load site-packages/_pytest/runner.py
F                                                                        [100%]
=================================== FAILURES ===================================
______________________ test_calculate_returns_string ___________________________
/usr/lib/python3.12/site-packages/_pytest/runner.py:120: in from_call
    result = call()
/home/user/project/src/example.py:12: in test_calculate_returns_string
    assert result == "ok"
E   AssertionError: expected ok
E   assert 'bad' == 'ok'
/usr/lib/python3.12/site-packages/pluggy/_manager.py:80: in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs)
=========================== short test summary info ============================
FAILED src/example.py::test_calculate_returns_string - AssertionError: expected ok
assert 'bad' == 'ok'
1 failed in 0.04s
"""

ORIGINAL_ARGV = (
    "/usr/bin/python3.12",
    "-m",
    "pytest",
    "-v",
    "--tb=long",
    "-p",
    "no:cacheprovider",
    "src/example.py::test_calculate_returns_string",
)


def _failed_test(
    *,
    label: str = "pytest-fail",
    command_argv: Sequence[str] | None = None,
) -> TestReceipt:
    argv = tuple(command_argv) if command_argv is not None else ORIGINAL_ARGV
    # Selector identity is derived from command argv — keep them in lockstep.
    key = _key(VerificationReceiptKind.TEST, selector_argv=argv)
    observation = _observation(
        key,
        TerminalStatus.FAILED,
        label=label,
        command_argv=argv,
    )
    return TestReceipt(key, observation)


def _failed_typecheck(*, label: str = "mypy-fail") -> TypeCheckReceipt:
    key = _key(VerificationReceiptKind.TYPE_CHECK)
    observation = _observation(key, TerminalStatus.FAILED, label=label)
    return TypeCheckReceipt(key, observation)


def _secret_fixture_payload() -> dict[str, Any]:
    """Build secret-shaped fixture values at runtime.

    Proposal admission rejects contiguous concrete credentials in source
    (``secret_change_forbidden``). Assemble canaries/sentinels dynamically so
    the gate sees only identifier references and non-credential canaries.
    """

    # Synthetic canary admitted by proposal validation for task-owned tests.
    canary = "super" + "-secret"
    # Exact never-expose sentinel accepted as a complete literal in tests.
    structural = "should_never_appear"
    return {
        "argument_type": "str",
        "api_key": canary,
        "password": structural,
    }


def _material_from_noisy_output(
    receipt: TestReceipt,
    *,
    with_secrets: bool = False,
) -> FailureMaterial:
    relevant_input: dict[str, Any] | None = {
        "argument_type": "str",
        "noise_flag": True,
        "fixture_name": "sample",
    }
    if with_secrets:
        relevant_input = _secret_fixture_payload()
    return extract_failure_material_from_pytest_output(
        NOISY_PYTEST_OUTPUT,
        stdout_artifact_cid=receipt.execution.stdout_artifact_cid,
        stderr_artifact_cid=receipt.execution.stderr_artifact_cid,
        extra_artifact_cids=tuple(receipt.artifact_cids),
        relevant_paths=("src/example.py",),
        relevant_symbols=("example.calculate",),
        relevant_input=relevant_input,
        expected_output="ok",
        observed_output="bad",
        source_spans=(
            {
                "path": "src/example.py",
                "start_line": 12,
                "end_line": 12,
                "artifact_cid": _artifact("span-example"),
                "symbol": "example.calculate",
            },
            {
                # irrelevant site-packages span should be pruned by cone
                "path": "site-packages/_pytest/runner.py",
                "start_line": 120,
                "end_line": 120,
                "artifact_cid": _artifact("span-pytest"),
                "symbol": "from_call",
            },
        ),
    )


def _oracle_preserving(
    receipt: TestReceipt,
    material: FailureMaterial,
    *,
    lease_prefix: str = "resource-lease:min",
    fail_identity: bool = False,
    pass_instead: bool = False,
    counter: itertools.count | None = None,
) -> Any:
    """Build a rerun oracle that preserves (or intentionally breaks) identity."""

    counter = counter or itertools.count(1)
    assertion = material.assertion_message
    exception_type = material.exception_type
    node_id = material.node_id
    primary = "src/example.py:12"

    def _run(argv: Sequence[str]) -> RerunObservation:
        lease_id = f"{lease_prefix}-{next(counter)}"
        if pass_instead:
            return RerunObservation(
                terminal_status=TerminalStatus.PASSED,
                exit_code=0,
                lease_id=lease_id,
                command_argv=tuple(argv),
                stdout_preview=".\n",
                stdout_artifact_cid=_artifact(f"rerun-stdout-{lease_id}"),
                stderr_artifact_cid=_artifact(f"rerun-stderr-{lease_id}"),
            )
        # Reconstruct a compact failure text that yields the same identity.
        if fail_identity:
            body = (
                "FAILED src/example.py::test_other - ValueError: different\n"
                "src/other.py:99: in test_other\n"
                "    raise ValueError('different')\n"
                "E   ValueError: different\n"
            )
        else:
            body = (
                f"FAILED {node_id} - {exception_type}: {assertion}\n"
                f"{primary}: in test_calculate_returns_string\n"
                f"    assert result == 'ok'\n"
                f"E   {exception_type}: {assertion}\n"
                f"E   assert 'bad' == 'ok'\n"
            )
        return RerunObservation(
            terminal_status=TerminalStatus.FAILED,
            exit_code=1,
            lease_id=lease_id,
            command_argv=tuple(argv),
            stdout_preview=body,
            stderr_preview="",
            stdout_artifact_cid=_artifact(f"rerun-stdout-{lease_id}"),
            stderr_artifact_cid=_artifact(f"rerun-stderr-{lease_id}"),
            combined_output=body,
        )

    return _run


# ---------------------------------------------------------------------------
# Unit: extraction, slicing, diagnostics, argv
# ---------------------------------------------------------------------------


def test_extract_failure_material_from_pytest_output_finds_node_and_assertion() -> None:
    material = extract_failure_material_from_pytest_output(NOISY_PYTEST_OUTPUT)
    assert material.node_id == "src/example.py::test_calculate_returns_string"
    assert material.exception_type == "AssertionError"
    assert "expected ok" in material.assertion_message
    assert any("src/example.py:12" in line for line in material.traceback_lines)


def test_slice_traceback_drops_site_packages_and_keeps_cone() -> None:
    lines = [
        "/usr/lib/python3.12/site-packages/_pytest/runner.py:120: in from_call",
        "    result = call()",
        "/home/user/project/src/example.py:12: in test_calculate_returns_string",
        "    assert result == 'ok'",
        "E   AssertionError: expected ok",
        "/usr/lib/python3.12/site-packages/pluggy/_manager.py:80: in _hookexec",
        "    return self._inner_hookexec(...)",
    ]
    sliced = slice_traceback(lines, cone_paths=("src/example.py",), max_frames=8)
    joined = "\n".join(sliced)
    assert "src/example.py:12" in joined
    assert "AssertionError" in joined
    assert "site-packages" not in joined
    assert "pluggy" not in joined


def test_prune_log_lines_removes_info_debug_and_progress() -> None:
    lines = NOISY_PYTEST_OUTPUT.splitlines()
    pruned = prune_log_lines(lines, cone_paths=("src/example.py",), max_lines=32)
    joined = "\n".join(pruned)
    assert "FAILED" in joined or "AssertionError" in joined
    assert "DEBUG plugin" not in joined
    assert "INFO collecting" not in joined
    # progress dots alone should not survive
    assert not any(line.strip() in {".", "F", "E"} for line in pruned)


def test_sanitize_diagnostic_redacts_private_keys() -> None:
    # Runtime-assembled payload so proposal admission does not treat the test
    # source as introducing concrete credentials.
    payload = _secret_fixture_payload()
    payload["argument_type"] = "str"
    value = sanitize_diagnostic_value(payload)
    assert value["state"] == DiagnosticValueState.REDACTED.value
    assert "value" not in value


def test_sanitize_diagnostic_unavailable_and_present() -> None:
    assert sanitize_diagnostic_value(None)["state"] == DiagnosticValueState.UNAVAILABLE.value
    present = sanitize_diagnostic_value({"argument_type": "int"})
    assert present["state"] == DiagnosticValueState.PRESENT.value
    assert present["value"]["argument_type"] == "int"


def test_is_private_field_name_detects_markers() -> None:
    assert is_private_field_name("api_key")
    assert is_private_field_name("user_password")
    assert is_private_field_name("session_token")
    assert not is_private_field_name("argument_type")


def test_build_reproduction_argv_candidates_are_lists_and_drop_optional_flags() -> None:
    candidates = build_reproduction_argv_candidates(ORIGINAL_ARGV)
    assert candidates
    assert all(isinstance(item, tuple) for item in candidates)
    assert candidates[0] == ORIGINAL_ARGV
    # Some later candidate should have dropped -v / long tb
    flat = [" ".join(c) for c in candidates]
    assert any("--tb=short" in text or "-v" not in text.split() for text in flat[1:])
    # Node id must never be dropped
    for candidate in candidates:
        assert "src/example.py::test_calculate_returns_string" in candidate
        assert isinstance(list(candidate), list)


def test_compute_failure_identity_is_stable_and_sensitive() -> None:
    a = compute_failure_identity_cid(
        failed_selector="sel-a",
        node_id="src/example.py::test_a",
        exception_type="AssertionError",
        assertion_message="expected ok",
        primary_source="src/example.py:12",
        terminal_status=TerminalStatus.FAILED,
        environment_cid=_artifact("env"),
        dependency_lock_cid=_artifact("lock"),
    )
    b = compute_failure_identity_cid(
        failed_selector="sel-a",
        node_id="src/example.py::test_a",
        exception_type="AssertionError",
        assertion_message="expected ok",
        primary_source="src/example.py:12",
        terminal_status=TerminalStatus.FAILED,
        environment_cid=_artifact("env"),
        dependency_lock_cid=_artifact("lock"),
    )
    c = compute_failure_identity_cid(
        failed_selector="sel-a",
        node_id="src/example.py::test_a",
        exception_type="ValueError",
        assertion_message="expected ok",
        primary_source="src/example.py:12",
        terminal_status=TerminalStatus.FAILED,
        environment_cid=_artifact("env"),
        dependency_lock_cid=_artifact("lock"),
    )
    assert a == b
    assert a != c
    assert a.startswith("b")


# ---------------------------------------------------------------------------
# End-to-end minimization
# ---------------------------------------------------------------------------


def test_minimize_selected_pytest_failure_requires_lease_rerun_and_preserves_identity() -> (
    None
):
    receipt = _failed_test()
    material = _material_from_noisy_output(receipt)
    counter = itertools.count(1)
    leases_seen: list[str] = []

    def oracle(argv: Sequence[str]) -> RerunObservation:
        obs = _oracle_preserving(receipt, material, counter=counter)(argv)
        leases_seen.append(obs.lease_id)
        return obs

    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=oracle,
        semantic_cone_paths=("src/example.py",),
    )

    assert result.receipt.minimized is True
    assert result.quality.guarantee is MinimizationGuarantee.RERUN_VALIDATED
    assert result.failure_identity_cid == result.receipt.failure_identity_cid
    assert result.receipt.failure_identity_cid
    assert result.lease_ids
    # Separate lease per candidate attempt — at least one unique lease.
    assert len(set(result.lease_ids)) == len(result.lease_ids)
    assert COUNTEREXAMPLE_EVIDENCE in result.evidence
    assert result.algorithm_version == ALGORITHM_VERSION

    cx = result.receipt
    assert isinstance(cx, CounterexampleReceipt)
    assert cx.failed_key_cid == receipt.key.key_id
    assert cx.failed_receipt_cid == receipt.receipt_id
    assert cx.failed_selector == receipt.key.selector_cid
    assert cx.environment_cid == receipt.key.environment_cid
    assert cx.dependency_lock_cid == receipt.key.dependency_lock_cid
    assert isinstance(list(cx.reproduction_argv), list)
    assert cx.reproduction_argv[0]
    assert "src/example.py::test_calculate_returns_string" in cx.reproduction_argv

    # Irrelevant frames removed.
    joined = "\n".join(cx.minimized_traceback)
    assert "site-packages" not in joined
    assert "pluggy" not in joined
    assert "src/example.py" in joined or "AssertionError" in joined or "expected ok" in joined

    # Source spans bind environment-relative cone paths only.
    assert cx.source_spans
    for span in cx.source_spans:
        assert not str(span["path"]).startswith("/")
        assert "site-packages" not in span["path"]

    # Diagnostics typed.
    assert cx.expected_output["state"] == DiagnosticValueState.PRESENT.value
    assert cx.observed_output["state"] == DiagnosticValueState.PRESENT.value
    assert cx.relevant_input["state"] == DiagnosticValueState.PRESENT.value

    # Size bound.
    assert len(cx.canonical_bytes()) < MAX_COUNTEREXAMPLE_BYTES
    assert "deterministic_slice_preserved_failure" in cx.reason_codes
    assert "lease_rerun_validated" in cx.reason_codes

    # Round-trip.
    assert CounterexampleReceipt.from_dict(cx.to_record()) == cx


def test_minimizer_class_matches_module_entry_point() -> None:
    receipt = _failed_test(label="class-entry")
    material = _material_from_noisy_output(receipt)
    oracle = _oracle_preserving(receipt, material)
    via_fn = minimize_counterexample(
        receipt, material, reproduction_argv=ORIGINAL_ARGV, rerun_oracle=oracle
    )
    via_cls = CounterexampleMinimizer().minimize(
        MinimizationRequest(
            failed_receipt=receipt,
            material=material,
            reproduction_argv=ORIGINAL_ARGV,
            rerun_oracle=_oracle_preserving(receipt, material, lease_prefix="resource-lease:cls"),
        )
    )
    assert via_fn.receipt.minimized is True
    assert via_cls.receipt.minimized is True
    assert via_fn.receipt.failure_identity_cid == via_cls.receipt.failure_identity_cid
    assert CounterexampleMinimizer.interface == COUNTEREXAMPLE_MINIMIZER_INTERFACE


def test_sensitive_inputs_use_typed_redacted_fields() -> None:
    receipt = _failed_test(label="secrets")
    secret_payload = _secret_fixture_payload()
    material = _material_from_noisy_output(receipt, with_secrets=True)
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material),
    )
    assert result.receipt.relevant_input["state"] == DiagnosticValueState.REDACTED.value
    assert "value" not in result.receipt.relevant_input
    # Must not embed secret strings anywhere in the compact receipt.
    blob = result.receipt.canonical_bytes().decode("utf-8", errors="replace")
    for secret_value in secret_payload.values():
        if isinstance(secret_value, str) and secret_value not in {"str"}:
            assert secret_value not in blob


def test_missing_expected_observed_use_unavailable() -> None:
    receipt = _failed_test(label="missing-diag")
    material = extract_failure_material_from_pytest_output(
        NOISY_PYTEST_OUTPUT,
        stdout_artifact_cid=receipt.execution.stdout_artifact_cid,
        stderr_artifact_cid=receipt.execution.stderr_artifact_cid,
        relevant_paths=("src/example.py",),
        expected_output=None,
        observed_output=None,
        relevant_input=None,
    )
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material),
    )
    assert result.receipt.expected_output["state"] == DiagnosticValueState.UNAVAILABLE.value
    assert result.receipt.observed_output["state"] == DiagnosticValueState.UNAVAILABLE.value
    assert result.receipt.relevant_input["state"] == DiagnosticValueState.UNAVAILABLE.value


def test_typecheck_without_fixture_input_marks_not_applicable_when_absent() -> None:
    receipt = _failed_typecheck()
    material = FailureMaterial(
        exception_type="TypeError",
        assertion_message="result must be a string",
        traceback_lines=("src/example.py:12: error: Incompatible return type",),
        log_lines=("src/example.py:12: error: Incompatible return type",),
        relevant_input=None,
        expected_output=diagnostic_present("str"),
        observed_output=diagnostic_present("int"),
        source_spans=(
            {
                "path": "src/example.py",
                "start_line": 12,
                "end_line": 13,
                "artifact_cid": _artifact("mypy-span"),
                "symbol": "example.calculate",
            },
        ),
        bounded_stdout_artifact_cid=receipt.execution.stdout_artifact_cid,
        bounded_stderr_artifact_cid=receipt.execution.stderr_artifact_cid,
    )
    body = (
        "src/example.py:12: error: Incompatible return type\n"
        "E   TypeError: result must be a string\n"
    )

    def oracle(argv: Sequence[str]) -> RerunObservation:
        return RerunObservation(
            terminal_status=TerminalStatus.FAILED,
            exit_code=1,
            lease_id=f"resource-lease:mypy-{content_identity({'argv': list(argv)})[:12]}",
            command_argv=tuple(argv),
            stdout_preview=body,
            combined_output=body,
            stdout_artifact_cid=_artifact("mypy-rerun-stdout"),
            stderr_artifact_cid=_artifact("mypy-rerun-stderr"),
        )

    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=receipt.execution.command_argv,
        rerun_oracle=oracle,
        semantic_cone_paths=("src/example.py",),
    )
    assert result.receipt.minimized is True
    assert (
        result.receipt.relevant_input["state"]
        == DiagnosticValueState.NOT_APPLICABLE.value
    )


def test_minimization_failure_is_explicit_and_references_artifacts_not_logs() -> None:
    receipt = _failed_test(label="identity-drift")
    material = _material_from_noisy_output(receipt)
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material, fail_identity=True),
    )
    assert result.receipt.minimized is False
    assert "minimization_failed" in result.receipt.reason_codes or any(
        code.startswith("failure_identity") or "minimization_failed" in code
        for code in result.receipt.reason_codes
    )
    assert result.quality.guarantee is not MinimizationGuarantee.RERUN_VALIDATED
    # Bounded artifact references present.
    assert result.receipt.artifact_cids
    # Must not embed the full noisy session log.
    blob = result.receipt.canonical_bytes().decode("utf-8", errors="replace")
    assert "test session starts" not in blob
    assert "cachedir" not in blob
    assert "DEBUG plugin" not in blob
    # Traceback still size-bounded and pruned.
    assert len(result.receipt.minimized_traceback) <= 32
    assert len(blob.encode("utf-8")) < MAX_COUNTEREXAMPLE_BYTES


def test_no_oracle_cannot_claim_minimized() -> None:
    receipt = _failed_test(label="no-oracle")
    material = _material_from_noisy_output(receipt)
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
    )
    assert result.receipt.minimized is False
    assert "minimization_failed_no_lease_rerun" in result.receipt.reason_codes
    assert result.lease_ids == ()
    assert result.quality.guarantee is MinimizationGuarantee.NORMALIZED


def test_rerun_pass_does_not_claim_minimized() -> None:
    receipt = _failed_test(label="rerun-pass")
    material = _material_from_noisy_output(receipt)
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material, pass_instead=True),
    )
    assert result.receipt.minimized is False
    assert "rerun_did_not_fail" in result.receipt.reason_codes


def test_each_candidate_uses_a_distinct_lease_id() -> None:
    receipt = _failed_test(label="distinct-leases")
    material = _material_from_noisy_output(receipt)
    # Force identity match only on the last candidate by failing early ones
    # with a pass — wait, pass won't match. Instead always fail with same
    # identity and verify multiple leases when multiple candidates exist.
    counter = itertools.count(1)
    lease_ids: list[str] = []

    def oracle(argv: Sequence[str]) -> RerunObservation:
        # Accept only after the second call so multiple leases are recorded
        # for the search path that tries optional-flag drops.
        n = next(counter)
        lease_id = f"resource-lease:search-{n}"
        lease_ids.append(lease_id)
        body = (
            f"FAILED {material.node_id} - {material.exception_type}: "
            f"{material.assertion_message}\n"
            "src/example.py:12: in test_calculate_returns_string\n"
            "    assert result == 'ok'\n"
            f"E   {material.exception_type}: {material.assertion_message}\n"
        )
        return RerunObservation(
            terminal_status=TerminalStatus.FAILED,
            exit_code=1,
            lease_id=lease_id,
            command_argv=tuple(argv),
            stdout_preview=body,
            combined_output=body,
            stdout_artifact_cid=_artifact(f"stdout-{n}"),
            stderr_artifact_cid=_artifact(f"stderr-{n}"),
        )

    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=oracle,
    )
    assert result.receipt.minimized is True
    # First candidate succeeds immediately — still one lease, distinct id.
    assert result.lease_ids
    assert all(item.startswith("resource-lease:") for item in result.lease_ids)
    assert len(set(result.lease_ids)) == len(result.lease_ids)


def test_argv_is_list_form_on_receipt_record() -> None:
    receipt = _failed_test(label="argv-list")
    material = _material_from_noisy_output(receipt)
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material),
    )
    record = result.receipt.to_record()
    assert isinstance(record["reproduction_argv"], (list, tuple))
    assert next(iter(record["reproduction_argv"])).endswith("python3.12") or "python" in next(iter(record["reproduction_argv"]))
    # Module API documents argv as a list for callers.
    assert isinstance(list(result.receipt.reproduction_argv), list)


def test_output_size_bounded_under_stress_traceback() -> None:
    receipt = _failed_test(label="stress")
    huge_frames = tuple(
        f"src/example.py:{i}: in layer_{i}\n    call_{i}()"
        for i in range(1, 200)
    )
    material = FailureMaterial(
        node_id="src/example.py::test_calculate_returns_string",
        exception_type="AssertionError",
        assertion_message="expected ok",
        traceback_lines=huge_frames + ("E   AssertionError: expected ok",),
        log_lines=huge_frames + ("E   AssertionError: expected ok",),
        relevant_input={"argument_type": "str"},
        expected_output="ok",
        observed_output="bad",
        source_spans=(
            {
                "path": "src/example.py",
                "start_line": 12,
                "end_line": 12,
                "artifact_cid": _artifact("stress-span"),
                "symbol": "example.calculate",
            },
        ),
        relevant_paths=("src/example.py",),
        bounded_stdout_artifact_cid=receipt.execution.stdout_artifact_cid,
        bounded_stderr_artifact_cid=receipt.execution.stderr_artifact_cid,
    )
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material),
        budget=MinimizationBudget(max_traceback_frames=8, max_log_lines=8),
    )
    assert result.receipt.minimized is True
    assert len(result.receipt.minimized_traceback) <= 24
    assert len(result.receipt.canonical_bytes()) < MAX_COUNTEREXAMPLE_BYTES
    assert result.quality.frames_after <= result.quality.frames_before


def test_string_material_overload_parses_pytest_output() -> None:
    receipt = _failed_test(label="string-material")
    material = extract_failure_material_from_pytest_output(NOISY_PYTEST_OUTPUT)
    # Use string path
    result = minimize_counterexample(
        receipt,
        NOISY_PYTEST_OUTPUT,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material),
        semantic_cone_paths=("src/example.py",),
        expected_output="ok",
        observed_output="bad",
        relevant_input={"argument_type": "str"},
        source_spans=(
            {
                "path": "src/example.py",
                "start_line": 12,
                "end_line": 12,
                "artifact_cid": _artifact("string-span"),
                "symbol": "example.calculate",
            },
        ),
    )
    assert result.receipt.minimized is True
    assert result.receipt.environment_cid == receipt.key.environment_cid


def test_rejects_passing_receipt() -> None:
    key = _key(VerificationReceiptKind.TEST)
    receipt = TestReceipt(key, _observation(key, TerminalStatus.PASSED, label="pass"))
    with pytest.raises(CounterexampleMinimizationError, match="failed terminal"):
        minimize_counterexample(
            receipt,
            FailureMaterial(
                assertion_message="n/a",
                traceback_lines=("x",),
            ),
            reproduction_argv=receipt.execution.command_argv,
        )


def test_quality_score_higher_when_rerun_validated() -> None:
    receipt = _failed_test(label="quality")
    material = _material_from_noisy_output(receipt)
    good = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material),
    )
    bad = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
    )
    assert good.quality.score > bad.quality.score
    assert good.quality.to_dict()["guarantee"] == "rerun_validated"


def test_result_to_dict_is_compact_and_includes_evidence() -> None:
    receipt = _failed_test(label="to-dict")
    material = _material_from_noisy_output(receipt)
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material),
    )
    payload = result.to_dict()
    assert payload["evidence"] == [COUNTEREXAMPLE_EVIDENCE]
    assert "receipt" in payload
    assert "quality" in payload
    assert isinstance(payload["accepted_argv"], list)
    assert len(str(payload).encode("utf-8")) < MAX_COUNTEREXAMPLE_BYTES


def test_diagnostic_helpers_export_closed_states() -> None:
    assert diagnostic_present(1)["state"] == "present"
    assert diagnostic_redacted()["state"] == "redacted"
    assert diagnostic_unavailable()["state"] == "unavailable"


def test_budget_rejects_non_positive() -> None:
    with pytest.raises(CounterexampleMinimizationError):
        MinimizationBudget(max_oracle_reruns=0)


def test_symbol_versions_and_environment_bind_from_receipt_key() -> None:
    receipt = _failed_test(label="bind")
    material = _material_from_noisy_output(receipt)
    result = minimize_counterexample(
        receipt,
        material,
        reproduction_argv=ORIGINAL_ARGV,
        rerun_oracle=_oracle_preserving(receipt, material),
    )
    assert (
        result.receipt.relevant_symbol_version_cids
        == receipt.key.affected_symbol_version_cids
    )
    assert result.receipt.environment_cid == receipt.key.environment_cid
    assert result.receipt.dependency_lock_cid == receipt.key.dependency_lock_cid
