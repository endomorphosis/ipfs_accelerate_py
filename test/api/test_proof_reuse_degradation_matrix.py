"""Proof-backed test reuse degradation matrix.

These tests deliberately exercise fail-open behavior at contract, cache, capability,
and pytest-plugin boundaries.  A degraded reuse service may decline to reuse a
result, but it must never prevent the real test body from running.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    ReuseAction,
    ReuseDecision,
    ReuseReasonCode,
    TestExecutionKey as ExecutionKey,
    TestLocatorKey as LocatorKey,
    decision_from_absence,
    decision_from_exception,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import (
    TestProofCache as ProofCache,
    TestProofCacheLookupStatus as ProofCacheLookupStatus,
)
from ipfs_accelerate_py.agent_supervisor.integrations.test_reuse_capabilities import (
    TestReuseCapabilityName as CapabilityName,
    TestReuseCapabilityProbe as CapabilityProbe,
    TestReuseCapabilityStatus as CapabilityStatus,
)
from ipfs_accelerate_py.testing.proof_reuse.reporting import (
    ProofReuseSessionMetrics,
)


@dataclass(frozen=True)
class DegradationCase:
    """One externally distinguishable reason that must fall open to execution."""

    condition: str
    reason_code: ReuseReasonCode
    exception_path: bool = False


DEGRADATION_MATRIX = (
    DegradationCase("plugin_absent", ReuseReasonCode.PLUGIN_UNAVAILABLE),
    DegradationCase("plugin_disabled", ReuseReasonCode.PLUGIN_UNAVAILABLE),
    DegradationCase("cache_absent", ReuseReasonCode.CACHE_UNAVAILABLE),
    DegradationCase("cache_unreachable", ReuseReasonCode.CACHE_UNAVAILABLE),
    DegradationCase("cache_read_only", ReuseReasonCode.CACHE_UNAVAILABLE),
    DegradationCase("locator_miss", ReuseReasonCode.CANDIDATE_MISSING),
    DegradationCase(
        "candidate_corrupt", ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED
    ),
    DegradationCase(
        "candidate_oversized", ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED
    ),
    DegradationCase(
        "candidate_path_escape", ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED
    ),
    DegradationCase(
        "multiformats_or_cid_missing", ReuseReasonCode.CID_PROVIDER_UNAVAILABLE
    ),
    DegradationCase(
        "datasets_zk_missing", ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE
    ),
    DegradationCase(
        "datasets_zk_incompatible", ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE
    ),
    DegradationCase("groth16_issuer_missing", ReuseReasonCode.CERTIFICATE_DEFERRED),
    DegradationCase("provekit_issuer_missing", ReuseReasonCode.CERTIFICATE_DEFERRED),
    DegradationCase("local_verifier_missing", ReuseReasonCode.VERIFIER_UNAVAILABLE),
    DegradationCase("verifier_key_missing", ReuseReasonCode.KEY_UNAVAILABLE),
    DegradationCase("verifier_circuit_missing", ReuseReasonCode.CIRCUIT_UNAVAILABLE),
    DegradationCase(
        "simulated_proof", ReuseReasonCode.CERTIFICATE_NON_ATTESTED
    ),
    DegradationCase("expired_certificate", ReuseReasonCode.EXPIRED_OR_REVOKED),
    DegradationCase("revoked_issuer", ReuseReasonCode.ISSUER_REVOKED),
    DegradationCase("wrong_issuer", ReuseReasonCode.TRUST_POLICY_REJECTED),
    DegradationCase("wrong_policy", ReuseReasonCode.POLICY_MISMATCH),
    DegradationCase("incomplete_trace", ReuseReasonCode.INCOMPLETE_TRACE),
    DegradationCase("changed_trace", ReuseReasonCode.INVALIDATION),
    DegradationCase("xdist_coordination_failure", ReuseReasonCode.COORDINATION_UNAVAILABLE),
    DegradationCase(
        "unexpected_exception",
        ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
        exception_path=True,
    ),
)


EXPECTED_DEGRADATION_REASONS: Mapping[str, ReuseReasonCode] = {
    case.condition: case.reason_code for case in DEGRADATION_MATRIX
}


def _degraded_decision(case: DegradationCase) -> ReuseDecision:
    if case.exception_path:
        return decision_from_exception(
            RuntimeError("provider-specific detail must not escape"),
            reason_code=case.reason_code,
            diagnostics={"matrix_case": case.condition},
        )
    return decision_from_absence(
        case.reason_code,
        diagnostics={"matrix_case": case.condition},
    )


def _execute_real_test(
    decision: ReuseDecision,
    test_body: Callable[[], None],
) -> None:
    """Model the only safe response to a non-authoritative reuse decision."""

    assert decision.action is ReuseAction.RUN
    assert decision.reason_code is not ReuseReasonCode.PROOF_CACHE_HIT
    assert not decision.receipt_cid
    assert not decision.certificate_cid
    test_body()


@pytest.mark.parametrize(
    "case",
    DEGRADATION_MATRIX,
    ids=lambda case: case.condition,
)
def test_each_degradation_reason_is_bounded_and_executes_real_test(
    case: DegradationCase,
) -> None:
    calls: list[str] = []
    metrics = ProofReuseSessionMetrics()
    decision = _degraded_decision(case)

    metrics.degraded(reason_code=decision.reason_code.value)
    _execute_real_test(decision, lambda: calls.append(case.condition))
    metrics.executed(reason_code="real_execution")

    assert calls == [case.condition]
    assert decision.reason_code is case.reason_code
    assert 0 < len(decision.reason_code.value) <= 96
    assert decision.reason_code.value.replace("_", "").isalnum()

    snapshot = metrics.snapshot().to_dict()
    assert snapshot["counts"] == {
        "predicted": 0,
        "verified": 0,
        "skipped": 0,
        "executed": 1,
        "deferred": 0,
        "degraded": 1,
    }
    assert snapshot["reasons"] == {
        case.reason_code.value: 1,
        "real_execution": 1,
    }


def test_matrix_population_covers_every_planned_degradation_row() -> None:
    assert EXPECTED_DEGRADATION_REASONS == {
        "plugin_absent": ReuseReasonCode.PLUGIN_UNAVAILABLE,
        "plugin_disabled": ReuseReasonCode.PLUGIN_UNAVAILABLE,
        "cache_absent": ReuseReasonCode.CACHE_UNAVAILABLE,
        "cache_unreachable": ReuseReasonCode.CACHE_UNAVAILABLE,
        "cache_read_only": ReuseReasonCode.CACHE_UNAVAILABLE,
        "locator_miss": ReuseReasonCode.CANDIDATE_MISSING,
        "candidate_corrupt": ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
        "candidate_oversized": ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
        "candidate_path_escape": ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
        "multiformats_or_cid_missing": ReuseReasonCode.CID_PROVIDER_UNAVAILABLE,
        "datasets_zk_missing": ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
        "datasets_zk_incompatible": ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
        "groth16_issuer_missing": ReuseReasonCode.CERTIFICATE_DEFERRED,
        "provekit_issuer_missing": ReuseReasonCode.CERTIFICATE_DEFERRED,
        "local_verifier_missing": ReuseReasonCode.VERIFIER_UNAVAILABLE,
        "verifier_key_missing": ReuseReasonCode.KEY_UNAVAILABLE,
        "verifier_circuit_missing": ReuseReasonCode.CIRCUIT_UNAVAILABLE,
        "simulated_proof": ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
        "expired_certificate": ReuseReasonCode.EXPIRED_OR_REVOKED,
        "revoked_issuer": ReuseReasonCode.ISSUER_REVOKED,
        "wrong_issuer": ReuseReasonCode.TRUST_POLICY_REJECTED,
        "wrong_policy": ReuseReasonCode.POLICY_MISMATCH,
        "incomplete_trace": ReuseReasonCode.INCOMPLETE_TRACE,
        "changed_trace": ReuseReasonCode.INVALIDATION,
        "xdist_coordination_failure": ReuseReasonCode.COORDINATION_UNAVAILABLE,
        "unexpected_exception": ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
    }
    assert len(EXPECTED_DEGRADATION_REASONS) == len(DEGRADATION_MATRIX)


def test_capability_probe_reports_all_missing_providers_without_side_effects() -> None:
    report = CapabilityProbe(
        find_spec=lambda _module_name: None,
        which=lambda _executable_name: None,
        path_is_file=lambda _path: False,
        path_is_dir=lambda _path: False,
        environ={},
    ).probe()
    executed: list[str] = []

    for capability in report.capabilities:
        assert capability.status is CapabilityStatus.MISSING
        assert capability.optional is True
        assert capability.blocking is False
        assert capability.test_action == "run"
        assert 0 < len(capability.reason_code) <= 128
        assert capability.reason_code.isprintable()
        executed.append(capability.name)

    assert executed == [
        CapabilityName.MULTIFORMATS.value,
        CapabilityName.DATASETS_ZK.value,
        CapabilityName.GROTH16.value,
        CapabilityName.PROVEKIT.value,
        CapabilityName.CACHE.value,
        CapabilityName.IPFS.value,
        CapabilityName.LOCAL_VERIFIER.value,
    ]

    payload = report.to_dict()
    assert payload["bounded"] is True
    assert payload["lazy"] is True
    assert payload["side_effect_free"] is True
    assert payload["network_attempted"] is False
    assert payload["daemon_started"] is False
    assert payload["cache_created"] is False


def _locator_key() -> LocatorKey:
    return LocatorKey(
        repository_id="repository:degradation-matrix",
        package_identity="package:ipfs-accelerate-py",
        node_id="test_degradation_matrix.py::test_real_body",
    )


def _execution_key(locator_key: LocatorKey) -> ExecutionKey:
    return ExecutionKey(
        locator_cid=locator_key.locator_id,
        repository_forest_cid="bafy-repository-forest",
        static_trace_root_cid="bafy-static-trace-root",
        runtime_trace_root_cid="bafy-runtime-trace-root",
        runtime_completeness_policy="complete-v1",
        policy_cid="bafy-policy",
    )


@pytest.mark.parametrize(
    ("candidate_provider", "expected_status", "expected_reason"),
    (
        (
            lambda _locator_key: (),
            ProofCacheLookupStatus.MISS,
            ReuseReasonCode.CANDIDATE_MISSING,
        ),
        (
            lambda _locator_key: (_ for _ in ()).throw(OSError("cache offline")),
            ProofCacheLookupStatus.ERROR,
            ReuseReasonCode.CACHE_UNAVAILABLE,
        ),
    ),
    ids=("locator-miss", "cache-unreachable"),
)
def test_cache_lookup_degradation_executes_real_test(
    candidate_provider: Callable[[LocatorKey], object],
    expected_status: ProofCacheLookupStatus,
    expected_reason: ReuseReasonCode,
) -> None:
    locator_key = _locator_key()
    result = ProofCache(candidate_provider=candidate_provider).lookup(
        locator_key,
        _execution_key(locator_key),
    )
    calls: list[str] = []

    assert result.status is expected_status
    assert result.decision.reason_code is expected_reason
    _execute_real_test(result.decision, lambda: calls.append("executed"))
    assert calls == ["executed"]


@dataclass(frozen=True)
class PytestModeCase:
    name: str
    mode: str
    load_plugin: bool = True
    required_audit: bool = False


PYTEST_MODE_CASES = (
    PytestModeCase("plugin-absent", "readwrite", load_plugin=False),
    PytestModeCase("mode-off", "off"),
    PytestModeCase("mode-shadow", "shadow"),
    PytestModeCase("mode-read", "read"),
    PytestModeCase("mode-write", "write"),
    PytestModeCase("mode-readwrite", "readwrite"),
    PytestModeCase("required-audit", "read", required_audit=True),
)


def _write_isolated_test(test_dir: Path) -> tuple[Path, Path]:
    test_file = test_dir / "test_real_execution.py"
    marker = test_dir / "executed.marker"
    test_file.write_text(
        "\n".join(
            (
                "import os",
                "from pathlib import Path",
                "",
                "def test_real_body():",
                '    Path(os.environ["PTR_EXECUTION_MARKER"]).write_text(',
                '        "executed", encoding="utf-8"',
                "    )",
                "",
            )
        ),
        encoding="utf-8",
    )
    return test_file, marker


def _write_optional_import_blocker(test_dir: Path) -> None:
    """Make optional provider absence deterministic in the child interpreter."""

    (test_dir / "sitecustomize.py").write_text(
        "\n".join(
            (
                "import builtins",
                "",
                "_real_import = builtins.__import__",
                "_blocked_roots = {",
                '    "groth16",',
                '    "ipfs_datasets_py",',
                '    "ipfshttpclient",',
                '    "ipfs_kit_py",',
                '    "multiformats",',
                '    "provekit",',
                "}",
                "",
                "def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):",
                "    root = name.partition('.')[0]",
                "    if level == 0 and root in _blocked_roots:",
                "        raise ModuleNotFoundError(",
                '            f"optional provider {root!r} blocked by degradation matrix"',
                "        )",
                "    return _real_import(name, globals, locals, fromlist, level)",
                "",
                "builtins.__import__ = _guarded_import",
                "",
            )
        ),
        encoding="utf-8",
    )


def _child_environment(test_dir: Path, marker: Path, mode: str) -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_parts = [str(test_dir), str(package_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PTR_EXECUTION_MARKER": str(marker),
            "IPFS_TEST_PROOF_REUSE_MODE": mode,
            "IPFS_TEST_PROOF_REUSE_DISABLE_GROTH16": "1",
            "IPFS_TEST_PROOF_REUSE_DISABLE_PROVEKIT": "1",
            "IPFS_TEST_PROOF_REUSE_DISABLE_CACHE": "1",
            "IPFS_TEST_PROOF_REUSE_DISABLE_IPFS": "1",
            "IPFS_TEST_PROOF_REUSE_DISABLE_LOCAL_VERIFIER": "1",
            "IPFS_TEST_PROOF_REUSE_DISABLE_MULTIFORMATS": "1",
            "IPFS_TEST_PROOF_REUSE_DISABLE_DATASETS_ZK": "1",
            "IPFS_TEST_PROOF_REUSE_CACHE_DIR": str(test_dir / "missing-cache"),
            "IPFS_TEST_PROOF_REUSE_VERIFIER_KEY": str(test_dir / "missing-key"),
            "IPFS_TEST_PROOF_REUSE_VERIFIER_CIRCUIT": str(
                test_dir / "missing-circuit"
            ),
        }
    )
    return environment


@pytest.mark.parametrize(
    "case",
    PYTEST_MODE_CASES,
    ids=lambda case: case.name,
)
def test_pytest_startup_and_real_execution_survive_missing_optional_providers(
    case: PytestModeCase,
    tmp_path: Path,
) -> None:
    test_file, marker = _write_isolated_test(tmp_path)
    _write_optional_import_blocker(tmp_path)
    command = [sys.executable, "-m", "pytest"]
    if case.load_plugin:
        command.extend(("-p", "ipfs_accelerate_py.testing.proof_reuse.plugin"))
        command.append(f"--proof-reuse-mode={case.mode}")
        if case.required_audit:
            command.append("--proof-reuse-required-audit")
    command.extend((str(test_file), "-q"))

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=_child_environment(tmp_path, marker, case.mode),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    combined_output = completed.stdout + completed.stderr

    assert marker.read_text(encoding="utf-8") == "executed", combined_output
    assert "1 passed" in completed.stdout, combined_output

    if case.required_audit and completed.returncode != 0:
        assert completed.returncode == 1, combined_output
        assert "proof reuse" in combined_output.lower()
        assert (
            "required-audit" in combined_output.lower()
            or "required audit" in combined_output.lower()
        )
    else:
        assert completed.returncode == 0, combined_output
        standard_output = completed.stdout.lower().replace("skipped=0", "")
        assert "skipped" not in standard_output
        assert "failed" not in completed.stdout.lower()

    if not case.load_plugin or case.mode == "off":
        assert "proof reuse:" not in combined_output.lower()
    else:
        assert "proof reuse:" in combined_output.lower()
        assert "skipped=0" in combined_output.lower()
        assert "executed=1" in combined_output.lower()
