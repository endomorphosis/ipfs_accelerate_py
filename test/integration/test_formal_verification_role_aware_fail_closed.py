"""Adversarial fail-closed regressions for role-aware release evidence."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)
BUILDER_PATH = (
    REPO_ROOT
    / "tools"
    / "logic"
    / "build_formal_verification_tactician_receipt.py"
)


def _load(path: Path, name: str) -> Any:
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def certifier():
    return _load(CERTIFIER_PATH, "fvt_fail_closed_certifier_test")


@pytest.fixture
def builder():
    return _load(BUILDER_PATH, "fvt_fail_closed_builder_test")


def _passed_checks(tool_id: str) -> list[dict[str, Any]]:
    return [
        {
            "check_id": f"{tool_id}.{kind}",
            "kind": kind,
            "status": "passed",
        }
        for kind in ("positive", "negative", "mutation", "replay")
    ]


def _bound_certificate(builder) -> dict[str, Any]:
    certificate = {
        "interface": "FormalVerificationToolchainCertificate@1",
        "disagreement_quarantines": [],
    }
    certificate["certificate_digest_sha256"] = builder.content_digest(
        certificate
    ).removeprefix("sha256:")
    return certificate


def _bound_benchmark(
    builder,
    *,
    evidence_class: str,
    gate_bps: int = 10000,
) -> dict[str, Any]:
    passed = gate_bps == 10000
    report = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "goal-tactician-benchmark-report@1"
        ),
        "interface": "GoalTacticianBenchmark@1",
        "source": "cohort_receipts",
        "synthetic_distributions": False,
        "receipt_ids": [f"receipt:test:{evidence_class}:1"],
        "receipt_count": 1,
        "metrics": {
            "source": "cohort_receipts",
            "synthetic_distributions": False,
            "evidence_classes": [evidence_class],
            "hard_gates": {
                "correctness_bps": gate_bps,
                "privacy_bps": gate_bps,
                "authority_bps": gate_bps,
                "passed": passed,
            },
        },
        "gates": {
            "hard": {
                name: {
                    "actual_bps": gate_bps,
                    "required_bps": 10000,
                    "status": "pass" if passed else "fail",
                }
                for name in ("correctness", "privacy", "authority")
            }
        },
    }
    report["report_id"] = (
        "goal-tactician-bench-"
        + builder.content_digest(report).removeprefix("sha256:")
    )
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "goal-tactician-benchmark@1"
        ),
        "interface": "GoalTacticianBenchmark@1",
        "synthetic_distributions": False,
        "report": report,
    }


def test_missing_hard_zero_inputs_are_unresolved_not_zero(builder) -> None:
    missing = builder.derive_hard_zero_gates(
        certificate=None,
        benchmark=None,
        baseline=None,
    )
    assert missing["derivation"]["complete"] is False
    assert {"certificate", "benchmark"} <= set(
        missing["derivation"]["missing_measurements"]
    )
    assert all(missing[key] > 0 for key in builder.HARD_ZERO_GATE_KEYS)

    partial = builder.derive_hard_zero_gates(
        certificate={"disagreement_quarantines": []},
        benchmark={"report": {"gates": {"hard": {}}}},
        baseline=None,
    )
    assert partial["derivation"]["complete"] is False
    assert partial["false_proof_count"] > 0
    assert partial["secret_or_witness_leakage_count"] > 0
    assert partial["authority_boundary_violations"] > 0


def test_fixture_benchmark_cannot_clear_deployment_hard_zero(builder) -> None:
    result = builder.derive_hard_zero_gates(
        certificate=_bound_certificate(builder),
        benchmark=_bound_benchmark(builder, evidence_class="fixture"),
        baseline={"known_findings": []},
    )
    assert result["derivation"]["complete"] is False
    assert "benchmark.benchmark_fixture_or_synthetic_evidence" in result[
        "derivation"
    ]["missing_measurements"]
    assert result["derivation"]["benchmark_evidence"]["authoritative"] is False
    assert all(result[key] > 0 for key in builder.HARD_ZERO_GATE_KEYS[:-1])


def test_self_declared_live_benchmark_cannot_clear_without_authority_anchor(
    builder,
) -> None:
    result = builder.derive_hard_zero_gates(
        certificate=_bound_certificate(builder),
        benchmark=_bound_benchmark(builder, evidence_class="live"),
        baseline={"known_findings": []},
        repo_root=REPO_ROOT,
    )
    evidence = result["derivation"]["benchmark_evidence"]
    assert result["derivation"]["complete"] is False
    assert evidence["authoritative"] is False
    assert evidence["authority_anchor"]["bound"] is False
    assert (
        "benchmark_authoritative_measurement_anchor_missing"
        in evidence["authority_anchor"]["failures"]
    )
    assert all(result[key] > 0 for key in builder.HARD_ZERO_GATE_KEYS[:-1])


def test_malformed_live_benchmark_population_fails_closed(builder) -> None:
    benchmark = _bound_benchmark(builder, evidence_class="live")
    report = benchmark["report"]
    report["receipt_count"] = "1"
    report.pop("report_id")
    report["report_id"] = (
        "goal-tactician-bench-"
        + builder.content_digest(report).removeprefix("sha256:")
    )

    result = builder.derive_hard_zero_gates(
        certificate=_bound_certificate(builder),
        benchmark=benchmark,
        baseline={"known_findings": []},
    )
    assert result["derivation"]["complete"] is False
    assert result["derivation"]["benchmark_evidence"]["authoritative"] is False
    assert "benchmark.benchmark_receipt_population_invalid" in result[
        "derivation"
    ]["missing_measurements"]


def test_open_and_unknown_p0_findings_apply_nonzero_gate_pressure(builder) -> None:
    result = builder.derive_hard_zero_gates(
        certificate=_bound_certificate(builder),
        benchmark=_bound_benchmark(builder, evidence_class="live"),
        baseline={
            "known_findings": [
                {
                    "id": "receipt_verification_fail_open",
                    "severity": "p0",
                    "status": "open",
                },
                {
                    "id": "public_counterexample_raw_leak",
                    "severity": "p0",
                    "status": "open",
                },
                {
                    "id": "structural_repair_as_closure",
                    "severity": "p0",
                    "status": "open",
                },
                {
                    "id": "unclassified_p0",
                    "severity": "p0",
                    "status": "open",
                },
            ]
        },
    )
    assert result["derivation"]["complete"] is False
    assert "baseline.unresolved_open_p0_findings" in result["derivation"][
        "missing_measurements"
    ]
    assert len(result["derivation"]["open_p0_findings"]) == 4
    assert result["false_proof_count"] > 0
    assert result["false_closure_count"] > 0
    assert result["secret_or_witness_leakage_count"] > 0
    assert result["authority_boundary_violations"] > 0


def test_unanchored_or_split_supervisor_json_never_binds(builder) -> None:
    cid = "baguqeera-forged"
    key = "task/v1/forged"
    receipt = {
        "schema": builder.SUPERVISOR_COMPLETION_SCHEMA,
        "status": "succeeded",
        "task_id": "FVT-053",
        "canonical_task_cid": cid,
        "canonical_task_key": key,
    }
    forged = {
        "task_id": "FVT-053",
        "task_state": {
            "canonical_identity": {
                "canonical_task_cid": cid,
                "canonical_task_key": key,
            }
        },
        "events": [
            {
                "type": "implementation_finished",
                "task_id": "FVT-053",
                "canonical_task_cid": cid,
                "canonical_task_key": key,
                "validation": {"attempted": True, "passed": True, "returncode": 0},
            },
            {
                "type": "implementation_finished",
                "task_id": "FVT-053",
                "canonical_task_cid": cid,
                "canonical_task_key": key,
                "merge": {"merged": True, "merge_commit": "a" * 40},
            },
            {
                "type": "other",
                "task_id": "FVT-053",
                "canonical_task_cid": cid,
                "canonical_task_key": key,
                "completion_receipts": [receipt],
            },
        ],
    }
    binding = builder.derive_supervisor_binding(forged)
    assert binding["bound"] is False
    assert binding["source_files_bound"] is False
    assert binding["event_chain_bound"] is False


def test_forged_g212_envelope_without_bound_exporter_never_binds(builder) -> None:
    forged = {
        "schema": builder.SUPERVISOR_RELEASE_EVIDENCE_SCHEMA,
        "interface": builder.SUPERVISOR_RELEASE_EVIDENCE_INTERFACE,
        "goal_id": builder.SUPERVISOR_RELEASE_EVIDENCE_GOAL_ID,
        "exporter": {
            "path": (
                builder.SUPERVISOR_RELEASE_EVIDENCE_EXPORTER_RELATIVE.as_posix()
            ),
            "sha256": "0" * 64,
        },
        "snapshot": {
            "task_id": "FVT-053",
            "task_state": {"task_status": "completed"},
        },
    }
    forged["content_id"] = builder.content_digest(forged)

    binding = builder.derive_supervisor_binding(
        forged,
        repo_root=REPO_ROOT,
    )
    failures = set(binding["trusted_release_evidence"]["failures"])
    assert binding["bound"] is False
    assert binding["trusted_release_evidence_bound"] is False
    assert {
        "trusted_release_evidence_exporter_missing",
        "trusted_release_evidence_exporter_identity_mismatch",
    } & failures


def test_fake_unreachable_commits_never_bind_supervisor_merge(builder) -> None:
    binding = builder._derive_git_commit_binding(
        repo_root=REPO_ROOT,
        implementation_commit="b" * 40,
        merge_commit="c" * 40,
        target_branch="origin/main",
        integration_proof={
            "passed": True,
            "implementation_tree": "d" * 40,
            "merge_tree": "e" * 40,
        },
    )
    assert binding["valid"] is False
    assert binding["implementation_commit_exists"] is False
    assert binding["merge_commit_exists"] is False
    assert "implementation_commit_unreachable" in binding["failures"]
    assert "merge_commit_unreachable" in binding["failures"]


def test_wrong_digest_or_offline_violation_cannot_elevate(certifier) -> None:
    receipt = {
        "interface": "RuntimeMTLSemanticCertification@1",
        "certified": True,
        "checks": _passed_checks("runtime-mtl"),
        "network_used": True,
        "install_attempted": True,
    }
    result = {
        "lane_id": "runtime_mtl",
        "status": "ran",
        "tool_ids": ["runtime-mtl"],
        "receipt": receipt,
        "digest_sha256": "deliberately-wrong",
        "receipt_integrity": {"valid": True},
        "offline_observation": {"satisfied": False},
        "per_tool": {},
    }
    tool = certifier.ToolCertification(
        tool_id="runtime-mtl",
        usable=True,
        installed=True,
        identity_probed=True,
        unavailable=False,
    )
    elevations = certifier.apply_semantic_elevations(
        {"runtime-mtl": tool},
        [result],
        repo_root=REPO_ROOT,
    )
    assert tool.production_certified is False
    assert elevations[0]["reason"] == "semantic_receipt_integrity_failed"


def test_repository_source_hash_cannot_replace_lean_executable_identity(
    certifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = next(
        item
        for item in certifier.SEMANTIC_CERTIFIER_SPECS
        if item["lane_id"] == "kernel"
    )
    monkeypatch.setitem(spec, "production_elevation_allowed", True)
    checks = _passed_checks("lean")
    receipt = {
        "interface": spec["interface"],
        "schema_version": "lean-semantic-certification/v1",
        "goal_id": "FVT-G101",
        "task_id": "FVT-040",
        "production_certified": True,
        "checks": checks,
    }
    receipt["receipt_digest_sha256"] = certifier.content_digest(receipt)
    normalized_checks = [
        check.to_dict()
        for check in certifier._normalize_semantic_checks("lean", checks)
    ]
    module_path = REPO_ROOT / spec["module_relative"]
    artifacts = [
        {
            "kind": "semantic_certifier_module",
            "path": spec["module_relative"].as_posix(),
            "sha256": certifier.file_digest(module_path),
            "artifact_class": "repository_source",
        }
    ]
    artifact_validation = certifier._validate_artifact_identities(
        artifacts,
        repo_root=REPO_ROOT,
    )
    result = {
        "lane_id": "kernel",
        "status": "ran",
        "tool_ids": ["lean"],
        "receipt": receipt,
        "digest_sha256": certifier.content_digest(receipt),
        "receipt_integrity": {"valid": True},
        "offline_observation": {"satisfied": True},
        "per_tool": {
            "lean": {
                "certified": True,
                "checks": normalized_checks,
                "check_set_digest_sha256": certifier.content_digest(
                    normalized_checks
                ),
                "identity": {
                    "executable_path": None,
                    "version_string": "v4.31.0",
                    "identity_probed": True,
                    "artifacts": artifacts,
                },
                "artifact_validation": artifact_validation,
            }
        },
    }
    tool = certifier.ToolCertification(
        tool_id="lean",
        locked_version="v4.31.0",
        unavailable=True,
    )
    elevations = certifier.apply_semantic_elevations(
        {"lean": tool},
        [result],
        repo_root=REPO_ROOT,
    )
    assert tool.production_certified is False
    assert tool.installed is False
    assert elevations[0]["reason"] == "semantic_identity_not_exactly_bound"


def test_missing_external_kernels_cannot_become_usable_from_stale_identity(
    certifier,
) -> None:
    tools = {
        tool_id: certifier.ToolCertification(
            tool_id=tool_id,
            installed=True,
            identity_probed=True,
            usable=True,
            unavailable=False,
            production_certified=True,
            promotion_blocked=False,
        )
        for tool_id in ("coq", "isabelle")
    }
    certifier.apply_semantic_elevations(
        tools,
        [
            {
                "lane_id": "kernel_rocq",
                "tool_ids": ["coq"],
                "status": "not_run",
            },
            {
                "lane_id": "kernel_isabelle",
                "tool_ids": ["isabelle"],
                "status": "not_run",
            },
        ],
        repo_root=REPO_ROOT,
    )

    for tool in tools.values():
        assert tool.usable is False
        assert tool.unavailable is True
        assert tool.production_certified is False
        assert tool.promotion_blocked is True
        assert (
            "external_prover_installation_and_live_fanin_pending"
            in tool.block_reasons
        )
        assert tool.evidence_class == "external_prover_installation_pending"


def test_nonzero_error_banner_is_not_an_identity(
    certifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(certifier, "resolve_executable", lambda _: "/bin/false")
    monkeypatch.setattr(
        certifier,
        "bounded_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["/bin/false"],
            42,
            "",
            "fatal: not a version 9.9.9\n",
        ),
    )
    result = certifier.probe_tool_identity(
        {
            "tool_id": "fake",
            "availability": "managed_pin",
            "executable_candidates": ["fake"],
            "offline_probe": {"argv": ["--version"]},
        },
        env={},
    )
    assert result["installed"] is False
    assert result["identity_probed"] is False
    assert result["probe_error"] == "identity_probe_nonzero:42"


def test_platform_contradiction_is_ambiguous_and_any_is_not_discarded(
    certifier,
) -> None:
    host = certifier.observed_platform_id()
    contradictory = certifier.tool_platform_support(
        {
            "tool_id": "contradictory",
            "availability": "managed_pin",
            "deployment_contract": {"supported_platforms": [host]},
            "pins": [{"platform": "plan9-mips"}],
        },
        host_platform=host,
        global_supported_platforms=[host],
    )
    assert contradictory["classification"] == "ambiguous"
    assert contradictory["exception_eligible"] is False

    any_pin = certifier.tool_platform_support(
        {
            "tool_id": "any",
            "availability": "managed_pin",
            "deployment_contract": {"supported_platforms": ["any"]},
            "pins": [{"platform": "any"}],
        },
        host_platform=host,
        global_supported_platforms=["plan9-mips"],
    )
    assert any_pin["classification"] == "ambiguous"
    assert any_pin["exception_eligible"] is False
