"""DCR-004 hermetic tests for deterministic repair capability receipts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.capabilities import (
    DETERMINISTIC_REPAIR_CAPABILITIES_INTERFACE,
    SOLVER_READINESS_INTERFACE,
    CapabilityEvidenceReceipt,
    CapabilityStatus,
    DeterministicRepairCapabilityProbe,
    LogicModuleRequirement,
    NetworkMode,
    ToolchainRequirement,
)

MODULE = "ipfs_datasets_py.logic.fixture_logic"


def _digest(label: str, value: bytes) -> str:
    return f"{label}:sha256:{hashlib.sha256(value).hexdigest()}"


def _evidence(
    *,
    evidence_id: str,
    evidence_kind: str,
    subject_id: str,
    subject_digest: str,
    subject_version: str,
) -> CapabilityEvidenceReceipt:
    transcript = f"{evidence_id}:{evidence_kind}:{subject_digest}".encode()
    return CapabilityEvidenceReceipt(
        evidence_id=evidence_id,
        evidence_kind=evidence_kind,
        subject_id=subject_id,
        subject_digest=subject_digest,
        subject_version=subject_version,
        transcript_digest=_digest("transcript", transcript),
        passed=True,
    )


def _module_source(
    *, marker: str = "", initialized: bool = True, reconstructed: bool = True
) -> str:
    return "\n".join(
        (
            marker,
            "class ExactLogic: pass",
            "def reconstruct(): return 'local'",
            f"INITIALIZED = {initialized!r}",
            f"RECONSTRUCTION_READY = {reconstructed!r}",
        )
    )


def _probe(tmp_path: Path, source: str, **overrides: object) -> DeterministicRepairCapabilityProbe:
    module_path = tmp_path / "fixture_logic.py"
    module_path.write_text(source, encoding="utf-8")
    executable = tmp_path / "fixture-prover"
    executable.write_bytes(b"local deterministic prover")
    executable.chmod(0o700)
    module_req = LogicModuleRequirement(
        module=MODULE,
        distribution="ipfs-datasets-py",
        expected_version="1.2.3",
        required_symbols=("ExactLogic", "reconstruct"),
    )
    tool_req = ToolchainRequirement(
        tool_id="fixture-prover",
        executable="fixture-prover",
        expected_version="4.0",
        self_test_id="self-test:fixture-prover",
        reconstruction_id="reconstruction:fixture-prover",
    )
    module_digest = _digest("module", source.encode())
    executable_digest = _digest("executable", executable.read_bytes())
    kwargs: dict[str, object] = {
        "module_requirements": (module_req,),
        "toolchain_requirements": (tool_req,),
        "find_spec": lambda name: (
            SimpleNamespace(origin=str(module_path)) if name == MODULE else None
        ),
        "distribution_version": lambda distribution: "1.2.3",
        "executable_finder": lambda name: str(executable) if name == "fixture-prover" else None,
        "initialization_evidence": {
            MODULE: _evidence(
                evidence_id=MODULE,
                evidence_kind="initialization",
                subject_id=MODULE,
                subject_digest=module_digest,
                subject_version="1.2.3",
            )
        },
        "self_test_evidence": {
            MODULE: _evidence(
                evidence_id=MODULE,
                evidence_kind="self_test",
                subject_id=MODULE,
                subject_digest=module_digest,
                subject_version="1.2.3",
            ),
            "self-test:fixture-prover": _evidence(
                evidence_id="self-test:fixture-prover",
                evidence_kind="self_test",
                subject_id="fixture-prover",
                subject_digest=executable_digest,
                subject_version="4.0",
            ),
        },
        "reconstruction_evidence": {
            MODULE: _evidence(
                evidence_id=MODULE,
                evidence_kind="reconstruction",
                subject_id=MODULE,
                subject_digest=module_digest,
                subject_version="1.2.3",
            ),
            "reconstruction:fixture-prover": _evidence(
                evidence_id="reconstruction:fixture-prover",
                evidence_kind="reconstruction",
                subject_id="fixture-prover",
                subject_digest=executable_digest,
                subject_version="4.0",
            ),
        },
        "executable_versions": {"fixture-prover": "4.0"},
    }
    kwargs.update(overrides)
    return DeterministicRepairCapabilityProbe(**kwargs)  # type: ignore[arg-type]


def test_exact_local_receipts_are_content_addressed_and_never_execute_tools(tmp_path: Path) -> None:
    calls: list[str] = []
    probe = _probe(
        tmp_path,
        _module_source(),
        executable_finder=lambda name: calls.append(name) or str(tmp_path / "fixture-prover"),
    )
    report = probe.probe()
    assert DETERMINISTIC_REPAIR_CAPABILITIES_INTERFACE == "DeterministicRepairCapabilities@1"
    assert SOLVER_READINESS_INTERFACE == "SolverReadiness@1"
    assert report.available
    assert report.network_mode is NetworkMode.OFFLINE
    assert report.module(MODULE).content_digest.startswith("module:sha256:")
    assert report.toolchain("fixture-prover").executable_digest.startswith("executable:sha256:")
    assert report.receipt_id.startswith("capability-inventory:sha256:")
    assert calls == ["fixture-prover"]
    assert report.to_dict()["probe_side_effects"] == "none"


@pytest.mark.parametrize(
    ("source", "overrides", "reason"),
    (
        (_module_source(marker="# TODO replace"), {}, "stub_todo_or_simulated_source"),
        (_module_source(initialized=False), {}, "capability_uninitialized_or_unattested"),
        (_module_source(reconstructed=False), {}, "reconstruction_unavailable_or_unattested"),
        (
            _module_source(),
            {"distribution_version": lambda distribution: "wrong"},
            "distribution_version_mismatch",
        ),
        (
            _module_source(),
            {"self_test_evidence": {"self-test:fixture-prover": True}},
            "self_test_missing_failed_or_unattested",
        ),
    ),
)
def test_bad_module_evidence_is_unavailable_and_cannot_select(
    tmp_path: Path, source: str, overrides: dict[str, object], reason: str
) -> None:
    report = _probe(tmp_path, source, **overrides).probe()
    capability = report.module(MODULE)
    assert capability.status is CapabilityStatus.UNAVAILABLE
    assert not capability.available
    assert not report.available
    assert reason in capability.reason_codes


def test_missing_symbols_and_missing_module_fail_closed_without_importing(tmp_path: Path) -> None:
    report = _probe(
        tmp_path, _module_source().replace("def reconstruct(): return 'local'", "")
    ).probe()
    assert "required_symbols_missing" in report.module(MODULE).reason_codes

    missing = _probe(tmp_path, _module_source(), find_spec=lambda name: None).probe()
    assert missing.module(MODULE).reason_codes == ("module_missing_or_non_file_origin",)


def test_toolchain_stub_missing_version_or_reconstruction_is_unavailable(tmp_path: Path) -> None:
    missing = _probe(tmp_path, _module_source(), executable_finder=lambda name: None).probe()
    assert "executable_missing" in missing.toolchain("fixture-prover").reason_codes

    wrong = _probe(
        tmp_path, _module_source(), executable_versions={"fixture-prover": "bad"}
    ).probe()
    assert "executable_version_mismatch" in wrong.toolchain("fixture-prover").reason_codes

    simulated = _probe(tmp_path, _module_source())
    (tmp_path / "fixture-prover").write_bytes(b"SIMULATED prover")
    report = simulated.probe()
    assert "stub_todo_or_simulated_executable" in report.toolchain("fixture-prover").reason_codes

    unproven = _probe(tmp_path, _module_source(), reconstruction_evidence={}).probe()
    assert (
        "reconstruction_unavailable_or_unattested"
        in unproven.toolchain("fixture-prover").reason_codes
    )


def test_plain_booleans_and_stale_content_receipts_never_authorize(
    tmp_path: Path,
) -> None:
    booleans = _probe(
        tmp_path,
        _module_source(),
        initialization_evidence={MODULE: True},
        self_test_evidence={MODULE: True, "self-test:fixture-prover": True},
        reconstruction_evidence={
            MODULE: True,
            "reconstruction:fixture-prover": True,
        },
    ).probe()
    assert not booleans.available
    assert "self_test_missing_failed_or_unattested" in booleans.module(MODULE).reason_codes

    stale = _probe(tmp_path, _module_source())
    (tmp_path / "fixture_logic.py").write_text(
        _module_source(marker="# changed bytes"), encoding="utf-8"
    )
    report = stale.probe()
    assert not report.available
    assert "capability_uninitialized_or_unattested" in report.module(MODULE).reason_codes


def test_networked_or_unspecified_mode_is_never_selectable(tmp_path: Path) -> None:
    for mode in (NetworkMode.NETWORKED, NetworkMode.UNSPECIFIED):
        report = _probe(tmp_path, _module_source(), network_mode=mode).probe()
        assert not report.available
        assert "network_mode_not_offline" in report.module(MODULE).reason_codes
        assert "network_mode_not_offline" in report.toolchain("fixture-prover").reason_codes
