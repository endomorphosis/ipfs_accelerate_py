"""DCR-090: hermetic cross-repository contract conformance fixtures.

Acceptance:
* Monorepo structural fixtures stay live_conformance=false without real servers.
* Mocks cannot echo requested capabilities or expected detector values.
* Incompatible implementations produce deterministic failing counterexamples.
* Standalone-clone skips cannot flip monorepo green.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.hermetic_conformance import (
    DEFAULT_HERMETIC_CONFORMANCE_PATH,
    HERMETIC_CONFORMANCE_INTERFACE,
    REQUIRED_MONOREPO_ROOTS,
    ConformanceMode,
    CounterexampleKind,
    HermeticConformanceError,
    HermeticConformanceReport,
    build_contract_graph_fixture,
    materialize_hermetic_conformance,
    validate_hermetic_conformance,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "external" / "ipfs_accelerate").is_dir() and (
            candidate / "swissknife"
        ).is_dir():
            return candidate
    return here.parents[4]


def test_interface_constants() -> None:
    assert HERMETIC_CONFORMANCE_INTERFACE == "HermeticConformance@1"
    assert HermeticConformanceReport.INTERFACE == HERMETIC_CONFORMANCE_INTERFACE
    assert "Mcp-Plus-Plus" in REQUIRED_MONOREPO_ROOTS
    assert "swissknife" in REQUIRED_MONOREPO_ROOTS


def test_monorepo_structural_ok_but_not_live() -> None:
    report = validate_hermetic_conformance(
        repo_root=_repo_root(),
        claim_live_conformance=False,
        real_server_available=False,
    )
    assert report.mode is ConformanceMode.MONOREPO
    assert report.structural_ok is True
    assert report.live_conformance is False
    assert report.runtime_model_calls == 0
    assert "live_conformance_false" in report.reason_codes
    payload = report.to_dict()
    assert payload["live_conformance"] is False
    assert payload["runtime_model_calls"] == 0
    assert payload["content_id"].startswith("sha256:")


def test_forged_live_green_without_server_is_rejected() -> None:
    report = validate_hermetic_conformance(
        repo_root=_repo_root(),
        claim_live_conformance=True,
        real_connector_available=True,
        real_server_available=False,
    )
    assert report.live_conformance is False
    kinds = {item["kind"] for item in report.counterexamples}
    assert CounterexampleKind.FORGED_LIVE_GREEN.value in kinds


def test_mock_echo_capabilities_produce_counterexample() -> None:
    requested = ["tools/list", "tools/call", "initialize"]
    report = validate_hermetic_conformance(
        repo_root=_repo_root(),
        requested_capabilities=requested,
        observed_implementations=[
            {
                "implementation_id": "mock:echo-server",
                "capabilities": list(requested),
            }
        ],
        real_server_available=False,
    )
    assert report.live_conformance is False
    echo = [
        item
        for item in report.counterexamples
        if item.get("kind") == CounterexampleKind.MOCK_ECHO.value
    ]
    assert echo
    assert any(item.get("reason") == "mock_echoed_requested_capabilities" for item in echo)


def test_mock_echo_detector_value_rejected() -> None:
    report = validate_hermetic_conformance(
        repo_root=_repo_root(),
        observed_implementations=[
            {
                "implementation_id": "mock:detector",
                "expected_detector_value": "profile-e",
                "detector_value": "profile-e",
            }
        ],
        real_server_available=False,
    )
    assert any(
        item.get("reason") == "mock_echoed_expected_detector_value"
        for item in report.counterexamples
    )


def test_incompatible_profile_counterexample() -> None:
    report = validate_hermetic_conformance(
        repo_root=_repo_root(),
        observed_implementations=[
            {
                "implementation_id": "impl:real-ish",
                "profile": "mcp++/experimental",
                "admitted_profiles": ["mcp++/default"],
            }
        ],
        real_server_available=False,
    )
    assert any(
        item.get("kind") == CounterexampleKind.INCOMPATIBLE_PROFILE.value
        for item in report.counterexamples
    )


def test_standalone_clone_cannot_claim_live() -> None:
    with pytest.raises(HermeticConformanceError):
        HermeticConformanceReport(
            mode=ConformanceMode.STANDALONE_CLONE,
            live_conformance=True,
            structural_ok=False,
            roots_present=(),
            roots_missing=REQUIRED_MONOREPO_ROOTS,
            module_origins={},
            counterexamples=(),
            reason_codes=("standalone",),
        )


def test_missing_root_counterexample(tmp_path: Path) -> None:
    # Empty tree → standalone, missing roots.
    report = validate_hermetic_conformance(
        repo_root=tmp_path,
        claim_live_conformance=False,
    )
    assert report.mode is ConformanceMode.STANDALONE_CLONE
    assert report.structural_ok is False
    assert report.live_conformance is False
    assert any(
        item.get("kind") == CounterexampleKind.MISSING_ROOT.value
        for item in report.counterexamples
    )


def test_contract_graph_fixture_is_hermetic() -> None:
    graph = build_contract_graph_fixture(snapshot_id="snap:test")
    assert graph["live_conformance"] is False
    assert graph["runtime_model_calls"] == 0
    assert graph["interface"] == "SwissKnifeMcpContractGraph@1"
    assert len(graph["nodes"]) >= 4
    assert len(graph["edges"]) >= 3
    assert graph["graph_cid"].startswith("sha256:")


def test_materialize_hermetic_conformance(tmp_path: Path) -> None:
    dest = tmp_path / "hermetic-conformance.json"
    payload = materialize_hermetic_conformance(
        repo_root=_repo_root(),
        destination=dest,
    )
    assert dest.is_file()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["interface"] == HERMETIC_CONFORMANCE_INTERFACE
    assert on_disk["live_conformance"] is False
    assert on_disk["runtime_model_calls"] == 0
    assert on_disk["report"]["structural_ok"] is True
    assert "contract_graph" in on_disk
    assert payload["live_conformance"] is False


def test_default_artifact_path_constant() -> None:
    assert DEFAULT_HERMETIC_CONFORMANCE_PATH.endswith("hermetic-conformance.json")
