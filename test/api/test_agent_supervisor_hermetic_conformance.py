"""DCR-090 hermetic cross-root fixture validation tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.hermetic_conformance import (
    HermeticConformanceDisposition,
    HermeticConformanceError,
    ImportedModuleOrigin,
    IndependentExpectedFact,
    McpProtocolObservation,
    validate_hermetic_conformance,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _module(root: Path, root_id: str) -> ModuleType:
    source = root / f"{root_id}_connector.py"
    source.write_text("IDENTITY = 'actual imported fixture'\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"dcr090_{root_id}", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation(root_id: str, *, fact_id: str = "catalog-stable") -> McpProtocolObservation:
    requests = {
        "initialize": _canonical({"method": "initialize", "params": {}}),
        "tools/list": _canonical({"method": "tools/list", "params": {}}),
        "tools/call": _canonical({"method": "tools/call", "params": {"name": "health"}}),
    }
    results = {step: _canonical({"result": {"step": step}}) for step in requests}
    errors = {step: _canonical({"error": None}) for step in requests}
    return McpProtocolObservation(
        root_id=root_id,
        schema_id="mcp/2025-03-26",
        profile_id="dcr090-fixture-profile",
        requests=requests,
        results=results,
        errors=errors,
        expected_fact_id=fact_id,
        observed_fact=_canonical({"tools": ["health"]}),
    )


def test_structural_imported_origins_are_not_projected_as_live_conformance(tmp_path: Path) -> None:
    origins = []
    observations = []
    for root_id in ("accelerate", "swissknife", "mcpplusplus"):
        root = tmp_path / root_id
        root.mkdir()
        origins.append(
            ImportedModuleOrigin.from_module(
                root_id=root_id, root=root, module=_module(root, root_id)
            )
        )
        observations.append(_observation(root_id))
    fact = IndependentExpectedFact.from_bytes(
        fact_id="catalog-stable",
        source=_canonical({"source": "reviewed-golden"}),
        value=_canonical({"tools": ["health"]}),
    )

    report = validate_hermetic_conformance(
        origins=origins, observations=observations, expected_facts=(fact,)
    )

    assert report.disposition is HermeticConformanceDisposition.INTEGRATION_PENDING
    assert report.reason_codes == ("structural_fixture_non_live",)
    assert report.report_cid
    assert report.to_dict()["model_call_count"] == 0
    assert report.to_dict()["network_call_count"] == 0
    assert report.to_dict()["structural_fixture"] is True
    assert report.to_dict()["live_conformance"] is False


def test_missing_real_swissknife_root_is_pending_not_green_skip(tmp_path: Path) -> None:
    root = tmp_path / "accelerate"
    root.mkdir()
    origin = ImportedModuleOrigin.from_module(
        root_id="accelerate", root=root, module=_module(root, "accelerate")
    )
    fact = IndependentExpectedFact.from_bytes(
        fact_id="catalog-stable",
        source=_canonical({"fixture": "golden"}),
        value=_canonical({"tools": ["health"]}),
    )

    report = validate_hermetic_conformance(
        origins=(origin,), observations=(), expected_facts=(fact,)
    )

    assert report.disposition is HermeticConformanceDisposition.INTEGRATION_PENDING
    assert "missing_real_root_swissknife" in report.reason_codes
    assert "protocol_observations_required" in report.reason_codes


def test_origin_outside_declared_root_and_echoed_expected_fact_fail(tmp_path: Path) -> None:
    inside = tmp_path / "accelerate"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    module = _module(outside, "outside")
    with pytest.raises(HermeticConformanceError, match="outside_declared_root"):
        ImportedModuleOrigin.from_module(root_id="accelerate", root=inside, module=module)

    requests = {
        "initialize": _canonical({"method": "initialize", "params": {}}),
        "tools/list": _canonical({"method": "tools/list", "params": {}}),
        "tools/call": _canonical(
            {"method": "tools/call", "params": {"detector": "catalog-stable"}}
        ),
    }
    with pytest.raises(HermeticConformanceError, match="echoes_expected_fact"):
        McpProtocolObservation(
            root_id="accelerate",
            schema_id="mcp/2025-03-26",
            profile_id="fixture",
            requests=requests,
            results={step: _canonical({"result": {}}) for step in requests},
            errors={step: _canonical({"error": None}) for step in requests},
            expected_fact_id="catalog-stable",
            observed_fact=_canonical({"tools": ["health"]}),
        )
