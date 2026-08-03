from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from tools.logic import certify_formal_verification_toolchains as certifier


REPO_ROOT = Path(__file__).resolve().parents[3]
SEALED_ROOT = Path("/opt/fvt-checked-vendor-test")
PNMR = ("positive", "negative", "mutation", "replay")


def _semantic_spec(lane_id: str) -> Mapping[str, Any]:
    return next(
        spec
        for spec in certifier.SEMANTIC_CERTIFIER_SPECS
        if spec["lane_id"] == lane_id
    )


def _semantic_module(lane_id: str) -> SimpleNamespace:
    identities = {
        "runtime_mtl": (
            "runtime-mtl-semantic-certification/v1",
            "FVT-G103",
            "FVT-039",
        ),
        "datalog_secpal": (
            "authorization-semantic-certification/v1",
            "FVT-G102",
            "FVT-038",
        ),
    }
    schema, goal, task = identities[lane_id]
    return SimpleNamespace(
        SCHEMA_VERSION=schema,
        GOAL_ID=goal,
        TASK_ID=task,
    )


def _checks(tool_id: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "check_id": f"{tool_id}.reference.{index}",
            "tool_id": tool_id,
            "kind": PNMR[index] if index < len(PNMR) else "positive",
            "status": "passed",
            "expected": "passed",
            "observed": "passed",
            "detail": "closed focused reference check",
        }
        for index in range(count)
    ]


def _runtime_reference_receipt() -> dict[str, Any]:
    manifest_relative = Path(
        "test/fixtures/formal_verification/toolchains/runtime_mtl/manifest.json"
    )
    manifest_path = REPO_ROOT / manifest_relative
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = len(manifest["case_recipes"])
    payload: dict[str, Any] = {
        "schema_version": "runtime-mtl-semantic-certification/v1",
        "interface": "RuntimeMTLSemanticCertification@1",
        "goal_id": "FVT-G103",
        "task_id": "FVT-039",
        "certified": True,
        "production_certified": True,
        "checks": _checks("runtime-mtl", count),
        "block_reasons": [],
        "policy": {
            "no_install": True,
            "no_download": True,
            "no_network": True,
        },
        "manifest": {
            "schema_version": manifest["schema_version"],
            "interface": manifest["interface"],
            "case_count": count,
            "path": str(manifest_path),
        },
        "source_tree": {
            "files": [
                {
                    "path": manifest_relative.as_posix(),
                    "exists": True,
                    "content_sha256": certifier._bare_file_digest(
                        manifest_path
                    ),
                }
            ]
        },
        "summary": {
            "checks_passed": count,
            "checks_skipped": 0,
            "checks_failed": 0,
            "checks_total": count,
        },
    }
    payload["certificate_digest_sha256"] = certifier.content_digest(payload)
    return payload


def _authorization_reference_receipt() -> dict[str, Any]:
    engines = [
        {
            "engine_id": tool_id,
            "certified": True,
            "checks": _checks(tool_id, 24),
            "block_reasons": [],
        }
        for tool_id in ("datalog-authorization", "secpal-authorization")
    ]
    payload: dict[str, Any] = {
        "schema_version": "authorization-semantic-certification/v1",
        "interface": "AuthorizationSemanticCertification@1",
        "goal_id": "FVT-G102",
        "task_id": "FVT-038",
        "certified": True,
        "engines": engines,
        "policy": {"in_process_only": True},
        "summary": {
            "checks_passed": 48,
            "checks_total": 48,
            "block_reasons": [],
        },
    }
    payload["certificate_digest_sha256"] = certifier.content_digest(payload)
    return payload


def _checked_receipt(lane_id: str) -> dict[str, Any]:
    spec = certifier.CHECKED_VENDOR_FANIN_SPECS[lane_id]
    return json.loads(
        (REPO_ROOT / spec["checked_receipt_relative"]).read_text(
            encoding="utf-8"
        )
    )


def _live_vendor_certificate(
    lane_id: str,
    *,
    nested_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = certifier.CHECKED_VENDOR_FANIN_SPECS[lane_id]
    count = int(spec["expected_vendor_checks"])
    checks = [
        {
            "check_id": f"{spec['vendor_tool_id']}.vendor.{index}",
            "engine_id": spec["vendor_tool_id"],
            "kind": PNMR[index] if index < len(PNMR) else "positive",
            "status": "passed",
            "expected": "passed",
            "observed": "passed",
            "detail": "fresh vendor check",
        }
        for index in range(count)
    ]
    vendor: dict[str, Any] = {
        "engine_id": spec["vendor_tool_id"],
        "certified": True,
        "usable": True,
        "block_reasons": [],
        "executable": "/opt/fvt-checked-vendor-test/bin/vendor",
        "checks": checks,
        "case_results": [{"case_id": "fresh.vendor.case"}],
        "is_vendor_build": True,
    }
    payload: dict[str, Any] = {
        "schema_version": spec["live_schema"],
        "interface": spec["interface"],
        "goal_id": spec["goal_id"],
        "task_id": spec["task_id"],
        "repair_task_id": spec["repair_task_id"],
        "certified": True,
        "objective_validation_repair": True,
        "install": None,
        "summary": {
            spec["summary_certified_key"]: True,
            "checks_passed": count,
            "checks_total": count,
            "block_reasons": [],
        },
        spec["vendor_section"]: vendor,
    }
    if lane_id == "runtime_mtl":
        payload.update(
            {
                "forbids_theorem_authority": True,
                "forbids_global_correctness_claim": True,
            }
        )
        vendor.update(
            {
                "is_hermetic_parity_engine": False,
                "no_python_reference_dispatch": True,
                "finite_trace_authority_only": True,
            }
        )
    else:
        payload.update(
            {
                "forbids_authorization_authority_on_shadows": True,
                "forbids_theorem_authority": True,
                "secpal_platform_exception": {
                    "exception": True,
                    "narrow_scope": True,
                    "classification": "unsupported_here",
                    "installed": False,
                    "complete": False,
                    "authoritative": False,
                    "production_certified": False,
                },
            }
        )
        vendor["is_hermetic_shadow"] = False
    payload["certificate_digest_sha256"] = (
        certifier._checked_vendor_outer_digest(
            payload,
            repo_root=REPO_ROOT,
            uses_public_projection=bool(
                spec["outer_digest_uses_public_projection"]
            ),
        )
    )
    payload["install_receipt"] = dict(
        nested_receipt or _checked_receipt(lane_id)
    )
    return payload


def _install_fake_vendor(
    monkeypatch: pytest.MonkeyPatch,
    lane_id: str,
    live: Mapping[str, Any],
) -> list[dict[str, Any]]:
    spec = certifier.CHECKED_VENDOR_FANIN_SPECS[lane_id]
    calls: list[dict[str, Any]] = []

    def fake_certifier(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(dict(kwargs))
        return live

    monkeypatch.setattr(
        certifier,
        "_load_module_from_path",
        lambda *_args, **_kwargs: SimpleNamespace(
            **{str(spec["callable_name"]): fake_certifier}
        ),
    )
    monkeypatch.setattr(
        certifier,
        "_checked_vendor_sealed_root_failures",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        certifier,
        "_runtime_mtl_sealed_path_failures",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        certifier,
        "observed_platform_id",
        lambda: "linux-aarch64",
    )
    return calls


@pytest.mark.parametrize(
    ("lane_id", "expected_targets"),
    [
        ("runtime_mtl", {"runtime-mtl"}),
        (
            "datalog_secpal",
            {"datalog-authorization", "secpal-authorization"},
        ),
    ],
)
def test_checked_vendor_fanin_unlocks_only_in_process_targets(
    monkeypatch: pytest.MonkeyPatch,
    lane_id: str,
    expected_targets: set[str],
) -> None:
    live = _live_vendor_certificate(lane_id)
    calls = _install_fake_vendor(monkeypatch, lane_id, live)
    receipt = (
        _runtime_reference_receipt()
        if lane_id == "runtime_mtl"
        else _authorization_reference_receipt()
    )
    spec = _semantic_spec(lane_id)
    fanin = certifier._build_checked_vendor_fanin(
        repo_root=REPO_ROOT,
        sealed_root=SEALED_ROOT,
        semantic_spec=spec,
        semantic_module=_semantic_module(lane_id),
        reference_receipt=receipt,
    )

    assert fanin["vendor_valid"] is True
    assert fanin["complete"] is True
    assert set(fanin["eligible_tool_ids"]) == expected_targets
    assert not expected_targets & {"runtime-mtl-external", "souffle", "secpal"}
    assert len(calls) == 1
    assert calls[0]["skip_install"] is True
    assert calls[0]["force_install"] is False
    assert calls[0]["lock_path"] == REPO_ROOT / certifier.DEFAULT_LOCK_RELATIVE
    assert calls[0]["write_receipt_path"] is None
    if lane_id == "runtime_mtl":
        assert calls[0]["install_root"] == SEALED_ROOT
    else:
        assert calls[0]["install_root"] == SEALED_ROOT / "souffle-vendor"
        assert calls[0]["dependency_prefix"] == (
            SEALED_ROOT
            / "build-dependencies/souffle/ubuntu-noble-arm64/root"
        )
        assert calls[0]["platform_id"] == "linux-aarch64"

    bound = certifier._bind_checked_vendor_fanin_to_receipt(
        receipt,
        semantic_spec=spec,
        fanin=fanin,
    )
    assert certifier._validate_semantic_receipt_integrity(
        bound,
        spec=spec,
        module=_semantic_module(lane_id),
    )["valid"]
    for tool_id in expected_targets:
        identity = certifier._semantic_tool_identity(
            tool_id,
            bound,
            selector=str(spec["selector"]),
            repo_root=REPO_ROOT,
        )
        assert certifier._validate_artifact_identities(
            identity["artifacts"],
            repo_root=REPO_ROOT,
        )["has_production_binding"]
    assert (
        certifier._recorded_checked_vendor_fanin_eligibility(
            repo_root=REPO_ROOT,
            semantic_spec=spec,
            result_fanin=fanin,
            receipt_fanin=bound["checked_vendor_fanin"],
        )
        == expected_targets
    )


def test_exact_nested_install_receipt_mismatch_blocks_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatched = _checked_receipt("runtime_mtl")
    mismatched["task_id"] = "tampered"
    live = _live_vendor_certificate(
        "runtime_mtl",
        nested_receipt=mismatched,
    )
    _install_fake_vendor(monkeypatch, "runtime_mtl", live)
    fanin = certifier._build_checked_vendor_fanin(
        repo_root=REPO_ROOT,
        sealed_root=SEALED_ROOT,
        semantic_spec=_semantic_spec("runtime_mtl"),
        semantic_module=_semantic_module("runtime_mtl"),
        reference_receipt=_runtime_reference_receipt(),
    )

    assert fanin["vendor_valid"] is False
    assert fanin["eligible_tool_ids"] == []
    assert "live_vendor_nested_install_receipt_mismatch" in fanin["failures"]


def test_failed_vendor_check_blocks_all_authorization_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _live_vendor_certificate("datalog_secpal")
    live["souffle"]["checks"][7]["status"] = "failed"
    _install_fake_vendor(monkeypatch, "datalog_secpal", live)
    fanin = certifier._build_checked_vendor_fanin(
        repo_root=REPO_ROOT,
        sealed_root=SEALED_ROOT,
        semantic_spec=_semantic_spec("datalog_secpal"),
        semantic_module=_semantic_module("datalog_secpal"),
        reference_receipt=_authorization_reference_receipt(),
    )

    assert fanin["vendor_valid"] is False
    assert fanin["eligible_tool_ids"] == []
    assert "live_vendor_full_check_set_invalid" in fanin["failures"]


def test_reference_failure_blocks_only_its_authorization_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _live_vendor_certificate("datalog_secpal")
    _install_fake_vendor(monkeypatch, "datalog_secpal", live)
    receipt = _authorization_reference_receipt()
    receipt["engines"][0]["checks"][3]["status"] = "failed"
    receipt["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in receipt.items()
            if key != "certificate_digest_sha256"
        }
    )
    fanin = certifier._build_checked_vendor_fanin(
        repo_root=REPO_ROOT,
        sealed_root=SEALED_ROOT,
        semantic_spec=_semantic_spec("datalog_secpal"),
        semantic_module=_semantic_module("datalog_secpal"),
        reference_receipt=receipt,
    )

    assert fanin["vendor_valid"] is True
    assert fanin["complete"] is False
    assert fanin["eligible_tool_ids"] == ["secpal-authorization"]
    assert fanin["per_tool_failures"]["datalog-authorization"]
    assert fanin["per_tool_failures"]["secpal-authorization"] == []


def test_missing_sealed_root_never_invokes_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoked = False

    def fail_if_loaded(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal invoked
        invoked = True
        raise AssertionError("vendor certifier must not be loaded")

    monkeypatch.setattr(certifier, "_load_module_from_path", fail_if_loaded)
    fanin = certifier._build_checked_vendor_fanin(
        repo_root=REPO_ROOT,
        sealed_root=None,
        semantic_spec=_semantic_spec("runtime_mtl"),
        semantic_module=_semantic_module("runtime_mtl"),
        reference_receipt=_runtime_reference_receipt(),
    )

    assert invoked is False
    assert fanin["vendor_valid"] is False
    assert fanin["eligible_tool_ids"] == []
    assert "sealed_vendor_root_unavailable" in fanin["failures"]


def test_forged_recorded_eligible_population_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _live_vendor_certificate("runtime_mtl")
    _install_fake_vendor(monkeypatch, "runtime_mtl", live)
    spec = _semantic_spec("runtime_mtl")
    fanin = certifier._build_checked_vendor_fanin(
        repo_root=REPO_ROOT,
        sealed_root=SEALED_ROOT,
        semantic_spec=spec,
        semantic_module=_semantic_module("runtime_mtl"),
        reference_receipt=_runtime_reference_receipt(),
    )
    forged = json.loads(json.dumps(fanin))
    forged["eligible_tool_ids"] = ["runtime-mtl-external"]
    forged["digest_sha256"] = certifier.content_digest(
        {key: value for key, value in forged.items() if key != "digest_sha256"}
    )

    assert not certifier._recorded_checked_vendor_fanin_eligibility(
        repo_root=REPO_ROOT,
        semantic_spec=spec,
        result_fanin=forged,
        receipt_fanin=forged,
    )
