from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from tools.logic import build_formal_verification_tactician_receipt as builder
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


def _compact_semantic_result(
    *,
    lane_id: str,
    receipt: Mapping[str, Any],
    fanin: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _semantic_spec(lane_id)
    per_tool: dict[str, Any] = {}
    for tool_id in spec["tool_ids"]:
        certified, raw_checks, block_reasons = (
            certifier._tool_certified_from_semantic_receipt(
                tool_id,
                receipt,
                certified_key=str(spec["certified_key"]),
                selector=str(spec["selector"]),
            )
        )
        normalized = certifier._normalize_semantic_checks(
            tool_id,
            raw_checks,
        )
        identity = certifier._semantic_tool_identity(
            tool_id,
            receipt,
            selector=str(spec["selector"]),
            repo_root=REPO_ROOT,
        )
        module_artifact = {
            "kind": "semantic_certifier_module",
            "path": Path(spec["module_relative"]).as_posix(),
            "sha256": certifier.file_digest(
                REPO_ROOT / Path(spec["module_relative"])
            ),
            "artifact_class": "repository_source",
        }
        identity["artifacts"].append(module_artifact)
        full_tool = {
            "certified": certified,
            "block_reasons": list(block_reasons),
            "check_kinds_present": sorted(
                {
                    str(check.get("kind"))
                    for check in raw_checks
                    if isinstance(check, Mapping)
                }
            ),
            "checks_retained_without_kind_collapse": True,
            "checks_passed": sum(
                check.status == "passed" for check in normalized
            ),
            "checks_total": len(normalized),
            "checks": [check.to_dict() for check in normalized],
            "check_set_digest_sha256": certifier.content_digest(
                [check.to_dict() for check in normalized]
            ),
            "identity": identity,
            "artifact_validation": certifier._validate_artifact_identities(
                identity["artifacts"],
                repo_root=REPO_ROOT,
            ),
            "handler_key": (
                f"{spec.get('property_lane_id') or lane_id}::{tool_id}"
            ),
        }
        per_tool[tool_id] = certifier._compact_semantic_tool_projection(
            full_tool
        )
    return {
        "lane_id": lane_id,
        "status": "ran",
        "receipt": dict(receipt),
        "checked_vendor_fanin": dict(fanin),
        "production_elevation_allowed": True,
        "evidence_class": certifier.CHECKED_VENDOR_FANIN_SPECS[lane_id][
            "evidence_class"
        ],
        "per_tool": per_tool,
    }


def _valid_semantic_result(
    monkeypatch: pytest.MonkeyPatch,
    lane_id: str,
) -> dict[str, Any]:
    original_module_loader = certifier._load_module_from_path
    live = _live_vendor_certificate(lane_id)
    _install_fake_vendor(monkeypatch, lane_id, live)
    reference = (
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
        reference_receipt=reference,
    )
    receipt = certifier._bind_checked_vendor_fanin_to_receipt(
        reference,
        semantic_spec=spec,
        fanin=fanin,
    )
    monkeypatch.setattr(
        certifier,
        "_load_module_from_path",
        original_module_loader,
    )
    return _compact_semantic_result(
        lane_id=lane_id,
        receipt=receipt,
        fanin=fanin,
    )


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


@pytest.mark.parametrize("lane_id", ["runtime_mtl", "datalog_secpal"])
def test_builder_freshly_replays_vendor_join_and_reference_pnmr(
    monkeypatch: pytest.MonkeyPatch,
    lane_id: str,
) -> None:
    live = _live_vendor_certificate(lane_id)
    _install_fake_vendor(monkeypatch, lane_id, live)
    reference = (
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
        reference_receipt=reference,
    )
    receipt = certifier._bind_checked_vendor_fanin_to_receipt(
        reference,
        semantic_spec=spec,
        fanin=fanin,
    )
    semantic_result = _compact_semantic_result(
        lane_id=lane_id,
        receipt=receipt,
        fanin=fanin,
    )
    monkeypatch.setattr(
        certifier,
        "_runtime_mtl_managed_prebuilt_binding",
        lambda *_args, **_kwargs: {
            "public": {"authenticated": True},
            "invocation": {"sealed_root": str(SEALED_ROOT)},
        },
    )
    monkeypatch.setattr(
        certifier,
        "_build_checked_vendor_fanin",
        lambda **_kwargs: dict(fanin),
    )
    monkeypatch.setattr(
        certifier,
        "_load_module_from_path",
        lambda *_args, **_kwargs: _semantic_module(lane_id),
    )

    policy = builder._audited_semantic_elevation_policy(
        certifier=certifier,
        repo_root=REPO_ROOT,
        spec=spec,
        semantic_result=semantic_result,
    )

    assert policy["valid"] is True
    assert policy["fanin_satisfied"] is True
    assert set(policy["eligible_tool_ids"]) == set(spec["tool_ids"])
    assert set(policy["production_allowed_tool_ids"]) == set(
        spec["tool_ids"]
    )
    assert all(
        audit["valid"] for audit in policy["reference_audits"].values()
    )


def test_builder_rejects_forged_central_vendor_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane_id = "runtime_mtl"
    live = _live_vendor_certificate(lane_id)
    _install_fake_vendor(monkeypatch, lane_id, live)
    spec = _semantic_spec(lane_id)
    reference = _runtime_reference_receipt()
    fanin = certifier._build_checked_vendor_fanin(
        repo_root=REPO_ROOT,
        sealed_root=SEALED_ROOT,
        semantic_spec=spec,
        semantic_module=_semantic_module(lane_id),
        reference_receipt=reference,
    )
    receipt = certifier._bind_checked_vendor_fanin_to_receipt(
        reference,
        semantic_spec=spec,
        fanin=fanin,
    )
    semantic_result = _compact_semantic_result(
        lane_id=lane_id,
        receipt=receipt,
        fanin=fanin,
    )
    forged = json.loads(json.dumps(fanin))
    forged["eligible_tool_ids"] = ["runtime-mtl-external"]
    forged["digest_sha256"] = certifier.content_digest(
        {key: value for key, value in forged.items() if key != "digest_sha256"}
    )
    semantic_result["checked_vendor_fanin"] = forged
    monkeypatch.setattr(
        certifier,
        "_runtime_mtl_managed_prebuilt_binding",
        lambda *_args, **_kwargs: {
            "public": {"authenticated": True},
            "invocation": {"sealed_root": str(SEALED_ROOT)},
        },
    )
    monkeypatch.setattr(
        certifier,
        "_build_checked_vendor_fanin",
        lambda **_kwargs: dict(fanin),
    )
    monkeypatch.setattr(
        certifier,
        "_load_module_from_path",
        lambda *_args, **_kwargs: _semantic_module(lane_id),
    )

    policy = builder._audited_semantic_elevation_policy(
        certifier=certifier,
        repo_root=REPO_ROOT,
        spec=spec,
        semantic_result=semantic_result,
    )

    assert policy["valid"] is False
    assert policy["eligible_tool_ids"] == []
    assert "checked_vendor_fanin_recording_disagrees_with_receipt" in (
        policy["failures"]
    )


@pytest.mark.parametrize(
    ("lane_id", "external_tool_id", "role", "certifying_role"),
    [
        (
            "runtime_mtl",
            "runtime-mtl-external",
            "authority",
            True,
        ),
        ("datalog_secpal", "souffle", "shadow", False),
    ],
)
def test_checked_vendor_projection_satisfies_readiness_without_authority(
    monkeypatch: pytest.MonkeyPatch,
    lane_id: str,
    external_tool_id: str,
    role: str,
    certifying_role: bool,
) -> None:
    semantic_result = _valid_semantic_result(monkeypatch, lane_id)
    projection = (
        certifier.build_checked_vendor_capability_readiness_projection(
            repo_root=REPO_ROOT,
            semantic_results=[semantic_result],
        )
    )
    entry = projection["tools"][external_tool_id]
    assert entry["ready"] is True
    assert entry["vendor_checks_passed"] == (
        37 if lane_id == "runtime_mtl" else 32
    )
    assert entry["production_certified"] is False
    assert entry["production_elevation_allowed"] is False
    assert entry["authority_granted"] is False
    assert entry["authority_requirement_satisfied"] is False
    assert entry["grants_theorem_authority"] is False
    assert entry["grants_global_correctness"] is False
    assert entry["grants_authorization_decision_authority"] is False
    assert entry["readiness_scope"] == (
        "differential_witness_only"
        if lane_id == "runtime_mtl"
        else "shadow_checker_only"
    )
    assert entry["declared_authority_role"] == role
    assert entry["declared_authority_ceiling"] == (
        "finite_trace" if certifying_role else "none"
    )
    assert entry[
        "declared_role_can_satisfy_certified_authority"
    ] is certifying_role

    lock = certifier.load_lock(
        REPO_ROOT / certifier.DEFAULT_LOCK_RELATIVE
    )
    all_tools = certifier.lock_tools_by_id(lock)
    selected_ids = [external_tool_id]
    if external_tool_id == "souffle":
        selected_ids.append("secpal")
    selected_tools = {
        tool_id: all_tools[tool_id] for tool_id in selected_ids
    }
    tool_certs = {
        external_tool_id: certifier.ToolCertification(
            tool_id=external_tool_id,
            production_certified=False,
        )
    }
    roles = {
        "tools": {
            external_tool_id: {
                "role": role,
                "authority_ceiling": (
                    "finite_trace" if certifying_role else "none"
                ),
                "can_satisfy_certified_authority": certifying_role,
            },
            "secpal": {
                "role": "shadow",
                "authority_ceiling": "none",
                "can_satisfy_certified_authority": False,
            },
        }
    }
    readiness = certifier.build_managed_deployment_readiness(
        lock=lock,
        tools_index=selected_tools,
        tool_certs=tool_certs,
        authority_roles=roles,
        repo_root=REPO_ROOT,
        semantic_results=[semantic_result],
        checked_vendor_capability_readiness=projection,
    )

    assert readiness["ready"] is True
    assert readiness["all_blockers"] == []
    assert readiness[
        "checked_vendor_capability_readiness_binding_valid"
    ] is True
    assert readiness[
        "ready_via_checked_vendor_capability_tool_ids"
    ] == [external_tool_id]
    assert tool_certs[external_tool_id].production_certified is False
    if external_tool_id == "souffle":
        assert [row["tool_id"] for row in readiness["platform_exceptions"]] == [
            "secpal"
        ]
        assert readiness["platform_exceptions"][0]["complete"] is False


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("authority_granted", True),
        ("authority_requirement_satisfied", True),
        ("readiness_scope", "finite_trace_authority"),
        ("declared_authority_ceiling", "none"),
    ],
)
def test_rehashed_forged_vendor_readiness_projection_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    forged_value: Any,
) -> None:
    semantic_result = _valid_semantic_result(monkeypatch, "runtime_mtl")
    projection = (
        certifier.build_checked_vendor_capability_readiness_projection(
            repo_root=REPO_ROOT,
            semantic_results=[semantic_result],
        )
    )
    forged = json.loads(json.dumps(projection))
    entry = forged["tools"]["runtime-mtl-external"]
    entry[field_name] = forged_value
    entry["digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in entry.items()
            if key != "digest_sha256"
        }
    )
    forged["digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged.items()
            if key != "digest_sha256"
        }
    )
    lock = certifier.load_lock(
        REPO_ROOT / certifier.DEFAULT_LOCK_RELATIVE
    )
    tool_id = "runtime-mtl-external"
    readiness = certifier.build_managed_deployment_readiness(
        lock=lock,
        tools_index={tool_id: certifier.lock_tools_by_id(lock)[tool_id]},
        tool_certs={
            tool_id: certifier.ToolCertification(tool_id=tool_id)
        },
        authority_roles={
            "tools": {
                tool_id: {
                    "role": "authority",
                    "authority_ceiling": "finite_trace",
                    "can_satisfy_certified_authority": True,
                }
            }
        },
        repo_root=REPO_ROOT,
        semantic_results=[semantic_result],
        checked_vendor_capability_readiness=forged,
    )

    assert readiness[
        "checked_vendor_capability_readiness_binding_valid"
    ] is False
    assert readiness["ready"] is False
    blocker = readiness["capability_blockers"][0]
    assert blocker["tool_id"] == tool_id
    assert "semantic_evidence_below_authority_ceiling" in blocker["reasons"]
    assert (
        "supported_managed_installation_missing_or_shim_only"
        in blocker["reasons"]
    )


def test_builder_rederives_and_rejects_forged_vendor_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_results = [
        _valid_semantic_result(monkeypatch, lane_id)
        for lane_id in ("runtime_mtl", "datalog_secpal")
    ]
    projection = (
        certifier.build_checked_vendor_capability_readiness_projection(
            repo_root=REPO_ROOT,
            semantic_results=semantic_results,
        )
    )
    semantic_audit = {
        "lanes": {
            lane_id: {
                "valid": True,
                "elevation_policy": {
                    "valid": True,
                    "vendor_claimed": True,
                    "fanin_satisfied": True,
                    "eligible_tool_ids": list(
                        certifier.CHECKED_VENDOR_FANIN_SPECS[
                            lane_id
                        ]["expected_reference_checks"]
                    ),
                    "sealed_root_authenticated": True,
                },
            }
            for lane_id in ("runtime_mtl", "datalog_secpal")
        }
    }
    tools = {
        tool_id: {"production_certified": False}
        for tool_id in ("runtime-mtl-external", "souffle", "secpal")
    }
    managed = {
        "checked_vendor_capability_readiness": projection,
        "checked_vendor_capability_readiness_binding_valid": True,
        "ready_via_checked_vendor_capability_tool_ids": [
            "runtime-mtl-external",
            "souffle",
        ],
    }
    audit = builder._audit_checked_vendor_capability_readiness(
        certifier=certifier,
        repo_root=REPO_ROOT,
        managed=managed,
        tools=tools,
        semantic_results=semantic_results,
        semantic_audit=semantic_audit,
    )
    assert audit["valid"] is True
    assert all(
        row["fresh_vendor_fanin_replayed"]
        for row in audit["lane_audits"].values()
    )

    forged = json.loads(json.dumps(managed))
    forged_entry = forged["checked_vendor_capability_readiness"][
        "tools"
    ]["souffle"]
    forged_entry["semantic_certification_ready"] = False
    forged_entry["digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged_entry.items()
            if key != "digest_sha256"
        }
    )
    forged_projection = forged["checked_vendor_capability_readiness"]
    forged_projection["digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged_projection.items()
            if key != "digest_sha256"
        }
    )
    rejected = builder._audit_checked_vendor_capability_readiness(
        certifier=certifier,
        repo_root=REPO_ROOT,
        managed=forged,
        tools=tools,
        semantic_results=semantic_results,
        semantic_audit=semantic_audit,
    )
    assert rejected["valid"] is False
    assert (
        "checked_vendor_capability_readiness_projection_mismatch"
        in rejected["failures"]
    )

    independently_forged = json.loads(json.dumps(projection))
    forged_runtime = independently_forged["tools"][
        "runtime-mtl-external"
    ]
    forged_runtime["authority_requirement_satisfied"] = True
    forged_runtime["declared_authority_ceiling"] = "none"
    forged_runtime["digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged_runtime.items()
            if key != "digest_sha256"
        }
    )
    independently_forged["digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in independently_forged.items()
            if key != "digest_sha256"
        }
    )
    independently_claimed = json.loads(json.dumps(managed))
    independently_claimed[
        "checked_vendor_capability_readiness"
    ] = independently_forged
    monkeypatch.setattr(
        certifier,
        "build_checked_vendor_capability_readiness_projection",
        lambda **_kwargs: independently_forged,
    )
    independently_rejected = (
        builder._audit_checked_vendor_capability_readiness(
            certifier=certifier,
            repo_root=REPO_ROOT,
            managed=independently_claimed,
            tools=tools,
            semantic_results=semantic_results,
            semantic_audit=semantic_audit,
        )
    )
    assert independently_rejected["valid"] is False
    assert independently_rejected["projection_matches"] is True
    runtime_audit = independently_rejected["lane_audits"][
        "runtime_mtl"
    ]
    assert runtime_audit["authority_flags_valid"] is False
