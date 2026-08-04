from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tools.logic import build_formal_verification_tactician_receipt as builder
from tools.logic import certify_formal_verification_toolchains as certifier


REPO_ROOT = Path(__file__).resolve().parents[3]
SEALED_ROOT = Path(
    "/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers"
)
TARGETS = ("hyperltl", "autohyper", "mchyper")


def _semantic_spec() -> dict[str, object]:
    return next(
        spec
        for spec in certifier.SEMANTIC_CERTIFIER_SPECS
        if spec["lane_id"] == "hyperltl"
    )


def _semantic_module():
    spec = _semantic_spec()
    return certifier._load_module_from_path(
        REPO_ROOT / Path(spec["module_relative"]),
        "test_checked_hyper_vendor_semantic_module",
    )


@pytest.fixture(scope="module")
def live_vendor_certificate() -> dict[str, object]:
    if not SEALED_ROOT.is_dir():
        pytest.skip("sealed Hyper vendor root is not available")
    module = _semantic_module()
    observed = module.certify_hyperproperty_vendor_toolchains(
        install_root=SEALED_ROOT,
        engines=TARGETS,
        force_install=False,
        skip_install=True,
        platform_id=certifier.observed_platform_id(),
        repo_root=REPO_ROOT,
        lock_path=REPO_ROOT / certifier.DEFAULT_LOCK_RELATIVE,
        dependency_roots=None,
        write_receipt_path=None,
    )
    assert observed["certified"] is True
    assert observed["summary"]["checks_passed"] == 66
    assert observed["summary"]["checks_total"] == 66
    return observed


def _vendor_loader(
    live_vendor_certificate: dict[str, object],
):
    original = certifier._load_module_from_path
    fake_vendor = SimpleNamespace(
        certify_hyperproperty_vendor_toolchains=lambda **_kwargs: copy.deepcopy(
            live_vendor_certificate
        )
    )

    def load(path: Path, module_name: str):
        if module_name == "fvt_checked_hyper_vendor_fanin":
            return fake_vendor
        return original(path, module_name)

    return load


@pytest.fixture(scope="module")
def lane_result(
    live_vendor_certificate: dict[str, object],
) -> dict[str, object]:
    specs = certifier.SEMANTIC_CERTIFIER_SPECS
    hyper_specs = tuple(spec for spec in specs if spec["lane_id"] == "hyperltl")
    with (
        mock.patch.object(
            certifier,
            "SEMANTIC_CERTIFIER_SPECS",
            hyper_specs,
        ),
        mock.patch.object(
            certifier,
            "_load_module_from_path",
            side_effect=_vendor_loader(live_vendor_certificate),
        ),
    ):
        result = certifier._run_semantic_lane_certifiers_with_prebuilt(
            repo_root=REPO_ROOT,
            env=certifier.offline_env({}),
            tool_certs={},
            runtime_mtl_prebuilt_bind={
                "authenticated": True,
                "bound": True,
            },
            runtime_mtl_prebuilt_invocation={
                "sealed_root": str(SEALED_ROOT),
                "timeout_seconds": 10.0,
            },
        )[0]
    return result


def _rebinding_result(
    source: dict[str, object],
    mutate,
) -> dict[str, object]:
    result = copy.deepcopy(source)
    receipt = result["receipt"]
    fanin = result["checked_vendor_fanin"]
    mutate(receipt, fanin)
    fanin["digest_sha256"] = certifier.content_digest(
        {key: value for key, value in fanin.items() if key != "digest_sha256"}
    )
    receipt["checked_vendor_fanin"] = copy.deepcopy(fanin)
    receipt = certifier._refresh_semantic_receipt_self_digests(receipt)
    result["receipt"] = receipt
    result["checked_vendor_fanin"] = fanin
    result["digest_sha256"] = certifier.content_digest(receipt)
    return result


def test_sealed_hyper_vendor_adapter_is_exact_and_bounded(
    lane_result: dict[str, object],
) -> None:
    assert lane_result["status"] == "ran"
    assert lane_result["certified"] is True
    assert lane_result["production_elevation_allowed"] is True
    assert lane_result["evidence_class"] == (
        certifier.CHECKED_HYPER_VENDOR_FANIN_EVIDENCE_CLASS
    )
    assert lane_result["elevated_tool_ids"] == list(TARGETS)

    receipt = lane_result["receipt"]
    fanin = lane_result["checked_vendor_fanin"]
    assert receipt["checked_vendor_fanin"] == fanin
    assert fanin["eligible_tool_ids"] == list(TARGETS)
    assert fanin["live_certificate"]["checks_passed"] == 66
    assert fanin["live_certificate"]["checks_total"] == 66
    assert fanin["live_certificate"]["case_results_total"] == 32
    assert receipt["independent_reference"]["available"] is False
    assert receipt["independent_reference"]["claimed"] is False
    assert receipt["authority_ceiling"] == "bounded"
    assert receipt["forbids_theorem_authority"] is True
    assert receipt["forbids_universal_claims_beyond_bounds"] is True

    for engine in receipt["engines"]:
        tool_id = engine["engine_id"]
        assert engine["certified"] is True
        assert len(engine["checks"]) == 22
        assert all(check["status"] == "passed" for check in engine["checks"])
        assert engine["independent_reference_available"] is False
        assert engine["authority_ceiling"] == "bounded"
        assert engine["authorizes_universal_proof"] is False
        assert engine["is_theorem_authority"] is False
        executable = Path(engine["executable"])
        assert executable.is_relative_to(SEALED_ROOT)
        assert certifier.file_digest(executable) == engine["executable_sha256"]
        kinds = {item["kind"] for item in engine["artifact_identities"]}
        assert "vendor_engine_executable" in kinds
        assert "checked_vendor_install_receipt" in kinds
        if tool_id == "mchyper":
            assert {
                "launcher_runtime",
                "launcher_target",
                "runtime_dependency_abc",
                "runtime_dependency_aigtoaig",
            } <= kinds


def test_hyper_vendor_receipt_identity_replaces_stale_discovery(
    lane_result: dict[str, object],
) -> None:
    tool_certs = {
        tool_id: certifier.ToolCertification(
            tool_id=tool_id,
            usable=False,
            unavailable=True,
            promotion_blocked=True,
            block_reasons=["stale_generic_discovery"],
        )
        for tool_id in TARGETS
    }
    elevations = certifier.apply_semantic_elevations(
        tool_certs,
        [copy.deepcopy(lane_result)],
        repo_root=REPO_ROOT,
    )
    assert {
        item["tool_id"] for item in elevations if item.get("elevated") is True
    } == set(TARGETS)
    for cert in tool_certs.values():
        assert cert.installed is True
        assert cert.identity_probed is True
        assert cert.usable is True
        assert cert.unavailable is False
        assert cert.production_certified is True
        assert cert.promotion_blocked is False
        assert cert.evidence_class == (
            certifier.CHECKED_HYPER_VENDOR_FANIN_EVIDENCE_CLASS
        )
        assert Path(cert.executable_path).is_relative_to(SEALED_ROOT)
        assert cert.block_reasons == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt, _fanin: receipt["engines"][0]["checks"].pop(),
        lambda _receipt, fanin: fanin["live_certificate"]["per_engine"][
            "autohyper"
        ].update({"checks_total": 21}),
        lambda receipt, _fanin: receipt["engines"][2]["artifact_identities"][
            0
        ].update({"sha256": "sha256:" + ("0" * 64)}),
    ],
    ids=("missing_check", "per_engine_count", "sealed_artifact_digest"),
)
def test_recorded_hyper_fanin_rejects_recomputed_tamper(
    lane_result: dict[str, object],
    mutate,
) -> None:
    tampered = _rebinding_result(lane_result, mutate)
    assert not certifier._recorded_checked_vendor_fanin_eligibility(
        repo_root=REPO_ROOT,
        semantic_spec=_semantic_spec(),
        result_fanin=tampered["checked_vendor_fanin"],
        receipt_fanin=tampered["receipt"]["checked_vendor_fanin"],
        semantic_receipt=tampered["receipt"],
    )


def test_section_list_disagreement_and_failed_check_fail_closed(
    live_vendor_certificate: dict[str, object],
) -> None:
    def section_only_mutation(live):
        section = copy.deepcopy(live["hyperltl"])
        section["version"] = "section-only"
        live["hyperltl"] = section

    def failed_check_mutation(live):
        section = copy.deepcopy(live["autohyper"])
        section["checks"][0]["status"] = "failed"
        live["autohyper"] = section
        listed = copy.deepcopy(live["engines"][1])
        listed["checks"][0]["status"] = "failed"
        live["engines"][1] = listed

    for mutation, expected_reason in (
        (
            section_only_mutation,
            "live_hyper_vendor_hyperltl_section_list_mismatch",
        ),
        (
            failed_check_mutation,
            "live_hyper_vendor_autohyper_full_check_set_invalid",
        ),
    ):
        tampered = copy.deepcopy(live_vendor_certificate)
        mutation(tampered)
        tampered["certificate_digest_sha256"] = (
            certifier._checked_vendor_outer_digest(
                tampered,
                repo_root=REPO_ROOT,
                uses_public_projection=True,
            )
        )
        with mock.patch.object(
            certifier,
            "_load_module_from_path",
            side_effect=_vendor_loader(tampered),
        ):
            result = certifier._build_checked_hyper_vendor_adapter(
                repo_root=REPO_ROOT,
                sealed_root=SEALED_ROOT,
                semantic_spec=_semantic_spec(),
                semantic_module=_semantic_module(),
            )
        assert result["fanin"]["vendor_valid"] is False
        assert expected_reason in result["fanin"]["failures"]
        assert result["fanin"]["eligible_tool_ids"] == []


def test_wrong_root_and_nested_receipt_tamper_fail_closed(
    tmp_path: Path,
    live_vendor_certificate: dict[str, object],
) -> None:
    wrong_root = certifier._build_checked_hyper_vendor_adapter(
        repo_root=REPO_ROOT,
        sealed_root=tmp_path,
        semantic_spec=_semantic_spec(),
        semantic_module=_semantic_module(),
    )
    assert wrong_root["fanin"]["vendor_valid"] is False
    assert wrong_root["fanin"]["eligible_tool_ids"] == []

    tampered = copy.deepcopy(live_vendor_certificate)
    tampered["install_receipt"]["task_id"] = "tampered"
    tampered["certificate_digest_sha256"] = certifier._checked_vendor_outer_digest(
        tampered,
        repo_root=REPO_ROOT,
        uses_public_projection=True,
    )
    with mock.patch.object(
        certifier,
        "_load_module_from_path",
        side_effect=_vendor_loader(tampered),
    ):
        result = certifier._build_checked_hyper_vendor_adapter(
            repo_root=REPO_ROOT,
            sealed_root=SEALED_ROOT,
            semantic_spec=_semantic_spec(),
            semantic_module=_semantic_module(),
        )
    assert "live_hyper_vendor_nested_receipt_mismatch" in (
        result["fanin"]["failures"]
    )
    assert result["fanin"]["eligible_tool_ids"] == []


def test_builder_freshly_replays_and_rejects_forged_fanin(
    monkeypatch: pytest.MonkeyPatch,
    lane_result: dict[str, object],
    live_vendor_certificate: dict[str, object],
) -> None:
    monkeypatch.setenv(
        certifier.RUNTIME_MTL_SEALED_ROOT_ENV,
        str(SEALED_ROOT),
    )
    projected = certifier._compact_semantic_lane_projection(
        certifier._project_semantic_lane_result(
            lane_result,
            repo_root=REPO_ROOT,
        )
    )
    with mock.patch.object(
        certifier,
        "_load_module_from_path",
        side_effect=_vendor_loader(live_vendor_certificate),
    ):
        audit = builder._audited_semantic_elevation_policy(
            certifier=certifier,
            repo_root=REPO_ROOT,
            spec=_semantic_spec(),
            semantic_result=projected,
        )
    assert audit["valid"] is True
    assert audit["fanin_satisfied"] is True
    assert audit["eligible_tool_ids"] == list(TARGETS)
    assert audit["production_allowed_tool_ids"] == list(TARGETS)
    assert audit["independent_reference_available"] is False
    assert audit["authority_ceiling"] == "bounded"
    assert all(
        item["valid"] is True for item in audit["reference_audits"].values()
    )

    forged = _rebinding_result(
        lane_result,
        lambda _receipt, fanin: fanin["live_certificate"]["per_engine"][
            "mchyper"
        ].update({"checks_passed": 21}),
    )
    forged_projected = certifier._compact_semantic_lane_projection(
        certifier._project_semantic_lane_result(
            forged,
            repo_root=REPO_ROOT,
        )
    )
    with mock.patch.object(
        certifier,
        "_load_module_from_path",
        side_effect=_vendor_loader(live_vendor_certificate),
    ):
        rejected = builder._audited_semantic_elevation_policy(
            certifier=certifier,
            repo_root=REPO_ROOT,
            spec=_semantic_spec(),
            semantic_result=forged_projected,
        )
    assert rejected["valid"] is False
    assert rejected["eligible_tool_ids"] == []
    assert "checked_hyper_vendor_fanin_fresh_replay_mismatch" in (
        rejected["failures"]
    )
