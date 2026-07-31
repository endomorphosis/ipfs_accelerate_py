"""TLC + Apalache toolchain installation and certification (FVT-042 / FVT-G120).

Covers:

* installer plugin selects exact TLC 1.8.0 and Apalache 0.58.3 under strict mode;
* invariant-holds, violation trace, mutated Next/invariant, replay, malformed
  model, timeout, and bound behavior cases pass offline via the certification
  corpus;
* receipts bind model, config, constants, bounds, and exact tool identities;
* Java is support only and cannot promote the TLA lane by itself;
* bounded model-checking never promotes to theorem authority;
* certification never installs, downloads, or opens the network;
* the TLA lane handler binds under the roles surface without editing the
  central multi-prover certificate.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "installers"
    / "state_model.py"
)
STATE_MODEL_CERT_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "state_model.py"
)
ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "StateModelToolchainCertification@1"
SCHEMA_VERSION = "state-model-toolchain-certification/v1"
CORPUS_SCHEMA = "state-model-toolchain-corpus/v1"
GOAL_ID = "FVT-G120"
TASK_ID = "FVT-042"
LOCKED_TLC_VERSION = "1.8.0"
LOCKED_APALACHE_VERSION = "0.58.3"
LOCKED_TLC_SHA256 = (
    "e22f8ffb4bacdea0a871f444dd94fe5fb0d8013b3388ae39e82e26f852c735d5"
)

REQUIRED_CASE_KINDS = {
    "invariant_holds",
    "violation_trace",
    "mutation",
    "replay",
    "malformed",
    "timeout",
    "bound",
    "version_mismatch",
}

REQUIRED_CASE_IDS = {
    "invariant_holds",
    "violation_trace",
    "mutated_next",
    "mutated_invariant",
    "deterministic_replay",
    "malformed_model",
    "timeout_bound",
    "bound_behavior",
    "version_mismatch",
}


def _ensure_import_paths() -> None:
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing expected output: {path}"
    _ensure_import_paths()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def installer():
    assert INSTALLER_PATH.is_file(), f"missing expected output: {INSTALLER_PATH}"
    _ensure_import_paths()
    from ipfs_datasets_py.logic.backends.installers import state_model as installer_mod

    return installer_mod


@pytest.fixture(scope="module")
def state_model_cert():
    return _load_module(STATE_MODEL_CERT_PATH, "tools_logic_certification_state_model")


@pytest.fixture(scope="module")
def receipt(state_model_cert) -> dict[str, Any]:
    return state_model_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        env=state_model_cert.offline_env(os.environ),
    )


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert STATE_MODEL_CERT_PATH.is_file()
    assert Path(__file__).is_file()


def test_module_constants(installer, state_model_cert) -> None:
    assert installer.TLC_VERSION == LOCKED_TLC_VERSION
    assert installer.APALACHE_VERSION == LOCKED_APALACHE_VERSION
    assert installer.GOAL_ID == GOAL_ID
    assert installer.TASK_ID == TASK_ID
    assert installer.IMPORT_INSTALLS_FORBIDDEN is True

    assert state_model_cert.INTERFACE == INTERFACE
    assert state_model_cert.SCHEMA_VERSION == SCHEMA_VERSION
    assert state_model_cert.GOAL_ID == GOAL_ID
    assert state_model_cert.TASK_ID == TASK_ID
    assert state_model_cert.LOCKED_TLC_VERSION == LOCKED_TLC_VERSION
    assert state_model_cert.LOCKED_APALACHE_VERSION == LOCKED_APALACHE_VERSION
    assert state_model_cert.LANE_ID == "tla"
    assert (
        state_model_cert.CERTIFICATION_SURFACE
        == "tools.logic.certification.state_model"
    )
    assert state_model_cert.AUTHORITY_SCOPE == "bounded_state_model_only"


# ---------------------------------------------------------------------------
# Installer: strict pin selection and fail-closed policy
# ---------------------------------------------------------------------------


def test_strict_install_selects_tlc_1_8_0_and_apalache_0_58_3(installer) -> None:
    lock = None
    if LOCK_PATH.is_file():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        assert lock["managed_pin_versions"]["tlc"] == LOCKED_TLC_VERSION
        assert lock["managed_pin_versions"]["apalache"] == LOCKED_APALACHE_VERSION

    tlc_pin = installer.select_strict_pin(
        "tlc",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    apalache_pin = installer.select_strict_pin(
        "apalache",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    assert tlc_pin.version == LOCKED_TLC_VERSION
    assert apalache_pin.version == LOCKED_APALACHE_VERSION
    assert tlc_pin.platform in {"any", "linux-x86_64"}
    assert apalache_pin.platform in {"any", "linux-x86_64"}
    assert tlc_pin.sha256 == LOCKED_TLC_SHA256
    assert tlc_pin.is_checksummed is True
    assert apalache_pin.sha256
    assert len(apalache_pin.sha256) == 64
    assert tlc_pin.artifact_url.startswith("https://")
    assert apalache_pin.artifact_url.startswith("https://")
    assert tlc_pin.requires_checksum_at_install is True


def test_strict_pin_rejects_wrong_version(installer) -> None:
    fake_lock = {
        "managed_pin_versions": {"tlc": "9.9.9"},
        "tools": [
            {
                "tool_id": "tlc",
                "pins": [
                    {
                        "tool_id": "tlc",
                        "version": "1.8.0",
                        "platform": "any",
                        "artifact_url": "https://example.invalid/tla2tools.jar",
                        "sha256": "a" * 64,
                        "is_checksummed": True,
                    }
                ],
            }
        ],
    }
    with pytest.raises(installer.StateModelInstallerError):
        installer.select_strict_pin(
            "tlc",
            platform_key="linux-x86_64",
            lock=fake_lock,
            allow_any_platform=True,
        )


def test_ensure_without_yes_is_blocked(
    installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer, "which_executable", lambda *_a, **_k: None)
    monkeypatch.setattr(installer, "java_is_available", lambda: True)

    tlc = installer.ensure_tlc(
        yes=False,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        require_java=False,
    )
    apalache = installer.ensure_apalache(
        yes=False,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        require_java=False,
    )
    assert tlc.selected_version == LOCKED_TLC_VERSION
    assert apalache.selected_version == LOCKED_APALACHE_VERSION
    assert tlc.status == "blocked"
    assert apalache.status == "blocked"
    assert "yes_required" in tlc.reason_codes or "dry_run" in tlc.reason_codes
    assert "yes_required" in apalache.reason_codes or "dry_run" in apalache.reason_codes


def test_ensure_dry_run_with_yes_selects_locked_pins(installer) -> None:
    receipt = installer.ensure_tlc(
        yes=True,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        require_java=False,
    )
    assert receipt.selected_version == LOCKED_TLC_VERSION
    assert receipt.pin is not None
    assert receipt.pin["version"] == LOCKED_TLC_VERSION
    assert receipt.bindings["java_is_support_only"] is True
    assert receipt.grants_theorem_authority is False
    assert receipt.phase == "dry_run"


def test_authorize_install_forbids_import_and_requires_yes(installer) -> None:
    with pytest.raises(installer.StateModelInstallerError):
        installer.authorize_plugin_install(
            "tlc",
            yes=True,
            import_context=True,
        )
    with pytest.raises(installer.StateModelInstallerError):
        installer.authorize_plugin_install(
            "apalache",
            yes=False,
        )


def test_plugin_manifest_declares_support_boundary(installer) -> None:
    manifest = installer.plugin_manifest()
    assert manifest["family"] == "state_model"
    assert manifest["locked_versions"]["tlc"] == LOCKED_TLC_VERSION
    assert manifest["locked_versions"]["apalache"] == LOCKED_APALACHE_VERSION
    assert manifest["java_is_support_only"] is True
    assert manifest["java_cannot_promote_tla_lane"] is True
    assert manifest["grants_theorem_authority"] is False
    assert manifest["roles"]["java"] == "support"
    assert manifest["roles"]["tlc"] == "authority"
    assert manifest["roles"]["apalache"] == "authority"
    assert manifest["policy"]["never_on_import"] is True
    assert manifest["policy"]["requires_explicit_yes"] is True
    assert manifest["policy"]["strict_selects_locked_versions"] is True
    assert {entry["tool_id"] for entry in manifest["entries"]} >= {
        "tlc",
        "apalache",
    }


def test_portfolio_dry_run_selects_both(installer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "which_executable", lambda *_a, **_k: None)
    portfolio = installer.ensure_state_model_portfolio(
        yes=True,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert portfolio["both_selected"] is True
    assert portfolio["java_is_support_only"] is True
    assert portfolio["grants_theorem_authority"] is False
    assert portfolio["tlc"]["selected_version"] == LOCKED_TLC_VERSION
    assert portfolio["apalache"]["selected_version"] == LOCKED_APALACHE_VERSION


# ---------------------------------------------------------------------------
# Certification corpus and offline semantics
# ---------------------------------------------------------------------------


def test_corpus_schema_and_required_cases(state_model_cert) -> None:
    manifest = state_model_cert.default_corpus_manifest()
    assert manifest["schema_version"] == CORPUS_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["locked_tlc_version"] == LOCKED_TLC_VERSION
    assert manifest["locked_apalache_version"] == LOCKED_APALACHE_VERSION
    assert manifest["locked_artifact_digests"] == {
        "tlc": state_model_cert.LOCKED_TLC_SHA256,
        "apalache": state_model_cert.LOCKED_APALACHE_SHA256,
    }
    assert manifest["policy"]["java_is_support_only"] is True
    assert manifest["policy"]["java_cannot_promote_tla_lane"] is True
    assert manifest["policy"]["never_theorem_authority"] is True
    assert manifest["policy"]["no_install"] is True
    assert manifest["policy"]["no_network"] is True

    cases = manifest["cases"]
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds
    case_ids = {case["case_id"] for case in cases}
    assert REQUIRED_CASE_IDS <= case_ids


def test_offline_semantic_cases_pass(state_model_cert) -> None:
    for case in state_model_cert.corpus_cases():
        outcome = state_model_cert.evaluate_corpus_case(case)
        assert outcome.matched is True, (
            f"{outcome.case_id}: expected {outcome.expect}, "
            f"got {outcome.status} ({outcome.reason_codes})"
        )
        assert outcome.grants_theorem_authority is False


def test_invariant_holds_violation_and_mutations(state_model_cert) -> None:
    by_id = {
        case["case_id"]: state_model_cert.evaluate_corpus_case(case)
        for case in state_model_cert.corpus_cases()
    }
    assert by_id["invariant_holds"].status == "passed"
    assert by_id["violation_trace"].status == "counterexample"
    assert by_id["violation_trace"].counterexample is not None
    assert by_id["mutated_next"].status == "counterexample"
    assert by_id["mutated_next"].status != "passed"
    assert by_id["mutated_invariant"].status == "counterexample"
    assert by_id["malformed_model"].status != "passed"
    assert by_id["timeout_bound"].status == "timed_out"
    assert by_id["bound_behavior"].status == "passed"
    assert "bounded_only" in by_id["bound_behavior"].reason_codes or (
        "finite_trace_only" in by_id["bound_behavior"].reason_codes
    )
    assert by_id["version_mismatch"].status == "blocked"


def test_deterministic_replay_digests(state_model_cert) -> None:
    by_id = {
        case["case_id"]: state_model_cert.evaluate_corpus_case(case)
        for case in state_model_cert.corpus_cases()
    }
    holds = by_id["invariant_holds"]
    replay = by_id["deterministic_replay"]
    assert holds.output_digest == replay.output_digest
    assert holds.model_digest == replay.model_digest
    assert holds.config_digest == replay.config_digest
    assert holds.status == replay.status == "passed"


def test_receipt_binds_model_config_constants_bounds_and_tools(
    receipt: dict[str, Any],
) -> None:
    bindings = receipt["bindings"]
    assert bindings["locked_artifact_digests"] == {
        "tlc": receipt["bindings"]["binaries"]["tlc"][
            "locked_artifact_sha256"
        ],
        "apalache": receipt["bindings"]["binaries"]["apalache"][
            "locked_artifact_sha256"
        ],
    }
    assert "model" in bindings
    assert bindings["model"]["module_name"]
    assert "config" in bindings
    assert "constants" in bindings
    assert bindings["constants"]
    assert "bounds" in bindings
    assert bindings["bounds"]["network"] is False
    assert bindings["bounds"]["install"] is False
    assert bindings["bounds"]["unbounded_proof"] is False
    assert "binaries" in bindings
    assert bindings["binaries"]["tlc"]["locked_version"] == LOCKED_TLC_VERSION
    assert (
        bindings["binaries"]["apalache"]["locked_version"] == LOCKED_APALACHE_VERSION
    )
    assert bindings["binaries"]["java"]["support_only"] is True
    assert bindings["binaries"]["java"]["can_promote_tla_lane"] is False
    assert bindings["authority"]["java_is_support_only"] is True
    assert bindings["authority"]["never_theorem"] is True
    assert bindings["authority"]["scope"] == "bounded_state_model_only"

    check = next(c for c in receipt["checks"] if c["check_id"] == "state_model.bindings")
    assert check["status"] == "passed"


def test_offline_policy_never_installs(receipt: dict[str, Any]) -> None:
    assert receipt["network_used"] is False
    assert receipt["install_attempted"] is False
    assert receipt["download_attempted"] is False
    policy = receipt["policy"]
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert policy["java_is_support_only"] is True
    assert policy["never_theorem_authority"] is True
    assert policy["does_not_edit_central_certificate"] is True

    offline = next(
        c for c in receipt["checks"] if c["check_id"] == "state_model.offline_policy"
    )
    assert offline["status"] == "passed"


def test_semantic_corpus_passed_even_without_live_tools(receipt: dict[str, Any]) -> None:
    assert receipt["semantic_corpus_passed"] is True
    by_id = {case["case_id"]: case for case in receipt["cases"]}
    for case_id in REQUIRED_CASE_IDS:
        assert by_id[case_id]["matched"] is True, case_id
    for check in receipt["checks"]:
        if check["kind"] in REQUIRED_CASE_KINDS:
            assert check["status"] == "passed", check


def test_java_cannot_promote_tla_lane(state_model_cert, receipt: dict[str, Any]) -> None:
    boundary = state_model_cert.java_cannot_promote_tla_lane()
    assert boundary["support_only"] is True
    assert boundary["promotion_allowed"] is False
    assert boundary["can_satisfy_tla_requirement"] is False
    assert boundary["blocks_alone"] is True
    assert boundary["role"] == "support"
    assert boundary["authority_ceiling"] == "none"
    assert boundary["grants_theorem_authority"] is False

    check = next(
        c for c in receipt["checks"] if c["check_id"] == "java.support_only_boundary"
    )
    assert check["status"] == "passed"


def test_bounded_never_theorem_authority(
    state_model_cert, receipt: dict[str, Any]
) -> None:
    boundary = state_model_cert.bounded_checking_never_theorem_authority()
    assert boundary["never_theorem_authority"] is True
    assert boundary["bounded_model_checking_only"] is True
    assert boundary["tlc"]["is_bounded"] is True
    assert boundary["apalache"]["is_bounded"] is True
    assert boundary["apalache"]["finite_trace_only"] is True
    assert boundary["apalache"]["checks_liveness"] is False

    check = next(
        c
        for c in receipt["checks"]
        if c["check_id"] == "state_model.never_theorem_authority"
    )
    assert check["status"] == "passed"
    assert receipt["grants_theorem_authority"] is False
    assert receipt["bounded_model_checking_only"] is True


def test_version_mismatch_case_blocks(state_model_cert) -> None:
    case = next(
        item
        for item in state_model_cert.corpus_cases()
        if item["case_id"] == "version_mismatch"
    )
    outcome = state_model_cert.evaluate_corpus_case(case)
    assert outcome.status == "blocked"
    assert "locked_version_mismatch" in outcome.reason_codes


def test_corpus_artifact_digest_manifest_fails_closed(
    state_model_cert,
) -> None:
    manifest = state_model_cert.default_corpus_manifest()
    manifest["locked_artifact_digests"]["apalache"] = "0" * 64

    cert = state_model_cert.run_certification_suite(
        repo_root=REPO_ROOT,
        manifest=manifest,
    )

    check = next(
        item
        for item in cert.checks
        if item.check_id == "state_model.artifact_digest_manifest"
    )
    assert check.status == "failed"
    assert "artifact_digest_manifest_mismatch" in cert.block_reasons
    assert cert.production_certified is False


def test_live_identity_mismatch_blocks_production_cert(
    state_model_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        state_model_cert,
        "probe_tlc_identity",
        lambda **_kwargs: {
            "tool_id": "tlc",
            "path_present": True,
            "executable_path": "/fixture/tlc",
            "version_string": "TLC 1.7.0",
            "identity_probed": True,
            "version_match": False,
            "locked_version": LOCKED_TLC_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    monkeypatch.setattr(
        state_model_cert,
        "probe_apalache_identity",
        lambda **_kwargs: {
            "tool_id": "apalache",
            "path_present": True,
            "executable_path": "/fixture/apalache-mc",
            "version_string": "Apalache 0.40.0",
            "identity_probed": True,
            "version_match": False,
            "locked_version": LOCKED_APALACHE_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    monkeypatch.setattr(
        state_model_cert,
        "probe_java_identity",
        lambda **_kwargs: {
            "tool_id": "java",
            "path_present": True,
            "executable_path": "/fixture/java",
            "version_string": "openjdk 17",
            "identity_probed": True,
            "version_match": True,
            "support_only": True,
            "can_promote_tla_lane": False,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    cert = state_model_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.production_certified is False
    assert cert.promotion_blocked is True
    assert cert.tlc_version_match is False
    assert all(case.matched for case in cert.cases)


def test_production_certified_when_live_tools_usable(
    state_model_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        state_model_cert,
        "probe_tlc_identity",
        lambda **_kwargs: {
            "tool_id": "tlc",
            "path_present": True,
            "executable_path": "/fixture/tlc",
            "version_string": f"TLC2 Version {LOCKED_TLC_VERSION}",
            "identity_probed": True,
            "version_match": True,
            "managed_identity_verified": True,
            "managed_identity": {
                "usable": True,
                "artifact_digest_verified": True,
                "payload_digest_verified": True,
                "launchers_structurally_valid": True,
                "manifest_valid": True,
            },
            "locked_version": LOCKED_TLC_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    monkeypatch.setattr(
        state_model_cert,
        "probe_apalache_identity",
        lambda **_kwargs: {
            "tool_id": "apalache",
            "path_present": True,
            "executable_path": "/fixture/apalache-mc",
            "version_string": f"Apalache version {LOCKED_APALACHE_VERSION}",
            "identity_probed": True,
            "version_match": True,
            "managed_identity_verified": True,
            "managed_identity": {
                "usable": True,
                "artifact_digest_verified": True,
                "distribution_tree_verified": True,
                "payload_digest_verified": True,
                "launchers_structurally_valid": True,
                "manifest_valid": True,
            },
            "locked_version": LOCKED_APALACHE_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    monkeypatch.setattr(
        state_model_cert,
        "probe_java_identity",
        lambda **_kwargs: {
            "tool_id": "java",
            "path_present": True,
            "executable_path": "/fixture/java",
            "version_string": "openjdk version \"17.0.0\"",
            "identity_probed": True,
            "version_match": True,
            "support_only": True,
            "can_promote_tla_lane": False,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    cert = state_model_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.tlc_usable is True
    assert cert.apalache_usable is True
    assert cert.java_usable is True
    assert cert.production_certified is True
    assert cert.promotion_blocked is False
    assert cert.block_reasons == []
    assert cert.grants_theorem_authority is False
    assert all(case.matched for case in cert.cases)


def test_lane_handler_binds_under_roles_without_editing_central_certificate(
    state_model_cert,
) -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface not present in this worktree")
    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles_for_state_model")
    policy = roles.build_role_aware_policy(register_placeholders=True)
    roles.bind_lane_handler(
        "tla", state_model_cert.lane_handler, policy=policy, replace=True
    )
    handler = policy.get_lane_handler("tla")
    assert callable(handler)
    result = handler(repo_root=REPO_ROOT)
    assert result["owner_module"] == "tools.logic.certification.state_model"
    assert result["handler_id"] == state_model_cert.HANDLER_ID
    assert result["lane_id"] == "tla"
    assert "production_certified" in result
    assert result["authority_scope"] == "bounded_state_model_only"
    assert result["java_support_only"] is True
    assert result["java_cannot_promote_alone"] is True
    assert result["grants_theorem_authority"] is False


def test_receipt_digest_present(receipt: dict[str, Any]) -> None:
    assert receipt["interface"] == INTERFACE
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["receipt_digest_sha256"]
    assert len(receipt["receipt_digest_sha256"]) == 64
