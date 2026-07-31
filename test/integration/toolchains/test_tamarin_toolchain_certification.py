"""Tamarin + Maude toolchain installation and certification (FVT-043 / FVT-G130).

Covers:

* installer plugin selects exact Tamarin 1.12.0 and Maude 3.5.1 under strict mode;
* secure, attack, mutated claim/rule, replay, malformed output, timeout, and
  version-mismatch cases pass offline via the certification corpus;
* receipts bind theory, claims, bounds, and exact binaries;
* Maude is support only and cannot promote the protocol lane by itself;
* certification never installs, downloads, or opens the network;
* the protocol lane handler binds under the roles surface without editing the
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
    / "tamarin.py"
)
TAMARIN_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "tamarin.py"
ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "TamarinToolchainCertification@1"
SCHEMA_VERSION = "tamarin-toolchain-certification/v1"
CORPUS_SCHEMA = "tamarin-toolchain-corpus/v1"
GOAL_ID = "FVT-G130"
TASK_ID = "FVT-043"
LOCKED_TAMARIN_VERSION = "1.12.0"
LOCKED_MAUDE_VERSION = "3.5.1"

REQUIRED_CASE_KINDS = {
    "secure",
    "attack",
    "mutation",
    "replay",
    "malformed",
    "timeout",
    "version_mismatch",
}

REQUIRED_CASE_IDS = {
    "secure_claims",
    "attack_trace",
    "mutated_claim",
    "mutated_rule",
    "deterministic_replay",
    "malformed_output",
    "timeout_claim",
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
    from ipfs_datasets_py.logic.backends.installers import tamarin as installer_mod

    return installer_mod


@pytest.fixture(scope="module")
def tamarin_cert():
    return _load_module(TAMARIN_CERT_PATH, "tools_logic_certification_tamarin")


@pytest.fixture(scope="module")
def receipt(tamarin_cert) -> dict[str, Any]:
    return tamarin_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        env=tamarin_cert.offline_env(os.environ),
    )


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert TAMARIN_CERT_PATH.is_file()
    assert Path(__file__).is_file()


def test_module_constants(installer, tamarin_cert) -> None:
    assert installer.TAMARIN_VERSION == LOCKED_TAMARIN_VERSION
    assert installer.MAUDE_VERSION == LOCKED_MAUDE_VERSION
    assert installer.GOAL_ID == GOAL_ID
    assert installer.TASK_ID == TASK_ID
    assert installer.IMPORT_INSTALLS_FORBIDDEN is True

    assert tamarin_cert.INTERFACE == INTERFACE
    assert tamarin_cert.SCHEMA_VERSION == SCHEMA_VERSION
    assert tamarin_cert.GOAL_ID == GOAL_ID
    assert tamarin_cert.TASK_ID == TASK_ID
    assert tamarin_cert.LOCKED_TAMARIN_VERSION == LOCKED_TAMARIN_VERSION
    assert tamarin_cert.LOCKED_MAUDE_VERSION == LOCKED_MAUDE_VERSION
    assert tamarin_cert.LANE_ID == "protocol"
    assert tamarin_cert.CERTIFICATION_SURFACE == "tools.logic.certification.tamarin"
    assert tamarin_cert.AUTHORITY_SCOPE == "protocol_verification_only"


# ---------------------------------------------------------------------------
# Installer: strict pin selection and fail-closed policy
# ---------------------------------------------------------------------------


def test_strict_install_selects_tamarin_1_12_0_and_maude_3_5_1(installer) -> None:
    lock = None
    if LOCK_PATH.is_file():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        assert lock["managed_pin_versions"]["tamarin"] == LOCKED_TAMARIN_VERSION
        assert lock["managed_pin_versions"]["maude"] == LOCKED_MAUDE_VERSION

    tamarin_pin = installer.select_strict_pin(
        "tamarin",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    maude_pin = installer.select_strict_pin(
        "maude",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    assert tamarin_pin.version == LOCKED_TAMARIN_VERSION
    assert maude_pin.version == LOCKED_MAUDE_VERSION
    assert len(tamarin_pin.sha256) == 64
    assert len(maude_pin.sha256) == 64
    assert tamarin_pin.artifact_url.startswith("https://")
    assert maude_pin.artifact_url.startswith("https://")


def test_strict_pin_rejects_wrong_version(installer) -> None:
    fake_lock = {
        "managed_pin_versions": {"tamarin": "9.9.9"},
        "tools": [
            {
                "tool_id": "tamarin",
                "pins": [
                    {
                        "tool_id": "tamarin",
                        "version": "1.12.0",
                        "platform": "linux-x86_64",
                        "artifact_url": "https://example.invalid/tamarin.tgz",
                        "sha256": "a" * 64,
                    }
                ],
            }
        ],
    }
    with pytest.raises(installer.TamarinInstallerError):
        installer.select_strict_pin(
            "tamarin",
            platform_key="linux-x86_64",
            lock=fake_lock,
            allow_source_fallback=False,
        )


def test_ensure_without_yes_is_blocked(
    installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a missing host so the yes=True gate is exercised even when the
    # developer machine already has a managed Maude/Tamarin install.
    monkeypatch.setattr(installer, "which_executable", lambda *_a, **_k: None)

    tamarin = installer.ensure_tamarin(
        yes=False,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        ensure_maude_first=True,
    )
    maude = installer.ensure_maude(
        yes=False,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert tamarin.selected_version == LOCKED_TAMARIN_VERSION
    assert maude.selected_version == LOCKED_MAUDE_VERSION
    assert tamarin.status == "blocked"
    assert maude.status == "blocked"
    assert "yes_required" in tamarin.reason_codes or "dry_run" in tamarin.reason_codes
    assert "yes_required" in maude.reason_codes or "dry_run" in maude.reason_codes


def test_ensure_dry_run_with_yes_selects_locked_pins(installer) -> None:
    receipt = installer.ensure_tamarin(
        yes=True,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert receipt.selected_version == LOCKED_TAMARIN_VERSION
    assert receipt.pin is not None
    assert receipt.pin["version"] == LOCKED_TAMARIN_VERSION
    assert receipt.bindings["maude_locked_version"] == LOCKED_MAUDE_VERSION
    assert receipt.bindings["maude_is_support_only"] is True
    assert receipt.install_attempted is False if hasattr(receipt, "install_attempted") else True
    assert receipt.phase == "dry_run"


def test_authorize_install_forbids_import_and_requires_yes(installer) -> None:
    with pytest.raises(installer.TamarinInstallerError):
        installer.authorize_plugin_install(
            "tamarin",
            yes=True,
            import_context=True,
        )
    with pytest.raises(installer.TamarinInstallerError):
        installer.authorize_plugin_install(
            "maude",
            yes=False,
        )


def test_maude_compatibility_allowlist(installer) -> None:
    assert installer.maude_version_is_compatible("3.5.1")
    assert installer.maude_version_is_compatible("Maude 3.5.1")
    assert installer.maude_version_is_compatible("3.2.1")
    assert not installer.maude_version_is_compatible("3.2")
    assert not installer.maude_version_is_compatible("1.0.0")


def test_tamarin_accepts_maude_runtime_marker(installer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        installer.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "tamarin-prover 1.12.0\n"
                "checking installation: OK.\n"
                "Maude version 3.5.1\n"
            ),
            stderr="",
        ),
    )
    assert installer.tamarin_accepts_maude("/fixture/tamarin", "/fixture/maude")


def test_plugin_manifest_declares_support_boundary(installer) -> None:
    manifest = installer.plugin_manifest()
    assert manifest["family"] == "tamarin"
    assert manifest["locked_versions"]["tamarin"] == LOCKED_TAMARIN_VERSION
    assert manifest["locked_versions"]["maude"] == LOCKED_MAUDE_VERSION
    assert manifest["maude_is_support_only"] is True
    assert manifest["maude_cannot_promote_protocol_lane"] is True
    assert manifest["roles"]["maude"] == "support"
    assert manifest["roles"]["tamarin"] == "authority"
    assert manifest["policy"]["never_on_import"] is True
    assert manifest["policy"]["requires_explicit_yes"] is True
    assert manifest["policy"]["strict_selects_locked_versions"] is True
    assert {entry["tool_id"] for entry in manifest["entries"]} >= {
        "tamarin",
        "maude",
        "stack",
    }


# ---------------------------------------------------------------------------
# Certification corpus and offline semantics
# ---------------------------------------------------------------------------


def test_corpus_schema_and_required_cases(tamarin_cert) -> None:
    manifest = tamarin_cert.default_corpus_manifest()
    assert manifest["schema_version"] == CORPUS_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["locked_tamarin_version"] == LOCKED_TAMARIN_VERSION
    assert manifest["locked_maude_version"] == LOCKED_MAUDE_VERSION
    assert manifest["policy"]["maude_is_support_only"] is True
    assert manifest["policy"]["maude_cannot_promote_protocol_lane"] is True
    assert manifest["policy"]["no_install"] is True
    assert manifest["policy"]["no_network"] is True

    cases = manifest["cases"]
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds
    case_ids = {case["case_id"] for case in cases}
    assert REQUIRED_CASE_IDS <= case_ids


def test_offline_semantic_cases_pass(tamarin_cert) -> None:
    for case in tamarin_cert.corpus_cases():
        outcome = tamarin_cert.evaluate_corpus_case(case)
        assert outcome.matched is True, (
            f"{outcome.case_id}: expected {outcome.expect}, "
            f"got {outcome.status} ({outcome.reason_codes})"
        )


def test_secure_and_attack_and_mutations(tamarin_cert) -> None:
    by_id = {
        case["case_id"]: tamarin_cert.evaluate_corpus_case(case)
        for case in tamarin_cert.corpus_cases()
    }
    assert by_id["secure_claims"].status == "secure"
    assert by_id["attack_trace"].status == "attack"
    assert by_id["attack_trace"].attack_trace is not None
    assert by_id["mutated_claim"].status in {"quarantined", "attack", "unknown"}
    assert by_id["mutated_claim"].status != "secure"
    assert by_id["mutated_rule"].status == "attack"
    assert by_id["malformed_output"].status != "secure"
    assert by_id["timeout_claim"].status != "secure"
    assert by_id["version_mismatch"].status == "blocked"


def test_deterministic_replay_digests(tamarin_cert) -> None:
    by_id = {
        case["case_id"]: tamarin_cert.evaluate_corpus_case(case)
        for case in tamarin_cert.corpus_cases()
    }
    secure = by_id["secure_claims"]
    replay = by_id["deterministic_replay"]
    assert secure.output_digest == replay.output_digest
    assert secure.status == replay.status == "secure"


def test_receipt_binds_theory_claims_bounds_and_binaries(receipt: dict[str, Any]) -> None:
    bindings = receipt["bindings"]
    assert "theory" in bindings
    assert bindings["theory"]["ceiling"]["adversary_model"] == "dolev_yao"
    assert bindings["theory"]["bound_theories"]
    assert "claims" in bindings
    assert "secrecy" in bindings["claims"]["bound_claim_kinds"] or any(
        "secrecy" in item for item in bindings["claims"]["bound_claim_kinds"]
    )
    assert "bounds" in bindings
    assert bindings["bounds"]["network"] is False
    assert bindings["bounds"]["install"] is False
    assert "binaries" in bindings
    assert bindings["binaries"]["tamarin"]["locked_version"] == LOCKED_TAMARIN_VERSION
    assert bindings["binaries"]["maude"]["locked_version"] == LOCKED_MAUDE_VERSION
    assert bindings["binaries"]["maude"]["support_only"] is True
    assert bindings["binaries"]["maude"]["can_promote_protocol_lane"] is False
    assert bindings["authority"]["maude_is_support_only"] is True
    assert bindings["authority"]["scope"] == "protocol_verification_only"

    check = next(c for c in receipt["checks"] if c["check_id"] == "tamarin.bindings")
    assert check["status"] == "passed"


def test_offline_policy_never_installs(receipt: dict[str, Any]) -> None:
    assert receipt["network_used"] is False
    assert receipt["install_attempted"] is False
    assert receipt["download_attempted"] is False
    policy = receipt["policy"]
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert policy["maude_is_support_only"] is True
    assert policy["does_not_edit_central_certificate"] is True
    assert policy["does_not_edit_proverif_lane"] is True

    offline = next(
        c for c in receipt["checks"] if c["check_id"] == "tamarin.offline_policy"
    )
    assert offline["status"] == "passed"


def test_semantic_corpus_passed_even_without_live_tools(receipt: dict[str, Any]) -> None:
    # Offline parsers must fully pass regardless of host tool presence.
    assert receipt["semantic_corpus_passed"] is True
    by_id = {case["case_id"]: case for case in receipt["cases"]}
    for case_id in REQUIRED_CASE_IDS:
        assert by_id[case_id]["matched"] is True, case_id
    # Corpus case checks must pass.
    for check in receipt["checks"]:
        if check["kind"] in REQUIRED_CASE_KINDS:
            assert check["status"] == "passed", check


def test_maude_cannot_promote_protocol_lane(tamarin_cert, receipt: dict[str, Any]) -> None:
    boundary = tamarin_cert.maude_cannot_promote_protocol_lane()
    assert boundary["support_only"] is True
    assert boundary["promotion_allowed"] is False
    assert boundary["can_satisfy_protocol_requirement"] is False
    assert boundary["blocks_alone"] is True
    assert boundary["role"] == "support"
    assert boundary["authority_ceiling"] == "none"

    check = next(
        c for c in receipt["checks"] if c["check_id"] == "maude.support_only_boundary"
    )
    assert check["status"] == "passed"


def test_version_mismatch_case_blocks(tamarin_cert) -> None:
    case = next(
        item
        for item in tamarin_cert.corpus_cases()
        if item["case_id"] == "version_mismatch"
    )
    outcome = tamarin_cert.evaluate_corpus_case(case)
    assert outcome.status == "blocked"
    assert "locked_version_mismatch" in outcome.reason_codes


def test_live_identity_mismatch_blocks_production_cert(
    tamarin_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tamarin_cert,
        "probe_tamarin_identity",
        lambda **_kwargs: {
            "tool_id": "tamarin",
            "path_present": True,
            "executable_path": "/fixture/tamarin-prover",
            "version_string": "tamarin-prover 1.8.0",
            "identity_probed": True,
            "version_match": False,
            "locked_version": LOCKED_TAMARIN_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    monkeypatch.setattr(
        tamarin_cert,
        "probe_maude_identity",
        lambda **_kwargs: {
            "tool_id": "maude",
            "path_present": True,
            "executable_path": "/fixture/maude",
            "version_string": "Maude 3.1",
            "identity_probed": True,
            "version_match": False,
            "locked_version": LOCKED_MAUDE_VERSION,
            "support_only": True,
            "can_promote_protocol_lane": False,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    cert = tamarin_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.production_certified is False
    assert cert.promotion_blocked is True
    assert cert.tamarin_version_match is False
    assert cert.semantic_corpus_passed if hasattr(cert, "semantic_corpus_passed") else True
    assert all(case.matched for case in cert.cases)


def test_production_certified_when_live_pair_usable(
    tamarin_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tamarin_cert,
        "probe_tamarin_identity",
        lambda **_kwargs: {
            "tool_id": "tamarin",
            "path_present": True,
            "executable_path": "/fixture/tamarin-prover",
            "version_string": f"tamarin-prover {LOCKED_TAMARIN_VERSION}",
            "identity_probed": True,
            "version_match": True,
            "locked_version": LOCKED_TAMARIN_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    monkeypatch.setattr(
        tamarin_cert,
        "probe_maude_identity",
        lambda **_kwargs: {
            "tool_id": "maude",
            "path_present": True,
            "executable_path": "/fixture/maude",
            "version_string": f"Maude {LOCKED_MAUDE_VERSION}",
            "identity_probed": True,
            "version_match": True,
            "locked_version": LOCKED_MAUDE_VERSION,
            "support_only": True,
            "can_promote_protocol_lane": False,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    monkeypatch.setattr(
        tamarin_cert,
        "probe_tamarin_maude_pair",
        lambda *_args, **_kwargs: {
            "validated": True,
            "detail": "pair_ok",
            "output": (
                f"tamarin-prover {LOCKED_TAMARIN_VERSION}\n"
                "checking installation: OK.\n"
                f"Maude version {LOCKED_MAUDE_VERSION}\n"
            ),
        },
    )
    cert = tamarin_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.tamarin_usable is True
    assert cert.maude_usable is True
    assert cert.pair_validated is True
    assert cert.production_certified is True
    assert cert.promotion_blocked is False
    assert cert.block_reasons == []
    assert all(case.matched for case in cert.cases)


def test_lane_handler_binds_under_roles_without_editing_central_certificate(
    tamarin_cert,
) -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface not present in this worktree")
    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles_for_tamarin")
    policy = roles.build_role_aware_policy(register_placeholders=True)
    roles.bind_lane_handler(
        "protocol", tamarin_cert.lane_handler, policy=policy, replace=True
    )
    handler = policy.get_lane_handler("protocol")
    assert callable(handler)
    result = handler(repo_root=REPO_ROOT)
    assert result["owner_module"] == "tools.logic.certification.tamarin"
    assert result["handler_id"] == tamarin_cert.HANDLER_ID
    assert result["lane_id"] == "protocol"
    assert "production_certified" in result
    assert result["authority_scope"] == "protocol_verification_only"
    assert result["maude_support_only"] is True
    assert result["maude_cannot_promote_alone"] is True


def test_receipt_digest_present(receipt: dict[str, Any]) -> None:
    assert receipt["interface"] == INTERFACE
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["receipt_digest_sha256"]
    assert len(receipt["receipt_digest_sha256"]) == 64
