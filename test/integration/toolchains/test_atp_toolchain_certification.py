"""Vampire + E ATP toolchain installation and certification (FVT-048 / FVT-G140).

Covers:

* installer plugin selects exact Vampire 5.0.1 and E 3.2.5 under strict mode;
* theorem, non-theorem, premise/conclusion mutation, proof-output binding,
  replay, malformed output, timeout, reconstruction, and version-mismatch
  cases pass offline via the SZS certification corpus;
* receipts bind TPTP sources, SZS outcomes, bounds, and exact binaries;
* ATP results remain candidates unless independent kernel reconstruction
  elevates them;
* certification never installs, downloads, or opens the network;
* the ATP lane handler binds under the roles surface without editing the
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
    / "atp.py"
)
ATP_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "atp.py"
ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "ATPToolchainCertification@1"
SCHEMA_VERSION = "atp-toolchain-certification/v1"
CORPUS_SCHEMA = "atp-toolchain-corpus/v1"
GOAL_ID = "FVT-G140"
TASK_ID = "FVT-048"
LOCKED_VAMPIRE_VERSION = "5.0.1"
LOCKED_EPROVER_VERSION = "3.2.5"

REQUIRED_CASE_KINDS = {
    "theorem",
    "non_theorem",
    "mutation",
    "proof_binding",
    "replay",
    "malformed",
    "timeout",
    "version_mismatch",
    "reconstruction",
}

REQUIRED_CASE_IDS = {
    "theorem_proved",
    "non_theorem",
    "mutated_premise",
    "mutated_conclusion",
    "proof_output_binding",
    "deterministic_replay",
    "malformed_output",
    "timeout_claim",
    "version_mismatch",
    "kernel_reconstruction_requires_receipt",
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
    from ipfs_datasets_py.logic.backends.installers import atp as installer_mod

    return installer_mod


@pytest.fixture(scope="module")
def atp_cert():
    return _load_module(ATP_CERT_PATH, "tools_logic_certification_atp")


@pytest.fixture(scope="module")
def receipt(atp_cert) -> dict[str, Any]:
    return atp_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        env=atp_cert.offline_env(os.environ),
    )


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert ATP_CERT_PATH.is_file()
    assert Path(__file__).is_file()


def test_module_constants(installer, atp_cert) -> None:
    assert installer.VAMPIRE_VERSION == LOCKED_VAMPIRE_VERSION
    assert installer.EPROVER_VERSION == LOCKED_EPROVER_VERSION
    assert installer.GOAL_ID == GOAL_ID
    assert installer.TASK_ID == TASK_ID
    assert installer.IMPORT_INSTALLS_FORBIDDEN is True

    assert atp_cert.INTERFACE == INTERFACE
    assert atp_cert.SCHEMA_VERSION == SCHEMA_VERSION
    assert atp_cert.GOAL_ID == GOAL_ID
    assert atp_cert.TASK_ID == TASK_ID
    assert atp_cert.LOCKED_VAMPIRE_VERSION == LOCKED_VAMPIRE_VERSION
    assert atp_cert.LOCKED_EPROVER_VERSION == LOCKED_EPROVER_VERSION
    assert atp_cert.LANE_ID == "atp"
    assert atp_cert.CERTIFICATION_SURFACE == "tools.logic.certification.atp"
    assert atp_cert.AUTHORITY_SCOPE == "atp_candidate_until_kernel_reconstruction"
    assert atp_cert.AUTHORITY_CEILING == "reconstruction"


# ---------------------------------------------------------------------------
# Installer: strict pin selection and fail-closed policy
# ---------------------------------------------------------------------------


def test_strict_install_selects_vampire_5_0_1_and_e_3_2_5(installer) -> None:
    lock = None
    if LOCK_PATH.is_file():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        assert lock["managed_pin_versions"]["vampire"] == LOCKED_VAMPIRE_VERSION
        assert lock["managed_pin_versions"]["eprover"] == LOCKED_EPROVER_VERSION

    vampire_pin = installer.select_strict_pin(
        "vampire",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    eprover_pin = installer.select_strict_pin(
        "eprover",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    assert vampire_pin.version == LOCKED_VAMPIRE_VERSION
    assert eprover_pin.version == LOCKED_EPROVER_VERSION
    assert len(vampire_pin.sha256) == 64
    assert len(eprover_pin.sha256) == 64
    assert vampire_pin.artifact_url.startswith("https://")
    assert eprover_pin.artifact_url.startswith("https://")
    # E ships a portable pin under platform "any".
    assert eprover_pin.platform in {"any", "linux-x86_64", "source", "portable"}


def test_strict_pin_rejects_wrong_version(installer) -> None:
    fake_lock = {
        "managed_pin_versions": {"vampire": "9.9.9"},
        "tools": [
            {
                "tool_id": "vampire",
                "pins": [
                    {
                        "tool_id": "vampire",
                        "version": "5.0.1",
                        "platform": "linux-x86_64",
                        "artifact_url": "https://example.invalid/vampire.zip",
                        "sha256": "a" * 64,
                    }
                ],
            }
        ],
    }
    with pytest.raises(installer.ATPInstallerError):
        installer.select_strict_pin(
            "vampire",
            platform_key="linux-x86_64",
            lock=fake_lock,
            allow_any_platform=False,
        )


def test_ensure_without_yes_is_blocked(
    installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer, "which_executable", lambda *_a, **_k: None)

    vampire = installer.ensure_vampire(
        yes=False,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    eprover = installer.ensure_eprover(
        yes=False,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert vampire.selected_version == LOCKED_VAMPIRE_VERSION
    assert eprover.selected_version == LOCKED_EPROVER_VERSION
    assert vampire.status == "blocked"
    assert eprover.status == "blocked"
    assert "yes_required" in vampire.reason_codes or "dry_run" in vampire.reason_codes
    assert "yes_required" in eprover.reason_codes or "dry_run" in eprover.reason_codes


def test_ensure_dry_run_with_yes_selects_locked_pins(
    installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a missing host so dry-run pin selection is exercised even when the
    # developer machine already has Vampire/E on PATH.
    monkeypatch.setattr(installer, "which_executable", lambda *_a, **_k: None)

    receipt = installer.ensure_vampire(
        yes=True,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert receipt.selected_version == LOCKED_VAMPIRE_VERSION
    assert receipt.pin is not None
    assert receipt.pin["version"] == LOCKED_VAMPIRE_VERSION
    assert receipt.bindings["results_are_candidates_without_reconstruction"] is True
    assert receipt.bindings[
        "kernel_reconstruction_required_for_theorem_authority"
    ] is True
    assert receipt.phase == "dry_run"

    e_receipt = installer.ensure_eprover(
        yes=True,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert e_receipt.selected_version == LOCKED_EPROVER_VERSION
    assert e_receipt.phase == "dry_run"


def test_authorize_install_forbids_import_and_requires_yes(installer) -> None:
    with pytest.raises(installer.ATPInstallerError):
        installer.authorize_plugin_install(
            "vampire",
            yes=True,
            import_context=True,
        )
    with pytest.raises(installer.ATPInstallerError):
        installer.authorize_plugin_install(
            "eprover",
            yes=False,
        )


def test_plugin_manifest_declares_candidate_authority_boundary(installer) -> None:
    manifest = installer.plugin_manifest()
    assert manifest["family"] == "atp"
    assert manifest["locked_versions"]["vampire"] == LOCKED_VAMPIRE_VERSION
    assert manifest["locked_versions"]["eprover"] == LOCKED_EPROVER_VERSION
    assert manifest["results_are_candidates_without_reconstruction"] is True
    assert manifest["kernel_reconstruction_required_for_theorem_authority"] is True
    assert manifest["authority_ceiling"] == "reconstruction"
    assert manifest["roles"]["vampire"] == "authority"
    assert manifest["roles"]["eprover"] == "authority"
    assert manifest["policy"]["never_on_import"] is True
    assert manifest["policy"]["requires_explicit_yes"] is True
    assert manifest["policy"]["strict_selects_locked_versions"] is True
    assert {entry["tool_id"] for entry in manifest["entries"]} >= {
        "vampire",
        "eprover",
    }


def test_portfolio_dry_run(installer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer, "which_executable", lambda *_a, **_k: None)
    portfolio = installer.ensure_atp_portfolio(
        yes=True,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert portfolio["vampire"].selected_version == LOCKED_VAMPIRE_VERSION
    assert portfolio["eprover"].selected_version == LOCKED_EPROVER_VERSION


# ---------------------------------------------------------------------------
# Certification corpus and offline semantics
# ---------------------------------------------------------------------------


def test_corpus_schema_and_required_cases(atp_cert) -> None:
    manifest = atp_cert.default_corpus_manifest()
    assert manifest["schema_version"] == CORPUS_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["locked_vampire_version"] == LOCKED_VAMPIRE_VERSION
    assert manifest["locked_eprover_version"] == LOCKED_EPROVER_VERSION
    assert manifest["policy"]["results_are_candidates_without_reconstruction"] is True
    assert manifest["policy"][
        "kernel_reconstruction_required_for_theorem_authority"
    ] is True
    assert manifest["policy"]["kernel_reconstruction_receipt_validated"] is False
    assert (
        manifest["policy"]["boolean_reconstruction_claim_cannot_elevate"]
        is True
    )
    assert manifest["policy"]["no_install"] is True
    assert manifest["policy"]["no_network"] is True
    assert manifest["policy"]["szs_status_only"] is True

    cases = manifest["cases"]
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds
    case_ids = {case["case_id"] for case in cases}
    assert REQUIRED_CASE_IDS <= case_ids


def test_offline_semantic_cases_pass(atp_cert) -> None:
    for case in atp_cert.corpus_cases():
        outcome = atp_cert.evaluate_corpus_case(case)
        assert outcome.matched is True, (
            f"{outcome.case_id}: expected {outcome.expect}, "
            f"got {outcome.status} ({outcome.reason_codes})"
        )


def test_theorem_non_theorem_mutations_and_timeout(atp_cert) -> None:
    by_id = {
        case["case_id"]: atp_cert.evaluate_corpus_case(case)
        for case in atp_cert.corpus_cases()
    }
    assert by_id["theorem_proved"].status == "theorem_candidate"
    assert by_id["theorem_proved"].authority == "candidate"
    assert by_id["non_theorem"].status == "non_theorem_candidate"
    assert by_id["mutated_premise"].status != "theorem_candidate"
    assert by_id["mutated_premise"].status != "theorem_authority"
    assert by_id["mutated_conclusion"].status != "theorem_candidate"
    assert by_id["malformed_output"].status == "quarantined"
    assert by_id["timeout_claim"].status == "timeout"
    assert by_id["version_mismatch"].status == "blocked"
    reconstruction = by_id["kernel_reconstruction_requires_receipt"]
    assert reconstruction.status == "theorem_candidate"
    assert reconstruction.authority == "candidate"
    assert reconstruction.independent_kernel_reconstruction is False
    assert reconstruction.kernel_reconstruction_claimed is True
    assert (
        "kernel_reconstruction_receipt_required"
        in reconstruction.reason_codes
    )


def test_proof_output_binding(atp_cert) -> None:
    case = next(
        item
        for item in atp_cert.corpus_cases()
        if item["case_id"] == "proof_output_binding"
    )
    outcome = atp_cert.evaluate_corpus_case(case)
    assert outcome.matched is True
    assert outcome.proof_bound is True
    assert outcome.status == "theorem_candidate"
    assert outcome.authority == "candidate"


def test_deterministic_replay_digests(atp_cert) -> None:
    by_id = {
        case["case_id"]: atp_cert.evaluate_corpus_case(case)
        for case in atp_cert.corpus_cases()
    }
    theorem = by_id["theorem_proved"]
    replay = by_id["deterministic_replay"]
    assert theorem.output_digest == replay.output_digest
    assert theorem.status == replay.status == "theorem_candidate"


def test_free_form_proof_found_is_not_theorem(atp_cert) -> None:
    # Free-form phrases without SZS must never grant theorem status.
    outcome = atp_cert.classify_szs_outcome(
        "Proof found!!!\nTheorem\nRefutation found\n"
    )
    assert outcome["status"] == "quarantined"
    assert outcome["authority"] == "candidate"


def test_boolean_reconstruction_claim_never_grants_theorem_authority(
    atp_cert,
) -> None:
    outcome = atp_cert.classify_szs_outcome(
        "% SZS status Theorem for claimed_reconstruction\n"
        "% SZS output start Proof for claimed_reconstruction\n"
        "fof(1, plain, p).\n"
        "% SZS output end Proof for claimed_reconstruction\n",
        independent_kernel_reconstruction=True,
        require_proof_body=True,
    )

    assert outcome["status"] == "theorem_candidate"
    assert outcome["authority"] == "candidate"
    assert outcome["result_status"] == "candidate"
    assert "kernel_reconstruction_claim_unvalidated" in outcome["reason_codes"]
    assert "kernel_reconstruction_receipt_required" in outcome["reason_codes"]


def test_receipt_binds_sources_bounds_and_binaries(receipt: dict[str, Any]) -> None:
    bindings = receipt["bindings"]
    assert "adapter" in bindings
    assert bindings["adapter"]["szs_status_only"] is True
    assert "bounds" in bindings
    assert bindings["bounds"]["network"] is False
    assert bindings["bounds"]["install"] is False
    assert "binaries" in bindings
    assert bindings["binaries"]["vampire"]["locked_version"] == LOCKED_VAMPIRE_VERSION
    assert bindings["binaries"]["eprover"]["locked_version"] == LOCKED_EPROVER_VERSION
    assert bindings["authority"]["ceiling"] == "reconstruction"
    assert bindings["authority"][
        "results_are_candidates_without_reconstruction"
    ] is True
    assert bindings["authority"][
        "kernel_reconstruction_required_for_theorem_authority"
    ] is True
    assert bindings["authority"]["scope"] == "atp_candidate_until_kernel_reconstruction"

    check = next(c for c in receipt["checks"] if c["check_id"] == "atp.bindings")
    assert check["status"] == "passed"


def test_offline_policy_never_installs(receipt: dict[str, Any]) -> None:
    assert receipt["network_used"] is False
    assert receipt["install_attempted"] is False
    assert receipt["download_attempted"] is False
    policy = receipt["policy"]
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert policy["results_are_candidates_without_reconstruction"] is True
    assert policy["does_not_edit_central_certificate"] is True
    assert policy["does_not_edit_cec_semantics"] is True

    offline = next(
        c for c in receipt["checks"] if c["check_id"] == "atp.offline_policy"
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


def test_candidate_until_reconstruction_boundary(atp_cert, receipt: dict[str, Any]) -> None:
    boundary = atp_cert.atp_results_remain_candidates_without_reconstruction()
    assert boundary["results_are_candidates_without_reconstruction"] is True
    assert boundary["kernel_reconstruction_required_for_theorem_authority"] is True
    assert boundary["boundary_holds"] is True
    assert (
        boundary["sample_without_reconstruction"]["status"] == "theorem_candidate"
    )
    assert (
        boundary["sample_with_reconstruction"]["status"]
        == "theorem_candidate"
    )
    assert (
        boundary["sample_with_reconstruction"]["authority"]
        == "candidate"
    )
    assert (
        "kernel_reconstruction_receipt_required"
        in boundary["sample_with_reconstruction"]["reason_codes"]
    )
    assert boundary["kernel_reconstruction_receipt_validated"] is False
    assert boundary["boolean_reconstruction_claim_cannot_elevate"] is True

    check = next(
        c
        for c in receipt["checks"]
        if c["check_id"] == "atp.candidate_until_reconstruction"
    )
    assert check["status"] == "passed"


def test_version_mismatch_case_blocks(atp_cert) -> None:
    case = next(
        item
        for item in atp_cert.corpus_cases()
        if item["case_id"] == "version_mismatch"
    )
    outcome = atp_cert.evaluate_corpus_case(case)
    assert outcome.status == "blocked"
    assert "locked_version_mismatch" in outcome.reason_codes


def test_live_identity_mismatch_blocks_production_cert(
    atp_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        atp_cert,
        "probe_vampire_identity",
        lambda **_kwargs: {
            "tool_id": "vampire",
            "path_present": True,
            "executable_path": "/fixture/vampire",
            "version_string": "Vampire 4.5.1",
            "identity_probed": True,
            "version_match": False,
            "locked_version": LOCKED_VAMPIRE_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    monkeypatch.setattr(
        atp_cert,
        "probe_eprover_identity",
        lambda **_kwargs: {
            "tool_id": "eprover",
            "path_present": True,
            "executable_path": "/fixture/eprover",
            "version_string": "E 2.6",
            "identity_probed": True,
            "version_match": False,
            "locked_version": LOCKED_EPROVER_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    cert = atp_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.production_certified is False
    assert cert.promotion_blocked is True
    assert cert.vampire_version_match is False
    assert all(case.matched for case in cert.cases)


def test_production_certified_when_live_tools_usable(
    atp_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        atp_cert,
        "probe_vampire_identity",
        lambda **_kwargs: {
            "tool_id": "vampire",
            "path_present": True,
            "executable_path": "/fixture/vampire",
            "version_string": f"Vampire {LOCKED_VAMPIRE_VERSION}",
            "identity_probed": True,
            "version_match": True,
            "locked_version": LOCKED_VAMPIRE_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    monkeypatch.setattr(
        atp_cert,
        "probe_eprover_identity",
        lambda **_kwargs: {
            "tool_id": "eprover",
            "path_present": True,
            "executable_path": "/fixture/eprover",
            "version_string": f"E {LOCKED_EPROVER_VERSION} Konstanz",
            "identity_probed": True,
            "version_match": True,
            "locked_version": LOCKED_EPROVER_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    cert = atp_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.vampire_usable is True
    assert cert.eprover_usable is True
    assert cert.production_certified is True
    assert cert.promotion_blocked is False
    assert cert.results_are_candidates_without_reconstruction is True
    assert cert.block_reasons == []
    assert all(case.matched for case in cert.cases)


def test_lane_handler_binds_under_roles_without_editing_central_certificate(
    atp_cert,
) -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface not present in this worktree")
    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles_for_atp")
    policy = roles.build_role_aware_policy(register_placeholders=True)
    roles.bind_lane_handler(
        "atp", atp_cert.lane_handler, policy=policy, replace=True
    )
    handler = policy.get_lane_handler("atp")
    assert callable(handler)
    result = handler(repo_root=REPO_ROOT)
    assert result["owner_module"] == "tools.logic.certification.atp"
    assert result["handler_id"] == atp_cert.HANDLER_ID
    assert result["lane_id"] == "atp"
    assert "production_certified" in result
    assert result["authority_scope"] == "atp_candidate_until_kernel_reconstruction"
    assert result["results_are_candidates_without_reconstruction"] is True
    assert result["kernel_reconstruction_required_for_theorem_authority"] is True


def test_receipt_digest_present(receipt: dict[str, Any]) -> None:
    assert receipt["interface"] == INTERFACE
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["receipt_digest_sha256"]
    assert len(receipt["receipt_digest_sha256"]) == 64


def test_observed_version_helpers(installer) -> None:
    assert installer.observed_version_matches_lock(
        f"Vampire {LOCKED_VAMPIRE_VERSION}", LOCKED_VAMPIRE_VERSION
    )
    assert installer.observed_version_matches_lock(
        f"E {LOCKED_EPROVER_VERSION} Konstanz (eprover)", LOCKED_EPROVER_VERSION
    )
    assert not installer.observed_version_matches_lock("Vampire 4.5.1", LOCKED_VAMPIRE_VERSION)
