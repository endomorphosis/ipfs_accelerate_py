"""Isabelle installation and session/kernel certification (FVT-049 / FVT-G151).

Covers:

* installer plugin selects exact Isabelle2025-2 under strict mode;
* checked theory/session, bad proof, assumption/conclusion mutation, replay,
  replay mismatch, malformed output, timeout, and wrong installation cases
  pass offline via the certification corpus;
* receipts bind theory heap, session, imports, source, property, and exact
  tool identity;
* Hammer remains proposal-only until independent kernel reconstruction;
* certification never installs, downloads, or opens the network;
* large-download/storage budget is observed by the installer;
* the kernel lane handler binds under the roles surface without editing the
  central multi-prover certificate.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
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
    / "isabelle.py"
)
ISABELLE_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "isabelle.py"
ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "IsabelleToolchainCertification@1"
SCHEMA_VERSION = "isabelle-toolchain-certification/v1"
CORPUS_SCHEMA = "isabelle-toolchain-corpus/v1"
GOAL_ID = "FVT-G151"
TASK_ID = "FVT-049"
LOCKED_VERSION = "Isabelle2025-2"

REQUIRED_CASE_KINDS = {
    "positive",
    "negative",
    "mutation",
    "replay",
    "replay_mismatch",
    "malformed",
    "timeout",
    "version_mismatch",
    "fail_closed",
    "policy",
}

REQUIRED_CASE_IDS = {
    "checked_theory_session",
    "bad_proof",
    "assumption_mutation",
    "conclusion_mutation",
    "deterministic_replay",
    "replay_mismatch",
    "malformed_output",
    "timeout_case",
    "wrong_installation",
    "sorry_escape",
    "hammer_proposal_only",
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
    from ipfs_datasets_py.logic.backends.installers import isabelle as installer_mod

    return installer_mod


@pytest.fixture(scope="module")
def isabelle_cert():
    return _load_module(ISABELLE_CERT_PATH, "tools_logic_certification_isabelle")


@pytest.fixture(scope="module")
def receipt(isabelle_cert) -> dict[str, Any]:
    return isabelle_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        env=isabelle_cert.offline_env(os.environ),
    )


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert ISABELLE_CERT_PATH.is_file()
    assert Path(__file__).is_file()


def test_module_constants(installer, isabelle_cert) -> None:
    assert installer.ISABELLE_VERSION == LOCKED_VERSION
    assert installer.GOAL_ID == GOAL_ID
    assert installer.TASK_ID == TASK_ID
    assert installer.IMPORT_INSTALLS_FORBIDDEN is True
    assert installer.MAX_DOWNLOAD_BYTES > 0
    assert installer.MIN_FREE_STORAGE_BYTES > installer.MAX_DOWNLOAD_BYTES

    assert isabelle_cert.INTERFACE == INTERFACE
    assert isabelle_cert.SCHEMA_VERSION == SCHEMA_VERSION
    assert isabelle_cert.GOAL_ID == GOAL_ID
    assert isabelle_cert.TASK_ID == TASK_ID
    assert isabelle_cert.LOCKED_VERSION == LOCKED_VERSION
    assert isabelle_cert.LANE_ID == "kernel"
    assert isabelle_cert.CERTIFICATION_SURFACE == "tools.logic.certification.isabelle"
    assert isabelle_cert.AUTHORITY_SCOPE == "kernel_proof_checking_only"


# ---------------------------------------------------------------------------
# Installer: strict pin selection and fail-closed policy
# ---------------------------------------------------------------------------


def test_strict_install_selects_isabelle_2025_2(installer) -> None:
    lock = None
    if LOCK_PATH.is_file():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        assert lock["managed_pin_versions"]["isabelle"] == LOCKED_VERSION

    pin = installer.select_strict_pin(
        "isabelle",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    assert pin.version == LOCKED_VERSION
    assert pin.platform == "linux-x86_64"
    assert len(pin.sha256) == 64
    assert pin.artifact_url.startswith("https://")
    assert "Isabelle2025-2" in pin.artifact_url


def test_strict_pin_rejects_wrong_version(installer) -> None:
    fake_lock = {
        "managed_pin_versions": {"isabelle": "Isabelle2099-9"},
        "tools": [
            {
                "tool_id": "isabelle",
                "pins": [
                    {
                        "tool_id": "isabelle",
                        "version": "Isabelle2025-2",
                        "platform": "linux-x86_64",
                        "artifact_url": "https://example.invalid/isabelle.tgz",
                        "sha256": "a" * 64,
                    }
                ],
            }
        ],
    }
    with pytest.raises(installer.IsabelleInstallerError):
        installer.select_strict_pin(
            "isabelle",
            platform_key="linux-x86_64",
            lock=fake_lock,
            allow_source_fallback=False,
        )


def test_ensure_without_yes_is_blocked(
    installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer, "which_executable", lambda *_a, **_k: None)

    receipt = installer.ensure_isabelle(
        yes=False,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        skip_storage_budget=True,
    )
    assert receipt.selected_version == LOCKED_VERSION
    assert receipt.status == "blocked"
    assert "yes_required" in receipt.reason_codes or "dry_run" in receipt.reason_codes


def test_ensure_dry_run_with_yes_selects_locked_pin(
    installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a missing host binary so dry-run pin selection is exercised even
    # when the developer machine already has a managed Isabelle install.
    monkeypatch.setattr(installer, "which_executable", lambda *_a, **_k: None)

    receipt = installer.ensure_isabelle(
        yes=True,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        skip_storage_budget=True,
    )
    assert receipt.selected_version == LOCKED_VERSION
    assert receipt.pin is not None
    assert receipt.pin["version"] == LOCKED_VERSION
    assert receipt.bindings["hammer_is_proposal_only"] is True
    assert receipt.bindings["hammer_cannot_grant_kernel_authority"] is True
    assert receipt.install_attempted is False
    assert receipt.phase == "dry_run"
    assert receipt.authority_tool is True


def test_authorize_install_forbids_import_and_requires_yes(installer) -> None:
    with pytest.raises(installer.IsabelleInstallerError):
        installer.authorize_plugin_install(
            "isabelle",
            yes=True,
            import_context=True,
        )
    with pytest.raises(installer.IsabelleInstallerError):
        installer.authorize_plugin_install(
            "isabelle",
            yes=False,
        )


def test_storage_budget_gate(installer) -> None:
    report = installer.check_storage_budget(
        installer.expand_user_local_root(),
        min_free_bytes=1,
        max_download_bytes=installer.MAX_DOWNLOAD_BYTES,
        expected_archive_bytes=1024,
    )
    assert report["ok"] is True
    assert report["archive_within_download_cap"] is True

    over_cap = installer.check_storage_budget(
        installer.expand_user_local_root(),
        min_free_bytes=1,
        max_download_bytes=100,
        expected_archive_bytes=200,
    )
    assert over_cap["ok"] is False
    assert "archive_exceeds_download_cap" in over_cap["reason_codes"]


def test_plugin_manifest_declares_hammer_and_budget_boundary(installer) -> None:
    manifest = installer.plugin_manifest()
    assert manifest["family"] == "isabelle"
    assert manifest["locked_versions"]["isabelle"] == LOCKED_VERSION
    assert manifest["hammer_is_proposal_only"] is True
    assert manifest["hammer_cannot_grant_kernel_authority"] is True
    assert manifest["roles"]["isabelle"] == "authority"
    assert manifest["roles"]["hammer"] == "advisor"
    assert manifest["policy"]["never_on_import"] is True
    assert manifest["policy"]["requires_explicit_yes"] is True
    assert manifest["policy"]["strict_selects_locked_versions"] is True
    assert manifest["policy"]["observes_large_download_storage_budget"] is True
    assert manifest["policy"]["does_not_edit_shared_lock"] is True
    assert manifest["policy"]["does_not_edit_central_certificate"] is True
    assert {entry["tool_id"] for entry in manifest["entries"]} >= {"isabelle"}


def test_observed_version_matching(installer) -> None:
    assert installer.observed_version_matches_lock(
        "Isabelle2025-2", LOCKED_VERSION
    )
    assert installer.observed_version_matches_lock(
        "Isabelle 2025: Isabelle2025-2", LOCKED_VERSION
    )
    assert not installer.observed_version_matches_lock(
        "Isabelle2021-1", LOCKED_VERSION
    )


# ---------------------------------------------------------------------------
# Certification corpus and offline semantics
# ---------------------------------------------------------------------------


def test_corpus_schema_and_required_cases(isabelle_cert) -> None:
    manifest = isabelle_cert.default_corpus_manifest()
    assert manifest["schema_version"] == CORPUS_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["locked_version"] == LOCKED_VERSION
    assert manifest["policy"]["hammer_is_proposal_only"] is True
    assert manifest["policy"]["no_install"] is True
    assert manifest["policy"]["no_network"] is True
    assert manifest["policy"]["authority_is_kernel_proof_checking_only"] is True

    cases = manifest["cases"]
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds
    case_ids = {case["case_id"] for case in cases}
    assert REQUIRED_CASE_IDS <= case_ids


def test_offline_semantic_cases_pass(isabelle_cert) -> None:
    outcomes: dict[str, Any] = {}
    for case in isabelle_cert.corpus_cases():
        outcome = isabelle_cert.evaluate_corpus_case(
            case, reference_outcomes=outcomes
        )
        outcomes[outcome.case_id] = outcome
        assert outcome.matched is True, (
            f"{outcome.case_id}: expected {outcome.expect}, "
            f"got {outcome.status} ({outcome.reason_codes})"
        )


def test_checked_theory_and_failures(isabelle_cert) -> None:
    outcomes: dict[str, Any] = {}
    by_id = {}
    for case in isabelle_cert.corpus_cases():
        outcome = isabelle_cert.evaluate_corpus_case(
            case, reference_outcomes=outcomes
        )
        outcomes[outcome.case_id] = outcome
        by_id[outcome.case_id] = outcome

    assert by_id["checked_theory_session"].accepted is True
    assert by_id["bad_proof"].accepted is False
    assert by_id["assumption_mutation"].accepted is False
    assert by_id["conclusion_mutation"].accepted is False
    assert by_id["malformed_output"].accepted is False
    assert by_id["timeout_case"].accepted is False
    assert by_id["wrong_installation"].status == "blocked"
    assert "locked_version_mismatch" in by_id["wrong_installation"].reason_codes
    assert by_id["sorry_escape"].accepted is False
    assert "sorry_or_oops" in by_id["sorry_escape"].reason_codes
    assert by_id["hammer_proposal_only"].status == "proposal_only"


def test_deterministic_replay_and_mismatch(isabelle_cert) -> None:
    outcomes: dict[str, Any] = {}
    by_id = {}
    for case in isabelle_cert.corpus_cases():
        outcome = isabelle_cert.evaluate_corpus_case(
            case, reference_outcomes=outcomes
        )
        outcomes[outcome.case_id] = outcome
        by_id[outcome.case_id] = outcome

    positive = by_id["checked_theory_session"]
    replay = by_id["deterministic_replay"]
    mismatch = by_id["replay_mismatch"]
    assert positive.accepted and replay.accepted
    assert positive.source_digest == replay.source_digest
    assert positive.output_digest == replay.output_digest
    assert mismatch.accepted is False
    assert mismatch.output_digest != positive.output_digest
    assert "replay_output_mismatch" in mismatch.reason_codes


def test_source_scan_rejects_sorry_and_axiomatization(isabelle_cert) -> None:
    assert "sorry_or_oops" in isabelle_cert.scan_isabelle_incomplete_or_unreviewed(
        "theorem t: \"True\" sorry\n"
    )
    assert "sorry_or_oops" in isabelle_cert.scan_isabelle_incomplete_or_unreviewed(
        "theorem t: \"True\" oops\n"
    )
    assert (
        "unreviewed_axiomatization"
        in isabelle_cert.scan_isabelle_incomplete_or_unreviewed(
            "axiomatization bad where bad_ax: \"False\"\n"
        )
    )
    assert isabelle_cert.scan_isabelle_incomplete_or_unreviewed(
        'theorem from_eq: "n = m ⟹ n = m" by simp\n'
    ) == ()


def test_receipt_binds_theory_session_imports_source_property_identity(
    receipt: dict[str, Any],
) -> None:
    bindings = receipt["bindings"]
    assert "theory_heap" in bindings
    assert bindings["theory_heap"]["theory_name"]
    assert bindings["theory_heap"]["session"]
    assert "session" in bindings
    assert "process_command_template" in bindings["session"]
    assert "imports" in bindings
    assert bindings["imports"]
    assert "source" in bindings
    assert bindings["source"]["source_digest"]
    assert bindings["source"]["primary_path"].endswith(".thy")
    assert "property" in bindings
    assert bindings["property"]["theorem_name"]
    assert "tool_identity" in bindings
    assert bindings["tool_identity"]["tool_id"] == "isabelle"
    assert bindings["tool_identity"]["locked_version"] == LOCKED_VERSION
    assert bindings["authority"]["scope"] == "kernel_proof_checking_only"
    assert bindings["authority"]["hammer_is_proposal_only"] is True

    check = next(c for c in receipt["checks"] if c["check_id"] == "isabelle.bindings")
    assert check["status"] == "passed"


def test_offline_policy_never_installs(receipt: dict[str, Any]) -> None:
    assert receipt["network_used"] is False
    assert receipt["install_attempted"] is False
    assert receipt["download_attempted"] is False
    policy = receipt["policy"]
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert policy["hammer_is_proposal_only"] is True
    assert policy["does_not_edit_central_certificate"] is True
    assert policy["does_not_edit_shared_lock"] is True

    offline = next(
        c for c in receipt["checks"] if c["check_id"] == "isabelle.offline_policy"
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


def test_hammer_remains_proposal_only(isabelle_cert, receipt: dict[str, Any]) -> None:
    boundary = isabelle_cert.hammer_remains_proposal_only()
    assert boundary["proposal_only"] is True
    assert boundary["can_grant_kernel_authority"] is False
    assert boundary["promotion_allowed_from_hammer_alone"] is False
    assert boundary["requires_independent_kernel_reconstruction"] is True
    assert boundary["reconstruction_kernel"] == "isabelle"

    check = next(
        c for c in receipt["checks"] if c["check_id"] == "isabelle.hammer_proposal_only"
    )
    assert check["status"] == "passed"
    assert receipt["hammer_proposal_only"] is True


def test_wrong_installation_blocks_production_cert(
    isabelle_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        isabelle_cert,
        "probe_isabelle_identity",
        lambda **_kwargs: {
            "tool_id": "isabelle",
            "path_present": True,
            "executable_path": "/fixture/isabelle",
            "version_string": "Isabelle2021-1",
            "identity_probed": True,
            "installed": True,
            "version_match": False,
            "locked_version": LOCKED_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    cert = isabelle_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.production_certified is False
    assert cert.promotion_blocked is True
    assert cert.version_match is False
    assert cert.semantic_corpus_passed is True
    assert all(case.matched for case in cert.cases)


def test_production_certified_when_live_pin_usable(
    isabelle_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        isabelle_cert,
        "probe_isabelle_identity",
        lambda **_kwargs: {
            "tool_id": "isabelle",
            "path_present": True,
            "executable_path": "/fixture/isabelle",
            "version_string": LOCKED_VERSION,
            "identity_probed": True,
            "installed": True,
            "version_match": True,
            "locked_version": LOCKED_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    cert = isabelle_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.usable is True
    assert cert.version_match is True
    assert cert.semantic_corpus_passed is True
    assert cert.production_certified is True
    assert cert.promotion_blocked is False
    assert cert.block_reasons == []
    assert all(case.matched for case in cert.cases)
    assert cert.bindings["tool_identity"]["locked_version"] == LOCKED_VERSION
    assert cert.bindings["theory_heap"]["theory_name"]


def test_lane_handler_binds_under_roles_without_editing_central_certificate(
    isabelle_cert,
) -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface not present in this worktree")
    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles_for_isabelle")
    policy = roles.build_role_aware_policy(register_placeholders=True)
    roles.bind_lane_handler(
        "kernel", isabelle_cert.lane_handler, policy=policy, replace=True
    )
    handler = policy.get_lane_handler("kernel")
    assert callable(handler)
    result = handler(repo_root=REPO_ROOT)
    assert result["owner_module"] == "tools.logic.certification.isabelle"
    assert result["handler_id"] == isabelle_cert.HANDLER_ID
    assert result["lane_id"] == "kernel"
    assert "production_certified" in result
    assert result["authority_scope"] == "kernel_proof_checking_only"
    assert result["hammer_proposal_only"] is True


def test_receipt_digest_present(receipt: dict[str, Any]) -> None:
    assert receipt["interface"] == INTERFACE
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["receipt_digest_sha256"]
    assert len(receipt["receipt_digest_sha256"]) == 64
