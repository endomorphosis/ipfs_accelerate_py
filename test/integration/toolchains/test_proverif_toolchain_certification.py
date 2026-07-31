"""ProVerif + isolated OPAM toolchain installation and certification (FVT-044 / FVT-G131).

Covers:

* installer plugin selects exact OPAM 2.5.2 support and ProVerif 2.05 under strict mode;
* repository-local isolated OPAM root contract (never global switch mutation);
* secure, attack, mutated claim/model, replay, malformed output, cancellation, and
  version-mismatch cases pass offline via the certification corpus;
* receipts bind model, claims, bounds, exact binaries, and isolated root;
* OPAM is support only and cannot promote the protocol lane by itself;
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
    / "proverif.py"
)
PROVERIF_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "proverif.py"
ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "ProVerifToolchainCertification@1"
SCHEMA_VERSION = "proverif-toolchain-certification/v1"
CORPUS_SCHEMA = "proverif-toolchain-corpus/v1"
GOAL_ID = "FVT-G131"
TASK_ID = "FVT-044"
LOCKED_PROVERIF_VERSION = "2.05"
LOCKED_OPAM_VERSION = "2.5.2"

REQUIRED_CASE_KINDS = {
    "secure",
    "attack",
    "mutation",
    "replay",
    "malformed",
    "cancellation",
    "version_mismatch",
}

REQUIRED_CASE_IDS = {
    "secure_claims",
    "attack_trace",
    "mutated_claim",
    "mutated_model",
    "deterministic_replay",
    "malformed_output",
    "cancelled_query",
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
    from ipfs_datasets_py.logic.backends.installers import proverif as installer_mod

    return installer_mod


@pytest.fixture(scope="module")
def proverif_cert():
    return _load_module(PROVERIF_CERT_PATH, "tools_logic_certification_proverif")


@pytest.fixture(scope="module")
def receipt(proverif_cert) -> dict[str, Any]:
    return proverif_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        env=proverif_cert.offline_env(os.environ),
    )


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert PROVERIF_CERT_PATH.is_file()
    assert Path(__file__).is_file()


def test_module_constants(installer, proverif_cert) -> None:
    assert installer.PROVERIF_VERSION == LOCKED_PROVERIF_VERSION
    assert installer.OPAM_VERSION == LOCKED_OPAM_VERSION
    assert installer.GOAL_ID == GOAL_ID
    assert installer.TASK_ID == TASK_ID
    assert installer.IMPORT_INSTALLS_FORBIDDEN is True

    assert proverif_cert.INTERFACE == INTERFACE
    assert proverif_cert.SCHEMA_VERSION == SCHEMA_VERSION
    assert proverif_cert.GOAL_ID == GOAL_ID
    assert proverif_cert.TASK_ID == TASK_ID
    assert proverif_cert.LOCKED_PROVERIF_VERSION == LOCKED_PROVERIF_VERSION
    assert proverif_cert.LOCKED_OPAM_VERSION == LOCKED_OPAM_VERSION
    assert proverif_cert.LANE_ID == "protocol"
    assert proverif_cert.CERTIFICATION_SURFACE == "tools.logic.certification.proverif"
    assert proverif_cert.AUTHORITY_SCOPE == "protocol_verification_only"


# ---------------------------------------------------------------------------
# Installer: strict pin selection and isolated OPAM root
# ---------------------------------------------------------------------------


def test_strict_install_selects_opam_2_5_2_and_proverif_2_05(installer) -> None:
    lock = None
    if LOCK_PATH.is_file():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        assert lock["managed_pin_versions"]["proverif"] == LOCKED_PROVERIF_VERSION
        assert lock["managed_pin_versions"]["opam"] == LOCKED_OPAM_VERSION

    proverif_pin = installer.select_strict_pin(
        "proverif",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    opam_pin = installer.select_strict_pin(
        "opam",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    assert proverif_pin.version == LOCKED_PROVERIF_VERSION
    assert opam_pin.version == LOCKED_OPAM_VERSION
    assert len(proverif_pin.sha256) == 64
    assert len(opam_pin.sha256) == 64
    assert proverif_pin.artifact_url.startswith("https://")
    assert opam_pin.artifact_url.startswith("https://")


def test_strict_pin_rejects_wrong_version(installer) -> None:
    fake_lock = {
        "managed_pin_versions": {"proverif": "9.9.9"},
        "tools": [
            {
                "tool_id": "proverif",
                "pins": [
                    {
                        "tool_id": "proverif",
                        "version": "2.05",
                        "platform": "any",
                        "artifact_url": "https://example.invalid/proverif.tgz",
                        "sha256": "a" * 64,
                    }
                ],
            }
        ],
    }
    with pytest.raises(installer.ProVerifInstallerError):
        installer.select_strict_pin(
            "proverif",
            platform_key="linux-x86_64",
            lock=fake_lock,
            allow_source_fallback=False,
            allow_any_platform=False,
        )


def test_isolated_opam_root_contract(installer) -> None:
    repo_root = installer.default_isolated_opam_root(repo_root=REPO_ROOT)
    assert "opam-roots" in str(repo_root)
    assert "proverif" in str(repo_root)
    assert installer.is_forbidden_global_opam_root(Path.home() / ".opam") is True
    with pytest.raises(installer.ProVerifInstallerError):
        installer.assert_isolated_opam_root(Path.home() / ".opam")
    ok = installer.assert_isolated_opam_root(repo_root)
    assert ok == repo_root.resolve()
    env = installer.isolated_opam_env(repo_root)
    assert env["OPAMROOT"] == str(repo_root.resolve())
    assert env["OPAMROOT"] != str((Path.home() / ".opam").resolve())


def test_ensure_without_yes_is_blocked(
    installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer, "which_executable", lambda *_a, **_k: None)

    proverif = installer.ensure_proverif(
        yes=False,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        ensure_opam_first=True,
    )
    opam = installer.ensure_opam(
        yes=False,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert proverif.selected_version == LOCKED_PROVERIF_VERSION
    assert opam.selected_version == LOCKED_OPAM_VERSION
    assert proverif.status == "blocked"
    assert opam.status == "blocked"
    assert "yes_required" in proverif.reason_codes or "dry_run" in proverif.reason_codes
    assert "yes_required" in opam.reason_codes or "dry_run" in opam.reason_codes
    assert proverif.isolated_opam_root
    assert opam.isolated_opam_root
    assert not installer.is_forbidden_global_opam_root(proverif.isolated_opam_root)


def test_ensure_dry_run_with_yes_selects_locked_pins(installer) -> None:
    receipt = installer.ensure_proverif(
        yes=True,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert receipt.selected_version == LOCKED_PROVERIF_VERSION
    assert receipt.pin is not None
    assert receipt.pin["version"] == LOCKED_PROVERIF_VERSION
    assert receipt.bindings["opam_locked_version"] == LOCKED_OPAM_VERSION
    assert receipt.bindings["opam_is_support_only"] is True
    assert receipt.bindings["global_switch_mutation_forbidden"] is True
    assert receipt.phase == "dry_run"
    assert receipt.isolated_opam_root
    assert "opam-roots" in receipt.isolated_opam_root


def test_authorize_install_forbids_import_and_requires_yes(installer) -> None:
    with pytest.raises(installer.ProVerifInstallerError):
        installer.authorize_plugin_install(
            "proverif",
            yes=True,
            import_context=True,
        )
    with pytest.raises(installer.ProVerifInstallerError):
        installer.authorize_plugin_install(
            "opam",
            yes=False,
        )


def test_plugin_manifest_declares_support_boundary(installer) -> None:
    manifest = installer.plugin_manifest()
    assert manifest["family"] == "proverif"
    assert manifest["locked_versions"]["proverif"] == LOCKED_PROVERIF_VERSION
    assert manifest["locked_versions"]["opam"] == LOCKED_OPAM_VERSION
    assert manifest["opam_is_support_only"] is True
    assert manifest["opam_cannot_promote_protocol_lane"] is True
    assert manifest["isolated_opam_root_required"] is True
    assert manifest["global_switch_mutation_forbidden"] is True
    assert manifest["roles"]["opam"] == "support"
    assert manifest["roles"]["proverif"] == "authority"
    assert manifest["policy"]["never_on_import"] is True
    assert manifest["policy"]["requires_explicit_yes"] is True
    assert manifest["policy"]["strict_selects_locked_versions"] is True
    assert manifest["policy"]["never_mutate_global_opam_switch"] is True
    tool_ids = {entry["tool_id"] for entry in manifest["entries"]}
    assert "proverif" in tool_ids


# ---------------------------------------------------------------------------
# Certification corpus and offline semantics
# ---------------------------------------------------------------------------


def test_corpus_schema_and_required_cases(proverif_cert) -> None:
    manifest = proverif_cert.default_corpus_manifest()
    assert manifest["schema_version"] == CORPUS_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["locked_proverif_version"] == LOCKED_PROVERIF_VERSION
    assert manifest["locked_opam_version"] == LOCKED_OPAM_VERSION
    assert manifest["policy"]["opam_is_support_only"] is True
    assert manifest["policy"]["opam_cannot_promote_protocol_lane"] is True
    assert manifest["policy"]["isolated_opam_root_required"] is True
    assert manifest["policy"]["never_mutate_global_opam_switch"] is True
    assert manifest["policy"]["no_install"] is True
    assert manifest["policy"]["no_network"] is True

    cases = manifest["cases"]
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds
    case_ids = {case["case_id"] for case in cases}
    assert REQUIRED_CASE_IDS <= case_ids


def test_offline_semantic_cases_pass(proverif_cert) -> None:
    for case in proverif_cert.corpus_cases():
        outcome = proverif_cert.evaluate_corpus_case(case)
        assert outcome.matched is True, (
            f"{outcome.case_id}: expected {outcome.expect}, "
            f"got {outcome.status} ({outcome.reason_codes})"
        )


def test_secure_and_attack_and_mutations(proverif_cert) -> None:
    by_id = {
        case["case_id"]: proverif_cert.evaluate_corpus_case(case)
        for case in proverif_cert.corpus_cases()
    }
    assert by_id["secure_claims"].status == "secure"
    assert by_id["attack_trace"].status == "attack"
    assert by_id["attack_trace"].attack_trace is not None
    assert by_id["mutated_claim"].status in {"quarantined", "attack", "unknown"}
    assert by_id["mutated_claim"].status != "secure"
    assert by_id["mutated_model"].status == "attack"
    assert by_id["malformed_output"].status != "secure"
    assert by_id["cancelled_query"].status != "secure"
    assert by_id["cancelled_query"].status == "quarantined"
    assert by_id["version_mismatch"].status == "blocked"


def test_deterministic_replay_digests(proverif_cert) -> None:
    by_id = {
        case["case_id"]: proverif_cert.evaluate_corpus_case(case)
        for case in proverif_cert.corpus_cases()
    }
    secure = by_id["secure_claims"]
    replay = by_id["deterministic_replay"]
    assert secure.output_digest == replay.output_digest
    assert secure.status == replay.status == "secure"


def test_receipt_binds_model_claims_bounds_and_binaries(receipt: dict[str, Any]) -> None:
    bindings = receipt["bindings"]
    assert "model" in bindings or "theory" in bindings
    theory = bindings.get("model") or bindings["theory"]
    assert theory["ceiling"]["adversary_model"] == "dolev_yao"
    assert theory["bound_theories"]
    assert "claims" in bindings
    assert "secrecy" in bindings["claims"]["bound_claim_kinds"] or any(
        "secrecy" in item for item in bindings["claims"]["bound_claim_kinds"]
    )
    assert "bounds" in bindings
    assert bindings["bounds"]["network"] is False
    assert bindings["bounds"]["install"] is False
    assert "binaries" in bindings
    assert bindings["binaries"]["proverif"]["locked_version"] == LOCKED_PROVERIF_VERSION
    assert bindings["binaries"]["opam"]["locked_version"] == LOCKED_OPAM_VERSION
    assert bindings["binaries"]["opam"]["support_only"] is True
    assert bindings["binaries"]["opam"]["can_promote_protocol_lane"] is False
    assert bindings["authority"]["opam_is_support_only"] is True
    assert bindings["authority"]["scope"] == "protocol_verification_only"
    assert bindings["isolated_opam_root"]["global_switch_mutation_forbidden"] is True

    check = next(c for c in receipt["checks"] if c["check_id"] == "proverif.bindings")
    assert check["status"] == "passed"


def test_offline_policy_never_installs(receipt: dict[str, Any]) -> None:
    assert receipt["network_used"] is False
    assert receipt["install_attempted"] is False
    assert receipt["download_attempted"] is False
    assert receipt.get("global_opam_mutation_attempted") is False
    policy = receipt["policy"]
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert policy["opam_is_support_only"] is True
    assert policy["never_mutate_global_opam_switch"] is True
    assert policy["does_not_edit_central_certificate"] is True
    assert policy["does_not_edit_tamarin_lane"] is True

    offline = next(
        c for c in receipt["checks"] if c["check_id"] == "proverif.offline_policy"
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


def test_opam_cannot_promote_protocol_lane(proverif_cert, receipt: dict[str, Any]) -> None:
    boundary = proverif_cert.opam_cannot_promote_protocol_lane()
    assert boundary["support_only"] is True
    assert boundary["promotion_allowed"] is False
    assert boundary["can_satisfy_protocol_requirement"] is False
    assert boundary["blocks_alone"] is True
    assert boundary["role"] == "support"
    assert boundary["authority_ceiling"] == "none"

    check = next(
        c for c in receipt["checks"] if c["check_id"] == "opam.support_only_boundary"
    )
    assert check["status"] == "passed"


def test_version_mismatch_case_blocks(proverif_cert) -> None:
    case = next(
        item
        for item in proverif_cert.corpus_cases()
        if item["case_id"] == "version_mismatch"
    )
    outcome = proverif_cert.evaluate_corpus_case(case)
    assert outcome.status == "blocked"
    assert "locked_version_mismatch" in outcome.reason_codes


def test_live_identity_mismatch_blocks_production_cert(
    proverif_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        proverif_cert,
        "probe_proverif_identity",
        lambda **_kwargs: {
            "tool_id": "proverif",
            "path_present": True,
            "executable_path": "/fixture/proverif",
            "version_string": "ProVerif 2.00",
            "identity_probed": True,
            "version_match": False,
            "locked_version": LOCKED_PROVERIF_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    monkeypatch.setattr(
        proverif_cert,
        "probe_opam_identity",
        lambda **_kwargs: {
            "tool_id": "opam",
            "path_present": True,
            "executable_path": "/fixture/opam",
            "version_string": "2.1.0",
            "identity_probed": True,
            "version_match": False,
            "locked_version": LOCKED_OPAM_VERSION,
            "support_only": True,
            "can_promote_protocol_lane": False,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    cert = proverif_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.production_certified is False
    assert cert.promotion_blocked is True
    assert cert.proverif_version_match is False
    assert all(case.matched for case in cert.cases)


def test_production_certified_when_live_pair_usable(
    proverif_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        proverif_cert,
        "probe_proverif_identity",
        lambda **_kwargs: {
            "tool_id": "proverif",
            "path_present": True,
            "executable_path": "/fixture/proverif",
            "version_string": f"ProVerif {LOCKED_PROVERIF_VERSION}",
            "identity_probed": True,
            "version_match": True,
            "locked_version": LOCKED_PROVERIF_VERSION,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    monkeypatch.setattr(
        proverif_cert,
        "probe_opam_identity",
        lambda **_kwargs: {
            "tool_id": "opam",
            "path_present": True,
            "executable_path": "/fixture/opam",
            "version_string": LOCKED_OPAM_VERSION,
            "identity_probed": True,
            "version_match": True,
            "locked_version": LOCKED_OPAM_VERSION,
            "support_only": True,
            "can_promote_protocol_lane": False,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    cert = proverif_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.proverif_usable is True
    assert cert.opam_usable is True
    assert cert.isolated_root_validated is True
    assert cert.production_certified is True
    assert cert.promotion_blocked is False
    assert cert.block_reasons == []
    assert all(case.matched for case in cert.cases)


def test_lane_handler_binds_under_roles_without_editing_central_certificate(
    proverif_cert,
) -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface not present in this worktree")
    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles_for_proverif")
    policy = roles.build_role_aware_policy(register_placeholders=True)
    roles.bind_lane_handler(
        "protocol", proverif_cert.lane_handler, policy=policy, replace=True
    )
    handler = policy.get_lane_handler("protocol")
    assert callable(handler)
    result = handler(repo_root=REPO_ROOT)
    assert result["owner_module"] == "tools.logic.certification.proverif"
    assert result["handler_id"] == proverif_cert.HANDLER_ID
    assert result["lane_id"] == "protocol"
    assert "production_certified" in result
    assert result["authority_scope"] == "protocol_verification_only"
    assert result["opam_support_only"] is True
    assert result["opam_cannot_promote_alone"] is True


def test_receipt_digest_present(receipt: dict[str, Any]) -> None:
    assert receipt["interface"] == INTERFACE
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["receipt_digest_sha256"]
    assert len(receipt["receipt_digest_sha256"]) == 64
