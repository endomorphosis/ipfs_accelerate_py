"""Rocq/Coq + isolated OPAM toolchain installation and certification (FVT-045 / FVT-G150).

Covers:

* installer plugin selects exact Rocq/Coq 9.1.1 (package ``rocq-prover.9.1.1``)
  and OPAM 2.5.2 under strict mode;
* repository-local isolated OPAM root contract (never global switch mutation);
* true proof, false proof, hypothesis/conclusion mutation, deterministic replay,
  admit/Admitted/Axiom escapes, malformed input, and version-mismatch cases
  pass offline via the certification corpus;
* receipts bind imports, source, theorem, assumptions, and exact kernel identity;
* OPAM is support only and cannot promote the kernel lane by itself;
* certification never installs, downloads, or opens the network;
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
    / "rocq.py"
)
ROCQ_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "rocq.py"
ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "RocqToolchainCertification@1"
SCHEMA_VERSION = "rocq-toolchain-certification/v1"
CORPUS_SCHEMA = "rocq-toolchain-corpus/v1"
GOAL_ID = "FVT-G150"
TASK_ID = "FVT-045"
LOCKED_VERSION = "9.1.1"
LOCKED_OPAM_VERSION = "2.5.2"
PACKAGE_IDENTITY = "rocq-prover.9.1.1"

REQUIRED_CASE_KINDS = {
    "positive",
    "negative",
    "mutation",
    "replay",
    "fail_closed",
    "malformed",
    "version_mismatch",
}

REQUIRED_CASE_IDS = {
    "true_theorem",
    "false_proof",
    "hypothesis_mutation",
    "conclusion_mutation",
    "deterministic_replay",
    "admit_escape",
    "admitted_escape",
    "axiom_escape",
    "malformed_input",
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
    from ipfs_datasets_py.logic.backends.installers import rocq as installer_mod

    return installer_mod


@pytest.fixture(scope="module")
def rocq_cert():
    return _load_module(ROCQ_CERT_PATH, "tools_logic_certification_rocq")


@pytest.fixture(scope="module")
def receipt(rocq_cert) -> dict[str, Any]:
    return rocq_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        env=rocq_cert.offline_env(os.environ),
    )


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert ROCQ_CERT_PATH.is_file()
    assert Path(__file__).is_file()


def test_module_constants(installer, rocq_cert) -> None:
    assert installer.COQ_VERSION == LOCKED_VERSION
    assert installer.ROCQ_VERSION == LOCKED_VERSION
    assert installer.OPAM_VERSION == LOCKED_OPAM_VERSION
    assert installer.PACKAGE_IDENTITY == PACKAGE_IDENTITY
    assert installer.GOAL_ID == GOAL_ID
    assert installer.TASK_ID == TASK_ID
    assert installer.IMPORT_INSTALLS_FORBIDDEN is True

    assert rocq_cert.INTERFACE == INTERFACE
    assert rocq_cert.SCHEMA_VERSION == SCHEMA_VERSION
    assert rocq_cert.GOAL_ID == GOAL_ID
    assert rocq_cert.TASK_ID == TASK_ID
    assert rocq_cert.LOCKED_VERSION == LOCKED_VERSION
    assert rocq_cert.LOCKED_OPAM_VERSION == LOCKED_OPAM_VERSION
    assert rocq_cert.PACKAGE_IDENTITY == PACKAGE_IDENTITY
    assert rocq_cert.LANE_ID == "kernel"
    assert rocq_cert.CERTIFICATION_SURFACE == "tools.logic.certification.rocq"
    assert rocq_cert.AUTHORITY_SCOPE == "kernel_proof_checking_only"


# ---------------------------------------------------------------------------
# Installer: strict pin selection and isolated OPAM root
# ---------------------------------------------------------------------------


def test_strict_install_selects_rocq_9_1_1_and_opam_2_5_2(installer) -> None:
    lock = None
    if LOCK_PATH.is_file():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        assert lock["managed_pin_versions"]["coq"] == LOCKED_VERSION
        assert lock["managed_pin_versions"]["opam"] == LOCKED_OPAM_VERSION

    coq_pin = installer.select_strict_pin(
        "coq",
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
    assert coq_pin.version == LOCKED_VERSION
    assert opam_pin.version == LOCKED_OPAM_VERSION
    assert coq_pin.package_identity == PACKAGE_IDENTITY
    assert coq_pin.identity_kind == "opam_package"
    assert len(opam_pin.sha256) == 64
    assert opam_pin.artifact_url.startswith("https://")


def test_strict_pin_rejects_wrong_version(installer) -> None:
    fake_lock = {
        "managed_pin_versions": {"coq": "9.9.9"},
        "tools": [
            {
                "tool_id": "coq",
                "identity_kind": "opam_package",
                "pins": [
                    {
                        "tool_id": "rocq",
                        "version": "9.1.1",
                        "platform": "any",
                        "artifact_url": "",
                        "sha256": "",
                        "package_identity": "rocq-prover.9.1.1",
                    }
                ],
            }
        ],
    }
    with pytest.raises(installer.RocqInstallerError):
        installer.select_strict_pin(
            "coq",
            platform_key="linux-x86_64",
            lock=fake_lock,
            allow_source_fallback=False,
            allow_any_platform=False,
        )


def test_isolated_opam_root_contract(installer) -> None:
    repo_root = installer.default_isolated_opam_root(repo_root=REPO_ROOT)
    assert "opam-roots" in str(repo_root)
    assert "rocq" in str(repo_root)
    assert "proverif" not in str(repo_root)
    assert installer.is_forbidden_global_opam_root(Path.home() / ".opam") is True
    with pytest.raises(installer.RocqInstallerError):
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
    monkeypatch.setattr(installer, "resolve_coq_executable", lambda **_k: None)

    coq = installer.ensure_coq(
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
    assert coq.selected_version == LOCKED_VERSION
    assert opam.selected_version == LOCKED_OPAM_VERSION
    assert coq.status == "blocked"
    assert opam.status == "blocked"
    assert "yes_required" in coq.reason_codes or "dry_run" in coq.reason_codes
    assert "yes_required" in opam.reason_codes or "dry_run" in opam.reason_codes
    assert coq.isolated_opam_root
    assert opam.isolated_opam_root
    assert not installer.is_forbidden_global_opam_root(coq.isolated_opam_root)
    assert coq.package_identity == PACKAGE_IDENTITY


def test_ensure_dry_run_with_yes_selects_locked_pins(
    installer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer, "resolve_coq_executable", lambda **_k: None)

    receipt = installer.ensure_coq(
        yes=True,
        strict=True,
        dry_run=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert receipt.selected_version == LOCKED_VERSION
    assert receipt.pin is not None
    assert receipt.pin["version"] == LOCKED_VERSION
    assert receipt.pin["package_identity"] == PACKAGE_IDENTITY
    assert receipt.bindings["opam_locked_version"] == LOCKED_OPAM_VERSION
    assert receipt.bindings["opam_is_support_only"] is True
    assert receipt.bindings["global_switch_mutation_forbidden"] is True
    assert receipt.bindings["package_identity"] == PACKAGE_IDENTITY
    assert receipt.phase == "dry_run"
    assert receipt.isolated_opam_root
    assert "opam-roots" in receipt.isolated_opam_root
    assert "rocq" in receipt.isolated_opam_root


def test_authorize_install_forbids_import_and_requires_yes(installer) -> None:
    with pytest.raises(installer.RocqInstallerError):
        installer.authorize_plugin_install(
            "coq",
            yes=True,
            import_context=True,
        )
    with pytest.raises(installer.RocqInstallerError):
        installer.authorize_plugin_install(
            "opam",
            yes=False,
        )


def test_plugin_manifest_declares_support_boundary(installer) -> None:
    manifest = installer.plugin_manifest()
    assert manifest["family"] == "rocq"
    assert manifest["locked_versions"]["coq"] == LOCKED_VERSION
    assert manifest["locked_versions"]["opam"] == LOCKED_OPAM_VERSION
    assert manifest["package_identity"] == PACKAGE_IDENTITY
    assert manifest["opam_is_support_only"] is True
    assert manifest["opam_cannot_promote_kernel_lane"] is True
    assert manifest["isolated_opam_root_required"] is True
    assert manifest["global_switch_mutation_forbidden"] is True
    assert manifest["roles"]["opam"] == "support"
    assert manifest["roles"]["coq"] == "authority"
    assert manifest["policy"]["never_on_import"] is True
    assert manifest["policy"]["requires_explicit_yes"] is True
    assert manifest["policy"]["strict_selects_locked_versions"] is True
    assert manifest["policy"]["never_mutate_global_opam_switch"] is True
    tool_ids = {entry["tool_id"] for entry in manifest["entries"]}
    assert "coq" in tool_ids


def test_observed_version_matching(installer) -> None:
    assert installer.observed_version_matches_lock(
        "The Coq Proof Assistant, version 9.1.1", LOCKED_VERSION
    )
    assert installer.observed_version_matches_lock(
        "rocq-prover 9.1.1", LOCKED_VERSION
    )
    assert not installer.observed_version_matches_lock(
        "The Coq Proof Assistant, version 8.18.0", LOCKED_VERSION
    )


# ---------------------------------------------------------------------------
# Certification corpus and offline semantics
# ---------------------------------------------------------------------------


def test_corpus_schema_and_required_cases(rocq_cert) -> None:
    manifest = rocq_cert.default_corpus_manifest()
    assert manifest["schema_version"] == CORPUS_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["locked_version"] == LOCKED_VERSION
    assert manifest["locked_opam_version"] == LOCKED_OPAM_VERSION
    assert manifest["package_identity"] == PACKAGE_IDENTITY
    assert manifest["policy"]["opam_is_support_only"] is True
    assert manifest["policy"]["opam_cannot_promote_kernel_lane"] is True
    assert manifest["policy"]["isolated_opam_root_required"] is True
    assert manifest["policy"]["never_mutate_global_opam_switch"] is True
    assert manifest["policy"]["no_install"] is True
    assert manifest["policy"]["no_network"] is True
    assert manifest["policy"]["authority_is_kernel_proof_checking_only"] is True

    cases = manifest["cases"]
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds
    case_ids = {case["case_id"] for case in cases}
    assert REQUIRED_CASE_IDS <= case_ids


def test_offline_semantic_cases_pass(rocq_cert) -> None:
    outcomes: dict[str, Any] = {}
    for case in rocq_cert.corpus_cases():
        outcome = rocq_cert.evaluate_corpus_case(
            case, reference_outcomes=outcomes
        )
        outcomes[outcome.case_id] = outcome
        assert outcome.matched is True, (
            f"{outcome.case_id}: expected {outcome.expect}, "
            f"got {outcome.status} ({outcome.reason_codes})"
        )


def test_true_false_mutations_and_escapes(rocq_cert) -> None:
    outcomes: dict[str, Any] = {}
    by_id = {}
    for case in rocq_cert.corpus_cases():
        outcome = rocq_cert.evaluate_corpus_case(
            case, reference_outcomes=outcomes
        )
        outcomes[outcome.case_id] = outcome
        by_id[outcome.case_id] = outcome

    assert by_id["true_theorem"].accepted is True
    assert by_id["false_proof"].accepted is False
    assert by_id["hypothesis_mutation"].accepted is False
    assert by_id["conclusion_mutation"].accepted is False
    assert by_id["malformed_input"].accepted is False
    assert by_id["version_mismatch"].status == "blocked"
    assert "locked_version_mismatch" in by_id["version_mismatch"].reason_codes
    assert by_id["admit_escape"].accepted is False
    assert "admit_or_admitted" in by_id["admit_escape"].reason_codes
    assert by_id["admitted_escape"].accepted is False
    assert "admit_or_admitted" in by_id["admitted_escape"].reason_codes
    assert by_id["axiom_escape"].accepted is False
    assert (
        "unreviewed_axiom" in by_id["axiom_escape"].reason_codes
        or "open_assumptions_or_axioms" in by_id["axiom_escape"].reason_codes
    )


def test_deterministic_replay_digests(rocq_cert) -> None:
    outcomes: dict[str, Any] = {}
    by_id = {}
    for case in rocq_cert.corpus_cases():
        outcome = rocq_cert.evaluate_corpus_case(
            case, reference_outcomes=outcomes
        )
        outcomes[outcome.case_id] = outcome
        by_id[outcome.case_id] = outcome

    positive = by_id["true_theorem"]
    replay = by_id["deterministic_replay"]
    assert positive.accepted and replay.accepted
    assert positive.source_digest == replay.source_digest
    assert positive.output_digest == replay.output_digest


def test_source_scan_rejects_admit_and_axiom(rocq_cert) -> None:
    assert "admit_or_admitted" in rocq_cert.scan_rocq_incomplete_or_unsafe(
        "Theorem t : True.\nProof.\n  admit.\nQed.\n"
    )
    assert "admit_or_admitted" in rocq_cert.scan_rocq_incomplete_or_unsafe(
        "Theorem t : True.\nAdmitted.\n"
    )
    assert "unreviewed_axiom" in rocq_cert.scan_rocq_incomplete_or_unsafe(
        "Axiom bad : False.\nTheorem t : False.\nProof. exact bad. Qed.\n"
    )
    assert rocq_cert.scan_rocq_incomplete_or_unsafe(
        "Theorem from_eq : forall n m : nat, n = m -> n = m.\n"
        "Proof. intros n m H. exact H. Qed.\n"
    ) == ()


def test_receipt_binds_imports_source_theorem_assumptions_kernel(
    receipt: dict[str, Any],
) -> None:
    bindings = receipt["bindings"]
    assert "imports" in bindings
    assert "source" in bindings
    assert bindings["source"]["source_digest"]
    assert bindings["source"]["primary_path"].endswith(".v")
    assert "theorem" in bindings
    assert bindings["theorem"]["name"]
    assert "assumptions" in bindings
    assert "kernel_identity" in bindings
    assert bindings["kernel_identity"]["tool_id"] == "coq"
    assert bindings["kernel_identity"]["locked_version"] == LOCKED_VERSION
    assert bindings["kernel_identity"]["package_identity"] == PACKAGE_IDENTITY
    assert "binaries" in bindings
    assert bindings["binaries"]["coq"]["locked_version"] == LOCKED_VERSION
    assert bindings["binaries"]["opam"]["locked_version"] == LOCKED_OPAM_VERSION
    assert bindings["binaries"]["opam"]["support_only"] is True
    assert bindings["binaries"]["opam"]["can_promote_kernel_lane"] is False
    assert bindings["authority"]["scope"] == "kernel_proof_checking_only"
    assert bindings["authority"]["opam_is_support_only"] is True
    assert bindings["isolated_opam_root"]["global_switch_mutation_forbidden"] is True

    check = next(c for c in receipt["checks"] if c["check_id"] == "rocq.bindings")
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
    assert policy["does_not_edit_shared_lock"] is True

    offline = next(
        c for c in receipt["checks"] if c["check_id"] == "rocq.offline_policy"
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


def test_opam_cannot_promote_kernel_lane(rocq_cert, receipt: dict[str, Any]) -> None:
    boundary = rocq_cert.opam_cannot_promote_kernel_lane()
    assert boundary["support_only"] is True
    assert boundary["promotion_allowed"] is False
    assert boundary["can_satisfy_kernel_requirement"] is False
    assert boundary["blocks_alone"] is True
    assert boundary["role"] == "support"
    assert boundary["authority_ceiling"] == "none"

    check = next(
        c for c in receipt["checks"] if c["check_id"] == "opam.support_only_boundary"
    )
    assert check["status"] == "passed"


def test_version_mismatch_case_blocks(rocq_cert) -> None:
    case = next(
        item
        for item in rocq_cert.corpus_cases()
        if item["case_id"] == "version_mismatch"
    )
    outcome = rocq_cert.evaluate_corpus_case(case)
    assert outcome.status == "blocked"
    assert "locked_version_mismatch" in outcome.reason_codes


def test_live_identity_mismatch_blocks_production_cert(
    rocq_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rocq_cert,
        "probe_rocq_identity",
        lambda **_kwargs: {
            "tool_id": "coq",
            "path_present": True,
            "executable_path": "/fixture/coqc",
            "version_string": "The Coq Proof Assistant, version 8.18.0",
            "identity_probed": True,
            "installed": True,
            "version_match": False,
            "locked_version": LOCKED_VERSION,
            "package_identity": PACKAGE_IDENTITY,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    monkeypatch.setattr(
        rocq_cert,
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
            "can_promote_kernel_lane": False,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": "locked_version_mismatch",
        },
    )
    cert = rocq_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.production_certified is False
    assert cert.promotion_blocked is True
    assert cert.version_match is False
    assert cert.semantic_corpus_passed is True
    assert all(case.matched for case in cert.cases)


def test_production_certified_when_live_pin_usable(
    rocq_cert, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rocq_cert,
        "probe_rocq_identity",
        lambda **_kwargs: {
            "tool_id": "coq",
            "path_present": True,
            "executable_path": "/fixture/coqc",
            "version_string": f"The Coq Proof Assistant, version {LOCKED_VERSION}",
            "identity_probed": True,
            "installed": True,
            "version_match": True,
            "locked_version": LOCKED_VERSION,
            "package_identity": PACKAGE_IDENTITY,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    monkeypatch.setattr(
        rocq_cert,
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
            "can_promote_kernel_lane": False,
            "network_used": False,
            "install_attempted": False,
            "download_attempted": False,
            "probe_error": None,
        },
    )
    cert = rocq_cert.run_certification_suite(repo_root=REPO_ROOT)
    assert cert.usable is True
    assert cert.version_match is True
    assert cert.semantic_corpus_passed is True
    assert cert.isolated_root_validated is True
    assert cert.production_certified is True
    assert cert.promotion_blocked is False
    assert cert.block_reasons == []
    assert all(case.matched for case in cert.cases)
    assert cert.bindings["kernel_identity"]["locked_version"] == LOCKED_VERSION
    assert cert.bindings["kernel_identity"]["package_identity"] == PACKAGE_IDENTITY
    assert cert.bindings["theorem"]["name"]


def test_lane_handler_binds_under_roles_without_editing_central_certificate(
    rocq_cert,
) -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface not present in this worktree")
    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles_for_rocq")
    policy = roles.build_role_aware_policy(register_placeholders=True)
    roles.bind_lane_handler(
        "kernel", rocq_cert.lane_handler, policy=policy, replace=True
    )
    handler = policy.get_lane_handler("kernel")
    assert callable(handler)
    result = handler(repo_root=REPO_ROOT)
    assert result["owner_module"] == "tools.logic.certification.rocq"
    assert result["handler_id"] == rocq_cert.HANDLER_ID
    assert result["lane_id"] == "kernel"
    assert "production_certified" in result
    assert result["authority_scope"] == "kernel_proof_checking_only"


def test_receipt_digest_present(receipt: dict[str, Any]) -> None:
    assert receipt["interface"] == INTERFACE
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["receipt_digest_sha256"]
    assert len(receipt["receipt_digest_sha256"]) == 64
