"""Advisor role certification tests (FVT-050 / FVT-G160).

``AdvisorRoleCertification@1``

Covers:

* installer plugin selects locked SymbolicAI and ErgoAI identities under strict mode;
* SymAI, ErgoAI, Leanstral, autoencoder, and Hammer proposals are bounded,
  sanitized, source-bound, deterministic or replay-bound, and failure-explicit;
* confidence / similarity / generated text / advisor availability never become
  proof without deterministic compilation and independent solver/kernel validation;
* hammer-lane handler binds under the roles surface without editing the central
  multi-prover certificate;
* certification never installs over the network or mutates model runtimes.
"""

from __future__ import annotations

import hashlib
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
    / "advisors.py"
)
ADVISORS_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "advisors.py"
ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "AdvisorRoleCertification@1"
SCHEMA_VERSION = "advisor-role-certification/v1"
CORPUS_SCHEMA = "advisor-role-corpus/v1"
GOAL_ID = "FVT-G160"
TASK_ID = "FVT-050"
LANE_ID = "hammer"
HANDLER_ID = "advisor_role_certification@1"
LOCKED_SYMBOLICAI_VERSION = ">=1.14.0,<2.0.0"
LOCKED_ERGOAI_VERSION = "3.0"

ADVISOR_TOOL_IDS = {
    "symbolicai",
    "ergoai",
    "leanstral",
    "autoencoder",
    "hammer",
}

REQUIRED_CASE_KINDS = {
    "positive",
    "negative",
    "mutation",
    "replay",
    "malformed",
    "acceptance",
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
    from ipfs_datasets_py.logic.backends.installers import advisors as installer_mod

    return installer_mod


@pytest.fixture(scope="module")
def advisors_cert():
    return _load_module(ADVISORS_CERT_PATH, "tools_logic_certification_advisors")


@pytest.fixture(scope="module")
def install_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("advisor-role-cert-install")


@pytest.fixture(scope="module")
def receipt(advisors_cert, install_root) -> dict[str, Any]:
    return advisors_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        env=advisors_cert.offline_env(os.environ),
        install_root=install_root,
    )


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert INSTALLER_PATH.is_file()
    assert ADVISORS_CERT_PATH.is_file()
    assert Path(__file__).is_file()


def test_module_constants(installer, advisors_cert) -> None:
    assert installer.SYMBOLICAI_VERSION == LOCKED_SYMBOLICAI_VERSION
    assert installer.ERGOAI_VERSION == LOCKED_ERGOAI_VERSION
    assert installer.GOAL_ID == GOAL_ID
    assert installer.TASK_ID == TASK_ID
    assert installer.IMPORT_INSTALLS_FORBIDDEN is True
    assert set(installer.ADVISOR_INSTALL_TOOLS) == {"symbolicai", "ergoai"}

    assert advisors_cert.INTERFACE == INTERFACE
    assert advisors_cert.SCHEMA_VERSION == SCHEMA_VERSION
    assert advisors_cert.GOAL_ID == GOAL_ID
    assert advisors_cert.TASK_ID == TASK_ID
    assert advisors_cert.LOCKED_SYMBOLICAI_VERSION == LOCKED_SYMBOLICAI_VERSION
    assert advisors_cert.LOCKED_ERGOAI_VERSION == LOCKED_ERGOAI_VERSION
    assert advisors_cert.LANE_ID == LANE_ID
    assert advisors_cert.HANDLER_ID == HANDLER_ID
    assert advisors_cert.CERTIFICATION_SURFACE == "tools.logic.certification.advisors"
    assert advisors_cert.AUTHORITY_SCOPE == "candidate_generation_only"
    assert set(advisors_cert.ADVISOR_TOOL_IDS) == ADVISOR_TOOL_IDS


# ---------------------------------------------------------------------------
# Installer: strict pin selection and fail-closed policy
# ---------------------------------------------------------------------------


def test_strict_install_selects_locked_symbolicai_and_ergoai(installer) -> None:
    lock = None
    if LOCK_PATH.is_file():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        versions = lock.get("managed_pin_versions") or {}
        assert versions.get("symbolicai") == LOCKED_SYMBOLICAI_VERSION
        assert versions.get("ergoai") == LOCKED_ERGOAI_VERSION

    symai_pin = installer.select_strict_pin(
        "symbolicai",
        platform_key="any",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    ergo_pin = installer.select_strict_pin(
        "ergoai",
        platform_key="linux-x86_64",
        repo_root=REPO_ROOT,
        lock=lock,
    )
    assert symai_pin.version == LOCKED_SYMBOLICAI_VERSION
    assert ergo_pin.version == LOCKED_ERGOAI_VERSION
    assert symai_pin.identity_kind in {"python_package", "pypi_package"}
    assert ergo_pin.identity_kind in {
        "immutable_release_tag",
        "release_installer",
        "immutable_source_tag",
    }


def test_ergoai_linux_aarch64_pin_is_exact_and_checksummed(installer) -> None:
    pin = installer.select_strict_pin(
        "ergoai",
        platform_key="linux-aarch64",
        repo_root=REPO_ROOT,
        allow_source_fallback=False,
    )
    assert pin.version == LOCKED_ERGOAI_VERSION
    assert pin.platform == "linux-aarch64"
    assert pin.release_tag == "v3.0_release"
    assert pin.is_checksummed is True
    assert pin.sha256 == installer.ERGOAI_RELEASE_SHA256
    assert pin.artifact_size_bytes == installer.ERGOAI_RELEASE_SIZE_BYTES


def test_ergoai_unsupported_platform_fails_closed(installer, install_root) -> None:
    receipt = installer.ensure_ergoai(
        yes=True,
        strict=False,
        force=True,
        install_root=install_root / "unsupported-ergoai",
        repo_root=REPO_ROOT,
        platform_key="darwin-arm64",
        hermetic_shim=False,
    )
    assert not receipt.ok
    assert receipt.phase == "pin_selection"
    assert "pin_selection_failed" in receipt.reason_codes
    assert receipt.install_attempted is False


def test_ergoai_live_path_rejects_bad_prefetched_checksum(
    installer, install_root
) -> None:
    artifact = install_root / "bad-ergoAI_3.0.run"
    artifact.write_bytes(b"not the official release")
    receipt = installer.ensure_ergoai(
        yes=True,
        strict=False,
        force=True,
        install_root=install_root / "bad-checksum",
        repo_root=REPO_ROOT,
        platform_key="linux-aarch64",
        hermetic_shim=False,
        artifact_path=artifact,
    )
    assert not receipt.ok
    assert receipt.phase == "checksum"
    assert "download_or_checksum_failed" in receipt.reason_codes
    assert receipt.install_attempted is False


def test_ergoai_live_request_never_substitutes_offline_shim(
    installer, install_root, monkeypatch
) -> None:
    root = install_root / "offline-live-request"
    monkeypatch.setenv("FORMAL_VERIFICATION_CERTIFY_OFFLINE", "1")
    receipt = installer.ensure_ergoai(
        yes=True,
        strict=False,
        force=True,
        install_root=root,
        repo_root=REPO_ROOT,
        platform_key="linux-aarch64",
        hermetic_shim=False,
    )
    assert not receipt.ok
    assert receipt.phase == "offline_policy"
    assert "offline_policy_blocks_live_install" in receipt.reason_codes
    assert not (root / "advisors" / "ergoai" / "3.0" / "identity.json").exists()


def test_strict_pin_rejects_wrong_symbolicai_version(installer) -> None:
    fake_lock = {
        "managed_pin_versions": {"symbolicai": ">=9.9.0,<10.0.0"},
        "tools": [
            {
                "tool_id": "symbolicai",
                "pins": [
                    {
                        "tool_id": "symbolicai",
                        "version": ">=1.14.0,<2.0.0",
                        "platform": "any",
                        "artifact_url": "https://pypi.org/project/symbolicai/",
                        "sha256": "",
                        "identity_kind": "python_package",
                    }
                ],
            }
        ],
    }
    with pytest.raises(installer.AdvisorInstallerError):
        installer.select_strict_pin(
            "symbolicai",
            platform_key="any",
            lock=fake_lock,
            allow_source_fallback=False,
        )


def test_strict_pin_rejects_wrong_ergoai_version(installer) -> None:
    fake_lock = {
        "managed_pin_versions": {"ergoai": "9.9.9"},
        "tools": [
            {
                "tool_id": "ergoai",
                "pins": [
                    {
                        "tool_id": "ergoai",
                        "version": "3.0",
                        "platform": "linux-x86_64",
                        "artifact_url": "https://example.invalid/ergo.run",
                        "sha256": "",
                        "identity_kind": "immutable_release_tag",
                    }
                ],
            }
        ],
    }
    with pytest.raises(installer.AdvisorInstallerError):
        installer.select_strict_pin(
            "ergoai",
            platform_key="linux-x86_64",
            lock=fake_lock,
            allow_source_fallback=False,
        )


def test_ensure_without_yes_is_blocked(installer, install_root) -> None:
    # force=True so host package presence cannot short-circuit the yes gate.
    symai = installer.ensure_symbolicai(
        yes=False,
        strict=True,
        force=True,
        dry_run=False,
        install_root=install_root / "no-yes-symai",
        hermetic_marker=True,
        repo_root=REPO_ROOT,
    )
    ergo = installer.ensure_ergoai(
        yes=False,
        strict=True,
        force=True,
        dry_run=False,
        install_root=install_root / "no-yes-ergo",
        hermetic_shim=True,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert symai.selected_version == LOCKED_SYMBOLICAI_VERSION
    assert ergo.selected_version == LOCKED_ERGOAI_VERSION
    assert symai.status in {"blocked", "refused"}
    assert ergo.status in {"blocked", "refused"}
    assert "yes_required" in symai.reason_codes
    assert "yes_required" in ergo.reason_codes
    assert symai.grants_proof_authority is False
    assert ergo.grants_proof_authority is False


def test_ensure_dry_run_with_yes_selects_locked_pins(installer, install_root) -> None:
    symai = installer.ensure_symbolicai(
        yes=True,
        strict=True,
        force=True,
        dry_run=True,
        install_root=install_root / "dry-symai",
        repo_root=REPO_ROOT,
    )
    ergo = installer.ensure_ergoai(
        yes=True,
        strict=True,
        force=True,
        dry_run=True,
        install_root=install_root / "dry-ergo",
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
    )
    assert symai.selected_version == LOCKED_SYMBOLICAI_VERSION
    assert ergo.selected_version == LOCKED_ERGOAI_VERSION
    assert symai.pin is not None
    assert ergo.pin is not None
    assert symai.phase == "dry_run"
    assert ergo.phase == "dry_run"
    assert symai.install_attempted is False
    assert ergo.install_attempted is False


def test_ensure_hermetic_install_records_identities(installer, install_root) -> None:
    root = install_root / "hermetic"
    symai = installer.ensure_symbolicai(
        yes=True,
        strict=True,
        force=True,
        install_root=root,
        repo_root=REPO_ROOT,
        hermetic_marker=True,
        test_mode=True,
    )
    ergo = installer.ensure_ergoai(
        yes=True,
        strict=True,
        force=True,
        install_root=root,
        repo_root=REPO_ROOT,
        platform_key="linux-x86_64",
        hermetic_shim=True,
        test_mode=True,
    )
    assert symai.ok
    assert ergo.ok
    assert symai.selected_version == LOCKED_SYMBOLICAI_VERSION
    assert ergo.selected_version == LOCKED_ERGOAI_VERSION
    assert symai.role == "advisor"
    assert ergo.role == "advisor"
    assert symai.authority_ceiling == "advisory"
    assert ergo.authority_ceiling == "advisory"
    assert symai.grants_theorem_authority is False
    assert ergo.grants_theorem_authority is False
    assert Path(ergo.executable_path).is_file()
    # Version probe must report the pinned identity.
    import subprocess

    completed = subprocess.run(
        [ergo.executable_path, "--version"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    banner = (completed.stdout or "") + (completed.stderr or "")
    assert LOCKED_ERGOAI_VERSION in banner
    assert "ergoai" in banner.casefold() or "hermetic" in banner.casefold()


def test_live_ergoai_certifier_requires_provenance_and_real_semantics(
    installer, advisors_cert, install_root, monkeypatch
) -> None:
    root = install_root / "live-ergoai-fixture"
    executable = root / "bin" / "ergoai"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import re, sys\n"
        "if len(sys.argv) > 1 and sys.argv[1] in {'--version', '-v', 'version'}:\n"
        "    print('ErgoAI 3.0 (managed linux-aarch64; v3.0_release)')\n"
        "    raise SystemExit(0)\n"
        "data = sys.stdin.read()\n"
        "match = re.search(r\"load\\{'([^']+)'\\}\", data)\n"
        "program = Path(match.group(1)).read_text() if match else ''\n"
        "verdict = 'No' if 'fvt_ergo_absent' in data or 'fvt_ergo_mutated' in program else 'Yes'\n"
        "print('Yes')\n"
        "print(verdict)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    xsb = root / "advisors" / "ergoai" / "3.0" / "vendor" / "XSB" / "config" / "aarch64-unknown-linux-gnu" / "bin" / "xsb"
    xsb.parent.mkdir(parents=True)
    xsb.write_bytes(b"fixture-xsb-aarch64")
    xsb.chmod(0o755)
    release = root / "downloads" / "ergoAI_3.0.run"
    release.parent.mkdir(parents=True)
    release.write_bytes(b"fixture-official-release")
    release_digest = hashlib.sha256(release.read_bytes()).hexdigest()
    monkeypatch.setattr(installer, "ERGOAI_RELEASE_SHA256", release_digest)
    monkeypatch.setattr(installer, "ERGOAI_RELEASE_SIZE_BYTES", release.stat().st_size)

    identity_path = root / "advisors" / "ergoai" / "3.0" / "identity.json"
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": "ergoai-managed-vendor-identity/v1",
        "tool_id": "ergoai",
        "version": "3.0",
        "selected_platform": "linux-aarch64",
        "release_tag": "v3.0_release",
        "release_url": installer.ERGOAI_RELEASE_URL,
        "release_artifact_path": str(release),
        "release_artifact_sha256": release_digest,
        "release_artifact_size_bytes": release.stat().st_size,
        "vendor_executable": str(executable),
        "vendor_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "xsb_executable": str(xsb),
        "xsb_executable_sha256": hashlib.sha256(xsb.read_bytes()).hexdigest(),
        "xsb_configuration": "aarch64-unknown-linux-gnu",
        "launcher_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "identity_digest_sha256": "fixture-identity",
        "license_components": ["Apache-2.0", "LGPL-2.0"],
        "checksum_verified": True,
        "is_live_vendor": True,
        "is_hermetic_advisor_shim": False,
        "grants_proof_authority": False,
    }
    identity_path.write_text(json.dumps(identity), encoding="utf-8")

    receipt = advisors_cert.certify_live_ergoai_vendor(
        executable=executable,
        install_root=root,
        repo_root=REPO_ROOT,
        platform_key="linux-aarch64",
    )
    assert receipt["interface"] == "LiveErgoAIAdvisorCertification@1"
    assert receipt["vendor_certified"] is True
    assert receipt["authoritative_live_evidence"] is True
    assert receipt["evidence_class"] == "checksummed_authoritative_vendor_execution"
    assert receipt["grants_proof_authority"] is False
    assert receipt["promotion_blocked"] is True
    assert not receipt["block_reasons"]
    by_id = {check["check_id"]: check for check in receipt["checks"]}
    for kind in ("positive", "negative", "mutation", "replay"):
        assert by_id[f"advisors.ergoai_live.{kind}"]["status"] == "passed"

    xsb.write_bytes(b"tampered-xsb")
    rejected = advisors_cert.certify_live_ergoai_vendor(
        executable=executable,
        install_root=root,
        repo_root=REPO_ROOT,
        platform_key="linux-aarch64",
    )
    assert rejected["vendor_certified"] is False
    assert rejected["authoritative_live_evidence"] is False
    assert "xsb_executable_digest_mismatch" in rejected["block_reasons"]


def test_authorize_install_forbids_import_and_requires_yes(installer) -> None:
    with pytest.raises(installer.AdvisorInstallerError):
        installer.authorize_plugin_install(
            "symbolicai",
            yes=True,
            import_context=True,
        )
    with pytest.raises(installer.AdvisorInstallerError):
        installer.authorize_plugin_install(
            "ergoai",
            yes=False,
        )


def test_install_forbidden_on_import_receipt(installer, install_root) -> None:
    receipt = installer.ensure_symbolicai(
        yes=True,
        strict=False,
        force=True,
        install_root=install_root / "import-blocked",
        hermetic_marker=True,
        import_context=True,
        test_mode=True,
    )
    assert not receipt.ok
    assert "forbidden_on_import" in receipt.reason_codes or receipt.status in {
        "refused",
        "failed",
    }


def test_plugin_manifest_declares_advisor_boundary(installer) -> None:
    manifest = installer.plugin_manifest()
    assert manifest["family"] == "advisors"
    assert manifest["locked_versions"]["symbolicai"] == LOCKED_SYMBOLICAI_VERSION
    assert manifest["locked_versions"]["ergoai"] == LOCKED_ERGOAI_VERSION
    assert manifest["roles"]["symbolicai"] == "advisor"
    assert manifest["roles"]["ergoai"] == "advisor"
    assert manifest["authority_ceiling"] == "advisory"
    assert manifest["policy"]["never_on_import"] is True
    assert manifest["policy"]["requires_explicit_yes"] is True
    assert manifest["policy"]["strict_selects_locked_versions"] is True
    assert manifest["policy"]["advisors_are_candidate_generation_only"] is True
    assert manifest["policy"]["confidence_never_yields_proof"] is True
    assert manifest["policy"]["does_not_edit_central_certificate"] is True
    assert {entry["tool_id"] for entry in manifest["entries"]} >= {
        "symbolicai",
        "ergoai",
    }


def test_python_version_range_helper(installer) -> None:
    assert installer.python_version_satisfies_range("1.14.0", LOCKED_SYMBOLICAI_VERSION)
    assert installer.python_version_satisfies_range("1.99.0", LOCKED_SYMBOLICAI_VERSION)
    assert not installer.python_version_satisfies_range(
        "2.0.0", LOCKED_SYMBOLICAI_VERSION
    )
    assert not installer.python_version_satisfies_range(
        "1.13.9", LOCKED_SYMBOLICAI_VERSION
    )


# ---------------------------------------------------------------------------
# Certification corpus and offline semantics
# ---------------------------------------------------------------------------


def test_corpus_schema_and_required_cases(advisors_cert) -> None:
    manifest = advisors_cert.default_corpus_manifest()
    assert manifest["schema_version"] == CORPUS_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["locked_symbolicai_version"] == LOCKED_SYMBOLICAI_VERSION
    assert manifest["locked_ergoai_version"] == LOCKED_ERGOAI_VERSION
    assert set(manifest["advisor_tool_ids"]) == ADVISOR_TOOL_IDS
    assert manifest["policy"]["advisors_never_promote_alone"] is True
    assert manifest["policy"]["confidence_never_yields_proof"] is True
    assert manifest["policy"]["no_install"] is True
    assert manifest["policy"]["no_network"] is True

    cases = manifest["cases"]
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds
    # Every advisor has at least positive + negative cases.
    advisors_with_positive = {
        case["advisor_id"]
        for case in cases
        if case["kind"] == "positive"
    }
    assert ADVISOR_TOOL_IDS <= advisors_with_positive


def test_offline_semantic_cases_pass(advisors_cert) -> None:
    failures = []
    for case in advisors_cert.corpus_cases():
        outcome = advisors_cert.evaluate_corpus_case(case)
        if not outcome.matched:
            failures.append(
                f"{outcome.case_id}: expected {outcome.expect}, "
                f"got {outcome.status} ({outcome.reason_codes})"
            )
    assert not failures, "\n".join(failures)


def test_confidence_never_yields_proof_for_all_advisors(advisors_cert) -> None:
    by_id = {
        case["case_id"]: advisors_cert.evaluate_corpus_case(case)
        for case in advisors_cert.corpus_cases()
        if case["kind"] == "negative"
    }
    assert by_id
    for case_id, outcome in by_id.items():
        assert outcome.matched, case_id
        assert outcome.status == "not_proved"
        assert outcome.authority == "unverified_candidate_only"


def test_mutation_rejects_authority_claims(advisors_cert) -> None:
    mutations = [
        advisors_cert.evaluate_corpus_case(case)
        for case in advisors_cert.corpus_cases()
        if case["kind"] == "mutation"
    ]
    assert mutations
    for outcome in mutations:
        assert outcome.matched, (outcome.case_id, outcome.status, outcome.reason_codes)
        assert outcome.status != "unverified_candidate" or "authority" in str(
            outcome.reason_codes
        )


def test_deterministic_replay_digests(advisors_cert) -> None:
    replays = [
        advisors_cert.evaluate_corpus_case(case)
        for case in advisors_cert.corpus_cases()
        if case["kind"] == "replay"
    ]
    assert replays
    for outcome in replays:
        assert outcome.matched, (outcome.case_id, outcome.reason_codes)
        assert outcome.status == "deterministic_replay"
        assert outcome.output_digest
        assert len(outcome.output_digest) == 64


def test_acceptance_requires_compilation_and_validation(advisors_cert) -> None:
    case = next(
        item
        for item in advisors_cert.corpus_cases()
        if item["case_id"] == "shared.acceptance_requires_compilation_and_validation"
    )
    outcome = advisors_cert.evaluate_corpus_case(case)
    assert outcome.matched
    assert outcome.status == "acceptance_gate"
    assert outcome.bindings["rejected"]["accepted"] is False
    assert outcome.bindings["compile_only"]["accepted"] is False
    assert outcome.bindings["full"]["accepted"] is True


def test_role_matrix_blocks_all_advisors(advisors_cert) -> None:
    report = advisors_cert.all_advisor_role_boundaries()
    assert report["all_blocked_from_certified_authority"] is True
    for tool_id in ADVISOR_TOOL_IDS:
        boundary = report["tools"][tool_id]
        assert boundary["is_advisor_or_candidate"] is True
        # Leanstral/autoencoder use ceiling=candidate; SymAI/ErgoAI/Hammer use advisory.
        assert boundary["ceiling_is_non_certifying"] is True
        assert boundary["authority_ceiling"] in {"advisory", "candidate", "none"}
        assert boundary["can_satisfy_certified_authority"] is False
        assert boundary["can_satisfy_certified_authority_requirement"] is False
        assert boundary["promotion_allowed"] is False
        assert boundary["blocks_alone"] is True
        assert boundary["lane_id"] == LANE_ID
        assert LANE_ID in boundary["lane_ids"]


def test_advisors_cannot_promote_hammer_lane(advisors_cert) -> None:
    boundary = advisors_cert.advisors_cannot_promote_hammer_lane()
    assert boundary["lane_id"] == LANE_ID
    assert boundary["promotion_allowed"] is False
    assert boundary["all_blocked_from_certified_authority"] is True
    assert boundary["authority_tool_ids_for_lane"] == []


def test_receipt_binds_identities_and_authority(receipt: dict[str, Any]) -> None:
    assert receipt["interface"] == INTERFACE
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["lane_id"] == LANE_ID
    assert receipt["authority_scope"] == "candidate_generation_only"
    assert receipt["authority_ceiling"] == "advisory"
    assert receipt["advisors_never_promote_alone"] is True
    assert receipt["promotion_blocked"] is True
    assert receipt["semantic_corpus_passed"] is True
    assert receipt["role_matrix_passed"] is True
    assert receipt["install_identity_passed"] is True
    assert receipt["production_certified"] is True

    bindings = receipt["bindings"]
    assert bindings["authority"]["scope"] == "candidate_generation_only"
    assert bindings["authority"]["confidence_never_yields_proof"] is True
    assert bindings["locked_versions"]["symbolicai"] == LOCKED_SYMBOLICAI_VERSION
    assert bindings["locked_versions"]["ergoai"] == LOCKED_ERGOAI_VERSION
    assert set(bindings["advisor_tool_ids"]) == ADVISOR_TOOL_IDS

    check_ids = {c["check_id"] for c in receipt["checks"]}
    assert "advisors.offline_policy" in check_ids
    assert "advisors.role_matrix" in check_ids
    assert "advisors.install_identities" in check_ids
    assert "advisors.confidence_never_yields_proof" in check_ids
    assert "advisors.hammer_lane_promotion_blocked" in check_ids

    offline = next(c for c in receipt["checks"] if c["check_id"] == "advisors.offline_policy")
    assert offline["status"] == "passed"
    install = next(
        c for c in receipt["checks"] if c["check_id"] == "advisors.install_identities"
    )
    assert install["status"] == "passed"


def test_offline_policy_never_installs_over_network(receipt: dict[str, Any]) -> None:
    assert receipt["network_used"] is False
    assert receipt["download_attempted"] is False
    policy = receipt["policy"]
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert policy["advisors_never_promote_alone"] is True
    assert policy["does_not_edit_central_certificate"] is True
    assert policy["does_not_change_model_runtimes"] is True


def test_semantic_corpus_passed_in_receipt(receipt: dict[str, Any]) -> None:
    assert receipt["semantic_corpus_passed"] is True
    by_id = {case["case_id"]: case for case in receipt["cases"]}
    assert by_id
    for case in receipt["cases"]:
        assert case["matched"] is True, case["case_id"]
    # All corpus case checks must pass.
    for check in receipt["checks"]:
        if check["kind"] in REQUIRED_CASE_KINDS:
            assert check["status"] == "passed", check


def test_lane_handler_binds_under_roles_without_editing_central_certificate(
    advisors_cert,
) -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface not present in this worktree")
    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles_for_advisors")
    policy = roles.build_role_aware_policy(register_placeholders=True)
    roles.bind_lane_handler(
        "hammer", advisors_cert.lane_handler, policy=policy, replace=True
    )
    handler = policy.get_lane_handler("hammer")
    assert callable(handler)
    result = handler(repo_root=REPO_ROOT)
    assert result["owner_module"] == "tools.logic.certification.advisors"
    assert result["handler_id"] == HANDLER_ID
    assert result["lane_id"] == LANE_ID
    assert "production_certified" in result
    assert result["authority_scope"] == "candidate_generation_only"
    assert result["advisors_never_promote_alone"] is True
    assert result["promotion_blocked"] is True
    assert set(result["advisor_tool_ids"]) == ADVISOR_TOOL_IDS


def test_receipt_digest_present(receipt: dict[str, Any]) -> None:
    assert receipt["receipt_digest_sha256"]
    assert len(receipt["receipt_digest_sha256"]) == 64


def test_roles_surface_predeclares_hammer_owner() -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface not present in this worktree")
    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles_owner_check")
    assert roles.LANE_HANDLER_OWNERS.get("hammer") == "tools.logic.certification.advisors"
