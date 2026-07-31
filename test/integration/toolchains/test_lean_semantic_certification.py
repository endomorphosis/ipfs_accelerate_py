"""Semantic certification of the pinned Lean kernel (FVT-040 / FVT-G101).

Exercises ``tools/logic/certification/lean.py`` and the Lean corpus fixture.

Acceptance covered:

* exact Lean v4.31.0 compiles a true theorem;
* false and malformed proofs are rejected;
* hypothesis and conclusion mutations are rejected;
* deterministic replay of the positive case;
* receipts bind imports, source tree, theorem, assumptions, toolchain, and
  output;
* sorry, admit, unsafe escape, shim mismatch, install, download, and network
  use fail closed;
* resulting authority is kernel proof checking only.
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
LEAN_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "lean.py"
MANIFEST_PATH = (
    REPO_ROOT
    / "test"
    / "fixtures"
    / "formal_verification"
    / "toolchains"
    / "lean"
    / "manifest.json"
)
ROLES_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "roles.py"
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "LeanSemanticCertification@1"
SCHEMA_VERSION = "lean-semantic-certification/v1"
CORPUS_SCHEMA = "lean-semantic-corpus/v1"
GOAL_ID = "FVT-G101"
TASK_ID = "FVT-040"
LOCKED_TOOLCHAIN = "leanprover/lean4:v4.31.0"
LOCKED_VERSION = "v4.31.0"
LOCKED_VERSION_NUMERIC = "4.31.0"

REQUIRED_CASE_KINDS = {
    "positive",
    "negative",
    "mutation",
    "replay",
    "fail_closed",
}


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing expected output: {path}"
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lean_cert():
    return _load_module(LEAN_CERT_PATH, "tools_logic_certification_lean")


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.is_file(), f"missing corpus manifest: {MANIFEST_PATH}"
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def receipt(lean_cert, manifest) -> dict[str, Any]:
    env = lean_cert.offline_env(
        {
            **os.environ,
            "ELAN_TOOLCHAIN": LOCKED_TOOLCHAIN,
            "ELAN_NO_AUTO_INSTALL": "1",
        }
    )
    return lean_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        manifest=manifest,
        env=env,
    )


# ---------------------------------------------------------------------------
# Expected outputs / fixture contract
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert LEAN_CERT_PATH.is_file()
    assert MANIFEST_PATH.is_file()
    assert Path(__file__).is_file()


def test_manifest_schema_and_corpus(manifest: dict[str, Any]) -> None:
    assert manifest["schema_version"] == CORPUS_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["tool_id"] == "lean"
    assert manifest["lane_id"] == "kernel"
    assert manifest["locked_toolchain"] == LOCKED_TOOLCHAIN
    assert manifest["locked_version"] == LOCKED_VERSION
    assert manifest["authority_ceiling"] == "kernel"
    assert manifest["authority_scope"] == "kernel_proof_checking_only"

    policy = manifest["policy"]
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert policy["elan_no_auto_install"] is True
    assert policy["shim_mismatch_fails_closed"] is True
    assert policy["sorry_admit_unsafe_fail_closed"] is True
    assert policy["authority_is_kernel_proof_checking_only"] is True

    cases = manifest["cases"]
    assert isinstance(cases, list) and cases
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds
    case_ids = {case["case_id"] for case in cases}
    assert "true_theorem" in case_ids
    assert "false_proof" in case_ids
    assert "malformed_proof" in case_ids
    assert "hypothesis_mutation" in case_ids
    assert "conclusion_mutation" in case_ids
    assert "sorry_escape" in case_ids
    assert "admit_escape" in case_ids
    assert "unsafe_escape" in case_ids
    assert "axiom_escape" in case_ids
    assert "deterministic_replay" in case_ids
    for case in cases:
        assert case["source"].strip()
        assert case["expect"] in {"accepted", "rejected"}


def test_module_constants(lean_cert) -> None:
    assert lean_cert.INTERFACE == INTERFACE
    assert lean_cert.SCHEMA_VERSION == SCHEMA_VERSION
    assert lean_cert.GOAL_ID == GOAL_ID
    assert lean_cert.TASK_ID == TASK_ID
    assert lean_cert.LOCKED_TOOLCHAIN == LOCKED_TOOLCHAIN
    assert lean_cert.LOCKED_VERSION == LOCKED_VERSION
    assert lean_cert.AUTHORITY_SCOPE == "kernel_proof_checking_only"
    assert lean_cert.LANE_ID == "kernel"
    assert lean_cert.CERTIFICATION_SURFACE == "tools.logic.certification.lean"


# ---------------------------------------------------------------------------
# Source scanning (fail closed without kernel)
# ---------------------------------------------------------------------------


def test_source_scan_rejects_sorry_admit_unsafe(lean_cert) -> None:
    assert "sorry_or_admit" in lean_cert.scan_lean_incomplete_or_unsafe(
        "theorem t : True := by sorry\n"
    )
    assert "sorry_or_admit" in lean_cert.scan_lean_incomplete_or_unsafe(
        "theorem t : True := by admit\n"
    )
    assert "unsafe_or_unreviewed_axiom" in lean_cert.scan_lean_incomplete_or_unsafe(
        "unsafe def x : Nat := 0\ntheorem t : True := trivial\n"
    )
    assert "unsafe_or_unreviewed_axiom" in lean_cert.scan_lean_incomplete_or_unsafe(
        "axiom bad : False\ntheorem t : False := bad\n"
    )
    assert lean_cert.scan_lean_incomplete_or_unsafe(
        "theorem from_eq (n m : Nat) (h : n = m) : n = m := h\n"
    ) == ()


def test_shim_mismatch_detector(lean_cert) -> None:
    assert lean_cert.detect_lean_shim_toolchain_mismatch(
        LOCKED_TOOLCHAIN,
        ["leanprover/lean4:v4.32.2"],
    )
    assert not lean_cert.detect_lean_shim_toolchain_mismatch(
        LOCKED_TOOLCHAIN,
        [LOCKED_TOOLCHAIN, "leanprover/lean4:v4.32.2"],
    )


# ---------------------------------------------------------------------------
# Live semantic suite against pinned Lean v4.31.0
# ---------------------------------------------------------------------------


def test_identity_is_exact_pin(receipt: dict[str, Any]) -> None:
    if not receipt.get("identity_probed"):
        pytest.skip(f"Lean identity unavailable: {receipt.get('block_reasons')}")
    assert receipt["locked_toolchain"] == LOCKED_TOOLCHAIN
    assert receipt["locked_version"] == LOCKED_VERSION
    assert receipt["selected_toolchain"] == LOCKED_TOOLCHAIN
    assert receipt["version_string"]
    assert LOCKED_VERSION_NUMERIC in receipt["version_string"]
    assert receipt["locked_version_mismatch"] is False
    assert receipt["shim_toolchain_mismatch"] is False
    assert receipt["usable"] is True


def test_offline_policy_never_installs_or_networks(receipt: dict[str, Any]) -> None:
    assert receipt["network_used"] is False
    assert receipt["install_attempted"] is False
    assert receipt["download_attempted"] is False
    policy = receipt["policy"]
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert policy["elan_no_auto_install"] is True
    assert policy["does_not_edit_central_certificate"] is True
    assert policy["does_not_select_alternate_elan_toolchain"] is True

    offline = next(
        check for check in receipt["checks"] if check["check_id"] == "lean.offline_policy"
    )
    assert offline["status"] == "passed"
    assert "ELAN_NO_AUTO_INSTALL=1" in offline["observed"]
    assert f"ELAN_TOOLCHAIN={LOCKED_TOOLCHAIN}" in offline["observed"]


def test_true_theorem_accepted(receipt: dict[str, Any]) -> None:
    if not receipt.get("usable"):
        pytest.skip("Lean pin not usable in this environment")
    by_id = {case["case_id"]: case for case in receipt["cases"]}
    positive = by_id["true_theorem"]
    assert positive["accepted"] is True
    assert positive["returncode"] == 0
    assert positive["theorem_name"] == "from_eq"
    check = next(c for c in receipt["checks"] if c["check_id"] == "lean.true_theorem")
    assert check["status"] == "passed"
    assert check["kind"] == "positive"


def test_false_and_malformed_rejected(receipt: dict[str, Any]) -> None:
    if not receipt.get("usable"):
        pytest.skip("Lean pin not usable in this environment")
    by_id = {case["case_id"]: case for case in receipt["cases"]}
    for case_id in ("false_proof", "malformed_proof"):
        case = by_id[case_id]
        assert case["accepted"] is False
        assert case["returncode"] in {1, None} or "kernel_rejected" in case["reason_codes"]
        check = next(c for c in receipt["checks"] if c["check_id"] == f"lean.{case_id}")
        assert check["status"] == "passed"
        assert check["kind"] == "negative"
        assert check["observed"] == "rejected"


def test_hypothesis_and_conclusion_mutations_rejected(receipt: dict[str, Any]) -> None:
    if not receipt.get("usable"):
        pytest.skip("Lean pin not usable in this environment")
    by_id = {case["case_id"]: case for case in receipt["cases"]}
    for case_id in ("hypothesis_mutation", "conclusion_mutation"):
        case = by_id[case_id]
        assert case["accepted"] is False
        check = next(c for c in receipt["checks"] if c["check_id"] == f"lean.{case_id}")
        assert check["status"] == "passed"
        assert check["kind"] == "mutation"
        assert check["observed"] == "rejected"


def test_sorry_admit_unsafe_axiom_fail_closed(receipt: dict[str, Any]) -> None:
    if not receipt.get("usable"):
        pytest.skip("Lean pin not usable in this environment")
    by_id = {case["case_id"]: case for case in receipt["cases"]}
    for case_id, reason in (
        ("sorry_escape", "sorry_or_admit"),
        ("admit_escape", "sorry_or_admit"),
        ("unsafe_escape", "unsafe_or_unreviewed_axiom"),
        ("axiom_escape", "unsafe_or_unreviewed_axiom"),
    ):
        case = by_id[case_id]
        assert case["accepted"] is False
        assert reason in case["reason_codes"]
        check = next(c for c in receipt["checks"] if c["check_id"] == f"lean.{case_id}")
        assert check["status"] == "passed"
        assert check["kind"] == "fail_closed"


def test_deterministic_replay(receipt: dict[str, Any]) -> None:
    if not receipt.get("usable"):
        pytest.skip("Lean pin not usable in this environment")
    by_id = {case["case_id"]: case for case in receipt["cases"]}
    positive = by_id["true_theorem"]
    replay = by_id["deterministic_replay"]
    assert positive["accepted"] is True
    assert replay["accepted"] is True
    assert positive["source_digest"] == replay["source_digest"]
    assert positive["output_digest"] == replay["output_digest"]
    assert positive["returncode"] == replay["returncode"]

    binding = next(
        c
        for c in receipt["checks"]
        if c["check_id"] == "lean.deterministic_replay_binding"
    )
    assert binding["status"] == "passed"


def test_bindings_cover_trust_inputs(receipt: dict[str, Any]) -> None:
    if not receipt.get("usable"):
        pytest.skip("Lean pin not usable in this environment")
    bindings = receipt["bindings"]
    assert "imports" in bindings
    assert bindings["source_tree"]["primary_path"] == "Main.lean"
    assert bindings["source_tree"]["source_digest"]
    assert bindings["theorem"]["name"] == "from_eq"
    assert "h : n = m" in bindings["theorem"]["assumptions"]
    assert bindings["assumptions"] == bindings["theorem"]["assumptions"]
    toolchain = bindings["toolchain"]
    assert toolchain["locked_toolchain"] == LOCKED_TOOLCHAIN
    assert toolchain["locked_version"] == LOCKED_VERSION
    assert toolchain["selected_toolchain"] == LOCKED_TOOLCHAIN
    assert toolchain["executable_path"]
    assert toolchain["version_string"]
    assert bindings["output"]["output_digest"]
    assert bindings["authority"]["ceiling"] == "kernel"
    assert bindings["authority"]["scope"] == "kernel_proof_checking_only"
    assert bindings["authority"]["not_advisor"] is True
    assert bindings["authority"]["not_install_authority"] is True

    check = next(c for c in receipt["checks"] if c["check_id"] == "lean.bindings")
    assert check["status"] == "passed"


def test_production_certified_when_pin_usable(receipt: dict[str, Any]) -> None:
    if not receipt.get("usable"):
        pytest.skip("Lean pin not usable in this environment")
    assert receipt["interface"] == INTERFACE
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["authority_scope"] == "kernel_proof_checking_only"
    assert receipt["production_certified"] is True
    assert receipt["promotion_blocked"] is False
    assert receipt["block_reasons"] == []
    assert receipt["receipt_digest_sha256"]
    assert len(receipt["receipt_digest_sha256"]) == 64
    # Every corpus case check must pass under a certified receipt.
    for check in receipt["checks"]:
        if check["kind"] in REQUIRED_CASE_KINDS or check["check_id"] in {
            "lean.bindings",
            "lean.deterministic_replay_binding",
            "lean.identity",
            "lean.offline_policy",
        }:
            assert check["status"] == "passed", check


# ---------------------------------------------------------------------------
# Fail-closed synthetic paths (no alternate toolchain selection)
# ---------------------------------------------------------------------------


def test_shim_mismatch_fails_closed_without_download(
    lean_cert, manifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lean_cert,
        "list_elan_installed_toolchains",
        lambda _env=None: ["leanprover/lean4:v4.32.2"],
    )
    env = lean_cert.offline_env({"ELAN_TOOLCHAIN": LOCKED_TOOLCHAIN})
    cert = lean_cert.run_semantic_suite(
        repo_root=REPO_ROOT,
        manifest=manifest,
        env=env,
    )
    assert cert.shim_toolchain_mismatch is True
    assert cert.production_certified is False
    assert cert.promotion_blocked is True
    assert "shim_toolchain_mismatch" in cert.block_reasons
    assert cert.install_attempted is False
    assert cert.download_attempted is False
    assert cert.network_used is False
    # Must not have selected a different installed toolchain for certification.
    assert cert.selected_toolchain == LOCKED_TOOLCHAIN


def test_version_mismatch_blocks_certification(
    lean_cert, manifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lean_cert,
        "list_elan_installed_toolchains",
        lambda _env=None: [LOCKED_TOOLCHAIN],
    )
    monkeypatch.setattr(
        lean_cert,
        "resolve_lean_executable",
        lambda _candidates=None: "/fixture/lean",
    )

    def fake_run(argv, *, timeout, env, **_kwargs):
        return lean_cert.subprocess.CompletedProcess(
            argv,
            0,
            stdout="Lean (version 4.32.2, fixture)\n",
            stderr="",
        )

    monkeypatch.setattr(lean_cert, "bounded_run", fake_run)
    cert = lean_cert.run_semantic_suite(
        repo_root=REPO_ROOT,
        manifest=manifest,
        env=lean_cert.offline_env(),
        executable="/fixture/lean",
    )
    assert cert.locked_version_mismatch is True
    assert cert.production_certified is False
    assert cert.promotion_blocked is True
    assert "locked_version_mismatch" in cert.block_reasons


def test_lane_handler_binds_under_roles_without_editing_central_certificate(
    lean_cert,
) -> None:
    if not ROLES_PATH.is_file():
        pytest.skip("roles certification surface not present in this worktree")
    roles = _load_module(ROLES_PATH, "tools_logic_certification_roles")
    policy = roles.build_role_aware_policy(register_placeholders=True)
    roles.bind_lane_handler("kernel", lean_cert.lane_handler, policy=policy, replace=True)
    handler = policy.get_lane_handler("kernel")
    assert callable(handler)
    # Invocation returns a semantic receipt rather than the pending placeholder.
    result = handler(repo_root=REPO_ROOT)
    assert result["owner_module"] == "tools.logic.certification.lean"
    assert result["handler_id"] == lean_cert.HANDLER_ID
    assert result["lane_id"] == "kernel"
    assert "production_certified" in result
    assert result["authority_scope"] == "kernel_proof_checking_only"
    # Central certificate path is intentionally not rewritten by this lane.
    certificate = (
        REPO_ROOT / "docs" / "architecture" / "formal_verification_toolchain_certificate.json"
    )
    # Presence is optional for this task; the policy is that lean.py must not
    # be the writer. We only assert the certifier module path is not the
    # central certifier.
    assert lean_cert.CERTIFICATION_SURFACE != (
        "tools.logic.certify_formal_verification_toolchains"
    )
    assert LOCK_PATH.is_file()
    _ = certificate  # documentation of non-ownership


def test_offline_env_pins_locked_toolchain(lean_cert) -> None:
    env = lean_cert.offline_env({"PATH": "/bin"})
    assert env["ELAN_TOOLCHAIN"] == LOCKED_TOOLCHAIN
    assert env["ELAN_NO_AUTO_INSTALL"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_INSTALL"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_NETWORK"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_DOWNLOAD"] == "1"
