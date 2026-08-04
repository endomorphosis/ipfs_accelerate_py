"""Replayed Runtime MTL monitor and advisor semantics (FVT-098 / FVT-G230).

``ReplayedMonitorAdvisorSemantics@1``

Acceptance covered:

* the independent Node/TypeScript Runtime MTL engine executes positive,
  negative, boundary, malformed, mutation, replay, timeout, and cross-runtime
  parity cases against the in-process monitor with disagreement quarantine;
* real ErgoAI and SymbolicAI execute positive/non-entailment/contradiction/
  mutation/replay/malformed/resource-bound advisory cases;
* exact package, lockfile, runtime, launcher, target, artifact, and executable
  identities are bound;
* Runtime MTL gains finite-trace authority only after parity, while advisors
  remain proposal-only until independent reconstruction;
* Stack and Temurin remain support-only and cannot satisfy public
  verification, semantic, or proof-authority requirements;
* offline replay never installs, downloads, or mutates ambient PATH /
  user-site / the source tree;
* hermetic/parser fixtures cannot satisfy the external Runtime MTL lane.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CERTIFIER_PATH = (
    REPO_ROOT
    / "tools"
    / "logic"
    / "certify_formal_verification_replayed_monitor_advisors.py"
)
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_replayed_monitor_advisor_semantics.json"
)
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
RUNTIME_MTL_EXTERNAL_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "runtime_mtl_external.py"
)
ADVISORS_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "advisors.py"

INTERFACE = "ReplayedMonitorAdvisorSemantics@1"
SCHEMA_VERSION = "formal-verification-replayed-monitor-advisor-semantics/v1"
GOAL_ID = "FVT-G230"
TASK_ID = "FVT-098"
PROGRAM = "formal-verification-tactician/replayed-monitor-advisor-semantics"

REQUIRED_RUNTIME_MTL_CATEGORIES = {
    "satisfied",
    "violated",
    "timestamp_boundary",
    "interval_mutation",
    "event_mutation",
    "shortest_violating_prefix",
    "malformed",
    "clean_prefix",
}
REQUIRED_RUNTIME_MTL_MUTATIONS = {"interval", "event"}
REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES = {
    "positive",
    "negative",
    "boundary",
    "malformed",
    "mutation",
    "replay",
    "timeout",
    "parity",
    "disagreement_quarantine",
}
REQUIRED_RUNTIME_MTL_IDENTITY_FIELDS = (
    "package_digest_sha256",
    "source_digest_sha256",
    "lockfile_digest_sha256",
    "runtime_digest_sha256",
    "launcher_digest_sha256",
    "launcher_target_digest_sha256",
    "executable_digest_sha256",
    "artifact_sha256",
)
REQUIRED_ERGOAI_CASE_KINDS = {
    "entailment",
    "non_entailment",
    "contradiction",
    "mutation",
    "replay",
    "malformed",
    "timeout",
    "resource_bound",
}
REQUIRED_SYMBOLICAI_CASE_KINDS = {
    "positive",
    "negative",
    "mutation",
    "replay",
    "malformed",
}

DEFAULT_SEALED_ROOT = Path(
    "/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers"
)
GENUINE_ERGOAI_ENV = "IPFS_DATASETS_PY_TEST_ERGOAI_MANAGED_ROOT"


def _load_module():
    assert CERTIFIER_PATH.is_file(), f"missing expected output: {CERTIFIER_PATH}"
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    name = "tools_logic_certify_formal_verification_replayed_monitor_advisors"
    spec = importlib.util.spec_from_file_location(name, CERTIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certifier():
    return _load_module()


@pytest.fixture(scope="module")
def sealed_root() -> Path | None:
    configured = os.environ.get("IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT", "").strip()
    candidate = Path(configured) if configured else DEFAULT_SEALED_ROOT
    if candidate.is_dir() and (candidate / "bin").is_dir():
        return candidate.resolve()
    return None


@pytest.fixture(scope="module")
def genuine_ergoai_root() -> Path | None:
    raw = os.environ.get(GENUINE_ERGOAI_ENV, "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            return path
    # Discover a non-hermetic managed ErgoAI tree when present under the
    # operator profile (provider-side evidence generation only).
    candidates = [
        Path.home()
        / ".local"
        / "share"
        / "ipfs_datasets_py"
        / "theorem-provers-ergoai-v10-runtime-binding-20260803",
        Path.home()
        / ".local"
        / "share"
        / "ipfs_datasets_py"
        / "theorem-provers-ergoai-v9-20260803",
    ]
    for candidate in candidates:
        identity = candidate / "advisors" / "ergoai" / "3.0" / "identity.json"
        runtime_tc = (
            candidate / "advisors" / "ergoai" / "3.0" / "runtime-toolchain-bin"
        )
        if not candidate.is_dir():
            continue
        if runtime_tc.is_dir() and not runtime_tc.is_symlink():
            return candidate.resolve()
        if identity.is_file():
            try:
                payload = json.loads(identity.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not bool(payload.get("is_hermetic_advisor_shim")):
                return candidate.resolve()
    return None


@pytest.fixture(scope="module")
def live_receipt(certifier, sealed_root, genuine_ergoai_root) -> dict[str, Any]:
    if sealed_root is None:
        pytest.skip("sealed managed prover root is not available")
    if genuine_ergoai_root is None:
        pytest.skip(
            "genuine ErgoAI managed root is not available "
            f"(set {GENUINE_ERGOAI_ENV})"
        )
    return certifier.certify_replayed_monitor_advisor_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=sealed_root,
        runtime_mtl_install_root=sealed_root,
        ergoai_install_root=genuine_ergoai_root,
        skip_install=True,
        force_install=False,
    )


@pytest.fixture(scope="module")
def receipt_document() -> dict[str, Any]:
    assert RECEIPT_PATH.is_file(), f"missing expected output: {RECEIPT_PATH}"
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _good_runtime_mtl_certificate() -> dict[str, Any]:
    digest = "a" * 64
    return {
        "certified": True,
        "interface": "ExternalRuntimeMTLVendorCertification@1",
        "schema_version": "external-runtime-mtl-vendor-certification/v1",
        "goal_id": "FVT-G210",
        "task_id": "FVT-056",
        "repair_task_id": "FVT-072",
        "authority_ceiling": "finite_trace",
        "certificate_digest_sha256": digest,
        "categories_exercised": sorted(REQUIRED_RUNTIME_MTL_CATEGORIES),
        "mutation_kinds": sorted(REQUIRED_RUNTIME_MTL_MUTATIONS),
        "policy": {
            "disagreement_quarantines_promotion": True,
            "independent_node_package_without_python_dispatch": True,
            "finite_trace_authority_only": True,
            "hermetic_parity_wrappers_cannot_satisfy_vendor": True,
        },
        "acceptance": {
            "locked_typescript_dependency_graph": True,
            "independent_node_package_without_python_dispatch": True,
            "package_source_lockfile_runtime_launcher_executable_artifact_digests_bound": True,
            "offline_certification_never_builds_or_downloads": True,
            "hermetic_parity_wrappers_cannot_satisfy_vendor": True,
            "finite_trace_authority_only": True,
        },
        "runtime_mtl_external": {
            "tool_id": "runtime-mtl-external",
            "version": "1.0.0-reviewed",
            "certified": True,
            "usable": True,
            "role": "authority",
            "authority_ceiling": "finite_trace",
            "is_vendor_build": True,
            "is_hermetic_parity_engine": False,
            "executable": (
                "/opt/ipfs-accelerate/formal-toolchains/x/bin/runtime-mtl"
            ),
            "package_identity": "@ipfs-datasets/logic-runtime-mtl",
            "package_digest_sha256": "b" * 64,
            "source_digest_sha256": "c" * 64,
            "lockfile_digest_sha256": "d" * 64,
            "runtime_digest_sha256": "e" * 64,
            "launcher_digest_sha256": "f" * 64,
            "launcher_target_digest_sha256": "1" * 64,
            "executable_digest_sha256": "2" * 64,
            "artifact_sha256": "3" * 64,
            "node_version": "18.19.1",
            "platform_id": "linux-aarch64",
            "checks": [
                {"check_id": "mtl.parity", "kind": "parity", "status": "passed"},
                {
                    "check_id": "mtl.disagreement",
                    "kind": "disagreement_quarantine",
                    "status": "passed",
                },
                {"check_id": "mtl.timeout", "kind": "timeout", "status": "passed"},
            ],
        },
        "hermetic_parity_shadow": {
            "is_hermetic_parity_engine": True,
            "is_vendor_build": False,
            "non_production_shadow_evidence": True,
            "cannot_satisfy_vendor": True,
            "executable": "/tmp/hermetic-runtime-mtl",
        },
        "summary": {
            "vendor_certified": True,
            "checks_passed": 37,
            "checks_total": 37,
            "categories_exercised": sorted(REQUIRED_RUNTIME_MTL_CATEGORIES),
            "mutation_kinds": sorted(REQUIRED_RUNTIME_MTL_MUTATIONS),
            "block_reasons": [],
        },
    }


def _good_ergoai_certificate() -> dict[str, Any]:
    return {
        "interface": "ErgoAILiveToolchainContract@1",
        "schema_version": "ergoai-live-toolchain-contract/v1",
        "goal_id": "FVT-G218",
        "task_id": "FVT-085",
        "tool_id": "ergoai",
        "locked_version": "3.0",
        "contract_passed": True,
        "structural_passed": True,
        "semantic_passed": True,
        "live_vendor_execution": True,
        "managed_vendor_live_evidence": True,
        "vendor_certified": True,
        "is_hermetic_advisor_shim": False,
        "production_certified": False,
        "authority_ceiling": "advisory",
        "grants_proof_authority": False,
        "grants_theorem_authority": False,
        "case_kinds": sorted(REQUIRED_ERGOAI_CASE_KINDS),
        "block_reasons": [],
        "receipt_digest_sha256": "4" * 64,
        "executable": "/opt/ipfs-accelerate/formal-toolchains/x/bin/ergoai",
        "host_platform": "linux-aarch64",
    }


def _good_symbolicai_certificate() -> dict[str, Any]:
    cases = []
    for kind in sorted(REQUIRED_SYMBOLICAI_CASE_KINDS):
        cases.append(
            {
                "case_id": f"symbolicai.{kind}",
                "kind": kind,
                "advisor_id": "symbolicai",
                "provider": "symai",
                "matched": True,
                "status": "passed",
            }
        )
    return {
        "interface": "AdvisorRoleCertification@1",
        "schema_version": "advisor-role-certification/v1",
        "goal_id": "FVT-G160",
        "task_id": "FVT-050",
        "production_certified": True,
        "semantic_corpus_passed": True,
        "authority_ceiling": "advisory",
        "grants_proof_authority": False,
        "locked_symbolicai_version": ">=1.14.0,<2.0.0",
        "cases": cases,
        "receipt_digest_sha256": "5" * 64,
    }


# ---------------------------------------------------------------------------
# Expected outputs / surface constants
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert CERTIFIER_PATH.is_file()
    assert RECEIPT_PATH.is_file()
    assert Path(__file__).is_file()
    assert LOCK_PATH.is_file()
    assert RUNTIME_MTL_EXTERNAL_PATH.is_file()
    assert ADVISORS_PATH.is_file()


def test_module_surface_constants(certifier) -> None:
    assert certifier.INTERFACE == INTERFACE
    assert certifier.SCHEMA_VERSION == SCHEMA_VERSION
    assert certifier.GOAL_ID == GOAL_ID
    assert certifier.TASK_ID == TASK_ID
    assert certifier.PROGRAM == PROGRAM
    assert set(certifier.REQUIRED_RUNTIME_MTL_CATEGORIES) == (
        REQUIRED_RUNTIME_MTL_CATEGORIES
    )
    assert set(certifier.REQUIRED_RUNTIME_MTL_MUTATIONS) == (
        REQUIRED_RUNTIME_MTL_MUTATIONS
    )
    assert set(certifier.REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES) == (
        REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES
    )
    assert tuple(certifier.REQUIRED_RUNTIME_MTL_IDENTITY_FIELDS) == (
        REQUIRED_RUNTIME_MTL_IDENTITY_FIELDS
    )
    assert set(certifier.REQUIRED_ERGOAI_CASE_KINDS) == REQUIRED_ERGOAI_CASE_KINDS
    assert set(certifier.REQUIRED_SYMBOLICAI_CASE_KINDS) == (
        REQUIRED_SYMBOLICAI_CASE_KINDS
    )
    assert certifier.RUNTIME_MTL_AUTHORITY_CEILING == "finite_trace"
    assert certifier.ADVISOR_AUTHORITY_CEILING == "advisory"
    assert certifier.SUPPORT_AUTHORITY_CEILING == "none"
    assert tuple(certifier.SUPPORT_TOOL_IDS) == ("stack", "temurin-jdk")
    assert (
        certifier.DEFAULT_RECEIPT_RELATIVE.as_posix()
        == "docs/architecture/formal_verification_replayed_monitor_advisor_semantics.json"
    )


def test_offline_env_blocks_install_and_network(certifier) -> None:
    env = certifier.offline_env({"PATH": "/usr/bin", "HOME": "/tmp/x"})
    assert env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_INSTALL"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_NETWORK"] == "1"
    assert env["FORMAL_VERIFICATION_REPLAYED_MONITOR_ADVISOR_OFFLINE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PIP_NO_INDEX"] == "1"


def test_force_install_with_skip_install_refused(certifier) -> None:
    with pytest.raises(certifier.ReplayedMonitorAdvisorError):
        certifier.certify_replayed_monitor_advisor_semantics(
            repo_root=REPO_ROOT,
            skip_install=True,
            force_install=True,
        )


def test_live_install_path_refused(certifier) -> None:
    with pytest.raises(certifier.ReplayedMonitorAdvisorError):
        certifier.certify_replayed_monitor_advisor_semantics(
            repo_root=REPO_ROOT,
            skip_install=False,
            force_install=False,
        )


# ---------------------------------------------------------------------------
# Checked-in receipt document
# ---------------------------------------------------------------------------


def test_checked_in_receipt_surface(receipt_document) -> None:
    assert receipt_document["schema_version"] == SCHEMA_VERSION
    assert receipt_document["interface"] == INTERFACE
    assert receipt_document["goal_id"] == GOAL_ID
    assert receipt_document["task_id"] == TASK_ID
    assert receipt_document["program"] == PROGRAM
    assert receipt_document["certified"] is True
    assert receipt_document["semantic_certification"] is True
    policy = receipt_document["policy"]
    assert policy["owns_runtime_mtl_and_advisor_replay_fanin"] is True
    assert policy["does_not_make_core_ergoai_depend_on_java"] is True
    assert policy["does_not_promote_advice_to_theorem_authority"] is True
    assert (
        policy["hermetic_parser_fixture_cannot_satisfy_external_runtime_mtl"] is True
    )
    assert policy["runtime_mtl_authority_ceiling"] == "finite_trace"
    assert policy["advisor_authority_ceiling"] == "advisory"
    assert policy["stack_and_temurin_support_only"] is True
    assert policy["finite_trace_authority_only_after_parity"] is True
    assert policy["advisors_proposal_only_until_independent_reconstruction"] is True


def test_checked_in_receipt_runtime_mtl_lane(receipt_document) -> None:
    runtime = receipt_document["runtime_mtl"]
    assert runtime["certified"] is True
    assert runtime["authority_ceiling"] == "finite_trace"
    assert runtime["finite_trace_authority_granted"] is True
    assert runtime["is_vendor_build"] is True
    assert runtime["is_hermetic_parity_engine"] is False
    assert runtime["hermetic_cannot_satisfy_vendor"] is True
    assert REQUIRED_RUNTIME_MTL_CATEGORIES <= set(runtime["categories_exercised"])
    assert REQUIRED_RUNTIME_MTL_MUTATIONS <= set(runtime["mutation_kinds"])
    assert REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES <= set(runtime["acceptance_axes"])
    identity = runtime["identity"]
    for field in REQUIRED_RUNTIME_MTL_IDENTITY_FIELDS:
        value = identity.get(field)
        assert value, field
        assert len(str(value).replace("sha256:", "")) == 64, field
    shadow = runtime["hermetic_parity_shadow"]
    assert shadow["cannot_satisfy_vendor"] is True
    assert shadow["is_hermetic_parity_engine"] is True
    assert shadow["is_vendor_build"] is False


def test_checked_in_receipt_advisors_lane(receipt_document) -> None:
    advisors = receipt_document["advisors"]
    assert advisors["certified"] is True
    assert advisors["authority_ceiling"] == "advisory"
    assert advisors["proposal_only_until_independent_reconstruction"] is True
    assert advisors["grants_proof_authority"] is False
    ergoai = advisors["ergoai"]
    assert ergoai["certified"] is True
    assert ergoai["live_vendor_execution"] is True
    assert ergoai["is_hermetic_advisor_shim"] is False
    assert ergoai["authority_ceiling"] == "advisory"
    assert ergoai["grants_proof_authority"] is False
    assert REQUIRED_ERGOAI_CASE_KINDS <= set(ergoai["case_kinds"])
    symbolicai = advisors["symbolicai"]
    assert symbolicai["certified"] is True
    assert symbolicai["authority_ceiling"] == "advisory"
    assert symbolicai["grants_proof_authority"] is False
    assert REQUIRED_SYMBOLICAI_CASE_KINDS <= set(symbolicai["case_kinds"])


def test_checked_in_receipt_support_lane(receipt_document) -> None:
    support = receipt_document["support"]
    assert support["certified"] is True
    assert support["support_only"] is True
    assert support["authority_ceiling"] == "none"
    assert support["cannot_satisfy_public_verification"] is True
    assert support["cannot_satisfy_semantic_authority"] is True
    assert support["cannot_satisfy_proof_authority"] is True
    tools = {item["tool_id"]: item for item in support["tools"]}
    assert set(tools) == {"stack", "temurin-jdk"}
    for tool_id, tool in tools.items():
        assert tool["support_only"] is True, tool_id
        assert tool["authority_ceiling"] == "none", tool_id
        assert tool["can_satisfy_public_verification"] is False, tool_id
        assert tool["can_satisfy_semantic_authority"] is False, tool_id
        assert tool["can_satisfy_proof_authority"] is False, tool_id


def test_checked_in_receipt_acceptance(receipt_document) -> None:
    acceptance = receipt_document["acceptance"]
    assert acceptance["goal_id"] == GOAL_ID
    assert acceptance["task_id"] == TASK_ID
    assert (
        acceptance[
            "independent_node_typescript_runtime_mtl_positive_negative_boundary_malformed_mutation_replay_timeout_parity"
        ]
        is True
    )
    assert (
        acceptance["cross_runtime_parity_with_disagreement_quarantine"] is True
    )
    assert acceptance["real_ergoai_and_symbolicai_advisory_cases"] is True
    assert (
        acceptance[
            "package_lockfile_runtime_launcher_target_artifact_executable_identities_bound"
        ]
        is True
    )
    assert (
        acceptance["runtime_mtl_finite_trace_authority_only_after_parity"] is True
    )
    assert (
        acceptance[
            "advisors_remain_proposal_only_until_independent_reconstruction"
        ]
        is True
    )
    assert (
        acceptance[
            "stack_and_temurin_support_only_cannot_satisfy_verification_semantic_or_proof"
        ]
        is True
    )


# ---------------------------------------------------------------------------
# Live managed-root replay (when genuine ErgoAI + sealed MTL are available)
# ---------------------------------------------------------------------------


def test_live_receipt_certified(live_receipt) -> None:
    assert live_receipt["schema_version"] == SCHEMA_VERSION
    assert live_receipt["interface"] == INTERFACE
    assert live_receipt["goal_id"] == GOAL_ID
    assert live_receipt["task_id"] == TASK_ID
    assert live_receipt["certified"] is True
    assert live_receipt["semantic_certification"] is True
    phase = live_receipt["certification_phase"]
    assert phase["offline"] is True
    assert phase["install"] is False
    assert phase["download"] is False
    assert phase["network"] is False
    assert phase["skip_install"] is True
    assert phase["ambient_path_mutated"] is False
    assert phase["source_tree_mutated"] is False


def test_live_runtime_mtl_lane_covers_acceptance(live_receipt) -> None:
    runtime = live_receipt["runtime_mtl"]
    assert runtime["certified"] is True
    assert runtime["authority_ceiling"] == "finite_trace"
    assert runtime["finite_trace_authority_granted"] is True
    assert runtime["forbids_theorem_authority"] is True
    assert runtime["is_vendor_build"] is True
    assert runtime["is_hermetic_parity_engine"] is False
    assert REQUIRED_RUNTIME_MTL_CATEGORIES <= set(runtime["categories_exercised"])
    assert REQUIRED_RUNTIME_MTL_MUTATIONS <= set(runtime["mutation_kinds"])
    assert REQUIRED_RUNTIME_MTL_ACCEPTANCE_AXES <= set(runtime["acceptance_axes"])
    assert runtime["checks_passed"] == runtime["checks_total"]
    assert runtime["checks_total"] >= 1
    identity = runtime["identity"]
    for field in REQUIRED_RUNTIME_MTL_IDENTITY_FIELDS:
        assert identity.get(field), field
        assert len(str(identity[field]).replace("sha256:", "")) == 64


def test_live_advisors_lane_covers_acceptance(live_receipt) -> None:
    advisors = live_receipt["advisors"]
    assert advisors["certified"] is True
    assert advisors["authority_ceiling"] == "advisory"
    assert advisors["grants_proof_authority"] is False
    ergoai = advisors["ergoai"]
    assert ergoai["certified"] is True
    assert ergoai["live_vendor_execution"] is True
    assert ergoai["is_hermetic_advisor_shim"] is False
    assert REQUIRED_ERGOAI_CASE_KINDS <= set(ergoai["case_kinds"])
    symbolicai = advisors["symbolicai"]
    assert symbolicai["certified"] is True
    assert REQUIRED_SYMBOLICAI_CASE_KINDS <= set(symbolicai["case_kinds"])


def test_live_support_lane(live_receipt) -> None:
    support = live_receipt["support"]
    assert support["certified"] is True
    assert support["cannot_satisfy_proof_authority"] is True
    tools = {item["tool_id"]: item for item in support["tools"]}
    assert "stack" in tools
    assert "temurin-jdk" in tools
    assert tools["stack"]["support_only"] is True
    assert tools["temurin-jdk"]["support_only"] is True


def test_live_acceptance_flags(live_receipt) -> None:
    acceptance = live_receipt["acceptance"]
    assert acceptance["goal_id"] == GOAL_ID
    assert all(bool(value) for key, value in acceptance.items() if key not in {"goal_id", "task_id"})


def test_live_receipt_digest_is_stable(certifier, live_receipt) -> None:
    basis = {
        key: value
        for key, value in live_receipt.items()
        if key not in {"receipt_digest_sha256", "certificate_digest_sha256"}
    }
    recomputed = certifier.content_digest(basis)
    assert live_receipt["receipt_digest_sha256"] == recomputed
    assert live_receipt["certificate_digest_sha256"] == recomputed


def test_certify_never_mutates_source_tree(
    certifier, sealed_root, genuine_ergoai_root, tmp_path: Path
) -> None:
    if sealed_root is None:
        pytest.skip("sealed managed prover root is not available")
    if genuine_ergoai_root is None:
        pytest.skip("genuine ErgoAI managed root is not available")
    before = {
        path: path.stat().st_mtime_ns
        for path in (
            LOCK_PATH,
            CERTIFIER_PATH,
            RUNTIME_MTL_EXTERNAL_PATH,
            ADVISORS_PATH,
        )
        if path.is_file()
    }
    out = tmp_path / "receipt.json"
    receipt = certifier.certify_replayed_monitor_advisor_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=sealed_root,
        runtime_mtl_install_root=sealed_root,
        ergoai_install_root=genuine_ergoai_root,
        write_receipt_path=out,
    )
    assert receipt["certified"] is True
    assert out.is_file()
    for path, mtime in before.items():
        assert path.stat().st_mtime_ns == mtime


def test_sealed_root_runtime_mtl_lane_without_genuine_ergoai(
    certifier, sealed_root
) -> None:
    """Sealed root alone must certify Runtime MTL and fail closed on hermetic ErgoAI."""

    if sealed_root is None:
        pytest.skip("sealed managed prover root is not available")
    receipt = certifier.certify_replayed_monitor_advisor_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=sealed_root,
        runtime_mtl_install_root=sealed_root,
        # Force hermetic-only ErgoAI path by pointing at the sealed root and
        # not providing a genuine override.
        ergoai_install_root=sealed_root,
        skip_install=True,
    )
    assert receipt["runtime_mtl"]["certified"] is True
    assert receipt["runtime_mtl"]["finite_trace_authority_granted"] is True
    assert receipt["support"]["certified"] is True
    # Hermetic ErgoAI under the sealed advisors tree must not pass.
    ergoai = receipt["advisors"]["ergoai"]
    assert ergoai["certified"] is False
    assert receipt["certified"] is False
    assert any(
        "hermetic" in reason or "live_vendor" in reason
        for reason in (ergoai.get("block_reasons") or [])
    )


# ---------------------------------------------------------------------------
# Injected-certificate isolation / fail-closed rules
# ---------------------------------------------------------------------------


def test_lane_failure_isolation_with_injected_certificates(certifier) -> None:
    managed = DEFAULT_SEALED_ROOT if DEFAULT_SEALED_ROOT.is_dir() else REPO_ROOT
    good_mtl = _good_runtime_mtl_certificate()
    good_ergo = _good_ergoai_certificate()
    good_symai = _good_symbolicai_certificate()

    both = certifier.certify_replayed_monitor_advisor_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=managed,
        runtime_mtl_certificate=good_mtl,
        ergoai_certificate=good_ergo,
        symbolicai_certificate=good_symai,
    )
    assert both["runtime_mtl"]["certified"] is True
    assert both["advisors"]["certified"] is True
    assert both["support"]["certified"] is True
    assert both["certified"] is True
    assert both["runtime_mtl"]["finite_trace_authority_granted"] is True

    bad_mtl = copy.deepcopy(good_mtl)
    bad_mtl["runtime_mtl_external"]["is_hermetic_parity_engine"] = True
    bad_mtl["runtime_mtl_external"]["is_vendor_build"] = False
    mtl_fail = certifier.certify_replayed_monitor_advisor_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=managed,
        runtime_mtl_certificate=bad_mtl,
        ergoai_certificate=good_ergo,
        symbolicai_certificate=good_symai,
    )
    assert mtl_fail["runtime_mtl"]["certified"] is False
    assert mtl_fail["advisors"]["certified"] is True
    assert mtl_fail["certified"] is False
    assert any(
        "hermetic" in reason
        for reason in mtl_fail["runtime_mtl"]["block_reasons"]
    )

    bad_ergo = copy.deepcopy(good_ergo)
    bad_ergo["is_hermetic_advisor_shim"] = True
    bad_ergo["live_vendor_execution"] = False
    ergo_fail = certifier.certify_replayed_monitor_advisor_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=managed,
        runtime_mtl_certificate=good_mtl,
        ergoai_certificate=bad_ergo,
        symbolicai_certificate=good_symai,
    )
    assert ergo_fail["runtime_mtl"]["certified"] is True
    assert ergo_fail["advisors"]["ergoai"]["certified"] is False
    assert ergo_fail["certified"] is False

    elevated = copy.deepcopy(good_ergo)
    elevated["grants_proof_authority"] = True
    proof_fail = certifier.certify_replayed_monitor_advisor_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=managed,
        runtime_mtl_certificate=good_mtl,
        ergoai_certificate=elevated,
        symbolicai_certificate=good_symai,
    )
    assert proof_fail["advisors"]["ergoai"]["certified"] is False
    assert "ergoai_incorrectly_grants_proof_authority" in (
        proof_fail["advisors"]["ergoai"]["block_reasons"]
    )
    assert proof_fail["certified"] is False

    # Missing identity bindings fail closed.
    unbound = copy.deepcopy(good_mtl)
    unbound["runtime_mtl_external"]["package_digest_sha256"] = None
    unbound["runtime_mtl_external"]["lockfile_digest_sha256"] = ""
    unbound_fail = certifier.certify_replayed_monitor_advisor_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=managed,
        runtime_mtl_certificate=unbound,
        ergoai_certificate=good_ergo,
        symbolicai_certificate=good_symai,
    )
    assert unbound_fail["runtime_mtl"]["certified"] is False
    assert any(
        "identity_unbound" in reason
        for reason in unbound_fail["runtime_mtl"]["block_reasons"]
    )


def test_support_tools_cannot_satisfy_authority(certifier) -> None:
    managed = DEFAULT_SEALED_ROOT if DEFAULT_SEALED_ROOT.is_dir() else REPO_ROOT
    receipt = certifier.certify_replayed_monitor_advisor_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=managed,
        runtime_mtl_certificate=_good_runtime_mtl_certificate(),
        ergoai_certificate=_good_ergoai_certificate(),
        symbolicai_certificate=_good_symbolicai_certificate(),
    )
    support = receipt["support"]
    assert support["cannot_satisfy_public_verification"] is True
    assert support["cannot_satisfy_semantic_authority"] is True
    assert support["cannot_satisfy_proof_authority"] is True
    for tool in support["tools"]:
        assert tool["role"] == "support"
        assert tool["support_only"] is True
        assert tool["can_satisfy_proof_authority"] is False


def test_checked_in_receipt_aligns_with_live_shape(
    receipt_document, live_receipt
) -> None:
    for key in (
        "schema_version",
        "interface",
        "goal_id",
        "task_id",
        "program",
        "handler_id",
    ):
        assert receipt_document[key] == live_receipt[key]
    assert receipt_document["runtime_mtl"]["authority_ceiling"] == (
        live_receipt["runtime_mtl"]["authority_ceiling"]
    )
    assert receipt_document["advisors"]["authority_ceiling"] == (
        live_receipt["advisors"]["authority_ceiling"]
    )
    assert set(receipt_document["support"]["tool_ids"]) == set(
        live_receipt["support"]["tool_ids"]
    )
