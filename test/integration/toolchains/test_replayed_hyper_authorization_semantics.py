"""Replayed hyperproperty and external authorization semantics (FVT-097 / FVT-G229).

``ReplayedHyperAuthorizationSemantics@1``

Acceptance covered:

* HyperLTL, AutoHyper, and MCHyper execute bounded satisfiable/violating
  information-flow hyperproperties with trace-pair witnesses, mutation,
  deterministic replay, malformed, timeout, and cross-engine disagreement
  handling against the managed vendor binaries;
* Soufflé executes allow, deny, unknown, conflict, delegation, rule/scope
  mutation, replay, malformed, timeout, and disagreement cases through the
  exact managed vendor binary;
* each receipt binds executable, runtime, source/artifact, host,
  policy/formula, bounds, parser decisions, and output digests;
* hyperproperty authority remains bounded and Soufflé remains an external
  authorization shadow;
* Microsoft SecPAL compatibility evidence is not interchangeable with
  Soufflé vendor semantics or hyperproperty authority;
* offline replay never installs, downloads, or mutates ambient PATH /
  user-site / the source tree.
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
    / "certify_formal_verification_replayed_hyper_authorization.py"
)
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_replayed_hyper_authorization_semantics.json"
)
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
HYPER_CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "hyperproperty.py"
)
AUTH_CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certification" / "authorization_external.py"
)

INTERFACE = "ReplayedHyperAuthorizationSemantics@1"
SCHEMA_VERSION = (
    "formal-verification-replayed-hyper-authorization-semantics/v1"
)
GOAL_ID = "FVT-G229"
TASK_ID = "FVT-097"
PROGRAM = "formal-verification-tactician/replayed-hyper-authorization-semantics"

HYPER_ENGINE_IDS = ("hyperltl", "autohyper", "mchyper")
REQUIRED_HYPER_CATEGORIES = {
    "satisfaction",
    "violation",
    "mutation",
    "replay",
    "malformed",
    "disagreement",
    "timeout",
    "bounds",
}
REQUIRED_HYPER_MUTATIONS = {"observation", "quantifier"}
REQUIRED_SOUFFLE_CATEGORIES = {
    "allow",
    "deny",
    "unknown",
    "conflict",
    "delegation",
}
REQUIRED_SOUFFLE_MUTATIONS = {"rule", "scope"}

DEFAULT_SEALED_ROOT = Path(
    "/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers"
)


def _load_module():
    assert CERTIFIER_PATH.is_file(), f"missing expected output: {CERTIFIER_PATH}"
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    name = "tools_logic_certify_formal_verification_replayed_hyper_authorization"
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
def live_receipt(certifier, sealed_root) -> dict[str, Any]:
    if sealed_root is None:
        pytest.skip("sealed managed prover root is not available")
    return certifier.certify_replayed_hyper_authorization_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=sealed_root,
        skip_install=True,
        force_install=False,
    )


@pytest.fixture(scope="module")
def receipt_document() -> dict[str, Any]:
    assert RECEIPT_PATH.is_file(), f"missing expected output: {RECEIPT_PATH}"
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# Expected outputs / surface constants
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert CERTIFIER_PATH.is_file()
    assert RECEIPT_PATH.is_file()
    assert Path(__file__).is_file()
    assert LOCK_PATH.is_file()
    assert HYPER_CERTIFIER_PATH.is_file()
    assert AUTH_CERTIFIER_PATH.is_file()


def test_module_surface_constants(certifier) -> None:
    assert certifier.INTERFACE == INTERFACE
    assert certifier.SCHEMA_VERSION == SCHEMA_VERSION
    assert certifier.GOAL_ID == GOAL_ID
    assert certifier.TASK_ID == TASK_ID
    assert certifier.PROGRAM == PROGRAM
    assert tuple(certifier.HYPER_ENGINE_IDS) == HYPER_ENGINE_IDS
    assert set(certifier.REQUIRED_HYPER_CATEGORIES) == REQUIRED_HYPER_CATEGORIES
    assert set(certifier.REQUIRED_HYPER_MUTATIONS) == REQUIRED_HYPER_MUTATIONS
    assert set(certifier.REQUIRED_SOUFFLE_CATEGORIES) == REQUIRED_SOUFFLE_CATEGORIES
    assert set(certifier.REQUIRED_SOUFFLE_MUTATIONS) == REQUIRED_SOUFFLE_MUTATIONS
    assert certifier.HYPER_AUTHORITY_CEILING == "bounded"
    assert certifier.SOUFFLE_AUTHORITY_CEILING == "none"
    assert (
        certifier.DEFAULT_RECEIPT_RELATIVE.as_posix()
        == "docs/architecture/formal_verification_replayed_hyper_authorization_semantics.json"
    )


def test_offline_env_blocks_install_and_network(certifier) -> None:
    env = certifier.offline_env({"PATH": "/usr/bin", "HOME": "/tmp/x"})
    assert env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_INSTALL"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_NETWORK"] == "1"
    assert env["FORMAL_VERIFICATION_REPLAYED_HYPER_AUTHORIZATION_OFFLINE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PIP_NO_INDEX"] == "1"


def test_force_install_with_skip_install_refused(certifier) -> None:
    with pytest.raises(certifier.ReplayedHyperAuthorizationError):
        certifier.certify_replayed_hyper_authorization_semantics(
            repo_root=REPO_ROOT,
            skip_install=True,
            force_install=True,
        )


def test_live_install_path_refused(certifier) -> None:
    with pytest.raises(certifier.ReplayedHyperAuthorizationError):
        certifier.certify_replayed_hyper_authorization_semantics(
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
    assert receipt_document["policy"]["secpal_compatibility_not_interchangeable"] is True
    assert (
        receipt_document["policy"]["hyperproperty_authority_ceiling"] == "bounded"
    )
    assert receipt_document["policy"]["souffle_authority_ceiling"] == "none"
    assert (
        receipt_document["policy"][
            "does_not_elevate_external_shadows_to_authorization_authority"
        ]
        is True
    )
    assert (
        receipt_document["policy"]["does_not_edit_legacy_secpal_artifact_intake"]
        is True
    )


def test_checked_in_receipt_lanes(receipt_document) -> None:
    hyper = receipt_document["hyperproperty"]
    auth = receipt_document["external_authorization"]
    assert hyper["certified"] is True
    assert hyper["authority_ceiling"] == "bounded"
    assert set(hyper["engine_ids"]) == set(HYPER_ENGINE_IDS)
    assert REQUIRED_HYPER_CATEGORIES <= set(hyper["categories_exercised"])
    assert REQUIRED_HYPER_MUTATIONS <= set(hyper["mutation_kinds"])
    assert auth["certified"] is True
    assert auth["authority_ceiling"] == "none"
    assert auth["external_authorization_shadow"] is True
    assert REQUIRED_SOUFFLE_CATEGORIES <= set(auth["categories_exercised"])
    assert REQUIRED_SOUFFLE_MUTATIONS <= set(auth["mutation_kinds"])
    secpal = auth["secpal_compatibility"]
    assert secpal["interchangeable_with_souffle_vendor"] is False
    assert secpal["interchangeable_with_hyperproperty_authority"] is False
    assert secpal["authoritative"] is False
    assert secpal["production_certified"] is False


# ---------------------------------------------------------------------------
# Live managed-root replay
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


def test_live_hyperproperty_lane_covers_acceptance(live_receipt) -> None:
    hyper = live_receipt["hyperproperty"]
    assert hyper["certified"] is True
    assert hyper["authority_ceiling"] == "bounded"
    assert hyper["forbids_theorem_authority"] is True
    assert hyper["forbids_universal_claims_beyond_bounds"] is True
    assert hyper["authorizes_universal_proof"] is False
    assert set(hyper["engine_ids"]) == set(HYPER_ENGINE_IDS)
    assert REQUIRED_HYPER_CATEGORIES <= set(hyper["categories_exercised"])
    assert REQUIRED_HYPER_MUTATIONS <= set(hyper["mutation_kinds"])
    assert hyper["checks_passed"] == hyper["checks_total"]
    assert hyper["checks_total"] >= 60  # 3 engines × ~22 checks

    engines = {item["tool_id"]: item for item in hyper["engines"]}
    assert set(engines) == set(HYPER_ENGINE_IDS)
    for tool_id, engine in engines.items():
        assert engine["certified"] is True, tool_id
        assert engine["authority_ceiling"] == "bounded"
        assert engine["authorizes_universal_proof"] is False
        assert engine["is_theorem_authority"] is False
        assert engine["is_vendor_build"] is True
        assert engine["is_hermetic_engine"] is False
        assert engine["executable_sha256"]
        assert len(engine["executable_sha256"]) == 64
        assert engine["source_archive_sha256"]
        assert engine["artifact_sha256"]
        assert engine["runtime_digest_sha256"]
        assert engine["host_platform"]
        assert engine["case_results_total"] >= 1
        # Policy/formula + parser + output digests must be bound per case.
        for case in engine["case_bindings"]:
            assert case.get("document_digest") or case.get("policy_or_formula_digest")
            assert case.get("output_digest")
            parser = case["parser_decisions"]
            assert "translation_preserved" in parser
            assert "quantifier_signature" in parser
            bounds = case["bounds"]
            assert bounds["authorizes_universal_proof"] is False
            assert bounds["is_theorem_authority"] is False


def test_live_souffle_lane_covers_acceptance(live_receipt) -> None:
    auth = live_receipt["external_authorization"]
    assert auth["certified"] is True
    assert auth["authority_ceiling"] == "none"
    assert auth["external_authorization_shadow"] is True
    assert auth["forbids_authorization_authority_on_shadows"] is True
    assert auth["grants_authorization_decision_authority"] is False
    assert REQUIRED_SOUFFLE_CATEGORIES <= set(auth["categories_exercised"])
    assert REQUIRED_SOUFFLE_MUTATIONS <= set(auth["mutation_kinds"])
    assert auth["checks_passed"] == auth["checks_total"]
    assert auth["checks_total"] >= 1

    souffle = auth["souffle"]
    assert souffle["tool_id"] == "souffle"
    assert souffle["certified"] is True
    assert souffle["authority_ceiling"] == "none"
    assert souffle["role"] in {"shadow", "authorization_shadow"}
    assert souffle["is_vendor_build"] is True
    assert souffle["is_hermetic_engine"] is False
    assert souffle["executable_sha256"]
    assert len(souffle["executable_sha256"]) == 64
    assert souffle["source_archive_sha256"]
    assert souffle["artifact_sha256"] or souffle["runtime_digest_sha256"]
    assert souffle["host_platform"]

    secpal = auth["secpal_compatibility"]
    assert secpal["tool_id"] == "secpal"
    assert secpal["interchangeable_with_souffle_vendor"] is False
    assert secpal["interchangeable_with_hyperproperty_authority"] is False
    assert secpal["authoritative"] is False
    assert secpal["production_certified"] is False
    assert secpal["complete"] is False


def test_live_acceptance_flags(live_receipt) -> None:
    acceptance = live_receipt["acceptance"]
    assert acceptance["goal_id"] == GOAL_ID
    assert acceptance["task_id"] == TASK_ID
    assert (
        acceptance[
            "hyperltl_autohyper_mchyper_bounded_information_flow_hyperproperties"
        ]
        is True
    )
    assert (
        acceptance[
            "trace_pair_witnesses_mutation_replay_malformed_timeout_disagreement"
        ]
        is True
    )
    assert (
        acceptance[
            "souffle_allow_deny_unknown_conflict_delegation_rule_scope_mutation_replay"
        ]
        is True
    )
    assert (
        acceptance[
            "receipts_bind_executable_runtime_source_host_policy_bounds_parser_output"
        ]
        is True
    )
    assert acceptance["hyperproperty_authority_remains_bounded"] is True
    assert acceptance["souffle_remains_external_authorization_shadow"] is True
    assert acceptance["microsoft_secpal_compatibility_not_interchangeable"] is True


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
    certifier, sealed_root, tmp_path: Path
) -> None:
    if sealed_root is None:
        pytest.skip("sealed managed prover root is not available")
    before = {
        path: path.stat().st_mtime_ns
        for path in (LOCK_PATH, CERTIFIER_PATH, HYPER_CERTIFIER_PATH, AUTH_CERTIFIER_PATH)
        if path.is_file()
    }
    out = tmp_path / "receipt.json"
    receipt = certifier.certify_replayed_hyper_authorization_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=sealed_root,
        write_receipt_path=out,
    )
    assert receipt["certified"] is True
    assert out.is_file()
    for path, mtime in before.items():
        assert path.stat().st_mtime_ns == mtime


def test_lane_failure_isolation_with_injected_certificates(certifier) -> None:
    """Hyper and Soufflé lanes fail independently; SecPAL cannot repair either."""

    good_hyper = {
        "certified": True,
        "interface": "HyperpropertyVendorToolchainCertification@1",
        "schema_version": "hyperproperty-vendor-toolchain-certification/v1",
        "goal_id": "FVT-G208",
        "task_id": "FVT-061",
        "certificate_digest_sha256": "a" * 64,
        "categories_exercised": sorted(REQUIRED_HYPER_CATEGORIES),
        "mutation_kinds": sorted(REQUIRED_HYPER_MUTATIONS),
        "summary": {"checks_passed": 66, "checks_total": 66},
        "engines": [
            {
                "engine_id": tool_id,
                "version": "test",
                "executable": "/opt/ipfs-accelerate/formal-toolchains/x/bin/" + tool_id,
                "usable": True,
                "certified": True,
                "role": "authority",
                "authority_ceiling": "bounded",
                "is_vendor_build": True,
                "is_hermetic_engine": False,
                "artifact_sha256": "b" * 64,
                "source_archive_sha256": "c" * 64,
                "source_archive_url": f"https://example.invalid/{tool_id}.tar.gz",
                "platform_id": "linux-aarch64",
                "checks": [
                    {
                        "check_id": f"{tool_id}.role",
                        "kind": "role",
                        "status": "passed",
                    }
                ],
                "case_results": [
                    {
                        "case_id": "case:ni_holds",
                        "engine_id": tool_id,
                        "expected": "satisfied",
                        "outcome": "satisfied",
                        "agreed": True,
                        "document_digest": "d" * 64,
                        "quantifier_signature": ["forall", "forall"],
                        "observation_fields": ["status"],
                        "translation_preserved": True,
                        "authority": "bounded",
                        "authorizes_universal_proof": False,
                        "is_theorem_authority": False,
                        "counterexample_traces": 0,
                    }
                ],
                "block_reasons": [],
            }
            for tool_id in HYPER_ENGINE_IDS
        ],
    }
    good_auth = {
        "certified": True,
        "souffle_vendor_certified": True,
        "interface": "ExternalAuthorizationVendorCertification@1",
        "schema_version": "external-authorization-vendor-certification/v1",
        "goal_id": "FVT-G209",
        "task_id": "FVT-055",
        "certificate_digest_sha256": "e" * 64,
        "categories_exercised": sorted(REQUIRED_SOUFFLE_CATEGORIES),
        "mutation_kinds": sorted(REQUIRED_SOUFFLE_MUTATIONS),
        "summary": {"checks_passed": 12, "checks_total": 12},
        "souffle": {
            "engine_id": "souffle",
            "version": "2.4.1",
            "executable": "/opt/ipfs-accelerate/formal-toolchains/x/bin/souffle",
            "usable": True,
            "certified": True,
            "role": "shadow",
            "authority_ceiling": "none",
            "is_vendor_build": True,
            "is_hermetic_shadow": False,
            "artifact_sha256": "f" * 64,
            "source_archive_sha256": (
                "08d9b19cb4a8f570ac75dea73016b6a326d87ac28fccd4afeba217ace2071587"
            ),
            "source_archive_url": "https://example.invalid/souffle.tar.gz",
            "platform_id": "linux-aarch64",
            "checks": [{"check_id": "souffle.role", "kind": "role", "status": "passed"}],
            "case_results": [
                {
                    "case_id": "case:allow",
                    "engine_id": "souffle",
                    "expected": "allow",
                    "outcome": "allow",
                    "agreed": True,
                    "document_digest": "1" * 64,
                    "authority": "none",
                }
            ],
            "block_reasons": [],
        },
        "secpal_platform_exception": {
            "tool_id": "secpal",
            "exception": True,
            "installed": False,
            "complete": False,
            "authoritative": False,
            "production_certified": False,
            "host_platform": "linux-aarch64",
            "classification": "unsupported_here",
        },
        "secpal_vendor_certified": False,
        "combined_external_authorization_certified": False,
    }

    both = certifier.certify_replayed_hyper_authorization_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=DEFAULT_SEALED_ROOT if DEFAULT_SEALED_ROOT.is_dir() else REPO_ROOT,
        hyper_certificate=good_hyper,
        authorization_certificate=good_auth,
    )
    assert both["hyperproperty"]["certified"] is True
    assert both["external_authorization"]["certified"] is True
    # Without live executable files under the injected paths, identity may
    # still bind via artifact digests; overall certificate remains True when
    # projection rules are satisfied.
    assert both["certified"] is True

    bad_hyper = copy.deepcopy(good_hyper)
    bad_hyper["certified"] = False
    bad_hyper["engines"][0]["certified"] = False
    hyper_fail = certifier.certify_replayed_hyper_authorization_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=DEFAULT_SEALED_ROOT if DEFAULT_SEALED_ROOT.is_dir() else REPO_ROOT,
        hyper_certificate=bad_hyper,
        authorization_certificate=good_auth,
    )
    assert hyper_fail["hyperproperty"]["certified"] is False
    assert hyper_fail["external_authorization"]["certified"] is True
    assert hyper_fail["certified"] is False

    bad_auth = copy.deepcopy(good_auth)
    bad_auth["souffle"]["is_hermetic_shadow"] = True
    bad_auth["souffle"]["is_hermetic_engine"] = True
    auth_fail = certifier.certify_replayed_hyper_authorization_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=DEFAULT_SEALED_ROOT if DEFAULT_SEALED_ROOT.is_dir() else REPO_ROOT,
        hyper_certificate=good_hyper,
        authorization_certificate=bad_auth,
    )
    assert auth_fail["hyperproperty"]["certified"] is True
    assert auth_fail["external_authorization"]["certified"] is False
    assert auth_fail["certified"] is False

    # Elevating SecPAL must fail closed rather than repair Soufflé.
    secpal_elevated = copy.deepcopy(good_auth)
    secpal_elevated["secpal_platform_exception"]["authoritative"] = True
    secpal_elevated["secpal_vendor_certified"] = True
    secpal_elevated["combined_external_authorization_certified"] = True
    secpal_fail = certifier.certify_replayed_hyper_authorization_semantics(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        managed_root=DEFAULT_SEALED_ROOT if DEFAULT_SEALED_ROOT.is_dir() else REPO_ROOT,
        hyper_certificate=good_hyper,
        authorization_certificate=secpal_elevated,
    )
    assert secpal_fail["external_authorization"]["certified"] is False
    assert "secpal_compatibility_treated_as_interchangeable" in (
        secpal_fail["external_authorization"]["block_reasons"]
    )
    assert secpal_fail["certified"] is False


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
    assert set(receipt_document["hyperproperty"]["engine_ids"]) == set(
        live_receipt["hyperproperty"]["engine_ids"]
    )
    assert receipt_document["external_authorization"]["authority_ceiling"] == (
        live_receipt["external_authorization"]["authority_ceiling"]
    )
