"""Replayed core external semantics aggregator (FVT-096 / FVT-G228).

``ReplayedCoreExternalSemantics@1``

Cross-family replay for state-model, protocol, proof-kernel, and ATP families:

* TLC and Apalache execute bounded state/safety/liveness cases;
* Tamarin and ProVerif execute protocol secrecy/authentication and mutation;
* Rocq/Coq and Isabelle check accepted/rejected proof objects in genuine kernels;
* Vampire and E execute theorem/non-theorem/resource-bound cases;
* each provider has independent positive, negative, mutation, replay, malformed,
  timeout, and disagreement evidence bound to the managed identity;
* Maude and OPAM remain support-only;
* no fixture, parser, wrapper, advisor, or other provider can supply a missing
  engine's semantic or authority axis.

The aggregator reuses family certifiers without changing their authority
ceilings or installers.
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
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_replayed_core_semantics.py"
)
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_replayed_core_external_semantics.json"
)

INTERFACE = "ReplayedCoreExternalSemantics@1"
SCHEMA_VERSION = "replayed-core-external-semantics/v1"
GOAL_ID = "FVT-G228"
TASK_ID = "FVT-096"
PROGRAM = "formal-verification-tactician/replayed-core-external-semantics"

REQUIRED_EVIDENCE_KINDS = (
    "positive",
    "negative",
    "mutation",
    "replay",
    "malformed",
    "timeout",
    "disagreement",
)
REQUIRED_PROVIDERS = (
    "tlc",
    "apalache",
    "tamarin",
    "proverif",
    "rocq",
    "isabelle",
    "vampire",
    "eprover",
)
FAMILY_IDS = ("state_model", "protocol", "kernel", "atp")
SUPPORT_TOOLS = ("maude", "opam")

FAMILY_CERT_PATHS = {
    "state_model": REPO_ROOT
    / "docs/architecture/formal_verification_state_model_live_certificate.json",
    "protocol": REPO_ROOT
    / "docs/architecture/formal_verification_protocol_live_certificate.json",
    "kernel": REPO_ROOT
    / "docs/architecture/formal_verification_kernel_live_certificate.json",
    "atp": REPO_ROOT
    / "docs/architecture/formal_verification_atp_live_certificate.json",
}
MANAGED_ENV_PATH = (
    REPO_ROOT
    / "docs/architecture/formal_verification_managed_environment_replay_receipt.json"
)


def _load_module():
    assert CERTIFIER_PATH.is_file(), f"missing expected output: {CERTIFIER_PATH}"
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    name = "tools_logic_certify_formal_verification_replayed_core_semantics"
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
def durable_receipt(certifier) -> dict[str, Any]:
    return certifier.certify_replayed_core_semantics(
        repo_root=REPO_ROOT,
        mode="reuse",
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
    for path in FAMILY_CERT_PATHS.values():
        assert path.is_file(), f"dependency family certificate missing: {path}"
    assert MANAGED_ENV_PATH.is_file()


def test_module_surface_constants(certifier) -> None:
    assert certifier.INTERFACE == INTERFACE
    assert certifier.SCHEMA_VERSION == SCHEMA_VERSION
    assert certifier.GOAL_ID == GOAL_ID
    assert certifier.TASK_ID == TASK_ID
    assert certifier.PROGRAM == PROGRAM
    assert tuple(certifier.REQUIRED_EVIDENCE_KINDS) == REQUIRED_EVIDENCE_KINDS
    assert tuple(certifier.REQUIRED_PROVIDER_IDS) == REQUIRED_PROVIDERS
    assert tuple(certifier.FAMILY_IDS) == FAMILY_IDS
    assert tuple(certifier.SUPPORT_TOOL_IDS) == SUPPORT_TOOLS
    assert (
        certifier.DEFAULT_RECEIPT_RELATIVE.as_posix()
        == "docs/architecture/formal_verification_replayed_core_external_semantics.json"
    )


# ---------------------------------------------------------------------------
# Durable receipt document (checked-in evidence term)
# ---------------------------------------------------------------------------


def test_receipt_document_schema(receipt_document) -> None:
    assert receipt_document["interface"] == INTERFACE
    assert receipt_document["schema_version"] == SCHEMA_VERSION
    assert receipt_document["goal_id"] == GOAL_ID
    assert receipt_document["task_id"] == TASK_ID
    assert receipt_document["certified"] is True
    assert receipt_document["production_certified"] is True
    assert set(receipt_document["required_evidence_kinds"]) == set(
        REQUIRED_EVIDENCE_KINDS
    )
    assert list(receipt_document["required_provider_ids"]) == list(REQUIRED_PROVIDERS)
    assert list(receipt_document["family_ids"]) == list(FAMILY_IDS)
    assert list(receipt_document["support_tool_ids"]) == list(SUPPORT_TOOLS)
    policy = receipt_document["policy"]
    assert policy["reuses_family_certifiers"] is True
    assert policy["does_not_change_authority_ceilings"] is True
    assert policy["does_not_edit_installers"] is True
    assert policy["fixture_parser_wrapper_advisor_cannot_substitute"] is True
    assert policy["sibling_provider_cannot_substitute"] is True
    assert policy["maude_opam_support_only"] is True
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert receipt_document["receipt_digest_sha256"]
    assert receipt_document["certificate_digest_sha256"]


def test_receipt_document_providers_complete(receipt_document) -> None:
    providers = receipt_document["providers"]
    for provider_id in REQUIRED_PROVIDERS:
        row = providers[provider_id]
        assert row["provider_id"] == provider_id
        assert row["certified"] is True
        assert row["support_only"] is False
        assert row["semantic"] is True
        evidence = row["evidence"]
        assert evidence["complete"] is True
        present = evidence["present"]
        for kind in REQUIRED_EVIDENCE_KINDS:
            assert present[kind] is True, f"{provider_id} missing {kind}"
        bindings = row["bindings"]
        assert bindings["managed_identity_bound"] is True
        independence = row["independence"]
        assert independence["ok"] is True
        assert independence["fixture_cannot_substitute"] is True
        assert independence["parser_cannot_substitute"] is True
        assert independence["advisor_cannot_substitute"] is True
        assert independence["sibling_provider_cannot_substitute"] is True


def test_receipt_document_families(receipt_document) -> None:
    families = receipt_document["families"]
    expected = {
        "state_model": ("tlc", "apalache"),
        "protocol": ("tamarin", "proverif"),
        "kernel": ("rocq", "isabelle"),
        "atp": ("vampire", "eprover"),
    }
    for family_id, provider_ids in expected.items():
        family = families[family_id]
        assert family["family_id"] == family_id
        assert family["certified"] is True
        assert family["independence_ok"] is True
        assert family["production_certified"] is True
        assert list(family["provider_ids"]) == list(provider_ids)
        for provider_id in provider_ids:
            assert family["providers"][provider_id]["certified"] is True
        assert family["policy"]["reuses_family_certifier"] is True
        assert family["policy"]["does_not_change_authority_ceiling"] is True
        assert family["policy"]["does_not_edit_installer"] is True


def test_receipt_document_support_only(receipt_document) -> None:
    support = receipt_document["support_tools"]
    for tool_id in SUPPORT_TOOLS:
        row = support[tool_id]
        assert row["tool_id"] == tool_id
        assert row["support_only"] is True
        assert row["semantic"] is False
        assert row["non_semantic"] is True
        assert row["non_authoritative"] is True
        assert row["grants_semantic_certification"] is False
        assert row["grants_authority"] is False
        assert row["cannot_supply_missing_engine_semantics"] is True
        assert row["cannot_supply_missing_engine_authority"] is True


def test_receipt_document_acceptance(receipt_document) -> None:
    acceptance = receipt_document["acceptance"]
    assert acceptance["goal_id"] == GOAL_ID
    assert acceptance["task_id"] == TASK_ID
    assert acceptance["tlc_apalache_bounded_state_safety_liveness"] is True
    assert acceptance["tamarin_proverif_protocol_secrecy_authentication_mutation"] is True
    assert acceptance["rocq_isabelle_genuine_kernel_proof_objects"] is True
    assert acceptance["vampire_e_theorem_nontheorem_resource_bound"] is True
    assert (
        acceptance[
            "each_provider_independent_positive_negative_mutation_replay_malformed_timeout_disagreement"
        ]
        is True
    )
    assert acceptance["maude_opam_support_only"] is True
    assert (
        acceptance["no_fixture_parser_wrapper_advisor_other_provider_substitution"]
        is True
    )
    assert acceptance["managed_identity_bound"] is True


# ---------------------------------------------------------------------------
# Live aggregator over durable family certificates
# ---------------------------------------------------------------------------


def test_durable_reuse_certifies(durable_receipt) -> None:
    assert durable_receipt["interface"] == INTERFACE
    assert durable_receipt["certified"] is True
    assert durable_receipt["production_certified"] is True
    assert not durable_receipt.get("receipt_validation_failures")
    assert durable_receipt["summary"]["certified"] is True
    assert set(durable_receipt["summary"]["providers_certified"]) == set(
        REQUIRED_PROVIDERS
    )
    assert set(durable_receipt["summary"]["families_certified"]) == set(FAMILY_IDS)
    assert durable_receipt["managed_environment"]["present"] is True


def test_validate_receipt_accepts_certified(certifier, durable_receipt) -> None:
    failures = certifier.validate_receipt(durable_receipt)
    assert failures == []


def test_kind_mapping_covers_required_axes(certifier) -> None:
    for family_id, spec in certifier.FAMILY_SPECS.items():
        mapped = set(spec["kind_map"].values())
        # Every required canonical kind must be reachable either by map or
        # derived disagreement for families without native disagreement rows.
        for kind in REQUIRED_EVIDENCE_KINDS:
            if kind == "disagreement":
                continue
            assert kind in mapped, f"{family_id} kind_map missing {kind}"


def test_content_digest_stable(certifier) -> None:
    payload = {"a": 1, "b": ["x", "y"], "c": {"z": True}}
    first = certifier.content_digest(payload)
    second = certifier.content_digest(payload)
    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# Fail-closed: missing provider evidence, substitution, missing family
# ---------------------------------------------------------------------------


def _load_family(family_id: str) -> dict[str, Any]:
    return json.loads(FAMILY_CERT_PATHS[family_id].read_text(encoding="utf-8"))


def test_missing_provider_cases_fail_closed(certifier) -> None:
    families = {fid: _load_family(fid) for fid in FAMILY_IDS}
    # Strip all TLC cases so positive/negative evidence vanishes.
    families["state_model"] = copy.deepcopy(families["state_model"])
    families["state_model"]["cases"] = [
        case
        for case in families["state_model"].get("cases") or []
        if case.get("tool_id") != "tlc"
    ]
    receipt = certifier.certify_replayed_core_semantics(
        repo_root=REPO_ROOT,
        mode="reuse",
        family_receipts=families,
    )
    assert receipt["certified"] is False
    tlc = receipt["providers"]["tlc"]
    assert tlc["certified"] is False
    assert tlc["evidence"]["complete"] is False
    assert "positive" in tlc["evidence"]["missing_kinds"] or not tlc["evidence"][
        "present"
    ]["positive"]
    # Sibling apalache must not inherit success into TLC.
    assert receipt["providers"]["apalache"]["certified"] is True
    assert "providers_incomplete" in receipt["block_reasons"]


def test_fixture_cannot_substitute_missing_engine(certifier) -> None:
    family = certifier.evaluate_family("atp", _load_family("atp"))
    proof = certifier.prove_substitution_fail_closed(
        family,
        missing_provider="vampire",
        substitute_source="fixture",
    )
    assert proof["substitution_allowed"] is False
    assert proof["fail_closed"] is True
    assert proof["simulated_provider_certified"] is False
    assert proof["family_remains_certified"] is False


def test_parser_wrapper_advisor_sibling_cannot_substitute(certifier) -> None:
    family = certifier.evaluate_family("kernel", _load_family("kernel"))
    for source in (
        "parser",
        "wrapper",
        "advisor",
        "sibling_provider",
        "other_provider",
        "hermetic_parser",
        "canned_text",
    ):
        proof = certifier.prove_substitution_fail_closed(
            family,
            missing_provider="rocq",
            substitute_source=source,
        )
        assert proof["fail_closed"] is True
        assert proof["substitution_allowed"] is False


def test_missing_family_fail_closed(certifier, durable_receipt) -> None:
    families = durable_receipt["families"]
    for family_id in FAMILY_IDS:
        proof = certifier.prove_missing_family_fail_closed(families, family_id)
        assert proof["fail_closed"] is True
        assert proof["aggregator_certified"] is False
        assert proof["stale_receipt_cannot_repair"] is True
        assert proof["missing_family"] == family_id


def test_support_tools_cannot_grant_semantics(certifier) -> None:
    protocol = _load_family("protocol")
    for tool_id in SUPPORT_TOOLS:
        row = certifier.evaluate_support_tool(tool_id, protocol_payload=protocol)
        assert row["support_only"] is True
        assert row["grants_semantic_certification"] is False
        assert row["grants_authority"] is False
        assert row["cannot_supply_missing_engine_semantics"] is True


def test_demoted_family_certificate_blocks(certifier) -> None:
    families = {fid: _load_family(fid) for fid in FAMILY_IDS}
    families["protocol"] = copy.deepcopy(families["protocol"])
    families["protocol"]["production_certified"] = False
    families["protocol"]["live_semantic_certified"] = False
    receipt = certifier.certify_replayed_core_semantics(
        repo_root=REPO_ROOT,
        mode="reuse",
        family_receipts=families,
    )
    assert receipt["certified"] is False
    assert receipt["families"]["protocol"]["certified"] is False
    # Other families remain independently certified.
    assert receipt["families"]["state_model"]["certified"] is True
    assert receipt["families"]["atp"]["certified"] is True
    assert receipt["families"]["kernel"]["certified"] is True


def test_stale_receipt_cannot_repair_missing_provider(certifier, durable_receipt) -> None:
    """A previously certified aggregator receipt cannot repair a fresh failure."""

    families = {fid: _load_family(fid) for fid in FAMILY_IDS}
    families["atp"] = copy.deepcopy(families["atp"])
    families["atp"]["cases"] = [
        case
        for case in families["atp"].get("cases") or []
        if case.get("tool_id") != "eprover"
    ]
    failed = certifier.certify_replayed_core_semantics(
        repo_root=REPO_ROOT,
        mode="reuse",
        family_receipts=families,
    )
    assert failed["certified"] is False
    # Stale success evidence exists but must not flip the failed run.
    assert durable_receipt["certified"] is True
    assert failed["providers"]["eprover"]["certified"] is False
    assert failed["receipt_digest_sha256"] != durable_receipt["receipt_digest_sha256"]


# ---------------------------------------------------------------------------
# Independence and authority ceilings preserved
# ---------------------------------------------------------------------------


def test_authority_ceilings_preserved(durable_receipt) -> None:
    ceilings = {
        family["family_id"]: family["authority_ceiling"]
        for family in durable_receipt["families"].values()
    }
    assert ceilings["state_model"] == "bounded"
    assert ceilings["protocol"] == "protocol"
    assert ceilings["kernel"] == "kernel"
    assert ceilings["atp"] == "reconstruction_candidate"


def test_family_goal_and_interface_bindings(durable_receipt) -> None:
    expected = {
        "state_model": (
            "FVT-G204",
            "StateModelLiveSemanticCertification@1",
        ),
        "protocol": (
            "FVT-G205",
            "ProtocolLiveSemanticCertification@1",
        ),
        "kernel": ("FVT-G206", "KernelLiveSemanticFanIn@1"),
        "atp": ("FVT-G207", "ATPLiveSemanticCertification@1"),
    }
    for family_id, (goal_id, interface) in expected.items():
        family = durable_receipt["families"][family_id]
        assert family["goal_id"] == goal_id
        assert family["interface"] == interface


def test_managed_environment_dependency_bound(durable_receipt) -> None:
    managed = durable_receipt["managed_environment"]
    assert managed["goal_id"] == "FVT-G226"
    assert managed["task_id"] == "FVT-094"
    assert managed["interface"] == "FormalVerificationManagedEnvironmentReplay@1"
    assert managed["present"] is True


def test_substitution_proofs_embedded(durable_receipt) -> None:
    proofs = durable_receipt["substitution_fail_closed"]
    assert proofs
    for proof in proofs.values():
        assert proof["fail_closed"] is True
        assert proof["substitution_allowed"] is False
    missing = durable_receipt["missing_family_fail_closed"]
    assert set(missing) == set(FAMILY_IDS)


def test_offline_policy_env(certifier) -> None:
    env = certifier.offline_env({"PATH": "/opt/tools/bin", "HOME": "/tmp/home"})
    assert env["FORMAL_VERIFICATION_FORBID_INSTALL"] == "1"
    assert env["FORMAL_VERIFICATION_FORBID_NETWORK"] == "1"
    assert env["FORMAL_VERIFICATION_CERTIFY_OFFLINE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["FORMAL_VERIFICATION_REPLAYED_CORE_SEMANTICS_OFFLINE"] == "1"


def test_write_receipt_roundtrip(certifier, durable_receipt, tmp_path) -> None:
    target = tmp_path / "replayed_core.json"
    certifier.write_receipt(target, durable_receipt)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["interface"] == INTERFACE
    assert loaded["certified"] is durable_receipt["certified"]
    assert loaded["receipt_digest_sha256"] == durable_receipt["receipt_digest_sha256"]


def test_cli_main_reuse(certifier, tmp_path) -> None:
    output = tmp_path / "out.json"
    code = certifier.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--mode",
            "reuse",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["certified"] is True
    assert payload["interface"] == INTERFACE


def test_map_case_kind_helpers(certifier) -> None:
    assert certifier.map_case_kind("invariant_holds", {"invariant_holds": "positive"}) == (
        "positive"
    )
    assert certifier.map_case_kind("unknown", {"theorem": "positive"}) is None
    assert certifier.map_case_kind("secure", certifier.FAMILY_SPECS["protocol"]["kind_map"]) == (
        "positive"
    )
    assert certifier.map_case_kind("attack", certifier.FAMILY_SPECS["protocol"]["kind_map"]) == (
        "negative"
    )
    assert certifier.map_case_kind("fail_closed", certifier.FAMILY_SPECS["kernel"]["kind_map"]) == (
        "disagreement"
    )


def test_compact_receipt_does_not_embed_full_stdout(durable_receipt) -> None:
    """Admission prefers compact receipts over bulk golden dumps."""

    text = json.dumps(durable_receipt)
    assert "raw_szs_output" not in text or text.count("raw_szs_output") < 3
    # No multi-megabyte embedding of family certificates.
    assert len(text) < 1_000_000
    for provider in durable_receipt["providers"].values():
        by_kind = provider["evidence"]["by_kind"]
        for rows in by_kind.values():
            assert len(rows) <= 2
            for row in rows:
                assert "stdout" not in row
                assert len(str(row.get("detail") or "")) <= 240
