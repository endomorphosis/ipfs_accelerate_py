"""Live Tamarin + ProVerif protocol semantic certification (FVT-058 / FVT-G205).

Exercises ``ProtocolLiveSemanticCertification@1``:

* both pinned binaries execute secrecy/authentication protocols;
* concrete attacks, premise/conclusion and protocol mutations, replay,
  malformed models, timeout, disagreement, and bounded-search cases;
* receipts bind tool and dependency identities, source, query, assumptions,
  bound, witnesses/traces, and raw output;
* parser fixtures remain non-production;
* neither engine may stand in for the other.
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
TAMARIN_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "tamarin.py"
PROVERIF_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "proverif.py"
CERTIFICATE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_protocol_live_certificate.json"
)

LIVE_INTERFACE = "ProtocolLiveSemanticCertification@1"
LIVE_SCHEMA = "protocol-live-semantic-certification/v1"
LIVE_CORPUS_SCHEMA = "protocol-live-semantic-corpus/v1"
LIVE_GOAL_ID = "FVT-G205"
LIVE_TASK_ID = "FVT-058"

REQUIRED_CASE_KINDS = {
    "secure",
    "attack",
    "mutation",
    "replay",
    "malformed",
    "timeout",
    "disagreement",
    "bounded_search",
}

REQUIRED_CASE_IDS = {
    "live_secure_secrecy_auth",
    "live_attack_leak",
    "live_mutated_claim",
    "live_mutated_protocol",
    "live_deterministic_replay",
    "live_malformed_model",
    "live_timeout",
    "live_disagreement",
    "live_bounded_search",
}

BINDING_KEYS = {
    "tool",
    "dependency",
    "source",
    "query",
    "assumptions",
    "bound",
    "witnesses_traces",
    "raw_output",
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
def tamarin_cert():
    return _load_module(TAMARIN_CERT_PATH, "tools_logic_certification_tamarin_live")


@pytest.fixture(scope="module")
def proverif_cert():
    return _load_module(PROVERIF_CERT_PATH, "tools_logic_certification_proverif_live")


@pytest.fixture(scope="module")
def tamarin_live(tamarin_cert) -> dict[str, Any]:
    return tamarin_cert.build_live_semantic_receipt(
        repo_root=REPO_ROOT,
        env=tamarin_cert.offline_env(os.environ),
    )


@pytest.fixture(scope="module")
def proverif_live(proverif_cert) -> dict[str, Any]:
    return proverif_cert.build_live_semantic_receipt(
        repo_root=REPO_ROOT,
        env=proverif_cert.offline_env(os.environ),
    )


@pytest.fixture(scope="module")
def protocol_certificate(tamarin_cert, tamarin_live, proverif_live) -> dict[str, Any]:
    return tamarin_cert.build_protocol_live_certificate(
        repo_root=REPO_ROOT,
        tamarin_receipt=tamarin_live,
        proverif_receipt=proverif_live,
        env=tamarin_cert.offline_env(os.environ),
    )


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert TAMARIN_CERT_PATH.is_file()
    assert PROVERIF_CERT_PATH.is_file()
    assert Path(__file__).is_file()
    assert CERTIFICATE_PATH.is_file(), (
        f"missing live certificate evidence: {CERTIFICATE_PATH}"
    )


def test_live_module_constants(tamarin_cert, proverif_cert) -> None:
    for mod in (tamarin_cert, proverif_cert):
        assert mod.LIVE_INTERFACE == LIVE_INTERFACE
        assert mod.LIVE_SCHEMA_VERSION == LIVE_SCHEMA
        assert mod.LIVE_CORPUS_SCHEMA == LIVE_CORPUS_SCHEMA
        assert mod.LIVE_GOAL_ID == LIVE_GOAL_ID
        assert mod.LIVE_TASK_ID == LIVE_TASK_ID
        assert mod.EVIDENCE_CLASS_LIVE == "live"
        assert mod.EVIDENCE_CLASS_PARSER_FIXTURE == "parser_fixture"
        assert mod.parser_fixture_evidence_class() == "parser_fixture"

    assert tamarin_cert.TOOL_ID == "tamarin"
    assert proverif_cert.TOOL_ID == "proverif"
    assert tamarin_cert.LANE_ID == proverif_cert.LANE_ID == "protocol"


def test_live_corpus_schema_and_required_cases(tamarin_cert, proverif_cert) -> None:
    for mod, tool_id in ((tamarin_cert, "tamarin"), (proverif_cert, "proverif")):
        manifest = mod.default_live_corpus_manifest()
        assert manifest["schema_version"] == LIVE_CORPUS_SCHEMA
        assert manifest["interface"] == LIVE_INTERFACE
        assert manifest["goal_id"] == LIVE_GOAL_ID
        assert manifest["task_id"] == LIVE_TASK_ID
        assert manifest["tool_id"] == tool_id
        assert manifest["policy"]["parser_fixtures_are_non_production"] is True
        assert manifest["policy"]["live_binary_required_for_semantic_proof"] is True
        assert manifest["policy"]["engines_are_independent"] is True
        assert manifest["policy"]["no_install"] is True
        assert manifest["policy"]["no_network"] is True

        cases = manifest["cases"]
        kinds = {case["kind"] for case in cases}
        assert REQUIRED_CASE_KINDS <= kinds
        case_ids = {case["case_id"] for case in cases}
        assert REQUIRED_CASE_IDS <= case_ids
        for case in cases:
            assert str(case.get("source") or "").strip(), case["case_id"]
            assert case["expect"] in {
                "secure",
                "attack",
                "quarantined",
                "blocked",
                "rejected_or_quarantined",
            }


# ---------------------------------------------------------------------------
# Live execution evidence
# ---------------------------------------------------------------------------


def test_tamarin_live_semantic_certified(tamarin_live: dict[str, Any]) -> None:
    assert tamarin_live["interface"] == LIVE_INTERFACE
    assert tamarin_live["schema_version"] == LIVE_SCHEMA
    assert tamarin_live["goal_id"] == LIVE_GOAL_ID
    assert tamarin_live["task_id"] == LIVE_TASK_ID
    assert tamarin_live["tool_id"] == "tamarin"
    assert tamarin_live["parser_fixtures_are_non_production"] is True
    assert tamarin_live["cannot_substitute_proverif"] is True
    assert tamarin_live["network_used"] is False
    assert tamarin_live["install_attempted"] is False
    assert tamarin_live["download_attempted"] is False

    if not tamarin_live.get("tamarin_usable"):
        pytest.skip("pinned Tamarin binary unavailable on this host")

    assert tamarin_live["live_semantic_certified"] is True
    by_id = {case["case_id"]: case for case in tamarin_live["cases"]}
    assert REQUIRED_CASE_IDS <= set(by_id)
    for case_id in REQUIRED_CASE_IDS:
        case = by_id[case_id]
        assert case["matched"] is True, case
        assert case["evidence_class"] == "live"
        assert case["live_executed"] is True
        assert case["source_digest"]
        assert case["output_digest"] or case["kind"] in {"timeout", "malformed"}
        assert "query" in case
        assert isinstance(case.get("assumptions"), list)
        assert isinstance(case.get("bounds"), dict)

    assert by_id["live_secure_secrecy_auth"]["status"] == "secure"
    assert by_id["live_attack_leak"]["status"] == "attack"
    assert by_id["live_attack_leak"]["attack_trace"] is not None
    assert by_id["live_mutated_claim"]["status"] != "secure"
    assert by_id["live_mutated_protocol"]["status"] == "attack"
    assert by_id["live_malformed_model"]["status"] == "quarantined"
    assert by_id["live_timeout"]["status"] == "quarantined"
    assert by_id["live_disagreement"]["status"] != "secure"
    assert by_id["live_bounded_search"]["status"] == "quarantined"

    bindings = tamarin_live["bindings"]
    assert BINDING_KEYS <= set(bindings)
    assert bindings["tool"]["tool_id"] == "tamarin"
    assert bindings["tool"]["locked_version"] == "1.12.0"
    assert bindings["dependency"]["tool_id"] == "maude"
    assert bindings["dependency"]["support_only"] is True
    assert bindings["dependency"]["can_promote_protocol_lane"] is False
    assert bindings["source"]["source_digest"]
    assert bindings["authority"]["parser_fixtures_are_non_production"] is True
    assert bindings["authority"]["not_proverif"] is True


def test_proverif_live_semantic_certified(proverif_live: dict[str, Any]) -> None:
    assert proverif_live["interface"] == LIVE_INTERFACE
    assert proverif_live["schema_version"] == LIVE_SCHEMA
    assert proverif_live["goal_id"] == LIVE_GOAL_ID
    assert proverif_live["task_id"] == LIVE_TASK_ID
    assert proverif_live["tool_id"] == "proverif"
    assert proverif_live["parser_fixtures_are_non_production"] is True
    assert proverif_live["cannot_substitute_tamarin"] is True
    assert proverif_live["network_used"] is False
    assert proverif_live["install_attempted"] is False
    assert proverif_live["download_attempted"] is False
    assert proverif_live.get("global_opam_mutation_attempted") is False

    if not proverif_live.get("proverif_usable"):
        pytest.skip("pinned ProVerif binary unavailable on this host")

    assert proverif_live["live_semantic_certified"] is True
    by_id = {case["case_id"]: case for case in proverif_live["cases"]}
    assert REQUIRED_CASE_IDS <= set(by_id)
    for case_id in REQUIRED_CASE_IDS:
        case = by_id[case_id]
        assert case["matched"] is True, case
        assert case["evidence_class"] == "live"
        assert case["live_executed"] is True
        assert case["source_digest"]
        assert case["source_format"] == "pv"

    assert by_id["live_secure_secrecy_auth"]["status"] == "secure"
    assert by_id["live_attack_leak"]["status"] == "attack"
    assert by_id["live_attack_leak"]["attack_trace"] is not None
    assert by_id["live_mutated_claim"]["status"] != "secure"
    assert by_id["live_mutated_protocol"]["status"] == "attack"
    assert by_id["live_malformed_model"]["status"] == "quarantined"
    assert by_id["live_timeout"]["status"] == "quarantined"
    assert by_id["live_disagreement"]["status"] != "secure"
    assert by_id["live_bounded_search"]["status"] == "quarantined"

    bindings = proverif_live["bindings"]
    assert BINDING_KEYS <= set(bindings)
    assert bindings["tool"]["tool_id"] == "proverif"
    assert bindings["tool"]["locked_version"] == "2.05"
    assert bindings["dependency"]["tool_id"] == "opam"
    assert bindings["dependency"]["support_only"] is True
    assert bindings["dependency"]["can_promote_protocol_lane"] is False
    assert bindings["authority"]["not_tamarin"] is True
    assert bindings["authority"]["parser_fixtures_are_non_production"] is True


def test_engines_are_independent(
    tamarin_live: dict[str, Any], proverif_live: dict[str, Any]
) -> None:
    assert tamarin_live["tool_id"] != proverif_live["tool_id"]
    assert tamarin_live.get("cannot_substitute_proverif") is True
    assert proverif_live.get("cannot_substitute_tamarin") is True
    # Sources and digests must be engine-local (spthy vs pv).
    t_cases = {c["case_id"]: c for c in tamarin_live.get("cases") or []}
    p_cases = {c["case_id"]: c for c in proverif_live.get("cases") or []}
    if "live_secure_secrecy_auth" in t_cases and "live_secure_secrecy_auth" in p_cases:
        assert (
            t_cases["live_secure_secrecy_auth"]["source_format"]
            != p_cases["live_secure_secrecy_auth"]["source_format"]
        )
        assert (
            t_cases["live_secure_secrecy_auth"]["source_digest"]
            != p_cases["live_secure_secrecy_auth"]["source_digest"]
        )


def test_parser_fixtures_remain_non_production(
    tamarin_cert, proverif_cert, tamarin_live, proverif_live
) -> None:
    # Offline canned corpus still passes, but is labeled non-production.
    for mod in (tamarin_cert, proverif_cert):
        assert mod.parser_fixture_evidence_class() == "parser_fixture"
        for case in mod.corpus_cases():
            # Offline cases may carry canned stdout — never evidence_class live.
            assert case.get("evidence_class", "parser_fixture") != "live"
            outcome = mod.evaluate_corpus_case(case)
            assert outcome.matched is True, case["case_id"]

    assert tamarin_live["parser_fixtures_are_non_production"] is True
    assert proverif_live["parser_fixtures_are_non_production"] is True
    for receipt in (tamarin_live, proverif_live):
        for case in receipt.get("cases") or []:
            assert case["evidence_class"] == "live"
            assert case["live_executed"] is True


def test_protocol_live_certificate_aggregate(
    protocol_certificate: dict[str, Any],
) -> None:
    cert = protocol_certificate
    assert cert["interface"] == LIVE_INTERFACE
    assert cert["schema_version"] == LIVE_SCHEMA
    assert cert["goal_id"] == LIVE_GOAL_ID
    assert cert["task_id"] == LIVE_TASK_ID
    assert cert["policy"]["parser_fixtures_are_non_production"] is True
    assert cert["policy"]["engines_are_independent"] is True
    assert cert["policy"]["no_engine_stands_in_for_other"] is True
    assert cert["policy"]["live_binary_required_for_semantic_proof"] is True
    assert set(cert["required_case_kinds"]) >= REQUIRED_CASE_KINDS
    assert "tamarin" in cert["tools"]
    assert "proverif" in cert["tools"]
    assert cert["engine_independence"]["independence_ok"] is True

    tamarin_ok = bool(cert["tools"]["tamarin"].get("live_semantic_certified"))
    proverif_ok = bool(cert["tools"]["proverif"].get("live_semantic_certified"))
    if not (tamarin_ok and proverif_ok):
        pytest.skip("one or both protocol engines unavailable for live certification")

    assert cert["live_semantic_certified"] is True
    assert cert["production_certified"] is True
    assert cert["promotion_blocked"] is False
    assert cert["certificate_digest_sha256"]
    assert len(cert["certificate_digest_sha256"]) == 64


def test_checked_in_certificate_matches_interface() -> None:
    payload = json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))
    assert payload["interface"] == LIVE_INTERFACE
    assert payload["schema_version"] == LIVE_SCHEMA
    assert payload["goal_id"] == LIVE_GOAL_ID
    assert payload["task_id"] == LIVE_TASK_ID
    assert payload["policy"]["parser_fixtures_are_non_production"] is True
    assert payload["policy"]["engines_are_independent"] is True
    assert "tamarin" in payload["tools"]
    assert "proverif" in payload["tools"]
    assert set(payload.get("required_case_kinds") or []) >= REQUIRED_CASE_KINDS

    # When the checked-in certificate claims certification, both tools must
    # carry live case evidence with the required kinds.
    if payload.get("live_semantic_certified"):
        for tool_id in ("tamarin", "proverif"):
            tool = payload["tools"][tool_id]
            assert tool.get("live_semantic_certified") is True
            kinds = {case.get("kind") for case in tool.get("cases") or []}
            assert REQUIRED_CASE_KINDS <= kinds
            for case in tool.get("cases") or []:
                assert case.get("evidence_class") == "live"
                assert case.get("live_executed") is True
                assert case.get("source_digest")


def test_write_protocol_live_certificate_roundtrip(
    tamarin_cert, protocol_certificate, tmp_path: Path
) -> None:
    out = tmp_path / "formal_verification_protocol_live_certificate.json"
    path = tamarin_cert.write_protocol_live_certificate(
        protocol_certificate,
        repo_root=REPO_ROOT,
        output=out,
    )
    assert path == out
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["interface"] == LIVE_INTERFACE
    assert loaded["goal_id"] == LIVE_GOAL_ID
    assert loaded["certificate_digest_sha256"] == protocol_certificate[
        "certificate_digest_sha256"
    ]


def test_offline_policy_never_installs_during_live(
    tamarin_live: dict[str, Any], proverif_live: dict[str, Any]
) -> None:
    for receipt in (tamarin_live, proverif_live):
        assert receipt["network_used"] is False
        assert receipt["install_attempted"] is False
        assert receipt["download_attempted"] is False
        policy = receipt.get("policy") or {}
        assert policy.get("no_install") is True
        assert policy.get("no_download") is True
        assert policy.get("no_network") is True
