"""PTR-002: executable guards for the test-proof reuse authority doctrine.

These tests intentionally validate the normative threat-model artifact without
importing an optional CID, cache, datasets, IPFS, or ZK implementation. Doctrine
must remain enforceable even when all of those capabilities are unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
THREAT_MODEL = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "TEST_PROOF_REUSE_ZK_THREAT_MODEL.md"
)


@pytest.fixture(scope="module")
def doctrine() -> str:
    assert THREAT_MODEL.is_file(), f"missing PTR ZK threat model: {THREAT_MODEL}"
    text = THREAT_MODEL.read_text(encoding="utf-8")
    assert text.strip(), "PTR ZK threat model must not be empty"
    return text.lower()


def _section(text: str, heading: str) -> str:
    """Return one level-two Markdown section, failing on absent/empty doctrine."""

    match = re.search(
        rf"(?ms)^## {re.escape(heading.lower())}\s*$\n(.*?)(?=^## |\Z)",
        text,
    )
    assert match is not None, f"missing doctrine section: {heading}"
    section = match.group(1).strip()
    assert section, f"empty doctrine section: {heading}"
    return section


def _prose(text: str) -> str:
    """Normalize Markdown wrapping while preserving the asserted wording."""

    return re.sub(r"\s+", " ", text).strip()


def test_document_pins_versioned_statement_and_binary_decision(doctrine: str) -> None:
    assert "zkthreatmodel@1" in doctrine
    assert "testpassstatementv1" in doctrine
    assert "decision actions: `run` or `skip`" in doctrine
    assert "zero stale or false authoritative skips" in doctrine
    assert "`skip` is a fail-closed allow decision" in doctrine
    assert "`run` is the safe default" in doctrine


@pytest.mark.parametrize(
    "required_rule",
    (
        "zk proves possession of the exact trusted pass receipt",
        "ast similarity never means pass",
        "a cid only identifies bytes",
        "simulated zk never skips",
        "every uncertainty executes the test",
    ),
)
def test_non_negotiable_doctrine_is_explicit(
    doctrine: str, required_rule: str
) -> None:
    rules = _section(doctrine, "non-negotiable doctrine")
    assert required_rule in rules


def test_zk_claim_is_receipt_possession_not_changed_code_correctness(
    doctrine: str,
) -> None:
    rules = _prose(_section(doctrine, "non-negotiable doctrine"))
    statement = _prose(
        _section(doctrine, "`testpassstatementv1` claim boundary")
    )

    assert "does not prove that changed code passes" in rules
    assert "not a proof of general program correctness" in rules
    assert "private canonical receipt bytes hash" in statement
    assert "public receipt cid" in statement
    assert "setup, call, and teardown outcome bits are all pass" in statement
    assert "disqualifying bits are clear" in statement
    assert "issuer signature or commitment" in statement


def test_ast_traces_and_cids_have_no_pass_authority(doctrine: str) -> None:
    lattice = _prose(_section(doctrine, "authority lattice and allow decision"))
    trace = _prose(
        _section(doctrine, "trace completeness and semantic similarity")
    )

    assert "valid cid over retained canonical bytes" in lattice
    assert "establish exact byte identity only" in lattice
    assert "never establish meaning, trust, freshness, or pass" in lattice
    assert "static ast and runtime traces" in trace
    assert "they are not positive outcome evidence" in trace
    for forbidden_authority in (
        "similarity threshold",
        "embedding score",
        "model verdict",
        "runtime overlap",
        "unchanged-line heuristic",
    ):
        assert forbidden_authority in trace
    assert "participates in the `skip` authority calculation" in trace


def test_skip_requires_the_complete_conjunctive_authority_chain(
    doctrine: str,
) -> None:
    lattice = _prose(_section(doctrine, "authority lattice and allow decision"))

    required_bindings = (
        "retained canonical receipt and certificate bytes",
        "trusted under the admitted runner/issuer policy",
        "setup, call, and teardown",
        "freshly recomputed exact current execution-key cid",
        "trace policy is admitted and reports complete",
        "reuse policy",
        "circuit",
        "verification key",
        "real approved backend",
        "independently verified by the local pinned verifier",
        "bounded verification",
    )
    for binding in required_bindings:
        assert binding in lattice, f"authority chain missing binding: {binding}"

    assert "all required checks below succeed conjunctively" in lattice
    assert "this is a conjunction, not a score or majority vote" in lattice
    assert "failure or uncertainty in any item" in lattice
    assert "chooses `run`" in lattice


def test_simulated_zk_is_permanently_non_authoritative(doctrine: str) -> None:
    lattice = _prose(_section(doctrine, "authority lattice and allow decision"))
    backend = _prose(
        _section(doctrine, "backend qualification and downgrade resistance")
    )

    assert "simulated, mock, provider-asserted, unavailable, or unverified" in lattice
    assert "always `run`" in lattice
    assert "its authority is `non_attested`" in backend
    assert "cannot be upgraded by a provider flag" in backend
    assert "failure of a real prover or verifier does not retry through a" in backend
    assert "simulated backend" in backend
    assert "executes the test" in backend


def test_every_named_uncertainty_and_attack_fails_to_run(doctrine: str) -> None:
    threats = _section(doctrine, "threats and required fail-closed responses")

    required_threats = (
        "receipt forgery",
        "receipt substitution",
        "proof replay or rollback",
        "ast similarity promoted to pass",
        "trace incompleteness",
        "cid semantic confusion",
        "cross-profile cid confusion",
        "circuit or verification-key confusion",
        "issuer or trust-domain confusion",
        "public-input omission",
        "simulated-backend mislabeling / downgrade",
        "provider authority spoofing",
        "malformed, oversized, or malleable proof",
        "witness leakage",
        "cache/index poisoning",
        "mutable-state race",
        "optional capability failure",
        "verification timeout or resource exhaustion",
    )
    for threat in required_threats:
        matching_row = next(
            (line for line in threats.splitlines() if f"| {threat} |" in line),
            None,
        )
        assert matching_row is not None, f"missing threat row: {threat}"
        assert "`run`" in matching_row, f"threat does not explicitly choose RUN: {threat}"


def test_public_inputs_prevent_replay_substitution_and_key_confusion(
    doctrine: str,
) -> None:
    bindings = _prose(
        _section(doctrine, "protected assets and data classification")
    )
    for required_binding in (
        "statement schema and version",
        "proof-system",
        "circuit",
        "public-input schema",
        "setup manifest",
        "verification-key cids",
        "receipt cid",
        "exact current execution-key cid",
        "trace roots",
        "reuse-policy cid",
        "issuer/key commitment",
        "revocation epoch",
        "repository-forest",
        "verifier domain",
        "nonce or challenge",
        "expiry",
    ):
        assert required_binding in bindings, f"missing public binding: {required_binding}"

    replay = _prose(_section(doctrine, "replay, substitution, and freshness"))
    assert "rebuild the statement from the current item" in replay
    assert "not from an index record or prover-provided statement" in replay
    assert "time alone is not freshness" in replay


def test_private_witness_is_excluded_from_every_public_boundary(
    doctrine: str,
) -> None:
    assets = _prose(
        _section(doctrine, "protected assets and data classification")
    )
    leakage = _prose(_section(doctrine, "witness leakage controls"))

    for forbidden_surface in (
        "public certificate",
        "receipt index",
        "cas metadata record",
        "prompt",
        "trace",
        "log",
        "event",
        "metric label",
        "exception",
        "crash report",
        "pytest skip reason",
    ):
        assert forbidden_surface in assets
    assert "never sent to the verifier" in assets
    assert "never written to a proving-request cache" in assets
    assert "never cache a proving request containing private material" in leakage
    assert "allowlist public reason codes" in leakage


def test_decision_procedure_has_no_unknown_to_skip_transition(doctrine: str) -> None:
    decision = _section(doctrine, "decision procedure and audit contract")

    assert "every exact check succeeds                   => skip" in decision
    assert decision.count("=> run") >= 6
    assert "there is no implicit truthy result" in decision
    assert "no `unknown -> skip` transition" in decision
    assert "runs the test normally" in decision
    assert "ipfs_test_proof_reuse_mode=off" in _section(
        doctrine, "required validation population"
    )
