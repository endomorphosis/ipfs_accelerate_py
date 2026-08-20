"""FACP-057: evidence-checked documentation claims."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.evidence_checked_documentation import (
    BUNDLE,
    CLAIM_IR_REQUIREMENTS,
    ClaimCheckDisposition,
    ClaimMode,
    DocsClaimsError,
    DocumentationClaimKind,
    EVIDENCE_SCHEMA,
    FORBIDDEN_EVIDENCE_KINDS,
    GOAL_ID,
    SCHEMA,
    STRONG_CLAIM_TOKENS,
    TASK_ID,
    VOCAB_SCHEMA,
    check_documentation_claims,
    claim_ir_coverage,
    documentation_claim_fixtures,
    evaluate_claim,
    normalize_claim_token,
    parse_controlled_claim,
    parse_controlled_claims,
    parse_exact_evidence_ref,
    render_narrowed_statement,
    requirements_for_token,
)


def _current_verified_envelope(**overrides: str) -> dict[str, str]:
    payload = {
        "origin": "live_observed",
        "integrity": "signature_valid",
        "authority": "valid",
        "policy": "allowed",
        "proof": "verified",
        "freshness": "current",
        "effect": "observed",
        "environment": "live",
        "review": "human_reviewed",
    }
    payload.update(overrides)
    return payload


def _candidate_envelope(**overrides: str) -> dict[str, str]:
    payload = {
        "origin": "hermetic_observed",
        "integrity": "structurally_valid",
        "authority": "unchecked",
        "policy": "unchecked",
        "proof": "candidate",
        "freshness": "stale",
        "effect": "started",
        "environment": "hermetic",
        "review": "unreviewed",
    }
    payload.update(overrides)
    return payload


def test_schema_task_bundle_and_evidence_subset_are_declared() -> None:
    report = check_documentation_claims([])
    assert report.schema == SCHEMA == "facp/docs-claims@1"
    assert EVIDENCE_SCHEMA == "facp/docs-claims@1"
    assert report.task_id == TASK_ID == "FACP-057"
    assert report.goal_id == GOAL_ID == "FACP-G810"
    assert report.bundle == BUNDLE == "facp/release/documentation"
    assert report.vocab_schema == VOCAB_SCHEMA
    assert set(report.evidence_subset) == set(STRONG_CLAIM_TOKENS)
    expected = {
        "supports",
        "production-ready",
        "formally verified",
        "live",
        "current",
        "complete",
        "authenticated",
        "content-addressed",
        "filing-ready",
        "zero-knowledge",
        "cryptographically proven",
    }
    assert set(STRONG_CLAIM_TOKENS) == expected


def test_claim_ir_maps_every_evidence_subset_token() -> None:
    coverage = claim_ir_coverage()
    assert set(coverage) == set(STRONG_CLAIM_TOKENS)
    for token in sorted(STRONG_CLAIM_TOKENS):
        requirement = requirements_for_token(token)
        assert requirement.token == token
        assert requirement.required_predicates
        assert requirement.narrower_statement_template
        assert token in CLAIM_IR_REQUIREMENTS


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("production_ready", "production-ready"),
        ("Formally Verified", "formally verified"),
        ("cryptographically-proven", "cryptographically proven"),
        ("content addressed", "content-addressed"),
        ("zk", "zero-knowledge"),
        ("supported", "supports"),
    ],
)
def test_normalize_claim_token_aliases(raw: str, canonical: str) -> None:
    assert normalize_claim_token(raw) == canonical


def test_unsupported_strong_claim_without_evidence_fails_in_fail_mode() -> None:
    claim = parse_controlled_claim(
        {
            "claim_id": "c:no-evidence-fail",
            "token": "production-ready",
            "raw_text": "Service is production-ready.",
            "mode": "fail",
        }
    )
    result = evaluate_claim(claim)
    assert result.disposition is ClaimCheckDisposition.REJECTED
    assert "production-ready" in result.rendered_text
    assert "missing_exact_evidence" in result.reason_codes
    assert result.evidence_links == ()


def test_unsupported_strong_claim_without_evidence_narrows() -> None:
    claim = parse_controlled_claim(
        {
            "claim_id": "c:no-evidence-narrow",
            "token": "formally verified",
            "raw_text": "Module is formally verified.",
            "subject": "module",
            "mode": "narrow",
        }
    )
    result = evaluate_claim(claim)
    assert result.disposition is ClaimCheckDisposition.NARROWED
    assert "not formally verified" in result.rendered_text.casefold()
    assert "evidence" in result.rendered_text.casefold()
    # Must not retain an unqualified strong assertion.
    assert result.rendered_text.casefold() != "module is formally verified."


def test_formally_verified_with_only_candidate_proof_narrows() -> None:
    report = check_documentation_claims(
        [
            {
                "claim_id": "c:candidate-proof",
                "token": "formally verified",
                "raw_text": "The theorem is formally verified.",
                "mode": ClaimMode.NARROW,
                "evidence": [
                    {
                        "evidence_id": "evidence:candidate",
                        "kind": "proof",
                        "digest": "sha256:cand",
                        "freshness": "current",
                        "envelope": _candidate_envelope(
                            proof="candidate", freshness="current"
                        ),
                    }
                ],
            }
        ]
    )
    result = report.results[0]
    assert result.disposition is ClaimCheckDisposition.NARROWED
    assert result.freshness == "current"
    assert result.evidence_links[0]["evidence_id"] == "evidence:candidate"
    assert result.evidence_links[0]["digest"] == "sha256:cand"
    assert "proof.verified" in result.missing_predicates
    assert "not formally verified" in result.rendered_text.casefold()


def test_live_and_production_ready_reject_simulation_as_live() -> None:
    for token in ("live", "production-ready"):
        result = evaluate_claim(
            parse_controlled_claim(
                {
                    "claim_id": f"c:{token}-sim",
                    "token": token,
                    "raw_text": f"Backend is {token}.",
                    "mode": "narrow",
                    "evidence": [
                        {
                            "evidence_id": "evidence:sim",
                            "kind": "capability",
                            "freshness": "current",
                            "envelope": _candidate_envelope(
                                origin="simulated",
                                environment="hermetic",
                                freshness="current",
                            ),
                        }
                    ],
                }
            )
        )
        assert result.disposition is ClaimCheckDisposition.NARROWED
        assert "simulation_as_live" in result.reason_codes or (
            "hermetic_not_live" in result.reason_codes
        )
        assert "not" in result.rendered_text.casefold()


def test_current_claim_with_stale_freshness_fails_or_narrows() -> None:
    narrowed = evaluate_claim(
        parse_controlled_claim(
            {
                "claim_id": "c:stale-current",
                "token": "current",
                "raw_text": "Receipt is current.",
                "mode": "narrow",
                "evidence": [
                    {
                        "evidence_id": "evidence:stale",
                        "kind": "receipt",
                        "freshness": "stale",
                        "envelope": {"freshness": "stale"},
                    }
                ],
            }
        )
    )
    assert narrowed.disposition is ClaimCheckDisposition.NARROWED
    assert narrowed.freshness == "stale"
    assert "stale_evidence" in narrowed.reason_codes
    assert narrowed.evidence_links[0]["evidence_id"] == "evidence:stale"

    rejected = evaluate_claim(
        parse_controlled_claim(
            {
                "claim_id": "c:stale-current-fail",
                "token": "current",
                "mode": "fail",
                "evidence": [
                    {
                        "evidence_id": "evidence:stale",
                        "kind": "receipt",
                        "freshness": "stale",
                        "envelope": {"freshness": "stale"},
                    }
                ],
            }
        )
    )
    assert rejected.disposition is ClaimCheckDisposition.REJECTED


def test_supported_claim_links_exact_current_evidence() -> None:
    report = check_documentation_claims(
        [
            {
                "claim_id": "c:ok-verified",
                "token": "formally verified",
                "subject": "release gate",
                "raw_text": "The release gate is formally verified.",
                "evidence": [
                    {
                        "evidence_id": "evidence:proof-1",
                        "kind": "proof",
                        "digest": "sha256:ok",
                        "tree_id": "tree:abc",
                        "artifact_path": "proofs/gate.lean",
                        "freshness": "current",
                        "envelope": _current_verified_envelope(),
                    }
                ],
            }
        ]
    )
    result = report.results[0]
    assert result.disposition is ClaimCheckDisposition.ACCEPTED
    assert result.freshness == "current"
    link = result.evidence_links[0]
    assert link["evidence_id"] == "evidence:proof-1"
    assert link["digest"] == "sha256:ok"
    assert link["freshness"] == "current"
    assert link["tree_id"] == "tree:abc"
    assert "formally verified under exact evidence" in result.rendered_text
    assert report.ok is True


def test_markdown_history_and_prose_cannot_serve_as_evidence() -> None:
    for kind in sorted(FORBIDDEN_EVIDENCE_KINDS):
        result = evaluate_claim(
            parse_controlled_claim(
                {
                    "claim_id": f"c:bad-{kind}",
                    "token": "current",
                    "mode": "fail",
                    "evidence": [
                        {
                            "evidence_id": f"evidence:{kind}",
                            "kind": kind,
                            "freshness": "current",
                            "envelope": {"freshness": "current"},
                        }
                    ],
                }
            )
        )
        assert result.disposition is ClaimCheckDisposition.REJECTED
        assert "forbidden_evidence_kind" in result.reason_codes


def test_human_and_heuristic_conclusions_remain_labeled() -> None:
    report = check_documentation_claims(
        [
            {
                "claim_id": "c:human",
                "raw_text": "Operators judge this ready.",
                "conclusion_type": "human",
                "labels": ["heuristic"],
            }
        ]
    )
    claim = report.claims[0]
    result = report.results[0]
    assert claim.kind is DocumentationClaimKind.HUMAN_HEURISTIC
    assert result.disposition is ClaimCheckDisposition.HUMAN_LABELED
    assert "human_conclusion" in result.labels
    assert "heuristic" in result.labels
    assert result.rendered_text.lower().startswith("[human/heuristic]")
    # Must not be rewritten as proof / formally verified.
    assert "formally verified" not in result.rendered_text.casefold()
    assert "proof.verified" not in result.rendered_text.casefold()


def test_renderer_never_strengthens_claims() -> None:
    claim = parse_controlled_claim(
        {
            "claim_id": "c:no-upgrade",
            "token": "formally verified",
            "raw_text": "candidate proof exists",
            "mode": "narrow",
            "evidence": [
                {
                    "evidence_id": "evidence:cand",
                    "kind": "proof",
                    "freshness": "current",
                    "envelope": _candidate_envelope(
                        proof="candidate", freshness="current"
                    ),
                }
            ],
        }
    )
    narrowed = evaluate_claim(claim)
    assert narrowed.disposition is ClaimCheckDisposition.NARROWED
    # Strengthening would emit accepted formally-verified wording.
    assert "under exact evidence" not in narrowed.rendered_text
    assert "not formally verified" in narrowed.rendered_text.casefold()

    with pytest.raises(DocsClaimsError):
        render_narrowed_statement(
            claim,
            disposition=ClaimCheckDisposition.ACCEPTED,
            evidence=None,
        )


def test_content_addressed_accepts_digest_without_requiring_current() -> None:
    result = evaluate_claim(
        parse_controlled_claim(
            {
                "claim_id": "c:cid",
                "token": "content-addressed",
                "evidence": [
                    {
                        "evidence_id": "evidence:digest",
                        "kind": "digest",
                        "digest": "sha256:fff",
                        "freshness": "stale",
                        "envelope": {"integrity": "digest_valid", "freshness": "stale"},
                    }
                ],
            }
        )
    )
    assert result.disposition is ClaimCheckDisposition.ACCEPTED
    assert result.freshness == "stale"
    assert result.evidence_links[0]["digest"] == "sha256:fff"


def test_fixtures_round_trip_through_report() -> None:
    fixtures = documentation_claim_fixtures()
    report = check_documentation_claims(fixtures)
    payload = report.to_dict()
    assert payload["schema"] == SCHEMA
    assert payload["task_id"] == TASK_ID
    assert payload["counts"]["claims"] == len(fixtures)
    assert payload["counts"]["accepted"] >= 1
    assert payload["counts"]["narrowed"] >= 1
    assert payload["counts"]["human_labeled"] >= 1
    assert payload["counts"]["rejected"] >= 1
    assert payload["ok"] is False

    markdown_result = next(
        item
        for item in report.results
        if item.claim_id == "fixture:markdown-evidence-rejected"
    )
    assert markdown_result.disposition is ClaimCheckDisposition.REJECTED
    assert "forbidden_evidence_kind" in markdown_result.reason_codes

    # Forbidden kinds parse as refs but never satisfy claims.
    forbidden_ref = parse_exact_evidence_ref(
        {"evidence_id": "evidence:readme", "kind": "markdown", "freshness": "current"}
    )
    assert forbidden_ref.is_forbidden_kind is True


def test_parse_exact_evidence_ref_requires_identity() -> None:
    with pytest.raises(DocsClaimsError):
        parse_exact_evidence_ref({"kind": "receipt", "freshness": "current"})
    with pytest.raises(DocsClaimsError):
        parse_exact_evidence_ref({"evidence_id": "e1", "kind": ""})


def test_unknown_token_is_fail_closed() -> None:
    with pytest.raises(DocsClaimsError):
        requirements_for_token("totally-assured")
    claim = parse_controlled_claim(
        {"claim_id": "c:plain", "raw_text": "No strong vocabulary here."}
    )
    assert claim.kind is DocumentationClaimKind.NOT_A_CLAIM
    result = evaluate_claim(claim)
    assert "not_a_claim" in result.reason_codes


def test_batch_parse_preserves_order_and_modes() -> None:
    claims = parse_controlled_claims(
        [
            {"claim_id": "a", "token": "live", "mode": "fail"},
            {"claim_id": "b", "token": "supports", "mode": "narrow"},
        ]
    )
    assert [claim.claim_id for claim in claims] == ["a", "b"]
    assert claims[0].mode is ClaimMode.FAIL
    assert claims[1].mode is ClaimMode.NARROW
