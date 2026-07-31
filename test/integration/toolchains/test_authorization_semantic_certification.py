"""Semantic certification for reference Datalog and SecPAL authorization.

FVT-038 / FVT-G102 — ``AuthorizationSemanticCertification@1``.

Acceptance covered:

* both in-process engines exercise allow, deny, unknown, conflict, scoped
  delegation, revocation, negative, and malformed inputs;
* rule, principal, scope, and delegation mutations change or quarantine the
  verdict;
* counterexamples replay deterministically;
* receipts bind the exact policy digest and engine identity;
* certification grants authorization-decision authority only, never theorem
  authority.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CERTIFIER_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "authorization.py"
MANIFEST_PATH = (
    REPO_ROOT
    / "test"
    / "fixtures"
    / "formal_verification"
    / "toolchains"
    / "authorization"
    / "manifest.json"
)

INTERFACE = "AuthorizationSemanticCertification@1"
SCHEMA_VERSION = "authorization-semantic-certification/v1"
MANIFEST_SCHEMA = "authorization-semantic-certification-manifest/v1"
GOAL_ID = "FVT-G102"
TASK_ID = "FVT-038"
LANE_ID = "datalog_secpal"
HANDLER_ID = "authorization_semantic_certification@1"

REQUIRED_ENGINES = {"datalog-authorization", "secpal-authorization"}
REQUIRED_CATEGORIES = {
    "allow",
    "deny",
    "unknown",
    "conflict",
    "delegation",
    "revocation",
    "negative",
    "malformed",
}
REQUIRED_MUTATIONS = {"rule", "principal", "scope", "delegation"}


def _load_certifier():
    assert CERTIFIER_PATH.is_file(), f"missing certifier: {CERTIFIER_PATH}"
    # Ensure datasets package is importable the same way as the validation command.
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (REPO_ROOT, datasets_root):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    spec = importlib.util.spec_from_file_location(
        "authorization_semantic_certification",
        CERTIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certifier():
    return _load_certifier()


@pytest.fixture(scope="module")
def certificate(certifier) -> dict[str, Any]:
    return certifier.certify_authorization_semantics(
        manifest_path=MANIFEST_PATH,
    )


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.is_file(), f"missing manifest: {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Artifact presence / identity
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert CERTIFIER_PATH.is_file()
    assert MANIFEST_PATH.is_file()


def test_certifier_interface_constants(certifier) -> None:
    assert certifier.INTERFACE == INTERFACE
    assert certifier.SCHEMA_VERSION == SCHEMA_VERSION
    assert certifier.GOAL_ID == GOAL_ID
    assert certifier.TASK_ID == TASK_ID
    assert certifier.LANE_ID == LANE_ID
    assert certifier.HANDLER_ID == HANDLER_ID
    assert certifier.AUTHORITY_CEILING == "authorization"
    assert set(certifier.REFERENCE_ENGINES) == REQUIRED_ENGINES


def test_manifest_schema_and_recipes(manifest: dict[str, Any]) -> None:
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["interface"] == INTERFACE
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["lane_id"] == LANE_ID
    assert manifest["handler_id"] == HANDLER_ID
    assert manifest["authority_ceiling"] == "authorization"
    assert manifest["forbids_theorem_authority"] is True
    assert set(manifest["engines"]) == REQUIRED_ENGINES
    assert set(manifest["required_categories"]) == REQUIRED_CATEGORIES
    assert set(manifest["required_mutation_kinds"]) == REQUIRED_MUTATIONS
    assert manifest["policy"]["authorization_decision_authority_only"] is True
    assert manifest["policy"]["no_central_certificate_edit"] is True
    assert manifest["policy"]["no_external_shadow_install"] is True

    recipes = manifest["case_recipes"]
    assert isinstance(recipes, list) and recipes
    categories = {item["category"] for item in recipes}
    assert REQUIRED_CATEGORIES <= categories
    mutation_kinds = {
        item["mutation_kind"]
        for item in recipes
        if item.get("category") == "mutation" and item.get("mutation_kind")
    }
    assert mutation_kinds == REQUIRED_MUTATIONS
    # Compact recipes: no bulk IR dumps.
    for item in recipes:
        assert "authorization_ir" not in item
        assert "document" not in item
        assert item["recipe"]


# ---------------------------------------------------------------------------
# Full semantic certification
# ---------------------------------------------------------------------------


def test_both_engines_are_semantically_certified(
    certificate: dict[str, Any],
) -> None:
    assert certificate["schema_version"] == SCHEMA_VERSION
    assert certificate["interface"] == INTERFACE
    assert certificate["goal_id"] == GOAL_ID
    assert certificate["task_id"] == TASK_ID
    assert certificate["lane_id"] == LANE_ID
    assert certificate["certified"] is True
    assert certificate["authority_ceiling"] == "authorization"
    assert certificate["forbids_theorem_authority"] is True
    assert certificate["policy"]["grants_theorem_authority"] is False
    assert certificate["policy"]["grants_authorization_decision_authority"] is True
    assert set(certificate["engine_ids"]) == REQUIRED_ENGINES

    engines = {item["engine_id"]: item for item in certificate["engines"]}
    assert set(engines) == REQUIRED_ENGINES
    for engine_id, entry in engines.items():
        assert entry["usable"] is True, engine_id
        assert entry["certified"] is True, engine_id
        assert entry["authority_ceiling"] == "authorization"
        assert entry["block_reasons"] == []
        assert entry["checks"], engine_id
        assert all(check["status"] == "passed" for check in entry["checks"]), engine_id
        assert all(check["is_theorem_authority"] is False for check in entry["checks"])


def test_required_categories_exercised(
    certificate: dict[str, Any], certifier
) -> None:
    categories = set(certificate["categories_exercised"])
    assert REQUIRED_CATEGORIES <= categories

    for engine in certificate["engines"]:
        case_ids = {item["case_id"] for item in engine["case_results"]}
        # At least one record per required category (malformed uses case:malformed).
        for category in REQUIRED_CATEGORIES:
            assert any(
                category in case_id or case_id.endswith(category) or f":{category}" in case_id
                or case_id == f"case:{category}"
                for case_id in case_ids
            ), (engine["engine_id"], category, sorted(case_ids))


@pytest.mark.parametrize("category", sorted(REQUIRED_CATEGORIES - {"malformed"}))
def test_category_outcomes_match_expected(
    certifier, category: str
) -> None:
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category == category
    ]
    assert specs, category
    for engine_id in REQUIRED_ENGINES:
        for spec in specs:
            document, query, expected = certifier.materialize_case(spec)
            record = certifier.run_engine_case(
                engine_id, spec.case_id, document, query
            )
            assert record.outcome == expected, (engine_id, spec.case_id)
            assert record.authority == "authorization"
            assert record.is_theorem_authority is False
            assert record.policy_digest
            assert record.engine_id == engine_id


def test_malformed_inputs_fail_closed_never_allow(certifier) -> None:
    for engine_id in REQUIRED_ENGINES:
        record = certifier.run_engine_case(
            engine_id,
            "case:malformed",
            None,
            None,
            expect_error=True,
        )
        assert record.outcome == "error"
        assert record.status in {"error", "quarantined"}
        assert record.outcome != "allow"
        assert record.is_theorem_authority is False


@pytest.mark.parametrize("mutation_kind", sorted(REQUIRED_MUTATIONS))
def test_mutations_change_or_quarantine_verdict(certifier, mutation_kind: str) -> None:
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category == "mutation" and spec.mutation_kind == mutation_kind
    ]
    assert specs, mutation_kind
    for engine_id in REQUIRED_ENGINES:
        for spec in specs:
            base = certifier._fixture_by_id(spec.base_fixture_id)
            baseline = certifier.run_engine_case(
                engine_id,
                f"{spec.case_id}:baseline",
                base.document,
                base.query,
            )
            document, query, expected = certifier.materialize_case(spec)
            mutated = certifier.run_engine_case(
                engine_id, spec.case_id, document, query
            )
            assert mutated.outcome != baseline.outcome, (
                engine_id,
                mutation_kind,
                baseline.outcome,
                mutated.outcome,
            )
            assert mutated.outcome == expected
            assert mutated.policy_digest != baseline.policy_digest
            assert mutated.authority == "authorization"
            assert mutated.is_theorem_authority is False


def test_counterexamples_replay_deterministically(certifier) -> None:
    replay_categories = {"deny", "conflict", "unknown", "revocation", "negative"}
    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category in replay_categories
    ]
    assert specs
    for engine_id in REQUIRED_ENGINES:
        for spec in specs:
            document, query, _expected = certifier.materialize_case(spec)
            first = certifier.run_engine_case(
                engine_id, spec.case_id, document, query
            )
            second = certifier.run_engine_case(
                engine_id, f"{spec.case_id}:replay", document, query
            )
            assert first.outcome == second.outcome
            assert first.policy_digest == second.policy_digest
            assert first.explanation_digest == second.explanation_digest
            assert first.authority == second.authority == "authorization"
            assert first.is_theorem_authority is False


def test_receipts_bind_exact_policy_and_engine(certificate: dict[str, Any]) -> None:
    for engine in certificate["engines"]:
        engine_id = engine["engine_id"]
        for record in engine["case_results"]:
            if record["outcome"] == "error":
                continue
            assert record["engine_id"] == engine_id
            assert record["policy_digest"]
            assert len(record["policy_digest"]) == 64
            assert record["request_digest"]
            assert record["receipt_id"]
            assert record["authority"] == "authorization"
            assert record["is_theorem_authority"] is False


def test_certification_never_grants_theorem_authority(
    certificate: dict[str, Any],
) -> None:
    assert certificate["forbids_theorem_authority"] is True
    assert certificate["policy"]["grants_theorem_authority"] is False
    for engine in certificate["engines"]:
        for check in engine["checks"]:
            assert check["authority"] == "authorization"
            assert check["is_theorem_authority"] is False
        for record in engine["case_results"]:
            assert record["is_theorem_authority"] is False
            assert record["authority"] == "authorization"


def test_lane_handler_reports_certified(certifier) -> None:
    result = certifier.authorization_lane_handler()
    assert result["lane_id"] == LANE_ID
    assert result["handler_id"] == HANDLER_ID
    assert result["certified"] is True
    assert result["status"] == "certified"
    assert result["authority_ceiling"] == "authorization"
    assert result["grants_theorem_authority"] is False
    assert set(result["engine_ids"]) == REQUIRED_ENGINES
    assert result["certificate_digest_sha256"]
    assert len(result["certificate_digest_sha256"]) == 64


def test_certificate_digest_is_stable(certifier) -> None:
    first = certifier.certify_authorization_semantics(manifest_path=MANIFEST_PATH)
    second = certifier.certify_authorization_semantics(manifest_path=MANIFEST_PATH)
    assert first["certificate_digest_sha256"] == second["certificate_digest_sha256"]
    assert first["certified"] is True
    assert second["certified"] is True


def test_datalog_and_secpal_agree_on_semantic_corpus(certifier) -> None:
    """Both reference engines must agree on non-error semantic outcomes."""

    specs = [
        spec
        for spec in certifier.default_case_specs()
        if spec.category != "malformed"
    ]
    for spec in specs:
        document, query, expected = certifier.materialize_case(spec)
        outcomes = {}
        for engine_id in REQUIRED_ENGINES:
            record = certifier.run_engine_case(
                engine_id, spec.case_id, document, query
            )
            outcomes[engine_id] = record.outcome
            assert record.outcome == expected
        assert len(set(outcomes.values())) == 1, (spec.case_id, outcomes)


def test_revocation_flips_allow_to_deny(certifier) -> None:
    document, query, expected = certifier.build_revocation_case()
    assert expected.value == "deny"
    for engine_id in REQUIRED_ENGINES:
        # Baseline allow.
        allow = certifier._fixture_by_category("allow")
        baseline = certifier.run_engine_case(
            engine_id, "revocation-baseline", allow.document, allow.query
        )
        assert baseline.outcome == "allow"
        revoked = certifier.run_engine_case(
            engine_id, "revocation", document, query
        )
        assert revoked.outcome == "deny"
        assert revoked.policy_digest != baseline.policy_digest


def test_scoped_delegation_out_of_scope_is_unknown(certifier) -> None:
    from ipfs_datasets_py.logic.software_verification.authorization import (
        DecisionQuery,
    )

    deleg = certifier._fixture_by_category("delegation")
    out_of_scope = DecisionQuery(
        "query:delegation-out-of-scope",
        principal_id="principal:bob",
        action="read",
        resource="docs/secret/payroll",
        source_ref_ids=deleg.query.source_ref_ids,
        span_ids=deleg.query.span_ids,
    )
    for engine_id in REQUIRED_ENGINES:
        allowed = certifier.run_engine_case(
            engine_id, "delegation-in-scope", deleg.document, deleg.query
        )
        denied = certifier.run_engine_case(
            engine_id, "delegation-out-of-scope", deleg.document, out_of_scope
        )
        assert allowed.outcome == "allow"
        assert denied.outcome == "unknown"


def test_bind_authorization_lane_when_roles_available(certifier) -> None:
    # Roles certification is a dependency (FVT-037); bind when importable.
    try:
        from tools.logic.certification.roles import (
            build_role_aware_policy,
        )
    except Exception:
        pytest.skip("roles certification surface not importable in this worktree")

    policy = build_role_aware_policy(register_placeholders=True)
    bound = certifier.bind_authorization_lane(policy, replace=True)
    handler = bound.get_lane_handler(LANE_ID)
    assert handler is not None
    result = handler()
    assert result["certified"] is True
    assert result["handler_id"] == HANDLER_ID
    assert result["grants_theorem_authority"] is False
