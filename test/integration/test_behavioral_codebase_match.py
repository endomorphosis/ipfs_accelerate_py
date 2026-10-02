"""Native v2 intent matching with exact discovery and actual finite checkers."""
from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import behavioral_codebase_match as behavior
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as finite
from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as native
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from test.integration.test_terminal_codebase_semantic_index import prepared
from test.api.test_finite_integer_codebase import finite_tools, finite_text


@pytest.fixture
def options(prepared, finite_tools, tmp_path):
    index, repository, head, descriptor = prepared
    catalog = IntentCodebaseCatalog(index)
    catalog.publish(repository, expected_head=head, manifest_cid=descriptor["manifest_cid"],
        operation_id="discover-exact-contract")
    return dict(catalog=catalog,
        checked_cache=FiniteCheckedCache(FormalVerificationCache(tmp_path / "proof"), index.artifacts),
        repository=repository, expected_head=head, semantic_manifest_cid=descriptor["manifest_cid"],
        intent_document=finite.build_finite_integer_intent(finite_text()), source_text=finite_text(),
        output=tmp_path / "match", tool_policy=finite_tools)


def test_genuine_finite_behavior_keeps_counterexample_and_complete_inventory(options):
    result = behavior.match_behavioral_intent(**options)
    assert result["schema"] == "intent-codebase-match@2"
    assert result["satisfied_requirements"] == [finite.TYPE_STATEMENT_ID]
    assert result["residual_requirements"] == [finite.OFFSET_STATEMENT_ID]
    assert len(result["counterexamples"]) == 5
    assert result["checked_cache"]["status"] == "refuted"
    assert result["complete_inventory"]["inventory_entries"] == 4
    assert len(result["unsupported_units"]) == 3
    assert result["interpretation"]["full_instruction_alignment"]
    assert result["interpretation"]["theorem_domain"] == [-2, -1, 0, 1, 2]
    assert result["model"] == {"enabled": False, "identity": "explicit-model-off@1"}
    assert result["whole_program_verified"] is result["proof_cache_grants_source_authority"] is False
    assert result["reduced_task_population_authorized"] is False
    assert result["match_cid"] == cid_for_structured({key: value for key, value in result.items() if key != "match_cid"})


@pytest.mark.parametrize("change", ["predicate", "argument", "modality"])
def test_valid_but_unsupported_meaning_retains_every_requirement(options, monkeypatch, change):
    value = options["intent_document"].to_dict()
    statement = value["statements"][0]
    if change == "predicate": statement["predicate"] = "absence_of_vulnerability"
    if change == "argument": statement["arguments"][-1] = "99"
    if change == "modality": statement["modality"] = "prohibited"
    def forbidden(**kwargs):
        pytest.fail("unsupported full meaning executed a checker")
    monkeypatch.setattr(native, "observe_finite_integer_source", forbidden)
    result = behavior.match_behavioral_intent(**{**options, "intent_document": value})
    assert result["status"] == "open" and result["current_facts"] == []
    assert result["satisfied_requirements"] == []
    assert result["residual_requirements"] == sorted([finite.TYPE_STATEMENT_ID, finite.OFFSET_STATEMENT_ID])
    assert result["assurance"] == "unresolved"
    assert result["discovery"] is result["checked_cache"] is None


def test_missing_or_tampered_discovery_cannot_satisfy_a_requirement(options):
    catalog = options["catalog"]
    # Corrupt selector membership must fail, including when it would turn the
    # exact native query into an empty answer. The native audit owns refusal.
    catalog._cx.execute("DELETE FROM intent_codebase.selectors WHERE path='calc.py'")
    with pytest.raises(ValueError, match="selector"):
        behavior.match_behavioral_intent(**options)


def test_unselected_contract_does_not_trigger_hidden_fallback(options):
    text = finite_text(offset=1)
    with pytest.raises(ValueError, match="nomination"):
        behavior.match_behavioral_intent(**{**options,
            "source_text": text, "intent_document": finite.build_finite_integer_intent(text)})


def test_changed_source_refuses_prior_matches(options):
    behavior.match_behavioral_intent(**options)
    (options["repository"] / "calc.py").write_text("def increment(n: int) -> int:\n    return n + 2\n")
    with pytest.raises(ValueError):
        behavior.match_behavioral_intent(**{**options, "output": options["output"].parent / "stale"})


def test_caller_evidence_cannot_bypass_owner_execution(options):
    for name in ("receipt", "current_facts", "checked_evidence", "model_prediction", "proof_authority"):
        with pytest.raises(TypeError):
            behavior.match_behavioral_intent(**options, **{name: True})


def test_deleted_observation_rows_cannot_be_relabelled_as_complete(options, monkeypatch):
    original = finite.match_finite_integer_intent
    def altered(**kwargs):
        result = original(**kwargs)
        result["clause_results"].pop()
        result["match_cid"] = cid_for_structured({key: value for key, value in result.items() if key != "match_cid"})
        return result
    monkeypatch.setattr(finite, "match_finite_integer_intent", altered)
    with pytest.raises(ValueError, match="requirement"):
        behavior.match_behavioral_intent(**options)
