"""Bounded observation tests; literal identities provide no native authority.

The fixture follows the native worker/capsule/admission wire shapes. It does
not run the semantic producer, authenticate CIDs, or qualify source freshness.
Those independent checks remain the preparation caller's responsibility.
"""

from copy import deepcopy
from enum import Enum
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import context_selection_observation as observation


PRIVATE_OBJECTIVE = "PRIVATE_OBJECTIVE café 🧪 must remain outside the sidecar"
PRIVATE_QUERY = "PRIVATE_QUERY target helper"
PRIVATE_SOURCE = "# PRIVATE_SOURCE_BODY café 🧪\n"
_DEFAULT_DIGEST = object()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def capsule(name, source_cid, *, dependencies=(), raw_required=False):
    """A producer-shaped literal, deliberately not an authenticated capsule."""
    symbol = "literal-symbol:" + name
    value = {
        "schema": "ipfs-datasets.software-contracts.semantic-capsule@1",
        "stable_symbol_id": symbol,
        "version_cid": "literal-version:" + name,
        "semantic_index_schema": "literal-semantic-index@1",
        "extractor_version": "literal-extractor@1",
        "capsule_schema": "ipfs-datasets.software-contracts.semantic-capsule@1",
        "capsule_compiler_version": "literal-capsule-compiler@1",
        "source_slice_path": name + ".py",
        "source_cid": source_cid,
        "symbol_fact_cid": "literal-fact:" + name,
        "signature": {"authored_hint": "PRIVATE_SIGNATURE_BODY"},
        "annotations": {}, "defaults": {}, "decorators": [],
        "contracts": {}, "effects": [], "exception_behavior": {},
        "schema_relations": [], "serialization_relations": [],
        "test_refs": [], "fixture_refs": [],
        "proof_obligation_refs": ["literal-obligation:" + name],
        "dependency_stable_ids": list(dependencies),
        "dependency_version_cids": ["literal-dependency-version:" + name],
        "dependency_fact_cids": ["literal-dependency-fact:" + name],
        "dependency_link_ids": ["literal-dependency-link:" + name],
        "confidence": "exact", "relevant_binding_projection_cid": None,
        "docstring_hint": "PRIVATE_DOCSTRING_HINT",
        "metadata": {}, "capsule_cid": "literal-capsule:" + name,
    }
    admission = {
        "schema": "ipfs-accelerate.semantic-capsule-admission@1",
        "interface": "SemanticCapsuleAdmission@1",
        "ref": {
            "capsule_cid": value["capsule_cid"],
            "semantic_state_root_cid": "literal-semantic-root",
            "stable_symbol_id": symbol, "version_cid": value["version_cid"],
            "source_cid": source_cid, "confidence": "exact",
            "validity_bindings": ["literal-validity:" + name],
            "raw_source_required": raw_required,
        },
        "admission": "raw_source_required" if raw_required else "exact_substitute",
        "freshness": "fresh", "caveats": [],
        "assessment_cid": "literal-assessment:" + name,
    }
    return value, admission


@pytest.fixture
def payload():
    sources = {
        "dependency.py": "def helper(value):\n    return value * 2\n",
        "instruction.md": PRIVATE_SOURCE,
        "target.py": "def target(value):\n    return value + 1\n",
    }
    manifest = {path: {"sha256": sha(text.encode()),
                       "source_cid": "literal-source:" + path}
                for path, text in sources.items()}
    target, target_admission = capsule("target", manifest["target.py"]["source_cid"],
        dependencies=("literal-symbol:dependency", "literal-symbol:outside-scope"),
        raw_required=True)
    dependency, dependency_admission = capsule(
        "dependency", manifest["dependency.py"]["source_cid"])
    return {
        "schema": "supervisor-semantic-worker-context@2",
        "task_id": "LITERAL-OBSERVATION-001", "objective": PRIVATE_OBJECTIVE,
        "scope_cid": "literal-scope", "semantic_root_cid": "literal-semantic-root",
        "manifest": manifest, "required_raw_paths": ["instruction.md", "target.py"],
        "program_paths": ["dependency.py", "target.py"],
        "preparation_bounds": {"max_symbols": 256, "context_input_tokens": 8192},
        "doctor_snapshot_id": "literal-doctor-snapshot",
        "doctor_manifest_cid": "literal-doctor-manifest",
        "reconstruction": {
            "schema": "supervisor-scoped-semantic-reconstruction@2",
            "scope_cid": "literal-scope", "semantic_root_cid": "literal-semantic-root",
            "nomination_matched": True, "reuse": "cold", "full_repository": False,
            "semantic_acceptance_authority": False, "completion_authority": False,
            "program_paths": ["dependency.py", "target.py"],
            "semantic_symbol_count": 4, "captured_source_count": 3,
            "program_source_count": 2, "support_source_count": 1,
            "doctor_source_paths": ["dependency.py", "target.py"],
        },
        "refresh_lineage": None, "completion_authority": False,
        "capsules": [target, dependency],
        "admissions": [target_admission, dependency_admission],
        "raw_sources": {path: sources[path] for path in ("instruction.md", "target.py")},
        "pack": {
            "schema": "ipfs-accelerate.context-pack-result@1",
            "interface": "ContextPack@1", "pack_cid": "literal-pack",
            "pack": {
                "objective": PRIVATE_OBJECTIVE, "target_source_cid": "literal-scope",
                "surrounding_source_cid": "literal-scope", "test_source_cid": "literal-raw-scope",
                "dependency_capsule_cids": ["literal-capsule:dependency"],
                "obligation_cids": [], "counterexample_cids": [],
                "delta_cid": "literal-delta", "interface_cids": [],
                "assumptions": ["Initial scoped context, not a full-repository world snapshot."],
                "exclusions": [], "token_totals": {"invariant": 100},
                "estimator_version": "context-compiler-calibrated_utf8@1",
                "risk": "low", "route": "literal-route", "escalation_recommendation": "",
            },
            "references": [], "token_estimate": {
                "totals": {"invariant": 100}, "total": 100,
                "estimator_version": "context-compiler-calibrated_utf8@1",
            },
            "coverage_satisfied": True, "production_slice_cid": None,
            "budget_exceeded": False, "decisions": [],
        },
        "worker_projection": {
            "schema": "supervisor-semantic-worker-projection@1",
            "selector": "lexical-query-symbol-name@1", "query": PRIVATE_QUERY,
            "max_capsules": 8, "max_bytes": 32768,
            "full_capsule_count": 4, "selected_capsule_count": 2,
            "omitted_capsule_count": 2, "exact_byte_budget_repacked": True,
            "selected_symbols": [
                {"path": value["source_slice_path"], "qualified_name": name,
                 "stable_symbol_id": value["stable_symbol_id"]}
                for name, value in (("target", target), ("helper", dependency))
            ],
            "capsule_index_cid": "literal-capsule-index",
            "raw_source_fetch_required": {"dependency.py": manifest["dependency.py"]},
            "omission_reason": "explicit bounded worker projection; full source and capsule index retained",
            "raw_required_caveat": "Read exact repository source before editing.",
            "semantic_equivalence_claimed": False, "completion_authority": False,
        },
    }


def build(payload, *, role="coding", payload_bytes=None):
    return observation.build_context_selection_observation(payload=payload,
        payload_bytes=encoded(payload) if payload_bytes is None else payload_bytes, role=role)


def verify(value, payload, *, role="coding", raw=None, expected_sha256=_DEFAULT_DIGEST):
    raw = encoded(value) if raw is None else raw
    return observation.verify_context_selection_observation(raw=raw,
        expected_sha256=sha(raw) if expected_sha256 is _DEFAULT_DIGEST else expected_sha256,
        payload=payload, payload_bytes=encoded(payload), role=role)


@pytest.mark.parametrize("role", ["coding", "planning", "review"])
def test_round_trip_binds_exact_payload_role_and_producer_references(payload, role):
    raw = encoded(payload)
    result = build(payload, role=role)
    assert result["schema"] == "supervisor-context-selection-observation@1"
    assert result["observation_stage"] == "native_semantic_context_preparation"
    assert result["role"] == role and result["role_basis"] == "owner_declared"
    assert result["payload"] == {"schema": payload["schema"], "sha256": sha(raw), "bytes": len(raw)}
    assert result["task_id"] == payload["task_id"]
    assert result["objective_sha256"] == sha(PRIVATE_OBJECTIVE.encode())
    assert result["bindings"] == {
        "scope_cid": payload["scope_cid"], "semantic_root_cid": payload["semantic_root_cid"],
        "pack_cid": payload["pack"]["pack_cid"],
        "doctor_snapshot_id": payload["doctor_snapshot_id"],
        "doctor_manifest_cid": payload["doctor_manifest_cid"],
    }
    assert observation.encode_context_selection_observation(result) == encoded(result)
    assert verify(result, payload, role=role) == result


def test_indexed_selection_retains_snapshot_identity_without_authority(payload):
    projection = payload["worker_projection"]
    projection.update(selector="duckdb-query-symbol-name@1",
        candidate_snapshot_cid="literal-indexed-candidate-snapshot")
    before = deepcopy(payload)
    result = build(payload)
    assert result["selection"]["selector"] == projection["selector"]
    assert result["selection"]["candidate_snapshot_cid"] == projection["candidate_snapshot_cid"]
    assert result["selection"]["ranked_symbol_ids"] == []
    assert result["selection"]["retrieval_sha256"] is None
    assert all(value is False for value in result["authority"].values())
    assert verify(result, payload) == result
    assert payload == before
    # Rehashing an observation cannot replace the producer's candidate scope.
    result["selection"]["candidate_snapshot_cid"] = "literal-foreign-candidates"
    with pytest.raises(ValueError, match="native selection replay"):
        verify(result, payload)


@pytest.mark.parametrize("damage", [
    "missing_snapshot", "empty_snapshot", "nonstring_snapshot", "lexical_snapshot",
    "retrieval_snapshot", "implicit_program_scope", "extra_field",
])
def test_indexed_observation_requires_closed_snapshot_and_program_scope(payload, damage):
    projection = payload["worker_projection"]
    projection.update(selector="duckdb-query-symbol-name@1",
        candidate_snapshot_cid="literal-indexed-candidate-snapshot")
    if damage == "missing_snapshot":
        del projection["candidate_snapshot_cid"]
    elif damage == "empty_snapshot":
        projection["candidate_snapshot_cid"] = ""
    elif damage == "nonstring_snapshot":
        projection["candidate_snapshot_cid"] = True
    elif damage == "lexical_snapshot":
        projection["selector"] = "lexical-query-symbol-name@1"
    elif damage == "retrieval_snapshot":
        projection.update(schema="supervisor-semantic-worker-projection@2",
            selector="verified-retrieval-symbol-rank@1", ranked_symbol_ids=[],
            retrieval={"artifact": "selection.json", "sha256": "1" * 64},
            unranked_policy="lexical-query-symbol-name@1")
    elif damage == "implicit_program_scope":
        payload["schema"] = "supervisor-semantic-worker-context@1"
        del payload["program_paths"]
        payload["reconstruction"]["schema"] = "supervisor-scoped-semantic-reconstruction@1"
    else:
        projection["sql_rank_authorizes_omission"] = True
    with pytest.raises(ValueError):
        build(payload)


def test_native_reference_metadata_tuples_preserve_the_exact_json_array_wire(payload):
    # ContextPack's reference metadata can retain immutable coverage_ids from
    # ContextReference.to_dict(). JSON renders that tuple as an ordinary array.
    payload["pack"]["references"] = [{
        "schema": "ipfs_accelerate_py/agent-supervisor/context-reference@1",
        "contract_version": 1, "reference_id": "literal-required-reference",
        "kind": "target_source", "tier": "invariant",
        "referenced_content_id": "literal-source:target.py",
        "repository_id": "literal-repository", "tree_id": "literal-scope",
        "path": "target.py", "summary": "PRIVATE_REFERENCE_SUMMARY",
        "byte_count": 0, "token_count": 0,
        "metadata": {"required": True, "coverage_ids": ("target_source",)},
    }]
    before = deepcopy(payload)
    raw = encoded(payload)
    normalized = json.loads(raw)
    assert normalized["pack"]["references"][0]["metadata"]["coverage_ids"] == ["target_source"]
    result = build(payload, payload_bytes=raw)
    assert result == build(normalized, payload_bytes=raw)
    assert result["payload"]["sha256"] == sha(raw)
    assert verify(result, payload) == verify(result, normalized) == result
    assert payload == before
    assert type(payload["pack"]["references"][0]["metadata"]["coverage_ids"]) is tuple
    assert "PRIVATE_REFERENCE_SUMMARY" not in encoded(result).decode()


def test_native_reference_tier_enum_is_normalized_at_the_existing_json_wire_boundary(payload):
    # The native ContextTier is a (str, Enum), and ContextReference.to_dict()
    # retains it because the existing canonicalizer accepts strings first.
    class NativeTier(str, Enum):
        INVARIANT = "invariant"

    payload["pack"]["references"] = [{
        "schema": "ipfs_accelerate_py/agent-supervisor/context-reference@1",
        "contract_version": 1, "reference_id": "literal-required-reference",
        "kind": "target_source", "tier": NativeTier.INVARIANT,
        "referenced_content_id": "literal-source:target.py",
        "repository_id": "literal-repository", "tree_id": "literal-scope",
        "path": "target.py", "summary": "PRIVATE_ENUM_REFERENCE_SUMMARY",
        "byte_count": 0, "token_count": 0,
        "metadata": {"required": True, "coverage_ids": ["target_source"]},
    }]
    before = deepcopy(payload)
    raw = encoded(payload)
    normalized = json.loads(raw)
    assert type(normalized["pack"]["references"][0]["tier"]) is str
    assert normalized["pack"]["references"][0]["tier"] == "invariant"
    # The native caller already emitted these exact bytes. The observer reads
    # their plain JSON snapshot, without broadening its Python input contract.
    with pytest.raises(ValueError):
        build(payload, payload_bytes=raw)
    result = build(normalized, payload_bytes=raw)
    assert result["payload"]["sha256"] == sha(raw)
    assert verify(result, normalized) == result
    with pytest.raises(ValueError):
        verify(result, payload)
    assert payload == before
    assert payload["pack"]["references"][0]["tier"] is NativeTier.INVARIANT
    assert "PRIVATE_ENUM_REFERENCE_SUMMARY" not in encoded(result).decode()


def test_plain_enum_is_not_coerced_into_native_json_string(payload):
    class NonJsonTier(Enum):
        INVARIANT = "invariant"

    payload["pack"]["references"] = [{"tier": "invariant"}]
    raw = encoded(payload)
    payload["pack"]["references"][0]["tier"] = NonJsonTier.INVARIANT
    # A matching value attribute does not make this object JSON-backed.
    with pytest.raises(ValueError):
        build(payload, payload_bytes=raw)


@pytest.mark.parametrize("kind", ["list", "tuple", "dict", "int", "str"])
def test_tuple_compatibility_does_not_admit_arbitrary_json_type_subclasses(payload, kind):
    class CustomList(list):
        pass

    class CustomTuple(tuple):
        pass

    class CustomDict(dict):
        pass

    class CustomInt(int):
        pass

    class CustomStr(str):
        pass

    values = {
        "list": CustomList(["literal"]), "tuple": CustomTuple(("literal",)),
        "dict": CustomDict({"literal": 1}), "int": CustomInt(1),
        "str": CustomStr("literal"),
    }
    payload["capsules"][0]["metadata"]["custom_type"] = values[kind]
    # These subclasses serialize through json.dumps, so byte equality alone
    # must not broaden the bounded caller-object traversal contract.
    raw = encoded(payload)
    with pytest.raises(ValueError):
        build(payload, payload_bytes=raw)


def test_final_selection_preserves_order_and_distinguishes_unresolved_dependencies(payload):
    result = build(payload)
    assert result["selection"]["full_capsule_count"] == 4
    assert result["selection"]["selected_capsule_count"] == 2
    assert result["selection"]["omitted_capsule_count"] == 2
    assert result["selection"]["exact_byte_budget_repacked"] is True
    assert result["selection"]["query_sha256"] == sha(PRIVATE_QUERY.encode())
    assert [row["stable_symbol_id"] for row in result["capsules"]] == [
        "literal-symbol:target", "literal-symbol:dependency"]
    assert [row["ordinal"] for row in result["capsules"]] == [0, 1]
    target = result["capsules"][0]
    assert target["dependency_selection"] == {
        "selected_stable_ids": ["literal-symbol:dependency"],
        "not_selected_or_unresolved_stable_ids": ["literal-symbol:outside-scope"],
    }
    assert target["proof_obligation_refs"] == ["literal-obligation:target"]
    assert target["admission"]["sha256"] == sha(encoded(payload["admissions"][0]))
    assert target["admission"]["raw_source_required"] is True


def test_required_support_and_fetch_handles_retain_distinct_meanings(payload):
    result = build(payload)
    assert [row["path"] for row in result["sources"]] == sorted(payload["manifest"])
    rows = {row["path"]: row for row in result["sources"]}
    assert rows["instruction.md"] == {
        "path": "instruction.md", **payload["manifest"]["instruction.md"],
        "inline_raw": True, "required_raw": True, "fetch_required": False,
        "program_source": False, "selected_capsule_requires_raw": False,
    }
    assert rows["target.py"]["selected_capsule_requires_raw"] is True
    assert rows["target.py"]["inline_raw"] is True
    assert rows["dependency.py"]["program_source"] is True
    assert rows["dependency.py"]["fetch_required"] is True
    assert rows["dependency.py"]["inline_raw"] is False


def test_observation_neither_discloses_bodies_nor_promotes_native_authority(payload):
    result = build(payload)
    assert result["authority"] == {name: False for name in (
        "omission_authority", "proof_authority", "completion_authority", "execution_authority",
        "mutation_authority", "provider_delivery_verified", "source_freshness_verified",
        "semantic_completeness_claimed", "full_dependency_graph_claimed")}
    assert result["provider_token_usage"] is None and result["token_savings"] is None
    assert result["modality_evidence"]["codebase"] == {
        "status": "structural_producer_observed", "semantic_root_cid": payload["semantic_root_cid"]}
    for modality in ("intent", "security", "ui_ux"):
        assert result["modality_evidence"][modality] == {"status": "not_observed_in_this_payload"}
    text = encoded(result).decode()
    for marker in (PRIVATE_OBJECTIVE, PRIVATE_QUERY, "PRIVATE_SOURCE_BODY",
                   "PRIVATE_SIGNATURE_BODY", "PRIVATE_DOCSTRING_HINT"):
        assert marker not in text
    assert any("does not authenticate" in caveat for caveat in result["scope_caveats"])
    assert any("not selected does not mean" in caveat for caveat in result["scope_caveats"])


def test_pure_observer_does_not_claim_cid_authentication_or_touch_files(payload, monkeypatch):
    import builtins
    from pathlib import Path

    def forbidden(*args, **kwargs):
        raise AssertionError("observation must not acquire source or catalog ownership")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    before = deepcopy(payload)
    result = build(payload)
    assert payload == before
    assert verify(result, payload) == result
    payload["capsules"][0]["dependency_stable_ids"].append("literal-symbol:later-mutation")
    payload["manifest"]["target.py"]["source_cid"] = "literal-source:later-mutation"
    assert result["capsules"][0]["dependency_selection"]["not_selected_or_unresolved_stable_ids"] == [
        "literal-symbol:outside-scope"]
    assert next(row for row in result["sources"] if row["path"] == "target.py")["source_cid"] == "literal-source:target.py"


def test_all_capsules_removed_by_final_budget_is_observed_without_inventing_excluded_handles(payload):
    payload["capsules"] = []
    payload["admissions"] = []
    payload["worker_projection"].update(max_capsules=0, selected_capsule_count=0,
        omitted_capsule_count=4, selected_symbols=[])
    result = build(payload)
    assert result["capsules"] == []
    assert result["selection"]["selected_capsule_count"] == 0
    assert result["selection"]["omitted_capsule_count"] == 4
    assert all(row["selected_capsule_requires_raw"] is False for row in result["sources"])
    assert verify(result, payload) == result


def test_unprojected_legacy_payload_has_no_invented_query_or_program_scope(payload):
    payload["schema"] = "supervisor-semantic-worker-context@1"
    payload.pop("program_paths")
    payload.pop("worker_projection")
    payload["reconstruction"] = {
        "schema": "supervisor-scoped-semantic-reconstruction@1",
        "scope_cid": payload["scope_cid"], "semantic_root_cid": payload["semantic_root_cid"],
        "nomination_matched": True, "reuse": "cold", "full_repository": False,
        "semantic_acceptance_authority": False, "completion_authority": False,
    }
    result = build(payload)
    selection = result["selection"]
    assert selection["selector"] == "full_producer_population"
    assert selection["projection_schema"] is None and selection["query_sha256"] is None
    assert selection["full_capsule_count"] == selection["selected_capsule_count"] == 2
    assert selection["omitted_capsule_count"] == 0
    assert all(row["program_source"] is None for row in result["sources"])
    assert verify(result, payload) == result


def test_retrieval_projection_records_ranking_without_upgrading_its_authority(payload):
    selection = {"artifact": ".semantic/retrieval.json", "sha256": "1" * 64}
    payload["worker_projection"].update(schema="supervisor-semantic-worker-projection@2",
        selector="verified-retrieval-symbol-rank@1", retrieval=selection,
        ranked_symbol_ids=["literal-symbol:dependency"], unranked_policy="lexical-query-symbol-name@1")
    result = build(payload)
    assert result["selection"]["retrieval_sha256"] == sha(encoded(selection))
    assert result["selection"]["ranked_symbol_ids"] == ["literal-symbol:dependency"]
    assert result["authority"]["source_freshness_verified"] is False
    assert verify(result, payload) == result


@pytest.mark.parametrize("role", [None, True, 1, "", "Coding", " coding", "coding ", "execution", [], {}])
def test_only_explicit_closed_roles_are_accepted(payload, role):
    with pytest.raises(ValueError):
        build(payload, role=role)


@pytest.mark.parametrize("mutation", ["unknown_field", "unknown_schema", "untyped_authority"])
def test_native_payload_schema_cannot_be_extended_or_downgraded(payload, mutation):
    if mutation == "unknown_field":
        payload["caller_supplied_omission_authority"] = True
    elif mutation == "unknown_schema":
        payload["schema"] = "supervisor-semantic-worker-context@999"
    else:
        payload["completion_authority"] = 0
    with pytest.raises(ValueError):
        build(payload)


@pytest.mark.parametrize("representation", ["pretty", "trailing_newline", "different_payload", "text", "bytearray"])
def test_payload_bytes_must_be_exact_canonical_bytes_for_the_supplied_payload(payload, representation):
    raw = encoded(payload)
    if representation == "pretty":
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2).encode()
    elif representation == "trailing_newline":
        raw += b"\n"
    elif representation == "different_payload":
        other = deepcopy(payload)
        other["task_id"] = "FOREIGN-TASK"
        raw = encoded(other)
    elif representation == "text":
        raw = raw.decode()
    else:
        raw = bytearray(raw)
    with pytest.raises(ValueError):
        build(payload, payload_bytes=raw)


@pytest.mark.parametrize("field", ["task_id", "scope_cid", "semantic_root_cid", "doctor_snapshot_id", "doctor_manifest_cid", "objective"])
def test_saved_observation_cannot_be_reused_for_another_native_payload(payload, field):
    result = build(payload)
    payload[field] = "literal-foreign-binding"
    with pytest.raises(ValueError):
        verify(result, payload)


@pytest.mark.parametrize("tamper", ["role", "task", "source", "selected", "dependency", "authority", "unknown", "usage", "count_type"])
def test_rehashing_tampered_observation_does_not_establish_provenance(payload, tamper):
    result = build(payload)
    if tamper == "role":
        result["role"] = "planning"
    elif tamper == "task":
        result["task_id"] = "FOREIGN-TASK"
    elif tamper == "source":
        result["sources"][0]["sha256"] = "0" * 64
    elif tamper == "selected":
        result["capsules"].reverse()
    elif tamper == "dependency":
        result["capsules"][0]["dependency_selection"]["not_selected_or_unresolved_stable_ids"] = []
    elif tamper == "authority":
        result["authority"]["omission_authority"] = True
    elif tamper == "unknown":
        result["invented_verified_fact"] = True
    elif tamper == "usage":
        result["provider_token_usage"] = 1
    else:
        result["selection"]["selected_capsule_count"] = True
    # A newly computed SHA authenticates none of these producer bindings.
    with pytest.raises(ValueError):
        verify(result, payload)


@pytest.mark.parametrize("value", [0, None, "false", [], {}])
def test_authority_flags_require_actual_false_not_false_like_values(payload, value):
    result = build(payload)
    result["authority"]["proof_authority"] = value
    with pytest.raises(ValueError):
        verify(result, payload)


@pytest.mark.parametrize("raw", [b"[]", b"null", b"{", b"\xff", b'{"schema":NaN}', b'{"schema":Infinity}'])
def test_invalid_observation_wire_is_refused_even_with_matching_digest(payload, raw):
    with pytest.raises(ValueError):
        verify({}, payload, raw=raw)


def test_duplicate_observation_fields_are_refused_even_when_identical(payload):
    result = build(payload)
    raw = encoded(result)
    raw = b'{"role":"coding",' + raw[1:]
    with pytest.raises(ValueError):
        verify(result, payload, raw=raw)


def test_duplicate_nested_observation_fields_are_refused_even_when_identical(payload):
    result = build(payload)
    raw = encoded(result).replace(b'"authority":{', b'"authority":{"proof_authority":false,', 1)
    with pytest.raises(ValueError):
        verify(result, payload, raw=raw)


@pytest.mark.parametrize("digest", ["0" * 64, "A" * 64, "a" * 63, "", None, True])
def test_observation_digest_is_explicit_and_exact(payload, digest):
    result = build(payload)
    with pytest.raises(ValueError):
        verify(result, payload, expected_sha256=digest)


@pytest.mark.parametrize("mutation", ["raw_sha", "missing_required", "unknown_raw", "capsule_source", "admission_source", "admission_capsule", "admission_root", "count", "count_bool", "selection_identity", "duplicate_capsule", "admission_length", "escaping_path", "projection_authority"])
def test_structural_source_and_final_selection_inconsistencies_are_refused(payload, mutation):
    if mutation == "raw_sha":
        payload["manifest"]["target.py"]["sha256"] = "0" * 64
    elif mutation == "missing_required":
        payload["raw_sources"].pop("instruction.md")
    elif mutation == "unknown_raw":
        payload["raw_sources"]["not-captured.py"] = "invented"
    elif mutation == "capsule_source":
        payload["capsules"][0]["source_cid"] = "literal-foreign-source"
    elif mutation == "admission_source":
        payload["admissions"][0]["ref"]["source_cid"] = "literal-foreign-source"
    elif mutation == "admission_capsule":
        payload["admissions"][0]["ref"]["capsule_cid"] = "literal-foreign-capsule"
    elif mutation == "admission_root":
        payload["admissions"][0]["ref"]["semantic_state_root_cid"] = "literal-foreign-root"
    elif mutation == "count":
        payload["worker_projection"]["selected_capsule_count"] += 1
    elif mutation == "count_bool":
        payload["worker_projection"]["selected_capsule_count"] = True
    elif mutation == "selection_identity":
        payload["worker_projection"]["selected_symbols"][0]["stable_symbol_id"] = "literal-foreign-symbol"
    elif mutation == "duplicate_capsule":
        payload["capsules"][1] = deepcopy(payload["capsules"][0])
        payload["admissions"][1] = deepcopy(payload["admissions"][0])
    elif mutation == "admission_length":
        payload["admissions"].pop()
    elif mutation == "escaping_path":
        payload["manifest"]["../private.py"] = payload["manifest"].pop("dependency.py")
    else:
        payload["worker_projection"]["semantic_equivalence_claimed"] = True
    with pytest.raises(ValueError):
        build(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected_even_in_unreported_native_metadata(payload, value):
    payload["capsules"][0]["metadata"]["irrelevant_number"] = value
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    with pytest.raises(ValueError):
        build(payload, payload_bytes=raw)


def test_payload_depth_is_bounded_before_observation(payload):
    value = "PRIVATE_DEPTH_CONTENT"
    for _ in range(observation.MAX_DEPTH + 1):
        value = [value]
    payload["capsules"][0]["metadata"]["nested"] = value
    with pytest.raises(ValueError):
        build(payload)


def test_payload_node_population_is_bounded_even_when_bytes_fit(payload):
    payload["capsules"][0]["metadata"]["many_small_nodes"] = [0] * (observation.MAX_NODES + 1)
    raw = encoded(payload)
    assert len(raw) < observation.MAX_PAYLOAD_BYTES
    with pytest.raises(ValueError):
        build(payload, payload_bytes=raw)


def test_payload_size_is_bounded_before_observation(payload):
    raw = b" " * (observation.MAX_PAYLOAD_BYTES + 1)
    with pytest.raises(ValueError):
        build(payload, payload_bytes=raw)


def test_observation_size_is_bounded_before_decoding(payload):
    raw = b" " * (observation.MAX_OBSERVATION_BYTES + 1)
    with pytest.raises(ValueError):
        verify({}, payload, raw=raw)


def test_more_than_the_bounded_source_population_is_refused(payload):
    for index in range(observation.MAX_SOURCES):
        payload["manifest"][f"extra_{index:03}.py"] = {
            "sha256": "0" * 64, "source_cid": "literal-extra-source:" + str(index)}
    with pytest.raises(ValueError):
        build(payload)
