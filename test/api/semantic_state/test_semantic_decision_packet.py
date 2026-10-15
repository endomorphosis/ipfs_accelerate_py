"""Explicit capsule selection preserves required evidence and parent custody.

Counters and resolvers here are authored fixtures. These contracts establish
neither source-equivalence proofs nor provider usage or session token savings.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextTier
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import (
    SemanticProjectionBindings,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import (
    SemanticDecisionPolicy,
    build_semantic_decision_packet,
    expand_semantic_decision_packet,
    semantic_decision_packet_references,
    verify_semantic_decision_packet,
)


REQUIRED = "b" + "a" * 58
OPTIONAL = "b" + "c" * 58
SELECTED = "b" + "d" * 58


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def bind(text):
    parent = json.loads(text)
    return SemanticProjectionBindings(
        artifact=".runtime/semantic/worker-context.json",
        expected_sha256=hashlib.sha256(text.encode()).hexdigest(),
        task_id=parent["task_id"],
        semantic_root_cid=parent["semantic_root_cid"],
        scope_cid=parent["scope_cid"],
        repository_id="authored-repository",
        tree_id="authored-current-tree",
    )


@pytest.fixture
def parent():
    source = "# Keep café and trailing spaces.  \ndef target(value):\n    return value + 1\n"

    def capsule(identity, name):
        return {
            "capsule_cid": identity,
            "stable_symbol_id": "symbol-" + name,
            "confidence": "conservative",
            "metadata": {
                "future_requirement": {"literal": "Keep every caveat", "required": True},
                "name": name,
            },
            "caveats": ["Raw source required before editing"],
        }

    return {
        "schema": "supervisor-semantic-worker-context@1",
        "task_id": "DECISION-1",
        "objective": "Preserve every literal instruction.  \n",
        "scope_cid": "scope-current",
        "semantic_root_cid": "root-current",
        "manifest": {
            "target.py": {
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "source_cid": "source-current",
            }
        },
        "required_raw_paths": ["target.py"],
        "preparation_bounds": {"max_symbols": 256, "context_input_tokens": 8192},
        # Selection preserves producer order, independently of policy ordering.
        "capsules": [
            capsule(SELECTED, "selected"),
            capsule(REQUIRED, "required"),
            capsule(OPTIONAL, "deferred"),
        ],
        "admissions": [
            {
                "admission": "raw_source_required",
                "ref": {
                    "capsule_cid": identity,
                    "stable_symbol_id": "symbol-" + name,
                    "raw_source_required": True,
                },
                "caveats": ["Do not replace current source with inferred behavior"],
            }
            for identity, name in (
                (SELECTED, "selected"), (REQUIRED, "required"), (OPTIONAL, "deferred")
            )
        ],
        "raw_sources": {"target.py": source},
        "pack": {
            "schema": "ipfs-accelerate.context-pack-result@1",
            "interface": "ContextPack@1",
            "pack_cid": "pack-current",
            "references": [{"coverage_ids": ["ACCEPT-1"], "required": True}],
            "pack": {
                "objective": "Literal pack objective",
                "token_totals": {"target_source": 12},
                "assumptions": ["Scoped, not full repository"],
                "exclusions": ["No inferred equivalence"],
                "obligation_cids": ["obligation-current"],
                "counterexample_cids": ["counterexample-current"],
                "risk": "conservative",
                "future_requirement": {"literal": "Keep security condition", "required": True},
            },
            "token_estimate": {"totals": {"target_source": 12}, "estimator_version": "authored@1", "total": 12},
            "coverage_satisfied": True,
            "budget_exceeded": False,
            "decisions": ["raw source required"],
        },
        "doctor_snapshot_id": "doctor-current",
        "doctor_manifest_cid": "doctor-manifest-current",
        "reconstruction": {
            "schema": "supervisor-scoped-semantic-reconstruction@1",
            "full_repository": False,
            "semantic_acceptance_authority": False,
            "completion_authority": False,
        },
        "refresh_lineage": {"cause": "source_changed_before_dispatch", "source_delta": {}},
        "completion_authority": False,
    }


def policy():
    return SemanticDecisionPolicy(
        required_capsule_ids=(REQUIRED,),
        optional_capsule_ids=(OPTIONAL, SELECTED),
        selected_optional_capsule_ids=(SELECTED,),
    )


def case(parent, selected_policy=None):
    text = encoded(parent)
    bindings = bind(text)
    view = build_semantic_decision_packet(
        original_text=text, bindings=bindings, policy=selected_policy or policy()
    )
    return text, bindings, view


def test_selection_preserves_native_order_and_every_other_decision_field(parent):
    before = copy.deepcopy(parent)
    text, bindings, view = case(parent)
    expected = copy.deepcopy(parent)
    expected["capsules"] = expected["capsules"][:2]
    del expected["pack"]["token_estimate"]
    del expected["pack"]["pack"]["token_totals"]
    assert view["schema"] == "supervisor-semantic-decision-packet@1"
    assert view["projection_revision"] == "explicit-capsule-selection@1"
    assert view["required_context"] == expected
    assert view["origin"] == bindings.to_dict()
    assert parent == before and encoded(parent) == text
    assert view["execution_authority"] is view["completion_authority"] is False
    assert view["semantic_equivalence_claimed"] is False
    assert view["selection_policy"] == {
        "required_capsule_ids": [REQUIRED],
        "optional_capsule_ids": [OPTIONAL, SELECTED],
        "selected_optional_capsule_ids": [SELECTED],
    }
    assert view["selection_summary"] == {
        "parent_capsule_count": 3,
        "retained_capsule_count": 2,
        "deferred_capsule_count": 1,
        "parent_worker_projection_preserved": True,
    }
    assert verify_semantic_decision_packet(
        view=view, original_text=text, bindings=bindings, policy=policy()
    ) == view
    assert len(view["deferred_references"]) == 1
    handle = view["deferred_references"][0]
    assert handle["tier"] == "expansion" and handle["metadata"]["required"] is False
    assert encoded(parent["capsules"][2]) not in encoded(view)


@pytest.mark.parametrize(
    "arguments",
    [
        {"required_capsule_ids": [REQUIRED]},
        {"required_capsule_ids": (REQUIRED, REQUIRED)},
        {"optional_capsule_ids": (OPTIONAL, OPTIONAL)},
        {"selected_optional_capsule_ids": (SELECTED, SELECTED)},
        {"required_capsule_ids": (True,)},
        {"required_capsule_ids": ("",)},
        {"required_capsule_ids": (REQUIRED, OPTIONAL)},
        {"selected_optional_capsule_ids": (REQUIRED,)},
    ],
)
def test_policy_requires_explicit_disjoint_tuples_and_optional_selection(parent, arguments):
    values = {
        "required_capsule_ids": (REQUIRED,),
        "optional_capsule_ids": (OPTIONAL, SELECTED),
        "selected_optional_capsule_ids": (SELECTED,),
    }
    values.update(arguments)
    with pytest.raises(ValueError):
        case(parent, SemanticDecisionPolicy(**values))


@pytest.mark.parametrize("mutation", ["unclassified", "unknown", "duplicate", "missing", "wrong_type"])
def test_native_capsules_and_policy_form_one_complete_unique_partition(parent, mutation):
    if mutation == "unclassified":
        parent["capsules"].append({"capsule_cid": "b" + "e" * 58})
    elif mutation == "unknown":
        parent["capsules"].pop()
    elif mutation == "duplicate":
        parent["capsules"].append(copy.deepcopy(parent["capsules"][0]))
    elif mutation == "missing":
        del parent["capsules"][0]["capsule_cid"]
    else:
        parent["capsules"][0]["capsule_cid"] = True
    with pytest.raises(ValueError):
        case(parent)


def test_all_capsules_selected_without_telemetry_needs_no_archive(parent):
    del parent["pack"]["token_estimate"]
    del parent["pack"]["pack"]["token_totals"]
    all_required = SemanticDecisionPolicy(
        required_capsule_ids=(OPTIONAL, REQUIRED, SELECTED),
        optional_capsule_ids=(),
        selected_optional_capsule_ids=(),
    )
    _, _, view = case(parent, all_required)
    assert view["required_context"] == parent
    assert view["deferred_references"] == []


def test_native_worker_projection_remains_literal_parent_metadata(parent):
    parent["worker_projection"] = {
        "full_capsule_count": 7,
        "selected_capsule_count": 3,
        "omitted_capsule_count": 4,
        "selected_symbols": [capsule["stable_symbol_id"] for capsule in parent["capsules"]],
        "raw_required_caveat": "Native preparation inventory; read raw source before editing",
        "semantic_equivalence_claimed": False,
        "completion_authority": False,
    }
    _, _, view = case(parent)
    assert view["required_context"]["worker_projection"] == parent["worker_projection"]
    assert len(view["required_context"]["capsules"]) == 2
    assert view["selection_summary"]["parent_capsule_count"] == 3


@pytest.mark.parametrize("mutation", ["capsule_id", "symbol_id", "duplicate", "missing"])
def test_capsule_admissions_cannot_be_reassigned_or_removed(parent, mutation):
    if mutation == "capsule_id":
        parent["admissions"][0]["ref"]["capsule_cid"] = OPTIONAL
    elif mutation == "symbol_id":
        parent["admissions"][0]["ref"]["stable_symbol_id"] = "foreign-symbol"
    elif mutation == "duplicate":
        parent["admissions"].append(copy.deepcopy(parent["admissions"][0]))
    else:
        parent["admissions"].pop()
    with pytest.raises(ValueError):
        case(parent)


def test_self_consistent_packet_cannot_replace_the_trusted_caller_policy(parent):
    text = encoded(parent)
    bindings = bind(text)
    replacement = SemanticDecisionPolicy(
        required_capsule_ids=(SELECTED,),
        optional_capsule_ids=(OPTIONAL, REQUIRED),
        selected_optional_capsule_ids=(),
    )
    altered = build_semantic_decision_packet(
        original_text=text, bindings=bindings, policy=replacement
    )
    assert verify_semantic_decision_packet(
        view=altered, original_text=text, bindings=bindings, policy=replacement
    ) == altered
    with pytest.raises(ValueError):
        verify_semantic_decision_packet(
            view=altered, original_text=text, bindings=bindings, policy=policy()
        )
    with pytest.raises(ValueError):
        semantic_decision_packet_references(
            view=altered, original_text=text, bindings=bindings, policy=policy()
        )
    with pytest.raises(ValueError):
        expand_semantic_decision_packet(
            view=altered, reference_id=altered["deferred_references"][0]["reference_id"],
            bindings=bindings, policy=policy(), resolve_original=lambda **_request: text,
            count_tokens=lambda _wire: 1,
        )


@pytest.mark.parametrize(
    "mutation", ["source", "admission", "obligation", "caveat", "capsule", "handle", "authority", "extra"]
)
def test_changed_required_evidence_and_repaired_packet_fields_refuse(parent, mutation):
    text, bindings, view = case(parent)
    core = view["required_context"]
    if mutation == "source":
        core["raw_sources"]["target.py"] += "# forged\n"
    elif mutation == "admission":
        core["admissions"][0]["ref"]["raw_source_required"] = False
    elif mutation == "obligation":
        core["pack"]["pack"]["obligation_cids"].clear()
    elif mutation == "caveat":
        core["capsules"][0]["caveats"].clear()
    elif mutation == "capsule":
        core["capsules"].pop()
    elif mutation == "handle":
        view["deferred_references"][0]["metadata"]["required"] = True
    elif mutation == "authority":
        view["completion_authority"] = True
    else:
        view["future_authority"] = True
    with pytest.raises(ValueError):
        verify_semantic_decision_packet(
            view=view, original_text=text, bindings=bindings, policy=policy()
        )


def test_required_chunks_retain_coverage_and_only_embed_deferred_handles(parent):
    text, bindings, view = case(parent)
    refs = semantic_decision_packet_references(
        view=view, original_text=text, bindings=bindings, policy=policy(),
        coverage_ids=("ACCEPT-1", "SOURCE-1"), chunk_bytes=512,
    )
    assert len(refs) > 1
    assert all(ref.required and ref.tier is ContextTier.INVARIANT for ref in refs)
    assert all(ref.kind == "semantic-context" and ref.path == bindings.artifact for ref in refs)
    assert all(ref.byte_count <= 512 for ref in refs)
    assert all(set(ref.coverage_ids) >= {"ACCEPT-1", "SOURCE-1"} for ref in refs)
    assert json.loads("".join(ref.summary for ref in refs)) == view


def test_expansion_returns_exact_deferred_body_and_counts_complete_reference(parent):
    text, bindings, view = case(parent)
    requests, counted = [], []

    def resolve(**request):
        requests.append(request)
        return text

    def count(wire):
        counted.append(wire)
        return 31

    expanded = expand_semantic_decision_packet(
        view=view, reference_id=view["deferred_references"][0]["reference_id"],
        bindings=bindings, policy=policy(), resolve_original=resolve, count_tokens=count,
    )
    assert len(requests) == 2 and requests[0] == requests[1] == {
        "artifact": bindings.artifact,
        "expected_sha256": bindings.expected_sha256,
        "task_id": bindings.task_id,
    }
    assert counted == [encoded(expanded.to_dict())]
    assert expanded.tier is ContextTier.EVIDENCE
    assert expanded.required is False and expanded.token_count == 0
    assert json.loads(expanded.summary) == {
        "schema": "supervisor-semantic-decision-archive@1",
        "fields": [
            {"field_selector": ["pack", "token_estimate"], "value": parent["pack"]["token_estimate"]},
            {"field_selector": ["pack", "pack", "token_totals"], "value": parent["pack"]["pack"]["token_totals"]},
            {"field_selector": ["capsules", 2], "value": parent["capsules"][2]},
        ],
    }


def test_targeted_expansion_returns_only_requested_capsules_in_native_order(parent):
    deferred_policy = SemanticDecisionPolicy(
        required_capsule_ids=(REQUIRED,),
        optional_capsule_ids=(OPTIONAL, SELECTED),
        selected_optional_capsule_ids=(),
    )
    text, bindings, view = case(parent, deferred_policy)
    requests, counted = [], []

    def resolve(**request):
        requests.append(request)
        return text

    def count(wire):
        counted.append(wire)
        return 31

    reference_id = view["deferred_references"][0]["reference_id"]
    subset = expand_semantic_decision_packet(
        view=view, reference_id=reference_id, bindings=bindings, policy=deferred_policy,
        resolve_original=resolve, count_tokens=count, capsule_ids=(OPTIONAL, SELECTED),
    )
    assert len(requests) == 2 and requests[0] == requests[1]
    assert counted == [encoded(subset.to_dict())]
    assert json.loads(subset.summary) == {
        "schema": "supervisor-semantic-decision-archive@1",
        "fields": [
            {"field_selector": ["capsules", 0], "value": parent["capsules"][0]},
            {"field_selector": ["capsules", 2], "value": parent["capsules"][2]},
        ],
    }
    digest = hashlib.sha256(subset.summary.encode()).hexdigest()
    assert subset.to_dict()["metadata"]["expanded_capsule_ids"] == [SELECTED, OPTIONAL]
    assert subset.metadata["expanded_sha256"] == digest
    assert subset.metadata["complete_archive"] is False
    assert subset.referenced_content_id == "sha256:" + digest
    assert subset.reference_id != reference_id and digest in subset.reference_id
    assert subset.tier is ContextTier.EVIDENCE
    assert subset.required is False and subset.token_count == 0


def test_explicit_none_preserves_complete_archive_expansion(parent):
    text, bindings, view = case(parent)
    arguments = {
        "view": view,
        "reference_id": view["deferred_references"][0]["reference_id"],
        "bindings": bindings,
        "policy": policy(),
        "resolve_original": lambda **_request: text,
        "count_tokens": lambda _wire: 1,
    }
    default = expand_semantic_decision_packet(**arguments)
    explicit = expand_semantic_decision_packet(**arguments, capsule_ids=None)
    assert default.to_dict() == explicit.to_dict()
    assert len(json.loads(default.summary)["fields"]) == 3


@pytest.mark.parametrize(
    "capsule_ids",
    [
        (), [], "not-a-tuple", (REQUIRED,), (SELECTED,), ("unknown",),
        (OPTIONAL, OPTIONAL), (True,), ("",),
    ],
)
def test_invalid_targeted_selection_refuses_before_resolver_or_counter(parent, capsule_ids):
    _, bindings, view = case(parent)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid capsule selection performed callback work")

    with pytest.raises(ValueError):
        expand_semantic_decision_packet(
            view=view, reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings, policy=policy(), resolve_original=forbidden,
            count_tokens=forbidden, capsule_ids=capsule_ids,
        )


def test_selected_complete_wire_can_fit_when_full_archive_exceeds_callers_bound(parent):
    parent["capsules"][0]["metadata"]["unrequested_body"] = "unrequested " * 1000
    deferred_policy = SemanticDecisionPolicy(
        required_capsule_ids=(REQUIRED,),
        optional_capsule_ids=(OPTIONAL, SELECTED),
        selected_optional_capsule_ids=(),
    )
    text, bindings, view = case(parent, deferred_policy)
    arguments = {
        "view": view,
        "reference_id": view["deferred_references"][0]["reference_id"],
        "bindings": bindings,
        "policy": deferred_policy,
        "resolve_original": lambda **_request: text,
        "count_tokens": lambda _wire: 1,
    }
    full = expand_semantic_decision_packet(**arguments, max_bytes=65_536)
    subset = expand_semantic_decision_packet(
        **arguments, capsule_ids=(OPTIONAL,), max_bytes=65_536
    )
    selected_wire_bytes = len(encoded(subset.to_dict()).encode())
    assert selected_wire_bytes < len(encoded(full.to_dict()).encode())
    assert json.loads(subset.summary)["fields"] == [
        {"field_selector": ["capsules", 2], "value": parent["capsules"][2]}
    ]
    with pytest.raises(ValueError):
        expand_semantic_decision_packet(**arguments, max_bytes=selected_wire_bytes)
    fitting = expand_semantic_decision_packet(
        **arguments, capsule_ids=(OPTIONAL,), max_bytes=selected_wire_bytes
    )
    assert fitting.to_dict() == subset.to_dict()
    with pytest.raises(ValueError):
        expand_semantic_decision_packet(
            **arguments, capsule_ids=(OPTIONAL,), max_bytes=selected_wire_bytes - 1
        )


@pytest.mark.parametrize("timing", ["initial", "closing_changed", "closing_stale"])
@pytest.mark.parametrize("capsule_ids", [None, (OPTIONAL,)], ids=("whole_archive", "targeted"))
def test_expansion_rechecks_same_current_parent_after_callback_work(parent, timing, capsule_ids):
    text, bindings, view = case(parent)
    altered = copy.deepcopy(parent)
    altered["raw_sources"]["target.py"] += "changed\n"
    calls = 0

    def resolve(**_request):
        nonlocal calls
        calls += 1
        if timing == "closing_stale" and calls > 1:
            raise ValueError("existing source loader reports stale")
        return encoded(altered) if timing == "initial" or calls > 1 else text

    with pytest.raises(ValueError):
        expand_semantic_decision_packet(
            view=view, reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings, policy=policy(), resolve_original=resolve, count_tokens=lambda _wire: 1,
            capsule_ids=capsule_ids,
        )


@pytest.mark.parametrize("result", [True, False, 0, -1, 4097, 1.0, "1", None])
def test_expansion_counter_must_return_a_bounded_actual_integer(parent, result):
    text, bindings, view = case(parent)
    with pytest.raises(ValueError):
        expand_semantic_decision_packet(
            view=view, reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings, policy=policy(), resolve_original=lambda **_request: text,
            count_tokens=lambda _wire: result,
        )


@pytest.mark.parametrize("max_view_bytes", [False, 0, 1, 2_000_001])
def test_required_context_overflow_refuses_instead_of_truncating(parent, max_view_bytes):
    text = encoded(parent)
    with pytest.raises(ValueError):
        build_semantic_decision_packet(
            original_text=text, bindings=bind(text), policy=policy(), max_view_bytes=max_view_bytes,
        )


def test_archive_must_fit_complete_expanded_descriptor_at_construction(parent):
    parent["capsules"][2]["metadata"]["large_optional_body"] = "é" * 32768
    text = encoded(parent)
    with pytest.raises(ValueError):
        build_semantic_decision_packet(original_text=text, bindings=bind(text), policy=policy())


@pytest.mark.parametrize(
    "field,value", [("max_bytes", False), ("max_bytes", 1), ("max_bytes", 65_537), ("max_tokens", 0)]
)
def test_expansion_limits_apply_to_complete_returned_reference(parent, field, value):
    text, bindings, view = case(parent)
    with pytest.raises(ValueError):
        expand_semantic_decision_packet(
            view=view, reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings, policy=policy(), resolve_original=lambda **_request: text,
            count_tokens=lambda _wire: 1, **{field: value},
        )


def test_initial_cancellation_prevents_resolver_and_counter_calls(parent):
    _, bindings, view = case(parent)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cancelled expansion performed callback work")

    with pytest.raises(ValueError, match="cancelled"):
        expand_semantic_decision_packet(
            view=view, reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings, policy=policy(), resolve_original=forbidden, count_tokens=forbidden,
            cancelled=lambda: True,
        )


def test_cancellation_during_counting_prevents_returning_expanded_evidence(parent):
    text, bindings, view = case(parent)
    cancelled = False

    def count(_wire):
        nonlocal cancelled
        cancelled = True
        return 1

    with pytest.raises(ValueError, match="cancelled"):
        expand_semantic_decision_packet(
            view=view, reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings, policy=policy(), resolve_original=lambda **_request: text,
            count_tokens=count, cancelled=lambda: cancelled,
        )


def test_unknown_handle_cannot_materialize_an_archive(parent):
    text, bindings, view = case(parent)
    counted = []
    with pytest.raises(ValueError):
        expand_semantic_decision_packet(
            view=view, reference_id="unknown", bindings=bindings, policy=policy(),
            resolve_original=lambda **_request: text,
            count_tokens=lambda wire: counted.append(wire) or 1,
        )
    assert counted == []


def test_pure_packet_operations_do_not_open_ambient_files(parent, monkeypatch):
    text, bindings, view = case(parent)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ambient file operation used")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("os.open", forbidden)
    assert build_semantic_decision_packet(original_text=text, bindings=bindings, policy=policy()) == view
    assert verify_semantic_decision_packet(
        view=view, original_text=text, bindings=bindings, policy=policy()
    ) == view
    semantic_decision_packet_references(
        view=view, original_text=text, bindings=bindings, policy=policy()
    )
    expand_semantic_decision_packet(
        view=view, reference_id=view["deferred_references"][0]["reference_id"],
        bindings=bindings, policy=policy(), resolve_original=lambda **_request: text,
        count_tokens=lambda _wire: 1,
    )
