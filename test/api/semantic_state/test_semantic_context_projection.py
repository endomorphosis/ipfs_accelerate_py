"""Role projection contracts; authored counters/resolvers grant no authority."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextTier
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import (
    PROJECTION_REVISION,
    SCHEMA,
    SemanticProjectionBindings,
    build_semantic_context_projection,
    expand_semantic_context_projection,
    semantic_context_projection_references,
    verify_semantic_context_projection,
)


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
    source = (
        "# Keep café, literal s0 and trailing spaces.  \ndef target(value):\n    return value + 1\n"
    )
    manifest = {
        "target.py": {
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_cid": "authored-source-cid",
        },
        "dependency.py": {"sha256": "d" * 64, "source_cid": "dependency-source-cid"},
    }
    return {
        "schema": "supervisor-semantic-worker-context@1",
        "task_id": "ROLE-1",
        "objective": "Preserve every literal requirement.  \n",
        "scope_cid": "scope-current",
        "semantic_root_cid": "root-current",
        "manifest": manifest,
        "required_raw_paths": ["target.py"],
        "preparation_bounds": {"max_symbols": 256, "context_input_tokens": 8192},
        "capsules": [
            {
                "stable_symbol_id": "target-symbol",
                "confidence": "conservative",
                "metadata": {"version_evidence": {"op": "Add", "literal": "s0"}},
                "caveats": ["not a source-equivalence proof"],
                "source_span": {"start": 2, "end": 3},
            }
        ],
        "admissions": [
            {
                "admission": "raw_source_required",
                "ref": {"raw_source_required": True},
                "caveats": ["force_raw_source"],
            }
        ],
        "raw_sources": {"target.py": source},
        "pack": {
            "schema": "ipfs-accelerate.context-pack-result@1",
            "interface": "ContextPack@1",
            "pack_cid": "authored-pack-cid",
            "references": [{"coverage_ids": ["ACCEPT-1"], "required": True}],
            "pack": {
                "objective": "Literal pack objective",
                "token_totals": {"target_source": 12},
                "assumptions": ["Scoped, not full repository"],
                "exclusions": ["optional lookup"],
                "obligation_cids": ["obligation-current"],
                "counterexample_cids": ["counterexample"],
                "interface_cids": ["interface-current"],
                "risk": "conservative",
                "route": "model",
                "escalation_recommendation": "keep caveats",
            },
            "token_estimate": {
                "totals": {"target_source": 12},
                "estimator_version": "authored@1",
                "total": 12,
            },
            "coverage_satisfied": True,
            "production_slice_cid": None,
            "budget_exceeded": False,
            "decisions": ["raw source required"],
        },
        "doctor_snapshot_id": "doctor-current",
        "doctor_manifest_cid": "doctor-manifest-current",
        "reconstruction": {
            "schema": "supervisor-scoped-semantic-reconstruction@1",
            "scope_cid": "scope-current",
            "semantic_root_cid": "root-current",
            "full_repository": False,
            "semantic_acceptance_authority": False,
            "completion_authority": False,
        },
        "refresh_lineage": {
            "cause": "source_changed_before_dispatch",
            "previous_scope_cid": "scope-old",
            "previous_semantic_root_cid": "root-old",
            "attempt_id": "ROLE-1:2",
            "source_delta": {
                "target.py": {"before": {"sha256": "a" * 64}, "after": manifest["target.py"]}
            },
            "semantic_acceptance_authority": False,
            "completion_authority": False,
        },
        "worker_projection": {
            "omitted_capsule_count": 2,
            "selected_capsule_count": 1,
            "raw_source_fetch_required": {"dependency.py": manifest["dependency.py"]},
            "raw_required_caveat": "Read original current source before editing",
            "semantic_equivalence_claimed": False,
            "completion_authority": False,
        },
        "completion_authority": False,
    }


def case(parent):
    text = encoded(parent)
    bindings = bind(text)
    view = build_semantic_context_projection(original_text=text, bindings=bindings)
    return text, bindings, view


def test_only_two_telemetry_fields_deferred_and_every_decision_field_retained(parent):
    original = copy.deepcopy(parent)
    text, bindings, view = case(parent)
    expected = copy.deepcopy(parent)
    del expected["pack"]["token_estimate"]
    del expected["pack"]["pack"]["token_totals"]
    assert view["schema"] == SCHEMA and view["projection_revision"] == PROJECTION_REVISION
    assert view["origin"] == bindings.to_dict() and view["required_context"] == expected
    assert (
        view["required_coverage_sha256"] == hashlib.sha256(encoded(expected).encode()).hexdigest()
    )
    assert parent == original and encoded(parent) == text
    assert view["required_context"]["raw_sources"] == original["raw_sources"]
    assert view["required_context"]["capsules"] == original["capsules"]
    assert view["required_context"]["refresh_lineage"] == original["refresh_lineage"]
    assert view["required_context"]["reconstruction"] == original["reconstruction"]
    assert (
        view["execution_authority"]
        is view["completion_authority"]
        is view["semantic_equivalence_claimed"]
        is False
    )
    assert (
        verify_semantic_context_projection(view=view, original_text=text, bindings=bindings) == view
    )
    handle = view["deferred_references"][0]
    assert handle["tier"] == "expansion" and handle["metadata"]["required"] is False
    assert handle["metadata"]["field_selectors"] == [
        ["pack", "token_estimate"],
        ["pack", "pack", "token_totals"],
    ]
    assert "authored@1" not in handle["summary"]


@pytest.mark.parametrize("missing", [(), ("estimate",), ("totals",), ("estimate", "totals")])
def test_absent_and_empty_telemetry_never_drops_other_material(parent, missing):
    parent["pack"]["token_estimate"] = {"totals": {}, "estimator_version": "empty@1", "total": 0}
    parent["pack"]["pack"]["token_totals"] = {}
    if "estimate" in missing:
        del parent["pack"]["token_estimate"]
    if "totals" in missing:
        del parent["pack"]["pack"]["token_totals"]
    text, bindings, view = case(parent)
    assert len(view["deferred_references"]) == (0 if len(missing) == 2 else 1)
    assert (
        verify_semantic_context_projection(view=view, original_text=text, bindings=bindings) == view
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.update(extra=True),
        lambda v: v.update(projection_revision="unknown@2"),
        lambda v: v.update(execution_authority=0),
        lambda v: v.update(completion_authority=True),
        lambda v: v["origin"].update(expected_sha256="0" * 64),
        lambda v: v["origin"].update(tree_id="foreign"),
        lambda v: v["origin"].update(repository_id="foreign"),
        lambda v: v.update(required_coverage_sha256="0" * 64),
        lambda v: v["required_context"].update(objective="summarized"),
        lambda v: v["required_context"]["required_raw_paths"].clear(),
        lambda v: v["required_context"]["raw_sources"].update(**{"target.py": "stripped source"}),
        lambda v: v["required_context"]["capsules"][0]["metadata"].clear(),
        lambda v: v["required_context"]["admissions"][0]["ref"].update(raw_source_required=False),
        lambda v: v["required_context"]["pack"]["pack"]["obligation_cids"].clear(),
        lambda v: v["required_context"]["pack"]["pack"]["assumptions"].clear(),
        lambda v: v["required_context"]["pack"]["pack"].update(risk="none"),
        lambda v: v["required_context"]["worker_projection"]["raw_source_fetch_required"].clear(),
        lambda v: v["required_context"]["reconstruction"].update(full_repository=True),
        lambda v: v["required_context"]["refresh_lineage"]["source_delta"].clear(),
        lambda v: v["deferred_references"][0].update(tier="invariant"),
        lambda v: v["deferred_references"][0]["metadata"].update(required=True),
        lambda v: v["deferred_references"][0]["metadata"].update(field_selectors=[["raw_sources"]]),
        lambda v: v["deferred_references"][0]["metadata"].update(archived_sha256="0" * 64),
        lambda v: v["deferred_references"].clear(),
    ],
)
def test_tampering_requiredness_downgrade_and_forged_archive_refused(parent, mutate):
    text, bindings, view = case(parent)
    mutate(view)
    with pytest.raises(ValueError):
        verify_semantic_context_projection(view=view, original_text=text, bindings=bindings)


def test_required_chunks_preserve_original_coverage_and_do_not_select_archive(parent):
    text, bindings, view = case(parent)
    refs = semantic_context_projection_references(
        view=view,
        original_text=text,
        bindings=bindings,
        coverage_ids=("ACCEPT-1", "SOURCE-1"),
        chunk_bytes=512,
    )
    assert len(refs) > 1 and all(ref.required and ref.tier is ContextTier.INVARIANT for ref in refs)
    assert all(ref.kind == "semantic-context" and ref.path == bindings.artifact for ref in refs)
    assert json.loads("".join(ref.summary for ref in refs)) == view
    assert all(
        set(ref.coverage_ids)
        == {
            "ACCEPT-1",
            "SOURCE-1",
            "semantic-projection-required:" + view["required_coverage_sha256"],
        }
        for ref in refs
    )
    assert not any(ref.kind == "semantic-estimator-archive" for ref in refs)


def test_expansion_exact_selectors_complete_wire_counter_and_parent_reread(parent):
    text, bindings, view = case(parent)
    requested, counted = [], []

    def resolver(**request):
        requested.append(request)
        return text

    def counter(wire):
        counted.append(wire)
        return 31

    expanded = expand_semantic_context_projection(
        view=view,
        reference_id=view["deferred_references"][0]["reference_id"],
        bindings=bindings,
        resolve_original=resolver,
        count_tokens=counter,
    )
    assert len(requested) == 2 and requested[0] == requested[1] == {
        "artifact": bindings.artifact,
        "expected_sha256": bindings.expected_sha256,
        "task_id": bindings.task_id,
    }
    assert counted == [encoded(expanded.to_dict())]
    assert (
        expanded.tier is ContextTier.EVIDENCE
        and expanded.required is False
        and expanded.token_count == 0
    )
    archive = json.loads(expanded.summary)
    assert archive["fields"] == [
        {"field_selector": ["pack", "token_estimate"], "value": parent["pack"]["token_estimate"]},
        {
            "field_selector": ["pack", "pack", "token_totals"],
            "value": parent["pack"]["pack"]["token_totals"],
        },
    ]


@pytest.mark.parametrize("timing", ["initial", "after_counter", "closing_stale"])
def test_changed_or_stale_original_never_expands(parent, timing):
    text, bindings, view = case(parent)
    altered = copy.deepcopy(parent)
    altered["raw_sources"]["target.py"] += "changed\n"
    calls = 0

    def resolver(**_request):
        nonlocal calls
        calls += 1
        if timing == "closing_stale" and calls > 1:
            raise ValueError("existing source loader reports stale")
        return encoded(altered) if timing == "initial" or calls > 1 else text

    with pytest.raises(ValueError):
        expand_semantic_context_projection(
            view=view,
            reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings,
            resolve_original=resolver,
            count_tokens=lambda _wire: 1,
        )


@pytest.mark.parametrize("result", [True, False, 0, -1, 4097, 1.0, "1", None])
def test_expansion_requires_actual_bounded_integer_counter(parent, result):
    text, bindings, view = case(parent)
    with pytest.raises(ValueError):
        expand_semantic_context_projection(
            view=view,
            reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings,
            resolve_original=lambda **_request: text,
            count_tokens=lambda _wire: result,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_original_bytes", 0),
        ("max_original_bytes", True),
        ("max_original_bytes", 2_000_001),
        ("max_view_bytes", False),
        ("max_view_bytes", 0),
        ("max_view_bytes", 1),
    ],
)
def test_build_bounds_and_required_overflow_refuse(parent, field, value):
    text = encoded(parent)
    with pytest.raises(ValueError):
        build_semantic_context_projection(original_text=text, bindings=bind(text), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_bytes", 0),
        ("max_bytes", True),
        ("max_bytes", 65_537),
        ("max_bytes", 1),
        ("max_tokens", False),
        ("max_tokens", 0),
        ("max_tokens", 128001),
    ],
)
def test_expansion_bounds_cannot_be_bypassed(parent, field, value):
    text, bindings, view = case(parent)
    with pytest.raises(ValueError):
        expand_semantic_context_projection(
            view=view,
            reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings,
            resolve_original=lambda **_request: text,
            count_tokens=lambda _wire: 1,
            **{field: value},
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact", "../foreign"),
        ("artifact", "/absolute"),
        ("artifact", "a/./b"),
        ("artifact", "."),
        ("expected_sha256", "A" * 64),
        ("task_id", ""),
        ("semantic_root_cid", True),
        ("scope_cid", ""),
    ],
)
def test_binding_identity_is_explicit_and_closed(parent, field, value):
    with pytest.raises(ValueError):
        replace(bind(encoded(parent)), **{field: value})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema="unknown"),
        lambda p: p.update(extra="unknown"),
        lambda p: p.update(completion_authority=0),
        lambda p: p["pack"].update(schema="unknown"),
        lambda p: p["pack"]["token_estimate"].update(total=True),
        lambda p: p["pack"]["token_estimate"].update(total=99),
        lambda p: p["pack"]["pack"]["token_totals"].update(target_source=False),
        lambda p: p["raw_sources"].clear(),
    ],
)
def test_unsupported_original_shapes_never_get_archive_classification(parent, mutate):
    mutate(parent)
    text = encoded(parent)
    with pytest.raises(ValueError):
        build_semantic_context_projection(original_text=text, bindings=bind(text))


def test_duplicate_json_nonfinite_and_preallocation_bounds(parent):
    text = encoded(parent)
    for wrong in [
        text.replace('"task_id":"ROLE-1"', '"task_id":"ROLE-1","task_id":"ROLE-1"'),
        text.replace('"total":12', '"total":NaN'),
    ]:
        with pytest.raises(ValueError):
            build_semantic_context_projection(original_text=wrong, bindings=bind(text))
    with pytest.raises(ValueError):
        build_semantic_context_projection(original_text="[" * 2_000_001, bindings=bind(text))
    cyclic = {}
    cyclic["cycle"] = cyclic
    with pytest.raises(ValueError):
        verify_semantic_context_projection(view=cyclic, original_text=text, bindings=bind(text))


def test_cancelled_or_unknown_handles_never_resolve(parent):
    text, bindings, view = case(parent)

    def forbidden(**_request):
        raise AssertionError("cancelled expansion called resolver")

    with pytest.raises(ValueError, match="cancelled"):
        expand_semantic_context_projection(
            view=view,
            reference_id="unknown",
            bindings=bindings,
            resolve_original=forbidden,
            count_tokens=lambda _wire: 1,
            cancelled=lambda: True,
        )
    with pytest.raises(ValueError, match="absent"):
        expand_semantic_context_projection(
            view=view,
            reference_id="unknown",
            bindings=bindings,
            resolve_original=lambda **_request: text,
            count_tokens=lambda _wire: 1,
        )


def test_pure_projection_and_expansion_have_no_ambient_file_effects(parent, monkeypatch):
    text, bindings, view = case(parent)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ambient operation used")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("os.open", forbidden)
    assert build_semantic_context_projection(original_text=text, bindings=bindings) == view
    assert (
        verify_semantic_context_projection(view=view, original_text=text, bindings=bindings) == view
    )
    semantic_context_projection_references(view=view, original_text=text, bindings=bindings)
    expand_semantic_context_projection(
        view=view,
        reference_id=view["deferred_references"][0]["reference_id"],
        bindings=bindings,
        resolve_original=lambda **_request: text,
        count_tokens=lambda _wire: 1,
    )


def test_actual_native_preparation_current_loader_and_stale_expansion(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
        load_semantic_worker_context,
        prepare_semantic_context,
    )

    (tmp_path / "target.py").write_text("# Keep café.\ndef target(value):\n    return value + 1\n")
    result = prepare_semantic_context(
        repository=tmp_path,
        paths=["target.py"],
        required_raw_paths=["target.py"],
        objective="Preserve target",
        task_id="NATIVE-1",
        output=tmp_path / ".runtime/semantic",
    )
    request = {
        "repository": tmp_path,
        "artifact": ".runtime/semantic/worker-context.json",
        "expected_sha256": result["worker_payload_sha256"],
        "task_id": "NATIVE-1",
    }
    text = load_semantic_worker_context(**request)
    bindings = bind(text)
    view = build_semantic_context_projection(original_text=text, bindings=bindings)
    original = json.loads(text)
    assert view["required_context"]["capsules"] == original["capsules"]
    assert view["required_context"]["admissions"] == original["admissions"]
    assert view["required_context"]["raw_sources"] == original["raw_sources"]
    def resolver(**kwargs):
        return load_semantic_worker_context(repository=tmp_path, **kwargs)
    expanded = expand_semantic_context_projection(
        view=view,
        reference_id=view["deferred_references"][0]["reference_id"],
        bindings=bindings,
        resolve_original=resolver,
        count_tokens=lambda _wire: 10,
    )
    assert len(json.loads(expanded.summary)["fields"]) == 2
    (tmp_path / "target.py").write_text("def target(value): return value - 1\n")
    with pytest.raises(ValueError, match="stale"):
        expand_semantic_context_projection(
            view=view,
            reference_id=view["deferred_references"][0]["reference_id"],
            bindings=bindings,
            resolve_original=resolver,
            count_tokens=lambda _wire: 10,
        )
