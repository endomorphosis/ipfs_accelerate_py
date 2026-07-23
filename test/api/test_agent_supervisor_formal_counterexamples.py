from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_counterexamples import (
    COUNTEREXAMPLE_CAPSULE_SCHEMA,
    COUNTEREXAMPLE_GRAPH_SCHEMA,
    FORMAL_COUNTEREXAMPLE_SCHEMA,
    CounterexampleBindings,
    CounterexampleEdgeKind,
    CounterexampleKnowledgeGraph,
    CounterexampleKind,
    CounterexampleLimits,
    CounterexampleStore,
    CounterexampleValidationError,
    FormalCounterexample,
    RepairClass,
    build_counterexample_context_capsule,
    build_counterexample_graph,
    deduplicate_counterexamples,
    normalize_counterexample,
    normalize_dcec_contradiction,
    normalize_hypertrace,
    normalize_kernel_error,
    normalize_protocol_attack,
    normalize_runtime_mtl_violation,
    normalize_smt_model,
    normalize_tdfol_contradiction,
    normalize_tla_trace,
    normalize_unsat_core,
)
from ipfs_accelerate_py.agent_supervisor.runtime_temporal_monitor import (
    monitor_event_trace,
)


@pytest.mark.parametrize(
    ("normalizer", "raw", "expected_kind", "payload_key"),
    [
        (
            normalize_smt_model,
            {"model": {"lease_owner": "worker-a", "epoch": 4}},
            CounterexampleKind.SMT_MODEL,
            "assignments",
        ),
        (
            normalize_unsat_core,
            {"unsat_core": ["lease-exclusive", "two-live-owners"]},
            CounterexampleKind.SMT_UNSAT_CORE,
            "core",
        ),
        (
            normalize_dcec_contradiction,
            {
                "contradiction": {
                    "left": "Permitted(worker, merge)",
                    "right": "Forbidden(worker, merge)",
                },
                "premises": ["permit", "forbid"],
            },
            CounterexampleKind.DCEC_CONTRADICTION,
            "contradiction",
        ),
        (
            normalize_tdfol_contradiction,
            {
                "contradiction": {
                    "formula": "Started(t) and not DependenciesComplete(t)"
                }
            },
            CounterexampleKind.TDFOL_CONTRADICTION,
            "contradiction",
        ),
        (
            normalize_tla_trace,
            {
                "trace": [
                    {"state": 0, "owner": ""},
                    {"state": 1, "owner": "a"},
                    {"state": 2, "owner": "b"},
                ],
                "invariant": "SingleOwner",
            },
            CounterexampleKind.TLA_TRACE,
            "steps",
        ),
        (
            normalize_protocol_attack,
            {
                "attack_trace": [
                    {"event": "claim", "actor": "attacker"},
                    {"event": "accept", "actor": "authority"},
                ],
                "query": "attacker cannot publish",
            },
            CounterexampleKind.PROTOCOL_ATTACK,
            "steps",
        ),
        (
            normalize_hypertrace,
            {
                "trace_refs": ["b-left", "b-right"],
                "observed_fields": ["output.digest"],
                "differences": [
                    {
                        "field": "output.digest",
                        "left_digest": "sha256:left",
                        "right_digest": "sha256:right",
                    }
                ],
                "observation_policy_id": "policy:public-output",
            },
            CounterexampleKind.HYPERTRACE,
            "trace_refs",
        ),
        (
            normalize_kernel_error,
            {
                "failure_code": "statement_mismatch",
                "kernel_id": "lean:4.19",
                "stderr": "unbounded and untrusted",
            },
            CounterexampleKind.KERNEL_ERROR,
            "failure_code",
        ),
        (
            normalize_runtime_mtl_violation,
            {
                "events": [
                    {"type": "merge_started"},
                    {"type": "proof_missing"},
                ],
                "partition": {"task_id": "REF-288"},
            },
            CounterexampleKind.RUNTIME_MTL_VIOLATION,
            "steps",
        ),
    ],
)
def test_canonical_ir_represents_every_required_failure_family(
    normalizer, raw, expected_kind, payload_key
) -> None:
    value = normalizer(
        raw,
        violated_property="obligation:no-unsafe-transition",
        bindings={"task_id": "REF-288", "obligation_id": "obl:safety"},
    )

    assert value.kind is expected_kind
    assert payload_key in value.payload
    assert value.minimized
    assert value.counterexample_id == value.semantic_id
    assert value.to_dict()["schema"] == FORMAL_COUNTEREXAMPLE_SCHEMA
    assert value.to_dict()["contains_private_material"] is False
    assert value.to_dict()["contains_raw_prover_output"] is False
    assert FormalCounterexample.from_dict(value.to_record()) == value
    assert json.loads(value.to_json()) == value.to_dict()


def test_graph_binds_every_required_plan_and_evidence_identity() -> None:
    bindings = CounterexampleBindings(
        plan_ids=("plan:288",),
        task_ids=("task:288",),
        tree_ids=("tree:abc",),
        ast_scope_ids=("symbol:normalizer",),
        assumption_ids=("assumption:lease",),
        obligation_ids=("obligation:exclusive",),
        provider_ids=("provider:z3",),
        receipt_ids=("receipt:solver",),
        invalidated_evidence_ids=("evidence:old-proof",),
        policy_ids=("policy:formal-plan",),
    )
    value = normalize_smt_model(
        {"model": {"owners": 2}},
        violated_property="exclusive lease",
        bindings=bindings,
    )

    graph = build_counterexample_graph([value])
    by_target = {edge.target_id: edge.kind for edge in graph.edges}

    assert graph.to_dict()["schema"] == COUNTEREXAMPLE_GRAPH_SCHEMA
    assert by_target == {
        "plan:288": CounterexampleEdgeKind.COUNTEREXAMPLE_TO,
        "task:288": CounterexampleEdgeKind.AFFECTS,
        "tree:abc": CounterexampleEdgeKind.OBSERVED_ON,
        "symbol:normalizer": CounterexampleEdgeKind.SCOPED_TO,
        "assumption:lease": CounterexampleEdgeKind.USES_ASSUMPTION,
        "obligation:exclusive": CounterexampleEdgeKind.COUNTEREXAMPLE_TO,
        "provider:z3": CounterexampleEdgeKind.PRODUCED_BY,
        "receipt:solver": CounterexampleEdgeKind.RECORDED_BY,
        "evidence:old-proof": CounterexampleEdgeKind.INVALIDATES,
        "policy:formal-plan": CounterexampleEdgeKind.GOVERNED_BY,
    }
    assert {item.node_id for item in graph.neighbors(value.semantic_id)} == set(
        by_target
    )
    assert graph.counterexamples_for("task:288") == (value,)
    assert CounterexampleKnowledgeGraph.from_dict(graph.to_record()) == graph


def test_minimization_and_semantic_dedup_ignore_order_stutter_and_lineage() -> None:
    left = normalize_unsat_core(
        {"unsat_core": ["b", "a", "b"], "provider_id": "z3"},
        violated_property="consistent premises",
        bindings={"task_id": "REF-288", "receipt_id": "receipt:z3"},
    )
    right = normalize_unsat_core(
        {"unsat_core": ["a", "b"], "provider_id": "cvc5"},
        violated_property="consistent premises",
        bindings={"task_id": "REF-288", "receipt_id": "receipt:cvc5"},
    )

    assert left.payload["core"] == ["a", "b"]
    assert left.semantic_id == right.semantic_id
    deduplicated = deduplicate_counterexamples((left, right))
    assert len(deduplicated) == 1
    assert deduplicated[0].bindings.provider_ids == ("cvc5", "z3")
    assert deduplicated[0].bindings.receipt_ids == (
        "receipt:cvc5",
        "receipt:z3",
    )

    trace = normalize_tla_trace(
        {
            "trace": [
                {"state": "start", "timestamp": "first"},
                {"state": "start", "timestamp": "second"},
                *({"state": f"s{index}"} for index in range(30)),
                {"state": "violated"},
            ]
        },
        violated_property="bounded transition",
        limits=CounterexampleLimits(max_trace_steps=5),
    )
    assert len(trace.payload["steps"]) == 5
    assert trace.payload["steps"][0]["state"] == "start"
    assert trace.payload["steps"][-1]["state"] == "violated"
    assert trace.truncated


def test_private_material_raw_output_and_source_are_discarded_before_identity() -> None:
    common = {
        "model": {
            "public_state": "double-owner",
            "hidden_witness": "DO-NOT-STORE",
            "credential": "DO-NOT-STORE",
            "note": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            "unrelated_source": "complete repository source",
            "raw_output": "solver output" * 10000,
        },
        "source_excerpt": "unrelated source",
        "stdout": "unbounded transcript",
    }
    first = normalize_smt_model(
        common,
        violated_property="exclusive lease",
    )
    changed_secret = json.loads(json.dumps(common))
    changed_secret["model"]["hidden_witness"] = "A-DIFFERENT-SECRET"
    changed_secret["model"]["credential"] = "ANOTHER-CREDENTIAL"
    second = normalize_smt_model(
        changed_secret,
        violated_property="exclusive lease",
    )
    encoded = first.to_json().lower()

    assert first.semantic_id == second.semantic_id
    assert first.redaction.dropped_fields >= 5
    assert first.redaction.redacted_values >= 1
    assert "do-not-store" not in encoded
    assert "different-secret" not in second.to_json().lower()
    assert "hidden_witness" not in encoded
    assert "credential" not in encoded
    assert "unrelated_source" not in encoded
    assert "source_excerpt" not in encoded
    assert "raw_output" not in encoded
    assert "stdout" not in encoded
    assert "abcdefghijklmnopqrstuvwxyz" not in encoded


def test_capsule_filters_graph_and_enforces_byte_and_confidentiality_limits() -> None:
    values = [
        normalize_smt_model(
            {
                "model": {
                    "state": index,
                    "description": "bounded public diagnostic " * 20,
                    "api_key": f"credential-{index}",
                    "source_code": "unrelated source",
                }
            },
            violated_property=f"obligation:{index}",
            bindings={
                "task_id": "task:selected" if index < 3 else "task:unrelated",
                "plan_id": f"plan:{index}",
                "provider_id": "z3",
            },
        )
        for index in range(10)
    ]
    limits = CounterexampleLimits(
        max_capsule_bytes=3000,
        max_capsule_counterexamples=2,
        max_graph_nodes=8,
        max_graph_edges=8,
    )
    capsule = build_counterexample_context_capsule(
        build_counterexample_graph(values),
        target_ids=("task:selected",),
        limits=limits,
    )
    encoded = capsule.to_json().lower()

    assert capsule.to_dict()["schema"] == COUNTEREXAMPLE_CAPSULE_SCHEMA
    assert capsule.byte_size <= limits.max_capsule_bytes
    assert capsule.usage.encoded_bytes == capsule.byte_size
    assert capsule.usage.counterexamples <= 2
    assert capsule.usage.omitted_counterexamples == 1
    assert {edge.target_id for edge in capsule.edges} <= {"task:selected"}
    assert "credential-" not in encoded
    assert "api_key" not in encoded
    assert "source_code" not in encoded
    assert "task:unrelated" not in encoded
    assert capsule.to_dict()["contains_private_material"] is False
    assert capsule.to_dict()["contains_source"] is False
    assert type(capsule).from_dict(capsule.to_record()) == capsule


def test_persistence_normalizes_first_and_semantically_deduplicates(tmp_path) -> None:
    path = tmp_path / "counterexamples.jsonl"
    store = CounterexampleStore(path)
    raw = {
        "unsat_core": ["premise:b", "premise:a"],
        "hidden_witness": "never persist",
        "stdout": "unbounded prover output",
    }
    first, inserted = store.persist(
        raw,
        kind=CounterexampleKind.SMT_UNSAT_CORE,
        violated_property="consistent plan",
        bindings={"task_id": "REF-288"},
    )
    replay, replay_inserted = store.persist(
        {**raw, "unsat_core": ["premise:a", "premise:b"]},
        kind=CounterexampleKind.SMT_UNSAT_CORE,
        violated_property="consistent plan",
        bindings={"task_id": "REF-288"},
    )

    assert inserted
    assert not replay_inserted
    assert replay.semantic_id == first.semantic_id
    assert store.load() == (first,)
    assert store.load_graph().counterexamples == (first,)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    durable = path.read_text(encoding="utf-8").lower()
    assert "never persist" not in durable
    assert "hidden_witness" not in durable
    assert "stdout" not in durable


def test_runtime_mtl_contract_is_consumed_without_raw_monitor_transcript() -> None:
    report = monitor_event_trace(
        [
            {
                "type": "merge_started",
                "event_id": "merge",
                "task_id": "REF-288",
                "lane_id": "g12-s4",
                "repository_tree_id": "tree:abc",
                "restart_epoch": "epoch:a",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ],
        now="2026-01-01T00:00:01Z",
    )
    assert report.counterexamples
    runtime = normalize_counterexample(report.counterexamples[0])

    assert runtime.kind is CounterexampleKind.RUNTIME_MTL_VIOLATION
    assert runtime.property_class == "runtime_trace"
    assert runtime.violated_property == report.counterexamples[0].property_id
    assert runtime.bindings.task_ids == ("REF-288",)
    assert runtime.bindings.tree_ids == ("tree:abc",)
    assert runtime.bindings.policy_ids == (report.counterexamples[0].policy_id,)
    assert runtime.payload["steps"]


def test_malformed_or_private_prebuilt_contracts_fail_closed() -> None:
    valid = normalize_kernel_error(
        {"failure_code": "statement_mismatch"},
        violated_property="fixed theorem",
    )
    with pytest.raises(CounterexampleValidationError, match="private"):
        replace(valid, payload={"hidden_witness": "secret"})
    with pytest.raises(CounterexampleValidationError, match="private"):
        replace(valid, payload={"note": "Bearer abcdefghijklmnopqrstuvwxyz"})
    with pytest.raises(CounterexampleValidationError, match="minimized"):
        replace(valid, minimized=False)
    with pytest.raises(CounterexampleValidationError, match="identity"):
        FormalCounterexample.from_dict(
            {**valid.to_dict(), "counterexample_id": "forged"}
        )


def test_inference_and_repair_classes_are_typed_and_actionable() -> None:
    value = normalize_counterexample(
        {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "hypertrace-counterexample@1"
            ),
            "trace_refs": ["b-left", "b-right"],
            "observed_fields": ["route"],
            "differences": [
                {
                    "field": "route",
                    "left_digest": "sha256:left",
                    "right_digest": "sha256:right",
                }
            ],
            "property_id": "noninterference",
        }
    )

    assert value.kind is CounterexampleKind.HYPERTRACE
    assert value.property_class == "hyperproperty"
    assert RepairClass.CONSTRAIN_SCOPE in value.repair_classes
    assert RepairClass.ADD_OBLIGATION in value.repair_classes


def test_retained_portfolio_counterexample_attempt_is_selected_and_bound() -> None:
    value = normalize_counterexample(
        {
            "verdict": "disproved",
            "counterexample_attempt_id": "attempt:z3",
            "plan": {
                "plan_id": "plan:portfolio",
                "obligation": {
                    "obligation_id": "obligation:finite",
                    "property_kind": "finite_constraint",
                    "premise_ids": ["premise:one"],
                },
            },
            "attempts": [
                {
                    "attempt_id": "attempt:other",
                    "prover_id": "cvc5",
                    "effective_outcome": "unknown",
                    "conclusive": False,
                    "evidence": {},
                },
                {
                    "attempt_id": "attempt:z3",
                    "prover_id": "z3",
                    "effective_outcome": "counterexample",
                    "conclusive": True,
                    "evidence": {
                        "model": {"owners": 2, "hidden_witness": "never"},
                        "stdout": "unbounded output",
                    },
                },
            ],
        }
    )

    assert value.kind is CounterexampleKind.SMT_MODEL
    assert value.violated_property == "obligation:finite"
    assert value.bindings.plan_ids == ("plan:portfolio",)
    assert value.bindings.obligation_ids == ("obligation:finite",)
    assert value.bindings.provider_ids == ("z3",)
    assert value.bindings.receipt_ids == ("attempt:z3",)
    assert value.assumption_ids == ("premise:one",)
    assert value.payload == {"assignments": {"owners": 2}}
