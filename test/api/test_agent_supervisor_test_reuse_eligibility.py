"""PTR-022 contract tests for conservative reuse eligibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    AnalysisASTIndex,
    build_analysis_ast_index,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_execution_identity import (
    mint_content_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_reuse_eligibility import (
    DEFAULT_ROLLOUT_SCOPE,
    ROLLOUT_SCOPE_REPOSITORY_FOREST,
    TEST_REUSE_ELIGIBILITY_DECISION_INTERFACE,
    TEST_REUSE_ELIGIBILITY_DECISION_SCHEMA,
    TEST_REUSE_ELIGIBILITY_EVALUATOR_INTERFACE,
    DirtyStateEvidence,
    EligibilityDenyReason,
    TestReuseEligibilityError,
    TestReuseEligibilityEvaluator,
    TestReuseEligibilityPolicy,
    evaluate_reuse_eligibility,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_runtime_dependency_trace import (
    RuntimeTestDependencyTracer,
)
from ipfs_accelerate_py.agent_supervisor.analysis.test_static_dependency_trace import (
    STATIC_TEST_DEPENDENCY_TRACE_INTERFACE,
    STATIC_TEST_DEPENDENCY_TRACE_SCHEMA,
    StaticTestDependencyTrace,
    StaticTestDependencyTracer,
    UnknownDependencyFrontier,
    trace_static_dependencies,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    build_python_ast_blob_record,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    EligibilityClass,
    ReuseAction,
    ReuseReasonCode,
)
from multiformats import CID


def _forest_cid(label: str = "forest-v1") -> str:
    return mint_content_identity(
        {
            "schema": "test/repository-forest-fixture@1",
            "label": label,
        }
    ).cid


def _adapter_cid(label: str) -> str:
    return mint_content_identity({"schema": "test/adapter@1", "label": label}).cid


def _index(
    root: Path,
    sources: Mapping[str, str],
    *,
    order: Iterable[str] | None = None,
) -> AnalysisASTIndex:
    records = []
    for relative in order if order is not None else sources:
        source = sources[relative]
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        records.append(
            (
                relative,
                build_python_ast_blob_record(
                    source,
                    blob_identity=f"blob:{relative}:{hashlib.sha256(source.encode()).hexdigest()}",
                ),
            )
        )
    return build_analysis_ast_index(records)


def _static_pure(tmp_path: Path):
    sources = {
        "tests/test_pure.py": "def test_pure(answer):\n    assert answer == 42\n",
        "tests/conftest.py": "@pytest.fixture\ndef answer():\n    return 42\n",
    }
    index = _index(tmp_path, sources)
    return trace_static_dependencies(
        index,
        "tests/test_pure.py",
        repository_root=tmp_path,
        test_symbol="test_pure",
    )


def _static_with_effects(tmp_path: Path | None = None):
    """Synthetic complete-except-effect trace for adapter-closure tests.

    Real static tracing of ``subprocess`` / stdlib imports also emits
    ``missing_file`` frontiers for non-repository modules.  Eligibility
    still fail-closes those as incomplete analysis; these fixtures isolate
    the uncontrolled-effect + adapter path.
    """

    del tmp_path
    frontier = UnknownDependencyFrontier(
        kind="uncontrolled_effect",
        source_path="tests/test_effect.py",
        source_symbol="test_effect",
        target="subprocess",
        line_start=3,
        line_end=3,
    )
    payload = {
        "schema": STATIC_TEST_DEPENDENCY_TRACE_SCHEMA,
        "interface": STATIC_TEST_DEPENDENCY_TRACE_INTERFACE,
        "root": {"path": "tests/test_effect.py", "symbol": "test_effect"},
        "analysis_ast_index_id": "fixture-index",
        "limits": {"interface": "StaticTraceLimits@1"},
        "analyzer": {"interface": "StaticTraceAnalyzer@1"},
        "analyzer_cid": _adapter_cid("static-analyzer"),
        "dependencies": {
            "nodes": [
                {
                    "path": "tests/test_effect.py",
                    "kind": "source",
                    "content_sha256": "0" * 64,
                }
            ],
            "edges": [
                {
                    "kind": "effect",
                    "source_path": "tests/test_effect.py",
                    "source_symbol": "test_effect",
                    "target_path": "",
                    "target_symbol": "subprocess",
                    "line_start": 3,
                    "line_end": 3,
                }
            ],
        },
        "unknown_frontier": [frontier.to_dict()],
        "health": {
            "complete": False,
            "source_hashes_verified": True,
            "parser_healthy": True,
            "analysis_bounds_reached": [],
            "analyzed_file_count": 1,
            "dependency_edge_count": 1,
            "unknown_frontier_count": 1,
        },
    }
    expected = canonical_json_bytes(payload)
    identity = mint_content_identity(payload)
    assert identity.canonical_bytes == expected
    return StaticTestDependencyTrace(
        content_identity=identity,
        retained_canonical_bytes=expected,
        unknown_frontier=(frontier,),
        analyzed_file_count=1,
        dependency_edge_count=1,
    )


def _static_incomplete(tmp_path: Path):
    sources = {
        "tests/test_dyn.py": (
            "from importlib import import_module\n\n"
            "def test_dyn(name):\n"
            "    return import_module(name)\n"
        ),
    }
    index = _index(tmp_path, sources)
    return trace_static_dependencies(
        index,
        "tests/test_dyn.py",
        repository_root=tmp_path,
        test_symbol="test_dyn",
    )


def _runtime_complete(tmp_path: Path, *, profile: str = "pure"):
    tracer = RuntimeTestDependencyTracer(
        allowed_roots={"repo": tmp_path},
        capture_code_objects=False,
        eligibility_profile=profile,
    )
    with tracer:
        pass
    assert tracer.result is not None
    return tracer.result


def _runtime_incomplete(tmp_path: Path):
    tracer = RuntimeTestDependencyTracer(
        allowed_roots={"repo": tmp_path},
        capture_code_objects=False,
    )
    tracer.start()
    tracer.record_subprocess("curl", ["curl", "https://example.invalid"])
    return tracer.stop()


def _assert_profile_cid(trace_cid: str, canonical_bytes: bytes) -> None:
    parsed = CID.decode(trace_cid)
    assert parsed.version == 1
    assert parsed.base.name == "base32"
    assert parsed.codec.name == "dag-json"
    assert parsed.hashfun.name == "sha2-256"
    assert bytes(parsed.raw_digest) == hashlib.sha256(canonical_bytes).digest()


def test_pure_item_is_reusable_and_binds_repository_forest(tmp_path: Path) -> None:
    static = _static_pure(tmp_path)
    runtime = _runtime_complete(tmp_path)
    forest = _forest_cid()

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=forest,
    )

    assert decision.interface == TEST_REUSE_ELIGIBILITY_DECISION_INTERFACE
    assert decision.schema == TEST_REUSE_ELIGIBILITY_DECISION_SCHEMA
    assert decision.reusable is True
    assert decision.eligibility_class is EligibilityClass.PURE
    assert decision.action is ReuseAction.RUN
    assert decision.is_run is True
    assert decision.is_skip is False
    assert decision.to_dict()["authorizes_skip"] is False
    assert decision.repository_forest_cid == forest
    assert decision.rollout_scope == ROLLOUT_SCOPE_REPOSITORY_FOREST
    assert decision.to_dict()["binds_repository_forest"] is True
    assert decision.static_trace_root_cid == static.cid
    assert decision.runtime_trace_root_cid == runtime.cid
    assert decision.verify() is decision
    assert (
        json.dumps(
            decision.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        == decision.canonical_bytes
    )
    _assert_profile_cid(decision.cid, decision.canonical_bytes)


def test_v1_default_scope_is_repository_forest_and_missing_forest_runs(
    tmp_path: Path,
) -> None:
    policy = TestReuseEligibilityPolicy()
    assert policy.rollout_scope == DEFAULT_ROLLOUT_SCOPE == ROLLOUT_SCOPE_REPOSITORY_FOREST

    static = _static_pure(tmp_path)
    runtime = _runtime_complete(tmp_path)
    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        # intentionally omit forest
    )

    assert decision.reusable is False
    assert decision.eligibility_class is EligibilityClass.NON_REUSABLE
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.MISSING_REPOSITORY_FOREST.value in decision.reason_codes


def test_incomplete_static_analysis_always_runs(tmp_path: Path) -> None:
    static = _static_incomplete(tmp_path)
    runtime = _runtime_complete(tmp_path)
    assert not static.complete

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid(),
    )

    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.INCOMPLETE_STATIC_ANALYSIS.value in decision.reason_codes
    assert decision.as_reuse_reason() is ReuseReasonCode.INCOMPLETE_TRACE


def test_incomplete_runtime_analysis_always_runs(tmp_path: Path) -> None:
    static = _static_pure(tmp_path)
    runtime = _runtime_incomplete(tmp_path)
    assert not runtime.complete

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid(),
        effect_adapters=("subprocess",),
    )

    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.INCOMPLETE_RUNTIME_ANALYSIS.value in decision.reason_codes


def test_uncontrolled_effects_without_adapters_always_run(tmp_path: Path) -> None:
    static = _static_with_effects()
    runtime = _runtime_complete(tmp_path, profile="repository_forest_bound")
    assert any(item.kind == "uncontrolled_effect" for item in static.unknown_frontier)

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid(),
        effect_adapters=(),
    )

    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.UNCONTROLLED_EFFECT.value in decision.reason_codes
    assert EligibilityDenyReason.MISSING_EFFECT_ADAPTER.value in decision.reason_codes


def test_effect_adapters_close_uncontrolled_effect_frontiers(tmp_path: Path) -> None:
    static = _static_with_effects()
    runtime = _runtime_complete(tmp_path, profile="repository_forest_bound")
    forest = _forest_cid("with-adapters")

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=forest,
        effect_adapters=("subprocess",),
    )

    assert decision.reusable is True
    assert decision.action is ReuseAction.RUN
    assert decision.eligibility_class in {
        EligibilityClass.REPOSITORY_FOREST_BOUND,
        EligibilityClass.SNAPSHOT_BOUND,
    }
    assert decision.repository_forest_cid == forest
    assert EligibilityDenyReason.UNCONTROLLED_EFFECT.value not in decision.reason_codes


def test_missing_snapshot_adapter_for_snapshot_profile_runs(tmp_path: Path) -> None:
    static = _static_with_effects()
    runtime = _runtime_complete(tmp_path, profile="snapshot_bound")

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid(),
        effect_adapters=("subprocess",),
        snapshot_adapters={},
    )

    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.MISSING_SNAPSHOT_ADAPTER.value in decision.reason_codes


def test_snapshot_bound_with_adapters_and_snapshot_identities(tmp_path: Path) -> None:
    static = _static_with_effects()
    adapter = _adapter_cid("tool")
    snapshot = _adapter_cid("snapshot")
    tracer = RuntimeTestDependencyTracer(
        allowed_roots={"repo": tmp_path},
        capture_code_objects=False,
        eligibility_profile="snapshot_bound",
        subprocess_allowlist={"true": adapter},
    )
    tracer.start()
    tracer.record_service("postgres", adapter_identity=adapter, snapshot_identity=snapshot)
    runtime = tracer.stop()
    assert runtime.complete

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid("snapshot"),
        effect_adapters=("subprocess", "service"),
        snapshot_adapters={"subprocess": snapshot, "postgres": snapshot},
    )

    assert decision.reusable is True
    assert decision.eligibility_class is EligibilityClass.SNAPSHOT_BOUND
    assert decision.repository_forest_cid
    assert decision.to_dict()["binds_repository_forest"] is True


def test_unsupported_parameters_always_run(tmp_path: Path) -> None:
    static = _static_pure(tmp_path)
    runtime = _runtime_complete(tmp_path)

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid(),
        parameters_supported=False,
        parameter_non_reusable_reason="unsupported_pytest_parameter_float",
    )

    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.UNSUPPORTED_PARAMETERS.value in decision.reason_codes
    assert (
        decision.to_dict()["diagnostics"]["parameter_non_reusable_reason"]
        == "unsupported_pytest_parameter_float"
    )


def test_unaccounted_dirty_state_always_runs(tmp_path: Path) -> None:
    static = _static_pure(tmp_path)
    runtime = _runtime_complete(tmp_path)

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid(),
        dirty_state=DirtyStateEvidence(
            dirty=True,
            dirty_accounted=False,
            unaccounted_paths=("scratch/generated.bin",),
            reason_codes=("unaccounted_generated",),
        ),
    )

    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.UNACCOUNTED_DIRTY_STATE.value in decision.reason_codes

    # Dirty but fully accounted is permitted when overlay identity is bound.
    accounted = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid(),
        dirty_state={
            "dirty": True,
            "dirty_accounted": True,
            "dirty_overlay_cid": _adapter_cid("dirty-overlay"),
            "unaccounted_paths": (),
        },
    )
    assert accounted.reusable is True


def test_heuristic_similarity_never_authorizes_reuse(tmp_path: Path) -> None:
    static = _static_pure(tmp_path)
    runtime = _runtime_complete(tmp_path)
    forest = _forest_cid()

    # Even with "perfect" similarity on an otherwise pure item, heuristics deny.
    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=forest,
        similarity_score=1.0,
        embedding_score=0.99,
        model_verdict="pass",
        runtime_overlap=1.0,
        unchanged_line_heuristic=True,
    )

    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert (
        EligibilityDenyReason.HEURISTIC_SIMILARITY_REJECTED.value in decision.reason_codes
    )
    # Incomplete pure-looking case with high similarity still cannot reuse.
    incomplete = evaluate_reuse_eligibility(
        static_trace=_static_incomplete(tmp_path),
        runtime_trace=runtime,
        repository_forest_cid=forest,
        similarity_score=0.999,
    )
    assert incomplete.reusable is False
    assert incomplete.action is ReuseAction.RUN


def test_missing_traces_always_run() -> None:
    decision = evaluate_reuse_eligibility(repository_forest_cid=_forest_cid())
    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.MISSING_STATIC_TRACE.value in decision.reason_codes
    assert EligibilityDenyReason.MISSING_RUNTIME_TRACE.value in decision.reason_codes


def test_decision_is_deterministic_and_order_insensitive(tmp_path: Path) -> None:
    static = _static_pure(tmp_path)
    runtime = _runtime_complete(tmp_path)
    forest = _forest_cid("deterministic")

    first = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=forest,
        effect_adapters=("environment", "filesystem"),
        diagnostics={"b": 2, "a": 1},
    )
    second = TestReuseEligibilityEvaluator().evaluate(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=forest,
        effect_adapters=("filesystem", "environment"),
        diagnostics={"a": 1, "b": 2},
    )

    assert first.cid == second.cid
    assert first.canonical_bytes == second.canonical_bytes


def test_evaluator_interface_and_policy_validation() -> None:
    evaluator = TestReuseEligibilityEvaluator()
    assert evaluator.interface == TEST_REUSE_ELIGIBILITY_EVALUATOR_INTERFACE
    with pytest.raises(TestReuseEligibilityError, match="repository_forest"):
        TestReuseEligibilityPolicy(rollout_scope="dependency_root")
    with pytest.raises(TestReuseEligibilityError, match="boolean"):
        TestReuseEligibilityPolicy(allow_pure="yes")  # type: ignore[arg-type]


def test_forest_object_binding(tmp_path: Path) -> None:
    static = _static_pure(tmp_path)
    runtime = _runtime_complete(tmp_path)
    forest_id = _forest_cid("object")

    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest={"forest_id": forest_id, "schema": "fixture"},
    )
    assert decision.reusable is True
    assert decision.repository_forest_cid == forest_id


def test_malformed_evidence_fails_closed_to_run(tmp_path: Path) -> None:
    decision = TestReuseEligibilityEvaluator().evaluate(
        static_trace="not-a-trace",  # type: ignore[arg-type]
        runtime_trace=_runtime_complete(tmp_path),
        repository_forest_cid=_forest_cid(),
    )
    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.MALFORMED_EVIDENCE.value in decision.reason_codes


def test_repository_forest_bound_class_for_adapted_non_pure_without_snapshot(
    tmp_path: Path,
) -> None:
    """Adapted effects without snapshot profile fall to forest-bound (v1 default)."""

    static = _static_with_effects()
    runtime = _runtime_complete(tmp_path, profile="repository_forest_bound")
    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid("forest-class"),
        effect_adapters=("subprocess",),
    )
    assert decision.reusable is True
    assert decision.eligibility_class is EligibilityClass.REPOSITORY_FOREST_BOUND


def test_real_static_effect_plus_stdlib_import_is_incomplete_analysis(
    tmp_path: Path,
) -> None:
    """Live AST traces of stdlib subprocess keep missing_file + effect frontiers."""

    sources = {
        "tests/test_effect.py": (
            "import subprocess\n\n"
            "def test_effect():\n"
            "    subprocess.run(['true'])\n"
        ),
    }
    index = _index(tmp_path, sources)
    static = trace_static_dependencies(
        index,
        "tests/test_effect.py",
        repository_root=tmp_path,
        test_symbol="test_effect",
    )
    runtime = _runtime_complete(tmp_path)
    decision = evaluate_reuse_eligibility(
        static_trace=static,
        runtime_trace=runtime,
        repository_forest_cid=_forest_cid(),
        effect_adapters=("subprocess",),
    )
    assert decision.reusable is False
    assert decision.action is ReuseAction.RUN
    assert EligibilityDenyReason.INCOMPLETE_STATIC_ANALYSIS.value in decision.reason_codes


def test_eligibility_never_emits_skip_action(tmp_path: Path) -> None:
    static = _static_pure(tmp_path)
    runtime = _runtime_complete(tmp_path)
    for kwargs in (
        {"static_trace": static, "runtime_trace": runtime, "repository_forest_cid": _forest_cid()},
        {"repository_forest_cid": _forest_cid()},
        {
            "static_trace": _static_incomplete(tmp_path),
            "runtime_trace": runtime,
            "repository_forest_cid": _forest_cid(),
            "similarity_score": 1.0,
        },
    ):
        decision = evaluate_reuse_eligibility(**kwargs)
        assert decision.action is ReuseAction.RUN
        assert decision.to_dict()["action"] == "RUN"
        assert decision.to_dict()["authorizes_skip"] is False


def test_static_tracer_class_path_matches_module_helper(tmp_path: Path) -> None:
    sources = {"test_one.py": "def test_one():\n    assert True\n"}
    index = _index(tmp_path, sources)
    via_helper = trace_static_dependencies(
        index, "test_one.py", repository_root=tmp_path, test_symbol="test_one"
    )
    via_class = StaticTestDependencyTracer(index, tmp_path).trace(
        "test_one.py", test_symbol="test_one"
    )
    runtime = _runtime_complete(tmp_path)
    forest = _forest_cid("parity")

    first = evaluate_reuse_eligibility(
        static_trace=via_helper,
        runtime_trace=runtime,
        repository_forest_cid=forest,
    )
    second = evaluate_reuse_eligibility(
        static_trace=via_class,
        runtime_trace=runtime,
        repository_forest_cid=forest,
    )
    assert first.reusable and second.reusable
    assert first.eligibility_class == second.eligibility_class
