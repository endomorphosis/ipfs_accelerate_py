"""Tests for runtime context revalidation against retained candidates (PTR-136).

Acceptance covered:

* Lookup starts from a stable locator only.
* Every candidate dependency named by the retained trace is freshly resolved
  and content-addressed.
* Current source, AST, fixtures, hooks, parameters, locks, distributions,
  environment, capabilities, repository forest, policy and external snapshots
  must match.
* Incomplete, unresolvable, changed or uncontrolled facts return RUN.
* A verified unchanged context may proceed to certificate verification without
  executing fixtures or the test body.
* A normal miss executes setup/call/teardown exactly once before capturing and
  publishing its observed runtime trace.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_candidate_context_store import (
    TestCandidateContextStore,
)
from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
    CandidateExecutionContext,
    CurrentExecutionContext,
    RuntimeReuseAction,
)
from ipfs_accelerate_py.testing.proof_reuse.runtime_revalidation import (
    CANDIDATE_COMPARISON_INTERFACE,
    POST_PASS_RUNTIME_TRACE_CAPTURE_INTERFACE,
    RUNTIME_CONTEXT_REVALIDATOR_INTERFACE,
    RUNTIME_DEPENDENCY_TRACE_INTERFACE,
    CandidateComparison,
    FilesystemDependencyResolver,
    LifecyclePhase,
    PostPassRuntimeTraceCapture,
    RevalidationAction,
    RevalidationReason,
    RuntimeContextRevalidator,
    StaticCurrentContextProvider,
    build_runtime_context_revalidator,
    compare_candidate_to_current,
    resolve_retained_runtime_frontier,
)


NOW_S = 20.0
NOW_MS = 20_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/test/revalidation-label@1",
            "label": label,
        }
    )


def _component_bytes(label: str) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "ipfs_accelerate_py/test/candidate-component@1",
            "label": label,
            "version": 1,
        }
    )


def _component_cid(label: str) -> str:
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/test/candidate-component@1",
            "label": label,
            "version": 1,
        }
    )


def _complete_runtime_trace(
    *,
    files: list[dict[str, Any]] | None = None,
    modules: list[dict[str, Any]] | None = None,
    environment: list[dict[str, Any]] | None = None,
    complete: bool = True,
    extra_kinds: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    dependencies: dict[str, list[dict[str, Any]]] = {
        "modules": list(modules or []),
        "code_objects": [],
        "files": list(files or []),
        "environment": list(environment or []),
        "subprocesses": [],
        "services": [],
        "policies": [],
        "capabilities": [],
    }
    if extra_kinds:
        for key, value in extra_kinds.items():
            dependencies[str(key)] = list(value)
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/runtime-test-dependency-trace@1",
        "interface": "RuntimeTestDependencyTrace@1",
        "eligibility_profile": "pure",
        "completeness": {
            "status": "complete" if complete else "incomplete",
            "complete": complete,
            "reasons": [] if complete else ["private_event"],
        },
        "dependencies": dependencies,
        "health": {
            "audit_hook_healthy": True,
            "profile_healthy": True,
            "started": True,
            "stopped": True,
            "observed_event_count": sum(len(v) for v in dependencies.values()),
            "recorded_fact_count": sum(len(v) for v in dependencies.values()),
            "dropped_event_count": 0,
            "unsupported_event_kinds": [],
            "private_event_kinds": [],
            "internal_failure_kinds": [],
        },
    }


def _candidate(
    *,
    tag: str = "alpha",
    component_cids: dict[str, str] | None = None,
    external_snapshot_cids: tuple[str, ...] = (),
    **overrides: Any,
) -> CandidateExecutionContext:
    labels = {
        "execution_key": f"ek-{tag}",
        "static_trace": f"static-{tag}",
        "runtime_trace": f"runtime-{tag}",
        "repository_forest": f"forest-{tag}",
        "environment": f"env-{tag}",
        "policy": f"policy-{tag}",
        "pass_receipt": f"receipt-{tag}",
        "test_ast": f"ast-{tag}",
        "dependency_lock": f"lock-{tag}",
        "installed_distributions": f"dist-{tag}",
        "capability_root": f"cap-{tag}",
        "fixtures": f"fixtures-{tag}",
        "hooks": f"hooks-{tag}",
        "parameters": f"params-{tag}",
        "source": f"source-{tag}",
    }
    cids = {name: _component_cid(label) for name, label in labels.items()}
    defaults = dict(
        locator_cid=_cid(f"locator-{tag}"),
        execution_key_cid=cids["execution_key"],
        pass_receipt_cid=cids["pass_receipt"],
        repository_forest_cid=cids["repository_forest"],
        test_ast_cid=cids["test_ast"],
        static_trace_root_cid=cids["static_trace"],
        runtime_trace_root_cid=cids["runtime_trace"],
        environment_cid=cids["environment"],
        policy_cid=cids["policy"],
        dependency_lock_cid=cids["dependency_lock"],
        installed_distributions_cid=cids["installed_distributions"],
        capability_root_cid=cids["capability_root"],
        component_cids={
            "execution_key": cids["execution_key"],
            "static_trace": cids["static_trace"],
            "runtime_trace": cids["runtime_trace"],
            "repository_forest": cids["repository_forest"],
            "environment": cids["environment"],
            "policy": cids["policy"],
            "pass_receipt": cids["pass_receipt"],
            "test_ast": cids["test_ast"],
            "fixtures": cids["fixtures"],
            "hooks": cids["hooks"],
            "parameters": cids["parameters"],
            "source": cids["source"],
            **(component_cids or {}),
        },
        external_snapshot_cids=external_snapshot_cids,
        retained_at_ms=NOW_MS,
    )
    defaults.update(overrides)
    return CandidateExecutionContext(**defaults)


def _current_from_candidate(
    candidate: CandidateExecutionContext,
    **overrides: Any,
) -> CurrentExecutionContext:
    defaults = dict(
        locator_cid=candidate.locator_cid,
        execution_key_cid=candidate.execution_key_cid,
        repository_forest_cid=candidate.repository_forest_cid,
        test_ast_cid=candidate.test_ast_cid,
        static_trace_root_cid=candidate.static_trace_root_cid,
        runtime_trace_root_cid=candidate.runtime_trace_root_cid,
        environment_cid=candidate.environment_cid,
        policy_cid=candidate.policy_cid,
        dependency_lock_cid=candidate.dependency_lock_cid,
        installed_distributions_cid=candidate.installed_distributions_cid,
        platform_cid=candidate.platform_cid,
        capability_root_cid=candidate.capability_root_cid,
        component_cids=dict(candidate.component_cids),
        external_snapshot_cids=tuple(candidate.external_snapshot_cids),
        rebuild_source="fresh_live_rebuild",
        rebuilt_at_ms=NOW_MS,
    )
    defaults.update(overrides)
    return CurrentExecutionContext(**defaults)


def _matching_bundle(
    tmp_path: Path,
    *,
    tag: str = "alpha",
    mutate_file: bool = False,
) -> tuple[
    CandidateExecutionContext,
    CurrentExecutionContext,
    dict[str, bytes],
    dict[str, Any],
    Path,
]:
    """Build candidate + current + runtime trace + live file under tmp_path."""

    payload = b"fixture-payload-bytes"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(exist_ok=True)
    data = fixtures / "payload.bin"
    data.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if mutate_file:
        data.write_bytes(payload + b"-changed")

    runtime_trace = _complete_runtime_trace(
        files=[
            {
                "root_id": "repo",
                "path": "fixtures/payload.bin",
                "size_bytes": len(payload),
                "content_sha256": digest,
            }
        ]
    )
    candidate = _candidate(tag=tag)
    current = _current_from_candidate(candidate)
    components = {
        "execution_key": _component_bytes(f"ek-{tag}"),
        "static_trace": _component_bytes(f"static-{tag}"),
        "runtime_trace": canonical_json_bytes(runtime_trace),
        "repository_forest": _component_bytes(f"forest-{tag}"),
        "environment": _component_bytes(f"env-{tag}"),
        "policy": _component_bytes(f"policy-{tag}"),
        "pass_receipt": _component_bytes(f"receipt-{tag}"),
        "test_ast": _component_bytes(f"ast-{tag}"),
    }
    # Align runtime_trace root CID with actual bytes when using rehash path.
    runtime_cid = content_identity(runtime_trace)
    candidate = CandidateExecutionContext(
        locator_cid=candidate.locator_cid,
        execution_key_cid=candidate.execution_key_cid,
        pass_receipt_cid=candidate.pass_receipt_cid,
        repository_forest_cid=candidate.repository_forest_cid,
        test_ast_cid=candidate.test_ast_cid,
        static_trace_root_cid=candidate.static_trace_root_cid,
        runtime_trace_root_cid=runtime_cid,
        environment_cid=candidate.environment_cid,
        policy_cid=candidate.policy_cid,
        dependency_lock_cid=candidate.dependency_lock_cid,
        installed_distributions_cid=candidate.installed_distributions_cid,
        capability_root_cid=candidate.capability_root_cid,
        component_cids={
            **dict(candidate.component_cids),
            "runtime_trace": runtime_cid,
        },
        external_snapshot_cids=candidate.external_snapshot_cids,
        retained_at_ms=candidate.retained_at_ms,
    )
    current = _current_from_candidate(
        candidate,
        runtime_trace_root_cid=runtime_cid,
        component_cids={
            **dict(candidate.component_cids),
            "runtime_trace": runtime_cid,
        },
    )
    return candidate, current, components, runtime_trace, tmp_path


# ---------------------------------------------------------------------------
# Interfaces / symbols
# ---------------------------------------------------------------------------


def test_public_interfaces_and_symbols_are_stable() -> None:
    assert RUNTIME_CONTEXT_REVALIDATOR_INTERFACE == "RuntimeContextRevalidator@1"
    assert CANDIDATE_COMPARISON_INTERFACE == "CandidateComparison@1"
    assert POST_PASS_RUNTIME_TRACE_CAPTURE_INTERFACE == "PostPassRuntimeTraceCapture@1"
    assert RUNTIME_DEPENDENCY_TRACE_INTERFACE == "RuntimeDependencyTrace@1"

    revalidator = build_runtime_context_revalidator(require_runtime_frontier=False)
    assert revalidator.interface == RUNTIME_CONTEXT_REVALIDATOR_INTERFACE
    assert revalidator.may_authorize_skip is False
    assert "resolve_bounded_candidate_descriptor" in revalidator.authority_sequence_prefix


# ---------------------------------------------------------------------------
# Locator-only lookup
# ---------------------------------------------------------------------------


def test_lookup_starts_from_stable_locator_only(tmp_path: Path) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(tmp_path)
    store = TestCandidateContextStore(tmp_path / "store", clock=lambda: NOW_S)
    put = store.publish(candidate, components, locator_cid=candidate.locator_cid)
    assert put.stored

    revalidator = RuntimeContextRevalidator(
        candidate_store=store,
        current_context_provider=StaticCurrentContextProvider(current),
        allowed_roots={"repo": root},
        clock=lambda: NOW_S,
    )

    # Locator string is the only key; no node id / path fallback.
    result = revalidator.revalidate(candidate.locator_cid)
    assert result.is_proceed
    assert result.reason is RevalidationReason.CONTEXT_UNCHANGED
    assert result.fixtures_executed is False
    assert result.test_body_executed is False
    assert result.may_authorize_skip is False
    assert result.may_proceed_to_certificate_verification is True
    assert result.lookup_hit is True

    # Non-locator garbage cannot discover candidates.
    miss = revalidator.revalidate(None)
    assert miss.is_run
    assert miss.reason is RevalidationReason.LOCATOR_MISSING

    wrong = revalidator.revalidate(_cid("other-locator"))
    assert wrong.is_run
    assert wrong.reason is RevalidationReason.CANDIDATE_MISSING


def test_locator_object_with_locator_cid_is_accepted(tmp_path: Path) -> None:
    candidate, current, components, _, root = _matching_bundle(tmp_path)
    store = TestCandidateContextStore(tmp_path / "store", clock=lambda: NOW_S)
    store.publish(candidate, components, locator_cid=candidate.locator_cid)
    revalidator = RuntimeContextRevalidator(
        candidate_store=store,
        current_context_provider=StaticCurrentContextProvider(current),
        allowed_roots={"repo": root},
    )
    hint = SimpleNamespace(locator_cid=candidate.locator_cid)
    result = revalidator.revalidate(hint)
    assert result.is_proceed


# ---------------------------------------------------------------------------
# Fresh dependency resolution
# ---------------------------------------------------------------------------


def test_retained_file_dependency_is_freshly_resolved_and_content_addressed(
    tmp_path: Path,
) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(tmp_path)
    resolver = FilesystemDependencyResolver(allowed_roots={"repo": root})
    report = resolve_retained_runtime_frontier(runtime_trace, resolver)
    assert report.matched
    assert report.complete
    assert len(report.facts) == 1
    assert report.facts[0].status.value == "matched"
    assert report.facts[0].current_digest == runtime_trace["dependencies"]["files"][0][
        "content_sha256"
    ]


def test_changed_file_dependency_returns_run(tmp_path: Path) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(
        tmp_path, mutate_file=True
    )
    revalidator = RuntimeContextRevalidator(
        current_context_provider=StaticCurrentContextProvider(current),
        allowed_roots={"repo": root},
    )
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        component_bytes=components,
        retained_runtime_trace=runtime_trace,
    )
    assert result.is_run
    assert result.reason is RevalidationReason.DEPENDENCY_CHANGED
    assert result.comparison is not None
    assert result.comparison.changed_dependencies


def test_unresolvable_file_dependency_returns_run(tmp_path: Path) -> None:
    runtime_trace = _complete_runtime_trace(
        files=[
            {
                "root_id": "repo",
                "path": "fixtures/missing.bin",
                "size_bytes": 1,
                "content_sha256": "ab" * 32,
            }
        ]
    )
    candidate = _candidate()
    current = _current_from_candidate(candidate)
    revalidator = RuntimeContextRevalidator(
        current_context_provider=StaticCurrentContextProvider(current),
        allowed_roots={"repo": tmp_path},
    )
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        retained_runtime_trace=runtime_trace,
    )
    assert result.is_run
    assert result.reason is RevalidationReason.DEPENDENCY_UNRESOLVABLE


def test_uncontrolled_subprocess_fact_returns_run(tmp_path: Path) -> None:
    runtime_trace = _complete_runtime_trace(
        extra_kinds={
            "subprocesses": [
                {"name": "python", "content_sha256": "cd" * 32},
            ]
        }
    )
    # Override dependencies properly — extra_kinds merges into dict.
    runtime_trace["dependencies"]["subprocesses"] = [
        {"name": "python", "content_sha256": "cd" * 32}
    ]
    candidate = _candidate()
    current = _current_from_candidate(candidate)
    revalidator = RuntimeContextRevalidator(
        current_context_provider=StaticCurrentContextProvider(current),
        allowed_roots={"repo": tmp_path},
    )
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        retained_runtime_trace=runtime_trace,
    )
    assert result.is_run
    assert result.reason is RevalidationReason.DEPENDENCY_UNCONTROLLED


def test_incomplete_runtime_trace_returns_run(tmp_path: Path) -> None:
    runtime_trace = _complete_runtime_trace(complete=False)
    candidate = _candidate()
    current = _current_from_candidate(candidate)
    revalidator = RuntimeContextRevalidator(
        current_context_provider=StaticCurrentContextProvider(current),
        allowed_roots={"repo": tmp_path},
    )
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        retained_runtime_trace=runtime_trace,
    )
    assert result.is_run
    assert result.reason is RevalidationReason.TRACE_INCOMPLETE


# ---------------------------------------------------------------------------
# Identity dimension matching
# ---------------------------------------------------------------------------


def test_all_identity_dimensions_must_match_for_proceed(tmp_path: Path) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(tmp_path)
    revalidator = RuntimeContextRevalidator(
        allowed_roots={"repo": root},
    )

    # Matching current → proceed.
    ok = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        current=current,
        component_bytes=components,
        retained_runtime_trace=runtime_trace,
    )
    assert ok.is_proceed
    assert ok.fixtures_executed is False
    assert ok.test_body_executed is False

    mutations = {
        "test_ast_cid": _cid("ast-mutated"),
        "static_trace_root_cid": _cid("static-mutated"),
        "runtime_trace_root_cid": _cid("runtime-mutated"),
        "environment_cid": _cid("env-mutated"),
        "policy_cid": _cid("policy-mutated"),
        "repository_forest_cid": _cid("forest-mutated"),
        "dependency_lock_cid": _cid("lock-mutated"),
        "installed_distributions_cid": _cid("dist-mutated"),
        "capability_root_cid": _cid("cap-mutated"),
        "execution_key_cid": _cid("ek-mutated"),
    }
    for field_name, value in mutations.items():
        mutated = _current_from_candidate(candidate, **{field_name: value})
        # Keep runtime frontier happy by reusing retained matching file.
        result = revalidator.revalidate(
            candidate.locator_cid,
            candidate=candidate,
            current=mutated,
            component_bytes=components,
            retained_runtime_trace=runtime_trace,
        )
        assert result.is_run, field_name
        assert result.reason is RevalidationReason.IDENTITY_MISMATCH, field_name


def test_fixture_hook_parameter_component_mismatch_returns_run(
    tmp_path: Path,
) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(tmp_path)
    revalidator = RuntimeContextRevalidator(allowed_roots={"repo": root})
    for key in ("fixtures", "hooks", "parameters"):
        mutated_components = dict(current.component_cids)
        mutated_components[key] = _cid(f"{key}-changed")
        mutated = _current_from_candidate(
            candidate, component_cids=mutated_components
        )
        result = revalidator.revalidate(
            candidate.locator_cid,
            candidate=candidate,
            current=mutated,
            component_bytes=components,
            retained_runtime_trace=runtime_trace,
        )
        assert result.is_run
        assert key in (result.comparison.mismatched_dimensions if result.comparison else ())


def test_external_snapshot_mismatch_returns_run(tmp_path: Path) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(tmp_path)
    candidate = CandidateExecutionContext(
        **{
            **{
                field: getattr(candidate, field)
                for field in (
                    "locator_cid",
                    "execution_key_cid",
                    "pass_receipt_cid",
                    "repository_forest_cid",
                    "test_ast_cid",
                    "static_trace_root_cid",
                    "runtime_trace_root_cid",
                    "environment_cid",
                    "policy_cid",
                    "dependency_lock_cid",
                    "installed_distributions_cid",
                    "capability_root_cid",
                    "component_cids",
                    "retained_at_ms",
                )
            },
            "external_snapshot_cids": (_cid("snap-a"),),
        }
    )
    current = _current_from_candidate(
        candidate, external_snapshot_cids=(_cid("snap-b"),)
    )
    revalidator = RuntimeContextRevalidator(allowed_roots={"repo": root})
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        current=current,
        retained_runtime_trace=runtime_trace,
        component_bytes=components,
    )
    assert result.is_run
    assert result.comparison is not None
    assert "external_snapshots" in result.comparison.mismatched_dimensions


def test_historical_current_context_is_rejected(tmp_path: Path) -> None:
    candidate, _, components, runtime_trace, root = _matching_bundle(tmp_path)
    with pytest.raises(Exception):
        # CurrentExecutionContext itself rejects non-fresh rebuild sources.
        CurrentExecutionContext(
            locator_cid=candidate.locator_cid,
            execution_key_cid=candidate.execution_key_cid,
            repository_forest_cid=candidate.repository_forest_cid,
            test_ast_cid=candidate.test_ast_cid,
            static_trace_root_cid=candidate.static_trace_root_cid,
            runtime_trace_root_cid=candidate.runtime_trace_root_cid,
            environment_cid=candidate.environment_cid,
            policy_cid=candidate.policy_cid,
            rebuild_source="historical_trace",
        )


def test_compare_candidate_to_current_never_authorizes_skip() -> None:
    candidate = _candidate()
    current = _current_from_candidate(candidate)
    comparison = compare_candidate_to_current(candidate, current)
    assert comparison.matched is True
    assert comparison.may_authorize_skip is False
    assert comparison.interface == CANDIDATE_COMPARISON_INTERFACE
    assert comparison.to_dict()["may_authorize_skip"] is False


# ---------------------------------------------------------------------------
# Proceed without executing fixtures / body
# ---------------------------------------------------------------------------


def test_verified_unchanged_context_proceeds_without_fixtures_or_body(
    tmp_path: Path,
) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(tmp_path)
    fixture_calls = {"n": 0}
    body_calls = {"n": 0}

    class CountingProvider(StaticCurrentContextProvider):
        def compile_current(self, **kwargs: Any) -> CurrentExecutionContext | None:
            # Must not run fixtures to rebuild current context.
            assert fixture_calls["n"] == 0
            assert body_calls["n"] == 0
            return super().compile_current(**kwargs)

    revalidator = RuntimeContextRevalidator(
        current_context_provider=CountingProvider(current),
        allowed_roots={"repo": root},
    )
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        component_bytes=components,
        retained_runtime_trace=runtime_trace,
    )
    assert result.action is RevalidationAction.PROCEED_TO_CERTIFICATE_VERIFICATION
    assert result.fixtures_executed is False
    assert result.test_body_executed is False
    assert result.may_authorize_skip is False
    # Revalidation maps to RUN at the sealed disposition boundary (not SKIP).
    disposition = result.to_disposition()
    assert disposition.action is RuntimeReuseAction.RUN
    assert disposition.collection_failed is False


# ---------------------------------------------------------------------------
# Normal miss: setup/call/teardown exactly once + capture/publish
# ---------------------------------------------------------------------------


def test_normal_miss_executes_lifecycle_exactly_once_and_captures(
    tmp_path: Path,
) -> None:
    counters = {"setup": 0, "call": 0, "teardown": 0}
    published: list[Any] = []

    def setup() -> None:
        counters["setup"] += 1

    def call() -> str:
        counters["call"] += 1
        return "ok"

    def teardown() -> None:
        counters["teardown"] += 1

    def publisher(observation: Any, **kwargs: Any) -> dict[str, Any]:
        published.append(observation)
        return {
            "published": True,
            "observation_id": observation.observation_id,
            "may_authorize_skip": False,
        }

    capture = PostPassRuntimeTraceCapture(
        locator_cid=_cid("locator-miss"),
        execution_key_cid=_cid("ek-miss"),
        pass_receipt_cid=_cid("receipt-miss"),
        publisher=publisher,
        clock=lambda: NOW_S,
    )
    assert capture.interface == POST_PASS_RUNTIME_TRACE_CAPTURE_INTERFACE
    assert capture.may_authorize_skip is False

    outcome = capture.execute_lifecycle_once(
        setup=setup,
        call=call,
        teardown=teardown,
        runtime_trace_root_cid=_cid("runtime-observed"),
        pass_receipt_cid=_cid("receipt-miss"),
        locator_cid=_cid("locator-miss"),
        execution_key_cid=_cid("ek-miss"),
    )
    assert outcome["passed"] is True
    assert counters == {"setup": 1, "call": 1, "teardown": 1}
    assert capture.setup_call_count == 1
    assert capture.test_call_count == 1
    assert capture.teardown_call_count == 1
    assert capture.phase is LifecyclePhase.COMPLETE
    assert outcome["observation"] is not None
    assert outcome["observation"].test_call_count == 1
    assert outcome["observation"].duplicate_test_call_forbidden is True
    assert outcome["observation"].observation_source == "post_pass_lifecycle"

    # Capture forbids a second body execution path.
    with pytest.raises(RuntimeError, match="already captured|exactly one|duplicate"):
        capture.note_call()

    pub = capture.publish_observed_trace()
    assert pub["published"] is True
    assert len(published) == 1
    assert published[0].may_authorize_skip is False if hasattr(published[0], "may_authorize_skip") else True


def test_duplicate_setup_call_teardown_is_rejected() -> None:
    capture = PostPassRuntimeTraceCapture(
        locator_cid=_cid("loc"),
        execution_key_cid=_cid("ek"),
        pass_receipt_cid=_cid("rcpt"),
    )
    capture.note_setup()
    with pytest.raises(RuntimeError, match="duplicate"):
        capture.note_setup()
    capture2 = PostPassRuntimeTraceCapture(
        locator_cid=_cid("loc"),
        execution_key_cid=_cid("ek"),
        pass_receipt_cid=_cid("rcpt"),
    )
    capture2.note_setup()
    capture2.note_call()
    with pytest.raises(RuntimeError, match="duplicate"):
        capture2.note_call()


def test_capture_without_full_lifecycle_is_rejected() -> None:
    capture = PostPassRuntimeTraceCapture(
        locator_cid=_cid("loc"),
        execution_key_cid=_cid("ek"),
        pass_receipt_cid=_cid("rcpt"),
    )
    capture.note_setup()
    with pytest.raises(RuntimeError, match="exactly one"):
        capture.capture_observed_runtime_trace(
            runtime_trace_root_cid=_cid("rt"),
        )


def test_miss_path_via_revalidator_factory_and_store(tmp_path: Path) -> None:
    """Candidate miss → RUN; then lifecycle capture publishes once."""

    revalidator = build_runtime_context_revalidator(
        candidate_store=TestCandidateContextStore(tmp_path / "store", clock=lambda: NOW_S),
        allowed_roots={"repo": tmp_path},
        require_runtime_frontier=True,
    )
    result = revalidator.revalidate(_cid("no-such-locator"))
    assert result.is_run
    assert result.reason is RevalidationReason.CANDIDATE_MISSING

    counters = {"setup": 0, "call": 0, "teardown": 0}
    capture = revalidator.new_post_pass_capture(
        locator_cid=_cid("loc"),
        execution_key_cid=_cid("ek"),
        pass_receipt_cid=_cid("rcpt"),
    )
    capture.execute_lifecycle_once(
        setup=lambda: counters.__setitem__("setup", counters["setup"] + 1),
        call=lambda: counters.__setitem__("call", counters["call"] + 1),
        teardown=lambda: counters.__setitem__("teardown", counters["teardown"] + 1),
        runtime_trace_root_cid=_cid("rt"),
    )
    assert counters == {"setup": 1, "call": 1, "teardown": 1}
    assert capture.observation is not None


# ---------------------------------------------------------------------------
# Store integration: rehash + publish round-trip
# ---------------------------------------------------------------------------


def test_store_lookup_rehash_and_identity_match_round_trip(tmp_path: Path) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(tmp_path)
    store = TestCandidateContextStore(tmp_path / "store", clock=lambda: NOW_S)
    put = store.publish(candidate, components, locator_cid=candidate.locator_cid)
    assert put.stored
    assert put.may_authorize_skip is False

    revalidator = RuntimeContextRevalidator(
        candidate_store=store,
        current_context_provider=StaticCurrentContextProvider(current),
        allowed_roots={"repo": root},
        clock=lambda: NOW_S,
    )
    result = revalidator.revalidate(candidate.locator_cid)
    assert result.is_proceed
    assert result.candidate is not None
    assert result.candidate.candidate_context_id == candidate.candidate_context_id
    assert result.comparison is not None
    assert result.comparison.matched is True
    assert isinstance(result.comparison, CandidateComparison)


def test_component_integrity_failure_returns_run(tmp_path: Path) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(tmp_path)
    # Tamper one component while keeping the claimed CID.
    bad = dict(components)
    bad["static_trace"] = _component_bytes("static-tampered")
    revalidator = RuntimeContextRevalidator(
        current_context_provider=StaticCurrentContextProvider(current),
        allowed_roots={"repo": root},
    )
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        component_bytes=bad,
        retained_runtime_trace=runtime_trace,
    )
    assert result.is_run
    assert result.reason is RevalidationReason.COMPONENT_INTEGRITY_FAILED


def test_missing_current_context_returns_run(tmp_path: Path) -> None:
    candidate, _, components, runtime_trace, root = _matching_bundle(tmp_path)
    revalidator = RuntimeContextRevalidator(allowed_roots={"repo": root})
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        component_bytes=components,
        retained_runtime_trace=runtime_trace,
    )
    assert result.is_run
    assert result.reason is RevalidationReason.CURRENT_CONTEXT_UNAVAILABLE


def test_empty_complete_frontier_with_matching_identities_proceeds(
    tmp_path: Path,
) -> None:
    runtime_trace = _complete_runtime_trace()
    runtime_cid = content_identity(runtime_trace)
    candidate = _candidate()
    candidate = CandidateExecutionContext(
        locator_cid=candidate.locator_cid,
        execution_key_cid=candidate.execution_key_cid,
        pass_receipt_cid=candidate.pass_receipt_cid,
        repository_forest_cid=candidate.repository_forest_cid,
        test_ast_cid=candidate.test_ast_cid,
        static_trace_root_cid=candidate.static_trace_root_cid,
        runtime_trace_root_cid=runtime_cid,
        environment_cid=candidate.environment_cid,
        policy_cid=candidate.policy_cid,
        dependency_lock_cid=candidate.dependency_lock_cid,
        installed_distributions_cid=candidate.installed_distributions_cid,
        capability_root_cid=candidate.capability_root_cid,
        component_cids={
            **dict(candidate.component_cids),
            "runtime_trace": runtime_cid,
        },
        retained_at_ms=NOW_MS,
    )
    current = _current_from_candidate(candidate)
    revalidator = RuntimeContextRevalidator(allowed_roots={"repo": tmp_path})
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        current=current,
        retained_runtime_trace=runtime_trace,
    )
    assert result.is_proceed
    assert result.fixtures_executed is False
    assert result.test_body_executed is False


def test_environment_fact_fresh_resolution(tmp_path: Path) -> None:
    value = "UTC"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    runtime_trace = _complete_runtime_trace(
        environment=[{"name": "TZ", "value_sha256": digest}]
    )
    runtime_cid = content_identity(runtime_trace)
    candidate = _candidate()
    candidate = CandidateExecutionContext(
        locator_cid=candidate.locator_cid,
        execution_key_cid=candidate.execution_key_cid,
        pass_receipt_cid=candidate.pass_receipt_cid,
        repository_forest_cid=candidate.repository_forest_cid,
        test_ast_cid=candidate.test_ast_cid,
        static_trace_root_cid=candidate.static_trace_root_cid,
        runtime_trace_root_cid=runtime_cid,
        environment_cid=candidate.environment_cid,
        policy_cid=candidate.policy_cid,
        dependency_lock_cid=candidate.dependency_lock_cid,
        installed_distributions_cid=candidate.installed_distributions_cid,
        capability_root_cid=candidate.capability_root_cid,
        component_cids=dict(candidate.component_cids),
        retained_at_ms=NOW_MS,
    )
    current = _current_from_candidate(candidate)
    revalidator = RuntimeContextRevalidator(
        allowed_roots={"repo": tmp_path},
        environ={"TZ": value},
    )
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        current=current,
        retained_runtime_trace=runtime_trace,
    )
    assert result.is_proceed

    # Changed env → RUN.
    revalidator_changed = RuntimeContextRevalidator(
        allowed_roots={"repo": tmp_path},
        environ={"TZ": "Europe/Paris"},
    )
    changed = revalidator_changed.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        current=current,
        retained_runtime_trace=runtime_trace,
    )
    assert changed.is_run
    assert changed.reason is RevalidationReason.DEPENDENCY_CHANGED


def test_result_to_dict_is_json_safe_and_non_authoritative(tmp_path: Path) -> None:
    candidate, current, components, runtime_trace, root = _matching_bundle(tmp_path)
    revalidator = RuntimeContextRevalidator(allowed_roots={"repo": root})
    result = revalidator.revalidate(
        candidate.locator_cid,
        candidate=candidate,
        current=current,
        component_bytes=components,
        retained_runtime_trace=runtime_trace,
    )
    payload = result.to_dict()
    assert payload["may_authorize_skip"] is False
    assert payload["action"] == "PROCEED_TO_CERTIFICATE_VERIFICATION"
    # Ensure JSON serializable.
    json.dumps(payload)
