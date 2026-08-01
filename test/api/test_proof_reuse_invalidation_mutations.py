"""PTR-091: invalidation mutation population for proof-backed test reuse.

Acceptance:

* Each relevant mutation changes or invalidates the exact execution context
  and executes the real test body under a ``RUN`` decision.
* Unrelated locator-index candidates cannot override current identity.
* The authoritative stale-skip count across the population is zero.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    ReuseAction,
    ReuseReasonCode,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import (
    TestProofCache,
    TestProofCacheLookupStatus,
)
from ipfs_accelerate_py.testing.proof_reuse.lookup import ProofReuseLookup
from ipfs_accelerate_py.testing.proof_reuse.reporting import ProofReuseSessionMetrics


def _load_mutations_fixture():
    """Load ``test/fixtures/proof_reuse_mutations.py`` without requiring a package init."""

    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "proof_reuse_mutations.py"
    )
    module_name = "proof_reuse_mutations_fixture"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, fixture_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load mutation fixture from {fixture_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_mutations = _load_mutations_fixture()

BaselineArtifacts = _mutations.BaselineArtifacts
INVALIDATION_MUTATIONS = _mutations.INVALIDATION_MUTATIONS
MutationSpec = _mutations.MutationSpec
MutationTarget = _mutations.MutationTarget
NOW_MS = _mutations.NOW_MS
ProofReuseMutationCorpus = _mutations.ProofReuseMutationCorpus
REQUIRED_MUTATION_CATEGORIES = _mutations.REQUIRED_MUTATION_CATEGORIES
StaleSkipTracker = _mutations.StaleSkipTracker
apply_mutation = _mutations.apply_mutation
assert_no_stale_proof_skip = _mutations.assert_no_stale_proof_skip
build_baseline_artifacts = _mutations.build_baseline_artifacts
mutation_changes_execution_context = _mutations.mutation_changes_execution_context
unrelated_locator_candidate = _mutations.unrelated_locator_candidate


class _CandidateStore:
    """Locator-index hint store; candidates have no authority on their own."""

    def __init__(self, candidates: Any) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int]] = []

    def lookup(self, locator_cid: str, *, max_candidates: int) -> Any:
        self.calls.append((locator_cid, max_candidates))
        return self.candidates


class _Verifier:
    def __init__(self, result: Any = True) -> None:
        self.result = result
        self.calls = 0

    def as_cache_verifier(self):
        def _verify(*_args: Any, **_kwargs: Any) -> Any:
            self.calls += 1
            return self.result

        return _verify


def _execute_real_test(decision: Any, body: Any) -> None:
    assert decision.action is ReuseAction.RUN
    assert decision.reason_code is not ReuseReasonCode.PROOF_CACHE_HIT
    body()


@pytest.fixture(scope="module")
def corpus() -> Any:
    population = ProofReuseMutationCorpus()
    population.ensure_complete()
    return population


@pytest.fixture
def baseline() -> Any:
    return build_baseline_artifacts()


def test_corpus_covers_required_evidence_categories(corpus: Any) -> None:
    assert corpus.missing_required_categories() == frozenset()
    assert REQUIRED_MUTATION_CATEGORIES <= corpus.categories()
    assert len(corpus) == len(INVALIDATION_MUTATIONS)
    assert len(corpus) >= len(REQUIRED_MUTATION_CATEGORIES)


def test_baseline_warm_context_authorizes_exact_skip(baseline: Any) -> None:
    """Control: unchanged identity with matching candidate still hits."""

    cache = TestProofCache(
        current_policy=baseline.policy,
        verifier=lambda *_a, **_k: True,
        clock=lambda: NOW_MS,
    )
    result = cache.lookup(
        baseline.locator,
        baseline.execution_key,
        candidates=(baseline.candidate,),
        now_ms=NOW_MS,
    )
    assert result.status is TestProofCacheLookupStatus.HIT
    assert result.decision.action is ReuseAction.SKIP
    assert result.decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT


@pytest.mark.parametrize(
    "mutation",
    INVALIDATION_MUTATIONS,
    ids=lambda item: item.name,
)
def test_each_mutation_changes_context_and_executes(
    mutation: Any,
    baseline: Any,
) -> None:
    assert mutation_changes_execution_context(baseline, mutation)

    current_key, current_policy = apply_mutation(baseline, mutation)
    if mutation.target is MutationTarget.EXECUTION_KEY:
        assert current_key.execution_key_id != baseline.execution_key.execution_key_id
    else:
        assert current_policy != baseline.policy
        assert current_key.execution_key_id == baseline.execution_key.execution_key_id

    cache = TestProofCache(
        current_policy=current_policy,
        verifier=lambda *_a, **_k: True,
        clock=lambda: NOW_MS,
    )
    # Index still surfaces the baseline candidate for the same locator.
    result = cache.lookup(
        baseline.locator,
        current_key,
        candidates=(baseline.candidate,),
        now_ms=NOW_MS,
    )

    executed: list[str] = []
    assert_no_stale_proof_skip(
        result.decision,
        case=mutation.name,
        expected_reason=mutation.expected_reason,
    )
    _execute_real_test(result.decision, lambda: executed.append(mutation.name))
    assert executed == [mutation.name]
    assert result.status is not TestProofCacheLookupStatus.HIT
    assert result.decision.is_run


def test_population_authoritative_stale_skip_count_is_zero(corpus: Any) -> None:
    executed: list[str] = []
    tracker = corpus.evaluate_population(execute=executed.append)

    assert tracker.authoritative_stale_skip_count == 0
    assert tracker.stale_skip_count == 0
    assert tracker.executed_count == len(corpus)
    assert executed == [item.name for item in corpus.mutations]
    assert all(action == "RUN" for _case, action, _reason in tracker.decisions)
    assert all(
        reason != ReuseReasonCode.PROOF_CACHE_HIT.value
        for _case, _action, reason in tracker.decisions
    )


def test_unrelated_locator_index_candidates_cannot_override_current_identity(
    baseline: Any,
) -> None:
    """A different node's valid certificate under the same index cannot skip."""

    _other_locator, _other_key, other_candidate = unrelated_locator_candidate(
        baseline=baseline
    )
    # Pollute the index with an unrelated warm candidate plus the baseline one.
    polluted = (other_candidate, baseline.candidate)

    # Mutate current identity away from baseline.
    mutation = next(
        item
        for item in INVALIDATION_MUTATIONS
        if item.name == "test_function"
    )
    current_key, current_policy = apply_mutation(baseline, mutation)

    cache = TestProofCache(
        current_policy=current_policy,
        verifier=lambda *_a, **_k: True,
        clock=lambda: NOW_MS,
    )
    result = cache.lookup(
        baseline.locator,
        current_key,
        candidates=polluted,
        now_ms=NOW_MS,
    )

    tracker = StaleSkipTracker()
    executed: list[str] = []
    assert_no_stale_proof_skip(
        result.decision,
        tracker=tracker,
        case="unrelated-locator-index",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
    )
    _execute_real_test(result.decision, lambda: executed.append("ran"))
    assert executed == ["ran"]
    assert tracker.authoritative_stale_skip_count == 0
    assert result.decision.action is ReuseAction.RUN


def test_lookup_plugin_path_rejects_stale_index_hint_after_mutation(
    baseline: Any,
) -> None:
    """ProofReuseLookup@1 must treat the locator index as a non-authoritative hint."""

    mutation = next(item for item in INVALIDATION_MUTATIONS if item.name == "fixture_definition")
    current_key, current_policy = apply_mutation(baseline, mutation)
    store = _CandidateStore((baseline.candidate,))
    verifier = _Verifier(True)
    lookup = ProofReuseLookup(
        store,
        verifier,
        current_policy=current_policy,
        max_candidates=32,
        timeout_seconds=1.0,
    )

    decision = lookup.lookup(
        baseline.locator,
        current_key,
        now_ms=NOW_MS,
    )

    tracker = StaleSkipTracker()
    executed: list[str] = []
    assert_no_stale_proof_skip(
        decision,
        tracker=tracker,
        case="plugin-lookup-mutation",
        expected_reason=ReuseReasonCode.EXECUTION_KEY_MISMATCH,
    )
    _execute_real_test(decision, lambda: executed.append("ran"))
    assert executed == ["ran"]
    assert tracker.authoritative_stale_skip_count == 0
    assert store.calls  # index was consulted
    assert decision.action is ReuseAction.RUN


def test_metrics_record_executions_without_stale_skips(
    corpus: Any,
    baseline: Any,
) -> None:
    metrics = ProofReuseSessionMetrics()
    tracker = StaleSkipTracker()

    for mutation in corpus:
        current_key, current_policy = apply_mutation(baseline, mutation)
        cache = TestProofCache(
            current_policy=current_policy,
            verifier=lambda *_a, **_k: True,
            clock=lambda: NOW_MS,
        )
        result = cache.lookup(
            baseline.locator,
            current_key,
            candidates=(baseline.candidate,),
            now_ms=NOW_MS,
        )
        assert_no_stale_proof_skip(
            result.decision,
            tracker=tracker,
            case=mutation.name,
            expected_reason=mutation.expected_reason,
        )
        metrics.degraded(reason_code=result.decision.reason_code.value)
        metrics.executed(reason_code="real_execution")

    snapshot = metrics.snapshot().to_dict()
    assert snapshot["counts"]["skipped"] == 0
    assert snapshot["counts"]["executed"] == len(corpus)
    assert snapshot["counts"]["degraded"] == len(corpus)
    assert tracker.authoritative_stale_skip_count == 0
    assert "proof_cache_hit" not in snapshot["reasons"]


def test_assert_no_stale_proof_skip_rejects_skip_decision() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
        reuse_skip,
    )

    decision = reuse_skip(
        certificate_cid="cid:certificate",
        receipt_cid="cid:receipt",
    )
    tracker = StaleSkipTracker()
    with pytest.raises(AssertionError, match="stale proof skip"):
        assert_no_stale_proof_skip(decision, tracker=tracker, case="forged-hit")
    assert tracker.authoritative_stale_skip_count == 1
