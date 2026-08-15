# agent_supervisor.verification

**Layer:** Mid · **Board:** `incremental-verification-planner-v1` (IVP)  
**Primary module:** `IncrementalVerificationPlanner`  
**Release report:** [`docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md`](../../../docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md)

## Purpose

Select and execute the minimum defensible verification work for a proposed
patch: reuse only **exact** valid receipts, produce compact counterexamples,
recommend a **provider-neutral** repair route, and emit a structural Merkle
commitment over admitted receipts.

This package is **not** an end-user application, model provider, ZK VM, payment
surface, or distributed scheduler.

## Production-shaped APIs

```python
from ipfs_accelerate_py.agent_supervisor.verification.planner import (
    create_verification_plan,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    choose_model_route,
)
from ipfs_accelerate_py.agent_supervisor.verification.bundle import (
    build_verification_commitment,
)

plan = create_verification_plan(
    repository_state,
    invalidation_plan,
    context_pack,
    patch_delta,
    policy,
)
route = choose_model_route(
    context_pack,
    plan,
    prior_attempts,
    available_models,
    route_policy,
)
commitment = build_verification_commitment(verification_bundle)
```

Final root-package lazy export freeze is owned by **IVP-019**. Imports above
use the package modules directly and remain side-effect free on import.

## Modules

| Module | Path | Responsibility |
| --- | --- | --- |
| contracts | `contracts.py` | Statuses, keys, receipts, plans, bundles, summaries, commitments |
| datasets adapter | `datasets_adapter.py` | Lazy `RepositoryState` / `InvalidationPlan` / `SemanticCapsule` / `ContextPack` normalization |
| receipt store | `receipt_store.py` | Durable immutable receipts + CAS index (hermetic / optional ipfs-kit) |
| receipt cache | `receipt_cache.py` | Exact-key production admission, tombstones, historical preservation |
| process runner | `process_runner.py` | Single admitted hermetic subprocess + process-tree cancellation |
| pytest adapter | `adapters/pytest_adapter.py` | Exact node IDs / full-suite oracle |
| mypy adapter | `adapters/mypy_adapter.py` | Explicit mypy argv execution |
| prover adapters | `adapters/prover_adapters.py` | Z3 + registry-admitted proof assistants |
| selection | `selection.py` | Pure semantic-edge affected-check selection |
| planner | `planner.py` | `create_verification_plan` / `IncrementalVerificationPlanner` |
| counterexamples | `counterexamples.py` | Minimized compact failure diagnostics |
| model route | `model_route.py` | Deterministic/small/medium/frontier/human routes |
| bundle | `bundle.py` | Bundle, ContextPack summary, commitment builder |
| executor | `executor.py` | Plan execution orchestration |
| evaluation | `evaluation.py` | Controlled selected-vs-full differential evaluation |

## Terminal statuses

Closed set (case-sensitive):

```text
passed, failed, proved, disproved, unknown, timeout, unavailable,
not_modeled, stale, invalid, cancelled, simulated
```

`timeout` / `unavailable` never project to success. `simulated` never satisfies
production. Cancellation fences late success publication.

## Exact cache key

`VerificationReceiptKey@1` binds repository tree CID, semantic-state root CID,
affected symbol-version CIDs, environment CID, dependency-lock CID, selector
CID, proof-obligation CID (or `not_applicable`), tool name/version,
configuration CID, fixture-data CIDs, network policy, receipt-schema version,
receipt kind, and adapter schema. Any component mutation changes the key.
Identity is re-derived by `VerificationIdentityCompiler` from observed inputs;
unchecked caller CID strings are not authority.

## Invalidation and reuse

- Relevant changes select/invalidate; unrelated edits preserve the old receipt
  under its **old key** only.
- Environment, lock, tool, configuration, fixture, or schema changes invalidate.
- Cross-tree reuse of an “unaffected” receipt is **rejected** (exact full-tree
  binding). The release target for incompatible cross-tree unaffected reuse is
  intentionally **unmet**; see the report.
- Stale and simulated receipts are never production-accepted.

## Selection, fallback, and routing

- Selection walks supplied semantic / test / fixture / config / proof edges.
- Uncertainty broadens toward conservative or full-suite fallbacks.
- Acceptance requires production-admissible required success and no pending
  mandatory fallback for required obligations.
- `ModelRouteDecision` chooses a **capability class** only:
  `deterministic_only`, `small_local_model`, `medium_model`, `frontier_model`,
  or `human_review_required`. Provider/vendor resolution is out of scope.

## Commitment non-claims

`build_verification_commitment` builds a SHA-256 Merkle tree over canonical
DAG-JSON leaves with domain tags `IVP-LEAF@1`, `IVP-NODE@1`, and `IVP-EMPTY@1`.

**Mandatory non-claims:**

1. The commitment is not a ZK proof (not a zero-knowledge proof).
2. Signatures need trusted issuers — signed receipts do not prove test
   execution unless the issuer is trusted.
3. Structural validation is not cryptographic validation of execution.

`VerificationCommitment.IS_ZERO_KNOWLEDGE_PROOF` is `False`.

### Exact future ZK step

Before any external membership/aggregation circuit:

1. Freeze the admitted receipt leaf codec and trust policy.
2. Publish deterministic cross-implementation Merkle vectors.
3. Then — and only then — add an external circuit over that root without
   changing ordinary verification authority.

## Operations

| Task | Command |
| --- | --- |
| Generate benchmark | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 benchmarks/agent_supervisor/incremental_verification.py --output artifacts/agent_supervisor/incremental_verification/benchmark.json` |
| Validate benchmark | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/benchmarks/test_incremental_verification_planner_benchmark.py` |
| Validate this report | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_incremental_verification_report.py` |
| Focused planner tests | `PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q --timeout=300 test/api/test_agent_supervisor_incremental_verification_planner.py` |

Importing this package performs no network I/O and never installs packages.

## Trust doctrine

No cache presence, provider text, signature alone, CID string, historical pass,
or structural validation creates verification authority. Production acceptance
requires current exact-key leaves that pass/prove with no required unresolved
obligation, mandatory fallback, or human-review gate remaining (advisory
obligations may stay unresolved without upgrade).

## Evidence

| Evidence | Meaning |
| --- | --- |
| `ivp/documentation@1` | This README + architecture report |
| `ivp/release-report@1` | `INCREMENTAL_VERIFICATION_PLANNER_REPORT.md` |
| `ivp/benchmark@1` | Current-tree benchmark artifact (IVP-017) |
| `ivp/verification-commitment@1` | Commitment builder / contract |

## Limitations (honest)

- Hard conformance false-negative gate is IVP-016/IVP-019; benchmark/report may
  be **red** while still recording complete metrics.
- Controlled fixtures may leave static/proof execution `not_measured`.
- Missing provers are typed `unavailable` / `not_measured`, never fabricated.
- Full ZK execution is out of scope until the frozen leaf codec, trust policy,
  and cross-implementation Merkle vectors exist.

## See also

- [INCREMENTAL_VERIFICATION_PLANNER_REPORT.md](../../../docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_REPORT.md)
- Plan (operator-protected): `docs/architecture/INCREMENTAL_VERIFICATION_PLANNER_PLAN.md`
- Benchmark harness: `benchmarks/agent_supervisor/incremental_verification.py`
