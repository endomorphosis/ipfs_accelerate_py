# Semantic Compression Harness — Release Report

**Interface:** `SemanticStateHarnessRelease@1`  
**Task:** `SCH-018`  
**Goal:** `SCH-G050`  
**Board namespace:** `semantic-compression-harness-v1`  
**Bundle:** `sch/release`  
**Package:** `ipfs_accelerate_py.agent_supervisor.semantic_state`  
**Status:** implementation complete for the harness MVP; **not** a production-readiness or ZK claim

This document is the final release evidence report for the Python semantic-compression
coding-agent harness. It documents architecture, modules, commands, tests, measured
benchmark results, honest limitations, bottlenecks, and exact remaining work before
ZK aggregation and production integration. Figures below are taken from committed
receipts and the checked-in benchmark report; they are not aspirational.

Normative plan (read-only for workers):  
[`docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md`](../architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md)  
Companion board:  
[`docs/architecture/semantic_compression_harness.todo.md`](../architecture/semantic_compression_harness.todo.md)  
Measured benchmark narrative:  
[`docs/benchmarks/semantic_compression_harness_results.md`](../benchmarks/semantic_compression_harness_results.md)

---

## 1. Traceability (commits, roots, receipts)

### 1.1 Dependency seal (SCH-000)

Seal schema: `ipfs-accelerate.agent-supervisor.semantic-state-dependency-seal@2`  
Seal path: `config/semantic_state_dependencies.seal.json`  
Status: **sealed**

| Authority role | Repository | Commit (exact) | Tree (exact) |
|---|---|---|---|
| `accelerate_harness` | `endomorphosis/ipfs_accelerate_py` | `271e331af802f37d759c000666282631a99f7aab` | `5859208bdab59338eab67a5cd0102c193ca6c388` |
| `incremental_semantic_index` | `endomorphosis/ipfs_datasets_py` | `1330038f626ef92993f03d46f21e1a57719e9c25` | `c1686dfce8e14ebd32327a0214c0f62ff6a5c7d6` |
| `semantic_state_contracts` | `endomorphosis/ipfs_datasets_py` | `1330038f626ef92993f03d46f21e1a57719e9c25` | `c1686dfce8e14ebd32327a0214c0f62ff6a5c7d6` |
| `kit_state_roots` | `endomorphosis/ipfs_kit_py` | `df2f9cc092456329de9724c45a50c54b410875d1` | `d3f2d9ae8b1cbf0145c7d54114a5408d90b49fd0` |
| `mcp_plus_plus` | `endomorphosis/Mcp-Plus-Plus` | `dc3164653a48d059ae9812078359daeafb451c07` | `6560c3d0c926be12df860afb7d7c82043a1769ba` |

Toolchain projection bound by the seal: Python **3.12**, pytest, closed environment
with auto-install disabled. Policy name: `exact_clean_head` (local object identity
only; no remote-ref reachability claim).

### 1.2 Implementation tree (this workspace)

| Field | Value |
|---|---|
| Worktree `HEAD` at report generation | `e53cae58384f182a2a80c25503b125b0f3032252` |
| Worktree tree | `6445e97630f140337e7c5b910b5259f83458b70a` |
| Objective id | `SCH-018` |
| Objective revision | `baguqeerarclol5emeyw3ek4puqipo4ll24jjhe7uci4ymotmxbcxuwlhvkqa` |

### 1.3 Benchmark receipts (SCH-017)

| Field | Value |
|---|---|
| Report path (JSON) | `docs/benchmarks/semantic_compression_harness_results.json` |
| Report path (Markdown) | `docs/benchmarks/semantic_compression_harness_results.md` |
| Interface | `SemanticStateBenchmark@1` |
| Schema | `ipfs_accelerate_py/semantic-state/benchmark-report@1` |
| Bundle | `sch/benchmark@1` |
| Corpus | `semantic-state-benchmark-corpus-v1` (40 tasks) |
| Fixture corpus | `semantic-state-controlled-repo-v1` |
| Tokenizer / estimator | `sch-fixture/token-estimator@1` / `semantic-state-token-estimator-v1` |
| Deterministic digest (timing stripped) | `sha256:15bddb87fcf7af223caaf43f579bbc6e38342356ec6d7ec7acc2c4f541823dd6` |
| Content digest | `sha256:17809670c11261338e3d0db01959f1717c51c49879284063ee50d2067d970805` |

Replay check:

```bash
python3.12 benchmarks/semantic_state/run_benchmark.py --check
```

### 1.4 Root manifests and wire receipts

- Root manifest schema: `ipfs-accelerate.semantic-state-root-manifest@1`
  (`SemanticStateRootManifest` in `contracts.py`).
- Root CAS token: `RootRef` (root CID + monotonic generation).
- Verification receipts: `SemanticVerificationReceipt@1` /
  `ipfs-accelerate.semantic-verification-receipt@1` (MCP++ Profile B payloads).
- Wire boundary: MCP++ Profiles **A / B / F** via
  `canonicalize_artifact` + Kubo-compatible `cid_for_bytes` (real CIDv1).
- Closed interface resource:
  `ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/semantic-state-harness.interface.json`
  (`semantic-state-harness` / namespace `ipfs-accelerate.agent-supervisor`).

Production promotion never treats operational/scheduler receipts as correctness
proofs. Stale, incomplete, simulated (`sim:` / `degraded:`), OFF/SIMULATED/DEGRADED,
fallback, or unadmitted-replay evidence is non-authoritative and cannot advance a
production root.

---

## 2. Architecture

The harness is a **focused local coding loop**, not a second agent framework, MCP
server, dashboard, prover, or multi-language platform. It consumes datasets-owned
semantic state, projects admitted capsules into existing context/compiler surfaces,
routes model work through injected providers gated for production, validates and
applies patches in fenced worktrees, executes sealed test/proof selections, emits
MCP++ receipts, and CAS-promotes generation-bearing state roots through kit.

```text
Git/tree snapshot
      |
      v
SemanticStateProvider (datasets) ------------------+
  state + Merkle DAG + capsules + delta            |
  + invalidation + exact tree-bound source         |
      |                                             |
      +--> CapsuleAdmission --> DatasetsSelection  |
      |           |                 |               |
      +-----------+--> ContextPacker                |
                          |                          |
                    ModelRouting                     |
                          |                          |
                 SchedulingAdapter                   |
                          |                          |
             Fenced disposable worktree              |
                          |                          |
              patch -> rescan -> verification        |
                          |                          |
                   MCP++ receipts/event              |
                          |                          |
                DurableSemanticStatePort <-----------+
                          |
              expected-old CAS state root
```

### 2.1 Authority reuse (no second sources of truth)

| Concern | Authority | Harness decision |
|---|---|---|
| Symbols, graph, capsules, selection, source | pinned `ipfs_datasets_py` semantic-state APIs | Adapt only; never re-derive AST/graph facts |
| Durable root CAS / WAL | pinned `ipfs_kit_py` generation-bearing port | Narrow protocol: put/get/has/read_root/CAS/recover |
| Wire envelopes + CIDv1 | MCP++ Profile A/B/F + accelerate Kubo helpers | Local payload schemas only |
| Context budgeting | existing `ContextCompiler` / production context slice | Project capsules; no second optimizer |
| Resources / cancellation | `ResourceScheduler` | Wrap, do not replace |
| Provider execution | `ProviderExecutionGateway` | Single gateway path + stricter promotion gate |
| Leases / worktrees | `LeaseCoordinator` / `WorktreeLifecycleStore` | Fence every mutation |
| Validation / proofs | `ValidationScheduler` / `ProofScheduler` | Explicit sealed commands; no reselection |

### 2.2 Fourteen-step harness loop

`SemanticCompressionHarness@1` (`harness.py`) composes:

1. acquire worktree  
2. materialize context pack  
3. invoke model (or halt)  
4. validate proposal  
5. enforce scope  
6. apply patch  
7. rescan changed symbols  
8. recompute delta/invalidation  
9. run static checks  
10. run selected tests  
11. run proofs  
12. optional oracle  
13. store artifacts and manifest  
14. compare-and-swap root  

Invariants include: rejection leaves root unchanged; acceptance requires fresh
non-simulated receipts; production requires a real provider when a model is needed;
human review never invokes or publishes; bootstrap is indexed not verified.

### 2.3 Production gate (non-negotiable)

Production requires ENFORCE mode, AVAILABLE coordination, real coordinator and
invoker, verified attribution, matching provider identity, and a non-simulated
reservation. Rejected paths (nonzero / never verified / never root-committed):

- `sim:` / `degraded:` reservation identities  
- OFF / SIMULATED / DEGRADED / denied / cancelled / failed phases  
- non-ENFORCE modes (off, observe, shadow, assist)  
- fallback reason codes (local/cross-provider fallback, degraded, simulated, …)  
- unadmitted replay  
- missing coordinator, invoker, attribution, or provider  
- development simulation labels  

Any `llm_router.generate_text` adapter call forces
`allow_local_fallback=False` and `allow_cross_provider_fallback=False` and checks
the effective provider.

---

## 3. Packages and modules

Owning package:

```text
ipfs_accelerate_py/agent_supervisor/semantic_state/
```

| Module | Interface / role |
|---|---|
| `__init__.py` | Public re-exports; cold-import safe |
| `contracts.py` | Closed deterministic records (`HarnessMode`, `ContextPack`, `RootRef`, `SemanticStateRootManifest`, …); schema `semantic-state-harness@1` |
| `wire.py` | MCP++ A/B/F codec; `semantic-state-harness` interface descriptor |
| `datasets_adapter.py` | `SemanticStateProvider@1` over pinned datasets APIs |
| `durable_state.py` | Narrow kit durable-root port |
| `scheduling.py` / `scheduling_contracts.py` | `SemanticSchedulingAdapter@1` / `SemanticWorkScheduling@1` |
| `capsules.py` | `SemanticCapsuleAdmission@1` |
| `context_pack.py` | `ContextPack@1` compilation over admitted capsules + raw source |
| `routing.py` | `ModelRouting@1` (deterministic route decision) |
| `providers.py` | `ModelProvider@1` + `ProductionProviderGate` |
| `selection_execution.py` | `SelectionExecutionAdapter@1` (no graph reselection) |
| `verification.py` | `SemanticVerification@1` |
| `receipts.py` | `SemanticVerificationReceipt@1` + freshness admission |
| `worktree.py` | `IsolatedPatchWorktree@1` |
| `harness.py` | `SemanticCompressionHarness@1` 14-step loop |
| `session.py` | `SemanticStateSession@1` incremental sessions / watch |
| `cli.py` | `SemanticStateCLI@1` console entrypoint |
| `benchmark.py` | `SemanticStateBenchmark@1` corpus runner |
| `schemas/semantic-state-harness.interface.json` | Packaged closed interface schema |

Tests: `test/api/semantic_state/`  
Controlled fixture repo: `test/fixtures/semantic_state_harness/controlled_repo/`  
Benchmark corpus: `benchmarks/semantic_state/tasks/` (exactly 40 tasks)

Console entry (also packaged in wheels):

```text
semantic-state = ipfs_accelerate_py.agent_supervisor.semantic_state.cli:main
```

---

## 4. Commands and examples

Deterministic JSON by default. Unavailable optional dependencies return a typed
error envelope and a **nonzero** exit code. Production `apply-patch` never falls
back to simulation. Imports and `--help` start no watchers, processes, databases,
network clients, or package installers.

```text
semantic-state scan <repo>
semantic-state watch <repo>
semantic-state status <repo>
semantic-state graph <repo> [--symbol ID]
semantic-state explain-symbol <repo> <symbol>
semantic-state explain-impact <repo> <symbol-or-file>...
semantic-state invalidate <old-state> <new-state>
semantic-state select-tests <repo> <symbol-or-file>...
semantic-state pack-context <repo> <task> <target>
semantic-state verify <repo> [--full-suite]
semantic-state apply-patch <repo> <patch-or-task>
semantic-state compare-full-suite <fixture-or-repo>
semantic-state benchmark [--corpus PATH]
semantic-state interface-schema
```

### 4.1 Library example (observational / development)

```python
from ipfs_accelerate_py.agent_supervisor.semantic_state import (
    HarnessMode,
    SemanticCompressionHarness,
    harness_loop_descriptor,
)

print(harness_loop_descriptor()["interface"])  # SemanticCompressionHarness@1
# Construct with injected durable port + providers; production mode requires a
# real provider path and never admits simulated gateway results.
```

### 4.2 Provider gate example

```python
from ipfs_accelerate_py.agent_supervisor.semantic_state.providers import (
    ProductionProviderGate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessMode

gate = ProductionProviderGate(
    expected_provider_id="provider-alpha",
    coordinator_present=True,
    invoker_present=True,
    admitted_production_receipt_ids=("receipt:prod-1",),
)
# Evaluate a ProviderExecutionGateway result; sim:/degraded:/OFF/fallback reject.
```

### 4.3 Benchmark check

```bash
python3.12 benchmarks/semantic_state/run_benchmark.py --check
```

---

## 5. Tests and results

### 5.1 Focused semantic suite

Primary validation command (Python 3.12):

```bash
python3.12 -m pytest -q \
  test/api/semantic_state \
  test/api/test_agent_supervisor_context_compiler.py \
  test/api/test_agent_supervisor_production_context_slice.py \
  test/api/test_agent_supervisor_provider_execution.py \
  test/api/test_agent_supervisor_resource_scheduler.py \
  test/api/test_agent_supervisor_lease_coordination.py \
  test/api/test_agent_supervisor_worktree_lifecycle.py \
  test/api/test_agent_supervisor_proposal_validation.py \
  test/api/test_agent_supervisor_validation_scheduler.py \
  test/api/test_agent_supervisor_proof_scheduler.py \
  test/api/test_agent_supervisor_hermetic_validation.py \
  && python3.12 benchmarks/semantic_state/run_benchmark.py --check
```

Semantic-state coverage areas include: contracts/wire, datasets adapter, durable
CAS, capsules, context pack, routing, providers, scheduling, selection execution,
verification, receipts, worktree, harness loop, session/watch, CLI, production
gates, acceptance matrix, concurrency/recovery, fixture repository, wheel install,
import safety, and provider regressions.

Named existing supervisor regressions required by SCH-018: ContextCompiler,
production context slice, provider execution, resource scheduler, lease
coordination, worktree lifecycle, proposal validation, validation scheduler, proof
scheduler, and hermetic validation.

### 5.2 Import safety (SCH-018)

`test/api/semantic_state/test_import_safety.py` proves:

- ordinary package and module imports are side-effect free (no installer, network,
  process spawn, thread start, database open, environment mutation, or cwd file
  creation);
- static AST scan of the owning package rejects legacy mock hardware / mock
  inference import surfaces.

### 5.3 Provider regressions (SCH-018)

`test/api/semantic_state/test_provider_regressions.py` proves real, absent,
default-simulated, degraded, off, replayed, and fallback dispositions: production
unavailable/simulation/fallback paths are **nonzero** and **never**
`can_verify` / `can_commit`.

### 5.4 Acceptance matrix highlights (SCH-015 / controlled fixtures)

Bounded invalidation, raw-source fallback for opaque behavior, stale-receipt
rejection, zero controlled-fixture selection false negatives, full-suite fallback,
deterministic roots, safe recovery, single-winner generation CAS, and strict
production simulation rejection.

---

## 6. Benchmark and task-type token reductions

Source: committed `docs/benchmarks/semantic_compression_harness_results.json`
(deterministic digest `sha256:15bddb87fcf7af223caaf43f579bbc6e38342356ec6d7ec7acc2c4f541823dd6`).

| Gate | Result |
|---|---|
| `task_count_is_40` | PASS |
| `median_reduction_at_least_30_percent` | PASS (median **58.90%**) |
| `zero_controlled_false_negatives` | PASS |
| `zero_coverage_omissions` | PASS |
| `zero_stale_admissions` | PASS |
| `zero_simulated_admissions` | PASS |
| `all_production_eligible_false` | PASS |
| `no_model_receipts` | PASS |
| `no_production_root_advanced` | PASS |

Overall context reduction (same tokenizer for raw and semantic modes):

| Metric | Value |
|---|---:|
| Median reduction | **58.90%** |
| Mean reduction | **52.28%** |
| Range | 4.68% … 68.43% |

Reduction by task type:

| Category | Count | Median | Mean |
|---|---:|---:|---:|
| `api_adapter` | 6 | 58.46% | 57.37% |
| `multi_file_refactor` | 6 | 52.81% | 48.79% |
| `rejection_or_escalation` | 6 | 47.09% | 41.55% |
| `schema_migration` | 6 | 58.14% | 57.47% |
| `small_bug_fix` | 10 | 59.08% | 49.34% |
| `test_repair` | 6 | 61.96% | 61.12% |

Checked-in candidate patches are **oracle/replay fixtures only**
(`production_eligible=false`). They never produce a model receipt and never advance
a production root. Failed and escalated tasks remain in the denominator. Wall-clock
latency is observational and excluded from `--check` equality.

---

## 7. Test-selection precision and recall

| Metric | Value |
|---|---:|
| Overall precision | **36.22%** (3622 bp) |
| Overall recall | **100.00%** (10000 bp) |
| Controlled false negatives | **0** |
| False positives (extras kept visible) | 81 |
| Coverage omissions | 0 |
| Stale admissions | 0 |
| Simulated admissions | 0 |
| Production-eligible true rows | 0 |

Fallback distribution (producer directive retained; accelerate does not reselect):

| Fallback | Count |
|---|---:|
| `none` | 31 |
| `full_pytest` | 8 |
| `both` | 1 |

Route distribution (measured on the corpus; not a production provider claim):

| Route | Count |
|---|---:|
| `deterministic_only` | 10 |
| `medium_model` | 16 |
| `human_review_required` | 13 |
| `frontier_model` | 1 |

Candidate verification outcomes: pass 34 / reject 4 / escalate 2.  
Production acceptance for oracle/replay rows: not_applicable 34 / rejected 4 /
blocked 2 (never accepted).

---

## 8. Known unsoundness and opaque cases

These limitations are intentional and fail closed. They lower confidence or force
raw source / full verification; they never become exact claims.

1. **Python reflection and dynamic dispatch** — monkey patches, dynamic imports,
   and runtime-bound names may be opaque; required raw code is retained.
2. **Static call / test selection is bounded, not complete** — precision is
   intentionally conservative; recall on controlled fixtures is 100%.
3. **Pytest plugin / config / dependency fallback** — producer may force
   `full_pytest` / `full_proofs` / `both`; accelerate must not weaken that.
4. **Proof tools may be unavailable** — typed unavailable obligation/receipt;
   never reported as a passed proof.
5. **Token estimates depend on a declared estimator** — corpus uses
   `sch-fixture/token-estimator@1`; alternate estimators are not interchangeable
   without re-measuring digests.
6. **External network, filesystem, and native effects** cannot be proven absent
   by the analyzer; high-risk / security / side-effect cases escalate or retain
   raw context.
7. **Heuristic / stale capsules** are not substituted as exact facts; raw source
   is retrieved when confidence is insufficient.
8. **No ZK aggregation** is implemented (see §10).

---

## 9. Performance bottlenecks

Observed and structural bottlenecks (honest, not optimized away in this MVP):

| Bottleneck | Notes |
|---|---|
| Full-suite fallback | 8+ corpus tasks force broader pytest/proof work; dominates end-to-end wall time when selected. |
| Opaque / dynamic surfaces | Low reduction (≈5%) when nearly all raw code must stay in the pack. |
| Datasets scan + capsule admission | Tree-bound rescans after each patch; cold start pays full provider scan cost. |
| Worktree acquire / fence / apply | Git worktree lifecycle and lease fencing add fixed overhead per attempt. |
| Provider reservation path | Real production path pays ResourceScheduler + ProviderExecutionGateway latency; simulation is forbidden as a shortcut. |
| Proof probing | Capability probing for unavailable provers still costs scheduling rounds. |
| Token estimation | Dual-mode (raw vs semantic) estimation is cheap vs scan/test, but must stay deterministic for digests. |

Wall-clock figures in the benchmark report are **observational** and are not part
of the deterministic gate.

---

## 10. Exact work remaining before ZK and production integration

This release does **not** claim complete Python analysis, universal verification,
ZK support, guaranteed provider availability, or production readiness.

### 10.1 Before production integration

1. Operator-bound clean worktrees for all five sealed authority roles with
   `exact_clean_head` validation on the target deployment hosts.
2. Real (non-injected) provider fleet registration under ENFORCE with continuous
   coordination health; typed unavailable must remain nonzero and non-verifying.
3. Production context-slice and proposal-admission budgets enforced at the
   supervisor daemon boundary for live tasks (already reused, still operator-gated).
4. Runbook for recovery: root CAS conflict, lease loss, interrupted apply-patch,
   stale receipt inventory, and watcher fence replay.
5. Explicit production policy: no auto-install on import; no silent local/cross
   provider fallback; no simulation admission.
6. Capacity and SLO measurement on real repositories beyond the controlled
   40-task corpus (the corpus is not a portfolio-wide readiness proof).
7. Independent security review of patch apply, worktree isolation, and receipt
   admission before enabling mutation-authoritative modes.

### 10.2 Before ZK aggregation

No ZK circuit or aggregator is present. Future work requires, at minimum:

1. Frozen receipt / root-manifest circuit and input schema.  
2. Deterministic verifier semantics aligned with MCP++ Profile B admission.  
3. Proof-system and trusted-setup (or transparent alternative) selection.  
4. Aggregation rules over selected/full verification receipts and root CAS steps.  
5. Key lifecycle, rotation, and revocation policy.  
6. Independent cryptographic security review.  
7. Explicit non-goals retained: ZK must not launder simulated/stale evidence or
   replace producer selection authority.

---

## 11. Explicit non-claims

- Complete or sound whole-program Python analysis.  
- Universal verification of all dependents.  
- ZK proofs or a new theorem prover.  
- Always-available model providers.  
- Production readiness or continuous portfolio scanning.  
- Mock hardware / mock inference as a production fallback.  
- Auto-install or network access on import.

---

## 12. Validation evidence checklist

| Evidence | Path / command | Role |
|---|---|---|
| Dependency seal | `config/semantic_state_dependencies.seal.json` | Five-role pins |
| Benchmark JSON | `docs/benchmarks/semantic_compression_harness_results.json` | Measured reductions / gates |
| Benchmark MD | `docs/benchmarks/semantic_compression_harness_results.md` | Human-readable twin |
| Import safety | `test/api/semantic_state/test_import_safety.py` | Cold import + static mock ban |
| Provider regressions | `test/api/semantic_state/test_provider_regressions.py` | Production gate dispositions |
| Release report | `docs/semantic_state/SEMANTIC_COMPRESSION_HARNESS.md` | This document |
| Focused suite | SCH-018 validation command in §5.1 | ContextCompiler … hermetic |

---

*Generated for SCH-018 (`SemanticStateHarnessRelease@1`). Document measured
receipts exactly; do not treat this file as authority over sealed commits or
benchmark digests—the seal JSON and benchmark JSON remain the content-addressed
sources of truth.*
