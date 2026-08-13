# Semantic Compression Harness — Release Report

**Interface:** `SemanticStateHarnessRelease@1`  
**Bundle:** `sch/release@1`  
**Predicted symbol:** `release_report`  
**Board:** `semantic-compression-harness-v1`  
**Track:** release (`SCH-G050` / `SCH-018`)

This document is the product-facing completion report for the focused Python 3.12
and pytest semantic-compression coding-agent harness under
`ipfs_accelerate_py.agent_supervisor.semantic_state`. It documents architecture,
modules, commands, tests, the exactly-40-task benchmark, limitations,
bottlenecks, and remaining pre-ZK / production work. Claims are bound to
committed receipts, root-manifest contracts, and the sealed dependency pins.

It does **not** claim complete Python analysis, universal verification, ZK
support, ambient provider availability, or production readiness.

---

## 1. Authority pins and evidence roots

Exact sealed authority pins (`config/semantic_state_dependencies.seal.json`,
schema `ipfs-accelerate.agent-supervisor.semantic-state-dependency-seal@2`):

| Role | Repository | Commit | Tree |
| --- | --- | --- | --- |
| `accelerate_harness` (runtime authority) | `endomorphosis/ipfs_accelerate_py` | `271e331af802f37d759c000666282631a99f7aab` | `5859208bdab59338eab67a5cd0102c193ca6c388` |
| `incremental_semantic_index` | `endomorphosis/ipfs_datasets_py` | `1330038f626ef92993f03d46f21e1a57719e9c25` | `c1686dfce8e14ebd32327a0214c0f62ff6a5c7d6` |
| `semantic_state_contracts` | `endomorphosis/ipfs_datasets_py` | `1330038f626ef92993f03d46f21e1a57719e9c25` | `c1686dfce8e14ebd32327a0214c0f62ff6a5c7d6` |
| `kit_state_roots` | `endomorphosis/ipfs_kit_py` | `df2f9cc092456329de9724c45a50c54b410875d1` | `d3f2d9ae8b1cbf0145c7d54114a5408d90b49fd0` |
| `mcp_plus_plus` | `endomorphosis/Mcp-Plus-Plus` | `dc3164653a48d059ae9812078359daeafb451c07` | `6560c3d0c926be12df860afb7d7c82043a1769ba` |

Harness implementation evidence (this checkout):

| Identity | Value |
| --- | --- |
| Report tree at documentation write | `git rev-parse HEAD` / `HEAD^{tree}` of the implementing branch |
| Dependency seal | `config/semantic_state_dependencies.seal.json` |
| Seal validator | `scripts/validate_semantic_state_dependencies.py` |
| Control plan | `docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md` |
| Objectives | `docs/architecture/semantic_compression_harness.objectives.md` |
| Taskboard | `docs/architecture/semantic_compression_harness.todo.md` |
| Benchmark JSON receipt | `docs/benchmarks/semantic_compression_harness_results.json` |
| Benchmark human summary | `docs/benchmarks/semantic_compression_harness_results.md` |

Benchmark content digests (committed receipt; observational wall-clock excluded
from `--check`):

| Field | Value |
| --- | --- |
| Interface | `SemanticStateBenchmark@1` |
| Schema | `ipfs_accelerate_py/semantic-state/benchmark-report@1` |
| Corpus | `semantic-state-benchmark-corpus-v1` (exactly **40** tasks) |
| Fixture corpus | `semantic-state-controlled-repo-v1` |
| Tokenizer / estimator | `sch-fixture/token-estimator@1` / `semantic-state-token-estimator-v1` |
| Deterministic digest | `sha256:15bddb87fcf7af223caaf43f579bbc6e38342356ec6d7ec7acc2c4f541823dd6` |
| Content digest | `sha256:17809670c11261338e3d0db01959f1717c51c49879284063ee50d2067d970805` |

Root-manifest contract (`SemanticStateRootManifest@1`) transitively names
graph/capsule/delta/obligation/selection/receipt and environment bindings. A
published root is a generation-bearing `RootRef(root_cid, generation)` advanced
only after every gate succeeds; rejected candidates remain unreachable from the
current root.

---

## 2. Architecture

```text
Git/tree snapshot
      |
      v
SemanticStateProvider (datasets-owned state / Merkle / capsules / delta /
                       invalidation / tree-bound source)
      |
      +--> CapsuleAdmission --> DatasetsSelection
      |           |                 |
      +-----------+--> ContextPacker (ContextCompiler + ProductionContextSlice)
                          |
                    RoutingPolicy (assurance-aware)
                          |
                 SchedulingAdapter (ResourceScheduler / ProviderExecutionGateway)
                          |
             Fenced disposable worktree (WorktreeLifecycleStore + LeaseCoordinator)
                          |
              patch -> rescan -> ValidationScheduler.run_staged / ProofScheduler
                          |
                   MCP++ receipts / runtime.event_log
                          |
                DurableSemanticStatePort (kit DurableCoordinationStore + root CAS)
                          |
              expected-old generation-bearing CAS state root
```

Design invariants:

- Datasets owns scanner identity, symbol Merkle DAG, capsules, selection, and
  confidence. Accelerate never reconstructs AST/graph/capsule/selection authority.
- Wire artifacts use `canonicalize_artifact` + Kubo-compatible `cid_for_bytes`
  (never `compute_artifact_cid` pseudo-CIDs).
- `PersistentTaskQueue` is not an authority.
- Production never falls back to mock hardware, mock inference, silent
  simulation, or unadmitted replay.
- Ordinary package imports perform no I/O, install, network, process, or
  environment mutation.

---

## 3. Exact packages and modules

Owning package:

```text
ipfs_accelerate_py/agent_supervisor/semantic_state/
    __init__.py                 # public harness surface; side-effect free import
    contracts.py                # closed deterministic records
    wire.py                     # MCP++ Profile A/B/F codec + interface descriptor
    datasets_adapter.py         # lazy SemanticStateProvider over sealed datasets
    durable_state.py            # kit durable-root / CAS port
    scheduling_contracts.py     # work request/result / bindings
    scheduling.py               # existing-supervisor scheduling adapter
    capsules.py                 # capsule admission / projection
    context_pack.py             # assurance-aware ContextPack
    routing.py                  # model route policy
    providers.py                # injected providers + ProductionProviderGate
    selection_execution.py      # datasets selection -> validation/proof projection
    verification.py             # staged verification composition
    receipts.py                 # MCP++ verification receipts / admission
    worktree.py                 # isolated patch worktree + scope validation
    harness.py                  # 14-step semantic patch loop
    session.py                  # restartable watcher/session coordinator
    cli.py                      # semantic-state console entry
    benchmark.py                # exactly-40-task benchmark runner helpers
    schemas/
        semantic-state-harness.interface.json
```

Supporting surfaces (reused, not reimplemented):

| Concern | Existing authority |
| --- | --- |
| Context budgeting | `agent_supervisor.context.context_compiler.ContextCompiler`, `todo_daemon.production_context_slice` |
| Resource admission | `runtime.resource_scheduler.ResourceScheduler` |
| Provider execution | `runtime.provider_execution.ProviderExecutionGateway` |
| Worktree / lease | `worktree_lifecycle.WorktreeLifecycleStore`, `merge.lease_coordination.LeaseCoordinator` |
| Proposal admission | `validation.proposal_validation` |
| Validation / proof | `ValidationScheduler.run_staged`, `ProofScheduler` |
| Journal | `runtime.event_log` |

Tests: `test/api/semantic_state/`, focused supervisor regressions under
`test/api/test_agent_supervisor_*.py` listed in §6.  
Controlled fixture: `test/fixtures/semantic_state_harness/controlled_repo/`.  
Benchmark corpus: `benchmarks/semantic_state/tasks/` +
`benchmarks/semantic_state/run_benchmark.py`.

Console entry (packaged in `setup.py` / `pyproject.toml`):

```text
semantic-state=ipfs_accelerate_py.agent_supervisor.semantic_state.cli:main
```

---

## 4. Commands and examples

CLI interface: `SemanticStateCLI@1`. Default output is deterministic JSON.
Exit codes: `0` ok, `1` error, `2` usage, `3` unavailable, `4` production gate.

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

Examples (local, no IPFS daemon required for help/status of pure commands):

```bash
# Bounded help: no watchers, processes, network, or install
semantic-state --help
semantic-state apply-patch --help

# Deterministic JSON status against a scanned repository root
semantic-state status /path/to/controlled_repo

# Pack minimum-sufficient context for a task (assurance-aware)
semantic-state pack-context /path/to/repo "fix VALUE guard" pkg/target.py

# Production apply-patch refuses simulation/fallback (exit 4 on gate failure)
semantic-state apply-patch /path/to/repo ./candidate.patch --mode production

# Replay the sealed 40-task benchmark and check deterministic digests
python3.12 benchmarks/semantic_state/run_benchmark.py --check
```

Typed unavailability (missing optional provider/prover) returns a stable JSON
error body and a **nonzero** exit. Production `apply-patch` never falls back to
simulation.

---

## 5. End-to-end loop (14-step summary)

1. Obtain deterministic repository state (datasets scan / semantic view).
2. Open verified read-only `SemanticStateView` via injected `get_block`.
3. Admit exact/conservative capsules; retrieve raw source for heuristic/opaque/stale.
4. Consume deltas and explicit invalidation obligations.
5. Select tests/proofs via datasets `select_tests_and_proofs` (previous+current views).
6. Build `ContextPack` through existing ContextCompiler / ProductionContextSlice.
7. Route by assurance (deterministic / small / medium / frontier / human review).
8. Acquire lease + create fenced disposable worktree.
9. Request patch from injected provider (or accept explicit patch text).
10. Strict proposal/preimage/scope validation before Git apply.
11. Rescan candidate; run static + selected pytest + available provers.
12. Compile MCP++ verification receipt bound to exact inputs/environment.
13. Admit receipt (simulated/stale never verify or promote).
14. Generation-bearing CAS of `SemanticStateRootManifest` on acceptance only.

---

## 6. Tests and validation

### 6.1 Focused semantic-state suite

```bash
python3.12 -m pytest -q test/api/semantic_state
```

Coverage includes wire, datasets adapter, durable state, scheduling, capsules,
context pack, routing, providers, selection execution, verification, receipts,
worktree, harness, session, CLI, fixtures, acceptance matrix, production gates,
benchmark corpus, wheel install, concurrency/recovery, and dependency seal.

Release-owned regressions added by `SCH-018`:

| File | Role |
| --- | --- |
| `test/api/semantic_state/test_import_safety.py` | Cold import + static rejection of legacy mock hardware/inference |
| `test/api/semantic_state/test_provider_regressions.py` | Production dispositions: real / absent / simulated / degraded / off / replayed / fallback |

### 6.2 Named existing supervisor regressions

```bash
python3.12 -m pytest -q \
  test/api/test_agent_supervisor_context_compiler.py \
  test/api/test_agent_supervisor_production_context_slice.py \
  test/api/test_agent_supervisor_provider_execution.py \
  test/api/test_agent_supervisor_resource_scheduler.py \
  test/api/test_agent_supervisor_lease_coordination.py \
  test/api/test_agent_supervisor_worktree_lifecycle.py \
  test/api/test_agent_supervisor_proposal_validation.py \
  test/api/test_agent_supervisor_validation_scheduler.py \
  test/api/test_agent_supervisor_proof_scheduler.py \
  test/api/test_agent_supervisor_hermetic_validation.py
```

### 6.3 Benchmark check

```bash
python3.12 benchmarks/semantic_state/run_benchmark.py --check
```

### 6.4 Import safety

Ordinary imports of `ipfs_accelerate_py.agent_supervisor.semantic_state` and its
modules must not:

- start threads or subprocesses;
- open sockets or databases;
- mutate `os.environ` or process CWD;
- invoke package installers (`pip`, auto-install hooks);
- access the network.

Static AST audit of the harness package rejects imports of legacy mock hardware
and mock inference coordinator surfaces so production cannot silently simulate
capacity.

---

## 7. Benchmark and token reductions

Committed receipt:
`docs/benchmarks/semantic_compression_harness_results.json` /
`.md`.

### 7.1 Gates (all PASS on committed receipt)

| Gate | Result |
| --- | --- |
| `task_count_is_40` | PASS |
| `median_reduction_at_least_30_percent` | PASS |
| `zero_controlled_false_negatives` | PASS |
| `zero_coverage_omissions` | PASS |
| `zero_stale_admissions` | PASS |
| `zero_simulated_admissions` | PASS |
| `all_production_eligible_false` | PASS |
| `no_model_receipts` | PASS |
| `no_production_root_advanced` | PASS |

### 7.2 Overall context reduction

| Metric | Value |
| --- | --- |
| Median reduction | **58.90%** |
| Mean reduction | **52.28%** |
| Range | 4.68% … 68.43% |

### 7.3 Reduction by task type

| Category | Count | Median reduction | Mean reduction |
| --- | ---: | ---: | ---: |
| `api_adapter` | 6 | 58.46% | 57.37% |
| `multi_file_refactor` | 6 | 52.81% | 48.79% |
| `rejection_or_escalation` | 6 | 47.09% | 41.55% |
| `schema_migration` | 6 | 58.14% | 57.47% |
| `small_bug_fix` | 10 | 59.08% | 49.34% |
| `test_repair` | 6 | 61.96% | 61.12% |

### 7.4 Selection precision / recall

| Metric | Value |
| --- | --- |
| Overall precision | **36.22%** (3622 bp) |
| Overall recall | **100.00%** (10000 bp) |
| Controlled false negatives | **0** |
| False positives (extras kept visible) | 81 |
| Coverage omissions | **0** |
| Stale admissions | **0** |
| Simulated admissions | **0** |
| Production-eligible true rows | **0** |

Recall is complete on the controlled fixture; precision is intentionally
conservative (extras remain visible rather than omitted). Required
target/test/opaque source is never dropped to improve reduction.

### 7.5 Outcomes and routes

| Dimension | Distribution |
| --- | --- |
| Candidate verification | pass 34 / reject 4 / escalate 2 |
| Production acceptance | not_applicable 34 / rejected 4 / blocked 2 (never accepted for oracle/replay) |
| Routes | medium_model 16 / human_review_required 13 / deterministic_only 10 / frontier_model 1 |
| Fallback | none 31 / full_pytest 8 / both 1 |

Checked-in candidates are oracle/replay fixtures only (`production_eligible=false`).
They never produce a model receipt and never advance a production root. Failed
and escalated tasks remain in the denominator.

---

## 8. Known unsoundness and opaque cases

These limitations lower confidence or force raw-source / full-suite verification.
They never become exact claims:

1. **Python reflection and dynamic dispatch** — `importlib`, string-built names,
   and runtime attribute binding can be opaque to static symbol identity.
2. **Bounded static call / test selection** — selection is datasets-owned and
   conservative; it is not a complete interprocedural proof of impact.
3. **Pytest plugin / conftest behavior** — plugin registration and fixture
   discovery can require full-suite fallback (`full_pytest` / `both`).
4. **Monkey patches and generated bindings** — low reduction on tasks such as
   `sch-bench-06` / `sch-bench-07` / `sch-bench-08` when opacity forces raw source.
5. **Native / opaque extensions** — frontier or human-review routes; no claim of
   absence of side effects.
6. **Proof tools may be unavailable** — typed `UnavailableResult`, nonzero exit;
   absence is never treated as proof success.
7. **Token estimates** depend on the declared estimator
   (`sch-fixture/token-estimator@1`); they are comparative, not absolute.
8. **External network, filesystem, and native effects** cannot be proven absent
   by this analyzer.

---

## 9. Performance bottlenecks

Observed / structural bottlenecks (honest, not optimized away by weakening gates):

| Bottleneck | Effect | Mitigation in harness |
| --- | --- | --- |
| Full-suite fallback | Dominates wall-clock when selection is uncertain | Explicit fallback reason codes; keep selected path when recall-safe |
| Opaque / dynamic modules | Near-baseline token counts | Retrieve raw source; escalate route |
| Capsule invalidation cones on multi-file renames | Larger packs, more tests | Datasets previous+current selection retains delete/rename evidence |
| Provider admission / reservation fencing | Extra coordination round-trips | Reuse ProviderExecutionGateway; fail closed rather than silent fallback |
| Hermetic worktree create/destroy | Git worktree + lease cost per attempt | Disposable fenced worktrees; recover by durable attempt identity |
| Proof scheduler probes | Latency when tools absent | Typed unavailability; never invent success |

Wall-clock latencies in the benchmark receipt are **observational** and excluded
from `--check` equality.

---

## 10. Work remaining before ZK aggregation

No ZK aggregation is implemented. Future ZK work requires, at minimum:

1. Frozen receipt circuit / input schema bound to `VerificationReceipt` fields.
2. Deterministic verifier semantics and public transcript format.
3. Proof-system and trusted/setup selection (or transparent alternative) with
   documented soundness assumptions.
4. Aggregation rules over per-stage static/test/proof receipts.
5. Key lifecycle, revocation, and independent security review.
6. Explicit non-goals preserved: model text and simulated gates never enter a
   proof witness as authority.

---

## 11. Work remaining before production integration

Implementation completeness on the controlled fixture is **not** deployment
certification. Remaining production work:

1. Supported live provider adapters with credential isolation (no secrets in
   receipts or scheduler observations).
2. Sandbox hardening for worktree validation (seccomp/landlock/platform tests).
3. Resource quotas and multi-tenant admission policies beyond local hermetic runs.
4. Platform-specific worktree/process-fence tests (Linux/macOS/Windows where
   claimed).
5. Release signing and artifact provenance for wheels and seal receipts.
6. Migration / version policy for root-manifest and interface schema evolution.
7. Operational rollback/recovery runbook for interrupted CAS and lease loss.
8. Continuous seal revalidation against the five clean authority roots.
9. Explicit operator refusal of any silent simulation, mock hardware, or mock
   inference path in production policy.

Production dispositions that **must** remain nonzero and never verified:

| Disposition | Production verify? | Production commit? |
| --- | --- | --- |
| Real ENFORCE + AVAILABLE + admitted | Yes (when all gates pass) | Yes |
| Absent / unavailable provider | **No** (exit ≠ 0) | **No** |
| Default development simulation | **No** | **No** |
| `sim:` / `degraded:` reservation | **No** | **No** |
| OFF / SIMULATED / DEGRADED mode or phase | **No** | **No** |
| Fallback reason codes | **No** | **No** |
| Unadmitted replay | **No** | **No** |

---

## 12. Companion artifacts

| Path | Role |
| --- | --- |
| `docs/architecture/SEMANTIC_COMPRESSION_HARNESS_PLAN.md` | Reviewed design (protected) |
| `docs/architecture/semantic_compression_harness.objectives.md` | Goal heap (protected) |
| `docs/architecture/semantic_compression_harness.todo.md` | Taskboard (protected) |
| `config/semantic_state_dependencies.seal.json` | Five-role pin seal (protected) |
| `scripts/validate_semantic_state_dependencies.py` | Seal validator (protected) |
| `docs/benchmarks/semantic_compression_harness_results.json` | Benchmark machine receipt |
| `docs/benchmarks/semantic_compression_harness_results.md` | Benchmark human summary |
| `test/api/semantic_state/test_import_safety.py` | Import safety regressions |
| `test/api/semantic_state/test_provider_regressions.py` | Provider production-gate regressions |

---

## 13. Non-claims (explicit)

- Not a multi-language verification platform.
- Not a network service, dashboard, MCP server, or general agent framework.
- Does not auto-rewrite dependent source.
- Does not promote model-generated summaries to exact facts.
- Does not implement ZK proofs or a new prover.
- Does not claim production readiness solely because tests are green.
- Does not weaken a failing gate to close the board.
