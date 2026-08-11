# Deterministic Doctor — Operator Guide

This guide covers **rollout controls, feature flags, safety floors, kill
switch, promotion, and rollback** for the agent-supervisor **no-LLM
deterministic doctor** (`LPR-G110`, board namespace
`agent-supervisor-tactician-hammer-logic-repair-v1`).

Normative architecture lives in the protected Tactician-Hammer plan and
objective heap. This document is the human operator surface for
deterministic-doctor staged rollout (**LPR-041**).

Related: [Tactician-Hammer Logic Repair Guide](TACTICIAN_HAMMER_LOGIC_REPAIR_GUIDE.md),
[Proof-Gated Contract Repair Guide](PROOF_GATED_CONTRACT_REPAIR_GUIDE.md).

## Trust boundary

**Trust is root-bound and fail-closed.**

Only exact evidence snapshots, proof-cache revalidation, native kernel
reconstruction, complete impact closure, sandbox isolation, writer leases,
atomic SCC transactions, and joint program+logic fixed points may participate
in admission or write authority. Discovery (vector hits, knowledge-graph
edges, Tactician ranking, Hammer candidates, cache metadata, embedding
nearest neighbours) is **not** authority.

| Surface | May nominate? | May admit / write? |
| --- | --- | --- |
| Vector / embedding nearest neighbours | Yes | **No** |
| Knowledge-graph / GraphRAG edges | Yes | **No** |
| Tactician ranking | Yes | **No** (advisory) |
| Hammer candidate packages | Yes | Only after independent reconstruction |
| LLM / remote model-provider text | **No** in deterministic mode | **No** |
| Analytical reconstructed transforms | Yes | Only under reviewed narrow-auto |
| Partial SCC / open frontier | No | **Never** |

Stale, ambiguous, unsupported, incomplete, or unreconstructed evidence **fails
closed** (abstain / reject / roll back).

## Safety boundary

**Models do not run in deterministic mode.**

Language models, remote model providers, remote embeddings, and network access
remain **hard-off** under every deterministic-doctor rollout policy. The
doctor never:

- invokes an LLM router or remote model provider;
- treats embeddings / KG / vector / Tactician / Hammer candidates as semantic
  authority;
- completes a task or claims fixed-point success without independent
  reconstruction;
- overrides a failed obligation or nonzero safety floor;
- mutates sources without an admitted plan, writer lease, checkpoint, and
  compensating rollback strategy.

## Feature flags and rollout stages

Immutable config defaults to **report-only**. Independent flags keep
**narrow autonomous mutation**, **LLM router**, **LLM invocations**,
**remote model-provider calls**, **remote embeddings**, **network access**,
and **target-code import** off until an explicit scoped policy elevates them
(and model/network flags stay forbidden even then in deterministic mode).

| Stage | Mode value | Behavior | How enabled |
| --- | --- | --- | --- |
| **Report-only** (default) | `report_only` | Inspect / explain / plan receipts; **no** write-path mutation | Built-in default |
| **Plan** | `plan` | Materialize admitted plans without source writes | **Manual** promotion; explicit scoped policy |
| **Sandbox auto** | `sandbox_auto` | Sandboxed candidate execution under lease/checkpoint; no source write | Manual promotion; explicit scoped policy |
| **Narrow auto** | `narrow_auto` | Automated mutation **only** when every gate holds and mutation is authorized | Manual promotion; explicit scoped policy; `narrow_autonomous_mutation_enabled=true` |

### Promotion is manual and monotonic

Promotion never auto-elevates. Operators construct an explicit scoped policy
document with repository/program/policy scope and advance **one stage at a
time**:

```text
report_only -> plan -> sandbox_auto -> narrow_auto
```

Skip-ahead without intermediate allow flags fails closed. The independent
**kill switch** forces `report_only` and blocks any elevation until cleared.

### Narrow-auto limits

Narrow-auto may authorize mutation only when **all** of the following hold:

1. Mode is `narrow_auto` under an explicit scoped policy.
2. `enabled`, `allow_narrow_auto`, `narrow_autonomous_mutation_enabled`, and
   `mutation_authorized` are true.
3. Kill switch is **not** engaged.
4. Every safety floor is zero.
5. Unique target value placement, native reconstruction, complete impact
   closure, enforced sandbox, writer lease + checkpoint, atomic SCC
   transaction, and logic/program fixed-point readiness all hold.
6. The change class is not approval-required (TCB, stateful, public API/schema,
   dynamic/generated, native/FFI, cross-repository, new dependency, unsupported
   memory/lifetime claim).

## Hard gates (always true)

| Gate | Meaning |
| --- | --- |
| Exact evidence snapshot | Diagnosis binds only sealed snapshot identities |
| Proof-cache revalidation | Cache hits revalidate binding and preimage |
| Native reconstruction | Kernel/path reconstruction is independent |
| Complete impact / all callers | Required consumers closed before mutation |
| Enforced sandbox | Target execution stays isolated |
| Writer lease + checkpoint | Repair needs lease and checkpoint refs |
| Atomic SCC transaction | Partial groups cannot merge |
| Logic and program fixed point | Re-index, re-resolve, re-diff, re-prove |
| Compensating rollback | Every write path has a restoration plan |

## Resource limits

Bounded defaults (configurable only as positive integers of the same closed
key set):

| Limit | Default |
| --- | --- |
| `max_findings` | 256 |
| `max_candidates_per_finding` | 64 |
| `max_graph_nodes_per_query` | 2048 |
| `max_proof_routes_per_goal` | 32 |
| `max_operators_per_finding` | 32 |
| `max_plan_steps` | 256 |
| `max_fixed_point_iterations` | 8 |
| `max_changed_files` | 128 |
| `max_changed_bytes` | 1_048_576 |
| `max_processes` | 8 |
| `max_wall_time_seconds` | 3600 |
| `max_cpu_time_seconds` | 1800 |
| `max_memory_bytes` | 4_294_967_296 |

Resource bounds are **capacity limits**, not memory-safety proofs.

## Safety floors and rollback

Absolute-zero floors include missed mandatory caller, authority promotion,
stale proof/CID admission, out-of-scope sandbox write, partial transaction,
rollback failure, nondeterministic render, false fixed point, LLM router /
model-provider invocation, root/schema/capability drift, embedding canary
failure, reconstruction/isolation loss, transaction/rollback failure, and
material resource regression.

**Any nonzero floor, root/schema/capability drift, embedding canary failure,
reconstruction/isolation loss, transaction/rollback failure, or material
resource regression rolls back one stage or disables auto** (narrow auto is
always revoked). Kill switch and nonzero floors force `report_only`.

## Kill switch

Engage the kill switch to:

1. force effective mode to `report_only`;
2. disable `enabled`, `allow_*`, and `narrow_autonomous_mutation_enabled`;
3. revoke mutation authorization;
4. block further promotion until the switch is cleared under a new reviewed
   policy document.

## Lifecycle doctor

The ordinary lifecycle doctor remains **read-only and idempotent**. It may
inspect health, status, and sealed receipts. It never mutates sources, merges
work, or claims completion authority. Re-running doctor/status with the same
policy yields identity-stable decisions.

## Optional providers

Optional retrieval, prover, embedding, and model-provider modules may be
absent. Absence is **actionable** (recorded as degradation / abstention
signals) but **does not block report-only startup**. Report-only diagnosis and
operator validation must succeed with cold imports and no optional providers
loaded.

## Operator commands

Config: `config/agent_supervisor_deterministic_doctor.json`

Python control surface:

```text
ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_rollout
```

Validate (LPR-041):

```bash
python scripts/ops/agent_supervisor/validate_deterministic_doctor.py --check-all
python scripts/ops/agent_supervisor/validate_deterministic_doctor.py status --json
python scripts/ops/agent_supervisor/validate_deterministic_doctor.py policy --json
python scripts/ops/agent_supervisor/validate_deterministic_doctor.py check-flags
python scripts/ops/agent_supervisor/validate_deterministic_doctor.py check-rollback
python scripts/ops/agent_supervisor/validate_deterministic_doctor.py check-lifecycle
python scripts/ops/agent_supervisor/validate_deterministic_doctor.py check-providers
```

Thin doctor ops facade (service delegation; LPR-039):

```bash
python scripts/ops/agent_supervisor/deterministic_doctor.py status
python scripts/ops/agent_supervisor/deterministic_doctor.py inspect --mode report_only
```

## Recovery

1. **Detect** — `validate_deterministic_doctor.py doctor` or `status`.
2. **Contain** — engage kill switch; effective mode becomes report-only.
3. **Diagnose** — inspect floor breaches, drift, canary, isolation, and
   transaction receipts (read-only).
4. **Restore** — compensating rollback under an admitted plan/lease/checkpoint.
5. **Re-admit** — only after every floor is zero and gates hold; promote
   manually and monotonically under a new reviewed policy document.

## Interfaces

| Interface | Role |
| --- | --- |
| `DeterministicDoctorMode` | report_only / plan / sandbox_auto / narrow_auto |
| `DeterministicDoctorRolloutPolicy@1` | Immutable bounded policy |
| `DeterministicDoctorRolloutDecision@1` | Effective mode after gates |
| `DeterministicDoctorRollbackGate@1` | Demotion receipts |
| `DeterministicDoctorOperationsValidator@1` | Operator validation report |
| `DeterministicDoctorMetrics@1` | Benchmark floors (LPR-040) |
| `DeterministicDoctorService@1` | Control service (LPR-039) |
