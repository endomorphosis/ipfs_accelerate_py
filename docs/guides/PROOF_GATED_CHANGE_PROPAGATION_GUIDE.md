# Proof-Gated Change Propagation — Operator Guide

This guide covers **operations, metrics, feature flags, transaction health,
and rollback** for the agent-supervisor proof-gated **change-propagation**
extension (`RPR-G110` / `RPR-G220`, board namespace
`agent-supervisor-proof-gated-contract-repair-v1`).

Normative architecture lives in
[`AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md`](../architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md)
(§20–§23). The objective heap and taskboard are machine-ingestible sources of
intent; this document is the human operator surface for propagation rollout.

Related: [Proof-Gated Contract Repair Guide](PROOF_GATED_CONTRACT_REPAIR_GUIDE.md)
(legacy RPR-020 single-site repair operations).

## Trust boundary

**Trust is root-bound and fail-closed.**

Only exact **repository**, **graph**, **index**, **model**, **translator**,
**toolchain**, **policy**, and **proof** roots may participate in admission or
completion. Discovery (package presence, solver binaries, GraphRAG hits, vector
nearest neighbours) is **not** authority.

| Surface | May nominate? | May admit / complete? |
| --- | --- | --- |
| Vector / embedding nearest neighbours | Yes | **No** |
| Graph / index discovery | Yes | Only after exact root seal + reconstruction |
| LLM free-text proposals | Yes | **No** |
| Analytical reconstructed transforms | Yes | Only under reviewed narrow-auto policy |
| Independent reconstructed formal proof | Eligible | Still fail-closed if stale/incomplete |
| Partial SCC / open frontier | No | **Never** |

Stale, ambiguous, unsupported, incomplete, or unreconstructed evidence **fails
closed** (abstain / reject / roll back). Solver and model candidates are never
authoritative without independent reconstruction.

## Safety and memory boundaries

**Models propose; they do not admit.**

Language models may draft nominations, rankings, explanations, and edit text.
They never:

- admit an impact closure, value source, or atomic plan;
- authorize a write path or transaction merge;
- complete a task or claim fixed-point success;
- override a failed obligation;
- invent behaviour without independent authority;
- promote unsupported memory-safety claims.

**Vector, test, type, and resource evidence does not prove memory safety.**

| Evidence | Role | Proves memory safety? |
| --- | --- | --- |
| Vector / embedding nearest neighbours | Recall nomination only | **No** |
| Lexical / graph / history signals | Nomination and ranking features | **No** |
| Unit / integration **test** results | Regression signal | **No** |
| **Type** / schema / mypy / tsc gates | Structural compatibility | **No** |
| **Resource** bounds (`max_memory_bytes`, CPU, quotas) | Capacity limits | **No** |
| LLM free-text claims | Proposal text | **No** |
| Coverage / compile cleanliness after partial migration | Non-transitive signal | **No** |
| Reconstructed formal proof over `MemorySafetyFacet@1` | Formal argument (when independently reconstructed) | Eligible only after reconstruction; still fail-closed if stale or incomplete |

A clean compile after a partial migration is **not** transitive completion
evidence. Fixed-point validation must re-index, re-resolve, re-diff, re-prove,
and discharge every original obligation.

## Runtime order (reminder)

```text
contract / API / call-site delta
  -> impact frontier and graph/index closure
  -> consumer inventory (no missed resolved consumer)
  -> value-source retrieval and proof
  -> analytical transform synthesis (supported Python first)
  -> atomic plan admission (complete frontier required)
  -> checkpointed transaction in candidate worktree
  -> re-index, re-resolve, re-diff, re-prove to fixed point
  -> PropagationCompletionReceipt@1 or rollback
```

LLM routes are considered only after analytical paths abstain, and remain
approval-gated under default and narrow-auto policies.

## Feature flags and rollout stages

Feature flags are **per repository / program / policy**. The default policy is
always **shadow**.

| Stage | Mode value | Behavior | How enabled |
| --- | --- | --- | --- |
| **Shadow** (default) | `shadow` | Record deltas, closures, proofs, and proposed plans; **no** write-path mutation | Built-in default; no policy document required |
| **Assist** | `assist` | Show exact obligations and admitted plans to a human; implementation remains approval-gated | **Explicit** scoped policy document + repository/program/policy identity |
| **Narrow auto** | `narrow_auto` | Automated mutation **only** for complete-frontier, unique, reconstructed, analytical supported-Python transforms with exact write authority | Explicit scoped policy; `mutation_authorized=true` |
| **Expanded auto** | `expanded_auto` | Broader families only after benchmark floors, toolchain policy, rollback drills, and independent review | Explicit scoped policy + separate review; initially **disabled** |

### Assist requires explicit policy

Assist is **not** a silent upgrade from shadow. Constructing
`ChangePropagationRolloutPolicy(mode=assist)` without an explicit scoped policy
document and repository/program/policy scope fails closed.

### Narrow-auto limits (initial)

Narrow-auto may authorize mutation only when **all** of the following hold:

1. Mode is `narrow_auto` under an **explicit scoped** policy.
2. Transform is an analytical supported-Python kind in the initial set:
   `add_argument`, `rename_argument`, `reorder_argument`, `thread_parameter`,
   `add_import`, `add_export`.
3. The admitted target is **unique**.
4. Obligations were **independently reconstructed**.
5. The impact **frontier is complete** (no open frontier).
6. The path is **analytical** (not model-authored).
7. Language/shape is **supported Python** with exact write authority.

### Always approval-gated (until separate review)

The following remain **approval-gated** even under narrow-auto:

- **model-authored** / LLM-bounded edits;
- **stateful** new behaviour and complex service changes;
- **public schema / public API** rewrites;
- **dynamic**, **generated**, and **native** / FFI frontiers;
- **cross-root** / cross-repository mutations;
- adapters, constructor rewrites, serializers, fixtures, generated manifests,
  and registration/plugin wiring outside the closed analytical set.

Automation expands only with measured coverage and safety evidence for the
exact change family.

## Transaction and recovery boundaries

### Transactions

Each admitted `AtomicPropagationPlan@1` executes only inside an isolated
checkpointed candidate worktree:

- steps and SCC groups produce content-addressed receipts;
- **partial groups cannot merge**;
- lease, tree, plan, and step identities are sealed;
- any failed group rolls back to the prior checkpoint.

### Recovery

Recovery rebuilds indexes and graphs, recomputes delta and closure, discharges
each original obligation once, discovers second-order impacts, reconstructs
proofs, and runs dependency-complete policy tools/tests. Bound exhaustion,
bypass, weakened tests/contracts, skipped tools, omitted consumers, unknown
required frontier, or new unplanned delta **prevents completion**.

Operators may **doctor**, **inspect**, **replay**, **shadow**, **assist**,
narrowly automate analytical changes, and **roll back** on capability,
coverage, proof, or metric regression. Rollback receipts are evidence only;
they are never completion or mutation authority.

## Rollback gates

The runtime **demotes** toward the previous stage (and always revokes mutation)
when any of the following fire:

| Trigger | Reason code | Typical cause |
| --- | --- | --- |
| Capability regression | `capability_regression` | Required graph/index/logic/toolchain capability leaves `available` |
| Stale root | `stale_root` | Code/index/graph/model/translator/toolchain/policy root drift |
| Open frontier | `open_frontier` | Required impact frontier incomplete or unknown |
| Reconstruction failure | `reconstruction_failure` | Proof cannot be independently reconstructed |
| Proof loss | `proof_loss` | Admitted proof no longer reconstructs after edit |
| Wrong value | `wrong_value` | Wrong or unproved value-source admission |
| Missed consumer | `missed_consumer` | Missed resolved impacted consumer |
| Partial plan | `partial_plan` | Partial SCC group or incomplete plan completion |
| False completion | `false_completion` | False fixed-point / false completion claim |
| Metric / floor breach | `metric_breach` | Any release safety floor ≠ 0 |
| Coverage loss | `coverage_loss` | Graph/index coverage regression |
| Explicit operator | `explicit_operator` | Manual demotion |

## Release safety floors

These rates must remain **absolute zero** on the adversarial corpus (legacy +
propagation):

**Legacy**

- wrong-path automated mutation rate
- failed-obligation override rate
- stale / forged / poisoned authoritative admission rate
- unsupported memory-safety claim promotion rate

**Propagation**

- missed resolved impacted-consumer rate
- unproved or wrong value-source admission rate
- invented-behaviour-without-authority rate
- partial propagation completion rate
- stale graph/index plan-admission rate
- fixed-point false-completion rate

## Operator metrics

`ChangePropagationMetrics@1` projects `ChangePropagationBenchmarkMetrics@1`
and exposes:

| Metric | Meaning |
| --- | --- |
| `stage_counts` / `benchmark_stages` | Every stage: delta, graph_closure, consumer_inventory, value_retrieval, proof, plan_admission, implementation, transaction, fixed_point |
| `analytical_coverage` / `model_rate` / `llm_rate` | Analytical vs model split |
| `analytical_model_split` | Compact analytical/model projection |
| `tokens` / `context_bytes` | Provider token and context byte counters |
| `fixed_point_iterations` / `fixed_point_iterations_total` | Fixed-point iteration counts |
| `impact_recall`, `consumer_precision`, `proof_eligible_value_recall` | Closure and retrieval quality |
| `plan_completeness`, `closure_success_rate`, `completion_success_rate` | Plan and fixed-point health |
| `scc_rollback_count` | Partial-group rollbacks observed |
| `safety_floors` / `safety_absolute` | Zero-tolerance release gates |

Benchmark measurement itself is owned by RPR-045
(`scripts/benchmark_change_propagation.py`).

## Validation CLI

Primary entry point:

```bash
python scripts/validate_change_propagation.py --check-all
python scripts/validate_change_propagation.py --check-all --json
```

### Commands

| Command | Purpose |
| --- | --- |
| `doctor` / `--check-all` | Exact sources, capabilities, graph/index coverage, proof reconstruction, transaction health, supervisor state, benchmark floors, feature flags, rollback gates, guide boundaries |
| `status` | Current mode, bindings, graph/index, transaction, supervisor projection |
| `replay --receipt PATH` | Recompute receipt identity; re-evaluate auto authorization and rollback |
| `check-dag` | Plan + objective heap + task dependency DAG (requires RPR-G220 / RPR-046) |
| `check-bindings` | Exact content identities for control-plane and propagation sources |
| `check-capabilities` | Fail-closed capability probe (non-authoritative) |
| `check-graph-index` | Fixture and module graph/index coverage |
| `check-proof-reconstruction` | Proof reconstruction surfaces and reconstruction-required policy |
| `check-transaction` | Transaction module health (checkpoint / partial-group rollback) |
| `check-supervisor` | Master PID and lane supervisor/task state consistency |
| `check-benchmark-floors` | Re-run or verify absolute-zero safety floors |
| `check-flags` | Shadow default; assist/narrow-auto require scope; auto transform limits |
| `check-rollback` | Capability / stale / open-frontier / proof / zero-tolerance demotion drills |
| `check-guide` | Trust, safety, memory, transaction, recovery language |
| `metrics` | Emit `ChangePropagationMetrics` projected from the adversarial benchmark |
| `policy` | Emit the default shadow `ChangePropagationRolloutPolicy` |

Examples:

```bash
python scripts/validate_change_propagation.py doctor --json
python scripts/validate_change_propagation.py status
python scripts/validate_change_propagation.py replay --receipt /path/to/receipt.json --json
python scripts/validate_change_propagation.py metrics --json
python scripts/validate_change_propagation.py policy --json
python scripts/validate_change_propagation.py check-graph-index --json
python scripts/validate_change_propagation.py check-transaction --json
```

Useful flags:

- `--repo-root PATH` — override repository root
- `--skip-benchmark` — skip the corpus benchmark (faster doctor loops)
- `--skip-capabilities` — skip the capability probe
- `--json` — machine-readable reports

Library entry points:

- `ipfs_accelerate_py.agent_supervisor.validation.change_propagation_rollout`
  (`ChangePropagationRolloutPolicy`, `ChangePropagationMetrics`,
  `ChangePropagationRollbackGate`)

## Interfaces

| Interface | Role |
| --- | --- |
| `ChangePropagationRolloutPolicy@1` | Feature flags and auto gates |
| `ChangePropagationMetrics@1` | Operator metrics projection |
| `ChangePropagationRollbackGate@1` | Demotion receipts |
| `ChangePropagationBenchmarkMetrics@1` | Adversarial corpus aggregates (RPR-045) |
| `AtomicPropagationPlan@1` | Admitted atomic plan |
| `PropagationCompletionReceipt@1` | Fixed-point completion evidence only |

## Parallel supervisor launcher

The isolated multi-lane supervisor wrapper remains:

```bash
scripts/proof_gated_contract_repair_supervisor.sh doctor
scripts/proof_gated_contract_repair_supervisor.sh preflight
scripts/proof_gated_contract_repair_supervisor.sh start
scripts/proof_gated_contract_repair_supervisor.sh status
scripts/proof_gated_contract_repair_supervisor.sh stop
```

The launcher owns process lifecycle. The propagation validation CLI owns
**policy-level** health, metrics, flags, coverage, transaction, and rollback
checks and can run whether or not the supervisor is currently up. A **stopped**
master is a valid process state; a lane that claims `running` with a dead PID
is not.
