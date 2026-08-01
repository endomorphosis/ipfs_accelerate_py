# Tactician-Hammer Logic Repair — Operator Guide

This guide covers **operations, metrics, feature flags, transaction health,
and rollback** for the agent-supervisor **Tactician-Hammer logic repair**
program (`LPR-G060`, board namespace
`agent-supervisor-tactician-hammer-logic-repair-v1`).

Normative architecture lives in
[`AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md`](../architecture/AGENT_SUPERVISOR_TACTICIAN_HAMMER_LOGIC_REPAIR_PLAN.md).
The objective heap and taskboard are machine-ingestible sources of intent;
this document is the human operator surface for logic-repair rollout
(LPR-020).

Related: [Proof-Gated Change Propagation Guide](PROOF_GATED_CHANGE_PROPAGATION_GUIDE.md),
[Proof-Gated Contract Repair Guide](PROOF_GATED_CONTRACT_REPAIR_GUIDE.md).

## Trust boundary

**Trust is root-bound and fail-closed.**

Only exact **accelerator** and **datasets** gitlink revisions, plus graph,
index, model, translator, toolchain, policy, and proof roots may participate
in admission or completion. Discovery (package presence, solver binaries,
GraphRAG hits, vector nearest neighbours, Tactician ranking) is **not**
authority.

| Surface | May nominate? | May admit / complete? |
| --- | --- | --- |
| Vector / embedding nearest neighbours | Yes | **No** |
| Knowledge-graph / GraphRAG edges | Yes | **No** |
| Tactician plan ranking | Yes | **No** (advisory) |
| Raw countermodels / solver candidates | Yes | Only after independent validation / reconstruction |
| LLM free-text proposals | Yes | **No** without approval-gated model-edit |
| Analytical reconstructed transforms | Yes | Only under reviewed narrow-auto policy |
| Independent native Hammer reconstruction | Eligible | Still fail-closed if stale/incomplete |
| Partial SCC / open frontier | No | **Never** |

Stale, ambiguous, unsupported, incomplete, or unreconstructed evidence **fails
closed** (abstain / reject / roll back). Four isolated lanes share **one merge
queue**; partial groups cannot merge.

## Safety and memory boundaries

**Models propose; they do not admit.**

Language models may draft nominations, rankings, explanations, and edit text.
They never:

- admit a logic goal, premise corpus, or atomic plan;
- authorize a write path or transaction merge;
- complete a task or claim fixed-point success;
- override a failed obligation;
- invent behaviour without independent authority;
- promote unsupported memory-safety claims.

**Vector, test, type, and resource evidence does not prove memory safety.**

| Evidence | Role | Proves memory safety? |
| --- | --- | --- |
| Vector / embedding nearest neighbours | Recall nomination only | **No** |
| Lexical / graph / history / KG signals | Nomination and ranking features | **No** |
| Unit / integration **test** results | Regression signal | **No** |
| **Type** / schema / mypy / tsc gates | Structural compatibility | **No** |
| **Resource** bounds (`max_memory_bytes`, CPU, quotas) | Capacity limits | **No** |
| Tactician ranking / learned ranking | Advisory plan order | **No** |
| LLM free-text claims | Proposal text | **No** |
| Reconstructed formal proof over memory facets | Formal argument (when independently reconstructed) | Eligible only after reconstruction; still fail-closed if stale |

A clean compile after a partial migration is **not** transitive completion
evidence. Fixed-point validation must re-index, re-resolve, re-diff, re-prove,
and attach a current `LogicFixedPointEvidenceAttachment@1` to an existing
`PropagationCompletionReceipt@1`.

## Runtime order (reminder)

```text
task / broken trace / contract delta
  -> independent goal + premise corpus
  -> Tactician plan (advisory)
  -> Hammer lowering + native reconstruction / validated countermodel
  -> analytical transform synthesis (supported Python first)
  -> atomic plan admission (complete frontier required)
  -> checkpointed transaction in candidate worktree
  -> re-index, re-resolve, re-diff, re-prove to joint program+logic fixed point
  -> PropagationCompletionReceipt@1 + LogicFixedPointEvidenceAttachment@1
     or rollback / abstain
```

LLM routes are considered only after analytical paths abstain, and remain
approval-gated under default and narrow-auto policies.

## Feature flags and rollout stages

Feature flags are **per repository / program / policy**. The default policy is
always **shadow**. Independent flags keep **logic prediction**, **learned
ranking**, **Hammer execution**, **counterexample refinement**, **LLM router**,
and **narrow autonomous mutation** off until an explicit scoped policy elevates
them.

| Stage | Mode value | Behavior | How enabled |
| --- | --- | --- | --- |
| **Doctor / replay** | `doctor_replay` | Control-plane doctor, receipt replay; **no** write-path mutation | Always available; non-mutating |
| **Shadow** (default) | `shadow` | Record goals, proofs, and proposed plans; **no** write-path mutation | Built-in default |
| **Assist** | `assist` | Show exact obligations and admitted plans to a human | **Explicit** scoped policy |
| **Narrow auto** | `narrow_auto` | Automated mutation **only** for complete-frontier, unique, reconstructed, analytical supported-Python transforms with fixed-point readiness | Explicit scoped policy; `mutation_authorized=true`; flag `narrow_autonomous_mutation_enabled` |
| **Model edit** | `model_edit` | Approval-gated behavior-complete model proposals analyzed as overlays before mutation | Explicit scoped policy; **never** auto-mutates |

### Assist requires explicit policy

Assist is **not** a silent upgrade from shadow. Constructing
`LogicRepairRolloutPolicy(mode=assist)` without an explicit scoped policy
document and repository/program/policy scope fails closed.

### Narrow-auto limits (initial)

Narrow-auto may authorize mutation only when **all** of the following hold:

1. Mode is `narrow_auto` under an **explicit scoped** policy.
2. Transform is an analytical supported-Python kind in the initial set:
   `add_argument`, `rename_argument`, `reorder_argument`, `thread_parameter`,
   `add_import`, `add_export`, `deterministic_rename`,
   `deterministic_substitution`.
3. The admitted target is **unique**.
4. Obligations were **independently reconstructed**.
5. The impact **frontier is complete** (no open frontier).
6. The path is **analytical** (not model-authored).
7. Language/shape is **supported Python** with exact write authority.
8. Joint program+logic **fixed-point** readiness is present.

### Always approval-gated (until separate review)

The following remain **approval-gated** even under narrow-auto:

- **model-authored** / LLM-bounded / behavior-complete model edits;
- **stateful** new behaviour and complex service / support-type changes;
- **public schema / public API** rewrites;
- **dynamic**, **generated**, and **native** / FFI frontiers;
- **cross-root** / cross-repository mutations;
- **new-dependency** / new external dependency introduction.

## Transaction and recovery boundaries

### Transactions

Each admitted plan executes only inside an isolated checkpointed candidate
worktree:

- steps and SCC groups produce content-addressed receipts;
- **partial groups cannot merge**;
- lease, tree, plan, and step identities are sealed;
- any failed group rolls back to the prior checkpoint.

### Recovery

Recovery rebuilds indexes and graphs, recomputes delta and closure, regenerates
corpus/goals/Tactician/Hammer receipts, discharges each original obligation
once, discovers second-order impacts, and re-proves to a fixed point. Bound
exhaustion, bypass, weakened tests/contracts, skipped tools, omitted callers,
unknown frontiers, or reconstruction/countermodel-validation loss **roll
back**.

## Isolation and supervisor launch

The protected LPR-000 bootstrap launcher
(`scripts/tactician_hammer_logic_repair_supervisor.sh`) remains the lifecycle
surface:

| Command | Behavior |
| --- | --- |
| `doctor` | Control-plane + binding + capability preflight (non-mutating) |
| `start` | Idempotent; refuses a second master when already running |
| `status` | Reports master and four-lane state |
| `restart` | Owned stop + start |
| `stop` | Identity-safe; refuses unowned live PIDs; never kills unrelated processes |

Secrets come from the environment only and **must not** appear in argv or logs.

Strict **four-lane** sharding, isolated state/worktrees, **one merge queue**,
bounded retries, and **one refill owner** (objective/codebase refill disabled)
are sealed in the scheduler configuration.

## Rollback gates

Any of the following demotes toward shadow and **revokes mutation**:

- nonzero safety floor or metric breach;
- authority root drift / stale roots;
- reconstruction or countermodel-validation loss;
- proof loss, wrong value, missed caller;
- partial plan / transaction failure;
- isolation or budget regression;
- corpus / receipt inconsistency.

## Release validator

```bash
python scripts/validate_tactician_hammer_logic_repair.py --check-all
python scripts/validate_tactician_hammer_logic_repair.py doctor --json
python scripts/validate_tactician_hammer_logic_repair.py status --json
python scripts/validate_tactician_hammer_logic_repair.py policy --json
python scripts/validate_tactician_hammer_logic_repair.py end-to-end --json
```

The validator composes the protected bootstrap board/DAG doctor with exact
two-repository bindings, import-isolation and native-execution permits,
platform resource/network isolation strength, capability health, and
benchmark floors. It does **not** mutate the completed RPR board or edit
protected planning artifacts.

## Safety floors (absolute zero)

Release requires zero rates for missed resolved callers, unreconstructed or
raw-countermodel admission, unauthorized axioms, invented behavior, wrong
value/source/placement, stale root/corpus/receipt, failed-obligation override,
LLM scope/semantic escape, partial transaction completion, and false fixed
point completion.

## End-to-end expectations

Seeded **explicit** and **ordinary-proposal** two-to-three-argument cases plus
complex support-type fixtures must update all resolved callers and reach an
existing completion receipt with a current logic fixed-point attachment, or
**abstain**. Stateful support types remain approval-required. A healthy
isolated supervisor must drain the board without dependency, provider,
protected-path, merge, or lifecycle blockage.
