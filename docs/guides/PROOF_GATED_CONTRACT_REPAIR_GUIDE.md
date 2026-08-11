# Proof-Gated Contract Repair — Operator Guide

This guide covers **operations, metrics, feature flags, and rollback** for the
agent-supervisor proof-gated contract-repair program (`RPR-` /
`agent-supervisor-proof-gated-contract-repair-v1`).

Normative architecture lives in
[`AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md`](../architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md).
The objective heap and taskboard are machine-ingestible sources of intent; this
document is the human operator surface.

## Model and evidence boundaries

**Models propose; they do not admit.**

Language models may draft nominations, rankings, explanations, and edit text.
They never:

- admit a repair target;
- authorize a write path;
- complete a task or merge;
- override a failed obligation;
- promote unsupported memory-safety claims.

**Vector, test, type, and resource evidence does not prove memory safety.**

The following evidence classes may *nominate*, *explain*, or *gate engineering
checks*, but they are **not** memory-safety proofs:

| Evidence | Role | Proves memory safety? |
| --- | --- | --- |
| Vector / embedding nearest neighbors | Recall nomination only | **No** |
| Lexical / graph / history signals | Nomination and ranking features | **No** |
| Unit / integration **test** results | Regression signal | **No** |
| **Type** / schema / mypy / tsc gates | Structural compatibility | **No** |
| **Resource** bounds (`max_memory_bytes`, CPU, quotas) | Capacity limits | **No** |
| LLM free-text claims | Proposal text | **No** |
| Reconstructed `ipfs_datasets_py.logic` proof over `MemorySafetyFacet@1` | Formal argument (when independently reconstructed) | Eligible only after reconstruction; still fail-closed if stale or incomplete |

`ProgramContract@1.max_memory_bytes` is a resource bound. It is **not**
ownership, lifetime, unsafe, FFI, or allocator evidence. Those claims use
`MemorySafetyFacet@1` and remain unsupported until reconstructed proof says
otherwise.

Stale, ambiguous, unsupported, incomplete, or unreconstructed evidence **fails
closed** (abstain / reject). Solver candidates are never authoritative without
independent reconstruction.

## Runtime order (reminder)

```text
broken trace
  -> authoritative sender requirement
  -> snapshot-bound candidate nomination
  -> receiver/placement contracts
  -> ipfs_datasets_py logic obligations and reconstruction
  -> hard eligibility filter
  -> deterministic rerank
  -> admitted target decision
  -> bounded repair packet
  -> implementation
  -> re-index, re-resolve, re-prove, validate
```

Vector search is a recall mechanism only. No candidate path reaches the
implementation agent unless its target decision is admitted under the exact
repository, graph, index, translator, toolchain, policy, and proof roots.

## Feature flags and rollout stages

Feature flags are **per repository / program / policy**. The default policy is
always **shadow**.

| Stage | Mode value | Behavior | How enabled |
| --- | --- | --- | --- |
| **Shadow** (default) | `shadow` | Generate candidates, proofs, rankings, and decisions; **no** prompt or write-path mutation | Built-in default; no policy document required |
| **Assist** | `assist` | Show admitted target and contract to a human; implementation remains approval-gated | Explicit scoped policy document + repository/program/policy identity |
| **Narrow auto** | `narrow_auto` | Automated mutation **only** for unique, reconstructed, supported pure renames / closed substitutions in supported Python shapes with exact write authority | Explicit scoped policy; `mutation_authorized=true`; strategy ∈ `{rename_substitution}` |
| **Expanded auto** | `expanded_auto` | Adapters or insertion only after benchmark floors, toolchain policy, rollback drills, and independent review | Explicit scoped policy + separate review; initially **disabled** |

### Auto limits (initial)

Narrow-auto may authorize mutation only when **all** of the following hold:

1. Mode is `narrow_auto` under an **explicit scoped** policy.
2. Strategy is a supported **rename / substitution** (not adapter, not new site).
3. The admitted target is **unique**.
4. Obligations were **independently reconstructed**.
5. The shape is a **supported** Python pattern with exact write authority.

Anything else stays assist-or-shadow. Expanded auto is not a silent upgrade.

## Rollback gates

The runtime **demotes** toward the previous stage (and always revokes mutation)
when any of the following fire:

| Trigger | Reason code | Typical cause |
| --- | --- | --- |
| Capability regression | `capability_regression` | Required logic/toolchain capability leaves `available` |
| Stale root | `stale_root` | Code/index/model/translator/toolchain/policy root drift |
| Reconstruction failure | `reconstruction_failure` | Solver result cannot be independently reconstructed |
| Metric / floor breach | `metric_breach` | Any release safety floor ≠ 0 or wrong-path rate > 0 |
| Elevated abstention error | `elevated_abstention_error` | Operator-configured abstention quality breach |
| Explicit operator | `explicit_operator` | Manual demotion |

Rollback receipts are content-addressed evidence. They are **not** completion
or mutation authority.

## Release safety floors

These rates must remain **absolute zero** on the adversarial corpus:

- wrong-path automated mutation rate
- failed-obligation override rate
- stale / forged / poisoned authoritative admission rate
- unsupported memory-safety claim promotion rate

Configured mirrors live in
`config/agent_supervisor_proof_gated_contract_repair_scheduler.json` under
`release_safety_floors` and `proof_policy`.

## Operator metrics

`ContractRepairMetrics` exposes (rates as integer parts-per-million):

| Metric | Meaning |
| --- | --- |
| `recall_at_k` | Nomination recall@K |
| `proof_eligible_recall_at_k` | Recall among proof-eligible candidates |
| `admitted_precision` | Admitted target precision |
| `wrong_path_rate` | Wrong-path automated mutation rate (must be 0) |
| `abstention_count` / `abstention_rate` | Abstention volume and rate |
| `proof_latency_ms` / `cache_latency_ms` | Proof and cache latency budgets |
| `cache_hit_rate` | Proof/index cache reuse |
| `tokens` / `context_bytes` | Provider token and context byte counters |
| `safety_floors` / `safety_absolute` | The four release gates |

Benchmark measurement itself is owned by RPR-019
(`scripts/benchmark_contract_repair.py`).

## Validation CLI

Primary entry point:

```bash
python scripts/validate_proof_gated_contract_repair.py --check-all
python scripts/validate_proof_gated_contract_repair.py --check-all --json
```

### Commands

| Command | Purpose |
| --- | --- |
| `doctor` / `--check-all` | Plan/objective/task DAG, exact source bindings, capability health, supervisor/process state, benchmark floors, feature flags, rollback gates, guide boundaries |
| `status` | Current mode, bindings, supervisor master/lane projection |
| `replay --receipt PATH` | Recompute receipt identity; re-evaluate auto authorization and rollback |
| `check-dag` | Plan + objective heap + task dependency DAG + scheduler proof policy |
| `check-bindings` | Exact content identities for plan, objectives, todo, scheduler, launcher |
| `check-capabilities` | Fail-closed capability probe (non-authoritative) |
| `check-supervisor` | Master PID and lane supervisor/task state consistency |
| `check-benchmark-floors` | Re-run or verify absolute-zero safety floors |
| `check-flags` | Shadow default; assist/narrow-auto require scope; auto strategy limits |
| `check-rollback` | Capability / stale / reconstruction / metric demotion drills |
| `metrics` | Emit `ContractRepairMetrics` projected from the adversarial benchmark |
| `policy` | Emit the default shadow `ContractRepairRolloutPolicy` |

Examples:

```bash
python scripts/validate_proof_gated_contract_repair.py doctor --json
python scripts/validate_proof_gated_contract_repair.py status
python scripts/validate_proof_gated_contract_repair.py replay --receipt /path/to/receipt.json --json
python scripts/validate_proof_gated_contract_repair.py metrics --json
python scripts/validate_proof_gated_contract_repair.py policy --json
```

Useful flags:

- `--repo-root PATH` — override repository root
- `--skip-benchmark` — skip the corpus benchmark (faster doctor loops)
- `--skip-capabilities` — skip the capability probe
- `--json` — machine-readable reports

## Parallel supervisor launcher

The isolated multi-lane supervisor wrapper is:

```bash
scripts/proof_gated_contract_repair_supervisor.sh doctor
scripts/proof_gated_contract_repair_supervisor.sh preflight
scripts/proof_gated_contract_repair_supervisor.sh start
scripts/proof_gated_contract_repair_supervisor.sh status
scripts/proof_gated_contract_repair_supervisor.sh stop
```

The launcher owns process lifecycle. The validation CLI owns **policy-level**
health, metrics, flags, and rollback checks and can run whether or not the
supervisor is currently up. A **stopped** master is a valid process state; a
lane that claims `running` with a dead PID is not.

Protected control-plane files (never edited by implementation tasks):

- `docs/architecture/AGENT_SUPERVISOR_PROOF_GATED_CONTRACT_REPAIR_PLAN.md`
- `docs/architecture/agent_supervisor_proof_gated_contract_repair.objectives.md`
- `docs/architecture/agent_supervisor_proof_gated_contract_repair.todo.md`
- `config/agent_supervisor_proof_gated_contract_repair_scheduler.json`
- `scripts/proof_gated_contract_repair_supervisor.sh`

## Exact source bindings

Doctor and `check-bindings` seal content identities for the plan, objective
heap, taskboard, scheduler JSON, and launcher script. Operations must reproduce
those bindings on every replay. Drift fails closed.

## Related commands

| Need | Command / module |
| --- | --- |
| Safety benchmark | `python scripts/benchmark_contract_repair.py` |
| Capability probe | `python -m ipfs_accelerate_py.agent_supervisor.integrations.contract_repair_capabilities` |
| Dependency pins | `python -m ipfs_accelerate_py.agent_supervisor.integrations.contract_repair_dependencies --print-env` |
| Post-edit validation | `ipfs_accelerate_py.agent_supervisor.validation.contract_repair_validation` |
| General supervisor ops | [`AGENT_SUPERVISOR_GUIDE.md`](AGENT_SUPERVISOR_GUIDE.md) |

## Definition of operator readiness

You are ready to leave pure documentation mode when:

1. `python scripts/validate_proof_gated_contract_repair.py --check-all` is green.
2. Default policy mode is `shadow` with `mutation_authorized=false`.
3. Benchmark safety floors are absolute zero.
4. Capability probe completes (individual backends may be unavailable; none may
   claim candidate authority).
5. Assist / narrow-auto policies exist only as **explicit scoped** documents.
6. Rollback drills for capability regression, stale roots, reconstruction
   failure, and metric breach demote and revoke mutation.
7. Operators understand that **vector / test / type / resource evidence does not
   prove memory safety**, and that models only propose.

## AST / interface symbols

| Symbol | Role |
| --- | --- |
| `ContractRepairRolloutPolicy` | Feature flags and mode admission |
| `ContractRepairMetrics` | Operator metrics projection |
| `RollbackReceipt` | Content-addressed demotion evidence |
| `ExactSourceBinding` | Sealed control-plane identities |
| `RepairTargetDecision@1` | Admitted/abstained target (upstream) |
| `ContractRepairCompletionReceipt` | Post-edit completion (upstream) |
| `BenchmarkMetrics` | Adversarial corpus aggregates (RPR-019) |
