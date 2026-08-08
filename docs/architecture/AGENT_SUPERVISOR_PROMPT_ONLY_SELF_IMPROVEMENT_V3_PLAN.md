# Agent Supervisor Prompt-Only Self-Improvement v3 Plan

Status: proposed successor program  
Audit baseline: 2026-08-08, current `main` at `34420f615`  
Executable projections:

- `agent_supervisor_prompt_only_self_improvement_v3.objectives.md`
- `agent_supervisor_prompt_only_self_improvement_v3.todo.md`
- task prefix `## ASE3-`
- board namespace `agent-supervisor-prompt-only-self-improvement-v3`

## 1. Outcome

Make the supported product path accept a prompt and return a durable run handle:

```bash
ipfs-accelerate supervisor run "Improve the agent supervisor without weakening safety gates"
ipfs-accelerate supervisor steer RUN_ID "Prioritize the stalled-run recovery path"
ipfs-accelerate supervisor status RUN_ID
ipfs-accelerate supervisor follow RUN_ID
ipfs-accelerate supervisor doctor RUN_ID
```

```python
from ipfs_accelerate_py import Supervisor

supervisor = Supervisor.open()
run = supervisor.run("Improve the agent supervisor without weakening safety gates")
```

Equivalent MCP and MCP++ tools expose `run`, `preview`, `steer`, `status`,
`follow`, `explain`, and `doctor`. A normal `run` or `preview` caller supplies
only a prompt. The implementation derives operational arguments from trusted
local or server context, converts intent into a canonical hierarchy of goals,
subgoals, and tasks, starts conflict-free tasks in parallel, refills residual
work while objectives remain open, and keeps a progress-aware watchdog alive
until evidence-authorized completion or a bounded terminal failure.

This is a convergence program, not a rewrite of the low-level supervisor. It
composes the existing prompt planner, plan admission, objective graph, Markdown
and DuckDB task sources, scheduler, backlog refinery, Doctor, rescue planner,
lease/fence coordinator, and lifecycle services behind one supported facade.

## 2. Audit finding and why v3 is required

The v1 and v2 plans describe most of the desired architecture, but the product
contract is not available on current `main`:

- the product CLI has no `supervisor` command group;
- the package does not export a high-level `Supervisor` facade;
- MCP exposes low-level catalog operations, not prompt-only lifecycle tools;
- the current inference runtime is a prototype with heuristic profile trust and
  a receipt-only “launch”; it must be replaced by canonical resolver
  composition, not promoted as-is;
- the current launch profile emits arguments the real implementation-supervisor
  parser does not accept, and the prompt workflow fixes ready width at one even
  when the invocation budget admits more lanes;
- an empty recovered execution slice can be interpreted as the full board,
  allowing a restarted lane to select work owned by another lane;
- the decisive v2 facade, CLI, context adapter, launch guard, local profile,
  provider route, run-registry backend, runtime factory, and MCP files exist on
  `rescue/local-main-worktree-20260801`, not on current `main`;
- that rescue branch is 36 commits ahead of its merge base and 597 commits
  behind current `main`, so a blind merge or cherry-pick chain is unsafe;
- its default facade reports an in-memory run as completed and its default
  runtime effect is a no-op, so even the preserved implementation is not a
  production self-improvement path;
- the previous v2 scheduler has no live supervisor or worker, yet stale lane
  projections still say `running`; its rollout is staged but inactive;
- saved v2 projections disagree: one eligible index says all eight tasks are
  complete, the bundle index says only four are complete, and the source board
  still says all eight are `todo`;
- historical work was accepted on a branch without proving reachability from
  the current integration tree, and the drained run performed no successor
  refill.

The existing checkout also contains unrelated user work. No effectful v3 run
may start from that dirty checkout. ASE3-000 must first create a clean isolated
integration worktree, record the exact base/tree identities, preserve user
changes, and classify every rescue-branch file and commit as port, rewrite,
already-superseded, or discard.

## 3. Product contract

### 3.1 One facade, four transports

All transports call one `SupervisorIntentService` and return the same typed
result envelope. Transport adapters may authenticate and bind context, but may
not implement independent planning, authorization, provider, scheduling, or
mutation policy.

| Operation | Python | CLI | MCP / MCP++ | Required user input |
|---|---|---|---|---|
| create/run | `Supervisor.run(prompt)` | `supervisor run PROMPT` | `agent_supervisor_run` | prompt |
| preview | `Supervisor.preview(prompt)` | `supervisor preview PROMPT` | `agent_supervisor_preview` | prompt |
| steer | `run.steer(prompt)` | `supervisor steer [RUN] PROMPT` | `agent_supervisor_steer` | prompt; run only if ambiguous |
| observe | `status`, `follow` | matching commands | matching tools | run only if ambiguous |
| diagnose | `explain`, `doctor` | matching commands | matching tools | run only if ambiguous |
| bootstrap | `Supervisor.init_local()` | `supervisor init` | server policy, not remote key creation | explicit one-time consent |

Advanced overrides remain available for experts, but they are optional,
validated, and represented in the same resolution receipt. Existing `agent`
commands and low-level MCP operations remain compatible.

### 3.2 Trust boundary

The prompt is intent, never authority. It may describe desired outcomes and
constraints, but it cannot select a filesystem root, grant write/network/merge
permission, name an authenticated principal, choose a provider, expand a
budget, set arbitrary validation commands, weaken policy, or bypass a lease.

Inference may eliminate configuration ceremony; it may not invent permission.
When no valid local profile or server grant exists, `run` returns a safe preview
and one typed continuation directing the operator to `supervisor init` or an
administrator-owned target alias. That is the only intentional exception to
the one-prompt happy path.

## 4. Argument resolution

Resolution gathers evidence once into an immutable `InvocationContext`, then
applies deterministic precedence: explicit authorized override, authenticated
transport context, verified installed profile, bounded ambient observation,
safe product default. Material ambiguity stops before effects.

| Runtime input | Inferred source | Fail-closed rule |
|---|---|---|
| repository target | nearest unique enclosing Git root for local CLI; embedder allowlist for Python; server-owned alias for MCP | prompt paths, arbitrary remote paths, symlink escapes, and multiple roots do not launch |
| caller/authority | verified signed local profile, authenticated server policy, or MCP++ UCAN | credentials, environment strings, and prompt claims are not grants |
| effect ceiling | signed profile/server policy intersected with operation requirements | inference may narrow but never widen effects |
| state root | platform state directory keyed by repository and profile identity | repository files cannot redirect mutable state outside the allowed root |
| run identity | new idempotency key for `run`; unique compatible registry entry for implicit observe/steer | zero or multiple compatible runs returns a typed continuation |
| objectives/task source | admitted prompt plan or the unique compatible active program | filename similarity and stale runtime copies are not authority |
| provider route | installed policy and fresh typed provider-capacity evidence | prompt-selected models and generic failures cannot trigger fallback |
| worker count | minimum of policy ceiling, host capacity, provider capacity, and ready conflict-free lanes | never exceed a lease, resource, or budget ceiling |
| validation | repository policy/profile plus admitted task checks | model-proposed shell text is data until allowlisted and compiled |
| resume/adopt | DuckDB revision, process birth identity, lease/fence, and launch-plan equality | PID existence or a JSON status file alone is insufficient |

Every resolved field records value, source, freshness, alternatives, confidence,
policy decision, and a content identity. `explain` shows this receipt with
secrets and raw prompt content redacted.

## 5. Prompt-to-taskboard compilation

The runtime pipeline is:

```text
prompt
  -> transient prompt broker
  -> trusted invocation-context resolution
  -> preview and authority/effect intersection
  -> objective planner
  -> root goal + subgoal DAG + atomic task DAG
  -> plan lint/admission
  -> DuckDB authoritative projection
  -> bounded Markdown operator projection
  -> immutable IPLD/Parquet history receipt
  -> launch/adopt and parallel dispatch
```

Raw prompts remain transient. Persist a prompt digest, redacted summary, and
derived contracts; do not place secrets or full prompt bodies in taskboards,
argv, logs, branch names, or public immutable replicas.

Each generated goal contains a stable goal ID, parent IDs, goal dependencies,
priority, track, evidence requirements, evidence-source policy, producing task
IDs, outputs, validation, and acceptance. Each task contains the fields already
consumed by the supervisor: status, completion mode, priority, track,
dependencies, goal ID, outputs, validation, board namespace, bundle, parallel
lane, resource class, predicted files, conflict policy, preconditions, effects,
evidence subset, and acceptance.

Admission rejects duplicate IDs, unknown parents or dependencies, cycles,
unbounded scope, missing evidence ownership, unsafe validation, undeclared
effects, and a plan with no executable path to the root goal. A canonical plan
root binds the goal population, task population, policy, target tree, profile,
provider route, and validation policy.

DuckDB is the sole mutable source of run heads, task claims, CAS revisions,
leases, and fences. Markdown is a bounded human-readable projection. Parquet,
IPLD, and IPFS are immutable evidence/history and read replicas; none may claim
tasks or authorize effects.

## 6. Parallel execution

Parallelism is compiled, not requested as a bare worker count:

1. compute ready tasks from the admitted dependency DAG;
2. build a conflict graph from predicted files, scope paths, semantic resource
   locks, repository/submodule ownership, provider slots, and validation locks;
3. choose a deterministic maximal conflict-free set within inferred resource
   and budget ceilings;
4. claim each task by canonical CID under the shared DuckDB lease/fence
   coordinator;
5. run each task in an isolated worktree with a distinct attempt identity;
6. validate and review independently, then serialize accepted changes through
   a merge queue;
7. rebase/revalidate remaining workers after every accepted integration change.

The admitted `InvocationBudget.max_lanes` must propagate to ready-width and
runtime lane limits; no prompt workflow may silently hard-code width one. Lane
restarts preserve their exact plan revision and execution slice. An empty slice
means no tasks; full-board execution requires an explicit typed mode. Idle lanes
may work-steal only a newly fenced ready task from the same admitted revision.

Independent tasks must overlap in wall-clock time in the end-to-end gate.
Tasks that touch shared exports, CLI registration, package metadata, or common
fixtures are deliberately assigned to later fan-in tasks. File predictions are
rechecked against actual diffs; an undeclared overlap fences both attempts and
replans rather than racing.

The seed ASE3 board exposes up to three parallel implementation lanes in its
first and third implementation waves and two transport lanes later. Refilled
work is subject to the same dependency and conflict compiler.

## 7. Bounded goal and task refill

Refill is part of normal lifecycle, not an optional operator afterthought.
Enable the existing objective backlog refinery through one new runtime
controller and trigger it when any of these conditions holds:

- ready plus active work falls below the configured low-water mark;
- the taskboard drains while any goal or evidence requirement remains open;
- validation, review, merge, or current-tree reachability rejects a completion;
- Doctor finds a newly actionable gap or drift invalidates old evidence;
- a task exhausts attempts and its parent goal still has a viable refinement;
- a rollout or self-improvement benchmark misses an admitted threshold.

Refill first refreshes current-tree evidence and reconciles goal completion.
It derives the smallest residual gaps, deduplicates them by goal/evidence/scope
CID, and emits bounded child goals or atomic tasks with full lineage. It may
append only through an epoch/revision CAS; it never silently rewrites the
canonical board a live scheduler is reading.

Guardrails are mandatory: maximum findings per scan, maximum new work per
epoch, maximum refinement depth, cooldown, retry and provider budgets,
equivalence/deduplication, oscillation detection, and a circuit breaker.
Healthy exhaustion is allowed only when the root goal is evidence-complete.
Otherwise the run enters `refilling`, `blocked`, or `failed`, never `complete`.

When enabled by the signed profile, the controller also registers the existing
Planner Doctor and self-improvement epoch evaluator as real production hooks.
They evaluate baseline and candidate on exact isolated trees and may propose a
successor revision; they cannot change their own gates, award completion, or
promote a candidate without the ordinary validation, review, and rollout path.

## 8. Completion authority

Task completion requires all of the following on one exact run revision:

- the claimed task CID and attempt receipt match;
- declared outputs and actual diff are within scope;
- required validation and independent review receipts pass;
- the accepted commit is reachable from the current integration head;
- post-merge validation passes on that head;
- no stale lease, fence, tree, profile, policy, or provider identity exists.

The durable state progression is `proposed -> ready -> claimed -> implemented
-> merged -> validation_passed -> review_pending -> accepted`, with explicit
rejected, blocked, and quarantined exits. Only `accepted` tasks satisfy a task
or goal dependency. Recovery convenience states such as “soft completed” never
mutate authoritative completion.

The root goal may close only when every descendant goal's evidence is current,
no actionable residual remains after a final forced scan, all accepted task
commits are reachable from the release candidate, cross-transport and real
lifecycle gates pass on that exact tree, and the signed rollout decision is
active. “All Markdown tasks say completed,” “no ready tasks,” “the provider
returned success,” or “the process exited zero” is never sufficient.

If branch-local work exists but is not integrated, the supervisor creates or
reopens a convergence task. This explicitly closes the failure mode that left
v2 work only on its rescue branch.

## 9. Progress monitoring and deterministic recovery

The monitor evaluates progress, not just process liveness. It watches:

- supervisor and worker process-birth identities;
- heartbeat age and monotonic event cursor;
- DuckDB run revision and registry ownership;
- task claim, attempt, log, validation, review, and merge progress ages;
- ready/active/waiting/blocked counts and conflict reasons;
- lease/fence freshness and current integration tree;
- provider attempts, resource saturation, and refill scan outcomes;
- branch reachability and rollout state.

Default operating targets are a 5-second heartbeat, a 30-second stale-control
threshold after startup grace, and a 300-second no-progress threshold for an
active implementation unless the task declares a longer bounded operation.
Thresholds are policy values recorded in the launch plan, not prompt values.

The public health states are `starting`, `running`, `refilling`, `blocked`,
`stalled`, `recovering`, `complete`, and `failed`. Required incident responses:

| Observation | Classification | Bounded response |
|---|---|---|
| status says running, process identity is dead | stale projection | fence old owner, repair projection from registry/events, adopt or restart |
| ready tasks exist but no worker is claimed | scheduler stall | refresh conflicts/resources, restart scheduler once, then Doctor/rescue |
| active task has no event/log/revision progress | worker stall | capture diagnostics, terminate exact process tree, retry/rescue within budget |
| no ready tasks and an objective is open | false idle | force completion reconciliation and refill scan |
| tasks complete but commits are not reachable | soft completion | reopen convergence work; prohibit closeout |
| lease/fence/tree changes during an effect | stale authority | deny effect, quarantine attempt, re-resolve |
| repeated equivalent rescue/refill | oscillation | open one incident, trip circuit breaker, require operator decision |

`status` returns a bounded snapshot, `follow` streams durable events from a
cursor, `explain` reports inference and scheduling decisions, and `doctor`
returns findings plus previewable repairs. Recovery actions are idempotent,
rate-limited, receipt-bound, and never gain broader authority than the launch
plan. Lane PID/status files are projections and are repaired when contradicted
by process-birth and registry evidence.

## 10. Implementation waves

The companion taskboard is the executable source. Its dependency graph creates
these conflict-safe waves:

```text
Wave 0:            ASE3-000 current-main truth and salvage manifest
Wave 1 (parallel): ASE3-001 trusted context/inference
                   ASE3-002 provider/authority policy
                   ASE3-003 durable run state/effect guard
Wave 2:            ASE3-004 prompt-to-goal/task materializer
Wave 3 (parallel): ASE3-005 real runtime/lifecycle saga
                   ASE3-006 conflict-aware parallel scheduler
                   ASE3-007 bounded objective/task refill
Wave 3b (parallel): ASE3-006 adaptive scheduler continuation
                    ASE3-018 canonical context/resolver hardening
                    ASE3-019 signed authority/provider-attempt hardening
Wave 3c:           ASE3-021 durable production refill wiring
Wave 3d:           ASE3-020 transactional run truth and crash-safe saga
Wave 4:            ASE3-008 progress watchdog/Doctor/recovery
Wave 5:            ASE3-009 Python facade and package exports
Wave 6 (parallel): ASE3-010 prompt-first CLI
                   ASE3-011 MCP and MCP++ tools
Wave 7:            ASE3-012 cross-transport/security/chaos gates
Wave 8:            ASE3-013 self-hosted improvement canary
Wave 9:            ASE3-014 canonical materialization and staged cutover
```

Wave 3b starts only after the configured primary provider has a verified,
non-expired local authentication record. Its ready set is deliberately
three-way sharded: ASE3-006 to lane 0, ASE3-019 to lane 1, and ASE3-018 to
lane 2. After ASE3-019 is accepted, the bootstrap process tree must be fenced
and relaunched in the same namespace before ASE3-021 starts so the hardened
provider-attempt daemon code is loaded. The legacy objective and codebase
refill flags remain disabled; ASE3-021 owns the later scoped-v3 adoption and
refill transition.

ASE3-000 selectively ports or rewrites preserved v2 work. No task may claim
historical ASE/ASE2 completion based on old state files. Fresh task IDs and
content identities prevent stale v2 receipts from satisfying v3 dependencies.

## 11. Verification gates

The release candidate must prove:

- cold `--help` and package import perform no IPFS/provider/process startup;
- Python, CLI, MCP, and MCP++ compile equivalent canonical requests and results;
- one prompt produces a valid root goal, subgoals, atomic tasks, and dual
  DuckDB/Markdown projections with matching identities;
- at least two independent tasks execute concurrently, while conflicting tasks
  never overlap and all claims are fenced;
- a real supported `run` starts or adopts a real lifecycle process; no default
  in-memory-completed or no-op-effect path is reachable in production mode;
- low-water, drained-open-goal, failed-validation, and drift cases refill;
- a drained evidence-complete goal does not refill;
- stale PID, stale heartbeat, frozen worker, dead scheduler, duplicate launch,
  crash-at-boundary, lease loss, branch-only completion, and corrupt projection
  cases recover or fail closed deterministically;
- raw prompt/secret leak scans, path/authority injection, UCAN attenuation,
  provider fallback, retry, resource, and budget adversarial tests pass;
- existing expert CLI and low-level MCP compatibility suites pass;
- final evidence is produced on the exact current integration tree and the
  canary stays healthy through a sustained observation window.

Quantitative release targets: 100% public-operation parity, zero unauthorized
effects, zero duplicate task effects, zero accepted branch-unreachable commits,
zero false-complete chaos cases, at least two observed parallel workers on a
synthetic independent DAG, deterministic replay for unchanged context, and a
bounded recovery/refill decision for every injected incident.

## 12. Rollout and rollback

1. **Inventory only:** archive the old v1/v2 state as evidence, invalidate its
   liveness projections, and build the port/rewrite manifest on current `main`.
2. **Preview:** ship the common service and facade behind an opt-in profile;
   compare all transports without starting effects.
3. **Assist:** enable real isolated-worktree runs with explicit local profiles;
   keep expert entrypoints available as rollback.
4. **Self-host canary:** use the new prompt path to perform one bounded
   improvement against this package while a separate monitor observes progress,
   refill, branch convergence, and recovery.
5. **Local auto:** promote only after exact-tree conformance, chaos, load, and
   sustained canary gates pass. Remote mutation remains separately authorized.

Rollback disables the prompt-only profile, fences its active run revision,
stops only processes whose birth identities match, preserves immutable evidence
and worktrees, and restores the prior expert entrypoints. It must not delete
user work or treat cleanup as authorization to reset a checkout.

## 13. Immediate execution rule

Do not restart the old prompt-only scheduler and do not activate its staged
rollout. Begin with ASE3-000 in a new clean isolated integration worktree. Once
the v3 board is materialized and its runtime profile is validated, launch the
bootstrap waves through the existing expert supervisor, monitor them with the
v3 health contract, and use the newly completed prompt facade for ASE3-013's
self-hosted canary.
