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
- the transient prompt broker has no production consumer and its default
  process-local body store cannot recover a planning continuation after crash;
- prompt planning can call a provider before a durable logical-effect
  reservation exists, so retry after an unknown outcome can duplicate planning
  or produce a different goal graph;
- prompt materialization has no current-tree proof that a fresh generated,
  non-ASE3 board reaches the real configured scheduler and implementation
  daemon rather than a wrapper, fixture, or preseeded board;
- generic implementation auto-routing imports upward into entrypoints and owns
  ranking/classification behavior that belongs in `llm_router`, violating both
  the package dependency direction and the single routing-policy boundary;
- the prospective ASE3-019/023 integration exposes eleven imports from six
  runtime/todo-daemon files into five entrypoint modules; shared authority,
  execution-slice, invocation-budget, and provider-capacity records must be
  lowered into a neutral side-effect-free contracts package before routing
  policy can be consolidated without another import cycle;
- the accepted control-plane capsule does not yet include the full CID helper
  closure: `utils/cid_utils.py` imports the optional `multiformats` package,
  which is unavailable to a real `python -I` capsule even when present in the
  user site; adaptive-runtime acceptance therefore needs an in-tree canonical
  CID implementation and a Git-bound recursive capsule dependency proof;
- durable refill primitives are not joined to real scheduler events,
  active-plan invalidation, recompilation, and descendant dispatch, while the
  protected production refill flags remain disabled;
- monitor construction is not yet a detached durable lifecycle effect, so a
  client disconnect or monitor death can leave a run without autonomous stall
  detection and recovery;
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

`Supervisor.open()` resolves one installed production service registry and
returns a body-free content-addressed composition manifest. Python, CLI, MCP,
and MCP++ must report the same composition CID and bind `preview` and `run` to
the same resolver, prompt broker, planner, materializer, scheduler, refill,
monitor, and run-registry backends. In-process fake-service injection and
schema-only tool registration do not satisfy product conformance.

RUNNING is a joined evidence state, not a launch acknowledgment. The shared
composition returns it only when immutable registry history proves same-revision
lifecycle and monitor process births, leases, fences, fresh heartbeats, and
monotonic cursors. The durable monitor is guarded by the independently reviewed
host-namespace `ReviewedHostNamespaceReconciler`; no transport session or
monitor self-report may substitute for that join.

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
| provider route | `llm_router` intersects installed policy with fresh typed provider-capacity observations | prompt-selected models and generic failures cannot trigger fallback; lower layers cannot select, classify, or authorize |
| worker count | minimum of policy ceiling, host capacity, provider capacity, and ready conflict-free lanes | never exceed a lease, resource, or budget ceiling |
| validation | repository policy/profile plus admitted task checks | model-proposed shell text is data until allowlisted and compiled |
| resume/adopt | DuckDB revision, process birth identity, lease/fence, and launch-plan equality | PID existence or a JSON status file alone is insufficient |

Every resolved field records value, source, freshness, alternatives, confidence,
policy decision, and a content identity. `explain` shows this receipt with
secrets and raw prompt content redacted.

Planning policy is not another field on `SignedSupervisorProfile`. A separately
versioned, signed, content-addressed planning-policy artifact binds planning
route bounds, prompt-retention limits, replay behavior, expiry, signer
generation, and revocation. The local profile may identify the authorized
signer and maximum effect ceiling, but changing planning policy cannot mutate
the profile schema or its signed bytes.

## 5. Prompt-to-taskboard compilation

The runtime pipeline is:

```text
prompt
  -> run-bound encrypted prompt broker continuation
  -> trusted invocation-context resolution
  -> preview and authority/effect intersection
  -> separately signed planning-policy verification
  -> multiprocess durable broker lease and planning-specific effect CAS
  -> llm_router-owned planning route and terminal-output adoption
  -> typed objective planner
  -> root goal + subgoal DAG + atomic task DAG
  -> plan lint/admission
  -> DuckDB authoritative program revision CAS
  -> bounded Markdown operator projection
  -> immutable IPLD/Parquet history receipt
  -> revision-fenced generated-source observer
  -> namespace-independent generated per-run scheduler profile
  -> real configured-scheduler launch/adopt and parallel dispatch
```

Raw prompts remain confidential and short-lived, but crash recovery cannot rely
on process memory. Before planning, a multiprocess-hardened broker stores a
run/context/policy-bound encrypted intent behind an expiring, single-use,
fenced continuation lease. Its durable store uses bounded reads, private
ownership/modes, no-follow regular-file or transactional-database invariants,
interprocess exclusion, and atomic commit/recovery. A planning-specific CAS,
distinct from provider-fallback attempt accounting, reserves the logical
planning attempt before any provider effect and advances through `RESERVED`,
`EFFECT_STARTED`, `TERMINAL_OBSERVED`, and `ADMITTED`. A crash adopts the exact
terminal output and program root. If an effect may have happened but no exact
terminal receipt is recoverable, the CAS records `UNKNOWN`, denies any second
provider call, and returns `PROMPT_REPLAY_REQUIRED`. Prompt bytes and bearer
capabilities are zeroized after admitted materialization or bounded expiry.
Persist only a prompt digest, redacted summary, and derived contracts; never
place full bodies or capabilities in taskboards, argv, logs, branch names,
receipts, or public immutable replicas.

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

DuckDB is the sole mutable source of planning reservations, authoritative
program/run heads, task claims, CAS revisions, leases, and fences. The program
revision is transactionally committed before Markdown and immutable
projections and is adopted without reinvoking the planner. A revision-fenced
generated-source observer exposes that authoritative source directly to the
configured runtime; generated Markdown need not be Git-tracked, clean, or even
present for execution. Markdown is a bounded human-readable projection.
Parquet, IPLD, and IPFS are immutable evidence/history and read replicas; none
may claim tasks or authorize effects. Embedded task CIDs and subgoal ownership
survive storage, observation, claim, and receipts without being recomputed from
a projection. `GeneratedBoardRuntimeProfile` derives its namespace from the
admitted revision and accepts arbitrary valid namespaces; it cannot encode an
ASE3 seed assumption. The generated program must then drive genuine configured
scheduler, implementation-supervisor, and implementation-daemon subprocesses;
a wrapper, monkeypatch/in-process callback, fake claim store, Git-tracked-board
requirement, hard-coded namespace, or preseeded taskboard is not a production
materialization proof.

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
first and third implementation waves and two transport lanes later. A fresh
prompt-generated board must receive the same adaptive plan, exact task-ID/CID
slices, claims, worktrees, fences, validation, and merge behavior without an
ASE3-specific prefix or seed-board fallback. Refilled work is subject to the
same dependency and conflict compiler.

## 7. Bounded goal and task refill

Refill is part of normal lifecycle, not an optional operator afterthought.
Compose the existing objective backlog refinery through a durable production
event adapter and trigger it when any of these conditions holds:

- ready plus active work falls below the configured low-water mark;
- the taskboard drains while any goal or evidence requirement remains open;
- validation, review, merge, or current-tree reachability rejects a completion;
- Doctor finds a newly actionable gap or drift invalidates old evidence;
- a task exhausts attempts and its parent goal still has a viable refinement;
- a rollout or self-improvement benchmark misses an admitted threshold.

Refill first refreshes current-tree evidence and reconciles goal completion.
It derives the smallest residual gaps, deduplicates them by goal/evidence/scope
CID, and emits bounded child goals or atomic tasks with the same canonical
schema as the initial generated program. It may append only through an
epoch/revision CAS and one durable refill saga cursor. The exact progression is
`EVALUATING -> APPEND_RESERVED -> APPENDED -> PLAN_INVALIDATED -> RECOMPILED
-> DISPATCHED`, with `ADOPTED` as the terminal recovery alternative when the
same append, plan, or dispatch already won. Each transition binds its
predecessor, current tree, program/plan roots, reservation, and a phase-specific
monitor deadline before work begins. A crash resumes the first incomplete
phase or adopts the identical winner; it never skips a phase or blindly
replays an uncertain effect. It never silently rewrites the canonical board a
live scheduler is reading.

The sealed compact cursor name is
`EVALUATING→APPEND_RESERVED→APPENDED→PLAN_INVALIDATED→RECOMPILED→DISPATCHED/ADOPTED`;
all transports, the registry, the configured scheduler, and the monitor use
that same sequence and phase identity.

Guardrails are mandatory: maximum findings per scan, maximum new work per
epoch, maximum refinement depth, cooldown, retry and provider budgets,
equivalence/deduplication, oscillation detection, and a circuit breaker.
Healthy exhaustion is allowed only when the root goal is evidence-complete.
Otherwise the run enters `refilling`, `blocked`, or `failed`, never `complete`.

ASE3-021 lands this complete durable cursor/deadline path dormant. ASE3-008
depends directly on it so the monitor cannot ship against process-local refill
progress. Only a validated operator-owned ASE3-026 pre-effect authorization
may be consumed to enable scoped prompt-program/objective refill and reload the
runtime generation; that authorization does not prove the effect ran. Broad
legacy codebase refill remains disabled, and a separate post-activation
observation must prove actual refill dispatch or adoption.

When enabled by that signed protected profile, the controller also registers the existing
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

ASE3-020 makes DuckDB authoritative for immutable append-only run-history
vectors, monotonic lifecycle/monitor/refill cursor vectors, and monitor-ready
effect reservations. A history entry binds its predecessor and content; a
cursor cannot move backward or fork. If an external effect may have happened
but cannot be identified exactly after a crash, the reservation becomes
durable `UNKNOWN`: recovery may adopt exact observed evidence or return typed
operator action, but may not replay the effect.

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

The monitor is a detached durable lifecycle effect, not a client-owned loop or
its own survival authority. `ReviewedHostNamespaceReconciler` is the named,
independently reviewed host-namespace guardian. It consumes the exact run
registry and durable-process contracts to start or adopt one monitor generation
even when the CLI, MCP connection, Python object, lifecycle coordinator, or
monitor itself disappears. The monitor cannot attest to or restart its own
guardian.

`run` persists a monitor-ready effect reservation before any external effect.
It may return RUNNING only after one same-revision join proves both lifecycle
and monitor process births, leases, fences, fresh heartbeats, and monotonic
event cursors. Any missing, stale, cross-generation, synthetic, projection-only,
or self-attested member denies RUNNING. Monitor death has one bounded guardian
adoption/restart winner, unknown effect outcomes are adopted or become typed
operator action without replay, and a terminal receipt stops only the exact
owned monitor generation. The configured scheduler publishes semantic phase
and cursor movement; the monitor watches:

- lifecycle, monitor, supervisor, and worker process-birth identities;
- joined lifecycle/monitor lease, fence, heartbeat age, and monotonic cursors;
- DuckDB immutable history vectors, run revision, and registry ownership;
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
| active task has no semantic phase/cursor/revision progress | worker stall | capture diagnostics, terminate exact process tree, retry/rescue within budget |
| no ready tasks and an objective is open | false idle | force completion reconciliation and refill scan |
| tasks complete but commits are not reachable | soft completion | reopen convergence work; prohibit closeout |
| lease/fence/tree changes during an effect | stale authority | deny effect, quarantine attempt, re-resolve |
| provider is saturated but bounded work remains | resource wait | persist deadline/cursor, re-evaluate capacity after bounded backoff |
| monotonic source rolls back or cursor regresses | clock/cursor corruption | fail closed, preserve history, require a fresh verified clock generation |
| merge or refill phase exceeds its durable deadline | merge/refill stall | adopt an exact winner or recover once from the persisted saga phase |
| client disconnects or monitor dies | guardian continuity | keep lifecycle alive; one host-guardian adoption/restart winner |
| repeated equivalent rescue/refill | oscillation | open one incident, trip circuit breaker, require operator decision |

`status` returns a bounded snapshot, `follow` streams durable events from a
cursor, `explain` reports inference and scheduling decisions, and `doctor`
returns findings plus previewable repairs. Recovery actions are idempotent,
rate-limited, receipt-bound, and never gain broader authority than the launch
plan. Lane PID/status files are projections and are repaired when contradicted
by process-birth and registry evidence.

### 9.1 ASE3-026 protected activation authorization and observation

ASE3-026 is a two-evidence protected
transition. First, an operator-reviewed commit validates a signed
`ipfs_accelerate_py.agent_supervisor.protected-runtime-activation-authorization@1`
receipt bound to the inactive exact tree, old generation, target old+1 CAS/lease,
guardian, flags, quiescence proof, and expiry. It must state
`authorization_effect_observed: false` and cannot claim a birth, heartbeat,
cursor, refill, reload, or completion. Only after that commit validates may one
CAS/lease winner activate the exact old+1 generation. A separate
`ipfs_accelerate_py.agent_supervisor.protected-runtime-post-activation-observation@1`
receipt then joins actual lifecycle/monitor births, leases, fences, heartbeats,
cursors, and refill dispatch/adoption to that generation. Authorization alone
never proves activation and cannot make public facades selectable.

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
Wave 3b repair (parallel): ASE3-027 production resolver composition
                           ASE3-019 operator-salvaged authority/provider repair
Wave 3b identity:  ASE3-030 hermetic control-plane CID dependency closure
Wave 3b native:    ASE3-031 sealed reviewed DuckDB extension
Wave 3b database:  ASE3-032 configuration-locked DuckDB connection policy
Wave 3b adaptive:  ASE3-023 production adaptive-scheduler acceptance
Wave 3c gate:      ASE3-022 operator-owned provider-generation reload
Wave 3d:           ASE3-029 neutral shared-contract lowering and import-DAG gate
Wave 3e:           ASE3-028 single router-owned provider decision
Wave 3f:           ASE3-024 crash-safe prompt/planning transaction
Wave 3g:           ASE3-025 generated-board production runtime proof
Wave 3h:           ASE3-021 durable refill event/append/replan/dispatch wiring
Wave 3i:           ASE3-020 transactional run truth and crash-safe saga
Wave 4:            ASE3-008 autonomous progress watchdog/Doctor/recovery
Wave 4b gate:      ASE3-026 protected refill/monitor activation and reload
Wave 5:            ASE3-009 Python facade, service composition, package exports
Wave 6 (parallel): ASE3-010 prompt-first CLI
                   ASE3-011 MCP and MCP++ tools
Wave 7:            ASE3-012 black-box cross-transport/security/chaos gates
Wave 8:            ASE3-013 fresh-state prompt-generated self-host canary
Wave 9:            ASE3-014 canonical materialization and staged cutover
```

### 10.1 ASE3-031/032 protected native DuckDB acceptance chain

ASE3-031 and ASE3-032 close two separate native-state prerequisites without
changing the protected ASE3-019/022/023/027 task blocks. They use a sequential
acceptance design so every ordinary task dependency is satisfied on a strictly
earlier committed manifest phase: ASE3-030 follows accepted ASE3-019, ASE3-031
follows accepted ASE3-030, and ASE3-032 follows accepted ASE3-031. ASE3-031 owns
only the reviewed two-path implementation in
`ipfs_accelerate_py/llm_router.py` and
`test/api/test_agent_supervisor_native_dependency_pin.py`; the reviewed
candidate commit is
`25fedf091dad928dad1f83c9f81a54c2d401eabe`, which is implementation evidence
and not acceptance authority. That commit descends from and preserves the exact
reviewed ASE3-030 product delta, and its path-free inspection pin binds the exact
DuckDB distribution/engine version, extension filename, CPython cache tag and
SOABI, platform and machine, Python executable digest, payload size and digest,
ELF identity, ordered `DT_NEEDED`, and dependency content ID. Inspection is
evidence only. Sealing requires both the independently reviewed expected pin
and the exact accepted authorization ID; the resulting private executable
memfd is bound by descriptor/stat/seal facts and is the only allowed `_duckdb`
origin for the isolated preload. Ambient loader variables, source replacement,
unsealed or rebound descriptors, malformed or ambient-loader ELF structures,
second preload, and in-process retry after preload begins or fails are denied.

The distinct signed P-phase artifact is
`data/agent_supervisor/prompt_only_self_improvement_v3/convergence/native_dependency_launch_authorization.json`
with schema
`ipfs_accelerate_py.agent_supervisor.ase3-031-native-dependency-launch-authorization@1`.
It must be signed by an accepted local profile and bind the namespace, exact
source HEAD/tree, accepted ASE3-030 receipt, exact native pin/dependency ID,
Python/ABI/platform, purpose, expiry, nonce, and zero prior launch effects.
Neither inspection nor a successful local test may mint it. Its
`accepted_authorization_id` is equality-bound into the sealed launch JSON. The
authorization is pre-launch authority only: it cannot claim that preload,
process birth, a query, task acceptance, scheduler dispatch, or any other
effect occurred.

ASE3-032 starts only after the committed A031 status/manifest transition and owns exactly
`ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py`,
`ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py`,
`ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py`, and
`test/api/test_agent_supervisor_duckdb_connection_policy.py`.
`connect_duckdb_with_policy` is the sole connection-birth helper in that scope.
It supplies `autoinstall_known_extensions=false`,
`autoload_known_extensions=false`, `enable_external_access=false`, and
`allow_unsigned_extensions=false` atomically, applies only bounded `threads`
and `memory_limit` tuning, inserts `lock_configuration=true` last, then verifies
the exact five typed boolean settings and closes the connection on mismatch.
DuckDBConnection (and therefore its MergeQueue/resolver consumers), the
generated DuckDB task source, every LeaseCoordinator connection, and
coordination compaction use that helper. Before replacement, compaction seals
the complete persistent catalog inventory of schemas, tables, views,
sequences, macros, custom types, and indexes. Production compaction contains
zero `ATTACH` statements and preserves that catalog plus all live data and
metadata under the existing locked atomic-replacement semantics. No caller may
override protected settings, unlock policy, install or fetch extensions, load
dynamic/external extension bytes, retain an ambient direct connection, or
widen external access. A reviewed statically linked module may answer `LOAD`,
and DuckDB may allow `ATTACH ':memory:'`; neither is evidence that external
extension bytes or an `ATTACH`-based production compactor are allowed.

ASE3-032 is deliberately bounded to those four product/test paths; it is not a
claim that every historical raw connection in the repository is removed.
ASE3-025 must route every generated-board and planning-reachable connection,
including formal-plan compilation, through `connect_duckdb_with_policy`.
ASE3-020 must do the same for run-registry and runtime-history connections.
ASE3-012 then builds an installed-tree import/call graph plus AST inventory
from every public prompt-product launch root and rejects any launch-reachable
raw `duckdb.connect`. Any raw site that remains must be proven non-reachable
from that product and explicitly classified as legacy or proof-only; a global
source-count assertion cannot substitute for this reachability gate.

The protected transition is `Q→R(root pin)→P019(witness+provider auth@2+manifest)→A019→A030→P031(native auth+manifest)→A031→A032→A023/027→L(ASE3-022 reload authorization)`.
Each arrow is an exact parent/manifest transformation. A019 accepts the operator-salvaged
ASE3-019; A030 adds the strict ASE3-030 validator, manifest binding, receipt,
and status on a committed parent where ASE3-019 is already accepted. P031 then
binds that committed A030 receipt into the strict signed native launch
authorization validator, still-pre-effect artifact, and manifest binding. The
P031 artifact is required before any ASE3-023 capsule/native subprocess
end-to-end runtime effect used as acceptance evidence, while
`authorization_may_claim_launch_effect: false` remains exact. A031 retains
those exact P031 bytes and adds the strict validator, manifest binding, receipt,
and status for ASE3-031. Its receipt is
`data/agent_supervisor/prompt_only_self_improvement_v3/convergence/sealed_native_dependency_acceptance_receipt.json`
with schema
`ipfs_accelerate_py.agent_supervisor.ase3-031-native-dependency-acceptance@1`;
A032 has accepted A031 as its parent and adds the ASE3-032 A receipt
`data/agent_supervisor/prompt_only_self_improvement_v3/convergence/duckdb_connection_policy_acceptance_receipt.json`
with schema
`ipfs_accelerate_py.agent_supervisor.ase3-032-duckdb-connection-policy-acceptance@1`.
A023/027 then accepts the two independent repair tasks together; neither has a
same-phase dependency on the other. L follows both and owns only the ASE3-022
reload authorization. No phase may treat a status it writes as a dependency it
already read, and L cannot retroactively authorize P031 or any A phase. All
three new paths remain absent, ASE3-030/031/032 remain `todo`, and the
refill/monitor/activation flags remain dormant in this roadmap commit. ASE3-023
implementation may proceed independently, but neither its acceptance nor the
ASE3-022 transition is valid until the accepted ASE3-030, ASE3-031, and
ASE3-032 receipts are all present, strictly validated, manifest-bound, and
reachable through the exact sequential phase chain.

### 10.2 Existing repair, transition, and downstream ordering

Wave 3b starts only after the prospective operator authorization in
`provider_fallback_policy_authorization_20260808.json` validates against its
exact source HEAD/tree and this namespace. Each logical implementation attempt
still tries Grok 4.5 first. The bootstrap route accepts only a nonce-bound
typed authentication-unavailable finding, or a hard-quota finding plus
independent confirmation, from the fixed no-tools primary probe; verifies that
the pre-effect workspace fingerprint is unchanged; and dispatches exact Codex
`gpt-5.6-terra` at `high` reasoning inside the pinned external Docker boundary
at most once per runner in the same daemon attempt.
Direct authentication fallback recognizes only the exact normalized signals
`not signed in` and `not authenticated`. Bare or ambiguous `401`, `403`,
`forbidden`, or `unauthorized` signals do not directly authorize authentication
fallback; they may continue only when separately classified and independently
confirmed as hard quota. ASE3-019 may expand the authentication signal policy
only through its signed typed policy.
Before any bootstrap dispatch, the canonical implementation route plan and the
typed fallback decision must be owned and exported only by
`ipfs_accelerate_py.llm_router`. The scheduler
supplies an immutable route profile as input; the runner consumes the router's
plan and decision and owns only isolation, process effects, and the terminal
outcome; the daemon consumes that outcome and owns only task retry accounting.
Provider/model/trigger/effort tuples, authentication and quota classification,
and fallback allow/deny logic must not be independently duplicated in those
layers. The canonical plan and terminal outcome must carry the verified board
namespace, authorization-artifact SHA-256, authorization kind, source HEAD,
and source tree into daemon accounting. The ambient six-field
provider/model/trigger/effort route tuple is profile input only and cannot
authorize fallback by itself. At bootstrap, a task that explicitly requires
independent Codex review hard-denies Codex implementation fallback; no reviewer
identity is inferred or fabricated. ASE3-019 must later bind a nonempty signed
reviewer identity and provider and prove they differ from the chosen implementer.
That bounded authorization lets Wave 3b proceed; it does not claim that the
bootstrap daemon already has a durable cross-process/restart reservation or
full signed invocation/task/prompt/scope/budget/authority/provider equality.
ASE3-019 must implement and prove that durable once-only CAS, crash adoption,
fresh typed evidence, and exact signed equality before it can be accepted.
Arbitrary errors, rate limits, transport failures, invalid requests, unknown
failures, a changed workspace, or post-effect evidence do not authorize
fallback. The route neither charges nor repairs an attempt counter, including
through provider-capacity restoration. The
historical post-wave-3 residual report remains immutable evidence and is not
reclassified by this prospective policy.

The first Wave 3b run exposed two false-positive completions and one legitimate
failed validation. ASE3-006's standalone compiler and SQLite ledger were not
consumed by the production scheduler, while ASE3-018's default resolver still
accepted repository-only, caller-shaped profile, and caller-constructed UCAN
evidence. Their green tests and completed projections remain immutable evidence,
but they do not satisfy product acceptance. The manifest-bound
`false_completion_recovery_20260808.json` maps those residuals to ASE3-023 and
ASE3-027 without rewriting counters or runtime history. ASE3-019 retained its
exact task identity, charged attempt 1, and preserved failed candidate for the
authorized attempt-2 prior-attempt seed.

Attempt 2 applied that exact seed on launch HEAD
`e6f8e4a7771907372fc93b0f35cfde30170c2b2a`, then failed before either Grok or
Codex performed a provider effect. The accepted daemon invoked
`python -m ipfs_accelerate_py.agent_supervisor.grok_cli_runner` with the seeded
candidate as its current directory, so Python imported the candidate runner
and router instead of the accepted provider control plane. The candidate router
required new signed implementer/reviewer fields that the accepted scheduler
could not yet supply and returned
`agent implementation route binding fields are invalid`. The runner subprocess
was started, the attempt was therefore charged, and neither its counter nor its
queue history may be restored. The manifest-bound
`failed_pre_dispatch_event_ase3_019_attempt_2_20260808.json`,
`failed_pre_dispatch_log_ase3_019_attempt_2_20260808.txt`, and
`self_host_seed_failure_ase3_019_attempt_2_20260808.json` preserve the exact
event, log, Git/blob provenance, accounting, and fence facts.

ASE3-019 remains the same manual, canonical, todo task. Its next authorized
path is an operator-owned, no-provider reconciliation of the immutable
`eb68ff2a20e0719388f60ffef1f5bfcb90b79263` rescue delta on the then-current
accepted integration tree. The future manifest-bound operator-salvage receipt
must prove the implementation, merge, current-tree validation, and independent
bootstrap review chains and must include an `accepted_control_plane` object.
That object must bind the exact accepted generation tree, router/runner/daemon
blobs, isolated absolute runner origin, candidate workspace identity, and a
seed-shadow regression proving candidate code cannot replace the provider
control plane. It may not claim future signed-provider-review authority, model
dispatch, counter restoration, runtime-state edits, or queue-history edits.

The second repair launch was deliberately three-way sharded: ASE3-027 to lane
0, ASE3-019 to lane 1, and ASE3-023 to lane 2. Their authority remains
disjoint: ASE3-023 is forbidden from touching `implementation_daemon.py` or
provider authority files, and ASE3-027 is forbidden from touching ASE3-019
profile, router, or daemon surfaces. That process generation is now fenced;
interrupted ASE3-023/027 candidates require controlled acceptance or rejection.
The next-wave readiness audit additionally found that ASE3-019's control-plane
capsule omits a hermetic CID dependency closure: `utils/cid_utils.py` delegates
to optional user-site `multiformats`, which `python -I` cannot load. Because the
ASE3-019/022/023/027 task blocks are protected identity contracts, ASE3-030,
ASE3-031, and ASE3-032 are added without rewriting them. Their ordinary status
dependencies follow the committed acceptance sequence: ASE3-030 depends on
ASE3-019, ASE3-031 depends on ASE3-030, and ASE3-032 depends on ASE3-031. The
scheduler config contains a separately sealed
acceptance-prerequisite join preventing ASE3-023
acceptance or ASE3-022 transition until ASE3-030 proves that the Git-bound
capsule recursively includes every in-tree dependency, imports every allowed
module from its sealed fd, and mints/validates canonical CIDv1 lowercase-base32
raw and DAG-JSON sha2-256 identities under `python -I` without user site or
`PYTHONPATH`. Because the protected task blocks cannot be rewritten, ASE3-031
and ASE3-032 extend this prerequisite as new canonical tasks: the first seals
the reviewed native dependency and the second locks every scoped DuckDB
connection at birth. The path
`data/agent_supervisor/prompt_only_self_improvement_v3/convergence/hermetic_control_plane_identity_acceptance_receipt.json`
is reserved and must remain absent while its schema
`ipfs_accelerate_py.agent_supervisor.ase3-030-hermetic-identity-acceptance@1`
lacks a strict validator and convergence-manifest binding. ASE3-030 remains
`todo` until an immediately preceding protected acceptance commit adds that
validator, binds source HEAD/tree/blob/archive/root and suite digests, and
atomically accepts the task at A030. The following P031, A031, and A032 commits
then authorize and accept the native dependency and connection policy in strict
parent order before A023/027 and the ASE3-022 L reload authorization. ASE3-023
implementation may be inspected independently, but cannot be accepted early.

ASE3-022 keeps ASE3-021 machine-blocked until the operator-salvaged unchanged-
CID ASE3-019, ASE3-030 hermetic identity closure, ASE3-031 sealed native
dependency, ASE3-032 locked DuckDB policy, and the ASE3-023/027 repair chains
are accepted and jointly revalidated. Its final receipt must bind the attempt-2 incident, operator
salvage, accepted control plane, exact stopped generation, and transition
daemon/scheduler blobs. The same namespace is then relaunched once from that
completion commit. ASE3-029 then removes all eleven audited upward-import sites
by lowering shared authority DTO/verifier, invocation-budget/execution-slice,
and non-authoritative provider-capacity contracts into a neutral package. It
updates the six current lower-domain importers while retaining local-profile
key/lifecycle effects in `local_profile`, orchestration and compatibility
re-exports in entrypoints, and provider decisions in `llm_router`; an exact AST
inventory must report zero runtime/todo-daemon imports of entrypoints. ASE3-028
then focuses only on removing duplicated provider ranking and classification
from `implementation_provider_auto` and `capability_resolver`, with both
consuming one content-addressed `RouterOwnedProviderDecision` from
`llm_router`. ASE3-024 installs a separate signed planning-policy artifact, a
multiprocess durable encrypted prompt broker, and a planning-specific once-only
CAS with durable `UNKNOWN`/`PROMPT_REPLAY_REQUIRED` behavior. ASE3-025 commits
the authoritative program revision to DuckDB first, preserves embedded task
CIDs and subgoal owners through a generated-source observer, constructs a
namespace-independent runtime profile, and proves genuine configured runtime
subprocess execution without a Git-tracked board. ASE3-021 lands the durable
refill saga cursor and monitor phase deadlines dormant; ASE3-020 closes the
actual-parser fan-in plus immutable history/cursor vectors, monitor-ready
reservations, and UNKNOWN adoption; and ASE3-008 directly depends on both,
owns the configured-scheduler semantic-progress integration, and composes the
reviewed `ReviewedHostNamespaceReconciler` guardian with the detached monitor.
The legacy objective and codebase refill flags and the monitor flag remain
disabled through those implementation waves.

ASE3-026 is a second, distinct operator boundary. It is a canonical producer
for the monitoring goal but remains blocked, non-schedulable, and review-only
because it owns protected authorization/config validation and the later
post-effect observation. One operator-reviewed pre-effect commit may add only
strict authorization validation and manifest binding for the inactive exact
tree; its signed receipt cannot claim the generation already ran. After that
commit validates, one target-old+1 CAS/lease winner may enable only scoped
prompt-program/objective refill plus the autonomous monitor, disable legacy
hash sharding for active slices, retain broad legacy codebase refill as false,
and reload via the reviewed host guardian. A later separately bound observation
must prove the actual replacement lifecycle, monitor, heartbeat/cursor, and
refill effects. ASE3-009 cannot become ready from authorization alone.

ASE3-000 selectively ports or rewrites preserved v2 work. No task may claim
historical ASE/ASE2 completion based on old state files. Fresh task IDs and
content identities prevent stale v2 receipts from satisfying v3 dependencies.

## 11. Verification gates

The release candidate must prove:

- cold `--help` and package import perform no IPFS/provider/process startup;
- Python, CLI, MCP, and MCP++ compile equivalent canonical requests and results;
- one prompt produces a valid root goal, subgoals, atomic tasks, and dual
  DuckDB/Markdown projections with matching identities, after verification of
  an independently signed planning-policy artifact and one planning-specific
  durable pre-effect CAS; two processes share one broker/effect winner, and an
  unknown provider outcome becomes durable `UNKNOWN` plus
  `PROMPT_REPLAY_REQUIRED` without provider replay;
- the exact package AST has zero runtime/todo-daemon imports of entrypoints,
  while neutral DTO/re-export identity and serialized compatibility remain
  unchanged and all provider policy remains in `llm_router`;
- the exact Git-bound control-plane capsule recursively includes its in-tree
  identity dependency closure, imports every allowed module solely from its
  sealed fd, and mints/validates known raw and DAG-JSON CID vectors under
  `python -I` with user site and `PYTHONPATH` unavailable; missing, substituted,
  extra, or externally resolved members fail before effects;
- the P-phase native launch authorization is signed and manifest-bound, is not
  derivable from inspection, binds the exact reviewed DuckDB/Python/ABI/ELF
  pin, and claims zero launch effects; a fresh `python -I` process exact-matches
  that authorization ID, imports `_duckdb` only from the inherited sealed fd,
  executes a real query, and cannot retry after preload begins or fails;
- every ASE3-032-scoped DuckDB connection atomically disables extension
  autoinstall/autoload, external access, and unsigned extensions, applies only
  bounded threads/memory tuning before `lock_configuration=true`, verifies the
  exact typed settings, closes on mismatch, blocks external extension bytes and
  external-path access, and preserves schemas, tables, views, sequences,
  macros, custom types, indexes, data, and metadata through production
  compaction containing zero `ATTACH`; statically linked `LOAD` and
  `ATTACH ':memory:'` observations grant no broader authority;
- every prompt-product launch-reachable generated-board, planning,
  run-registry, and runtime-history connection delegates to
  `connect_duckdb_with_policy`; an installed-tree import/call-graph plus AST
  gate rejects raw reachable `duckdb.connect` and explicitly classifies
  surviving non-reachable legacy/proof-only sites;
- a fresh generated arbitrary-namespace board, rather than this seed board or
  a wrapper, is committed authoritatively to DuckDB before projections and is
  consumed through a revision-fenced generated-source observer by genuine real
  configured scheduler, implementation-supervisor, and implementation-daemon
  subprocesses without requiring Git-tracked Markdown;
- embedded task CIDs and subgoal owners are identical from admission through
  DuckDB, observation, claim, effect, and receipt;
- at least two independent tasks execute concurrently, while conflicting tasks
  never overlap and all claims are fenced;
- a real supported `run` starts or adopts real lifecycle and monitor processes;
  RUNNING requires their same-revision births, leases, fences, fresh heartbeats,
  and monotonic cursors, and no default in-memory-completed, projection-only,
  or no-op-effect path is reachable in production mode;
- low-water, drained-open-goal, failed-validation, and drift cases each prove
  `EVALUATING -> APPEND_RESERVED -> APPENDED -> PLAN_INVALIDATED -> RECOMPILED
  -> DISPATCHED/ADOPTED` with durable predecessor links and phase deadlines;
- a drained evidence-complete goal does not refill;
- stale PID, stale heartbeat, frozen worker, dead scheduler, duplicate launch,
  dead monitor, client disconnect, provider saturation, monotonic-clock rollback,
  merge/refill stalls, repeated oscillation, crash-at-boundary, lease loss,
  branch-only completion, UNKNOWN effect outcome, and corrupt projection cases
  recover/adopt once or fail closed deterministically without replay;
- raw prompt/secret leak scans, path/authority injection, UCAN attenuation,
  provider fallback, retry, resource, and budget adversarial tests pass;
- existing expert CLI and low-level MCP compatibility suites pass;
- installed-wheel Python, CLI subprocess, MCP JSON-RPC, and MCP++ black-box
  paths report one production composition CID and common preview/run backends;
- the self-host canary begins without an objective/taskboard/task-source seed,
  derives every program/effect CID from the prompt root, dispatches a refill
  descendant, and accepts a reviewed non-sentinel repository change;
- final evidence is produced on the exact current integration tree and the
  canary stays continuously healthy after its final recovery for at least the
  signed config `monitor_policy.canary_observation_seconds: 900` window, proven
  by monotonic start, end, and health evidence; prompt input and wall-clock-only
  claims cannot shorten or satisfy the window, and an unhealthy sample resets
  it.

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
4. **Protected authorization:** after the dormant refill and autonomous monitor
   implementations pass together, bind a signed pre-effect authorization for
   an inactive exact tree, old-generation quiescence, target old+1 CAS/lease,
   reviewed guardian, and scoped flags. It must not claim an effect ran.
5. **Activation and observation:** one CAS/lease winner consumes that
   authorization, fences the old generation, activates the scoped flags and
   replacement through the guardian, then a separate post-effect receipt proves
   actual lifecycle, monitor, heartbeat/cursor, and refill dispatch/adoption
   evidence.
6. **Self-host canary:** use the new prompt path from a fresh empty state
   namespace to perform one bounded
   improvement against this package while a separate monitor observes progress,
   refill, branch convergence, and recovery; after the final recovery it must
   remain continuously healthy for at least the signed config
   `monitor_policy.canary_observation_seconds: 900` window.
7. **Local auto:** promote only after exact-tree conformance, chaos, load, and
   sustained canary gates pass. Remote mutation remains separately authorized.

Rollback disables the prompt-only profile, fences its active run revision,
stops only processes whose birth identities match, preserves immutable evidence
and worktrees, and restores the prior expert entrypoints. It must not delete
user work or treat cleanup as authorization to reset a checkout.

## 13. Immediate execution rule

Do not restart the old prompt-only scheduler and do not activate its staged
rollout. Preserve the accepted ASE3-000 convergence base and complete the
ASE3-019/027 recovery, ASE3-030 hermetic identity closure, ASE3-031 sealed
native dependency, ASE3-032 locked DuckDB connection policy, ASE3-023 adaptive
acceptance, and ASE3-022 operator transition first. The exact protected order is
Q -> R -> P019 -> A019 -> A030 -> P031 -> A031 -> A032 -> A023/027 -> L;
every ordinary dependency is accepted on a strictly earlier committed phase,
A023/027 has no internal dependency edge, and acceptance may not bypass the
three-prerequisite join. Then execute
ASE3-029 -> ASE3-028 -> ASE3-024 -> ASE3-025 -> ASE3-021 -> ASE3-020 ->
ASE3-008. Keep refill and the detached monitor dormant until operator-owned
ASE3-026 validates pre-effect authorization, one exact old+1 CAS/lease winner
activates the generation, and a separate post-activation observation proves
the joined lifecycle/monitor/refill effects. Only then build the public facades
and run ASE3-013, which directly depends on ASE3-026, from no seed board and
requires 900 uninterrupted healthy seconds after its final injected recovery.
