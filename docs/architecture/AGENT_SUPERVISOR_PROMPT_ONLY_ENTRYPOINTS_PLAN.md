# Agent Supervisor Prompt-Only Entrypoints and Automatic Target Inference Plan

## Outcome

Make the normal agent-supervisor experience:

```bash
ipfs-accelerate supervisor run "Improve validation-cache correctness"
ipfs-accelerate supervisor steer RUN_ID "Prioritize the load tests"
ipfs-accelerate supervisor follow RUN_ID
```

with equivalent Python, MCP, and MCP++ entrypoints. A prompt-only invocation
must inspect the current target, resolve a conservative supervisor profile,
create or adopt a durable run, plan and admit work, materialize a canonical
task source, start healthy lanes, and return a resumable run handle. Advanced
flags remain available as overrides, but they are not the normal path.

This plan is the executable successor to:

- `AGENT_SUPERVISOR_PROMPT_BOOTSTRAP_AND_RESCUE_PLAN.md`; and
- `AGENT_SUPERVISOR_PLAN_CREATE_AND_STEER_PLAN.md`.

Those plans supplied the prompt workflow, create/steer model, deterministic
admission, task-source parity, lifecycle, and rescue foundations. This plan
closes the remaining usability and runtime-composition gap. Its objective heap
and taskboard are:

- `agent_supervisor_prompt_only_entrypoints.objectives.md`; and
- `agent_supervisor_prompt_only_entrypoints.todo.md`.

The audited 2026-08-03 refresh is recorded in:

- `agent_supervisor_prompt_only_entrypoints_v2_delta.objectives.md`; and
- `agent_supervisor_prompt_only_entrypoints_v2_delta.todo.md`.

The delta is deliberately versioned rather than edited into the live v1
taskboard: task metadata contributes to canonical task CIDs, and a v1 bundle
supervisor is still consuming that materialized projection. The delta adds
the missing ambient-evidence composition layer, signed local bootstrap,
DuckDB run-registry convergence, effect-bound launch-plan revalidation, and
the exact quota-only provider route. It must be merged into a newly
materialized v2 projection before execution; it must not rewrite v1 task
identity or completion evidence in place.

The v1 heap and board use the `ASE-G…` and `ASE-…` namespaces; the delta uses
`ASE2-G…` and `ASE2-…`. The proposed
`ASI-172` through `ASI-185` identifiers in the earlier create/steer plan are
retired as planning aliases because the live ASI board subsequently reused
some of them for unrelated completed work.

## Current baseline

The repository already has strong low-level pieces:

- `PromptSupervisorService.preview`, `materialize`, `start`, and `bootstrap`;
- content-addressed prompt, scan, goal, task, materialization, run, and rescue
  contracts;
- deterministic repository scanning, goal planning fallback, plan admission,
  Markdown and DuckDB task sources, lifecycle orchestration, and rescue;
- one canonical `OperationRequest` / `OperationResult` control contract across
  Python, CLI, and MCP;
- repository forest, checkout authority, repository snapshot, provider
  selection, resource scheduling, conflict scheduling, and multi-lane runners.

The installed experience is nevertheless not prompt-only:

1. `ipfs-accelerate agent` requires nine target bindings unless the caller
   supplies a complete request: repository root, state root, repository ID,
   tree ID, objective ID and revision, policy ID and revision, and caller.
2. Real mutation additionally requires authorization, idempotency, exact
   effects, a lease, and a fencing epoch.
3. The default CLI/MCP `RepositorySupervisorBackend` implements bounded reads;
   it does not install prompt, materialization, lifecycle, rescue, validation,
   or objective mutation handlers.
4. CLI prompt parsing computes a safe prompt identity and drops the body, but
   no transient body channel connects the installed command to the live
   planner.
5. Lifecycle orchestration needs an immutable launch profile that no standard
   runtime factory currently compiles or selects.
6. Prompt workflow continuation relies on process-local state unless an
   embedding supplies durable receipt and artifact loading.
7. The implementation supervisor and daemon still expose large,
   program-specific flag surfaces and inconsistent state-root defaults.
8. Current MCP tools accept a complete canonical request. MCP++ has no
   supervisor-specific prompt, run-handle, or steering contract.

The principal problem is therefore not argument parsing. It is the absence of
one trusted invocation runtime that resolves identity, compiles configuration,
installs real service handlers, carries transient prompt data, and resumes the
whole saga.

## User journeys

### CLI

```bash
# Positional text is convenient for non-sensitive prompts.
ipfs-accelerate supervisor run "Add a bounded cache inspection API"

# File/stdin avoid exposing sensitive text in a process listing.
ipfs-accelerate supervisor run --prompt-file request.md
printf '%s\n' "Fix the failing graph query tests" |
  ipfs-accelerate supervisor run --stdin

# The current repository and the sole compatible run are inferred.
ipfs-accelerate supervisor status
ipfs-accelerate supervisor follow
ipfs-accelerate supervisor steer "Run the integration tests before benchmarks"

# Explain every inferred value without starting work.
ipfs-accelerate supervisor explain
ipfs-accelerate supervisor doctor
ipfs-accelerate supervisor preview "Refactor the provider adapter"
```

The canonical convenience group is `ipfs-accelerate supervisor`. Add
`ipfs-accelerate agent run|steer|follow|explain|doctor` as compatibility
aliases without changing the existing expert `agent` operations. An optional
`ipfs-accelerate-supervisor` console alias invokes the same parser and
service.

### Python

```python
from ipfs_accelerate_py.agent_supervisor.entrypoints import Supervisor

supervisor = Supervisor.open()
run = supervisor.run("Improve validation-cache correctness")
supervisor.steer(run.run_id, "Prioritize concurrent cache tests")
for event in supervisor.follow(run.run_id):
    print(event.summary)
```

`Supervisor.open()` may infer locations and a conservative profile. It never
manufactures remote identity or authority. Embedders may instead construct it
with explicit allowlists, principal, policy catalog, and lease backend.

### MCP and MCP++

```json
{
  "prompt": "Improve validation-cache correctness",
  "target_hint": "optional allowlisted target alias",
  "mode": "worktree",
  "budget": {}
}
```

Expose:

- `agent_supervisor_run`;
- `agent_supervisor_preview`;
- `agent_supervisor_steer`;
- `agent_supervisor_status`;
- `agent_supervisor_follow`;
- `agent_supervisor_explain`; and
- `agent_supervisor_doctor`.

MCP target selection is limited to roots configured by the server. MCP++
additionally validates UCAN invocation capability. Neither transport-level
permission replaces the inner effect-bound mutation decision.

## Design rules

1. **Infer identity and configuration, never authority.** Location, current
   tree identity, state namespace, provider capability, and resource ceilings
   are discoverable. Caller identity, UCAN authority, policy relaxation,
   destructive effects, and completion evidence are not.
2. **One high-level contract, one low-level control plane.** The convenience
   service compiles high-level intent into the existing canonical operations;
   it does not create a second mutation or completion system.
3. **A prompt is data, not configuration.** Prompt text cannot select roots,
   executable argv, environment, credentials, policy, authority, merge/push
   rights, or lease backends.
4. **Every inference is explicit after resolution.** A content-addressed
   receipt records selected value, source, alternatives, evidence identity,
   confidence/disposition, and override policy for every field.
5. **Safe useful work is the default.** A trusted local-development profile
   may authorize state writes, isolated worktree creation, in-scope edits, and
   allowlisted validation. It does not authorize merging, pushing, deployment,
   secret access, current-checkout rewrites, or destructive cleanup.
6. **Ambiguity degrades to useful preview.** Read, scan, and proposal work can
   continue where safe. Mutation pauses with one concise typed question only
   when multiple candidates would materially change the target or effects.
7. **Explicit canonical requests stay exact.** `--request-json` and
   `--request-file` bypass inference and retain current validation, errors,
   and exit codes.
8. **Durable intent, transient bodies.** Routine state stores prompt CIDs and
   capability-protected references. Raw prompt text, credentials, source
   bodies, and model transcripts do not enter receipts or logs.
9. **Idempotent adoption before launch.** A compatible healthy run is adopted;
   concurrent invocations cannot create duplicate process trees.
10. **Steering creates history.** A steer is a revision-bound append or
    `PlanDelta`; it cannot rewrite completed, accepted, claimed, or historical
    records.
11. **Discovery remains cold.** Imports, parser construction, `--help`, and MCP
    tool listing do not scan repositories, load providers, open DuckDB, or
    start processes.
12. **The inferred configuration is reproducible.** Replaying an inference
    receipt against unchanged evidence yields the same launch plan.
13. **Parallel work is compiled and durably coordinated.** Dependencies,
    predicted-file conflicts, resources, provider limits, shard ownership,
    leases, and fencing determine ready lanes. A lane label or prompt request
    cannot manufacture concurrency.
14. **Mutable coordination and immutable distribution are separate.** A
    single-writer DuckDB coordination shard owns current claims and fences.
    Parquet epochs and IPLD/CAR objects distribute immutable task, event,
    receipt, and result history through IPFS. An eventually consistent IPFS or
    IPNS view never grants a lease or authorizes an effect.

## High-level architecture

```text
CLI / Python / MCP / MCP++
           |
           v
  SupervisorIntentService
           |
           +--> PromptBodyBroker (transient/capability protected)
           |
           +--> TargetResolver
           |      +-- repository/worktree snapshot
           |      +-- objective/task-source/run discovery
           |      +-- policy/principal boundary
           |      `-- provider/resource/profile selection
           |
           +--> TargetResolutionReceipt + LaunchPlan
           |
           +--> StandardSupervisorRuntimeFactory
           |      +-- SupervisorControlService handlers
           |      +-- PromptSupervisorService
           |      +-- lifecycle + rescue + validation
           |      +-- task sources + run registry
           |      +-- DuckDB coordination-shard owner
           |      `-- Parquet/IPLD/IPFS epoch replication
           |
           `--> resolve -> preview -> authorize -> materialize
                         -> start/adopt -> monitor -> steer/resume
```

Add a highest-layer `agent_supervisor.entrypoints` package. Lower domain
packages must not import it. It may compose `control`, `prompt`, `objectives`,
`planning`, `runtime`, `rescue`, and `todo_daemon` lazily. Update the package
DAG and cold-import tests so this composition layer does not create cycles or
eager side effects.

## Contract set

### `SupervisorInvocationRequest`

Required:

- exactly one transient prompt body or capability-protected prompt reference;
- mode: `preview`, `worktree`, or a policy-approved stronger mode;
- bounded budget.

Optional hints:

- repository, scope, run, objective, task source, profile, provider, output
  mode, and lane ceiling;
- explicit overrides for advanced users;
- expected target or resolution receipt for replay.

Unknown fields and unbounded text fail closed. Prompt bodies are removed before
durable canonical serialization.

### `TargetCandidate` and `TargetInferenceDecision`

Each decision contains:

- field name and selected canonical value;
- resolution source and source precedence;
- evidence CID/revision;
- considered alternatives and rejection reason codes;
- `unique`, `defaulted`, `ambiguous`, `unavailable`, or `denied` disposition;
- whether selection affects only identity, creates an effect, or requires
  authority;
- whether an explicit override is accepted;
- freshness deadline and revalidation rule.

### `TargetResolutionReceipt`

Bind:

- invocation CID and prompt CID;
- repository, checkout, scope, HEAD tree and dirty-overlay identities;
- submodule and nested-repository population;
- state root and run namespace;
- objective, plan, task-source and revision, or the decision to create them;
- policy, principal, authority source, and effect ceiling;
- output mode and paths;
- implementation provider, capability report, resource budget, lane ceiling,
  merge target, worktree strategy, validation profile, coordination shard,
  lease/fence owner, and replication policy;
- every inference decision and unresolved ambiguity;
- configuration and capability catalog roots;
- resolution timestamp/freshness plus a receipt CID.

The receipt is evidence about resolution, not authorization.

### `ResolvedSupervisorProfile` and `LaunchPlan`

Compile the existing verbose daemon/supervisor configuration into one immutable
profile. Include exact argv, environment-name allowlist, task source, state and
run paths, provider route, resource ceilings, validation policy, lifecycle
health contract, lease backend, and expected effects. Credentials remain
external handles. The profile CID must change when any behavioral field
changes.

### `RunHandle`

A durable run handle contains:

- run ID/CID and target-resolution receipt;
- prompt/workflow, scan, plan, materialization, task-source, lifecycle profile,
  and process identities;
- objective and plan revision;
- lease/fencing generation;
- state, health and event cursors;
- continuation action and pending approval/ambiguity, when any;
- timestamps excluded from semantic identity.

The run registry supports exact lookup, current-run selection, adoption,
compare-and-swap revision, bounded listing, event replay, and reconstruction
after process restart.

### `SteeringRequest`, `SteeringEvent`, and `SteeringResult`

Steering binds:

- run ID and expected run/plan/task-source revisions;
- instruction prompt CID and transient reference;
- closed intent kind: append requirement, answer question, narrow scope,
  reprioritize, request replan, pause, resume, cancel, or request status;
- affected population bound and deadline;
- authorization/effect requirements;
- resulting plan delta, deferred successor tasks, or lifecycle request;
- event cursor and idempotency identity.

Free-form text may propose the closed intent kind, but deterministic code
validates it. If text is ambiguous between materially different mutations, the
service emits a bounded question rather than guessing.

## Resolution model

### Precedence

Use one deterministic merge:

```text
complete canonical request (inference disabled)
  > explicit high-level override
  > existing run binding when resuming/steering
  > authenticated transport and server policy
  > signed local/user supervisor profile
  > reviewed repository hints
  > deterministic repository/runtime discovery
  > conservative built-in defaults
```

A lower-precedence source cannot widen an upper-precedence allowlist,
authority, policy, resource ceiling, or effect set. Environment variables and
repository files are typed sources with explicit trust classification, not
unstructured overrides.

### Field inference

| Field | Primary evidence | Default behavior |
| --- | --- | --- |
| Repository root | explicit target, current Git worktree, repository forest | Select only one canonical enclosing root; reject symlink/nested ambiguity |
| Repository ID | repository descriptor and normalized remote/local identity | Reuse existing repository identity helpers |
| Tree ID | HEAD tree plus staged, dirty, deleted, admitted-untracked and submodule roots | Always observe current worktree; never use HEAD alone |
| Scope | explicit hint, current directory below root, prompt-mentioned path validated against scan | Default to repository root without widening allowlists |
| State root | trusted profile or platform state directory keyed by repository ID | Keep runtime state outside the source tree by default |
| Run | explicit run, unique compatible healthy run in target namespace | Adopt unique match; otherwise create a new run or report ambiguity |
| Objective | explicit objective, run binding, unique compatible active objective | New prompt creates a new content-addressed objective when no unique match |
| Task source | run binding, existing canonical projection, capability policy | DuckDB runtime index plus Markdown mirror when available; Markdown fallback is explicit |
| Policy | server/embedding policy or signed local profile | Built-in local worktree policy; repository policy is a constraint, never authority |
| Caller | authenticated MCP/MCP++ principal or local OS/key identity | Never derive from prompt or repository prose |
| Provider | capability catalog, credential handles, exact model identity, fresh typed quota evidence, budget and policy | Select `grok-4.5`; permit one `gpt-5.6-terra`/`medium` implementation fallback only after verified Grok quota exhaustion and before any repository effect |
| Lanes | ready width, conflict graph, host resources and provider quotas | ResourceScheduler computes a bounded ceiling, never prompt text |
| Branch/worktree | run namespace, repository authority and merge policy | Create an isolated run worktree; never edit/merge/push main by default |
| Validation | task acceptance, AST impact, repository metadata and reviewed command policy | Compile allowlisted argv; never execute prompt-authored shell |
| Coordination | deployment profile, topology, shard map and authenticated owner | Existing DuckDB lease/fence coordinator per writable shard; remote workers call the owner rather than sharing the database file |
| Replication | committed event cursor, epoch policy and IPFS capability | Export immutable Parquet plus DAG-JSON/IPLD manifests, optionally CAR-pack and pin through `ipfs_kit_py`/IPFS |

### Profiles and configuration

Support a small declarative profile instead of exposing daemon flags:

```toml
schema = "ipfs_accelerate_py/agent-supervisor/profile@1"
mode = "worktree"
output = "both"
max_lanes = 4
primary_provider = "grok"
primary_model = "grok-4.5"
quota_fallback_provider = "codex"
quota_fallback_model = "gpt-5.6-terra"
quota_fallback_reasoning = "medium"
quota_fallback_reasons = ["quota_exhausted"]
coordination_backend = "duckdb-ipld"
coordination_shards = 1
replication = "parquet-ipld-ipfs"

[effects]
edit_isolated_worktree = true
run_validation = true
merge = false
push = false
deploy = false
```

Profiles compile to current typed contracts; daemon argv is an internal
projection. Provide:

- built-in `preview`, `local-worktree`, and `ci-worker` profiles;
- user profile search in the platform configuration directory;
- optional reviewed repository hints;
- server-owned MCP profiles;
- signed/content-addressed production profiles.

Do not read arbitrary command strings or secrets from repository profiles.

### Ambient evidence composition

Leaf resolvers are not themselves a prompt-only entrypoint. Add one
`InferenceEvidenceCollector` that adapts bounded current process, CWD,
repository, installed-profile, authenticated-server, run-registry,
provider-capacity, host-resource, validation-policy, and topology evidence
into an immutable `InvocationContext`. A `SupervisorResolutionService` calls
the leaf resolvers in dependency order and emits one complete profile and
root-linked receipt. It performs observation only: provider dispatch,
repository mutation, process start, and transport authorization remain beyond
this boundary.

Transport adapters supply different trusted inputs to the collector. Local
CLI uses the nearest unambiguous enclosing Git root and an installed signed
profile; Python uses embedder allowlists or that same local bootstrap; MCP
maps server-owned target aliases; MCP++ additionally binds verified UCAN
attenuation. Prompt content cannot supply a root allowlist, caller, provider,
policy, authority, validation argv, or effect ceiling.

### Default provider route

The built-in implementation route is exact and policy-selected, never
prompt-selected:

```text
grok-4.5
  -> verified quota_exhausted before any repository effect
  -> gpt-5.6-terra with medium reasoning, once
  -> typed unavailable/deferred result
```

`auto` and the prompt-only facade require an authenticated, policy-allowed
`grok-4.5` primary. The only automatic implementation fallback is exact
`gpt-5.6-terra` with `medium` reasoning after fresh, typed evidence proves
that Grok quota is exhausted. The fallback is prohibited for mere
unavailability, capacity pressure, authentication errors, network errors,
timeouts, a bare status code, a nonzero exit, or an unclassified failure. It
is also prohibited after any repository effect. The receipt binds both exact
models, reasoning effort, quota evidence, prompt CID, scope, authorization,
budget, attempt/worktree identity, and effect boundary. Prompt or repository
prose cannot select or relax this route.

Fallback does not erase provider-role separation. If Terra implements after
verified Grok quota exhaustion, that attempt cannot satisfy its own review
obligation. A separately authorized, independent reviewer must be selected or
the review-bound effect remains deferred; the route does not unconditionally
require or enable a particular Codex review model.

There are two implementation routes to harden, and their evidence must not be
conflated:

- the current legacy/raw-command `auto` route pins `grok-4.5` and permits a
  bounded, shell-free replay of the same prompt to `gpt-5.6-terra` at medium
  reasoning only when its transcript yields verified quota-exhaustion
  evidence. Explicit Grok selection and every non-quota failure remain
  fail-closed. This compatibility path still needs the same durable
  pre-effect/effect-boundary receipt used by the production claim;
- the production packet route currently has Grok-implementation and
  Codex-review roles, so it needs a new typed
  `codex-implementation-fallback` role, bounded proposal/effect admission,
  fallback receipt, and independent-review continuation before it can make
  the same claim.

The production route may dispatch that fallback only for a verified
`quota_exhausted` reason. It never replays after an observed repository
effect, never expands the authorized path/effect set, and permits at most one
fallback dispatch for a task revision. Exact provider/model/reasoning and
executable identity, attempt/process identity, authorization, budget, quota
evidence, prompt/scope equality, and review separation are committed before
the fallback starts.

## Standard runtime factory

Implement one `StandardSupervisorRuntimeFactory` that:

1. receives explicit repository/state allowlists and an authenticated
   principal;
2. creates bounded artifact, receipt, event, task-source, run, DuckDB
   coordination-shard, and immutable replication stores;
3. registers real prompt preview/materialization, objective, lifecycle,
   validation, rescue, retry, and status handlers with
   `SupervisorControlService`;
4. wires the transient prompt body to the planner while persisting only the
   prompt reference;
5. admits the plan and generates a canonical DuckDB task source automatically;
   canonical Markdown is generated only within its 24-task admission bound or
   as root-linked epochs, never hand-authored;
6. compiles an immutable lifecycle profile from the resolved profile,
   including verified task-source kind and expected plan/repository roots;
7. requires a complete `LaunchPlan` and re-observes tree, authority, policy,
   provider route, run revision, task-source root, idempotency key, DuckDB
   owner lease, and fence immediately before every provider, write, or process
   effect;
8. resumes saga and process identity after restart;
9. adopts a compatible healthy process before launching;
10. exposes capability degradation instead of pretending unavailable handlers
   exist.

The read-only backend remains available for discovery and restricted servers.
The convenience entrypoint selects the standard runtime only when its trusted
profile and stores can be constructed.

## Prompt-to-run saga

```text
RECEIVED
  -> RESOLVING
  -> RESOLVED
  -> PREVIEWING
  -> ADMITTED | NEEDS_INPUT | REJECTED
  -> AUTHORIZING
  -> MATERIALIZING
  -> STARTING | ADOPTING
  -> RUNNING
  -> DRAINED | COMPLETED | BLOCKED | QUARANTINED | CANCELLED
```

Persist intent before each effect and an observed-effect receipt afterward.
Every transition is idempotent and compares the current tree, objective,
policy, run, task-source, lease, and fencing roots. A crash resumes at the
first uncommitted stage. A partial stage exposes one continuation token and a
human-readable next action.

`worktree` mode may proceed without interactive confirmation only when a
trusted local profile authorizes the exact bounded effects. Stronger effects
remain separate operations and approval boundaries.

## Steering behavior

1. Resolve or select the exact run.
2. Snapshot current run, plan, task-source, attempts, accepted evidence, tree,
   policy, event cursor, lease, and fence.
3. Classify the directive into the closed steering vocabulary.
4. Generate and deterministically admit a revision-bound `PlanDelta`.
5. Preserve completed and claimed task specifications.
6. Apply changes only to unstarted work; create successor/deferred tasks for
   claimed work.
7. CAS-publish a child plan revision to Markdown/DuckDB projections.
8. Notify lanes through the canonical event stream.
9. Return the new run/plan revision and observable consequences.

Concurrent steering requests serialize by run revision and lease/fence.
Semantically identical requests replay. Conflicting stale requests return a
typed conflict and a fresh preview; they are not silently rebased.

## Authorization and UCAN

### Local development

`ipfs-accelerate supervisor init` may generate an Ed25519 `did:key` local
development authority and install a signed, content-addressed
`local-worktree` profile. Private material is stored outside the repository
with mode 0600; profile load verifies its signature, repository/state roots,
effect ceiling, expiry, and revocation state. The same command family can
inspect, rotate, and revoke the profile. Thereafter prompt-only work can run
without repeated flags. The profile grants only:

- repository reads;
- supervisor-state writes;
- isolated worktree/branch creation;
- edits below admitted paths;
- allowlisted validation and local process management.

Merge, push, deployment, secret/key access, arbitrary network effects,
current-checkout replacement, and broad deletion remain denied.

### MCP/MCP++

The server owns repository/state allowlists and profile catalog. Derive the
caller from authenticated transport context. For MCP++, verify issuer,
audience, signature, expiry, revocation, attenuation, target scope, operation,
and effect capabilities. Then separately obtain the canonical inner
authorization decision bound to exact current roots, effects, lease, fence,
and idempotency.

Never store bearer UCANs or credentials in run receipts. Store bounded proof
references and decision identities.

## DuckDB, Parquet, IPLD, and IPFS coordination

Use the repository's existing DuckDB lease/fencing and CID-native distributed
lane contracts as the foundation. The storage model has three explicit planes:

1. **Authoritative mutable shard.** One owner process writes each
   `coordination.duckdb` shard. Short transactions plus the existing
   process-shared lock atomically claim, renew, release, fence, publish, and
   quarantine. Mutable run heads, run-revision CAS, adoption keys,
   idempotency keys, and event cursors use a `RunRegistryBackend` on this same
   coordination plane; the current JSON/file-lock registry becomes a bounded
   import and rollback adapter, not a second production authority. A
   deterministic shard map partitions repositories/runs/tasks so independent
   shards and lanes can progress concurrently.
2. **Immutable exchange and history.** After commit, export bounded,
   schema-versioned Parquet fragments for tasks, dependencies, events, leases,
   receipts, artifacts, metrics, and terminal results. Build a canonical
   DAG-JSON/IPLD epoch manifest containing the Parquet CIDs, previous epoch
   CID, shard identity, committed cursor, maximum fencing generation, schema
   roots, row counts, and integrity digests. CAR bundles are optional transport
   packages for the same DAG.
3. **Local query replicas.** Workers and observers fetch verified epoch objects
   through the IPFS backend router, which prefers `ipfs_kit_py`, and query the
   Parquet directly with DuckDB or reconstruct a read-only DuckDB projection.
   A replica never edits or replaces the authoritative shard.

The existing lease schema does not yet expose the committed cursor required by
that model. Add a transactionally advanced shard commit sequence under the
same DuckDB transaction and process-shared lock as the accepted mutation.
Freeze the logical tables, primary/sort keys, null/time/binary/JSON
normalization and per-store cursor/root bindings that form an epoch. An
exporter snapshots only a complete committed sequence; it cannot combine
unrelated latest views from the broker, run registry, event log and CAS.
Logical row-set digests and referential checks prove round-trip parity.
Deterministic Parquet byte identity is a separate, version-pinned option rather
than an assumed property of Parquet.

All content identities pass through the strict CIDv1 multiformats profile.
The replication adapter computes the expected raw or DAG-JSON CID locally,
requires the backend to return the same validated CID, fetches and rehashes the
bytes before admission, and capability-gates CAR export. The Hugging Face
cache adapter is local/cache transport until it satisfies this contract; its
current synthetic `bafy...` identifier is never admitted as an IPLD CID.
Likewise, an `ipfs_kit_py` add operation is not assumed to preserve a requested
codec. MCP++ compaction hashes and runtime-CAS identifiers are bridged through
explicit `IdentityLink` records and never treated as coordination-epoch
authority.

Remote workers fetch immutable lane input by CID and submit signed,
capability-bound results to the authenticated shard owner over MCP++. The owner
rechecks the current lease, logical epoch, fence, capability, policy, expected
effects, and result CID before accepting a terminal publication. Late,
partitioned, foreign, or conflicting results remain immutable but are
quarantined.

Before any remote result or discovery head is trusted, a transport-neutral
signature contract verifies issuer DID, key identifier, algorithm, audience,
resource, ability, shard/profile/epoch/fence binding, expiry, nonce and
revocation state. Content-bound claimant strings alone are not authentication.

Publish a signed `CoordinationHead` that points to the latest immutable epoch.
It may be advertised through MCP++, IPNS, pubsub, Hugging Face, or an IPFS
pinset, but it is discovery and replication state—not lease authority.
Read-side convergence can be eventual because every epoch is immutable and
linked. Writable ownership handoff must be explicitly fenced and
linearizable. On one host the existing DuckDB transaction/file-lock boundary
provides that serialization. In a multi-host deployment, all mutations route
to one authenticated owner per shard; high-availability owner election needs
a separately configured consensus/authority service. Without an unambiguous
owner, side-effecting work fails closed rather than treating IPFS/IPNS
convergence as a lock.

`CoordinationHead` includes shard/profile identity, monotonic sequence,
current and previous epoch CIDs, fence, signer DID/key ID/algorithm, issued and
expiry times, and signature. Import rejects rollback, same-sequence
equivocation, broken ancestry, revoked keys and unauthorized rotation.

Replication is subject to an explicit disclosure policy before encoding or
publication. It classifies every field, allowlists exportable columns,
redacts paths/task/provider material where required, supports private encrypted
epochs, binds recipient/key policy without publishing bearer capabilities,
scans canary secrets, and records retention/unpin policy. Public IPFS, IPNS or
Hugging Face availability never implies that task text, repository identities,
receipts or model payloads are safe to disclose. A failed or degraded Parquet
export, leak scan, signature, CID check or encryption step prevents head
publication.

The initial single-writer DuckDB shard mode is POSIX/local-filesystem only
because its lock uses `fcntl`. Launch-profile admission rejects network/shared
database paths and unsupported platforms. A future cross-platform lock or
linearizable multi-host owner service is an explicit capability, not inferred.

Pure, isolated, idempotent computation may execute speculatively under an
explicit degraded policy, but only the current fenced owner can accept its
result or authorize downstream effects. This preserves IPFS distribution
without claiming that content addressing alone prevents split brain.

## Observability

Every run exposes:

- resolution decisions and ambiguities;
- selected profile/provider/task source/lanes and why;
- preferred/selected/fallback provider and the typed fallback reason;
- DuckDB shard owner, committed cursor, current IPLD epoch/head and replica
  lag;
- current saga stage and continuation;
- objective, plan and task-source revisions;
- ready/running/blocked/completed tasks;
- provider/resource/lease health;
- retries, recovery and rescue actions;
- expected versus observed effects;
- event cursor and bounded artifact links.

`explain` renders configuration provenance without prompt/source bodies.
`doctor` checks dependencies, providers, credentials handles, writable state,
Git/worktree support, task-source integrity, lease backend, policy/authority,
ports, process identity, and stale state. It distinguishes required,
degraded, and optional capabilities and suggests the smallest safe remedy.

## Compatibility and migration

- Preserve all existing `ipfs-accelerate agent` expert commands and canonical
  request schemas.
- Preserve low-level console scripts while changing their repository-specific
  defaults into deprecation warnings and profile projections.
- Make one source authoritative for console-script registration; verify
  `pyproject.toml` and legacy `setup.py` parity until legacy packaging retires.
- Reuse current repository/snapshot, prompt, admission, task-source, provider,
  resource, lifecycle, and runner implementations.
- Add adapters for existing taskboards, status files, run directories and
  launch profiles.
- Reconcile the current `--start` mismatch so start is an explicit saga stage,
  not an unrecognized materialization parameter.
- Do not silently migrate or delete existing state. Preview import, validate
  identity, write a new run registry, then atomically switch the selected
  profile.

## Verification strategy

### Unit and property tests

- precedence, exact replay, stable identities, and provenance receipts;
- Git worktree, dirty tree, submodule, nested repository, symlink, non-Git and
  deleted/untracked cases;
- state-root and run-namespace collision resistance;
- objective/task-source/run unique, absent and ambiguous populations;
- exact `grok-4.5` selection; verified quota classification; once-only,
  pre-effect `gpt-5.6-terra`/`medium` fallback; same-prompt/scope/budget
  proofs; non-quota and post-effect denials; provider/attempt identity; and
  independent-review separation;
- resource and lane ceiling computation;
- profile parsing, signing, catalog selection and forbidden fields;
- prompt secret/body non-persistence;
- transactionally bound coordination commit sequences, frozen logical epoch
  schemas, canonical CID verification, Parquet logical parity, signed-head
  ancestry/rotation, disclosure/encryption policy and leak scanning;
- steering delta lifecycle invariants and stale CAS.

### Transport conformance

Run identical frozen requests through Python, CLI, MCP, and MCP++ and require
the same resolution, run, steer, error, effect, and event records. Imports,
help, discovery, schema listing, and tool registration remain side-effect
free.

### Real subprocess E2E

Create temporary fixture repositories and invoke the installed product CLI
with only a prompt. Exercise deterministic and model planning,
Markdown/DuckDB/both projections, real materialization, process start, task
claim, validation, steering, follow, restart, resume, completion, Parquet/IPLD
epoch export/import, IPFS fetch, and quarantine. Test explicit override
equivalence and old expert commands.

### Security and adversarial tests

- prompt attempts to select roots, commands, authority, credentials, policy,
  completion, merge/push/deploy, or destructive effects;
- malicious repository profiles and CI metadata;
- path/symlink/submodule/worktree escapes;
- forged caller, profile, receipt, CID, UCAN, authorization, lease and fence;
- forged or equivocating coordination heads, revoked/rotated signing keys,
  synthetic backend CIDs and codec substitution;
- prompt/source secrets in argv, environment, logs, events, state and errors;
- sensitive task, path, repository, provider or receipt fields in public
  replication epochs;
- stale tree/objective/policy/plan/task-source/run replay;
- duplicate launches and conflicting concurrent steering.

### Chaos and load

- crash before and after every saga intent/effect/receipt boundary;
- killed parent and orphaned descendants;
- corrupt/partial state, stale locks, expired leases and fencing loss;
- provider, DuckDB owner, Parquet export, IPLD/CAR verification, IPFS,
  replica-lag, head equivocation/key rotation and disk degradation;
- many repositories, runs, lanes, events and simultaneous steering requests;
- bounded CPU, memory, descriptors, storage, provider usage and recovery
  attempts.

### Quantitative promotion gates

- at least 95% of single-repository fixture invocations reach an admitted,
  materialized, healthy run from only prompt plus ambient authenticated
  context;
- 100% deterministic resolution replay on unchanged evidence;
- zero unauthorized or out-of-scope effects;
- zero raw prompt/secret leakage in the inspected durable surfaces;
- zero duplicate process trees for one run/profile/fencing generation;
- zero accepted duplicate claims/effects across DuckDB shards, partitions,
  stale replicas, or owner restart;
- exact DuckDB-to-Parquet-to-IPLD-to-DuckDB replay parity for every committed
  coordination epoch, including logical row-set digests and referential
  integrity;
- zero unclassified or disallowed fields in public replication objects and
  zero accepted CID/codec mismatches;
- exact `grok-4.5` is selected whenever admitted, while every
  `gpt-5.6-terra`/`medium` fallback is once-only, pre-effect, prompt/scope
  preserving, and backed by verified quota-exhaustion evidence;
- Python/CLI/MCP/MCP++ canonical parity for the closed fixture population;
- bounded time to first run handle and time to first useful event, with
  published baselines and regression thresholds;
- unchanged expert requests retain their existing canonical results and exit
  behavior.

## Parallel delivery

The `ASE` taskboard encodes dependencies, predicted files, resource classes,
and conflict policies. Its intended execution waves are:

```text
Wave 0: baseline inventory and UX acceptance fixtures
Wave 1: entrypoint package/contracts, safety matrix, steering contracts
Wave 2: repository, state, objective, policy, provider and profile resolvers
Wave 3: run registry, prompt broker, authority adapters, runtime factory
Wave 4: prompt-to-run saga, profile compiler, steering runtime, DuckDB shard
        coordination, exact quota-only provider fallback, strict CID adapter,
        shard
        commit sequence, signing, and disclosure policy in parallel where
        their declared files do not overlap
Wave 5: immutable Parquet/IPLD/IPFS replication fan-in plus Python, CLI,
        MCP/MCP++ and observability entrypoints as their dependencies permit
Wave 6: conformance, E2E, adversarial, chaos and load gates
Wave 7: compatibility migration, documentation, staged rollout and closeout
```

Tasks in a wave may run concurrently only when their declared dependencies are
complete and the conflict/resource scheduler admits their predicted files.
Integration joins are deliberately separate from leaf implementation lanes.

## Rollout

1. `off`: existing expert operations only.
2. `observe`: resolve and compare configuration; no prompt planning or writes.
3. `preview`: scan, plan and explain; no task-source or process effects.
4. `assist`: separately approve worktree materialization/start.
5. `local-auto`: trusted local profile automatically performs only bounded
   isolated-worktree effects.
6. `distributed-assist`: MCP++/UCAN remote workers, DuckDB shard owners, and
   verified Parquet/IPLD/IPFS replication with explicit apply authority.
7. `policy-auto`: later, independently evaluated automatic operation within a
   signed deployment profile.

Any authority, scope, identity, parity, duplicate-process, secret, stale-state,
or unexpected-effect regression returns the affected mode to `preview` or
`off`. Rollback never erases run history or evidence.

## Definition of done

The program is complete only when:

- a newly installed CLI can enter a supported repository and start useful,
  isolated supervisor work from one prompt without constructing target,
  task-source, provider, lane, lifecycle, or daemon arguments;
- the standard runtime installs real handlers rather than returning
  `unavailable` for the advertised prompt workflow;
- every inferred value is reproducible and explained by a
  `TargetResolutionReceipt`;
- prompt bodies reach the planner safely but do not enter durable routine
  records;
- a durable run handle survives process restart and supports status, follow,
  resume, and semantic steering;
- local automatic authority is explicitly installed and effect-bounded;
- MCP and MCP++ respect server allowlists, authenticated identity, UCAN, and
  the separate inner mutation decision;
- Python, CLI, MCP, and MCP++ produce equivalent canonical outcomes;
- concurrent launch/steer and crash/recovery tests prove idempotency, CAS,
  lease and fencing behavior;
- `grok-4.5` is the verified default implementation model and exact
  `gpt-5.6-terra` with medium reasoning is available once, only on verified
  pre-effect Grok quota exhaustion;
- committed coordination state is queryable in DuckDB, distributable as
  Parquet/IPLD/CAR through IPFS, and exactly reconstructible without granting
  authority to a replica;
- the old expert interfaces remain compatible; and
- all quantitative gates pass on a later fresh repository root.
