# ExternalAgentAutonomousExecutionFabric implementation plan

This is the human architectural overview for the canonical machine board in
[`task_board.json`](task_board.json) and its exhaustive human projection in
[`TASK_BOARD.md`](TASK_BOARD.md). The source authority is
[`source_reconciliation_manifest.json`](source_reconciliation_manifest.json),
with a human projection in
[`reconciliation_report.md`](reconciliation_report.md); cross-package
compatibility is frozen in
[`stack_compatibility_manifest.json`](stack_compatibility_manifest.json).
The canonical board explicitly marks `OBJECTIVES.md`, `TASK_BOARD.md` and
`task_board.json` as generator-owned projections, and marks this plan, the
stack compatibility manifest, source-reconciliation manifest and reconciliation
report as reviewed source-owned control artifacts. The append-only
`bootstrap_materialization_attempts.json` evidence ledger is also
reviewed-source-owned, preserves every attempted materialization disposition
and is never a worker task output.

## Evidence boundary

The planning baseline is not `origin/main` alone. The reviewed integration
roots preserve the accelerator lease-state forward repair, the real two-parent
datasets UI/UX-IR provenance merge, the accepted kit baseline, and the narrow
MCP++ backend-role clarification. All other diverged branches and dirty
worktrees remain preserved and classified; none was force-pushed, squashed,
deleted, or silently overwritten.

External history is context and evidence, never authority. Imported messages,
tool results, patches, approvals and receipts retain provenance and trust
classification. Only a locally reverified result or an independently admitted
external receipt may satisfy a completion gate. The design neither requests
nor represents hidden model chain-of-thought.

## Control-plane architecture

```text
authenticated client / exported session / exact repository
                         |
                         v
      bounded normalization + quarantine + authority decision
                         |
                         v
       semantic indexing and logic-governed FormalWorkPlan
                         |
                         v
 conflict-free task frontier + isolated policy-qualified OCI worktrees
                         |
                         v
 signed effect envelopes -> bounded Quack append/read transport
                         -> one fenced local owner -> private DuckDB transactions
                                                  |
                                                  v
                      immutable outbox -> DuckLake / Parquet / IPLD / CAR/IPFS
                         |
                         v
  verification + proof + merge admission + semantic refresh + replanning
                         |
                         v
             content-addressed typed terminal report
```

DuckDB and Quack jointly form the mutable orchestration plane. DuckDB owns the
transactional records, schema, CAS predicates, fencing epochs and authoritative
outbox cursor. Exactly one admitted local state-owner service opens a shard's
private DuckDB file. Quack 1.5.5 is used only as the bounded multi-writer
append ingress and read-only projection transport: it never serves the
operational relations. Before applying a command, the local owner independently
verifies its principal, effect-bound authority, shard, epoch, live lease, fence,
idempotency key, deadline, typed operation and expected-version/CAS predicate.
Quack authentication or possession of its token is not mutation authority.
There is no direct remote file writer and no fallback from a failed Quack
service to shared direct-file access.

DuckLake is deliberately separate: it receives immutable post-commit epochs,
event/task/audit history, snapshots, lineage, benchmarks and recovery
manifests. DuckLake lag or loss cannot create or change a current claim, lease,
fence, write owner, resume right, merge authority or finalization decision.

## Goal graph and execution order

The root goal `EAAEF-G000` owns nineteen ordered epic subgoals. Contracts are
frozen before consumers, while tasks with disjoint files/symbols/effects run in
parallel inside each admitted epic:

```text
S host-gated bootstrap admission evidence
  -> A reconciliation and compatibility
  -> B handoff protocol
  -> C repository transfer
  -> D principal, authority and disclosure
  -> E ProjectAdapter onboarding
  -> F policy-qualified OCI execution (rootless where supported)
  -> G federated retrieval
  -> H logic-governed planning
  -> I conflict-free parallel work
  -> J DuckDB + Quack authority and DuckLake history
  -> K closed-loop replanning
  -> L Python / CLI / MCP / existing MCP++ profiles
  -> M security
  -> N observability
  -> O fault and end-to-end qualification
  -> P performance benchmark
  -> Q packaging and deployment
  -> R blocking CI, terminal seal and go/no-go
```

The machine board contains 116 stable tasks. Its content-addressed validation
report records the exact current dependency, owned-path and overlap-contract
counts; repeated ownership is permitted only through an explicit serialized
forward-extension contract and direct predecessor dependency. Every task
records the owning repository, exact
source commits/trees, source semantic root, source control-plane schema,
dependencies, read/write/effect scopes, context/capsules, resources, container
profile, model/provider route, tests, proofs, completion contract, leases and
fences, idempotency key, compensation, evidence, terminal state and artifact
identities.

## Bootstrap and rebind

Tasks `EAAEF-180` through `EAAEF-191` plus `EAAEF-000` through `EAAEF-009`
are the bounded initial population. Epic S is the host-controlled evidence
frontier that closes every current live-launch no-go; its ready tasks are
`EAAEF-180` through `EAAEF-183`. Manual host-controlled task `EAAEF-000`
depends on `EAAEF-191` and independently verifies signed EAAEF provider
authorization, a signed task-capable worker image and SBOM, at least five
exact worker slots, signed per-attempt internal-network/proxy authorizations,
the exact engine mode, bounded resources, the immutable materialization
receipt and the exact DuckDB 1.5.5/Quack 1.5.5 command-ingress decision. Rootless execution is
preferred and required
where the host supports it. An unsupported host may use only an independently
approved rootful host daemon with a nonroot worker, no mounted engine socket,
and otherwise identical isolation controls. Missing or invalid evidence produces
a typed no-go and starts no supervisor. Tasks `EAAEF-001` through `EAAEF-005`
depend on that admission;
`EAAEF-006` produces a reviewed compatibility proposal rather than mutating the
frozen R1 input; `EAAEF-007` builds the exact multi-repository semantic root;
`EAAEF-008` renews the Quack owner and immutable control-plane capsule against
the reconciled source and semantic roots; and
`EAAEF-009` performs the CAS-guarded Plan R2/population transition. All later
tasks are blocked templates carrying `REBIND_REQUIRED_BY_EAAEF-009`, not
schedulable work. Plan R2 must preserve completed R1 tasks, replace every
sentinel with the verified semantic root, and materialize only the B frontier.

### Reviewed Plan R1 ownership erratum

The reviewed Plan R1 source assigns the reusable CASF bootstrap opener, typed
owner adapters, reconciliation lifecycle, offline population projection,
owner-local R1/Plan-R2 services and their focused validation to `EAAEF-000`.
The offline population implementation and its focused validation are one
ownership unit; the corrected projection therefore contains 397 owned paths
and 430 task-output contracts. This is a source-ownership correction,
not a live admission, Plan R2 transition, terminal task result or signature.
It retains the `EAAEF-PLAN-R1` revision alias while producing new
content-addressed source, board, task and plan identities; any previously
prepared signing material is stale and a later ceremony must bind the new
identities.

`EAAEF-191` continues to own only its separate
`scripts/issue_eaaef_admission_bundle.py` ceremony CLI and immutable admission
bundle receipts. No lifecycle or Plan-R2 request path is shared between
`EAAEF-000` and `EAAEF-191`: `EAAEF-000` already depends on `EAAEF-191`, so a
reverse repeated-path dependency would create a cycle. The CASF canonical board
and tip remain unchanged. The EAAEF CASF import manifest is instead a reviewed,
source-owned EAAEF board input and cannot be mutated by a worker task.

The preserved worker-profile and execution-profile `@1` contracts cannot
represent Grok's required prompt, policy and provider-home mounts. Reviewed
source now defines versioned `@2` contracts with those exact mount kinds,
targets and modes, and the fail-closed launcher reverifies them before every
engine boundary. Qualified Grok nevertheless remains a typed pre-create no-go:
no external independently signed `@2` artifact, admitted engine or live
container preflight receipt exists. Unsigned extra mounts or silent widening
remain prohibited.

The embedded bootstrap is not continuous-operation authority. Multi-process
execution remains fail closed until the exact DuckDB/Quack 1.5.5 profile, the
signed-command ingress, the single private DuckDB owner and every mutable
repository gateway are qualified. A bare `StateCommand`, shared Quack token or
successful append can never authorize a mutation. Live
DuckLake remains a separate later promotion after catalog, binding, parity,
security, outage and restore evidence.

The bounded bootstrap gateway is narrower than the later generic daemon but is
not a three-read demonstration. The preserved structural
`EAAEFBootstrapDaemonCapability@1` and `EAAEFBootstrapDaemonGateway@1` bind the
closed 31-operation task, coordination, execution, provider, effect and
validation vocabulary used by EAAEF-001 through EAAEF-009; their original
source-evidence projection remains two admitted reads plus 29 typed no-go
dispositions and grants no production authority. The new
`DatasetsAuthoritativeEAAEFOperationalProfile@2`, independently signed
`EAAEFBootstrapDaemonOperationalCapability@2`, stable
`EAAEFBootstrapGatewayBinding@1`, command-authorization service contract and
`EAAEFBootstrapBorrowedTransactionOperationHandler@1` implement candidate
owner-transaction paths for all 31 operations without widening the generic
39-operation gateway. They exclude offline task materialization, broad task
enumeration, host merge admission and the three Plan-R2 operations. Every
mutable operation is required to execute inside the sole owner's already-active
DuckDB transaction and bind exact request, principal, board/task and lane
authority, live lease, fence, expected version, idempotency and receipt state.

Reviewed source now implements the R1 source-verified runtime factory rather
than a placeholder: `AgentSupervisorNativeDependencyAdmission@1`, the V2
lane/verifier/merge chain, `EAAEFExactEnvelopeJournal@1`, qualified-input-only
Quack and container-dispatcher factories, and exact per-birth supervisor/daemon
wiring construct the typed proxy and dispatch path from source-reverified
inputs. Direct proxy construction remains prohibited. The source status is
`r1_source_verified_runtime_factory_implemented`, not live admission. Actual
independently signed native-dependency, V2 lane, Quack-client, dispatcher-service
and per-birth lane artifacts, deployed signed command-authorizer/Quack/dispatcher
endpoints, a qualified DuckDB/Quack extension, host-merge evidence and real
admitted Docker/container/provider/network authority are absent. The
command-authorization client contains no private signing key, callback, token
or local-database fallback. EAAEF-000 therefore terminates `no_go`, creates no
supervisor process and must not leave EAAEF-001 through EAAEF-009 as a claimed
or blocked worker population.

Promotion is deliberately two-phase and acyclic. Historical Quack owner
evidence may support only the R1 bootstrap. After bootstrap admission, a new
Promotion@2 receipt must carry independent reviewer, operator and security
signatures and bind the exact three-operation Plan-R2 dispatcher, authorization
policy, build, lease/fence identities, atomic guarantees and current evidence.
That receipt can authorize the R2 transition but cannot promote the broader
daemon vocabulary. A separate gateway qualification is required for every
generic task/coordinator/execution/merge operation.

The distinct process-remote R2 seam is also source-complete. Its independently
signed `PlanR2ProcessRemoteOwnerCapability@1`, canonical request/response wire,
durable exact-envelope client journal, owner service and
`PlanR2ProcessRemoteOwnerGateway@1` preserve the three-operation
`prepare -> apply -> observe` partition without carrying R1, merge, generic
state-command, path, token, callback or database authority. Its status remains
`source_complete_external_signed_channel_required`: no actual independently
signed remote-owner capability, qualified process-remote channel factory or
supervisor repository wiring has been deployed.

## Isolated execution and authority

Each implementation attempt gets one isolated Git worktree and one isolated,
policy-qualified OCI container. Network-disabled work uses `--network=none`;
an approved provider attempt receives a collision-free internal network and a
CONNECT-only proxy that admits only the signed provider hostname on port 443.
The engine is rootless where supported; an
explicitly evidenced unsupported host needs separate independent approval for
a rootful-host-daemon/nonroot-worker fallback. The source-addressed full
container-execution-profile loader and launcher are implemented fail closed and
reverify the signed source artifact before inspect, create, start and restart.
No external independently signed profile artifact or admitted engine exists at
this freeze, so this code grants no launch authority. The host supervisor
retains authorization, run registry, Quack ownership, leases, secrets,
provider policy, merge admission and result acceptance. A worker receives only
its exact work packet, writable worktree, bounded resources and short-lived
capabilities. It does not receive the Docker socket, unrestricted host
environment or credentials, and it cannot approve its own work.

The logical claim key is:

```text
task ID + active plan revision + repository base tree
        + semantic-state root + task-spec CID + idempotency key
```

Multiple attempts may execute, but a fenced CAS permits exactly one accepted
logical result. Unknown write/effect scope conflicts conservatively. Merge,
network, secrets, dependency installation, broad disclosure, destructive
cleanup and publication require exact authenticated approval.

## Qualification and terminal policy

Required suites record exact argv, collected/passed/skipped/failed/xfailed
populations, duration, stdout/stderr hashes, source generation and environment.
A zero exit with a required skip or xfail is failure. Historical, simulated or
unavailable checks cannot count as current qualification.

The target is `supervised_external_pilot`, not a universal autonomy claim. It
may be assigned only if real supported clients can hand off and disconnect, a
new process can reattach, work executes in qualified containers, DuckDB/Quack
fencing and DuckLake non-authority survive faults, and current tests/proofs and
the terminal seal verify. Other repositories receive typed outcomes such as
`preview_only`, `unsupported_language`, `unsafe_repository`,
`insufficient_validation_profile`, `human_configuration_required` or
`mutation_not_admitted`.

At source freeze the structural board is valid and the direct
single-supervisor DuckDB authority propagation regression has been repaired,
and the source-addressed loader/launcher, operational profile, signed-capability
verifier, 31-operation borrowed-transaction handler, R1 runtime factory,
per-birth wiring and distinct R2 process-remote owner seam are implemented and
focused source-tested, but live launch remains intentionally unauthorized.
The signed-command-fabric profile is frozen at `@2` and binds the exact EAAEF
board namespace plus `control-shard-0` through the materializer, admission,
capsule, scheduler and runner; this source-level continuity grants no runtime
authority.
The isolated materializer's board-validation child also reopens only the exact
approved dependency root from the already verified bootstrap runtime binding,
while retaining isolated/no-site/no-bytecode flags and excluding ambient
`PYTHONPATH`; this makes offline validation usable without widening execution
authority.
The actual independently signed native-dependency, V2 lane/verifier/merge,
Quack-client, dispatcher-service, per-birth and Plan-R2 remote-owner artifacts;
deployed signed command-authorizer/Quack/dispatcher endpoints; qualified
DuckDB/Quack extension; signed `@2` container-execution profile; admitted
engine, image/SBOM, provider and effect-bound network authority; and current
host-merge receipts are absent. The only admitted next
action is the deterministic, host-controlled `EAAEF-000` preflight. It must
bind the EAAEF-scoped provider authorization, exact task-capable OCI
image/SBOM and engine mode, five effect-bound internal-network/proxy lanes,
resource policy, materialization identity and signed-command Quack decision in
an immutable independently signed admission receipt. Until then,
the result is a typed no-go and no supervisor process is launched. Continuous
Quack and live DuckLake still require their later exact-profile, gateway,
security, outage and restore qualification; neither may be inferred from a
successful bootstrap.

Focused lifecycle evidence covers exact claim retention/release, cleanup-only
recovery after a durable completed publication, poisoned pool-sidecar rejection,
changed-integration response loss and no repeated provider launch. It does not
claim full crash closure: owner failover, network partition, DuckLake outage,
supervisor crash and eight-worker restart qualification remain later gates.
