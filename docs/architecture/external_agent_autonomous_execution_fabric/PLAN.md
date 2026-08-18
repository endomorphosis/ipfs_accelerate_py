# ExternalAgentAutonomousExecutionFabric implementation plan

This is the human architectural overview for the canonical machine board in
[`task_board.json`](task_board.json) and its exhaustive human projection in
[`TASK_BOARD.md`](TASK_BOARD.md). The source authority is
[`source_reconciliation_manifest.json`](source_reconciliation_manifest.json);
cross-package compatibility is frozen in
[`stack_compatibility_manifest.json`](stack_compatibility_manifest.json).

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
  conflict-free task frontier + isolated rootless OCI worktrees
                         |
                         v
 typed authenticated clients -> Quack fenced owner -> DuckDB transactions
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
outbox cursor. Exactly one admitted Quack state-owner service opens a shard's
DuckDB file. All other readers and writers use bounded typed authenticated
Quack methods carrying principal, authority, epoch, lease, fence, idempotency,
deadline and expected-version/CAS data. There is no direct remote file writer
and no fallback from a failed Quack service to shared direct-file access.

DuckLake is deliberately separate: it receives immutable post-commit epochs,
event/task/audit history, snapshots, lineage, benchmarks and recovery
manifests. DuckLake lag or loss cannot create or change a current claim, lease,
fence, write owner, resume right, merge authority or finalization decision.

## Goal graph and execution order

The root goal `EAAEF-G000` owns eighteen ordered epic subgoals. Contracts are
frozen before consumers, while tasks with disjoint files/symbols/effects run in
parallel inside each admitted epic:

```text
A reconciliation and compatibility
  -> B handoff protocol
  -> C repository transfer
  -> D principal, authority and disclosure
  -> E ProjectAdapter onboarding
  -> F rootless OCI execution
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

The machine board contains 104 stable tasks, 237 dependency edges and 228
non-overlapping owned paths. Every task records the owning repository, exact
source commits/trees, source semantic root, source control-plane schema,
dependencies, read/write/effect scopes, context/capsules, resources, container
profile, model/provider route, tests, proofs, completion contract, leases and
fences, idempotency key, compensation, evidence, terminal state and artifact
identities.

## Bootstrap and rebind

Tasks `EAAEF-000` through `EAAEF-009` are the bounded initial population, but
the sole initial ready frontier is manual host-controlled task `EAAEF-000`.
That task independently verifies signed EAAEF provider authorization, signed
bootstrap image and SBOM identities, rootless/default-deny network and resource
policy, the immutable materialization receipt and an explicit Quack authority
decision. Missing or invalid evidence produces a typed no-go and starts no
supervisor. Tasks `EAAEF-001` through `EAAEF-005` depend on that admission;
`EAAEF-006` produces a reviewed compatibility proposal rather than mutating the
frozen R1 input; `EAAEF-007` builds the exact multi-repository semantic root;
`EAAEF-008` admits the Quack owner and immutable control-plane capsule; and
`EAAEF-009` performs the CAS-guarded Plan R2/population transition. All later
tasks are blocked templates carrying `REBIND_REQUIRED_BY_EAAEF-009`, not
schedulable work. Plan R2 must preserve completed R1 tasks, replace every
sentinel with the verified semantic root, and materialize only the B frontier.

The embedded bootstrap is not continuous-operation authority. Multi-process
execution remains fail closed until the exact DuckDB/Quack 1.5.5 profile, the
single Quack owner and every mutable repository gateway are qualified. Live
DuckLake remains a separate later promotion after catalog, binding, parity,
security, outage and restore evidence.

## Isolated execution and authority

Each implementation attempt gets one isolated Git worktree and one isolated
rootless OCI container. The host supervisor retains authorization, run
registry, Quack ownership, leases, secrets, provider policy, merge admission
and result acceptance. A worker receives only its exact work packet, writable
worktree, bounded resources and short-lived capabilities. It does not receive
the Docker socket, unrestricted host environment or credentials, and it cannot
approve its own work.

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

At planning freeze the structural board is valid and the direct
single-supervisor DuckDB authority propagation regression has been repaired,
but live launch remains intentionally unauthorized. The only admitted next
action is the deterministic, host-controlled `EAAEF-000` preflight. It must
bind the EAAEF-scoped provider authorization, rootless OCI image/SBOM,
network-deny/resource policy, materialization identity and explicit Quack
decision in an immutable independently signed admission receipt. Until then,
the result is a typed no-go and no supervisor process is launched. Continuous
Quack and live DuckLake still require their later exact-profile, gateway,
security, outage and restore qualification; neither may be inferred from a
successful bootstrap.
