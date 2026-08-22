# ExternalAgentAutonomousExecutionFabric qualification report

This report records the evidence boundary for the reviewed bootstrap and
planning checkpoint. It is not a claim that Epics B through R have been
implemented, nor that arbitrary repositories may be autonomously modified.
The canonical implementation population remains the machine board in
[`task_board.json`](task_board.json).

## Exact source revisions and integration branches

The reviewed input roots are:

| Repository | Integration branch | Reviewed root |
| --- | --- | --- |
| `ipfs_accelerate_py` | `integration/external-agent-autonomous-execution-fabric-v1` | `0085dc719686bf4cd077c8099170bdd55fa2cf99` |
| `ipfs_datasets_py` | `integration/external-agent-autonomous-execution-fabric-v1` | `41533721c5559ad68cecfe226fa6ba5f76f8a15d` |
| `ipfs_kit_py` | `integration/external-agent-autonomous-execution-fabric-v1` | `2564aea1ae35061f2165872aff91e8a40801ab7e` |
| `Mcp-Plus-Plus` | `integration/external-agent-autonomous-execution-fabric-v1` | `5bf87beba3acf18d705c5c8ee3174e5e16ab5e04` |

The exact post-review accelerator implementation commit and tree are recorded
by the final clean-source materialization receipt. The reviewed roots above
remain ancestry/provenance inputs rather than a claim that `origin/main`
contained every completed change.

## Unmerged-work reconciliation report

[`source_reconciliation_manifest.json`](source_reconciliation_manifest.json)
and [`reconciliation_report.md`](reconciliation_report.md) inventory local and
remote lineages, worktrees, non-main commits, duplicate campaigns, dirty
overlays, superseded work, generated/runtime artifacts, schema conflicts and
recommended merge treatment. Preserved refs and dirty worktrees were not
force-pushed, squashed, discarded or silently overwritten.
Eight committed runtime artifacts (PID, WAL, log, certificate and private-key-
shaped files) were forward-removed from the integration tree without rewriting
their history. Replacement self-signed private keys are generated ephemerally,
and publication now fails unless the key can be restricted to mode `0600`.

## Compatibility manifest

[`stack_compatibility_manifest.json`](stack_compatibility_manifest.json) binds
the reviewed package roots and existing logic, semantic-state, ContextPack,
run-registry, receipt, proof, MCP++, DuckDB/Quack and DuckLake roles. Contract
changes require a superseding manifest and PlanRevision; workers cannot alter
the compatibility ceiling.

## External handoff schemas and client adapters

The transport-neutral schemas, Codex adapter, Claude Code adapter, Gemini CLI
adapter and generic JSON/MCP format are specified as tasks EAAEF-010 through
EAAEF-015. They are not implemented or qualified by this bootstrap checkpoint.
No hidden private chain-of-thought is requested or represented as exportable.

## Repository transfer implementation

Complete Git bundle, snapshot, dirty-overlay, quarantine and reconciliation
work is specified by EAAEF-020 through EAAEF-024. It is not yet qualified.
Remote callers therefore cannot nominate arbitrary host paths and no claim of
complete arbitrary-repository transfer is made at this checkpoint.

## Caller authority and source disclosure

The bootstrap uses source-addressed signed route authorization, distinct
operator/security review, exact effect and resource bounds, one-use identities,
and create-once admission/capsule artifacts. Conversation history, repository
text, model output, a CID, payment or a run ID cannot grant effects. The full
principal, disclosure, approval and MCP++ delegation surface remains assigned
to EAAEF-030 through EAAEF-034.

## Container security design

Worker policy requires a nonroot user, read-only root filesystem, dropped
capabilities, no-new-privileges, bounded PID/CPU/RAM/disk/GPU, no Docker socket,
no unrestricted environment and default-deny networking. Provider egress uses
an independently signed per-attempt authorization, a worktree-derived internal
network, and a bounded CONNECT-only proxy. Exact image/profile/network
identities are reverified before create, start and restart. The host currently
lacks an admitted rootless engine and no independent rootful-daemon fallback
approval has been supplied.

The EAAEF provider path accepts only the in-image
`/opt/eaaef/bin/grok` and `/opt/eaaef/bin/codex` identities. Explicit primary
Grok and fallback routing use the implemented fail-closed, source-addressed
full container-execution-profile loader and launcher. It re-reads and verifies
the signed profile before inspect, create, start and restart and rejects host
endpoint, image, mount, resource or security-policy drift before an engine or
provider effect. No external independently signed launch-profile artifact or
admitted engine is present, and the provider-neutral effect CAS plus
containerized Grok preflight receipt remain unavailable. Routing therefore
still stops before invocation signing, protected-attempt latching, Docker
access or a provider effect, preserving clean retry state instead of turning a
pre-effect refusal into an ambiguous provider attempt.

The preserved signed mount schema `@1` cannot express the prompt, policy and
provider-home mounts required by the Grok runtime. Reviewed source now provides
worker-profile and execution-profile `@2` contracts with exact mount kinds,
targets and modes; this is an implemented fail-closed schema extension, not a
live admission. A combined in-session source-only authority, network, bootstrap
preflight and safe-Grok run reported 172 passing tests and one deliberately
deselected live-container qualification case. A separate route/VGO/fallback run
reported 78 passing tests and two fork deprecation warnings. These unsealed
diagnostic observations are not a green live gate. No external signed `@2`
artifact or admitted engine exists, and no Docker container, provider or network
effect was invoked for those observations.

Two reproducible unsigned worker candidates were inspected during this working
session. These are unsealed diagnostic observations, not immutable promotion or
current-qualification receipts. The full candidate is intentionally rejected
because its inherited filesystem contains unresolved credential-like material.
The minimal validation-capable candidate was observed with image identity
`sha256:50bb50ce32cfe2287683e2ab68f761890eaeef58a033835e0f847c198a802fb0`;
its isolated Python validation toolchain is
`sha256:77aaf62266a0a59936adfe3a9a6ddcc68fb9a1b1e0acb3c37c1f9274060a9b32`,
its detached SBOM is
`sha256:4f696aac7aa1eda61919276f42f793261a19ddd3a665adb78ab0ac26f16f2bee`,
and its bounded merged-filesystem scan reported no unresolved
credential-material findings. Those observations must be repeated and sealed
against the final source tree before they can count as qualification. The image
remains zero-capacity until independent image/SBOM, provider, network and
engine-mode authorization is admitted.

## ProjectAdapter support matrix

No ProjectAdapter is qualified by the bootstrap population. The generic
read-only adapter and the first complete Python mutation adapter are tasks
EAAEF-040 through EAAEF-044. Until those gates pass, repositories are
preview-only or receive a typed limitation; validation commands are never
fabricated from README, CI or imported-agent assertions.

## Context ingestion and retrieval architecture

The board keeps repository truth, imported assertions, verified receipts, user
requirements, documentation, policy/legal data and model hypotheses in
separate provenance domains. EAAEF-060 through EAAEF-064 compose AST,
semantic-state, capsule, BM25, vector, sparse GraphRAG, knowledge-graph, proof
and counterexample retrieval without allowing similar untrusted transcript
text to override current source. This architecture is planned, not yet
end-to-end qualified.

## Logic-governed planning design

EAAEF-070 through EAAEF-074 reuse the existing FormalWorkPlan and admitted
logic/prover surfaces. Models may propose goals, alternatives and tasks, while
deterministic admission checks coverage, acyclicity, authority, resources,
conflicts, proofs and merge feasibility. No planner-specific theorem family is
introduced.

## Parallel scheduling and conflict model

Each task declares exact files, projected execution paths, read/write/effect
scope, resources, dependencies, leases, fences and completion evidence.
Identical file ownership is permitted only through an explicit ordered
forward-extension merge contract and a dependency edge; unknown scopes
serialize. EAAEF-080 through EAAEF-085 implement the full semantic conflict
graph, conflict-free frontier and one-accepted-result rule. The bootstrap
configuration caps a later admitted frontier at five lanes.

## DuckDB/Quack control-plane design

DuckDB is the private transactional authority. One fenced local owner alone
opens each operational shard. Quack is bounded authenticated append/read
transport; its token and transport authorization are not mutation authority.
Signed commands bind principal, independent authority, shard/store,
generation, live lease, fence, effect, deadline, nonce, idempotency and CAS.
Request/nonce consumption, mutation, authoritative receipt and idempotency are
committed in one private-owner transaction; the served projection is
rebuildable and non-authoritative. There is no remote operational-table UPDATE,
arbitrary SQL or direct-file fallback.

The exact process-remote Quack ingress can carry a bounded signed daemon
operation intent and return its typed receipt. Plan-R2 has a dedicated atomic
three-operation owner dispatcher. Promotion@2 is post-bootstrap and requires
independent reviewer, operator and security signatures over one exact nonce,
source, policy, owner/fence and evidence statement. Historical owner evidence
cannot promote Plan-R2, and Plan-R2 promotion cannot promote the generic daemon
gateway. Any generic operation that lacks a canonical transaction-safe handler
remains a typed no-go.

The canonical generic-daemon handler recognizes all 39 frozen operation names.
Only `task.get`, `task.list` and `task.ready` currently map safely into the
authoritative control schema and are admitted inside the existing owner
transaction; the other 36 produce durable operation-specific rejection
receipts. Arbitrary injected handlers and generic StateCommand fallthrough are
rejected. When either Plan-R2 or daemon capability is configured, every bare
generic command is rejected; the remaining base transaction seam is limited to
the exact `CLAIM -> claimed` transition and cannot turn `OBSERVE` or `MIGRATE`
authority into a status mutation. This partial safe vocabulary is not generic-daemon production
qualification.

The live bootstrap trace retains the smaller structural
`EAAEFBootstrapDaemonCapability@1` and `EAAEFBootstrapDaemonGateway@1` contract.
That original source-evidence projection is exact but not production-admitted:
it records the 31-operation vocabulary, two admitted reads and 29 typed no-go
dispositions. Initial materialization, broad task enumeration, host merge and
the three Plan-R2 operations are deliberately excluded; EAAEF-009 must use the
independently promoted `prepare -> apply -> observe` Plan-R2 gateway after
EAAEF-008.

Reviewed source now adds the distinct
`DatasetsAuthoritativeEAAEFOperationalProfile@2`, signed
`EAAEFBootstrapDaemonOperationalCapability@2`, stable
`EAAEFBootstrapGatewayBinding@1`, `EAAEFCommandAuthorizationService@1`, exact
task/lane/recovery authority schemas and
`EAAEFBootstrapBorrowedTransactionOperationHandler@1`. The owner fabric accepts
that typed capability and handler only on a mutually exclusive EAAEF path; the
generic 39-operation capability is unchanged. Candidate borrowed-transaction
implementations cover all 31 operations inside the sole owner's active
transaction and cross-join exact principal, board/task, lease, lane, process,
fence, idempotency and receipt state.

Reviewed source now implements the positive R1 construction path from exact
verified inputs. `AgentSupervisorNativeDependencyAdmission@1`, the V2
lane/verifier/merge chain, independently signed Quack-client and dispatcher
factory qualification schemas, `EAAEFExactEnvelopeJournal@1`, the lazy runtime
dependency factory and per-birth supervisor/daemon wiring construct
`EAAEFBootstrapExecutionRepositoryProxy@2` and its dispatch path without a raw
callback, path, token or database handle. Direct construction remains
prohibited. The exact status is
`r1_source_verified_runtime_factory_implemented`, not production admission.

The R2 path is separately source-complete. The signed
`PlanR2ProcessRemoteOwnerCapability@1`, `PlanR2CanonicalWireChannel@1`,
`PlanR2RemoteOwnerService@1`, `PlanR2ProcessRemoteOwnerGateway@1` and durable
`PlanR2RemoteExactEnvelopeJournal@1` carry only prepare/apply/observe. The
status `source_complete_external_signed_channel_required` records that the
actual remote capability, qualified wire-channel factory and supervisor
repository wiring have not been deployed.

No current source seam authorizes launch. Actual independently signed native,
V2 lane/verifier/merge, Quack-client, dispatcher-service, per-birth and Plan-R2
artifacts are absent, as are deployed signed command-authorizer/Quack/dispatcher
endpoints, a qualified DuckDB/Quack extension, independent host-merge evidence,
and real admitted Docker/container/image/profile/SBOM/provider/network
authority. The authorization service remains an independently deployed signer
boundary; no callback, shared token, signer key, Portal object or local database
path may be serialized into a child. The checkpoint therefore remains a
terminal typed no-go rather than a blocked worker population.

The container worker dispatcher additionally requires the owner to reserve the
exact task, attempt, plan, source/semantic state, worktree, image/profile,
network, lease, fence and idempotency identities before launch. A worker may
return only patch, artifact, test and proof identities. Acceptance and host
merge remain separate independently reviewed effects, and an ambiguous crash
after reservation cannot be automatically relaunched. The required production
reservation and independent receipt-verifier adapters remain fail-closed.

## DuckLake role and qualification

DuckLake is assigned immutable epochs, event/task/audit history, snapshots,
lineage, benchmarks and recovery manifests. It cannot grant claims, leases,
fences, write ownership or merge authority. Live catalog, outage, restore and
replication qualification remains EAAEF-094 through EAAEF-097.

## Multi-supervisor and subagent results

Independent analysis, implementation and verification lanes were used to
reconcile sources and harden bootstrap code. This is not the required live
three-supervisor/eight-worker qualification. No EAAEF supervisor or provider
process was started because admission evidence is incomplete.

## Crash and partition results

Focused control-plane tests cover stale fences, duplicate identities,
post-commit/pre-projection failure, projection loss/rebuild, restart and exact
Plan-R2 rollback/readback. The frozen lifecycle slices also cover exact claim
retention and release, cleanup-only recovery after durable `merge_completed`
publication, pool-sidecar poisoning, changed-integration response loss, exact
pool-entry adoption and no repeated provider launch. These are focused source
tests, not full crash closure. Full owner failover, network partition, DuckLake
outage, supervisor crash and eight-worker restart qualification remains open.

## Security findings

The current design fails closed on unsigned, named-but-not-signing or
self-approved evidence, mutable
or symlinked authority files, stale/expired identities, image/profile drift,
host provider execution, direct DuckDB access, Portal/Markdown fallback,
unbounded Docker options, Docker-socket mounts, unsafe proxy destinations,
untrusted imported success claims and worker self-acceptance. The existing
content-addressed host-merge admission and per-attempt verification result do
not cryptographically prove participation by the named independent reviewer;
they remain production no-go pending separately signed, trust-rooted contracts.
The source exact-envelope journals, runtime factories and wiring now exist, but
actual independent signing, per-birth artifacts, deployed signed service
endpoints, a qualified extension, rootless/runtime admission, provider
credentials/quota and exact live container/network evidence remain real
external dependencies.

## Performance and parallelism results

Configurations A through D have not been benchmarked. No wall-time, 60%
parallel-efficiency, 70% utilization or 50% reuse claim is made. The board
records these as targets for EAAEF-150 through EAAEF-153 and requires actual
results even when targets are missed.

## Packaging results

Clean wheels, versioned OCI images, locks, schema bundles, migrations and the
three required deployment profiles remain EAAEF-160 through EAAEF-164. The
unsigned image candidates are diagnostic artifacts, not release images.

## CI results

The source-freeze evidence records the following exact outcomes. None launched
an EAAEF supervisor, provider, network or production container effect:

- canonical two-lane recovery matrix: 13 collected and 13 passed, with zero
  skips or xfails;
- configured scheduler: 109 collected and 109 passed;
- todo-daemon port: 567 collected and 567 passed;
- frozen boundary slice: 121 collected and 121 passed;
- frozen validation slice: 56 collected and 56 passed;
- incremental runtime: 32 collected and 32 passed;
- native admission, V2 lane/runtime, Plan-R2 remote owner and supervisor birth
  wiring: 26 collected and 26 passed in 6.53 seconds under
  `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider --tb=short
  test/api/test_agent_supervisor_native_dependency_admission.py
  test/api/test_eaaef_lane_gateway_runtime.py
  test/api/test_plan_r2_remote_owner.py
  test/api/test_eaaef_supervisor_daemon_birth_wiring.py`;
- native-admission plus native-pin matrix: 50 passed and one environment-only
  skip in 2.66 seconds. The skipped node was
  `test_real_aarch64_duckdb_loads_from_sealed_fd_under_isolated_python`, because
  the reviewed aarch64 DuckDB/Python fixture was unavailable. That slice is not
  represented as zero-skip green;
- ordinary Plan-R2/state execution: 10 passed and three explicit dependency
  skips. The skipped nodes were
  `test_canonical_quack_owner_applies_and_reads_back_plan_r2_atomically`,
  `test_canonical_quack_owner_exact_prepare_replay_is_one_durable_result` and
  `test_canonical_quack_owner_rollback_replay_crash_and_live_fences`, each
  requiring DuckDB 1.5.5 and `QUACK_155_EXTENSION_PATH`;
- the same Plan-R2/state selection passed 13/13 under a temporary exact
  DuckDB/Quack 1.5.5 diagnostic environment. That temporary extension and
  virtual environment were unsealed and are not a qualified or deployed live
  artifact.

The independent Volta final review found no remaining P0/P1 source issue and
reported `git diff --check` clean. This is a source-review verdict, not an
external admission signature or production qualification. Required B-through-R
production lanes do not exist yet and therefore cannot be called green.

Earlier, now-superseded broad todo-daemon diagnostics were explicitly non-green.
One run hit its
`--maxfail=20` ceiling after 183 cases passed and 20 failed. An earlier broad
matrix collected 791 cases: 675 passed, 112 failed and four skipped (111
todo-daemon failures plus one adapter timing fixture that was subsequently
fixed). These unsealed observations neither satisfy the
zero-failure/zero-required-skip policy nor qualified the daemon. They are
retained as history and are not represented as the frozen result above.

For reproducibility of the uncommitted working-tree review only, the last
command-fabric boundary snapshot reported these SHA-256 file identities:

- `control_plane_transactions.py`:
  `6506aa84f0d67374bd317b117aa9045fc9f17d08519d2e557be7cf22395062ac`;
- `quack_command_authorization.py`:
  `78711cf8905ba998faf1a330d8ade06983a66038c800d1be1476461074f7bca4`;
- `quack_command_fabric.py`:
  `e90221fd9dead033ba42bfcb4626156a0c5f0f4764a6cef693255e8f4c981159`;
- `test_eaaef_borrowed_transaction.py`:
  `7bb66f0eaba01225fec6e56ec2fd2d2e540deb5400196e1ddc260f521c94aef6`;
- `test_eaaef_quack_command_fabric.py`:
  `29b15d6df2f4bbbc7b5657b21d6c315cdd80282256de0ad456a8b8d165ab867f`.

In-session diagnostics at that snapshot reported 10 focused fabric cases, 24
adapter-plus-fabric cases, 103 generic/EAAEF/gateway/Plan-R2 cases, 231 broad
non-todo boundary cases and five repeated all-29 parallel runs without an
observed failure. They lack immutable argv/environment/count/output-hash/duration
receipts and therefore are diagnostic observations only, not current CI or
promotion evidence.

A later isolated materializer diagnostic reported 20 focused cases passing.
The exact `/usr/bin/python3.12 -I -S -B` launch-plan path advanced through board
validation after reopening only the runtime binding's verified import root, then
stopped at the expected dirty-source guard. Independent source review reported no
P0/P1 finding. This observation is likewise unsealed, started no supervisor and
does not satisfy a release or promotion gate.

## Qualification level and recommendation

Current level: `research_demo` with source R1/R2 seams implemented and live
execution fail closed.

Recommendation: **NO-GO for external-agent handoff or live autonomous
execution** until actual independently signed native, V2 lane/verifier/merge,
Quack-client, dispatcher-service, per-birth and Plan-R2 remote-owner artifacts;
deployed signed command-authorizer/Quack/dispatcher endpoints; a qualified
DuckDB/Quack extension; independent host-merge evidence; and admitted
Docker/container engine, route, image/profile/SBOM, provider and effect-bound
network authority exist. The source-addressed EAAEF-000 admission must then
verify and every remaining board gate must execute. No live launch or effect
occurred. It is a GO only for reviewed source integration, clean offline
materialization, read-only verification and external evidence preparation.

The eventual qualified claim must remain narrow: supported clients may hand
off legitimately exportable visible history and an exact Git repository; only
repositories admitted by a qualified ProjectAdapter may receive bounded
containerized mutation. Unsupported codebases remain preview-only or require
human configuration.

## Current overlay implementation status (EAAEF-174)

This section extends the bootstrap checkpoint above. It does not rewrite
reviewed roots, unmerged-work classification, compatibility bindings, or the
historical diagnostic observations already recorded. Those remain provenance.

Contract modules for epics B through R now exist in-process (handoff API and
run handle, repository-transfer contracts, authority/disclosure, container
execution/lease/OCI/checkpoint records, context and planning, conflict-free
frontier, Quack owner and recovery, fixed-point termination, Python/CLI/MCP
surfaces, security and packaging admission, CI receipts). Host-gated bootstrap
admission receipts (epic S) remain separate typed evidence and are not a live
launch.

Qualification `evidence_mode` is `contract_fail_closed` unless a live receipt
exists. No live eight-container campaign was run in this overlay. Receipts
therefore bind:

- `evidence_mode`: `contract_fail_closed`
- `live_runtime_invoked`: `false`
- `live_eight_container_qualification`: `false`

In-process overlay suites exercise existing APIs and fail closed on the live
path rather than converting a simulation into live evidence.

### Epic status

| Epic | Goal | Status | Live qualified |
| --- | --- | --- | --- |
| A | Unmerged-work reconciliation and release baseline | `implemented_contracts` | no |
| B | External agent-session handoff protocol | `implemented_contracts` | no |
| C | Complete Git repository transfer | `implemented_contracts` | no |
| D | Caller identity, capability and disclosure policy | `implemented_contracts` | no |
| E | Project onboarding and codebase classification | `implemented_contracts` | no |
| F | OCI container execution fabric | `implemented_contracts` | no |
| G | Handoff context and federated retrieval | `implemented_contracts` | no |
| H | Logic-governed goal and task compilation | `implemented_contracts` | no |
| I | Conflict-free multi-agent parallel execution | `implemented_contracts` | no |
| J | Production DuckDB, Quack and DuckLake plane | `implemented_contracts` | no |
| K | Closed-loop execution and adaptive replanning | `implemented_contracts` | no |
| L | Python, CLI, MCP and MCP++ surfaces | `implemented_contracts` | no |
| M | Security hardening | `implemented_contracts` | no |
| N | Observability and accounting | `implemented_contracts` | no |
| O | End-to-end and fault qualification | `implemented_contracts` | no |
| P | Performance and parallelism benchmark | `implemented_contracts` | no |
| Q | Packaging and external deployment | `implemented_contracts` | no |
| R | Blocking CI and qualification release | `implemented_contracts` | no |
| S | Host-gated bootstrap admission evidence | `implemented_contracts` | no |

Live eight-container qualification: **not run**. Real external clients, isolated
worker containers, and unsupervised autonomy are **not** qualified. Unsupported
codebases remain preview-only or human-configured. The current recommendation
is recorded in [`FINAL_RECOMMENDATION.md`](FINAL_RECOMMENDATION.md).
