# Native supervision integration

The security extension is documented in [SYMBOLIC_SECURITY.md](SYMBOLIC_SECURITY.md):
source-bound header repair, separate code autoencoder training, SecurityIR
declarations, proof-index invalidation, and verified DuckLake learning catalogs.
Its local guard proof has a narrower scope than the impact-closed keyword repair
described below.

The integration path now connects scoped producer capsules, verified retry
reconstruction, intent-backed world observations, native planning, and the
Doctor tactician/proof/synthesis/impact/transaction pipeline. The container
qualification executes these components without model calls. It does not
measure Terminal-Bench success or qualify the complete production scheduler.

## Automatic native repair and post-publication state

The current provider-free integration entrypoint is:

```bash
PYTHONPATH=/path/to/ipfs_accelerate:/path/to/ipfs_datasets:/path/to/ipfs_kit \
DOCTOR_COMPOSITION_LEAN=/path/to/lean/bin/lean \
python3 -B -m benchmarks.agent_supervisor.container_coding.native_doctor_supervision \
  --output /path/to/new-results
```

This creates an independently declared, initially failing single-file task. It
builds and hydrates native AST/vector metadata and DuckLake catalogs, constructs
producer capsules and an intent-backed world snapshot, then automatically
selects the supported direct-local keyword repair from the actual source AST.
Real Lean/Z3 proof receipts, synthesis, impact closure and the Doctor transaction
produce an immutable candidate handoff. The native supervisor START allocates
the worker worktree, applies that exact candidate, validates, merges, and records
owner-authorized completion. The driver neither restores candidate bytes into
the canonical checkout nor directly marks a task completed.

An opted-in signed launch rebuilds context after successful native STOP, with a
separate refresh budget. `runtime.refresh_after_stop()` and post-STOP `observe()`
return the successor nomination. Source, exact accepted publication, completed
task revision, owner and intent watermark are rechecked. Original bundles stay
immutable. Capsules, Doctor observations, world records and metadata catalogs are
rebuilt; cached successors are rejected after source or intent drift.

The successor also projects goal/subgoal links, exact task statuses/revisions and
current completion-receipt identities from the same native world capture. Counts
explicitly describe the observed task population; a task-filtered capture cannot
establish that every task under a goal was observed. Actual goal status is
preserved. This progress view does not evaluate or replace the separate formal
goal-completion contract, and incomplete/cyclic hierarchies remain unavailable.

`published_retrieval_policy="lexical-tfidf-symbols@1"` pins the initial native
index, configuration, query, task and scope in the signed launch. Its built-in
rebuilder preserves the lexical configuration and projects a new source-bound
index into DuckLake. Vocabulary changes are explicitly unavailable.
`published_retrieval_policy="local-safetensors-symbols@1"` instead pins the
learned predecessor's result, manifest, protected local model snapshot,
configuration and native canary. It uses the actual embeddings router to rebuild
the changed corpus and query, then projects the new index into DuckLake. It
cannot download weights or substitute a lexical backend. Successful and failed
inference attempts retain their measured local calls and text counts; interrupted
work without a receipt stays unknown. The full benchmark arm selects the matching
policy and includes refresh time in its original total budget.
New files outside the original semantic scope are explicitly reported as
unindexed; publication does not silently widen the scope.

The Terminal-Bench full arm now attempts this Doctor workflow before model
coding. Supported candidates use the native worker/validation/merge path;
unsupported, ambiguous or unproved repairs retain residual planning proposals
and continue through the configured LLM router. A native DoctorPlanRefill and
PlannerDoctorContextCapsule carry verified residual evidence into that same
coding invocation. The advisory is bound to the existing task, source, permitted
outputs and acceptance checks; it does not create another provider call. A proof cannot bypass impact
closure: an unwritable dependent consumer still causes abstention. Residual
proposals do not automatically gain plan admission or task-completion authority.
If the native secret screen prevents optional Doctor evidence collection, the
workflow preserves that guard and emits a sanitized capability-gap refill for
the already admitted task. It does not manufacture a diagnostic snapshot or proof
and does not copy the flagged content into the advisory. Source, task and
admission integrity failures still stop the workflow.
The fixed-duration benchmark refresh runs only after successful STOP and an empty
process tree. It retains a finalization reserve, records deferred refresh when
insufficient time remains, and uses a deadline that optional-component error
handlers cannot swallow.

This qualifies one admitted task through the real daemon, not autonomous
multi-goal closure or a parallel benchmark speedup. The authored declaration is
not presented as model-generated planning, and the closed keyword operator is
not general Python program synthesis.

## Reversible semantic transport through the LLM router

`router_implementation_runner --semantic-repository /canonical/repository`
verifies that the allocated worker belongs to the same Git repository and checks
the producer-backed capsule inventory against current source bytes. A typed,
content-addressed translation table aliases repeated semantic symbol/CID
identifiers at the transport boundary. It preserves executable source, names,
literals, comments, paths, task scope and authority fields. This is reversible
representation compression, not a proof of semantic equivalence or source-code
alpha-renaming.

Explicit structured semantic replies decode through the same table and enter
the native residual-candidate grammar. Unknown aliases, mismatched scope or
changed source reject the response; actual provider usage is retained. Ordinary
prose is returned unchanged. Receipts separately identify native, translated and
actual model input bytes. The final benchmark audit can reconstruct historical
translated input without falsely asserting current source freshness.

The real-producer router qualification measured one prompt at 56,632 native
bytes and 31,725 transport bytes. That is a fixture byte measurement, not measured
token savings, correctness improvement, or a Terminal-Bench result.

Native content identities reuse only the canonical CID encoding of an already
freshly computed multihash, under the fixed process CID profile. The bounded
cache holds immutable strings, not source bytes, structured objects or verified
results. Every call still validates and serializes its payload and recomputes
the actual hash; both cold source reconstructions and native verification remain
in place. Changing source cannot reuse a previous content hash.

Worker capsule selection reads one fully verified immutable capsule index per
call. It still verifies every visited capsule's native identity and schema, and
uses native freshness/admission for every capsule that could fit. Capsules whose
bytes already exceed the remaining budget are omitted before those additional
checks. Exact selected payload, ordering, admissions and final packing are
preserved; this creates no cache across source generations. The two producer
scans, complete bundle verification and final source checks remain unchanged.

## Prepare a task's context

For an existing task in the native intent database, prepare both semantic and
world evidence from an explicit list of permitted repository files:

```bash
PYTHONPATH=.:/path/to/ipfs_datasets:/path/to/ipfs_kit \
python3 scripts/ops/agent_supervisor/prepare_task_context.py \
  --repository /path/to/task-repository \
  --intent-database /path/to/control.duckdb --task-cid TASK_CID \
  --file src/target.py --file tests/test_target.py \
  --required-raw-file src/target.py --required-raw-file tests/test_target.py \
  --output /path/to/task-repository/.runtime/context/TASK_CID
```

The output contains task-bound artifact metadata for the existing daemon
context compiler. It does not write canonical task status or admit a new plan.
Use the native task-source owner to bind metadata when applying it to an
authoritative task. Never include hidden benchmark tests or reference solutions
in the permitted input list.

Semantic context is checked against current source bytes before dispatch.
Explicit `Semantic context refresh: true` enables immutable scoped refresh;
otherwise stale source is rejected. Retry refresh retains the failure receipt
and the compiler's immutable task core and budget. Captured world evidence
remains an observation: missing operational owners do not become fabricated
scheduling or completion authority.

`generate_prompt_goal_graph_with_world` supplies verified, current intent
observations to the native prompt planner through its bounded context API.
The returned plan still goes through normal admission.

The full Terminal-Bench arm now builds its permitted AST/vector/DuckLake index,
semantic capsules and real empty-intent world capture before the first planning
call. Bounded summaries of those artifacts enter the native planner's constraint
context. Source, capsule inventory, retrieval and empty-owner evidence are
revalidated immediately before provider dispatch and after its response. Once
the proposed graph is independently admitted, source indexes and capsules are
reused and a new task-backed world snapshot is captured. Reuse does not perform
another embedding call or accept stale source. Cold preparation and verification
are measured inside the agent's total time budget.

`task_context_bundle.write_task_context_bundle` writes these nominations outside
the canonical task body. The signed launch pins the bundle digest; the worker
matches the actual database task CID and alias before resolving it. This keeps
world capture stable rather than changing its own task projection to attach
the capture's digest. Only context artifact keys are supported, with no provider
or completion-policy overrides.

The public service composition also forwards explicitly supplied
`optional_analysis` and `admission_request_factory` adapters to the prompt
service. Supplying them does not create independent admission evidence.

## Isolated local profile and pending acceptance

`runtime/local_planning_admission.py` admits bounded local coding tasks against
an independently signed source/output/check manifest. Planning retains each
acceptance check as a mandatory post-execution requirement. A planning receipt
cannot complete a task, and the native intent owner rejects contract removal,
changed checks, stale results and completion bypass flags. This local policy
does not supply external domain proofs or alter production admission.

`entrypoints/isolated_benchmark_runtime.py` qualifies real native supervisor
START/OBSERVE/STOP through signed local configuration, resource leases,
process identity and native heartbeat checks. Its original empty-queue profile
has implementation disabled and grants no task execution authority.

For the joined indexed repair lifecycle, run:

```bash
PYTHONPATH=.:/path/to/ipfs_datasets:/path/to/ipfs_kit \
DOCTOR_COMPOSITION_LEAN=/path/to/lean/bin/lean \
python3 benchmarks/agent_supervisor/container_coding/indexed_doctor_lifecycle.py \
  --output /path/to/new-results
```

This exercises actual failing acceptance, local admission, hydrated vectors,
semantic capsules and world context in the native worker prompt, an actual
Doctor transaction, refreshed metadata after repair, passing acceptance and
canonical task completion. Its task graph and reviewed repair are authored
qualification inputs; it does not claim live model planning or daemon dispatch.
Add `--model-snapshot /path/to/existing/snapshot --model-revision EXACT_REVISION`
to use the pinned local learned embedding path instead of lexical vectors.

## Hydrate complete permitted code indexes

`vector_index_preflight.py` builds native lexical vectors;
`learned_vector_preflight.py` uses an explicit existing safetensors snapshot
through `embeddings_router` and the native pinned embedding policy/canary.
Both serialize and reopen the native index, replay every row's AST facts, and
project/query the catalogs through DuckLake. Every supplied file must parse
successfully without truncation; unsupported or malformed inputs fail the run.

Large AST fact sets are addressed by exact content identity and reconstructed
from their bound native AST record. Earlier valid inline records remain
readable. External feature references retain the existing byte/count limits.
The learned qualification uses a declared 32,768-byte row bound for its actual
384-dimensional MiniLM vectors. These are retrieval nominations, not proof or
task-completion authority. The learned lane is a separate host qualification;
the minimal Docker image does not contain PyTorch or model weights.

`prepare_supervised_task_context` accepts the native snapshot and search result
together. It verifies ranking, complete current AST membership and every fact
reference before binding the retrieval artifact to the task. The native daemon
consumes it as required prompt evidence. Changed/deleted source removes stale
hits; a new index must be prepared for current retrieval after a repair.
Large native snapshots use complete hash-bound persistence references instead
of embedding their vectors in the bounded task envelope. Ranking and every
source-bound fact still replay before dispatch. Batched metadata insertion
retains exact single-link identities and commits all rows atomically.

## Database attempt lifecycle

The database runner passes its attempt limit to the real Portal bridge.
Ordinary typed-owner execution reserves and admits the exact claim with live
owner credentials and explicit lane fields. Closed pre-effect deferrals write
queue evidence before task CAS and settle only that exact claim; replay covers
lost queue/CAS responses. The successor receives the bounded previous failure
through the native feedback reader. The configured attempt limit is a
coordination safety cap, not a provider-token or model-call counter.

Dispatched candidates with unknown callback outcomes retain their claims,
including after lease expiry. General active-worker reconciliation and
cross-lane virgin transfer are still unqualified. Ordinary home-lane claim
handling refuses transfer lineage and configured transfer policies; it does
not manufacture the missing transfer receipts.

## Run the offline Docker qualification

The runner uses real DuckDB/DuckLake, source scans, Z3, a pinned Lean executable,
native context compilation and actual isolated Git repair transactions:

```bash
python3 benchmarks/agent_supervisor/container_coding/run_supervision_docker.py \
  --build --ducklake-extension-dir /path/to/.duckdb/extensions/v1.5.5/linux_arm64 \
  --lean-toolchain /path/to/lean-4.34.1 \
  --datasets-source /path/to/ipfs_datasets --kit-source /path/to/ipfs_kit \
  --output /path/to/fresh-results-directory
```

Use DuckLake, httpfs and Quack extensions matching the container's architecture.
The image build downloads its base and Python/OS packages; extensions come from the
explicit build context. Native `LOAD` verifies version/architecture
compatibility. The test runs have networking disabled, read-only source and
toolchain mounts, and no host credentials or Docker socket. The runner records
the resolved image ID, source hashes, test results and repair receipts.
It runs as nonroot `1000:0`; pinned Python/prover deployments remain outside the
worker's write authority. Group access is used only for the new result directory
under rootless Docker's bind-mount mapping. Running this proof path as root
correctly fails the native executable-ownership check.

The joined lifecycle repairs a keyword argument using a closed operator.
Its proof concerns preservation of that keyword's value, with separate AST
and behavioral checks. The native transaction validates its candidate before
ref publication. That proof does not authorize task completion: the separate
signed acceptance check must actually pass on current source. Unsupported
findings still abstain.

## Benchmark accounting

`full_supervisor_benchmark.py` prepares and executes one-shot Harbor trials for
the `full` and `no-index` arms. Both use the original task Dockerfile and hidden
verifier and Codex 0.158.0 through the shared model router, with `gpt-6.1-sol`
and high reasoning in the current comparison profile. Historical trials retain
their recorded models. The default agent limit is 300 seconds; the explicit
`source384-5cpu-16gib-extended@1` development profile allows 840 seconds of
supervisor work, 60 seconds of cleanup and a 960-second outer agent limit.
Planning and cold indexing count against the supervisor work budget. Native Codex's separate adapter
is retained in `native_codex_baseline.py`.

The local container profile uses an owner UID for native state and signing,
and a separate worker UID for model tools, candidate validation, and automatic
materialization commands. Root-owned launchers and the signed runtime bind
their exact code, namespace identities, and private paths. The owner accepts
only its independently chosen checks and actual native publication receipts.
Cleanup covers detached worker processes. This initial profile supports one
active worker; a parallelism experiment requires separate worker boundaries.
Full owner state contains signing material and must not be exported as an
ordinary benchmark log directory. The Harbor adapter exports only curated
results and native usage receipts.

Deployment installs the entire pinned native Codex executable directory,
including `codex-code-mode-host`. Both the CLI version and the helper startup
are checked before a model call. Copying only the CLI binary can otherwise
return a completed model response while every tool command fails to start.
The coding router describes the allocated worktree's mapping from logical
`/app` paths and records separate hashes for native input and actual model
input; the router preserves the planner's prepared input bytes. Bounded native tool-event
counts and failure classifications are retained without exporting raw tool
arguments or output. Missing usage and incomplete classifications remain unknown.

The driver reserves a shutdown window within its total agent budget. On entry
to native STOP it replaces the work alarm with that cleanup deadline, including
when a work timeout is already unwinding. A verifier pass, native completion,
and successful process cleanup are recorded separately.

Full-arm finalization now collects a bounded context-input audit inside the
container after STOP, worker cleanup, and invocation accounting. It rehydrates
the persisted native capsule and reconstructs each coding call's complete model
input using that call's recorded workspace advisory. Exact hashes and byte
counts must match for that invocation; one match cannot validate an unmatched
retry. Only context identities, hashes and check results are exported. The
audit child excludes the task working directory from Python's automatic import
path and has at most three seconds within the remaining driver budget. Missing,
tampered or unavailable evidence remains unknown and cannot change completion
or erase usage. This post-v8 collector cannot recover trial 06's lost capsule.

Context preparation also exports monotonic stage timings for vector
qualification, persisted snapshot reopening, combined semantic/world context,
bundle persistence, and Doctor eligibility/report persistence. Initial function imports/admission are
separate from the existing inner `seconds` timer. Nested vector helper timings
overlap their parent interval and must not be added to it. These post-v8
measurements apply to future runs; earlier context intervals remain undivided.

The admitted owner also binds a fixed Git author and committer to its signed
launch environment. Candidate commits and native two-parent publication work
without borrowing a user Git profile; worker execution still clears owner
identity variables. Shutdown uses its signed lifecycle grant, process identity,
and live fenced lease even during the gap between publication and owner
validation. Source acceptance remains required for START and observation.

Run preparation and execution with the installed Harbor Python and an explicit
runtime archive produced by `terminal_deployment.py`. Each preparation uses a
fresh directory, hashes original task inputs and runtime sources, runs Harbor's
dry-run, and refuses a second execution in the same directory. Failed and
interrupted calls remain in usage accounting; missing counters remain unknown.
An official verifier pass is distinct from native supervisor completion.

Native retry deltas are verified and retained locally. A stateless provider
receives the reconstructed full context unless a future transport proves
retention of the exact parent. The daemon records final prompt bytes and the
configured tokenizer estimate after appended guidance. Provider-native token
usage is a separate measurement; a smaller local delta is not a token-saving
result.

The full indexed/native Codex comparison uses the isolated local profile to
join admitted tasks, real owner-bound daemon dispatch and router execution.
Production activation is not a prerequisite for the local benchmark. Signed
post-STOP context renewal is integrated. General active callback reconciliation,
independent admission of derived tasks, whole-goal completion authority and
multi-worker execution remain outside this local profile. See
[INDEXED_ABLATIONS.md](INDEXED_ABLATIONS.md). Keep cold build/reconstruction cost,
warm retrieval cost, all model attempts and independent benchmark outcomes in
that comparison. This qualification must not be relabeled as a full-system
benchmark result.

The targeted qualification excludes the old synthetic transaction API fixtures.
Nine failures in `test_agent_supervisor_deterministic_doctor_transaction.py`
were reproduced against unchanged HEAD: those fixtures expect the former
`@1` contract or successful rollback/commit without the durable `@2` worktree
evidence. Current native worktree and live-transaction tests exercise the new
validation gate. Passing the targeted integration suite is not a claim that
the whole repository test suite passes.
