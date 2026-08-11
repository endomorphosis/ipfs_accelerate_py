# Python semantic-compression coding-agent harness plan

Status: reviewed implementation plan; supervisor launch is intentionally gated by
`SCH-000`.

This plan defines a focused Python 3.12 and pytest harness in
`ipfs_accelerate_py.agent_supervisor.semantic_state`. It consumes the semantic
state contracts implemented in `endomorphosis/ipfs_datasets_py`, uses a narrow
durability port backed by `endomorphosis/ipfs_kit_py`, and reuses the existing
`ipfs_accelerate_py` supervisor runtime. It does not create another agent
framework, MCP server, user interface, theorem prover, or general multi-language
verification platform.

The companion supervisor inputs are:

- `docs/architecture/semantic_compression_harness.objectives.md`
- `docs/architecture/semantic_compression_harness.todo.md`

## 1. Revision and launch seal

The `ipfs_accelerate_py` runtime authority remains unresolved. Candidate commit
`8a1136d1114cd83a0c7a9cdcc03a41c4ed81ed14` supplied the intended real-CID and
process-liveness surface, but independent review rejected it for five reasons:

1. a live owner without heartbeat can create split brain while a lost fence is
   swallowed;
2. an expired owner can overwrite a newer active task index;
3. an empty or unavailable process snapshot fails open;
4. whitespace validation omits untracked and submodule outputs that a later
   commit stages; and
5. fast-zombie birth capture can leak a lease.

The seal records all five reasons. The candidate is evidence for the required
surface, not a sealed authority. The MCP++ wire authority inspected is
`Mcp-Plus-Plus` commit
`dc3164653a48d059ae9812078359daeafb451c07`.

The repaired `ipfs_kit_py` generation-bearing durable-root authority is pinned
at commit `05ba9375923cd5fb52e2c9c18b98b530d57d077f`. The reviewed phase-one
`ipfs_datasets_py` baseline was
`a2f5400b7cb89c8481819379a1b7b9959fe81d45`; neither the final repaired
incremental-index closeout nor the final semantic-state/Merkle/capsule closeout
is assumed here.

The dependency seal is schema
`ipfs-accelerate.agent-supervisor.semantic-state-dependency-seal@2`. The
accelerator runtime pin is `UNRESOLVED_REPAIRED_ACCELERATE_COMMIT`. Its eventual
Profile-B canonicalizer, Kubo CIDv1 helper, DAG-JSON CID authority, process-tree
fence, and process-birth fence remain explicit required blobs and
source-extracted contracts. The independent real-CID/process-liveness vector
and regressions for live-owner heartbeat/fence propagation, task-index
publication, unavailable process snapshots, dirty/clean untracked and
submodule output whitespace, and fast-zombie lease cleanup are mandatory
producer evidence for the replacement pin.

`SCH-000` must be completed manually before any implementation supervisor is
launched. The kit authority is already pinned, but the accelerator and both
datasets values must remain unresolved until their respective repaired
closeouts supply exact 40-hex commits:

```text
IPFS_ACCELERATE_RUNTIME_AUTHORITY_COMMIT          = UNRESOLVED_REPAIRED_ACCELERATE_COMMIT
IPFS_DATASETS_INCREMENTAL_SEMANTIC_INDEX_COMMIT = UNRESOLVED_FINAL_ISI_COMMIT
IPFS_DATASETS_SEMANTIC_STATE_COMMIT              = UNRESOLVED_FINAL_DSS_COMMIT
IPFS_KIT_DURABLE_ROOT_COMMIT                     = 05ba9375923cd5fb52e2c9c18b98b530d57d077f
```

The gate fails closed if the accelerator or either datasets value is unresolved,
or if any bound checkout is not the canonical, clean worktree root whose `HEAD`,
commit object, tree, origin, and required blobs equal the operator-owned seal,
or does not pass its producer's contract tests. The policy is named
`exact_clean_head`: it
proves the exact local Git object reviewed and separately verifies the configured
origin URL; it deliberately does not claim that a remote ref advertises the
commit.
The dependency seal records all four repository origins and all five authority
roles (accelerate, MCP++, datasets ISI, datasets semantic state, and kit),
plus interface/schema fingerprints and the Python 3.12 toolchain. A
deterministic seal validator must check those values; `git
rev-parse` alone is not validation. Workers may not infer a pin from a mutable
branch, current checkout, editable install, submodule pointer, or ambient
`PYTHONPATH` ordering.

The operator also supplies one absolute Python 3.12 executable. The seal binds
its binary SHA-256, exact CPython patch version, the installed pytest version
and complete distribution digest, and a closed environment with no inherited
variables. Every sealed argv names that exact executable. Tests run from a new
mode-0700 full-tree materialization reconstructed from the pinned commit, with a
private HOME and only that materialization on `PYTHONPATH`; source checkouts are
never the execution directories.

The validator, rather than the JSON document, owns the exact five-role order,
repository/origin mapping, commit and tree pins, required path sets, argv-only
test tuples, bounded timeouts, target/wire schemas, and API signatures. Each
complete authority fingerprint binds all of those fields and every required
blob OID. A producer cannot replace its manifest with a smoke test and
re-fingerprint the weakened document. Interface evidence is extracted directly
from pinned Python AST assignments/signatures and JSON vector fields rather
than accepted as a document-authored hash. The entire commit tree is the closed
blob/import/test-dependency fallback, while required blobs name the reviewed
surface. This explicitly binds the kit MCP artifact vector layout and the MCP++
Profile A descriptor fields, Profile B envelope/receipt vector fields, and
Profile F event fields.

A sealed check requires `--run-tests`, `--python`, and a fresh absolute
`--receipt-dir`. Each command gets a distinct private materialization and process
group. Timeout or surviving descendants are fenced and fail the gate. Before
and after every command the validator revalidates all five source roots,
including every tracked working byte and rejection of `assume-unchanged` or
`skip-worktree`. A successful command writes a mode-0600, SHA-256-addressed,
closed per-role producer-test receipt binding argv, toolchain,
environment policy, full-tree closure, stdout/stderr digests, and all five
pre/post roots.

The two datasets authorities share an origin but never a checkout. The final
phase-one and phase-two closeouts must be supplied through distinct clean roots,
as must all other roles:

```text
SCH_ACCELERATE_CHECKOUT  # exact future repaired runtime authority
SCH_ISI_CHECKOUT         # exact final phase-one closeout
SCH_DSS_CHECKOUT         # exact final phase-two closeout
SCH_KIT_CHECKOUT         # exact 05ba937... durable-root authority
SCH_MCP_PLUS_PLUS_CHECKOUT
```

The final manual command is equivalent to:

```text
/home/barberb/lift_coding/.venv/bin/python \
  scripts/validate_semantic_state_dependencies.py \
  --check config/semantic_state_dependencies.seal.json \
  --python /home/barberb/lift_coding/.venv/bin/python \
  --receipt-dir /absolute/fresh/private/sch-receipts \
  --repo accelerate_harness=${SCH_ACCELERATE_CHECKOUT} \
  --repo incremental_semantic_index=${SCH_ISI_CHECKOUT} \
  --repo semantic_state_contracts=${SCH_DSS_CHECKOUT} \
  --repo kit_state_roots=${SCH_KIT_CHECKOUT} \
  --repo mcp_plus_plus=${SCH_MCP_PLUS_PLUS_CHECKOUT} \
  --run-tests
```

An AST audit of the current harness working tree additionally rejects local scanner, symbol
graph, semantic-state view/capsule/selection, durable-root, generic MCP++
envelope/event/receipt, canonical-byte, or CID authority implementations,
including semantic aliases, dynamic imports, reflected mutation, local
canonicalizers/hashers, reversed names, and forged provider method bodies.
Only direct datasets-provider delegation and imports of the two sealed
accelerator wire helpers are allowed. `SemanticCapsuleRef` and
`TestSelectionRef` remain reference/admission records only, and the adapter
opens the datasets-owned `SemanticStateView` through its injected block reader.

## 2. Scope and completion outcome

The release is a complete local loop over a controlled Python fixture repository
or a deliberately selected stable kernel in `ipfs_datasets_py`. It must:

1. obtain a deterministic repository state from the repaired phase-one scanner;
2. consume the datasets-owned symbol-level Merkle DAG and confidence
   classifications;
3. request and reuse datasets-owned semantic capsules for unchanged
   dependencies;
4. consume state deltas and propagate explicit invalidation obligations;
5. select affected pytest tests and configured proof obligations;
6. build a minimum-sufficient, assurance-aware `ContextPack`;
7. create a disposable fenced Git worktree and request a patch from a configured
   provider;
8. validate patch syntax and file scope before applying it;
9. rescan, run static checks, selected tests, and available provers;
10. publish MCP++-conformant, content-addressed receipts and compare-and-swap the
    accepted semantic-state root.

The harness never imports or executes a target repository for analysis. Test and
proof execution occurs only in the isolated validation worktree after the exact
commands, environment bindings, and selected inputs have been recorded.

### Non-goals

- arbitrary language frontends or verification beyond Python 3.12 and pytest;
- automatic rewriting of dependent source;
- model-generated summaries promoted to exact facts;
- ZK proofs or a new prover;
- a network service, MCP server, dashboard, or interactive UI;
- semantic context packing unrelated to this local coding loop;
- replacing the existing agent supervisor, resource scheduler, leases, event
  log, provider gateway, or worktree lifecycle implementation;
- scanning or verifying the whole repository portfolio in the MVP;
- production fallback to mock hardware, mock inference, replayed model text, or
  simulation.

## 3. Authorities inspected and reuse decisions

| Concern | Existing authority | Decision |
|---|---|---|
| Python repository state, symbols, typed graph, Merkle manifest, capsules, deltas, invalidation, explanations, exact source retrieval | Final pinned repaired `ipfs_datasets_py.logic.software_contracts` semantic-state public API | Consume through `SemanticStateProvider`; never reconstruct AST identity, graph semantics, capsule facts, or source identity in accelerate. |
| Semantic content identity | Final pinned semantic-index records and their `software_contracts.content` CIDs | Preserve supplied CIDs and verify round trips; never translate them into a second identity. |
| MCP++ wire shapes | MCP-IDL Profile A, CID-native artifacts Profile B, and Event DAG Profile F at `dc316465…` | Encode a local interface descriptor and conformance vectors against those profiles. No semantic-specific upstream schema was found, so local payload schemas remain namespaced extensions carried by the MCP++ envelope. |
| MCP++ artifact bytes and real CIDv1 | `ipfs_accelerate_py.mcp_server.mcplusplus.artifacts.canonicalize_artifact` plus `ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid.cid_for_bytes` | Use this pair for harness wire artifacts. Do not use the SHA-256 string returned by `artifacts.compute_artifact_cid`. Add vectors proving Kubo-compatible CIDv1. |
| Immutable coordination storage | Lazy `ipfs_kit_py.mcp_server.mcplusplus.coordination_storage.DurableCoordinationStore`, `BlockBackend`, and `IPFSHeliaBlockBackend` | Inject through a small protocol. Local store is hermetic; remote block transport is optional. Do not use Iroh/bucket CAS or the accelerate `VerifiedIPLDBackend` as the semantic root authority. |
| Mutable current-root CAS and WAL | Generic generation-bearing root-CAS surface delivered at the final `ipfs_kit_py` pin | `SCH-003` binds only to `put`, `get`, `get_bytes`, `has`, `read_root`, `compare_and_swap_root`, and `recover`; its concrete import path is sealed in `SCH-000`. |
| Context budgeting and provider-visible source coverage | `agent_supervisor.context.context_compiler.ContextCompiler`, `context_contracts`, and `todo_daemon.production_context_slice` | Project semantic capsules/raw source into existing context references and production slices; do not build a second generic context optimizer. |
| Resource admission and cancellation | `agent_supervisor.runtime.resource_scheduler.ResourceScheduler` and cancellation-aware work contracts | Wrap these types rather than introducing a scheduler. |
| Provider execution | `agent_supervisor.runtime.provider_execution.ProviderExecutionGateway` | Use its reservation/idempotency path, with a stricter harness promotion gate that rejects simulated/degraded/replayed-as-new results in production. |
| Worktree ownership and fencing | `agent_supervisor.worktree_lifecycle.WorktreeLifecycleStore` and `merge.lease_coordination.LeaseCoordinator` | Acquire before `git worktree add`, fence every mutation/publication, and recover by durable attempt identity. The eventual mandatory runtime vector independently binds process birth and process-group cleanup. |
| Proposal and patch admission | `validation.proposal_validation` plus `todo_daemon.production_context_slice.assert_proposal_covered_by_context` | Reuse strict parsing, immutable scope, preimage coverage, and rejection receipts before Git application. |
| Validation execution | `validation.validation_commands`, `validation.validation_runtime`, and `validation.validation_scheduler.ValidationScheduler.run_staged` | The semantic selector supplies explicit commands; do not invoke the legacy `run_impact_selected` selector or add a subprocess scheduler. |
| Proof execution | `proof.proof_scheduler.ProofScheduler` and formal-verification capability records | Reuse capability probing and typed unavailable results; do not build a prover. |
| Restart/replay journal | `runtime.event_log` | Append bounded semantic-session events and checkpoint replay cursors. It is a journal, not a task-queue or semantic-state authority. |

`PersistentTaskQueue` is deliberately not an authority or dependency of this
feature. The harness is one bounded session coordinator over existing resource,
lease, worktree, event, and provider mechanisms.

## 4. Smallest owning package

All focused implementation belongs under:

```text
ipfs_accelerate_py/agent_supervisor/semantic_state/
    __init__.py
    contracts.py
    wire.py
    datasets_adapter.py
    durable_state.py
    scheduling.py
    capsules.py
    context_pack.py
    routing.py
    selection_execution.py
    verification.py
    receipts.py
    worktree.py
    harness.py
    session.py
    cli.py
    benchmark.py
    schemas/
        semantic-state-harness.interface.json
```

Names may be combined only where doing so reduces surface without mixing
authorities. Tests live in the pytest-discovered `test/api/semantic_state/`, the
controlled repository in `test/fixtures/semantic_state_harness/controlled_repo/`,
and the exactly-40-task corpus in `benchmarks/semantic_state/tasks/`.

The root `pyproject.toml` and `setup.py` receive only the matching
`semantic-state` console entry and package the closed interface schema for
`importlib.resources` access. A built-wheel smoke test must prove both the
schema and console entry are present. No legacy CLI or MCP server is expanded.

## 5. Architecture and data flow

```text
Git/tree snapshot
      |
      v
SemanticStateProvider ------------------------------+
  state + Merkle DAG + capsules + delta             |
  + invalidation + exact tree-bound source          |
      |                                              |
      +--> CapsuleAdmission --> DatasetsSelection   |
      |           |                 |                |
      +-----------+--> ContextPacker                 |
                          |                           |
                    RoutingPolicy                    |
                          |                           |
                 SchedulingAdapter                   |
                          |                           |
             Fenced disposable worktree              |
                          |                           |
              patch -> rescan -> verification         |
                          |                           |
                   MCP++ receipts/event               |
                          |                           |
                DurableSemanticStatePort <------------+
                          |
              expected-old CAS state root
```

Every arrow crosses a typed, bounded contract. The scanner remains canonical;
filesystem watcher events only request another scan.

## 6. Closed harness contracts

`contracts.py` defines deterministic immutable records with `to_dict` and
strict `from_dict` behavior. Durable collections are sorted and duplicate IDs
are rejected. At minimum:

- `HarnessMode`: `development` or `production`;
- `WorkKind`: task parsing, scan, capsule compilation, test selection, context
  packing, model invocation, static check, pytest, prover, persistence;
- `Availability`: `available`, `unavailable`;
- `UnavailableResult`: operation, provider/adapter ID, stable reason code,
  retryable flag, and bounded diagnostic text;
- `SemanticCapsuleRef`: an admission-only reference containing the
  datasets-owned capsule CID, producing semantic-state-root CID, stable symbol
  ID, version/source CIDs, confidence, validity bindings, and raw-source
  requirement; it never copies authoritative semantic facts from the capsule;
- `ContextPack`: objective, exact raw target/surrounding/test code, dependency
  capsules, obligations/counterexamples/delta/interfaces/assumptions,
  exclusions, token accounting, risk, route, and escalation recommendation;
- `ModelRoute`: exactly `deterministic_only`, `small_local_model`,
  `medium_model`, `frontier_model`, or `human_review_required`;
- `PatchProposal`: provider identity, mode, base tree/root, unified diff bytes
  CID, declared paths, and generation result;
- `TestSelectionRef`: the datasets-owned selection CID plus its previous
  semantic-state-root CID (or `None`) and current semantic-state-root CID; the
  selected node IDs, proof IDs, reason paths, fallback, universe, and coverage
  facts remain solely in the referenced datasets `TestSelection`;
- `VerificationReceipt`: exact tree/config/dependency/policy/interface/root
  bindings, command identity, selected nodes, exit result, output artifact CIDs,
  simulation flag, freshness, and acceptance eligibility;
- `HarnessResult`: accepted/rejected/unavailable, old/new roots, patch, receipts,
  obligations, event head, and stable reasons.
- `RootRef`: root CID plus monotonic generation used as the complete CAS token;
- `SemanticStateRootManifest`: repository identity, base/candidate Git tree,
  datasets state and Merkle-root CIDs, capsule-index/delta/invalidation/
  obligation/selection/receipt CIDs, dependency/configuration/policy/interface/
  toolchain bindings, event head, versions, and acceptance disposition.

Timestamps, local paths, process IDs, wall-clock durations, and provider billing
observations may appear in operational receipts but never alter a deterministic
semantic-state root. Secret values, prompts, raw model output, and repository
source bodies are never copied into small scheduler observations.

## 7. Semantic-state provider boundary

`datasets_adapter.py` is the only import boundary to `ipfs_datasets_py`. It
loads the pinned dependency lazily on first operation, checks the expected
contract/schema/extractor versions, and exposes:

```python
class SemanticStateProvider(Protocol):
    def scan_repository(self, repo_path, previous_state=None): ...
    def diff_repository_states(self, previous_state, current_state): ...
    def calculate_invalidation(self, previous_state, current_state, delta): ...
    def explain_symbol(self, repository_state, symbol_id): ...
    def explain_impact(self, repository_state, changed_symbol_ids): ...
    def watch_repository(self, repo_path, callback, *, debounce_ms): ...
    def build_semantic_state(self, semantic_index, *, environment_bindings=(), previous_bundle=None): ...
    def verify_semantic_state_bundle(self, bundle): ...
    def open_semantic_state(self, root_cid, get_block) -> SemanticStateView: ...
    def compile_semantic_capsule(self, semantic_index, symbol_id, *, relevant_bindings): ...
    def assess_capsule_freshness(self, capsule, *, current_state, invalidation=None): ...
    def read_required_source(self, semantic_index, symbol_id, *, expected_producer_state_cid): ...
    def extend_semantic_invalidation(
        self, previous_index, current_index, delta, plan, previous_state, current_state
    ): ...
    def select_tests_and_proofs(
        self,
        previous_state: SemanticStateView | None,
        current_state: SemanticStateView,
        invalidation,
        *,
        policy,
        explicit_rules=(),
    ): ...
    def compare_test_selection_oracle(
        self, selection, *, baseline_full, selected_run, candidate_full, authored_oracle=None
    ): ...
```

The adapter validates returned state/Merkle/capsule/version/selection CIDs and
closed confidence values (`exact`, `conservative`, `heuristic`, `opaque`).
`open_semantic_state` receives only an injected `get_block(cid) -> bytes`
function and yields a verified, read-only, storage-neutral `SemanticStateView`;
no put, CAS, WAL, provider, or network behavior enters the datasets package.
Selection always supplies the previous view (when one exists), current view,
and producer invalidation so deletion and rename evidence is not discarded.
The adapter does not reach into the semantic-index implementation to recover
facts absent from its public records. Missing capabilities produce
`UnavailableResult`, never empty/exact state. Git/tree or deterministic
filesystem snapshots remain authoritative; watch notifications never become
mutations or state. Source retrieval reads the exact scanned Git tree/blob and
verifies the expected source CID. Reading the current filesystem after a scan
is forbidden; a source race produces typed staleness and a rescan request.

The harness consumes the scanner's modules, imports, symbols, signatures,
annotations, decorators, exceptions, state reads/writes, calls, inheritance,
schemas/serialization edges, tests/markers/fixtures/config, generated relations,
proof edges, and source spans. It does not claim complete dynamic dispatch.

Policy and MCP-interface inputs are explicit content-addressed artifacts. A
policy CID change invalidates cached policy decisions. An interface descriptor
CID change invalidates descriptor receipts and client-adapter obligations. These
are harness bindings layered on the phase-one delta; they do not mutate the
phase-one graph authority.

## 8. Semantic capsule authority and admission

The datasets semantic-state producer compiles one capsule per requested symbol
version. Accelerate's `capsules.py` verifies, binds, caches, and projects those
capsules without re-extracting or restating authoritative semantic facts:

- normalized AST facts and stable/version identities;
- public signature, annotations, defaults, decorators, and explicit contracts;
- declared and observed exception/effect sets;
- bounded static relations and schema/serialization relations;
- relevant tests, fixtures, configurations, and proof obligations;
- raw-source CID and spans needed to retrieve exact code;
- the symbol's analysis confidence and every reason it was reduced.

Docstrings are stored separately as non-authoritative hints. Optional model
summaries are stored only in `heuristic_annotations`, bind the exact input
capsule CID and model/provider version, and can never raise confidence or
satisfy a verification obligation.

Capsule reuse requires equality of capsule CID, symbol-version CID, extractor/schema
versions, dependency/policy/interface/configuration bindings, and non-stale
status. `exact` and `conservative` capsules may substitute for unchanged
dependency code with visible caveats. `heuristic`, `opaque`, missing,
invalidated, or unknown capsules force raw source retrieval. `CapsuleCache` is
only an index over `DurableSemanticStatePort`; it does not introduce another
filesystem, database, block, CID, or mutable-root authority.

## 9. Incremental invalidation and obligations

The harness delegates source-semantic invalidation to
`calculate_invalidation` and preserves its explanation paths. It additionally
binds execution artifacts to configuration, dependency lock, policy,
interface-descriptor, provider, toolchain, and proof identities. Admission
implements these explicit rules:

- body changes stale that capsule, direct proof obligations, and relevant
  tests; callers remain reusable if signature/effects/exceptions are stable;
- signature changes stale callers, adapters, interface descriptions, schemas,
  and tests;
- effect changes stale purity/security assumptions and callers bound to the
  prior effect set;
- exception changes stale recovery assumptions and exception-contract tests;
- schema changes stale serializers, deserializers, storage/API adapters, and
  tests;
- dependency/lock changes stale dependent summaries and verification receipts;
- fixture/test configuration changes stale bound test receipts;
- policy changes stale policy decisions and security admission receipts;
- MCP interface changes stale interface descriptions and client adapters;
- opaque behavior emits a raw-source-required obligation.

The output is always a sorted obligation set. No task automatically rewrites an
arbitrary caller, adapter, or schema consumer.

## 10. Datasets-owned test/proof selection and execution projection

Test/proof selection remains datasets semantic authority. Through SCH-002 the
harness calls the sealed `select_tests_and_proofs(previous_state,
current_state, invalidation, *, policy, explicit_rules=())`, stores the returned
datasets `TestSelection`, and carries only a `TestSelectionRef` in harness wire
records. The previous/current root bindings are mandatory so deleted and
renamed-symbol evidence cannot be lost.

`selection_execution.py` verifies and dereferences that selection, then maps
its already-selected pytest node IDs and proof IDs to bounded
`ValidationScheduler.run_staged` and `ProofScheduler` commands. It also enforces
the producer's `none`/`full_pytest`/`full_proofs`/`both` fallback directive and
explicit harness assurance policy. It never traverses semantic edges, chooses a
second affected set, calls `run_impact_selected`, imports/collects target tests,
guesses node IDs, or weakens a producer fallback. The producer-owned reason
paths, ambiguity, unresolved obligations, universe, and coverage facts remain
available through the referenced selection rather than being copied into a
competing harness contract.

`compare-full-suite` runs selected tests and then the same fixture's full suite
as an oracle, then supplies normalized run facts and the referenced producer
selection to datasets `compare_test_selection_oracle`. A false negative is any
full-suite failure attributable to the mutation that was absent from the
selected run. The controlled fixture release gate requires zero false negatives
and reports precision, recall, fallback rate, and selected/full test counts
without redefining an unaffected passing test as a true positive.

Proof execution is optional and capability-probed. An unavailable prover yields
a typed unavailable obligation/receipt; it is never reported as a passed proof.

## 11. Assurance-aware context packing

`context_pack.py` adapts semantic inputs into
`ContextCompiler`/`ContextReference` records and a verified
`ProductionContextSliceManifest`. The existing compiler optimizes minimum
context subject to the additional semantic coverage and assurance constraints.
The resulting pack always contains:

- task/objective;
- exact raw target code and exact surrounding edit context;
- exact directly edited tests;
- reusable semantic capsules for unchanged dependencies;
- unresolved obligations and minimized counterexamples;
- current repository-state delta;
- relevant MCP/public interface schemas;
- explicit assumptions and confidence caveats;
- an explanation for every excluded source region;
- token totals by category and estimator version;
- route and escalation recommendation.

Coverage is a hard constraint, not a ranking score. Exact target/edit spans are
never compressed. Conservative capsules remain visibly conservative. Heuristic
capsules can guide retrieval but cannot replace source. Opaque, invalid, stale,
or unknown facts include raw source. If the budget cannot satisfy coverage, the
packer recommends escalation or human review instead of silently truncating.

## 12. Scheduling and model routing

`scheduling.py` adapts harness jobs to `ResourceScheduler`,
`ProviderExecutionGateway`, `LeaseCoordinator`, `WorktreeLifecycleStore`, and
`runtime.event_log`:

- deterministic task parsing and datasets capsule/summary projection are
  explicit scheduled work kinds; optional model summaries remain heuristic;
- resource admission occurs before parsing batches, capsule compilation, tests,
  provers, and provider invocation;
- leases include attempt identity and fencing tokens;
- cancellation is propagated to queued/local subprocess/provider work;
- terminal outcomes are idempotent and replayable without repeating provider
  charges or publishing stale results;
- restart reads fenced lifecycle records and a verified event cursor;
- unavailable capacity/provider/tooling returns `UnavailableResult`.

The adapter does not use `PersistentTaskQueue` as an authority and does not use
legacy mock hardware or mock inference coordinators.

`routing.py` scores only the declared inputs: context size, lowest relevant
confidence, risk class, affected dependency cone, unresolved obligations,
prior repair failures, and available proofs. Routing results use the five
required classes. Providers are injected by typed capability; none is hardcoded.
`deterministic_only` means no model invocation, and `human_review_required`
halts before invocation or root publication. Production uses
`ProviderExecutionMode.ENFORCE` and requires available coordination, a real
coordinator/invoker, verified provider attribution, and a non-simulated
reservation. It rejects `sim:`/`degraded:` reservations, off/simulated/degraded
phases, fallback reason codes, mismatched effective providers, and replay
without a previously admitted production receipt. Any `llm_router.generate_text`
adapter passes `allow_local_fallback=False` and
`allow_cross_provider_fallback=False`. Development
simulation is labeled in every result and can never satisfy production
verification or state-root acceptance.

## 13. Durable artifacts, MCP++ wire, and freshness

`wire.py` publishes a Profile A interface descriptor for scan/status/graph/
explain/invalidate/select/pack/verify/apply/compare/benchmark operations. Payload
schemas are closed, versioned harness extensions. Calls and results are carried
by Profile B execution envelopes, and session transitions form a Profile F
parent-linked event DAG.

Harness wire CIDs are calculated as:

```text
canonical bytes = mcp_server.mcplusplus.artifacts.canonicalize_artifact(payload)
CIDv1           = mcp_server.mcplusplus.kubo_cid.cid_for_bytes(canonical bytes)
```

The older `compute_artifact_cid` SHA-256 label is explicitly forbidden for new
harness artifacts. Conformance tests compare exact bytes/CIDs with MCP++ and
Kubo-compatible vectors. A phase-one semantic-state CID remains its original
CID and is referenced, not recomputed.

`durable_state.py` defines this narrow protocol:

```python
class DurableSemanticStatePort(Protocol):
    def put(self, artifact, *, expected_cid, codec="dag-json"): ...
    def get(self, cid): ...
    def get_bytes(self, cid): ...
    def has(self, cid): ...
    def read_root(self, repository_id) -> RootRef | None: ...
    def compare_and_swap_root(
        self, repository_id, expected: RootRef | None, new_root_cid
    ) -> RootRef: ...
    def recover(self): ...
```

The local implementation uses the pinned `DurableCoordinationStore` plus the
pinned generic root-CAS/WAL surface and needs no daemon. The optional backend is
lazy and injected. Writes store and verify all immutable blocks first, append a
prepared WAL transition, perform expected-old CAS, mark committed, and expose
the new root atomically. Recovery either completes an already-published valid
root or retains the old root; corruption fails closed. Two distinct concurrent
writers from one expected root cannot both succeed. The monotonic generation is
part of the expected token, so an A-to-B-to-A sequence cannot admit an ABA-stale
writer. The CAS target is always a stored and transitively verifiable
`SemanticStateRootManifest`, never a bare unexplained repository-state CID.
Observed and rejected candidate manifests may remain as immutable blocks, but
only an accepted manifest is reachable from the current `RootRef`. The initial
scan uses an explicit `None -> bootstrap manifest` CAS and labels that manifest
indexed rather than patch-verified; later verification receipts cannot be
inferred from bootstrap state.

`receipts.py` binds every receipt to the exact pre/post tree, semantic roots,
selection, commands, toolchain, dependency/lock/config/policy/interface CIDs,
provider mode, and output blocks. A receipt is admissible only when all bindings
match, all required stages passed, all required artifacts rehash, the event
parent is current, and `simulation == false`. Stale receipts remain inspectable
but can never satisfy acceptance.

## 14. Fenced isolated patch workflow

`worktree.py` composes `managed_git_worktree`/`GitWorktreeSession`,
`WorktreeLifecycleStore`, `LeaseCoordinator`, strict proposal validation, and
the production-context preimage gate. It asserts that the worktree base is the
exact scanned commit/tree; if the shared helper cannot select that base, only a
narrow reviewed base-ref extension is permitted. `worktree.py` and `harness.py`
implement exactly this acceptance sequence:

1. acquire a fenced attempt and create a disposable Git worktree;
2. materialize the accepted `ContextPack` by CID;
3. invoke the configured model adapter through the production gate;
4. validate the untrusted proposal through `proposal_validation`, prove every
   patch preimage was visible in the bound production context, parse a bounded
   unified diff, and run `git apply --check`;
5. reject paths outside the declared allowlist, binary patches, symlink escapes,
   submodule changes, and control/runtime/state paths;
6. apply the already-checked patch with Git in the disposable worktree;
7. rescan and identify actually changed symbols;
8. recompute the delta and invalidation obligations;
9. run declared static checks;
10. run selected pytest nodes;
11. run configured available proof obligations;
12. optionally run the full suite as oracle, mandatorily when policy escalates;
13. store the graph, capsule index, context pack, delta, obligations, patch,
    immutable receipts, next event node, and complete root manifest;
14. compare-and-swap the stored manifest CID using the prior `RootRef` only
    after every acceptance gate passes.

Rejection leaves the current root unchanged and records a bounded rejection
receipt. Cleanup is fenced and recoverable. No accepted source commit is made in
the user's working checkout; the result identifies the retained/cleaned
worktree and patch artifact according to configured policy.

## 15. Incremental sessions and watcher behavior

`session.py` coordinates repeated local runs. Watch callbacks debounce only to
schedule a fresh canonical scan. Concurrent watchers coalesce equal snapshot
CIDs, and fenced attempts prevent stale callbacks from publishing a root.
Restart replays event-log pages, verifies immutable artifacts, reconciles WAL
state, and resumes only nonterminal attempts whose lease/fence still matches.

No watcher event, task queue row, DuckDB projection, model assertion, or receipt
alone is semantic truth. The pinned scanner state and current-root CAS are the
truth boundaries.

## 16. CLI

The dedicated `semantic-state` command exposes deterministic JSON by default:

```text
semantic-state scan <repo>
semantic-state watch <repo>
semantic-state status <repo>
semantic-state graph <repo> [--symbol ID]
semantic-state explain-symbol <repo> <symbol>
semantic-state explain-impact <repo> <symbol-or-file>...
semantic-state invalidate <old-state> <new-state>
semantic-state select-tests <repo> <symbol-or-file>...
semantic-state pack-context <repo> <task> <target>
semantic-state verify <repo> [--full-suite]
semantic-state apply-patch <repo> <patch-or-task>
semantic-state compare-full-suite <fixture-or-repo>
semantic-state benchmark [--corpus PATH]
```

Commands that need an unavailable optional dependency/provider return a stable
typed error on stdout/stderr and nonzero exit. Production `apply-patch` never
falls back to simulation. CLI imports do not start watchers, processes,
databases, daemons, networks, or package installation.

## 17. Controlled fixtures and acceptance matrix

The fixture repository is small enough for a full-suite oracle and contains
committed snapshots/mutations for:

1. local function body change;
2. public signature change;
3. cross-module call change;
4. dataclass/schema change;
5. exception behavior change;
6. fixture dependency change;
7. pytest configuration change;
8. dynamic import;
9. monkey patch;
10. opaque native dependency;
11. unrelated formatting change;
12. deleted symbol;
13. renamed symbol;
14. generated file;
15. stale receipt;
16. failed root CAS;
17. interrupted state transition;
18. concurrent watchers/writers;
19. out-of-scope model patch.

The same controlled repository also supplies explicit end-to-end mutations for
side-effect/security-policy changes, dependency/lockfile changes, policy
changes, MCP interface/client-adapter changes, and a post-scan source race.

The release suite proves bounded invalidation, all known dependent
invalidations, raw-source fallback for opaque behavior, stale-receipt rejection,
zero controlled-fixture test-selection false negatives, full-suite fallback,
deterministic state roots, safe recovery, single-winner CAS, and the production
simulation gate. It also runs cold-import tests with all automatic install
features disabled and checks that imports do not change environment, create
files/processes, or access the network.

## 18. Exactly-40-task benchmark

The checked-in corpus contains exactly 40 reproducible maintenance tasks:

| Task type | Count |
|---|---:|
| Small bug fixes | 10 |
| Test repairs | 6 |
| API adapters | 6 |
| Schema migrations | 6 |
| Multi-file changes and refactors | 6 |
| Expected rejection or frontier/human escalation | 6 |

Every task records baseline raw/retrieval context tokens, semantic `ContextPack`
tokens, exact raw code excluded, capsule count, invalidation cone, selected and
full tests, proof obligations, model route, acceptance/rejection, assumptions,
and uncertainty. Checked-in candidate patches are replay/oracle fixtures only:
they are always marked `production_eligible=false`, never produce a model
receipt, and never advance a production root. Candidate verification outcome is
reported separately from production acceptance. Both context modes use the
same pinned tokenizer/estimator and
coverage policy. Required target/test/opaque source cannot be removed to improve
the result.

The initial gates are at least 30% median input-context reduction overall, zero
known stale receipts admitted, zero controlled-fixture selection false
negatives, deterministic state roots, and all uncertainty represented in each
pack. Results include reduction by task type, precision/recall, fallback rate,
latency by stage, artifact bytes, and route distribution. `--check` compares
only deterministic semantic fields; wall-clock latency is explicitly
observational. Failed and escalated
tasks remain in the denominator.

## 19. Parallel delivery plan

The exact dependency DAG is encoded in the taskboard. Its waves are:

```text
A0  SCH-000
A1  SCH-001 | SCH-002 | SCH-003 | SCH-014
A2  SCH-004 | SCH-006 | SCH-016
A3  SCH-005
A4  SCH-007 | SCH-008 | SCH-010
A5  SCH-009
A6  SCH-011
A7  SCH-012 | SCH-017
A8  SCH-013 | SCH-015
A9  SCH-018
```

Task output ownership is deliberately narrow. The three reviewed control files
are protected from implementation workers. Parallel work may proceed only after
`SCH-000` is manually sealed with the final dependency commits.

## 20. Release evidence and honest limitations

The final report is generated from committed receipts and benchmark results and
must include architecture, exact packages/modules, commands/examples, tests and
results, benchmark and task-type token reductions, test-selection precision and
recall, known unsoundness/opaque cases, performance bottlenecks, and exact work
remaining before ZK aggregation and production integration.

Expected limitations remain explicit: Python reflection and dynamic dispatch
can be opaque; static call/test selection is bounded rather than complete;
pytest plugin behavior can require full-suite fallback; proof tools may be
unavailable; token estimates depend on a declared estimator; external network,
filesystem, and native effects cannot be proven absent by this analyzer. These
limitations lower confidence or force source/full verification. They never
become exact claims.

No ZK aggregation is implemented. Future ZK work requires a frozen receipt
circuit/input schema, deterministic verifier semantics, proof-system and setup
selection, aggregation rules, key lifecycle, and independent security review.
Production integration additionally requires supported live provider adapters,
credential isolation, sandbox hardening, resource quotas, platform-specific
worktree/process tests, release signing, migration/version policy, and an
operational rollback/recovery runbook.
