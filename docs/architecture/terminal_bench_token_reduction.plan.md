# Terminal Bench token reduction using the existing supervisor infrastructure

Status: implementation plan, based on the canonical workspace audit of 2026-10-11. This document adds no execution, source-substitution, proof, or completion authority. The next outcome is a measured reduction in the tokens needed to complete the same Terminal Bench tasks with the same acceptance checks.

## Implemented measurement seams (2026-10-12)

The first measurement changes are available in the existing runtime/context
interfaces: a pure Codex cumulative-session reconciler, corrected native
lifecycle completion handling, and optional counter forwarding through
`compile_manifest_to_capsule`. The
[accounting and counter guide](../semantic_state/TOKEN_USAGE_RECONCILIATION.md)
documents their entry points, scopes, missing-data behavior and 138 fresh guarded
tests. Historical completion flags remain unverified, cached/reasoning/tool
attribution is not double-counted, and compiler totals remain conservative.
The standard Harbor/supervisor path now forwards an explicitly supplied
container-local descriptor/hash and existing budget to planning and model-routed
coding before the signed launch is created. Final reports and Harbor metrics
use the shared session reconciliation. Conflicting log receipts remain visible,
and accounting failures preserve readable task results.

The standard launch path also forwards the opt-in versioned identifier policy
and estimator-only role projection before signing the coding command. Planning
and provider-free Doctor routes keep their existing behavior. The router
compares complete native, legacy, V2, projected V2 and eligible packed messages; a supplied
counter must show strictly fewer tokens before an alternative can win. See
the [compact launch guide](../semantic_state/COMPACT_CONTEXT_LAUNCH.md).

These changes establish measurement and representation plumbing. Production
tokenizer/owner composition, portable asset staging, a registered expansion tool
and a matched provider token-savings comparison remain open. The
[compact semantic design](terminal_bench_compact_semantic_projection.design.md)
specifies existing projection reuse, explicit codec revisions and preserved
required coverage. Broader evidence projection still requires explicit coverage
rules; the implemented policy defers only estimator telemetry.

Lossless reference columns are also implemented as a separate transport
revision, preserving existing table/CID and identifier policies. The opt-in V2
router derives packed alternatives from verified encodings and keeps the old
tie preferences. In one complete-message fixture, packed V2 shrank from 32,006
to 30,122 bytes (5.9%), including instructions; one helper invocation took
15.6 ms. **237 tests passed** under a separate unchanged 166-path guard, with
an independent exact pre-V2 replay. Provider-token savings remain unknown.
See the [reference wire guide](../semantic_state/REFERENCE_WIRE_PACKING.md)
and [current packing evidence](../../../../artifacts/supervisor-reference-packing-20261012/README.md).

## Objective and evidence boundaries

Keep the complete source, task, metadata, capsule, and verification evidence in the existing databases and content-addressed stores. Send each supervisor model only the current mandatory task context and selected relevant evidence. Resolve additional evidence through bounded, authenticated expansion when needed. Count the costs of every expansion, tool result, retry, planning call, and coding call before claiming a reduction in total usage.

The preceding native preparation work established race and source checks, and measured approximately 99 ms per authored fixture preparation. It did not establish token savings. Its two native root reads account for most preparation time; reducing those checks is outside this plan. Preparation latency and model token consumption are separate measurements.

Available compression evidence has limited scope:

| Evidence | Observed result | Limitation |
| --- | --- | --- |
| Retained local probe, [independent byte/identity observation](../../../../artifacts/supervisor-token-context-audit-20261011/terminal-bench-token-reduction-review-20261011/retained-byte-observation-r3/README.md) | Final prompt bytes 27,619 to 16,726, approximately 39.44% fewer bytes; recorded aliases restore the original prompt exactly | A 429-byte toy source, two complete capsules, and a 14,760-byte compact worker artifact; complete model-token counts and provider/session usage are unknown. Historical byte/identity verification does not revalidate producer admission or current source freshness. |
| Checked-in 40-task [semantic compression benchmark](../benchmarks/semantic_compression_harness_results.md) | Approximately 58.9% median estimated reduction | `sch-fixture/token-estimator@1`, synthetic capsule-cost bounds, no production-eligible candidates or model receipts. This is not a Terminal Bench savings measurement. |
| Native owner/materializer validation | Source, root, admission, and refusal behavior exercised with an authored local tokenizer | Fixture authority and tokenizer compatibility do not qualify a production launch or remote model. |

Byte savings can identify a promising transport. They cannot establish savings for a tokenizer, billing, or the complete agent session. No global savings target or successful-task claim is inferred from these observations.

### Canonical source, compact views, and edits

Keep three interfaces distinct: exact restoration from retained original source
bytes; compact read views bound to current source; and supported mapped edits
translated into an isolated canonical-source candidate. Retaining a complete
repository permits original-byte restoration. Arbitrary edits to a minified
tree still need an inverse transformation, which the current narrow mapped-edit
interfaces do not provide for general programs. Neither property establishes
token savings. A compact view helps only when it displaces larger model input
after counting its legend, bindings, wrappers, source expansion, and later replay.

Use DuckDB joins and admitted Quack operations to retrieve the task's source,
symbols, dependencies, interfaces, open obligations, counterexamples, and proof
handles. Keep DuckLake history and full capsule provenance available for audit;
send a bounded decision view and authenticated expansion handles to the model.
Task-specific mandatory graph closure must precede retrieval ranking. Rank only
independently optional evidence; an unknown dependency cannot justify omission.

Intent, Legal, and Security IR have explicit typed roots in the current context
contracts. Map Codebase IR nominations into source-backed program, symbol, and
dependency facts; an explicit UI/UX IR adapter remains a gap. Learned outputs
nominate facts and relevant evidence. Source validators and applicable proof
checkers establish admitted claims. A latent vector alone is neither a compact
source program understood by the supervisor model nor a proof of behavior.

The acceptance experiment must hold task, source tree, model, and verification
checks fixed and retain failed tasks. Measure complete provider input plus
output over planning, coding, expansions, and retries. Local tokenizer counts
can select a representation and expose replay break-even; they cannot replace
that whole-task measurement.

Preserve the canonical datasets source tree at `/home/barberb/lift_coding/external/ipfs_datasets` and all import-origin checks. Do not edit the HACC checkout or `hallucinate_app`, download weights or tokenizer assets, increase context windows, or change temperature from 0. Only an actual `lake build <Lib>` grants a Lean admission. A stored row, capsule, compile, solver result, reconstruction loss, or successful transport round trip does not. The Constitution remains unformalized and receives no `roundtrip_ok` promotion.

## Reuse the existing components

| Component | Existing seam | Intended use |
| --- | --- | --- |
| DuckDB metadata | [`SupervisorMetaIndex.compose_for_subject`](../../ipfs_accelerate_py/agent_supervisor/runtime/supervisor_meta_index.py) | Join bounded task/content identities to AST, BM25, graph, vector, proof, and capsule catalogs. Return nominations and handles rather than full indexes. |
| Quack owner | Existing `QuackStateServer`, [`QuackStateClient`](../../ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py), and typed owner gateways | Coordinate closed, named, parameter-bound operations through the already admitted owner; preserve generation, namespace, and fence checks. Do not open a competing owner or pass database credentials to model children. |
| DuckLake | `SupervisorMetaIndex.project_ducklake` | Preserve observational metadata/history for analysis and cross-system discovery. History is not current admission or live control authority. |
| Database context | [`DatabaseContextManifest.model_packet`, `LLMContextFrontier`, `build_context_delta`, `DatabaseContextStore.page_members`](../../ipfs_accelerate_py/agent_supervisor/context/database_context.py) | Project task history into invariant context, distinct current failures, changed paths, pending obligations, selected references, and paged omitted context. |
| Semantic producer | [`IpfsDatasetsSemanticStateProvider`](../../ipfs_accelerate_py/agent_supervisor/semantic_state/datasets_adapter.py), source retrieval, freshness assessment, `admit_capsule` | Authenticate nominated dependencies against the complete current producer view before substitution or use. |
| Capsule selection | [`select_worker_capsules`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_capsule_selection.py) | Preserve native capsule bytes and admissions while choosing a bounded relevant set. Improve relevance using existing indexes and dependency links. |
| Context compiler | [`ContextCompiler`](../../ipfs_accelerate_py/agent_supervisor/context/context_compiler.py) | Preserve the mandatory core, rank optional evidence, enforce complete-input budgets, expose expansion handles, and compile parent-bound retry deltas and stable prefixes. |
| Reversible representation | [`semantic_router_translation`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py) | Alias repeated typed producer identifiers and restore them exactly. Retain literal code, names, comments, paths, signatures, and requirements. |
| Exact prompt selection | [`select_model_prompt`](../../ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py) | Compare the full rendered native and encoded user messages with the explicit provider/model counter; use encoding only when strictly smaller. |
| Deterministic work | [`PlannerDoctorContext`](../../ipfs_accelerate_py/agent_supervisor/context/planner_doctor_context.py), Terminal Doctor dispatch | Complete eligible local work through existing validation, or send only its unresolved residual to a model. A candidate remains subject to ordinary publication and completion checks. |
| Replay | `TerminalAttemptIndex.reserve`, `resolve`, and `publish` | Avoid duplicate provider work for the exact replay identity; changed source, model, owner, tokenizer, or context requires current checks and the applicable new identity. |

The meta-index explicitly refuses attaching exclusive `control.duckdb`; retain that rule. A `quack://` locator is not permission to run arbitrary SQL. The external Quack facade still documents unqualified owner-dispatch operations that fail closed; this plan must use existing admitted operations or record a production gap. No new remote SQL or generic update endpoint is required for compression.

## Current integration gaps

1. The standard Terminal Bench path accepts and forwards `--prompt-tokenizer-artifact`, `--prompt-tokenizer-sha256` and the existing token budget for already deployed child-local assets. Explicit assets retain priority. The automatic resolver now uses reviewed installed model mappings and cached ranks where supported, otherwise labels its fallback as an estimate. Grok and unmapped Codex models must not inherit an exact-compatibility claim from that fallback. The archive builder does not yet stage portable tokenizer assets/runtime, and production compatibility has not been established. Estimated selection is available, but its budget guarantee remains false; final provider usage is a separate measurement.
2. [`implementation_daemon`](../../ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py) currently wraps the complete semantic, code-retrieval, and intent-world artifacts in `required=True` text chunks. The context compiler cannot defer an optional retrieval hit or provenance detail inside a required chunk. Removing the whole chunk or changing all three to optional would lose required coverage.
3. [`semantic_context_runtime`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py) combines manifest, complete capsules, admissions, raw sources, a context pack, reconstruction, and refresh provenance in one worker JSON payload. Much of this is necessary retained evidence; not all of it must be repeated in the model's decision input. Native capsules can contain a full version-evidence AST and exceed the source text they describe.
4. Database context member budgeting still uses `_estimate_tokens`; `compile_manifest_to_capsule` now accepts an optional tokenizer/counter and forwards it to the compiler. Remaining work is caller wiring under the production owner binding and final provider-bound remeasurement. Initial member estimates and conservative capsule accounting retain their distinct scopes; the helper still returns only the capsule, without the compiler receipt. Do not relabel `manifest.total_tokens` or additive member counts as an exact count of the full wire.
5. The router identifier codec consumes the existing compiled worker-evidence format. The newer `OwnerContextMaterializer` harness packet is a different format. Supporting both requires an explicit versioned adapter with exact restoration tests, not silent format substitution.
6. There is no demonstrated complete production composition of source/import/environment owner admission with model routing, calibrated context selection, expansion tools and whole-session usage reconciliation. Local fixture assets are available; exact production tokenizer compatibility is not established. Cached foreign-model tokenizers may inform explicitly labelled estimates, never exact compatibility or provider usage.

### Explicit capsule decision packet (2026-10-13)

The [native selection measurement and reporter update](../../../../artifacts/supervisor-token-accounting-20261013-02/README.md)
now separates retrieval and representation effects with an installed offline
tokenizer. In its retained authored fixture, the complete full native input is
21,231 proxy tokens; the two known reachable capsules use 6,895, or 5,292 with
V2 identifier aliases, while every raw source/task file remains present. The
75.07% first-input estimate does not establish provider or whole-task savings.
The lexical seed omits a known helper, and even the known closure retains an
unresolved `global:helper_operation` edge. Neither is admitted as complete
task-specific mandatory coverage. This supports prioritizing source-backed
task/dependency selection while keeping omission policy explicit.

The residual global edge is a binding-state obligation, not a duplicate helper
identity. The Python extractor emits both `CALLS` to the known implementation
and `READS_STATE` to the global binding used to obtain it. A selected definition
does not establish that reassignment or dynamic mutation cannot change the
binding. The next source-backed adapter should consume verified typed links
through `SemanticStateView.semantic_links_for_symbol`, preserving each relation,
edge identity, source span, extractor, and root-bound fact/version identities.
Represent binding reads separately, identified by snapshot/root, owning module,
and name; `global:name` alone is not a cross-module identity. Retain declaration,
import, reassignment, and dynamic-mutation evidence or explicit uncertainty.
Attach those facts through `build_semantic_dependency_graph(nodes=..., edges=...)`
and join admitted task, acceptance, test, and policy requirements before optional
ranking. Validate module collisions, shadowing, rebinding, stale source, and
missing evidence. Declared call-graph coverage is not semantic or proof closure.

### Implemented typed source dependency proposal

`runtime/semantic_typed_dependency_proposal.py` now projects verified native
relations into the existing `SemanticDependencyGraph` contract. It reuses
`audit_selected_capsule_relations` and the native typed-link accessor rather
than deriving semantics from flattened capsule dependencies. Call observations,
state reads/writes, artifact targets, and unresolved binding obligations remain
distinct, with native edge/link/fact/version identities, spans, extraction
provenance, and metadata hashes. Module-global binding identities include the
source root, module path, and binding name. Other unknown references remain
edge-specific. Recursive calls use observation nodes rather than direct
self-edges.

The nomination traverses every recorded outgoing resolved-symbol relation from
explicit seed IDs, including proof, test, fixture, and serialization relations.
Incoming callers and other incoming relations remain in the audit; they are not
automatically declared task requirements. All root symbols and typed links remain
in the body-free graph. Native analysis limitations, unresolved state, artifact
coverage, and task relevance stay explicit. Bounds refuse a successful partial
nomination: at most 32 nominees, 2,048 graph nodes, 8,192 graph edges, and a
2,000,000-byte complete report. Large codebases still need admitted task scopes
and indexed/sharded discovery; these limits do not qualify whole-repository
semantic closure.

`runtime/semantic_typed_source_context.py` adds the current-source boundary:

```python
proposal = project_source_bound_typed_links(
    repository=repository,
    artifact=".runtime/full/worker-context.json",
    expected_sha256=independently_held_artifact_hash,
    task_id=task_id,
    expected_semantic_root_cid=independently_held_source_root,
    graph_root_id=decision_graph_root,
    seed_symbol_ids=(explicit_target_symbol_id,),
)
```

It requires the complete native worker artifact and exact full capsule
population. It verifies task, root, retained producer, complete source frontier,
and current bytes through the existing native verifier before and after
projection. Every graph symbol must bind to the manifest's exact path/source CID.
Declared required raw paths must be unique manifest members with literal source
delivery. The result carries the original required/raw-source path inventory and
manifest, without copying source bodies into the graph. Source freshness means
the invocation's opening/closing checks passed; dispatch still needs its normal
current checks.

Every generated node and edge has `PROPOSAL_ONLY` authority, and every edge is
nonmandatory. Verified native custody does not admit task coverage. The adapter
creates no decision policy, proof, execution permission, or source-omission
authority. An owner-held task policy must independently join these facts to
explicit task/acceptance/test/policy obligations before existing proof-directed
retrieval and decision-context compilation. Only that independently admitted
policy may classify capsule bodies as required or optional. The bounded bridge
below consumes that classification; it does not infer it from graph membership.

The retained native integration nominates the target, helper, and linked public
acceptance test from eleven symbols, while retaining both unresolved global
binding obligations and the original raw-source inventory. This is source/graph
integration evidence, not a model token or Terminal Bench outcome measurement.
The [typed source qualification record](../../../../artifacts/supervisor-typed-source-context-20261013-01/README.md)
records exact source versions, focused controls, and validation limits.

### Typed dependencies within owner policy (2026-10-14)

`runtime/semantic_typed_owner_policy.py` now connects the current-source proposal
to the existing owner nomination and launch-policy formats:

```python
result = build_typed_owner_decision_nomination(
    native_prompt=final_compiled_native_prompt,
    repository=canonical_source_repository,
    workspace=allocated_candidate_workspace,
    nomination_artifact=independent_owner_declaration,
    nomination_sha256=independently_held_declaration_hash,
    task_id=admitted_task_id,
    expected_parent_sha256=independently_held_parent_hash,
    expected_semantic_root_cid=independently_held_source_root,
    graph_root_id=decision_graph_root,
)
```

Every owner-required capsule supplies a graph seed. The declaration must
completely partition the actual native capsule inventory. Known outgoing
dependency nominees classified optional are added to immediate optional
selection in owner order. Required and optional classifications, their order,
and the existing selected-optional prefix remain exact. A disconnected required
capsule remains selected; graph relevance cannot demote it. Unresolved binding
obligations remain in the separate receipt, without being resolved by a known
helper or turned into proof/task authority.

The result contains detached `nomination`, `launch_policy`, and `receipt` values.
Its nomination uses the existing `supervisor-semantic-decision-nomination@1`
schema. Retain the complete receipt outside model input; serialize the nomination
to an owner-controlled file outside the candidate and pin the SHA-256 of those
exact bytes. Existing host nomination-source delivery or child-local owner-time
binding can consume it. The library writes no file and launches no worker.
Normal owner publication and worker loading still perform their current checks.
After building the result, the bridge rechecks native current source/parent
bytes and the independently held owner declaration. Changes refuse success.

The [owner-policy qualification record](../../../../artifacts/supervisor-typed-owner-policy-20261014-01/README.md)
records **54 passed focused tests** with normal mandatory AST seals and a separate
native integration. The authored full-parent fixture has ten capsules: two
required and eight optional. It retains one prior optional selection and adds
the helper and public acceptance-test bodies. The real owner publishes the
descriptor, and the real worker loader reconstructs the exact selected policy.
Literal raw source, admissions, required paths, and manifest remain unchanged;
both global binding obligations remain unresolved. Complete rendered user
inputs measure 19,150 versus 17,604 offline `o200k_base` proxy tokens (8.1% less).
There is no model call, acceptance outcome, remote tokenizer qualification, or
whole-session savings result.

That first qualification required a complete native parent and a revision `@1`
owner declaration. The subsequent projection-aware implementation below reuses
the complete retained source graph while keeping the actual worker partition
distinct. A full-root nomination still cannot be applied silently to a
shortened inventory.

### Full source graph with query-selected workers (2026-10-14)

`project_worker_source_bound_typed_links(...)` now reads the complete native
graph from the query-selected preparation's existing immutable blocks. It
accepts the same explicit artifact/hash, task, source root, seeds and bounds as
the full-only source wrapper. The original preparation already retains every
native source and capsule-index block; this join requires neither another scan
nor a fabricated full worker artifact. The full-only API continues refusing
projected parents and preserves its original result contract.

The new result separates `full_symbol_ids` from the ordered `worker_inventory`
actually delivered. Its `worker_projection_binding` validates the root capsule
index, exact full/selected/omitted counts, selected symbol order and native
path/name/identity facts, count and byte bounds, and the complete optional raw
source-fetch frontier. The graph's symbol source CIDs must match the complete
current manifest. Native source/producer verification still runs before and
after projection. These are custody and source observations, not admitted task
coverage or independently reconstructed admissions.

`build_typed_owner_decision_nomination(...)` uses this wrapper when its actual
compiled native parent contains a supported `worker_projection`. The owner
policy must partition that actual inventory, not the larger root. It checks
each delivered symbol/capsule identity against the full graph and traverses
recorded relations from every owner-required seed. A known nominee absent from
the worker causes explicit refusal before launch-policy construction, even
when its raw source remains present. It never silently drops the dependency,
backfills a capsule, or mutates the parent. Available optional nominees remain
within the existing order-preserving selection policy.

Revision `@2` is now supported for authentic projected parents. The existing
launch/packet builders validate exact instruction/query equality and the
separate objective/pack-objective equality. The original source path and literal
targets remain intact; no whitespace normalization or revision downgrade is
performed by the bridge. The existing full-parent route remains revision `@1`.

The [projection-aware qualification record](../../../../artifacts/supervisor-typed-projected-owner-20261014-01/README.md)
records **192 passed tests**, including 46 new join cases and the existing
full-parent/task-text suites, with normal mandatory AST seals and 30 unchanged
source guards. One actual query-selected preparation retains a ten-symbol root
and delivers five capsules under the existing 32,768-byte projection ceiling.
Real owner publication, worker `@2` policy loading, and exact selected required
context restoration pass. A target-only control reuses the same native bundle
and freshly packs its own admissions; it refuses the missing helper/test
nominees despite complete raw-source delivery. Both unresolved global binding
obligations remain explicit. There are no model calls or acceptance outcomes.

This fixture also exposes representation overhead: all five delivered capsules
remain selected, so the plain `@2` packet grows the complete local user-input
estimate from 12,041 to 14,072 offline proxy tokens (16.9%). It does not demonstrate
token savings. The existing dialogue compares complete native and first-round
packet inputs and chooses native when it fits and is no larger. Preserve that
measured selection rather than forcing packet framing when little can be
deferred; whole-task savings remain dependent on reconciled live usage.

An actual offline replay on that same published projected `@2` policy confirms
native selection: 12,041 estimated tokens versus 14,449 for the complete first
packet round, under per-call and total ceilings of 12,041. One explicit synthetic
callback receives the exact original prompt; current source checks remain in
place. There are no expansions, new scans, CAS writes, model calls, or provider
usage claims. The replay and immutable pins are retained with the qualification
record above. This preserves the existing measured native-input behavior; it
does not add a second selector or establish live whole-task savings.

### Pre-worker owner hook and retained lineage (2026-10-14)

The Terminal nomination route now wires typed selection into
`prepare_decision_owner_dispatch(...)`, after the owner authenticates final
compiled stdin and before it publishes the worker launch descriptor. The
original independently pinned declaration remains the input; callers need not
construct or stage a derived nomination themselves. The bridge preserves the
owner partition and prior selection while adding available known dependencies.
Missing delivered dependencies still refuse before publication. Required seeds,
unresolved bindings, literal source, task text, and native parent remain exact.

This optimization defaults on for nomination source/artifact routes. The public
`--disable-semantic-typed-nomination` switch and Python
`disable_semantic_typed_nomination=False` option are carried through Harbor,
preparation configuration, and the terminal driver. The signed implementation
argv emits the explicit owner pair `--semantic-decision-typed-nomination`
`enabled|disabled`; it is consumed before the worker starts. Prebound-policy and
native-input routes retain their behavior. A legacy direct owner call without
the new pair remains disabled. Requested configuration and the independently
computed `semantic_decision_typed_nomination_expected` report field remain
separate from printed owner observations.

Owner publication writes the derived nomination and complete typed lineage to
immutable content-addressed files outside the candidate. The existing compact
owner receipt retains `nomination_artifact` and `nomination_sha256` as the
**original declaration** and keeps final task, stdin, parent, launch-policy, and
selected-policy commitments. Its separate `typed_nomination` binding pins the
derived nomination and lineage files by exact path and SHA-256. The receipt
remains bounded at 65,536 bytes; the full lineage is bounded at 2,000,000 bytes
and contains exact origin bytes plus the typed result, including graph evidence.

`runtime/semantic_typed_nomination_lineage.py` reloads actual published files and
replays native graph selection to verify the transformation. It checks exact
owner partitions, the prior selected prefix, optional append order, source/CID
bindings, derived nomination, and final descriptor. Live owner admission checks
current source. Historical input auditing replays retained native/CAS evidence
with current freshness explicitly false; a printed verification marker does
not establish this join. Final context auditing now precedes usage accounting,
which consumes only the observer's compact verification subtree. Omitted or
foreign verification retains observed invocation evidence while withholding
typed owner coverage. Neither graph traversal nor lineage custody grants proof,
task acceptance, or completion authority.

The selected runtime archive/source-pin inventory now declares **31 sources**,
including all three typed modules, the lineage verifier, relation audit leaves,
and dependency graph source. **217 distinct cases passed**, comprising 102 new
cases and 115 existing/compatibility cases, with mandatory fresh AST seals.
The native four-file probe automatically publishes original-to-derived lineage
for a ten-symbol root and five-capsule worker, loads the actual `@2` descriptor,
restores required context, verifies historical leaf-observer joins, and checks
source drift, explicit opt-out, and missing dependency refusal. All 58 native
implementation pins and both source/CAS input inventories remain unchanged.
Its one synthetic callback makes no provider call. Native rendered input costs
11,971 offline proxy tokens; the complete first packet round costs 14,397, so
the existing dialogue chooses native. Whole-session savings remain unknown.
The [handoff qualification record](../../../../artifacts/supervisor-typed-owner-handoff-20261014-01/README.md)
retains exact passing subsets, failed import-guard attempts, sources and evidence.

The [fresh archive qualification](../../../../artifacts/supervisor-typed-container-qualification-20261014-01/README.md) now verifies every captured source/extension tar member and both runtime profiles. The post-cache archive has 12,721 members and 35/38 admitted source pins (normal/retrieval-reference profiles). Actual signed container owner/worker handoff including refusal and interruption controls, a complete historical Terminal audit, and a matched provider trial remain separate qualification gates. Complete-session savings still require unchanged task acceptance and reconciled provider usage. Full Terminal context-audit qualification needs all three admitted context kinds and their retained preparation, admission and base-capsule artifacts. Its current retrieval-scope check requires
exactly `bottle.py`; the operations fixture does not exercise that complete audit
or its bounded subprocess. These library and synthetic callback records do not
qualify signed deployment, provider usage or task acceptance. Existing frozen
campaign artifacts remain historical evidence.

### Fresh-reader index reuse (2026-10-14)

A genuine Bottle scan retains 531 capsules and delivers one 27,276-byte worker context in the explicit lexical qualification. Signed owner diagnostics exposed CPU work before owner publication: full-root typed relation checks repeatedly reconstruct complete sorted fact, capsule, node, and link indexes even for a small delivered subset. The datasets API now reuses up to 16 immutable parsed indexes, each backed by at most 1 MiB of returned bytes, keyed by the exact CID and returned builtin bytes. Every lookup still calls its actual block reader. Changed, missing, foreign, or non-byte answers cannot use a prior parsed result; cache misses retain the original CID and native schema verification. The current-source opening/closing fences and original-to-derived owner lineage checks remain in force.

This optimization defaults on; `IPFS_DATASETS_SEMANTIC_INDEX_CACHE=0` opts out. 56 API, binding, and freshness cases passed with fresh mandatory AST seals, including warm-reader tampering/missing/foreign-index refusals, enabled/disabled native parity, and parsing reuse with a fresh reader call on every lookup. The immutable archive was rebuilt after this change. A matched check on the actual 531-entry, 69,114-byte native index measured 100 fresh-reader lookups at 0.3411/0.3515 seconds with reuse disabled and 0.01072/0.01128 seconds with reuse enabled (31.82×/31.17×). Each enabled group includes one parse miss and 99 hits, and each group preserves all 100 backend reads. These local timings do not qualify full owner or inference latency. The fresh post-cache signed run still exceeded its 90-second verification window before owner publication; STOP drained the process tree and the container was removed. The next bottleneck is repeated full-bundle and fact/capsule/link verification in graph nomination and lineage replay. Full derived-policy worker dispatch and the bounded historical observer remain unqualified. The failed cold runs and interrupted lifecycle custody are retained rather than reset. This is local verification reuse, not a token-savings, learned inference, model-head, proof, or task-acceptance result.

### Full-root verification reuse and signed opt-out qualification (2026-10-14)

Three further optimizations now default on. Successful canonicality checks for exact DAG-JSON byte strings up to 16 KiB are cached in a 4,096-entry LRU, retaining at most 64 MiB of raw keys plus Python overhead. Every verification still validates the CID and freshly hashes the observed bytes; failures and larger blocks are not memoized. Structured identity loading uses that fresh verification and returns a newly parsed, detached dictionary. Typed nomination builds one private, fully verified relation inventory per call and shares it between seed diagnostics, graph construction and nominee diagnostics. Opening and closing checks require the same root and complete literal CID-to-bytes population, including rejection of equal bytearray, bytes-subclass and key-subclass substitutions. Native source, relation, coverage and authority checks remain required.

The explicit opt-outs are `IPFS_DATASETS_SEMANTIC_BLOCK_CACHE=0`, `IPFS_DATASETS_SEMANTIC_IDENTITY_REUSE=0`, and `IPFS_ACCELERATE_TYPED_INVENTORY_REUSE=0`. Disabling block memoization retains the equivalent direct hashing of canonical observed bytes. The earlier index cache separately supports `IPFS_DATASETS_SEMANTIC_INDEX_CACHE=0`. The new qualification contains 251 distinct passing selected cases: 94 low-level block/identity/API/binding/freshness cases and 157 relation/proposal/current-source compatibility cases, with fresh mandatory AST seals and zero failures, errors or skips.

The genuine 531-symbol Bottle root exposes a separate graph capacity issue. A timed baseline owner replay ended after 78.553 seconds with `typed proposal graph exceeds node bound`; it was not a successful typed handoff. Matched isolated native replays of the same 6,261 captured CAS blocks, delivered seed, root and unchanged 2,048-node bound measured the previous indexed-cache baseline versus the optimized implementation. Root/bundle construction plus nomination took 77.672 versus 26.318 seconds, then 76.409 versus 26.947 seconds in a repeat without concurrent artifact export: 2.95× and 2.84× faster. Nomination alone took 73.558 versus 23.544 seconds and 72.355 versus 24.184 seconds. Both versions preserved the exact native capacity refusal. These are local verification/proposal timings, not inference, signed default-route latency or token-savings measurements.

The rebuilt archive has 12,724 verified members and the existing normal/retrieval declarations and admissions of 31/34 and 35/38 source pins. A fresh explicit typed-opt-out route passed actual DID-signed START, unchanged original owner nomination, protected UID1001 worker pre-invoke budget refusal, STOP and complete process drain. Both unmodified full Terminal observers checked all actual capsule contexts, reconstructed the original native input and joined one owner/dialogue pair without issues. Aggregate model delivery remains unverified because the worker made zero provider attempts; provider token totals remain unknown, and task settlement and acceptance remain unclaimed. This qualifies the explicit opt-out route only. Default typed large-root dispatch, CUDA and all learned head performance still require their own qualification.

The next graph milestone must preserve the 2,048-node and 8,192-edge limits and introduce an explicitly versioned task-scoped authenticated graph. Retain the complete native root, source frontier, sorted indexes, relation ledger and full population commitments as verification evidence. Materialize the outgoing resolved-symbol closure of independently nominated required seeds, with their relation observations, artifact endpoints and unresolved targets. Preserve the owner's required/optional partition, incoming-boundary diagnostics and missing-body refusals. Declare the graph's scope and omitted membership references; a subset cannot use the existing full-graph schema or completeness claims. Update owner publication and historical lineage replay together so both reconstruct the same scoped graph from the full native inventory, refuse incomplete frontiers or exceeded bounds, and retain current-source/parent checks before and after publication. Qualify the same native 531-root signed route and full observers, followed by wrong-origin and interrupted-publication controls. Graph membership remains advisory.

Evidence and retained failures are in [the full-root verification record](../../../../artifacts/supervisor-full-root-verification-20261014-01/README.md). The earlier timed-out container and the archive built after its reservation expired remain recorded; a fresh reserved build supersedes that unaccepted archive attempt.

### Task-scoped authenticated graph (2026-10-14)

The graph milestone above is now implemented across native nomination, source-bound wrappers, owner selection and historical lineage replay. Owner nomination defaults to `task_closure`; `IPFS_ACCELERATE_TYPED_TASK_GRAPH=0` selects the legacy full-root typed graph. The pure API still defaults to `graph_scope="full_root"`. Existing full-root proposal, adapter and wrapper `@1` schemas retain their previous output; an explicit closure uses proposal/adapter and source-bound wrapper `@2`. Node, edge and selected-symbol ceilings remain 2,048, 8,192 and 32, and exceeded bounds refuse without a fallback.

Both scopes verify the complete native root. Closure graphs materialize every outgoing recorded relation of the independently required seeds' resolved-symbol closure, including artifact endpoints and unresolved observations. Reports retain complete body-free root symbol metadata as verification evidence outside provider input, explicit materialized and omitted symbol IDs, and count/SHA-256 commitments for full, materialized and omitted relations and referenced artifact endpoints. The scope commitment also binds root/index identities, full capsule and relation-ledger digests, complete symbol-record count/digest, incoming-boundary diagnostics and graph bytes. Incoming callers remain diagnostic. A scoped graph grants no task coverage, proof, execution or source-substitution authority.

The source wrapper checks the complete root metadata against the current source frontier, including symbols outside the closure. Owner selection preserves the independently declared required/optional partition and accepts delivered optional capsules outside the graph closure; a known nominee absent from the delivered inventory still refuses. Historical replay selects scope from the exact retained version and literal scope, independently of the observer's current environment, then reconstructs and compares the complete proposal from native evidence.

The [retained native replay](../../../../artifacts/supervisor-task-scoped-graph-20261014-01/native02/task_closure-result.json) verified the genuine 531-symbol, 4,652-relation Bottle root and returned a five-node, four-edge graph for its actual seed. It retains both `lexical:handler` and `state:environ[]` as unresolved obligations. The other 530 symbols and 4,650 relations remain committed; this seed has no incoming boundary. Native root/bundle construction plus closure nomination took 27.527 seconds; the same fresh full-root route took 27.717 seconds and retained its node-bound refusal. This establishes bounded successful materialization, without a further latency speedup claim. The [fresh archive](../../../../artifacts/supervisor-task-scoped-graph-20261014-01/build01/build-result.json) verifies 12,724 members with normal/retrieval declarations of 31/34 and admissions of 35/38 sources.

The default signed route now passed on a fresh native Bottle preparation: actual DID-signed START, immutable scoped typed lineage preserving the original nomination, protected UID1001 pre-provider budget refusal, STOP and full process drain. The unmodified full in-process Terminal auditor reconstructed the original input, replayed the native scoped lineage and joined the actual owner/dialogue records. The cold bounded auditor returned `audit_timeout` at the unchanged ten-second ceiling; its availability and qualification remain false, and the combined full-observer gate remains unqualified. All 327 selected regression cases passed with fresh mandatory AST seals, unchanged guarded sources, no reused or skipped cases, and no degraded cases. The two earlier test-fixture assumption failures remain retained alongside the corrected full run. Wrong-origin and interrupted-publication native controls remain separate gates. Model inference, CUDA and head performance, provider token totals, whole-task savings and task acceptance are unmeasured by this graph work.

The next verification milestone is an authenticated cache that a separate observer process can reuse. Key eligibility by exact native root and observed block bytes, validator/source code pins, producer/profile bindings and enforced limits. Recheck backend bytes and current-source/parent fences independently, preserve complete root membership and all refusal behavior, and admit cached results only from an independently authenticated owner-controlled verification record. Printed success flags, learned confidence or an unverified sidecar are insufficient. Cold misses remain bounded/unavailable until normal full verification completes under the existing resource pool. Qualify genuine cross-process reuse against changed blocks, omitted-record corruption, validator/profile changes, stale roots, concurrent source changes and unavailable caches without increasing the audit window.

`runtime/semantic_decision_packet.py` adds an opt-in library packet derived from
the authenticated native parent. Its independently held `SemanticDecisionPolicy`
must completely partition native `capsule_cid` identities into required and
optional sets, and explicitly select any optional bodies to retain. Retrieval
scores, confidence and raw-source flags do not establish optionality. An
unclassified or foreign capsule causes refusal.

All raw source, admissions, pack decisions/obligations/caveats, manifests,
reconstruction limitations and refresh records remain literal. Only unselected
optional capsule bodies and existing estimator telemetry move into a retained,
parent-bound archive. The original `worker_projection` remains parent preparation
provenance; separate counts describe the decision packet. Exact regeneration
requires the trusted caller's expected policy again, so a self-consistent packet
cannot reclassify required evidence on its own.

`semantic_projection_dispatch.project_semantic_router_prompt(decision_policy=...)`
authenticates the complete producer and current source before and after packet
construction. `expand_semantic_router_decision_packet` binds expansion to the
same full verifier on both sides of complete returned-descriptor counting.
Construction refuses an archive that cannot fit the hard expansion bound;
expansion also enforces caller byte/token ceilings and cancellation.
Expansion accepts an explicit nonempty `capsule_ids` tuple to return only those
deferred optional bodies, without estimator telemetry. Required, already retained,
unknown or duplicate identities refuse. The returned subset has its own content
digest and declares that it is not the complete archive. Fetching the entire
archive remains available, but can erase the initial prompt savings.

This revision is not admitted by the existing CLI codec or selected automatically.
The supervisor-managed dialogue below provides a separate opt-in library bridge,
using native identities and independently retained policy rather than adding
decision packets to the identifier codec. Existing telemetry-only projection and
CLI behavior remain unchanged. Terminal Bench launch wiring, Grok tokenizer
compatibility, task acceptance and session savings remain unestablished.

The current Grok host runner admits file tools and disables generic MCP/terminal
execution; the pinned task container has a distinct terminal-tool profile. A
registered MCP adapter alone therefore does not deliver expansion to Grok.
Qualify an explicit bridge under the existing provider profile, with immutable
attempt-bound policy custody, bounded per-capsule delivery and observable usage.
Source changes require fresh preparation and a fresh policy over the new capsule
identities. Do not widen provider allowlists just to make a projection usable.

An authored eight-helper source example using the real producer, identical
6144-byte chunking and workspace instructions measured these local `o200k_base`
proxy estimates for the Grok model nomination: original prompt 17,141; packet
12,658; packet plus one returned capsule descriptor 13,960; packet plus the whole
archive descriptor 18,730. The source stayed unchanged. These are representation
component sums, not a conversation simulation or provider/session measurement:
tool framing, repeated input history, output and retries are absent. The example
shows both the opportunity and the expansion cost; it does not establish a
Grok tokenizer mapping or benchmark success.

### Supervisor-managed expansion dialogue (2026-10-13)

`runtime/semantic_decision_dialogue.py` retains the caller's policy and original
parent bindings outside model responses. A model can return a closed expansion
request identifying 1-4 capsule IDs, or a closed final response containing its
literal output. Session, generation and context commitments bind both responses
to the current round. Unknown fields, duplicate keys, foreign/stale bindings,
required or already delivered capsules and repeated requests refuse. No response
can choose source paths, policy, budgets or tool permissions.

The supervisor checks the complete native producer/source before and after each
call, expands only requested optional bodies through the existing verified API,
and starts a fresh bound round. Every round preserves the required task packet,
literal instructions and all deliveries already made. The initial selection
counts protocol and workspace overhead and retains the native prompt when it is
not larger. Each full model prompt uses the unchanged byte/token ceiling; an
overall deadline and at most four calls bound the continuation. Cancellation
prevents further calls and final delivery; it does not claim to terminate a
provider process already running.
The optional `max_total_input_tokens` limit checks the sum of complete repeated
user prompts before starting each invocation, using the declared local tokenizer
or estimate basis. It cannot guarantee provider internal tool-turn or output
usage. Prepared expansions record the invocation attempts in which their bytes
were included; preparation alone is not model delivery.

`runtime/semantic_decision_provider.run_shared_router_decision_dialogue` composes
that loop with pinned shared-router adapters. Codex reuses the existing coding
runner in a separate allocated Git worktree. Grok uses one structured CLI turn,
no tools, no cross-provider fallback and a fresh native session per call. This
Grok path produces a response for owner validation; it does not add editing tools
or qualify the Harbor implementation route. Adapter entry checks preserve the
worktree/source relation and reject inherited database authority.
Grok also reuses the existing trusted executable resolver and refusal of project
MCP, hook, plugin and LSP configuration. A workspace-controlled executable cannot
establish the tool-free profile just by accepting familiar command flags.

Every attempted invocation retains a bounded receipt, including failures,
malformed responses, cancellation and source changes. The dialogue's internal
summary reconciles bound receipts. Final Terminal accounting also retains
independently parseable native usage from unbound wrappers as known evidence,
while withholding an exact total because coverage failed. Codex native receipts
are unwrapped and reconciled by session, so cumulative snapshots are not summed. Grok's missing
timeout usage stays unknown. Provider dispatch observations are distinct from
adapter attempts. The coding runner's optional receipt observer cannot suppress
provider failure or prevent environment restoration.

The local input sum counts complete repeated prompts. The earlier packet-plus-
descriptor illustration does not include this replay: a second stateless call
can erase its apparent savings. Provider outputs, internal tool turns, cached
subsets and all retries still require actual provider/session accounting.
In the retained final eight-helper example, a two-round dialogue measured 13,035
and 14,374 local proxy input tokens, totalling 27,409 versus a single native input
of 17,141. This is local full-prompt replay accounting, not actual Grok usage or a
matched task-success comparison. A smaller first packet alone does not justify
promoting a continuation policy.
Automatic tokenizer selection remains model-aware where supported and explicitly
estimated otherwise. Trusted library callers and the explicit worker launch route
below can use this bridge. Automatic policy selection, runtime archive deployment
and matched live success/usage comparisons remain separate work.

### Explicit Terminal worker route (2026-10-13)

`runtime/semantic_decision_worker.py` connects the dialogue to the existing
Codex coding worker. Planning keeps its existing route. Host preparation, Harbor,
the container supervisor and the signed implementation command accept:

```text
--semantic-decision-policy-artifact /absolute/child/local/policy.json
--semantic-decision-policy-sha256 <exact-sha256>
--semantic-decision-max-calls 2
--semantic-decision-max-total-input-tokens 18000
```

The artifact/hash pair selects the route. Existing launches preserve their
defaults. The route requires the full indexed arm and cannot combine with the
identifier-codec/role-projection comparison. Doctor candidate routes stay
provider-free; `--doctor-dispatch-policy model-only` selects a model-route
comparison explicitly. Per-call `--max-input-tokens` and the existing tokenizer
options also apply. Grok remains the separately available tool-free library
adapter; this change does not qualify a Grok Harbor editing worker.

An independently held descriptor uses schema
`supervisor-semantic-decision-launch-policy@1` and contains exactly `task_id`,
`parent_artifact`, `parent_sha256` and the explicit three-list selection `policy`.
`build_decision_launch_policy(native_prompt=..., policy=..., repository=...)`
authenticates the complete native producer and selection partition before
returning that descriptor. The owner serializes it, pins the SHA of those exact
bytes, and deploys it outside the writable candidate at the supplied child-local
path. The worker rejects changed bytes, duplicate/extra fields, symlinks, a
foreign task/parent, incomplete selection and stale canonical source.

For the prebound artifact variant, policy creation and staging remain caller
responsibilities. A descriptor must match the actual task and semantic parent
generated for that attempt. The owner-time nomination variant below removes
that dynamic descriptor prerequisite while keeping capsule classification
explicit. Neither variant infers policy from relevance scores or supplies
portable delivery of caller-owned input assets.
Selected configurations also require current bytes for the declared worker and
direct runtime dependencies in both the archive manifest and regular tar members
before the Harbor dry-run. Retained source pins are rechecked at execution. This
extends the existing deadline-source gate; it does not claim verification of the
complete transitive import closure. Older archives require a rebuild before this
route can run, rather than discovering a missing worker after paid planning.

Doctor residual and public-instruction loaders validate against the original
compiled stdin. Their literal suffixes are appended before full-message
selection/counting and replayed on continuation. `worker_launch` records the
policy hash, original input identity, suffix identity and tokenizer provenance.
Known preparation refusals report no provider call. Unexpected failures keep
dispatch status unknown; dialogue failures retain all attempted-call receipts.

Terminal and Harbor accounting retain `semantic_decision_dialogues`, unwrap
original Codex invocation identities and deduplicate exact separately printed
receipts. Changed observations remain conflicts. Missing later receipts,
unbound attempts, unsupported Grok usage or an interrupted launch missing its
expected dialogue record prevent exact totals while retaining known subtotals.
Local prompt estimates and outer CLI counters never become additional charges.
At this milestone, historical context auditing did not yet reconstruct complete
dialogue rounds; unverified round inputs did not establish coverage. The
historical dialogue-input audit below adds that reconstruction.

Offline validation retained 180 distinct successful checks (219 physical test
calls across the main and focused runs), with normal mandatory AST seals and
43 current source/test/runner guards. The main run passed 177 checks and had one
fixture setup error: pytest expanded a 256 KB parameter into `PYTEST_CURRENT_TEST`,
so Git could not start (`E2BIG`). Explicit short parameter names fixed that case.
The focused run passed all 42 checks, including both input-refusal cases, new
foreign-receipt call-count checks, and legacy accounting. Only the conservative
call-count correction and its tests changed between runs; the worker test also
received bounded parameter names. No live provider calls, deployable archive
builds, or benchmark executions occurred; synthetic archive fixtures were tested.
Evidence: `artifacts/supervisor-decision-launch-20261013-01/validation-summary.json`
at workspace root, SHA-256
`8ac2781909065960e50e651d28ce49eaffb0cd52161d9ad92c97d40a20c5cbb2`.

### Owner-time policy binding (2026-10-13)

`runtime/semantic_decision_owner.py` binds a pinned nomination at the existing
owner-to-worker handoff, after the final native prompt has been compiled and
before a model worker starts. Select this alternative with:

```text
--semantic-decision-nomination-artifact /absolute/child/local/nomination.json
--semantic-decision-nomination-sha256 <exact-sha256>
--semantic-decision-max-calls 2
--semantic-decision-max-total-input-tokens 18000
```

The nomination uses schema `supervisor-semantic-decision-nomination@1` and exactly
one `policy` containing the existing required, optional and selected-optional
capsule-ID lists. `build_decision_nomination(policy)` creates this representation
from independently chosen classifications. It contains no precomputed task ID
or semantic-parent digest. This variant assumes an explicit nomination at an
already deployed child-local path. The host-source delivery variant below can
stage that static input; neither variant infers a classification.

The container supervisor adds the admitted task key to the signed owner command.
The owner launcher captures bounded UTF-8 stdin, checks that task key, verifies
the complete native producer/current source and the nomination's complete
partition, then constructs the existing task/parent-bound launch descriptor.
Unknown capsule IDs or changed source/inventory require a new valid nomination;
scores, confidence and raw-source flags never make a body optional.

Generated descriptors live under the owner-controlled canonical repository's
`.runtime/semantic-decision-policies/`, outside the writable candidate. Publication
uses digest-addressed, single-link `0444` regular files, `0755` directories that
only the owner may write, atomic creation and fsync. A validated owner-only
`0600` `.runtime/.semantic-decision-publish.lock` serializes publication and
concurrent reuse. Exact immutable objects can be reused; conflicting bytes,
symlinks, hardlinks or writable directories refuse. Owner
private state remains private. The original prompt bytes, including literal
suffixes and newline spelling, are replayed exactly through the existing worker
monitor; ordinary routes retain inherited stdin and existing cleanup behavior.
The worker independently revalidates the generated descriptor and current source.

An owner-binding receipt commits to the nomination, signed task, native stdin,
semantic parent and published policy. These records are retained separately from
model receipts and add no calls or tokens. On the selected route, accounting
requires the independently pinned task/nomination and a one-to-one match to each
dialogue's original stdin and policy identities. Missing, conflicting, foreign,
reused or orphaned owner bindings preserve known usage subtotals but prevent
exact totals and complete call counts. `provider_called: false` on a binding
describes the binding operation only; it cannot establish that a later child
made no model calls. Archive capability checks include the owner module.

Offline validation qualified 215 distinct checks across three sealed runs. The
broad run passed 203 checks and retained nine fixture failures: authored source
directories inherited the host's `0002` umask and became `0775`, which the
publisher correctly refused. Explicit `0755` fixture baselines fixed those
cases without weakening publication checks. The complete owner suite then
passed all 59 checks, including two native producer fixtures, a writable-root
refusal and forced overlapping publication. A final three-check run verified
that pending-file cleanup failures still close all directory and lock
descriptors, and repeated sync-failure and concurrency checks. There were 265
passing physical calls and nine historical failed calls, with every check's
latest result passing.

The artifact records 50 current declared source/test/runner guards across the
three runs; these guards are not a complete import-closure proof. Only the owner
publication module and its test file changed during incremental revalidation.
Normal mandatory AST seals remained active and guarded sources stayed unchanged
within each run. No live provider calls, deployable archive builds or benchmark
executions occurred. At that milestone, static nomination delivery, explicit
capsule classification, Grok Harbor editing qualification and a matched live
comparison of task success and whole-session token usage remained open. The
host-source delivery variant below addresses static nomination transport.
Evidence: `artifacts/supervisor-decision-owner-20261013-01/validation-summary.json`
at workspace root, SHA-256
`67291ef59a6e8f5e8a3c25592e9940170c30ab19b6ee4e446baf0cadf95cb328`.

### Historical dialogue-input audit (2026-10-13)

`semantic_decision_input_audit.py` reconstructs up to four recorded dialogue
inputs from the captured original context, SHA-pinned launch policy, retained
producer blocks and required suffix artifacts. It reuses the packet, subset
expansion and round renderers. Historical producer verification remains required;
current source and a surviving candidate worktree are not required. Replay grants
no freshness, execution or completion authority and exports no prompt contents.

The existing final-context audit forwards a bounded, closed observation envelope
containing dialogue and owner-binding receipts and their expected task/nomination
pins. Its `semantic_decision_inputs` result checks every round's generation,
prompt/model hashes and byte lengths, expansion descriptor and delivery schedule,
then joins each qualified round to the separately collected native invocation.
Identical dialogue observations deduplicate; changed observations, ambiguous
joins and missing expected receipts keep coverage incomplete. Interrupted or
failed attempts retain their known input observations. Prepared expansions that
never reached an invocation contribute no input. Existing ordinary audit requests
retain their list format and default behavior.

Each dialogue may report a historical local-input comparison against one native
input when recorded counts, count kinds, selected representation and per-call /
cumulative budgets are internally consistent. These counts are not recomputed,
do not establish tokenizer compatibility, and are not provider usage. The authored
regression deliberately records native input 1,000, first packet input 100 and
second input 1,200: first-input reduction coexists with total input 1,300. This is
a contract example, not a measured model result or task-success comparison.
Provider usage reconciliation remains separate and local counts are never added
to it. The existing component census does not decompose the dialogue packet;
that limitation remains visible even when exact round bytes are verified.

The selected-route archive check pins `terminal_context_audit.py`,
`semantic_decision_input_audit.py` and `automatic_tokenizer_audit.py` alongside
the existing worker and command adapter. The declared selected list contains 21
sources plus the four deadline sources. Both the manifest and regular tar bytes
must match the current owner sources before preparation; execution rechecks the
retained pins. This is a direct capability check, not a complete dependency
closure or a deployable-archive qualification.

Offline validation passed all 182 distinct checks: 169 main historical
reconstruction, collector/join and existing context/tokenizer/translation checks,
then 13 archive checks. Mandatory AST seals were active; all declared source
guards stayed unchanged within each run, and all 54 final source/test/runner
hashes matched current files. Only the archive capability list changed between
runs. The new reconstruction suite shares one native producer/allocated Git
fixture with four authored callback variants; collector joins and combined
suffix-loader ordering use authored fixtures. Actual bounded owner children
exercise the observation envelope and trusted sibling loading. These checks
establish offline contract behavior, not live provider delivery, remote tokenizer
compatibility, task success or token savings. No live provider calls, downloads,
installs, deployable archive builds or benchmark executions occurred.
Evidence: `artifacts/supervisor-decision-input-audit-20261013-01/validation-summary.json`
at workspace root, SHA-256
`86f79af2990e785f03f4ab8ad7732d8a3dbccff6b482347b73ed2756df0b880d`.

### Host nomination delivery (2026-10-13)

The full benchmark accepts a distinct host input:

```text
--semantic-decision-nomination-source /absolute/host/nomination.json
--semantic-decision-nomination-sha256 <exact-raw-file-sha256>
```

This option is mutually exclusive with child-local nomination and prebound policy
artifacts. The existing artifact options retain their meanings. Preparation
authenticates a bounded regular file, rejects symlinks in every path component,
checks the closed nomination schema, and captures its exact raw bytes in a new
`0444` snapshot under the fresh trial output. Configuration points to that host
snapshot. Preparation and execution recheck its path, digest, size, owner,
permissions and single-link identity; editing or deleting the original input
after capture does not change the prepared snapshot.

The Harbor adapter captures a fresh run snapshot and charges delivery operations
to the original execution deadline. Root prepares an exclusive per-run directory
under `/opt/ipfs-supervisor/decision-nominations/`, Harbor uploads the snapshot,
and root revalidates the exact bytes before publishing a separate immutable
inode. The final digest-named file is `0444`; its directories are `0755` and only
root may write them. Existing private or writable parents, collisions, aliases,
changed bytes and malformed input refuse without changing existing permissions.
An open write descriptor to the upload cannot alter the separate published inode.
Prepare and publish receipts explicitly distinguish their operation and published
status. The driver receives only the derived child-local artifact and SHA, and
starts after the publication receipt matches. Delivery receipts add no calls or
tokens. A failed stage retains bounded diagnostics and the existing finalization
path; interrupted staging may retain its isolated per-run files.

Owner-time task/parent binding and full producer/source/partition checks still
run after delivery. Capsule classification remains explicit. Tokenizer and
prebound-policy assets still use their existing child-local interfaces. The
selected archive gate also pins the delivery helper and Harbor adapter, bringing
the selected capability list to 23 sources plus four deadline sources. This
implementation does not qualify a full runtime deployment or live savings.

Validation passed all 141 distinct tests with normal mandatory AST seals and 53
unchanged, current declared source/test/runner guards. Host and adapter fixtures
use authored preflight/archive/environment responses. Local publisher tests
adapt only the required owner UID to the nonroot test runner; a separate native
producer fixture verifies publication through owner binding and the existing
worker loader. The actual unmodified publisher was also exercised as UID 0 in an
isolated container using a locally installed Python image, no network, a read-only
root filesystem, bounded tmpfs storage, and no host mounts or privileged mode.

The first direct Docker-copy probe failed before publication and remains in the
evidence. A second probe used the installed, unmodified Harbor
`UnixOps.upload_file` through a narrow bridge to that probe's container. Docker
copy could not locate the prepared tmpfs directory; Harbor's existing tar-stream
fallback succeeded without relaxing the isolation restrictions. All 11 checks
passed, including exact root-owned publication, a separate immutable final inode,
held upload-write-FD isolation, nonroot final-file reads and prepare refusal,
directory collisions and private-parent refusal. Both owned test containers were
removed. This qualifies the publisher and upload method through the bridge, not
the full Harbor Compose environment or supervisor runtime deployment.

No provider calls, downloads, installs, deployable archive builds or benchmark
executions occurred. Explicit capsule classification, current capsule identities,
full runtime deployment and a matched live comparison remain open.
Evidence: `artifacts/supervisor-nomination-delivery-20261013-01/validation-summary.json`
at workspace root, SHA-256
`fe985911d5f501f8e6e67d0df21fd05f99d72f28b941fccafc7e4b60fd60c915`.

### Full indexed native-input control (2026-10-13)

The default full indexed route can select the legacy identifier codec when its
complete input count is smaller. It is an automatic representation control,
not a guaranteed literal native-input baseline. The no-index arm also removes
indexed context and Doctor selection, so it cannot isolate packet compression.

The explicit `--semantic-native-input` option keeps the full indexed arm,
canonical repository, source/producer admission, cold preparation and planning,
literal public requirements, Doctor suffix, workspace advisory, tokenizer and
input-budget checks. It independently verifies the complete native semantic
artifact and constructs no codec candidate or translation table. Source checks
repeat after counting and provider return; failure usage remains retained.
This is an opt-in coding control. Planning retains its existing input route.
Decision policies/nominations, context projection, nonlegacy slot policies,
unbound coding and the no-index arm cannot combine with it. Existing defaults
and argv shapes remain unchanged.

Host preparation and execution verify the declared runtime capability sources
against both regular archive members and current owner bytes. Native-control
pins are retained separately as `runtime_native_input_source_sha256`. This
reuses the selected-route source list; it is not a complete import-closure or
runtime-deployment qualification.

For the first matched transport experiment, prepare two fresh outputs with
the same dataset, current runtime archive, model, reasoning, CLI version,
timeout, tokenizer policy and requirements. Both use `--arm full` and
`--doctor-dispatch-policy model-only`. The control adds
`--semantic-native-input`. The candidate instead adds the exact host nomination
source/hash and explicitly bounded dialogue calls/input. Independently classify
the current capsule inventory under the actual `/app` source identity before
launch. An advisory dependency closure cannot supply this classification.
The current worker route remains Codex coding; Grok Harbor editing is not
qualified by this change.

Obtain the capsule inventory with existing provider-free
`terminal_indexed_preparation.prepare`, `initial_context` and
`terminal_initial_context.load_initial_context` in a separate disposable task
container at `/app`, using the matching source and selected asset profile.
Partition `verified["semantic"]["capsules"]`, whose original is
`/app/.runtime/terminal-initial-context/semantic/worker-context.json`; the full
root index can contain additional unselected capsules. Exact scoped bytes,
canonical `/app` identity and scanner/compiler profile determine capsule IDs;
fresh surrounding task/parent bindings still require a new owner launch policy.
Reuse a classification only after checking its actual current inventory.
The standard driver requires fresh preparation, so restart from the original
task image for each real trial. Standard exported host reports do not contain
the full capsule inventory. Owner nomination validation currently follows
planning, which makes this separate provider-free check useful before paying
for a trial that would reject the partition.

The existing retained-receipt reporter now accepts two supervisor variants
without requiring a separate native Codex harness trial:

```bash
python -m benchmarks.agent_supervisor.container_coding.benchmark_comparison \
  --supervisor /absolute/native-trial/receipt.json \
  --supervisor /absolute/packet-trial/receipt.json \
  --output /absolute/fresh-comparison
```

Configured native-input, decision-packet and automatic-codec groups receive
separate totals through the existing strict native session reconciler. Failed
and interrupted costs remain visible. A native session reused across variants
withholds aggregate totals and subtotals. Configuration labels are not proof of
the actual provider wire: a decision dialogue may fall back to native. The
report does not infer source matching, complete campaign coverage, matched task
success or paired savings merely from shared task/model settings. The separate
`context_projection_comparison` API retains its stricter paired contract.

Offline qualification passed 350 distinct checks across two sealed runs. The
initial run retained 266 passes and two failures: rereading a purpose-less
planning log could manufacture an inferred coding observation and hide the
missing-coding receipt marker. The shared usage collector now compares exact
native bodies independently of incidental phase, preserves the original owner
observation, and retains changed counters/body. Later purpose-less changed
observations inherit only an unambiguous established phase for that invocation.
Explicit native purpose conflicts remain separate evidence. The affected
accounting/collector/reporting suites and both original failures were rerun:
all 122 revalidation checks passed. Four new phase regressions cover this fix.
The initial failures remain in the evidence; unrelated code stayed byte-identical.
Mandatory AST seals remained active and all 61 final declared source/test/runner
pins matched current files. A standalone reporter CLI check on authored receipts
also passed; it is separate from the test count and from actual provider usage.

No live provider experiment or token-savings result is established by these
checks. Evidence:
`artifacts/supervisor-native-input-control-20261013-01/validation-summary.json`
at workspace root. Runtime rebuilding and deployment qualification are a
separate campaign and are not counted in this offline summary.

### Upfront capsule delivery versus reactive replay (2026-10-13)

The existing `selected_optional_capsule_ids` policy supports a bounded upfront
selection without a new representation. A new provider-free probe reuses the
retained eight-helper native producer, exact source/parent, unchanged required
and optional partition, task/suffix/model bindings and full dialogue renderers.
It supplies upfront exactly the one capsule requested by the prior authored
reactive trace. Fixed authored protocol UUIDs control irrelevant tokenizer
variation. This is an oracle-like selection comparison, not a learned selector
or task-success result.

With the installed `o200k_base` tokenizer used as an explicitly unverified
Grok 4.7 proxy, complete rendered input counts were:

| Variant | Full input estimates | Calls |
| --- | ---: | ---: |
| Native input | 17,141 | 1 |
| Reactive packet plus expansion | 13,035 + 14,374 = 27,409 | 2 |
| Packet with the requested capsule upfront | 13,778 | 1 |

Upfront delivery adds 743 tokens to the first reactive input and avoids the
14,374-token second replay. That is 13,631 fewer input estimates (49.73%) than
this reactive schedule, and 3,363 fewer than the one native input. Authored
response envelopes are counted separately: reactive 125 + 87, upfront 94. None
of these counts is native provider usage or billing. The callbacks make no
provider/tool/coding calls and perform no task acceptance; required coverage
classification is supplied, not inferred. No real selection quality, internal
provider framing or whole-session savings is established.

All 57 declared source/fixture pins stayed unchanged; the probe is not a
complete import-closure proof. Evidence:
`artifacts/supervisor-native-input-control-20261013-01/prefetch-probe/report.json`
at workspace root, SHA-256
`32a9172acc0dd41119fe53d719c76b75cbe38ab2a3efdf264a300f2618d1745e`.
The next live comparison should therefore test task-relevant upfront selection
separately from reactive expansion, charging every delivered round and failed
attempt. Fetching every optional capsule upfront is not implied by this result.

### Original-task deployment and current inventory (2026-10-14)

A rebuilt runtime archive passed the existing original-task deployment
qualification on ARM64. The deployment exercised Python 3.12, DuckDB 1.5.5,
native DuckLake/HTTPFS/Quack extension loads, pinned Codex CLI 0.158.0, empty
native ownership and unchanged original `/app` source/Git identity. Its archive
SHA-256 is `025bfa6f6802eb2f3967be3335356965f91f7f3d4eb37d1b05fcc831a3e8e9dc`.
All 27 declared capability/deadline sources matched current files and actual
archive members. These are selected sources, not a complete import closure.
This separate campaign installed runtime packages/images; it downloaded no new
model weights or tokenizer assets and made no provider calls.

A second independently fresh deployment passed the same qualification before
controlled provider-free preparation. Existing preparation, initial context and
strict loader APIs then captured the actual worker inventory under `/app`, with
the cached MiniLM revision and a new private state namespace. Both owned
containers were removed; neither prepared container is eligible for a paid
trial. Foreign containers were left untouched.

The inventory callback used the API default `train_autoencoder=False`; the
production full arm enables local CodebaseIR training. It therefore did not
qualify that training path or its timing. In the subsequent native live trial,
both reported initial and coding-context semantic roots matched the captured
inventory root exactly. Native capsule compilation/source scope is separate
from the trained learner metadata. A fresh candidate must still pass its own
current inventory/owner checks; the earlier capture alone is insufficient.

The actual worker artifact is 27,276 bytes and contains **one of 531 root
capsules**, `bottle.FapwsServer.run.app`. The existing lexical selector and exact
byte repacking had already omitted the other 530. Its body describes two source
statements: set `environ['wsgi.multiprocess'] = False`, then delegate to `handler`.
It has no extracted contracts, exception behavior, tests, fixtures or proof
obligations. This is a concrete nomination-quality concern; a trained code index
does not itself establish useful capsule selection.

The current `lexical-query-symbol-name@1` selector counts distinct regex-word
overlap between the whole instruction and each qualified name/module path. It
does not split CamelCase or underscores and has no minimum relevance score.
`app`, `bottle`, `py` and `run` give this shim four matches; paths and the suggested
test command therefore influence its rank. Learned vector retrieval is prepared
separately and its hits do not enter this capsule selector. The configured cap is
eight, not one: preliminary size accounting skips large capsules, then exact
payload repacking removes capsules from the end of the ranked selection until
the byte/pack budgets fit. The final artifact does not retain the earlier
selected/dropped population, so that intermediate count is unknown. Repacking
does not backfill cheaper candidates. A future isolated relevance change should
compose existing structured retrieval nominations with native admission and
exact final budgeting; confidence or rank still cannot establish requiredness.

Byte attribution of that unchanged worker artifact found 15,474 canonical JSON
value bytes in four task-text fields (`objective`, `pack.pack.objective`, literal
instruction and projection query), versus 2,712 bytes in the capsule body. The
two objective strings are identical, as are the literal instruction and query.
This is artifact-component evidence, not full rendered input or native provider
usage. A future versioned model-facing projection could preserve one literal
instruction and all distinct restrictions while retaining canonical duplicates
and provenance outside the message; it must establish explicit field coverage
first. The current pilot does not make that change. Evidence:
`inventory/payload-byte-attribution.json` under the live campaign directory.

The pilot explicitly classifies this supplemental body as optional and initially
deferred, after two independent public-source reviews. This is not proof of its
irrelevance or safety. Literal instruction/smoke inputs, admissions and caveats,
manifest, pack references and the exact full `bottle.py` source-fetch-before-edit
obligation remain required. Deferred facts are unavailable until delivered.
The policy partitions only the actual worker inventory; it does not reclassify
the other 530 capsules. The live owner must verify the exact current capsule ID
and source, refusing a mismatch.

Before either model run, both Harbor dry-runs passed with matched task/source,
archive, model (`codex_cli` / `gpt-5.6-sol`), CLI, reasoning, setup/execution limits,
tokenizer policy, per-call input ceiling and model-only Doctor setting. The
candidate adds a pinned nomination, at most two dialogue calls and a 65,536
cumulative local input estimate ceiling. The one-pair population includes
failures; the native control is scheduled first and the candidate second in
fresh original-task environments. No general efficiency or causal claim can
follow from one pair, and configured packet selection may fall back to native.

Evidence at workspace root:
`artifacts/supervisor-native-input-live-20261013-01/validation-summary.json`,
`inventory/validation-summary.json` and `matched-pilot/matched-preparation.json`
under that campaign directory. The frozen nomination SHA-256 is
`ad645eac7c8cf1b8e9ec22b06fd0429e5698459f0cf5ccffb9de7a345cfd0516`;
`matched-pilot/policy-review.json` records its public-source rationale and limits.

The frozen first pair finished with asymmetric outcomes. Native input passed the
original verifier (reward 1.0) and reported supervisor completion. Two complete
observed native sessions reconciled to 259,278 input + 2,888 output = **262,166
tokens**; 212,992 cached input tokens are already included in input. Planning
accounted for 22,193 and coding for 239,973 of that total. The initial coding
message was 63,396 bytes / 15,849 UTF-8-div4 token estimates, not an exact model
tokenizer count. The complete coding-session counters include subsequent model
turns/framing and must not be equated with that initial message. Execution took
285.96 seconds; coding-input audit verified all observed model inputs.

The packet environment qualified in 616.63 seconds, but nomination publication
then failed before instruction/driver/model dispatch. It has no verifier reward
or provider-native usage receipt. Unknown receipt totals are not zero. Both
owned containers were removed, and neither arm was retried. The existing strict
paired reporter withholds every savings metric because task outcomes differ
and candidate usage/acceptance are unavailable. The adapter's generic publisher
exception discarded remote stderr, leaving the precise rejected guard unknown.
A provider-free actual Harbor transfer/publisher reproduction is required before
changing the transport or launching another pair. Evidence:
`matched-pilot/execution-summary.json` and `matched-pilot/comparison/paired.json`.

### Harbor nomination ingress correction (2026-10-14)

A provider-free reproduction on the actual cached original task image, using
unmodified Harbor Compose `upload_file`, isolated the staging failure. Compose
copy succeeded but retained host UID/GID 1000; the publisher required UID 0.
The 258-byte nomination, digest, regular-file type, single link and read-only
mode all matched. The earlier bridge qualification exercised the tar fallback,
which resets ownership to root, and therefore missed this transport composition.
The first prebuilt-image diagnostic startup was incompatible with ARM64 and
was cleaned; the successful reproduction used the same cached Dockerfile-build
path as the actual pilot.

Publication now normalizes only the opened upload descriptor to root ownership.
Before changing custody it checks non-following regular-file access, a single
link, exact bounded size and current descriptor/path identity. It then verifies
the same inode, size, mode, modification time and link count, plus normalized
ownership, before the existing byte/hash/schema/currentness checks. The final
file is still a separately created root-owned `0444` inode. The adapter also
retains bounded private publisher diagnostics before receipt parsing, so a
future nonzero exit cannot discard its stderr.

The fixed actual Compose probe passed normal publication, a held writable
ingress-descriptor isolation case, and tampered-byte, malformed-JSON and wrong-pin
refusals. UID 1000 ingress became a new UID 0/single-link/read-only final with
the exact digest; mutation of the old unlinked upload inode did not alter the
final. The owned container was removed. No authentication, runtime installation,
providers or original verifier were used. Evidence:
`artifacts/supervisor-native-input-live-20261013-01/nomination-publish-repro/cause-summary.json`
and `nomination-publish-fixed-repro/validation-summary.json` under that campaign;
the latter SHA-256 is
`0b199b6d0b60d76cea50907382b23a38c29c452158b3d995efb2ec46a22908f5`.

A new one-pair intention keeps the original reviewed nomination and task/model
settings. Its runtime preserves the earlier immutable packaging snapshot with
only the publisher and host-diagnostic source overlays. Both arms require fresh
preparation and the same new archive; the earlier successful native run is not
silently reused as an exact-archive control. The failed first pair and its
observed provider costs remain separate retained evidence. The new intention
is `artifacts/supervisor-nomination-publish-fix-20261014-01/matched-pilot/intent.json`
at workspace root; source/runtime qualification must finish before launch.

Fixed-source qualification passed all 54 focused publisher/adapter checks with
normal mandatory AST seals; all 64 declared source/test/runner guards remained
stable. The rebuilt archive has 27,908 regular members: exactly the two released
source changes, 27,906 unchanged members and no additions/removals. All 12,709
earlier packaging inputs were copied exactly before those overlays, and all 27
current capability/deadline pins verified against actual tar bytes. Archive
SHA-256: `bede6aba313c411270a0843559c05129446922d7a8ab99c9400cbae2458d71dc`.
Both fresh Harbor dry-runs and matched bindings passed before the new pair was
released. Evidence: `qualification-summary.json`, `runtime-campaign/build-result.json`
and `matched-pilot/matched-preparation.json` under
`artifacts/supervisor-nomination-publish-fix-20261014-01` at workspace root.

The fresh fixed-archive pair completed successfully on both arms:

| Configured arm | Actual coding transport | Native input | Native output | Native total | Original reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Native control | Native input | 275,422 | 3,049 | 278,471 | 1.0 |
| Decision packet | Native fallback; no expansions | 400,806 | 3,650 | 404,456 | 1.0 |

Each arm has two complete, separately owned native sessions and verified coding
inputs. Initial and coding-context roots match the captured inventory. In the
candidate's own complete-message comparison, native input was estimated at
15,800 tokens and the packet at 16,039 using `utf8-div4@1`. These are labeled
fallback estimates, not exact model-tokenizer counts. The guard therefore chose
native before its single coding call. Nomination publication succeeded in the
actual lifecycle, and both owned task containers were removed.

The configured packet path consumed 125,985 more observed session tokens
(45.24%), but **no compressed packet reached the provider**. This is not a
compression-effect comparison. Actual native prompts/trajectories are not
byte-identical across fresh runs; provider caching and host workloads are also
uncontrolled. The earlier successful native trial used 262,166 tokens, further
illustrating run variation. The strict paired API can report this revision's
accepted-task usage, while a compression-effect claim remains ineligible.
No token reduction is established by these experiments.

All four trial records across both revisions remain in the retained reporter,
including the staging failure. The known observed provider subtotal is 945,093
tokens; the complete overall total remains unavailable rather than treating the
missing failed-trial receipt as zero. Cached input and reasoning output remain
subset categories and are not added again. Dollar billing and external receipt
authenticity are not established. Evidence:
`matched-pilot/comparison/paired.json`, `all-retained/comparison.md` and
`final-evidence.json` under the fixed-publisher campaign. Final evidence SHA-256:
`c8120c797c2e860f16d6722691c1c0e95c8312837a1e77bb15a73a4eb36c6337`.

The next isolated change should reduce verified duplicate task/provenance text
in the model-facing view and improve source-bound capsule nominations. The
four task fields occupy 15,474 bytes versus the sole capsule's 2,712 bytes;
deferring that small body cannot pay for the current packet envelope. Preserve
one literal public instruction and every distinct requirement, retain complete
canonical originals outside the message, and verify explicit field coverage
before changing the projection. Connecting existing learned retrieval to the
capsule selector should be a separate ablation. Whole-repository minify/unminify
is not required for either change.

### Exact required task-text references (2026-10-14)

The optional `SemanticDecisionPolicy.task_text_source_path` selects a separate
`@2` nomination, launch descriptor and packet revision. Omitting it preserves
the exact three-list `@1` policy shape, packet schema and existing argv. No new
Harbor flag is required: the independently pinned nomination bytes carry the
profile. Owner/worker loaders and the historical input audit require schema and
policy revisions to match; the host and standalone root publisher accept the
same two closed shapes.

The new packet keeps **two literal roots**: the required public instruction in
`raw_sources` and the top-level objective. It replaces only two exact duplicate
fields with explicit immediately available required references:

| Duplicate field | Retained literal target |
| --- | --- |
| `worker_projection.query` | `raw_sources[task_text_source_path]` |
| `pack.pack.objective` | Top-level `objective` |

The nominated source path must be canonical, explicitly required, and bound to
its literal bytes by the unchanged source manifest hash. Both comparisons use
exact string equality. Missing, differently spaced or distinct text refuses
the profile; projection performs no whitespace normalization. Unknown fields
and distinct constraints remain literal. All admissions, caveats, required raw
sources, source-fetch obligations, manifest, reconstruction and outer compiled
goal/scope/authority/acceptance are preserved. Optional capsule-body selection
and telemetry deferral retain their existing rules.

Verification regenerates the whole view against the independently held policy
and authenticated parent. `restore_semantic_decision_required_context` first
verifies that view, then restores the two fields from their in-packet literal
targets and requires exact equality with selected native required context.
The `@2` packet commits both its referenced-view coverage and the canonical
selected-required coverage separately. This restores selected required fields,
not a whole repository or deferred optional bodies, and grants no new task or
source authority.

A provider-free probe on the previously verified real captured Bottle parent
used the same authored outer compiled context, suffix, workspace and fixed round
bindings for each variant. Counts include actual 6,144-byte chunk descriptors,
the dialogue protocol and workspace advisory:

| Variant | Complete model-input bytes | Local token estimates | Estimated selection |
| --- | ---: | ---: | --- |
| Native input | 36,789 | 9,198 | Native |
| Existing `@1` packet | 37,693 | 9,424 | Native fallback |
| Exact-text-reference `@2` packet | 29,729 | 7,433 | Decision packet |

The `@2` message is 7,060 bytes / 1,765 estimates smaller than native and restores
all selected required material exactly. These counts use the automatic labeled
`utf8-div4@1` fallback; the installed tiktoken version is outside the loader's
reviewed version binding. No assets were downloaded. The outer context is
authored and differs from the live trial; the native parent was verified at
capture, with no fresh producer/source admission in this probe. This is not
provider usage, billing, task-success or whole-session savings evidence.
Report: `artifacts/supervisor-task-text-projection-20261014-01/offline-probe/report.json`
at workspace root, SHA-256
`4aee559b7e8be3d6c249eb3d571914c65f0f12330b1d4e37db98cce2a0a36db0`.

Actual provider-free Harbor Compose custody also passed for the closed `@2`
nomination: UID 1000 ingress became a distinct root-owned/read-only/single-link
final inode with the exact 268-byte payload and digest. Unsafe paths and schema
mismatches were refused by host and publisher. Frozen sources stayed unchanged;
the owned environment was cleaned without auth, runtime installation, providers
or verifier execution. Evidence: `compose-probe/validation-summary.json` under
that campaign, SHA-256
`b5996977c59c02745e9fe165fe17204f710641b99418aead6c654d9abc041416`.
All 204 targeted contract/compatibility checks passed with normal mandatory AST
seals; all 71 declared source/test/runner guards stayed unchanged. These cover
exact restoration, different-text refusal, forged references, source/policy
custody, legacy packets, actual native producer/owner/worker replay, stale-source
refusal, historical input audit and nomination delivery. Evidence:
`offline-qualification.json` and `run01/verification.json` under the campaign.
The deployable archive build uses the prior immutable packaging snapshot with
exactly five allowlisted runtime source overlays. A fresh live `@2` comparison
still requires archive verification and two matched Harbor preparations.

That archive qualification subsequently passed: all 27,908 regular members were
checked, with exactly five released source changes, 27,903 unchanged members and
no additions/removals. The prior 12,709 packaging inputs remained intact, and
all 27 current capability/deadline pins verified against actual bytes. Archive
SHA-256 is `d88f032fc347ba9f16f7b01c2665941f090d5efbbc9e5ca3441dbd055afb864e`.
Both fresh Harbor dry-runs and shared task/source/archive/model/budget bindings
passed before release of a new sequential one-pair trial. The policy adds only
the explicit text-reference profile to the original capsule partition; all
earlier trial outcomes and costs remain retained. The frozen pretty-JSON live
nomination SHA-256 is
`c955e825b7eef6552bfa859079748bcfa9fb18161fcae8dffd13cfcf1e802346`;
its raw formatting differs from the compact Compose-probe nomination while
the closed policy is identical. Evidence: `runtime-campaign/build-result.json`,
`matched-pilot/intent.json` and `matched-pilot/matched-preparation.json` under
`artifacts/supervisor-task-text-projection-20261014-01` at workspace root.

That `@2` pair is inconclusive. Native input passed the original verifier and
supervisor completion with two complete observed sessions totaling 412,186
tokens. The candidate failed during pinned Codex installation: `nvm install 22`
could not resolve version 22. This is the observed remote installer failure,
not evidence that Node 22 does not exist. It occurred before qualified deployment,
nomination publication, driver or model execution. Candidate native usage and
reward remain unavailable; the strict paired reporter withholds savings metrics.
Both owned containers were removed without retry. Across all six retained trials,
the known observed subtotal is 1,357,279 tokens; the complete total remains
unknown because missing receipts are not zero. Evidence: `pilot-result.json`
under that campaign, SHA-256
`474c32ffb37767c968d1db2ea87b3c18684f473a37f00da11bb6df35e1767f28`.

The next setup revision uses an explicit optional cached native CLI bundle to
remove NVM resolution from that profile. Exact cached Codex 0.158.0 ARM64 package
integrity and its two previously exposed native-bin members were independently
verified; no new download is needed. The initial GNU target assumption was
incorrect and is retained as unqualified metadata. A new immutable manifest
uses the package's actual `aarch64-unknown-linux-musl` vendor path with identical
binary bytes. ELF architecture alone does not establish execution compatibility;
the actual original-image version/helper probe and sealed deployment tests
must pass before that profile is qualified. Other package resources remain
outside the existing two-bin coding profile. Evidence at workspace root:
`artifacts/supervisor-portable-codex-20261014-01/corrected-musl-runtime-qualification.json`.

That optional setup revision passed 54 focused tests with normal mandatory AST
seals and 69 unchanged source guards. Its separate original-image probe executed
the actual cached Codex `0.158.0 --version` and helper `--help` successfully;
nine invalid binding/asset cases refused before binary invocation. The original
task's tracked hashes, modes, Git HEAD and status stayed unchanged, and the owned
probe container was removed. The package is admitted by an independently pinned
closed manifest and exact executable hashes, rather than a cache-directory name.
Default installation behavior remains unchanged. Selected native/decision
archive admission now includes `terminal_deployment.py`: 24 direct capability
sources plus four deadline sources, without claiming complete import closure.
Evidence: `pre-build-qualification.json` and
`native-executable-probe/validation-summary.json` under that campaign; actual
probe SHA-256 is
`ed8df4119444803b98c92c633e727cdfc8a832bf73ec22fd8e233996466eabc4`.
The rebuilt portable archive also qualified: all 27,911 regular members were
verified, with exactly two source changes, three pinned native CLI members,
27,906 unchanged members and no removals. All 28 current capability/deadline
pins passed; prior source inputs, archive and cached assets stayed unchanged.
Archive SHA-256 is
`f5dcbbf0ca2c442848eec119150f6c788ddc06ffb9ec00316882bdc443cb428c`.
Both fresh Harbor dry-runs and shared task/source/archive/model/budget checks
passed before one new sequential native-then-`@2` pair was released. The same
portable archive is used in both arms; the original capsule partition and
`c955e825...` nomination are unchanged. No live token reduction is established
by the setup repair itself. Evidence: `runtime-campaign/build-result.json` and
`matched-pilot/{intent,matched-preparation}.json` under that campaign.

The native arm then exposed a missed integration boundary. Cached CLI deployment
qualified, but `install_worker_boundary` unconditionally called the legacy
exposure helper without the packaged binding. Its NVM lookup failed before
the supervisor driver or any provider-attempt metadata. The native failure is
retained with unknown usage and reward; the planned packet arm was not invoked.
The owned container was removed. Across seven retained trials, the known
observed subtotal remains 1,357,279 tokens and the complete total remains
unknown. Evidence: `pilot-result.json` and
`matched-pilot/execution-summary.json` under that campaign.

The follow-up repair explicitly forwards the validated deployment binding into
worker setup and revalidates existing exact protected binaries before execution.
A provider-free full deployment plus actual worker permission/termination check
must pass before another frozen live pair. It preserves the original capsule
policy and `@2` text projection. Evidence is kept in the separate
`artifacts/supervisor-portable-codex-worker-fix-20261014-01` campaign so the failed
portable setup and its partial pair remain immutable.

That handoff repair passed 38 focused tests, including 24 new cases, with normal
mandatory AST seals and 69 unchanged source guards. Its archive census verified
exactly three released source changes, 27,908 unchanged regular members and no
additions/removals; the cached CLI assets and settings are identical to the
previous portable archive. All 28 current declared capability/deadline pins
passed. Archive SHA-256 is
`f89c7ace1f4387bcb020f223758609dac607c8e6258c208189bbd8dec2f652f8`.

Full provider-free original-image qualification then passed in 143.86 seconds:
both deployment stages, repeated native CLI/helper checks, unchanged protected
binary inodes/hashes/modes, actual UID1001 candidate writes, six access denials,
worker termination, unchanged original task/Git state and owned-container
cleanup. Only a synthetic `{}` auth placeholder was used; no provider or verifier
ran, and no Node/npm or new model/tokenizer assets were installed. Routine pinned
runtime packages were installed. Qualification SHA-256 is
`047a76654208afd1be684e17c1eb8867d812e0a205f2ca4cb13b746c95fe9f63`.
Both fresh Harbor dry-runs, shared bindings and ready checks passed before one
new sequential matched pair was released on that same qualified archive.
Evidence: `qualification/validation-summary.json`, `run01/verification.json` and
`matched-pilot/live-release.json` under the worker-fix campaign. Live packet
transport and whole-session results were subsequently observed in that pair.

Both arms passed the original verifier (reward 1.0) and supervisor completion.
The candidate actually delivered the `@2` decision packet with one coding
dialogue attempt and zero expansions. Both initial/coding semantic roots matched
the retained inventory, coding inputs and historical packet rounds verified,
and all native invocation bindings matched without issues. Each arm supplied
two complete reconciled native sessions. Native input used 720,840 total tokens
(716,351 input + 4,489 output); packet input used 445,333 (441,111 + 4,222).
The strict paired reporter is eligible and records 275,507 fewer observed tokens,
or 38.22%, in this one pair. Cached input (652,032 and 390,144) is already included
in input and is not added again.

The first coding user message was 63,104 bytes for native input and 56,442 for
the packet, a 10.56% observed byte difference. Within the candidate's own
complete-alternative comparison, the local `utf8-div4` estimate was 15,828 for
native and 14,111 for the packet. Those estimates are not a verified model
tokenizer. Coding tool activity differed too: 13 native calls versus nine packet
calls, with 14 versus ten cumulative usage observations. The observed total
delta therefore cannot be assigned solely to exact task-text deduplication.
This one sequential Codex pair establishes an observed result, not causal or
general efficiency, Grok results, or configured promotion acceptance. Historical
round audits do not re-certify source freshness after publication.

Both owned containers were removed, and all 69 guarded sources remained unchanged
through live completion. All nine trial records, including three earlier failures
with missing usage, remain retained. Their known observed token subtotal is
2,523,452; the complete history total remains unknown. Evidence:
`matched-pilot/comparison/{paired,binding-evidence}.json`,
`all-retained/comparison.json` and `pilot-result.json` under the worker-fix campaign.
Final result SHA-256 is
`617a541ca0b2a80ca651a1de31940805436cfd21550da825f28836726c779629`.

An independent artifact-only duplication audit confirmed the historical
semantic packet's four exact task strings occupy 15,474 canonical JSON string
bytes; the two distinct literal roots retained by `@2` occupy 7,737 bytes.
These are field-level structural counts, not complete-message or native-token
measurements. The bounded retrieval loader also returns `query_text`, which
enters required worker chunks; its full vector snapshot stays outside them.
The historical inventory did not retain that loader envelope, so its query
equality and byte cost remain unknown. Persisted descriptor/receipt duplicates
do not establish model costs. Evidence: `duplicate-text-audit/report.json` under
the worker-fix campaign, SHA-256
`994b5aed7fa6297412bba8f7c874c598deaf5b2343c53d8c160daa35eb33a7c2`.

### Provider-free retrieval and selection audit (2026-10-14)

A fresh original-image capture now retains the authenticated raw retrieval
parent, its referenced snapshot, exact bounded loader text, native chunk
records, all 531 native capsules and their freshly assessed admissions, and
6,261 verified native CAS blocks. Target Codebase IR preparation used the
cached model and `train_autoencoder=True`, matching production preparation.
All 6,289 exported file hashes were independently checked. Thirty explicit
deployed source files match the released runtime; the nine selection/repack
leaves also match the current canonical sources. These are declared partial
inventories, not a full import-closure proof.

Capture qualification verifies both parents and source/task bindings before
and after export. Preparation intentionally adds the public instruction and
smoke test in the disposable repository; original tracked bytes/modes are
preserved, and the resulting prepared HEAD, status and tracked files remain
stable throughout capture. The original host task is untouched. No provider,
credentials, verifier, new model/tokenizer assets or Node/npm were used. The
owned container was removed. The first attempt's fresh-state-directory refusal
is preserved; a separate second attempt corrected only the export directory
placement and completed in 234.63 seconds. Evidence:
`artifacts/supervisor-retrieval-capture-20261014-01/run02/validation-summary.json`,
SHA-256 `953306204279948ff22755b1a387c78c35494b44893e294b61f04d20da2223b1`.

The current retrieval envelope is 8,178 bytes and contains five hits. Its
3,873-byte `query_text` exactly equals the required public instruction in the
semantic packet. An artifact-only required-reference prototype retains that
literal once in an existing `@2` packet, keeps every worker capsule required,
and replaces only the duplicate retrieval query. All status, source, index,
query, result, hit, stale-path and authority fields remain intact. Its 6,092-byte
view restores the exact current-loader envelope and complete original chunk
records, including order and coverage. All 23 authored adverse cases refuse;
Unicode restoration also passes. No capsule optionality is inferred.

The complete authored alternative, including identical core, suffix, workspace
advisory and required semantic target, measures 40,227 bytes for native retrieval
and 37,376 for the reference: 2,851 fewer bytes, or 7.09%. The automatic local
resolver for `grok_cli` / `grok-4.7` uses its labelled `utf8-div4@1` fallback
(`unsupported_tiktoken_version`), yielding estimates of 10,057 and 9,344.
These are standalone authored checks and historical-message estimates, not
sealed repository tests, a captured full admitted/provider message, a compatible
remote tokenizer, task acceptance or whole-session token savings. Evidence:
`query-reference-probe/report.json` under that campaign, SHA-256
`7ef0c44072d3f0e7ff983da4006f84d03b26c7b214f577324dc3a5dad2651cf0`.

The relevance audit reproduces the captured worker's exact 27,276 bytes. Its
lexical selector ranks `bottle.FapwsServer.run.app` first because the instruction
contains `app`, `bottle`, `py` and `run`. Of 531 symbols, 473 tie at two matches;
the five learned retrieval hits occupy lexical positions 174, 129, 173, 399 and
47. Each joins uniquely by exact path/name and source hash. The preliminary
lexical selection contains three capsules, but final serialization exceeds
32,768 bytes and drops the tail until only the first remains. Retrieval-first
ordering similarly retains only one capsule, at 30,577 bytes. Neither ordering
establishes relevance to the actual defect or sufficient dependency coverage.
Required raw paths, source-fetch obligations and existing pack obligations stay
unchanged. Evidence: `relevance-audit/report.json`, SHA-256
`ffde8a7ed2dd8fefac415a9bd5fa0619bb60f0b1492e4ba788f9516b0cb7d753`.

An additional complete-size packing probe confirms that the tail-drop step
discards a later capsule that fits: adding `bottle.BaseRequest.app` to the
captured prefix costs 4,900 bytes, producing 32,176 bytes with 592 remaining.
Both additive backfill and exact-size lexical skip/continue retain two capsules;
retrieval-first still retains one. The canonical pack adds required conservative
assurance markers for each selected capsule. The first utility incorrectly
required its complete assumption list to remain identical; its failure and the
intermediate strict result are preserved. The corrected comparison retains and
counts exact native admission-derived assurance, requires all original
assumptions as a subset for additive backfill, and keeps literal task, scope,
source, obligation and source-fetch fields unchanged. All held source/input
pins remain stable. This demonstrates capacity, not relevance, sufficient
coverage or reduced input/session tokens; adding a capsule makes the worker
payload larger. Evidence: `relevance-audit/backfill-probe/run03/report.json`,
SHA-256 `b24f58327a3d4681aac4e07a7f5bfbc3a61019adeceac1f4497f41da0bcaa5c5`.

These results support exact sizing and nominations as separate experiments.
They do not require moving the whole repository into a minified state: preserve
canonical source and construct reversible, independently bound decision views.
A compact view saves input only when its full rendered form displaces more
text than its references, wrappers and later expansions add.

Production query references need additional parent custody. The implementation
daemon authenticates the raw retrieval SHA, then chunk rendering retains only
derived-text identity. The existing task-context rebind already authenticates
the raw path/SHA; carry that independent pair through the signed owner launch,
router projection, current dialogue checks and historical reconstruction. Reuse
the current owner and round interfaces rather than introducing another state
owner. A packet-only rewrite cannot establish that binding. This campaign changes
no production selector, packet policy, default, compiler or dialogue code, and
does not change the earlier live pair. Public-instruction suffix deduplication
still requires separate review of the mandatory text contract.

## Opt-in retrieval query references (2026-10-14)

The production integration now carries the independent raw retrieval path, SHA-256
and task identity through the existing signed owner launch. The new
`semantic_retrieval_query_reference` option defaults to false. It requires the
full indexed arm, an owner nomination with explicit required task text, and
`disable_semantic_typed_nomination=True`. The existing typed nomination path
continues to use its existing schemas and source inventory.

The host obtains the authenticated task-context bundle from native preparation
or rebind. It signs the complete owner-only triplet
`--semantic-decision-context-bundle-artifact`,
`--semantic-decision-context-bundle-sha256` and
`--semantic-decision-task-cid`; callers supply no independent raw retrieval
path through benchmark configuration. The owner verifies the admitted task
CID/key, the semantic parent and the bundle's raw retrieval pair before
publishing a closed `supervisor-semantic-decision-launch-policy@3` descriptor
outside the candidate. It checks the bundle, nomination and both parents again
after publication. The policy-only legacy loader refuses this descriptor
rather than silently dropping its extra binding.

The packet remains the existing `@2` semantic decision packet. Only retrieval
`query_text` that exactly equals its required literal task text becomes a
required reference to that target. Complete retrieval status, source/index/query/
result identities, scores, hits, stale-path information, coverage and authority
flags remain present. Native reference records, their positions and independent
raw identity are retained; exact restoration is checked. No capsule is made
optional by this feature, and selection, packing and mandatory instruction
suffixes retain their existing contracts.

Current projection and dialogue revalidate both parents around complete-message
counts, calls, native fallback and expansion. The existing router still selects
a candidate only under its bound complete rendered counter and budgets.
Historical input audit authenticates the raw retrieval artifact, referenced
snapshot, native search replay and original retained CAS source bytes. It
reconstructs historical input without claiming current source freshness.
This representation is independent of tokenizer choice; local estimates and
remote compatibility retain their existing labels and limits.

The opted-in declared archive inventory adds the new leaf and its two existing
retrieval/task-bundle dependencies. Its 34-source inventory exceeds the old
32-source population limit, so that finite limit now derives from the largest
declared inventory. The default inventory remains 31 sources; all source names,
byte bounds, manifest/archive membership and SHA checks still apply. These are
explicit direct-source inventories, not a full imported dependency-closure proof.

Validation qualifies 464 actual passed calls across all
four new suites and ten affected legacy suites, with zero skips or setup/teardown
failures. Each retained call has a mandatory AST seal; completed attempts have
517 stable before/after source guards. Complete successful suites from earlier
stopped-on-failure attempts are reused only after checking that every later
changed source is absent from their recorded closures. Two unchanged low-level
producer/task-text suites are outside this run; the new fixtures exercise the
genuine native producer and task-text behavior with authored vectors, counters
and provider callbacks.

The failed/interrupted attempts remain available. The first interpreter failed
before collection; two optional-tracing runs were terminated with partial
evidence. The focused runs then identified and corrected a proof-JSON encoder
being used for decimal retrieval scores, two invalid legacy-policy test
constructors, and the stale archive source-population ceiling. The last terminal check found
an existing owner-launch assertion that omitted the typed flag already present
in the captured baseline; its two literal expectations were corrected while
preserving the production default and checking the explicit disabled form. The retrieval
receipt now uses the same finite JSON encoding as its inverse and historical
audit; formal proof float rules remain unchanged.

Evidence is in
`artifacts/supervisor-retrieval-query-reference-20261014-01/`:
`baseline-sources.json`, `implementation.patch`,
`qualified-completed-suites.json`, each run's release/result and JUnit files,
and `implementation-result.json`. The source census contains eleven changed
existing production files, one corrected legacy test, one new helper and four
new test files. Other captured code sources and all captured permission modes
remain unchanged. This campaign invokes no real provider,
downloads no model/tokenizer assets, runs no benchmark, rebuilds no runtime
archive and deploys nothing. An archive including the current opted-in source
pins is still needed before a live trial. The earlier live pair and the
7.09% authored byte comparison retain their original scope; this change
establishes no new accepted-task or whole-session token-savings result.

## Model-facing context contract

Create a versioned projection of already verified inputs. Keep its complete canonical origin and coverage records outside the prompt, bound by identity and retrievable through the supplied owner/resolver. A projection must declare which items are required, selected, deferred, unavailable, stale, or rejected, and why. A compact projection is not itself the full source/capsule artifact.

| Context | Required in the effective decision context | Possible optional projection |
| --- | --- | --- |
| Task and owner | Literal public instruction; complete goal, authority, allowed files/scope, acceptance, prohibitions, task/root/policy bindings | Supervisor accounting and internal signed admission blobs may remain outside the model message after owner verification. Required restrictions remain explicit. |
| Source | Required target, surrounding, test, configuration, and raw-required source inputs; exact originals before edits | Optional unrelated source bodies may use content-addressed handles. The resolver must return the correct current bytes before relying on or editing them. |
| Dependencies | Dependencies needed to satisfy declared coverage; non-substitutable, stale, opaque, heuristic, or raw-required dependencies remain raw or cause refusal | Other dependency capsules may be nominated by relevance and dependency closure, then freshly admitted. Their optional bodies may be deferred with exact native identities. |
| Retrieval | Required coverage or a hit already necessary to resolve the current decision | Other BM25/graph/vector hits are nominations, not proof. Keep bounded paths, symbols, locations, and sidecar references; defer bodies and full indexes. |
| Formal evidence | Pending proof obligations, supported semantic-family requirements, counterexamples, and authoritative acceptance checks | Full solver logs, already closed history, and duplicate derivation provenance may be handles when the decision does not require them. A family router must send a solver only semantics it supports. |
| World and retry history | Current task/plan roots, required dependency status, latest distinct actionable failure, relevant changed paths | Unrelated task history, duplicate failures, and superseded observations may be paged rather than replayed. Stale status and missing components stay visible. |

Do not demote authoritative required inputs merely because they are expensive. Derive the required set from the admitted task and source/capsule contracts before ranking. Preserve dependencies and raw-fetch obligations across selection, expansion, refresh, and retry. If the mandatory set cannot fit the existing ceiling, refuse or use an already permitted task split; never truncate silently or increase the window.

Formal logic should structure obligations, dependency links, effects, and residual uncertainty so the supervisor does less repeated reasoning. The capsule producer's freshness/substitutability rules remain authoritative. Existing Lake and semantic acceptance gates determine success; metadata or a compressed formula cannot replace those gates.

## Staged implementation

| Stage | Files and changes | Meaningful validation and acceptance | Opportunity cost and risk |
| --- | --- | --- | --- |
| 1. Establish accounting | Extend existing router receipts/context audit with component identities, complete rendered candidate counts, tokenizer provenance, selected representation, and a task/attempt/session ledger. Keep raw prompts private. | Reconcile native receipts with recorded cumulative usage, retain missing categories as unknown, deduplicate repeated receipts for the same request/session, charge every fresh retry/expansion session, and distinguish complete from interrupted sessions. No savings claim with missing denominators. | Low runtime overhead if computed once at existing boundaries. Accounting must not double-count cached input, cumulative records, or already attributed calls. |
| 2. Bind production inputs | Explicit descriptor/hash/budget forwarding through Harbor, planning and signed coding commands is implemented for already deployed child-local inputs. Automatic installed-model mapping and labelled fallback estimates are available. Add portable authenticated asset delivery and compose the supplied production owner admission. | Wrong explicit model/runtime/assets, signature, path, owner generation, or source/import/environment binding refuses before dispatch. Counter failure cannot enable a compressed candidate. No discovery/download or credential inheritance. | Compatibility is an external input, not something these code changes prove. Native fallback remains useful when optional comparison is unavailable; estimates cannot establish a strict token ceiling. |
| 3. Expose database measurement | Optional counter forwarding through `database_context.compile_manifest_to_capsule` is implemented and tested. Wire the production-bound caller, retain conservative selection estimates, and count the final full provider message after wrappers and suffixes. | Existing database-context and compiler tests plus complete prompt-boundary and overflow cases. The invariant core and required coverage remain identical; misleading member token hints cannot bypass limits. | Moderate plumbing cost. Approximate ranking is acceptable only when labeled and followed by exact final measurement under the production binding. |
| 4. Separate required evidence from optional evidence | Estimator-only role projection remains available. A separate explicit-policy capsule packet and bounded producer/source-verified expansion API support optional capsule-body deferral through trusted library callers and the opt-in Codex Terminal worker. Owner-time task/parent policy construction, pinned host delivery, and the default-on typed nomination hook with immutable lineage are implemented and qualified at library/owner/accounting boundaries. The runtime archive has been rebuilt and the explicit typed-opt-out signed worker and full historical observers are qualified; derived-policy signed container deployment remains pending. The explicit opt-in required retrieval-query reference and signed extra-parent launch are implemented and locally qualified. Broader retrieval/world projections and capsule classification remain open. | Tampering, changed independently held policy, omitted required source/dependency/acceptance, requiredness downgrade, stale parents, foreign scope, or unsupported expansion refuses. Preserve all admissions and exercise native producer/source races. | Packet and handle overhead can outweigh savings; expansion can erase them. Production callers must derive the required/optional policy from admitted task coverage and retain policy custody outside model input. |
| 5. Improve nominations | Compose bounded metadata hits from `SupervisorMetaIndex`, existing retrieval results, and dependency links; preserve provenance/diversity and raw-fetch requirements. Reuse existing capsule admission after nomination. | The same task/source scope, deterministic tie breaks, no full vector matrices in prompts, dependency closure maintained, corrupt/stale rows rejected or labeled before use. | Better selection can save more than byte minification, but extra queries and expansion calls can erase savings. Start with existing lexical policy as a matched control. |
| 6. Reuse existing codec and minification | Apply compact canonical JSON and the current reversible identifier codec only to supported typed packets. Add a versioned harness adapter only if needed. Compare complete alternatives after required suffixes and workspace instructions. | Exact native restoration; unchanged code/comments/names/literals/requirements; no forged table or missing source inventory; measured count strictly smaller before selection. | Lower semantic risk than rewriting source. Alias-table and instruction overhead can make small inputs larger; retain native then. |
| 7. Reduce retries and unnecessary calls | Use existing Doctor residual route, parent-bound retry deltas, bounded question-specific expansions, and stable prefix rendering. Retain actual provider cache observations separately. | No unchanged required context lost; changed scope/model/root invalidates reuse; exact replay avoids duplicate calls; deterministic candidates still pass existing completion/publication checks. | Larger potential whole-session benefit. Retries and tool results often dominate; delta storage alone does not mean a fresh stateless provider receives fewer tokens. |
| 8. Run matched Terminal Bench comparisons | Compare the same task population, provider/model/reasoning/temperature, context ceilings, hardware/concurrency, validation, and versioned source snapshot. Include failures, replays, expansions, and interrupted runs. | Complete required coverage and unchanged task acceptance; fully reconciled all-round input/output/cache/reasoning categories; report per-task and aggregate outcomes, bytes, calls, wall time, and uncertainty. Apply existing configured promotion requirements without weakening them. | Real runs cost time and provider tokens. Proceed after offline contract tests and production bindings, and avoid repeating expensive campaigns without a specific new hypothesis. |

Changes belong in the listed existing interfaces. Avoid introducing a second context compiler, capsule schema, retrieval service, state owner, or compression database unless the audit identifies a concrete unsupported responsibility.

## Tests and evidence to extend

Reuse current tests rather than replacing the source-provenance guard or inventing fixture authority:

- `test/api/test_router_prompt_selection.py`, `test_router_local_prompt_tokenizer.py`, `test_local_prompt_tokenizer.py`, and `test_router_implementation_runner.py`: complete-message counts, suffixes, unavailable/failed measurement, binding mismatch, exact boundary and overflow, and native-versus-encoded choice.
- `test/api/test_agent_supervisor_database_context.py` and `test_agent_supervisor_context_compiler.py`: frontiers/pages/deltas, unchanged invariant core, authoritative coverage, actual counter plumbing, stale parent rejection, and budget refusal.
- `test/api/test_supervisor_meta_index.py` and `test_agent_supervisor_quack_state_client.py`: bounded discovery, unknown/stale identities, closed named operations, owner fences, DuckLake observational status, and refusal to attach exclusive control state.
- `test/api/semantic_state/test_semantic_context_runtime.py`, `test_semantic_capsule_selection.py`, `test_code_retrieval_context.py`, and `test_intent_world_snapshot.py`: native capsule integrity/admission, required raw paths, explicit omissions, dependency closure, and unchanged world requiredness.
- `test/api/semantic_state/test_semantic_decision_packet.py` and `test/api/test_semantic_decision_dispatch.py`: independently held selection policy, complete capsule partition, unchanged admissions/source/assurance, authenticated native producer, bounded full/subset expansion, concurrent source changes and CLI refusal until a provider policy/tool bridge exists.
- `test/api/test_semantic_decision_dialogue.py`, `test_semantic_decision_provider.py` and `test_semantic_decision_bridge.py`: strict supervisor continuation, initial native fallback, repeated-input measurement, bounded callbacks, stale final rejection, failed/unbound receipt retention, native cumulative reconciliation and pinned tool-free Grok/existing Codex adapters. Provider calls are authored fixtures.
- Existing native harness/materializer and Terminal Bench context/translation audits: fresh owner/source checks, exact restoration, no provider effect during preparation, changed identity requiring a new attempt, and accurate historical-versus-current evidence labels.

Use real pinned producer modules and isolated artifact-local owners. Tests must run with normal conftests, source seals, and fresh evidence catalogs. Provider-free fixtures establish contract behavior only; they do not prove remote token compatibility, model fidelity, task correctness, or a Lean admission.

## End-to-end ledger and reporting

Key each observation by task, attempt, invocation, parent/context identity, source/root revision, provider/model, tokenizer revision, and selected transport. Preserve baseline and candidate population membership, acceptance status, input scope, and completeness.

Record complete rendered user-message bytes/counts; invariant and selected/deferred evidence identities; expansion input/output and resolved references; retry and planning/coding calls; native input, cached-input, output, reasoning, and total categories when actually supplied; wall time; task validation and completion outcome; and unavailable attribution. Parent-bound deltas and content identities are useful linkage, not proof that the provider reused context or a cache.

Use the existing `recover_codex_usage` latest cumulative observation for the exact thread and working directory; deduplicate a session/thread identity referenced by multiple attempts or receipts while charging every genuine new retry or expansion session. Conflicting or nonmonotonic snapshots remain incomplete/quarantined evidence rather than an invented exact total. Component counts and local expansion/prompt counts explain attribution; do not add them again on top of their native cumulative usage. Do not add cached input to input or reasoning output to output when they are subset categories. Do not sum cumulative usage snapshots. Tool-result text is commonly already counted in subsequent provider input; the separate `ProviderTokenUsage.total_tokens = input + output + tool_tokens` convention must not be reused blindly as the provider denominator. Use observed provider input plus provider output as the primary token total when those categories are complete, and retain any independently charged non-provider tool usage separately with explicit evidence.

Do not infer absent values as zero or invent dollar costs. For incomplete sessions, keep the observed lower bound and its reason. Report prompt-content savings separately from whole-session usage and from provider billing.

The primary comparison uses the same assigned task population and charges all attempts, failures, cancellations, retries, and expansions, with unchanged required coverage. Report success rate, retries, expansion counts, and tokens per accepted task/criterion alongside per-task raw results so a smaller first prompt cannot conceal worse completion or extra work. A matched-success subset can be a labeled secondary analysis; excluding failed tasks from the primary denominator would introduce survivorship bias. A matched run with unreconciled usage can demonstrate transport bytes and acceptance observations; it cannot establish total token savings.

The first practical experiment is an isolated provider-free comparison of full evidence, bounded database projection, admitted dependency capsules, reversible identifier encoding, and retry deltas for one unchanged task scope. Establish exact restoration, required coverage, and component accounting first. Production Terminal Bench comparisons remain dependent on the supplied production owner and compatible coding-model/tokenizer binding.


## Runtime deployment and first query-reference trial (2026-10-14)

The opt-in implementation now has a complete deployed runtime. The rebuilt
archive contains 27,914 members: fourteen changed source members, three added
source members and no removals. All retained non-source assets, bytes and modes
match the previous portable Codex runtime. The archive SHA-256 is
`cc68fcb153497b2e11a2473128a6f4a377fe99c05b17fd9baf17e4cfd2f1c47c`;
its manifest SHA-256 is
`a58684a00e4b56a1aeff3dbd16a0fb8c939240ffed955db28fe230a4583d472b`.
Actual full-image provider-free deployment and worker handoff passed, including
38 selected/deadline source pins, UID 1001 worker permissions, protected native
binary preservation, worker termination and exact owned-container cleanup.
The 464 source-sealed contract tests are retained from the preceding integration
campaign; this runtime campaign did not rerun them or qualify later shared edits.

One fresh sequential pair used the same original `fix-code-vulnerability` task,
immutable archive, Codex CLI 0.158.0, `gpt-5.6-sol` with high reasoning, model-only
Doctor dispatch, explicit nomination, disabled typed nomination, 600-second
agent budget and 1,800-second setup budget. Only the candidate enabled retrieval
query references. Local prompt counts remained explicitly labeled UTF-8
estimates; native session usage supplied the following complete totals.

| Attempt | Official reward / completed | Native input | Native output | Total | Host source check |
| --- | --- | ---: | ---: | ---: | --- |
| Plain packet control | 0 / false | 504,435 | 3,868 | 508,303 | Failed after concurrent shared edits |
| Query-reference candidate | 1 / true | 544,293 | 4,145 | 548,438 | Passed on the frozen worktree |

The control coding provider returned exit code zero, but its dialogue response
was rejected with `ValueError`; the supervisor remained incomplete. The precise
rejection cause is unresolved. Its public target file remained unchanged and
the required report was absent. The later shared edits do not establish the
cause of that container failure. All failed-control costs and original records
remain retained; no control or candidate retry was performed.

Before invoking the still-unrun candidate, execution moved into a genuine
detached Git worktree. It retains 2,819 pinned source files, all 386 prelive
checks and all 19 qualification pins. Two concurrently changed runtime files
were restored only inside that worktree. Main source edits remained untouched;
canonical datasets and kit roots stayed in use. A new explicit release mapped
source identities and checked actual Git/module origins while retaining the
original preparation/configuration and failed control qualification.

The candidate passed acceptance and its accounting covers two complete native
sessions. Its packet receipt contains matching nonempty retrieval and launch
bindings. However, the independent historical input audit timed out under the
final collector's at-most-three-second budget. The adapter retained the report,
admission and declared public outputs but exported no exact native state,
capsule or CAS parents. After normal container cleanup those inputs are
unavailable for an honest offline replay. Actual query-reference delivery thus
remains independently unverified; missing verification does not prove absence.

These two attempts consumed 1,056,741 native input-plus-output tokens, including
the failed control. Cache and reasoning subset counters are not added again.
There is no eligible accepted-task paired savings result: control acceptance and
source qualification failed, and candidate historical delivery is unverified.
The candidate's observed total is larger. Earlier successful native/packet
results and authored byte savings retain their original scopes; whole-history
usage remains incomplete and Grok was not evaluated here.

Evidence is in `artifacts/supervisor-retrieval-query-runtime-20261014-01/`:
`runtime-campaign/build-result.json`, `qualification/result.json`,
`isolated-host/manifest.json`, both diagnostic reports, the immutable original
pilot records, `relocated-pilot/` readiness/observation/cleanup and `result.json`.
The next focused work is to budget the independent audit within the existing
finalization deadline, retain bounded authentic audit inputs for replay, and
record private stable check-phase/reason codes for response rejection. Do not
start another paid comparison until those evidence gaps can be resolved.
Canonical source can remain intact throughout: reversible decision views do
not require moving the entire codebase into a minified state.


## Audit replay evidence fixes (2026-10-14)

The audit child now has up to 30 seconds within the existing driver deadline.
Before live observation, the driver allows up to 15 seconds for a closed replay
bundle. The adapter gives its transfer five seconds inside a 25-second
finalization reserve. Capture, transfer and audit failures preserve task
completion and provider accounting. Nonblocking child I/O bounds elapsed time
and output; timeout and cancellation clean up only owned process groups.

The bundle retains authenticated native capsules, transitive CAS inputs,
independently pinned task/launch/retrieval artifacts and minimal signed public
Git history at their original namespace. Private state trees, credentials and
provider transcripts are excluded. Four stable rejection phases record a
reason code and exception class while preserving the original exception,
receipt, usage and retry behavior.

Fourteen source/test paths were published after isolated qualification. The
eight recorded peer files were unchanged during publication; a separate Quack
fabric revision is included in the current main snapshot. Both checkouts
passed all 332 affected tests with normal conftests, mandatory AST seals and
proof reuse disabled. The isolated run guarded 2,851 sources;
the canonical run guarded 2,855. These are explicit source guards,
with recorded imports and per-test AST closures, not a whole-repository claim.

The provider-free authored fixture used the genuine native producer, signed
public metadata, referenced vector-snapshot persistence and production child
collector. It changed current Bottle after recording model input, preserved all original
dependencies during capture, and exported 28 files (143,360 bytes). Deleting the
original repository and state still allowed the model-input digests to match in
historical replay. Current-source freshness remains false. The single provider
callback was authored; no model, task-source code or benchmark verifier ran.
All seven disposable probe containers were removed. This qualifies the fixture
contract; original-image deployment and worker handoff need fresh evidence.

The preceding paid pair and archive remain unchanged. Candidate historical
delivery remains unknown, and no token savings are established. Direct source
inventories are now 32 by default and 35 with query references, plus the existing
deadline sources. Rebuild and qualify a runtime containing these changes before
another matched accepted-task comparison. Canonical source remains intact;
reversible decision views do not require a global minified codebase state.

Evidence: `artifacts/supervisor-context-evidence-fix-20261014-01/result.json`,
`validation/run05/`, `canonical-main-validation/run02/`, `publication.json`,
`publication-review02/` and `offline-probe/fixture-summary02.json`. Earlier failed
fixture/setup attempts remain retained with their original scopes.


## Native session cost and literal-response transport (2026-10-14)

A fresh runtime containing the preceding evidence fixes passed genuine original
task deployment, worker handoff, packaged-source replay and exact ARM64 image
admission. The new paid plain-packet control retained a verified replay archive
of 255 files (5,877,760 bytes). Independent audit verified the actual delivered
packet, historical input bindings and complete native session usage. The task
remained incomplete with reward zero: `response_contract_validation` recorded
`response_contract_refused` and `ValueError`. Its exact response condition is
unknown because only the reply digest was retained. The gated query candidate
was skipped; no retry or second paid arm was run.

The new control consumed 1,122,727 provider tokens: 22,060 planning and 1,100,667
coding. Across the three retained attempts, the known total is 2,179,468 tokens,
with 96.94% in coding. Their rewards remain 0 / 1 / 0; the earlier accepted
candidate still lacks independent historical delivery verification. Cached input
and reasoning counters are subsets, not additional tokens. These are cumulative
native session totals; a smaller initial packet does not bound repeated context
or tool output. No eligible paired savings result or whole-history billing total
is established. Canonical source can remain intact: task-scoped reversible
views do not require moving the whole repository into a minified state.

An isolated patch now preserves the Codex final-message UTF-8 bytes for semantic
decisions, including markup, entities and original line endings. It refuses a
missing or invalid UTF-8 final file rather than substituting JSONL event text.
This fixes a demonstrated transport bug; it is not established as the cause of
the paid rejection. Twelve closed response rejection reasons preserve the
original exception and existing strict response, receipt and source checks.

All 277 selected tests passed across 12 suites with normal conftests, mandatory
per-test AST seals and proof reuse off. All 2,864 explicit source guards were
unchanged. The qualification retains the earlier failed attempts. Two existing
fixtures were corrected: finalization budgets now use a controlled clock, and
fallback tests isolate automatic model discovery while retaining their original
assertions and production retry behavior. The qualifier now joins suite coverage
through actual absolute test paths. These are explicit source guards and actual
per-test closures, not a complete imported-dependency guard claim. Authored CLI
fixtures qualify the transport and accounting boundary; no model call, deployment
or runtime archive rebuild occurred in this patch campaign.

The four production edits, two new tests and two fixture corrections remain in
the isolated checkout. The bundle applies and reverses with exact bytes and
modes against its frozen baseline. Shared router, runner and fallback-test peer
edits remain untouched. Current-main integration proposals are separate and
unapplied: the runner's receipt callback interface and combined native schema /
literal response validation need qualification against the peer revision.
Before another paid comparison, integrate those small changes, qualify and
package the resulting runtime, and establish an accepted control. The next
token-reduction experiment should target cumulative coding context and tool
output; no native hard cumulative token cap is yet verified.

Evidence: `artifacts/supervisor-context-evidence-runtime-20261014-01/result.json`,
`artifacts/supervisor-decision-response-fix-20261014-01/result.json`,
`validation/version03/run03/`, `patch-bundle-version02/`,
`known-attempt-token-analysis.json` and `integration-proposal/` in that patch
campaign. Earlier failures and unverified delivery outcomes retain their scopes.


## Owner-local authenticated cold-audit reuse (2026-10-14)

The typed source wrapper now attempts verification reuse by default;
`IPFS_ACCELERATE_TYPED_VERIFICATION_CACHE=0` opts out. It caches only the
deterministic native dependency proposal. Each lookup freshly captures every
CAS block, including omitted and unreferenced blocks, and binds their CID, byte
length and SHA-256 population to the exact parent, repository identity, task,
ordered seeds, graph scope, projection mode, limits, complete native/CID source
populations and interpreter. Parent/source verification, the full symbol/source
frontier, delivered worker joins and independent lineage/policy reconstruction
remain fresh. Closing evidence-read failures refuse on both hits and misses.
An unavailable optional code cache permits only actual full native fallback.

Authentication uses a fixed private HMAC key in the protected repository
namespace. Keys and records are single-link owner files with mode 0400 in a
0700 directory; generation-addressed immutable records permit deliberate key
rotation. This authenticates the local supervisor owner, rather than a portable
federation DID or kernel proof. Actual signed admission remains independently
verified. Reuse requires a non-root Linux owner without effective/permitted/
ambient capabilities, root-owned non-writable validator files and ancestors,
the admitted startup boundary, and no native bytecode or alternate pycache.
Editable checkouts and unsupported platforms fall back to full native validation.
No additional database or service is introduced.

The final isolated archive retains the previously qualified runtime plus the
new source-wrapper implementation: 12,724 members, normal/retrieval declarations
31/34 and actual admissions 35/38. This does not qualify concurrent unrelated
canonical runtime changes. Its SHA-256 is
`0b56f36ad5b2a9141d28bf6ba917c36a6465bbb214012cf3e75b55c759ef37ec`.
The genuine 531-symbol/4,652-relation signed route produced the scoped owner
lineage, isolated UID1001 pre-provider budget refusal and complete STOP/drain.
Both unmodified full observers verified the actual original input and owner/
dialogue join. The cold subprocess qualified in 3.30255 seconds under its
unchanged ten-second limit; the full in-process observer took 30.58583 seconds.
UID1001 was unable to open the private key. All 101 focused cases passed with
fresh mandatory AST seals, unchanged guarded sources, no skipped/reused/deferred
cases and no degraded cases. Earlier fixture/seal failures, a source-guard
rejection and the genuine native cache-miss timeout remain separately retained.

Evidence is in `artifacts/supervisor-authenticated-verification-reuse-20261014-01/`: 
`qualification04/verification.json`, `build04/build-result.json`, and
`container04/typed-evidence/full-terminal-audit.json`. The public replay contains
6,280 files including the complete 6,261-block native CAS and excludes the
private cache key. Every owned container was removed; each existing MAIN
reservation closed with zero leases/waiters and unchanged TTL/configuration.

Next, instrument private cache-eligibility reasons and native verification
phases to explain the remaining in-process latency, then qualify native
wrong-origin and interrupted-publication controls on the scoped route.
Inference, CUDA/head performance, provider usage, accepted-task token savings
and task acceptance remain separate qualifications.


## Shared-router literal response and tokenizer compatibility integration (2026-10-14)

The qualified response fix is now integrated into the current shared checkout,
while preserving its newer world packets, native output schemas, source/numerical
brokers, metadata census and repository configuration. Publication changed four
production files, added two authored test files, and corrected two assertions in
one existing planning fixture. Every other captured source/configuration row
remained unchanged. Exact before/after bytes and rollback copies are retained.

Semantic Codex calls explicitly preserve the literal final response, including
JSON escape spelling, tags, entities and line endings. The native schema guard
still validates structured replies. Twelve closed rejection reasons preserve
strict acceptance checks and the original exception. Optional receipt callbacks
run after environment restoration; a callback interruption cannot replace an
already pending provider failure.

The runner again supports both complete-input/two-value and rendered-input/
four-value prompt selection, with mixed argument forms refused before counting.
Counter quality and compatibility provenance survive caller/owner/automatic
bindings. Ordinary coding calls resolve an already available offline tokenizer
or an explicitly labelled estimate; owner descriptors and injected counters
retain precedence. Existing world, semantic, census and explicitly selected
source/tool profiles retain their measurement defaults. The semantic decision
provider already resolves its counter independently. Estimated counts stay out
of measured/native usage fields, and local estimates do not guarantee a provider
budget. Complete-input accounting retains the separate native output schema.

A fresh qualification on the genuine shared checkout passed all 649 eligible
cases across 25 test files, with normal conftests, mandatory per-test AST seals,
proof reuse off, 1,947 passing setup/call/teardown reports and 11,246 unchanged
source guards. Earlier failed attempts remain intact. The isolated assessment
retains 625 genuinely unaffected passes plus 24 fresh fixture/remaining checks;
the final shared qualification re-executed every eligible case without reuse.
The fixture now expects the native schema refusal and its actual provider phase,
while retaining usage, schema custody, temporary-file cleanup and independent
planning admission assertions. Scratch allocation is outside ancestor Git roots.

Two legacy comparator cases remain explicitly deferred and unqualified. They
require the removed SLOT_POLICY_V2/slot_policy_revision codec contract. Old and
current @2 transports have different meanings, so a constant alias cannot restore
that contract. Repair needs a separate comparator/codec migration. No shared codec
or comparator was changed, and the result is not a full 651-case qualification.
The three benchmark-specific launch/input-audit/usage suites remain outside this
router scope: displaced driver/audit/helper/transfer APIs are mapped separately.

No paid benchmark model call, runtime archive build or deployment occurred in
this integration. Authored CLI fixtures establish response/accounting behavior;
they do not establish remote tokenizer equivalence, accepted-task token savings,
a complete imported-dependency guard closure or a cumulative native session cap.
Canonical source remains intact; whole-codebase reversible minification is not
required for task-scoped decision views. The remaining measurement target is
cumulative coding/tool/history tokens per independently accepted task.

Evidence: `artifacts/supervisor-decision-response-integration-20261014-01/result.json`,
`publication-result-version03.json`, `validation/phase5-aggregate/run01/`,
`canonical-validation/version04/run01/`, `legacy-comparison-deferral-basis.json`
and `benchmark-restoration-map.json` in that campaign. Before another paid pair,
repair and qualify the displaced benchmark audit forwarding/export against the
current peer interfaces, package the resulting runtime, and establish an accepted
control with complete native usage before admitting a candidate savings claim.


## Current benchmark audit, evidence and usage integration (2026-10-15)

The displaced benchmark observation chain is repaired against the current
shared runtime interfaces. Publication changed five existing production files,
corrected five existing fixtures, and added one shared accounting helper and
four test files. The two independently advancing typed-source files were
captured separately and preserved. All 11,233 other captured source rows stayed
unchanged during publication. Isolated evidence retains its original identity;
the later peer revisions have a separate publication baseline and fresh shared
checkout qualification. Exact after bytes, rollback copies and a forward/reverse
Git patch probe are retained.

Decision audit forwarding carries the seven closed dialogue/owner/task/nomination
fields. Independent historical verification joins original native prompt bytes,
receipts and owner bindings without promoting printed claims to verification.
The audit uses the genuine package roots and bounded concurrent child I/O with
process cleanup. Its 30-second ceiling clamps huge integer requests before float
arithmetic and charges preparation time. Evidence capture refuses unsupported
world, formalization and typed-lineage export families explicitly; those refusals
do not establish portable replay coverage for those families.

The adapter uses one physical deadline for the selected execution profile,
prioritizes report/native usage export, and bounds optional capture and transfer.
Attested evidence bundles pass closed size, digest, manifest and origin checks
before exclusive publication, with owned rollback on interruption. The worker
capability now pins the five restored audit/accounting consumers. Ordinary
Grok/native usage behavior is preserved. Selected decision accounting reconstructs
retained Codex cumulative sessions and independent coverage; unsupported/mixed
provider decision totals remain unknown while separate native observations are
retained. Accounting is saved before audit and refreshed after successful audit,
so newly verified typed coverage reaches the stored result and Harbor projection.
An interrupted audit preserves its exception and the earlier conservative usage.

Both fresh isolated and shared checkout runs passed all 571 cases across the
same 13 suites, with normal conftests, mandatory per-test AST seals, proof reuse
off, no deselections/skips, 1,713 passing setup/call/teardown phases and 11,251
unchanged source guards in each run. Four earlier failed attempts remain intact.
These explicit guards and actual test seals do not establish a complete frozen
closure of all imported dependencies. Authored counters, archives and observers
are unit fixtures; no paid Terminal Bench task, deployable runtime archive build
or deployment occurred. No accepted-task token savings or billing total is
established. The earlier two legacy codec comparator cases remain deferred.

The missing legacy decision producer/configuration/argument bridge remains a
separate migration; this repair does not enable those removed route flags.
Before another paid pair, qualify that bridge against current interfaces and
build the matching runtime archive, then establish an independently accepted
control with complete native usage. Canonical source remains the editing
authority. Task-scoped compact views do not require minifying and restoring the
entire codebase; they must earn savings in total provider tokens per accepted
task, including coding, tools, history, retries and amortized index preparation.

Evidence: `artifacts/supervisor-benchmark-audit-integration-20261014-01/result.json`,
`validation/version05/run01/`, `canonical-validation/version04/run01/`,
`publication-baseline-version01.json`, `publication-result-version01.json`,
`publication-proposal-version04/`, `fixture-corrections-version03.json` and
`finalization-order-fix/applied.json` in that campaign.


## Namespace-aware authenticated verification reuse (2026-10-15)

The remaining in-process latency was a cache-eligibility miss: fresh admission
loads `ipfs_datasets_py.logic.intent_ir.formalize`, a legitimate Python namespace
without a file origin. A diagnostic calling the real verifiers identified that
exact rejection. Its full in-process audit took 28.51218 seconds; its cold
subprocess took 3.27637 seconds.

The wrapper now admits standard native namespace packages only at the exact
single canonical directory implied by their module name. Directory membership
comes from the complete fresh immutable tree capture and its existing hash
commitment. Exact spec, loader, namespace-path and bound search-callback
provenance are checked before iteration. Foreign/multiple roots, symlinks,
uncaptured directories, regular-package initializers and spoofed callbacks
disable reuse. Unsupported private import layouts retain native fallback.
Loaded-module subsets remain outside the deterministic key. Both opening and
closing source, full CAS/code, owner and authenticated-record fences remain.
Reuse is default-on, with `IPFS_ACCELERATE_TYPED_VERIFICATION_CACHE=0` as opt-out.

The fresh 531-symbol/4,652-relation signed route passed both unchanged full
auditors: in-process 2.41751 seconds and cold subprocess 3.35608 seconds under the
unchanged ten-second cap. This is an approximately 11.8-fold in-process reduction
on this single public fixture. A separate diagnostic observed authenticated hits
at both fences. A real wrong-namespace-path control refused caching, invoked the
actual native bundle/proposal builders, and passed both full auditors. UID1001
could not open the owner's private key. Provider calls on the signed route were
zero; source bytes and the stopped worker boundary were independently checked.

All 135 focused cases passed together across three files, including the exact
callback-metadata forgery control, with normal conftests, fresh mandatory AST
seals, proof reuse off and no skipped/deferred/degraded cases. The isolated rerun
guarded all 12,721 runtime source files and every copied normal test/config file.
An earlier 134-pass canonical run was rejected because three peer benchmark
files changed across its guard; those concurrent edits were preserved. That run,
two intentionally interrupted collection attempts and a recovered readonly-copy
staging error remain retained and are excluded from final qualification. Local
isolated Git clones reused existing committed history without creating commits.
These fresh executions qualify test behavior and AST/source guards. The borrowed
histories include uninitialized gitlinks; reusable test receipts and proof-factory
forest admission are outside this qualification. No test receipt was reused.
The final build03 archive has 12,724 verified
members and uses the previous isolated normal31/retrieval34 declarations and
35/38 admissions, plus this source-wrapper change. Concurrent unrelated shared
runtime changes require their own integration qualification.

Evidence is in `artifacts/supervisor-namespace-verification-reuse-20261015-01/`:
`qualification-result.json`, `native-comparison.json`, `qualification03/`,
`qualification04/`, and `container03/typed-evidence/full-terminal-audit.json`.
The complete public replay contains 6,280 files and excludes the private key.
Every owned container and the existing MAIN reservation closed cleanly, with
the frozen resource configuration unchanged.

Next qualify native interrupted-publication controls and restore/qualify the
current canonical benchmark forwarding/export interfaces before another paid
comparison. CUDA and 8D/384D/768D/4096D model inference, complete native provider
usage, independently accepted tasks and whole-task token savings remain separate
qualifications. These verification timings establish no model or token savings.


## Interrupted authenticated-cache publication (2026-10-15)

Completed process-interruption-safe publication of owner-local authenticated
verification records. The previous named-temporary hardlink method could leave
both names with link count two if the process died between link and unlink.
An actual SIGKILL reproduction confirmed that the strict reader refused both
the stranded key and subsequent publication attempts. That baseline used
synthetic custody bytes and establishes no semantic verification authority.

Publication now writes and syncs an anonymous Linux O_TMPFILE inode, sets mode
0400, and links the complete inode once through its held process descriptor.
It preserves no-overwrite behavior and independently checks the published inode.
Unsupported publication takes the existing full native verification fallback.
The cache remains default-on with IPFS_ACCELERATE_TYPED_VERIFICATION_CACHE=0 as
opt-out. Opening and closing source, code, CAS, owner and MAC fences remain.
Existing invalid or hardlinked cache records remain unmodified and refused;
this prevents new crash residues rather than migrating or repairing old ones.

Four genuine native process SIGKILL controls passed: before/after publication of
the key and before/after publication of the entry. Every visible record remained
complete, immutable and single-link. The three incomplete-entry retries invoked
the actual native bundle/proposal builders. The complete-entry retry observed
authenticated hits at both fences, invoked neither builder, and took 1.330
seconds. An O_TMPFILE=None diagnostic exercised full native fallback with no
published files; it simulates unsupported publication, while the other four
cases exercise the actual deployment filesystem. The original cache namespace
was restored before the independent full auditors ran again: 2.626 seconds in
process and 3.519 seconds cold, under the unchanged ten-second cap. The signed
531-symbol/4,652-relation route made zero provider calls. UID1001 could not open
the owner key. No private key or diagnostic-case contents were exported.

All 142 focused cases passed together, including seven new atomic-publication
cases covering concurrency, collisions and write/link/directory-sync failures.
Normal conftests, fresh mandatory AST seals and before/after guards covered the
12,721 captured runtime files and all normal copied test/config inputs. No test
receipt was reused and no case was skipped, deferred or degraded. Existing Git
histories were reused in isolated clones without creating commits. Their missing
gitlinks leave reusable-receipt and proof-factory forest admission unqualified.
The isolated normal31/retrieval34 declarations and 35/38 actual admissions are
qualified here; concurrent shared benchmark changes are a separate integration.

Evidence: artifacts/supervisor-crash-safe-verification-publication-20261015-01/,
including qualification-result.json, container01/native-publication-sigkill.json,
the full audits and a 6,280-file public replay. The owned runtime and source
containers were removed and the existing MAIN reservation closed empty with
its frozen configuration unchanged. The evidence manifest uses SHA-256 and raw
CIDv1 integrity; private cache keys are excluded.

The read-only canonical review found missing source-pin APIs and semantic-option
forwarding in the current benchmark/Harbor/container entry points. Reconcile
these with the existing public-replay custody path, then qualify the actual
current canonical runtime before a paid comparison. Its current thirty-second
audit cap is distinct from this isolated ten-second qualification. Machine or
power-loss recovery, other filesystems, CUDA and 8D/384D/768D/4096D inference,
complete provider usage, accepted tasks and whole-task token savings remain
separate qualifications. No model or token savings are established here.


## Current canonical semantic option plumbing and native component qualification (2026-10-15)

Restored the current canonical option path through benchmark configuration,
preparation, Harbor, the terminal driver, Doctor argv generation and the protected
owner/worker boundary. Explicit tokenizer ceilings, native semantic input,
versioned identifier/projection controls, independently pinned decision policy
and nomination inputs, and the default-on typed nomination opt-out are forwarded
again. Host nominations use the independently hashed immutable snapshot and root
publication protocol. The shared route guard refuses unqualified combinations
with alternative renderer controls; decision-policy routes retain their Codex
boundary. Existing provider, physical resource, source-view, replay export and
usage-accounting work is preserved. Archive packaging now retains required
parent package initializers. Provider-free Doctor routes retain requested options
as observations without claiming those model-rendering options were applied.
Combinations of the decision worker with newer source-formalization, broker or
budget-profile controls remain refused or unqualified; this milestone adds no
support for those combinations.

The fresh current build03 contains 13,101 source files and 13,104
verified archive members. Its normal/retrieval declarations contain 32/35 source
pins, and the actual admission closures contain 36/39. All 563 collected test
cases passed exactly once with the normal repository conftests and fresh
mandatory AST catalog seals containing each actual test file. Every runtime
source and all 50 copied normal test/config files had independent
matching SHA guards. The normal optional receipt plugin explicitly ran in mode
off: no receipt read, write or skip path was admitted, and every test body ran.
These executions establish behavior and source/AST custody; they establish no
reusable test receipt, repository-forest admission or reusable-proof coverage.

The separate public Bottle native component fixture retains 531 symbols and
4,652 relations, with one nominated closure materializing five graph nodes and
four edges. Actual DID signature verification, altered-argv refusal, protected
UID1000 owner to UID1001 worker handoff and a complete-input-budget refusal all
ran before any provider invocation. The current independent full auditors
verified retained native input, typed lineage and owner coverage in
2.44911 seconds in-process and 3.40940
seconds in a cold subprocess. This run uses the current native 30-second audit
cap. The earlier ten-second cap belongs to its earlier frozen qualification;
that result is not transferred to this current source profile.

The current native Codex 0.160.0 Linux ARM64 executables came from the exact
official npm package. Its tar SHA256 is
`9286a7e01d500ab224c9e5b4b223b7adf5dd901fba23efd6a438316426798b17`;
SHA512 matched the independently pinned registry metadata. Complete member
inventory, per-file hashes, remote uploaded bytes, and the unchanged installer's
real version/help checks passed. This qualifies native executable staging and
version compatibility without credentials or a provider call. The older 0.158
portable cache was not substituted. The model-worker UID could not read the
owner's private cache key. The whitelist public replay contains 6,281 files and
excludes that key. Both owned containers closed; all 3 existing MAIN coordinators
closed with zero leases/waiters and the frozen resource configuration unchanged.

Earlier failed staging, obsolete helper-argument, observer and focused-test
attempts remain retained and excluded from final qualification. The stopped
qualification05 attempt with optional receipt generation selected is retained
and is not counted as a completed result. The failed qualification06 run is also
retained and excluded. The selected qualification09 is the final complete
execution with optional receipt generation off. Earlier complete executions are retained separately and are not added to the selected final count.
No failed task was reset or
accepted from an earlier fixture.

This milestone qualifies canonical option plumbing and the actual native
components exercised above. It does not establish a complete canonical Harbor
lifecycle, live provider inference, paid comparison, accepted benchmark task,
CUDA/model-head inference, provider token savings or whole-task token savings.
Next run the actual canonical Harbor lifecycle with replay delivery/custody,
then a matched provider comparison with measured usage and independent task
acceptance. Qualify CUDA and the 8D, 384D, 768D and 4096D heads separately.

Evidence is retained in
`artifacts/supervisor-canonical-semantic-forwarding-20261015-01/`:
`qualification-result.json`, `qualification09/verification.json`,
`build03/build-result.json`,
`container01/typed-evidence/full-terminal-audit.json`,
`container01/public-replay/manifest.json`, and `plan-update.json`.
The final artifact inventory and CIDv1 integrity record are written separately
as `evidence-manifest.json` and `evidence-integrity.json` after finalization.


## Actual canonical Harbor lifecycle, replay custody and advisory 8D CPU training (2026-10-15)

The actual Harbor SDK created and removed one exact owned container, and the
unchanged FullSupervisorAgent setup and run methods exercised canonical
planning, signed admission, typed nomination, the driver, isolated worker and
replay delivery. This uses the existing public Bottle ARM image pinned by
digest, its independently checked public source and the public task resource
declaration: one CPU, 2,048 MiB memory and zero GPUs. The original published
image recipe was not executed; the configured 10,240 MiB storage declaration is
recorded without claiming storage-limit enforcement. Setup made no cached
command substitution. Its native Codex 0.160.0 version check ran against the
actual installed executable. External container networks were detached before
the agent run.

This is an explicitly authless worker fixture. No real host credential was
read or transferred; a protected authored empty file satisfied the installer
input boundary. Authentication readiness remains unqualified. The independently
reviewed public semantic contract is an authored administrative interpretation,
not a proof that the code or security requirements are correct. Native DID
admission and the signed implementation grant were actually verified. The
typed owner/worker route refused a complete prompt exceeding the one-token
input ceiling before any provider invocation. The refusal did not complete or
accept the task. Unknown provider token counters remain unknown; this result
claims no measured provider usage or savings.

The genuine native historical typed-nomination verification joins the exact
owner, lineage, derived and original nominee, native input, semantic parent,
policy and launch digests. It grants no source-freshness or proof authority.
The overall input audit can remain unverified because no provider round was
dispatched; this result does not relabel that status. The versioned typed replay
profile retains the closed lineage, derived nominee and exact original bytes
embedded in that lineage. The retained origin is not mounted or dispatched;
the existing replay mounts and base profile remain unchanged.
The genuine slim public context result pins its one public initial-context
descriptor by exact path and SHA. Replay retains only that bound descriptor and
does not follow nested world, index, model or learner references. Canonical
manifest bytes are bounded at 4 MiB for the typed profile and 1 MiB for the
unchanged base profile.

This typed replay qualification covers the exercised public Bottle owner packet.
Indexed-candidate typed snapshots and typed retrieval-reference launch@3 remain
explicitly refused and require separate closure work and native qualification.
Existing untyped launch@2/@3 behavior remains unchanged. The optional diagnostic
capture of already public prepared/context-result leaves has no qualification
authority and never substitutes for canonical replay delivery.

The actual default codebase autoencoder trained in this run with task_profile
unset. The independently SHA-bound receipt reports 24
epochs, 2 sources and 358 samples;
the natively observed checkpoint has latent width 8 and loaded no Legal IR
weights. The training descriptor, receipt and checkpoint digest agree. Its
security-code@1 result remains an unverified advisory candidate with no proof
or formalization authority and no mutation of the legal state. The separately
selected CPU runtime wheel is torch==2.13.0+cpu, SHA256 `6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604`. Actual
wheel metadata, interpreter/platform compatibility, native archive verification
and installed bytes were checked. This wheel supplies runtime code; learned
model weights were not bundled or substituted. Build time imported no Torch
and performed no training; the actual advisory 8D training happened later in
the SDK container.

The fresh build contains 13,104 source files and
13,108 archive members, including three native DuckDB
extensions and the exact CPU wheel. The real canonical replay was exported,
downloaded to its named host destination and independently verified. Final
stdlib closure freshly hashes every one of its 6,512
manifest-listed files and its 14,928,004 content bytes. Replay
excludes the private state tree and provider transcripts. It grants no
execution or completion authority. Protected installed runtime files had exact
opening and closing hashes, public Bottle bytes stayed unchanged, UID1001
worker processes drained, and the actual SDK stop deleted the exact container.

The selected qualification04 execution passed all 563 collected
cases across 20 selected files, using the original normal
conftests. Each test body executed once and had a fresh mandatory AST seal
containing its actual test file. All 13,104 frozen
runtime source files and 52 copied normal inputs had
independent matching before/after guards. Actual DuckDB 1.5.5 supplied the native
extension environment. Optional test proof reuse was explicitly off; no receipt
skip, reusable receipt or repository-forest admission is qualified. Counts from
failed, incomplete or superseded attempts are retained separately and are not
added to this selected execution.

The actual Harbor implementation ran from a byte-equivalent frozen capture of
the installed SDK. All 458 SDK files are retained
and freshly hashed; original installed-source hardlinks were not relabeled as
single-link evidence. Frozen build/runtime/test source custody is the qualified
scope. Closing observations of unrelated live checkout edits are retained as
drift, without claiming that a concurrently used global repository head stayed
unchanged. Both owned production fixes, for the CPU wheel and typed replay
export, must still match their qualified sources.

All 3 owned MAIN reservations closed without changing
the frozen scheduler configuration or unrelated reservations. local01 released
its own lease and joined its heartbeat before its obsolete global-empty check
encountered another job; that reviewed failure remains retained. Later closures
verify their own acquired lease and waiter are absent. A globally empty pool is
neither required nor claimed.

This qualifies the actual provider-free canonical SDK/adapter lifecycle, replay
custody and the exercised advisory 8D CPU training. It establishes no accepted
benchmark task, authenticated provider readiness, model/provider inference,
paid comparison, CUDA qualification, 384D/768D/4096D head qualification, token
savings, source-semantics proof or reusable test-proof coverage. Next qualify
an independently accepted task with a matched provider comparison and measured
usage, then measure the remaining model heads and CUDA independently.
Separately close and qualify indexed-candidate typed snapshots and typed
retrieval-reference launch@3 before extending this replay scope.

Evidence: `artifacts/supervisor-canonical-harbor-lifecycle-20261015-01/`,
`build03/build-result.json`, `qualification04/verification.json`,
`lifecycle05/qualification.json`, `lifecycle05/actual-replay-manifest.json`,
`lifecycle05/actual-codebase-training-receipt.json`, `sdk-source/capture.json`,
and `plan-update.json`. The final evidence manifest and raw CIDv1 are written
separately after all closure checks pass. Earlier failed or partial attempts
remain retained as unqualified history.


## Typed retrieval-reference launch and canonical replay qualification (2026-10-15)

Default-on typed nomination now composes with independently admitted retrieval query references. The trusted owner derives the retrieval artifact, digest and task binding from the signed task bundle; workers cannot override those parents. New typed result, receipt and lineage revisions bind the exact native launch@3 descriptor. Legacy typed and explicit opt-out paths retain their existing contracts. On the retained 426-row, 268D, five-hit public Bottle retrieval workload, five measured searches per build after one warmup had median times of 639.954 ms before and 194.794 ms after (3.29x ratio), with identical canonical result bytes. This is a CPU search-kernel measurement, excluding model/encoder work, imports, loading and result serialization. Retrieval search now reuses the full snapshot identity within each call while the public validator independently revalidates it; export byte accounting uses running totals without skipping any source reads. The new combined export may use the existing 30-second collector bound within the unchanged remaining task deadline.

A dedicated typed retrieval replay profile retains the exact raw retrieval parent, snapshot, typed origin, derived nominee and lineage. Native historical verification reconstructs the graph, policy and query reference and checks its inverse against retained source evidence. Replay retains only the reviewed public CAS closure; indexed candidate contexts remain explicitly refused pending their separate source-scope bridge.

Qualification: 842 actual passing cases across 25 normal test files, fresh mandatory AST seals and optional proof reuse off; 13105 runtime sources independently guarded. Execution used three disjoint file groups in separate actual pytest processes, each with fresh full source guards, mandatory AST seals and its own finite existing-MAIN reservation. The checker matched their exact node union to the independently pinned planned collection; partial earlier runs were retained and excluded. Cumulative shard execution time is 2073.337 seconds; this does not qualify single-process order interactions. The actual Harbor SDK and unchanged FullSupervisorAgent setup/run exercised signed admission, the default typed retrieval owner and UID1001 worker, and delivered a verified 6512-file replay archive. One-token complete-input refusal kept provider calls at zero. Default advisory 8D CPU training also ran. The exact owned container and all owned existing-MAIN reservations were closed.

This explicit authless diagnostic does not qualify authentication, accepted task completion, provider inference, measured whole-task token savings or CUDA/head inference. Existing public Bottle ARM image and original task CPU/memory/GPU declarations were retained. New evidence: artifacts/supervisor-typed-retrieval-replay-20261015-01/qualification-result.json and evidence-integrity.json. Prior sealed evidence remains immutable.

Remaining work: separately bridge indexed context@2 program/support scope and bounded replay scratch; refresh current-source CUDA qualification for retained trained 8D/384D/768D heads, and qualify the 4096D Leanstral checkpoint with a declared batching/parity contract. The saved head-qualification-followup.json distinguishes prior trained, diagnostic and untrained evidence. A separate metadata-optimization-followup.json records transaction batching, connection reuse and narrowly scoped historical read-only replay opportunities; the normal regression and Harbor lifecycle retained default metadata behavior and the regression retained mandatory AST seals. Only the isolated search microbenchmark used the existing production read-only metadata context, as recorded in its result.
