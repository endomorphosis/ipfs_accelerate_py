# Owner-injected harness context materializer

[`OwnerContextMaterializer`](../../ipfs_accelerate_py/agent_supervisor/semantic_state/owner_context_materializer.py)
implements the `HarnessContextMaterializer` protocol through explicitly injected
owner admission, native producer reads and local tokenization. It creates no
launch authority. `MemoryContextMaterializer` and `CapturingMaterializer` remain
explicit test authorities. Authored owner and model-compatibility fixtures do
not qualify a production launch.

The adapter connects existing producer admission and local tokenization to
the semantic harness. It creates no scanner, capsule compiler,
database connection or source of semantic identity. Correct materialization is
a prerequisite for comparing compression; this design establishes no token
savings, program equivalence, task completion or Lean admission.

## Implemented owner port

Construct `OwnerContextMaterializer` with keyword arguments `provider`,
`producer_revision_cid`, `implementation_revision_cid`, `authority`, `policy`
and a tuple of `MaterializerModelBinding` instances. Inject it with
`SemanticCompressionHarness(context_materializer=adapter, ...)`. The default
is still `None`, which refuses model dispatch. There is no automatic wiring.

Each model binding supplies `provider_id`, `route`, `mode`, `model`,
`model_revision_cid` and an actual `OwnerPromptTokenizer`. Duplicate mappings
and tokenizer provider/model mismatches refuse construction. The adapter's
composite revision binds both implementation revisions, authority revision,
the captured policy, all model mappings and tokenizer descriptor/asset/runtime
provenance, and the capsule transport encoding. Changes require a new adapter
and a new replay context.

The owner implements `MaterializerOwnerAuthority`:

```python
def admit(*, bindings, producer_revision_cid) -> OwnerMaterializationSnapshot:
    ...

def verify_snapshot(*, snapshot, bindings, producer_revision_cid) -> bool:
    ...
```

`bindings` is a canonical JSON value containing repository/task/attempt/lane,
objective, scope, expected root, base commit/tree, repository path, all five
environment CIDs and the captured policy, full context pack and its harness
CID, selected model/tokenizer mapping, adapter and authority revisions.
The owner must independently admit these values and current canonical producer
source/import provenance before supplying any reader. A digest of caller input
or an always-true callback is not a production authority.

`OwnerMaterializationSnapshot` supplies `binding_cid` (the harness wire CID of
that complete binding value), owner `admission_cid`, `semantic_state_root_cid`,
`producer_state_cid`, native `semantic_index`, explicit `read_source_blob` and
`get_block` readers, `source_symbols` (the three required kinds),
`capsule_symbols` (selected capsule CID to stable symbol), and optional
applicable `invalidation`. Nominations are copied and frozen. The mutable index,
readers and invalidation must remain owner-admitted; freezing the wrapper alone
does not establish their currentness. Reader wrappers bound accepted bytes
after retrieval; owners must also bound their readers' allocation and I/O.

Owner verification brackets all producer reads and must detect intervening
changes, including changes which revert before the second check, or hold an
owner-sealed currentness lease. `verify(prompt=...)` repeats admission, all
source reads, capsule lookup and freshness assessments, then independently
reconstructs the complete canonical prompt and compares exact bytes. It stores
no last-prompt or last-admission cache across requests. Admission and producer
references remain visible alongside pack assumptions and exclusions.
Evidence must be stable for repeated admission of identical bindings and state;
minting a different lease nonce for every call makes exact verification refuse.
Owner-imposed raw-source requirements beyond producer invalidation must also
cause owner admission to refuse a selected capsule.

Current producer capsules include nested AST version evidence that can exceed
the harness depth limit even for a small function. The adapter preserves each
entire native `capsule.to_dict()` as canonical JSON text in the explicitly
labeled transport `{"encoding": "canonical-json-utf8@1", "body": "..."}`.
Decode `body` to recover the unchanged complete capsule, including its native
CID, metadata and evidence. No producer field is removed or summarized; the
outer `capsule_cid` remains the authenticated producer identity, not the hash
of a new wrapper. Fresh verification reconstructs the same complete text.
This is lossless serialization, not additional semantic admission. The 8,192
byte per-text ceiling still applies to the entire body, so larger capsules
refuse rather than being truncated. The harness depth limit remains eight and
applies to the outer transport, as do its item and traversal limits. The native
AST inside the JSON text can remain deeper; encoding it does not prove that it
meets those structural bounds or grants it any new semantic qualification.

The adapter scopes the existing `public_replay_without_metadata()` suppression
around CID validation and materialization: constructor, producer capability and capsule admission mirrors
cannot open an ambient metadata owner in that thread. This suppression confers
no admission and does not replace owner verification. No database connection,
goal insertion, reservation, network request or model call occurs here.

The local tokenizer's 256,000-byte ceiling intersects the existing harness
limits; no context limit increases. Full source blobs exceeding 8,192 bytes
refuse rather than silently become spans. The harness still checks structural,
serialized, route and provider input limits and counts the exact dispatched
prompt. Local encoding proves the count under the declared asset; it does not
prove a remote model uses that tokenizer.

Use the public harness run entry point with the admitted durable port. Native
owner execution already enters `IpfsKitNativeDurableStateAdapter.metadata_scope`,
which suppresses optional CID-validation mirrors. Calling the private rendering
method directly with a memory fixture outside that scope can open the configured
advisory metadata database; that is not evidence of production native-owner
overhead. Do not remove ordinary metadata mirroring based on that fixture alone.

## Public preparation without execution

With the owner, providers, model mapping and materializer already injected,
call `harness.prepare_model_context(request)`. The request must supply its
owner-admitted `base_tree`; preparation never creates a worktree to discover it.
This runs the existing source/capsule verification, canonical rendering, actual
local token counting and input limits. It brackets the operation with reads of
the same durable owner and refuses root changes, owner/materializer/provider
configuration replacement or cancellation. It checks an explicitly requested
root against the observed root, rather than treating the requested value as
current evidence.

```python
prepared = harness.prepare_model_context(request)
metadata = prepared.to_dict()  # Counts, hashes and bindings; excludes prompt.
packet = prepared.prompt       # Complete verified prompt, not an estimate.
```

The result is a `PreparedModelContext`, exported from `semantic_state`. It is
one fresh observation. Its metadata explicitly reports no execution, semantic,
tokenizer-compatibility or completion authority. `include_prompt=True` adds the
complete prompt to the dictionary; otherwise metadata contains only its byte
length and SHA256 alongside counts and bindings. The object is not accepted as
a dispatch token. A later `harness.run(request)` repeats its normal admission,
reservation, rendering and execution gates.

Preparation reads injected ports but never consults terminal replay, reserves
an attempt, stores blocks or roots, creates a worktree or invokes a provider.
It enters the native durable CID scope and suppresses optional metadata-owner
activation. Reader and availability implementations retain their own authority
and I/O responsibilities. Deterministic and human-review routes do not request
a provider prompt and therefore refuse this operation. No tokenizer, producer,
owner or model is discovered automatically.

## Existing dispatch contract

The protocol and dispatch boundary live in
[`semantic_state/harness.py`](../../ipfs_accelerate_py/agent_supervisor/semantic_state/harness.py):

- `HarnessContextMaterializer` requires `revision_cid`, `materialize`, `verify`
  and `count_tokens`.
- `SemanticCompressionHarness.context_materializer` defaults to `None`.
- `_materialize_model_prompt` passes the request, pack, pack CID, base tree,
  selected provider ID, materializer revision, route and mode to the adapter.
- Materialization must return exactly `required_sources`,
  `dependency_capsules` and `references`. The three required sources are
  `target_source`, `surrounding_source` and `test_source`, each with exactly
  `source_cid` and `text`. Every selected dependency capsule must have its
  producer CID and complete capsule body. This adapter uses the explicitly
  labeled lossless JSON-text transport described above.
- The harness renders one canonical packet containing the objective, scope,
  environment constraints, policy, pack and materialized data. It requires
  `verify(prompt=...) is True`, then counts that exact complete prompt before
  calling the selected provider. Missing admission, invalid material or an
  overflow refuses dispatch.

The harness checks bounded transport and bindings. It does not itself establish
source freshness, capsule substitutability or environment qualification. The
adapter must supply these checks through already admitted producer and launch
authorities. A nonempty JSON object, reference, stored row or matching digest
alone cannot provide those decisions.

## Inputs the launch owner must supply

| Input | Required binding | What it does not establish by itself |
| --- | --- | --- |
| Producer adapter | The admitted `IpfsDatasetsSemanticStateProvider` and its implementation/source revision | A bare import or path is not source admission |
| Source index and sealed reader | Expected producer state CID, stable symbols and exact source bytes | A mutable filesystem path is not a sealed source view |
| Current semantic view | Expected semantic root and read-only authenticated producer blocks | A retained root is not evidence that it is still current |
| Required-source nominations | Each of the three source kinds maps to a stable symbol and the corresponding pack source CID | Caller nominations are not producer membership proofs |
| Dependency nominations | Each selected capsule CID maps to a stable symbol authenticated by the current root's capsule index | SQL lookup or a cache wrapper is not capsule authority |
| Current launch admission | Actual tree, repository/task/objective, scope, environment and policy/interface bindings | Source bytes alone do not qualify the execution environment |
| Tokenizer mapping | Each allowed provider ID, route and mode maps to an explicit provider/model/tokenizer binding | A route label does not identify a model tokenizer |
| Adapter revision | An opaque CID binding implementation, producer authority and tokenizer/model revisions | The revision CID is not a fresh source or completion assertion |

Launch admission must be a concrete owner-controlled authority supplied at
construction, with its identity and revision bound into the adapter. It must
not be reconstructed from model text, source comments, advisory metadata or
DuckDB/Quack/DuckLake rows. If this authority is unavailable, the initial
adapter must refuse; an always-true callback is suitable only for a labeled
fixture. Storage readers may locate bytes, but all producer checks still run.

There is no automatic discovery of repositories, owners, assets, credentials,
tokenizers or models. Use the canonical datasets source tree at
`/home/barberb/lift_coding/external/ipfs_datasets`; importing the editable HACC
installation first is a different process with different producer code. Keep
the existing source-admission/import guards and do not update HACC or
`hallucinate_app` to reconcile them.

## Producer APIs to reuse

The Accelerate delegation boundary is
[`IpfsDatasetsSemanticStateProvider`](../../ipfs_accelerate_py/agent_supervisor/semantic_state/datasets_adapter.py).
It forwards producer operations and maps source failures without minting
Datasets identities.

| Operation | Current entry point | Required use |
| --- | --- | --- |
| Open authenticated semantic state | `open_verified_view(root_cid, get_block)` | Use the owner's admitted read-only block reader and exact expected root |
| Retrieve exact source | `read_required_source(semantic_index, symbol_id, expected_producer_state_cid=..., read_source_blob=...)` | Supply a sealed reader; preserve typed wrong-state, corrupt, unavailable and race failures |
| Retrieve a producer capsule | `view.capsule(stable_symbol_id)` | Authenticate root index membership and compare the returned CID with the selected CID |
| Assess current capsule reuse | `assess_capsule_freshness(capsule, current_state=view, invalidation=...)` | Supply the current admitted view and applicable invalidation obligations |
| Project reuse admission | `semantic_state.capsules.admit_capsule(..., assessment=...)` | Always pass the producer assessment; apply additional raw-source requirements without promotion |
| Check allowed substitution | `semantic_state.capsules.capsule_may_substitute(admission)` | Require a fresh exact or conservative admission, with visible caveats |

The underlying current datasets implementations are
[`semantic_state/api.py`](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/semantic_state/api.py),
[`semantic_state/source.py`](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/semantic_state/source.py)
and
[`semantic_state/freshness.py`](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/semantic_state/freshness.py).
`VerifiedSourceMaterialization` separates full `source_bytes` from
`span_bytes` and carries producer state, source CID, stable symbol and byte
offset evidence. Do not silently replace a full blob with its span while
retaining a claim that the text hashes to the full source CID.

Two compatibility traps need explicit handling:

1. [`context_pack.load_datasets_context_pack_authority`](../../ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py)
   imports `DatasetsContextPackAuthority` and `get_authority`. Those legacy
   symbols are absent from the current canonical
   [`proof_context/context_pack.py`](../../../ipfs_datasets/ipfs_datasets_py/proof_context/context_pack.py).
   Do not use this helper as an implicit production admission path or mask its
   failure with a locally minted envelope. The current module provides
   `admit_datasets_context_pack` for `DatasetsContextPack@1`; its
   `build_context_pack` remains the v0.1 compatibility builder. Neither
   construction call independently proves that nominated source is current.
   The harness's existing `contracts.ContextPack` is also a separate
   compatibility contract; do not relabel or replace its CID with a Datasets
   pack CID.
2. [`admit_capsule`](../../ipfs_accelerate_py/agent_supervisor/semantic_state/capsules.py)
   defaults to `fresh` when no assessment is supplied. The new adapter must
   require a real producer assessment, rather than allowing this compatibility
   default to authorize source substitution.

## Implemented adapter checks

These checks describe the adapter and its delegated checks. They do not qualify
a supplied production owner, model mapping or producer source/import authority.

- [x] Add one owner-injected adapter module; leave the harness default as
  `None`. Document constructor inputs and the owners of their identities.
- [x] Admit the exact request and current tree/environment before reading
  source or capsules. Require the producer source index and semantic view to
  agree with the owner-authorized current state; do not infer currentness from
  the fact that a retained block exists.
- [x] Initially support complete UTF-8 source blobs whose producer CIDs equal
  the three pack source CIDs. Return all bytes exactly, including whitespace
  and empty text when the producer admits it. Reject unsupported aggregated
  handles, missing symbol nominations, invalid UTF-8 and oversized blobs.
- [x] Authenticate every selected dependency against the producer capsule
  index, obtain its current assessment and require substitution admission.
  Preserve conservative caveats and existing pack assumptions/exclusions.
  A stale, opaque, heuristic or raw-required selected capsule must cause
  refusal or explicit preparation of a new admissible pack through the owner;
  do not silently drop it or pretend its missing raw source was supplied.
- [x] Use `references` for producer source evidence, freshness/admission
  records and explicit omission explanations. Do not promote reference
  presence into evidence that the referenced content was admitted.
- [x] Validate the exact canonical prompt during `verify`: objective, scope,
  tree, pack and CID, environment constraints, policy and materialized body.
  Recheck source/capsule/owner bindings so mutation between materialization and
  verification cannot become an admitted prompt. Avoid a mutable shared
  "last prompt" cache between parallel calls.
- [x] Count the complete verified prompt with the explicitly selected local
  tokenizer binding. Do not use `pack.token_totals`, the compatibility
  estimator, word counts or a different model's tokenizer as actual counts.
- [x] Change `revision_cid` whenever adapter, producer, launch-admission,
  tokenizer, special-token policy or model implementation bindings change.
  Preserve existing historical replay semantics: an old terminal result
  grants no new currentness qualification.
- [x] Keep storage read-only inside this adapter. Create no connections,
  roots, reservations, leases, supervisor goals or provider calls. The harness
  continues to own its durable attempt protocol and execution gates.

Subsequent support for explicit source spans needs producer-backed coverage and
span evidence, with an unambiguous statement of whether a source CID names the
full blob or the slice. It should be a separate extension, rather than a
shortcut to fit the initial byte ceiling.

## Provider/model binding and limits

[`ProviderCapabilitySpec`](../../ipfs_accelerate_py/agent_supervisor/semantic_state/providers.py)
contains a provider ID, capabilities and a context limit; it has no model or
tokenizer identity. The owner must therefore bind the actual model behind each
provider instance to the adapter's permitted `(provider_id, route, mode)` entry.
Refuse an absent or mismatched entry. Do not guess from `small_local_model`,
`medium_model`, `frontier_model` or the provider ID spelling.

The new
[`local_prompt_tokenizer.load_owner_prompt_tokenizer`](../../ipfs_accelerate_py/agent_supervisor/runtime/local_prompt_tokenizer.py)
can supply a counter from an explicit local descriptor/hash and pinned
provider/model. `OwnerPromptTokenizer.binding()` retains provider, model,
tokenizer identity/revision and special-token policy; `count()` checks the
complete untruncated text and rechecks local asset bindings. Its owner
declaration does not verify compatibility with an unpublished remote
tokenizer. Preserve that limitation and record local provenance distinctly
from observed provider usage.

Current harness ceilings come from
[`ContextBudget`](../../ipfs_accelerate_py/agent_supervisor/context/context_contracts.py):
8,192 input tokens, 2,048 reserved output tokens, 512 reserved tool tokens,
8,192 UTF-8 bytes per text string, 262,144 serialized bytes, depth 8 and at most
128 items per bounded container. The harness additionally permits at most
4,096 traversed structural nodes and applies the selected provider's context
window and route policy limits. The effective limit may therefore be smaller
than 8,192 input tokens.

The local tokenizer adapter has its own 256,000-byte prompt ceiling, so use the
stricter intersected byte bound. Its descriptor and asset ceilings are
16,384 bytes and 32 MiB respectively; these are tokenizer-input bounds, not
permission to expand the harness context. Refuse overflow before serialization
when possible and before provider dispatch in every case. Do not trim required
text, raise the context window, download weights/assets or change temperature
from 0 to make a case pass.

## Validation required before production use

The current local integration still lacks two production inputs. No exact
coding-model tokenizer binding was found in the audited local assets; the tiny
WordLevel tokenizer used by tests is explicitly authored. The existing
Datasets adapter checks API/schema versions but does not authenticate canonical
producer source/import bytes. The legal-tree pin covers its legal compiler,
decompiler and parser, not these software-contract producer modules. Existing
native source custody, signed launch admission and Accelerate import guards
are separate pieces; they do not collectively supply this materializer's
complete owner authority without an explicit admitted integration. A path or
revision digest alone must not fill that gap.

Reuse these existing entry points when completing that integration:

| Existing component | Entry point | Binding still required |
| --- | --- | --- |
| [Native repository observation](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_ir.py) | `RepositoryCodebaseIndex.observe_current()` | Retain the complete catalog generation and source/index view; an old head digest is insufficient |
| [Source custody](../../ipfs_accelerate_py/agent_supervisor/planning/finite_integer_source_custody.py) | `capture_source_custody()` and `FrozenSourceCustody.require_current()` | Keep the admitted reader and source/AST/Git inventory current throughout materialization |
| [Signed local launch admission](../../ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py) | `verify_local_benchmark_admission()` | Independently bind task, source and profile to the materializer request |
| [Receiving execution owner](../../ipfs_accelerate_py/agent_supervisor/runtime/codebase_inventory_execution.py) | `reserve_inventory_execution()`, `require_prelaunch_current()` and `require_spawn_fence()` | Preserve its inventory-worker ABI and receiving-side fence; it is not a generic materializer authority |

A future facade should compose these supplied owners and sealed readers,
construct a producer view from one captured admitted state, and recheck the
complete generation and custody. It also needs separately admitted Datasets
producer imports, execution-environment currentness and the exact provider/model
tokenizer mapping. Source observation alone cannot supply those inputs. Do not
rescan an unrelated tree or reconstruct admission from exported receipts to
complete the facade.

Existing fixture dispatch gates are in
[`test_harness_materialized_dispatch.py`](../../test/api/semantic_state/test_harness_materialized_dispatch.py).
Preparation-only checks are in
[`test_harness_prepare_context.py`](../../test/api/semantic_state/test_harness_prepare_context.py).
Public dispatch, persistent native-owner reopen and historical replay checks
are in
[`test_harness_native_materializer.py`](../../test/api/semantic_state/test_harness_native_materializer.py).
Actual authored local tokenizer integration is in
[`test_router_offline_semantic_roundtrip.py`](../../test/api/test_router_offline_semantic_roundtrip.py).
Their scopes must remain explicit: a fixture owner/provider/tokenizer is not
evidence of a production launch or remote tokenizer compatibility.

Add focused checks that execute the current native producer APIs, with
authorities and imports pinned to the canonical tree:

- [ ] Exact source, capsule and local-tokenizer material reaches the actual
  harness provider boundary; each body is checked against the producer and
  the complete counted prompt equals the dispatched prompt.
- [ ] Wrong producer state, missing/corrupt source, changed source between
  reads, invalid UTF-8 and incorrect required-source nominations refuse before
  a provider call.
- [ ] Wrong root/index membership, mutated capsule, absent freshness
  assessment, current-state mismatch and raw-required invalidation refuse.
  Fresh conservative material retains all caveats; it cannot acquire exact
  confidence.
- [ ] Changed tree, objective, scope, environment, policy/interface or launch
  admission revision refuses; a matching storage row cannot override it.
- [ ] Tampering with any required text, capsule, reference, constraint or pack
  between materialization and verification refuses. Parallel requests cannot
  exchange a cached admission.
- [ ] Missing/mismatched tokenizer mapping, changed descriptor/assets,
  truncation/padding and invalid counts refuse. Test the exact effective token
  ceiling and one-token overflow with zero provider invocations.
- [ ] Per-text, complete-byte, depth/item/node bounds remain enforced without
  silent truncation or a larger context window.
- [ ] Old terminal replay remains historical and makes no new provider call;
  changed materializer/model/tokenizer/admission revision requires a new
  attempt binding. Pending reservations and integrity failures remain closed.
- [ ] Run mandatory source guards and appropriate acceptance regressions with
  actual test execution. Report source provenance, actual passed/skipped/reused
  counts and fixture limits; compilation alone is not test evidence.

The fresh public integration run passed **78 distinct executed cases**: 23
preparation, 16 native-owner and 39 existing dispatch regressions. Normal
conftests and the mandatory AST seal were enabled; there were no skipped,
reused, deferred, degraded or failed results, and watched source hashes stayed
unchanged. The native run uses real producer checks, local encoding, DuckDB
persistence/fences and reopen. Its launch/model/provider declarations are
fixtures, and simulated dispatch correctly rejects production acceptance.
See the [validation receipt and scope](../../../../artifacts/supervisor-token-context-audit-20261011/public-preparation-native-validation/README.md)
for scripts, JUnit output, provenance and retained preliminary failures.

Seven warm public preparations on that isolated native fixture had a median
wall time of **97.95 ms**, with identical 9,688-byte prompts and 1,194
authored-tokenizer tokens. Native rows/roots and source hashes stayed unchanged;
no provider or forbidden effect occurred. These figures are fixture preparation
cost, not production token savings or legal-IR bridge timing. The attribution
probe identifies repeated producer state-CID computation as a potential next
optimization, while preserving state/race checks. Its native/SQL profile
coverage is incomplete, so missing owner frames must not be interpreted as
zero cost or permission to remove fencing.

The subsequent canonical-producer optimization computes a native state CID
once per source admission. It is restricted to exact frozen repository,
record and span types; arbitrary views and outer/nested subclasses retain
their original getter behavior. No CID/currentness decision is cached, and
source rereads, byte/CID/span checks and owner verifications remain unchanged.
The fresh follow-up run passed **178 executed cases**, including 52 new
producer-source compatibility tests. See the
[follow-up evidence](../../../../artifacts/supervisor-token-context-audit-20261011/source-state-cid-single-read-validation/README.md).

Paired diagnostics reduced state-CID computations per public preparation
from **14 to 8**, retaining two owner admissions and four verifications.
They have **not demonstrated an end-to-end speedup**. The original seven-pair
medians were 100.79 ms baseline and 101.46 ms candidate; one bounded repeat
after test workers exited measured 98.68 ms and 100.32 ms across fifteen pairs.
Exact prompts, counts, bindings and native rows/roots stayed identical.
These remain fixture timings, with no production token-savings claim.

External wall-clock attribution subsequently measured a **99.15 ms** median
over seven preparations: the two complete native root reads accounted for
**85.80 ms, or 86.61%**. First materialization, fresh verification (including
its second materialization), counting and the remaining work accounted for
5.66 ms, 5.29 ms, 1.65 ms and 0.56 ms respectively. These are independently
summarized disjoint phases; median phase times need not sum to median total
time. Root validation includes live owner fencing, the full transition chain,
root/manifest validation and present linked blocks; it is not isolated SQL
cost. This fixture has one historical transition and eleven linked fields.
The scalar telemetry originally omitted five missing environment bindings;
the saved-catalog supplement confirms that all sixteen link occurrences are
checked. Static accounting predicts 119 SQL executions per root read, 238
across the pair. These are inferred call counts, not instrumented SQL timings.
SDK methods, locks and connections, exact prompts/counts, native rows/roots and
68 source hashes stayed unchanged. See the
[attribution evidence and constraints](../../../../artifacts/supervisor-token-context-audit-20261011/native-preparation-attribution-20261011/README.md).

The missing profiler frames were reproduced with private in-memory DuckDB
1.5.5 and `SELECT 1`, without SDK or harness imports. The underlying mechanism
is unproven; their absence cannot justify removing a check. A separate bounded
microbenchmark estimates the four duplicate outer JSON parses at **9.10
microseconds, 0.0092%** of that preparation time. It excludes mandatory I/O,
hashing, canonicality validation and fencing. No runtime change was made for
this negligible fixture cost, and this estimate is not an end-to-end speedup.
Larger payloads and histories require their own measurements.

Keep both root reads, all callback configuration captures and every fresh
owner/history/present-block check. Native `has()` already authenticates bytes;
the following `get_bytes()` still provides an independent owner/content check
that can reject an intervening change. Do not replace these paired calls with
one result, cache currentness from a CID or add an outer lock around methods
that acquire a plain `Lock` themselves.

Ten new native paired-read regressions passed in a fresh 82.92-second run,
with normal conftests, nonempty mandatory AST seals and unchanged source
hashes; no results were skipped, reused, deferred or degraded. They cover
actual SDK A-to-B-to-A revision changes, owner/namespace drift and same-CID
linked/root/older-transition corruption between tokenization and the closing
read. Each verifies exact refusal mapping and that preparation adds no
provider call or write beyond the deliberately authored fixture mutation.
See
[`test_harness_native_preparation_races.py`](../../test/api/semantic_state/test_harness_native_preparation_races.py)
and the [fresh receipt](../../../../artifacts/supervisor-token-context-audit-20261011/native-preparation-attribution-20261011/guarded-native-races-r2/verification.json).
The initial fixture-setup failure and its native metadata-scope correction are
retained separately; their preliminary passes do not increase the final count.

The adapter does not loosen semantic patch acceptance or modality gates.
Only an actual `lake build <Lib>` can grant Lean admission. A compiler result,
decompiler sentence, loss, bridge target, NCA cell or DuckDB row cannot do so.
The US Constitution remains unformalized and its spans must not be marked
`roundtrip_ok` by this work.
