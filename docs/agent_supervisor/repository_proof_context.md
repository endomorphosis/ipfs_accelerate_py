# Checked local proof evidence in repository context

`RepositoryProofContextAdapter` joins one existing repository context adapter
to one live native header-proof-cache admission. It adds the complete local
proof conditions as required evidence and a bounded, optional proof handle.
Compilation, expansion, retry and rendering retain the current source generation
and recheck the original proof owners. A copied receipt, a capsule digest or an
index row saying `SATISFIED` cannot supply this admission.

The [qualification record](evidence/repository-proof-context-20261014/README.md)
describes the executed tests and their authored prover seam. This is the first
backend of the [indexed-proof context proposal](evidence/native-source-worker-spawn-20261014/indexed-proof-context-followup.json).
It uses the existing DuckDB repository index and context compiler. It creates
no new database, model, decoder, training process or solver owner.

## Admission and source identities

The trusted host supplies the original `RepositorySourceSession`, its existing
`RepositoryContextAdapter`, the original signed scoped analysis, a live
`HeaderProofCacheAdmission`, and independently reviewed expected identities:
the complete `DoctorProofCacheKey`, receipt ID, cache-owner path hash, signed
manifest CID, task CID, scope ID and source generation. Expected values taken
from an untrusted cache reply do not constitute independent review.

`bind_repository_proof_source` first joins the complete signed regular-file
population to the retained source session, including files omitted from the
local theorem's scope. Source bytes, executable bits and the directory closure
must match. Additional files, symlinks, gitlinks, nested Git roots and unmatched
empty directories refuse. The retained native index, package, materialization,
CAS and physical owner identities remain bound. This initial backend supports
a fully signed dirty candidate; it does not establish a committed Git snapshot.

The identities have different meanings:

| Identity | Meaning and use |
| --- | --- |
| Header key `tree` / `candidate_tree`, receipt `repository_tree_id` | Original signed task-scope CID. Keep its native namespace unchanged. |
| Signed local source-tree ID | Complete source population admitted by the signed analysis. Includes scoped and omitted originals. |
| Session `source_tree_sha256` | Source-view digest used by the existing repository adapter and capsule `tree_id`. |
| Session/index repository ID | Native index identity derived from its private state owner; it can differ from the proof repository ID. |
| Git tree object ID | A committed Git tree. This backend makes no equivalence claim for it. |

The public binding records both repository identities, both source identities,
the task scope and the generation. It explicitly sets
`git_tree_equivalence_verified=False`. A generic proof key containing a Git tree
needs a separate verified correspondence backend before it can be admitted.
Relabeling the session's SHA256 digest as a Git tree or task-scope CID is invalid.

## Using the retained owners

The implementation is in
[repository_proof_context.py](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_proof_context.py),
[repository_proof_source_binding.py](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_proof_source_binding.py)
and the pure
[public projection](../../ipfs_accelerate_py/agent_supervisor/runtime/repository_proof_context_projection.py).
The projection formatter does not authenticate a receipt or establish currentness.

After native import and independent review, the host binds the existing owners:

```python
source = bind_repository_proof_source(
    session=session, scoped_analysis=scoped_analysis,
    generation=generation,
    expected_manifest_cid=reviewed_manifest_cid,
    expected_task_cid=reviewed_task_cid,
    expected_scope_id=reviewed_scope_id,
)
proof_context = RepositoryProofContextAdapter(
    context_adapter=context_adapter, gate=gate, admission=admission,
    source_binding=source, scoped_analysis=scoped_analysis,
    expected_key=reviewed_complete_key,
    expected_receipt_id=reviewed_receipt_id,
    expected_cache_owner_path_sha256=reviewed_cache_owner_hash,
    generation=generation,
    expected_manifest_cid=reviewed_manifest_cid,
    expected_task_cid=reviewed_task_cid,
    expected_scope_id=reviewed_scope_id,
)
```

`compile(**arguments)` accepts the existing adapter's compilation arguments.
Declare the complete caller-required reference and coverage populations. The
wrapper preserves their canonical bytes and adds its own required conditions.
Use the returned original capsule with this wrapper's `validate_current`,
`expand_for_question`, `retry` and `render` methods. A detached but identical
capsule or a capsule issued by another wrapper refuses. Ordinary adapter
validation does not replace these proof-aware consumer boundaries.

`render(parent, generation=...)` returns canonical bytes of the existing
`provider_input_payload`. The caller accounts for provider framing and usage,
and repeats validation after asynchronous work before relying on a result.
The caller also retains ownership of the session, index, compiler and proof
cache. `proof_context.close()` retires only the wrapper's retained parents.

Refresh and edits require a newly reviewed binding and wrapper. A refreshed
session with identical source bytes still does not revive the old generation.
There is no automatic repair of an expired, poisoned, quarantined, tombstoned
or missing cache entry. Refusal leaves the proof obligation unresolved.

## Required conditions and bounded handles

The required invariant contains the original local scope, complete public
theorem, complete public source premises, assumptions, residual obligations,
open frontiers, source correspondence and original proof identities. Independent
SecurityIR obligations remain unresolved. The wrapper does not create execution,
mutation, completion, omission or whole-program proof authority.

The optional `repository-proof:` reference is a digest handle pointing to its
required `repository-proof-conditions:` invariant. The compiler may defer that
handle under its existing selection policy. The wrapper accepts only the
compiler's exact deferred representation with its declared exclusion reason.
Deferral cannot remove or truncate the required conditions. An explicit retry
can request the original proof handle while preserving all required evidence.
Both reference prefixes are reserved against caller replacement or promotion.

The required conditions are bounded to 32 KiB of summary and the complete public
source-binding/theorem/premise projection to 256 KiB. Oversized conditions refuse rather than silently dropping
premises or residuals. Larger proofs will need a reviewed paginated condition
contract whose dependency closure remains mandatory; that backend is pending.

## Closing checks and limits

The wrapper retains the exact original native owners, source admission, complete
key and receipt bytes, compiler policy, budget, estimator and tokenizer identity.
It snapshots caller inputs before callback-bearing operations and checks them
again before returning. Parent bytes, caller-required references and complete
conditions cannot change during compilation, expansion or retry.

Native source observations can invoke SQL, CAS, scheduling or caller callbacks.
Closing proof eligibility is checked after those observations, followed by a
complete filesystem-only signed source fence. Short-held entry locks reject
concurrent or recursive use without holding a wrapper lock over callbacks.
These are sequential currentness checks, not a filesystem writer lease or an
atomic transaction across independent owners. Validate at each consumer boundary.

The initial backend reuses native admission checks; it does not rerun Lean or
Z3 during context retrieval. Generic `FormalVerificationCache.lookup` restores
the declared full key and receipt, but that alone does not independently attest
an external kernel execution. `CodeProofQuery` digest annotations likewise do
not replace a reviewed complete key and live admission.

## Next integrations and measurements

1. Integrate this optional bridge into a trusted supervisor consumer after
   joining the actual native task, allocation, signed profile and compiled
   instruction. Qualify the permission and cross-process handoffs in the
   [existing deployment proposal](evidence/native-source-worker-spawn-20261014/permission-handoff-design.json).
   Preserve the existing source restoration and edit-generation paths.
   First close the [reviewed router adoption gap](evidence/repository-proof-context-20261014/router-adoption-followup.json):
   incoming instruction `@5` replay precedes the new working-set `request_page`
   callback. Repeat the independent complete signed/profile/compiled-instruction
   replay after that request and before provider dispatch, then qualify this
   combined path with selected and omitted-source drift in both checkouts.
   The five proof-context suites do not qualify that router combination.
2. Add independently verified identity backends for committed Git snapshots
   and other proof families. Keep decoded logic proposals, checked local proofs
   and task acceptance distinct. Add multi-receipt composition only with explicit
   dependency closure, conflicts, residuals and complete mandatory conditions.
3. Reuse each IR family's existing datasets, embeddings and decoder checkpoints.
   CodebaseIR, IntentIR, SecurityIR, LegalIR and UI/UXIR retain separate family,
   schema/version, decoder-task and 8D/384D/768D inventories. A source generation,
   encoder revision and exact training asset lineage belong in each artifact's
   identity. Neither dimension nor low training loss establishes formal validity.
   The [preserved LegalIR review](evidence/repository-proof-context-20261014/preserved-legalir-review.json)
   records trained-parent reuse and a published conditions-head comparison whose
   fresh-panel exact IR score was zero. Its 384D inputs were placeholders with
   latent conditioning disabled. Keep that lineage separate; require actual
   encoder/embedding conditioning and fresh held-out semantic/text reconstruction
   before teacher promotion or distillation. Those receipts do not establish
   the complete family/dimension/schema/task registry or ModelManager imports.
4. Freeze matched Terminal Bench traces for baseline, indexed source retrieval,
   semantic capsules, reversible source views and proof-context treatments.
   Measure complete input/output usage across planning, coding, lookup, expansion,
   refresh, verification, retries and failures at matched official task success.
   Include the bridge's repeated source scans and cache checks in latency and
   resource accounting. Evaluate retrieval granularity and mandatory-condition
   size with the actual provider tokenizer before choosing defaults.

Exact retained-source restoration, exact semantic-IR serialization, learned
text reconstruction, checked proof status and total-session token cost are
separate metrics. A reversible minified codebase permits supported source round
trips; it still needs a matched end-to-end trial to establish token savings.
This bridge supplies no new learned reconstruction, model registration,
Hugging Face publication or complete-session savings result.
