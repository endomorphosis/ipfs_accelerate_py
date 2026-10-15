# Compact semantic context for Terminal Bench

Status: implementation design from the canonical workspace audit of 2026-10-12. This document describes the next change; it records no new provider run, token savings, source-substitution admission, or task success.

The practical first step is a wider reversible representation of repeated typed identifiers. Keep the complete verified source, capsules and task context restorable. A later role projection can defer archival bookkeeping through the existing context-reference and expansion interfaces once the required coverage and original-artifact resolver are bound explicitly.

## Existing work to reuse

| Existing seam | Behavior to preserve |
| --- | --- |
| [`intent_world_snapshot.minify_intent_world_worker_context`](../../ipfs_accelerate_py/agent_supervisor/semantic_state/intent_world_snapshot.py) | Removes internal signed planning-contract blobs after verified loading, retains their identities and exact task specifications, and labels the result as a worker projection. The daemon already uses it. |
| [`code_retrieval_context.load_code_retrieval_context`](../../ipfs_accelerate_py/agent_supervisor/runtime/code_retrieval_context.py) | Replays native retrieval, checks the complete source/index binding, projects at most 20 hits, and returns explicit stale status with no hits when sources changed. Full vector indexes remain outside the worker packet. |
| [`semantic_context_runtime`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py) and [`select_worker_capsules`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_capsule_selection.py) | Retain the complete producer root and source blocks, select a bounded capsule set, and preserve required raw inputs, omission counts and raw-fetch obligations. Refresh remains confined to the original nominated scope. |
| [`ContextCompiler`](../../ipfs_accelerate_py/agent_supervisor/context/context_compiler.py) | Supports required `ContextReference` values, explicit deferred references, bounded expansion, and parent-bound retry reconstruction. Its required coverage must survive any projection. |
| [`semantic_router_translation`](../../ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py) | Authenticates producer capsules and source inventory, aliases typed identifiers, and restores the entire native prompt exactly. Literal source, names, signatures, comments, instructions and authority remain unchanged. |
| [`router_implementation_runner.select_model_prompt`](../../ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py) | Compares complete rendered candidates, including suffixes and workspace instructions. Selection requires a working provider/model-bound counter and strictly fewer tokens; unknown counts retain the native representation. |

Do not introduce a second context compiler, semantic index, capsule producer or storage owner. Metadata can nominate evidence; it does not establish source freshness or permission to substitute it.

## First patch: extend closed identifier slots

The current codec's `_slots` covers capsule top-level identifiers, admission reference identifiers, and selected top-level fields of pack references. Several repeated identifiers remain literal in other typed locations. Extend the existing codec using explicit field sets rather than recursively replacing any string that resembles a CID.

Candidates for an audited slot policy are:

- Top-level source-scope and semantic-root identifiers, and declared Doctor evidence identifiers.
- `manifest[path].source_cid`.
- `pack.pack_cid`, `pack.production_slice_cid`, and the inner `pack.pack` target/surrounding/test/delta identifiers and dependency/obligation/counterexample/interface identifier lists.
- Known typed reference metadata such as `capsule_cid`.
- Worker-projection capsule-index identifiers, selected-symbol stable identifiers, and raw-fetch source identifiers.
- Declared reconstruction or refresh root/scope identifiers where their exact schema provides a closed identifier slot.

Validate the exact supported record shape before considering its fields. Unknown schemas, arbitrary capsule metadata, source strings, path strings, prose summaries and AST literals stay literal. Preserve all current structure and size bounds. Do not shorten, strip or rename executable code.

The retained authored toy probe at `artifacts/supervisor-token-context-audit-20261011/probe/attempt-r3` in the enclosing workspace contains 17 missed closed-slot occurrences; 11 can reuse aliases already present in its retained table. A read-only standard-library diagnostic changed only those 11 occurrences and reduced the translated semantic body from 13,710 to 13,269 bytes: **441 bytes**, approximately 3.2%. This excludes full-wire, reference-template, instruction and revision overhead. No tokenizer, producer admission or current source freshness was revalidated. It is a historical byte opportunity, not token savings or a new qualified encoding.

### Codec revision and replay

Changing the slot policy changes the deterministic table and translation identity. `restore_semantic_router_prompt` currently recomputes the encoder and compares the result with the supplied table, so silently expanding the default policy would reject retained tables.

Bind an explicit slot-policy revision to new encodings and dispatch restoration through that revision. Keep the legacy policy available for existing tables without interpreting missing revision fields as the new policy. Use the existing codec module and a versioned contract extension; do not reinterpret old table identities or silently regenerate them. An unsupported revision must refuse. Historical reconstruction remains historical and cannot become operational response-decoding authority.

The new encoder must prove exact restoration before returning its candidate. The runner then compares the complete rendered alternatives using its supplied counter. Local fixture counts are diagnostic unless the coding-model/tokenizer binding has been established. Temperature, context limits and native acceptance checks remain unchanged.

## Later patch: an explicit role projection

The daemon currently wraps the full semantic artifact in `required=True` chunks. The codec verifies that their joined text equals the nominated artifact bytes. Dropping fields or marking the complete chunk optional would violate those contracts. A compact role view must therefore be a declared representation of the verified artifact, not a shortened object masquerading as `supervisor-semantic-worker-context@1`.

### Required context

Keep these effective decision inputs intact:

| Context | Required material |
| --- | --- |
| Task | Literal public instruction, objective, scope, allowed outputs, prohibitions, acceptance and validation commands, native policy/core and required suffixes. |
| Sources | Complete required raw inputs and their manifest bindings; required source paths and any raw-fetch obligation before relying on or editing omitted source. |
| Capsules and dependencies | Every selected complete native capsule, selected dependency identities, all freshness/admission decisions, confidence and conservative caveats, raw-required flags, and relevant invalidation obligations. |
| Coverage | Required references and coverage identifiers, pack assumptions/exclusions, obligations, counterexamples, interface requirements, coverage status, risk and route decisions. |
| Currentness | Current source/root/scope bindings; world component current/stale/unavailable status, event cursor, plan roots and exact relevant task specifications; retrieval current/stale status and nominated hits required for the present decision. |
| Omission | Selected/deferred/unavailable distinctions, omitted counts, raw-fetch requirements and bounded resolution handles. |

For the first role-view implementation, do not trim capsule metadata or reinterpret its embedded AST/version evidence as optional. Do not demote task or dependency coverage because it is expensive. If the required set exceeds the existing budget, retain the existing refusal or an already authorized task split.

Potential archival bookkeeping includes reconstruction audit records, refresh-history bookkeeping and estimator-category telemetry. Classify specific fields against the admitted decision rather than blanket-dropping provenance. Preserve any currentness, limitation or raw-source caveat in the model view. Keep the complete original artifact unchanged.

### Original-artifact and resolver contract

Construct a role view only after the existing dispatch loader has successfully verified the original task, digest, nominated scope and current sources. Bind the view to the exact original artifact path/digest, semantic root, task, source scope, projection revision and required-coverage identity. Report the view as a derived role projection with no execution or completion authority.

Use the existing `ContextReference` and frontier/expansion surfaces to identify deferred content. A deferred handle must resolve through an explicitly supplied owner/resolver to the correct original field or record, under the current task/tree/root and declared size/token limits. Recheck the original binding and freshness at expansion; reject missing, changed, foreign or stale parents. An ambient filesystem path, metadata row or digest match alone cannot supply source-substitution authority.

Resolve required content before the dependent decision or mutation. For each expansion and retry, preserve required coverage and parent linkage. Keep complete originals and projection receipts separately; never overwrite a producer capsule, source block or retained worker artifact. Local projection and expansion counts are attribution evidence, not additional token charges on top of native cumulative session usage.

The codec needs an explicit adapter for the role view: restore the declared representation and original binding before its existing artifact/source verification. Do not relax `raw == text` for arbitrary abbreviated inputs. The refresh/cache path must always reload current source evidence, and world-versus-semantic coherence checks must still use the same dispatch observation.

## Staged validation

1. Extend `test/api/test_semantic_router_translation.py` with real pinned producer fixtures exercising new typed slots, unchanged literals, exact complete prompt restoration, deterministic policy identity, legacy-table replay, unsupported revision and tampered replacement paths/table refusal. Retain current source-inventory, capsule-integrity and current-versus-historical checks.
2. Extend `test_router_prompt_selection.py` and local-tokenizer runner tests to compare the complete new wire with the native candidate, including suffixes and workspace advisory. An unknown, failed or mismatched counter cannot enable compression. Test a candidate whose additional alias/instruction overhead makes it larger and preserves native selection.
3. Before a role view is integrated, use `test/api/semantic_state/test_semantic_context_runtime.py`, `test_semantic_context_retry.py`, `test_semantic_worker_projection.py`, `test_supervised_task_context.py` and `test_code_retrieval_context.py` to prove unchanged required source/task/dependency/coverage inputs and retained originals. Test stale source, refresh scope, world coherence, explicit omission and requiredness downgrade refusal.
4. Exercise the existing reference/expansion compiler with missing or corrupt originals, foreign task/root, stale parents, cancelled or oversized expansion, and retry reconstruction. Deferred archival fields must remain resolvable without granting admission or eliminating currentness checks.
5. Run an isolated paired fixture measurement on identical source/task bindings. Record full-message bytes, bound-counter provenance/counts, expanded/retried inputs and wall time. Historical bytes and authored-tokenizer fixture results stay explicitly labeled. A real Terminal Bench comparison needs the production owner/model binding and complete usage over the same assigned task population, including failures and fresh retry/expansion sessions.

Use normal conftests, mandatory source seals and fresh evidence catalogs. No provider, live database owner or model run is needed for these contract tests. Only actual `lake build <Lib>` grants a Lean admission; representation restoration, metadata rows and passing tests do not. Preserve the canonical datasets import tree, download no weights/assets, and keep the Constitution unformalized.
