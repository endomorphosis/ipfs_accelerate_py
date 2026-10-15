# Audit the relations behind a capsule selection

`semantic_selection_relation_audit.audit_selected_capsule_relations` explains
selection omissions using the links committed by a native semantic-state root.
It adds a read-only diagnostic beside the current selector. It changes no
worker policy, producer, source, database owner or dispatch path.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_relation_audit import (
    audit_selected_capsule_relations,
)

report = audit_selected_capsule_relations(
    bundle=immutable_native_bundle,
    root_cid=expected_native_root_cid,
    selected_symbol_ids=selected_ids,
    max_symbols=1024,
    max_links=8192,
    max_bundle_bytes=32_000_000,
    max_report_bytes=2_000_000,
)
```

The bundle must be the exact native `SemanticStateBundle`, with immutable
blocks and the expected root. The audit checks every root index and record,
link source and target fact/version bindings, complete recorded node adjacency,
and capsule relation projections. It delegates relation projection to the
supported native capsule compiler helpers and compares effects and exceptions
with their merge from sealed symbol metadata. This is a consistency check on
recorded producer evidence; it does not prove complete extraction or current
repository bytes.

Capsule `dependency_link_ids` contain original **edge IDs**. The native link
index maps those IDs to **link wrapper CIDs**, which appear in node adjacency.
The report retains both identities. Reconstructing each original edge must
reproduce its edge ID, preserving relation, span, extraction method,
confidence and metadata.

For unresolved targets, reconstruction requires the exact
`metadata["unresolved_target"]` recorded by the native resolver. A valid native
bundle built directly from an unresolved edge can lack that metadata. This
diagnostic refuses such a bundle because target provenance is unavailable;
it does not infer a name or declare the native producer invalid. A resolved
link may also retain `unresolved_target` as provenance. Classification uses
`target_kind`, so that metadata alone never makes a resolved target unknown.
The wrapper's kind must agree with the reconstructed target's membership in
the complete root inventory, with the native compiler's symbol-first rule.
A wrapper cannot turn a known symbol or artifact into an unresolved reference.
Resolution accepts only native status labels or absence, and original target
text must satisfy the native bounded, printable, normalized string rules.
Structured provenance values refuse instead of entering the report.

## What a declared dependency closure leaves out

The existing closure helper traverses outgoing `dependency_stable_ids`.
The relation audit separately reports known call targets, other dependencies,
outgoing test and other relation targets, incoming omitted callers, resolved
artifact relations and unresolved state or other references. It also exposes
coverage-field counts, native analysis limitations and raw-source reason
counts. Counts and recorded relations do not establish effective test,
fixture, proof, schema, serialization or task coverage.

The authored Python fixture illustrates three different facts:

| Recorded fact | What selection can establish |
| --- | --- |
| `calls` to the known helper symbol | Adding the helper removes this known call-target omission. |
| `reads_state` of `global:helper_operation` | The global binding remains unresolved. A helper definition does not fix its runtime binding. |
| `tested_by` the public test symbol | This test capsule remains omitted until it is explicitly selected. It is outside the dependency-closure fields. |

The same fixture also records an incoming call from its test to the target.
The report exposes that omission without interpreting every incoming edge as
a mandatory task dependency.

Two finite CPython observations give a concrete reason to preserve the state
binding: `target_operation(4)` returns `10` with the original helper and `22`
after rebinding the helper to multiply by five. These observations concern
only the authored fixture. They are neither a universal source-equivalence
proof nor execution of the public acceptance command.

## Bounds and authority

The audit checks byte and structural bounds before native traversal. Empty
selections return `refused_empty_selection`; malformed identities, foreign or
duplicate selected IDs, unavailable target provenance and exceeded bounds
raise errors. An oversized report refuses instead of truncating evidence.
Output contains identities and typed relation rows, without source/AST or
capsule body fields. Exact logical target strings and source-span paths remain
present; arbitrary metadata is represented by a digest. It is not provider
input. This audit does not recompile unrelated capsule producer fields such
as signatures, defaults and decorators.

`declared_known_call_closure_selected` and
`selected_declared_relation_coverage_complete` describe only recorded links
at this root. `selection_eligibility` always remains experimental or refused.
Source freshness, complete dependency semantics, task acceptance, proof,
program equivalence, source substitution, deployment and runtime activation
remain unverified. A ledger of an old root still describes that old root after
source changes. Repeat existing source/current-root admission checks at the
actual use boundary; the probe confirms that existing transport refuses a
changed source.

The retained evidence is in
`artifacts/supervisor-relation-audit-20261012-01` in the enclosing workspace.
It uses native authored bundles and one private native preparation, with all
raw sources retained. It executes no provider, tokenizer, Terminal Bench task
or live owner workflow. All 75 final cases pass with mandatory AST seals and
current source guards. The earlier 59-case trial is retained separately,
before the two provenance/classification fixes. Model-token counts for the new
selection and whole-session savings remain unknown.

The current checked-in full indexed Harbor harness pins `codex_cli` and
`gpt-5.6-sol`. That is not a binding for an unspecified future campaign. The
scoped readiness observation found no verified compatible descriptor, and the
current counter's required `tokenizers` distribution is absent from `.venv`.
Installed `tiktoken` does not establish compatibility with that counter or
model. `tokenizer-readiness.json` retains the exact source and inspection
scope. Existing provider usage receipts can be replayed separately without a
local tokenizer; they do not supply token counts for these new prompts.

The supplemental `usage-replay/report.json` passes four original router
invocation records from two retained September 29 trials unchanged to the
existing cumulative reconciler. Its observed input-plus-output subtotals are
350,588 for the full arm and 266,001 for the no-index arm, with two distinct
sessions in each. Failed coding usage remains included; cached input and
reasoning output remain subsets. Both trials have acceptance reward zero.
Recorded task inputs and host source pins match, while runtime archives and
manifests differ. The legacy records lack strict final-usage lifecycle
evidence, so `provider_total_tokens` remains `None` and campaign completeness
remains unknown. These are historical observations, without a minification
comparison or savings claim.

This report makes the next selection experiment reviewable: known dependency
and test omissions can be addressed separately, while unresolved bindings and
other coverage gaps stay explicit. Live policy still needs current-source
admission, an owner-approved coverage contract, model-bound token measurement
and matched whole-session outcomes. See
[closure nomination](SEMANTIC_CAPSULE_CLOSURE_NOMINATION.md),
[selection comparison](SEMANTIC_SELECTION_COMPARISON.md) and
[usage reconciliation](TOKEN_USAGE_RECONCILIATION.md).
