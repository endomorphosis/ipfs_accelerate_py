# Bounded parent pages, selected rows and producer-chain replay

Three deterministic qualifications preserve complete supplied populations and original evidence; full-corpus and legal-semantic acceptance remain partial.

## LAC-S9-011 Qualify bounded parent Parquet page preflight

- Status: todo
- Completion: auto
- Is schedulable: true
- Priority: P0
- Goal id: LACB-G011
- Depends on:
- Track: parent-pages
- Provider role: deterministic-only
- Context budget tokens: 0
- Objective: Verify the independently pinned full parent file, bounded footer/schema and page-header populations before any value decode. Retain exact physical layout and supported/refused encoding observations under finite page, row and uncompressed-byte profiles.
- Outputs: data/legal_autoformalization_parent_pages/LAC-S9-011/result.json
- Predicted files: data/legal_autoformalization_parent_pages/LAC-S9-011/result.json
- Validation: python3 scripts/legal_autoformalization_parent_pages.py materialize --task LAC-S9-011
- Acceptance: The full supplied parent descriptor and page census independently replay; unsupported profiles remain explicit and cannot authorize a selected-row read.
- Allowed effects: Only the declared result, owned temporary validation files and native operational records; original inputs remain read-only.
- Board namespace: legal-autoformalization-parent-pages-20261014
- Bundle: lac/parent-pages/lac-s9-011
- Parallel lane: lac-s9-011
- Conflict policy: Disjoint results and native serialized merges.

## LAC-S9-024 Qualify exact selected physical parent rows

- Status: completed
- Completion: auto
- Is schedulable: true
- Priority: P0
- Goal id: LACB-G024
- Depends on:
- Track: parent-pages
- Provider role: deterministic-only
- Context budget tokens: 0
- Objective: Read each independently selected physical parent row through the qualified bounded page decoder. Retain complete original row fields, physical ordinals, per-column/page evidence and independently pinned row hashes. Refuse incomplete or unsupported representations without substituting inferred values.
- Outputs: data/legal_autoformalization_parent_pages/LAC-S9-024/result.json
- Predicted files: data/legal_autoformalization_parent_pages/LAC-S9-024/result.json
- Validation: python3 scripts/legal_autoformalization_parent_pages.py materialize --task LAC-S9-024
- Acceptance: The complete selected physical-row population independently replays from pinned original bytes; source correspondence and legal authority remain unresolved.
- Allowed effects: Only the declared result, owned temporary validation files and native operational records; original inputs remain read-only.
- Board namespace: legal-autoformalization-parent-pages-20261014
- Bundle: lac/parent-pages/lac-s9-024
- Parallel lane: lac-s9-024
- Conflict policy: Disjoint results and native serialized merges.

## LAC-S9-044 Qualify complete producer chains with page evidence

- Status: todo
- Completion: auto
- Is schedulable: true
- Priority: P0
- Goal id: LACB-G044
- Depends on:
- Track: parent-pages
- Provider role: deterministic-only
- Context budget tokens: 0
- Objective: Regenerate all twenty physical cache records and complete producer provenance, then reconcile independently pinned producer witnesses with complete qualified physical parent rows and original page custody. Preserve every unmatched or unsupported row and all prior evidence.
- Outputs: data/legal_autoformalization_parent_pages/LAC-S9-044/result.json
- Predicted files: data/legal_autoformalization_parent_pages/LAC-S9-044/result.json
- Validation: python3 scripts/legal_autoformalization_parent_pages.py materialize --task LAC-S9-044
- Acceptance: All twenty original records independently replay with the exact page-qualified parent population. Local chain evidence never establishes model invocation, original character occurrence, source edition correspondence or legal meaning.
- Allowed effects: Only the declared result, owned temporary validation files and native operational records; original inputs remain read-only.
- Board namespace: legal-autoformalization-parent-pages-20261014
- Bundle: lac/parent-pages/lac-s9-044
- Parallel lane: lac-s9-044
- Conflict policy: Disjoint results and native serialized merges.
