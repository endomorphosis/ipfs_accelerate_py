# Reversible source-view qualification — 2026-10-12

The [codec and repository guide](../../reversible_source_views.md) describes
read-only compact candidates with complete KEEP/DELETE byte maps. Original
bytes remain in the existing retained source inventory; sidecars copy no deleted
text. Cold replay joins every view to the complete source population, rebuilds
the exact profile and verifies unchanged-source restoration.

This is historical byte integrity, without mapped edits, source execution,
filesystem restoration, current-source admission, proof or coding-model access.
Raw candidate bytes exclude provider guidance/wrappers/tools and establish no
token savings. Existing model weights and embeddings remain unchanged.

## Test evidence

The final source passed **40 distinct cases**: 21 codec cases and 19 repository
inventory cases, with no failures or skips. Both use normal AST sealing and
isolated temporary state against immutable datasets
`c18390b350c7ab40f5bd342578204a03dc34fe05`.

- [Final command/environment/hash receipt](final-02-commands.json)
- [Exact node population](final-02-nodeids.json)
- [Final test log](final-02-tests.log) and [JUnit](final-02-junit.xml)
- [Final collection log](final-02-collection.log)
- [Qualification](qualification.json) and [source manifest](source-manifest.json)

Production and test bytes match before/after the final run. Controls cover
Unicode byte offsets, CRLF and literal/docstring preservation, protected comments,
default comment retention, f-string/encoding/token/piece identity fallback,
edited views and forged maps, full committed population and Gitlink frontier,
opt-in profiles using the same source assets, missing/corrupt source/view/sidecars,
codec/policy/input/store drift and root-publication poisoning. A separate process
restores original and view bytes after the authored original repository is
deleted, without source reacquisition or creating another owner after admission.

The [initial run](initial-commands.json) passed 39 cases and failed one malformed
fixture's reason assertion. An unterminated parenthesis failed tokenization
before AST parsing, so its correct fallback reason was `unsupported_tokens`.
The fixture was changed to tokenizable invalid syntax to exercise `invalid_python`.
The [initial log](initial-tests.log) and [JUnit](initial-junit.xml) are retained.
Production code did not change. Final [node IDs](final-02-nodeids.json) match the
initial population exactly; repeat cases are not added to the distinct count.

## Scope

Default Python views preserve comments; ordinary-comment removal is explicit.
Valid f-string sources remain identity views. Source/parser binding is strict,
with source-file hashes and reported runtime identities; it is not a complete
interpreter/toolchain attestation. Structural checks grant no semantic relevance
or executable-equivalence authority.

The same caller-owned CAS stores raw changed views and bounded structured
sidecars/pages. Unchanged views reference original source frames, including
large unsupported files; there is no second content/database owner. Partial
immutable artifacts can remain on failure. Checks are sequential, without an
atomic lease, rollback or permanent availability guarantee. Revalidation and
metadata pages remain historical, with no new Quack operation or model tool.

Local implementation/source hash checks and test evidence are separate from
worker archive/deployment and complete Terminal Bench qualification. No provider,
model, training or prover call was performed by this test phase.
