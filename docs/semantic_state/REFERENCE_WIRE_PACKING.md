# Lossless reference columns in semantic router inputs

Reference packing reduces repeated descriptor field names and shared values
inside an already verified V2 semantic input. It preserves complete reference
rows and native prompt reconstruction. It does not change source code, select
less evidence, create a new database owner or certify task success.

## Entry points and launch behavior

Use the existing `semantic_slot_policy_revision="closed-typed-identifiers@2"`
or `--semantic-slot-policy-revision closed-typed-identifiers@2` option. The
standard V2 router comparison offers `packed_v2` when eligible references can
be packed, alongside native, legacy and unpacked V2. With semantic role
projection enabled, it also offers `packed_projected_v2` when eligible.
Default legacy launches retain their existing wire and table identities.

Complete rendered messages include workspace instructions, packing guidance
and literal suffixes. The supplied compatible counter must show strictly fewer
tokens than native for any alternative to win. Equal candidate counts prefer
legacy, V2, projected V2, packed V2, then packed projected V2. An unavailable,
failed or inconsistent counter retains native; strict ceilings still require
valid counts. Every logical counter call remains visible to consistency checks.

The pure codec entry point is:

```python
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import (
    encode_semantic_router_prompt,
    pack_semantic_router_prompt,
)

verified = encode_semantic_router_prompt(
    prompt=native_prompt,
    repository=canonical_repository,
    slot_policy_revision="closed-typed-identifiers@2",
)
candidate = pack_semantic_router_prompt(encoded=verified)
```

Packing derives from the verified encoding without another producer pass. The
original table and translation CID remain unchanged. The pure step inherits
the input's recorded freshness and grants no new freshness. Operational
restoration and structured replies freshly verify the native producer, then
require exact deterministic re-packing. Historical replay stays historical.

## Versioned format and coverage

The packed transport declares `supervisor-semantic-router-input@2` and
`reference_packing="context-reference-columns@1"`. Existing unpacked transports
remain `supervisor-semantic-router-input@1`; existing table schemas and identifier
slot policies are not reinterpreted.

Only these declared reference lists can be represented as columns:

- `translated_semantic.pack.references` in a native worker context.
- `translated_semantic.required_context.pack.references` in a role view.

The descriptor schema is `supervisor-context-reference-columns@1`. Its closed
fields are `schema`, `shared`, `columns` and `rows`. Shared fields apply to every
row; each row's cells correspond to the ordered column names. Their disjoint
union contains all 13 original `ContextReference` fields. Columns are sorted
and unique; row count, width and order are exact. Shared values are compared
using canonical JSON bytes, so values such as `False` and `0` stay distinct.

Summaries, paths, source literals, required flags, coverage IDs, metadata and
alias objects retain their exact values. Objects which look like descriptors
inside summaries, arbitrary metadata or unrelated fields stay literal. Empty,
singleton, mixed and unknown-field lists remain unchanged. Only a descriptor
whose canonical bytes shrink can become a packing candidate.

Unpacking precedes the existing translated-semantic SHA check and identifier
restoration. Native reconstruction must reproduce the original prompt SHA.
Current restoration and structured reply decoding also compare the freshly
verified table and entire deterministically packed wire. The native candidate
grammar, source freshness checks, admission rules and qualification gates stay
in force.

## Bounds, telemetry and opportunity cost

Input and virtual expanded payloads retain the existing two-megabyte,
100,000-node and depth-40 bounds, with at most 4,096 reference rows. Expansion
counts repeated shared values before constructing rows. Pending traversal work
is bounded before allocation. A malformed declaration or invalid original
input refuses. If only the optional wrapper exceeds the envelope, the already
verified unpacked candidate remains available; no limit is raised.

Encoding receipts record `reference_packing`, `reference_packing_profile`,
the original provider-wire hash/bytes and the packed provider-wire hash/bytes.
The invocation receipt's `semantic_reference_packing_selected` identifies the
packing actually sent. `prompt_selection.comparison_candidates` and
`selected_candidate` retain complete-message measurements and the winning
variant. Projection selection includes packed projected V2 when appropriate.
Neither a constructed descriptor nor a smaller component proves token savings.

Packing adds canonical serialization, bounded expansion and restoration work;
comparing additional alternatives also adds counter calls. The offline probe
records complete-message bytes and one pure helper timing separately. Provider
tokens, model readability, retry costs and complete-task savings require a
matched production run with the same acceptance checks and whole-session
accounting.

See the [launch guide](COMPACT_CONTEXT_LAUNCH.md),
[whole-session accounting](TOKEN_USAGE_RECONCILIATION.md) and
[packing evidence](../../../../artifacts/supervisor-reference-packing-20261012/README.md).
Only an actual `lake build <Lib>` grants a Lean admit; transport restoration and
stored evidence do not. The Constitution remains unformalized.
