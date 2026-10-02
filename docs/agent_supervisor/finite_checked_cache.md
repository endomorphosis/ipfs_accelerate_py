# Durable finite table evidence

`proof.finite_checked_cache.FiniteCheckedCache` adds a finite evidence domain to
the existing `FormalVerificationCache` local DuckDB owner and datasets
`ImmutableCAS`. It accepts the same owner inputs as
[finite cache correspondence](finite_cache_correspondence.md): the current
native repository index/head, an explicit integer-offset contract, a sorted
finite input list, and pinned native Python/Lean tools. It does not accept a
caller-supplied checker receipt.

`check_and_store(owner_inputs=...)` invokes the native finite observer, validates
its complete artifact inventory, derives a table-specific `CodeProofObligation`,
and routes positive receipts through `FormalVerificationCache.put`. Its obligation
is the **exact generated finite-table Lean theorem bundle**. The corresponding
Python observations are retained evidence; Lean does not prove their origin or
universal CPython semantics. Source-runtime equivalence, worker admission,
mutation permission and task completion remain unproved and unauthorized.

`lookup(owner_inputs=...)` rederives all 16 canonical dimensions from current
source and native owners. It reconstructs every stored artifact body from CAS,
validates the exact SQL projection and reverse dependencies, derives the expected
obligation/receipt again, and calls the existing trust-aware
`FormalVerificationCache.lookup`. A positive result additionally requires a
**new native Python and Lean run** with the same table, source, domain and tool
bindings. A serialized result is historical evidence; copying its positive flag
does not create a live capability. This profile makes no claim of saving checker
calls or improving throughput.

The original identity-only key bridge remains unchanged and proof-ineligible.
The new execution key carries the entire original correspondence plus the exact
table statement, premise identities, artifact identities, and additional direct
execution/storage producer pins. No assurance is inferred from a hash, schema
marker, language-model prediction or successful syntax-only build.

## Durable boundaries

The companion tables retain immutable record CIDs, explicit dispositions,
complete key/receipt payloads and reverse references for every canonical
dimension, source, snapshot, contract, premise, model-off declaration and pinned
producer. All eleven native positive artifact bodies are persisted in CAS,
including source, compilation, process receipts, observations, Lean source,
compiled `.olean` and certificate. Temporary checking directories can disappear
without making reconstruction depend on an in-memory dictionary.

`positive`, `refuted`, `python_failed` and `lean_failed` remain distinct.
Unsupported source refuses during the preceding supported-source key preparation.
Nonpositive entries contain no authoritative proof-cache receipt. They are
historical diagnostics and cannot discharge a task. A subsequent differing
checker disposition refuses reuse; this owner does not silently overwrite
history or refresh expired proof entries.

The schema, columns and constraints are checked independently of stored schema
hashes. Exact lookups admit one row; duplicate records, incomplete dependencies,
unknown fields, changed producer/tool/source identities, missing bodies and
content corruption refuse. Default capacity is 256 records, maximum 4,096;
record and combined artifact bodies are bounded to 4 MiB. Reverse lookup returns
at most 64 explicit historical references and refuses overflow instead of
truncating it. Each reference must subsequently pass exact lookup before use.

Publication uses the existing native single-flight lease to serialize writers
for one complete request. Shared outcomes carry only an immutable record CID.
Each caller retains its own fresh check and reconstructs the stored result after
coordination. CAS, typed proof-cache insertion and companion index publication
are separate durable steps. Interruption can leave orphaned immutable artifacts
or a valid table receipt; the companion returns no hit without the complete
index record and dependencies. The native coordinator retains a failed outcome
for its bounded TTL; a fresh retry after expiry may complete publication.

This qualification is for the direct local serialized DuckDB owner. It does not
qualify Quack snapshot transactions, cross-machine replication, DuckLake
outboxes, generic proof-discovery pagination, revocation-service joins, or all
IR/logic families. It preserves all existing frozen source/compiler/cache owners.

## Qualification

The focused tests use actual Python and Lean, including complete artifact
omission controls, rehashed statement tampering, explicit real process failures,
fresh-process DuckDB/CAS reconstruction, concurrent publication, interrupted
publication recovery and existing typed-cache regressions. The retained
[qualification evidence](evidence/finite-checked-cache-20261002/README.md)
contains the exact test results, source pins and standalone reproducer.
