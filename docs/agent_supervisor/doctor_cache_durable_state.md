# Durable Doctor cache refusals and historical discovery

`DoctorProofCacheGate` keeps proof receipts in the existing
`FormalVerificationCache`. Its additive `doctor_cache_control_*` tables in that
same database retain canonical Doctor keys, reverse root dependencies, root
revocations, per-key tombstones, quarantines and observed receipt identities.
The existing cache connection factory owns locking and transactions; this
adapter does not open an independent raw DuckDB handle.

The first opening migrates exact Doctor keys from older formal cache entries.
It derives no new proof authority during migration. Every lookup still
reconstructs the typed receipt, validates its complete key and checks current
denial state. A final state check follows receipt reconstruction or storage.
Repeated invalidation is idempotent, and revoking an unseen root also refuses a
later key registration. TTL cleanup never clears revocations or quarantines.
An unchanged mathematical receipt can remain stored for audit while the Doctor
gate refuses its use. An independent policy still decides which roots to revoke;
this change does not add a source watcher or infer semantic equivalence.

State is queried from the owner for every eligibility decision, so two existing
gate objects and a fresh process see the same refusals. Canonical key/root
mismatches and unsupported schema versions fail closed. All new control data is
negative eligibility state; it cannot supply a proof, admit execution or complete
a task. Diagnostics remain the pre-existing process-local channel.

This implementation qualifies the serialized native file owner. The current
Quack transport queues writes and reads a prior snapshot, so it cannot substitute
for the required read/modify/read transaction. The adapter explicitly refuses
that handle. Supporting it requires a typed atomic owner command and separate
qualification; the adapter never bypasses a live owner by opening its file.

`BoardControlPlane.ingest_proof_cache_page(namespace, path, after_key_id=...,
page_size=...)` reads at most 512 ordered rows and returns a continuation key,
row counts and explicit omissions. It indexes historical candidates, even when
their stored payload says `proved`. Artifact identities include the complete
source locator and key, preventing basename and truncated-key collisions.

The compatible `ingest_proof_cache_files()` integer-returning method now walks
pages up to `max_pages` per file (default 128). Its complete report is available
as `last_proof_cache_ingestion`, retained as a board artifact, and included in
the codebase-ingestion result. Missing files, unreadable tables, malformed rows
and page-budget exhaustion are observable. The continuation key allows an
explicit later page request.

These pages are discovery records, not a consistent snapshot across calls.
Every report states `snapshot_consistent=False` and denies proof/completion
authority. Concurrent inserts before a cursor require a fresh scan. The existing
automatic filename search still has its separate 16-file cap; callers needing a
complete file inventory must provide and qualify that inventory separately.
Current source observation and native receipt eligibility remain mandatory.

Regression coverage includes fresh-process restart, migration, late writes,
concurrent equivocation, invalidation during cache operations, schema/root
corruption, more than 512 discovered rows, resumable bounded scans, unavailable
stores and identifier collisions. This increment does not close the whole
repository proof-index backlog or qualify distributed source-to-proof reuse.

## Cross-package canonical identity bridge

`proof.canonical_cache_key_bridge.bridge_canonical_proof_cache_key()` requires
the exact datasets `CanonicalProofCacheKey` class and an explicitly supplied
native supervisor `ProofCacheKey`. Its closed, versioned obligation envelope
retains all sixteen datasets identity dimensions, their schema/interface and
optional source CID. A fixed field correspondence records their exact locations.
The original supervisor obligation is retained inside the envelope; every other
supervisor execution field remains unchanged. The inverse API requires both an
expected native semantic request and the expected execution context, replays the
complete representation, and uses native datasets cache admission to reject
cross-environment or other semantic identity drift.

This adapter supports identity retention and coordination only. It does not
infer that a source digest identifies a candidate tree, assumptions correspond
to premises, logical bounds equal resource budgets, or a checker name denotes
kernel evidence. Its envelope deliberately lacks a top-level supervisor
`obligation_id`; the existing `FormalVerificationCache` refuses positive receipt
admission through that key. A separate native obligation-linking adapter remains
necessary for positive cross-package proof reuse. No proof receipt, live handle,
assurance upgrade or task completion is created by this bridge.

Focused tests exercise every native identity dimension, every supervisor
execution dimension, missing fields, ignored metadata, source CID drift,
authority tampering, exact-type requirements and a real DuckDB cache that retains
an independently eligible receipt while refusing to promote the bridged key.
