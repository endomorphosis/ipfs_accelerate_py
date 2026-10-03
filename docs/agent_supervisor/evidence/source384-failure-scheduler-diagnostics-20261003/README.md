# Actual failure-only scheduler diagnostics

The applied A helper, context probe and full driver passed **39 controlled tests
in 0.84 seconds**, with zero skips and one deliberate deselection of the existing
actual-host-resource-sampler case. The modules loaded from the real checkout,
without bootstrap or import overrides. No native model or Docker run was needed.

The first actual run executed 24 tests and reused 15 existing tests through the
mandatory pytest AST seal. Its raw result is retained. The final run used the
supported fresh per-run seal database and executed all 39 selected tests. The
preceding off-tree 39-pass generation is also retained; these are the same 39
distinct controls, not additive tests. The only helper change after the off-tree
run clarifies the docstring to say no *direct* ledger reads; previous helper bytes
are included for exact generation correspondence.

Failure paths now append a bounded source-free scheduler observation alongside
the existing post-unwind resource sample. Only fixed finite capacity/allocation,
lease/waiter counts, proof backoff and proof recovery fields are exported. Unknown
reason/phase strings map to a fixed enum value. Paths, arbitrary labels, source,
lease capabilities and diagnostic exception messages are excluded. Collection
failures preserve the original task error, failure phase, result and cleanup.

The helper selects exactly one already-imported native facade through the existing
private registry and nonblocking process lock. It never creates or configures an
owner. Missing, busy, ambiguous or changed layouts yield unavailable diagnostics.
It then uses the supported snapshot API, which may recover stale leases normally.
The current D source shape is independently inspected in `native-shape-review.json`;
that is source review, not a live host-pressure test. The helper bounds output to
4096 bytes and retains the native snapshot's existing ledger-locking semantics.

The snapshot is labeled `after_unwind`: the failing call has unwound, but full task
shutdown may still be pending. Its backoff can outlive a request. It is neither
the failed request's exact admission sample nor causal proof of why that request
failed. Resource limits, deadlines, admission thresholds and configuration remain
unchanged. No benchmark score, inference completion or proof authority is claimed.

Command and source paths in the receipts describe the original development
machine. `final-sources/` contains exact current bytes; `before/` and
`previous-helper/` retain preceding producer generations. No state database,
benchmark input, model weights, authentication data or runtime archive is included.
