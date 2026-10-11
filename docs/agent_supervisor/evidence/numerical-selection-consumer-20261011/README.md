# Learned numerical supervisor qualification

[The ledger](qualification.json) records 541 distinct passing tests across seven
successful suites: 554 test observations include 13 repeated driver controls.
All selected suites have zero failures, errors or skips. Each retains its own
command, tested source hashes and JUnit file; these are not 541 executions at a
single revision. The final merged host, archive and census suites ran 252 checks
on supervisor `9dd3d25945c3139152a195dc8ef10f668e4e8295` and datasets
`68374b51032b7a9fa795a79829e4d8eb2f2a5ca4`.

| Suite | Passing tests | Evidence scope |
| --- | ---: | --- |
| Datasets owner | 28 | Actual cached GTE/learned-head CPU inference, source inventory, mismatch, token deferral and replay |
| Datasets budgets | 2 | Extended deadline forwarding and pre-cancellation |
| Final merged host | 111 | Real checkpoint to planner, worker and historical audit; authored provider replies; merged formal-owner compatibility |
| Final archive | 101 | Tiny asset fixtures, actual source pins, archive bytes and Harbor selection |
| Final census driver | 40 | Authored driver seams using the Harbor interpreter |
| Formalization transport | 160 | Required evidence, native warm owners, audit/comparison and STOP retention |
| Driver guards | 112 | No implicit training, incompatible inputs, empty populations and preserved budgets |

The real inference tests use the already published development checkpoint;
they do not train it, download assets, repair mismatched predictions or replace
the learned result with deterministic expected IR. The full source-to-worker
test forbids repeated inference and rejects source/config/checkpoint/receipt
tampering before an authored provider seam. Source-aligned fragments still
discharge zero obligations and do not prove general numerical semantics.

The first datasets run revealed a Torch metadata/runtime build-suffix mismatch;
the owner now binds the actual version source without loading the model in the
parent. A first consumer assertion compared source hashes to full manifest rows;
the fixture now projects their hashes. Archive attempts exposed the inherited
producer-size bound and then refused concurrent source edits; the final archive
qualification used stable producers with the explicit 4-MiB source-file cap.
Collection failures used the wrong interpreter for Harbor; the final suites
separate native inference and Harbor environments. Failed and superseded local
runs remain identified by file digests in the ledger.

The [Docker compatibility probe](docker-02/stdout.log) used the existing preview
image with networking disabled, read-only mounts and no credentials. It verified
the release checkpoint and nine GTE files but found missing
`sentence-transformers`, `transformers` and `tokenizers`. It did not execute
inference. The first failed import probe is retained separately. No runtime
packages were installed, image rebuilt or live provider contacted. The temporary
containers were removed.

This evidence establishes local advisory transport and offline packaging
contracts. It supplies no new Terminal Bench score, token-efficiency comparison,
training result, theorem discharge or learned successor-refresh policy. The
previous Grok authentication failure was not reprobed.
