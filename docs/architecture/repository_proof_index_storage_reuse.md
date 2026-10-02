# Existing codebase evidence storage to reuse

This is a read-only inventory from 2026-10-02. The inspected original datasets
worktree is `/home/barberb/lift_coding/external/ipfs_datasets`; the released
integration is `/tmp/ir-release-datasets-20261001`. No catalog was ported and no
original tests were rerun for this inventory. The original worktree may contain
concurrent uncommitted work; the source hashes below identify the reviewed bytes.

Reuse these two existing domains before proposing another proof/evidence store.
They preserve historical conditional evidence, with current-source fences where
specified. They do not supply a live checker handle, kernel proof, source-runtime
equivalence, authoritative cache admission, task permission or completion.

## Owners and APIs

| Owner | Existing APIs | Persistence and limits | Caller obligations |
| --- | --- | --- | --- |
| `duckdb_control.codebase_evidence_index.CodebaseEvidenceIndex(catalog)` | `publish(receipt_cid, expected_head=...)`, `publish_many(receipt_cids, expected_head=...)`, `get`, `lookup(binding, expected_head=..., limit=32)`, `dependents(kind, value, expected_head=..., limit=32)` | `codebase_evidence` schema on the structural catalog's native connection; immutable CAS bodies; default 2,048 records, batches of 64, maximum 128 query results; overflow refuses | Require a fresh wrapper source observation and fresh checking before use. An exact durable head is not a live checkout observation. |
| `duckdb_control.codebase_verification_catalog.CodebaseVerificationCatalog(index)` | `publish(repository, expected_head=..., verification_cid=..., operation_id=..., applicability_cid=None)`, `lookup_current(repository, expected_head=..., path=..., contract_id=..., expected_contract_cid=None, expected_key_id=None, domain_id=None, expected_domain_cid=None, limit=16)` | Separate `codebase_verification_control` schema on the same native store/CAS; default 4,096 projections, 32,768 entries, 8,192 operations and 16 lookup results | Supply exact owner/source/head and bounded resource admission. Both calls observe current source on entry and successful exit. Lookup reconstructs historical artifacts; it does not establish checker authenticity. |

Both use the structural owner's serialized transaction API and fail on schema,
owner, process, CAS or complete-head mismatch. Neither opens a second competing
write connection. Both retain older generations and refuse to rebind them after
an ABA source restoration. Counts fail closed rather than pruning replay history.
Queries validate immutable bodies and exact SQL projections; a stored key or
status does not bypass reconstruction.

`CodebaseEvidenceIndex` has explicit reverse dependencies for `source`,
`snapshot`, `contract`, `compiled`, `profile`, `environment` and `compiler`.
Publication uses a real head-row write inside the transaction to conflict with
concurrent structural writers; cancellation rolls back the batch. Historical
lookup checks the sealed source/AST/compilation/receipt records without running
the compiler or solvers. The index accepts only terminal `proved`/`refuted`
conditional integer-offset observations, retaining all authority ceilings.

`CodebaseVerificationCatalog` retains native canonical keys and authored versus
lowered contract identities separately. Optional applicability records preserve
exact requested-domain identity and its query outcomes. The operation ID binds
the full publication request. Its historical loader recompiles native
ProgramIR/VC/SMT representations and recorded classifications without launching
solvers. A race after SQL publication can leave history, but final source
observation prevents delivery as a current result.

The structural `CodebaseCatalog`, `RepositoryCodebaseIndex`, `DuckDBASTStore` and
`DuckDBASTIngestor` matched the released source bytes during this review. These
owners require native file-backed DuckDB. This is not qualification of the
supervisor Quack snapshot transport or a typed atomic Quack command.

## Dependencies and compatibility

The narrow evidence index requires the original, currently unported
`software_contracts/codebase_property_cache.py` and
`software_contracts/codebase_integer_verification.py`. The verifier's explicit
implementation inventory also requires `codebase_integer_batch.py` and
`codebase_integer_workers.py`. `CodebaseIntegerVerifier.verify_many` already
coordinates bounded fresh workers with owner-thread-only SQL/CAS publication;
it is the existing integration path to reuse.

The richer verification catalog requires unported
`software_contracts/codebase_verification.py`. Requested-domain support also
requires `software_contracts/codebase_applicability.py` and
`software_verification/applicability.py`.

The original integer profile and finite observer import the original
`software_verification.pipeline`, `source_adapters` and `backends.process`.
The released integration deliberately uses additive `codebase_pipeline`,
`codebase_source_adapters` and `codebase_process` instead. The original richer
verification/applicability owners embed those original imports and producer pin
inventories. A future port must adapt imports and pins explicitly, preserve the
frozen modules, and qualify new records. Old receipt identities cannot silently
be relabeled as results of the released producer generation.

The historical key conventions are also distinct from the new
[finite owner-derived key correspondence](../agent_supervisor/finite_cache_correspondence.md).
Do not equate an operational `bounds` record with a finite semantic input domain,
or equate the original `SMT_CANDIDATE`/conditional solver scope with the new
declaration-only finite key. Reuse must preserve each complete key and profile.

## Remaining acceptance gates

| Backlog criterion | Reusable portion | Still missing from these owners |
| --- | --- | --- |
| RPI-003 | Separate native state domains, CAS, exact owner/head bindings, restart reconstruction, uniqueness, fences and bounded reads | Explicit versioned schema migration and a reviewed cross-domain intent/model/root correspondence |
| RPI-007 | Complete historical keys/artifacts, exact lookup, reverse dependencies and corruption/refusal controls | Positive checked-proof admission, explicit negative/tombstone/quarantine status domain, proof/CodeEvidencePlane projection and its trust-aware service integration |
| RPI-017 | Existing operation history is useful input | Actual DuckLake journal/outbox delivery, response-loss recovery across transport, snapshot retention/lag/GC and crash-window qualification |
| RPI-032 | Exact current-head historical selectors and complete-record reconstruction | Root-bound cursor pagination/frontiers, consumed-record commitments in the required service path, model dependency kind, revocation joins, intent/residual queries and freeze-to-admission qualification |

Neither owner alone closes any of those complete production criteria. Reusing
them avoids duplicate persistence while retaining the need for the independent
native checker and admission paths.

## Existing qualification sources

The original `tests/integration/logic/software_contracts/test_codebase_evidence_index.py`
contains native persistence/restart, exact key and reverse-dependency, forged
body, stale generation/ABA, rollback, capacity, bounded artifact read, concurrent
writer-conflict and inherited-process refusal controls.

The original `tests/unit/duckdb_control/test_codebase_verification_catalog.py`
contains exact native key/domain selectors, restart, operation replay/conflict,
CAS and SQL corruption, caps, cancellation and source changes after publication
or historical reconstruction. Related modules test verification,
applicability, property-cache and integer-batch behavior.

`docs/software_contracts/CODEBASE_IR_FOUNDATION.md` links the original retained
`workspace/codebase-batch-qualification-20261002` and
`workspace/codebase-conditional-verification-20261002` reports. Their reported
outcomes are historical evidence for their own producer bytes; this inventory
does not import their counts into the released acceptance ledger.

## Reviewed source identities

Paths below are relative to the original datasets worktree named above. SHA-256
identifies file bytes, not author authenticity or an approval to replace a
released owner.

| Path | SHA-256 |
| --- | --- |
| `ipfs_datasets_py/duckdb_control/codebase_verification_catalog.py` | `1a8041f0bee49eaa788d37c80e757ffd4abf4a1b9242e3159bb9c878094acf71` |
| `ipfs_datasets_py/duckdb_control/codebase_evidence_index.py` | `68b83b5921e2577d260c1f9e97ed12fe815135c180ec68247c571e41dac79ebd` |
| `ipfs_datasets_py/duckdb_control/codebase_catalog.py` | `e7c85582d965a1e72eb97795e711b77856daf3767cde02f4152127a1fa876140` |
| `ipfs_datasets_py/logic/software_contracts/codebase_ir.py` | `0fc2945f5c42eb2cdf068fab23885560dd8add30c3d4e751f3862f02fce72ba4` |
| `ipfs_datasets_py/logic/software_contracts/codebase_property_cache.py` | `d8d3614e435607de9ce1bb7b910af150dfbf6359dcebd3778593e411c3f6cbf1` |
| `ipfs_datasets_py/logic/software_contracts/codebase_integer_verification.py` | `6308995c77628f069385b0c3cac2a3a19e07488f7e90b4963eb39dce2e52d919` |
| `ipfs_datasets_py/logic/software_contracts/codebase_integer_batch.py` | `b63468aa1f123beee3ac0e44368b1ae2474ebefdf51d6711d3ed84e5b71deccd` |
| `ipfs_datasets_py/logic/software_contracts/codebase_integer_workers.py` | `1f5e1ff5ae6aaaa09f85db314bac1fa1fe04395ea978aadada1f66a03ec9d6cd` |
| `ipfs_datasets_py/logic/software_contracts/codebase_integer_profile.py` | `9baf67451294b29b7c9e205477293246b1dde46f6d4bebe627a4e3145832acf3` |
| `ipfs_datasets_py/logic/software_contracts/codebase_finite_integer_observation.py` | `5613f32adc9212dab0a68c2f1ffe161255f4b27fae3a948f2d01bbd32229778b` |
| `ipfs_datasets_py/logic/software_contracts/codebase_verification.py` | `9d735f56c3624dfaf9bd3ffbabc107ea6118057bfc85d8787b13077382cc6eb3` |
| `ipfs_datasets_py/logic/software_contracts/codebase_applicability.py` | `f930724d5e7e939605a83e8e453c28827595890bbff5ba191bfe68bae775c080` |
| `ipfs_datasets_py/logic/software_verification/applicability.py` | `64d2b9b784775aa0ec6eaa7352102ffb171570e22da9e798602af89bb344b884` |
| `ipfs_datasets_py/logic/software_verification/pipeline.py` | `97048ce560b367b4651def82dcb345d3ee34edc60373c15dbaba551ca459e9da` |
| `ipfs_datasets_py/logic/software_verification/source_adapters.py` | `f903d036b38199f1cfd3ca72b065d2fec70e23ed6065795b11423e991d6f2fa2` |
| `ipfs_datasets_py/logic/backends/process.py` | `5c569b34afd75e1942f11fcb9ecf20d57799d2762927d9379813ce94aca437e1` |
| `tests/integration/logic/software_contracts/test_codebase_evidence_index.py` | `9dcc88d360765480c34103cca159395ac26331c3dbfc9ddfaa42e60729926609` |
| `tests/unit/duckdb_control/test_codebase_verification_catalog.py` | `d6b23b61716c6a197e0a217fbcf6c3565500e6bba0f9418656886c952e4f039b` |
| `docs/software_contracts/CODEBASE_IR_FOUNDATION.md` | `2817363fb0a38494d8d8bec39a5be7f7d8d8be95427d2d7dde16e5cabcde49e0` |
