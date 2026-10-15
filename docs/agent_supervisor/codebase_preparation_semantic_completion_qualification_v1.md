# Resumable preparation semantic completion and native publication qualification v1

Date: 2026-10-12. A complete admitted preparation prefix can now assemble the entire structural semantic inventory and publish one atomic native DuckDB catalog operation. The default route reuses staged Python frontend records and has an exact-boolean opt-out. In the actual authored qualification, it eliminated **33 redundant frontend extractions** during completion while matching the independent uninterrupted graph, projections, native tables, head and operation receipt. Python semantic and pytest syntax parsing still runs.

## Implemented behavior

`RepositoryCodebasePreparation.complete_current(repository, root_cid, cursor, operation_id=..., admitted_page_cid=..., optimized_frontend_reuse=True, parent_lease=...)` requires the entire contiguous admitted inventory, its current qualified implementation context, and the exact matching catalog, AST-store and CAS owners. It intersects the original preparation limits with catalog entry/source/AST/manifest limits and the actual scheduler reservation before publication. The CAS remains outside the source repository.

The scanner accepts complete source/context-bound prepared Python records. It detaches the canonical records and context maps before caller checkpoints, verifies every model-safe Python source binding, and resolves the graph over the full inventory. Python semantic analysis retains its syntax parse; pytest analysis retains its own syntax parse. Pytest identities, fixture visibility, configuration edges and cross-file call resolution are assembled globally after all pages are present. Page graphs are never concatenated as though they were independent.

Reuse defaults on. `optimized_frontend_reuse=False` performs the ordinary frontend extraction route, and the actual controls compare the resulting full manifest, semantic state and SQL rows. This option is separate from the ingestor's existing parser-shard cache and scanner prior-state reuse policies. Completion supplies no prior semantic state. Custom staged frontend options that differ from the scanner's default profile refuse default reuse; explicit opt-out still validates admitted staging context and computes fresh semantics.

Completion seals every bare AST body and the complete structural manifest before one `CodebaseCatalog.publish` call. New operations require full source-membership/byte, implementation-context and expected-head observations immediately before publication. The catalog's native transaction performs the complete AST batch, head change, supersession and durable operation receipt together. An incomplete cursor cannot publish a normal head. Failure can leave immutable orphan CAS objects, but the head/receipt transaction rolls back.

Exact operation retries deterministically reassemble and authenticate their request, then resolve the existing catalog operation before requiring the former live-head observation. They perform no new AST/head/operation writes. A returned receipt can be historical after source edits or a successor head; it does not establish present currentness. This milestone does not add a fast retry companion or skip semantic reassembly.

Cancellation before commit is cooperative and rolls back the transaction. Cancellation observed after a successful catalog return preserves the committed receipt. A resource cleanup error after commit raises `CodebasePreparationCommittedError`, carrying the durable `.receipt`, so callers can distinguish it from precommit failure. Completion does not manufacture the ingestor's process-local publication history or latest-publication statistics.

## Actual owned qualification

The fixture has **34 inventory entries: 33 Python files and one `pyproject.toml`**, partitioned into 16, 16 and 2 entries. It contains imports and calls across preparation boundaries, root/conftest fixture dependencies, tests in the final page, and pytest configuration applying to both tests. The source modules, tests and fixtures in the repository under test are never imported or executed.

Five separate isolated, nonoptimized, no-bytecode Python 3.12 processes ran sequentially under the existing unchanged MAIN configuration. The parent reserved 2 CPU / 2,048 MiB / 2 child slots; each native child reserved 2 CPU / 1,536 MiB / 2 slots. Actual nested preparation operations used 1 CPU / 512 MiB / 1 slot under the declared 1,024 MiB maximum. The configured lease TTL stayed the exact float `120.0`, with automatic renewal. Total campaign elapsed time was **131.7363038409967 seconds**.

| Process | Actual PID | Kernel birth ticks | Child elapsed seconds |
| --- | ---: | ---: | ---: |
| producer | 1338492 | 80125805 | 1.832430 |
| resumer | 1339069 | 80126008 | 5.227253 |
| reference | 1340106 | 80126553 | 2.497790 |
| controls | 1341040 | 80126824 | 89.974342 |
| regression | 1367646 | 80135837 | 31.028730 |

The producer stages the first **16** Python members off-store and transfers the full root, cursor, immutable CAS closure, complete first-page projections and authored source bytes. The owner independently validates the complete transfer bytes and producer birth, then creates the receiving admission record. Lease capability travels only through ephemeral bounded stdin.

The fresh resumer restores those 16 projections with **zero parser calls**, parses exactly the remaining **17** Python files, and retains the configuration member as explicitly unindexed AST input. It constructs its own private file-backed DuckDB/CAS/catalog owners, then calls `complete_current` with default reuse. Semantic completion performs **zero additional native frontend extractions**.

The independent reference uses the existing `RepositoryCodebaseIndex.prepare_current` route without calling the new completion method. It performs **33 ingestion frontend extractions and 33 semantic frontend extractions**. Only the ingestion module's local clock attribute is replaced by a delegating fixture clock to preserve `1700000000.0`; shared `time.time` and scheduler clocks remain real. The reference's process-local ingestion publication count is one; the staged resumer's is zero. Those route-specific observations intentionally differ.

All complete normalized parity objects and serialized bytes match: source snapshot, `CodebaseIRManifest`, `RepositoryState`, all 33 AST projections, all rows of the 13 AST families, catalog metadata, head, operation/receipt rows and selected cross-page graph witnesses. Only the physical CAS-root field of native catalog metadata is normalized; each actual path remains recorded and binds its child's private owner. Full publication documents retain different route observations. The shared preparation root is `baguqeeraw7l63se3dydf75j6jtej6qtq26t2gwr4nimtx3wo65tar72nivnq`.

The 13 families are source revisions, source files, AST blobs, AST nodes, scopes, symbols, imports, references, calls, effects, interfaces, diagnostics and invalidations. The full projection document is **732,116 bytes**, SHA-256 `2524769972d6aa1da1784c32a27a823b42c1544f25031ff859f8808dbb4c311a`, RAW CIDv1 `bafkreibfer3js4wwvio2c6cmgkrhvar3ilavitzfamp7qwpyqcg3wtbrdi`. The complete canonical parity document is **1,611,741 bytes**, SHA-256 `3e18087cc932c70daa2724d9276dc8fada3174b464d6e2522fc3fff625e9fa0c`, RAW CIDv1 `bafkreib6daehzsjsy4g2ujze3etw3sh23iyxjnde23rfel6d773cl2p2bq`. Original projection clocks are preserved.

**All 54 actual controls passed:** 22 native semantic/SQL controls and 32 separately executed inert regressions (14 live AST staging plus 18 pure protocol controls). The inert regression process installs no DuckDB importer and verifies SQL/model packages stay absent. Controls include default/opt-out full parity; cross-page calls/fixtures/configuration; incomplete prefix, exact flags and owner refusal; entry/source/AST/manifest/memory bounds; source edit/add/delete/rename/configuration drift; parser/options/implementation changes; independent cursor/page admission; missing/corrupt ASTs; cancellation during assembly, after sealing, after AST application and after commit; before-commit rollback; competing heads; same-operation and historical retries; operation-ID conflict; committed cleanup errors; profile mismatch and detached callback aliases. No skipped or expected-failure cases receive credit.

All five actual process births and private workspaces were independently joined to native lifecycle observations. Leaders were reaped, descendant/group observations completed, readers/stdin writers joined, workspaces removed and owned child/parent reservations released. Source opening/closing pins matched and MAIN was idle afterward. The successful receipt is `artifacts/codebase_ir_terminal_bench/preparation-semantic-completion-owned-checks-20261012-03/receipt.json`, **126,618 bytes**, SHA-256 `7def291d6fe44dbfd9b41ee9b2c6ee6643b0dbe96548e0c4f17695642e445c0e`. Eleven retained actual outputs total **4,736,566 bytes**.

The databases were actual private file-backed native DuckDB owners, closed by their children. The child records an ephemeral closed-image fingerprint, but the owner receives and retains complete SQL readback documents rather than database binaries. Workspace cleanup removes those database files. This campaign qualifies the producer/resumer AST restart and native SQL publication within each child; it does not qualify a fresh-process database reopen.

## Source application and bounded evidence

| Module | Bytes | SHA-256 |
| --- | ---: | --- |
| `codebase_preparation_runtime` | 40496 | `2495b0bbffa6f525fae426f3d76deaa9624402e61dfcf17260213440b8dda7ab` |
| `semantic_index.python_analysis` | 58826 | `d17a56003b525949ae60ed778e1f6f52e07991be1997aeb872d4e30a1f592856` |
| `semantic_index.pytest_analysis` | 31108 | `15e71419ec045f18b73db32ca4516586bc882da85d608d44955d6dff7551959c` |
| `semantic_index.scanner` | 38909 | `c95fd8ec9a2a6491ef74bc8b9fc76f56a13f9870f8a2c5d32da27eecbbb7ba2d` |

The same exact qualified 22-control source was applied to `external/ipfs_datasets/tests/unit/logic/software_contracts/test_codebase_preparation_completion.py` (30,532 bytes, SHA-256 `512d2b2663ab314f6001d94e35c4c636adcdd26f165b846626bc5e81ebd2fbaa`). Its ordinary CI fixture helper is byte-equal to the captured 5,176-byte fixture. `preparation-semantic-completion-candidate-20261012-03/application.json` binds all six byte-equal live/frozen source pairs and the successful receipt. The package-wide CI entrypoint was not replayed; exact selected source ran in the owned native campaign.

The native plan binds **123 inputs / 61,133,421 bytes**, including 25 project leaves / 1,241,935 bytes, 48 multiformats dependency members / 670,027 bytes, 39 DuckDB package leaves / 450,381 bytes, the exact archived 54,541,064-byte native extension, its captured 4,245-byte DuckDB 1.5.5 distribution metadata, the captured Git discovery executable, loaders, fixtures and all test registries. Direct inputs stay within 160 members / 80 MiB, ordinary leaves within 4 MiB, and only that exact archived native extension receives the 64 MiB exception. Loaded project source is bounded to 40 leaves / 16 MiB. No site initialization, ambient package fallback, network access or model execution is admitted by the selected loaders.

Transfer bundle ceiling is 16 MiB; projection/publication ceilings are 4 MiB each; observations are at most 1 MiB and owner receipt at most 4 MiB. Private workspace ceiling is 256 MiB / 32,768 entries, with a 300-second per-mode timeout. Two earlier native qualification attempts are retained with **zero campaign credit**: the first lacked selected distribution metadata; the second used an unquoted reserved SQL table identifier in verification readback. Both completed owned cleanup and left MAIN idle. Corrective candidates are immutable and retain all preceding scripts and actual diagnostics. The production runtime and analyzers stayed unchanged through these harness corrections; the ordinary test helper's SQL quoting was corrected before successful qualification.

The final integrity reader retains every one of the preceding **1,533 full pins**, the preceding final close, all frozen candidates and both failed campaigns. A prospective **1,632-member / 640 MiB cumulative-read** profile was declared before the third native attempt, with no retirement and the same sole native-extension exception. Exact union, two full held-byte passes, MAIN-state and output reserves are recomputed before reader execution. Mutable resumable work notes remain observations outside the immutable input closure. The integrity reader replays no semantic, SQL, Git, model, training or CUDA workload.

## Usage limits and next work

Construct matching `DuckDBASTStore`, `DuckDBASTIngestor`, `ImmutableCAS` and `CodebaseCatalog` owners. Capture/stage pages using `RepositoryCodebasePreparation`, retain the independently admitted root/cursor/page chain, and call `complete_current` only after full inventory completion. Use the returned catalog receipt and separately reobserve currentness when planning or executing against the live repository. Supply `optimized_frontend_reuse=False` for fresh semantic frontend extraction. Source/current-head observations are bounded point checks rather than a filesystem lock or continuous custody.

Global semantic assembly is still one bounded complete operation; it is not resumable or persisted as per-source semantic fact/index pages. Full inventory capture also remains bounded rather than paginated. Larger complete-300 qualification, bounded global/index-page representation and signed supervisor worker receiving/dispatch remain subsequent milestones. Native catalog limits must be explicitly admitted for a larger profile.

This structural fixture performs **no learned model inference, on-demand training, federation, gradient synchronization, CUDA qualification, signed worker dispatch or formal proof checking**. Its manifest records zero formalized/checked properties. It does not qualify any 8D, 384D, 768D or Leanstral 4096D model head or grant solver/execution authority. Process timings cover different staging/reference workloads and do not establish a general throughput speedup. See [the preparation design](codebase_resumable_preparation_design_v1.md) and [the preceding AST restart qualification](codebase_live_preparation_restart_qualification_v1.md).
