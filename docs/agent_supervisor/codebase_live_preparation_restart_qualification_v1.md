# Live repository preparation and fresh-process AST restart qualification v1

Date: 2026-10-12. This milestone connects the existing immutable preparation ledger to owned live repository capture and bounded AST extraction. It stages pages outside the repository and active index, then restores complete projections after an actual interpreter restart. Cross-file semantic assembly and catalog publication remain subsequent stages.

## Implemented behavior

`RepositoryCodebasePreparation` exposes `start_current`, `prepare_page`, `validate_current`, and `load_projections`. A root seals the complete captured inventory, repository identity, admitted catalog head when supplied, qualified frontend/options/schema context, implementation source identities, and original finite creation clock. Each page covers the exact next contiguous slice. The default profile accepts at most 1,024 members, 64 KiB per source and 16 members per page (up to 64 explicitly); one operation shares a 128 MiB artifact validation payload budget. Full inventory capture is bounded but is not itself paginated in this milestone.

`DuckDBASTIngestor.prepare_snapshot_subset` extracts against the complete root identity without installing revisions, changing latest-publication pointers, generating active invalidations, or writing the AST store. Ordinary parser shard caching remains a separate policy. A stored native AST envelope preserves the extraction decision and qualified parser context, while a complete projection codec preserves all 13 relational families, metadata, diagnostics, synthetic failure metadata for the codec-only control, and hexadecimal binary-float clocks. Runtime pages contain native Python AST records; codec-only authored failure/partial fixtures are explicitly distinct from native extractor outcomes.

`optimized_resume=True` is default. `False` re-extracts and compares completed pages; it does not silently change the ingestor's independent shard-cache policy. Live entry and closing observations recheck full source membership/bytes, parser context and the expected catalog head. Cache storage root, namespaces and codec shards are checked against symlink redirection and must stay outside the scanned repository. These are bounded point observations, not continuous filesystem custody.

A receiving cursor requires an independently admitted preceding page CID. Recomputing a CID authenticates bytes, not the claimed analyzer. The native owner therefore retains and verifies the producer bundle fingerprint, root/cursor/page identity, source fingerprints, complete first-page projection fingerprint, and actual producer process birth before forwarding a separately created public admission record. Parent lease capability travels only over bounded ephemeral stdin and is not retained in the handoff.

No source module in the repository under test is executed. Git is used only to discover snapshot ownership for the authored private fixture. Selected project package parents remain inert, and SQL/model packages are unavailable to this campaign.

## Actual owned qualification

Four separate isolated, nonoptimized, no-bytecode Python 3.12 processes ran sequentially under the existing unchanged MAIN configuration. The owner retained a 2 CPU / 2,048 MiB / 2 process-slot parent reservation; each child used 2 CPU / 1,024 MiB / 2 slots, with page operations taking bounded nested subleases. No service was reconfigured. The whole campaign took **56.13272001699079 seconds**.

| Process | Actual PID | Kernel birth ticks | Child elapsed seconds |
| --- | ---: | ---: | ---: |
| producer | 3350518 | 79398029 | 2.666850 |
| resumer | 3350858 | 79398311 | 4.414660 |
| reference | 3352623 | 79398765 | 1.943350 |
| controls | 3353008 | 79398971 | 46.560542 |

The producer parsed the first 16 of a complete 34-member inventory. The new resumer restored those 16 projections with **zero parser calls** and parsed exactly the remaining **18** across pages `[16,32)` and `[32,34)`. An independent full reference parsed all **34** members. All complete normalized projection objects and serialized bytes were identical between the resumer and reference, including all saved clocks. The first 16 producer projection objects also matched both full results.

The 13 compared families were source revisions, source files, AST blobs, AST nodes, scopes, symbols, imports, references, calls, effects, interfaces, diagnostics and invalidations. The full projection document was **668,214 bytes**, SHA-256 `4f7402fc2c42e7b0606927a1313b97e6829db2e209b6246d7df507cab33fd68f`, RAW CIDv1 `bafkreicpoqbpylcc46yga2jhueytxf7gqko3fyqjwysg27pva7flgp6wr4`.

The shared full root was `baguqeera4ziq7fjll2dagkeksaqzyalxyw5ctuwiol652nsy5vpxl72a3biq`. The fixture clock was `1700000000.0`; restoration preserves that original clock rather than assigning a resume timestamp.

**All 32 actual controls passed:** 14 integration controls plus the preceding 18 protocol controls rerun against the shared-reader factoring. Integration coverage includes full reference parity, default/opt-out prefix behavior, complete codec metadata/clocks, failed/opaque/unsupported dispositions, edited/added/deleted/renamed source refusal, changed frontend/context/options/implementation refusal, missing or changed expected-head refusal, corrupt/missing artifact and cursor refusal, independent admission refusal, cancellation before/after sealing, closing observation fences, unqualified frontend refusal, and namespace/shard symlink refusal before any write. Expected-head mutation controls use an explicit test double; they do not qualify a real catalog publication.

The owner independently joined each actual process birth and private workspace to the native lifecycle observer. All leaders were reaped, tracked descendants and process-group observations were complete, output readers/stdin writers joined, workspaces removed, own child and parent reservations released, and MAIN was idle afterward. All selected source opening/closing pins matched. The successful owner receipt is `artifacts/codebase_ir_terminal_bench/live-preparation-extraction-owned-checks-20261012-01/receipt.json`, **86,758 bytes**, SHA-256 `07f7afec0b0261505d5741460a28b3784e09aacd64eb56eb4bba98b37b870278`.

There were no failed native attempts for this milestone. Nine retained actual outputs total **2,501,977 bytes**. The plan binds **72 source inputs / 5,796,360 bytes**, including 17 project leaves / 977,872 bytes, 48 dependency members / 670,027 bytes, the captured Git binary, loaders and both exact test registries. Direct source reads are bounded to 128 members / 8 MiB with 4 MiB leaves; loaded project leaves to 32 / 8 MiB. The child-to-owner bundle cap is 16 MiB, projection cap 4 MiB, observation cap 1 MiB, and private workspace cap 256 MiB / 32,768 entries. There is no broad checkout traversal or unrelated native replay in qualification.

## Applied source

| Module | Bytes | SHA-256 |
| --- | ---: | --- |
| `codebase_preparation.py` | 30,865 | `3407ad816120a848d044fafd9a1e532f969e2696325419c1f68ef3dcf1d2ebd7` |
| `codebase_preparation_runtime.py` | 29,970 | `c9614cfda6db9fa186c9c6d1c7b39c0dca286b1056aae00f1ce7c6cd56dba8bf` |
| `codebase_preparation_projection.py` | 6,320 | `c9f3fddc15d169aaa911d00f388b2106532175f930d2c77891e5f19732c9377d` |
| `duckdb_ingest.py` | 113,529 | `8ca005c4a396917168088dab431596d1e7a16be0b832c94b5dc587369b221f96` |

The exact qualified 14-control source was applied as `external/ipfs_datasets/tests/unit/logic/software_contracts/test_codebase_preparation_runtime.py` (34,353 bytes, SHA-256 `d916fdf7cb169b78000de27e3419032c9847c7cf0b53af797ea2bc47945f0ee7`). The retained application record `live-preparation-extraction-candidate-20261012-01/application.json` binds byte-equal live/frozen source pairs and their metadata. The ordinary package-wide CI entrypoint was not replayed; the same exact selected test source ran in the owned native campaign.

## Usage and next milestone

Allocate an existing scheduler reservation appropriate to the preparation profile and keep the CAS outside the repository under test. Create an ingestor, construct `RepositoryCodebasePreparation(artifacts=cas, ingestor=ingestor)`, and seal a root with `start_current(..., repository_id=..., parent_lease=...)`. Persist the returned root CID separately. Call `prepare_page` with the saved cursor and independently admitted preceding page CID, retain the returned `next_cursor` property, and repeat until inventory completion. A new process can reconstruct the owners, call `validate_current` to reobserve the live repository, and use `load_projections` to restore the historical prefix. Set `optimized_resume=False` for the reference extraction opt-out.

Historical `load_projections` verifies admitted artifacts and current implementation context; it does not itself establish live repository currentness. Late cancellation or a failed closing fence can leave immutable orphan artifacts, but no cursor is advanced or active head published to the caller. A retry can legitimately emit different decision metadata after parser-cache warming while preserving complete projection semantics.

Next, assemble source-local and cross-file semantics over the entire verified inventory, qualify equivalence to an uninterrupted assembly, and admit a bounded final revision/head transaction only after complete source/context/head observations. Worker handoff and supervisor planner integration must retain the same independent admission chain. Inventory completion and AST restoration do not grant formal proof or execution authority.

This milestone performs **no semantic assembly, native SQL catalog write, active revision/head publication, signed worker dispatch, formal proof verification, training, inference or CUDA qualification**. It makes no general throughput claim. The 8D, 384D, 768D and 4096D Leanstral model-head qualification work remains distinct. See [the resumable preparation design](codebase_resumable_preparation_design_v1.md) and [the preceding pure protocol qualification](codebase_preparation_protocol_qualification_v2.md).
