# Paged evidence and formal limitations for committed chunk projections

`chunked_reconstruction.reconstruct_chunked_semantic_state` is an explicit
consumer of the datasets streaming producer at `30e76cefc93635a0f7d3e80a190174aed7f14785`.
It builds on the formal limitation consumer
`af811cae6743138cb0afaa2e86c044993e8dea5d`, paging consumer
`eae35e0811dd2f04fa9c2a4db792032fca43cc2a`, and retains the captured consumer
`c705c3751b11dcd6fc693258970708e0f5582ef9` unchanged. Both source revisions
and actual imported bytes must be qualified by the native launcher. The existing
`reconstruct_semantic_state` API and snapshot schemas retain their behavior.

The caller supplies exact repository, commit, tree and chunked snapshot CID,
plus qualified `admission_limits` (bounded defaults). It accepts a typed manifest
or an immutable CID-to-bytes mapping. Object limits and population/chunk counts
are checked before serialization. Closed DAG admission checks every canonical
page, reference, count, ordering and population identity against fresh Git
metadata before blob reads. Both accepted input forms produce the same bundle.

The consumer reprojects Git content, verifies complete raw-path coverage and
the projected object/source identities, and scans only captured bounded inputs.
Known opaque inputs and scanner analysis failures become formal datasets
`AnalysisLimitation` records with opaque confidence. An opaque input missing
from the scanner's artifact population is a refusal.

The closed datasets limitation model exposes code, message, subject and
confidence. Each subject therefore names a separate provenance artifact in the
root artifact index. Its structured metadata binds repository identity,
commit/tree, population CID, chunked and projected snapshot CIDs, exact raw path,
Git mode/object, source CID/size, snapshot entry CID, paged snapshot root CID, original opaque artifact
fact CID, projection/stream limits, and the reason analysis is unavailable.
Verified blob content and unanalyzed semantics are separate explicit facts.
Gitlinks never claim verification of nested repository bytes.

Before semantic-state serialization, the explicit consumer replaces the
scanner's full snapshot-evidence artifact with a small reference to bounded entry
pages. The original snapshot and entry CIDs remain unchanged. The new evidence
DAG is independently parsed, including every entry CID and the reconstructed
original snapshot CID; all pages are persisted in the returned bundle. The
ordinary scanner and captured snapshot-evidence path remain unchanged.

The augmented state and the formal index are rebuilt together. The producer's
source-manifest CID binds the chunk manifest, whose complete metadata DAG is
included in the returned bundle. The consumer checks the exact formal index,
every expected limitation leaf, original/provenance and snapshot-evidence
artifact leaves and their root index membership, and every chunk manifest and
paged snapshot block. These checks also apply
to nominated bundles: a matching root with missing evidence is refused, as is a
schema-valid nomination whose limitation index was cleared.

Observations report `analysis_coverage=incomplete` when known limitations exist.
An empty known-limitation index reports `not_established`, and every result keeps
complete-analysis, semantic-acceptance and completion authority false. No
nomination, empty index or content hash establishes positive analysis coverage.

The default retained projection keeps its 4 MiB per-file and 128 MiB aggregate
source ceilings. Returned semantic and manifest metadata blocks retain a 1 MiB
ceiling. A complete
15,770-entry synthetic population now fits bounded snapshot/manifest pages; no
full live SPAR snapshot was acquired. Original snapshot-CID reconstruction and
scanner acquisition metadata still use bounded full metadata in memory. Other
oversized state indices, artifact facts or AST blocks may still require paging;
this consumer refuses them. Oversized code still needs separately
qualified analysis before its limitation can disappear.

Passing typed `streaming_limits=StreamingScanLimits()` opts into the new cold
ordinary-source path. It consumes one committed blob at a time, retains no source
bytes in the snapshot, and runs static Python/pytest extraction and final graph
assembly in owned, bounded children. Canonical facts and final state have a
32 MiB aggregate bound; individual records retain a 1 MiB bound. AST counts,
worker address space/CPU/time and pipe sizes also have fixed limits. Failures
raise `StreamingReconstructionError` with a structured producer refusal. An
oversized final bundle block is likewise a structured refusal. The existing
captured consumer remains byte-identical and the default chunk path is preserved.

When streaming is selected, configuration schema v3 binds the explicit mode,
fact/AST/time limits and the observed process profile, including worker source
hash. Worker source and committed Git fences must remain stable throughout the
scan. Runtime RSS measurements appear only in observations and never alter
canonical identities. A small mixed fixture produces exactly the same paged
bundle as the retained path; its configuration digest differs because execution
bounds differ. Separate producer and full-consumer tests consume a real
138,410,910-byte population of 66 ordinary Python files and preserve every source
CID without converting any input to opaque.

These are canonical fact bounds, not a hard cap on total parent RSS. The producer
report measures parent Python allocations separately from analyzer/assembly
child RSS; Git decoding retains its independent process cap. The current
32 MiB fact bound, other oversized semantic indices/ASTs, oversized opaque code,
and native qualification remain possible refusals for the actual SPAR population.
No full SPAR acquisition was performed.

This change does not schedule providers, alter boards/holds, update live source
pins, or grant native admission. Durable storage/admission, independent producer
qualification, owner/source fences, complete forest acquisition and runtime/goal
settlement remain native responsibilities. No real SPAR population was acquired
to validate this source change.
