# Formal limitations for committed chunk projections

`chunked_reconstruction.reconstruct_chunked_semantic_state` is an explicit
consumer of the datasets chunk producer introduced in commit
`cd8c041573091e40d4ae622f3975f0ae3465734a`. It builds on accelerate's captured
consumer commit `c705c3751b11dcd6fc693258970708e0f5582ef9`. Both source revisions
and actual imported bytes must be qualified by the native launcher. The existing
`reconstruct_semantic_state` API retains its behavior.

The caller supplies exact repository, commit, tree and chunked snapshot CID.
The consumer reprojects Git content, verifies complete raw-path coverage and
the projected object/source identities, and scans only captured bounded inputs.
Known opaque inputs and scanner analysis failures become formal datasets
`AnalysisLimitation` records with opaque confidence. An opaque input missing
from the scanner's artifact population is a refusal.

The closed datasets limitation model exposes code, message, subject and
confidence. Each subject therefore names a separate provenance artifact in the
root artifact index. Its structured metadata binds repository identity,
commit/tree, population CID, chunked and projected snapshot CIDs, exact raw path,
Git mode/object, source CID/size, snapshot entry CID, original opaque artifact
fact CID, projection/stream limits, and the reason analysis is unavailable.
Verified blob content and unanalyzed semantics are separate explicit facts.
Gitlinks never claim verification of nested repository bytes.

The augmented state and the formal index are rebuilt together. The producer's
source-manifest CID binds the chunk manifest, whose complete metadata DAG is
included in the returned bundle. The consumer checks the exact formal index,
every expected limitation leaf, original/provenance artifact leaves and their
root index membership, and every chunk manifest block. These checks also apply
to nominated bundles: a matching root with missing evidence is refused, as is a
schema-valid nomination whose limitation index was cleared.

Observations report `analysis_coverage=incomplete` when known limitations exist.
An empty known-limitation index reports `not_established`, and every result keeps
complete-analysis, semantic-acceptance and completion authority false. No
nomination, empty index or content hash establishes positive analysis coverage.

The 4 MiB per-file and 128 MiB retained-source ceilings remain enforced. Returned
semantic and manifest metadata blocks have a 1 MiB ceiling. Larger ordinary
source populations still require a streaming scanner/pytest frontend. Large
legacy snapshot-evidence artifacts and other oversized state blocks still need
paging; this consumer refuses them. Oversized code still needs separately
qualified analysis before its limitation can disappear.

This change does not schedule providers, alter boards/holds, update live source
pins, or grant native admission. Durable storage/admission, independent producer
qualification, owner/source fences, complete forest acquisition and runtime/goal
settlement remain native responsibilities. No real SPAR population was acquired
to validate this source change.
