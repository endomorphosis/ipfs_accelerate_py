# Bounded local code evidence queries

`analysis.repository_code_evidence.RepositoryCodeEvidence` composes the exact
native `IntentCodebaseCatalog`, `RepositoryCodebaseIndex` and
`FiniteCheckedCache`. It creates no new database or source/proof head. Every
query rechecks the complete source generation and scan policy, replays the
semantic manifest and discovery nominations, and checks producer bytes before
returning. Existing source captures and active AST projections remain the owners
of bytes, parser identity and source spans.

`RelevantUnitsQuery(paths=(), page_size=16)` returns complete inventory or an
explicit path selection. It never infers semantic relevance from a name.
Unmodeled entries remain visible. `UnitEvidenceQuery(path, inputs, page_size=32)`
requires 1..32 explicit inputs and adds exact scoped cache lookup and reverse dependency references for an explicit
integer domain. `resolve_intent(...)` invokes the fresh behavioral matcher and
returns every requirement and residual in its bounded supported profile. Saved
facts or match reports are not accepted as input.

Pages bind the full query, source head, manifest, producer pins, discovery
record IDs and ordered row commitments. Evidence pages also commit the cache
record, full canonical key, artifact bundle and scoped receipt, when present.
A cursor from a different source, query, producer or record population fails.
Missing rows and overflows fail closed. `selection_complete` is true only when
one page contains the entire selected population. The final page of a longer
query reports `page_is_last`; its cursor prefix is explicitly not evidence that
the caller consumed earlier pages.

The local plane reuses native `EvidenceNodeRow`, `EvidenceEdgeRow`, AST projection
and `ImpactGraph` types. All serialized nodes and edges have `authoritative=False`.
Positive table-proof eligibility is a separate result of fresh native cache
lookup, which reruns the finite Python/Lean observation on every positive page.
Negative and reverse-reference rows are historical; missing evidence cannot
become success. This does not save checker calls or prove general Python
equivalence.

The existing `CodeEvidencePlane` requires a DQP-039/Quack release identity. This
local profile has no independently verified release owner and refuses that
projection explicitly. It does not construct a fixture release identity.
The local projection is deterministically rebuilt from durable native stores,
including in a fresh process.

The [qualification](evidence/repository-code-evidence-20261002/README.md) covers
28 distinct tests. It includes exact AST pages, current dirty source refusal,
formatting recapture with changed spans, full-key/domain isolation, positive and
refuted history, deleted artifact and typed-receipt refusal, ignored model-off
asset changes, preserved intent residuals and cold-process replay. The ignored
asset test concerns an explicitly unconsumed model in a model-off profile; it
does not establish reuse across arbitrary learned-model changes.

RPI-023/032 remain open for the genuine DQP/Quack release route, broader model and
dependency profiles, all requested submodule/worktree/rename combinations at
this service boundary, and mandatory freeze-to-admission integration. This
query layer grants no task omission, execution or completion authority.
