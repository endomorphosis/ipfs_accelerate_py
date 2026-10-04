# Symbolic security benchmark profile

The full profile builds the initial source, capsule, vector and world indexes before
the planning router generates the admitted goal/subgoal/task graph. It also trains
a small code AST autoencoder and hydrates its checkpoint, features and ranked
observations into native world and metadata DuckLake catalogs. Admission reuses
these exact artifacts without another training run. Training, persistence and
verification are charged to agent time.

The autoencoder reuses the modal training engine's numerical reconstruction loss,
gradient and batching code. Its `security-code@1` feature and checkpoint schemas
are separate from Legal IR. The security transfer profile forks the compatible
`token:*` embedding rows from an explicitly pinned LegalIR checkpoint, using the
native tokenizer and preserving their dimensions and floating-point values.
Legal classifier heads, sample memory and legal views are excluded. New code and
CVE projection parameters start at zero; this profile does not initialize random
weights. The original LegalIR checkpoint is read-only, and existing output
directories cannot be overwritten. Checkpoint loading verifies
source hashes, implementation hashes, parameters and actual inference replay.
Reconstruction error supplies advisory nominations; it cannot exclude code from
analysis, infer a valid contract, or grant proof/publication authority. Its training
is transductive, with no held-out vulnerability accuracy claim. Legacy archives
without a selected initializer retain the earlier random-initialized pilot.

Host qualification retains a source snapshot and replays exact extraction. The
container receives only the portable initializer and its pinned manifest; it
checks artifact integrity and schema, rather than claiming to repeat extraction
from a LegalIR checkpoint that is absent from the container. The canonical
checkpoint used in qualification has 3,242 token rows of width eight. Its training
history and quality metrics are unavailable; checkpoint lineage is not evidence
of effective transfer.

The reviewed `wsgi-header-controls@1` profile runs a deterministic AST adapter over
the declared Python modifications. It recognizes a bounded normalizer/store/header
emission path under explicit WSGI and Python assumptions. It emits native
SecurityIR declarations, formalization artifacts and obligations, then synthesizes
CR/LF/NUL guards while retaining accepted-input normalization. A native Tactician
and Hammer execute real Z3 and Lean checks. Exact source/candidate/contract and
toolchain identities bind the sealed local proof receipt and proof index.
The compiled SecurityIR obligations are not themselves discharged by these
solvers. The solver inputs express the separately reviewed abstract guard;
SecurityIR artifact identities bind provenance, while exact AST replay and the
declared assumptions connect that template to the Python candidate.

The candidate goes through the ordinary allocated worker, validation, publication,
and native task completion machinery. The local theorem does not close the whole
program Doctor impact graph. Dynamic dispatch, monkeypatching, alternate flows and
full HTTP grammar remain explicit frontiers. Unsupported analysis cannot promote
a neural score or solver result to a proof. The general source-to-SMT adapter now
rejects unmodeled control flow and non-SSA bodies that previously permitted false
proofs from contradictory linearized returns.

After publication and successful STOP, the native proof index invalidates receipts
whose source dependencies changed. Semantic and retrieval refresh has its own
receipts. The initial autoencoder is retained as historical; a future task must
revalidate or train a successor before treating its nominations as current.

Security, functional and temporal projections have distinct contracts. This
profile implements the local header/security projection. TLA+ inference and model
checking are explicitly unsupported; autoencoder reconstruction is not a temporal
specification or verified source-to-logic translation.

## CVE training source

The Publicus dataset is
[`Publicus/cvefixes-security-ir-graphrag`](https://huggingface.co/datasets/Publicus/cvefixes-security-ir-graphrag),
pinned to revision `6fd5918bed34f8851430e74a149502587a953fe2`.
`security_cve_training_source.py` uses the native complete-release verifier and
bounded derived graph shards. Raw original vulnerability/fixed-code records are
not needed by this reader. The graph publication omits canonical SecurityIR
training records and code-body hash provenance, so its observations are quarantined
from training. It does not fabricate those missing records from retrieval text.
Benchmark exclusions include the entire Bottle repository family, filenames and
permitted source hashes; missing provenance never counts as proven disjointness.

`security_cve_canonical_export.py` now builds that missing canonical export. It
selects a repository and row through the pinned release's native routing and graph
indexes before fetching a bounded original row. The transport endpoint is not
revision-pinned: the adapter recomputes the native row CID and requires it to match
the immutable index. It verifies reviewed source provenance and exclusions, then
exports typed SourceRecord, CodeUnit and PolicyCandidate records with body hashes
and code-derived lexical/AST inputs. Portable artifacts exclude raw bodies,
excerpts and diff headers; their closed schemas reject extra raw files or fields.

The first admitted export contains one non-Bottle `django-s3file` CVE row: 37
canonical records and two vulnerable/fixed training pairs. Both bodies lack an
admitted AST sample, which is explicitly reported; lexical inputs remain usable.
A separate supervised head learns audit/CWE/polarity classification targets
alongside code reconstruction. Target graph features are never training inputs.
Predictions are candidate observations with no policy, proof or execution
authority. Two training pairs qualify the plumbing, not vulnerability detection
accuracy or a general neural source-to-logic translator.

To select this profile, pass `security_initializer`, `canonical_cve_export` and
`canonical_cve_manifest_sha256` to `terminal_deployment.build_runtime_archive`.
The full Harbor arm forwards those pinned assets to initial indexing; the
no-index arm does not. The planner sees bounded lineage summaries. Admission
independently revalidates the selected assets and reuses the same checkpoint.

## Interpretation of results

This operator was developed using the benchmark's public source and instruction;
the task is a development benchmark, not a held-out security evaluation. The
official verifier remains separate. Retained trials include failures, setup,
cold index/training time, planning tokens, coding tokens and cache counters.
The no-index arm removes both indexed context and Doctor selection, so it is a
combined-system ablation and cannot isolate the benefit of the autoencoder.
The profile uses one worker and makes no parallel-speedup or dollar-savings claim.
