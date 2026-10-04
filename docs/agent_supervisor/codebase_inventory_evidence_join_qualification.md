# Current inventory and conditional evidence planning join

The datasets `scan_current_codebase_evidence` API joins the complete captured
inventory and frozen CodebaseIR features to the existing native conditional
evidence catalog. The supervisor `preview_current_inventory_evidence_plan` API
binds its advisory references into the native model-off planning snapshot. These
are additive Python APIs. Production task preparation and signed admission keep
their separate qualification gates.

The [datasets producer](../../../ipfs_datasets/ipfs_datasets_py/logic/software_contracts/codebase_inventory_evidence.py)
calls the default optimized inventory scanner; `optimized=False` selects the
existing reference path. It then makes a bounded paginated query through
`CodebaseVerificationCatalog.query_current`. It retains the full source ledger
even when numerical inference or evidence retrieval reaches its budget. Source
identity, authored contract, requested domain, complete canonical proof key,
verification identity and evidence epoch remain distinct bindings. Learned
features do not identify a checked property.

| Evidence disposition | Meaning for the exact query selector |
| --- | --- |
| `matched_complete` | The complete query returned indexed conditional evidence for this member. |
| `no_exact_indexed_conditional_evidence` | The complete query returned no indexed evidence for this member. |
| `matched_partial` | Returned evidence exists, but the bounded query is incomplete. |
| `unknown_budget` | No evidence was retained for this member before the query stopped. |

An incomplete query never establishes absence. A retained continuation binds the
full source head, selector, inventory CID and epoch. If the first page exceeds
the byte budget, the result has zero retained pages, `complete=False` and
`next_cursor=None`: receiving clients restart at the sealed query root. This
empty result charges zero retained query bytes. Dependency selectors are refused
by this profile pending independent receiving-side membership qualification.

Evidence summaries retain base conditional verification and the selected
contract's applicability separately, including refutation, empty domains and
ambiguity. Multiple exact native records remain separate entries. None becomes
an observed current fact, runtime correctness claim or authoritative cache hit.

The immutable numerical record uses finite canonical native JSON with a raw
CIDv1. Float-free advisory references use the existing structured identity
codec. Both formats preserve their exact schemas and authority ceilings.
Recomputing a checksum does not attest execution of a scan or a checker.

Receiving `validate_current_inventory_evidence` checks complete captured source
membership, source/AST identities, current model ancestry and checkpoint layout,
then replays every retained native query page and reconstructs its evidence
summary. This rejects a rehashed record with invented proof-status metadata even
when its epoch remains current. It runs no forward inference, fitting or solver.
Numerical features remain advisory: a raw CID and exact model identity alone do
not authenticate arbitrary caller-rehashed numerical outputs.

Closing fences check native evidence identity, fresh full source and historical
CAS bindings, ancestry rows and artifacts, the closed model registry and installed
implementation generation. A final native evidence query catches catalog
changes during the intervening model checks. These observations are sequential
and do not provide an atomic lock across the checkout and catalogs.

The [supervisor consumer](../../ipfs_accelerate_py/agent_supervisor/planning/codebase_inventory_evidence_context.py)
validates the retained join before and after the actual
`preview_repository_plan` call. It adds the float-free references under the
reserved `codebase_inventory_evidence` material key and checks the native
snapshot and preview receipt. Independently authored intent, producers, goals,
predicates and task candidates retain their semantic bindings. All runtime
requirements remain residual; no task is removed. A changed advisory record
changes the planning input snapshot without granting execution authority.

One real datasets parent lease and a cooperative admission-inclusive deadline
cover the nested work. The supervisor also applies the request's own latency
budget. The current scan profile accepts at most 256 inventory members and
64 KiB per source file. Default evidence limits are sixteen rows per page,
sixteen pages, 256 entries, 16 MiB retained query bytes and 32 MiB output bytes.
Callers can narrow these limits; evidence page size may increase to 64 within
the unchanged page and entry ceilings. Larger resumable scans remain a separate
milestone. Default retained-byte phase estimates remain within one quarter of the
1,024 MiB reservation: 244 MiB during the nested scan and 200 MiB during query
processing. These are serialized-byte estimates, not RSS limits or kernel
containment. Native catalog result and normalized inventory limits are each
bounded at 64 MiB for this profile. Cancellation produces no successful record.

The [fresh native harness](../../benchmarks/agent_supervisor/container_coding/qualify_codebase_inventory_evidence_join.py)
uses a new private 28-entry source population, a two-epoch root and one-epoch
same-head child, and five genuine native verification/applicability publications.
Its authored examples include proved and refuted conditional properties, an
empty requested domain and two ambiguous records for one source. Fitting is
disabled after setup. Positive scans, receiving replay and native generic
previews preserve the frozen model and owner exports. Fault controls deliberately
mutate only this disposable fixture, then restore its baseline; they do not
implement a production rollback or recovery API.

The [pinned native result](../../../../artifacts/codebase_ir_terminal_bench/inventory-evidence-join-qualification-20261003-04/result.json)
qualified in 281.595 recorded seconds. Five scan profiles, two receiving
validations, four native planning previews and all twenty refusal controls
passed. Complete queries retain five native records across four source members;
all 28 source entries remain accounted for. Partial queries establish no
absence. Deferred inference preserves the independent evidence ledger, and the
exact canonical-key selector returns its single intended record. The four
independently authored runtime tasks remain, with zero current facts and no
task removal.

The model has eight latent dimensions and 53 feature columns. State SHA,
completed epochs and all four Adam steps remain unchanged after the three setup
epochs; post-setup fitting attempts are zero. Source/model/evidence owner
exports match baseline, and owned leases and waiters drain. All 46 selected
staged and retained sources match; their
[ending working-source observation](../../../../artifacts/codebase_ir_terminal_bench/inventory-evidence-join-selected-ending-observation-20261003-04.json)
also matches at that point. Parent auditing retained 24,747 launch events and
6,700,539 serialized bytes without overflow or policy failure.

The independent [whole-stage audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-evidence-integration-stage-readonly-audit-20261003-02.json)
passed 328,775 checks over every staged file, permission mode and recorded
namespace. All 14,399 original source locators also matched their captured
bytes and modes; current package membership had no added, missing or unknown
entries. These are sequential observations, not a deployment qualification.
The independent [primary-archive audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-evidence-join-readonly-audit-20261003-02.json)
passed 83,653 checks, including exact content identities, all five join ledgers,
four native previews, fifty native execution observations, twenty refusals and
unchanged owner exports. Both audits preserved their observed namespaces,
including bytes, modes and modification times, and imported no native packages
or owners. Earlier audit receipts remain historical; final receipts correct
only the reader's handling of excluded cache files and regular submodule Git
marker files. The [machine review](../architecture/repository_proof_index_and_codebase_ir.inventory_evidence_join_review.json)
records these pins, costs, coverage and remaining acceptance gates.

The producer's 147 focused cases pass, as do the consumer's 57 actual cases with
mandatory AST sealing enabled against a fresh seal database. An earlier sealed
run that skipped all 57 cases remains a separate historical receipt and is not
counted as passing execution. The 205 existing inventory inference cases also
passed after the restart. Together with the fifteen harness cases below, these
are 424 distinct passing focused cases.

The [first native attempt](../../../../artifacts/codebase_ir_terminal_bench/inventory-evidence-join-qualification-20261003-01/result.json)
remains failed historical evidence: 53.419 seconds and three completed setup
epochs, with no unknown fitting cost. All five native evidence publications
completed before the new harness's process observer rejected the legitimate
native `prlimit` wrapper around the inference worker. That observer assumed a
bare Python launch. The correction validates the native limiter arguments and
exact isolated worker command; it changes no producer, resource limit or
historical artifact. The failed run drained its owned leases and waiters.

The [second native attempt](../../../../artifacts/codebase_ir_terminal_bench/inventory-evidence-join-qualification-20261003-02/result.json)
also remains failed: 182.265 seconds and three known setup epochs. Five positive
joins and four native planning previews passed, followed by eight refusal
controls. Preparation of the next corruption control used the default
`dag-json` CAS path lookup for a `raw` source CID and refused before mutation.
The correction supplies the native owner's existing `source=True` argument.
All 46 selected sources still matched their retained copies after failure;
leases and waiters drained. The remaining twelve controls were unqualified by
that attempt. The final fresh run has a separate 600-second overall reserve;
individual native operation limits remain at 120 seconds, with the authored
planning request's 90-second limit also enforced. This development reserve does
not qualify the deployed trial budget.

The [third native attempt](../../../../artifacts/codebase_ir_terminal_bench/inventory-evidence-join-qualification-20261003-03/result.json)
failed its source-generation fence after 229.308 seconds and three known setup
epochs. Five joins, four previews and sixteen controls completed before
concurrent workspace changes added `query_many_current` and its request type to
the catalog/query modules. Four controls remained unqualified. The source fence
refused the changed generation; restored owner exports matched baseline through
the last completed control, and leases and waiters drained. All three failed
attempts remain immutable.

The final qualification uses an isolated source/resource snapshot of the three
local packages and exact harness/helper/test inputs. Streaming copy, per-file
content identities, complete namespace checks and ending source comparisons
precede sealing. Copied paths contain no links to mutable working source;
one internal font-resource alias is dereferenced with its exact target guarded.
Original read/execute permissions are preserved while write permissions are
removed. This qualifies the pinned integration generation. Later workspace
differences are reported separately and do not inherit its qualification.
The separately developed native batch API remains available in that generation;
the join uses its backward-compatible single-request entrypoint and makes no
batch performance claim.

The [sealed integration manifest](../../../../artifacts/codebase_ir_terminal_bench/inventory-evidence-join-integration-20261003-01/integration-manifest.json)
accounts for 14,399 files and 1,297,593,612 bytes. Copy/closure checking took
62.234 seconds and the final repeat seal took 4.443 seconds, with no fitting or
native jobs during staging. Native runs 01–04 together cost 746.588 recorded
seconds and twelve setup epochs; the failed attempts contribute 464.993 seconds
and nine epochs. No attempt has unknown fitting cost. Staging and unit/audit
time remain separate from these native run totals.

Fifteen focused harness regression cases pass for the retained native command,
changed limiter/worker/library arguments, event-count and byte overflow, and
rejected processes whose errors a bounded runner could otherwise mask. The
corrected Linux observer retains at most 32,768 parent launch events and 16 MiB
of their exact serialized array. Overflow and launch-policy failures invalidate
both negative controls and the final qualification. Parent observations do not
observe solver descendants inside native verification workers; the genuine
setup phase receipts retain that separate evidence.

The [restart readiness receipt](../../../../artifacts/codebase_ir_terminal_bench/inventory-inference-reuse-restart-readiness-20261003-01.json)
and [read-only historical audit](../../../../artifacts/codebase_ir_terminal_bench/inventory-inference-reuse-restart-archive-audit-20261003-01.json)
preserve the preceding inference experiment. Two current resource modules
changed after that experiment; its 37.8% latency reduction remains a result for
its retained source generation. This join is a functional qualification, not a
new throughput comparison. It qualifies neither 384D/CUDA CodebaseIR inference,
learned semantic decoding, general source/runtime correctness, signed admission,
worker execution, automatic recovery nor production activation. All 32
production acceptance tasks remain open.
