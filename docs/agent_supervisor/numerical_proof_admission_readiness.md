# Numerical proof admission readiness

The next numerical phase should retain a typed qualification record for planning
and retrieval. Accepted kernel-proof admission comes later, after an independently
checked theorem, source correspondence and complete custody exist. The current
spectral reports cannot become `KERNEL_VERIFIED` receipts by conversion or by
reusing the header bridge's admission handles.

This plan reuses the reviewed spectral candidate, existing observations and
original model assets. It proposes no training, embedding recomputation, model
download, checkpoint replacement or Hugging Face publication in the next phase.
It also leaves native validation, publication and task completion with their
existing owners.

## Current structural-obligation retrieval

The opt-in `python-numerical-structure@1` context now exposes bounded retrieval
of its pending source obligations. This is separate from the spectral
qualification record proposed below. It reuses the existing source-replayed
reports, metadata DuckLake and world DuckLake; it creates no accepted proof.
Select it during benchmark preparation with
`--numerical-formalization-profile python-numerical-structure@1` and a newly
qualified archive containing both current producers. It remains opt-in; the
no-index arm records the selection but does not consume the indexed context.

The 8,192-byte planning summary retains `pending_obligation_kinds`, an exact
count by kind over the full retained population, even when all individual rows
are omitted. `obligation_retrieval` identifies the API, limits and a bounded
list of source paths with an explicit omission count. It also includes the
isolated worker command and its usage in the mandatory coding evidence, so
discovery guidance participates in complete-input accounting. The complete selected
path population remains in the owner receipt. Detailed rows continue to live
in the already retained per-source reports.

`ipfs_datasets_py.logic.software_verification.numerical_obligation_retrieval`
owns source-to-report replay, exact function/kind filtering and deterministic
paging. The accelerator consumer additionally validates the selected receipt,
source population, producer pins, complete artifact inventory and lake replay.
It checks currentness again after constructing each page. The following uses
the receipt selected by `load_task_context_selection`, which validates the
task/context binding before returning `numerical_formalization`:

```python
from ipfs_accelerate_py.agent_supervisor.runtime.numerical_formalization_context import (
    retrieve_numerical_obligations,
)

receipt = selected_context["numerical_formalization"]
cursor = None
while True:
    page = retrieve_numerical_obligations(
        repository=repository,
        expected_receipt=receipt,
        path="eigen.py",  # An exact path from receipt["source_hashes"].
        cursor=cursor,
        limit=4,
        max_bytes=8192,
    )
    consume_pending_rows(page["rows"])
    cursor = page["next_cursor"]
    if cursor is None:
        break
```

Optional `function_id` and `kind` filters match exactly. Keep filters, row limit
and byte budget fixed across continuations. The complete UTF-8 response,
including owner binding and continuation, must fit the requested byte bound;
rows are never cut in half. An insufficient bound raises an error. Empty
filters report zero matches and the full unfiltered count; this is not proof
of an obligation-free program. `exhausted` means that this query has no more
rows, not that its obligations were discharged.

Each continuation binds the context receipt, source path/hash, complete native
report and query. An obligation ID alone is insufficient: local IDs can remain
the same across different source bytes. Source edits, changed reports, different
owners or changed query budgets require a new validated query. The byte count
is a transport measurement, not a provider token estimate.

Planning remains a single provider request. During isolated coding, the owner
wrapper now provides a read-only numerical page service for that invocation.
The existing `llm_router` session can run this pinned, standard-library-only
command from the allocated workspace:

```sh
/opt/ipfs-supervisor/venv/bin/python -I \
  /opt/ipfs-supervisor/source/ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py \
  --path eigen.py --limit 4 --max-bytes 8192
```

Add `--function-id`, `--kind`, or `--cursor '<next_cursor JSON>'` as needed.
The owner supplies a protected descriptor through explicit launcher arguments;
the router validates its task, receipt, workspace and boundary before exposing
it to child tools. The worker cannot nominate a private catalog or receipt.
Kernel peer credentials must identify a descendant of the exact launched
worker process birth. Matching the worker UID alone is insufficient.

The service checks the selected canonical and allocated source snapshots before
and after native retrieval. Source edits therefore require refreshed evidence;
old indexed rows cannot be read as current advice after an edit. The complete
JSON response envelope fits the requested budget, at most 8,192 bytes, with
at most eight complete rows. Four-byte framing and the CLI's trailing newline
are transport delimiters outside that JSON budget. A session accepts at most
64 connections and prepares at most 524,288 response bytes. Its lifetime is
bounded by the coding timeout, independently of the existing 90-second startup
handoff. Shutdown closes the socket and removes its descriptor by inode.

Owner session audits retain hashes, byte counts, row counts and delivery
outcomes. The benchmark collects these into
`numerical_obligation_retrieval_observation` after worker cleanup, matched to
the admitted task and router receipt. Missing or incomplete selected evidence
yields unavailable counts. An in-flight shutdown cannot claim complete
accounting. These observations carry no source-correctness, completion or
proof authority, and bytes do not substitute for native provider tokens.

The provider-free worker qualification uses an authored `generate_text`
response to drive the actual router, launcher and worker CLI. It does not
establish that a live model chose to invoke retrieval or that complete-run
tokens decreased. Those require a separately controlled benchmark comparison.
The retained producer pins also mean old receipts require their original
producer snapshot; this change does not make them current under new code.

## Audited baseline and available evidence

The source audit used accelerator revision
`648d911be1047378afb032c48ecaf5a8fdcc330b` and the immutable datasets snapshot
`050e08d67db70817903dce750ee76b6ba839c41e`. No numerical routine, solver, target
probe, model or benchmark was executed during this audit. These are the exact
four audited source byte identities:

| Source | SHA256 |
| --- | --- |
| [`doctor_spectral_contract.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/doctor_spectral_contract.py) | `119c8556e291bd179627f12d0b36950382b624ee5e6f400d35903d1d29c3ae3c` |
| [`spectral_target_qualification.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/spectral_target_qualification.py) | `b0516c2f0db2d8f12f3d5d06fc788c299881e67bd120fe631b82b7d8e826dabe` |
| [`spectral_kernel_proof.py`](../../ipfs_accelerate_py/agent_supervisor/analysis/spectral_kernel_proof.py) | `529835978f49d57bd7234925c1dbe2da159c49ebf3edeca9ebbe8326ded11a32` |
| [`spectral_eigen_kernel.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/spectral_eigen_kernel.py) | `a63e2b74a1923ea26493e2cc26d45c1c08da1c2f1423f141fd2933585b257105` |

The [retained spectral qualification](evidence/terminal-spectral-symbolic-20261007/README.md)
records an earlier fresh run of 113 tests, with no failures, errors or skips.
Those records remain historical evidence. The four source files above still
match their retained hashes; other integration files, including dispatch, have
subsequent changes. The old suite does not qualify every current integration
file or a newly selected target environment.

Keep the evidence classes separate:

| Evidence | What is established | What remains open |
| --- | --- | --- |
| 13 Z3 checks | Conditional ideal algebra and bounded computed-magnitude selection | Independent kernel acceptance and correspondence to Python/backend execution |
| 22 finite numerical fixtures | Observed eigenpair behavior for the fixed fixture population in the selected interpreter | All-matrix correctness, exact spectral certification and target performance |
| Signed candidate binding | Exact public instruction, source preimage, task revision, candidate bytes and permitted output | Proof of the complete task and permission to publish or complete it |
| Retained local timing | The reviewed candidate failed its consistent-speed gate | A faster recipe and qualification in the actual target environment |

The public `largest-eigenval` contract defines dominant by largest magnitude,
allows nonsymmetric real float64 matrices and complex eigenpairs, bounds sizes
to 1–10, and requires consistently faster execution than the reference while
meeting the residual check. Current selection uses an explicit
`dominant-eigenpair-f64-small@1` profile and exact signed source/specification
bindings. It does not infer eligibility from the task name.

The reviewed public instruction SHA256 is
`6fa938389f99648508e5a9772e0d5e95d1102738c68d1ff61bd65469d76c52ab`;
the original `eigen.py` preimage SHA256 is
`bbd5be97fb11e9ed46fd5f1fb9d407efb19b265b549d3c36e2cfba923efe3d9d`.
`prepare_spectral_kernel_candidate` joins these to the signed single task and
permitted `eigen.py` modification, and rechecks source and task revision before
handoff. The current [Doctor dispatch](../../benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py)
retains `spectral_performance_unqualified` and the residual model route. A
numerically qualified candidate is retained as a nested observation; it is not
promoted to benchmark execution.

## The proof boundary that must remain explicit

`prove_spectral_adapter_lemmas` checks two packed complex-vector sign relations,
one nonzero relation under a normalized-norm assumption, and ten bounded maximum
relations. Its assumptions state that the backend supplies a complete spectrum,
packed columns satisfy ideal backend eigenvector equations, and selector inputs
are finite nonnegative computed magnitudes. The normalization check assumes the
norm relation; it does not prove a floating-point normalization algorithm.

The report stores each SMT problem's hash and length plus the solver result and
version. It does not retain the canonical problem body, a kernel-checkable proof
object or independent kernel replay. Attaching a candidate-source digest to that
report associates the lemma with a recipe; it does not establish a semantic
lowering from the Python program to the proposition.

The target probe separately records the candidate and probe digests, selected
and resolved interpreter, interpreter executable digest, actual dependency
version strings and finite fixture count. It does not retain complete
NumPy/SciPy/LAPACK binary custody, a dependency environment lock, native process
transcripts or a sealed proof chain. NumPy and SciPy may share numerical backend
dependencies, so agreement remains numerical cross-checking.

The spectral workflow currently produces none of the complete accepted-proof
inputs: `DoctorReviewedTheorem`, exact lowering, full `DoctorProofCacheKey`,
independent pinned kernel receipt, authoritative sealed proof reference or
restart replay. The candidate schema's legacy `proof_receipt_id` can identify
the content-addressed finite/conditional check binding. The
[candidate materializer](../../ipfs_accelerate_py/agent_supervisor/runtime/doctor_contract_candidate_runner.py)
does not reinterpret that field as generic proof authority.

## Stage 1: retain qualification for planning and retrieval

Define a bounded, closed, versioned numerical qualification contract before
adding any accepted-cache bridge. Its schema and implementation are planned;
they do not exist merely because this guide names the required fields.

The record should retain:

- Signed task, revision, specification, source scope and manifest identities;
  original instruction/source bytes or immutable references; exact candidate
  bytes and the reviewed operator/domain interpretation.
- Every canonical SMT problem preimage, declared assumptions, expected check
  name, actual solver/version and executable/library identities, requested
  timeout/budget, result and available proof/counterexample artifacts.
- Exact target probe recipe and fixture population, selected/resolved
  interpreter, executable and numerical dependency identities, actual bounded
  process observations, result count and per-case outcomes. Requested limits
  remain declarations unless enforcement is separately evidenced.
- Distinct conditional-algebra, finite-numerical and performance statuses;
  explicit unresolved Python/backend/floating-point/convergence/representation
  conditions; original evidence provenance and immutable retained references.

Existing canonical proof contracts reject floats. Preserve numerical observations
with a reviewed exact encoding, such as dtype/shape plus original IEEE bytes, or
explicitly specified decimal text and units. Do not silently round a matrix,
tolerance, timing or result to fit the serialization boundary.

Use the existing caller-admitted owner interfaces when retaining records. A new
typed record may be sealed in an explicitly supplied native owner after strict
schema review; a seal proves retained custody, not mathematical correctness or
cryptographic proof attestation. Do not open an owner from a receipt locator,
scan for a likely database, or reconstruct lost evidence from a digest alone.
Observational DuckDB/Quack and DuckLake inventories can index these immutable
references while keeping them separate from accepted Doctor proof-cache entries.

A semantic capsule can describe the capability, domain, evidence levels and
remaining obligations. Its initial purpose is nomination and planning. Keep
required facts inline until a bounded expansion interface is qualified. An
embedding match, `UNSAT` summary, finite test pass or HMAC seal cannot authorize
`KERNEL_VERIFIED`, task completion, execution, publication or source replacement.

Stage completion requires strict round-trip identity, complete problem/fixture
populations, current-source replay and explicit refusal of missing evidence. It
does not require running a benchmark or promoting the candidate.

## Stage 2: review a theorem, lowering and source adapter

Choose an exact conditional theorem goal, independently of a task name or
desired benchmark verdict. A first theorem can concern ideal packed-vector
reconstruction or bounded maximum selection. Review its statement, premises,
assumptions and consequence, then retain a native lowering that preserves every
semantic anchor.

Review the source-to-formal correspondence separately. It must bind the exact
candidate bytes and relevant selection/reconstruction operations to the theorem,
retain input-domain and numerical conditions, and reject unsupported source or
backend recipes. Source matching alone is not a proof of correspondence.
Converting the existing checks to Lean or another proof language does not prove
LAPACK or IEEE behavior.

Assign stable obligation identities only to actual typed obligations. Retain
original descriptions for frontiers that have not been formalized. If the
correspondence remains an explicit reviewed assumption, preserve that assumption
in the theorem and downstream scope; never report it as discharged by the
algebraic proof.

Stage completion requires an exact reviewed statement/lowering and independently
replayed source correspondence. Finite numerical and speed observations remain
separate, with their original limits.

## Stage 3: execute the independent kernel and retain complete custody

Use actual admitted solver/kernel commands and retain the resulting proof object
or kernel-checkable certificate, exact canonical requests/results, transcript
identities and concrete executable/toolchain/dependency pins. Solver success
must not stand in for a separately executed kernel. Retain actual results;
do not fabricate upstream reconstruction records or accepted assurance flags.

Construct the complete native key from independently current inputs: signed
forest/tree/overlay and candidate scope, AST/import graph, corpus, exact goal,
premises, translation, solver, kernel, toolchain, registry, policy, requested
budget, sandbox and environment. Diagnostic placeholders or version strings
alone cannot supply those profiles. Retain profile canonical preimages and
implementation identities with the proof preparation.

Seal the actual native execution, kernel result and final authoritative proof
in the same explicitly admitted owner. Retain its final typed references and
complete key preparation at proof creation. For restart, reopen only the
original admitted owner and private seal key, independently rebuild current
inputs, reload the full retained lineage and rerun the actual pinned kernel.
Missing records remain unavailable; they cannot be reconstructed from today's
defaults. Do not export seal keys or private proving data.

Stage completion requires successful actual kernel replay and exact
source/profile/key/owner bindings. Kernel assurance establishes only the reviewed
conditional theorem. All-matrix accuracy and speed remain open unless separately
proved or qualified by the appropriate evidence.

## Stage 4: qualify a spectral generic receipt bridge

Implement a separate spectral producer bridge after Stage 3. Reuse the canonical
contracts and qualified custody principles, not `HeaderProofCacheAdmission` or
the header theorem's kernel namespace. The bridge must review any correspondence
between its actual native kernel label and pinned verifier identity.

Construct a `ScopedProofReceipt` only after fresh native replay. Bind the exact
theorem, source premise, full key, original sealed proof/kernel references,
requested budget and independently derived kernel evidence. Preserve typed
assumptions, local proof scope and actual residual obligation IDs, plus visible
unresolved frontier descriptions. Deterministic receipt identity must survive
repeated replay of the same original evidence.

The existing `proof-receipt@2` boundary preserves local kernel assurance while
denying task completion. Scope-blind claim, policy, planning and completion
projections cannot discard its conditions. Do not infer `ATTESTED` from a local
HMAC seal. Admit to an existing Doctor gate only through exact independently
expected key/receipt joins, current source/custody checks and native denial-state
validation. Planned spectral support is not available through the current
header-specific bridge.

## Stage 5: qualify a producer-aware Terminal binding

The [current Terminal typed admission interface](terminal_header_proof_admissions.md)
accepts exact header admission handles. A spectral record cannot satisfy that
interface by duck typing, subclass substitution, a boolean currentness callback
or reuse of a header profile. Review a versioned producer-agnostic interface, or
an explicitly typed spectral lane, before integrating numerical proofs.

The future lane must retain each producer's actual bound guard, exact signed
analysis and owner objects, independently expected full key/receipt population,
current deterministic receipt and denial-state checks. Native lookup/render
results must match the expected receipt identity; agreement between two
substituted results is insufficient. Missing or partial bindings refuse.

Carry the original handles and owner identities across both Terminal observation
passes. Use private lifecycle control and a strong original nested-report
reference. Drift, revocation, caller errors, report replacement/tampering and
cleanup failures must clear escaped accepted counts/currentness and restore
closed authority denials. Same-handle recursive operations refuse. Sequential
checks remain observations rather than an atomic owner lease.

Expose bounded counts and population digests, keeping raw paths, theorem bodies,
private references and seal keys out of worker input. Disclose direct adapter
calls separately from internal source hashing, SQL, lineage and stage checks;
these counters are not token usage. Preserve current defaults and require the
full typed producer binding before counting its scoped receipts.

## Stage 6: measure actual reuse and task benefit

Admit an actual current task proof inventory before reporting numerical accepted
coverage. Measure current receipt population, lookup latency, retained lineage
cost and repeated-status material with its complete required-fact closure.
Receipt byte bounds do not bound lineage-prefix work or arbitrary executable
filesystem effects.

Qualify bounded source/receipt expansion before replacing context. Preserve
property, verdict, assurance, assumptions, source/dependency identities, residual
obligations and unresolved frontiers. Misses, solver-only checks, finite tests,
unsupported work and accepted kernel proofs retain distinct meanings.

The optimization task still needs a faster recipe and the actual target's
numerical and consistent-speed validation. Account for imports, compilation and
setup as well as per-call behavior. Plan an official benchmark only after those
requirements qualify through their native owners; algebraic proof admission
does not override a failed performance gate.

Finally, census the complete coding input and native lookup cost, then compare
matched runs using total native planning/coding/lookup/repair tokens and reward.
Track failed and successful attempts, toolchain/environment identities and
whether required evidence remained accessible. Smaller capsules or fewer model
calls alone do not establish total-token savings or improved task reward.

## Planned verification and refusal controls

These controls are required qualification work, not tests executed by this plan:

| Boundary | Required controls |
| --- | --- |
| Qualification record | Missing/duplicate/reordered problem or fixture populations; changed source/problem/result bytes; unknown/sat/timeouts; wrong target/dependency pins; exact numeric encoding and bounds |
| Theorem and source adapter | Wrong theorem or premise; changed assumptions; altered candidate/selector/reconstruction; unsupported domain/backend; undeclared source; independently replayed correspondence |
| Native proof custody | Solver-only or authored transcripts cannot qualify real kernel execution; forged refs/seals; wrong/missing profiles; changed executable/toolchain/budget; source drift; owner/key-file replacement; cold restart and fresh actual kernel |
| Generic admission | Internally consistent wrong receipt; wrong full key; stripped scope/assumptions/residuals; incomplete canonical body; duplicate evidence; no task-completion or attestation promotion |
| Terminal lifecycle | Wrong producer/handle/analysis/owner; incomplete expected population; scoped raw-owner fallback; later-key revocation; recursive callbacks; entry/exit substitution; escaped-report tampering and cleanup errors |
| Measurement | No accepted population inferred from observational checks; complete lookup/internal cost accounting; required-fact expansion availability; target numerical/speed gate; matched total tokens and reward |

Keep authored/mock controls, real solver/kernel checks, finite target probes and
live benchmark observations in separate evidence records. Required sealing and
source inventories remain enabled. Retain failed attempts and original owner
custody; never clear quarantine or replace an owner to conceal receipt identity
changes. Version profile/key migrations explicitly when a reviewed emitter,
adapter, solver or dependency changes.

## Existing IR and model inventories

LegalIR, SecurityIR, CodebaseIR and IntentIR remain separate intermediate
representations. Their parallel 8D, 384D and 768D paths keep their own
family/schema/version/decoder-task inventories, databases, model repositories
and checkpoint identities. Reuse the original embeddings, weights and evidence
references as retrieval assets. A proof capability record is not a new decoder
checkpoint, and learned reconstruction quality is not formal acceptance.

Future retrieval experiments can nominate the reviewed numerical capability
through those existing indexes. Exact current contracts and evidence still
decide eligibility. Numerical proof readiness should be completed and measured
without adding training to this next phase.
