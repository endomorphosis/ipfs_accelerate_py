# Largest-eigenval diagnosis and symbolic call reduction

Regular Codex completed normally but received official reward **0.0**. The
official eigenpair and dominant-modulus checks passed; the sole recorded failure
was `test_speedup[9]`: **18.856488 microseconds/call** against a strict bound of
**15.760510 microseconds/call**, about **19.64% slower**. The stdout reports
26 passing and one failing parameterized cases; CTRF groups those cases into
three test families. These are different groupings, not contradictory results.
See the [bounded native diagnosis](native-runtime-diagnosis.json) and the
[original matched-arm observation](../terminal-codex-eigenvalue-baseline-20261005/README.md).

The supplied public evaluator measures sizes 2, 4, 6, 8 and 10. However, Codex
also ran a custom probe including size 9 and reported 9.30 microseconds versus
15.66 for NumPy. Its sampled correctness probe reported 20,153 passing matrices.
Therefore the failure is **unestablished speed robustness**, not evidence of an
untested size 9 or a known mathematical error. The exact cause remains unknown:
matrix populations, timing methodology, process warm-up and numerical backend
behavior have not been isolated. Earlier commentary claiming no size-9 probe
was observed was corrected after the complete tool-outcome inventory.

The retained implementation compiled a C extension at import, resolved NumPy's
LAPACK `DGEEV` symbol and integer ABI, computed eigenvalues and all right
eigenvectors, then selected the largest complex modulus. NumPy also uses LAPACK
`_geev` for general eigendecomposition ([NumPy documentation](https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html)).
The fixed workspace size of 4096 doubles, all-vector computation, ABI handling
and data-copy overhead are review targets; none is established as the cause of
the official timing failure. Generated source was reconstructed from a
successful tool write after the container was deleted and remains private.

## A demonstrated zero-provider planning route

The existing `intent-plan-requirement-contract@2` selects `intent_symbolic`.
The obligation compiler, symbolic candidate planner, critic and formal plan
compiler check declared requirement coverage, effects and ordering before the
normal signed admission and native IntentRepository storage. The contract does
not invoke a planning provider. Version 1 still uses an LLM; version 3 has
header-specific applicability and is unsuitable for this numeric task.

The [qualification producer](qualify_symbolic_planning.py) binds the exact
public instruction hash to **eight agent-authored candidate atoms**: entrypoint,
dominant modulus, input domain, nonsymmetric/complex output, speed requirement,
residual predicate, median timing and retained Python entrypoint. It accounts
for the optional evaluator suggestion separately. It does not claim automatic
semantic extraction or human review. Optional package/language permission is
preserved in the original source and does not become an installation duty.

The [native qualification](symbolic-planning-qualification.json) demonstrates:

- Zero planning-provider calls, model selection disabled, no provider request.
- Eight covered requirement atoms, two goals and one stored coding task.
- Deterministic replay of the selection receipt, verified admission and native
  storage, and unchanged original program bytes.
- No coding execution, numerical proof, official reward or training update.

The source interpretation and operation contract retain false semantic, proof,
execution and completion authority. Each atom maps to an administrative coding
task and the existing structural smoke. That smoke verifies program structure,
not eigenpair behavior or speed. Full source-bearing ledgers, owner keys and
manifests remain in ignored local artifacts. The benchmark is excluded from
training and weight publication. Source and invocation provenance are retained
in [qualification provenance](symbolic-planning-provenance.json).

## What the call saving means

The [historical phase accounting](historical-phase-accounting.json) binds the
successful supervisor trial: its planning invocation used **23,766 tokens** and
74.62 provider seconds; coding used **267,666 tokens**. Removing planning
projects **two router/CLI sessions to one**, avoiding a component equal to
**8.15% of that historical 291,432-token total**. The coding session still uses
the canonical `llm_router`. These session counts are not counts of internal
model turns or API requests. Regular Codex already used one CLI session and
has no separate supervisor planning invocation to eliminate.

This local planning qualification is not a fresh official supervisor trial.
Actual new coding tokens, end-to-end time, Source384 interaction and reward
remain unmeasured. Contract authoring/review has a cost, and an authored contract
is reusable only for compatible, source-bound tasks. The local planning time
cannot be subtracted directly from a container trial with different setup.

| Component | Provider sessions | Current evidence |
| --- | ---: | --- |
| Historical supervisor planning + coding | 2 | Official reward 1.0; 291,432 tokens |
| Native symbolic planning alone | 0 | Covered, replayed, admitted and stored; no coding |
| Symbolic planning + router coding | 1 expected | Fresh official arm not run |
| Reviewed numeric operator + deterministic validation | 0 possible | Numeric operator not implemented or qualified |

## How to reduce further coding calls

1. **Use the existing symbolic contract transport first.** Pass a reviewed,
   source-bound version-2 contract to `full_supervisor_benchmark prepare` through
   `--intent-requirement-contract`. Keep no-provider selection fail-closed,
   preserve resource and immutable input controls, and route the remaining
   coding call through `llm_router`. Qualify a fresh arm separately before
   claiming end-to-end savings or benchmark success.
2. **Add a general numeric operator, not a benchmark answer lookup.** Define a
   typed `DominantEigenpair` operation with explicit admissible shapes, dtype,
   finite values, layout, nonempty size bounds, mutation policy, output type and
   backend provenance. Some are stronger operator preconditions, not literal
   requirements extracted from the task. Inapplicable or unsupported inputs
   must abstain or use a compatible fallback. Bind source hash, template,
   compiler, NumPy/BLAS/LAPACK ABI, resource profile and numerical validation.
3. **Use theorem provers for the parts their semantics cover.** Check shape and
   index bounds, exact argmax selection, complex-conjugate column
   reconstruction, nonzero-vector obligations, file effects and operator
   preconditions. The current Python verification frontend excludes floating
   and complex semantics, and existing integer/rename/header operators cannot
   prove or implement this task. Proof reuse must bind the exact assumptions,
   kernel contract and implementation; a plan coverage certificate is not an
   eigenpair certificate. Symmetric `eigh`/Jacobi and unchecked power iteration
   are not general nonsymmetric substitutes.
4. **Execute independent public numerical gates without an LLM.** Preserve the
   public predicate; check finite/nonzero outputs, input preservation, complex
   eigenpairs, residual and dominant modulus independently, with sizes 1–10,
   structured matrices and supported strides. A residual alone admits a zero
   vector or a nondominant eigenpair. Finite sampling supplies evidence, not a
   universal floating-point theorem. Use paired inputs, randomized timing order,
   warm-up, repeated seeds and timing distributions with explicit backend and
   thread controls; retain both strict wins and margins. Timing must be measured.
5. **Select or reject bounded candidates deterministically.** A reviewed
   numerical kernel or generated template can be retained when its applicability
   and independent gates pass. Otherwise provide only residual obligations to
   one bounded router coding session. Record sessions, internal model turns,
   input/cache/output tokens, abstentions, wall time and proof/numerical scope
   separately. Reusing a historical model-generated candidate may avoid a new
   call but does not erase its original generation cost.

The 8D spaCy, 384D and 768D embeddings and possible Leanstral representations
can nominate operators, retrieve analogous contracts and rank proof reuse.
Their vectors, PCA projections and reconstruction losses do not prove semantic
equivalence or numerical correctness. The first demonstrated planning saving
requires no new embedding training. A numerical operator needs broad
independently reviewed development examples, with this benchmark kept outside
the training population.

The [incoming numeric proof-scope review](incoming-numeric-proof-scope.json)
also checks the concurrently published exact-real ranker source proofs. Those
cover scalar syntax evaluation, dot multiplication and a gradient-update
projection under mathematical assumptions. Binary64 refinement, the actual
objective/gradient implementation and the full training loop remain open;
there is no complex eigendecomposition, DGEEV or timing certificate. This
provides additional scalar proof reuse without closing the numeric operator
gap above.

The [bounded public replay](public-replay.json) records the recovered candidate
under a reconstructed public runtime. Its exact original task image was absent;
the replacement image and limits are recorded. This is a diagnostic finite
replay, not a new official reward or an exact verifier reproduction. No hidden
test/reference source body was read, no reward/input was changed, and no model
provider was invoked by the diagnostic replay.

The single measured replay completed in 2.36 seconds and passed **13,245**
finite correctness checks, with maximum scaled residual about **1.50e-15**.
The candidate was faster in all **350** paired timing-round comparisons across
sizes 1–10. Size-9 seed medians were **9.088–9.232 microseconds** for the
candidate versus **15.281–15.489** for the public NumPy wrapper, with median
ratio **0.596875**. The official slowdown therefore did not reproduce in this
warmed rebuilt runtime. It remains a valid observed failed trial; these new
finite samples do not identify the hidden timing mechanism, prove universal
behavior or revise its reward. Initial Docker reference and import-path
plumbing failures are retained separately; there was one measured replay and
no candidate optimization.
