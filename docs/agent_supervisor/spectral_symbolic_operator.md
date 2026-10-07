# Provider-free dominant eigenpair operator

The supervisor can reuse a reviewed numerical operator to compute a dominant
eigenpair without asking an LLM to generate a solver. Formal checks establish
conditional algebraic properties of its adapter; independent numerical checks
and native validation establish separate, narrower observations. General
planning stays in place.

The public `largest-eigenval` task asks for a **faster implementation**, and
defines dominant as largest **magnitude**. Its matrices are real float64,
square, at most 10 by 10, and may be nonsymmetric with complex eigenpairs.
The original public implementation already computes eigenpairs with NumPy.
Replacing runtime numerical computation alone does not solve the optimization
task. The new candidate has passed finite numerical checks but has **not**
passed its speed gate, so Doctor dispatch does not promote it for benchmark
execution.

| Observation | Current scope |
| --- | --- |
| Numerical operator | Fixed SciPy DGEEV recipe, finite nonempty matrices of sizes 1–10 |
| Conditional formal checks | 13 Z3 checks: packed conjugate reconstruction, normalized nonzero assumption, bounded computed-magnitude maximum |
| Local numerical qualification | 22 authored fixtures; 53 numerical unit tests |
| Explicit target probe | Isolated Python invocation, actual NumPy/SciPy versions, source/probe/executable digests, 22 authored fixtures |
| Native candidate | Signed task, revision and source preimage bound; staged output only |
| Performance | Failed local consistent-speed gate; target benchmark unqualified |
| Model use | No provider calls in these computation/check/candidate paths; general planning is unchanged |

## Numerical and formal responsibilities

[`spectral_eigen_kernel.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/spectral_eigen_kernel.py)
emits a standalone `eigen.py` implementation. It requests right eigenvectors,
selects a maximum computed modulus, and reconstructs the selected complex
vector from the real packed columns. LAPACK specifies the adjacent-column
conjugate convention and convergence status used here; see the
[official DGEEV contract](https://www.netlib.org/lapack/explore-html/d4/d68/group__geev_ga7d8afe93d23c5862e238626905ee145e.html)
and [SciPy wrapper](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.dgeev.html).
DGEEV computes all requested right vectors. This candidate avoids converting
all of them to complex Python outputs; it does not compute only one vector.
Owned copies preserve caller input. Extreme input scaling avoids a defect
observed in the locally installed SciPy/LAPACK combination during testing.
NumPy fallback is explicit and recorded as unoptimized.

The checker independently uses the NumPy full spectrum and requires finite
products, the public raw `np.allclose` residual, a normalized-vector residual
with the equation scaled, complex eigenvalue membership, nonzero vector and
modulus dominance. A residual by itself admits a zero vector and may admit a
nondominant eigenpair. Absolute tolerance alone can also admit a tiny wrong
vector; relative complex membership retains the eigenvalue's sign and phase.
NumPy and SciPy may share LAPACK dependencies, so this comparison is numerical
cross-checking rather than an independent exact spectral certificate.

[`spectral_kernel_proof.py`](../../ipfs_accelerate_py/agent_supervisor/analysis/spectral_kernel_proof.py)
checks ideal algebra under explicit backend equations and a complete-spectrum
assumption. It binds those reports to the candidate source digest. It does not
mechanically verify the Python program, LAPACK, IEEE floating-point accuracy,
all matrices, or timing. An unknown or missing solver leaves residual work.
For arbitrary nonsymmetric matrices, symmetric-only `eigh` and unrestricted
power iteration are insufficient reviewed replacements. For example, a real
rotation can have a pair of equally dominant imaginary eigenvalues.

## Native integration and current admission limit

[`doctor_spectral_contract.py`](../../ipfs_accelerate_py/agent_supervisor/runtime/doctor_spectral_contract.py)
provides `prepare_spectral_kernel_candidate`. It recognizes the reviewed public
instruction and original source digests, requires exactly the signed
`eigen.py` modification, and checks current task/source state again before
handoff. It does not infer eligibility from a task name. A missing target
interpreter or nonaffirmative target/proof report remains residual. The target
probe runs fixed owner-authored cases with `-I`, never evaluator files or
task-selected modules.

The affirmative reports must match the installed probe, resolved executable,
candidate, formal analysis module, and the complete declared lemma population.
The content-addressed candidate uses the existing Doctor contract runner and
allocated native staging path. This workflow never modifies canonical source,
publishes the task or marks it completed. Native validation and publication
retain their existing authority. The generic structural smoke check alone is
not numerical or speed qualification.

The owner-facing `prepare_terminal_doctor_dispatch` API accepts the explicit
profile `dominant-eigenpair-f64-small@1` and `target_interpreter`. It records a
numerically qualified candidate as a nested observation but returns
`spectral_performance_unqualified` with a residual implementation route.
There is no automatic task-name nomination or new benchmark CLI default. The
existing general planner, `llm_router`, and fallback policy continue to operate.
Intent requirement contract version 5 already describes finite schedules and
is not reused for spectral semantics.

Local timing retained both 31-case authored attempts. In the final attempt,
only two 1 by 1 cases were faster; sizes 2–10 took approximately 1.018–1.597
times the public NumPy reference. The complete candidate function, including
input checks, copies and reconstruction, was timed. Setup was recorded
separately. These local Python 3.12 results do not qualify the public target's
Python 3.13/NumPy 2.3 environment, which initially lacks SciPy. No official
benchmark was launched for this candidate and no token-savings claim follows.

## Improvement plan integration

Add reviewed mathematical operators to the existing autoformalization plan as
reusable typed capabilities. Keep these records in the native DuckDB/Quack
control plane and associate immutable evidence references with DuckLake
metadata where appropriate:

1. An operator contract records input domain, output interpretation and
   dependency requirements. Here the key distinction is maximum modulus and
   general real matrices, including complex outputs.
2. A source/task-bound evidence record stores instruction and preimage hashes,
   task revision, candidate digest, actual dependency versions, conditional
   proof scope, numerical qualification and performance qualification as
   separate fields.
3. A semantic capsule names the operator and its unresolved obligations. An
   embedding or the 8D, 384D and 768D autoencoder indexes can nominate this
   record. A Leanstral hidden-layer embedding could also nominate it after
   retrieval evaluation. Learned proximity must not grant proof or execution
   authority; exact reviewed contracts and current evidence select the route.
4. The general planning session consumes the small capsule and native state.
   Eligible reviewed work uses the deterministic operator. LLM calls handle
   remaining ambiguities or implementation work; token usage is measured over
   the complete retained workflow before any savings conclusion.

The next performance experiment should reduce wrapper overhead in a separately
reviewed recipe, account for compilation/import setup, and qualify the actual
target runtime before an official benchmark run. A closed pure numerical
extension or a bounded specialize-and-fallback recipe is a candidate for that
experiment. Promote it only after independent numerical checks and the task's
consistent speed requirement both pass. Exact or interval spectral
certificates and a mechanically verified adapter are deeper follow-on work;
the present conditional SMT checks do not substitute for them.
