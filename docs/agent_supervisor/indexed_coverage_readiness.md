# Indexed supervisor coverage and readiness

The October 13 follow-up fixes declared-output hash collection and loads the
released Intent384 checkpoint through a narrowly reviewed compatibility adapter.
The numerical source audit still finds **zero verified learned candidates out
of four functions**. These are readiness and coverage results; no new full
supervisor START, Terminal-Bench score or token improvement is established.
The [evidence bundle](evidence/indexed-coverage-20261013/README.md) retains the
qualification boundaries, source hashes and unsuccessful probes.

The earlier [native metadata pair](context_projection_pair.md) lacked final
output hashes. Its observer required a `0600` event log, while the native event
writer can create `0644` or `0640` logs inside the owner's `0700` attempt
directory. Production-writer tests reproduce that mismatch. The removed native
containers prevent retrospective confirmation of the exact failing stage or
recovery of their output bytes.

[`terminal_output_observation.py`](../../benchmarks/agent_supervisor/container_coding/terminal_output_observation.py)
now accepts an owner-owned event log without group or other write permission
after verifying private-directory custody. Binding privacy, regular-file and
single-link checks, bounded reads without symlink traversal, stable file
identity, native settlement joins and worker cleanup requirements remain in
force. Failures expose a fixed `evidence_stage` value;
[`benchmark_comparison.py`](../../benchmarks/agent_supervisor/container_coding/benchmark_comparison.py)
retains only recognized stages on the corresponding unavailable observation.
Historical receipts remain valid, and arbitrary exception text is discarded.

The successful offline Docker exercise used the admitted public
`largest-eigenval` profile, the production binding/event writers and the bounded
observer. It captured distinct candidate and canonical `eigen.py` hashes after
an authored comment edit. Owner state remained inaccessible to the separate
worker UID. Lifecycle records were authored fixtures; no provider or official
verifier ran. The observer, progress and comparison regressions contain **295
distinct passing test cases** after removing overlap between runs. Hashes
describe observed bytes, without establishing their writer, patch publication
or correctness.

The shared [Intent checkpoint loader](https://github.com/endomorphosis/ipfs_datasets_py/blob/afadfbcf8d33a67568c133bb761c43d42a9a133c/ipfs_datasets_py/logic/formalization/autoencoder/intent_action_checkpoint_384.py)
lives in `ipfs_datasets_py`; its
[Intent runtime](https://github.com/endomorphosis/ipfs_datasets_py/blob/afadfbcf8d33a67568c133bb761c43d42a9a133c/ipfs_datasets_py/logic/formalization/autoencoder/intent_action_runtime_384.py)
attaches the compatibility receipt to advice.
Accelerator [asset preparation](../../benchmarks/agent_supervisor/container_coding/terminal_deployment.py)
uses that same loader; its
[Intent advisor](../../ipfs_accelerate_py/agent_supervisor/runtime/intent_384_advisor.py)
continues consuming the shared runtime. The selected checkpoint SHA-256 is
`4f3fd17ea2d908fe36c57a444517f3cc0a983cab26f0f67f2e32371e25def3cd`.

The accepted transition is exactly
`implementation.shared_runtime.numerical.files.tree_pin.py` changing from
`587165942fb06ea9effc836b555e0ac0606c5876b013a374a1e76d25ee069e4d`
to `ec43fd9379f3c9198451e8d62d98db53826dcf115e6858039a73c3da7d72a1c2`:
the reviewed diagnostic output moves from stdout to stderr. All other
implementation pins remain exact. The adapter validates a detached runtime
view through the ordinary structured loader; checkpoint bytes, numerical
weights, target schema and training provenance remain unchanged. It permits no
unrelated decoder transition and performs no training or checkpoint rewrite.

[`terminal_intent_action_readiness.py`](../../benchmarks/agent_supervisor/container_coding/terminal_intent_action_readiness.py)
separates the following observations in the
[benchmark preparation receipt](../../benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py):

| Observation | What it establishes |
| --- | --- |
| Asset selection | Checkpoint and embedding assets match the selected archive binding. |
| Checkpoint eligibility | An explicitly supplied local checkpoint passes the current loader and compatibility checks; archive-only selection leaves this unchecked. |
| Input scope | The complete instruction fits the existing controlled action grammar. |
| Inference | Separately recorded execution of the embedding model and learned decoder. Readiness itself runs neither. |
| Source agreement | A candidate passes the supported source-contract comparison; asset selection alone establishes none. |
| Proof | A separately checked theorem discharges a stated source obligation. Loading, decoding or syntax compilation supplies no such result. |

Real-weight local qualification used the pinned checkpoint and `gte-small`.
An authored controlled-language scalar instruction produced
`semantic_candidate_advice`, with shared status
`source_supported_action_contract`; saved-advice replay agreed. Instrumentation
recorded four embedding calls and four structured inference calls across
preparation and replay. The unchanged 716-byte `largest-eigenval` instruction
returned `fail_open_input_out_of_scope` before either embedding or numerical
inference. The checkpoint's single-line explicit-contract grammar does not
cover that instruction. The qualification preserves the original instruction,
does not promote a default model, and performs neither a formal-logic projection
nor a Lake build. Asset packaging readiness is recorded separately from a built
archive or container deployment.

Two additional tests consumed the real checkpoint through goal preparation and
saved-advice replay with an authored local planner response. The combined
qualification contains **742 distinct passing tests**, after removing shared
transport cases from the per-suite totals. No optional tests remain unexecuted
in that selected set; the planner fixture is not a live provider trial.

The public-source audit explains why loading more weights alone cannot repair
the four numerical candidates. The existing Security384 source guard expects a
small scalar program with two distinct integer parameters. The eigenpair and
timing functions have one parameter. The assertion helper first encounters a
Unicode executable-token restriction, but removing it would still leave its
numerical body outside the scalar contract. The separate unary NumPy selection
head covers a direct `argmax`/`argmin` return, not the composed
`eig`–`abs`–`argmax`–tuple/index program.

Deterministic AST replay does recover useful structure independently of those
learned candidates:

| Public function | Structural status | Pending obligations |
| --- | --- | ---: |
| `find_dominant_eigenvalue_and_eigenvector` | Complete structure | 16 |
| `ref_solution` | Complete structure | 16 |
| `test_eigen_pair` | Four assertions retained; formatted failure message deferred | 13 |
| `test_speedup` | Loop bodies, conditional expressions and formatted output incomplete | 0 |

All four spans match the exact public source bytes. The **45 obligations remain
pending**, and the timing helper's zero count reflects missing coverage. ProgramIR,
first-order predicates, abstract SMT constraints and Lean definitions or
conditional lemmas are available projections. This audit reran neither neural
inference nor `lake build` and discharged no source obligations. Library identity,
floating-point behavior and performance remain outside the established semantics.

The next work is ordered to preserve those distinctions:

1. Add versioned numerical grammar for conditional expressions and formatted
   strings, preserving selected-branch evaluation, conversions, format specs,
   source spans and unknown effects.
2. Represent `for` bodies as conditional, repeated regions with iterator and
   target provenance. Model or explicitly refuse nested loops, exceptions,
   `break`, `continue`, early return and `for`–`else`; do not flatten their calls
   into unconditional effects.
3. Reuse shared 384D training infrastructure for a compositional numerical head
   with authored fixtures and held-out evaluation. Keep actual predictions
   distinct from deterministic targets and benchmark labels.
4. Define reviewed runtime, library, dtype, shape and tolerance witnesses plus
   source-to-model refinement before discharging numerical obligations. Replay
   source bindings and projections, and run Lake for syntax qualification while
   reporting proof coverage separately. TLA+, temporal, deontic, separation and
   concurrency projections require their own source-grounded interpretations.
