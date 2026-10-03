# Source384 header grammar: diagnosis and next slice

This is a development plan based on retained public input and existing code,
not a new qualification. No training, inference, solver, target-code execution,
or hidden benchmark inspection was performed for this review. `D` below means
the `ipfs_datasets_py` repository; owner paths and line numbers identify the
reviewed implementation as of 2026-10-03.

## What the Bottle canary established

The retained [inference report](../../../docs/agent_supervisor/evidence/terminal-source384-context-20261003/canaries/bottle-canary-04/inference.json)
inventoried 358 functions: 128 selected, 166 deferred by selection budget, and
64 with unsupported normalization. Of the selected units, one exceeded the
GTE token limit and 127 produced unverified model candidates. All 127 source
qualifications stopped with empty `checks`, before comparing the prediction
against source:

| First source-guard refusal | Count |
| --- | ---: |
| `two_distinct_parameters_required` | 52 |
| `function_signature_unsupported` | 49 |
| `explicit_integer_parameter_annotations_required` | 24 |
| `ast_resource_bound` | 2 |

These are first-failure counts, not a complete inventory of unsupported
constructs. Removing the first guard would expose further unsupported syntax.
The current guard accepts exactly two explicitly `int`-annotated parameters
and one binary arithmetic/comparison expression, returned directly or through
one fresh temporary. Calls, string operations, mutation and branches are outside
that contract. See D
`logic/formalization/autoencoder/security/source_program_binding_384.py:61,122,256`
(under `ipfs_datasets_py/`). The v2 owner at
`logic/formalization/autoencoder/security/source_program_binding_384_v2.py:162`
extends effect completeness only after the v1 qualification succeeds.

There is a separate model representability problem. The pinned checkpoint
`2ca38dfcc05536315fc3e2c0647b710b930ef4066b474061a7b4e5bfb9a258c5`
has a fixed binary `program_expression` template. Its four scalar heads choose
two first-operand references, one of nine arithmetic/comparison operators,
and `integer`/`boolean`; the second operand is always `expr:threshold`. It
cannot emit a string-normalization program. The fixed-tree/class restriction
is enforced by D `logic/formalization/autoencoder/structured_source_384.py:74,154`.
For example, `_hkey` predicted `position >= threshold` with integer result,
and `_hval` predicted `length + threshold`. Neither represents the source.
Thus zero source-qualified candidates is not a measured 0% reconstruction
score over 127 supported examples, and a parser-only change cannot fix it.

## Smallest useful scope and existing owners

Start with the public module's two top-level normalizers:
[`_hkey` and `_hval`](../../../docs/agent_supervisor/evidence/terminal-source384-context-20261003/canaries/public-bottle.py#L1560).
They convert through `touni` (line 124), then return either the converted string
or `title().replace('_', '-')`. Full response storage, list mutation, cookies,
encoding and dynamic dispatch at lines 1707–1749 remain outside this first
fragment. The public file SHA256 is
`761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba`.

Reuse these D owners rather than introducing another header analyzer:

| Owner under `ipfs_datasets_py/` | Existing responsibility |
| --- | --- |
| `logic/formalization/autoencoder/security/security_formula_grammar.py:18,70,219` | Bounded string/name/call/method/assignment/return/guard productions; composition from predicted productions without correcting them from source labels. |
| `logic/security_ir/doctor_header_contracts.py:65,154,186` | Explicit reviewed WSGI role premise, recognized conversion binding, and narrow normalizer/guard AST shapes. |
| `logic/security_ir/code_header_derivation.py:206,265,308` | Exact source-bound SecurityIR and SMT models, independent candidate AST comparison, and optional real solver checks. |
| `logic/formalization/autoencoder/security/security_formalization_pipeline.py:29` | Existing separation of learned production attribution, deterministic native lowering and caller-supplied specification. |

The existing exact public-input control is D
`tests/unit/logic/security_ir/test_code_header_derivation.py:196`: two modeled
helpers, 12 deterministic formulas, six SMT obligations, zero learned formulas.
This reviewed model begins **after successful conversion**. Conversion exceptions,
unmodified builtins/helper bindings, Unicode normalization/control preservation,
and receiver/callback identity remain explicit premises or open frontiers.
It is not a proof of arbitrary Python string behavior.

## Staged implementation and acceptance gates

1. **Add a versioned, opt-in header candidate profile in D.** Keep the existing
   scalar profile and checkpoint unchanged. Pass a closed, source-hash-bound
   module context and explicit `WsgiHeaderProtocolContract` to the shared
   qualifier: isolated function text alone cannot establish `touni`, store,
   property or callback bindings. Reuse native source-unit byte maps and full
   source replay; pin the protocol, grammar, producer and dependency identities
   in inference/replay receipts. Missing context, changed helpers or unsupported
   shapes must abstain. Publish deterministic header derivations separately;
   the existing weights still have zero representable header candidates.

2. **Fork a compatible 384D grammar head, then measure learning.** Reuse the
   inherited projection and existing compositional production vocabulary.
   Extend the target schema/runtime deliberately; do not relabel the old fixed
   binary checkpoint as header-capable. Source topology, identifier/literal
   bindings and AST teacher labels must have explicit attribution: the predictor
   sees permitted observations, never post-decode gold productions. Preserve
   wrong predictions, and compare independently reconstructed candidates with
   current source before native lowering. Keep legal and existing security
   checkpoints immutable and record parent, corpus, grammar and runtime hashes.

3. **Qualify reconstruction and contracts independently.** Use authored training
   functions with separate alpha/literal-normalized program-shape and repository
   holdouts; keep the public Bottle functions out of training/selection if they
   are reported as holdouts. Report support, selection, token deferrals, exact
   reconstruction, source matching and formal obligations separately over the
   complete inventory. Include model-off, zero/shuffled-weight and deliberately
   wrong-production controls; deterministic targets must never increase learned
   accuracy. Compare throughput with unchanged hardware, budgets and denominators.

4. **Extend existing negative and formal controls before supervisor use.** Reuse
   D `tests/unit/logic/security_ir/test_code_header_derivation.py:141,173,196`.
   Test renamed helpers/arguments, reordered methods, missing or weakened CR/LF/NUL
   guards, shadowed builtins, helper/source/protocol drift, forged authority,
   defaults/decorators, malformed candidates, and bytes/Unicode/exception frontiers.
   For the reviewed local string model, real Z3 must distinguish the original
   accepting behavior from the guarded repair and preserve safe normalization.
   Reuse native SecurityIR projections with a per-family supported/refused matrix;
   do not project unsupported string semantics into unrelated logics. Any Lean
   export must retain typed bindings and named premises, use a genuine `lake build`,
   and record missing capabilities explicitly. Compiling a formula establishes
   syntax/type validity, not source equivalence or a security proof.

The first end-to-end target is two source-bound local header candidates with
independently checked reconstruction and explicit model assumptions. It does
not require all 358 functions to become supported. Supervisor planning may
consume candidates and counterexamples, but admission, repair application,
proof acceptance and task completion retain their existing independent gates.
No coverage increase automatically upgrades proof or execution authority.

## Full original-container result

The subsequent [container qualification](SOURCE384_DOCKER_QUALIFICATION.md)
consumes the same pinned checkpoint over all 218 original files plus two
framework inputs. It completes Source384 in 82.872 seconds under its existing
90-second budget. The full index/context assembly takes 147.267 seconds;
these are separate scopes. The actual saved inference has one model load,
944 inventoried functions, 128 selected units, 127 decoded but unsupported
candidates, one token-limit deferral, 737 selection deferrals and 79 unsupported
normalizations. Container startup, inference publication and source-bound
validation now pass; the source grammar limitation above remains unchanged.

The first header slice should attach the existing deterministic derivation to
a complete captured module and explicit protocol premise. Its acceptance target
is two modeled helpers, 12 deterministic formulas and six SMT obligations, with
zero learned header formulas from the current checkpoint. Cold replay must
reject changed source/helper/protocol bindings and forged authority. Reuse pure
source validation for replay: the complete formalization pipeline rebuilds
through its assembler and can rerun decoding or solvers, so it must remain a
separate, explicitly budgeted operation.
