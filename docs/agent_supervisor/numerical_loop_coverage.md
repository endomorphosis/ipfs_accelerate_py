# Numerical loop coverage and supervisor retrieval

The explicitly selected `python-numerical-control-structure@3` profile represents
`for`, `while`, statement `if`, nested loops, `break`, `continue`, `else`, returns
and exceptional exits in the shared ProgramIR control-flow graph. On the
admitted public `largest-eigenval` files, complete **function structure** coverage
increases from three to four functions. All **105 obligations remain pending**.
The `eval.py` module still has explicit top-level assignment and main-guard
fences; this is not complete module translation or verified Python execution.

Select it through the existing benchmark argument:

```text
--numerical-formalization-profile python-numerical-control-structure@3
```

The shared implementation is
[numerical_source_v3.py](https://github.com/endomorphosis/ipfs_datasets_py/blob/6cfc5a640a4715dfc4da1a8f5d180719b4bc6fbe/ipfs_datasets_py/logic/software_verification/numerical_source_v3.py)
and
[numerical_logic_v3.py](https://github.com/endomorphosis/ipfs_datasets_py/blob/6cfc5a640a4715dfc4da1a8f5d180719b4bc6fbe/ipfs_datasets_py/logic/software_verification/numerical_logic_v3.py).
The supervisor's
[numerical context owner](../../ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py)
and [archive policy](../../benchmarks/agent_supervisor/container_coding/terminal_numerical_policy.py)
select and pin this implementation. Preparation, planner context, DuckLake
hydration, publication successors and worker retrieval use the same profile.
The historical v1 and v2 public reports, planning-summary fixtures, core grammar
files and structured-checkpoint implementation pins remain unchanged. Shared
retrieval/link and consumer implementation hashes change to admit v3 explicitly.
No checkpoint, training target or learned-decoder claim is changed by this work.

## Control and proof scope

A `for` activation evaluates its iterable and creates an iterator once. State
slots carry the iterator and yielded item between phases, preventing an operand
reference from reevaluating either producer. Each successful advance precedes
target binding and the body. Only advance-time `StopIteration` denotes exhaustion;
other exceptions, including initialization or body exceptions, escape. An inner
loop's `else` can transfer to an enclosing active loop. `while` and `if` edge
conditions inspect the evaluated test; their representation does not request
another test evaluation. Sixteen authored CPython examples and independent CFG
walks checked these distinctions against Python's
[loop statements](https://docs.python.org/3.12/reference/compound_stmts.html#the-for-statement)
and [break statement](https://docs.python.org/3.12/reference/simple_stmts.html#the-break-statement).
Benchmark source was parsed as data and never executed by these checks.

Every obligation retains its exact source, function, CFG, command and symbolic
activation/iteration context. Invariant initialization, preservation, exit
reasoning and termination have distinct proof phases and relevant entry,
backedge or exit references. Their source-expression anchors do not establish
phase applicability. First-order rows and extracted typed-expression metadata,
SMT goals where supported, and Lean bindings preserve that distinction. All
reachability, protocol, invariant, termination and source-semantic authority
flags remain false. CFG connectivity supplies no reviewed transition-system,
TLA+ or temporal interpretation; those families remain unsupported here.

The existing selected-eigenpair matcher receives only a reviewed temporary
straight-line view with a unique normal/return path and declared exceptional
exits. The report retains the full native CFG. Two compound contracts and four
local assertion requirements remain pending. Loop-bearing or guarded functions
are not admitted to that straight-line compound matcher.

## Qualification results

| Public function | v2 pending | v3 pending | v3 function structure |
| --- | ---: | ---: | --- |
| `find_dominant_eigenvalue_and_eigenvector` | 16 | 16 | Complete |
| `ref_solution` | 16 | 16 | Complete |
| `test_eigen_pair` | 23 | 23 | Complete |
| `test_speedup` | 20 | 50 | Complete, including both loops |

The two files contain 58 CFG blocks, 88 edges, two loop regions and eight loop
proof-phase obligations. Actual `lake build` accepted both public-source
artifacts and an authored control example; it rejected a deliberately false
theorem. This validates generated declaration syntax and types, not runtime
numerical correctness or discharge of source obligations.

A network-disabled Docker fixture exercised the production owner launcher,
UID1000/UID1001 separation, `llm_router` implementation runner and numerical
broker. Planning, provider selection and generation used authored hooks. Real
DuckLake preparation verified three catalogs, three links and one world record.
The worker retrieved all 105 complete rows exactly in 56 initial pages totaling
377,548 UTF-8 response bytes. The largest envelope was 8,180 bytes against the
8,192-byte limit. Full-session accounting recorded 64 requests and 385,404 response
bytes, including seven refusals and a successful restored-source recovery.
Byte counts exclude IPC frame prefixes and are not provider token measurements.

Two initial fixture attempts exhausted the existing 64-request session quota
while repeating filter reads after full retrieval. The completed fixture keeps
the full scan and all peer, cursor and source-staleness checks within that quota;
successful filter combinations are covered by the separate regression suite.
Production quotas remain unchanged. Five private owner paths were denied to the
worker; the descriptor, socket and temporary container were removed. These are
bounded custody checks with a writable container base and root setup for the
existing sudo transition, not a general hostile-worker security proof.

**1,096 distinct regression tests passed**: 426 shared-library and 670 accelerator
tests. All six selected native-Lake tests executed. Two learned-checkpoint
integration tests were skipped because their explicit checkpoint and cached GTE
paths were not supplied. Independent reviews checked source/CFG bindings,
projection scope, historical replay and public evidence. See the
[sealed evidence](evidence/numerical-loop-20261013/README.md).

Planning summaries stay within 8 KiB and sort loop proof phases before ordinary
pending rows when space permits. In this public fixture the final summary is
7,614 bytes: it retains both source/CFG/loop inventories and three assertions,
but omits all 105 inline pending rows, all four function rows, all four unit
links, both compound summaries and one assertion. Exact counters and references
remain; full obligation scopes require retrieval. Exhaustive retrieval now costs
more bytes because it includes loop structure and proof scope. A live token
ablation must use selective retrieval and charge that transport cost before
claiming savings. No live model, learned Source384/IntentIR inference, native
supervisor START, official verifier, checkpoint upload or new token score ran in
this qualification. The [earlier native comparison](context_projection_pair.md)
remains the latest performance evidence.

Further work includes reviewed operational semantics and refinement witnesses,
loop invariant/progress synthesis, module initialization, complex loop targets,
exception handlers and asynchronous control. Learned composition and proof
search remain separate from this deterministic grammar extension.
