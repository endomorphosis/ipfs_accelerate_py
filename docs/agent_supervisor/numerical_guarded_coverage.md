# Guarded numerical source coverage

The explicitly selected `python-numerical-guarded-structure@2` profile adds
Python conditional expressions, formatted strings and lazy assertion messages
to the shared datasets grammar. The supervisor stores and retrieves their
source-bound, unresolved obligations through its existing world and metadata
indexes. On the admitted public `largest-eigenval` source, complete structural
coverage increases from **two to three of four functions**. This is deterministic
AST coverage; learned decoder coverage and discharged proofs remain unchanged.

Select it with the existing benchmark argument:

```text
--numerical-formalization-profile python-numerical-guarded-structure@2
```

The historical `python-numerical-structure@1` profile remains available. Its
public-source report bytes and structured-checkpoint implementation pins were
checked against a frozen pre-change baseline. Existing checkpoint weights and
training targets were not modified. The shared unit-link and retrieval helpers
now dispatch by the report's exact profile and schema; their implementation
hashes consequently change.

The implementation remains owned by `ipfs_datasets_py` in
[numerical_source_v2.py](https://github.com/endomorphosis/ipfs_datasets_py/blob/9eb06b73658d8203628a697db6199f14ecf479e4/ipfs_datasets_py/logic/software_verification/numerical_source_v2.py) and
[numerical_logic_v2.py](https://github.com/endomorphosis/ipfs_datasets_py/blob/9eb06b73658d8203628a697db6199f14ecf479e4/ipfs_datasets_py/logic/software_verification/numerical_logic_v2.py). Accelerator's
[numerical context owner](../../ipfs_accelerate_py/agent_supervisor/runtime/numerical_formalization_context.py),
[archive policy](../../benchmarks/agent_supervisor/container_coding/terminal_numerical_policy.py)
and [publication successor owner](../../ipfs_accelerate_py/agent_supervisor/runtime/published_numerical.py)
consume that selected implementation. The selected profile participates in v2
tree identities, record identities and producer bindings. Refresh and retrieval
replay the same profile against exact source bytes.

## Evaluation and logic scope

ProgramIR's ordinary operands have eager evaluation order. A conditional's
eager operand therefore contains only its test; a required guarded-region table
retains the alternative branch roots. Assertion predicates require assertion
enablement, and messages additionally require normal predicate completion and
false truthiness. The table validates ownership, source spans, symbol scope,
reachability and conservative possible reads. Standalone ProgramIR is explicitly
insufficient to interpret this profile.

Formatting retains value evaluation, nested format specification, conversion
and formatting phases in that order. Earlier exceptions prevent later phases.
Seventeen independently authored CPython examples checked these evaluation
orders, including optimized-away assertions. They executed authored examples,
not benchmark source. See Python's
[conditional expressions](https://docs.python.org/3.12/reference/expressions.html#conditional-expressions),
[formatted strings](https://docs.python.org/3.12/reference/lexical_analysis.html#f-strings)
and [assert statement](https://docs.python.org/3.12/reference/simple_stmts.html#the-assert-statement).

Typed first-order projections carry guarded implications. SMT index/shape
goals require an unresolved context interpretation; no guard is asserted as an
assumption. Lean declarations retain unresolved guard and requirement
parameters. Existing selected-eigenpair compound contracts are retained only
where the reviewed straight-line matcher applies without guarded or formatting
constructs. Four local assertion requirements remain pending.

## Public-source qualification

| Public function | Previous pending | New pending | New structural coverage |
| --- | ---: | ---: | --- |
| `find_dominant_eigenvalue_and_eigenvector` | 16 | 16 | Complete |
| `ref_solution` | 16 | 16 | Complete |
| `test_eigen_pair` | 13 | 23 | Complete, including formatted failure message |
| `test_speedup` | 0 | 20 | Partial; loop bodies remain fenced |

All **75 obligations remain pending**. The larger count exposes previously
unrepresented conditions and effects. Actual `lake build` passed for the two
public-source artifacts and an authored guarded example. A deliberately
unsupported `False` theorem failed compilation. These observations establish
generated declaration syntax/type correctness, not Python numerical semantics.

A network-disabled Docker fixture used the production owner launcher, a
separate worker UID, `llm_router` runtime and the obligation broker. Provider
selection and response generation were authored hooks. Real DuckLake hydration
verified three catalogs, three links and one world record. The worker retrieved
all 75 obligations exactly across 21 initial pages; the largest response was
8,006 UTF-8 bytes against an 8,192-byte limit. Those initial responses totaled
147,671 bytes. The whole session additionally exercised filters, recovery and
seven refusal cases; byte counts exclude IPC frame prefixes.

Owner-private reads, unrelated peers, forged/crossed cursors and changed
canonical or worker files were rejected. Cleanup removed the broker descriptor,
socket and fresh container. Bounded planning summaries retain guard references
and omitted-row counts; full contexts remain available through retrieval. A
single row too large for the requested budget is refused, never truncated.

The [sealed evidence](evidence/numerical-guarded-20261013/README.md) includes
commands, source bindings, regression results, native Lake observations and
independent reviews. **947 distinct regression tests passed**: 338 shared-library
and 609 accelerator tests. All six selected optional native-Lake tests ran;
two separate learned-checkpoint integration tests were skipped because their
explicit checkpoint/embedding paths were not supplied. No live model, official Terminal-Bench verifier, training
or checkpoint upload ran in this qualification. It supplies no new token score
or measured token savings; the earlier
[native metadata comparison](context_projection_pair.md) remains the latest
performance evidence.

Further coverage needs explicit loop iteration and control-flow semantics,
including zero iterations, exceptions, `break`/`continue`, invariants and
termination obligations. Learned numerical composition and reviewed refinement
witnesses remain separate gaps. Formalization metadata is useful planning
context only when selected compactly; exhaustive retrieval bytes must be charged
in a future live ablation before claiming a token advantage.
