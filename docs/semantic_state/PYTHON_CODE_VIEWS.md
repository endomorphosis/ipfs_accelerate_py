# Bounded reversible Python code views

The prototype provides a model-facing view of simple Python arithmetic
functions and maps explicit expression edits back into exact native source.
Canonical files remain native Python. The controller retains the original
bytes privately and uses the existing isolated-worktree patch validator and
executor for mutations.

This fills the edited-view inversion gap in the compression plan. Metadata
packet restoration and unchanged-source restoration alone do not provide that
capability. The prototype does not supply a general whole-file inverse diff or
a whole-repository source transform, and it is not registered as a model tool
or enabled in production launch selection.

## Implementation and APIs

The nonexecuting
[view engine](../../ipfs_accelerate_py/agent_supervisor/semantic_refactoring/python_code_view.py)
exports:

```python
build_python_code_view(source=..., path=..., repository_id=..., tree_id=..., task_id=...)
restore_python_code_view(view=..., current_source=...)
replace_python_expression(view=..., slot_id=..., before_sha256=...,
                          expression=..., current_source=...)
```

`PythonCodeView` and `NativeCodeViewEdit` are immutable, closed artifacts.
Their constructors derive and compare the entire artifact; caller-supplied
maps, hashes, source spans and authority fields cannot replace the derivation.
`view.model_packet()` is the model-facing projection. `view.to_dict()` includes
the private native source and maps and must not be used as a compact prompt.

The engine uses the pinned datasets `PythonASTExtractor` and `SourceSpan` for
lexical provenance. It refuses frontend modules from another tree. The exact
repository binding remains intact; an explicit content-addressed repository
record ID satisfies the frontend's identifier contract, and both identities
are recorded in the private provenance mapping.

The
[worktree adapter](../../ipfs_accelerate_py/agent_supervisor/runtime/python_code_view_worktree.py)
exports:

```python
capture_worktree_code_view(worktree=..., scope=..., path=..., lease_id=..., fence=...)
read_worktree_code_view(worktree=..., scope=..., view=..., expected_view_id=...,
                       lease_id=..., fence=...)
prepare_code_view_expression(worktree=..., scope=..., view=..., expected_view_id=...,
                             slot_id=..., before_sha256=..., expression=...,
                             lease_id=..., fence=..., visible_sources=...)
apply_code_view_patch(worktree=..., scope=..., prepared=..., expected_view_id=...,
                     lease_id=..., fence=..., visible_sources=...)
```

Every operation requires a current native worktree owner and task/path scope.
Read and prepare verify the exact whole-source preimage, including unchanged
comments and other functions. Apply freshly reconstructs and validates the
same request before invoking the existing native patch executor. The private
patch contains full-file context to include all original lines in the Git
preimage check. Only reconstructed native source is eligible for execution.

`visible_sources` remains caller-provided evidence for the existing native
validator. A private original or reversible compact view does not establish
that the model read native source. Missing visibility or a compact-only view
therefore refuses where the native preimage is required. This prototype does
not change that gate. Any native-source expansion must count in the eventual
whole-session token comparison.

The adapter detects a post-apply mismatch and raises; it does not add an atomic
filesystem transaction or custom rollback. The controller retains ownership of
quarantine, worktree cleanup and ordinary task acceptance. Concurrent mutation
outside the worktree lease contract is not made safe by this representation.

## Supported profile

Source schema: `supervisor-python-code-view@1`.
Edit schema: `supervisor-python-code-view-edit@1`.
Revision: `python-arithmetic-source-view@1`.

Accepted files contain one to sixteen unique top-level functions with ordinary
positional parameters, optional literal docstrings, fresh simple local
assignments and a final value return. Expressions use integer literals,
available parameters or earlier locals, unary plus/minus, addition,
subtraction, multiplication, floor division and modulo. This is a syntax
restriction; it establishes no runtime purity, input type or arithmetic safety.

Native function names and parameter names remain unchanged when edits are
reconstructed. The compact view aliases parameters and locals and retains
their original names in its legend. Short aliases replace only
function-scoped NAME tokens. ASCII identifiers are required; Unicode comments
and literals remain intact. Imports, calls, defaults, annotations, decorators,
classes, closures, reflection, control flow, comprehensions, rebinding and new
identifiers are outside the profile. Unsupported input refuses rather than
discarding semantics.

The public packet contains compact source, binding/view identities, per-function
alias legends and concise expression slots. A replacement selects one slot
and the exact original-expression SHA-256, supplies a single-line expression
using available aliases, and may increase its length. Token-level inversion
restores original names. Byte splicing preserves all bytes outside the original
UTF-8 span; an independent AST comparison restricts the change to that slot.
Multiline or ambiguous-trivia slots remain noneditable. The original artifact
retains comments, literal tokens, CRLF and a missing final newline exactly.

| Bound | Limit |
| --- | ---: |
| Native source | 64 KiB |
| Tokens | 8,192 |
| Functions | 16 |
| AST nodes / depth | 4,096 / 40 |
| Replacement expression | 8 KiB |
| Integer literal digits | 1,024 |
| Private artifact bytes / nodes / depth | 2 MiB / 100,000 / 48 |

The worktree adapter additionally requires literal canonical ASCII `.py` paths
and reads through the existing confined, no-follow reader. Resource envelopes
are checked before copying or canonical serialization.

## Cost and qualification

The paired tests compare complete native and compact prompts using the same
task and source, including aliases, hashes and edit instructions. Source-body
bytes are reported separately from complete prompt bytes. A smaller source
body can still yield a larger complete message. No token estimate is inferred
from byte counts, and a single warm-process measurement is not a throughput
benchmark.

The next useful trial requires a compatible tokenizer and model, a registered
expansion/edit broker, native source visibility, unchanged acceptance tests,
and total planning/coding/read/search/edit/diagnostic/retry usage. Keep native
and compact representations in competition using exact complete-message
counts. Extend the grammar only when scope, source maps, inverse edits and
native validation have corresponding coverage.

The retained campaign is at
[supervisor-python-code-view-20261012](../../../../artifacts/supervisor-python-code-view-20261012/README.md).
Its Python tests and patch execution supply no Lean admit. Only an actual
`lake build <Lib>` grants that admit; the Constitution remains unformalized.
