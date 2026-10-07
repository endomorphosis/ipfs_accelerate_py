# Closed package import-alias repair

**Status:** Current

**Owner:** agent-supervisor maintainers

**Audience:** Developers extending symbolic task coverage

**Sources:** `runtime/doctor_alias_contract.py`, `runtime/doctor_task_workflow.py`,
and `analysis/program_dependency_graph.py`

**Last verified:** 2026-10-07

The Doctor can repair a unique direct call that uses an exported function's
original name after an explicit alias import from a local Python package.
For example, a captured `pkg/worker.py` containing
`from .helpers import transform as normalize` and `return transform(value)`
can become `return normalize(value)`. The signed source population must include
`pkg/helpers.py` and the inert `pkg/__init__.py`.

## Supported source contract

`closed-package-imported-alias-call@1` extends the existing flat-module operator.
Absolute, sibling-relative and parent-relative imports resolve against a
canonical inventory of captured paths. The contract records every module,
initializer and import edge in `closed-python-package-resolution@1`. Construction,
replay and candidate validation reconstruct that inventory from the exact source
bytes; claimed resolution metadata cannot substitute for reconstruction.

Regular packages must include every ancestor initializer, containing only an
optional initial docstring and `pass` statements. Donors must contain functions
and no imports. The existing finite function grammar admits immutable literal
defaults, positional-only and keyword-only arguments, and direct calls with
inert arguments. The repair changes one callee identifier while retaining the
argument expressions and their order. Donor computation is unchanged.

Namespace packages, reexports, import chains, dynamic initializers, wildcard
imports, module-attribute calls, cycles, ambiguous aliases, shadowed parameters,
module/package collisions, special metadata names and standard-library root
collisions remain unsupported. No import is inferred from an undeclared file.
Unsupported populations retain a residual for the existing router route.

Bounds include 256 captured files, eight package directories, 16 bindings per
ordinary module, 32 parameters, one megabyte per source file and four megabytes
for the population. Paths and identifiers use a finite ASCII grammar. Source
reads verify exact hashes and stable single-link regular files; symlinks,
hardlinks and source drift are rejected before candidate publication.

## Evidence and lifecycle

Independent AST replay establishes the finite source environment. Lean and Z3
check alias lookup, the absence of the original binding, argument preservation
and supplied-parameter binding. This assumes ordinary source-based imports with
no external shadowing, substituted bytecode, preloaded replacement modules or
import hooks. The provers do not interpret Python's import machinery or prove
whole-program correctness, donor computation, or equivalence to the unresolved
original call.

The dependency graph records relative and absolute donor dependencies plus
ancestor initializers. A module retains dependencies on its own initializers
even if it has only absolute imports or no imports. Ambiguous or missing package
context, wildcard imports and possible submodule bindings keep open frontiers.
These graph edges describe source dependencies, not runtime value-binding
proofs. Empty Python initializers now receive AST records correctly.

The existing signed admission, source ledger, implementation identity, task
revision, staged candidate, validation and publication gates remain in the
execution path. Package candidates carry the distinct operator identity through
Doctor dispatch and capability reporting. Index rows remain advisory and never
replace fresh source or proof checks.

## Qualification

The regression suite covers flat and packaged aliases, signature binding,
independently false proof claims, forged resolution metadata, stale source,
unsupported populations, dependency impact and initializer drift. Authored
fixtures import repaired packages in fresh Python processes.

The native lifecycle fixture uses signed indexed planning, two stored DuckLake catalog
records and two links, an attached lexical context bundle, real Lean/Z3 checks,
START, staged execution, validation, publication, task completion and STOP with
zero remaining process members. Published context is refreshed. Model entrypoints
are forbidden in both the parent process and candidate worker.

The signed public smoke validation is structural. Separate authored behavior
checks observe the original `NameError` and the repaired results. Intent
interpretation is authored, and the fixture uses lexical retrieval; it does not
qualify learned autoencoder inference or learned embeddings. It establishes no
new Terminal-Bench score or matched model-token saving ratio.

Run the package checks with the matching datasets checkout on `PYTHONPATH`:

```sh
python -m pytest -q test/api/test_doctor_package_alias_contract.py \
  test/api/test_program_dependency_package_imports.py \
  benchmarks/agent_supervisor/container_coding/test_terminal_package_alias_repair.py \
  test/integration/test_native_package_alias_repair.py
```

The native lifecycle requires the installed Quack transport, DuckLake support,
Lean and Z3. Retained source-bound results and environment limitations are in
[evidence/package-alias-20261007/README.md](evidence/package-alias-20261007/README.md).
