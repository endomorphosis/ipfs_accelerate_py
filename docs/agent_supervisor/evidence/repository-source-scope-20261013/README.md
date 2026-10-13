# Repository source-scope nomination qualification, 2026-10-13

The [nomination guide](../../repository_source_scope.md) describes an explicit
historical subset of a complete committed-source inventory. This increment adds
one pure adapter and its regression tests; it does not add native task admission,
repository sessions, model training or a new database.

## Executed qualification

The single [qualification invocation](qualification-01/commands.json) collected
and executed **219 distinct tests**, including **56 new cases**, across six
selected suites. All passed, with zero failures, errors or skips. There was one
pytest invocation, no separate collection and no failed attempt in this phase.
The [raw stdout](qualification-01/tests.stdout.log),
[stderr](qualification-01/tests.stderr.log) and
[JUnit identities](qualification-01/junit.xml) are retained. The
[qualification runner](qualification_runner.py) records the exact environment,
arguments, timing, output hashes and before/after source bindings.

The accelerator source base was
`5ad92f61bb643d4138749d3df16663a2af1c79c1`, plus the frozen new module and test.
The actual immutable datasets execution checkout was
`28b9ef1f0b9da51e87b1e68ee565b43681397109`; the kit checkout was
`8ac00bb1974b1378e12091d546d8425af34049e0`.
All **188 listed source bindings** (137 accelerator and 51 datasets) stayed
identical before and after execution. Datasets files also matched immutable Git
objects. This is a listed dependency inventory, not a complete import closure
or toolchain attestation. The two new files have exact
[source snapshots](qualification-01/source-snapshot).

The six suites cover the new nomination API, retained source inventory, code-view
inventory, current-source lookup, worker source views and semantic-context runtime.
Cases include full population/frontier accounting, missing or corrupt CAS frames,
callback mutation, unsupported paths and encodings, symlinks and gitlinks,
exact 64-file and 1,000,000-byte limits, duplicate per-path accounting, mandatory
supports, detached exports and bounded complete-row pages. Unknown or unsupported
selections and overflow refuse the whole nomination.

One fixture selects two paths from 71 committed originals, prepares the existing
semantic context, and checks the existing reader against authored semantic pins.
It verifies the selected scope, rejects an unrelated path and refuses changed
current bytes. It does **not** establish a live semantic-artifact custody handoff,
native task/allocation admission or worker adoption. Historical metadata continues
to report current-source and task-admission verification as false.

## Documentation and review records

The separate [documentation runner](documentation_runner.py) checks retained test
outputs and source custody without rerunning pytest, then extracts and executes
the four exact offline blocks from the pinned documentation workflow. Its
[receipt](documentation-checks.json) and [raw log](documentation-workflow.log)
record actual gate outcomes and relative-link/anchor checks for the three changed
guides and this README. The runner is outside the 188 listed source bindings and
has its own binding. Prior phase evidence is preserved unchanged.

The root-authored [review observations](review-observations.json) preserve
read-only collaboration feedback, its delivery and limits. They are not a
reviewer-authored execution receipt. The
[parallel-work survey](parallel-work-survey.json) binds the existing 21-commit
repository planning stack, compiler dependency, clean foreign worktrees and
current overlap. That stack and the current recipe class use an incompatible
same-named policy module/schema and CLI meaning. The guide gives the reconciliation
order; this increment neither duplicates nor merges those sessions or consumers.
Only the specifically queried matching remote branch names were absent; the
survey does not establish that the commits are absent from GitHub under other refs.

## Evidence limits

Cold loading checks every retained source blob and each page revalidates the
entire in-memory frontier. Pagination bounds output, not total host work.
Direct construction of the immutable value checks metadata self-consistency;
historical provenance requires the factory with independently pinned CAS inputs.
The complete signed task input inventory remains separate from the proposed
bounded semantic/read scope. Current source, task, allocation and proof owners
must independently admit any later handoff.

No model, provider, solver, Docker container, target-code execution or official
task verifier ran for this increment. No decoder weights, embeddings, model
registrations, private runtime databases or foreign worktrees were changed.
No task advantage, billed token reduction, minification savings, accepted proof,
completion or omission authority is established. Native integration and matched
full-session token/task measurements remain subsequent work.
