# Reviewed multi-task preparation and native binding

This increment joins a bounded, explicitly reviewed task profile to the existing
symbolic planner, signed admission and native task storage. It does not enable
the reviewed profile's contexts or worker execution.

`terminal-public-task-profile@3` permits 2–16 tasks with exact instruction,
requirement-contract, output-owner, operation, dependency and validation
identities. Version 2 is reserved by a separate active structured-data worktree;
those uncommitted changes were not imported. Frozen version 1 profile, spec and
smoke bytes remain compatible.

## Qualification and provenance

The authoritative summary is [qualification.json](qualification.json). Its
final selection includes the new profile/native controls and existing planning,
completion, inventory, context, dispatch and authenticated retrieval checks.
Baseline and focused runs overlap that selection and must not be added to its
distinct passing total. Exact commands, logs, JUnit outcomes, manual source
bindings and fresh local AST-seal counts are retained beside the summary.

The datasets checkout remains pinned to
`987cf856b2b902aa68c4587bb492b19b932b5d30`. Four previously used embedding or
decoder assets are hashed before and after each qualification run. This is a
control-path qualification with no model training, inference or download, and
does not inventory all models on the host. Provider calls, new weights and new
benchmark scores are not part of this increment.

[independent-review.json](independent-review.json) records the original review
and corrected findings; [publication-independent-review.json](publication-independent-review.json)
records final read-only review and the seven frozen implementation/test pins.
[before-qualification-integration.json](before-qualification-integration.json)
records the documentation-only upstream merge before the broad qualification.
[artifact-manifest.json](artifact-manifest.json) binds the retained files.

## Retained diagnostics

[owner-binding-red.json](owner-binding-red.json) records the real admission,
three-task materialization and Quack replay that accepted a substituted
requirement-contract CID. This ran with intermediate new profile/preparation
code against the original `7dda779` native owner. It was not a full unchanged
baseline with multi-task support. The original owner source snapshot and
diagnostic producer are retained; running that producer against the corrected
owner must refuse instead of recreating the historical success.

[intermediate-diagnostics.json](intermediate-diagnostics.json) retains the
initial native test failures and the broad run's three Doctor-driver fixture
failures. The same three Doctor nodes fail on the unchanged baseline. The
authored routing fixture now explicitly mocks its retrieval-policy boundary;
the production authentication helper is unchanged and separately exercised by
the actual current/stale empty-context tests.

## Qualification limits and remaining work

The native fixture has two ready roots and a third task with both dependency
edges. Meaning/operation inputs are explicitly authored reviewed candidates.
Their admission and storage do not establish semantic reconstruction quality
or checked proofs. Two currentness race controls inject source changes around
the real compiler or native insertion boundary; actual replay and transaction
rollback remain in use.

Fixed smoke selectors parse the selected task's outputs without executing
candidate code, with 1 MB per file and 4 MB per task. Aggregate multi-task
completion and oversized serialized admission remain separate gates.
Initial/context preparation and `AdmittedBenchmarkRuntime.create` refuse this
profile. Final process audits observe only each recorded fixture root or worker
PID set and issue no signals; they are not host-wide absence proofs.

Per-task contexts must retain separate CodebaseIR, SecurityIR, LegalIR and
IntentIR family/schema/version/decoder-task identities and parallel 8D, 384D
and 768D selections. They must bind token/span budgets, DuckDB/DuckLake
inventories, ModelManager identities and immutable Hugging Face/decoder and
embedding assets. Verified predecessor publications, cache reuse, cold restart
and then isolated worktree/lease execution, validation, review and merge
currentness are the next gates in the
[supervisor plan](../../terminal_symbolic_capabilities.md).
