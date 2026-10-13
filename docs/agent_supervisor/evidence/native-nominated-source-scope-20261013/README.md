# Native nominated source-scope qualification, 2026-10-13

The [native handoff guide](../../current_source_task_admission.md#bind-a-pinned-historical-nomination)
joins an independently replayed historical nomination to the existing live task
owner and current-source reader. It preserves the legacy native wire schema and
complete signed Terminal input inventory.

## Execution record

The final qualification collected and executed **168 distinct passing tests**
across six selected suites, including **22 new cases**, with zero failures, errors
or skips. All 194 listed source bindings stayed identical before and after the
final execution, and datasets files matched their immutable Git objects.

The [qualification commands](qualification-02/commands.json),
[raw stdout](qualification-02/tests.stdout.log),
[stderr](qualification-02/tests.stderr.log) and
[JUnit identities](qualification-02/junit.xml) record one final pytest invocation,
without a separate collection. Two qualification invocations were made in total.
The [initial run](qualification-01/commands.json) had 167 passes and one fixture
failure: it constructed an inventory limit above its hard ceiling before calling
the handoff. The [fixture correction](fixture-correction.json) records two test-line
changes to use a valid different limit; runtime code did not change. All initial
raw logs, JUnit and source snapshots remain retained unchanged. The [runner](qualification_runner.py) binds
arguments, environment, listed sources, timing and raw output hashes. Earlier
results are not reused as this execution's result. The three changed runtime/test
files have [exact source snapshots](qualification-02/source-snapshot).

The source base is accelerator
`f7cf87104e079d1159226c137449e29bf98bfd40`, plus the frozen admission change,
new tests and optional test-fixture population parameter. The actual datasets
execution revision is `7e60814a93fb81604092a0e0627fbc1996e644b7`, retaining the
parallel LegalIR temporal-attention publication. The preserved kit revision is
`8ac00bb1974b1378e12091d546d8425af34049e0`. The runner records 194 listed source
bindings: 143 accelerator and 51 datasets files, with datasets Git-object checks.
This is an explicit dependency inventory rather than a complete import closure
or toolchain attestation.

The suites cover the new handoff, existing native task guard, native worktree
allocation, source-worker channel, historical nomination and current-source
lookup. The new cases check independent pins and exact value type, self-consistent
forged metadata versus actual CAS replay, real grant revocation/lease release/task
body drift, selected and unselected source changes, corrupted unselected retained
source, callback mutation/reentry, closing-fence source edits, detached exports
and postpublication failures retaining bind-once state. The positive case reads
selected original bytes through the existing owner broker and refuses an
unselected path.

The native coordinator, claimed task, owner grants, DuckDB, Git worktree and CAS
are real. The Quack listener lifecycle and host deployment metadata remain the
existing authored fixture seam. Owner-side reads and same-UID process tests do
not qualify a separately authenticated coding child, full Terminal activation or
managed allocation ownership for the new handoff.

## Documentation and static review

The [documentation runner](documentation_runner.py) inspects retained qualification
and source custody without invoking pytest. It extracts and executes the four
exact offline blocks from the pinned documentation workflow and checks relative
links/anchors in the three changed guides and this README. The
[receipt](documentation-checks.json) and [raw log](documentation-workflow.log)
record outcomes. The documentation runner has its own binding outside the 194
listed source inventory. Prior phase evidence remains unchanged.

The root-authored [review observations](review-observations.json) preserve
collaboration feedback, source hashes, prequalification fixture/API corrections
and review limits. No product imports or tests were performed by those reviewers.
No separate reviewer-authored execution receipt is claimed. The parallel
repository planning/compiler stack remains unchanged; the fresh matching remote
branch query and same-named policy/schema collision remain separate integration
work. A missing matching ref is not evidence of commit absence under every ref.

## Remaining native and measurement work

Required and program roles are explicitly reviewed host policy here. The new
observation joins historical scope/frontier identities to the actual native task,
reader and context. It reports native task/binding verification, while allocation,
compiled context, coding process, proof, execution, completion and omission flags
remain false. The original historical nomination retains its own false current
admission flags. No detached observation is a serialized capability.

Full Terminal subset selection requires coordinated signed declarative selection,
preparation, partition/vector qualification, loaders, task-data deferral and
rebind changes. Preserve complete signed `input_paths` and mandatory supports.
Do not embed the postcommit source-inventory CID in its own indexed task profile;
join that inventory/nomination to the signed profile hash in a later checked
receipt. Native runtime bootstrap, actual child admission and allocation guards
remain distinct. The broker's 32-path ceiling remains narrower than the
nomination's 64-file/1,000,000-original-byte ceiling.

The handoff cold-replays the complete historical inventory, and observations
also replay the retained source and check the current committed population.
Bounded selection does not make these host costs lazy. All currentness checks are
sequential and cannot lock out arbitrary filesystem/task writers.

No model, provider, solver, Docker container, target-code execution or official
verifier is invoked by this qualification. Decoder checkpoints, embeddings,
model-manager entries, IR/dimension/schema/task inventories and foreign worktrees
are preserved. No learned reconstruction improvement, accepted proof, benchmark
success, token savings or billing reduction is established by this increment.

## Parallel live measurement retained separately

The incoming [live baseline report](https://github.com/endomorphosis/ipfs_accelerate_py/blob/4ea2271709bf697f175a198d3df02ac1e1a38205/docs/agent_supervisor/live_indexed_baseline.md)
is another session's evidence, with its own runtime pins, raw artifacts and
limits. It reports 113,883 native Codex tokens, 358,492 no-index supervisor tokens
and 333,973 indexed supervisor tokens, with one official pass in each arm.
Indexed was 6.84% below no-index but 2.93 times native. Source views were disabled,
so those runs do not measure this handoff or whole-codebase minification.
This increment's zero model/provider/Docker invocation statement applies only
to its own qualification; it does not relabel the incoming live experiments.
