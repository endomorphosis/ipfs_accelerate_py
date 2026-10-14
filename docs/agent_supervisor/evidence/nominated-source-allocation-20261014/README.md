# Nominated source allocation custody, 2026-10-14

This increment joins the existing historical source nomination to the actual
native managed worktree. It builds on published accelerator
`2ceac1afc3d66b54204656ee8d7f758780231ff1` in an owned execution worktree.
The [parallel survey](parallel-work-survey.json) retains separate active
capsule-budget and per-turn usage progress. No foreign worktree is changed.

## Behavior

The [allocation binder and observer](../../current_source_worktree_allocation.md#bind-a-pinned-historical-nomination)
reuse the sealed task admission, retained nomination, actual allocation, CAS and
source reader. Selected paths come from independent nomination replay; the
workspace comes from the retained allocation. Legacy and nominated binds share
whole-operation reentry protection. Failed prepublication checks permit retry;
an actually published reader stays consumed and marked after a closing failure.
A second allocation cannot adopt that reader.

`supervisor-native-nominated-source-worktree-allocation@1` adds verified
allocation ownership to the existing task/source observation. The task-only
observation keeps allocation ownership false. The new observer checks current
claim, grant, lifecycle, retained object joins and the complete allocated source
population after native callbacks, including omitted originals. These sequential
checks are not a filesystem writer lease. The [broker](../../source_lookup_worker.md#nominated-scope-and-managed-allocation)
independently checks canonical and allocated originals before its own grant and
around reads.

The nomination's 64-file/1,000,000-byte ceiling includes selected required
supports. The independent broker ceiling remains 32 paths. The new controls
exercise 32, 33 and 64 selected paths: oversized worker populations refuse before
a proxy grant without silently truncating a valid nomination.

## Qualification

The [frozen runner](qualification_runner.py), [eight-suite roster](targets.json)
and [listed dependency inventory](source-inventory.json) retain a new complete
execution. There is no separate collection or reuse of earlier test results.
The [execution record](qualification-01/commands.json), raw
[stdout](qualification-01/tests.stdout.log), [stderr](qualification-01/tests.stderr.log)
and [JUnit](qualification-01/junit.xml) retain original execution evidence and
changed-source snapshots. The single execution passed **215 distinct tests
across eight suites**, including 47 newly authored cases, with zero failures,
errors or skips. Collection and JUnit identities agree. All 215 source bindings,
both actual dependency execution revisions and the original reversible view/edit
mapper stayed unchanged. The runner observed 690.698 seconds.

The listed population binds 215 files and supporting records: 164 accelerator
and 51 immutable datasets objects. Actual datasets execution is
`d8f9e3dfc77901162e3d32343860aa4678374db0`; kit execution is
`8ac00bb1974b1378e12091d546d8425af34049e0`. These bindings are not a complete
import closure or installed-toolchain attestation. Qualification uses private
runtime state and preserves all earlier evidence at the published base.

The [datasets publication comparison](datasets-publication-compatibility.json)
checks those 51 files against the parent's newer datasets pin `ebdf6539a4`.
Their Git object bytes are equal. Actual execution remains at `d8f9e3dfc7`;
the comparison does not qualify the incoming decoder training or model assets.

The 47 newly authored cases cover pinned replay, native claim/grant loss,
selected and omitted source drift, CAS corruption, lifecycle renewal and loss,
serialized and retained-object substitutions, reentry, publication failures,
competing allocations and the broker boundaries. Claims, coordinator records,
CAS bytes, sealed Portal projections and allocated Git worktrees are real.
Quack listener/deployment metadata and support-profile examples are authored
test seams; those examples are not signed Terminal profiles.

The [documentation runner](documentation_runner.py) executes the four exact
local documentation workflow blocks and checks the maintained guides, plan and
this record. Its [receipt](documentation-checks.json) keeps those checks distinct
from hosted CI. No additional tests run through that runner.

## Remaining work

The [updated improvement plan](../../reversible_codebase_views_plan.md#nominated-allocation-custody-2026-10-14)
prioritizes authenticated cross-process native owner/coordinator provenance,
explicit worker permission transition, actual direct-child and signed compiled
context adoption, deployment qualification and complete-session measurement.
Source expansion needs a new admission and stale-capability retirement.

No coding child, provider, model, training, solver, Docker or official task
verifier is invoked by this qualification. No bootstrap, compiled-context,
coding-process, proof, execution, completion or omission authority is supplied
by the new receipt. There is no new reconstruction or token-saving result.
Existing family/schema/version/decoder-task and 8D/384D/768D checkpoints,
embeddings, model-manager entries and DuckDB/DuckLake inventories stay intact.

## Incoming normalization work

Upstream `04baaef0f8` adds stricter native IR normalization through
`3d0d3753ed`. The [incoming review](incoming-native-normalization-review.json)
checks its four changed paths and confirms they are outside this run's 215
listed bindings. Native task, allocation, broker and Terminal route files are
unchanged. The incoming normalization suites are not counted as executed here.
Publication preserves that work alongside this separately qualified increment.
