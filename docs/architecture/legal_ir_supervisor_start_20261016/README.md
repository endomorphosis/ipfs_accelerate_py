# LegalIR ordinary worker source-only bootstrap revision r03

This r03 proposal derives from the exact committed r02 package at
`d548a9353e5fb26b2ab40fdab0d5a50336bdb9f1`. It preserves the complete frozen
handoff and independent-acceptance gate while reviewing one additional bounded
engineering attempt. It does not reset or delete durable attempt history.

The r01 supervisor stopped after the strict attempt inventory rejected an
ephemeral authentication symlink. Its failed attempt and retained disk claim
remain preserved. That interruption does not prove no provider generation
occurred. The r02 native supervisor and daemon became active but selection
reported `task_attempt_limit_backpressure`: LIR-9001 already had one durable
attempt and the configured total cap was one. No productive provider dispatch
was observed in the r02 engineering metadata. Root requested STOP only through
its actually owned launcher, then confirmed native all-trees fencing, a quiescent
group, ordinary provider cleanup and native finalization. The failed/intentional
STOP record is preserved rather than relabeled as a productive task result.
See the [launch02 receipt](/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-ordinary-worker-bootstrap-20261016-launch02/receipt.json)
and exact engineering identities in `startup-status.json`.

This explicit reviewed revision raises `max_task_attempts` to **two total durable
attempts**. The first attempt remains counted; this permits one further bounded
engineering attempt. It does not change the implementation, validation or merge
retry budgets of one, the restart cap of one, provider order/fallback conditions,
900-second controller duration, source-only scope, or any acceptance/phase gate.
Future additional attempts require another explicitly reviewed revision.

The r03 master label and runtime directory isolate orchestration artifacts.
Canonical LIR-9001 identity and durable attempt history can carry across runtime
revisions, as the r02 backpressure observation demonstrates. They are not reset
by these path or label changes. Before all native imports, the owning controller
must set TMPDIR, TMP and TEMP to the
separately admitted and precharged provider directory
`/tmp/pytest-of-barberb/legal-ir-ordinary-worker-bootstrap-20261016-provider03`,
outside the strict attempt inventory. Actual temporary-file custody and cleanup
belong to that native owner. This proposal supplies no resource grant, symlink
exception, shared cap increase, task-state reset or claim release. Root must
verify the stopped r02 generation and current source before admitting r03.

This reviewed proposal derives from the immutable bootstrap package at
`651d2ea941c720726e234fbe4e319e650ad70f71`. It retains the 18 frozen
authoring/context source files and the complete task/goal/dependency population.
It uses the separately owned branch `agent/legal-ir-ordinary-worker-20261016`, execution namespace
`legal-ir-ordinary-worker-bootstrap-20261016-r03` and dedicated runtime paths. The original authoring
namespace is historical provenance; it is not a claim on another controller's
Quack catalog, database generation, task state, credentials or acceptance.

The ordinary worker image is source-pinned to
`sha256:a91a42c918514203bb6cfc7c38b320dfb68e9dfc0571bf01976c0db993e24e5e`, with label
`org.ipfs-accelerate.ordinary-worker=2026-10-16-ordinary-v1`. Its build inputs,
qualified mounted runtime, cleanup controls and limitations are recorded in
[`image-profile.json`](../../../containers/external-agent/ordinary-task-toolchain/image-profile.json).
The qualified image and source integration do not establish provider/model
access, supervisor execution, legal fidelity, training accuracy or a Lean build.
Protected routes retain their original image authority.

This staged execution package retains the complete 64-task, 34-goal handoff.
Only **LIR-9001** is `todo`. Every other task is explicitly phase-blocked, with
its original status and blocker recorded beside it. All task identities,
dependencies, parent crosswalks and goals are preserved. The original authoring
documents are copied unchanged under `authoring/`; their source checkout was
not edited.

## Files and installation layout

Install this explicitly reviewed revision at
`docs/architecture/legal_ir_supervisor_start_20261016/` in the separately owned
accelerator checkout. The logical result path remains
`data/legal_ir_span_graph_lean_20261016/LIR-9001/handoff-reconciliation.json`.

| File | Purpose |
| --- | --- |
| `tasks.todo.md` | Full native legacy-Markdown execution projection: one todo, 63 blocked. |
| `tasks.execution.json` | Exact projected authoring data; documentation, not a native JSON ingest schema. |
| `goals.objectives.md` | Byte-identical copy of all 34 authored goals. |
| `scheduler_config.json` | Reviewed ordinary-worker configured-board revision; source ancestor bound to the qualified integration commit, with no launch grant. |
| `validate_bootstrap.py` | Strict standard-library document and report checker, with no product/provider/Git/database calls. |
| `source-copy-manifest.json` | Exact pins for 18 unchanged authoring/context source files. |
| `CURRENT_SOURCE_BINDINGS.json` | Closed, bounded owner-supplied repository metadata; initially unavailable with null observations. |
| `authoring/`, `context/datasets/` | Frozen source/configuration and engineering metadata for the first task. |

The configured source route is `legacy-markdown`. The trusted primary executable
is `/home/barberb/.local/bin/grok`. The ordered provider contract is exactly Grok
CLI `grok-4.7`, with Codex `gpt-6.1-sol`, reasoning `high`, only
after `primary_quota_exhausted`. One lane is proposed. No provider was contacted
or credential read while staging this package.

## Before actual startup

The required current and merge branch is `agent/legal-ir-ordinary-worker-20261016`. The required
source ancestor is `93cfd32f65cd231df57e53736edf0f78b58af009`, which contains the qualified
ordinary image pin and route integration. The owning controller must verify that
ancestor and the exact current clean source before admission. This is an explicit
new package/verifier revision: branch, execution namespace, runtime paths and
protected image/build sources are bound by its new canonical config hash. The
immutable authoring and source-copy inventory hashes remain unchanged. Only a
reviewed replacement of the source ancestor is excluded from that config hash.

`CURRENT_SOURCE_BINDINGS.json` remains byte-identical pre-r03 metadata from
the committed r02 package in this proposal. Root must replace it with a fresh
actual owner-supplied observation of the current checkout and datasets source
before binding an r03 run. Its original origin, time and evidence must not be
relabeled as a current r03 observation.

The controller must commit the installed source package, bind actual current
provider/worker/source/resource capabilities, qualify genuine native parsing and
the launch path, and admit a finite owner before dispatch. Template fields and
copied historical pins supply no grant. Keep automatic objective/codebase refill
disabled. Use the genuine task execution slice for LIR-9001 where the chosen
launcher supports it; phase-blocking all other rows is an additional explicit
boundary. Manual completion alone does not stop dispatch.

Bound the controller to 900 seconds and one lane, with at most two total durable task attempts; this revision permits only one additional bounded engineering attempt.
Runtime worktrees, state, merge queue and logs use the dedicated
`workspace/legal-ir-ordinary-worker-20261016-r03/` namespace in the owning checkout.
Actual writable grants and resource lifecycle belong to the current native owner,
not these path strings. Do not reuse a foreign campaign's credentials, branch,
database, resource promise or execution authority.

The actual preparation owner may populate `CURRENT_SOURCE_BINDINGS.json` after
installation. Its closed metadata schema allows only the two named repositories,
attributed UTC observation/evidence metadata, observed checkout path/head/tree/
branch, bounded selected source-file path/size/SHA256 records and explicit
limitations. Unavailable repositories retain null identities and empty file lists.
The checker opens no referenced checkout or source body and authenticates no
observation; independent acceptance must establish factual origin and freshness.
Do not relabel copied historical pins as current source observations. The file
stays protected; update and commit it through the actual preparation owner before
binding the run. An observation of a pre-install source head retains its actual
time and limits rather than claiming to be a self-referential post-commit head.

## LIR-9001 worker scope

Read the frozen source/configuration and engineering metadata only. Inventory all
64 task IDs, 34 goal IDs, 24 original LIR parents and exact ordered dependencies.
Compare available current owner-supplied repository metadata and related LAC
headers, preserving collisions, unavailable authorities and the exact limits of
existing evidence. Record precise reuse/adoption recommendations; this task
adopts no commits and makes no code or test edits.

The only logical worker output is the LIR-9001 reconciliation JSON named above.
Native/controller lifecycle logs stay in the separately admitted attempt. Do not
open corpus/model/checkpoint/vector/latent/prediction/typed-payload bodies or
credentials; import product modules; run tests, Lean or numerical jobs; mutate
databases/ledgers/task state; or manufacture reviews, eligible gold, completion
or authority. Existing owners remain read-only context.

The report schema is `legal-ir-supervisor-handoff-reconciliation/v1`. The checker
defines its exact closed fields. It binds source-copy, authoring and execution
board hashes plus `current_source_bindings_sha256`; ordered task/goal populations;
the unchanged parent crosswalk;
attributed repository metadata; bounded LAC/collision/reuse/unresolved findings;
and explicit false effect declarations. Missing metadata must be recorded as
unavailable rather than invented. Report declarations are not authenticated
proof of behavior; independent acceptance remains mandatory.

## Verification and the closed independent-acceptance gate

The native daemon can mark a successfully merged task complete regardless of
`Completion: manual`. Manual mode alone does not fence merge or completion.
This ordinary-worker revision preserves the v02 closed executable report gate and fails it until a separately
reviewed independent-acceptance capability is actually bound in a future packet.
The model may produce a candidate report; the required native validation cannot
pass, so the candidate cannot complete or merge through that validation path.

After deliberate installation, the source-only structure command is:

```bash
python3 -B docs/architecture/legal_ir_supervisor_start_20261016/validate_bootstrap.py --check-all
```

It returns JSON with `valid: true` only for the exact expected projection and
frozen source copies. `ancestor_bound: false` means the native launch prerequisite
is still unresolved. This is documentation verification, not native preflight,
resource admission, provider readiness or task completion.

The first task's executable `Validation` command is:

```bash
python3 -B docs/architecture/legal_ir_supervisor_start_20261016/validate_bootstrap.py --check-report data/legal_ir_span_graph_lean_20261016/LIR-9001/handoff-reconciliation.json
```

It first checks the bounded report's shape and exact document correspondence.
Malformed or missing reports fail. Even a structurally valid report then returns
nonzero exit code **3**, with the closed reason
`independent_acceptance_capability_not_bound`. No self-authored receipt, report
declaration or CLI argument can turn that result into acceptance. There is no
review receipt writer or fake reviewer authority in this package.

After an actual candidate exists, the owner must prepare a new reviewed packet
with an independently qualified acceptance adapter. That adapter must establish
the factual repository observations, evidence scope, collisions and
recommendations, and bind the exact report/task/source/candidate identities under
actual independent authority. Only that later packet may change the closed gate.
Keep this immutable v02 evidence and the original staging package. Successful
`--check-all` verifies structure only and cannot complete LIR-9001.

## Subsequent phases

After the future capability is bound and LIR-9001 output is independently accepted,
the owning controller may prepare and review a new
execution revision for LIR-9002 capability mapping. Preserve this immutable
bootstrap and the untouched authoring plan. Do not clear all future blockers or
replace the complete board with a one-row document. Proceed through LIR-9002,
LIR-9003 bounded job packets, and LIR-9004 genuine full-population handoff
qualification before activating productive corpus/review/training/Lean stages.
Each transition needs current task revision/source roots, exact output ownership,
purpose-specific capability/resource admission and independent result acceptance.

This proposal was derived by bounded immutable Git source extraction and
ordinary source editing. Its syntax and declared derivation were reviewed without
importing product modules or executing the proposed checker. No supervisor,
native preflight, provider call, database mutation, model/corpus execution or task
launch was performed by this revision author. The owning native controller must
perform fresh structure, native preflight and finite lifecycle qualification.
