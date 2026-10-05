# Terminal supervisor expansion evidence — 2026-10-05

Expanded task profiles, symbolic repair, and Grok use `llm_router`.
The latest completed trial used
`c9357feb18226614ae8a9b246a01403a77d0f843`; datasets source is
`987cf856b2b902aa68c4587bb492b19b932b5d30`.
The later bounded-checker repair is datasets
`5171a632c6b9f0ecb2939d29d2ad74992cbfeb11`; the live archives remain pinned
to their original dependency. See the [contract repair](header-checker-contract.md).

## Qualification

The [qualification summary](qualification-summary.json) records **1,528 distinct
passes and three skips** across exact revisions, not all on the final commit.
Merged integration passed 420 tests; planner-budget qualification passed 257;
Routing passed 109 repeated cases; final merged integration passed 166.
The later lifecycle checks passed 124 and 12 cases; corrected dependency
compatibility passed all 67. The distinct total removes overlapping cases.
Skips concern checkpoint fixtures; separate checks exercised actual inference.
The datasets checker separately passed 86 tests with one optional public-fixture
skip; these development checks are not added to the aggregate.

Three profiles passed signed preparation and hydration without provider calls:

| Task | Indexed symbols | Immutable data |
| --- | ---: | --- |
| `tune-mjcf` | 4 | Reference XML |
| `llm-inference-batching-scheduler` | 21 | Two request JSONL files |
| `constraints-scheduling` | 0 | Three ICS calendars |

The first two consumed the pinned Source384 checkpoint once each and passed
current/historical scope replay. The calendar profile abstained before loading.

The scheduler retains 99,183 data bytes and 41 capsules. Its 32,751-byte semantic
context delivers **zero capsules** and data references; coding receives
verified retrieval context. Retention does not prove use. Five lifecycle
regressions cover publication refresh and immutable drift.

Lean/Z3 checks qualify finite local alias resolution and argument/signature
binding, not arithmetic behavior, neural formula fidelity, or whole-task
correctness. Structured-data semantics remain a Doctor residual. Merged `@3`
administrative and ready-root advisory qualification keeps native execution
guarded; parallel execution remains unqualified.

## Live Grok attempts

These `tune-mjcf` trials are separate from qualification. See the
[live trial summary](live-trial-summary.json) for bounded receipts.

| Attempt | Observed outcome | Official score | Usage |
| --- | --- | ---: | --- |
| 01 | Setup rejected native version format | Unavailable | No provider dispatch |
| 02 | Planning timed out at 90 seconds | 0 | Unknown |
| 03 | Native provider exited 1 | 0 | 28,566 observed native tokens; completeness unknown |
| 04 | Planning timed out at 90 seconds | 0 | Unknown |
| 05 | Native cancellation after 77.07 seconds despite 180-second cap | 0 | 30,296 observed native tokens; completeness unknown |
| 06 | Native end_turn at 139.49 seconds; JSON format validation failed before admission/START | 0 | 67,681 observed native tokens; completeness unknown |
| 07 | Native end_turn at 41.233 seconds; 153-byte response rejected as prose_wrapper | 0 | 15,338 observed native tokens; completeness unknown |

The [trial 06 tool observation](trial06-tool-policy-discrepancy.json) exposed 23 native tools despite the requested empty allowlist; three
`read_file` calls succeeded. Arguments/paths are unavailable. Requested
policy is not actual exposure. The corrected [readiness probe](ready-probe-03.json)
and [trial 07 observation](native-tune-mjcf-07-observations.json) independently observed zero tools.

| Task/attempt | Route | Result |
| --- | --- | --- |
| `largest-eigenval/01` | Source-bound symbolic planning, then Grok coding | Official score 0; coding timeout at 300.14 seconds; total tokens unknown |

This run admitted two goals and one task with zero planning-provider calls and
succeeded at native START. Its [lifecycle observation](eigen01-lifecycle-observation.json)
records 59 replacement-bootstrap errors after coding timed out. The task stayed
`in_progress`. STOP reported success and zero tracked processes, but runtime
closure still refused live launched-child custody. This is not a clean shutdown
qualification. The underlying bootstrap rejection was not retained; observed
source-validation phases completed, so source drift is not established as the
cause. The two latest task trials ran concurrently and are not a controlled
latency comparison.

The subsequent [post-START guard](native-bootstrap-progress.md) bounds this
failure loop without settling the task or granting retry authority. Bootstrap
diagnostics now classify only closed phases and reasons. Its 124 focused checks
and 12 diagnostic checks passed; the latter includes a real provider-free
replacement daemon, fresh birth-bound grant and exited launched processes after
STOP. That normal restart does not reproduce or resolve the specific live
timeout recovery failure. No live score is attributed to these later changes.

The new profile retains 900 seconds total, 840 seconds work, and 60 seconds
cleanup; existing profiles retain 90-second planning. The
[archive review](archive-review-07.json) binds the runtime package.
No success, savings, or Codex advantage is claimed. This export
contains bounded metadata; credentials, model bodies,
hidden verifier bodies, and solution bodies are excluded.

## Export verification

The [export safety review](live-export-safety-review.json) passed 26 local
metadata checks. The included scripts also support a portable fixture-only
check with no benchmark artifacts or source checkout:

```bash
python check_evidence_exports.py --fixtures-only
```

That check passes 22 synthetic metadata controls, not benchmark tasks. Use
`--artifact-root /path/to/terminal-expansion-20261005 --worktree /path/to/ipfs_accelerate_py`
for full local replay. New datasets revisions require the recorded tree, exact
revision, clean status and protected Python hashes; historical runs retain their
original, more limited evidence. See the [portability review](exporter-portability-review.json).
