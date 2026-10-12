# Isolated numerical obligation retrieval — 2026-10-12

The owner-side numerical pager is now reachable through a task-bound Unix
socket and a standalone worker CLI. The descriptor identifies one task,
context, allocated worktree and owner process. The broker checks the kernel
peer UID and its ancestry beneath the exact launched child, replays current
canonical and allocated source bindings, and returns bounded read-only pages.
It grants no proof, execution or completion authority.

## Docker qualification

The current [postmerge Docker run](docker-postmerge-01/qualification.json) passed
with UID1000 as owner and UID1001 as the isolated worker. Source and runner
hashes remained unchanged, and the owned container was removed. The image,
read-only source mounts, 5 CPU / 12 GiB limits and disabled network are recorded
in the [command](docker-postmerge-01/command.json) and
[actual container controls](docker-postmerge-01/container-controls.json).

Only the permitted public `largest-eigenval` inputs, `eigen.py` and `eval.py`,
were indexed. Native task preparation, context compilation, three metadata
DuckLake catalogs and the world record were real. The
[primary retrieval](docker-postmerge-01/probe-qualification.json) returned all
**45 unique pending obligations in 12 pages**:

| Public input | Unique obligations | Primary pages |
| --- | ---: | ---: |
| `eigen.py` | 16 | 4 |
| `eval.py` | 29 | 8 |

The primary responses contained 60,083 UTF-8 JSON-envelope bytes in total;
the largest was 6,470 bytes against the 8,192-byte limit. Seven exact filter
checks covered kind, function and their conjunction, including an empty join.
Five attempts to read private owner reports/catalogs were denied. Seven
negative requests were rejected: wrong UID, unrelated process with the same
worker UID, forged cursor, cursor reuse across path/filter changes, and stale
canonical/allocated source bytes.

After cleanup, the production collector joined the closed session to the
actual router receipt and selected task. Its
[persisted observation](docker-postmerge-01/session-observation.json) contains
36 requests and delivered responses, of which 29 were successful pages with
97 total row deliveries and 134,039 JSON response-body bytes. **97 counts
repeated filter queries and a freshness recheck; it is not a unique obligation
count.** Byte counts exclude the four-byte socket frame prefix per response.

## What ran

The generated production `OWNER_ENTRY` was unchanged. It sealed the prompt,
created the broker, registered the actual spawned sudo child, and closed the
broker during cleanup. The root-owned `WORKER_ENTRY` fixture inserted one
explicit branch for `numerical-ipc-authored`, installing an authored
`generate_text` response before entering the real router main. That fixture
called the production client CLI in worker subprocesses. The
[deployment observation](docker-postmerge-01/root-setup.stdout) records both the
production and fixture entry hashes, and the exact
[runner](runners/run_docker_qualification.py),
[owner probe](runners/owner_probe.py) and
[worker probe](runners/worker_probe.py) are retained.

The worker had no effective/permitted/ambient capabilities and enabled
no-new-privileges. The container used the existing root setup and sudo identity
transition; it did not claim an entirely read-only base or a global privilege
drop that would prevent that transition. Private owner files remained
inaccessible to the worker.

The [router observation](docker-postmerge-01/router-observation.json) records one
authored adapter call and restored descriptor environment variables. Its
`usage` object contains only fixture exit/schema metadata, **no provider token
or cost counters**. No external model, paid provider, solver, official verifier
or benchmark solution was run. Planning was authored locally, Intent
preprocessing was disabled, and no Source384 neural inference or full
supervisor START was qualified. Planning remains a single provider request
in the production harness; this feature qualifies the coding-worker retrieval
path. All 45 source obligations remain pending.
These measurements establish transport and retained accounting, not token
savings, model tool choice, task success or source correctness.

## Test gates and retained attempts

The [premerge gate](premerge-tests-01/qualification.json) passed 528 tests with
zero failures, errors or skips and unchanged sources. The expanded
[postmerge gate](postmerge-tests-01/qualification.json) passed **568 tests**,
also with zero failures, errors or skips and unchanged sources. Its
[command](postmerge-tests-01/command.json), [JUnit](postmerge-tests-01/junit.xml),
[log](postmerge-tests-01/pytest.log) and
[runner](runners/run_postmerge_tests.py) are retained. The suites overlap; their
counts are not summed as distinct tests.

The [independent code review](final-code-review.json) found no blocker. Its
premerge revision metadata is retained unchanged; every reviewed file hash
still matches the merged source.

The current Docker run binds merged accelerator revision
`cdd45312c9c7aee14c5ffa5241352eead0ce624e`, including the concurrent owner
current-source lookup changes. The earlier
[premerge Docker run](docker-final-01/qualification.json) remains retained;
both produced the same primary and session accounting totals.

## Provenance

[Three diagnostic attempts](diagnostic-attempt-accounting-01-03.json) remain
retained locally. The first two stopped on fixture setup permissions. The
third exercised retrieval but failed final router closure because the authored
adapter omitted required exit/schema metadata and returned prose under the
selected JSON response contract. The corrected fixture produced the passing
run. These were provider-free qualification attempts, not live model retries.

The [source release](root-release-postmerge-01.json) pins selected producers and
public inputs, while the [closed receipt review](closed-docker-review-postmerge.json)
checks arithmetic and cleanup. The [artifact manifest](artifact-manifest.json)
lists an explicit bounded publication allowlist. Owner databases, complete
source reports, raw broker audit, worker response payloads, prompt bodies and
credentials are excluded. This directory is not a complete import-origin or
runtime archive qualification.
