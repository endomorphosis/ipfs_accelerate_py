# Working-source refresh qualification — 2026-10-12

This qualifies the [working-candidate source reader](../../worker_source_views.md#read-working-files-after-edits)
through the isolated coding worker. It used an authored router provider hook,
with no model calls, native provider token measurements or official benchmark
verifier. Candidate source text does not refresh formal evidence.

## Docker edit and refresh

The [closed review](closed-docker-review.json) verifies the first passing attempt,
unchanged hashes for 84 production files and three runner scripts, and removal
of the owned container. The run took 34.14 seconds with networking disabled,
five CPUs, 12 GiB memory and a 128-process limit.

Actual worker UID 1001 edited and restored `eigen.py` in its allocated worktree.
The production owner launcher, router, source client and socket served a new
working snapshot. The worker expanded the edited original exactly, while the
canonical baseline and semantic admission remained unchanged. Editing `eigen.py`
invalidated an older working continuation for the unchanged `eval.py`. Three
distinct observed generations survived edit and restoration without reviving an
old cursor. All 18 negative checks passed, including stale generations, pins,
canonical drift, unadmitted paths and foreign peers. Cleanup retired the endpoint.

The [probe](docker-01/results/probe-qualification.json) checks that candidate
labels and exact pins survive compact CLI rendering. Its edited source was 677
bytes, with a 642-byte compact view. Default candidate JSON was 1,229 bytes;
the corresponding full envelope was 2,791 bytes. These are response bytes,
not measured tokens or a complete-session advantage.

The [collector](docker-01/results/session-observation.json) matched 39 requests,
20 successful pages, 68,977 full response bytes and 23,157 returned source bytes.
This includes repeated reads, refusals and two owner-driven peer checks.

| Request population | Requests | Successful pages | Full response bytes | Returned source bytes |
| --- | ---: | ---: | ---: | ---: |
| Baseline v1 | 25 | 13 | 45,500 | 16,650 |
| Working candidate v2 | 14 | 7 | 23,477 | 6,507 |

The split comes from the frozen authored call order joined to private owner audit
ordinals and response hashes. The public collector totals alone do not classify
requests by source state. The closed review records that provenance and verifies
all 12 retained primary response hashes. Repeated deliveries are not unique
source coverage. Full protocol bytes differ from default model-visible output.

## Validation and scope

The [integrated regression](final-tests-01/qualification.json) completed with
1,164 passes, one failure and no skips. The failure occurred before Doctor
dispatch: Elan's `stable` default resolved to a missing Lean 4.35.0 executable.
The [exact failed case](doctor-installed-prover-01/qualification.json) passed with
the already installed Lean 4.34.1 executable explicitly pinned, with no source
changes, global toolchain changes or skipped proof checks. Its command records
the executable hash and version. The [regression review](regression-review.json)
joins these results: 1,165 unique cases have a passing result across two runs;
the original combined run is retained as failed.

These cases exercise the final source together with baseline, numerical and
native source readers. Exact commands, pytest logs and JUnit reports are retained.
Source hashes remained unchanged during both runs. Focused client tests also
passed 185 cases; their population overlaps this regression and is not added to
its count. Local documentation and changed-file link checks passed.

The new source cache remains read-only and session-local. Unpinned first-page
working reads can rebuild it; pinned reads and continuations never refresh.
The admitted path population, 64-file/1-MiB bounds and canonical source admission
remain fixed. New unadmitted files are not scanned or served. Edits observed
by working-source checks invalidate old generations; this is not an atomic lease
or detection of every unobserved edit/revert cycle.

The Docker probe does not perform full live supervisor START, learned inference,
checkpoint training, solver calls, proof discharge, accepted-source publication,
formal-index rebuilding, capsule refresh or mapped compact edits. Separate
regression cases can exercise real provers on authored fixtures; they do not prove
the benchmark source. Required source remains in the initial prompt. A controlled
live run is still needed for token savings.

The [manifest](manifest.json) binds the reviewed publication. Private source
bodies, owner catalogs, IPC, full audit records, prompts and model logs are
excluded. Frozen runner snapshots and exact source hashes describe the tested
implementation independently of later Git labels.
