# Matched exact/compact source trace — 2026-10-13

Two isolated worker sessions executed the same authored read/edit sequence through
native generated owner/worker launchers, the source broker, and the default source
CLI. The exact arm used `identity@1`; the compact arm used `python-comments@1`.
Both requested `--view compact`, expanded exact original source before editing,
renamed `idx` to `i` in public `largest-eigenval/eigen.py`, checked generation
refresh and stale-read refusal, and restored the workspace. `eval.py` was unchanged.
The prescribed rename is an integration exercise, not a solution to the benchmark.

| Captured UTF-8 bytes | Exact | Compact |
| --- | ---: | ---: |
| Prepared router prompt | 53,933 | 53,940 |
| Default CLI stdout, including newlines | 25,429 | 25,289 |
| CLI stderr | 0 | 0 |
| One-pass prompt plus CLI output | 79,362 | 79,229 |
| Separate native response JSON | 39,645 | 39,709 |

The compact arm exposed **133 fewer bytes** in this scripted one-pass accounting.
The CLI difference was 140 bytes; longer profile guidance added 7 prompt bytes.
Transport JSON increased by 64 bytes and is a separate measurement. There is no
provider-token score, inferred token conversion, total-token savings claim, or
official reward. An authored `llm_router` hook replaced generation: no native
provider/model call occurred, despite exercising the `codex_cli` deployment route.

Both arms completed 11 logical operations with 12 CLI calls. Optional catalog
navigation cost 12,151 CLI bytes per arm, including two pages. The paths were
already known from the inline manifest and fixture; catalog navigation was an
explicit scripted step, not required to read or edit those files. Required raw
source remained in the initial semantic prompt. The source scope included the two
public files and three owner-generated harness support files.

One preflight supplied an identical native input prompt to both arms. Fresh worker
sessions ran sequentially, exact then compact, with separate descriptors and
working generations. The container had no network, an owner UID of 1000, and a
worker UID of 1001 with no capabilities or privilege escalation. Owner checks
replayed complete source and catalog bytes, confirmed clean/restored workspaces,
and found no remaining worker processes before the next session. The container
was removed and all 101 recorded source pins and three runner pins were unchanged.
This shared setup does not qualify cold-index performance or a timing advantage.

- [Strict comparison](collection/comparison.json), [shared controls](collection/controls.json), and [collection provenance](collection/collection-provenance.json).
- [Exact replay](collection/exact-owner-replay.json) and [compact replay](collection/compact-owner-replay.json).
- [Docker qualification](docker-02/qualification.json), [container controls](docker-02/container-controls.json), [command and source pins](docker-02/command.json), and [frozen release](docker-02/release-02.json).
- [Exact owner observation](docker-02/results/exact/owner-qualification.json), [compact owner observation](docker-02/results/compact/owner-qualification.json), and their adjacent router/session summaries.
- [Final regression](final-tests/qualification.json), [JUnit](final-tests/junit.xml), and [log](final-tests/stdout.txt): 97 distinct tests passed with no failures, errors, or skips; all 11 recorded source pins were unchanged. This includes the 41 focused checks.
- [Focused comparator checks](focused-tests/qualification.json) and [JUnit](focused-tests/junit.xml): 41 passed, no failures, errors, or skips.
- [Host mapper lineage](host-mapper-lineage.json) joins the current postimage to the earlier mapped-edit qualification. No new mapped application or numerical test ran in this pair.
- [Live prerequisites](live-prerequisites.json) is a separate read-only readiness audit. Its historical token numbers describe older experiments, not this source-view comparison; its command recipes were not executed here.
- [Disclosure checks](disclosure-checks.json) and [artifact checksums](manifest.json).

The worker edits used exact original bytes. Model-facing submission to the host
compact-edit mapper was not exercised. No checkpoint inference or proof-discharge stage ran. Static indexing,
capsule/world construction, and Doctor diagnostics did run; no source obligation
was discharged. This was not full supervisor START or an
official Terminal-Bench attempt. Owner replay hashes bind supplied evidence but
do not independently authenticate its origin.

Only content-free observations, public qualification scripts, commands, hashes,
and test output are published here. Raw prompts, full source-bearing worker/CLI
traces, private state/databases, IPC files, signing material, and private broker
audits remain outside the bundle. Their local artifact paths and hashes remain in
provenance records. The strict comparator replay therefore requires those retained
local captures; the public summaries alone cannot reproduce the full byte replay.
Scripts preserve absolute qualification-machine paths and need explicit path
adaptation elsewhere. The copied runner snapshot is the executed version; the
collection script binds the separately tested final comparator.
