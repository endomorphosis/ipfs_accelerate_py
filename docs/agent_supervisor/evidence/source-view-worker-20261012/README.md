# Source views through the isolated coding worker

This is source-bound, provider-free qualification of the
[read-only worker source reader](../../worker_source_views.md). The existing
semantic producer admits public files; the owner exposes compact views and exact
original expansion to the actual spawned worker. No model or provider was called.

## Docker result

The [closed review](closed-docker-review.json) records 23 passing checks,
unchanged hashes for 84 source files and three qualification scripts, and removal
of the owned container. The cached image ran with networking disabled, five CPUs,
12 GiB memory and a 128-process limit. Actual owner UID 1000 and worker UID 1001
used the generated owner launcher, router, client and socket. A root-owned
authored `generate_text` fixture invoked the source client in place of a model.

The [final probe](docker-final-01/results/probe-qualification.json) reconstructed
both public files and both representations exactly across six primary pages.

| Public file | Original bytes | Compact bytes | Compact profile |
| --- | ---: | ---: | --- |
| `eigen.py` | 642 | 607 | `python-comments@1` |
| `eval.py` | 2,246 | 2,246 | Exact fallback: unsupported f-strings |

The default CLI validates the full protocol response, then prints a smaller
model-facing projection. First-page output measurements preserve source content,
expansion hashes and continuations:

| Public file | Full JSON envelope bytes | Default displayed JSON bytes |
| --- | ---: | ---: |
| `eigen.py` | 2,497 | 938 |
| `eval.py` | 4,096 | 2,523 |

These are byte counts, not tokenizer measurements. The code transform saved only
35 bytes on this public population. Initial required source still remains in
the prompt, and no whole-session token advantage has been established.

Ten negative checks covered wrong UID, an unrelated worker process, unadmitted
paths, incorrect expansion pins, forged/replayed/foreign cursors, and changed
canonical or allocated source. A private owner database read was denied. The
broker stopped and removed its descriptor and socket.

The [durable observation](docker-final-01/results/session-observation.json)
matched the selected profile and actual router receipt after cleanup. Its 24
requests/responses included 13 successful pages and 16,650 source bytes with
repeated reads included. Full protocol JSON totaled 45,303 bytes. All 22 CLI
calls printed 41,624 JSON bytes; two peer probes account for the other requests.
The three default-output calls printed 3,507 bytes, including a 46-byte refusal.
These populations must not be combined or described as unique source coverage.

## Tests and retained scope

The final [integrated gate](final-tests-01/qualification.json) passed 1,059 tests
with no failures, errors or skips in 628.50 seconds. Its source hashes remained
unchanged throughout the run. The exact command, pytest log and JUnit report
are retained alongside the result.

Focused gates retain [broker tests](focused-broker/qualification.json),
[client and collector tests](focused-client-collector/qualification.json),
[source and profile tests](focused-policy/qualification.json), and a historical
[launcher/router diagnostic](focused-wiring/qualification.json). That diagnostic
passed 31 tests but detected collector source changes during its broad source
check; the final integrated gate supersedes it. These tests exercise real
codecs, Git worktrees, source hashes, semantic producers and metadata; provider
responses and selected boundary facts are authored fixtures where specified.
The Docker gate independently qualifies actual process identities and file
permissions. Focused counts overlap and are not an aggregate total.

The closed review retains both Docker attempts. The first completed source and
refusal exercises but reported unavailable collector evidence; its exact refusal
location was not traced before that container closed. The fixture's selected
profile and collector checks were made explicit for the final passing run. Both
attempts' elapsed time is retained; the initial result is not a successful gate.

No full live supervisor START, official Terminal Bench verifier, learned
Source384/IntentIR inference, training, solver call, proof discharge or mapped
compact edit is established here. Automatic refresh after edits and capsule
expansion remain subsequent work. Original files are the editing/execution
representation; stale indexed views refuse.

The publication contains bounded observations, commands, script snapshots and
hashes. Private owner stores, prompts, raw source responses, IPC data, credentials
and model logs are excluded. The [manifest](manifest.json) binds the retained files; source
hashes describe the tested implementation, not merely a Git HEAD label.
