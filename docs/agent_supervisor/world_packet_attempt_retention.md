# Durable repository world packet attempts

This opt-in path retains an attempt before router validation and saves the
complete accounting receipt before caller replay or output verification. A
later failure cannot silently remove an earlier attempt from the declared
population. Missing evidence stays unknown. The journal does not issue a live
packet or authorize execution, acceptance, proof, completion or publication.

The earlier [bounded repair trial](world_packet_quality.md) retained an initial
unreceipted refusal and a later provider rate limit. Its private driver also
wrote the structured attempt only after historical replay. That ordering could
lose a charged attempt if replay or source reading failed. This implementation
moves attempt retention into the strict router path without parsing stdout.
Defaults remain unchanged; no model is called by collection or replay.

## One journal for one declared slot

The supervisor creates a private owned directory outside the worker workspace
and constructs `RepositoryWorldAttemptJournal` with the exact slot identity and
observed control values. Construction exclusively writes and syncs `start.json`
before the runner can validate a packet, discover a provider or issue an
invocation ID. Unknown observed controls are null. Values asserted by the
caller do not become authenticated measurements merely because they are
hashed or retained.

The journal retains the entire finite canonical JSON router receipt as
`receipt.json`, separately from `terminal.json`. The terminal describes whether
the runner returned, raised or was interrupted; it does not describe task
acceptance. The receipt is copied before historical artifact processing or
caller output checks. Failed provider attempts retain native accounting when
the native accounting path supplies it. Failures before receipt construction or
during native accounting can leave the receipt absent, with usage unknown.

Use `run_repository_world_packet_attempt(...)` with an issued journal and
ordinary strict router options. It returns the provider output, the router
receipt and a separate retention diagnostic. It preserves the same raised
exception when retention fails. Successful provider output also survives a
retention failure; the diagnostic reports unavailable capture or terminal
closure. Retention failure never means a free model call. Output verification
and historical replay belong after this boundary.

```python
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_attempt_journal import (
    CONTROL_FIELDS, START_SCHEMA, RepositoryWorldAttemptJournal,
)
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_attempt_runner import (
    run_repository_world_packet_attempt,
)

# private_root already exists, is empty, owned, mode 0700 and outside workspace.
# Unknown controls stay null until independently observed.
start = dict(
    schema=START_SCHEMA, attempt_id="case-1:baseline:1", case_id="case-1",
    arm="baseline", parent_attempt_id=None,
    observed_controls=dict.fromkeys(CONTROL_FIELDS),
)
journal = RepositoryWorldAttemptJournal(private_root, workspace=workspace, start=start)
output, receipt, retention = run_repository_world_packet_attempt(
    journal=journal,
    runner_options=dict(
        prompt=task_text, provider="codex_cli", model=pinned_model,
        timeout=180, max_output_tokens=4096,
        world_task_packet=current_strict_packet,
        world_task_presentation=issued_presentation,  # Or None for the original.
    ),
)
```

The caller still supplies native admission and the correct workspace working
directory. Inspect the retention diagnostic even after a provider return. Do
not combine journal mode with the older `world_packet_artifact_dir`; historical
replay is a separate step after the durable attempt has closed.

The journal is checked before the original packet's final native source
checks. No journal write is inserted between those closing checks and the
provider call. The existing strict packet, supported route, validation commands,
effect scope and source checks remain required. The read-only observation
profile still refuses provider dispatch. This API does not enable a general
Harbor profile or change the supervisor's default transport.

## Filesystem and evidence scope

Directories must be owned and mode 0700, with no symlink in any component.
Writes use exclusive 0600 files, bounded exact readback and file/directory
syncs. Held original file identities are checked across stages, including
same-byte replacement, hardlink and root replacement cases. Failures preserve
partial evidence and foreign files. Existing records are never overwritten.

These checks concern the live held filesystem objects. Historical hashes bind
bytes; they do not recreate original inodes, establish authentic chronology,
prove an exhaustive campaign, prevent a privileged actor from modifying data,
or establish model ingestion or billing. Store receipt descriptors and their
expected hashes independently under supervisor custody. Raw receipts and start
metadata may be private; do not include them in model prompts or public logs.

## Collect all declared slots

`terminal_world_packet_attempt_collection.py` accepts independently pinned
declaration and ordered slot-census inputs. Each declared slot remains present
even if its start, receipt or terminal is missing. Absent or incomplete evidence
does not imply `not_attempted`, refusal or zero usage. Malformed evidence,
foreign identities and incorrect pins are refused rather than treated as
missing. Complete receipts survive missing terminal records.

Observed controls come from retained starts; absent starts produce null
controls, never values copied from the declaration. A generated source/output
hash is supplied explicitly as a separate retained observation. A returned
provider message is not a verified generated output. Verifier results must
still join the exact dispatched attempt and output identity. Missing verifier
results generate explicit unknown case/arm rows instead of shrinking the
population.

The collector composes the existing paired quality accounting. Cumulative
native usage is selected once per session and arm across all retries. Known
subtotals survive incomplete evidence; complete totals remain null when needed.
The scientific gate still requires eligible controls, non-regressing quality
and lower complete supplied usage. Call census and temporal-origin limitations
remain explicit. An amended declaration needs separately retained original and
amendment evidence; this collector does not authenticate amendments.

Use the composed `durable_population_gate` and `comparison_gate` when deciding
whether this path qualified. The nested `accounting_report` keeps the original
conditional accounting semantics and its independently replayable hash. A
complete native receipt and verified output can satisfy that conditional core
while the new durable gate still rejects a missing terminal record. The
collector does not modify the core report to hide that distinction.

The [supervisor benchmark session accounting](world_benchmark_session_accounting.md)
path consumes these records through a pinned projection and reconciles native
usage before Harbor export or comparison aggregation.

## Qualification

The frozen final suites pass 352 distinct tests: 82 journal cases, 25 runner
cases, 83 collector cases, 65 prior strict/presentation dispatch cases, and 97
prior accounting cases. Provider calls and counters in the runner suites are
authored; source/index/compiler owners and filesystem operations are genuine.

An additional offline replay consumed the independently pinned older actual
trial, preserving all three attempt slots, two full provider receipts, the
unreceipted attempt and known native token subtotals of 486,151 and 21,251.
Because that trial predates the journal, all three starts and terminals remain
explicitly missing. No start, terminal, observed control or timestamp was
reconstructed to suggest it existed before those calls. Complete totals remain
null and both the durable and combined comparison gates reject qualification.
A changed receipt is refused even when its enclosing census is re-pinned.

The replay's full canonical comparison SHA-256 is
`c05cd492014a866e8598d4b8fc69d022a46f028e6c38a2aceb06e876e847649c`.
Private evidence is retained under
`/home/barberb/lift_coding/artifacts/world-packet-attempt-retention-20261015-01`.
This stage makes no new provider call and does not upgrade the earlier
rate-limited trial into savings evidence. The next empirical step remains a
fresh paired run with provider availability and independently observed controls.
