# Preserve consumed attempts: isolated source proposal V01

Status: authored source; no product imports, unit tests, native jobs, Docker operations, provider calls, taskboard changes, live state reads/writes, or Git mutations were performed by the author. Parsing Python ASTs is the only validation completed here. The parent must publish and qualify the proposal under a genuine engineering owner before using it.

`preserve-task-attempts.patch` changes four existing source files. `proposed/` contains their complete proposed contents, and `BASE_PROVENANCE.json` records exact original and proposed SHA256 values. The original owned controller and canonical accelerator repositories were read only. `author_proposal.py` reproduces the source transformation from that exact reviewed base; it imports only standard-library source-processing tools.

The optional scheduler JSON field `preserve_task_attempts` defaults to `false`. Setting it to `true` emits `--preserve-task-attempts` at the configured-board and direct scheduler-profile boundaries. Both supervisor and daemon CLI parsers accept that flag. `SupervisorConfig`, daemon child argv, strict existing-child adoption, both ordinary execution constructors, the database daemon constructors, and the private database-to-Portal execution bridge carry the selection. Native process adoption rejects a missing flag, a flag unexpectedly present, abbreviations, duplicate flags, and assignment spellings. This changes no admission or acceptance authority.

When selected, the ordinary daemon preserves its display-ID and canonical-CID attempt counts and existing retry queue state across all three automatic exhaustion rearm paths, completed-repair resets, four automatic cooldown repairs, and unknown vanished-worker recovery. Refresh avoids calculating a new runtime recovery revision for these rearms. Ordinary implementation dispatch, finite attempt-limit partitioning, outcome recording, merge reconciliation, resource admission, and cleanup remain enabled. The active policy is included in both ordinary refresh result projections. When the flag is absent, the existing behavior is retained.

## Confirmed prelaunch refunds

`_restore_task_attempt` now has the keyword-only argument `zero_provider_execution=False`. In preserve mode it refuses a lower count or a pop unless the argument is exactly the boolean `True`; truthy strings and integers do not grant restoration authority. Existing protected attempt state/event latches still apply. Default mode retains the previous restoration behavior.

The five explicit no-provider call sites are bounded by their existing source observations:

| Caller | Evidence required to refund |
| --- | --- |
| `_record_provider_capacity_deferral` | The failure and an independent log classification both report `pre_dispatch_refusal_verified is True`; other capacity failures return through retained custody before the restore. |
| `_run_implementation`, worktree setup deferral | The existing setup-deferral branch is taken and the ephemeral result explicitly reports `provider_dispatched is False`. |
| `_run_implementation`, dependency preflight | A dependency preflight defers before the direct provider invocation. |
| `_run_implementation_in_ephemeral_worktree`, retry probe | The retry-probe branch's actual `provider_dispatched` value is exactly `False`. |
| `_run_implementation_in_ephemeral_worktree`, terminal setup deferral | The actual `provider_dispatched` value is exactly `False` and the deferral is classified as lifecycle setup/race, typed pre-dispatch, submodule setup, or dependency setup. An external protected-path deferral alone does not suffice. |

The three direct terminal verification/completion, timeout, and exception restores and the final ephemeral completion-durability restore receive no refund authority. Their consumed ordinal therefore stays consumed in preserve mode. Missing provider-birth evidence is treated conservatively. This option does not authorize recovery of a provider result or completion of a task.

## Authored controls and application requirements

`test_preserve_task_attempts.py` contains 51 parameterized unit cases. They execute exact selected function ASTs from the proposed or publisher-applied source, with unchanged function bodies and signatures. State and queue fixtures are ordinary in-memory objects; no ledger, native grant, Docker metadata, provider outcome, or native acceptance success is fabricated. Controls exercise unchanged default rearm/reset behavior, preservation across successive runtime revisions, unchanged foreign counters, finite ordinals two-to-three-to-stop, productive/unknown restoration refusal, confirmed no-provider refunds, both CLI option definitions, profile/configured-board translation, child policy/adoption, and all execution constructor boundaries. Separate source-contract controls check the nine actual restoration call sites. These unit controls are not full CLI imports or genuine native/provider qualification.

For a genuine admitted unit qualification, the parent can set `PRESERVE_ATTEMPTS_SOURCE_ROOT` to the applied owned accelerator tree and run this exact frozen test file with pytest. The variable otherwise selects `proposed/`. The publication owner must additionally qualify the complete configured-board process/CLI route against the applied source closure and bind receipts to the new source pins. No test command was run by this author, and no pass result is asserted.

The new LIR-9001 packet must keep the actual current canonical identity, original source history, runtime state lineage, both consumed attempts, and unchanged blocked descendants. A reviewed cap of three permits one additional consumed attempt only after the existing public state interface verifies the durable current count is two. This proposal neither changes the cap nor imports/copies/resets state. It does not repair a new namespace hiding the prior state projection, identity drift, the old report schema, pending independent acceptance, or the fixture-only nature of the Docker acceptance exercise. Those remain separate packet/application obligations.

The resource metadata ledger's real 8MiB limit currently prevents another native qualification. This proposal does not raise that limit, prune ledger history, or claim tests passed. Public archival/compaction is a separate owner-controlled change. Preserve the failed and successful prior attempt receipts when resuming qualification.
