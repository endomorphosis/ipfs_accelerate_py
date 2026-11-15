The full supervisor now carries an explicit Grok pruning selection through benchmark preparation, deployment, the container worker, `llm_router`, invocation receipts, and comparison reporting. This is the next controlled experiment after the [tool-event coverage diagnostic](grok_tool_event_coverage.md). It does not yet establish lower token usage: the new paired live benchmark remains pending, and the pinned client's inspection output does not expose effective pruning settings.

The default is `legacy`. It adds no pruning selection to historical configuration or receipt shapes and performs no pruning-specific file access or process launch. Explicit profiles require the Grok route, `declared-contract@1` planning, one fresh ordinary coding candidate, and no Doctor continuation, census, mapped edits, or structured candidate reply. Both full and no-index supervisor arms support these profiles. This restriction keeps model planning and continuation behavior out of the first pruning experiment.

| Setting | `grok-pruning-off@1` | `grok-pruning-on@1` | Unit |
|---|---:|---:|---|
| `enabled` | false | true | boolean |
| `keep_last_n_turns` | 3 | 3 | turns |
| `soft_trim_threshold` | 4000 | 4000 | characters |
| `soft_trim_head` | 1500 | 1500 | characters |
| `soft_trim_tail` | 1500 | 1500 | characters |
| `hard_clear_age_turns` | 10 | 10 | turns |

Only `enabled` differs. These numeric values are documented defaults, not token counts. Deployment writes `[compaction.pruning]` into the isolated worker's `/opt/ipfs-supervisor/worker-home/.grok/config.toml` and places a separate closed declaration at `/opt/ipfs-supervisor/grok-pruning.json`. Both files are root-owned and mode 0444. Existing authentication copying is unchanged. The worker home remains writable for native CLI state, so the preflight does not claim config immutability during the coding invocation.

The pinned Grok 1.0.46 documentation places pruning in the user config. Project configuration and the allowlisted `GROK_CONFIG` / `GROK_CONFIG_PATH` overlays do not carry this section. The [protocol audit](evidence/grok-pruning-20261015/pruning-config-protocol-audit.json) records the source ranges and hashes used to establish this. Five network-disabled, credential-free inspection containers completed and were removed. `grok inspect --json` worked, but on/off reports were byte-identical, contained no pruning fields, and reported an empty configuration-source list. This does not prove the user config was ignored; it means effective settings and file consumption remain unverified.

Before model dispatch, the explicit route checks exact config bytes, root ownership, read-only mode, a single file link, and symlink-free path bindings. It verifies the pinned executable hash, runs bounded `inspect --json`, validates the observed schema, and rechecks the config. Inspection is limited to 30 seconds and bounded stdout/stderr. Its owned process group is cleaned up on timeout, excess output, or a child retaining a capture pipe. Closed receipts retain sizes and hashes without native inspection text. Inspection time is deducted from the coding invocation timeout.

The shared CLI metadata layer retains the validated structured preflight receipt. The runner rejects an explicit-profile result with missing, invalid, or mismatched evidence, including a nominally successful provider exit. Failed invocations still retain observed native usage. The receipt always marks effective settings, config consumption, pruning behavior, and token savings as unverified. A future client schema exposing pruning values requires separate qualification before those claims can change.

For a paired run, rebuild and qualify the runtime archive first; preparation rejects archives whose relevant sources differ from the checkout. Add these options to the existing qualified full-supervisor preparation command, once for each profile, with separate output directories:

```text
--planning-mode declared-contract@1 --candidate-rounds 1
--grok-pruning-profile grok-pruning-off@1
```

Use `grok-pruning-on@1` for the second preparation. Keep dataset, task inputs, archive, provider/model, resource limits, turn budget, semantic profiles, setup policy, and all other controls identical. Execute and collect use the frozen preparation; those operations cannot change the pruning selection. Benchmark controls schema @10 binds the added profile while historical schemas remain supported.

The pure comparison API requires both collected receipts and their original frozen Harbor configurations:

```python
from benchmarks.agent_supervisor.container_coding.benchmark_comparison import (
    compare_grok_pruning_pair,
)

comparison = compare_grok_pruning_pair(
    off_receipt, on_receipt, off_config=off_config, on_config=on_config,
)
```

The helper recomputes configuration bindings and permits only `jobs_dir` and `agents[0].kwargs.grok_pruning_profile` to differ. It also checks task hashes, sealed controls, runtime archive identity, and each invocation's selected preflight. Same-arm full/full or no-index/no-index pairs are eligible; mixed-arm experiments require a separate comparison. Correctness differences are reported independently of configuration eligibility. Missing correctness, native token totals, effective settings, and complete billing remain unknown. No token-saving percentage is manufactured from incomplete subtotals, prompt bytes, or repeated output events.

Runtime commit `09cd34767b` passed **900 offline cases**: 383 core/router checks, 371 supervisor/deployment/comparison checks, nine authored inspection subprocess checks, and 137 metadata/Doctor regression checks. Source and dependency pins remained unchanged during the final batches; no model calls were made. Validation caught and fixed generic metadata stringification of the structured receipt. Review also added the pre-return receipt check and separated outcome differences from configuration eligibility. Two authored fixtures were corrected to preserve unknown completion after an empty timeout and supply the generic exit field separately from native usage observations. Earlier failed test records are retained alongside the passing qualification.

The [qualification receipt](evidence/grok-pruning-20261015/qualification-final01.json) and [evidence manifest](evidence/grok-pruning-20261015/manifest.json) bind the evidence. At qualification, free space was 12,687,036,416 bytes (about 11.8 GiB), below the unchanged 34,359,738,368-byte (32 GiB) admission floor. No new runtime archive or paired live benchmark was launched. The remaining work is to build and qualify the updated archive and run matched tasks when disk admission allows it, then compare official correctness, complete native usage where available, and elapsed time. Retained artifacts and admission limits are unchanged.
