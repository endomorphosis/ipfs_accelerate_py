# Grok in the full Terminal-Bench supervisor

The explicit provider profile `grok-4.7-cli-1.0.46@1` selects `grok_cli` / `grok-4.7` with high reasoning effort through `llm_router`. It uses the same admitted supervisor, planning, indexed context, candidate validation and native lifecycle as the Codex profile. See [public task profiles](TERMINAL_PUBLIC_TASK_PROFILES.md) and [suite resource profiles](../../benchmarks/agent_supervisor/container_coding/TERMINAL_SUITE_PROFILES.md) for the supported task boundaries and qualification limits.

This development profile binds the Linux aarch64 Grok executable to version/build `grok 1.0.46 (2765805b9442)`, 142,867,512 bytes and SHA-256 `45b0943e736f00a249b9cf02af2be9e0749d97c09a6f55cfcf3029a1a836f23e`. Its version output may append the managed-install channel label ` [stable]`; the standalone container binary omits that label. Only those two exact version/build forms are accepted, and the binary hash remains mandatory. The builder requires the resolved, regular executable, packages it as `providers/grok/grok`, and records its independent binding. Deployment verifies the archive, binary and version before exposing the root-owned executable.

Only the selected provider's bounded `auth.json` is transferred separately. The Harbor adapter defaults to the operator's `~/.grok/auth.json`; credentials never enter the archive or deployment receipts. The worker receives its own private credential copy, runs as UID 1001 with no new privileges, and cannot access the supervisor owner's state. Its provider argument must match the root-owned deployment manifest. Planning requests a singleton built-in allowlist and explicitly denies its sole tool. Coding enables the bounded file/terminal tool profile only after the container boundary is verified.

The benchmark profile disables automatic cross-provider fallback. A Codex attempt uses its own explicit profile and fresh trial evidence. The existing default benchmark profile remains Codex; select Grok explicitly in `prepare`.

## Build and run a fresh trial

Use a committed, clean checkout and the Harbor Python environment with the matching `ipfs_datasets_py` checkout importable. Replace the example dependency and asset paths with the independently pinned local paths. Keep build output and trials outside the source checkout. The selected Source384 configuration must match its frozen checkpoint, embedding snapshot and runtime archive.

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_deployment bundle \
  --output /path/to/new-grok-bundle \
  --source "$PWD" \
  --datasets /path/to/clean/ipfs_datasets_py \
  --kit /path/to/clean/ipfs_kit \
  --extension-dir /path/to/duckdb/v1.5.5/linux_arm64 \
  --lean-toolchain /path/to/pinned/lean-toolchain \
  --source384-config /path/to/source384-config.json \
  --torch-cpu-wheel /path/to/pinned/torch-cpu.whl \
  --torch-cpu-wheel-sha256 PINNED_WHEEL_SHA256 \
  --grok-binary /path/to/resolved/grok-1.0.46-linux-aarch64

python -m benchmarks.agent_supervisor.container_coding.terminal_profile_catalog \
  --dataset /path/to/terminal-bench-2 --task tune-mjcf \
  --output /path/to/new-tune-mjcf-profile.json

python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark prepare \
  --dataset /path/to/terminal-bench-2 --task tune-mjcf \
  --task-profile /path/to/new-tune-mjcf-profile.json \
  --archive /path/to/new-grok-bundle --output /path/to/new-grok-trial \
  --arm full --resource-profile source384-5cpu-16gib-extended@1 \
  --source384-config /path/to/source384-config.json \
  --provider-profile grok-4.7-cli-1.0.46@1

python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark execute \
  --output /path/to/new-grok-trial --task tune-mjcf
```

The Grok archive and trial reject Codex setup-cache policies. Container memory admission still applies; available memory must be measured after setup. The standalone deployment `qualify` command does not currently bootstrap generic task profiles. The full Harbor adapter performs that bootstrap.

The reviewed `tune-mjcf` profile treats `model_ref.xml` as immutable task data and declares creation of `model.xml`. XML parsing and source indexing establish input coverage, not physics equivalence or a symbolic repair contract. The Doctor retains an explicit residual when a data-consumer contract is unavailable. Official success requires the original native verifier; preparation, setup or a process exit code does not establish a benchmark score.

## Usage evidence

Grok receipts preserve numeric categories from the final native JSON envelope, including available counters on failed calls and timeouts. `input_tokens` excludes `cache_read_input_tokens`; the native `total_tokens` remains unchanged. Missing totals or categories remain unavailable, and cached input is not added a second time to the native total.

Harbor defines input tokens as including cache. Its input field is normalized
only when native input, cache reads, cache writes, output, and total counters
are all known and consistent; otherwise that field stays unknown. The native
receipt retains the original categories.

The scope is `native_final_envelope`, not an assertion of cumulative billing coverage across turns. A successful end-of-turn signal records native provider-session completion independently of token completeness or benchmark success. `usage_complete_observed` remains unknown when the CLI omits its completeness flag. Billing totals and dollar cost remain unverified. Comparisons must retain provider, model, CLI, resource, task and index selections; this route's availability alone establishes no efficiency advantage.

`native_provider_outcome` retains a closed set of native stop reasons and error
classifications even when token usage is absent. It distinguishes timeouts,
turn limits, token limits, refusals, cancellation, structured-output retries,
native execution errors and failed processes. Unknown or malformed values stay
unknown; contradictory stop-reason aliases are rejected. Plain-JSON errors that
omit a stop reason can match the pinned CLI's exact, bounded turn-limit message
templates. The receipt identifies that classification source and never exports
the message, model response, reasoning or debug log. These observations do not
grant completion authority or automatically change retry budgets.

`provider_invocation_policy` records the requested native settings: planning
uses at most two rounds, tools profile `none` and permission mode `dontAsk`; coding uses
at most 128 rounds, tools profile `isolated_coding` and `bypassPermissions` inside
the verified worker boundary. The `none` profile requests
`--tools read_file --disallowed-tools read_file,search_tool,use_tool`: the
nonempty allowlist activates filtering, then the deny list removes its sole tool
and Grok's otherwise retained MCP dispatch tools. Coding retains its six-tool
allowlist and denies only `search_tool,use_tool`. The receipt records
`native_tool_allowlist`, `native_tool_denylist` and `effective_toolset_verified=false`;
requested policy is not evidence of the native effective toolset.

In a retained development trial, `--tools ''` advertised 23 tools and performed
three `read_file` calls despite the requested `none` profile. The logs did not
retain their paths, so those reads cannot be classified against the task's input
boundary. The singleton-then-deny configuration subsequently exhibited zero
tools in a fresh readiness probe and full task trial; coding exhibited its six
requested tools. These [native observations](evidence/terminal-expansion-20261005/README.md)
qualify the pinned configuration without turning requested policy fields into
per-invocation attestations. This
deployment transfers no host MCP configuration. The generic
router retains its previous behavior when this option is omitted.
`provider_output_token_cap_enforced` remains false:
the router's requested output-token count is not an enforced native CLI limit.

The two-round planning cap is a bounded experiment under the selected planning
deadline; it does not extend that deadline. Native `cancelled` observations from
earlier one-round attempts do not establish that the turn cap caused the failure.

The retained full trials also leave two distinct gaps. Grok planning returned
text rejected by the strict JSON plan contract. A separate source-bound symbolic
plan reached native START and Grok coding, then hit the coding-call deadline and
a replacement-bootstrap failure loop. The subsequent progress guard observes
three new bootstrap errors without task, receipt or heartbeat progress and
requires a second unchanged sample before taking the ordinary STOP path. It
does not settle tasks, grant retry authority, or repair the underlying bootstrap
rejection. Closed phase/reason counters preserve that future diagnostic without
exporting exception messages or credentials. The live failed result remains
unchanged; these later guards need their own full-container qualification.
