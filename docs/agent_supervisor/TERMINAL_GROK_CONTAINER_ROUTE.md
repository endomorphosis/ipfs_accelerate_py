# Grok in the full Terminal-Bench supervisor

The explicit provider profile `grok-4.7-cli-1.0.46@1` selects `grok_cli` / `grok-4.7` with high reasoning effort through `llm_router`. It uses the same admitted supervisor, planning, indexed context, candidate validation and native lifecycle as the Codex profile. See [public task profiles](TERMINAL_PUBLIC_TASK_PROFILES.md) and [suite resource profiles](../../benchmarks/agent_supervisor/container_coding/TERMINAL_SUITE_PROFILES.md) for the supported task boundaries and qualification limits.

This development profile binds the Linux aarch64 Grok executable to version `grok 1.0.46 (2765805b9442) [stable]`, 142,867,512 bytes and SHA-256 `45b0943e736f00a249b9cf02af2be9e0749d97c09a6f55cfcf3029a1a836f23e`. The builder requires the resolved, regular executable, packages it as `providers/grok/grok`, and records its independent binding. Deployment verifies the archive, binary and exact version before exposing the root-owned executable.

Only the selected provider's bounded `auth.json` is transferred separately. The Harbor adapter defaults to the operator's `~/.grok/auth.json`; credentials never enter the archive or deployment receipts. The worker receives its own private credential copy, runs as UID 1001 with no new privileges, and cannot access the supervisor owner's state. Its provider argument must match the root-owned deployment manifest. Planning enables no tools. Coding enables the bounded file/terminal tool profile only after the container boundary is verified.

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
