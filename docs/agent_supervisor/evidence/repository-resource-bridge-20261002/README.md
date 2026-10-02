# Shared repository resource bridge qualification

The final run passed all 18 tests (103.76 seconds, no skips). Fifteen tests exercise closed CPU budgets and explicitly injected controls; three use the actual default datasets host authority and live sampler. They cover concurrent repository parents with nested consumer leases, dead-owner lease recovery after SIGKILL, and cancellation of a real bounded child process. The live tests do not substitute per-repository host capacity or scheduler files.

Earlier failures remain in this directory: one test-only `ToolRunLimits` keyword mismatch, then a real supervisor `host_cpu_high_watermark` refusal. Safety limits were kept; the complete suite was rerun successfully afterward. Counts from attempts are not added together.

This qualifies the additive CPU sampled parent bridge. RPI-022 remains open for shared disk/per-device/global queue authority, hard enforcement, gradual recovery, complete production call-site adoption, and per-parent validation capacity. The existing validation reserve is global: training may occupy its entire parent envelope and delay sibling validation. Native lease cleanup after process death is distinct from killing arbitrary untracked orphan processes.

`producer-sources.json` pins the tested bridge, its test file, and direct resource/process owners. `qualification.json` gives the exact scope. Neither leases serialized as receipts nor these test records confer task execution, completion, or proof authority.

Reproduce from the accelerate worktree with both release worktrees and the installed dependencies on `PYTHONPATH`, a fresh `IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB`, and:

```sh
python -m pytest -q -o log_cli=false test/api/test_repository_resource_bridge.py
```

Leave the datasets scheduler at its normal default state path and policy. The live tests can wait or refuse when the real host is busy.
