# Broader Terminal-Bench task profiles

The native supervisor runner accepts `--task NAME` and a source-bound public
`--task-profile PROFILE.json`. The original Bottle profile remains the default.
A profile declares exact existing inputs and files the task may create or modify.
It binds the normalized public instruction by SHA-256; the signed preparation
rechecks the profile, source inventory, generated structural validation and task
specification before planning and admission.

Example profile for the original `headless-terminal` public task:

```json
{
  "schema": "terminal-public-task-profile@1",
  "instruction_sha256": "004ac3efa98dd4c564f3888fe199962772d23de405c96ec9232c1eab1a2089e8",
  "input_paths": ["base_terminal.py"],
  "outputs": [
    {"path": "headless_terminal.py", "effect": "create", "media_type": "text/x-python"}
  ]
}
```

The profile grants bounded file operations, not an interpretation or proof of the
natural-language requirements. Its fixed validation checks regular output files,
size limits and Python syntax without running candidate code. The independent
Terminal-Bench verifier determines correctness.

```bash
python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark prepare \
  --dataset DATASET --task headless-terminal --task-profile PROFILE.json \
  --archive GENERIC_SOURCE384_ARCHIVE --output FRESH_OUTPUT --arm full \
  --resource-profile source384-5cpu-16gib-extended@1
python -m benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark execute \
  --output FRESH_OUTPUT --task headless-terminal
```

Build a new archive with generic `terminal-source384-config@1` for these tasks.
The previous `@2` header-applicability selection and reviewed Bottle intent
contract are task-specific. The generic route invokes the symbolic doctor with
no header contract; unsupported repairs become explicit residual work through
`ipfs_accelerate_py.llm_router`. The current comparison route pins `codex_cli`, model
`gpt-6.1-sol`, high reasoning and CLI `0.160.0`. The retained three-task pilot
used `gpt-5.6-sol` and CLI `0.158.0`; changing the model or CLI starts a
different experiment. This
container profile does not yet deploy the general supervisor's Grok route.
Runtime archives must carry the current CLI pin. The optional
`source384-native-aarch64-dontneed@1` setup-cache policy remains bound to
CLI `0.158.0` and its original binary hashes. Use the separately pinned
`source384-native-aarch64-dontneed@2` policy with CLI `0.160.0`; select it
identically during archive bundling and preparation. Cache advice remains
optional and grants no resource-admission authority. Planning and coding usage both
count. The worker permits at most 300 seconds per coding provider call; this is
separate from the 840-second supervisor work budget and 60-second cleanup reserve.
Missing token usage stays unknown, and no hard dollar/token ceiling is enforced.

The setup adapter records exact public `/app` inputs and initializes Git only in
the disposable task container when needed. It installs UV in an isolated harness
virtual environment, preserving the original task interpreter. Inputs may not
be links or special files. Native source/scan limits remain enforced: at most 252
original files, 262144 bytes per scanned file and the existing 4 MB source-total
bound. The nonempty vector route requires supported source and at most 64 files.
An explicit empty or zero-symbol retrieval population is available, but selecting
Source384, legacy security inference or local autoencoder training with that
population still refuses preparation until independent decoder abstention is
bound. Empty-context and no-index configurations need their own task qualification.

Other limitations remain explicit: workers cannot install system-wide packages,
files outside `/app` are not covered, write effects are exact create/modify paths,
there is one worker and one native attempt, and unsupported AST/index populations
fail rather than silently disappearing. Generic runs do not train on benchmark
inputs. No-index/native baselines require matched profiles before comparison.

Terminal-Bench 2.0 revision `2fd12b88aafdd04a52c298e3940bcb189f9766d6`
contains 89 tasks; 88 remain after the earlier Bottle task. Their native agent
budgets sum to 41.29 hours, excluding setup and verification. An extended 5 CPU /
16 GiB / 840-second work pilot is a distinct resource/time profile, not an official
full-suite native-budget score. Unprofiled and unsupported tasks must stay in
readiness inventories with unknown rewards; they are not executed failures.

The first [three-task full supervisor pilot](../../../docs/agent_supervisor/evidence/terminal-suite-pilot-20261004/README.md) records two official passes and one native-execution timeout, with token accounting and an 88-task readiness inventory. Its Docker source revision remains distinct from the later publication reconciliation.

The subsequent [supervisor recovery trial](../../../docs/agent_supervisor/evidence/terminal-supervisor-blockers-20261005/README.md)
passes `largest-eigenval` with CLI `0.160.0` and `gpt-6.1-sol` through
`llm_router`, consuming 291432 observed tokens. It verifies actual Source384
inference and native completion, and retains the remaining suite blockers.
It has no matched native or no-index baseline and establishes no efficiency advantage.
