# Source384 assets and common Harbor resources

Source384 shared-parent assets are an explicit offline selection. The runtime
config owner is
`ipfs_accelerate_py.agent_supervisor.runtime.source384_config`; deployment and
native consumers share its closed schema and validators. The config contains:

```json
{
  "schema": "terminal-source384-config@1",
  "mode": "pinned_parent",
  "checkpoint_path": "/absolute/local/checkpoint.json",
  "checkpoint_sha256": "<exact 64-character SHA256>",
  "embedding_snapshot": "/absolute/models--thenlper--gte-small/snapshots/<native pinned revision>",
  "embedding_revision": "<native pinned revision>",
  "embedding_assets": [{"name": "<native asset name>", "sha256": "<exact SHA256>", "bytes": 1}],
  "training_steps": 0,
  "download_calls": 0
}
```

The example is schematic: the complete ordered embedding manifest must equal
the datasets module's native GTE pins. `load_source384_config` checks bounded
regular checkpoint bytes through the existing Security Source384 v2
compatibility loader and verifies the local embedding snapshot. It preserves
the original checkpoint, including its exact SHA256. It performs no training,
inference or downloads.

Pass `--source384-config` to `terminal_deployment bundle`. The archive stores
regular copies under `models/source384/`, with fixed relocated paths and a
config hash. Archive verification checks every selected member before upload.
Source384 selection excludes legacy SecurityIR training, checkpoint, formula
and header selections. Intent384 remains an independent compatible selection.

For repeated container setup, `terminal_deployment bundle` also accepts a local
CPU PyTorch wheel with an independent SHA256:

```text
--torch-cpu-wheel /absolute/path/to/torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl
--torch-cpu-wheel-sha256 6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604
```

This example pin is for CPython 3.12 on Linux aarch64, from the
[official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/).
The explicit option also supports the corresponding x86_64 filename with its
own independently verified digest. Packaging checks the bounded regular file,
wheel metadata and archive binding; deployment rechecks the extracted bytes
and interpreter architecture before installing that exact local wheel. A
failed selected-wheel check stops deployment. The wheel does not enable a
model runtime that was not already selected.

This removes the large wheel download from each fresh container. PyTorch's
dependencies and other runtime packages still use their configured package
indexes, and the existing 600-second installation bound remains unchanged.
Omitting both options retains the ordinary pinned online installation.

Managed Python installs under `/opt/ipfs-supervisor/python-runtime/python`,
separate from the shipped `/opt/ipfs-supervisor/toolchains/lean` assets. Both
UV installation and virtual-environment creation use that managed root;
the runtime interpreter remains `/opt/ipfs-supervisor/venv/bin/python`.
This separation prevents UV 0.9.24's legacy sibling-`toolchains` migration
from relocating Lean. The [layout qualification](SOURCE384_DOCKER_QUALIFICATION.md#managed-python-and-lean-layout-collision)
records the reproduced collision and native offline regression control.

For `full_supervisor_benchmark prepare`, the optional `--source384-config`
rechecks the selected local config against the archive. A full arm containing
Source384 requires `--resource-profile source384-5cpu-12gib@1`. Select that same
resource profile for `native_codex_baseline prepare` and the no-index arm.
It declares 5 CPUs and 12288 MiB with Harbor CPU and memory limit enforcement;
the existing default retains the task's original 1 CPU/2048 MiB settings.
Common comparison controls include these limits and reject a declared profile
whose config contradicts them. The larger profile is needed to accommodate the
6144 MiB/3-slot managed parent while preserving the shared scheduler's 20%
headroom and separate validation slot. Actual admission still depends on live
resource observations.

The full Harbor adapter forwards the relocated `--source384-config`. The
no-index adapter records the packaged selection as disabled and never forwards
it. Selecting or transporting weights alone does not establish repository
coverage, proof authority, container admission, or an improved benchmark score.

The transport qualification uses explicit tiny asset and loader doubles. Its
transport tests cover config/asset rejection, archive tampering, actual
Harbor adapter argument forwarding, normalized three-arm declared controls,
and existing transport/deployment behavior. They invoke no model, provider or
benchmark verifier. Source384 inference, full supervisor operation and actual
container enforcement need their separate native qualifications.

The retained original `fix-code-vulnerability` Bottle file has 358 functions
and no parameter annotations. Current scalar Source384 source qualification
requires two explicit integer parameters; extracting function spans alone
cannot make those functions source-qualified. Unsupported coverage must remain
visible. No authored scalar cohort may be inserted into a benchmark checkout.

The optional original-container qualification now accepts the same named
profile and can run preparation before any planner/provider call:

```bash
python -m benchmarks.agent_supervisor.container_coding.terminal_deployment qualify \
  --task-dir /absolute/path/to/fix-code-vulnerability \
  --archive-dir /absolute/path/to/source384-runtime-bundle \
  --output /absolute/path/to/fresh-qualification \
  --resource-profile source384-5cpu-12gib@1 --source384-context --no-codex
```

This checks cgroup-v2 CPU quota and memory limit against 5 CPUs/12288 MiB,
then uses the packaged, relocated checkpoint to prepare the complete signed
original source population and create its initial context. It verifies native
inference, exact archive producer pins, and one warm observation without neural
replay. The probe has a 270-second bound; original population, byte, selection,
resource and inference bounds remain enforced. Receipts retain a refusal as a
failure. The default `qualify` path remains unchanged when these flags are absent.
The initial-context path does not run the official verifier, a coding provider,
training, or Lean; setup may download ordinary pinned runtime dependencies.
It supplies no task score or token comparison. The selected Source384 model's
offline setting does not claim that container installation is offline.
See [the Docker qualification record](SOURCE384_DOCKER_QUALIFICATION.md) for
the retained 8 GiB admission refusal and the explicit 12 GiB profile.
