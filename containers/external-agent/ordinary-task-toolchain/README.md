# Ordinary ARM64 task worker

This image replaces the unavailable historical image for the ordinary Grok/Codex
compatibility route. It has its own immutable source pin and version label.
Protected-route image and receipt checks keep their original identity.

The image is based on verified Ubuntu 24.04 ARM64 and contains nine hash-locked
validation packages under `/opt/ipfs-validation-site-packages`. The ordinary
runner supplies trusted host `/usr`, CA certificates and Python through explicit
read-only mounts, and mounts the Grok/Codex executables separately. These are
required runtime inputs, not files copied from a home or credential directory.
`python3` forwards through a regular launcher; staging creates no symlinks.

`stage_ordinary_worker_context.py` downloads the exact wheels in `wheels.lock.json`,
checks their sizes/SHA256 and complete wheel RECORD entries, rejects unsafe paths
and Python startup hooks, and writes a closed deterministic build-context archive,
installed-file inventory and SPDX SBOM. It executes no wheel code or installer.
Invoke it only within an actual admitted native build owner, with a fresh output
below that owner's canonical TMPDIR:

```sh
python3 -I -B stage_ordinary_worker_context.py --output "$TMPDIR/staged-worker"
docker --host=unix:///var/run/docker.sock image build --pull=false --network=none \
  --no-cache --force-rm --build-arg BASE_IMAGE=ubuntu:24.04 \
  --file Containerfile --tag ipfs-accelerate-ordinary-task-toolchain:20261016-v1 \
  - < "$TMPDIR/staged-worker/build-context.tar"
```

The owner must verify cached `ubuntu:24.04` resolves to the base image in
`image-profile.json` before and after construction. This recipe has no RUN
instruction. A rebuilt image must be qualified and source-pinned; a mutable tag
or caller environment cannot select an execution image.

The actual local build on 2026-10-16 produced:

```
ipfs-accelerate-ordinary-task-toolchain:20261016-v1
sha256:a91a42c918514203bb6cfc7c38b320dfb68e9dfc0571bf01976c0db993e24e5e
```

Docker reports 176,494,039 logical bytes. Physical daemon growth is not measured
by this value. Native named-root storage covers staging and evidence; the retained
Docker allocation is separately recorded, with no hard daemon disk quota claim.

A real network-disabled, read-only, nonroot container with the declared host
mounts verified all 692 installed package files and all nine package versions,
Python/Git/ripgrep startup, native Ed25519 signature validation, in-memory DuckDB
queries, denied writes to protected roots, and two passing subprocess pytest
cases. The independent legacy watchdog returned 0; the native parent separately
verified container absence and watchdog disappearance before finalization.
`qualify_runtime.py` is the container-side checker and requires a read-only
`/opt/installed-files.json` input plus an owned writable `/workspace`.

All 28 runner integration and existing confinement tests passed against commit
`93cfd32f65cd231df57e53736edf0f78b58af009`. Actual image inspection accepted the
new immutable identity and the protected image pin remained unchanged. Separate
real-container controls passed for normal cleanup (watchdog return code 0) and
an abruptly killed owned launcher (child return code -9, container, lease root
and watcher gone). The abrupt case cannot observe the watcher's return code;
it records absence and does not invent an exit status. Native owners completed
and finalized after their container/process absence checks.

Small qualification records and test results are committed in
[`evidence/20261016`](evidence/20261016); [`image-profile.json`](image-profile.json)
records the exact archive and evidence hashes. The local reloadable Docker
archive is 179,565,056 bytes, with SHA256
`9fced6382d3ad13f16c17aea382cdca43f20969c1d887d9fb16623a9b623ee61`, at:

```
/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-worker-image-20261016-integration02/ordinary-worker-arm64.docker.tar
```

To restore that archive on an ARM64 system daemon, verify its SHA256, then run
`docker --host=unix:///var/run/docker.sock image load --input PATH_TO_ARCHIVE`
and inspect the restored immutable identity and label before execution. The
large archive and downloaded wheels are retained locally, not committed to Git
or published to a container registry.

Earlier failed attempts remain separate local records: one storage lock wait,
one staging symlink rejection, and one pytest fixture symlink rejection. The
successful staging uses a regular launcher; the successful regression gives
pytest a separately charged fixture directory outside the strict artifact tree.
None of the failed attempts were relabeled as successful or used as completion
authority.

This qualification does not establish provider/model access, Legal IR task
completion, training accuracy or a Lean build. No Legal IR supervisor or provider
task was started during image qualification. No provider credential is embedded
in the image or build context.
