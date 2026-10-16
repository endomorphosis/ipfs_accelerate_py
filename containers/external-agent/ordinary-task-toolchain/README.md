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

Ordinary-route integration tests and abrupt-launcher cleanup qualification are
recorded separately after they run. This image qualification does not establish
provider/model access, Legal IR task completion, training accuracy or a Lean build.
No provider credential is embedded in the image or build context.
