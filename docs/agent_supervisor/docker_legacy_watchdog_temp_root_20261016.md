# Legacy Docker cleanup preserves its temporary root

The legacy Grok/Codex cleanup watchdog discarded `TMPDIR` when starting its
independent process. If the caller used an admitted temporary directory other
than `/tmp`, the watchdog compared its lease, private home and prompt paths to
`/tmp` and refused them. Normal cleanup could hide the loss of independent
cleanup protection.

`_LegacyDockerContainerLease.create()` now passes the exact allocated lease
parent as `TMPDIR` in its otherwise minimal Docker control environment. The
existing root and path-prefix checks remain in effect. This does not pass
provider credentials or Docker context/TLS variables to the cleanup process.

## Actual Docker validation on 2026-10-16

Both existing system and rootless daemons responded successfully on ARM64.
Neither held `ubuntu:24.04` or the required authority toolchain image. The system
endpoint was used because the ordinary supervised Grok/Codex route binds it.

The official Ubuntu 24.04 ARM64 manifest and configuration were downloaded,
SHA-256 verified, and assembled into a Docker archive under an admitted native
staging owner. Compressed layer and uncompressed `diff_id` hashes were checked
before import. The downloaded layer was 28,940,087 bytes compressed and
103,277,568 bytes uncompressed. System Docker retained `ubuntu:24.04` as:

```
manifest: sha256:08571ca13e00ca07a2a84eab83a959b4242e22cceb16486a11bef1428c9e93a7
image:    sha256:95d16dfcd4ab8b61154e2280ab9fe4afa2ef03e5eefa8a689e86571e8a1b32cf
```

A real container used the production legacy cleanup lease with its temporary
paths inside the admitted staging directory. It had no network or host mounts,
a read-only root, dropped capabilities, non-root UID, 0.5 CPU, 128 MiB memory,
and a 16-process limit. It was observed running, exited with code 0, and was
removed. The worker reaped the independent watchdog; the outer native owner
separately verified the same daemon, exact container absence and watchdog birth
identity disappearance before finalizing native storage/execution custody.

```
container: 0bb9069c4aaf1cb5861128830d0a786e2bd846f34e2c0288ac09aa8377bfaf29
```

This is an actual Docker lifecycle check, not a Legal IR task, provider/model
execution, training result, semantic evaluation or Lean build. No other daemon
was restarted and no foreign container/image was deleted.

## Retained evidence and remaining startup requirement

Local execution evidence is retained in
`external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-docker-start-20261016-preparation01/`:

- `receipt.json`: native root/phase/disk identities and parent cleanup fence.
- `docker-smoke-outcome.json`: actual container ID, successful exit and removal.
- `docker-allocation.json`: verified manifest/configuration and retained image.
- `ubuntu-arm64.docker.tar`: verified import archive.

Native named-root storage covers the staged archive and receipts. Docker image
storage is separately recorded and remains retained at `/var/lib/docker`.
Docker's 100,805,066-byte `.Size` is its logical image size, not the physical
storage delta. No hard daemon disk quota or shared-ledger scope expansion is
claimed.

The requested Legal IR Grok 4.7/Codex GPT-6.1-Sol route still requires the exact
custom toolchain image, including its expected authority label:

```
ipfs-accelerate-authority-validation:20260803-v2
sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6
```

That local configuration/image ID is absent from both engines. No verifiable
registry reference or original image archive has been found. A reference whose
manifest configuration matches this ID, or the original Docker/OCI archive, is
needed to restore it. The fixed toolchain check remains intact.

## Post-acquisition checks and qualification limit

The fresh `readiness03` native run completed and finalized. The actual production
gates returned `ubuntu_isolation_ready=true` and `codex_toolchain_ready=false`.
The custom toolchain remained absent on both daemons.

An additional regression was prepared to require watchdog return code 0 and
exercise cleanup after killing an actual owned launcher. Its first generation
failed before any container effect because an optional image-template field was
unavailable. The corrected generation was not admitted: the existing native
CPU pressure guard recorded `proof_cpu_stall`, and the five-minute admission wait
ended in `LeaseTimeoutError`. This regression is not qualified. The successful
initial smoke check establishes observed container lifecycle and final cleanup;
it does not substitute for the additional crash regression. No shared admission
threshold, resource cap or foreign lease was changed.
