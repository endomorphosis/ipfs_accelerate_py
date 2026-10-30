# Ordinary AMD64 task worker

This separately qualified profile supports the native AMD64 host. The existing
ARM64 ordinary profile and protected-route image identities remain separate.
The runner selects an exact source-approved image/platform/label from the host
architecture; environment-selected images and foreign architectures fail closed.

The frozen nine-wheel Python 3.12 closure is in `wheels.lock.json`. The stager
reuses the bounded wheel SHA256/RECORD verifier, with separate AMD64 lock/base
constants. It also extracts only `rg` and its copyright notice from a frozen
Ubuntu package. No wheel installer or Debian maintainer script executes.
`rg` lives in the image's `/opt/ipfs-task-tools/bin`; no host application directory
or host `/usr` write is needed. Its package and binary hashes are recorded in
`evidence/20261021/ripgrep-provenance.json`.

Build from an owned temporary directory:

```sh
python3 -I -B stage_ordinary_worker_context.py --output "$TMPDIR/staged-worker"
docker --host=unix:///var/run/docker.sock build --pull=false --network=none \
  --no-cache --force-rm --platform=linux/amd64 --build-arg BASE_IMAGE=ubuntu:24.04 \
  --tag ipfs-accelerate-ordinary-task-toolchain:20261021-amd64-v1 \
  --file "$TMPDIR/staged-worker/context/Containerfile" "$TMPDIR/staged-worker/context"
```

Verify the local Ubuntu base ID against `image-profile.json` before and after the
build. Docker's mutable tag alone grants no execution authority. A rebuilt image
with a different ID needs runtime qualification and an explicit source pin update.

The actual image passed seven checks in a network-disabled, read-only,
nonroot container with capabilities dropped. Checks covered all 692 Python
package files, nine package versions, native Ed25519/DuckDB behavior, denied
writes to protected roots, Python/Git/image-native-rg startup, and two isolated
subprocess pytest cases. The checker uses the source-frozen AMD64 lock and an
explicit expected nonroot UID (`--architecture amd64 --expected-uid 1003`).

Small qualification records and immutable IDs are committed under
`evidence/20261021` and `image-profile.json`. Large wheel/context artifacts remain
in the local artifact directory named by the profile. The image was not published
to a registry. Logical image size does not measure physical Docker storage.
This evidence qualifies the ordinary mounted toolchain; it does not establish
Grok authentication/model access, production privacy, or complaint completion.
