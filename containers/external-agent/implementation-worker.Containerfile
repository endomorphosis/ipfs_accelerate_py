# This image is built only from a script-owned temporary context.  The script
# checks this local tag's immutable ID before and after a network-disabled,
# pull-disabled build.  Do not add a syntax directive: a frontend fetch would
# violate the offline qualification boundary.
ARG BASE_IMAGE=ipfs-accelerate-authority-validation:20260803-v2
FROM ${BASE_IMAGE}

ARG CODEX_SHA256
ARG GROK_SHA256
ARG INPUT_MANIFEST_SHA256
ARG ROOTFS_TAR_SHA256
ARG SOURCE_DATE_EPOCH

LABEL org.opencontainers.image.title="EAAEF implementation worker candidate" \
      org.opencontainers.image.description="Closed, unsigned, zero-capacity offline qualification candidate" \
      org.opencontainers.image.base.digest="sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6" \
      org.ipfs-accelerate.eaaef.profile="implementation-worker-candidate-v1" \
      org.ipfs-accelerate.eaaef.codex.sha256="${CODEX_SHA256}" \
      org.ipfs-accelerate.eaaef.grok.sha256="${GROK_SHA256}" \
      org.ipfs-accelerate.eaaef.input-manifest.sha256="${INPUT_MANIFEST_SHA256}" \
      org.ipfs-accelerate.eaaef.rootfs-tar.sha256="${ROOTFS_TAR_SHA256}" \
      org.ipfs-accelerate.eaaef.source-date-epoch="${SOURCE_DATE_EPOCH}" \
      org.ipfs-accelerate.eaaef.worker-capacity="0" \
      org.ipfs-accelerate.eaaef.unsigned="true"

# The pinned base is GPU-capable and advertises all NVIDIA devices.  This
# candidate is CPU-only unless a later, independently admitted profile creates
# a different image and resource contract.
ENV NVIDIA_VISIBLE_DEVICES=void \
    NVIDIA_DRIVER_CAPABILITIES=""

# The script pre-verifies this sole context input as a canonical USTAR archive
# with an exact closed member set, numeric 65532 ownership, fixed modes and
# SOURCE_DATE_EPOCH mtimes.  Post-build qualification independently re-hashes
# and stats every extracted file.  No build process or package installer runs.
ADD worker-rootfs.tar /

USER 65532:65532
WORKDIR /workspace

# The default process is diagnostic only.  It clears the inherited environment
# and cannot dispatch work, contact a provider, or start a supervisor.
ENTRYPOINT ["/usr/bin/env", "-i", "BASH_ENV=", "CODEX_HOME=/opt/codex-home", "ENV=", "HOME=/opt/codex-home", "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "PATH=/opt/eaaef/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "TERM=dumb"]
CMD ["/usr/bin/python3", "-I", "-S", "-B", "-c", "import json,os;print(json.dumps({'gid':os.getgid(),'status':'closed_unsigned_candidate','uid':os.getuid(),'worker_capacity':0},sort_keys=True,separators=(',',':')))"]
