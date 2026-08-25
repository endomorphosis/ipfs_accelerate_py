# This is a distinct, closed minimal-image experiment.  Its qualification
# script binds both local source image IDs, constructs the sole build-context
# archive, and performs two pull-disabled/network-disabled clean builds.  Do
# not add a syntax directive: that could fetch a frontend.
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG CODEX_SHA256
ARG GROK_SHA256
ARG INPUT_MANIFEST_SHA256
ARG ROOTFS_TAR_SHA256
ARG SOURCE_DATE_EPOCH
ARG TOOL_SOURCE_IMAGE_ID
ARG VALIDATION_TOOLCHAIN_SHA256

LABEL org.opencontainers.image.title="EAAEF minimal implementation worker candidate" \
      org.opencontainers.image.description="Closed, unsigned, zero-capacity minimal offline candidate" \
      org.opencontainers.image.base.digest="sha256:ea17ec341c4211d1dd7f184a0dedf7dcb7945e92db20a5dde20544262214b84f" \
      org.ipfs-accelerate.eaaef.profile="implementation-worker-minimal-candidate-v1" \
      org.ipfs-accelerate.eaaef.tool-source.digest="${TOOL_SOURCE_IMAGE_ID}" \
      org.ipfs-accelerate.eaaef.codex.sha256="${CODEX_SHA256}" \
      org.ipfs-accelerate.eaaef.grok.sha256="${GROK_SHA256}" \
      org.ipfs-accelerate.eaaef.input-manifest.sha256="${INPUT_MANIFEST_SHA256}" \
      org.ipfs-accelerate.eaaef.rootfs-tar.sha256="${ROOTFS_TAR_SHA256}" \
      org.ipfs-accelerate.eaaef.python-validation-toolchain.sha256="${VALIDATION_TOOLCHAIN_SHA256}" \
      org.ipfs-accelerate.eaaef.source-date-epoch="${SOURCE_DATE_EPOCH}" \
      org.ipfs-accelerate.eaaef.worker-capacity="0" \
      org.ipfs-accelerate.eaaef.unsigned="true"

# The archive contains only the manifest-bound tool/runtime closure.  It was
# copied from one exact local image without starting that source container;
# source discovery is a separate inert, network-denied diagnostic.
ADD worker-rootfs-minimal.tar /

USER 65532:65532
WORKDIR /workspace

# Clear both inherited and host-provided environment.  The allowlisted shared
# libraries live outside the base system directories.
ENTRYPOINT ["/usr/bin/env", "-i", "BASH_ENV=", "CODEX_HOME=/opt/codex-home", "ENV=", "GIT_CONFIG_NOSYSTEM=1", "HOME=/opt/codex-home", "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "LD_LIBRARY_PATH=/opt/eaaef/lib", "PATH=/opt/eaaef/bin:/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", "TERM=dumb"]
CMD ["/usr/bin/python3", "-I", "-S", "-B", "-c", "import json,os;print(json.dumps({'gid':os.getgid(),'status':'closed_unsigned_minimal_candidate','uid':os.getuid(),'worker_capacity':0},sort_keys=True,separators=(',',':')))"]
