# The native builder verifies this cached tag's immutable config ID before and
# after its pull-disabled, network-disabled build. The wheel lock records that
# ID; this recipe does not consume the unavailable historical authority image.
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Legal IR ordinary task toolchain" \
      org.opencontainers.image.description="CPU-only host-toolchain-bound ordinary worker; no protected-route authority" \
      org.opencontainers.image.base.digest="sha256:95d16dfcd4ab8b61154e2280ab9fe4afa2ef03e5eefa8a689e86571e8a1b32cf" \
      org.ipfs-accelerate.ordinary-worker="2026-10-16-ordinary-v1" \
      org.ipfs-accelerate.worker-profile="ordinary-task-toolchain-v1"

COPY rootfs/ /

USER 65532:65532
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/env", "-i", "BASH_ENV=", "ENV=", "HOME=/opt/codex-home", "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "PATH=/opt/ipfs-task-tools/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "PYTHONPATH=/opt/ipfs-validation-site-packages", "TERM=dumb"]
CMD ["/bin/true"]
