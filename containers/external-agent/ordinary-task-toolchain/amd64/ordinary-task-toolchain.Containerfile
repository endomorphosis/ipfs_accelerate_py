ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Ordinary AMD64 task toolchain" \
      org.opencontainers.image.description="CPU-only host-toolchain-bound ordinary worker; no protected-route authority" \
      org.opencontainers.image.base.digest="sha256:6277ab668b2ad3c3ab851d370faa1690d324b30db29d342527f61fed28f20f36" \
      org.ipfs-accelerate.ordinary-worker="2026-10-21-ordinary-amd64-v1" \
      org.ipfs-accelerate.worker-profile="ordinary-task-toolchain-amd64-v1"

COPY rootfs/ /

USER 65532:65532
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/env", "-i", "BASH_ENV=", "ENV=", "HOME=/opt/codex-home", "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "PATH=/opt/ipfs-task-tools/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "PYTHONPATH=/opt/ipfs-validation-site-packages", "TERM=dumb"]
CMD ["/bin/true"]
