# The build script resolves this local-only tag immediately before and after
# the network-disabled build and requires the immutable image ID below.  A
# Dockerfile syntax directive is intentionally absent because frontends may be
# fetched even when build-stage networking is disabled.
ARG BASE_IMAGE=ipfs-accelerate-authority-validation:20260803-v2
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="EAAEF bootstrap reconciliation worker" \
      org.opencontainers.image.description="Non-launching, network-denied bootstrap evidence verifier" \
      org.opencontainers.image.base.digest="sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6" \
      org.ipfs-accelerate.eaaef.profile="bootstrap-reconciliation-v1"

USER 65532:65532
WORKDIR /workspace

# The entrypoint clears the inherited image environment before exec.  Runtime
# admission separately requires the same closed environment and rejects host
# inheritance; these values are not credentials or authority.
ENTRYPOINT ["/usr/bin/env", "-i", "BASH_ENV=", "CODEX_HOME=/opt/codex-home", "ENV=", "HOME=/opt/codex-home", "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "PATH=/opt/ipfs-task-tools/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "PYTHONPATH=/opt/ipfs-validation-site-packages", "TERM=dumb"]
CMD ["/usr/bin/python3", "-I", "-S", "-B", "-c", "import json,os; print(json.dumps({'status':'ready_for_host_admission','uid':os.getuid(),'gid':os.getgid()},sort_keys=True,separators=(',',':')))" ]
