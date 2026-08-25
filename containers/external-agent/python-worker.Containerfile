# EAAEF-052 python worker. Nonroot, no docker socket, digest-pinned at qualify time.
FROM debian:bookworm-slim
USER 65532:65532
ENTRYPOINT ["/usr/bin/true"]
