# EAAEF-052 prover image. Toolchain versions recorded at qualify time, not via mutable tags.
FROM debian:bookworm-slim
USER 65532:65532
ENTRYPOINT ["/usr/bin/true"]
