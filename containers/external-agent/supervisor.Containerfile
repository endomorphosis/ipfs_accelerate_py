# EAAEF-052 supervisor image. Digest/SBOM/arch are qualification outputs; tags are not authority.
FROM debian:bookworm-slim
USER 65532:65532
ENTRYPOINT ["/usr/bin/true"]
