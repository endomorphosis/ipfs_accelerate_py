# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# linux/arm64 child manifest for python:3.12.3-slim-bookworm.  The tag is
# descriptive only; the digest is the immutable base identity.
FROM docker.io/library/python:3.12.3-slim-bookworm@sha256:c27c26153fdf6863da2a1d85b474d1be004c42cc8ea2fd647004a8d4007a34d5 AS proof-context-build

ARG PROFILE=core
ARG ARTIFACT_MANIFEST_PATH=environment/artifact_hashes.json
ARG ARTIFACT_ROOT_PATH=frozen-artifacts

ENV PATH=/opt/proof-context-venv/bin:/usr/local/bin:/usr/bin:/bin \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INDEX=1 \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1

WORKDIR /opt/ipfs-accelerate

# The repository contributes only the frozen locks and the gate.  Package
# source trees are never copied into this image.
COPY packaging/proof_context/locks/ packaging/proof_context/locks/
COPY scripts/proof_context/test_clean_install.py scripts/proof_context/test_clean_install.py

# The caller must provide this named BuildKit context explicitly:
#   --build-context proof_context_artifacts=/path/to/downloaded/bundle
COPY --from=proof_context_artifacts / /opt/proof-context-artifacts/

# PCCE-053 currently makes this command exit 5 for every profile, before pip is
# invoked.  Keeping the install commands behind that gate prevents an image
# build from silently repairing the lock with locally derived wheels.
RUN case "$PROFILE" in \
      core|verification) ;; \
      *) echo "container profile must be core or verification" >&2; exit 2 ;; \
    esac \
    && case "$ARTIFACT_MANIFEST_PATH" in \
      ""|/*|..|../*|*/..|*/../*) echo "unsafe artifact manifest path" >&2; exit 2 ;; \
    esac \
    && case "$ARTIFACT_ROOT_PATH" in \
      ""|/*|..|../*|*/..|*/../*) echo "unsafe artifact root path" >&2; exit 2 ;; \
    esac \
    && python -I scripts/proof_context/test_clean_install.py \
      --artifacts "/opt/proof-context-artifacts/$ARTIFACT_MANIFEST_PATH" \
      --artifact-root "/opt/proof-context-artifacts/$ARTIFACT_ROOT_PATH" \
      --profile "$PROFILE" \
      --output /tmp/proof-context-clean-install.json \
      --require-qualified \
    && python -m venv /opt/proof-context-venv \
    && /opt/proof-context-venv/bin/python -I -m pip install \
      --no-cache-dir \
      --no-index \
      --find-links "/opt/proof-context-artifacts/$ARTIFACT_ROOT_PATH" \
      --only-binary=:all: \
      --require-hashes \
      --requirement "packaging/proof_context/locks/cpython312-linux-aarch64/$PROFILE.txt" \
    && cd /tmp \
    && /opt/proof-context-venv/bin/python -I -c \
      "import importlib.metadata as m; import ipfs_accelerate_py.proof_context; [m.version(n) for n in ('ipfs-accelerate-py','ipfs-datasets-py','ipfs-kit-py','mcp-plus-plus-contracts')]" \
    && /opt/proof-context-venv/bin/proof-context --help >/dev/null

# The runtime stage receives only the qualified virtual environment. Frozen
# archives, locks, the harness, and its report remain outside the final image.
FROM docker.io/library/python:3.12.3-slim-bookworm@sha256:c27c26153fdf6863da2a1d85b474d1be004c42cc8ea2fd647004a8d4007a34d5 AS proof-context-runtime

ENV PATH=/opt/proof-context-venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1

COPY --from=proof-context-build /opt/proof-context-venv/ /opt/proof-context-venv/

# Numeric identity avoids relying on mutable distribution account databases.
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT ["/opt/proof-context-venv/bin/proof-context"]
CMD ["--help"]
