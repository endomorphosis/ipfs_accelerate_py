# PCPR-054 declared-release-profile SBOMs

These files are declared SPDX-2.3 SBOMs of the Accelerate *release*
profile. They list the root package and the PCPR-053 declared PEP 508
direct requires. They are not a live Syft/CycloneDX/Trivy scan, not a
hashed transitive graph, not a signed SLSA attestation, and not a
closed PCPR release.

- `cpython312/release.sbom.json` is the canonical declared SBOM.
- Hash, license-concluded, and transitive identities stay typed
  unavailable. Hashes are never invented. `filesAnalyzed` is false.
- Named extras `libp2p`, `mcp-p2p`, `ipfs-transformers`, and
  `ipfs-model-manager` may carry source-checkout Git pins and are not
  this SBOM.
- `platform-catalog.json` binds Datasets, Kit, and Accelerate SBOM and
  provenance identities when those sibling trees are present.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools are not sealed-environment authority.
