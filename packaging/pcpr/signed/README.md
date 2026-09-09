# PCPR-055 declared signed tags and artifacts

These files are the Accelerate *release* signed-tag policy and artifact
checksum manifest. They are not a live GPG/SSH git tag, not a cosign or
minisign signature, not a published wheel or sdist, and not a closed
PCPR release.

- `cpython312/release.tag-policy.json` is the canonical declared tag
  policy. The intended annotated tag name is `ipfs_accelerate_py-v0.0.45`.
  Exact commit and tree are bound by the PCPR-055 receipt
  `current_tree_binding` because nested admission rewrites HEAD.
  `origin/main` is not the release identity.
- `cpython312/release.checksums.json` and `cpython312/SHA256SUMS` checksum
  the committed PCPR-053 lock and PCPR-054 SBOM/provenance files. Wheel,
  sdist, container, and signature identities stay typed unavailable.
  Hashes and signatures are never invented.
- `platform-catalog.json` binds Datasets, Kit, and Accelerate signed-tag
  identities when those sibling trees are present.
- Named extras `libp2p`, `mcp-p2p`, `ipfs-transformers`, and
  `ipfs-model-manager` may carry source-checkout Git pins and are not
  this signed release.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools and generated signing keys are not
sealed-environment authority.
