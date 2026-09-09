# PCPR-056 declared portfolio compatibility lock

These files are the *one* declared proof-carrying-platform-0.1.0
portfolio compatibility lock. They bind Accelerate, Datasets, and Kit
package versions, intended annotated tags, PCPR-053/054/055 artifact
CIDs, PCPR-040/041/042/043 contract and vector identities, the one
supported combination, hermetic candidate test suites, and named
qualification-receipt identities.

They are not a live GPG/SSH signature, not a freeze (PCPR-002), not
branch protection (PCPR-057), not a published wheel or sdist, and not a
closed PCPR release.

- `cpython312/release.lock.json` is the canonical declared lock.
  `lock` is true. Exact commit and tree are bound by the PCPR-056
  receipt `current_tree_binding` because nested admission rewrites HEAD.
  `origin/main` is not the release identity.
- `platform-catalog.json` binds Datasets and Kit lock-binding documents
  when those sibling trees are present. Sibling source is never required.
- Named extras `libp2p`, `mcp-p2p`, `ipfs-transformers`, and
  `ipfs-model-manager` may carry source-checkout Git pins and are not
  this lock.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools and generated signing keys are not
sealed-environment authority.
