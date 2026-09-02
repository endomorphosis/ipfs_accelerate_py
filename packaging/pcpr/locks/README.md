# PCPR-053 declared release-profile dependency locks

These files freeze the Accelerate *release* profile as declared PEP 508
specs from `requirements.txt`. They are not a hashed pip-tools / uv lock,
not a live PyPI resolution, and not a closed PCPR release.

- `cpython312/release.lock.json` is the canonical declared-spec lock.
- `cpython312/release.txt` is the pip-installable declared-spec list.
- Hash and transitive resolution remain typed unavailable until a live
  sealed-PATH resolver is admitted. Hashes are never invented.
- Named extras `libp2p`, `mcp-p2p`, `ipfs-transformers`, and
  `ipfs-model-manager` may carry source-checkout Git pins and are not
  this release lock.
- `platform-catalog.json` binds Datasets, Kit, and Accelerate lock
  identities when those sibling trees are present beside this checkout.

Sealed validation PATH is exactly `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin`.
Provider-local `~/.local` tools such as `uv` are not sealed-environment
authority.
