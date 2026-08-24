# Proof-context dependency locks

These locks cover one tested resolver target only: CPython 3.12 on Linux
`aarch64`. There is no implied support for another Python implementation,
version, operating system, or architecture.

Each profile lock is generated from a fresh pip JSON metadata-resolution report
with installed packages ignored, the cache disabled, and
`https://pypi.org/simple` as the sole index. Resolution used the exact admitted
wheel bytes for datasets, kit, accelerator, and the MCP++ contract package.
Every direct-package lock entry binds both the admitted wheel and sdist hashes.
Seven of the eight admitted archives are byte-verified. The exact kit sdist
identity remains bound but its bytes are unavailable after a bounded provider
replay failed to reproduce archive PAX subsecond timestamps; no mismatched
reconstruction is substituted. This availability fact is independent of the
successful exact-wheel resolution.

The adjacent resolver receipt binds the raw pip report hash, selected archive
URL and hash, Core Metadata dependency/license declarations,
source-distribution build-system metadata, and observed runner. Raw reports are
operational evidence and are intentionally not committed because package
descriptions make them large; their SHA-256 identities are retained.

The generated outer evidence preserves every SHA-256 and also derives its
canonical raw CIDv1 (`raw` multicodec plus `sha2-256` multihash). CIDs for the
seven available package archives, every lock, every resolver receipt, and each
environment document are labeled `bytes-verified`. The unavailable admitted
kit sdist CID is derived from its frozen digest and is explicitly labeled
`identity-derived-bytes-unavailable`; it is never described as byte-verified.
The builder decodes every emitted CID and compares its multihash digest before
writing or checking evidence.

A lock can be resolved through pip's hash gate without installing it by
materializing the exact selected archive named for every receipt entry. The
direct-package subset needs the four available exact admitted wheels; the
unavailable kit sdist is bound as an alternate identity but is not selected:

```bash
python -m pip install \
  --dry-run --ignore-installed --require-hashes \
  --no-index --find-links /path/to/exact-selected-archives \
  --no-build-isolation \
  -r packaging/proof_context/locks/cpython312-linux-aarch64/core.txt
```

Replace `core.txt` with `verification.txt`, `codex.txt`, `local-model.txt`, or
`evaluation.txt` for the other profiles. Core, verification, and codex passed
this exact dry-run hash gate. Local-model and evaluation failed closed while
preparing the selected native sdist because `scikit_build_core` was deliberately
not introduced outside the lock. No extra index or unhashed archive is
authorized. The `codex` profile specifies the command-adapter contract but
does not install, authenticate, or claim availability of a Codex executable.

The typed support claim is deliberately split. All five profiles are
`supported-hash-bound` for dependency resolution on the recorded runner. Every
profile remains clean-install `no-go-sdist-builds-not-qualified` because pip
selected the `varint` and `python-baseconv` sdists and PCCE-053 does not qualify
source builds. `local-model` and `evaluation` additionally carry
`no-go-native-sdist-only` because `llama-cpp-python` has no selected aarch64
wheel. No successful sdist or native build, server, model download, inference
run, or Codex availability is claimed. PCCE-054 inherits these clean-install
and native-build no-go dispositions.

The canonical builder verifies these files and materializes the outer evidence:

```bash
python external/ipfs_accelerate/scripts/proof_context/build_environment_manifest.py --check
```
