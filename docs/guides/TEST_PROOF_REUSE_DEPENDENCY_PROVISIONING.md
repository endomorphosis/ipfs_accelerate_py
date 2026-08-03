# Proof-reuse dependency provisioning

The accelerator keeps six different capability layers separate. Importing
`ipfs_accelerate_py`, its pytest plugin, or the lazy-installer module does not
import NLTK, download data, invoke pip/Cargo, connect to an endpoint, create a
cache, or generate cryptographic material.

| Layer | Provisioning | Missing-capability action |
| --- | --- | --- |
| Python packages (`nltk`, `jsonschema`, `multiformats`) | bounded pip allowlist; normal requirements metadata also declares them | `RUN` |
| exact datasets certificate verifier | closed reviewed Git blobs materialized directly into an owner-private content-addressed snapshot | `RUN` |
| NLTK tokenizer/tagger/chunker data | bounded `nltk.downloader` resource allowlist on explicit first use | `RUN` |
| datasets Groth16 binary | reviewed local datasets checkout plus `cargo build --locked --release` | `DEFERRED` |
| Groth16 endpoint | operator configuration only; it is never contacted by discovery | `DEFERRED` |
| Groth16 keys and circuit binding | operator/reviewed artifacts; trusted setup and circuit selection are never automatic | `DEFERRED` |

## Python runtime boundary

The accelerator's general package metadata remains Python 3.8+, but the
optional proof-reuse implementation currently uses Python 3.10-era type and
dataclass features. The exact reviewed datasets verifier is stricter and
requires Python 3.12+. Run the proof-reuse supervisor on Python 3.12 or newer.
On Python 3.10/3.11 the verifier capability resolves to typed `RUN` fallback
without provisioning; Python 3.8/3.9 environments should keep the optional
pytest plugin disabled (for example, with pytest entry-point autoload disabled)
until the proof-reuse compatibility floor is lowered.

## Policy and first use

Python installation requires both `IPFS_TEST_PROOF_REUSE_AUTO_INSTALL=1` and
`IPFS_ACCEL_AUTO_INSTALL=1` (the latter otherwise follows the virtualenv
default). The more consequential resource/native operations additionally
require their own explicit consent:

```bash
export IPFS_TEST_PROOF_REUSE_NLTK_DOWNLOAD=1
export IPFS_TEST_PROOF_REUSE_NLTK_DATA_DIR="$HOME/nltk_data"  # optional

export IPFS_TEST_PROOF_REUSE_GROTH16_BUILD=1
export IPFS_TEST_PROOF_REUSE_DATASETS_SOURCE=/reviewed/ipfs_datasets
export IPFS_TEST_PROOF_REUSE_PROVISION_DIR="$HOME/.cache/ipfs-proof-reuse"  # optional
```

After installing the package, operators may explicitly invoke the same bounded
lazy installer used by pytest:

```bash
# Installed console command. With no capability flags it requests both.
ipfs-accelerate-proof-reuse-provision

# Select one layer, or require a nonzero exit when it remains unavailable.
ipfs-accelerate-proof-reuse-provision --nltk-data
ipfs-accelerate-proof-reuse-provision --groth16-native --require-ready

# Equivalent source-checkout setuptools command (explicit only).
python setup.py proof_reuse_provision
```

Both commands emit one bounded JSON status document. They do not override the
four consent gates above: the two general auto-install policies and the
NLTK/Groth16-specific choice must still permit the requested operation. A
missing network, Cargo toolchain, reviewed source, cache, key, or circuit is a
typed `RUN`/`DEFERRED` result; it is nonzero only with `--require-ready`.

`pip install .`, `pip install '.[proof-reuse]'`, wheel/sdist construction and
package import never run either provisioner. NLTK is an ordinary declared
Python dependency, while its data is not. Groth16 is a reviewed Cargo-native
capability, not a package named `groth16` on PyPI. Native compilation never
runs trusted setup, and these commands never generate proving/verifying keys.

Call the narrow facade only when the capability is actually needed:

```python
from ipfs_accelerate_py import get_proof_reuse_bootstrap

bootstrap = get_proof_reuse_bootstrap()
nltk_status = bootstrap.ensure_nltk_data()
native_status = bootstrap.ensure_groth16_native_backend()
groth16_inputs = bootstrap.inspect_groth16_runtime()
```

`ensure_nltk_data()` accepts only the package identifiers returned by
`bootstrap.installer.dependency_plan()["nltk_data"]["resource_allowlist"]`.
The downloader runs in a bounded subprocess under thread and file locks.

The datasets verifier provisioner reads a closed 57-file manifest directly
from the exact reviewed Git commit, verifies every blob and the aggregate
digest, and atomically materializes it under an owner-private,
content-addressed target. It invokes no pip or VCS installer, executes no
package/build hook, initializes no submodule, performs no remote fetch, and
does not mutate global site-packages. A warm process revalidates the closed
snapshot and imports it without consulting Git or the source checkout.

The native provisioner materializes only the reviewed commit's `Cargo.toml`,
`Cargo.lock`, and Rust source blobs in an owner-private temporary tree, then
rehashes the snapshot under the build lock. It does not execute the datasets
`build.sh`, because that script also performs trusted setup. Cargo receives a
private `CARGO_HOME` and target directory plus a minimal environment with no
inherited Rust wrapper or target settings. It runs from the filesystem root so
mutable checkout/ancestor Cargo configuration is not discovered. Cargo has a
bounded timeout and runs with `--locked`; any source, toolchain, network, lock,
or build failure returns `DEFERRED` rather than aborting the real test.

Bundled binaries are accepted only when their reviewed digest and platform
both match the current host. A reviewed `linux-aarch64` binary on an x86_64
host is reported in `foreign_bundled_platforms`; it is never executed through
binfmt/QEMU as an implicit fallback. That host requires a native Cargo build or
an explicitly configured operator binary.

After a successful native build, the provisioner atomically copies the binary
to its owner-private cache and writes a receipt binding the exact reviewed
datasets revision and build-input fingerprint, native platform, fixed cache
path, and binary SHA-256. A later process reuses the binary only when that
receipt and digest still match; an unreceipted or modified build output is
never trusted. The validated path and reviewed artifacts root are activated
through `IPFS_DATASETS_GROTH16_BINARY` and
`GROTH16_BACKEND_ARTIFACTS_ROOT`. On Windows the executable is `groth16.exe`.

## Groth16 runtime inputs

Building a binary does not make a Groth16 issuer ready. Configure or inspect
these independently:

- `IPFS_TEST_PROOF_REUSE_GROTH16_ENDPOINT`: optional HTTP(S) proving endpoint.
- `IPFS_DATASETS_GROTH16_BINARY`: explicit operator-supplied native binary.
- `GROTH16_BACKEND_ARTIFACTS_ROOT`: versioned proving/verifying key root.
- `IPFS_TEST_PROOF_REUSE_GROTH16_CIRCUIT_REF`: an operator-reviewed test-pass
  v4 circuit binding, such as `test_pass@v4`; the generic
  `knowledge_of_axioms@*` family never grants test-certificate authority.

`inspect_groth16_runtime()` performs no network call or subprocess. Its
`ready=false` result identifies which binary/endpoint, keys, or circuit binding
is still missing. Key generation remains a separate security ceremony and is
never performed by dependency installation.

## Cache and store failures

An existing valid local proof cache/store is used atomically and requires no
installer. A missing, read-only, corrupt, or lock-contended cache/store yields
`RUN` (or `DEFERRED` for an explicitly external capability); it never turns
pytest collection into a failure. Dependency discovery and provisioning do not
create proof authority: a test is skipped only after exact identity,
certificate, provenance, and policy validation succeed.
