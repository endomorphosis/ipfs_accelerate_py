# LPC-150 Independent packaging and handshake scenarios

Hermetic required (no network, no optional solver install):

1. Install `ipfs_datasets_py` alone and import pure logic contracts.
2. Install `ipfs_accelerate_py` alone and import the supervisor package.
3. Install compatible released versions together and complete the
   `LogicPlatformManifest@1` handshake.
4. Install incompatible versions and receive a typed incompatibility result.
5. Run without sibling repository layout.
6. Run without Git metadata.
7. Run without optional solvers.

Installed-provider optional:

8. Run with one admitted local solver.

Network / OCI opt-in:

9. Run from an OCI or clean virtual environment.

Eliminate logic integration reliance on nested submodules, mutable main
dependencies, special relative paths, import-time repair, and arbitrary
`sys.path` modification. Worker worktrees now pin `ipfs_datasets_py` as
the datasets submodule rather than a sibling `external/ipfs_datasets` path.
