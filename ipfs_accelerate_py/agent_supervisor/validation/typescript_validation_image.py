"""Immutable TypeScript 5.9.3 validation-image contract.

The same local image is the sealed base for the Terra fallback container and
for host-dispatched authority validation.  Toolchain bytes live in the image;
neither runtime may bind a host compiler tree into the container.
"""

from __future__ import annotations

from typing import Final

TYPESCRIPT_VALIDATION_BASE_IMAGE: Final = (
    "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"
)
TYPESCRIPT_VALIDATION_IMAGE: Final = (
    "sha256:49cbd21cd2fee1fce22de57d11e7380071dbd76ac76ccc630e596488b947ff93"
)
TYPESCRIPT_TOOLCHAIN_ROOT: Final = "/opt/ipfs-validation-toolchains"
TYPESCRIPT_TOOLCHAIN_BIN: Final = f"{TYPESCRIPT_TOOLCHAIN_ROOT}/bin"
TYPESCRIPT_NODE_ROOT: Final = f"{TYPESCRIPT_TOOLCHAIN_ROOT}/node-v18.19.1"
TYPESCRIPT_NODE_BINARY: Final = f"{TYPESCRIPT_NODE_ROOT}/bin/node"
TYPESCRIPT_NODE_LAUNCHER: Final = f"{TYPESCRIPT_TOOLCHAIN_BIN}/node"
TYPESCRIPT_NODE_VERSION: Final = "v18.19.1"
TYPESCRIPT_NODE_SHA256: Final = (
    "2b0f6efd95c31c5538cc0a9042d5d13b7328cffcfdcc409f2e2ef336c4402086"
)
TYPESCRIPT_PACKAGE_ROOT: Final = f"{TYPESCRIPT_TOOLCHAIN_ROOT}/typescript-5.9.3"
TYPESCRIPT_NODE_MODULES: Final = f"{TYPESCRIPT_PACKAGE_ROOT}/node_modules"
TYPESCRIPT_COMPILER_JS: Final = (
    f"{TYPESCRIPT_NODE_MODULES}/typescript/lib/typescript.js"
)
TYPESCRIPT_PACKAGE_JSON: Final = f"{TYPESCRIPT_NODE_MODULES}/typescript/package.json"
TYPESCRIPT_VERSION: Final = "5.9.3"
TYPESCRIPT_COMPILER_SHA256: Final = (
    "3ae902c92cc44dace175c0e69e13a4b0899f6983c6121d76b9ab8dd5795e7675"
)
TYPESCRIPT_PACKAGE_SHA256: Final = (
    "822ef7ca6452205657b6288b066481ecf508bfbf43455d715cf7d3ec457561e6"
)
TYPESCRIPT_TOOLCHAIN_MANIFEST: Final = f"{TYPESCRIPT_TOOLCHAIN_ROOT}/manifest.json"
TYPESCRIPT_TOOLCHAIN_MANIFEST_SHA256: Final = (
    "d4acf45cef75ad49bb2742c2b27133dc2458858fe7876248e8310dbff5cc3f68"
)
TYPESCRIPT_CANARY: Final = f"{TYPESCRIPT_PACKAGE_ROOT}/canary.js"
TYPESCRIPT_CANARY_COMMAND: Final = f"{TYPESCRIPT_NODE_LAUNCHER} {TYPESCRIPT_CANARY}"
TYPESCRIPT_CANARY_SOURCE_SHA256: Final = (
    "6ff2ea1bc28c01fb19d18048da769c147109859d6a2e6d8f8921288d6775afbe"
)
TYPESCRIPT_TOOLCHAIN_ASSET_COUNT: Final = 28


def typescript_validation_toolchain_contract() -> dict[str, object]:
    """Return the JSON-safe immutable toolchain identity."""

    return {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor." "typescript-validation-toolchain@1"
        ),
        "image_id": TYPESCRIPT_VALIDATION_IMAGE,
        "base_image_id": TYPESCRIPT_VALIDATION_BASE_IMAGE,
        "root": TYPESCRIPT_TOOLCHAIN_ROOT,
        "manifest_path": TYPESCRIPT_TOOLCHAIN_MANIFEST,
        "manifest_sha256": TYPESCRIPT_TOOLCHAIN_MANIFEST_SHA256,
        "asset_count": TYPESCRIPT_TOOLCHAIN_ASSET_COUNT,
        "node": {
            "launcher_path": TYPESCRIPT_NODE_LAUNCHER,
            "binary_path": TYPESCRIPT_NODE_BINARY,
            "version": TYPESCRIPT_NODE_VERSION,
            "sha256": TYPESCRIPT_NODE_SHA256,
        },
        "typescript": {
            "version": TYPESCRIPT_VERSION,
            "compiler_path": TYPESCRIPT_COMPILER_JS,
            "compiler_sha256": TYPESCRIPT_COMPILER_SHA256,
            "package_path": TYPESCRIPT_PACKAGE_JSON,
            "package_sha256": TYPESCRIPT_PACKAGE_SHA256,
        },
        "canary": {
            "command": TYPESCRIPT_CANARY_COMMAND,
            "source_sha256": TYPESCRIPT_CANARY_SOURCE_SHA256,
        },
    }


__all__: Final = (
    "TYPESCRIPT_CANARY",
    "TYPESCRIPT_CANARY_COMMAND",
    "TYPESCRIPT_CANARY_SOURCE_SHA256",
    "TYPESCRIPT_COMPILER_JS",
    "TYPESCRIPT_COMPILER_SHA256",
    "TYPESCRIPT_NODE_BINARY",
    "TYPESCRIPT_NODE_LAUNCHER",
    "TYPESCRIPT_NODE_MODULES",
    "TYPESCRIPT_NODE_SHA256",
    "TYPESCRIPT_NODE_VERSION",
    "TYPESCRIPT_PACKAGE_JSON",
    "TYPESCRIPT_PACKAGE_SHA256",
    "TYPESCRIPT_TOOLCHAIN_ASSET_COUNT",
    "TYPESCRIPT_TOOLCHAIN_BIN",
    "TYPESCRIPT_TOOLCHAIN_MANIFEST",
    "TYPESCRIPT_TOOLCHAIN_MANIFEST_SHA256",
    "TYPESCRIPT_TOOLCHAIN_ROOT",
    "TYPESCRIPT_VALIDATION_BASE_IMAGE",
    "TYPESCRIPT_VALIDATION_IMAGE",
    "TYPESCRIPT_VERSION",
    "typescript_validation_toolchain_contract",
)
