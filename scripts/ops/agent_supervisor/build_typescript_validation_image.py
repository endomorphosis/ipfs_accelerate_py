#!/usr/bin/env python3
"""Build the sealed TypeScript validation image from local, pinned bytes.

The builder never pulls an image and gives Docker no build-time network.  It
derives from the already-sealed authority image and copies an independently
addressed Node runtime closure plus the TypeScript 5.9.3 compiler API into
``/opt/ipfs-validation-toolchains``.  Runtime code consumes only the immutable
image ID printed by this program; the local tag is a disposable operator aid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Final

BASE_IMAGE: Final = (
    "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"
)
BASE_IMAGE_LOCAL_REFERENCE: Final = "ipfs-accelerate-authority-validation:20260803-v2"
DOCKER_PATH: Final = Path("/usr/bin/docker")
DOCKER_SHA256: Final = (
    "414d9e16a30060770648522f8ecadef2f2b57b50b8c61d4b0ae9d3b8b64c2a02"
)
DOCKER_ENDPOINT: Final = "unix:///run/docker.sock"
LOCAL_TAG: Final = "ipfs-accelerate-authority-validation:typescript-5.9.3-20260808"

HOST_NODE: Final = Path("/usr/bin/node")
NODE_VERSION: Final = "v18.19.1"
NODE_SHA256: Final = "2b0f6efd95c31c5538cc0a9042d5d13b7328cffcfdcc409f2e2ef336c4402086"
HOST_TYPESCRIPT_ROOT: Final = Path(
    "/home/barberb/.local/share/ipfs_accelerate_py/contract-repair/"
    "typescript-5.9.3/node_modules/typescript"
)
HOST_TYPESCRIPT_JS: Final = HOST_TYPESCRIPT_ROOT / "lib/typescript.js"
HOST_TYPESCRIPT_PACKAGE: Final = HOST_TYPESCRIPT_ROOT / "package.json"
TYPESCRIPT_VERSION: Final = "5.9.3"
TYPESCRIPT_SHA256: Final = (
    "3ae902c92cc44dace175c0e69e13a4b0899f6983c6121d76b9ab8dd5795e7675"
)
TYPESCRIPT_PACKAGE_SHA256: Final = (
    "822ef7ca6452205657b6288b066481ecf508bfbf43455d715cf7d3ec457561e6"
)

TOOLCHAIN_ROOT: Final = Path("/opt/ipfs-validation-toolchains")
NODE_ROOT: Final = TOOLCHAIN_ROOT / "node-v18.19.1"
NODE_BINARY: Final = NODE_ROOT / "bin/node"
NODE_LIBRARY_ROOT: Final = NODE_ROOT / "lib"
NODE_SHARE_ROOT: Final = NODE_ROOT / "share/nodejs"
NODE_LOADER: Final = NODE_LIBRARY_ROOT / "ld-linux-aarch64.so.1"
TYPESCRIPT_ROOT: Final = TOOLCHAIN_ROOT / "typescript-5.9.3"
TYPESCRIPT_JS: Final = TYPESCRIPT_ROOT / "node_modules/typescript/lib/typescript.js"
TYPESCRIPT_PACKAGE: Final = TYPESCRIPT_ROOT / "node_modules/typescript/package.json"
NODE_WRAPPER: Final = TOOLCHAIN_ROOT / "bin/node"
CANARY: Final = TYPESCRIPT_ROOT / "canary.js"
MANIFEST: Final = TOOLCHAIN_ROOT / "manifest.json"

# ``ldd /usr/bin/node`` on the sealed host yields this complete runtime
# closure.  Named SONAME paths are copied as regular files so runtime lookup
# cannot escape the isolated directory through a symlink.
NODE_RUNTIME_ASSETS: Final = {
    "/lib/ld-linux-aarch64.so.1": (
        "393384096ffa869e1be20d2f91fdf08dfadb9f3e531dfe724085d8501d3f85d9"
    ),
    "/lib/aarch64-linux-gnu/libbrotlicommon.so.1": (
        "016600f673a5d1a565cf98f608d8425033593ab78827eccc3384596f70386c8c"
    ),
    "/lib/aarch64-linux-gnu/libbrotlidec.so.1": (
        "b7895f0fd4ad47408cb01fcb959ae6792804c15512526166f9bf48358714d0b5"
    ),
    "/lib/aarch64-linux-gnu/libbrotlienc.so.1": (
        "0e39569ce7768abb8fb525cad5b098cb571d4e1b922d099b4fcfdca8e54ed0e9"
    ),
    "/lib/aarch64-linux-gnu/libc.so.6": (
        "fe5966a43e068ad7cb389c3affa069f4ee6f296e07d7ccc0398a23cfde4f0b7e"
    ),
    "/lib/aarch64-linux-gnu/libcares.so.2": (
        "55d93f2870d498cc521afe990af515ca525fa525507e34b2be6c7ec6f9d54169"
    ),
    "/lib/aarch64-linux-gnu/libcrypto.so.3": (
        "3d51e64ad7113b2ab863b9582866ad2b31e545d1fe9f197c634a65a8f2611786"
    ),
    "/lib/aarch64-linux-gnu/libgcc_s.so.1": (
        "f2d3ad2bf0b61f6bc944cc37d7b6ab7f88d2582b41ff989ca164803f56cc5f20"
    ),
    "/lib/aarch64-linux-gnu/libicudata.so.74": (
        "96b1154b9ab94cfc908c0fe745edc128ee3f67e444c66e2d7a9cf06664df9255"
    ),
    "/lib/aarch64-linux-gnu/libicui18n.so.74": (
        "7bd69f1141097e7d127cdf3436fa7448dc7925bf2e3da0fdf028e94ec2e69243"
    ),
    "/lib/aarch64-linux-gnu/libicuuc.so.74": (
        "9d58137df08c36a03417445cc39576b0a95b538bdf0733f5455d2255deb9b799"
    ),
    "/lib/aarch64-linux-gnu/libm.so.6": (
        "d5b262e559d38769e4959f036a1cc2c4009fa60b4ec836a7a19ca8554aebf5a5"
    ),
    "/lib/aarch64-linux-gnu/libnghttp2.so.14": (
        "943724df4606c5eaeb47b2371da22d08fa3870249cfe90c84cef8e57ed6d8db5"
    ),
    "/lib/aarch64-linux-gnu/libnode.so.109": (
        "8156eeb6c3a461b266cc69eff2c4a8ceab4dbdb3be48ed382c622fac9c27a536"
    ),
    "/lib/aarch64-linux-gnu/libssl.so.3": (
        "f23c6de6c41fb0ecc159f1af052cae3b525de697fae242290e656a242e7bc25d"
    ),
    "/lib/aarch64-linux-gnu/libstdc++.so.6": (
        "6e3112d35cfc86db7ee85b27e1746f67408e2837ff8628b46386c3eabd5682a4"
    ),
    "/lib/aarch64-linux-gnu/libuv.so.1": (
        "241c344af1676bdc6f300528cd9dbf45678a3e31caae10fa2c88ca74052c5996"
    ),
    "/lib/aarch64-linux-gnu/libz.so.1": (
        "170380b4e7ab28ec86eb090b48df90f84089392cb72fecd5067e5b7a4dc5239f"
    ),
}

# Debian's shared libnode externalizes these five JavaScript builtins.  Keep
# their bytes in the isolated Node root and install only fixed compatibility
# symlinks at libnode's compile-time paths.
NODE_EXTERNALIZED_BUILTINS: Final = {
    "/usr/share/nodejs/acorn-walk/dist/walk.js": (
        "2bf2dbb2b7f0e4877eeffed1a9b51cec8cb341539c769da5e3ec83f5a8772672"
    ),
    "/usr/share/nodejs/acorn/dist/acorn.js": (
        "cb3b2d439857d6e4514d23a1219f6354c23498f5fa81fdf2691a5a9e53de43c0"
    ),
    "/usr/share/nodejs/cjs-module-lexer/dist/lexer.js": (
        "bfcd3684f197fffc2b9b3d05ba95818d69fe3a4e6c54463f000bcd0dbd97fd2d"
    ),
    "/usr/share/nodejs/cjs-module-lexer/lexer.js": (
        "0abd984d16d53a59af9978283f9fddaee1342edc6ccb7810e443d6ee0ff26fff"
    ),
    "/usr/share/nodejs/undici/undici-fetch.js": (
        "398ed64aebf6ee3b216049e5ff4df8297e4ad968a531aff0022bfaf9741266b4"
    ),
}

NODE_WRAPPER_BYTES: Final = b"""#!/bin/sh
set -eu
unset LD_AUDIT LD_DEBUG LD_LIBRARY_PATH LD_PRELOAD NODE_OPTIONS
export NODE_PATH=/opt/ipfs-validation-toolchains/typescript-5.9.3/node_modules
exec /opt/ipfs-validation-toolchains/node-v18.19.1/lib/ld-linux-aarch64.so.1 \\
  --inhibit-cache \\
  --library-path /opt/ipfs-validation-toolchains/node-v18.19.1/lib \\
  /opt/ipfs-validation-toolchains/node-v18.19.1/bin/node "$@"
"""

CANARY_BYTES: Final = b"""'use strict';
const crypto = require('node:crypto');
const fs = require('node:fs');

const manifestPath = '/opt/ipfs-validation-toolchains/manifest.json';
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const sha256 = (path) => crypto.createHash('sha256')
  .update(fs.readFileSync(path)).digest('hex');
for (const asset of manifest.assets) {
  const file = fs.lstatSync(asset.path);
  if (!file.isFile() || sha256(asset.path) !== asset.sha256) {
    throw new Error(`validation toolchain asset mismatch: ${asset.path}`);
  }
}
for (const link of manifest.compatibility_links) {
  const file = fs.lstatSync(link.path);
  if (!file.isSymbolicLink() || fs.readlinkSync(link.path) !== link.target) {
    throw new Error(`validation toolchain compatibility link mismatch: ${link.path}`);
  }
}

const ts = require('typescript');
const source = [
  'type Result<T> = { readonly ok: true; readonly value: T };',
  'const result: Result<number> = { ok: true, value: 42 };',
].join('\\n');
const sourceFile = ts.createSourceFile(
  'ipfs-validation-canary.ts',
  source,
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TS,
);
if (sourceFile.parseDiagnostics.length !== 0 || sourceFile.statements.length !== 2) {
  throw new Error('TypeScript canary parse failed');
}
if (ts.version !== manifest.typescript.version) {
  throw new Error(`TypeScript version mismatch: ${ts.version}`);
}
process.stdout.write(JSON.stringify({
  schema: 'ipfs_accelerate_py.agent_supervisor.typescript-validation-canary@1',
  node_version: process.version,
  node_sha256: sha256(manifest.node.binary_path),
  typescript_version: ts.version,
  compiler_sha256: sha256(manifest.typescript.compiler_path),
  package_sha256: sha256(manifest.typescript.package_path),
  asset_count: manifest.assets.length,
  parse_diagnostic_count: sourceFile.parseDiagnostics.length,
  statement_count: sourceFile.statements.length,
  source_sha256: crypto.createHash('sha256').update(source).digest('hex'),
}) + '\\n');
"""


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _verify_regular_asset(path: Path, expected_sha256: str) -> None:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"not a regular build input: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"build input digest mismatch for {path}: {observed}")


def _ldd_paths() -> set[str]:
    completed = subprocess.run(
        ["/usr/bin/ldd", str(HOST_NODE)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
        check=False,
        env={"HOME": "/nonexistent", "PATH": os.defpath},
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ldd failed: {completed.stdout}")
    found: set[str] = set()
    for line in completed.stdout.splitlines():
        match = re.search(r"(?:=>\s+)?(/[^\s(]+)", line)
        if match:
            found.add(match.group(1))
    return found


def _copy_asset(source: Path, destination: Path, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=True)
    destination.chmod(mode)


def _container_path(context_rootfs: Path, destination: Path) -> Path:
    return context_rootfs / destination.relative_to("/")


def _prepare_context(context: Path) -> tuple[dict[str, object], str]:
    _verify_regular_asset(DOCKER_PATH, DOCKER_SHA256)
    _verify_regular_asset(HOST_NODE, NODE_SHA256)
    _verify_regular_asset(HOST_TYPESCRIPT_JS, TYPESCRIPT_SHA256)
    _verify_regular_asset(
        HOST_TYPESCRIPT_PACKAGE,
        TYPESCRIPT_PACKAGE_SHA256,
    )
    observed_version = subprocess.run(
        [str(HOST_NODE), "--version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
        check=False,
        env={"HOME": "/nonexistent", "PATH": os.defpath},
    )
    if (
        observed_version.returncode != 0
        or observed_version.stdout.strip() != NODE_VERSION
    ):
        raise RuntimeError("host Node version is not the sealed version")
    package = json.loads(HOST_TYPESCRIPT_PACKAGE.read_text(encoding="utf-8"))
    if str(package.get("version") or "") != TYPESCRIPT_VERSION:
        raise RuntimeError("host TypeScript package is not version 5.9.3")
    ldd_paths = _ldd_paths()
    if ldd_paths != set(NODE_RUNTIME_ASSETS):
        raise RuntimeError(
            "host Node ldd closure changed: "
            f"missing={sorted(set(NODE_RUNTIME_ASSETS) - ldd_paths)}, "
            f"extra={sorted(ldd_paths - set(NODE_RUNTIME_ASSETS))}"
        )
    for raw_path, expected_sha256 in NODE_RUNTIME_ASSETS.items():
        _verify_regular_asset(Path(raw_path), expected_sha256)
    for raw_path, expected_sha256 in NODE_EXTERNALIZED_BUILTINS.items():
        _verify_regular_asset(Path(raw_path), expected_sha256)

    rootfs = context / "rootfs"
    assets: list[dict[str, object]] = []

    def add_bytes(
        payload: bytes,
        destination: Path,
        *,
        mode: int,
        role: str,
    ) -> None:
        target = _container_path(rootfs, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(mode)
        assets.append(
            {
                "mode": format(mode, "04o"),
                "path": str(destination),
                "role": role,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )

    def add_file(
        source: Path,
        destination: Path,
        *,
        mode: int,
        role: str,
        expected_sha256: str,
    ) -> None:
        target = _container_path(rootfs, destination)
        _copy_asset(source, target, mode)
        assets.append(
            {
                "mode": format(mode, "04o"),
                "path": str(destination),
                "role": role,
                "sha256": expected_sha256,
                "size_bytes": target.stat().st_size,
            }
        )

    add_file(
        HOST_NODE,
        NODE_BINARY,
        mode=0o755,
        role="node_binary",
        expected_sha256=NODE_SHA256,
    )
    for raw_path, expected_sha256 in sorted(NODE_RUNTIME_ASSETS.items()):
        source = Path(raw_path)
        add_file(
            source,
            NODE_LIBRARY_ROOT / source.name,
            mode=(0o755 if source == Path("/lib/ld-linux-aarch64.so.1") else 0o644),
            role=(
                "node_loader"
                if source == Path("/lib/ld-linux-aarch64.so.1")
                else "node_library"
            ),
            expected_sha256=expected_sha256,
        )
    compatibility_links: list[dict[str, str]] = []
    for raw_path, expected_sha256 in sorted(NODE_EXTERNALIZED_BUILTINS.items()):
        source = Path(raw_path)
        relative = source.relative_to("/usr/share/nodejs")
        destination = NODE_SHARE_ROOT / relative
        add_file(
            source,
            destination,
            mode=0o644,
            role="node_externalized_builtin",
            expected_sha256=expected_sha256,
        )
        link_path = _container_path(rootfs, source)
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(destination)
        compatibility_links.append({"path": str(source), "target": str(destination)})
    add_file(
        HOST_TYPESCRIPT_JS,
        TYPESCRIPT_JS,
        mode=0o644,
        role="typescript_compiler_api",
        expected_sha256=TYPESCRIPT_SHA256,
    )
    add_file(
        HOST_TYPESCRIPT_PACKAGE,
        TYPESCRIPT_PACKAGE,
        mode=0o644,
        role="typescript_package_manifest",
        expected_sha256=TYPESCRIPT_PACKAGE_SHA256,
    )
    add_bytes(
        NODE_WRAPPER_BYTES,
        NODE_WRAPPER,
        mode=0o755,
        role="node_isolated_launcher",
    )
    add_bytes(
        CANARY_BYTES,
        CANARY,
        mode=0o644,
        role="typescript_parse_canary",
    )
    assets.sort(key=lambda item: str(item["path"]))
    manifest: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "typescript-validation-toolchain-manifest@1"
        ),
        "base_image": BASE_IMAGE,
        "node": {
            "binary_path": str(NODE_BINARY),
            "loader_path": str(NODE_LOADER),
            "sha256": NODE_SHA256,
            "version": NODE_VERSION,
        },
        "typescript": {
            "compiler_path": str(TYPESCRIPT_JS),
            "compiler_sha256": TYPESCRIPT_SHA256,
            "package_path": str(TYPESCRIPT_PACKAGE),
            "package_sha256": TYPESCRIPT_PACKAGE_SHA256,
            "version": TYPESCRIPT_VERSION,
        },
        "assets": assets,
        "compatibility_links": compatibility_links,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    manifest_path = _container_path(rootfs, MANIFEST)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o644)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    dockerfile = f"""FROM {BASE_IMAGE_LOCAL_REFERENCE}
COPY --chown=0:0 rootfs/ /
ENV PATH=/opt/ipfs-validation-toolchains/bin:${{PATH}} \\
    NODE_PATH=/opt/ipfs-validation-toolchains/typescript-5.9.3/node_modules \\
    IPFS_ACCELERATE_TYPESCRIPT_JS={TYPESCRIPT_JS} \\
    IPFS_ACCELERATE_TYPESCRIPT_PACKAGE_JSON={TYPESCRIPT_PACKAGE} \\
    IPFS_ACCELERATE_TYPESCRIPT_VERSION={TYPESCRIPT_VERSION}
LABEL org.ipfs-accelerate.authority-validation=2026-08-08-typescript-5.9.3 \\
    org.ipfs-accelerate.authority-validation.base={BASE_IMAGE} \\
    org.ipfs-accelerate.validation-build-network=none \\
    org.ipfs-accelerate.node.version={NODE_VERSION} \\
    org.ipfs-accelerate.node.sha256={NODE_SHA256} \\
    org.ipfs-accelerate.typescript.version={TYPESCRIPT_VERSION} \\
    org.ipfs-accelerate.typescript.sha256={TYPESCRIPT_SHA256} \\
    org.ipfs-accelerate.typescript.package-sha256={TYPESCRIPT_PACKAGE_SHA256} \\
    org.ipfs-accelerate.toolchain-manifest.sha256={manifest_sha256}
"""
    (context / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    return manifest, manifest_sha256


def build(*, tag: str) -> dict[str, object]:
    with (
        tempfile.TemporaryDirectory(
            prefix="ipfs-typescript-validation-image-"
        ) as temporary_context,
        tempfile.TemporaryDirectory(
            prefix="ipfs-typescript-validation-docker-config-"
        ) as docker_config,
    ):
        context = Path(temporary_context)
        manifest, manifest_sha256 = _prepare_context(context)
        docker_environment = {
            "BUILDKIT_PROGRESS": "plain",
            "DOCKER_CONFIG": docker_config,
            "DOCKER_HOST": DOCKER_ENDPOINT,
            "HOME": "/nonexistent/ipfs-typescript-validation-image",
            "PATH": os.defpath,
        }
        base_inspect = subprocess.run(
            [
                str(DOCKER_PATH),
                f"--host={DOCKER_ENDPOINT}",
                "--config",
                docker_config,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                BASE_IMAGE_LOCAL_REFERENCE,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
            env=docker_environment,
        )
        if base_inspect.returncode != 0 or base_inspect.stdout.strip() != BASE_IMAGE:
            raise RuntimeError(
                "local predecessor tag is absent or does not resolve to the "
                f"sealed base image: {base_inspect.stdout}"
            )
        command = [
            str(DOCKER_PATH),
            f"--host={DOCKER_ENDPOINT}",
            "--config",
            docker_config,
            "build",
            "--pull=false",
            "--network=none",
            "--no-cache",
            "--tag",
            tag,
            str(context),
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
            check=False,
            env=docker_environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"offline Docker build failed:\n{completed.stdout}")
        inspected = subprocess.run(
            [
                str(DOCKER_PATH),
                f"--host={DOCKER_ENDPOINT}",
                "--config",
                docker_config,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                tag,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
            env=docker_environment,
        )
        image_id = inspected.stdout.strip()
        if inspected.returncode != 0 or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", image_id
        ):
            raise RuntimeError(f"built image identity unavailable: {inspected.stdout}")
        return {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "typescript-validation-image-build@1"
            ),
            "base_image": BASE_IMAGE,
            "build_network": "none",
            "image_id": image_id,
            "image_pull_allowed": False,
            "local_tag": tag,
            "manifest_sha256": manifest_sha256,
            "asset_count": len(manifest["assets"]),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=LOCAL_TAG)
    args = parser.parse_args()
    print(json.dumps(build(tag=str(args.tag)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
