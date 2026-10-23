#!/usr/bin/env python3
"""Stage the separately frozen AMD64 ordinary-worker closure.

Reuse the bounded wheel/RECORD/archive verifier without relaxing the historical
ARM64 profile. The builder must bind the exact cached base image and qualify the
result; staging alone grants no provider or protected-route authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile


LOCK_SHA256 = "4691b6e4c25f6e11f57c0434bc794595912325edfdff3d5551b1ee4295316f1a"
CONTAINERFILE_SHA256 = "6d9a3da6a4657f312c6694f96d3c4809779ea23cd7a0ccd5ec90ab68033616a9"
BASE_IMAGE_ID = "sha256:6277ab668b2ad3c3ab851d370faa1690d324b30db29d342527f61fed28f20f36"
RIPGREP_PACKAGE_SHA256 = "c5ae63c7bee915b1cfc9f0bd07c55b4ce7f2bcd1133cba4da56719aac26101a4"
RIPGREP_BINARY_SHA256 = "be2fbcdd985161603bfa268c7647eec670b9e94374f0e972da708d7de670a8fb"
RIPGREP_URL = "https://archive.ubuntu.com/ubuntu/pool/universe/r/rust-ripgrep/ripgrep_14.1.0-1_amd64.deb"


def load_stager():
    path = Path(__file__).resolve().parent.parent / "stage_ordinary_worker_context.py"
    spec = importlib.util.spec_from_file_location("ordinary_worker_frozen_stager", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("ordinary worker verifier unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.LOCK_SHA256 = LOCK_SHA256
    module.CONTAINERFILE_SHA256 = CONTAINERFILE_SHA256
    module.BASE_IMAGE_ID = BASE_IMAGE_ID
    module.PROFILE = "ordinary-task-toolchain-amd64-v1"
    module.LABEL = "2026-10-21-ordinary-amd64-v1"
    module.ARCHITECTURE = "amd64"
    module.SOURCE_DATE_EPOCH = 1792540800  # 2026-10-21 00:00:00 UTC.
    return module


def stage_ripgrep(module, *, output: Path, package: Path | None = None) -> dict:
    """Add only the frozen Ubuntu binary/license; never run package scripts."""

    package_record = module.WheelInput(
        "ripgrep", "14.1.0-1", "ripgrep_14.1.0-1_amd64.deb",
        RIPGREP_URL, 1551362, RIPGREP_PACKAGE_SHA256,
    )
    if package is None:
        package = output / package_record.filename
        module._download(package_record, package)
    if (package.is_symlink() or not package.is_file()
            or package.stat().st_size != package_record.size
            or hashlib.sha256(package.read_bytes()).hexdigest() != RIPGREP_PACKAGE_SHA256):
        raise module.BuildInputError("frozen_ripgrep_package_mismatch")
    # dpkg-deb only decompresses the verified archive. No package installer or
    # maintainer script is invoked, and the expanded closure is bounded below.
    expanded = subprocess.run(
        ["/usr/bin/dpkg-deb", "--fsys-tarfile", str(package)],
        check=True, capture_output=True, timeout=30,
    ).stdout
    if len(expanded) > 8 * 1024**2:
        raise module.BuildInputError("ripgrep_archive_byte_bound_exceeded")
    manifest_path = output / "installed-files.json"
    manifest = json.loads(manifest_path.read_text())
    inventory = manifest["files"]
    selected = {
        "./usr/bin/rg": ("opt/ipfs-task-tools/bin/rg", RIPGREP_BINARY_SHA256, 0o555),
        "./usr/share/doc/ripgrep/copyright": (
            "opt/ipfs-task-tools/share/licenses/ripgrep/copyright",
            "b8badd31db4a52167bf97284ce2ad9ae6fe3effe1df087db0c8687a02c898f9c", 0o444,
        ),
    }
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
        for source, (target, digest, mode) in selected.items():
            member = archive.getmember(source)
            if not member.isfile() or member.size > 6 * 1024**2:
                raise module.BuildInputError("ripgrep_member_not_admitted")
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise module.BuildInputError("ripgrep_member_digest_mismatch")
            module._write_generated(output / "context/rootfs", target, data, mode, inventory)
            inventory[target]["package"] = "ubuntu-ripgrep"
    manifest_path.write_bytes(module._canonical(manifest))
    lock = module.load_lock(Path(__file__).with_name("wheels.lock.json"))
    sbom = module._spdx(lock, inventory)
    sbom["packages"].append({
        "SPDXID": "SPDXRef-Package-ubuntu-ripgrep", "name": "ripgrep",
        "versionInfo": package_record.version, "downloadLocation": RIPGREP_URL,
        "filesAnalyzed": False,
        "checksums": [{"algorithm": "SHA256", "checksumValue": RIPGREP_PACKAGE_SHA256}],
        "licenseConcluded": "NOASSERTION", "licenseDeclared": "NOASSERTION",
        "copyrightText": "See opt/ipfs-task-tools/share/licenses/ripgrep/copyright",
    })
    (output / "sbom.spdx.json").write_bytes(module._canonical(sbom))
    context = output / "build-context.tar"
    context.unlink()
    module._context_tar(output / "context", context)
    report_path = output / "staging-receipt.json"
    report = json.loads(report_path.read_text())
    report.update(
        build_context_sha256=hashlib.sha256(context.read_bytes()).hexdigest(),
        build_context_bytes=context.stat().st_size,
        installed_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        sbom_sha256=hashlib.sha256((output / "sbom.spdx.json").read_bytes()).hexdigest(),
        installed_file_and_symlink_count=len(inventory),
        ripgrep_package_sha256=RIPGREP_PACKAGE_SHA256,
        ripgrep_binary_sha256=RIPGREP_BINARY_SHA256,
        package_scripts_executed=False,
    )
    report_path.write_bytes(module._canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ripgrep-deb", type=Path)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    module = load_stager()
    module.stage_worker_context(
        output=args.output,
        lock_path=directory / "wheels.lock.json",
        containerfile=directory / "ordinary-task-toolchain.Containerfile",
    )
    result = stage_ripgrep(module, output=args.output, package=args.ripgrep_deb)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
