"""Read-only DCR-011 multi-root forest and dirty-overlay manifest.

The portable identity deliberately excludes host paths while the companion host
projection records realpaths for local diagnostics.  It is an observation
artifact only: it neither imports target code nor performs network or write
operations.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..autonomous_repair.root_ownership import RepairRootOwnership
from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity

DCR_FOREST_SCHEMA = "ipfs_accelerate_py/agent-supervisor/deterministic-repair-forest@1"
DCR_FOREST_PORTABLE_SCHEMA = DCR_FOREST_SCHEMA + "/portable"
DCR_FOREST_HOST_SCHEMA = DCR_FOREST_SCHEMA + "/host"
DCR_REQUIRED_ROOT_IDS = (
    "orchestration",
    "swissknife",
    "mcp-plus-plus",
    "ipfs-accelerate",
    "ipfs-datasets",
    "ipfs-kit",
)


class DeterministicRepairForestError(ValueError):
    """A required root, policy/config binding, or forest replay is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _relative(workspace: Path, value: Path | str, *, field: str) -> str:
    candidate = Path(value).expanduser().resolve(strict=True)
    try:
        return candidate.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise DeterministicRepairForestError(f"{field} escapes workspace") from exc


def _file_binding(workspace: Path, value: Path | str, *, field: str) -> dict[str, str]:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise DeterministicRepairForestError(f"{field} is not a readable file")
    return {
        "path": _relative(workspace, path, field=field),
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise DeterministicRepairForestError("Git observation failed: " + " ".join(args))
    return result.stdout.strip()


def _gitlink_revision(orchestration: Path, pin_path: str) -> str:
    if not pin_path:
        return ""
    line = _git(orchestration, "ls-tree", "HEAD", "--", pin_path)
    fields = line.split(maxsplit=3)
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        raise DeterministicRepairForestError(f"required top-level gitlink is absent for {pin_path}")
    return fields[2]


def _normalize_exclusions(exclusions: Sequence[str] | None) -> tuple[str, ...]:
    result = []
    for value in exclusions or ():
        text = str(value).strip().replace("\\", "/")
        if not text or text.startswith("/") or ".." in Path(text).parts:
            raise DeterministicRepairForestError("exclusions must be safe relative paths")
        result.append(text)
    return tuple(sorted(set(result)))


def capture_deterministic_repair_forest(
    *,
    workspace_root: Path | str,
    root_policy_path: Path | str,
    config_paths: Sequence[Path | str] = (),
    exclusions: Sequence[str] = (),
) -> dict[str, Any]:
    """Capture all DCR roots with portable and host-specific projections."""

    workspace = Path(workspace_root).expanduser().resolve(strict=True)
    policy_path = Path(root_policy_path).expanduser().resolve(strict=True)
    try:
        ownership = RepairRootOwnership.from_file(policy_path, workspace_root=workspace)
    except ValueError as exc:
        raise DeterministicRepairForestError(
            "root policy does not bind every required root"
        ) from exc
    if tuple(root.root_id for root in ownership.roots) != DCR_REQUIRED_ROOT_IDS:
        raise DeterministicRepairForestError("policy does not declare the exact six DCR roots")
    bindings = ownership.capture_bindings()
    orchestration = ownership.root_path("orchestration")
    root_records: list[dict[str, Any]] = []
    host_roots: dict[str, dict[str, str]] = {}
    for root in ownership.roots:
        binding = bindings[root.root_id]
        binding_portable = {
            "root_id": binding.root_id,
            "head": binding.head,
            "tree": binding.tree,
            "dirty": binding.dirty,
            "dirty_overlay_digest": binding.overlay_digest,
        }
        root_records.append(
            {
                "root_id": root.root_id,
                "relative_path": root.relative_path,
                "role": root.role,
                "pin_path": root.pin_path,
                "head": binding.head,
                "tree": binding.tree,
                "dirty": binding.dirty,
                "dirty_overlay_digest": binding.overlay_digest,
                "content_digest": _sha256_bytes(canonical_json_bytes(binding_portable)),
                "planning_gitlink_revision": _gitlink_revision(orchestration, root.pin_path),
            }
        )
        host_roots[root.root_id] = {"realpath": binding.realpath}
    portable = {
        "schema": DCR_FOREST_PORTABLE_SCHEMA,
        "root_policy": _file_binding(workspace, policy_path, field="root_policy"),
        "config_roots": sorted(
            (_file_binding(workspace, path, field="config_root") for path in config_paths),
            key=lambda item: item["path"],
        ),
        "exclusions": list(_normalize_exclusions(exclusions)),
        "roots": root_records,
    }
    identity = content_identity(portable)
    return {
        "schema": DCR_FOREST_SCHEMA,
        "interface": "DeterministicRepairForest@1",
        "authoritative": True,
        "portable": portable,
        "portable_identity": identity,
        "host": {
            "schema": DCR_FOREST_HOST_SCHEMA,
            "workspace_realpath": str(workspace),
            "roots": host_roots,
        },
    }


def verify_deterministic_repair_forest(
    manifest: Mapping[str, Any],
    *,
    workspace_root: Path | str,
    root_policy_path: Path | str,
    config_paths: Sequence[Path | str] = (),
    exclusions: Sequence[str] = (),
) -> dict[str, Any]:
    """Re-capture and require exact portable identity, including overlays."""

    if not isinstance(manifest, Mapping) or manifest.get("schema") != DCR_FOREST_SCHEMA:
        raise DeterministicRepairForestError("unsupported deterministic repair forest manifest")
    if not manifest.get("authoritative", False):
        raise DeterministicRepairForestError("non-authoritative forest snapshot cannot verify")
    current = capture_deterministic_repair_forest(
        workspace_root=workspace_root,
        root_policy_path=root_policy_path,
        config_paths=config_paths,
        exclusions=exclusions,
    )
    if manifest.get("portable_identity") != current["portable_identity"]:
        raise DeterministicRepairForestError(
            "forest verification failed: root, gitlink, config, or dirty overlay changed"
        )
    return current


def non_authoritative_snapshot(portable: Mapping[str, Any], *, note: str) -> dict[str, Any]:
    """Create an explicitly non-verifiable checked-in planning snapshot."""

    value = dict(portable)
    return {
        "schema": DCR_FOREST_SCHEMA,
        "interface": "DeterministicRepairForest@1",
        "authoritative": False,
        "status": "non_authoritative_current_snapshot",
        "note": str(note),
        "portable": value,
        "portable_identity": content_identity(value),
        "host": {"schema": DCR_FOREST_HOST_SCHEMA, "roots": {}},
    }


__all__ = [
    "DCR_FOREST_HOST_SCHEMA",
    "DCR_FOREST_PORTABLE_SCHEMA",
    "DCR_FOREST_SCHEMA",
    "DCR_REQUIRED_ROOT_IDS",
    "DeterministicRepairForestError",
    "capture_deterministic_repair_forest",
    "non_authoritative_snapshot",
    "verify_deterministic_repair_forest",
]
