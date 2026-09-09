"""Kit-owned source-forest persistence on an already admitted native owner.

A source-forest CAS is one persistence component of completion. It cannot accept
semantic evidence, settle a goal, or turn a nominated report into acceptance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import RootRef
from .durable_state import _operation_id, verify_durable_root_cas_result

MANIFEST_SCHEMA = "ipfs_accelerate_py/agent-supervisor/kit-source-forest-manifest@1"
RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/kit-source-forest-persistence@1"
OWNER_FIELDS = (
    "store_id",
    "database_uuid",
    "generation",
    "fence_epoch",
    "process_birth_id",
)


def _digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
    )


def source_manifest(
    profile: Mapping[str, Any], profile_cid: str, source: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind actual observed Git commits without claiming semantic acceptance."""
    forest = source.get("source_forest")
    if (
        source.get("available") is not True
        or source.get("clean") is not True
        or not isinstance(forest, Mapping)
        or set(forest)
        != {
            "source_head",
            "nested_repositories",
            "cross_repository_writes",
            "source_forest_root",
        }
        or forest.get("cross_repository_writes") is not False
        or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", str(forest.get("source_head", ""))
        )
        or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", str(source.get("repository_tree_id", ""))
        )
        or forest["source_forest_root"]
        != _digest({k: v for k, v in forest.items() if k != "source_forest_root"})
    ):
        raise ValueError("current source forest is unavailable or invalid")
    nested = forest["nested_repositories"]
    expected = profile["nested_repositories"]
    if (
        not isinstance(nested, list)
        or len(nested) != len(expected)
        or any(
            not isinstance(row, Mapping)
            or any(
                row.get(key) != spec[key]
                for key in ("repository", "path", "planning_revision")
            )
            or row.get("planning_revision_is_ancestor") is not True
            for row, spec in zip(nested, expected)
        )
    ):
        raise ValueError("source forest differs from the sealed repository population")
    return {
        "schema": MANIFEST_SCHEMA,
        "board_namespace": profile["board_namespace"],
        "profile_cid": profile_cid,
        "bootstrap_receipt_id": profile["bootstrap_receipt_id"],
        "plan_root_cid": profile["plan_root_cid"],
        "bootstrap_repository_tree_id": profile["repository_tree_id"],
        "repository_tree_id": source["repository_tree_id"],
        "source_forest": copy.deepcopy(dict(forest)),
        "source_view": "git_commit_forest",
        "semantic_acceptance_authority": False,
        "completion_authority": False,
    }


def _load_kit(repository_root: str, forest: Mapping[str, Any]) -> Any:
    """Load only the two kit leaf modules from their observed commit bytes.

    Package initializers and site-installed kit copies are not part of this
    admission. Each file must match the actual selected Git commit before its
    captured bytes are executed, so a modified checkout cannot supply code.
    """
    specs = [s for s in forest["nested_repositories"] if s["repository"] == "ipfs_kit"]
    if len(specs) != 1:
        raise ValueError("source forest requires one exact kit repository")
    spec = specs[0]
    root = (Path(repository_root) / spec["path"]).resolve(strict=True)
    if not root.is_relative_to(Path(repository_root).resolve(strict=True)):
        raise ValueError("kit repository escapes source forest")
    captured = {}
    for name in ("coordination_storage", "duckdb_coordination_storage"):
        relative = f"ipfs_kit_py/mcp_server/mcplusplus/{name}.py"
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 262144:
            raise ValueError("kit native module is missing or outside its bound")
        data = path.read_bytes()
        result = subprocess.run(
            ["git", "cat-file", "blob", f"{spec['head']}:{relative}"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=5,
        )
        if result.stdout != data:
            raise ValueError("kit native module differs from observed commit")
        captured[name] = (path, data)
    package_name = "_native_kit_source_forest_" + spec["head"]
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    for name, (path, data) in captured.items():
        qualified = package_name + "." + name
        module = types.ModuleType(qualified)
        module.__package__, module.__file__ = package_name, str(path)
        sys.modules[qualified] = module
        try:
            exec(compile(data, str(path), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(qualified, None)
            raise
        setattr(package, name, module)
    return package.duckdb_coordination_storage


class KitSourceForestPersistence:
    """Launcher-owned producer; snapshot observation is strictly read-only."""

    def __init__(
        self,
        profile: Mapping[str, Any],
        profile_cid: str,
        store: Any,
        owner_identity: Mapping[str, Any],
    ):
        self.profile, self.profile_cid, self.store = (
            copy.deepcopy(dict(profile)),
            profile_cid,
            store,
        )
        self.namespace = f"source-forest/{profile['board_namespace']}/{profile_cid}"
        self.owner_identity = {key: owner_identity[key] for key in OWNER_FIELDS}

    @classmethod
    def bind(
        cls,
        profile: Mapping[str, Any],
        profile_cid: str,
        *,
        repository_root: str,
        source: Mapping[str, Any],
        connection: Any,
        transaction_lock: Any,
        owner_identity: Mapping[str, Any],
    ) -> "KitSourceForestPersistence":
        source_manifest(profile, profile_cid, source)
        module = _load_kit(repository_root, source["source_forest"])
        namespace = f"source-forest/{profile['board_namespace']}/{profile_cid}"
        store = module.DuckDBCoordinationStore(
            connection,
            transaction_lock=transaction_lock,
            owner_identity=owner_identity,
            namespace=namespace,
        )
        return cls(profile, profile_cid, store, owner_identity)

    def publish(self, source: Mapping[str, Any]) -> dict[str, Any]:
        manifest = source_manifest(self.profile, self.profile_cid, source)
        # The kit authority computes and checks its own canonical CID.
        stored = self.store.put(manifest, codec="dag-json", replicate=False)
        cid = stored["cid"]
        before = self.store.current_state_root(self.namespace)
        revision, predecessor = before["revision"], before["root_cid"]
        if predecessor == cid:
            # Replay the actual preceding operation, never bind an old key to
            # a later same-content generation after an A-to-B-to-A sequence.
            transition = self.store.get(before["transition_cid"])
            revision, predecessor = (
                transition["expected_revision"],
                transition["expected_root_cid"],
            )
        expected = (
            RootRef(root_cid=predecessor, generation=revision) if revision else None
        )
        result = self.store.compare_and_swap_state_root(
            self.namespace,
            expected_revision=revision,
            expected_root_cid=predecessor,
            new_root_cid=cid,
            operation_id=_operation_id(revision, cid),
        )
        verified = verify_durable_root_cas_result(
            self.namespace, expected, cid, result, read_artifact=self.store.get
        )
        observed = self.observe(source)
        if (
            observed.get("admitted") is not True
            or observed["transition_cid"] != verified.transition_cid
        ):
            raise ValueError(
                "published kit source forest is not independently readable"
            )
        return {**observed, "idempotent_replay": verified.idempotent_replay}

    def observe(self, source: Mapping[str, Any]) -> dict[str, Any]:
        result = {
            "schema": RECEIPT_SCHEMA,
            "admitted": False,
            "authority": "kit_source_forest_persistence",
            "completion_authority": False,
            "semantic_acceptance_authority": False,
            "owner_identity": self.owner_identity,
        }
        try:
            manifest = source_manifest(self.profile, self.profile_cid, source)
            current = self.store.current_state_root(self.namespace)
            if type(current.get("revision")) is not int or current["revision"] <= 0:
                raise ValueError("kit source forest root is absent")
            stored = self.store.get(current["root_cid"])
            transition = self.store.get(current["transition_cid"])
            bound = {
                "schema": "mcp++/coordination/state-root-transition@1",
                "namespace": self.namespace,
                "operation_id": _operation_id(
                    current["revision"] - 1, current["root_cid"]
                ),
                "new_root_cid": current["root_cid"],
                "new_revision": current["revision"],
                "expected_revision": current["revision"] - 1,
            }
            if (
                stored != manifest
                or current.get("namespace") != self.namespace
                or set(transition)
                != set(bound) | {"expected_root_cid", "created_at_ms"}
                or any(
                    type(transition.get(k)) is not type(v) or transition.get(k) != v
                    for k, v in bound.items()
                )
                or type(transition.get("created_at_ms")) is not int
                or transition["created_at_ms"] < 0
                or (current["revision"] == 1)
                != (transition["expected_root_cid"] is None)
            ):
                raise ValueError(
                    "kit source forest transition or current source binding differs"
                )
            result.update(
                admitted=True,
                namespace=self.namespace,
                manifest_cid=current["root_cid"],
                transition_cid=current["transition_cid"],
                root_revision=current["revision"],
                source_forest_root=manifest["source_forest"]["source_forest_root"],
                repository_tree_id=manifest["repository_tree_id"],
                source_head=manifest["source_forest"]["source_head"],
                transport="owner_bound_duckdb_quack",
            )
        except (
            Exception
        ) as error:  # noqa: BLE001 - failed component evidence is not acceptance
            result.update(
                reason="kit_current_source_forest_not_admitted",
                error_class=type(error).__name__,
            )
        return result
