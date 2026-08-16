"""DCR-040 registry is finite data and stays non-executing."""

from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
    RepairOperatorRegistryError,
)


def _raw() -> dict[str, object]:
    return {
        "operator_id": "operator:replace-catalog-registration",
        "kind": "replace_unique_registration",
        "input_schema": {
            "type": "object",
            "required": ["before_digest", "after_digest", "target_path"],
            "properties": {
                "before_digest": "sha256",
                "after_digest": "sha256",
                "target_path": "path",
            },
            "additional_properties": False,
        },
        "owner_root": "root:accelerate",
        "write_scope": ["ipfs_accelerate_py/mcp_server/catalog.py"],
        "before_predicates": ["predicate:unique-registration"],
        "after_predicates": ["predicate:registered-once"],
        "applicability_proofs": ["cid:applicability-proof"],
        "preview": {"kind": "metadata_only", "fields": ["operator_id", "input_cid"]},
        "inverse": {"kind": "restore_exact_before_bytes", "binding": "before_digest"},
        "validation_commands": [["pytest", "-q", "test/api/test_catalog.py"]],
    }


def test_manifest_pinned_descriptor_enumerates_and_previews_without_execution() -> None:
    descriptor = OperatorDescriptor.from_mapping(_raw())
    registry = OperatorRegistry(
        [descriptor], reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )
    preview = registry.enumerate()[0].preview_input(
        {
            "before_digest": "sha256:before",
            "after_digest": "sha256:after",
            "target_path": "ipfs_accelerate_py/mcp_server/catalog.py",
        }
    )
    report = registry.report()
    assert preview["execution_authorized"] is False
    assert preview["model_call_count"] == 0
    assert report["activation_status"] == "integration_pending_dcr035"
    assert report["operators"][0]["descriptor_id"] == descriptor.descriptor_id


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update({"source": "def unsafe(): pass"}),
        lambda raw: raw.update({"kind": "dynamic_import"}),
        lambda raw: raw.update({"write_scope": ["../escape.py"]}),
        lambda raw: raw.update({"inverse": {}}),
        lambda raw: raw.update({"validation_commands": [["sh", "-c", "echo hi; rm x"]]}),
    ],
)
def test_raw_source_shell_unknown_and_unbounded_descriptors_are_rejected(mutate) -> None:
    raw = _raw()
    mutate(raw)
    with pytest.raises(RepairOperatorRegistryError):
        OperatorDescriptor.from_mapping(raw)


def test_unreviewed_or_callable_manifest_paths_are_rejected() -> None:
    descriptor = OperatorDescriptor.from_mapping(_raw())
    with pytest.raises(RepairOperatorRegistryError):
        OperatorRegistry([descriptor], reviewed_manifest={})
    with pytest.raises(RepairOperatorRegistryError):
        OperatorRegistry([descriptor], reviewed_manifest={descriptor.operator_id: "cid:other"})
    raw = _raw()
    raw["preview"] = {"kind": "metadata_only", "fields": [lambda: None]}
    with pytest.raises(RepairOperatorRegistryError):
        OperatorDescriptor.from_mapping(raw)
