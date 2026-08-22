"""Frozen compatibility checks. No sibling-path search."""

from __future__ import annotations

from typing import Any, Mapping

# Frozen PCCE-007 matrix identity. The live JSON remains on the control
# checkout; this port only carries the pin identities needed to fail closed.
FROZEN_MATRIX = {
    "schema": "lift_coding.proof-carrying-context-engine.compatibility-matrix@1",
    "task_id": "PCCE-007",
    "content_id": "sha256:bfe49d9f3b6d2f472ae58d369b2138fc4e8e6320fccdd181e07a5564e075e920",
    "algorithm": "mcpp-jcs-v1",
    "repositories": {
        "endomorphosis/ipfs_datasets_py": {
            "commit": "b3669171b9bf34dac7e8f178bd0c2cc5936e57ae",
            "tree": "16ef68abe8a35a3033dfaf1ed4e8d6132600df8f",
        },
        "endomorphosis/ipfs_kit_py": {
            "commit": "81f11b6c2ee95ce49c88d07e5448380b66757478",
            "tree": "3fa93c380105221133ec14601c05696ea8a7f95c",
        },
        "endomorphosis/ipfs_accelerate_py": {
            "commit": "84a056e41e48a81d4484be43840196578d6c87da",
            "tree": "40f0771e77d394ac91d92cc1edb02f7860f6131b",
        },
        "endomorphosis/Mcp-Plus-Plus": {
            "commit": "85b8d03e767cd0ca7f4c8994013c27d31f03e817",
            "tree": "0b65dff9e2ff2be513ffac32c9fcd27838de4ead",
        },
    },
}

PSEUDO_CID_PREFIXES = ("sha256:", "urn:", "Qm")


class CompatibilityError(RuntimeError):
    reason = "incompatible"


def frozen_matrix() -> dict[str, Any]:
    return dict(FROZEN_MATRIX)


def reject_mutable_ref(ref: str) -> None:
    lowered = ref.strip().lower()
    if lowered in {"main", "master", "head"} or lowered.startswith("origin/"):
        raise CompatibilityError(f"mutable ref {ref!r} is not a v0.1 pin")


def reject_pseudo_cid(value: str) -> None:
    if value.startswith(PSEUDO_CID_PREFIXES) or not value.startswith("b"):
        raise CompatibilityError("pseudo-CID is not admitted in production")


def reject_mock(obj: Any) -> None:
    module = getattr(type(obj), "__module__", "")
    name = type(obj).__name__
    if "mock" in module.lower() or name.lower().startswith("mock"):
        raise CompatibilityError("production rejects mock dependencies")


def pin_for(repository: str) -> Mapping[str, str]:
    try:
        return FROZEN_MATRIX["repositories"][repository]
    except KeyError as exc:
        raise CompatibilityError(f"unknown repository {repository!r}") from exc
