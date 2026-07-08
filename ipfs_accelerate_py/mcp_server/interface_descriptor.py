"""Legacy Profile A interface descriptor surface adapted to canonical MCP++ IDL registry."""

from __future__ import annotations

from .mcplusplus.idl_registry import (
    CompatibilityVerdict,
    InterfaceDescriptorRegistry,
    build_descriptor,
    canonicalize_descriptor,
    compute_interface_cid,
)


def _canonicalize(obj):
    """Source-compatible canonicalization helper."""
    return canonicalize_descriptor(obj)


def compute_cid(content: bytes, *, prefix: str = "sha256:") -> str:
    """Source-compatible byte-content CID helper."""
    import hashlib

    digest = hashlib.sha256(bytes(content)).hexdigest()
    return f"{prefix}{digest}"


def _canonical_cid(obj) -> str:
    """Source-compatible canonical object CID helper."""
    return compute_interface_cid(dict(obj or {}))


class InterfaceRepository(InterfaceDescriptorRegistry):
    """Source-compatible repository facade over canonical descriptor registry."""

    def register(self, descriptor):
        return self.register_descriptor(descriptor)

    def list(self):
        return self.list_interfaces()

    def get(self, interface_cid: str):
        return self.get_descriptor(interface_cid)

    def compat(self, interface_cid: str, *, required_cid: str | None = None):
        if not required_cid:
            return super().compat(interface_cid)

        candidate = self.get_descriptor(interface_cid)
        required = self.get_descriptor(required_cid)
        if candidate is None:
            return CompatibilityVerdict(False, [f"unknown_interface:{interface_cid}"], [], [])
        if required is None:
            return CompatibilityVerdict(False, [f"unknown_required_interface:{required_cid}"], [], [])

        candidate_requires = set(candidate.get("requires", []))
        required_requires = set(required.get("requires", []))
        missing = sorted(required_requires - candidate_requires)
        if missing:
            return CompatibilityVerdict(
                compatible=False,
                reasons=["missing_required_capabilities"],
                requires_missing=missing,
                suggested_alternatives=[],
            )
        return CompatibilityVerdict(True, [], [], [])


def toolset_slice(cids, budget=None, sort_fn=None):
    result = list(cids)
    if sort_fn is not None:
        result.sort(key=sort_fn)
    if budget is not None:
        result = result[: max(0, int(budget))]
    return result


_global_repo = None


def get_interface_repository():
    global _global_repo
    if _global_repo is None:
        _global_repo = InterfaceRepository()
    return _global_repo


__all__ = [
    "CompatibilityVerdict",
    "InterfaceRepository",
    "build_descriptor",
    "compute_interface_cid",
    "canonicalize_descriptor",
    "_canonicalize",
    "compute_cid",
    "_canonical_cid",
    "toolset_slice",
    "get_interface_repository",
]
