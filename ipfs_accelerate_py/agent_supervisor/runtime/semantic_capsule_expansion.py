"""Bounded expansion of capsules from a source-current worker nomination.

Only explicitly requested native capsules leave this boundary. Block reads are
CID verified, local, and bounded; no repository scan, network or model call is
added by expansion. The worker loader retains its own source replay policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Sequence


MAX_EXPANSION_CAPSULES = 32
MAX_EXPANSION_BYTES = 65536
_MAX_BLOCK_BYTES = 2_000_000
_MAX_READ_BYTES = 8_000_000
_MAX_BLOCKS = 128
_MAX_SOURCE_BYTES = 1_000_000


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _read_source(repository, name):
    path = repository / name
    if (path.is_symlink() or not path.resolve(strict=True).is_relative_to(repository)
            or not path.is_file() or path.stat().st_size > _MAX_SOURCE_BYTES):
        raise ValueError("semantic expansion source path or size changed")
    with path.open("rb") as source:
        raw = source.read(_MAX_SOURCE_BYTES + 1)
    if len(raw) > _MAX_SOURCE_BYTES:
        raise ValueError("semantic expansion source exceeds byte bound")
    return raw


def _block_reader(directory: Path):
    """Use a call-local verified cache, never load the entire block store."""
    from ipfs_datasets_py.logic.software_contracts.content import validate_cid
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        verify_block_bytes,
    )

    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("semantic expansion block directory must be a real directory")
    directory = directory.resolve(strict=True)
    cache, used = {}, 0

    def get_block(cid):
        nonlocal used
        if (type(cid) is not str or len(cid) > 256
                or PurePosixPath(cid).name != cid or cid in {"", ".", ".."}):
            raise ValueError("semantic expansion block CID is invalid")
        validate_cid(cid)
        if cid in cache:
            return cache[cid]
        if len(cache) >= _MAX_BLOCKS:
            raise ValueError("semantic expansion block count exceeds bound")
        path = directory / cid
        if path.is_symlink() or path.resolve(strict=True).parent != directory:
            raise ValueError("semantic expansion block path escapes store")
        if not path.is_file() or path.stat().st_size > _MAX_BLOCK_BYTES:
            raise ValueError("semantic expansion block exceeds byte bound")
        with path.open("rb") as source:
            raw = source.read(min(_MAX_BLOCK_BYTES, _MAX_READ_BYTES - used) + 1)
        if len(raw) > _MAX_BLOCK_BYTES or used + len(raw) > _MAX_READ_BYTES:
            raise ValueError("semantic expansion block reads exceed byte bound")
        verify_block_bytes(cid, raw)
        cache[cid] = raw
        used += len(raw)
        return raw

    return get_block


def expand_semantic_capsules(
    *,
    repository: Path,
    artifact: str,
    expected_sha256: str,
    task_id: str,
    stable_symbol_ids: Sequence[str],
    max_bytes: int = 32768,
) -> dict:
    """Return exact requested capsules, or reject the complete expansion.

    ``artifact`` is the nominated repository-relative worker-context JSON; its
    producer block directory must remain beside it. All sources are rechecked
    before returning, including dependencies not requested for expansion.
    Capsule facts remain producer-owned and do not grant execution, proof or
    completion authority. Editing still requires the exact current raw source.
    """
    from ipfs_datasets_py.logic.software_contracts.content import (
        cid_for_bytes, validate_cid,
    )
    from ..semantic_state.capsules import admit_capsule
    from ..semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
    from ..semantic_state.wire import cid_for_payload
    from .semantic_context_runtime import (
        SemanticContextStale, _source_scope_payload, load_semantic_worker_context,
    )

    if (isinstance(stable_symbol_ids, (str, bytes))
            or not isinstance(stable_symbol_ids, Sequence)
            or not 1 <= len(stable_symbol_ids) <= MAX_EXPANSION_CAPSULES):
        raise ValueError("semantic expansion requires 1 to 32 explicit stable symbol IDs")
    requested = tuple(stable_symbol_ids)
    if (any(type(value) is not str or not value or len(value) > 256 for value in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError("semantic expansion stable symbol IDs must be unique bounded strings")
    for value in requested:
        validate_cid(value)
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_EXPANSION_BYTES:
        raise ValueError("semantic expansion max_bytes must be an integer from 1 to 65536")

    repository = Path(repository).resolve(strict=True)
    payload = json.loads(load_semantic_worker_context(repository=repository,
        artifact=artifact, expected_sha256=expected_sha256, task_id=task_id))
    manifest = payload["manifest"]
    program = payload.get("program_paths")
    if payload["scope_cid"] != cid_for_payload(_source_scope_payload(manifest, program)):
        raise ValueError("semantic expansion source scope identity differs")
    source_paths = set(manifest if program is None else program)
    artifact_path = (repository / artifact).resolve(strict=True)
    directory = artifact_path.parent / "blocks"
    if not directory.resolve(strict=True).is_relative_to(repository):
        raise ValueError("semantic expansion block directory escapes repository")
    provider = IpfsDatasetsSemanticStateProvider()
    view = provider.open_verified_view(payload["semantic_root_cid"], _block_reader(directory))
    if view.root.repository_id != cid_for_payload({"repository": str(repository)}):
        raise ValueError("semantic expansion root belongs to a different repository")
    projection = payload.get("worker_projection")
    if projection is not None and projection.get("capsule_index_cid") != view.root.capsule_index_cid:
        raise ValueError("semantic expansion capsule index differs from nominated root")

    response = {
        "schema": "supervisor-semantic-capsule-expansion@1",
        "task_id": task_id,
        "worker_payload_sha256": expected_sha256,
        "scope_cid": payload["scope_cid"],
        "semantic_root_cid": view.root.root_cid,
        "capsule_index_cid": view.root.capsule_index_cid,
        "requested_stable_symbol_ids": list(requested),
        "capsules": [],
        "admissions": [],
        "raw_source_fetch_required": {},
        "raw_source_policy": "Read exact repository source and verify SHA256 before editing.",
        "semantic_equivalence_claimed": False,
        "proof_authority": False,
        "execution_authority": False,
        "completion_authority": False,
    }
    sources, source_bytes = {}, 0
    for stable_id in requested:
        capsule = view.capsule(stable_id)
        name = capsule.source_slice_path
        if name not in source_paths or capsule.source_cid != manifest[name].get("source_cid"):
            raise ValueError("semantic expansion capsule source is outside nominated scope")
        # Version-one worker loading checks SHA256; expansion also binds native
        # producer source CIDs to those same current bytes.
        if name not in sources:
            sources[name] = _read_source(repository, name)
            source_bytes += len(sources[name])
            if source_bytes > _MAX_SOURCE_BYTES:
                raise ValueError("semantic expansion source scope exceeds byte bound")
        source = sources[name]
        if (hashlib.sha256(source).hexdigest() != manifest[name]["sha256"]
                or cid_for_bytes(source) != capsule.source_cid):
            raise SemanticContextStale("semantic expansion capsule source is stale: " + name)
        # The producer does not materialize every relevant-binding projection
        # in a bundle. Preserve its freshness policy for those optional blocks.
        projection_cid = capsule.relevant_binding_projection_cid
        if projection_cid is not None:
            optional_block = directory / projection_cid
            if optional_block.exists() or optional_block.is_symlink():
                # A present block must verify; the native assessor treats read
                # failure as optional absence, which cannot hide corruption.
                view.get_block(projection_cid)
        freshness = provider.assess_capsule_freshness(capsule, current_state=view)
        admission = admit_capsule(capsule, semantic_state_root_cid=view.root.root_cid,
            assessment=freshness, force_raw_source=name in payload["required_raw_paths"])
        if admission.freshness != "fresh":
            raise ValueError("semantic expansion capsule freshness could not be verified")
        response["capsules"].append(capsule.to_dict())
        response["admissions"].append(admission.to_dict())
        response["raw_source_fetch_required"][name] = dict(manifest[name])
        if len(_encoded(response)) > max_bytes:
            raise ValueError("semantic expansion response exceeds byte budget")

    # Catch dirty edits to any nominated dependency during expansion, not just
    # changes to an expanded symbol. Do not repeat expensive semantic replay.
    source_bytes = 0
    for name, binding in manifest.items():
        current = _read_source(repository, name)
        source_bytes += len(current)
        if source_bytes > _MAX_SOURCE_BYTES:
            raise ValueError("semantic expansion source scope exceeds byte bound")
        if hashlib.sha256(current).hexdigest() != binding["sha256"]:
            raise SemanticContextStale("semantic expansion source changed during expansion: " + name)
    return response


def main(argv=None):
    """Explicit terminal expansion tool; emits one compact JSON response."""
    import argparse
    from contextlib import redirect_stdout
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--symbol", required=True, action="append", dest="stable_symbol_ids")
    parser.add_argument("--max-bytes", type=int, default=32768)
    arguments = parser.parse_args(argv)
    try:
        # Optional producer imports may print diagnostics; stdout is the tool
        # result channel and must contain exactly one JSON document.
        with redirect_stdout(sys.stderr):
            response = expand_semantic_capsules(**vars(arguments))
    except (ValueError, OSError, KeyError) as error:
        print("semantic capsule expansion failed: " + str(error), file=sys.stderr)
        return 1
    print(_encoded(response).decode())
    return 0


__all__ = ["MAX_EXPANSION_CAPSULES", "MAX_EXPANSION_BYTES", "expand_semantic_capsules"]


if __name__ == "__main__":
    raise SystemExit(main())
