"""PCPR-032 quarantined pseudo-CID identity.

Hexadecimal SHA-256 slices and Qm-prefixed random strings are not IPFS CIDs.
Ordinary runtime mints CIDv1 from canonical bytes through
``CanonicalIPFSMultiformats``. Mock IPFS clients and random Qm generators
require the compatibility/simulation namespace. Simulated identities are
never live. Missing IPFS storage stays typed unavailable.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Final, Union

from ipfs_accelerate_py.assurance.capability_outcomes import (
    is_explicit_test_mode,
    select_simulation_namespace,
)
from ipfs_accelerate_py.assurance.content_identity import (
    CanonicalIPFSMultiformats,
    ContentIdentityError,
    IdentityErrorCode,
    get_cid,
    mint_content_identity,
    reject_pseudo_cid,
)

MOCK_IPFS_NAMESPACE: Final = "mock_ipfs"
PSEUDO_CID_NAMESPACE: Final = "mock_ipfs"
ORDINARY_RUNTIME: Final = False
SIMULATED_RESULTS_ARE_LIVE: Final = False
PSEUDO_CID_RANDOM_QM_REMOVED: Final = "pseudo_cid_random_qm_removed"
IPFS_STORE_UNAVAILABLE: Final = "ipfs_store_unavailable"


class PseudoCidIdentityError(RuntimeError):
    """Ordinary runtime attempted to mint or use a pseudo-CID."""

    def __init__(self, message: str, *, code: str = "pseudo_cid_quarantined"):
        super().__init__(message)
        self.code = code
        self.outcome = "Unavailable"
        self.live = False
        self.simulated = False
        self.production_authorized = False


def _explicit_simulation_requested(
    *,
    explicit_simulation: bool,
    explicit_test_mode: bool | None,
    environ: Mapping[str, str] | None,
) -> bool:
    if explicit_simulation is True:
        return True
    return is_explicit_test_mode(explicit_test_mode=explicit_test_mode, environ=environ)


def _require_simulation(
    namespace: str,
    *,
    explicit_simulation: bool,
    explicit_test_mode: bool | None,
    environ: Mapping[str, str] | None,
    label: str,
) -> None:
    if _explicit_simulation_requested(
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    ):
        selected = select_simulation_namespace(
            namespace,
            explicit_test_mode=True,
            environ=environ,
        )
        if selected.outcome != "Simulated":
            raise PseudoCidIdentityError(selected.message, code=str(selected.code))
        return
    refused = select_simulation_namespace(
        namespace,
        explicit_test_mode=False,
        environ=environ,
    )
    raise PseudoCidIdentityError(
        f"{label} is quarantined behind the compatibility/simulation namespace; "
        "ordinary runtime cannot instantiate it. Pass explicit_simulation=True "
        "or set IPFS_ACCELERATE_EXPLICIT_TEST_MODE. "
        f"{refused.message}",
        code=str(refused.code),
    )


def load_ordinary_multiformats() -> CanonicalIPFSMultiformats:
    """Ordinary runtime uses canonical CIDv1, never hex or Qm pseudo-CIDs."""

    return CanonicalIPFSMultiformats()


def mint_canonical_cid(data: Any) -> str:
    """Mint a verified CIDv1 from canonical bytes. Never returns hex or Qm."""

    return get_cid(data)


def unavailable_ipfs_store_result(*, reason: str = "ipfs_daemon_unavailable") -> dict[str, Any]:
    """Typed unavailable envelope for ordinary IPFS store claims."""

    return {
        "status": "unavailable",
        "outcome": "Unavailable",
        "origin": "unavailable",
        "live": False,
        "simulated": False,
        "production_authorized": False,
        "cid": None,
        "code": IPFS_STORE_UNAVAILABLE,
        "reason": reason,
    }


def ordinary_store_to_ipfs(data: bytes) -> dict[str, Any]:
    """Ordinary runtime cannot claim IPFS storage without a live daemon."""

    del data
    return unavailable_ipfs_store_result(
        reason="store_to_ipfs has no live IPFS daemon; missing storage stays typed unavailable"
    )


def simulate_store_to_ipfs(
    data: bytes,
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Explicit simulation mints a canonical CID and labels the store Simulated.

    The CID is real content identity from retained bytes. The store is not live
    IPFS and is never represented as live.
    """

    _require_simulation(
        MOCK_IPFS_NAMESPACE,
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
        label="Simulated IPFS store",
    )
    identity = mint_content_identity(data)
    return {
        "status": "simulated",
        "outcome": "Simulated",
        "origin": "simulated",
        "live": False,
        "simulated": True,
        "production_authorized": False,
        "cid": identity.cid,
        "digest_hex": identity.digest_hex,
        "codec": identity.codec,
        "code": "simulated_ipfs_store_not_live",
    }


def random_cid(
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Former Qm-prefixed random generator. Removed as CID authority.

    Even explicit simulation cannot mint identity without canonical bytes.
    Call ``mint_canonical_cid`` / ``mint_content_identity`` instead.
    """

    _require_simulation(
        MOCK_IPFS_NAMESPACE,
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
        label="random_cid",
    )
    raise PseudoCidIdentityError(
        "random_cid cannot mint identity without canonical bytes. "
        "Qm-prefixed random strings are not CIDs. Use mint_canonical_cid(data).",
        code=PSEUDO_CID_RANDOM_QM_REMOVED,
    )


class UnavailableIpfsClient:
    """Ordinary-runtime stand-in. Store and identity claims stay unavailable."""

    origin = "unavailable"
    live = False
    simulated = False
    production_authorized = False
    outcome = "Unavailable"

    def __init__(self) -> None:
        self.files: dict[str, Any] = {}
        self.pins: dict[str, Any] = {}
        self.mfs: dict[str, Any] = {}

    def add_file(self, path: str, wrap_with_directory: bool = False) -> dict[str, Any]:
        del path, wrap_with_directory
        return unavailable_ipfs_store_result(reason="mock IPFS client is not ordinary runtime")

    def cat(self, cid: str, offset: int = 0, length: int = -1) -> bytes:
        del offset, length
        raise PseudoCidIdentityError(
            f"IPFS cat is typed unavailable for {cid!r}; missing daemon is not empty content",
            code=IPFS_STORE_UNAVAILABLE,
        )


class MockIPFSClient:
    """Compatibility/simulation IPFS client. Not live.

    Content identity uses canonical CIDv1 from retained bytes. Random Qm
    strings are not emitted.
    """

    origin = "simulated"
    live = False
    simulated = True
    production_authorized = False
    outcome = "Simulated"

    def __init__(
        self,
        *,
        explicit_simulation: bool = False,
        explicit_test_mode: bool | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        _require_simulation(
            MOCK_IPFS_NAMESPACE,
            explicit_simulation=explicit_simulation,
            explicit_test_mode=explicit_test_mode,
            environ=environ,
            label="MockIPFSClient",
        )
        self.files: dict[str, Any] = {}
        self.pins: dict[str, Any] = {}
        self.mfs: dict[str, Any] = {}
        self._multiformats = CanonicalIPFSMultiformats()

    def _cid_for_bytes(self, payload: bytes) -> str:
        return self._multiformats.get_cid(payload)

    def add_file(self, path: str, wrap_with_directory: bool = False) -> dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        payload = PathBytes.read(path)
        cid = self._cid_for_bytes(payload)
        self.files[cid] = {
            "path": path,
            "size": len(payload),
            "name": os.path.basename(path),
            "wrapped": wrap_with_directory,
            "data": payload,
            "origin": "simulated",
            "live": False,
        }
        return {
            "Hash": cid,
            "Size": len(payload),
            "Name": os.path.basename(path),
            "origin": "simulated",
            "live": False,
            "outcome": "Simulated",
        }

    def cat(self, cid: str, offset: int = 0, length: int = -1) -> bytes:
        reject_pseudo_cid(cid)
        if cid not in self.files:
            raise PseudoCidIdentityError(
                f"simulated IPFS client has no retained bytes for {cid!r}",
                code="simulated_ipfs_missing_bytes",
            )
        data = self.files[cid].get("data")
        if not isinstance(data, (bytes, bytearray)):
            raise PseudoCidIdentityError(
                "simulated IPFS client cannot invent content for a missing payload",
                code="simulated_ipfs_missing_bytes",
            )
        payload = bytes(data)
        if offset > 0:
            payload = payload[offset:]
        if length > 0:
            payload = payload[:length]
        return payload

    def ls(self, cid: str) -> dict[str, Any]:
        reject_pseudo_cid(cid)
        return {
            "Objects": [{"Hash": cid, "Links": []}],
            "origin": "simulated",
            "live": False,
            "outcome": "Simulated",
        }

    def files_mkdir(self, path: str, parents: bool = False) -> None:
        del parents
        marker = mint_content_identity({"mfs": path, "type": "directory"})
        self.mfs[path] = {
            "type": "directory",
            "cid": marker.cid,
            "size": 0,
            "created": time.time(),
            "origin": "simulated",
            "live": False,
        }

    def files_stat(self, path: str) -> dict[str, Any]:
        if path.startswith("/ipfs/"):
            cid = path[6:]
            reject_pseudo_cid(cid)
            return {
                "Hash": cid,
                "Size": 0,
                "CumulativeSize": 0,
                "Blocks": 0,
                "Type": "file",
                "origin": "simulated",
                "live": False,
            }
        if path not in self.mfs:
            raise FileNotFoundError(f"File not found in simulated MFS: {path}")
        entry = self.mfs[path]
        return {
            "Hash": entry["cid"],
            "Size": entry["size"],
            "CumulativeSize": entry["size"],
            "Blocks": 1,
            "Type": entry["type"],
            "origin": "simulated",
            "live": False,
        }

    def files_write(
        self, path: str, data: bytes, create: bool = True, truncate: bool = True
    ) -> None:
        del create, truncate
        payload = data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
        identity = mint_content_identity(bytes(payload))
        self.mfs[path] = {
            "type": "file",
            "cid": identity.cid,
            "size": len(payload),
            "created": time.time(),
            "data": bytes(payload),
            "origin": "simulated",
            "live": False,
        }

    def files_read(self, path: str, offset: int = 0, count: int = -1) -> bytes:
        if path not in self.mfs:
            raise FileNotFoundError(f"File not found in MFS: {path}")
        if self.mfs[path]["type"] != "file":
            raise ValueError(f"Not a file: {path}")
        data = self.mfs[path].get("data")
        if not isinstance(data, (bytes, bytearray)):
            raise PseudoCidIdentityError(
                "simulated MFS cannot invent file bytes",
                code="simulated_ipfs_missing_bytes",
            )
        payload = bytes(data)
        if offset > 0:
            payload = payload[offset:]
        if count > 0:
            payload = payload[:count]
        return payload

    def pin_add(self, cid: str, recursive: bool = True) -> dict[str, Any]:
        reject_pseudo_cid(cid)
        self.pins[cid] = {
            "type": "recursive" if recursive else "direct",
            "pinned_at": time.time(),
        }
        return {"Pins": [cid], "origin": "simulated", "live": False}

    def pin_ls(self, cid: str | None = None) -> dict[str, Any]:
        if cid is not None:
            reject_pseudo_cid(cid)
            if cid not in self.pins:
                return {"Keys": {}, "origin": "simulated", "live": False}
            return {
                "Keys": {cid: {"Type": self.pins[cid]["type"]}},
                "origin": "simulated",
                "live": False,
            }
        result: dict[str, Any] = {"Keys": {}, "origin": "simulated", "live": False}
        for pin_cid, pin_info in self.pins.items():
            result["Keys"][pin_cid] = {"Type": pin_info["type"]}
        return result

    def pin_rm(self, cid: str, recursive: bool = True) -> dict[str, Any]:
        del recursive
        reject_pseudo_cid(cid)
        self.pins.pop(cid, None)
        return {"Pins": [cid], "origin": "simulated", "live": False}

    def id(self) -> dict[str, Any]:
        identity = mint_content_identity({"mock_ipfs_peer": "simulated", "live": False})
        return {
            "ID": identity.cid,
            "PublicKey": "",
            "Addresses": [],
            "AgentVersion": "ipfs-accelerate-py/simulated",
            "ProtocolVersion": "ipfs/0.1.0",
            "origin": "simulated",
            "live": False,
            "outcome": "Simulated",
        }

    def swarm_peers(self) -> dict[str, Any]:
        return {"Peers": [], "origin": "simulated", "live": False}

    def swarm_connect(self, addr: str) -> dict[str, Any]:
        return {
            "Strings": [f"simulated connection not live: {addr}"],
            "origin": "simulated",
            "live": False,
        }

    def pubsub_pub(self, topic: str, message: Union[str, bytes]) -> None:
        del topic, message

    def dht_findpeer(self, peer_id: str) -> dict[str, Any]:
        return {"Responses": [], "origin": "simulated", "live": False, "ID": peer_id}

    def dht_findprovs(self, cid: str, num_providers: int = 20) -> dict[str, Any]:
        del num_providers
        reject_pseudo_cid(cid)
        return {"Responses": [], "origin": "simulated", "live": False}

    def version(self) -> dict[str, Any]:
        return {
            "Version": "ipfs-accelerate-py/simulated",
            "Commit": "",
            "Repo": "unavailable",
            "System": "ipfs-accelerate-py/simulated",
            "Golang": "unavailable",
            "origin": "simulated",
            "live": False,
        }


class PathBytes:
    """Read exact file bytes for canonical CID minting."""

    @staticmethod
    def read(path: str) -> bytes:
        with open(path, "rb") as handle:
            payload = handle.read()
        if not payload:
            raise ContentIdentityError(
                "canonical bytes must be nonempty",
                code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
            )
        return payload


def load_ordinary_ipfs_client() -> UnavailableIpfsClient:
    """Ordinary runtime uses typed unavailable IPFS, never a mock client."""

    return UnavailableIpfsClient()


def instantiate_mock_ipfs_client(
    *,
    explicit_simulation: bool = False,
    explicit_test_mode: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> MockIPFSClient:
    """Construct a mock IPFS client only under explicit simulation."""

    return MockIPFSClient(
        explicit_simulation=explicit_simulation,
        explicit_test_mode=explicit_test_mode,
        environ=environ,
    )


__all__ = (
    "IPFS_STORE_UNAVAILABLE",
    "MOCK_IPFS_NAMESPACE",
    "ORDINARY_RUNTIME",
    "PSEUDO_CID_NAMESPACE",
    "PSEUDO_CID_RANDOM_QM_REMOVED",
    "SIMULATED_RESULTS_ARE_LIVE",
    "CanonicalIPFSMultiformats",
    "MockIPFSClient",
    "PseudoCidIdentityError",
    "UnavailableIpfsClient",
    "instantiate_mock_ipfs_client",
    "load_ordinary_ipfs_client",
    "load_ordinary_multiformats",
    "mint_canonical_cid",
    "ordinary_store_to_ipfs",
    "random_cid",
    "simulate_store_to_ipfs",
    "unavailable_ipfs_store_result",
)
