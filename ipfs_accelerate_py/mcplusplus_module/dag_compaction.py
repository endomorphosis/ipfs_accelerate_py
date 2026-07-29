"""Content-addressed DAG compaction for the MCP++ Event DAG.

When the EventDAG grows beyond a configurable threshold, older epochs are
compacted into a verifiable summary node. This allows:

1. **Memory efficiency**: Only recent events stay in hot memory
2. **Verifiability**: Compacted epochs always produce a Merkle root and a
   content-addressed integrity commitment
3. **Full recovery**: Cold events remain on disk and can be loaded on demand

Compaction Strategy:
- Events are grouped into epochs (default 1000 events per epoch)
- When an epoch completes, its events are hashed into a Merkle tree
- A real ZK certificate is attached only when the canonical
  ``ipfs_datasets_py`` provider both proves and verifies it
- Otherwise the summary is explicitly labelled ``hash-commitment-v1`` with
  ``zero_knowledge=False``
- The compacted epoch is replaced in memory by a single CompactionProof node
- Full event data is persisted to disk for on-demand retrieval

Module: ipfs_accelerate_py.mcplusplus_module.dag_compaction
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.dag_compaction")

# Compaction thresholds (configurable via environment variables)
EPOCH_SIZE = int(os.environ.get("MCPPP_EPOCH_SIZE", "1000"))
HOT_TIER_MAX = int(os.environ.get("MCPPP_HOT_TIER_MAX", "2000"))
COLD_TIER_DIR = os.environ.get("MCPPP_STORAGE_DIR", ".mcppp_dag_cold")
HASH_COMMITMENT_PROOF_SYSTEM = "hash-commitment-v1"


@dataclass
class MerkleNode:
    """A node in the Merkle tree used for epoch summarization."""
    hash: str
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    leaf_data: Optional[str] = None  # Only set for leaf nodes


@dataclass
class CompactionProof:
    """Certificate describing a compacted epoch.

    Contains:
    - merkle_root: Root hash of the Merkle tree over all epoch events
    - epoch_id: Sequential epoch identifier
    - event_count: Number of events in the compacted epoch
    - frontier_cids: CIDs of leaf events at epoch boundary (connect to next epoch)
    - root_cids: CIDs of root events in this epoch (connect from previous epoch)
    - proof: Hash commitment or serialized verifier-backed proof
    - proof_system: Exact proof/commitment scheme identifier
    - zero_knowledge: True only for a certificate accepted by a real verifier
    - validation_digest: Digest of the archived epoch's DAG-consistency summary
    - timestamp_range: (start, end) timestamps of epoch
    - cold_storage_path: File path where full epoch data is stored
    """
    merkle_root: str
    epoch_id: int
    event_count: int
    frontier_cids: List[str] = field(default_factory=list)
    root_cids: List[str] = field(default_factory=list)
    proof: str = ""
    proof_system: str = HASH_COMMITMENT_PROOF_SYSTEM
    zero_knowledge: bool = False
    validation_digest: str = ""
    verification_key_cid: str = ""
    verification_key_sha256: str = ""
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    cold_storage_path: str = ""
    verified: bool = False

    @property
    def cid(self) -> str:
        """Content-addressed ID for this compaction proof."""
        payload = json.dumps({
            "merkle_root": self.merkle_root,
            "epoch_id": self.epoch_id,
            "event_count": self.event_count,
            "frontier_cids": self.frontier_cids,
            "root_cids": self.root_cids,
            "proof": self.proof,
            "proof_system": self.proof_system,
            "zero_knowledge": self.zero_knowledge,
            "validation_digest": self.validation_digest,
            "verification_key_cid": self.verification_key_cid,
            "verification_key_sha256": self.verification_key_sha256,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "merkle_root": self.merkle_root,
            "epoch_id": self.epoch_id,
            "event_count": self.event_count,
            "frontier_cids": self.frontier_cids,
            "root_cids": self.root_cids,
            "proof": self.proof,
            "proof_system": self.proof_system,
            "zero_knowledge": self.zero_knowledge,
            "validation_digest": self.validation_digest,
            "verification_key_cid": self.verification_key_cid,
            "verification_key_sha256": self.verification_key_sha256,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "cold_storage_path": self.cold_storage_path,
            "verified": self.verified,
        }


# ---------------------------------------------------------------------------
# Merkle Tree Construction
# ---------------------------------------------------------------------------

def _hash_leaf(data: str) -> str:
    """Hash a leaf node (event CID + payload digest)."""
    return hashlib.sha256(data.encode()).hexdigest()


def _hash_pair(left: str, right: str) -> str:
    """Hash two child hashes to produce parent hash."""
    combined = (left + right).encode()
    return hashlib.sha256(combined).hexdigest()


def build_merkle_tree(event_cids: List[str]) -> Tuple[str, List[List[str]]]:
    """Build a Merkle tree from a list of event CIDs.

    Returns (root_hash, layers) where layers[0] = leaves, layers[-1] = [root].
    """
    if not event_cids:
        return hashlib.sha256(b"empty").hexdigest(), [[]]

    # Leaf layer
    current_layer = [_hash_leaf(cid) for cid in event_cids]
    layers = [current_layer[:]]

    # Build tree bottom-up
    while len(current_layer) > 1:
        next_layer = []
        for i in range(0, len(current_layer), 2):
            left = current_layer[i]
            right = current_layer[i + 1] if i + 1 < len(current_layer) else left
            next_layer.append(_hash_pair(left, right))
        current_layer = next_layer
        layers.append(current_layer[:])

    return current_layer[0], layers


def merkle_proof_for_cid(event_cid: str, event_cids: List[str], layers: List[List[str]]) -> List[Dict[str, str]]:
    """Generate a Merkle inclusion proof for a specific CID.

    Returns list of sibling hashes needed to verify the CID is in the tree.
    """
    try:
        idx = event_cids.index(event_cid)
    except ValueError:
        return []

    proof_path = []
    current_idx = idx

    for layer_idx in range(len(layers) - 1):
        layer = layers[layer_idx]
        # Find sibling
        if current_idx % 2 == 0:
            sibling_idx = current_idx + 1 if current_idx + 1 < len(layer) else current_idx
            proof_path.append({"side": "right", "hash": layer[sibling_idx]})
        else:
            sibling_idx = current_idx - 1
            proof_path.append({"side": "left", "hash": layer[sibling_idx]})
        current_idx = current_idx // 2

    return proof_path


def verify_merkle_proof(event_cid: str, proof_path: List[Dict[str, str]], expected_root: str) -> bool:
    """Verify a Merkle inclusion proof for a CID against a known root."""
    current_hash = _hash_leaf(event_cid)

    for step in proof_path:
        if step["side"] == "left":
            current_hash = _hash_pair(step["hash"], current_hash)
        else:
            current_hash = _hash_pair(current_hash, step["hash"])

    return current_hash == expected_root


# ---------------------------------------------------------------------------
# Profile F proof selection
# ---------------------------------------------------------------------------

def _is_sha256_hex(value: Any) -> bool:
    """Return whether *value* is a canonical lower-case SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_commitment(
    merkle_root: str,
    epoch_id: int,
    event_count: int,
    validation_digest: str,
) -> str:
    """Bind the public compaction metadata to a deterministic commitment."""
    commitment_input = json.dumps({
        "domain": "mcp++-event-dag-compaction",
        "proof_system": HASH_COMMITMENT_PROOF_SYSTEM,
        "merkle_root": merkle_root,
        "epoch_id": epoch_id,
        "event_count": event_count,
        "validation_digest": validation_digest,
    }, sort_keys=True, separators=(",", ":"))
    first_hash = hashlib.sha256(commitment_input.encode()).digest()
    return hashlib.sha256(first_hash).hexdigest()


def _generate_hash_commitment(
    epoch_events: List[Dict[str, Any]],
    merkle_root: str,
    epoch_id: int,
) -> Tuple[str, str]:
    validation_digest = _compute_validation_digest(epoch_events)
    commitment = _hash_commitment(
        merkle_root=merkle_root,
        epoch_id=epoch_id,
        event_count=len(epoch_events),
        validation_digest=validation_digest,
    )
    return commitment, validation_digest


def _profile_f_zk_certificate(event_cids: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Return a verifier-accepted certificate, or no ZK claim.

    ``MCPPP_PROFILE_F_ZK=required`` makes provider failure fatal. The legacy
    ``IPFS_DATASETS_ENABLE_GROTH16`` switch remains an opt-in alias, but it can
    never cause a hash commitment to be labelled as Groth16.
    """
    configured_mode = os.environ.get("MCPPP_PROFILE_F_ZK")
    if configured_mode is None:
        legacy_mode = os.environ.get("IPFS_DATASETS_ENABLE_GROTH16", "0")
        configured_mode = "1" if legacy_mode.strip().lower() in {"1", "true", "yes"} else "0"
    mode = configured_mode.strip().lower()
    if mode not in {"1", "true", "yes", "required"}:
        return None

    try:
        from ipfs_datasets_py.mcp_server.event_dag_zkp import (
            availability,
            prove_event_dag_compaction,
            verify_event_dag_compaction,
        )

        provider_status = availability()
        if not isinstance(provider_status, Mapping) or provider_status.get("available") is not True:
            raise RuntimeError("canonical Profile F verifier is unavailable")

        certificate = prove_event_dag_compaction(list(event_cids))
        if not isinstance(certificate, Mapping):
            raise RuntimeError("canonical Profile F prover returned a non-object certificate")
        proof_system = certificate.get("proof_system")
        if (
            not isinstance(proof_system, str)
            or not proof_system
            or proof_system == HASH_COMMITMENT_PROOF_SYSTEM
            or certificate.get("zero_knowledge") is not True
            or certificate.get("event_count") != len(event_cids)
        ):
            raise RuntimeError("canonical Profile F prover returned invalid certificate metadata")
        if provider_status.get("proof_system") not in {None, proof_system}:
            raise RuntimeError("Profile F provider and certificate proof systems differ")

        for key_field in ("verification_key_cid", "verification_key_sha256"):
            key_value = certificate.get(key_field)
            if not isinstance(key_value, str) or not key_value:
                raise RuntimeError(f"Profile F certificate is missing {key_field}")
            provider_value = provider_status.get(key_field)
            if provider_value is not None and provider_value != key_value:
                raise RuntimeError(f"Profile F certificate has an unexpected {key_field}")
        if not _is_sha256_hex(certificate["verification_key_sha256"]):
            raise RuntimeError("Profile F certificate has an invalid verification key digest")

        verification = verify_event_dag_compaction(certificate, list(event_cids))
        if not isinstance(verification, Mapping) or verification.get("valid") is not True:
            raise RuntimeError("canonical Profile F verifier rejected the generated certificate")
        return dict(certificate)
    except Exception as error:
        if mode == "required":
            raise RuntimeError("Profile F ZK proof is required but unavailable") from error
        logger.warning(
            "Profile F ZK proof unavailable; retaining %s: %s",
            HASH_COMMITMENT_PROOF_SYSTEM,
            error,
        )
        return None


def generate_compaction_proof(
    epoch_events: List[Dict[str, Any]],
    merkle_root: str,
    epoch_id: int,
) -> str:
    """Generate a deterministic content-addressed integrity commitment.

    This function deliberately never returns simulated ZK material. A real ZK
    certificate is selected separately so its proof-system metadata cannot be
    lost through this legacy string-returning API.
    """
    commitment, _ = _generate_hash_commitment(epoch_events, merkle_root, epoch_id)
    return commitment


def _compute_validation_digest(epoch_events: List[Dict[str, Any]]) -> str:
    """Summarize internal and cross-epoch references deterministically.

    This digest is commitment material, not an independent proof that the DAG
    is valid. ``verify_cold_epoch`` recomputes it from the durable archive.
    """
    all_cids = {e.get("cid", "") for e in epoch_events}
    cross_epoch_parents = set()

    for event in epoch_events:
        for parent_cid in event.get("parent_cids", []):
            if parent_cid not in all_cids:
                cross_epoch_parents.add(parent_cid)

    digest_input = json.dumps({
        "internal_event_count": len(all_cids),
        "cross_epoch_parent_count": len(cross_epoch_parents),
        "cross_epoch_parents": sorted(cross_epoch_parents)[:10],  # Cap for proof size
    }, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(digest_input.encode()).hexdigest()


def verify_compaction_proof(
    proof: CompactionProof,
    event_cids: Optional[Sequence[str]] = None,
) -> bool:
    """Verify a compaction certificate without trusting its persisted status."""
    if not isinstance(proof, CompactionProof) or not isinstance(proof.proof, str):
        return False
    if (
        type(proof.epoch_id) is not int
        or proof.epoch_id < 0
        or type(proof.event_count) is not int
        or proof.event_count < 0
        or not _is_sha256_hex(proof.merkle_root)
        or not _is_sha256_hex(proof.validation_digest)
    ):
        return False

    canonical_event_cids = None
    if event_cids is not None:
        try:
            canonical_event_cids = list(event_cids)
        except TypeError:
            return False
        if (
            len(canonical_event_cids) != proof.event_count
            or any(
                not isinstance(event_cid, str) or not event_cid
                for event_cid in canonical_event_cids
            )
        ):
            return False
        computed_root, _ = build_merkle_tree(canonical_event_cids)
        if not hmac.compare_digest(computed_root, proof.merkle_root):
            return False

    if proof.zero_knowledge is True:
        if (
            canonical_event_cids is None
            or proof.proof_system == HASH_COMMITMENT_PROOF_SYSTEM
            or not isinstance(proof.proof_system, str)
            or not proof.proof_system
            or not isinstance(proof.verification_key_cid, str)
            or not proof.verification_key_cid
            or not _is_sha256_hex(proof.verification_key_sha256)
        ):
            return False
        try:
            certificate = json.loads(proof.proof)
            if (
                not isinstance(certificate, Mapping)
                or certificate.get("proof_system") != proof.proof_system
                or certificate.get("zero_knowledge") is not True
                or certificate.get("event_count") != proof.event_count
                or certificate.get("verification_key_cid") != proof.verification_key_cid
                or certificate.get("verification_key_sha256") != proof.verification_key_sha256
            ):
                return False
            from ipfs_datasets_py.mcp_server.event_dag_zkp import verify_event_dag_compaction

            verification = verify_event_dag_compaction(
                certificate,
                canonical_event_cids,
            )
            return bool(
                isinstance(verification, Mapping)
                and verification.get("valid") is True
            )
        except Exception:
            return False

    if (
        proof.zero_knowledge is not False
        or proof.proof_system != HASH_COMMITMENT_PROOF_SYSTEM
        or not _is_sha256_hex(proof.proof)
    ):
        return False

    expected = _hash_commitment(
        merkle_root=proof.merkle_root,
        epoch_id=proof.epoch_id,
        event_count=proof.event_count,
        validation_digest=proof.validation_digest,
    )
    return hmac.compare_digest(proof.proof, expected)


# ---------------------------------------------------------------------------
# DAG Epoch Compactor
# ---------------------------------------------------------------------------

class DAGCompactor:
    """Manages epoch-based compaction of an EventDAG.

    Splits the DAG into hot (in-memory), cold (on-disk), and compacted
    (integrity commitment, optionally verifier-backed ZK) tiers.

    Usage:
        compactor = DAGCompactor(storage_dir="/path/to/dag_storage")
        # Called after each append:
        compaction = compactor.maybe_compact(dag_events, dag_children)
        if compaction:
            # Remove compacted events from hot tier
            for cid in compaction.compacted_cids:
                del dag_events[cid]
    """

    def __init__(self, storage_dir: str = COLD_TIER_DIR, epoch_size: int = EPOCH_SIZE):
        self.storage_dir = storage_dir
        self.epoch_size = epoch_size
        self._lock = threading.Lock()
        self._current_epoch_id = 0
        self._compaction_proofs: List[CompactionProof] = []
        self._total_compacted_events = 0
        self._max_storage_bytes = int(os.environ.get(
            "MCPPP_MAX_COLD_STORAGE_GB", "10")) * 1024 ** 3
        # Unique server identifier to prevent multi-server epoch collisions
        import socket
        self._server_id = f"{socket.gethostname()}_{os.getpid()}"

        os.makedirs(storage_dir, exist_ok=True)
        self._load_compaction_index()

    def _check_storage_quota(self) -> bool:
        """Check if cold storage is within quota. Returns False if over limit."""
        try:
            total_size = sum(
                os.path.getsize(os.path.join(self.storage_dir, f))
                for f in os.listdir(self.storage_dir)
                if os.path.isfile(os.path.join(self.storage_dir, f))
            )
            if total_size >= self._max_storage_bytes:
                logger.error(
                    "Cold storage quota exceeded: %.1f GB >= %d GB, compaction skipped",
                    total_size / (1024 ** 3), self._max_storage_bytes // (1024 ** 3),
                )
                return False
            return True
        except OSError:
            return True  # Graceful degradation: allow compaction if quota check fails

    def _load_compaction_index(self) -> None:
        """Load previously saved compaction proofs index."""
        index_path = os.path.join(self.storage_dir, "compaction_index.json")
        if os.path.isfile(index_path):
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                self._current_epoch_id = data.get("current_epoch_id", 0)
                self._total_compacted_events = data.get("total_compacted_events", 0)
                for pd in data.get("proofs", []):
                    loaded_proof = CompactionProof(
                        merkle_root=pd["merkle_root"],
                        epoch_id=pd["epoch_id"],
                        event_count=pd["event_count"],
                        frontier_cids=pd.get("frontier_cids", []),
                        root_cids=pd.get("root_cids", []),
                        proof=pd.get("proof", ""),
                        proof_system=pd.get(
                            "proof_system",
                            HASH_COMMITMENT_PROOF_SYSTEM,
                        ),
                        zero_knowledge=pd.get("zero_knowledge", False),
                        validation_digest=pd.get("validation_digest", ""),
                        verification_key_cid=pd.get("verification_key_cid", ""),
                        verification_key_sha256=pd.get(
                            "verification_key_sha256",
                            "",
                        ),
                        timestamp_start=pd.get("timestamp_start", 0.0),
                        timestamp_end=pd.get("timestamp_end", 0.0),
                        cold_storage_path=pd.get("cold_storage_path", ""),
                        verified=False,
                    )
                    loaded_proof.verified = verify_compaction_proof(loaded_proof)
                    if pd.get("verified") is True and not loaded_proof.verified:
                        logger.warning(
                            "Compaction proof for epoch %s failed fresh verification",
                            loaded_proof.epoch_id,
                        )
                    self._compaction_proofs.append(loaded_proof)
                logger.info(
                    "Loaded compaction index: %d epochs, %d total compacted events",
                    len(self._compaction_proofs), self._total_compacted_events,
                )
                # Validate that referenced epoch files actually exist
                missing = []
                for proof in self._compaction_proofs:
                    epoch_path = os.path.join(self.storage_dir, f"epoch_{proof.epoch_id:06d}.json")
                    if not os.path.isfile(epoch_path):
                        missing.append(proof.epoch_id)
                if missing:
                    logger.warning(
                        "Compaction index references %d missing epoch files: %s",
                        len(missing), missing[:10],
                    )
            except (json.JSONDecodeError, KeyError, OSError, IOError) as e:
                logger.warning("Failed to load compaction index: %s", e)

    def _save_compaction_index(self) -> None:
        """Save compaction proofs index to disk atomically (write-then-rename)."""
        index_path = os.path.join(self.storage_dir, "compaction_index.json")
        tmp_path = index_path + ".tmp"
        data = {
            "current_epoch_id": self._current_epoch_id,
            "total_compacted_events": self._total_compacted_events,
            "proofs": [p.to_dict() for p in self._compaction_proofs],
        }
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, index_path)
        except (OSError, IOError) as e:
            logger.error("Failed to save compaction index: %s", e)
            # Clean up partial write
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def should_compact(self, hot_event_count: int) -> bool:
        """Check if compaction should be triggered."""
        return hot_event_count >= HOT_TIER_MAX

    def compact_epoch(
        self,
        events: Dict[str, Any],
        children: Dict[str, List[str]],
    ) -> Optional["CompactionResult"]:
        """Compact the oldest epoch into a verified integrity certificate.

        Args:
            events: Dict of cid -> event data (mutable, will NOT be modified here)
            children: Dict of parent_cid -> [child_cids]

        Returns:
            CompactionResult with the CIDs that were compacted, or None if no compaction needed.
        """
        # Check cold storage quota before writing
        if not self._check_storage_quota():
            return None

        with self._lock:
            if len(events) < self.epoch_size:
                return None

            # Sort events by timestamp to identify the oldest epoch
            sorted_events = sorted(
                events.items(),
                key=lambda kv: kv[1].get("timestamp", 0) if isinstance(kv[1], dict)
                else getattr(kv[1], "timestamp", 0)
            )

            # Take the oldest epoch_size events
            epoch_items = sorted_events[:self.epoch_size]
            epoch_cids = [cid for cid, _ in epoch_items]

            # Serialize events for cold storage
            epoch_data = []
            for cid, event in epoch_items:
                if isinstance(event, dict):
                    serialized_event = dict(event)
                    serialized_event["cid"] = cid
                    epoch_data.append(serialized_event)
                else:
                    # Dataclass event — serialize
                    epoch_data.append({
                        "cid": cid,
                        "event_type": getattr(event, "event_type", "unknown"),
                        "parent_cids": getattr(event, "parent_cids", []),
                        "payload": getattr(event, "payload", {}),
                        "timestamp": getattr(event, "timestamp", 0),
                    })

            # Determine epoch boundaries
            frontier_cids = []
            root_cids = []
            epoch_cid_set = set(epoch_cids)
            for cid, event in epoch_items:
                parent_cids = (event.get("parent_cids", []) if isinstance(event, dict)
                               else getattr(event, "parent_cids", []))
                # Root: parents are not in this epoch
                if not parent_cids or all(p not in epoch_cid_set for p in parent_cids):
                    root_cids.append(cid)
                # Frontier: no children in this epoch
                child_list = children.get(cid, [])
                if not child_list or all(c not in epoch_cid_set for c in child_list):
                    frontier_cids.append(cid)

            # Build Merkle tree
            merkle_root, layers = build_merkle_tree(epoch_cids)

            # Persist cold epoch to disk (atomic write with flock to prevent multi-server collision)
            import fcntl
            index_path = os.path.join(self.storage_dir, "compaction_index.json")
            try:
                with open(index_path, "r+") as idx_f:
                    fcntl.flock(idx_f.fileno(), fcntl.LOCK_EX)
                    # Re-read epoch ID under lock (another server may have incremented)
                    idx_data = json.load(idx_f)
                    self._current_epoch_id = idx_data.get("current_epoch_id", self._current_epoch_id)
            except (OSError, json.JSONDecodeError):
                pass  # First epoch, no index yet

            cold_path = os.path.join(
                self.storage_dir, f"epoch_{self._current_epoch_id:06d}.json"
            )

            # Always generate a re-verifiable hash commitment. Replace its
            # proof bytes only when the canonical ZK provider returns a
            # certificate which it has also accepted.
            commitment, validation_digest = _generate_hash_commitment(
                epoch_data,
                merkle_root,
                self._current_epoch_id,
            )
            zk_certificate = _profile_f_zk_certificate(epoch_cids)
            proof_value = (
                json.dumps(zk_certificate, sort_keys=True, separators=(",", ":"))
                if zk_certificate
                else commitment
            )
            timestamps = [
                e.get("timestamp", 0) if isinstance(e, dict) else getattr(e, "timestamp", 0)
                for _, e in epoch_items
            ]
            compaction = CompactionProof(
                merkle_root=merkle_root,
                epoch_id=self._current_epoch_id,
                event_count=len(epoch_cids),
                frontier_cids=frontier_cids[:50],  # Cap for memory
                root_cids=root_cids[:50],
                proof=proof_value,
                proof_system=(
                    zk_certificate.get("proof_system")
                    if zk_certificate
                    else HASH_COMMITMENT_PROOF_SYSTEM
                ),
                zero_knowledge=bool(
                    zk_certificate
                    and zk_certificate.get("zero_knowledge") is True
                ),
                validation_digest=validation_digest,
                verification_key_cid=(
                    zk_certificate.get("verification_key_cid", "")
                    if zk_certificate
                    else ""
                ),
                verification_key_sha256=(
                    zk_certificate.get("verification_key_sha256", "")
                    if zk_certificate
                    else ""
                ),
                timestamp_start=min(timestamps) if timestamps else 0.0,
                timestamp_end=max(timestamps) if timestamps else 0.0,
                cold_storage_path=cold_path,
                verified=False,
            )
            compaction.verified = verify_compaction_proof(compaction, epoch_cids)
            if not compaction.verified:
                logger.error(
                    "Generated compaction certificate for epoch %d failed verification",
                    self._current_epoch_id,
                )
                return None

            tmp_cold_path = cold_path + ".tmp"
            try:
                with open(tmp_cold_path, "w") as f:
                    json.dump({
                        "epoch_id": self._current_epoch_id,
                        "merkle_root": merkle_root,
                        "events": epoch_data,
                        "merkle_layers": layers,
                    }, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_cold_path, cold_path)
            except (OSError, IOError) as e:
                logger.error("Failed to write cold epoch %d: %s", self._current_epoch_id, e)
                try:
                    os.unlink(tmp_cold_path)
                except OSError:
                    pass
                return None  # Abort compaction on disk failure

            self._compaction_proofs.append(compaction)
            self._current_epoch_id += 1
            self._total_compacted_events += len(epoch_cids)
            self._save_compaction_index()

            logger.info(
                "Compacted epoch %d: %d events → merkle_root=%s, proof=%s...",
                compaction.epoch_id, compaction.event_count,
                merkle_root[:16], compaction.proof[:16],
            )

            return CompactionResult(
                compacted_cids=epoch_cids,
                proof=compaction,
            )

    def load_cold_epoch(self, epoch_id: int) -> List[Dict[str, Any]]:
        """Load full event data from a cold epoch on disk.

        Used when provenance() needs to traverse beyond the hot tier.
        """
        cold_path = os.path.join(self.storage_dir, f"epoch_{epoch_id:06d}.json")
        if not os.path.isfile(cold_path):
            logger.warning("Cold epoch %d not found at %s", epoch_id, cold_path)
            return []

        try:
            with open(cold_path, "r") as f:
                data = json.load(f)
            events = data.get("events", [])
            if not isinstance(events, list):
                logger.error("Cold epoch %d has invalid events type: %s", epoch_id, type(events).__name__)
                return []
            return events
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("Cold epoch %d corrupted at %s: %s", epoch_id, cold_path, e)
            # Attempt recovery from .bak if available
            bak_path = cold_path + ".bak"
            if os.path.isfile(bak_path):
                try:
                    with open(bak_path, "r") as f:
                        data = json.load(f)
                    logger.info("Recovered cold epoch %d from backup", epoch_id)
                    return data.get("events", [])
                except Exception:
                    pass
            return []
        except OSError as e:
            logger.error("Cold epoch %d I/O error: %s", epoch_id, e)
            return []

    def verify_cold_epoch(self, epoch_id: int) -> bool:
        """Verify the archive and its certificate against each other."""
        # Find the proof for this epoch
        proof = None
        for p in self._compaction_proofs:
            if p.epoch_id == epoch_id:
                proof = p
                break

        if proof is None:
            return False

        # Load events and rebuild Merkle tree
        events = self.load_cold_epoch(epoch_id)
        if not events:
            return False

        if (
            not all(isinstance(event, dict) for event in events)
            or not _is_sha256_hex(proof.merkle_root)
            or not _is_sha256_hex(proof.validation_digest)
        ):
            proof.verified = False
            return False
        event_cids = [event.get("cid", "") for event in events]
        computed_root, _ = build_merkle_tree(event_cids)
        computed_validation_digest = _compute_validation_digest(events)
        is_valid = (
            hmac.compare_digest(computed_root, proof.merkle_root)
            and hmac.compare_digest(
                computed_validation_digest,
                proof.validation_digest,
            )
            and verify_compaction_proof(proof, event_cids)
        )
        proof.verified = is_valid
        return is_valid

    def find_epoch_for_cid(self, cid: str) -> Optional[int]:
        """Find which cold epoch (if any) contains a given CID.

        Checks frontier_cids and root_cids first (O(1) per proof),
        then falls back to scanning cold files with a CID index cache.
        """
        # Fast path: check proof metadata
        for proof in self._compaction_proofs:
            if cid in proof.frontier_cids or cid in proof.root_cids:
                return proof.epoch_id

        # Check CID→epoch index cache
        if not hasattr(self, '_cid_epoch_index'):
            self._cid_epoch_index: dict = {}

        if cid in self._cid_epoch_index:
            return self._cid_epoch_index[cid]

        # Scan cold epochs (expensive, but build index as we go)
        for proof in self._compaction_proofs:
            if proof.epoch_id in getattr(self, '_indexed_epochs', set()):
                continue
            events = self.load_cold_epoch(proof.epoch_id)
            if not hasattr(self, '_indexed_epochs'):
                self._indexed_epochs: set = set()
            self._indexed_epochs.add(proof.epoch_id)
            for event in events:
                ecid = event.get("cid")
                if ecid:
                    self._cid_epoch_index[ecid] = proof.epoch_id
                    if ecid == cid:
                        return proof.epoch_id
        return None

    @property
    def compaction_proofs(self) -> List[CompactionProof]:
        """All compaction proofs in order."""
        return list(self._compaction_proofs)

    @property
    def total_compacted_events(self) -> int:
        return self._total_compacted_events

    def summary(self) -> Dict[str, Any]:
        """Summary of compaction state."""
        return {
            "epochs_compacted": len(self._compaction_proofs),
            "total_compacted_events": self._total_compacted_events,
            "current_epoch_id": self._current_epoch_id,
            "storage_dir": self.storage_dir,
            "proofs": [p.to_dict() for p in self._compaction_proofs[-5:]],  # Last 5
        }


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    compacted_cids: List[str]
    proof: CompactionProof
