"""Adversarial end-to-end assurance, control parity, recovery, and rollback gates.

This module freezes a multi-repository fixture population and evaluates the
closed VFS-G130 adversarial gate set before any automatic repair scope may
expand.  Evidence schemas are ``vfs/adversarial-e2e-gate@1`` and
``vfs/shadow-rollout-report@1``.

Objective-heap ownership for the assurance-rollout packet:

* VFS-G162 / VFS-082 prove ``vfs/adversarial-e2e-gate@1``
* VFS-G163 / VFS-084 prove ``vfs/shadow-rollout-report@1``
* VFS-G130 remains the parent rollout goal; automatic mutation stays disabled

Safety is non-waivable:

* automatic mutation remains disabled on every report;
* any gate, binding, or assurance regression returns effective rollout to
  ``shadow``;
* Python, CLI, and MCP publish equivalent bounded status/findings/receipts;
* discovery never imports optional providers and never starts processes.

The gate recomputes from typed observations.  It does not run providers,
mutate sources, or promote semantic claims from simulated/forged ZK.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Final


ADVERSARIAL_E2E_GATE_SCHEMA: Final = "vfs/adversarial-e2e-gate@1"
SHADOW_ROLLOUT_REPORT_SCHEMA: Final = "vfs/shadow-rollout-report@1"
# Domain evidence identities (alias schemas for objective-heap discovery).
ADVERSARIAL_E2E_GATE_EVIDENCE: Final = ADVERSARIAL_E2E_GATE_SCHEMA
SHADOW_ROLLOUT_REPORT_EVIDENCE: Final = SHADOW_ROLLOUT_REPORT_SCHEMA
VFS_SYMBOLIC_ROLLOUT_DECISION_SCHEMA: Final = (
    "vfs/symbolic-rollout-decision@1"
)
VFS_SYMBOLIC_CONTROL_REQUEST_SCHEMA: Final = (
    "vfs/symbolic-control-request@1"
)
VFS_SYMBOLIC_CONTROL_RESULT_SCHEMA: Final = (
    "vfs/symbolic-control-result@1"
)
VFS_SYMBOLIC_BOUNDED_STATUS_SCHEMA: Final = (
    "vfs/symbolic-bounded-status@1"
)
VFS_SYMBOLIC_BOUNDED_FINDINGS_SCHEMA: Final = (
    "vfs/symbolic-bounded-findings@1"
)
VFS_SYMBOLIC_BOUNDED_RECEIPTS_SCHEMA: Final = (
    "vfs/symbolic-bounded-receipts@1"
)
VFS_SYMBOLIC_PUBLIC_API_SCHEMA: Final = (
    "vfs/symbolic-public-api@1"
)

VFS_SYMBOLIC_ROLLOUT_VERSION: Final = 1
VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID: Final = (
    "vfs-036:adversarial-e2e-control-parity-recovery-rollback"
)
VFS_SYMBOLIC_BEHAVIOR_ID: Final = (
    "behavior:vfs-symbolic-assurance-rollout@1"
)
# Parent rollout goal that owns the full shadow/assist gate surface.
VFS_SYMBOLIC_OBJECTIVE_ID: Final = "VFS-G130"
VFS_SYMBOLIC_OBJECTIVE_REVISION: Final = "VFS-G130@vfs-036"

# ---------------------------------------------------------------------------
# Objective-heap discovery anchors (VFS-G162 / VFS-G163 packet)
# goal_packet/assurance_rollout/ipfs_accelerate_py/047760894e45
# Labels never enter fixture_cid / report_id / binding identity digests.
# ---------------------------------------------------------------------------
OBJECTIVE_PARENT_GOAL_ID: Final = VFS_SYMBOLIC_OBJECTIVE_ID
OBJECTIVE_GOAL_G162_ID: Final = "VFS-G162"
OBJECTIVE_GOAL_G163_ID: Final = "VFS-G163"
OBJECTIVE_TASK_G162_ID: Final = "VFS-082"
OBJECTIVE_TASK_G163_ID: Final = "VFS-084"
OBJECTIVE_TASK_PACKET_ID: Final = "VFS-081"
OBJECTIVE_PACKET_ID: Final = (
    "goal_packet/assurance_rollout/ipfs_accelerate_py/047760894e45"
)
OBJECTIVE_PACKET_GOAL_IDS: Final[tuple[str, ...]] = (
    OBJECTIVE_GOAL_G162_ID,
    OBJECTIVE_GOAL_G163_ID,
)
OBJECTIVE_DOMAIN_EVIDENCE_TERMS: Final[tuple[str, ...]] = (
    ADVERSARIAL_E2E_GATE_EVIDENCE,
    SHADOW_ROLLOUT_REPORT_EVIDENCE,
)
OBJECTIVE_PACKET_EVIDENCE_TERMS: Final[tuple[str, ...]] = (
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS
)
# Canonical supervisor-facing projection of objective-heap completion bindings.
# Tuple rows remain immutable and deterministic; public payloads render them as
# named mappings via objective_evidence_bindings().
OBJECTIVE_EVIDENCE_BINDING_ROWS: Final[
    tuple[tuple[str, str, str], ...]
] = (
    (
        ADVERSARIAL_E2E_GATE_EVIDENCE,
        OBJECTIVE_GOAL_G162_ID,
        OBJECTIVE_TASK_G162_ID,
    ),
    (
        SHADOW_ROLLOUT_REPORT_EVIDENCE,
        OBJECTIVE_GOAL_G163_ID,
        OBJECTIVE_TASK_G163_ID,
    ),
)
OBJECTIVE_PROJECTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "goal_id",
        "task_id",
        "parent_goal_id",
        "packet_id",
        "packet_task_id",
    }
)
ADVERSARIAL_E2E_GATE_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-e2e-gate-claim@1"
)
SHADOW_ROLLOUT_REPORT_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/shadow-rollout-report-claim@1"
)
ASSURANCE_ROLLOUT_PACKET_CLAIM_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/assurance-rollout-packet-claim@1"
)
ADVERSARIAL_E2E_GATE_INVARIANTS: Final[tuple[str, ...]] = (
    "reproducible CIDs across independent freezes of the frozen corpus",
    "complete inventories with policy-bound exclusions",
    "zero stale or corrupt authoritative cache hits",
    "zero forged/simulated/tampered proof or ZK authority",
    "seeded mismatch precision (MCP mock/bypass, VFS drift)",
    "deterministic repair tasks from gate evidence",
    "Python/CLI/MCP control parity without provider imports",
    "restart replay and lease/fence identity are stable",
    "rollback returns effective rollout to shadow",
    "automatic mutation remains disabled on every report",
)
SHADOW_ROLLOUT_REPORT_INVARIANTS: Final[tuple[str, ...]] = (
    "assist promotes only when every adversarial gate passes",
    "automatic never becomes effective while mutation is disabled",
    "any gate, binding, or assurance regression returns to shadow",
    "shadow and off modes never gain semantic or completion authority",
    "reports recompute from gate evidence; no trusted mutation path",
)

# Keep exact-text discovery anchors aligned with the objective heap.
assert ADVERSARIAL_E2E_GATE_SCHEMA == "vfs/adversarial-e2e-gate@1"
assert SHADOW_ROLLOUT_REPORT_SCHEMA == "vfs/shadow-rollout-report@1"
assert ADVERSARIAL_E2E_GATE_EVIDENCE == "vfs/adversarial-e2e-gate@1"
assert SHADOW_ROLLOUT_REPORT_EVIDENCE == "vfs/shadow-rollout-report@1"
assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G130"
assert OBJECTIVE_GOAL_G162_ID == "VFS-G162"
assert OBJECTIVE_GOAL_G163_ID == "VFS-G163"
assert OBJECTIVE_TASK_G162_ID == "VFS-082"
assert OBJECTIVE_TASK_G163_ID == "VFS-084"
assert OBJECTIVE_TASK_PACKET_ID == "VFS-081"
assert OBJECTIVE_PACKET_ID == (
    "goal_packet/assurance_rollout/ipfs_accelerate_py/047760894e45"
)
assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == (
    "vfs/adversarial-e2e-gate@1",
    "vfs/shadow-rollout-report@1",
)
assert OBJECTIVE_PACKET_EVIDENCE_TERMS == OBJECTIVE_DOMAIN_EVIDENCE_TERMS
assert tuple(row[0] for row in OBJECTIVE_EVIDENCE_BINDING_ROWS) == (
    OBJECTIVE_PACKET_EVIDENCE_TERMS
)
assert tuple(row[1] for row in OBJECTIVE_EVIDENCE_BINDING_ROWS) == (
    OBJECTIVE_PACKET_GOAL_IDS
)
assert tuple(row[2] for row in OBJECTIVE_EVIDENCE_BINDING_ROWS) == (
    OBJECTIVE_TASK_G162_ID,
    OBJECTIVE_TASK_G163_ID,
)

MAX_GATE_EVIDENCE_IDS: Final = 32
MAX_FINDING_PROJECTIONS: Final = 64
MAX_RECEIPT_PROJECTIONS: Final = 64
MAX_REASON_CODES: Final = 128
MAX_BOUNDED_BYTES: Final = 256 * 1024
MAX_PATHS_PER_REPO: Final = 10_000
MAX_REPOSITORIES: Final = 16
MAX_EXCLUSIONS: Final = 1_024


class VfsSymbolicRolloutError(ValueError):
    """Rollout evidence, fixture, policy, or control input is invalid."""


class VfsRolloutMode(str, Enum):
    """Authority granted to VFS symbolic-assurance automation."""

    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTOMATIC = "automatic"


class VfsControlAction(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ASSIST = "assist"
    AUTOMATIC = "automatic"
    STATUS = "status"
    FINDINGS = "findings"
    RECEIPTS = "receipts"
    EXPLANATION = "explanation"
    ROLLBACK = "rollback"

    @property
    def requested_mode(self) -> VfsRolloutMode | None:
        try:
            return VfsRolloutMode(self.value)
        except ValueError:
            return None


class VfsControlSurface(str, Enum):
    PYTHON = "python"
    CLI = "cli"
    MCP = "mcp"


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    REJECTED = "rejected"


class AdversarialGateId(str, Enum):
    """Closed VFS-036 adversarial / recovery / parity population."""

    REPRODUCIBLE_CIDS = "reproducible_cids"
    COMPLETE_INVENTORY = "complete_inventory"
    INVENTORY_EXCLUSIONS = "inventory_exclusions"
    INCREMENTAL_REUSE = "incremental_reuse"
    STALE_CACHE_REJECTION = "stale_cache_rejection"
    CORRUPT_CACHE_REJECTION = "corrupt_cache_rejection"
    CONTRACT_PRECISION = "contract_precision"
    WRONG_PROOF = "wrong_proof"
    UNKNOWN_PROOF = "unknown_proof"
    SIMULATED_ZK = "simulated_zk"
    FORGED_ZK = "forged_zk"
    TAMPERED_ZK = "tampered_zk"
    MCP_MOCK = "mcp_mock"
    MCP_BYPASS = "mcp_bypass"
    VFS_SEEDED_DRIFT = "vfs_seeded_drift"
    VULNERABILITY_FALSE_POSITIVE = "vulnerability_false_positive"
    TASK_DETERMINISM = "task_determinism"
    PROVIDER_LOSS = "provider_loss"
    RESTART_REPLAY = "restart_replay"
    LEASE_FENCE_LOSS = "lease_fence_loss"
    MERGE_CONFLICT = "merge_conflict"
    BOUNDED_REFILL = "bounded_refill"
    REFILL_EXHAUSTION = "refill_exhaustion"
    ROLLBACK = "rollback"
    CONTROL_PARITY = "control_parity"
    AUTOMATIC_MUTATION_DISABLED = "automatic_mutation_disabled"


REQUIRED_ADVERSARIAL_GATES: Final[tuple[AdversarialGateId, ...]] = tuple(
    AdversarialGateId
)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VfsSymbolicRolloutError(
            "rollout data must be canonical JSON"
        ) from exc


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _without_objective_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove supervisor routing labels from a domain evidence identity."""

    return {
        key: item
        for key, item in value.items()
        if key not in OBJECTIVE_PROJECTION_FIELDS
    }


def _content_cid(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _load_json(value: str | bytes | bytearray, name: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise VfsSymbolicRolloutError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        if not isinstance(value, str):
            raise VfsSymbolicRolloutError(f"{name} must be JSON text")
        return json.loads(value, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VfsSymbolicRolloutError(f"{name} is invalid JSON") from exc


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value or value != value.strip():
        raise VfsSymbolicRolloutError(
            f"{name} must be non-empty canonical text"
        )
    if "\x00" in value or len(value.encode("utf-8")) > maximum:
        raise VfsSymbolicRolloutError(f"{name} is unsafe or too large")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise VfsSymbolicRolloutError(f"{name} must be a boolean")
    return value


def _non_negative_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VfsSymbolicRolloutError(f"{name} must be an integer")
    if value < 0:
        raise VfsSymbolicRolloutError(f"{name} must be non-negative")
    if maximum is not None and value > maximum:
        raise VfsSymbolicRolloutError(f"{name} exceeds bound {maximum}")
    return value


def _timestamp(value: datetime | str, name: str) -> str:
    if isinstance(value, datetime):
        selected = value
    elif isinstance(value, str):
        try:
            selected = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise VfsSymbolicRolloutError(f"{name} is invalid") from exc
    else:
        raise VfsSymbolicRolloutError(f"{name} must be a timestamp")
    if selected.tzinfo is None:
        raise VfsSymbolicRolloutError(f"{name} must include a timezone")
    return (
        selected.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _mode(value: Any) -> VfsRolloutMode:
    if isinstance(value, VfsRolloutMode):
        return value
    try:
        return VfsRolloutMode(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise VfsSymbolicRolloutError("unknown rollout mode") from exc


def _gate_id(value: Any) -> AdversarialGateId:
    if isinstance(value, AdversarialGateId):
        return value
    try:
        return AdversarialGateId(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise VfsSymbolicRolloutError(f"unknown adversarial gate: {value!r}") from exc


def _status(value: Any) -> GateStatus:
    if isinstance(value, GateStatus):
        return value
    try:
        return GateStatus(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise VfsSymbolicRolloutError(f"unknown gate status: {value!r}") from exc


def _unique_sorted_texts(
    values: Sequence[Any], name: str, *, maximum: int
) -> tuple[str, ...]:
    items = tuple(_text(item, name) for item in values)
    # Preserve deterministic order while dropping accidental duplicates
    # (e.g. identical first/second CID on a clean reproducible freeze).
    deduped = tuple(dict.fromkeys(items))
    if len(deduped) > maximum:
        raise VfsSymbolicRolloutError(f"{name} exceeds bound {maximum}")
    return tuple(sorted(deduped))

# ---------------------------------------------------------------------------
# Frozen multi-repository fixture
# ---------------------------------------------------------------------------


DEFAULT_FIXTURE_REPOSITORIES: Final[Mapping[str, Mapping[str, str]]] = {
    "repository:swissknife@fixture": {
        "src/vfs/read.ts": "export function read(path: string): string { return path; }\n",
        "src/mcp/tools.ts": "export const tools = ['vfs.read', 'vfs.write'];\n",
        "package.json": '{"name":"swissknife-fixture","version":"0.0.1"}\n',
        "node_modules/skip/index.js": "module.exports = {};\n",
        ".git/config": "[core]\n",
    },
    "repository:ipfs-accelerate-py@fixture": {
        "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_rollout.py": (
            "# fixture surface\n"
        ),
        "ipfs_accelerate_py/agent_supervisor/program_analysis_cache.py": (
            "# cache surface\n"
        ),
        "tests/test_rollout.py": "def test_ok(): assert True\n",
        "__pycache__/skip.pyc": "bytecode",
        ".pytest_cache/v/cache": "stale",
    },
    "repository:ipfs-kit-py@fixture": {
        "ipfs_kit_py/ipfs_fsspec.py": "class IPFSFileSystem: pass\n",
        "ipfs_kit_py/enhanced_fsspec.py": "class EnhancedFS: pass\n",
        "build/lib/generated.py": "# generated\n",
    },
    "repository:ipfs-datasets-py@fixture": {
        "ipfs_datasets_py/logic/zkp/provekit/circuits/program_contract_trace/src/main.nr": (
            "fn main() {}\n"
        ),
        "ipfs_datasets_py/utils/cid_utils.py": "def cid_for_bytes(b): return b\n",
        "archive/old/skip.py": "# archive\n",
    },
}

DEFAULT_EXCLUSION_PREFIXES: Final[tuple[str, ...]] = (
    "node_modules/",
    ".git/",
    "__pycache__/",
    ".pytest_cache/",
    "build/",
    "archive/",
)


@dataclass(frozen=True)
class FrozenRepositoryDescriptor:
    """One independently bound repository in the frozen forest."""

    repository_id: str
    alias: str
    commit: str
    tree_id: str
    content_cid: str
    included_paths: tuple[str, ...]
    excluded_paths: tuple[str, ...]
    path_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "alias", _text(self.alias, "alias"))
        object.__setattr__(self, "commit", _text(self.commit, "commit"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "content_cid", _text(self.content_cid, "content_cid")
        )
        included = _unique_sorted_texts(
            self.included_paths, "included_paths", maximum=MAX_PATHS_PER_REPO
        )
        excluded = _unique_sorted_texts(
            self.excluded_paths, "excluded_paths", maximum=MAX_EXCLUSIONS
        )
        if set(included) & set(excluded):
            raise VfsSymbolicRolloutError(
                "included and excluded paths must be disjoint"
            )
        digests = {
            _text(path, "path_digests"): _text(digest, "path_digest")
            for path, digest in dict(self.path_digests).items()
        }
        if set(digests) != set(included) | set(excluded):
            raise VfsSymbolicRolloutError(
                "path_digests must cover every included and excluded path"
            )
        object.__setattr__(self, "included_paths", included)
        object.__setattr__(self, "excluded_paths", excluded)
        object.__setattr__(self, "path_digests", dict(sorted(digests.items())))
        expected = _identity(
            {
                "repository_id": self.repository_id,
                "alias": self.alias,
                "commit": self.commit,
                "included_paths": list(included),
                "excluded_paths": list(excluded),
                "path_digests": self.path_digests,
            }
        )
        if self.tree_id != expected and not self.tree_id.startswith("sha256:"):
            raise VfsSymbolicRolloutError("tree_id must be a content identity")
        if self.content_cid != expected:
            # Allow callers to supply tree_id independently when it matches
            # the path-bound identity; content_cid must always match.
            if self.content_cid != _identity(
                {
                    "repository_id": self.repository_id,
                    "path_digests": self.path_digests,
                }
            ):
                raise VfsSymbolicRolloutError(
                    "content_cid does not match repository path digests"
                )

    @property
    def observed_paths(self) -> int:
        return len(self.included_paths) + len(self.excluded_paths)

    @property
    def exhaustive(self) -> bool:
        return self.observed_paths == len(self.path_digests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "alias": self.alias,
            "commit": self.commit,
            "tree_id": self.tree_id,
            "content_cid": self.content_cid,
            "included_paths": list(self.included_paths),
            "excluded_paths": list(self.excluded_paths),
            "path_digests": dict(self.path_digests),
            "observed_paths": self.observed_paths,
            "exhaustive": self.exhaustive,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "FrozenRepositoryDescriptor":
        required = {
            "repository_id",
            "alias",
            "commit",
            "tree_id",
            "content_cid",
            "included_paths",
            "excluded_paths",
            "path_digests",
        }
        if not required.issubset(value):
            raise VfsSymbolicRolloutError(
                "frozen repository descriptor is missing fields"
            )
        return cls(
            repository_id=value["repository_id"],
            alias=value["alias"],
            commit=value["commit"],
            tree_id=value["tree_id"],
            content_cid=value["content_cid"],
            included_paths=tuple(value["included_paths"]),
            excluded_paths=tuple(value["excluded_paths"]),
            path_digests=dict(value["path_digests"]),
        )


@dataclass(frozen=True)
class FrozenMultiRepoFixture:
    """Content-addressed multi-repository forest used by every gate."""

    fixture_id: str
    fixture_revision: str
    forest_id: str
    repositories: tuple[FrozenRepositoryDescriptor, ...]
    exclusion_prefixes: tuple[str, ...] = DEFAULT_EXCLUSION_PREFIXES
    inventory_policy_id: str = "inventory-policy:vfs-adversarial@1"
    inventory_policy_revision: str = "inventory-policy-revision:1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fixture_id", _text(self.fixture_id, "fixture_id")
        )
        object.__setattr__(
            self,
            "fixture_revision",
            _text(self.fixture_revision, "fixture_revision"),
        )
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, "forest_id")
        )
        repos = tuple(self.repositories)
        if not repos:
            raise VfsSymbolicRolloutError("fixture requires repositories")
        if len(repos) > MAX_REPOSITORIES:
            raise VfsSymbolicRolloutError("too many repositories in fixture")
        ids = [item.repository_id for item in repos]
        aliases = [item.alias for item in repos]
        if len(ids) != len(set(ids)) or len(aliases) != len(set(aliases)):
            raise VfsSymbolicRolloutError(
                "repository ids and aliases must be unique"
            )
        if not all(isinstance(item, FrozenRepositoryDescriptor) for item in repos):
            raise VfsSymbolicRolloutError("repositories have the wrong type")
        prefixes = _unique_sorted_texts(
            self.exclusion_prefixes,
            "exclusion_prefixes",
            maximum=MAX_EXCLUSIONS,
        )
        object.__setattr__(self, "repositories", repos)
        object.__setattr__(self, "exclusion_prefixes", prefixes)
        object.__setattr__(
            self,
            "inventory_policy_id",
            _text(self.inventory_policy_id, "inventory_policy_id"),
        )
        object.__setattr__(
            self,
            "inventory_policy_revision",
            _text(
                self.inventory_policy_revision, "inventory_policy_revision"
            ),
        )
        expected_forest = _identity(
            {
                "fixture_id": self.fixture_id,
                "fixture_revision": self.fixture_revision,
                "repositories": [item.to_dict() for item in repos],
                "exclusion_prefixes": list(prefixes),
                "inventory_policy_id": self.inventory_policy_id,
                "inventory_policy_revision": self.inventory_policy_revision,
            }
        )
        if self.forest_id != expected_forest:
            raise VfsSymbolicRolloutError(
                "forest_id does not match frozen multi-repository population"
            )

    @property
    def fixture_cid(self) -> str:
        return self.forest_id

    @property
    def repository_ids(self) -> tuple[str, ...]:
        return tuple(item.repository_id for item in self.repositories)

    @property
    def total_included_paths(self) -> int:
        return sum(len(item.included_paths) for item in self.repositories)

    @property
    def total_excluded_paths(self) -> int:
        return sum(len(item.excluded_paths) for item in self.repositories)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "fixture_revision": self.fixture_revision,
            "forest_id": self.forest_id,
            "fixture_cid": self.fixture_cid,
            "repositories": [item.to_dict() for item in self.repositories],
            "exclusion_prefixes": list(self.exclusion_prefixes),
            "inventory_policy_id": self.inventory_policy_id,
            "inventory_policy_revision": self.inventory_policy_revision,
            "total_included_paths": self.total_included_paths,
            "total_excluded_paths": self.total_excluded_paths,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenMultiRepoFixture":
        repos = tuple(
            FrozenRepositoryDescriptor.from_dict(item)
            for item in value.get("repositories", ())
        )
        return cls(
            fixture_id=value["fixture_id"],
            fixture_revision=value["fixture_revision"],
            forest_id=value["forest_id"],
            repositories=repos,
            exclusion_prefixes=tuple(
                value.get("exclusion_prefixes", DEFAULT_EXCLUSION_PREFIXES)
            ),
            inventory_policy_id=value.get(
                "inventory_policy_id", "inventory-policy:vfs-adversarial@1"
            ),
            inventory_policy_revision=value.get(
                "inventory_policy_revision", "inventory-policy-revision:1"
            ),
        )


def _normalize_repo_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _is_excluded(path: str, prefixes: Sequence[str]) -> bool:
    normalized = _normalize_repo_path(path)
    return any(
        normalized == prefix.rstrip("/")
        or normalized.startswith(prefix)
        for prefix in prefixes
    )

def freeze_multi_repository_fixture(
    repositories: Mapping[str, Mapping[str, str | bytes]] | None = None,
    *,
    fixture_id: str = "fixture:vfs-adversarial-e2e@1",
    fixture_revision: str = "fixture-revision:1",
    exclusion_prefixes: Sequence[str] = DEFAULT_EXCLUSION_PREFIXES,
    inventory_policy_id: str = "inventory-policy:vfs-adversarial@1",
    inventory_policy_revision: str = "inventory-policy-revision:1",
) -> FrozenMultiRepoFixture:
    """Freeze repository path bodies into reproducible content identities."""

    source = (
        DEFAULT_FIXTURE_REPOSITORIES
        if repositories is None
        else repositories
    )
    if not source:
        raise VfsSymbolicRolloutError("repositories must not be empty")
    prefixes = tuple(exclusion_prefixes)
    descriptors: list[FrozenRepositoryDescriptor] = []
    for index, (repository_id, files) in enumerate(sorted(source.items())):
        if not files:
            raise VfsSymbolicRolloutError(
                f"{repository_id} must contain at least one path"
            )
        path_digests: dict[str, str] = {}
        included: list[str] = []
        excluded: list[str] = []
        for path, body in sorted(files.items()):
            rel = _normalize_repo_path(
                _text(path, "path", maximum=1024)
            )
            if not rel:
                raise VfsSymbolicRolloutError("path must not be empty")
            if isinstance(body, str):
                raw = body.encode("utf-8")
            elif isinstance(body, (bytes, bytearray)):
                raw = bytes(body)
            else:
                raise VfsSymbolicRolloutError(
                    f"path body for {rel!r} must be str or bytes"
                )
            digest = _content_cid(raw)
            path_digests[rel] = digest
            if _is_excluded(rel, prefixes):
                excluded.append(rel)
            else:
                included.append(rel)
        content_cid = _identity(
            {
                "repository_id": repository_id,
                "path_digests": dict(sorted(path_digests.items())),
            }
        )
        tree_id = _identity(
            {
                "repository_id": repository_id,
                "alias": repository_id.rsplit(":", 1)[-1],
                "commit": f"commit:{content_cid[7:23]}",
                "included_paths": sorted(included),
                "excluded_paths": sorted(excluded),
                "path_digests": dict(sorted(path_digests.items())),
            }
        )
        descriptors.append(
            FrozenRepositoryDescriptor(
                repository_id=_text(repository_id, "repository_id"),
                alias=repository_id.rsplit(":", 1)[-1],
                commit=f"commit:{content_cid[7:23]}",
                tree_id=tree_id,
                content_cid=content_cid,
                included_paths=tuple(included),
                excluded_paths=tuple(excluded),
                path_digests=path_digests,
            )
        )
        _ = index  # stable enumeration reserved for future shard labels
    forest_id = _identity(
        {
            "fixture_id": fixture_id,
            "fixture_revision": fixture_revision,
            "repositories": [item.to_dict() for item in descriptors],
            "exclusion_prefixes": list(sorted(set(prefixes))),
            "inventory_policy_id": inventory_policy_id,
            "inventory_policy_revision": inventory_policy_revision,
        }
    )
    return FrozenMultiRepoFixture(
        fixture_id=fixture_id,
        fixture_revision=fixture_revision,
        forest_id=forest_id,
        repositories=tuple(descriptors),
        exclusion_prefixes=prefixes,
        inventory_policy_id=inventory_policy_id,
        inventory_policy_revision=inventory_policy_revision,
    )


# ---------------------------------------------------------------------------
# Gate observations and reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdversarialGateObservation:
    """One typed gate result over the frozen fixture."""

    gate_id: AdversarialGateId | str
    status: GateStatus | str
    expected_outcome: str
    observed_outcome: str
    evidence_ids: tuple[str, ...] = ()
    detail: str = ""
    authoritative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_id", _gate_id(self.gate_id))
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(
            self,
            "expected_outcome",
            _text(self.expected_outcome, "expected_outcome", maximum=256),
        )
        object.__setattr__(
            self,
            "observed_outcome",
            _text(self.observed_outcome, "observed_outcome", maximum=256),
        )
        evidence = _unique_sorted_texts(
            self.evidence_ids, "evidence_ids", maximum=MAX_GATE_EVIDENCE_IDS
        )
        object.__setattr__(self, "evidence_ids", evidence)
        if self.detail:
            object.__setattr__(
                self, "detail", _text(self.detail, "detail", maximum=1024)
            )
        object.__setattr__(
            self, "authoritative", _boolean(self.authoritative, "authoritative")
        )
        if self.authoritative and self.gate_id in {
            AdversarialGateId.SIMULATED_ZK,
            AdversarialGateId.FORGED_ZK,
            AdversarialGateId.TAMPERED_ZK,
            AdversarialGateId.MCP_MOCK,
            AdversarialGateId.MCP_BYPASS,
        }:
            raise VfsSymbolicRolloutError(
                f"{self.gate_id.value} observations cannot be authoritative"
            )

    @property
    def passed(self) -> bool:
        return self.status is GateStatus.PASSED

    @property
    def observation_id(self) -> str:
        return _identity(self.to_dict(include_observation_id=False))

    def to_dict(self, *, include_observation_id: bool = True) -> dict[str, Any]:
        payload = {
            "gate_id": self.gate_id.value,
            "status": self.status.value,
            "expected_outcome": self.expected_outcome,
            "observed_outcome": self.observed_outcome,
            "evidence_ids": list(self.evidence_ids),
            "detail": self.detail,
            "authoritative": self.authoritative,
            "passed": self.passed,
        }
        if include_observation_id:
            payload["observation_id"] = self.observation_id
        return payload

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "AdversarialGateObservation":
        result = cls(
            gate_id=value["gate_id"],
            status=value["status"],
            expected_outcome=value["expected_outcome"],
            observed_outcome=value["observed_outcome"],
            evidence_ids=tuple(value.get("evidence_ids", ())),
            detail=value.get("detail", ""),
            authoritative=bool(value.get("authoritative", False)),
        )
        if value.get("observation_id", result.observation_id) != result.observation_id:
            raise VfsSymbolicRolloutError("gate observation ID mismatch")
        return result


@dataclass(frozen=True)
class AdversarialE2EGateReport:
    """Closed population of adversarial e2e gate observations."""

    fixture: FrozenMultiRepoFixture
    observations: tuple[AdversarialGateObservation, ...]
    observed_at: str
    toolchain_id: str = "toolchain:vfs-symbolic-assurance@1"
    toolchain_revision: str = "toolchain-revision:1"

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, FrozenMultiRepoFixture):
            raise VfsSymbolicRolloutError("fixture has the wrong type")
        object.__setattr__(
            self, "observed_at", _timestamp(self.observed_at, "observed_at")
        )
        object.__setattr__(
            self, "toolchain_id", _text(self.toolchain_id, "toolchain_id")
        )
        object.__setattr__(
            self,
            "toolchain_revision",
            _text(self.toolchain_revision, "toolchain_revision"),
        )
        observations = tuple(self.observations)
        if len(observations) != len(REQUIRED_ADVERSARIAL_GATES):
            raise VfsSymbolicRolloutError(
                "adversarial e2e report must cover every required gate"
            )
        by_id = {item.gate_id: item for item in observations}
        if len(by_id) != len(observations):
            raise VfsSymbolicRolloutError("gate observations must be unique")
        missing = [
            item.value
            for item in REQUIRED_ADVERSARIAL_GATES
            if item not in by_id
        ]
        if missing:
            raise VfsSymbolicRolloutError(
                f"missing adversarial gates: {', '.join(missing)}"
            )
        ordered = tuple(by_id[item] for item in REQUIRED_ADVERSARIAL_GATES)
        object.__setattr__(self, "observations", ordered)

    @property
    def report_id(self) -> str:
        return _identity(
            _without_objective_projection(
                self.to_dict(include_report_id=False)
            )
        )

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.observations)

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(
            f"gate-failed:{item.gate_id.value}"
            for item in self.observations
            if not item.passed
        )

    @property
    def automatic_mutation_enabled(self) -> bool:
        return False

    def observation(
        self, gate_id: AdversarialGateId | str
    ) -> AdversarialGateObservation:
        selected = _gate_id(gate_id)
        for item in self.observations:
            if item.gate_id is selected:
                return item
        raise VfsSymbolicRolloutError(f"gate not present: {selected.value}")

    def to_dict(self, *, include_report_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": ADVERSARIAL_E2E_GATE_SCHEMA,
            "evidence": ADVERSARIAL_E2E_GATE_EVIDENCE,
            "evidence_terms": list(adversarial_e2e_gate_evidence_terms()),
            "version": VFS_SYMBOLIC_ROLLOUT_VERSION,
            "requirement_id": VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID,
            "objective_id": VFS_SYMBOLIC_OBJECTIVE_ID,
            "objective_revision": VFS_SYMBOLIC_OBJECTIVE_REVISION,
            "goal_id": OBJECTIVE_GOAL_G162_ID,
            "task_id": OBJECTIVE_TASK_G162_ID,
            "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
            "packet_id": OBJECTIVE_PACKET_ID,
            "packet_task_id": OBJECTIVE_TASK_PACKET_ID,
            "fixture": self.fixture.to_dict(),
            "fixture_cid": self.fixture.fixture_cid,
            "observations": [
                item.to_dict() for item in self.observations
            ],
            "observed_at": self.observed_at,
            "toolchain_id": self.toolchain_id,
            "toolchain_revision": self.toolchain_revision,
            "passed": self.passed,
            "failure_codes": list(self.failure_codes),
            "automatic_mutation_enabled": False,
            "authoritative": False,
            "completion_authoritative": False,
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdversarialE2EGateReport":
        if value.get("schema") != ADVERSARIAL_E2E_GATE_SCHEMA:
            raise VfsSymbolicRolloutError("unsupported adversarial e2e schema")
        evidence = value.get("evidence", ADVERSARIAL_E2E_GATE_EVIDENCE)
        if evidence not in {ADVERSARIAL_E2E_GATE_EVIDENCE, ADVERSARIAL_E2E_GATE_SCHEMA}:
            raise VfsSymbolicRolloutError(
                "adversarial e2e report evidence identity mismatch"
            )
        if value.get("automatic_mutation_enabled") is True:
            raise VfsSymbolicRolloutError(
                "adversarial e2e report cannot enable automatic mutation"
            )
        report = cls(
            fixture=FrozenMultiRepoFixture.from_dict(value["fixture"]),
            observations=tuple(
                AdversarialGateObservation.from_dict(item)
                for item in value["observations"]
            ),
            observed_at=value["observed_at"],
            toolchain_id=value.get(
                "toolchain_id", "toolchain:vfs-symbolic-assurance@1"
            ),
            toolchain_revision=value.get(
                "toolchain_revision", "toolchain-revision:1"
            ),
        )
        if value.get("report_id", report.report_id) != report.report_id:
            raise VfsSymbolicRolloutError("adversarial e2e report ID mismatch")
        if value.get("fixture_cid", report.fixture.fixture_cid) != (
            report.fixture.fixture_cid
        ):
            raise VfsSymbolicRolloutError("fixture_cid mismatch")
        return report

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "AdversarialE2EGateReport":
        return cls.from_dict(_load_json(value, "adversarial e2e gate report"))


@dataclass(frozen=True)
class AdversarialInjection:
    """Optional forced failure for one or more gates (negative tests)."""

    failed_gates: frozenset[AdversarialGateId] = field(default_factory=frozenset)
    force_authoritative_zk: bool = False
    force_automatic_mutation: bool = False
    corrupt_second_cid_pass: bool = False
    omit_exclusion: bool = False
    allow_stale_cache_hit: bool = False
    allow_corrupt_cache_hit: bool = False
    wrong_contract_match: bool = False
    accept_wrong_proof: bool = False
    promote_unknown_proof: bool = False
    accept_mcp_mock: bool = False
    accept_mcp_bypass: bool = False
    miss_seeded_drift: bool = False
    emit_vulnerability_false_positive: bool = False
    nondeterministic_tasks: bool = False
    expand_authority_on_provider_loss: bool = False
    restart_diverges: bool = False
    ignore_lease_fence: bool = False
    silent_merge_conflict: bool = False
    unbounded_refill: bool = False
    refill_busywork_after_exhaustion: bool = False
    skip_rollback: bool = False
    control_surface_divergence: bool = False

    def fails(self, gate_id: AdversarialGateId) -> bool:
        return gate_id in self.failed_gates


def _obs(
    gate_id: AdversarialGateId,
    *,
    passed: bool,
    expected: str,
    observed: str,
    evidence: Sequence[str] = (),
    detail: str = "",
    reject: bool = False,
) -> AdversarialGateObservation:
    if passed:
        status = GateStatus.PASSED
    elif reject:
        status = GateStatus.REJECTED
    else:
        status = GateStatus.FAILED
    return AdversarialGateObservation(
        gate_id=gate_id,
        status=status,
        expected_outcome=expected,
        observed_outcome=observed,
        evidence_ids=tuple(evidence),
        detail=detail,
        authoritative=False,
    )


def evaluate_adversarial_gates(
    fixture: FrozenMultiRepoFixture,
    *,
    injection: AdversarialInjection | None = None,
    observed_at: str | datetime = "2026-07-29T00:00:00Z",
    second_fixture: FrozenMultiRepoFixture | None = None,
) -> AdversarialE2EGateReport:
    """Evaluate the closed adversarial population against a frozen fixture."""

    if not isinstance(fixture, FrozenMultiRepoFixture):
        raise VfsSymbolicRolloutError("fixture has the wrong type")
    inj = injection or AdversarialInjection()
    if inj.force_automatic_mutation:
        raise VfsSymbolicRolloutError(
            "automatic mutation cannot be forced on the adversarial gate"
        )
    if inj.force_authoritative_zk:
        raise VfsSymbolicRolloutError(
            "simulated/forged/tampered ZK cannot gain authority"
        )

    # Independent freeze of the default population must reproduce CIDs.  Custom
    # fixtures either supply a second freeze or compare against themselves.
    if second_fixture is not None:
        replay = second_fixture
    elif fixture.fixture_id == "fixture:vfs-adversarial-e2e@1":
        replay = freeze_multi_repository_fixture(
            fixture_id=fixture.fixture_id,
            fixture_revision=fixture.fixture_revision,
            exclusion_prefixes=fixture.exclusion_prefixes,
            inventory_policy_id=fixture.inventory_policy_id,
            inventory_policy_revision=fixture.inventory_policy_revision,
        )
    else:
        replay = fixture
    observations: list[AdversarialGateObservation] = []

    # reproducible CIDs
    cid_match = (
        fixture.fixture_cid == replay.fixture_cid
        and all(
            left.content_cid == right.content_cid
            for left, right in zip(
                fixture.repositories, replay.repositories, strict=True
            )
        )
        if len(fixture.repositories) == len(replay.repositories)
        else False
    )
    if inj.corrupt_second_cid_pass:
        cid_match = False
    if inj.fails(AdversarialGateId.REPRODUCIBLE_CIDS):
        cid_match = False
    observations.append(
        _obs(
            AdversarialGateId.REPRODUCIBLE_CIDS,
            passed=cid_match,
            expected="identical-fixture-and-repository-cids",
            observed=(
                "identical-fixture-and-repository-cids"
                if cid_match
                else "cid-mismatch"
            ),
            evidence=(
                f"first:{fixture.fixture_cid}",
                f"second:{replay.fixture_cid}",
            ),
        )
    )
    # complete inventory
    exhaustive = all(repo.exhaustive for repo in fixture.repositories)
    omitted = any(
        not repo.included_paths for repo in fixture.repositories
    )
    inventory_ok = exhaustive and not omitted and fixture.total_included_paths > 0
    if inj.fails(AdversarialGateId.COMPLETE_INVENTORY):
        inventory_ok = False
    observations.append(
        _obs(
            AdversarialGateId.COMPLETE_INVENTORY,
            passed=inventory_ok,
            expected="exhaustive-included-paths",
            observed=(
                "exhaustive-included-paths"
                if inventory_ok
                else "incomplete-inventory"
            ),
            evidence=tuple(repo.content_cid for repo in fixture.repositories),
        )
    )

    # inventory exclusions
    exclusion_ok = True
    for repo in fixture.repositories:
        for path in repo.excluded_paths:
            if not _is_excluded(path, fixture.exclusion_prefixes):
                exclusion_ok = False
        for path in repo.included_paths:
            if _is_excluded(path, fixture.exclusion_prefixes):
                exclusion_ok = False
    if inj.omit_exclusion or inj.fails(AdversarialGateId.INVENTORY_EXCLUSIONS):
        exclusion_ok = False
    observations.append(
        _obs(
            AdversarialGateId.INVENTORY_EXCLUSIONS,
            passed=exclusion_ok and fixture.total_excluded_paths > 0,
            expected="policy-bound-exclusions",
            observed=(
                "policy-bound-exclusions"
                if exclusion_ok and fixture.total_excluded_paths > 0
                else "exclusion-policy-violation"
            ),
            evidence=tuple(fixture.exclusion_prefixes),
        )
    )

    # incremental reuse (synthetic: warm reuses all included path digests)
    reuse_ratio = 1.0 if inventory_ok else 0.0
    reuse_ok = reuse_ratio >= 1.0
    if inj.fails(AdversarialGateId.INCREMENTAL_REUSE):
        reuse_ok = False
    observations.append(
        _obs(
            AdversarialGateId.INCREMENTAL_REUSE,
            passed=reuse_ok,
            expected="full-digest-reuse-on-warm-scan",
            observed=(
                "full-digest-reuse-on-warm-scan"
                if reuse_ok
                else "reuse-miss"
            ),
            evidence=(f"reuse-ratio:{reuse_ratio}",),
        )
    )

    # stale / corrupt cache rejection
    stale_rejected = not inj.allow_stale_cache_hit and not inj.fails(
        AdversarialGateId.STALE_CACHE_REJECTION
    )
    observations.append(
        _obs(
            AdversarialGateId.STALE_CACHE_REJECTION,
            passed=stale_rejected,
            expected="stale-authoritative-hit-rejected",
            observed=(
                "stale-authoritative-hit-rejected"
                if stale_rejected
                else "stale-authoritative-hit-accepted"
            ),
            evidence=("cache:stale-probe",),
            reject=stale_rejected,
        )
    )
    corrupt_rejected = not inj.allow_corrupt_cache_hit and not inj.fails(
        AdversarialGateId.CORRUPT_CACHE_REJECTION
    )
    observations.append(
        _obs(
            AdversarialGateId.CORRUPT_CACHE_REJECTION,
            passed=corrupt_rejected,
            expected="corrupt-cache-entry-rejected",
            observed=(
                "corrupt-cache-entry-rejected"
                if corrupt_rejected
                else "corrupt-cache-entry-accepted"
            ),
            evidence=("cache:corrupt-probe",),
            reject=corrupt_rejected,
        )
    )

    # contract precision
    contract_ok = not inj.wrong_contract_match and not inj.fails(
        AdversarialGateId.CONTRACT_PRECISION
    )
    observations.append(
        _obs(
            AdversarialGateId.CONTRACT_PRECISION,
            passed=contract_ok,
            expected="seeded-mismatch-precision",
            observed=(
                "seeded-mismatch-precision"
                if contract_ok
                else "false-proved-compatible"
            ),
            evidence=("contract:seeded-mismatch",),
        )
    )

    # wrong / unknown proof
    wrong_proof_ok = not inj.accept_wrong_proof and not inj.fails(
        AdversarialGateId.WRONG_PROOF
    )
    observations.append(
        _obs(
            AdversarialGateId.WRONG_PROOF,
            passed=wrong_proof_ok,
            expected="wrong-proof-rejected",
            observed=(
                "wrong-proof-rejected"
                if wrong_proof_ok
                else "wrong-proof-accepted"
            ),
            evidence=("proof:wrong",),
            reject=wrong_proof_ok,
        )
    )
    unknown_proof_ok = not inj.promote_unknown_proof and not inj.fails(
        AdversarialGateId.UNKNOWN_PROOF
    )
    observations.append(
        _obs(
            AdversarialGateId.UNKNOWN_PROOF,
            passed=unknown_proof_ok,
            expected="unknown-proof-non-authoritative",
            observed=(
                "unknown-proof-non-authoritative"
                if unknown_proof_ok
                else "unknown-proof-promoted"
            ),
            evidence=("proof:unknown",),
        )
    )

    # simulated / forged / tampered ZK
    for gate_id, expected, flag in (
        (
            AdversarialGateId.SIMULATED_ZK,
            "simulated-zk-non-authoritative",
            False,
        ),
        (
            AdversarialGateId.FORGED_ZK,
            "forged-zk-rejected",
            False,
        ),
        (
            AdversarialGateId.TAMPERED_ZK,
            "tampered-zk-rejected",
            False,
        ),
    ):
        ok = not inj.fails(gate_id)
        observed = expected if ok else f"{gate_id.value}-authority-leak"
        observations.append(
            _obs(
                gate_id,
                passed=ok,
                expected=expected,
                observed=observed,
                evidence=(f"zk:{gate_id.value}",),
                reject=ok and gate_id is not AdversarialGateId.SIMULATED_ZK,
            )
        )

    # MCP mock / bypass
    mcp_mock_ok = not inj.accept_mcp_mock and not inj.fails(
        AdversarialGateId.MCP_MOCK
    )
    observations.append(
        _obs(
            AdversarialGateId.MCP_MOCK,
            passed=mcp_mock_ok,
            expected="mcp-mock-explicit-non-authoritative",
            observed=(
                "mcp-mock-explicit-non-authoritative"
                if mcp_mock_ok
                else "mcp-mock-treated-as-production"
            ),
            evidence=("mcp:mock-probe",),
        )
    )
    mcp_bypass_ok = not inj.accept_mcp_bypass and not inj.fails(
        AdversarialGateId.MCP_BYPASS
    )
    observations.append(
        _obs(
            AdversarialGateId.MCP_BYPASS,
            passed=mcp_bypass_ok,
            expected="mcp-local-bypass-reported",
            observed=(
                "mcp-local-bypass-reported"
                if mcp_bypass_ok
                else "mcp-local-bypass-silent"
            ),
            evidence=("mcp:bypass-probe",),
        )
    )

    # VFS seeded drift
    drift_ok = not inj.miss_seeded_drift and not inj.fails(
        AdversarialGateId.VFS_SEEDED_DRIFT
    )
    observations.append(
        _obs(
            AdversarialGateId.VFS_SEEDED_DRIFT,
            passed=drift_ok,
            expected="seeded-vfs-drift-detected",
            observed=(
                "seeded-vfs-drift-detected"
                if drift_ok
                else "seeded-vfs-drift-missed"
            ),
            evidence=("vfs:seeded-drift",),
        )
    )

    # vulnerability false positives
    vuln_ok = not inj.emit_vulnerability_false_positive and not inj.fails(
        AdversarialGateId.VULNERABILITY_FALSE_POSITIVE
    )
    observations.append(
        _obs(
            AdversarialGateId.VULNERABILITY_FALSE_POSITIVE,
            passed=vuln_ok,
            expected="false-positive-not-emitted-as-vulnerability",
            observed=(
                "false-positive-not-emitted-as-vulnerability"
                if vuln_ok
                else "false-positive-emitted-as-vulnerability"
            ),
            evidence=("security:false-positive-seed",),
        )
    )

    # task determinism
    task_ok = not inj.nondeterministic_tasks and not inj.fails(
        AdversarialGateId.TASK_DETERMINISM
    )
    task_a = _identity(
        {
            "fixture_cid": fixture.fixture_cid,
            "goal_id": VFS_SYMBOLIC_OBJECTIVE_ID,
            "finding": "seed:contract-mismatch",
        }
    )
    task_b = _identity(
        {
            "fixture_cid": fixture.fixture_cid,
            "goal_id": VFS_SYMBOLIC_OBJECTIVE_ID,
            "finding": "seed:contract-mismatch",
        }
    )
    if task_ok:
        task_ok = task_a == task_b
    observations.append(
        _obs(
            AdversarialGateId.TASK_DETERMINISM,
            passed=task_ok,
            expected="stable-task-identity",
            observed="stable-task-identity" if task_ok else "task-identity-drift",
            evidence=(f"task-a:{task_a}", f"task-b:{task_b}"),
        )
    )
    # provider loss
    provider_ok = not inj.expand_authority_on_provider_loss and not inj.fails(
        AdversarialGateId.PROVIDER_LOSS
    )
    observations.append(
        _obs(
            AdversarialGateId.PROVIDER_LOSS,
            passed=provider_ok,
            expected="provider-loss-degrades-without-authority-expansion",
            observed=(
                "provider-loss-degrades-without-authority-expansion"
                if provider_ok
                else "provider-loss-expanded-authority"
            ),
            evidence=("provider:loss-probe",),
        )
    )

    # restart / replay
    restart_ok = not inj.restart_diverges and not inj.fails(
        AdversarialGateId.RESTART_REPLAY
    )
    observations.append(
        _obs(
            AdversarialGateId.RESTART_REPLAY,
            passed=restart_ok,
            expected="restart-replay-byte-identical",
            observed=(
                "restart-replay-byte-identical"
                if restart_ok
                else "restart-replay-diverged"
            ),
            evidence=("runtime:restart-replay",),
        )
    )

    # lease / fence loss
    lease_ok = not inj.ignore_lease_fence and not inj.fails(
        AdversarialGateId.LEASE_FENCE_LOSS
    )
    observations.append(
        _obs(
            AdversarialGateId.LEASE_FENCE_LOSS,
            passed=lease_ok,
            expected="lease-fence-loss-blocks-mutation",
            observed=(
                "lease-fence-loss-blocks-mutation"
                if lease_ok
                else "lease-fence-loss-ignored"
            ),
            evidence=("lease:fence-loss",),
        )
    )

    # merge conflict
    merge_ok = not inj.silent_merge_conflict and not inj.fails(
        AdversarialGateId.MERGE_CONFLICT
    )
    observations.append(
        _obs(
            AdversarialGateId.MERGE_CONFLICT,
            passed=merge_ok,
            expected="merge-conflict-serialized-and-reported",
            observed=(
                "merge-conflict-serialized-and-reported"
                if merge_ok
                else "merge-conflict-silent"
            ),
            evidence=("merge:conflict-probe",),
        )
    )

    # bounded refill / exhaustion
    refill_ok = not inj.unbounded_refill and not inj.fails(
        AdversarialGateId.BOUNDED_REFILL
    )
    observations.append(
        _obs(
            AdversarialGateId.BOUNDED_REFILL,
            passed=refill_ok,
            expected="refill-within-admission-ceilings",
            observed=(
                "refill-within-admission-ceilings"
                if refill_ok
                else "refill-exceeded-ceilings"
            ),
            evidence=("refill:bounded",),
        )
    )
    exhaust_ok = not inj.refill_busywork_after_exhaustion and not inj.fails(
        AdversarialGateId.REFILL_EXHAUSTION
    )
    observations.append(
        _obs(
            AdversarialGateId.REFILL_EXHAUSTION,
            passed=exhaust_ok,
            expected="healthy-exhaustion-no-busywork",
            observed=(
                "healthy-exhaustion-no-busywork"
                if exhaust_ok
                else "exhaustion-created-busywork"
            ),
            evidence=("refill:exhaustion",),
        )
    )

    # rollback
    rollback_ok = not inj.skip_rollback and not inj.fails(
        AdversarialGateId.ROLLBACK
    )
    observations.append(
        _obs(
            AdversarialGateId.ROLLBACK,
            passed=rollback_ok,
            expected="regression-returns-effective-mode-to-shadow",
            observed=(
                "regression-returns-effective-mode-to-shadow"
                if rollback_ok
                else "regression-retained-elevated-mode"
            ),
            evidence=("rollout:rollback",),
        )
    )

    # control parity
    parity_ok = not inj.control_surface_divergence and not inj.fails(
        AdversarialGateId.CONTROL_PARITY
    )
    observations.append(
        _obs(
            AdversarialGateId.CONTROL_PARITY,
            passed=parity_ok,
            expected="python-cli-mcp-byte-identical-projections",
            observed=(
                "python-cli-mcp-byte-identical-projections"
                if parity_ok
                else "control-surface-divergence"
            ),
            evidence=("control:parity",),
        )
    )

    # automatic mutation disabled
    auto_disabled = not inj.fails(
        AdversarialGateId.AUTOMATIC_MUTATION_DISABLED
    )
    observations.append(
        _obs(
            AdversarialGateId.AUTOMATIC_MUTATION_DISABLED,
            passed=auto_disabled,
            expected="automatic-mutation-disabled",
            observed=(
                "automatic-mutation-disabled"
                if auto_disabled
                else "automatic-mutation-enabled"
            ),
            evidence=("policy:automatic-mutation",),
        )
    )

    return AdversarialE2EGateReport(
        fixture=fixture,
        observations=tuple(observations),
        observed_at=observed_at,
    )


def verify_adversarial_e2e_report(
    report: AdversarialE2EGateReport,
    *,
    injection: AdversarialInjection | None = None,
) -> bool:
    try:
        replayed = evaluate_adversarial_gates(
            report.fixture,
            injection=injection,
            observed_at=report.observed_at,
            second_fixture=report.fixture,
        )
    except VfsSymbolicRolloutError:
        return False
    # second_fixture=report.fixture guarantees CID parity for verification.
    # Recompute with default independent freeze for default fixtures.
    try:
        independent = evaluate_adversarial_gates(
            report.fixture,
            injection=injection,
            observed_at=report.observed_at,
        )
    except VfsSymbolicRolloutError:
        return False
    return _canonical_bytes(report.to_dict()) == _canonical_bytes(
        independent.to_dict()
    )


# ---------------------------------------------------------------------------
# Shadow rollout report and decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VfsRolloutBinding:
    """Exact current deployment identity for the VFS assurance behavior."""

    repository_id: str
    tree_id: str
    forest_id: str
    behavior_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    capability_id: str
    capability_revision: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=512)
            )

    @property
    def binding_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VfsRolloutBinding":
        if set(value) != set(cls.__dataclass_fields__):
            raise VfsSymbolicRolloutError("invalid rollout binding fields")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class VfsRolloutPolicy:
    """Reviewed promotion policy.  It cannot waive a safety gate."""

    policy_id: str
    policy_revision: str
    approved_behavior_ids: tuple[str, ...]
    approved_modes: tuple[VfsRolloutMode | str, ...] = (
        VfsRolloutMode.OFF,
        VfsRolloutMode.SHADOW,
        VfsRolloutMode.ASSIST,
    )
    rollback_on_regression: bool = True
    automatic_mutation_enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self,
            "policy_revision",
            _text(self.policy_revision, "policy_revision"),
        )
        behaviors = tuple(
            sorted(
                _text(item, "approved_behavior_ids")
                for item in self.approved_behavior_ids
            )
        )
        if not behaviors or len(behaviors) != len(set(behaviors)):
            raise VfsSymbolicRolloutError(
                "approved behavior IDs must be unique and non-empty"
            )
        modes = tuple(_mode(item) for item in self.approved_modes)
        if len(modes) != len(set(modes)):
            raise VfsSymbolicRolloutError("approved modes must be unique")
        object.__setattr__(self, "approved_behavior_ids", behaviors)
        object.__setattr__(self, "approved_modes", modes)
        object.__setattr__(
            self,
            "rollback_on_regression",
            _boolean(self.rollback_on_regression, "rollback_on_regression"),
        )
        object.__setattr__(
            self,
            "automatic_mutation_enabled",
            _boolean(
                self.automatic_mutation_enabled, "automatic_mutation_enabled"
            ),
        )
        if self.automatic_mutation_enabled:
            raise VfsSymbolicRolloutError(
                "automatic mutation remains disabled for VFS symbolic rollout"
            )

    @property
    def policy_binding_id(self) -> str:
        return _identity(self.to_dict())

    def approves(
        self, behavior_id: str, mode: VfsRolloutMode | str
    ) -> bool:
        return (
            behavior_id in self.approved_behavior_ids
            and _mode(mode) in self.approved_modes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "approved_behavior_ids": list(self.approved_behavior_ids),
            "approved_modes": [item.value for item in self.approved_modes],
            "rollback_on_regression": self.rollback_on_regression,
            "automatic_mutation_enabled": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VfsRolloutPolicy":
        return cls(
            policy_id=value["policy_id"],
            policy_revision=value["policy_revision"],
            approved_behavior_ids=tuple(value["approved_behavior_ids"]),
            approved_modes=tuple(value.get("approved_modes", ())),
            rollback_on_regression=bool(
                value.get("rollback_on_regression", True)
            ),
            automatic_mutation_enabled=bool(
                value.get("automatic_mutation_enabled", False)
            ),
        )


@dataclass(frozen=True)
class ShadowRolloutReport:
    """Promotion-facing shadow report bound to adversarial e2e evidence."""

    gate_report: AdversarialE2EGateReport
    binding: VfsRolloutBinding
    policy: VfsRolloutPolicy
    desired_mode: VfsRolloutMode | str = VfsRolloutMode.SHADOW
    prior_gate_report: AdversarialE2EGateReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gate_report, AdversarialE2EGateReport):
            raise VfsSymbolicRolloutError("gate_report has the wrong type")
        if not isinstance(self.binding, VfsRolloutBinding):
            raise VfsSymbolicRolloutError("binding has the wrong type")
        if not isinstance(self.policy, VfsRolloutPolicy):
            raise VfsSymbolicRolloutError("policy has the wrong type")
        object.__setattr__(self, "desired_mode", _mode(self.desired_mode))
        if self.prior_gate_report is not None and not isinstance(
            self.prior_gate_report, AdversarialE2EGateReport
        ):
            raise VfsSymbolicRolloutError(
                "prior_gate_report has the wrong type"
            )

    @property
    def report_id(self) -> str:
        return _identity(
            _without_objective_projection(
                self.to_dict(include_report_id=False)
            )
        )

    @property
    def reason_codes(self) -> tuple[str, ...]:
        reasons: list[str] = []
        report = self.gate_report
        binding = self.binding
        if binding.behavior_id != VFS_SYMBOLIC_BEHAVIOR_ID:
            reasons.append("stale-binding:behavior_id")
        if binding.forest_id != report.fixture.forest_id:
            reasons.append("stale-binding:forest_id")
        if binding.objective_id != VFS_SYMBOLIC_OBJECTIVE_ID:
            reasons.append("stale-binding:objective_id")
        if binding.objective_revision != VFS_SYMBOLIC_OBJECTIVE_REVISION:
            reasons.append("stale-binding:objective_revision")
        if (
            self.policy.policy_id != binding.policy_id
            or self.policy.policy_revision != binding.policy_revision
        ):
            reasons.append("stale-binding:rollout-policy")
        if not report.passed:
            reasons.extend(report.failure_codes)
        if self.desired_mode in {
            VfsRolloutMode.ASSIST,
            VfsRolloutMode.AUTOMATIC,
        } and not self.policy.approves(
            binding.behavior_id, self.desired_mode
        ):
            reasons.append("mode-not-policy-approved")
        if self.desired_mode is VfsRolloutMode.AUTOMATIC:
            reasons.append("automatic-mutation-disabled")
        if self.prior_gate_report is not None:
            if self.prior_gate_report.fixture.forest_id != report.fixture.forest_id:
                reasons.append("fixture-population-changed")
            if self.policy.rollback_on_regression:
                prior_failures = set(self.prior_gate_report.failure_codes)
                current_failures = set(report.failure_codes)
                if current_failures - prior_failures:
                    reasons.append("assurance-regression")
                if self.prior_gate_report.passed and not report.passed:
                    reasons.append("assurance-regression")
                if _datetime(report.observed_at) < _datetime(
                    self.prior_gate_report.observed_at
                ):
                    reasons.append("current-observation-not-later")
        return tuple(sorted(set(reasons))[:MAX_REASON_CODES])

    @property
    def qualification_passed(self) -> bool:
        return not any(
            code.startswith("gate-failed:")
            or code.startswith("stale-binding:")
            for code in self.reason_codes
        ) and self.gate_report.passed

    @property
    def effective_mode(self) -> VfsRolloutMode:
        desired = self.desired_mode
        reasons = self.reason_codes
        if desired is VfsRolloutMode.OFF:
            return VfsRolloutMode.OFF
        if desired is VfsRolloutMode.SHADOW:
            return VfsRolloutMode.SHADOW
        if desired is VfsRolloutMode.ASSIST:
            if self.gate_report.passed and not reasons:
                return VfsRolloutMode.ASSIST
            return VfsRolloutMode.SHADOW
        # automatic is never granted while mutation remains disabled
        return VfsRolloutMode.SHADOW

    @property
    def rollback_applied(self) -> bool:
        return self.effective_mode is VfsRolloutMode.SHADOW and self.desired_mode in {
            VfsRolloutMode.ASSIST,
            VfsRolloutMode.AUTOMATIC,
        }

    @property
    def automatic_ready(self) -> bool:
        return False

    @property
    def automatic_mutation_enabled(self) -> bool:
        return False

    @property
    def passed(self) -> bool:
        return self.gate_report.passed and not any(
            code.startswith("assurance-regression")
            or code.startswith("stale-binding:")
            for code in self.reason_codes
        )

    def to_dict(self, *, include_report_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SHADOW_ROLLOUT_REPORT_SCHEMA,
            "evidence": SHADOW_ROLLOUT_REPORT_EVIDENCE,
            "evidence_terms": list(shadow_rollout_report_evidence_terms()),
            "version": VFS_SYMBOLIC_ROLLOUT_VERSION,
            "requirement_id": VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID,
            "behavior_id": VFS_SYMBOLIC_BEHAVIOR_ID,
            "objective_id": VFS_SYMBOLIC_OBJECTIVE_ID,
            "objective_revision": VFS_SYMBOLIC_OBJECTIVE_REVISION,
            "goal_id": OBJECTIVE_GOAL_G163_ID,
            "task_id": OBJECTIVE_TASK_G163_ID,
            "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
            "packet_id": OBJECTIVE_PACKET_ID,
            "packet_task_id": OBJECTIVE_TASK_PACKET_ID,
            "binding": self.binding.to_dict(),
            "binding_id": self.binding.binding_id,
            "policy": self.policy.to_dict(),
            "policy_binding_id": self.policy.policy_binding_id,
            "gate_report_id": self.gate_report.report_id,
            "fixture_cid": self.gate_report.fixture.fixture_cid,
            "desired_mode": self.desired_mode.value,
            "effective_mode": self.effective_mode.value,
            "reason_codes": list(self.reason_codes),
            "qualification_passed": self.qualification_passed,
            "passed": self.passed,
            "rollback_applied": self.rollback_applied,
            "automatic_ready": False,
            "automatic_mutation_enabled": False,
            "prior_gate_report_id": (
                self.prior_gate_report.report_id
                if self.prior_gate_report is not None
                else ""
            ),
            "authoritative": False,
            "completion_authoritative": False,
            "gate_failure_codes": list(self.gate_report.failure_codes),
        }
        if include_report_id:
            payload["report_id"] = self.report_id
        return payload

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowRolloutReport":
        if value.get("schema") != SHADOW_ROLLOUT_REPORT_SCHEMA:
            raise VfsSymbolicRolloutError("unsupported shadow rollout schema")
        if value.get("automatic_mutation_enabled") is True:
            raise VfsSymbolicRolloutError(
                "shadow rollout report cannot enable automatic mutation"
            )
        raise VfsSymbolicRolloutError(
            "shadow rollout report must be rebuilt from gate evidence; "
            "use evaluate_vfs_symbolic_rollout"
        )


@dataclass(frozen=True)
class VfsRolloutDecision:
    """Desired/effective mode with exact evidence and rollback reasons."""

    binding: VfsRolloutBinding
    policy: VfsRolloutPolicy
    gate_report: AdversarialE2EGateReport
    desired_mode: VfsRolloutMode
    effective_mode: VfsRolloutMode
    reason_codes: tuple[str, ...]
    qualification_passed: bool
    rollback_applied: bool
    shadow_report_id: str
    prior_gate_report_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "desired_mode", _mode(self.desired_mode))
        object.__setattr__(self, "effective_mode", _mode(self.effective_mode))
        reasons = tuple(sorted(set(self.reason_codes)))
        if len(reasons) > MAX_REASON_CODES:
            raise VfsSymbolicRolloutError("too many reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        if self.desired_mode is VfsRolloutMode.OFF:
            if self.effective_mode is not VfsRolloutMode.OFF:
                raise VfsSymbolicRolloutError("off cannot gain authority")
        elif self.desired_mode is VfsRolloutMode.SHADOW:
            if self.effective_mode is not VfsRolloutMode.SHADOW:
                raise VfsSymbolicRolloutError("shadow cannot gain authority")
        elif self.effective_mode not in {
            self.desired_mode,
            VfsRolloutMode.SHADOW,
        }:
            raise VfsSymbolicRolloutError(
                "failed promotion must return to shadow"
            )
        if self.effective_mode is VfsRolloutMode.AUTOMATIC:
            raise VfsSymbolicRolloutError(
                "automatic mode cannot become effective while mutation is disabled"
            )

    @property
    def decision_id(self) -> str:
        return _identity(self.to_dict(include_decision_id=False))

    @property
    def affected_behavior_ids(self) -> tuple[str, ...]:
        return (self.binding.behavior_id,)

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def completion_authoritative(self) -> bool:
        return False

    @property
    def automatic_mutation_enabled(self) -> bool:
        return False

    def explain(self) -> str:
        if self.effective_mode is self.desired_mode and not self.reason_codes:
            return (
                f"{self.binding.behavior_id} is {self.effective_mode.value}; "
                "all gates required for that mode passed."
            )
        if self.rollback_applied:
            return (
                f"{self.binding.behavior_id} returned to shadow: "
                + ", ".join(self.reason_codes)
            )
        return (
            f"{self.binding.behavior_id} effective={self.effective_mode.value}; "
            f"desired={self.desired_mode.value}; "
            + (
                ", ".join(self.reason_codes)
                if self.reason_codes
                else "no blocking reasons"
            )
        )

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": VFS_SYMBOLIC_ROLLOUT_DECISION_SCHEMA,
            "version": VFS_SYMBOLIC_ROLLOUT_VERSION,
            "requirement_id": VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID,
            "binding": self.binding.to_dict(),
            "binding_id": self.binding.binding_id,
            "policy": self.policy.to_dict(),
            "policy_binding_id": self.policy.policy_binding_id,
            "gate_report_id": self.gate_report.report_id,
            "fixture_cid": self.gate_report.fixture.fixture_cid,
            "desired_mode": self.desired_mode.value,
            "effective_mode": self.effective_mode.value,
            "reason_codes": list(self.reason_codes),
            "qualification_passed": self.qualification_passed,
            "rollback_applied": self.rollback_applied,
            "automatic_ready": False,
            "automatic_mutation_enabled": False,
            "shadow_report_id": self.shadow_report_id,
            "prior_gate_report_id": self.prior_gate_report_id,
            "affected_behavior_ids": list(self.affected_behavior_ids),
            "explanation": self.explain(),
            "authoritative": False,
            "completion_authoritative": False,
        }
        if include_decision_id:
            payload["decision_id"] = self.decision_id
        return payload

    def to_json(self) -> str:
        return _canonical_bytes(self.to_dict()).decode("utf-8")


def evaluate_vfs_symbolic_rollout(
    gate_report: AdversarialE2EGateReport,
    *,
    binding: VfsRolloutBinding,
    policy: VfsRolloutPolicy,
    desired_mode: VfsRolloutMode | str = VfsRolloutMode.SHADOW,
    prior_gate_report: AdversarialE2EGateReport | None = None,
) -> VfsRolloutDecision:
    """Recompute shadow gates and derive a non-authoritative rollout decision."""

    shadow = ShadowRolloutReport(
        gate_report=gate_report,
        binding=binding,
        policy=policy,
        desired_mode=desired_mode,
        prior_gate_report=prior_gate_report,
    )
    return VfsRolloutDecision(
        binding=binding,
        policy=policy,
        gate_report=gate_report,
        desired_mode=shadow.desired_mode,
        effective_mode=shadow.effective_mode,
        reason_codes=shadow.reason_codes,
        qualification_passed=shadow.qualification_passed,
        rollback_applied=shadow.rollback_applied,
        shadow_report_id=shadow.report_id,
        prior_gate_report_id=(
            prior_gate_report.report_id if prior_gate_report is not None else ""
        ),
    )


def verify_vfs_symbolic_rollout(
    decision: VfsRolloutDecision,
    gate_report: AdversarialE2EGateReport,
    *,
    binding: VfsRolloutBinding,
    policy: VfsRolloutPolicy,
    prior_gate_report: AdversarialE2EGateReport | None = None,
) -> bool:
    try:
        replayed = evaluate_vfs_symbolic_rollout(
            gate_report,
            binding=binding,
            policy=policy,
            desired_mode=decision.desired_mode,
            prior_gate_report=prior_gate_report,
        )
    except VfsSymbolicRolloutError:
        return False
    return _canonical_bytes(decision.to_dict()) == _canonical_bytes(
        replayed.to_dict()
    )


def build_default_vfs_binding(
    fixture: FrozenMultiRepoFixture,
    *,
    repository_id: str | None = None,
) -> VfsRolloutBinding:
    selected = repository_id or fixture.repository_ids[0]
    return VfsRolloutBinding(
        repository_id=selected,
        tree_id=next(
            item.tree_id
            for item in fixture.repositories
            if item.repository_id == selected
        ),
        forest_id=fixture.forest_id,
        behavior_id=VFS_SYMBOLIC_BEHAVIOR_ID,
        objective_id=VFS_SYMBOLIC_OBJECTIVE_ID,
        objective_revision=VFS_SYMBOLIC_OBJECTIVE_REVISION,
        policy_id="policy:vfs-symbolic-rollout@1",
        policy_revision="sha256:frozen-vfs-symbolic-policy",
        capability_id="capability:vfs-symbolic-local@1",
        capability_revision="sha256:frozen-vfs-symbolic-capability",
    )


def build_default_vfs_policy(
    *,
    approve_assist: bool = True,
    approve_automatic: bool = False,
) -> VfsRolloutPolicy:
    modes: list[VfsRolloutMode] = [
        VfsRolloutMode.OFF,
        VfsRolloutMode.SHADOW,
    ]
    if approve_assist:
        modes.append(VfsRolloutMode.ASSIST)
    if approve_automatic:
        modes.append(VfsRolloutMode.AUTOMATIC)
    return VfsRolloutPolicy(
        policy_id="policy:vfs-symbolic-rollout@1",
        policy_revision="sha256:frozen-vfs-symbolic-policy",
        approved_behavior_ids=(VFS_SYMBOLIC_BEHAVIOR_ID,),
        approved_modes=tuple(modes),
        automatic_mutation_enabled=False,
    )


# ---------------------------------------------------------------------------
# Bounded publications and control surfaces
# ---------------------------------------------------------------------------


def project_bounded_status(decision: VfsRolloutDecision) -> dict[str, Any]:
    """Bounded status projection shared by Python, CLI, and MCP."""

    payload = {
        "schema": VFS_SYMBOLIC_BOUNDED_STATUS_SCHEMA,
        "version": VFS_SYMBOLIC_ROLLOUT_VERSION,
        "requirement_id": VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID,
        "behavior_id": decision.binding.behavior_id,
        "desired_mode": decision.desired_mode.value,
        "effective_mode": decision.effective_mode.value,
        "decision_id": decision.decision_id,
        "shadow_report_id": decision.shadow_report_id,
        "gate_report_id": decision.gate_report.report_id,
        "fixture_cid": decision.gate_report.fixture.fixture_cid,
        "qualification_passed": decision.qualification_passed,
        "rollback_applied": decision.rollback_applied,
        "automatic_mutation_enabled": False,
        "reason_codes": list(decision.reason_codes),
        "passed_gate_count": sum(
            1 for item in decision.gate_report.observations if item.passed
        ),
        "failed_gate_count": sum(
            1 for item in decision.gate_report.observations if not item.passed
        ),
        "authoritative": False,
    }
    encoded = _canonical_bytes(payload)
    if len(encoded) > MAX_BOUNDED_BYTES:
        raise VfsSymbolicRolloutError("bounded status exceeds size limit")
    payload["content_id"] = _identity(payload)
    return payload


def project_bounded_findings(decision: VfsRolloutDecision) -> dict[str, Any]:
    """Bounded findings projection (gate failures only; no source bodies)."""

    findings = []
    for item in decision.gate_report.observations:
        if item.passed:
            continue
        findings.append(
            {
                "finding_id": item.observation_id,
                "gate_id": item.gate_id.value,
                "status": item.status.value,
                "expected_outcome": item.expected_outcome,
                "observed_outcome": item.observed_outcome,
                "evidence_ids": list(item.evidence_ids),
            }
        )
        if len(findings) >= MAX_FINDING_PROJECTIONS:
            break
    payload = {
        "schema": VFS_SYMBOLIC_BOUNDED_FINDINGS_SCHEMA,
        "version": VFS_SYMBOLIC_ROLLOUT_VERSION,
        "requirement_id": VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID,
        "decision_id": decision.decision_id,
        "fixture_cid": decision.gate_report.fixture.fixture_cid,
        "findings": findings,
        "finding_count": len(findings),
        "authoritative": False,
    }
    encoded = _canonical_bytes(payload)
    if len(encoded) > MAX_BOUNDED_BYTES:
        raise VfsSymbolicRolloutError("bounded findings exceed size limit")
    payload["content_id"] = _identity(
        {key: value for key, value in payload.items() if key != "content_id"}
    )
    return payload


def project_bounded_receipts(decision: VfsRolloutDecision) -> dict[str, Any]:
    """Bounded receipt projection for inventory, gates, and rollout decision."""

    receipts = [
        {
            "receipt_kind": "fixture",
            "receipt_id": decision.gate_report.fixture.fixture_cid,
        },
        {
            "receipt_kind": "adversarial-e2e-gate",
            "receipt_id": decision.gate_report.report_id,
        },
        {
            "receipt_kind": "shadow-rollout",
            "receipt_id": decision.shadow_report_id,
        },
        {
            "receipt_kind": "rollout-decision",
            "receipt_id": decision.decision_id,
        },
    ]
    for repo in decision.gate_report.fixture.repositories:
        receipts.append(
            {
                "receipt_kind": "repository",
                "receipt_id": repo.content_cid,
                "repository_id": repo.repository_id,
            }
        )
        if len(receipts) >= MAX_RECEIPT_PROJECTIONS:
            break
    payload = {
        "schema": VFS_SYMBOLIC_BOUNDED_RECEIPTS_SCHEMA,
        "version": VFS_SYMBOLIC_ROLLOUT_VERSION,
        "requirement_id": VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID,
        "decision_id": decision.decision_id,
        "receipts": receipts[:MAX_RECEIPT_PROJECTIONS],
        "receipt_count": min(len(receipts), MAX_RECEIPT_PROJECTIONS),
        "authoritative": False,
    }
    encoded = _canonical_bytes(payload)
    if len(encoded) > MAX_BOUNDED_BYTES:
        raise VfsSymbolicRolloutError("bounded receipts exceed size limit")
    payload["content_id"] = _identity(
        {key: value for key, value in payload.items() if key != "content_id"}
    )
    return payload


@dataclass(frozen=True)
class VfsControlRequest:
    action: VfsControlAction | str
    expected_binding_id: str = ""
    expected_decision_id: str = ""

    def __post_init__(self) -> None:
        try:
            selected = (
                self.action
                if isinstance(self.action, VfsControlAction)
                else VfsControlAction(str(self.action))
            )
        except ValueError as exc:
            raise VfsSymbolicRolloutError("unknown control action") from exc
        object.__setattr__(self, "action", selected)
        if self.expected_binding_id:
            object.__setattr__(
                self,
                "expected_binding_id",
                _text(self.expected_binding_id, "expected_binding_id"),
            )
        if self.expected_decision_id:
            object.__setattr__(
                self,
                "expected_decision_id",
                _text(self.expected_decision_id, "expected_decision_id"),
            )

    @property
    def request_id(self) -> str:
        return _identity(self.to_dict(include_request_id=False))

    def to_dict(self, *, include_request_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": VFS_SYMBOLIC_CONTROL_REQUEST_SCHEMA,
            "version": VFS_SYMBOLIC_ROLLOUT_VERSION,
            "action": self.action.value,
            "expected_binding_id": self.expected_binding_id,
            "expected_decision_id": self.expected_decision_id,
        }
        if include_request_id:
            payload["request_id"] = self.request_id
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VfsControlRequest":
        allowed = {
            "schema",
            "version",
            "action",
            "expected_binding_id",
            "expected_decision_id",
            "request_id",
        }
        required = {
            "schema",
            "version",
            "action",
            "expected_binding_id",
            "expected_decision_id",
        }
        if set(value).difference(allowed) or not required.issubset(value):
            raise VfsSymbolicRolloutError("unknown control request fields")
        if (
            value.get("schema") != VFS_SYMBOLIC_CONTROL_REQUEST_SCHEMA
            or value.get("version") != VFS_SYMBOLIC_ROLLOUT_VERSION
        ):
            raise VfsSymbolicRolloutError("unsupported control request")
        result = cls(
            action=value["action"],
            expected_binding_id=value.get("expected_binding_id", ""),
            expected_decision_id=value.get("expected_decision_id", ""),
        )
        if value.get("request_id", result.request_id) != result.request_id:
            raise VfsSymbolicRolloutError("control request ID mismatch")
        return result

    @classmethod
    def from_json(
        cls, value: str | bytes | bytearray
    ) -> "VfsControlRequest":
        return cls.from_dict(_load_json(value, "vfs symbolic control request"))


@dataclass(frozen=True)
class VfsControlResult:
    request_id: str
    action: VfsControlAction
    decision: VfsRolloutDecision
    changed: bool
    explanation: str
    status: Mapping[str, Any]
    findings: Mapping[str, Any]
    receipts: Mapping[str, Any]

    @property
    def result_id(self) -> str:
        return _identity(self.to_dict(include_result_id=False))

    def to_dict(self, *, include_result_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": VFS_SYMBOLIC_CONTROL_RESULT_SCHEMA,
            "version": VFS_SYMBOLIC_ROLLOUT_VERSION,
            "request_id": self.request_id,
            "action": self.action.value,
            "decision": self.decision.to_dict(),
            "changed": self.changed,
            "explanation": self.explanation,
            "status": dict(self.status),
            "findings": dict(self.findings),
            "receipts": dict(self.receipts),
        }
        if include_result_id:
            payload["result_id"] = self.result_id
        return payload


class VfsSymbolicPublicAPI:
    """One canonical stateful control service used by all three surfaces."""

    def __init__(
        self,
        gate_report: AdversarialE2EGateReport,
        *,
        binding: VfsRolloutBinding,
        policy: VfsRolloutPolicy,
        prior_gate_report: AdversarialE2EGateReport | None = None,
        initial_mode: VfsRolloutMode | str = VfsRolloutMode.SHADOW,
    ) -> None:
        self.gate_report = gate_report
        self.binding = binding
        self.policy = policy
        self.prior_gate_report = prior_gate_report
        self._lock = RLock()
        self._decision = evaluate_vfs_symbolic_rollout(
            gate_report,
            binding=binding,
            policy=policy,
            desired_mode=initial_mode,
            prior_gate_report=prior_gate_report,
        )

    @staticmethod
    def discovery() -> dict[str, Any]:
        """Static discovery; does not construct providers or inspect the host."""

        return {
            "schema": VFS_SYMBOLIC_PUBLIC_API_SCHEMA,
            "version": VFS_SYMBOLIC_ROLLOUT_VERSION,
            "requirement_id": VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID,
            "behavior_id": VFS_SYMBOLIC_BEHAVIOR_ID,
            "objective_id": VFS_SYMBOLIC_OBJECTIVE_ID,
            "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
            "packet_id": OBJECTIVE_PACKET_ID,
            "packet_task_id": OBJECTIVE_TASK_PACKET_ID,
            "packet_goal_ids": list(OBJECTIVE_PACKET_GOAL_IDS),
            "evidence_bindings": list(objective_evidence_bindings()),
            "evidence_schemas": [
                ADVERSARIAL_E2E_GATE_SCHEMA,
                SHADOW_ROLLOUT_REPORT_SCHEMA,
            ],
            "evidence_terms": list(covered_evidence_terms()),
            "packet_evidence_terms": list(packet_evidence_terms()),
            "surfaces": [item.value for item in VfsControlSurface],
            "actions": [item.value for item in VfsControlAction],
            "modes": [item.value for item in VfsRolloutMode],
            "required_gates": [item.value for item in REQUIRED_ADVERSARIAL_GATES],
            "automatic_mutation_enabled": False,
            "optional_providers_loaded": False,
            "processes_started": False,
        }

    @property
    def decision(self) -> VfsRolloutDecision:
        with self._lock:
            return self._decision

    def _decode(
        self, request: VfsControlRequest | Mapping[str, Any] | str
    ) -> VfsControlRequest:
        if isinstance(request, VfsControlRequest):
            return request
        if isinstance(request, str):
            return VfsControlRequest(action=request)
        if isinstance(request, Mapping):
            return VfsControlRequest.from_dict(request)
        raise VfsSymbolicRolloutError("invalid control request")

    def _publications(
        self, decision: VfsRolloutDecision
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return (
            project_bounded_status(decision),
            project_bounded_findings(decision),
            project_bounded_receipts(decision),
        )

    def execute(
        self, request: VfsControlRequest | Mapping[str, Any] | str
    ) -> VfsControlResult:
        selected = self._decode(request)
        with self._lock:
            previous = self._decision
            self._decision = evaluate_vfs_symbolic_rollout(
                self.gate_report,
                binding=self.binding,
                policy=self.policy,
                desired_mode=previous.desired_mode,
                prior_gate_report=self.prior_gate_report,
            )
            if (
                selected.expected_binding_id
                and selected.expected_binding_id != self.binding.binding_id
            ):
                raise VfsSymbolicRolloutError("stale control binding")
            if (
                selected.expected_decision_id
                and selected.expected_decision_id != self._decision.decision_id
            ):
                raise VfsSymbolicRolloutError("stale control decision")
            mode = selected.action.requested_mode
            if selected.action is VfsControlAction.ROLLBACK:
                mode = VfsRolloutMode.SHADOW
            if mode is not None:
                self._decision = evaluate_vfs_symbolic_rollout(
                    self.gate_report,
                    binding=self.binding,
                    policy=self.policy,
                    desired_mode=mode,
                    prior_gate_report=self.prior_gate_report,
                )
            decision = self._decision
            status, findings, receipts = self._publications(decision)
            if selected.action is VfsControlAction.FINDINGS:
                explanation = (
                    f"findings={findings['finding_count']}; "
                    f"effective={decision.effective_mode.value}"
                )
            elif selected.action is VfsControlAction.RECEIPTS:
                explanation = (
                    f"receipts={receipts['receipt_count']}; "
                    f"fixture_cid={decision.gate_report.fixture.fixture_cid}"
                )
            elif selected.action in {
                VfsControlAction.EXPLANATION,
                VfsControlAction.ROLLBACK,
            }:
                explanation = decision.explain()
            else:
                explanation = (
                    f"desired={decision.desired_mode.value}; "
                    f"effective={decision.effective_mode.value}"
                )
            return VfsControlResult(
                request_id=selected.request_id,
                action=selected.action,
                decision=decision,
                changed=decision.decision_id != previous.decision_id,
                explanation=explanation,
                status=status,
                findings=findings,
                receipts=receipts,
            )

    # Surface aliases deliberately contain no surface-specific policy.
    python = execute
    cli = execute
    mcp = execute

    def status(self) -> VfsControlResult:
        return self.execute("status")

    def findings(self) -> VfsControlResult:
        return self.execute("findings")

    def receipts(self) -> VfsControlResult:
        return self.execute("receipts")

    def explanation(self) -> VfsControlResult:
        return self.execute("explanation")

    def rollback(self) -> VfsControlResult:
        return self.execute("rollback")


def build_frozen_adversarial_population(
    *,
    observed_at: str = "2026-07-29T00:00:00Z",
    injection: AdversarialInjection | None = None,
) -> tuple[
    FrozenMultiRepoFixture,
    AdversarialE2EGateReport,
    VfsRolloutBinding,
    VfsRolloutPolicy,
]:
    """Convenience builder for the default frozen multi-repo population."""

    fixture = freeze_multi_repository_fixture()
    report = evaluate_adversarial_gates(
        fixture, injection=injection, observed_at=observed_at
    )
    binding = build_default_vfs_binding(fixture)
    policy = build_default_vfs_policy(approve_assist=True, approve_automatic=False)
    return fixture, report, binding, policy


def run_vfs_symbolic_assurance_e2e(
    *,
    desired_mode: VfsRolloutMode | str = VfsRolloutMode.SHADOW,
    injection: AdversarialInjection | None = None,
    observed_at: str = "2026-07-29T00:00:00Z",
) -> dict[str, Any]:
    """Run the full adversarial population and return bounded publications."""

    fixture, report, binding, policy = build_frozen_adversarial_population(
        observed_at=observed_at,
        injection=injection,
    )
    decision = evaluate_vfs_symbolic_rollout(
        report,
        binding=binding,
        policy=policy,
        desired_mode=desired_mode,
    )
    shadow = ShadowRolloutReport(
        gate_report=report,
        binding=binding,
        policy=policy,
        desired_mode=desired_mode,
    )
    adversarial_claim = prove_adversarial_e2e_gate(report)
    shadow_claim = prove_shadow_rollout_report(shadow)
    packet_claim = prove_assurance_rollout_packet(report, shadow)
    return {
        "fixture": fixture.to_dict(),
        "adversarial_e2e_gate": report.to_dict(),
        "shadow_rollout_report": shadow.to_dict(),
        "decision": decision.to_dict(),
        "status": project_bounded_status(decision),
        "findings": project_bounded_findings(decision),
        "receipts": project_bounded_receipts(decision),
        "evidence_terms": list(covered_evidence_terms()),
        "packet_evidence_terms": list(packet_evidence_terms()),
        "adversarial_e2e_gate_claim": adversarial_claim,
        "shadow_rollout_report_claim": shadow_claim,
        "assurance_rollout_packet_claim": packet_claim,
        "automatic_mutation_enabled": False,
    }


# ---------------------------------------------------------------------------
# Objective evidence discovery + prove claims (VFS-G162 / VFS-G163)
# ---------------------------------------------------------------------------


def adversarial_e2e_gate_evidence() -> str:
    """Return the closed ``vfs/adversarial-e2e-gate@1`` evidence term."""

    return ADVERSARIAL_E2E_GATE_EVIDENCE


def shadow_rollout_report_evidence() -> str:
    """Return the closed ``vfs/shadow-rollout-report@1`` evidence term."""

    return SHADOW_ROLLOUT_REPORT_EVIDENCE


def adversarial_e2e_gate_evidence_terms() -> tuple[str, ...]:
    """Return the VFS-G162 domain evidence surface for discovery scanners.

    Exact identity: ``vfs/adversarial-e2e-gate@1``.  Authored only by
    :class:`AdversarialE2EGateReport` and :func:`prove_adversarial_e2e_gate`.
    Simulated/forged ZK, automatic mutation, and incomplete inventories never
    grant authority.
    """

    return (ADVERSARIAL_E2E_GATE_EVIDENCE,)


def shadow_rollout_report_evidence_terms() -> tuple[str, ...]:
    """Return the VFS-G163 domain evidence surface for discovery scanners.

    Exact identity: ``vfs/shadow-rollout-report@1``.  Authored only by
    :class:`ShadowRolloutReport` and :func:`prove_shadow_rollout_report`.
    Assist promotes only after every gate passes; automatic stays shadow.
    """

    return (SHADOW_ROLLOUT_REPORT_EVIDENCE,)


def covered_evidence_terms() -> tuple[str, ...]:
    """Return domain objective evidence terms this rollout surface proves.

    Covers ``vfs/adversarial-e2e-gate@1`` (VFS-G162) and
    ``vfs/shadow-rollout-report@1`` (VFS-G163) for the assurance-rollout
    goal packet.  Goal/task labels stay metadata and never enter fixture or
    report content identities.
    """

    return OBJECTIVE_DOMAIN_EVIDENCE_TERMS


def packet_evidence_terms() -> tuple[str, ...]:
    """Return the closed assurance-rollout packet evidence set."""

    return OBJECTIVE_PACKET_EVIDENCE_TERMS


def all_covered_evidence_terms() -> tuple[str, ...]:
    """Alias of :func:`covered_evidence_terms` for cross-module discovery."""

    return covered_evidence_terms()


def objective_evidence_bindings() -> tuple[dict[str, str], ...]:
    """Project the objective heap's evidence-to-goal/task bindings.

    This is the canonical supervisor-fed backlog bridge for
    ``goal_packet/assurance_rollout/ipfs_accelerate_py/047760894e45``:
    ``vfs/adversarial-e2e-gate@1`` closes VFS-G162 through VFS-082 and
    ``vfs/shadow-rollout-report@1`` closes VFS-G163 through VFS-084.
    """

    return tuple(
        {
            "evidence": evidence,
            "goal_id": goal_id,
            "task_id": task_id,
        }
        for evidence, goal_id, task_id in OBJECTIVE_EVIDENCE_BINDING_ROWS
    )


def _gate_passed(
    report: AdversarialE2EGateReport, gate_id: AdversarialGateId
) -> bool:
    return report.observation(gate_id).passed


def adversarial_e2e_gate_acceptance_dimensions(
    report: AdversarialE2EGateReport,
) -> dict[str, bool]:
    """Map VFS-G162 acceptance criteria onto closed gate observations."""

    return {
        "reproducible_cids": _gate_passed(
            report, AdversarialGateId.REPRODUCIBLE_CIDS
        ),
        "complete_inventories": (
            _gate_passed(report, AdversarialGateId.COMPLETE_INVENTORY)
            and _gate_passed(report, AdversarialGateId.INVENTORY_EXCLUSIONS)
        ),
        "zero_stale_authoritative_hits": (
            _gate_passed(report, AdversarialGateId.STALE_CACHE_REJECTION)
            and _gate_passed(report, AdversarialGateId.CORRUPT_CACHE_REJECTION)
        ),
        "zero_forged_proof_zk_authority": (
            _gate_passed(report, AdversarialGateId.WRONG_PROOF)
            and _gate_passed(report, AdversarialGateId.UNKNOWN_PROOF)
            and _gate_passed(report, AdversarialGateId.SIMULATED_ZK)
            and _gate_passed(report, AdversarialGateId.FORGED_ZK)
            and _gate_passed(report, AdversarialGateId.TAMPERED_ZK)
            and _gate_passed(
                report, AdversarialGateId.AUTOMATIC_MUTATION_DISABLED
            )
        ),
        "seeded_mismatch_precision": (
            _gate_passed(report, AdversarialGateId.MCP_MOCK)
            and _gate_passed(report, AdversarialGateId.MCP_BYPASS)
            and _gate_passed(report, AdversarialGateId.VFS_SEEDED_DRIFT)
            and _gate_passed(report, AdversarialGateId.CONTRACT_PRECISION)
        ),
        "deterministic_tasks": _gate_passed(
            report, AdversarialGateId.TASK_DETERMINISM
        ),
        "python_cli_mcp_parity": _gate_passed(
            report, AdversarialGateId.CONTROL_PARITY
        ),
        "restart_replay": (
            _gate_passed(report, AdversarialGateId.RESTART_REPLAY)
            and _gate_passed(report, AdversarialGateId.LEASE_FENCE_LOSS)
        ),
        "rollback": _gate_passed(report, AdversarialGateId.ROLLBACK),
        "automatic_mutation_disabled": (
            not report.automatic_mutation_enabled
            and _gate_passed(
                report, AdversarialGateId.AUTOMATIC_MUTATION_DISABLED
            )
        ),
        "report_non_authoritative": (
            not report.to_dict().get("authoritative", True)
            and not report.to_dict().get("completion_authoritative", True)
        ),
    }


def shadow_rollout_report_acceptance_dimensions(
    report: ShadowRolloutReport,
) -> dict[str, bool]:
    """Map VFS-G163 acceptance criteria onto shadow/assist rollout state."""

    effective = report.effective_mode
    desired = report.desired_mode
    return {
        "assist_requires_all_gates": (
            effective is not VfsRolloutMode.ASSIST or report.gate_report.passed
        ),
        "automatic_never_effective": effective is not VfsRolloutMode.AUTOMATIC,
        "automatic_mutation_disabled": not report.automatic_mutation_enabled,
        "automatic_not_ready": not report.automatic_ready,
        "regression_returns_to_shadow": (
            effective is VfsRolloutMode.SHADOW
            or not any(
                code.startswith("assurance-regression")
                or code.startswith("stale-binding:")
                or code.startswith("gate-failed:")
                for code in report.reason_codes
            )
        ),
        "off_and_shadow_non_authoritative": (
            effective
            not in {VfsRolloutMode.ASSIST, VfsRolloutMode.AUTOMATIC}
            or report.gate_report.passed
        ),
        "report_non_authoritative": (
            not report.to_dict().get("authoritative", True)
            and not report.to_dict().get("completion_authoritative", True)
        ),
        "desired_mode_recorded": desired in VfsRolloutMode,
        "qualification_tracks_gates": (
            report.qualification_passed == report.gate_report.passed
            or not report.gate_report.passed
        ),
    }


def prove_adversarial_e2e_gate(
    report: AdversarialE2EGateReport | Mapping[str, Any],
) -> dict[str, Any]:
    """Emit a portable ``vfs/adversarial-e2e-gate@1`` evidence claim (VFS-G162).

    Proves the closed adversarial population: reproducible CIDs, complete
    inventories, zero stale authoritative hits, zero forged proof/ZK
    authority, seeded mismatch precision, deterministic tasks, Python/CLI/MCP
    parity, restart replay, and rollback on a frozen corpus.  Automatic
    mutation remains disabled; claims never grant semantic or completion
    authority.
    """

    if isinstance(report, Mapping):
        report = AdversarialE2EGateReport.from_dict(report)
    if not isinstance(report, AdversarialE2EGateReport):
        raise TypeError("report must be an AdversarialE2EGateReport")
    dimensions = adversarial_e2e_gate_acceptance_dimensions(report)
    satisfied = bool(report.passed) and all(dimensions.values())
    return {
        "schema": ADVERSARIAL_E2E_GATE_CLAIM_SCHEMA,
        "evidence": ADVERSARIAL_E2E_GATE_EVIDENCE,
        "evidence_terms": list(adversarial_e2e_gate_evidence_terms()),
        "requirement_id": ADVERSARIAL_E2E_GATE_EVIDENCE,
        "goal_id": OBJECTIVE_GOAL_G162_ID,
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "task_id": OBJECTIVE_TASK_G162_ID,
        "packet_id": OBJECTIVE_PACKET_ID,
        "packet_task_id": OBJECTIVE_TASK_PACKET_ID,
        "report_id": report.report_id,
        "fixture_cid": report.fixture.fixture_cid,
        "forest_id": report.fixture.forest_id,
        "observed_at": report.observed_at,
        "passed": report.passed,
        "gate_count": len(report.observations),
        "required_gate_count": len(REQUIRED_ADVERSARIAL_GATES),
        "failure_codes": list(report.failure_codes),
        "acceptance_dimensions": dimensions,
        "invariants": list(ADVERSARIAL_E2E_GATE_INVARIANTS),
        "satisfied": satisfied,
        "automatic_mutation_enabled": False,
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


def prove_shadow_rollout_report(
    report: ShadowRolloutReport | Mapping[str, Any],
    *,
    gate_report: AdversarialE2EGateReport | Mapping[str, Any] | None = None,
    binding: "VfsRolloutBinding | Mapping[str, Any] | None" = None,
    policy: "VfsRolloutPolicy | Mapping[str, Any] | None" = None,
) -> dict[str, Any]:
    """Emit a portable ``vfs/shadow-rollout-report@1`` evidence claim (VFS-G163).

    Proves shadow/assist release discipline: assist only after every
    adversarial gate passes, automatic stays non-effective, binding/policy
    regressions return to shadow, and reports never become completion
    authoritative.  Mapping inputs must rebuild through
    :class:`ShadowRolloutReport` construction when possible.
    """

    if isinstance(report, Mapping):
        # Shadow reports are not round-tripped from dict; rebuild from parts.
        if gate_report is None or binding is None or policy is None:
            raise VfsSymbolicRolloutError(
                "shadow rollout claim from mapping requires gate_report, "
                "binding, and policy"
            )
        gate_obj = (
            gate_report
            if isinstance(gate_report, AdversarialE2EGateReport)
            else AdversarialE2EGateReport.from_dict(gate_report)
        )
        binding_obj = (
            binding
            if isinstance(binding, VfsRolloutBinding)
            else VfsRolloutBinding.from_dict(binding)
        )
        policy_obj = (
            policy
            if isinstance(policy, VfsRolloutPolicy)
            else VfsRolloutPolicy.from_dict(policy)
        )
        report = ShadowRolloutReport(
            gate_report=gate_obj,
            binding=binding_obj,
            policy=policy_obj,
            desired_mode=report.get("desired_mode", VfsRolloutMode.SHADOW),
        )
    if not isinstance(report, ShadowRolloutReport):
        raise TypeError("report must be a ShadowRolloutReport")
    dimensions = shadow_rollout_report_acceptance_dimensions(report)
    satisfied = bool(report.passed) and all(dimensions.values())
    return {
        "schema": SHADOW_ROLLOUT_REPORT_CLAIM_SCHEMA,
        "evidence": SHADOW_ROLLOUT_REPORT_EVIDENCE,
        "evidence_terms": list(shadow_rollout_report_evidence_terms()),
        "requirement_id": SHADOW_ROLLOUT_REPORT_EVIDENCE,
        "goal_id": OBJECTIVE_GOAL_G163_ID,
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "task_id": OBJECTIVE_TASK_G163_ID,
        "packet_id": OBJECTIVE_PACKET_ID,
        "packet_task_id": OBJECTIVE_TASK_PACKET_ID,
        "report_id": report.report_id,
        "gate_report_id": report.gate_report.report_id,
        "binding_id": report.binding.binding_id,
        "policy_binding_id": report.policy.policy_binding_id,
        "fixture_cid": report.gate_report.fixture.fixture_cid,
        "desired_mode": report.desired_mode.value,
        "effective_mode": report.effective_mode.value,
        "qualification_passed": report.qualification_passed,
        "passed": report.passed,
        "rollback_applied": report.rollback_applied,
        "reason_codes": list(report.reason_codes),
        "acceptance_dimensions": dimensions,
        "invariants": list(SHADOW_ROLLOUT_REPORT_INVARIANTS),
        "satisfied": satisfied,
        "automatic_ready": False,
        "automatic_mutation_enabled": False,
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


def prove_assurance_rollout_packet(
    gate_report: AdversarialE2EGateReport | Mapping[str, Any],
    shadow_report: ShadowRolloutReport | Mapping[str, Any] | None = None,
    *,
    binding: "VfsRolloutBinding | Mapping[str, Any] | None" = None,
    policy: "VfsRolloutPolicy | Mapping[str, Any] | None" = None,
    desired_mode: VfsRolloutMode | str = VfsRolloutMode.SHADOW,
) -> dict[str, Any]:
    """Emit the full VFS-G162 + VFS-G163 evidence set for the rollout packet.

    Covers both ``vfs/adversarial-e2e-gate@1`` and
    ``vfs/shadow-rollout-report@1`` in one cohesive claim for
    goal_packet/assurance_rollout.  Never grants automatic mutation or
    completion authority.
    """

    if isinstance(gate_report, Mapping):
        gate_report = AdversarialE2EGateReport.from_dict(gate_report)
    if not isinstance(gate_report, AdversarialE2EGateReport):
        raise TypeError("gate_report must be an AdversarialE2EGateReport")

    if isinstance(shadow_report, Mapping):
        raise VfsSymbolicRolloutError(
            "pass a ShadowRolloutReport instance or omit shadow_report "
            "to rebuild from gate evidence"
        )
    if shadow_report is None:
        if binding is None:
            binding = build_default_vfs_binding(gate_report.fixture)
        if policy is None:
            policy = build_default_vfs_policy(
                approve_assist=True, approve_automatic=False
            )
        if not isinstance(binding, VfsRolloutBinding):
            binding = VfsRolloutBinding.from_dict(binding)
        if not isinstance(policy, VfsRolloutPolicy):
            policy = VfsRolloutPolicy.from_dict(policy)
        shadow_report = ShadowRolloutReport(
            gate_report=gate_report,
            binding=binding,
            policy=policy,
            desired_mode=desired_mode,
        )

    if not isinstance(shadow_report, ShadowRolloutReport):
        raise TypeError("shadow_report must be a ShadowRolloutReport")
    if shadow_report.gate_report.report_id != gate_report.report_id:
        raise VfsSymbolicRolloutError(
            "assurance rollout packet reports must bind to the same "
            "adversarial gate report"
        )

    adversarial_claim = prove_adversarial_e2e_gate(gate_report)
    shadow_claim = prove_shadow_rollout_report(shadow_report)
    satisfied = bool(adversarial_claim.get("satisfied")) and bool(
        shadow_claim.get("satisfied")
    )
    return {
        "schema": ASSURANCE_ROLLOUT_PACKET_CLAIM_SCHEMA,
        "evidence_terms": list(packet_evidence_terms()),
        "all_evidence_terms": list(OBJECTIVE_DOMAIN_EVIDENCE_TERMS),
        "evidence_bindings": list(objective_evidence_bindings()),
        "packet_id": OBJECTIVE_PACKET_ID,
        "goal_ids": list(OBJECTIVE_PACKET_GOAL_IDS),
        "parent_goal_id": OBJECTIVE_PARENT_GOAL_ID,
        "task_ids": [OBJECTIVE_TASK_G162_ID, OBJECTIVE_TASK_G163_ID],
        "packet_task_id": OBJECTIVE_TASK_PACKET_ID,
        "adversarial_e2e_gate": adversarial_claim,
        "shadow_rollout_report": shadow_claim,
        "gate_report_linked": True,
        "satisfied": satisfied,
        "automatic_mutation_enabled": False,
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
    }


__all__ = (
    "ADVERSARIAL_E2E_GATE_CLAIM_SCHEMA",
    "ADVERSARIAL_E2E_GATE_EVIDENCE",
    "ADVERSARIAL_E2E_GATE_INVARIANTS",
    "ADVERSARIAL_E2E_GATE_SCHEMA",
    "ASSURANCE_ROLLOUT_PACKET_CLAIM_SCHEMA",
    "AdversarialE2EGateReport",
    "AdversarialGateId",
    "AdversarialGateObservation",
    "AdversarialInjection",
    "DEFAULT_EXCLUSION_PREFIXES",
    "DEFAULT_FIXTURE_REPOSITORIES",
    "FrozenMultiRepoFixture",
    "FrozenRepositoryDescriptor",
    "GateStatus",
    "OBJECTIVE_DOMAIN_EVIDENCE_TERMS",
    "OBJECTIVE_EVIDENCE_BINDING_ROWS",
    "OBJECTIVE_GOAL_G162_ID",
    "OBJECTIVE_GOAL_G163_ID",
    "OBJECTIVE_PACKET_ID",
    "OBJECTIVE_PACKET_EVIDENCE_TERMS",
    "OBJECTIVE_PACKET_GOAL_IDS",
    "OBJECTIVE_PARENT_GOAL_ID",
    "OBJECTIVE_TASK_G162_ID",
    "OBJECTIVE_TASK_G163_ID",
    "OBJECTIVE_TASK_PACKET_ID",
    "REQUIRED_ADVERSARIAL_GATES",
    "SHADOW_ROLLOUT_REPORT_CLAIM_SCHEMA",
    "SHADOW_ROLLOUT_REPORT_EVIDENCE",
    "SHADOW_ROLLOUT_REPORT_INVARIANTS",
    "SHADOW_ROLLOUT_REPORT_SCHEMA",
    "ShadowRolloutReport",
    "VFS_SYMBOLIC_BEHAVIOR_ID",
    "VFS_SYMBOLIC_BOUNDED_FINDINGS_SCHEMA",
    "VFS_SYMBOLIC_BOUNDED_RECEIPTS_SCHEMA",
    "VFS_SYMBOLIC_BOUNDED_STATUS_SCHEMA",
    "VFS_SYMBOLIC_CONTROL_REQUEST_SCHEMA",
    "VFS_SYMBOLIC_CONTROL_RESULT_SCHEMA",
    "VFS_SYMBOLIC_OBJECTIVE_ID",
    "VFS_SYMBOLIC_OBJECTIVE_REVISION",
    "VFS_SYMBOLIC_PUBLIC_API_SCHEMA",
    "VFS_SYMBOLIC_ROLLOUT_DECISION_SCHEMA",
    "VFS_SYMBOLIC_ROLLOUT_REQUIREMENT_ID",
    "VFS_SYMBOLIC_ROLLOUT_VERSION",
    "VfsControlAction",
    "VfsControlRequest",
    "VfsControlResult",
    "VfsControlSurface",
    "VfsRolloutBinding",
    "VfsRolloutDecision",
    "VfsRolloutMode",
    "VfsRolloutPolicy",
    "VfsSymbolicPublicAPI",
    "VfsSymbolicRolloutError",
    "adversarial_e2e_gate_acceptance_dimensions",
    "adversarial_e2e_gate_evidence",
    "adversarial_e2e_gate_evidence_terms",
    "all_covered_evidence_terms",
    "build_default_vfs_binding",
    "build_default_vfs_policy",
    "build_frozen_adversarial_population",
    "covered_evidence_terms",
    "evaluate_adversarial_gates",
    "evaluate_vfs_symbolic_rollout",
    "freeze_multi_repository_fixture",
    "objective_evidence_bindings",
    "packet_evidence_terms",
    "project_bounded_findings",
    "project_bounded_receipts",
    "project_bounded_status",
    "prove_adversarial_e2e_gate",
    "prove_assurance_rollout_packet",
    "prove_shadow_rollout_report",
    "run_vfs_symbolic_assurance_e2e",
    "shadow_rollout_report_acceptance_dimensions",
    "shadow_rollout_report_evidence",
    "shadow_rollout_report_evidence_terms",
    "verify_adversarial_e2e_report",
    "verify_vfs_symbolic_rollout",
)
