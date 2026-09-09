"""Independent contract tests for SPAR-045 frozen scale fixtures and corpus."""

from __future__ import annotations

import ast
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "benchmarks/agent_supervisor/semantic_refactoring/corpus_manifest.json",
    "benchmarks/agent_supervisor/semantic_refactoring/fixtures",
    "test/api/semantic_refactoring/test_benchmark_corpus.py",
)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
)
CAPSULE_TYPES = (
    "FunctionSemanticCapsule",
    "MethodSemanticCapsule",
    "ClassSemanticCapsule",
    "TopLevelBlockCapsule",
    "ModuleSemanticCapsule",
    "PackageSemanticCapsule",
    "CallsiteSemanticCapsule",
    "StateOwnerCapsule",
    "RegistrationCapsule",
    "ResourceLifecycleCapsule",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
MANIFEST_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/corpus_manifest.json"
)
FIXTURES_RELATIVE = "benchmarks/agent_supervisor/semantic_refactoring/fixtures"
PREREGISTRATION_RELATIVE = (
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json"
)

TASK_ID: str = "SPAR-045"
GOAL_ID: str = "SPAR-G081"
PROGRAM: str = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: str = "benchmark fixture generation"
AUTHORITY_OWNER: str = "ipfs_accelerate_py"
ANALYZER_ID: str = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.benchmark_corpus@1"
)
PREDECESSOR_TASK_IDS: tuple[str, ...] = ("SPAR-043",)
SPAR_BENCHMARK_CORPUS_INTERFACE: str = "SparBenchmarkCorpus@1"
SPAR_BENCHMARK_CORPUS_RECEIPT_INTERFACE: str = "SparBenchmarkCorpusReceipt@1"
FROZEN_SCALE_FIXTURE_INTERFACE: str = "FrozenScaleFixture@1"
SEALED_SPLIT_MANIFEST_INTERFACE: str = "SealedSplitManifest@1"
SPAR_BENCHMARK_CORPUS_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-benchmark-corpus@1"
)
SPAR_BENCHMARK_CORPUS_RECEIPT_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/spar-benchmark-corpus-receipt@1"
)
FROZEN_SCALE_FIXTURE_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/frozen-scale-fixture@1"
)
SEALED_SPLIT_MANIFEST_SCHEMA: str = (
    "ipfs_accelerate_py/agent-supervisor/sealed-split-manifest@1"
)
CORPUS_CONTRACT_VERSION: str = "1"
NETWORK_DENY: str = "deny"
CAN_AUTHORIZE_COMPLETION: bool = False
CAN_AUTHORIZE_TRANSITION: bool = False
CAN_CREATE_AUTHORITY: bool = False
GATE_WRITES_REPOSITORY: bool = False
GATE_IS_NOMINATION_ONLY: bool = True
VECTOR_SIMILARITY_IS_AUTHORITY: bool = False
PROJECTION_CLUSTERING_IS_AUTHORITY: bool = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: bool = True
TEST_PASS_IS_NOT_COMPLETION: bool = True
MARKDOWN_IS_NOT_COMPLETION: bool = True
WORKER_SELF_APPROVAL: bool = False
DUCKLAKE_IS_AUTHORITY: bool = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: bool = True
NETWORK_DENIED: bool = True
RAW_SOURCE_REQUIRED: bool = True

REQUIRED_DOMAINS: tuple[str, ...] = (
    "state",
    "initialization",
    "dynamics",
    "registries",
    "async",
    "exceptions",
    "resources",
    "apis",
    "seeded_defects",
)
DOMAIN_MODULES: dict[str, str] = {
    "state": "state.py",
    "initialization": "initialization.py",
    "dynamics": "dynamics.py",
    "registries": "registries.py",
    "async": "async_runtime.py",
    "exceptions": "exceptions.py",
    "resources": "resources.py",
    "apis": "apis.py",
    "seeded_defects": "seeded_defects.py",
}
DOMAIN_CLASSES: dict[str, str] = {
    "state": "StateOwner",
    "initialization": "InitializationOwner",
    "dynamics": "DynamicsOwner",
    "registries": "RegistryOwner",
    "async": "AsyncOwner",
    "exceptions": "ExceptionOwner",
    "resources": "ResourceOwner",
    "apis": "ApiOwner",
    "seeded_defects": "SeededDefectOwner",
}
SYNTHETIC_PROFILES: tuple[str, ...] = ("small", "medium", "large")
PAD_STATEMENT: str = "    value = value + 1"
IDENTITY_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "timestamp",
        "timestamps",
        "process_id",
        "pid",
        "local_path",
        "local_paths",
        "checkout_path",
        "model_output",
        "model",
        "provider",
        "prompt",
        "lease",
        "fence",
        "generation",
        "receipt",
        "acceptance",
        "wall_clock",
        "clock",
        "source",
        "source_text",
        "source_body",
        "file_contents",
        "repository_dump",
    }
)
_AUTHORITY_FLAGS: tuple[str, ...] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "writes_repository",
    "worker_self_approval",
    "projection_is_authority",
)


class BenchmarkCorpusError(RuntimeError):
    """Fail-closed SPAR-045 corpus or fixture contract violation."""


def provider_free_exports() -> tuple[str, ...]:
    return (
        "FrozenScaleFixture",
        "SealedSplitManifest",
        "SparBenchmarkCorpus",
        "SparBenchmarkCorpusReceipt",
        "generate_controlled_monolith",
        "load_benchmark_corpus_manifest",
        "nominate_benchmark_corpus",
        "seal_fixture_splits",
    )


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & set(CAPSULE_TYPES)
    if overlap:
        raise BenchmarkCorpusError(
            f"benchmark corpus must not define competing types: {sorted(overlap)}"
        )


def corpus_cid_profile() -> dict[str, str]:
    return {
        "codec": "dag-json",
        "hash": "sha2-256",
        "rule": "content-addressed over sealed fields; not universal meaning",
    }


def _load_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkCorpusError(f"{relative} must be a JSON object")
    return payload


def _count_loc(source: str) -> int:
    if not source:
        return 0
    text = source if source.endswith("\n") else source + "\n"
    return text.count("\n")


def _allocate_loc(target_loc: int, domains: Sequence[str]) -> dict[str, int]:
    names = tuple(domains)
    if not names:
        raise BenchmarkCorpusError("domains are required")
    if type(target_loc) is not int or target_loc < 1:
        raise BenchmarkCorpusError("target_loc must be a positive integer")
    base, remainder = divmod(target_loc, len(names))
    allocated = {name: base for name in names}
    allocated[names[-1]] += remainder
    if sum(allocated.values()) != target_loc:
        raise BenchmarkCorpusError("loc allocation is not exact")
    return allocated


def _header_lines(domain: str, defects: Sequence[str]) -> list[str]:
    class_name = DOMAIN_CLASSES[domain]
    module_name = DOMAIN_MODULES[domain]
    lines = [
        f'"""SPAR-045 controlled fixture module: {domain}."""',
        "from __future__ import annotations",
        f"DOMAIN = {domain!r}",
        f"MODULE = {module_name!r}",
        f"class {class_name}:",
        f"    domain = {domain!r}",
        "    def __init__(self) -> None:",
        "        self._value = 0",
        "    def read(self) -> int:",
        "        return self._value",
        "    def write(self, value: int) -> int:",
        "        self._value = value",
        "        return self._value",
    ]
    if domain == "registries":
        lines.extend(
            [
                "REGISTRY: dict[str, RegistryOwner] = {}",
                "def register_item(name: str, owner: RegistryOwner) -> RegistryOwner:",
                "    REGISTRY[name] = owner",
                "    return owner",
            ]
        )
    elif domain == "exceptions":
        lines.extend(
            [
                "class ExceptionOwnerError(RuntimeError):",
                "    pass",
                "def raise_controlled() -> None:",
                "    raise ExceptionOwnerError(DOMAIN)",
            ]
        )
    elif domain == "resources":
        lines.extend(
            [
                "class ResourceLease:",
                "    def __enter__(self) -> ResourceOwner:",
                "        return ResourceOwner()",
                "    def __exit__(self, *exc: object) -> bool:",
                "        return False",
            ]
        )
    elif domain == "async":
        lines.extend(
            [
                "async def async_entry(owner: AsyncOwner) -> int:",
                "    return owner.write(owner.read() + 1)",
            ]
        )
    elif domain == "initialization":
        lines.extend(
            [
                "INITIALIZED = True",
                "def initialization_entry(owner: InitializationOwner) -> int:",
                "    return owner.write(int(INITIALIZED))",
            ]
        )
    elif domain == "seeded_defects":
        for defect_id in defects:
            ident = defect_id.replace("-", "_")
            lines.extend(
                [
                    f"def seeded_{ident}() -> str:",
                    f'    marker = "SEEDED_DEFECT:{defect_id}"',
                    "    return marker",
                ]
            )
    else:
        ident = domain.replace("-", "_")
        lines.extend(
            [
                f"def {ident}_entry(owner: {class_name}) -> int:",
                "    return owner.write(owner.read() + 1)",
            ]
        )
    pad_name = domain.replace("-", "_")
    lines.extend(
        [
            f"def _pad_{pad_name}() -> int:",
            "    value = 0",
        ]
    )
    return lines


def _module_source(domain: str, loc: int, defects: Sequence[str]) -> str:
    header = _header_lines(domain, defects)
    footer = ["    return value"]
    minimum = len(header) + len(footer)
    if loc < minimum:
        raise BenchmarkCorpusError(
            f"{domain} target loc {loc} is below header minimum {minimum}"
        )
    pad_count = loc - minimum
    lines = header + [PAD_STATEMENT] * pad_count + footer
    source = "\n".join(lines) + "\n"
    if _count_loc(source) != loc:
        raise BenchmarkCorpusError(f"{domain} generated loc is not exact")
    return source


def load_benchmark_corpus_manifest() -> dict[str, Any]:
    payload = _load_json(MANIFEST_RELATIVE)
    if payload.get("schema") != SPAR_BENCHMARK_CORPUS_SCHEMA:
        raise BenchmarkCorpusError("unsupported corpus manifest schema")
    if payload.get("interface") != SPAR_BENCHMARK_CORPUS_INTERFACE:
        raise BenchmarkCorpusError("unsupported corpus manifest interface")
    if payload.get("task_id") != TASK_ID:
        raise BenchmarkCorpusError("corpus manifest task_id mismatch")
    if payload.get("nomination_only") is not True:
        raise BenchmarkCorpusError("corpus manifest must remain nomination_only")
    if payload.get("can_authorize_completion") is not False:
        raise BenchmarkCorpusError("corpus cannot authorize completion")
    if payload.get("network") != NETWORK_DENY:
        raise BenchmarkCorpusError("corpus network must remain deny")
    return payload


def load_preregistration() -> dict[str, Any]:
    payload = _load_json(PREREGISTRATION_RELATIVE)
    if payload.get("schema") != "spar/benchmark-preregistration@1":
        raise BenchmarkCorpusError("preregistration schema mismatch")
    return payload


def load_recipes() -> dict[str, Any]:
    manifest = load_benchmark_corpus_manifest()
    payload = _load_json(str(manifest["recipes_path"]))
    if payload.get("task_id") != TASK_ID:
        raise BenchmarkCorpusError("recipes task_id mismatch")
    if payload.get("rights_admitted") is not True:
        raise BenchmarkCorpusError("recipes must be rights-admitted")
    if payload.get("third_party_source") is not False:
        raise BenchmarkCorpusError("third-party source is not admitted")
    if payload.get("whole_repository_dump") is not False:
        raise BenchmarkCorpusError("repository dump is forbidden")
    if payload.get("network") != NETWORK_DENY:
        raise BenchmarkCorpusError("recipes network must remain deny")
    return payload


def load_split_policy() -> dict[str, Any]:
    manifest = load_benchmark_corpus_manifest()
    payload = _load_json(str(manifest["splits_path"]))
    if payload.get("keep_domain_module_together") is not True:
        raise BenchmarkCorpusError("split policy must keep domain modules together")
    if payload.get("unsafe_intra_domain_split") is not False:
        raise BenchmarkCorpusError("unsafe intra-domain split")
    if payload.get("network") != NETWORK_DENY:
        raise BenchmarkCorpusError("split policy network must remain deny")
    return payload


def load_seeded_defects() -> tuple[str, ...]:
    manifest = load_benchmark_corpus_manifest()
    payload = _load_json(str(manifest["seeded_defects_path"]))
    defects = tuple(str(item["defect_id"]) for item in payload["defects"])
    if len(defects) != len(set(defects)):
        raise BenchmarkCorpusError("seeded defect ids must be unique")
    return defects


def load_real_current_tree() -> dict[str, Any]:
    manifest = load_benchmark_corpus_manifest()
    payload = _load_json(str(manifest["real_current_tree_path"]))
    if payload.get("whole_repository_dump") is not False:
        raise BenchmarkCorpusError("real_current_tree must not dump the repository")
    if payload.get("synthetic") is not False:
        raise BenchmarkCorpusError("real_current_tree must not be synthetic")
    if payload.get("rights_basis") != "current_tree_owned":
        raise BenchmarkCorpusError("real_current_tree rights_basis mismatch")
    admitted = str(payload.get("admitted_path") or "")
    if not admitted or admitted.startswith("/") or ".." in Path(admitted).parts:
        raise BenchmarkCorpusError("real_current_tree path is not exact")
    return payload


@dataclass(frozen=True, slots=True)
class FixtureModule:
    """One domain module of a frozen synthetic monolith."""

    domain: str
    module_name: str
    loc: int
    source: str
    module_cid: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "loc": self.loc,
            "module_cid": self.module_cid,
            "module_name": self.module_name,
        }


@dataclass(frozen=True, slots=True)
class FrozenScaleFixture:
    """Frozen 5k/20k/100k controlled monolith nomination."""

    name: str
    target_loc: int
    loc: int
    kind: str
    rights_basis: str
    synthetic: bool
    modules: tuple[FixtureModule, ...]
    defect_ids: tuple[str, ...]
    fixture_cid: str
    interface: str = FROZEN_SCALE_FIXTURE_INTERFACE
    schema: str = FROZEN_SCALE_FIXTURE_SCHEMA

    def identity_payload(self) -> dict[str, Any]:
        return {
            "defect_ids": list(self.defect_ids),
            "fixture_cid": self.fixture_cid,
            "interface": self.interface,
            "kind": self.kind,
            "loc": self.loc,
            "modules": [item.identity_payload() for item in self.modules],
            "name": self.name,
            "rights_basis": self.rights_basis,
            "schema": self.schema,
            "synthetic": self.synthetic,
            "target_loc": self.target_loc,
        }


@dataclass(frozen=True, slots=True)
class SealedSplitManifest:
    """Domain-aligned sealed split over a frozen fixture."""

    profile: str
    split_kind: str
    members: tuple[dict[str, Any], ...]
    split_cid: str
    keep_domain_module_together: bool = True
    unsafe_intra_domain_split: bool = False
    interface: str = SEALED_SPLIT_MANIFEST_INTERFACE
    schema: str = SEALED_SPLIT_MANIFEST_SCHEMA

    def identity_payload(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "keep_domain_module_together": True,
            "members": [dict(item) for item in self.members],
            "profile": self.profile,
            "schema": self.schema,
            "split_cid": self.split_cid,
            "split_kind": self.split_kind,
            "unsafe_intra_domain_split": False,
        }


@dataclass(frozen=True, slots=True)
class SparBenchmarkCorpus:
    """Frozen SPAR-045 corpus bound to sealed preregistration profiles."""

    manifest: Mapping[str, Any]
    recipes_cid: str
    split_policy_cid: str
    seeded_defects_cid: str
    corpus_cid: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "corpus_cid": self.corpus_cid,
            "interface": SPAR_BENCHMARK_CORPUS_INTERFACE,
            "manifest": dict(self.manifest),
            "recipes_cid": self.recipes_cid,
            "schema": SPAR_BENCHMARK_CORPUS_SCHEMA,
            "seeded_defects_cid": self.seeded_defects_cid,
            "split_policy_cid": self.split_policy_cid,
            "task_id": TASK_ID,
        }


@dataclass(frozen=True, slots=True)
class SparBenchmarkCorpusReceipt:
    """Nomination-only SPAR-045 corpus receipt."""

    tree_id: str
    corpus_cid: str
    profiles: tuple[dict[str, Any], ...]
    split_cids: tuple[str, ...]
    defect_ids: tuple[str, ...]
    nominated: bool = True
    accepted: bool = False
    nomination_only: bool = True
    can_authorize_completion: bool = False
    can_authorize_transition: bool = False
    can_create_authority: bool = False
    writes_repository: bool = False
    worker_self_approval: bool = False
    projection_is_authority: bool = False
    frozen: bool = True
    rights_admitted: bool = True
    network: str = NETWORK_DENY
    analyzer_id: str = ANALYZER_ID
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    interface: str = SPAR_BENCHMARK_CORPUS_RECEIPT_INTERFACE
    schema: str = SPAR_BENCHMARK_CORPUS_RECEIPT_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "accepted",
            "analyzer_id",
            "can_authorize_completion",
            "can_authorize_transition",
            "can_create_authority",
            "corpus_cid",
            "defect_ids",
            "frozen",
            "goal_id",
            "interface",
            "network",
            "nominated",
            "nomination_only",
            "profiles",
            "projection_is_authority",
            "receipt_cid",
            "rights_admitted",
            "schema",
            "split_cids",
            "task_id",
            "tree_id",
            "worker_self_approval",
            "writes_repository",
        }
    )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "analyzer_id": self.analyzer_id,
            "can_authorize_completion": False,
            "can_authorize_transition": False,
            "can_create_authority": False,
            "corpus_cid": self.corpus_cid,
            "defect_ids": list(self.defect_ids),
            "frozen": True,
            "goal_id": self.goal_id,
            "interface": self.interface,
            "network": NETWORK_DENY,
            "nominated": True,
            "nomination_only": True,
            "profiles": [dict(item) for item in self.profiles],
            "projection_is_authority": False,
            "rights_admitted": True,
            "schema": self.schema,
            "split_cids": list(self.split_cids),
            "task_id": self.task_id,
            "tree_id": self.tree_id,
            "worker_self_approval": False,
            "writes_repository": False,
        }

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SparBenchmarkCorpusReceipt":
        excluded = set(data) & IDENTITY_EXCLUDED_FIELDS
        if excluded:
            raise BenchmarkCorpusError(
                f"observational fields are excluded from identity: {sorted(excluded)}"
            )
        unknown = set(data) - cls._FIELDS
        if unknown:
            raise BenchmarkCorpusError(
                f"unsupported SparBenchmarkCorpusReceipt fields: {sorted(unknown)}"
            )
        payload = dict(data)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != SPAR_BENCHMARK_CORPUS_RECEIPT_SCHEMA:
            raise BenchmarkCorpusError("unsupported corpus receipt schema")
        if payload.pop("interface") != SPAR_BENCHMARK_CORPUS_RECEIPT_INTERFACE:
            raise BenchmarkCorpusError("unsupported corpus receipt interface")
        if payload.pop("accepted") is not False:
            raise BenchmarkCorpusError("workers cannot self-approve SPAR-045")
        if payload.pop("nomination_only") is not True:
            raise BenchmarkCorpusError("receipt must remain nomination_only")
        if payload.pop("nominated") is not True:
            raise BenchmarkCorpusError("receipt must remain nominated")
        for flag in _AUTHORITY_FLAGS:
            if payload.pop(flag) is not False:
                raise BenchmarkCorpusError(f"receipt cannot claim {flag}")
        if payload.pop("network") != NETWORK_DENY:
            raise BenchmarkCorpusError("receipt network must remain deny")
        if payload.pop("frozen") is not True:
            raise BenchmarkCorpusError("corpus receipt must remain frozen")
        if payload.pop("rights_admitted") is not True:
            raise BenchmarkCorpusError("corpus receipt must remain rights-admitted")
        result = cls(
            tree_id=str(payload["tree_id"]),
            corpus_cid=str(payload["corpus_cid"]),
            profiles=tuple(dict(item) for item in payload["profiles"]),
            split_cids=tuple(str(item) for item in payload["split_cids"]),
            defect_ids=tuple(str(item) for item in payload["defect_ids"]),
            analyzer_id=str(payload["analyzer_id"]),
            task_id=str(payload["task_id"]),
            goal_id=str(payload["goal_id"]),
        )
        if claimed != result.receipt_cid:
            raise BenchmarkCorpusError("SparBenchmarkCorpusReceipt receipt_cid mismatch")
        return result


@functools.lru_cache(maxsize=8)
def generate_controlled_monolith(name: str) -> FrozenScaleFixture:
    """Expand one frozen synthetic profile to exact target LOC."""

    recipes = load_recipes()
    if name not in SYNTHETIC_PROFILES:
        raise BenchmarkCorpusError(f"unsupported synthetic profile {name!r}")
    profile = recipes["profiles"][name]
    target_loc = int(profile["target_loc"])
    domains = tuple(recipes["domains"])
    if domains != REQUIRED_DOMAINS:
        raise BenchmarkCorpusError("recipe domains must match required domains")
    defects = load_seeded_defects()
    allocated = _allocate_loc(target_loc, domains)
    modules: list[FixtureModule] = []
    for domain in domains:
        source = _module_source(domain, allocated[domain], defects)
        ast.parse(source)
        module = FixtureModule(
            domain=domain,
            module_name=DOMAIN_MODULES[domain],
            loc=allocated[domain],
            source=source,
            module_cid=cid_for_bytes(source.encode("utf-8")),
        )
        modules.append(module)
    loc = sum(item.loc for item in modules)
    if loc != target_loc:
        raise BenchmarkCorpusError(f"{name} generated loc is not exact")
    identity = {
        "defect_ids": list(defects),
        "kind": profile["kind"],
        "loc": loc,
        "modules": [item.identity_payload() for item in modules],
        "name": name,
        "rights_basis": profile["rights_basis"],
        "synthetic": True,
        "target_loc": target_loc,
    }
    return FrozenScaleFixture(
        name=name,
        target_loc=target_loc,
        loc=loc,
        kind=str(profile["kind"]),
        rights_basis=str(profile["rights_basis"]),
        synthetic=True,
        modules=tuple(modules),
        defect_ids=defects,
        fixture_cid=cid_for_dag_json(identity),
    )


def admit_real_current_tree() -> dict[str, Any]:
    """Nominate one rights-admitted current-tree module without dumping the repo."""

    spec = load_real_current_tree()
    relative = str(spec["admitted_path"])
    path = ROOT / relative
    if not path.is_file():
        raise BenchmarkCorpusError("admitted current-tree module is missing")
    source = path.read_text(encoding="utf-8")
    loc = _count_loc(source)
    if loc < 1:
        raise BenchmarkCorpusError("admitted current-tree module is empty")
    return {
        "admitted_path": relative,
        "kind": spec["kind"],
        "loc": loc,
        "module_cid": cid_for_bytes(source.encode("utf-8")),
        "name": "real_current_tree",
        "rights_basis": spec["rights_basis"],
        "synthetic": False,
        "target_loc": None,
        "whole_repository_dump": False,
    }


def seal_fixture_splits(
    fixture: FrozenScaleFixture,
    *,
    policy: Mapping[str, Any] | None = None,
) -> SealedSplitManifest:
    """Seal one domain-aligned split; intra-domain cuts fail closed."""

    loaded = dict(policy) if policy is not None else load_split_policy()
    if loaded.get("keep_domain_module_together") is not True:
        raise BenchmarkCorpusError("split policy must keep domain modules together")
    if loaded.get("unsafe_intra_domain_split") is not False:
        raise BenchmarkCorpusError("unsafe intra-domain split")
    expected = {item["domain"]: item["module"] for item in loaded["members"]}
    if tuple(expected) != tuple(item.domain for item in fixture.modules):
        raise BenchmarkCorpusError("split members must match fixture domains")
    members: list[dict[str, Any]] = []
    for module in fixture.modules:
        if expected[module.domain] != module.module_name:
            raise BenchmarkCorpusError("split module name mismatch")
        members.append(
            {
                "constraint_class": "scc",
                "domain": module.domain,
                "loc": module.loc,
                "module": module.module_name,
                "module_cid": module.module_cid,
            }
        )
    identity = {
        "keep_domain_module_together": True,
        "members": members,
        "profile": fixture.name,
        "split_kind": loaded["split_kind"],
        "unsafe_intra_domain_split": False,
    }
    return SealedSplitManifest(
        profile=fixture.name,
        split_kind=str(loaded["split_kind"]),
        members=tuple(members),
        split_cid=cid_for_dag_json(identity),
    )


def materialize_fixture(fixture: FrozenScaleFixture, directory: Path) -> None:
    """Write generated modules into an isolated directory."""

    directory.mkdir(parents=True, exist_ok=True)
    for module in fixture.modules:
        relative = Path(module.module_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise BenchmarkCorpusError("materialize path is not exact")
        target = directory / relative
        if target.resolve().parent != directory.resolve():
            raise BenchmarkCorpusError("materialize path is not exact")
        target.write_text(module.source, encoding="utf-8")


def nominate_benchmark_corpus() -> SparBenchmarkCorpusReceipt:
    """Nominate the frozen SPAR-045 corpus. Current authority remains separate."""

    manifest = load_benchmark_corpus_manifest()
    preregistration = load_preregistration()
    recipes = load_recipes()
    defects = load_seeded_defects()
    floors = dict(preregistration["zero_safety_floors"])
    if dict(manifest["zero_safety_floors"]) != floors:
        raise BenchmarkCorpusError("corpus floors must match preregistration")
    if set(defects) != set(floors):
        raise BenchmarkCorpusError("seeded defects must cover zero safety floors")
    declared = [item["name"] for item in manifest["profiles"]]
    registered = [item["name"] for item in preregistration["profiles"]]
    if declared != registered:
        raise BenchmarkCorpusError("corpus profiles must match preregistration")
    profile_rows: list[dict[str, Any]] = []
    split_cids: list[str] = []
    for item in manifest["profiles"]:
        name = str(item["name"])
        registered_item = next(
            row for row in preregistration["profiles"] if row["name"] == name
        )
        if item.get("target_loc") != registered_item.get("target_loc"):
            raise BenchmarkCorpusError(f"{name} target_loc must stay preregistered")
        if name in SYNTHETIC_PROFILES:
            fixture = generate_controlled_monolith(name)
            split = seal_fixture_splits(fixture)
            recipe_target = recipes["profiles"][name]["target_loc"]
            if fixture.loc != recipe_target:
                raise BenchmarkCorpusError(f"{name} loc is not exact")
            profile_rows.append(
                {
                    "defect_ids": list(fixture.defect_ids),
                    "domains": [module.domain for module in fixture.modules],
                    "fixture_cid": fixture.fixture_cid,
                    "kind": fixture.kind,
                    "loc": fixture.loc,
                    "name": fixture.name,
                    "rights_basis": fixture.rights_basis,
                    "split_cid": split.split_cid,
                    "synthetic": True,
                    "target_loc": fixture.target_loc,
                }
            )
            split_cids.append(split.split_cid)
        elif name == "real_current_tree":
            admitted = admit_real_current_tree()
            split_identity = {
                "keep_domain_module_together": True,
                "members": [
                    {
                        "admitted_path": admitted["admitted_path"],
                        "module_cid": admitted["module_cid"],
                    }
                ],
                "profile": name,
                "split_kind": "admitted_current_tree_module",
                "unsafe_intra_domain_split": False,
            }
            split_cid = cid_for_dag_json(split_identity)
            profile_rows.append(
                {
                    "admitted_path": admitted["admitted_path"],
                    "defect_ids": [],
                    "domains": [],
                    "fixture_cid": admitted["module_cid"],
                    "kind": admitted["kind"],
                    "loc": admitted["loc"],
                    "name": name,
                    "rights_basis": admitted["rights_basis"],
                    "split_cid": split_cid,
                    "synthetic": False,
                    "target_loc": None,
                    "whole_repository_dump": False,
                }
            )
            split_cids.append(split_cid)
        else:
            raise BenchmarkCorpusError(f"unsupported profile {name!r}")
    corpus = SparBenchmarkCorpus(
        manifest=manifest,
        recipes_cid=cid_for_dag_json(recipes),
        split_policy_cid=cid_for_dag_json(load_split_policy()),
        seeded_defects_cid=cid_for_dag_json(
            _load_json(str(manifest["seeded_defects_path"]))
        ),
        corpus_cid=cid_for_dag_json(
            {
                "manifest": manifest,
                "profiles": profile_rows,
                "task_id": TASK_ID,
                "tree_id": TREE_ID,
            }
        ),
    )
    return SparBenchmarkCorpusReceipt(
        tree_id=TREE_ID,
        corpus_cid=corpus.corpus_cid,
        profiles=tuple(profile_rows),
        split_cids=tuple(split_cids),
        defect_ids=defects,
    )


@functools.lru_cache(maxsize=1)
def _cached_nomination() -> SparBenchmarkCorpusReceipt:
    return nominate_benchmark_corpus()


def dry_run_benchmark_corpus() -> SparBenchmarkCorpusReceipt:
    return _cached_nomination()


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-045"
    assert GOAL_ID == "SPAR-G081"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert PREDECESSOR_TASK_IDS == ("SPAR-043",)
    assert SPAR_BENCHMARK_CORPUS_INTERFACE == "SparBenchmarkCorpus@1"
    assert SPAR_BENCHMARK_CORPUS_RECEIPT_INTERFACE == "SparBenchmarkCorpusReceipt@1"
    assert FROZEN_SCALE_FIXTURE_INTERFACE == "FrozenScaleFixture@1"
    assert SEALED_SPLIT_MANIFEST_INTERFACE == "SealedSplitManifest@1"
    assert CORPUS_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("benchmark_corpus@1")
    assert TEST_PATH.is_file()
    assert (ROOT / MANIFEST_RELATIVE).is_file()
    assert (ROOT / FIXTURES_RELATIVE).is_dir()
    assert WRITE_SCOPE == (
        MANIFEST_RELATIVE,
        FIXTURES_RELATIVE,
        "test/api/semantic_refactoring/test_benchmark_corpus.py",
    )
    for relative in (
        "benchmarks/agent_supervisor/semantic_refactoring/fixtures/recipes.json",
        "benchmarks/agent_supervisor/semantic_refactoring/fixtures/splits.json",
        "benchmarks/agent_supervisor/semantic_refactoring/fixtures/seeded_defects.json",
        "benchmarks/agent_supervisor/semantic_refactoring/fixtures/real_current_tree.json",
    ):
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "benchmark fixture generation"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert CAN_AUTHORIZE_COMPLETION is False
    assert CAN_AUTHORIZE_TRANSITION is False
    assert CAN_CREATE_AUTHORITY is False
    assert GATE_WRITES_REPOSITORY is False
    assert GATE_IS_NOMINATION_ONLY is True
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert RAW_SOURCE_REQUIRED is True
    profile = corpus_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    manifest = load_benchmark_corpus_manifest()
    assert manifest["nomination_only"] is True
    assert manifest["can_authorize_completion"] is False
    assert manifest["can_authorize_transition"] is False
    assert manifest["can_create_authority"] is False
    assert manifest["worker_self_approval"] is False
    assert manifest["model_output_is_proposal_only"] is True
    assert manifest["test_pass_is_not_completion"] is True


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(TEST_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "FrozenScaleFixture" in names
    assert "SealedSplitManifest" in names
    assert "SparBenchmarkCorpus" in names
    assert "SparBenchmarkCorpusReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "FrozenScaleFixture" in exports
    assert "SealedSplitManifest" in exports
    assert "nominate_benchmark_corpus" in exports
    functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for forbidden in (
        "authorize_completion",
        "promote_root",
        "dump_repository",
        "admit_third_party_source",
    ):
        assert forbidden not in functions
        assert forbidden not in names


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_corpus_manifest_binds_preregistered_profiles() -> None:
    manifest = load_benchmark_corpus_manifest()
    preregistration = load_preregistration()
    assert manifest["preregistration_path"] == PREREGISTRATION_RELATIVE
    assert manifest["sealed_before_tuning"] is True
    assert manifest["frozen"] is True
    assert [item["name"] for item in manifest["profiles"]] == [
        item["name"] for item in preregistration["profiles"]
    ]
    for declared, registered in zip(
        manifest["profiles"], preregistration["profiles"], strict=True
    ):
        assert declared["name"] == registered["name"]
        assert declared["target_loc"] == registered["target_loc"]
    assert manifest["required_domains"] == list(REQUIRED_DOMAINS)
    assert manifest["zero_safety_floors"] == preregistration["zero_safety_floors"]
    assert preregistration["sealed_before_tuning"] is True
    assert manifest["recipes_cid"] == cid_for_dag_json(load_recipes())
    assert manifest["split_policy_cid"] == cid_for_dag_json(load_split_policy())
    assert manifest["seeded_defects_cid"] == cid_for_dag_json(
        _load_json(str(manifest["seeded_defects_path"]))
    )
    assert manifest["real_current_tree_cid"] == cid_for_dag_json(
        _load_json(str(manifest["real_current_tree_path"]))
    )


def test_committed_fixtures_remain_compact_recipes() -> None:
    fixtures = ROOT / FIXTURES_RELATIVE
    assert list(fixtures.rglob("*.py")) == []
    recipes = load_recipes()
    assert recipes["synthetic"] is True
    assert recipes["rights_basis"] == "synthetic_original"
    assert tuple(recipes["domains"]) == REQUIRED_DOMAINS
    assert recipes["profiles"]["small"]["target_loc"] == 5000
    assert recipes["profiles"]["medium"]["target_loc"] == 20000
    assert recipes["profiles"]["large"]["target_loc"] == 100000


def test_synthetic_profiles_hit_exact_preregistered_loc() -> None:
    expected = {"small": 5000, "medium": 20000, "large": 100000}
    for name, target in expected.items():
        fixture = generate_controlled_monolith(name)
        assert fixture.loc == target
        assert fixture.target_loc == target
        assert fixture.synthetic is True
        assert fixture.rights_basis == "synthetic_original"
        assert tuple(item.domain for item in fixture.modules) == REQUIRED_DOMAINS
        assert sum(item.loc for item in fixture.modules) == target
        again = generate_controlled_monolith(name)
        assert again.fixture_cid == fixture.fixture_cid
        assert [item.module_cid for item in again.modules] == [
            item.module_cid for item in fixture.modules
        ]


def test_seeded_defects_cover_zero_safety_floors() -> None:
    preregistration = load_preregistration()
    defects = load_seeded_defects()
    assert set(defects) == set(preregistration["zero_safety_floors"])
    fixture = generate_controlled_monolith("small")
    seeded = next(item for item in fixture.modules if item.domain == "seeded_defects")
    for defect_id in defects:
        assert f"SEEDED_DEFECT:{defect_id}" in seeded.source
        assert f"def seeded_{defect_id}(" in seeded.source


def test_sealed_splits_keep_domain_modules_together() -> None:
    fixture = generate_controlled_monolith("small")
    split = seal_fixture_splits(fixture)
    assert split.keep_domain_module_together is True
    assert split.unsafe_intra_domain_split is False
    assert split.split_kind == "domain_scc_aligned"
    assert len(split.members) == len(REQUIRED_DOMAINS)
    assert [item["domain"] for item in split.members] == list(REQUIRED_DOMAINS)
    again = seal_fixture_splits(fixture)
    assert again.split_cid == split.split_cid
    mutated = dict(load_split_policy())
    mutated["unsafe_intra_domain_split"] = True
    with pytest.raises(BenchmarkCorpusError, match="unsafe intra-domain split"):
        seal_fixture_splits(fixture, policy=mutated)
    mutated = dict(load_split_policy())
    mutated["keep_domain_module_together"] = False
    with pytest.raises(BenchmarkCorpusError, match="keep domain modules together"):
        seal_fixture_splits(fixture, policy=mutated)


def test_real_current_tree_admits_owned_module_without_dump() -> None:
    admitted = admit_real_current_tree()
    assert admitted["name"] == "real_current_tree"
    assert admitted["synthetic"] is False
    assert admitted["whole_repository_dump"] is False
    assert admitted["target_loc"] is None
    assert admitted["rights_basis"] == "current_tree_owned"
    assert (
        admitted["admitted_path"]
        == "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/rollout.py"
    )
    assert (ROOT / admitted["admitted_path"]).is_file()
    assert admitted["loc"] == _count_loc(
        (ROOT / admitted["admitted_path"]).read_text(encoding="utf-8")
    )
    payload = _load_json(
        "benchmarks/agent_supervisor/semantic_refactoring/fixtures/real_current_tree.json"
    )
    assert "class ShadowPlanGate" not in json.dumps(payload)
    assert "def " not in json.dumps(payload)
    assert payload["whole_repository_dump"] is False


def test_small_fixture_materializes_and_parses(tmp_path: Path) -> None:
    fixture = generate_controlled_monolith("small")
    materialize_fixture(fixture, tmp_path)
    counted = 0
    for module in fixture.modules:
        path = tmp_path / module.module_name
        text = path.read_text(encoding="utf-8")
        ast.parse(text)
        counted += _count_loc(text)
        assert _count_loc(text) == module.loc
    assert counted == 5000


def test_nomination_receipt_cannot_complete_or_self_approve() -> None:
    receipt = _cached_nomination()
    assert receipt.nominated is True
    assert receipt.accepted is False
    assert receipt.nomination_only is True
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_authority is False
    assert receipt.writes_repository is False
    assert receipt.worker_self_approval is False
    assert receipt.network == NETWORK_DENY
    assert receipt.task_id == TASK_ID
    assert receipt.goal_id == GOAL_ID
    assert receipt.tree_id == TREE_ID
    assert [item["name"] for item in receipt.profiles] == [
        "small",
        "medium",
        "large",
        "real_current_tree",
    ]
    by_name = {item["name"]: item for item in receipt.profiles}
    assert by_name["small"]["loc"] == 5000
    assert by_name["medium"]["loc"] == 20000
    assert by_name["large"]["loc"] == 100000
    assert by_name["real_current_tree"]["target_loc"] is None
    assert by_name["real_current_tree"]["whole_repository_dump"] is False
    encoded = receipt.to_dict()
    restored = SparBenchmarkCorpusReceipt.from_dict(encoded)
    assert restored.receipt_cid == receipt.receipt_cid
    assert restored == receipt
    dry = dry_run_benchmark_corpus()
    assert dry.receipt_cid == receipt.receipt_cid
    mutated = dict(encoded)
    mutated["accepted"] = True
    with pytest.raises(BenchmarkCorpusError, match="self-approve"):
        SparBenchmarkCorpusReceipt.from_dict(mutated)
    mutated = dict(encoded)
    mutated["nomination_only"] = False
    with pytest.raises(BenchmarkCorpusError, match="nomination_only"):
        SparBenchmarkCorpusReceipt.from_dict(mutated)
    mutated = dict(encoded)
    mutated["can_authorize_completion"] = True
    with pytest.raises(BenchmarkCorpusError, match="can_authorize_completion"):
        SparBenchmarkCorpusReceipt.from_dict(mutated)
    mutated = dict(encoded)
    mutated["source"] = "forbidden"
    with pytest.raises(BenchmarkCorpusError, match="observational fields"):
        SparBenchmarkCorpusReceipt.from_dict(mutated)


def test_loc_mismatch_and_missing_domain_fail_closed() -> None:
    with pytest.raises(BenchmarkCorpusError, match="unsupported synthetic profile"):
        generate_controlled_monolith("real_current_tree")
    with pytest.raises(BenchmarkCorpusError, match="below header minimum"):
        _module_source("state", 1, load_seeded_defects())
    with pytest.raises(BenchmarkCorpusError, match="positive integer"):
        _allocate_loc(0, REQUIRED_DOMAINS)
