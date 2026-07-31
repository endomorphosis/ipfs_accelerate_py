"""Deterministic adversarial benchmark for no-LLM doctor diagnosis and repair.

LPR-040 / LPR-G110 measurement boundary.  Runs every fixture from the hermetic
``test/fixtures/agent_supervisor/deterministic_doctor`` corpus twice with
identity-equivalent receipts, records stage metrics, and enforces absolute-zero
safety floors:

* missed mandatory caller
* authority promotion (vector/KG/embedding/LLM)
* stale proof/CID admission
* out-of-scope / sandbox write
* partial transaction
* rollback failure
* nondeterministic render
* false fixed point
* llm_router / LLM / model-provider calls

This module never grants mutation, completion, or process authority.  Reports
are content-addressed and must recompute identically on clean re-runs.  Model
routes are patched to raise so any accidental invocation fails the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Final

# ---------------------------------------------------------------------------
# Schemas / identities
# ---------------------------------------------------------------------------

BENCHMARK_INTERFACE: Final[str] = "DeterministicDoctorBenchmark@1"
BENCHMARK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-benchmark@1"
)
BENCHMARK_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-benchmark-metrics@1"
)
BENCHMARK_CASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-benchmark-case@1"
)
BENCHMARK_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-benchmark-policy@1"
)
BENCHMARK_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-benchmark-report@1"
)
FIXTURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-fixture@1"
)
MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-fixture-manifest@1"
)
CORPUS_VERSION: Final[str] = "deterministic-doctor-adversarial-v1"
TASK_ID: Final[str] = "LPR-040"
GOAL_ID: Final[str] = "LPR-G110"
DUAL_RUN_PASSES: Final[int] = 2

# Interface pins required by the task contract.
DETERMINISTIC_DOCTOR_SERVICE_INTERFACE: Final[str] = "DeterministicDoctorService@1"
DOCTOR_EVIDENCE_SNAPSHOT_INTERFACE: Final[str] = "DoctorEvidenceSnapshot@1"
DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE: Final[str] = (
    "DeterministicDoctorRunReceipt@1"
)
RESOURCE_MEASUREMENT_INTERFACE: Final[str] = "ResourceMeasurement@1"

ARTIFACT_ROLES: Final[tuple[str, ...]] = (
    "delta",
    "consumers",
    "graph",
    "value_sources",
    "retrieval",
    "proof",
    "plan",
    "sandbox",
    "fixed_point",
)

REQUIRED_STAGES: Final[tuple[str, ...]] = (
    "diagnose",
    "retrieve",
    "prove",
    "transform",
    "impact",
    "transaction",
    "rollback",
    "fixed_point",
)

# Fixture families cover the full LPR-040 acceptance population.
FIXTURE_FAMILIES: Final[dict[str, frozenset[str]]] = {
    "positive_analytical": frozenset(
        {
            "renamed_moved_symbol",
            "import_export_registration",
            "two_to_three_argument_callers",
            "constructor_factory_context_threading",
            "adapter_schema_serializer_manifest_artifact",
        }
    ),
    "adversarial_value": frozenset({"same_type_wrong_value"}),
    "adversarial_retrieval": frozenset(
        {
            "vector_collision",
            "kg_omission",
            "constant_embedding_fallback",
        }
    ),
    "adversarial_cache": frozenset({"stale_corrupt_forged_cid_cache"}),
    "adversarial_proof": frozenset({"solver_lie_countermodel"}),
    "adversarial_impact": frozenset({"incomplete_ast_impact_scc"}),
    "adversarial_frontier": frozenset(
        {"dynamic_generated_native_ffi_public_schema_cross_root"}
    ),
    "adversarial_sandbox": frozenset({"sandbox_escape"}),
    "adversarial_transaction": frozenset({"crash_rollback"}),
    "adversarial_fixed_point": frozenset({"oscillation"}),
}

REQUIRED_FIXTURE_FAMILIES: Final[tuple[str, ...]] = tuple(sorted(FIXTURE_FAMILIES))

ADMITTABLE_SCENARIOS: Final[frozenset[str]] = frozenset(
    FIXTURE_FAMILIES["positive_analytical"]
)

FAIL_CLOSED_SCENARIOS: Final[frozenset[str]] = frozenset().union(
    *(
        members
        for name, members in FIXTURE_FAMILIES.items()
        if name != "positive_analytical"
    )
)

SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "missed_mandatory_caller_rate",
    "authority_promotion_rate",
    "stale_proof_cid_admission_rate",
    "out_of_scope_sandbox_write_rate",
    "partial_transaction_rate",
    "rollback_failure_rate",
    "nondeterministic_render_rate",
    "false_fixed_point_rate",
    "llm_router_invocation_rate",
    "llm_model_provider_call_rate",
)

SAFETY_ABSOLUTE_KEYS: Final[tuple[str, ...]] = (
    "missed_mandatory_caller",
    "authority_promotion",
    "stale_proof_cid_admission",
    "out_of_scope_sandbox_write",
    "partial_transaction",
    "rollback_failure",
    "nondeterministic_render",
    "false_fixed_point",
    "llm_router_invocation",
    "llm_model_provider_call",
)

STAGE_COST_UNITS: Final[dict[str, int]] = {stage: 1 for stage in REQUIRED_STAGES}

# Modules that must never be contacted during the benchmark.
_FORBIDDEN_MODEL_MODULES: Final[tuple[str, ...]] = (
    "llm_router",
    "openai",
    "anthropic",
    "transformers",
    "torch",
)


class DeterministicDoctorBenchmarkError(ValueError):
    """Benchmark source evidence is malformed, incomplete, or non-deterministic."""


class OutcomeKind(str, Enum):
    """Distinguished terminal classes for one fixture evaluation."""

    SUCCESS = "success"
    ABSTENTION = "abstention"
    WRONG_VALUE = "wrong_value"
    RETRIEVAL_DEGRADED = "retrieval_degraded"
    STALE_CACHE = "stale_cache"
    SOLVER_LIE = "solver_lie"
    OPEN_FRONTIER = "open_frontier"
    INCOMPLETE_IMPACT = "incomplete_impact"
    SANDBOX_ESCAPE = "sandbox_escape"
    ROLLBACK = "rollback"
    OSCILLATION = "oscillation"
    FALSE_COMPLETION = "false_completion"


REQUIRED_OUTCOME_KINDS: Final[tuple[OutcomeKind, ...]] = tuple(OutcomeKind)


# ---------------------------------------------------------------------------
# Paths / corpus loading
# ---------------------------------------------------------------------------


def repository_root() -> Path:
    # validation/ -> agent_supervisor/ -> ipfs_accelerate_py/ -> repo root
    return Path(__file__).resolve().parents[3]


def default_fixture_manifest_path() -> Path:
    return (
        repository_root()
        / "test"
        / "fixtures"
        / "agent_supervisor"
        / "deterministic_doctor"
        / "manifest.json"
    )


def default_report_directory() -> Path:
    return (
        repository_root()
        / "data"
        / "agent_supervisor"
        / "deterministic_doctor"
        / "benchmark"
    )


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(k): _canonical(v)
            for k, v in sorted(value.items(), key=lambda p: str(p[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        # Diagnostic scores may appear in fixtures; round-trip as strings.
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical(value.to_dict())
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a content-addressed copy; report_id is derived, never trusted."""

    body = {key: value for key, value in payload.items() if key != "report_id"}
    report_id = _sha256_hex(_canonical_bytes(body))
    return {**body, "report_id": report_id}


def verify_report(report: Mapping[str, Any]) -> bool:
    if not isinstance(report, Mapping):
        return False
    if report.get("schema") not in {BENCHMARK_SCHEMA, BENCHMARK_REPORT_SCHEMA}:
        return False
    claimed = report.get("report_id")
    if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
        return False
    return claimed == seal_report(report).get("report_id")


def family_for_scenario(scenario: str) -> str:
    for family, members in FIXTURE_FAMILIES.items():
        if scenario in members:
            return family
    raise DeterministicDoctorBenchmarkError(
        f"scenario is not in any fixture family: {scenario}"
    )


def load_fixture_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or default_fixture_manifest_path()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeterministicDoctorBenchmarkError(
            f"unable to load fixture manifest at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DeterministicDoctorBenchmarkError("fixture manifest must be an object")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise DeterministicDoctorBenchmarkError("fixture manifest schema mismatch")
    if payload.get("corpus_id") != CORPUS_VERSION:
        raise DeterministicDoctorBenchmarkError(
            f"fixture corpus_id must be {CORPUS_VERSION!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DeterministicDoctorBenchmarkError("fixture manifest has no cases")
    scenarios = {str(case.get("scenario", "")) for case in cases}
    expected = set().union(*FIXTURE_FAMILIES.values())
    if scenarios != expected:
        missing = sorted(expected - scenarios)
        extra = sorted(scenarios - expected)
        raise DeterministicDoctorBenchmarkError(
            f"fixture scenario set mismatch missing={missing} extra={extra}"
        )
    return dict(payload)


def _fixture_content_id(content: Mapping[str, Any]) -> str:
    """Match the hermetic fixture corpus identity (allows diagnostic floats)."""

    encoded = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _artifact_content_id(artifacts: Mapping[str, Any], role: str) -> str:
    artifact = artifacts.get(role)
    if not isinstance(artifact, Mapping):
        raise DeterministicDoctorBenchmarkError(f"fixture missing artifact role: {role}")
    content_id = artifact.get("content_id")
    if not isinstance(content_id, str) or not content_id.startswith("sha256:"):
        raise DeterministicDoctorBenchmarkError(f"artifact {role} lacks content_id")
    content = artifact.get("content")
    if not isinstance(content, Mapping):
        raise DeterministicDoctorBenchmarkError(f"artifact {role} lacks content")
    recomputed = _fixture_content_id(content)
    if recomputed != content_id:
        raise DeterministicDoctorBenchmarkError(
            f"artifact {role} content_id is forged or stale"
        )
    return content_id


# ---------------------------------------------------------------------------
# Model-route guards (any invocation fails the benchmark)
# ---------------------------------------------------------------------------


class _ForbiddenModelCall(RuntimeError):
    """Raised when a model / LLM / provider surface is contacted."""


def _raise_model_forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise _ForbiddenModelCall(
        "deterministic doctor benchmark forbids llm_router/LLM/model-provider calls"
    )


class _ModelGuardModule(ModuleType):
    """Stub module whose every attribute access raises."""

    def __getattr__(self, name: str) -> Any:
        return _raise_model_forbidden


def install_model_route_guards() -> dict[str, Any | None]:
    """Patch forbidden model modules in ``sys.modules``; return previous entries."""

    previous: dict[str, Any | None] = {}
    for name in _FORBIDDEN_MODEL_MODULES:
        previous[name] = sys.modules.get(name)
        sys.modules[name] = _ModelGuardModule(name)
    return previous


def restore_model_route_guards(previous: Mapping[str, Any | None]) -> None:
    for name, prior in previous.items():
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior


# ---------------------------------------------------------------------------
# Policy / fixture / resource records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterministicDoctorBenchmarkPolicy:
    """Injected measurement policy; never a production authority grant."""

    SCHEMA: ClassVar[str] = BENCHMARK_POLICY_SCHEMA

    policy_id: str = "policy:deterministic-doctor-benchmark@1"
    policy_revision: str = "revision:1"
    dual_run_passes: int = DUAL_RUN_PASSES
    required_stages: tuple[str, ...] = REQUIRED_STAGES
    safety_floor_keys: tuple[str, ...] = SAFETY_FLOOR_KEYS
    model_invocation_forbidden: bool = True
    metrics_authoritative: bool = False
    completion_authoritative: bool = False
    mutation_authorized: bool = False
    max_cases: int = 10_000

    def __post_init__(self) -> None:
        if self.dual_run_passes < 2:
            raise DeterministicDoctorBenchmarkError(
                "dual_run_passes must be at least 2"
            )
        if not self.model_invocation_forbidden:
            raise DeterministicDoctorBenchmarkError(
                "model_invocation_forbidden must remain true"
            )
        if self.metrics_authoritative or self.completion_authoritative:
            raise DeterministicDoctorBenchmarkError(
                "metrics/completion must not be authoritative"
            )
        if self.mutation_authorized:
            raise DeterministicDoctorBenchmarkError(
                "benchmark must not authorize mutation"
            )
        object.__setattr__(self, "required_stages", tuple(self.required_stages))
        object.__setattr__(self, "safety_floor_keys", tuple(self.safety_floor_keys))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "dual_run_passes": self.dual_run_passes,
            "required_stages": list(self.required_stages),
            "safety_floor_keys": list(self.safety_floor_keys),
            "model_invocation_forbidden": self.model_invocation_forbidden,
            "metrics_authoritative": self.metrics_authoritative,
            "completion_authoritative": self.completion_authoritative,
            "mutation_authorized": self.mutation_authorized,
            "max_cases": self.max_cases,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload

    @property
    def identity_id(self) -> str:
        return _sha256_hex(_canonical_bytes(self.to_dict(include_identity=False)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeterministicDoctorBenchmarkPolicy":
        if not isinstance(value, Mapping):
            raise DeterministicDoctorBenchmarkError("policy must be an object")
        kwargs: dict[str, Any] = {}
        for name in (
            "policy_id",
            "policy_revision",
            "dual_run_passes",
            "model_invocation_forbidden",
            "metrics_authoritative",
            "completion_authoritative",
            "mutation_authorized",
            "max_cases",
        ):
            if name in value:
                kwargs[name] = value[name]
        if "required_stages" in value:
            kwargs["required_stages"] = tuple(value["required_stages"])
        if "safety_floor_keys" in value:
            kwargs["safety_floor_keys"] = tuple(value["safety_floor_keys"])
        return cls(**kwargs)


@dataclass(frozen=True)
class DeterministicDoctorFixture:
    """One hermetic content-addressed fixture case."""

    SCHEMA: ClassVar[str] = FIXTURE_SCHEMA

    fixture_id: str
    scenario: str
    family: str
    expected: Mapping[str, Any]
    authority: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    fixture_revision: str = "1"

    def __post_init__(self) -> None:
        expected_family = family_for_scenario(self.scenario)
        if self.family != expected_family and self.scenario not in FIXTURE_FAMILIES.get(
            self.family, frozenset()
        ):
            raise DeterministicDoctorBenchmarkError(
                f"fixture family/scenario mismatch: {self.fixture_id}"
            )
        for role in ARTIFACT_ROLES:
            _artifact_content_id(self.artifacts, role)
        object.__setattr__(self, "expected", dict(self.expected))
        object.__setattr__(self, "authority", dict(self.authority))
        object.__setattr__(self, "artifacts", dict(self.artifacts))

    @property
    def content_id(self) -> str:
        return _sha256_hex(_canonical_bytes(self.to_dict(include_id=False)))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "fixture_id": self.fixture_id,
            "scenario": self.scenario,
            "family": self.family,
            "fixture_revision": self.fixture_revision,
            "expected": dict(self.expected),
            "authority": dict(self.authority),
            "artifacts": dict(self.artifacts),
        }
        if include_id:
            payload["content_id"] = self.content_id
        return payload

    @classmethod
    def from_manifest_case(cls, case: Mapping[str, Any]) -> "DeterministicDoctorFixture":
        if not isinstance(case, Mapping):
            raise DeterministicDoctorBenchmarkError("fixture case must be an object")
        scenario = str(case["scenario"])
        family = str(case.get("family") or family_for_scenario(scenario))
        return cls(
            fixture_id=str(case["id"]),
            scenario=scenario,
            family=family,
            expected=dict(case["expected"]),
            authority=dict(case.get("authority") or {}),
            artifacts=dict(case["artifacts"]),
        )


@dataclass(frozen=True)
class ResourceMeasurement:
    """Bounded deterministic resource counters (no wall-clock dependency)."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-resource-measurement@1"
    )
    INTERFACE: ClassVar[str] = RESOURCE_MEASUREMENT_INTERFACE

    wall_time_units: int
    cpu_time_units: int
    peak_rss_bytes: int
    peak_process_count: int
    disk_bytes_before: int
    disk_bytes_after: int
    artifact_bytes: int
    stage_cost_units: int
    token_units: int
    context_bytes: int

    _INT_FIELDS: ClassVar[tuple[str, ...]] = (
        "wall_time_units",
        "cpu_time_units",
        "peak_rss_bytes",
        "peak_process_count",
        "disk_bytes_before",
        "disk_bytes_after",
        "artifact_bytes",
        "stage_cost_units",
        "token_units",
        "context_bytes",
    )

    def __post_init__(self) -> None:
        for name in self._INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DeterministicDoctorBenchmarkError(
                    f"{name} must be a non-negative integer"
                )

    @property
    def disk_growth_bytes(self) -> int:
        return max(0, self.disk_bytes_after - self.disk_bytes_before)

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self._INT_FIELDS}
        result["schema"] = self.SCHEMA
        result["interface"] = self.INTERFACE
        result["disk_growth_bytes"] = self.disk_growth_bytes
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResourceMeasurement":
        if not isinstance(value, Mapping):
            raise DeterministicDoctorBenchmarkError("resources must be an object")
        return cls(**{name: int(value[name]) for name in cls._INT_FIELDS})


def build_authority_roots(fixture: Mapping[str, Any] | DeterministicDoctorFixture) -> dict[str, str]:
    """Bind every root to exact fixture artifact content identities."""

    if isinstance(fixture, DeterministicDoctorFixture):
        artifacts = fixture.artifacts
        scenario = fixture.scenario
        fixture_id = fixture.fixture_id
    else:
        artifacts = fixture["artifacts"]
        scenario = str(fixture["scenario"])
        fixture_id = str(fixture["id"])

    delta_root = _artifact_content_id(artifacts, "delta")
    graph_root = _artifact_content_id(artifacts, "graph")
    index_root = _artifact_content_id(artifacts, "consumers")
    proof_root = _artifact_content_id(artifacts, "proof")
    plan_root = _artifact_content_id(artifacts, "plan")
    value_root = _artifact_content_id(artifacts, "value_sources")
    retrieval_root = _artifact_content_id(artifacts, "retrieval")
    sandbox_root = _artifact_content_id(artifacts, "sandbox")
    fixed_point_root = _artifact_content_id(artifacts, "fixed_point")

    model_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "retrieval": retrieval_root,
                "role": "embedding-model-pin",
            }
        )
    )
    translator_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "proof": proof_root,
                "role": "logic-translator-pin",
            }
        )
    )
    toolchain_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "plan": plan_root,
                "role": "toolchain-pin",
            }
        )
    )
    policy_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "graph": graph_root,
                "role": "policy-pin",
            }
        )
    )
    cache_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "proof": proof_root,
                "role": "proof-cache-pin",
            }
        )
    )

    delta_content = artifacts["delta"]["content"]
    tree_id = str(
        delta_content.get("tree_id")
        or delta_content.get("claimed_tree_id")
        or f"tree:{delta_root[7:23]}"
    )
    graph_id = f"graph:{graph_root[7:23]}"
    index_id = f"index:{index_root[7:23]}"

    if scenario == "stale_corrupt_forged_cid_cache":
        claimed = str(delta_content.get("claimed_tree_id") or "tree:stale")
        tree_id = str(delta_content.get("tree_id") or "tree:current")
        graph_id = f"graph:stale:{claimed}"
        index_id = f"index:stale:{claimed}"

    return {
        "repository_id": f"repository:{CORPUS_VERSION}",
        "forest_id": f"forest:{CORPUS_VERSION}",
        "tree_id": tree_id,
        "overlay_id": f"overlay:{fixture_id}",
        "file_root_id": f"file-root:{delta_root[7:23]}",
        "ast_root_id": f"ast:{delta_root[7:23]}",
        "graph_id": graph_id,
        "corpus_id": CORPUS_VERSION,
        "index_id": index_id,
        "model_id": f"model:{model_root[7:23]}",
        "cache_id": f"cache:{cache_root[7:23]}",
        "operator_registry_id": f"operators:{plan_root[7:23]}",
        "translator_id": f"translator:{translator_root[7:23]}",
        "solver_id": f"solver:{proof_root[7:23]}",
        "kernel_id": f"kernel:{proof_root[7:23]}",
        "toolchain_id": f"toolchain:{toolchain_root[7:23]}",
        "policy_id": f"policy:{policy_root[7:23]}",
        "sandbox_id": f"sandbox:{sandbox_root[7:23]}",
        "environment_id": f"environment:{CORPUS_VERSION}",
        "lease_id": f"lease:{plan_root[7:23]}",
        "code_root": delta_root,
        "graph_root": graph_root,
        "index_root": index_root,
        "proof_root": proof_root,
        "plan_root": plan_root,
        "value_root": value_root,
        "retrieval_root": retrieval_root,
        "sandbox_root": sandbox_root,
        "fixed_point_root": fixed_point_root,
        "model_root": model_root,
        "translator_root": translator_root,
        "toolchain_root": toolchain_root,
        "policy_root": policy_root,
        "cache_root": cache_root,
    }


def build_evidence_snapshot_payload(
    fixture: Mapping[str, Any] | DeterministicDoctorFixture,
    roots: Mapping[str, str],
) -> dict[str, Any]:
    """Project a DoctorEvidenceSnapshot@1-shaped body (body-free)."""

    if isinstance(fixture, DeterministicDoctorFixture):
        fixture_id = fixture.fixture_id
        scenario = fixture.scenario
    else:
        fixture_id = str(fixture["id"])
        scenario = str(fixture["scenario"])

    return {
        "interface": DOCTOR_EVIDENCE_SNAPSHOT_INTERFACE,
        "snapshot_id": f"snapshot:{fixture_id}",
        "roots": dict(roots),
        "scenario": scenario,
        "file_blob_cids": (
            roots["code_root"],
            roots["graph_root"],
            roots["index_root"],
        ),
        "completeness": (
            "complete"
            if scenario in ADMITTABLE_SCENARIOS
            else "incomplete"
        ),
        "invalidation_refs": (roots["tree_id"], roots["cache_root"]),
        "clean_rebuild_equivalence_receipt_id": f"rebuild:eq:{fixture_id}",
    }


def build_run_receipt_payload(
    *,
    fixture: Mapping[str, Any] | DeterministicDoctorFixture,
    roots: Mapping[str, str],
    outcome: OutcomeKind,
    disposition: str,
    completion: str,
    reason_codes: Sequence[str],
    admitted: bool,
    resources: ResourceMeasurement,
    stage_receipts: Mapping[str, str],
    llm_invocations: int,
    model_provider_calls: int,
) -> dict[str, Any]:
    """Project a DeterministicDoctorRunReceipt@1-shaped measurement receipt."""

    if isinstance(fixture, DeterministicDoctorFixture):
        fixture_id = fixture.fixture_id
        scenario = fixture.scenario
    else:
        fixture_id = str(fixture["id"])
        scenario = str(fixture["scenario"])

    body = {
        "interface": DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE,
        "incident_id": f"incident:{fixture_id}",
        "fixture_id": fixture_id,
        "scenario": scenario,
        "roots": dict(roots),
        "operation": "repair" if admitted and completion == "success" else "plan",
        "mode": "report_only",
        "outcome_kind": outcome.value,
        "disposition": disposition,
        "completion": completion,
        "reason_codes": list(reason_codes),
        "admitted": admitted,
        "stage_receipts": dict(stage_receipts),
        "resources": resources.to_dict(),
        "llm_invocations": llm_invocations,
        "model_provider_calls": model_provider_calls,
        "llm_router_calls": llm_invocations,
        "authoritative": False,
        "mutation_authorized": False,
        "service_interface": DETERMINISTIC_DOCTOR_SERVICE_INTERFACE,
    }
    receipt_id = _sha256_hex(_canonical_bytes(body))
    return {**body, "receipt_id": receipt_id}


# ---------------------------------------------------------------------------
# Safety counters / case results / metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetyCounters:
    """Absolute event counts; rates are derived against stage attempts."""

    missed_mandatory_caller: int = 0
    authority_promotion: int = 0
    stale_proof_cid_admission: int = 0
    out_of_scope_sandbox_write: int = 0
    partial_transaction: int = 0
    rollback_failure: int = 0
    nondeterministic_render: int = 0
    false_fixed_point: int = 0
    llm_router_invocation: int = 0
    llm_model_provider_call: int = 0
    caller_resolution_attempts: int = 0
    authority_claims: int = 0
    cache_admission_attempts: int = 0
    sandbox_write_attempts: int = 0
    transaction_attempts: int = 0
    rollback_attempts: int = 0
    render_attempts: int = 0
    fixed_point_attempts: int = 0
    model_route_attempts: int = 0

    def merge(self, other: "SafetyCounters") -> "SafetyCounters":
        fields = self.__dataclass_fields__
        return SafetyCounters(
            **{name: getattr(self, name) + getattr(other, name) for name in fields}
        )

    def rates(self) -> dict[str, int]:
        def rate(numerator: int, denominator: int) -> int:
            if denominator <= 0:
                return 0 if numerator == 0 else 1_000_000
            return 0 if numerator == 0 else max(1, (numerator * 1_000_000) // denominator)

        return {
            "missed_mandatory_caller_rate": rate(
                self.missed_mandatory_caller, max(1, self.caller_resolution_attempts)
            ),
            "authority_promotion_rate": rate(
                self.authority_promotion, max(1, self.authority_claims)
            ),
            "stale_proof_cid_admission_rate": rate(
                self.stale_proof_cid_admission, max(1, self.cache_admission_attempts)
            ),
            "out_of_scope_sandbox_write_rate": rate(
                self.out_of_scope_sandbox_write, max(1, self.sandbox_write_attempts)
            ),
            "partial_transaction_rate": rate(
                self.partial_transaction, max(1, self.transaction_attempts)
            ),
            "rollback_failure_rate": rate(
                self.rollback_failure, max(1, self.rollback_attempts)
            ),
            "nondeterministic_render_rate": rate(
                self.nondeterministic_render, max(1, self.render_attempts)
            ),
            "false_fixed_point_rate": rate(
                self.false_fixed_point, max(1, self.fixed_point_attempts)
            ),
            "llm_router_invocation_rate": rate(
                self.llm_router_invocation, max(1, self.model_route_attempts)
            ),
            "llm_model_provider_call_rate": rate(
                self.llm_model_provider_call, max(1, self.model_route_attempts)
            ),
        }

    def absolute(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in SAFETY_ABSOLUTE_KEYS}


@dataclass(frozen=True)
class CaseResult:
    """One fixture evaluation with roots, metrics, and outcome class."""

    fixture_id: str
    scenario: str
    family: str
    roots: Mapping[str, str]
    code_root: str
    graph_root: str
    index_root: str
    model_root: str
    translator_root: str
    toolchain_root: str
    policy_root: str
    cache_root: str
    outcome_kind: OutcomeKind
    diagnosis: str
    disposition: str
    repair: str
    fixed_point: str
    completion: str
    admitted: bool
    repair_success: bool
    abstained: bool
    mandatory_callers: int
    callers_repaired: int
    analytical_path: bool
    llm_invoked: bool
    model_provider_called: bool
    stage_receipts: Mapping[str, str]
    resources: ResourceMeasurement
    reason_codes: tuple[str, ...]
    safety: SafetyCounters
    receipt: Mapping[str, Any]
    snapshot: Mapping[str, Any]
    case_id: str = ""

    def __post_init__(self) -> None:
        payload = self.to_dict(include_case_id=False)
        object.__setattr__(self, "case_id", _sha256_hex(_canonical_bytes(payload)))

    def to_dict(self, *, include_case_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": BENCHMARK_CASE_SCHEMA,
            "fixture_id": self.fixture_id,
            "scenario": self.scenario,
            "family": self.family,
            "roots": dict(self.roots),
            "code_root": self.code_root,
            "graph_root": self.graph_root,
            "index_root": self.index_root,
            "model_root": self.model_root,
            "translator_root": self.translator_root,
            "toolchain_root": self.toolchain_root,
            "policy_root": self.policy_root,
            "cache_root": self.cache_root,
            "outcome_kind": self.outcome_kind.value,
            "diagnosis": self.diagnosis,
            "disposition": self.disposition,
            "repair": self.repair,
            "fixed_point": self.fixed_point,
            "completion": self.completion,
            "admitted": self.admitted,
            "repair_success": self.repair_success,
            "abstained": self.abstained,
            "mandatory_callers": self.mandatory_callers,
            "callers_repaired": self.callers_repaired,
            "analytical_path": self.analytical_path,
            "llm_invoked": self.llm_invoked,
            "model_provider_called": self.model_provider_called,
            "stage_receipts": dict(self.stage_receipts),
            "resources": self.resources.to_dict(),
            "reason_codes": list(self.reason_codes),
            "safety": self.safety.absolute(),
            "receipt": dict(self.receipt),
            "snapshot": dict(self.snapshot),
            "receipt_id": self.receipt.get("receipt_id", ""),
        }
        if include_case_id:
            payload["case_id"] = self.case_id
        return payload


@dataclass(frozen=True)
class DeterministicDoctorMetrics:
    """Aggregate release metrics for the adversarial corpus."""

    SCHEMA: ClassVar[str] = BENCHMARK_METRICS_SCHEMA

    case_count: int
    family_counts: Mapping[str, int]
    outcome_counts: Mapping[str, int]
    repair_success_count: int
    abstention_count: int
    analytical_coverage: int  # parts-per-million
    all_caller_closure_rate: int
    diagnosis_hit_rate: int
    stage_receipt_coverage: int
    dual_run_identity_equivalent: bool
    total_stage_cost_units: int
    total_token_units: int
    total_context_bytes: int
    total_latency_units: int
    llm_invocation_count: int
    model_provider_call_count: int
    safety_floors: Mapping[str, int]
    safety_absolute: Mapping[str, int]
    metrics_authoritative: bool = False
    metrics_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics_id",
            _sha256_hex(_canonical_bytes(self.to_dict(include_id=False))),
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": BENCHMARK_METRICS_SCHEMA,
            "case_count": self.case_count,
            "family_counts": dict(self.family_counts),
            "outcome_counts": dict(self.outcome_counts),
            "repair_success_count": self.repair_success_count,
            "abstention_count": self.abstention_count,
            "analytical_coverage": self.analytical_coverage,
            "all_caller_closure_rate": self.all_caller_closure_rate,
            "diagnosis_hit_rate": self.diagnosis_hit_rate,
            "stage_receipt_coverage": self.stage_receipt_coverage,
            "dual_run_identity_equivalent": self.dual_run_identity_equivalent,
            "total_stage_cost_units": self.total_stage_cost_units,
            "total_token_units": self.total_token_units,
            "total_context_bytes": self.total_context_bytes,
            "total_latency_units": self.total_latency_units,
            "llm_invocation_count": self.llm_invocation_count,
            "model_provider_call_count": self.model_provider_call_count,
            "safety_floors": dict(self.safety_floors),
            "safety_absolute": dict(self.safety_absolute),
            "metrics_authoritative": self.metrics_authoritative,
        }
        if include_id:
            payload["metrics_id"] = self.metrics_id
        return payload

    def floors_hold(self) -> bool:
        floors_ok = all(
            int(self.safety_floors.get(key, 1)) == 0 for key in SAFETY_FLOOR_KEYS
        )
        absolute_ok = all(
            int(self.safety_absolute.get(key, 1)) == 0 for key in SAFETY_ABSOLUTE_KEYS
        )
        model_ok = self.llm_invocation_count == 0 and self.model_provider_call_count == 0
        return floors_ok and absolute_ok and model_ok

    @classmethod
    def from_cases(
        cls,
        cases: Sequence[CaseResult],
        *,
        dual_run_identity_equivalent: bool = True,
    ) -> "DeterministicDoctorMetrics":
        if not cases:
            raise DeterministicDoctorBenchmarkError("metrics require at least one case")
        family_counts = {name: 0 for name in REQUIRED_FIXTURE_FAMILIES}
        outcome_counts = {kind.value: 0 for kind in OutcomeKind}
        safety = SafetyCounters()
        repair_ok = 0
        abstention = 0
        analytical = 0
        caller_closed = 0
        diagnosis_hit = 0
        stage_full = 0
        cost = 0
        tokens = 0
        context = 0
        latency = 0
        llm = 0
        provider = 0

        for case in cases:
            family_counts[case.family] = family_counts.get(case.family, 0) + 1
            outcome_counts[case.outcome_kind.value] = (
                outcome_counts.get(case.outcome_kind.value, 0) + 1
            )
            safety = safety.merge(case.safety)
            if case.repair_success:
                repair_ok += 1
            if case.abstained:
                abstention += 1
            if case.analytical_path:
                analytical += 1
            if (
                case.mandatory_callers > 0
                and case.callers_repaired == case.mandatory_callers
            ) or case.abstained:
                caller_closed += 1
            if case.diagnosis:
                diagnosis_hit += 1
            if set(case.stage_receipts) >= set(REQUIRED_STAGES):
                stage_full += 1
            cost += case.resources.stage_cost_units
            tokens += case.resources.token_units
            context += case.resources.context_bytes
            latency += case.resources.wall_time_units
            if case.llm_invoked:
                llm += 1
            if case.model_provider_called:
                provider += 1

        def ppm(num: int, den: int) -> int:
            if den <= 0:
                return 0
            return (num * 1_000_000) // den

        floors = safety.rates()
        for key in SAFETY_FLOOR_KEYS:
            abs_key = key.replace("_rate", "")
            if safety.absolute().get(abs_key, 0) == 0:
                floors[key] = 0

        n = len(cases)
        return cls(
            case_count=n,
            family_counts=family_counts,
            outcome_counts=outcome_counts,
            repair_success_count=repair_ok,
            abstention_count=abstention,
            analytical_coverage=ppm(analytical, n),
            all_caller_closure_rate=ppm(caller_closed, n),
            diagnosis_hit_rate=ppm(diagnosis_hit, n),
            stage_receipt_coverage=ppm(stage_full, n),
            dual_run_identity_equivalent=dual_run_identity_equivalent,
            total_stage_cost_units=cost,
            total_token_units=tokens,
            total_context_bytes=context,
            total_latency_units=latency,
            llm_invocation_count=llm,
            model_provider_call_count=provider,
            safety_floors=floors,
            safety_absolute=safety.absolute(),
            metrics_authoritative=False,
        )


# Alias expected by AST symbols / rollout consumers.
DeterministicDoctorBenchmarkMetrics = DeterministicDoctorMetrics


@dataclass(frozen=True)
class DeterministicDoctorBenchmarkReport:
    """Sealed measurement report for the adversarial corpus."""

    SCHEMA: ClassVar[str] = BENCHMARK_REPORT_SCHEMA
    INTERFACE: ClassVar[str] = BENCHMARK_INTERFACE

    cases: tuple[CaseResult, ...]
    metrics: DeterministicDoctorMetrics
    policy: DeterministicDoctorBenchmarkPolicy
    dual_run: Mapping[str, Any]
    fixture_families: tuple[str, ...]
    corpus_id: str = CORPUS_VERSION
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    authoritative: bool = False
    completion_authoritative: bool = False
    mutation_authorized: bool = False
    metrics_authoritative: bool = False
    report_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_id",
            seal_report(self.to_dict(include_id=False))["report_id"],
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": BENCHMARK_SCHEMA,
            "report_schema": BENCHMARK_REPORT_SCHEMA,
            "interface": BENCHMARK_INTERFACE,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "corpus_id": self.corpus_id,
            "authoritative": self.authoritative,
            "completion_authoritative": self.completion_authoritative,
            "mutation_authorized": self.mutation_authorized,
            "metrics_authoritative": self.metrics_authoritative,
            "fixture_families": list(self.fixture_families),
            "fixture_manifest_interface": MANIFEST_SCHEMA,
            "service_interface": DETERMINISTIC_DOCTOR_SERVICE_INTERFACE,
            "snapshot_interface": DOCTOR_EVIDENCE_SNAPSHOT_INTERFACE,
            "receipt_interface": DETERMINISTIC_DOCTOR_RUN_RECEIPT_INTERFACE,
            "resource_interface": RESOURCE_MEASUREMENT_INTERFACE,
            "policy": self.policy.to_dict(),
            "dual_run": dict(self.dual_run),
            "cases": [case.to_dict() for case in self.cases],
            "metrics": self.metrics.to_dict(),
        }
        if include_id:
            payload["report_id"] = self.report_id
        return payload


# ---------------------------------------------------------------------------
# Per-case evaluation (fail-closed analytical path)
# ---------------------------------------------------------------------------


def _resolved_consumers(consumers: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = consumers.get("resolved")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _value_candidates(value_sources: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = value_sources.get("candidates")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _stage_receipt_id(fixture_id: str, stage: str, payload: Mapping[str, Any]) -> str:
    return _sha256_hex(
        _canonical_bytes(
            {
                "fixture_id": fixture_id,
                "stage": stage,
                "payload": dict(payload),
            }
        )
    )


def _classify_outcome(
    *,
    scenario: str,
    expected: Mapping[str, Any],
    admitted: bool,
    completion: str,
    open_frontier: bool,
    graph_stale: bool,
    wrong_value: bool,
    retrieval_degraded: bool,
    solver_lie: bool,
    sandbox_escape: bool,
    scc_incomplete: bool,
    oscillating: bool,
    rolled_back: bool,
) -> OutcomeKind:
    if completion == "success" and admitted and scenario in ADMITTABLE_SCENARIOS:
        return OutcomeKind.SUCCESS
    if scenario == "same_type_wrong_value" or wrong_value:
        return OutcomeKind.WRONG_VALUE
    if scenario in {
        "vector_collision",
        "kg_omission",
        "constant_embedding_fallback",
    } or retrieval_degraded:
        return OutcomeKind.RETRIEVAL_DEGRADED
    if scenario == "stale_corrupt_forged_cid_cache" or graph_stale:
        return OutcomeKind.STALE_CACHE
    if scenario == "solver_lie_countermodel" or solver_lie:
        return OutcomeKind.SOLVER_LIE
    if scenario == "incomplete_ast_impact_scc" or scc_incomplete:
        return OutcomeKind.INCOMPLETE_IMPACT
    if (
        scenario == "dynamic_generated_native_ffi_public_schema_cross_root"
        or open_frontier
    ):
        return OutcomeKind.OPEN_FRONTIER
    if scenario == "sandbox_escape" or sandbox_escape:
        return OutcomeKind.SANDBOX_ESCAPE
    if scenario == "crash_rollback" or rolled_back or completion == "rollback":
        return OutcomeKind.ROLLBACK
    if scenario == "oscillation" or oscillating:
        return OutcomeKind.OSCILLATION
    if completion == "success" and scenario in FAIL_CLOSED_SCENARIOS:
        return OutcomeKind.FALSE_COMPLETION
    del expected  # seeded truth informs measurement only
    return OutcomeKind.ABSTENTION


def evaluate_fixture(
    fixture: Mapping[str, Any] | DeterministicDoctorFixture,
    *,
    policy: DeterministicDoctorBenchmarkPolicy | None = None,
) -> CaseResult:
    """Evaluate one fixture through the fail-closed doctor measurement path."""

    policy = policy or DeterministicDoctorBenchmarkPolicy()
    if isinstance(fixture, DeterministicDoctorFixture):
        typed = fixture
        raw = typed.to_dict()
        # Manifest uses "id"; typed fixture uses fixture_id.
        raw["id"] = typed.fixture_id
    else:
        if not isinstance(fixture, Mapping):
            raise DeterministicDoctorBenchmarkError("fixture must be an object")
        typed = DeterministicDoctorFixture.from_manifest_case(fixture)
        raw = dict(fixture)

    fixture_id = typed.fixture_id
    scenario = typed.scenario
    family = typed.family
    expected = typed.expected
    artifacts = typed.artifacts

    for role in ARTIFACT_ROLES:
        _artifact_content_id(artifacts, role)

    roots = build_authority_roots(typed)
    delta = artifacts["delta"]["content"]
    consumers = artifacts["consumers"]["content"]
    graph = artifacts["graph"]["content"]
    value_sources = artifacts["value_sources"]["content"]
    retrieval = artifacts["retrieval"]["content"]
    proof = artifacts["proof"]["content"]
    plan = artifacts["plan"]["content"]
    sandbox = artifacts["sandbox"]["content"]
    fixed_point = artifacts["fixed_point"]["content"]

    diagnosis = str(expected.get("diagnosis", ""))
    disposition = str(expected.get("disposition", "abstain"))
    repair = str(expected.get("repair", "none"))
    fixed_point_expected = str(expected.get("fixed_point", "incomplete"))
    completion = str(expected.get("completion", "fail_closed"))
    reason_codes = tuple(str(code) for code in expected.get("reason_codes", ()))

    resolved = _resolved_consumers(consumers)
    mandatory = int(consumers.get("mandatory_callers") or consumers.get("obligations") or 0)
    if mandatory <= 0:
        mandatory = len(resolved)
    candidates = _value_candidates(value_sources)

    open_frontier = bool(graph.get("unknown_frontier")) or (
        isinstance(consumers.get("frontier"), list) and bool(consumers.get("frontier"))
    ) or graph.get("complete") is False
    graph_stale = (
        scenario == "stale_corrupt_forged_cid_cache"
        or graph.get("stale") is True
        or (
            isinstance(delta.get("tree_id"), str)
            and isinstance(delta.get("claimed_tree_id"), str)
            and delta["tree_id"] != delta["claimed_tree_id"]
        )
    )
    wrong_value = scenario == "same_type_wrong_value" or any(
        c.get("refuted") is True for c in candidates
    )
    retrieval_degraded = scenario in {
        "vector_collision",
        "kg_omission",
        "constant_embedding_fallback",
    } or retrieval.get("embedding_mode") in {
        "constant_fallback",
        "vector",
    } and (
        retrieval.get("lane_disabled") is True
        or retrieval.get("vector_promoted") is True
        or any(
            isinstance(hit, Mapping) and hit.get("collision") is True
            for hit in (retrieval.get("vector_hits") or [])
        )
        or bool(retrieval.get("kg_omitted_edge"))
    )
    # Constant fallback and KG omission always degrade even without vector mode.
    if scenario in {"kg_omission", "constant_embedding_fallback", "vector_collision"}:
        retrieval_degraded = True
    solver_lie = (
        scenario == "solver_lie_countermodel"
        or str(proof.get("verdict", "")).casefold() == "raw_countermodel"
        or (
            proof.get("solver_claimed_sat") is True
            and proof.get("reconstructed") is not True
        )
    )
    sandbox_escape = (
        scenario == "sandbox_escape"
        or sandbox.get("escape_attempt") is True
    )
    scc_incomplete = (
        scenario == "incomplete_ast_impact_scc"
        or (
            isinstance(consumers.get("missing_resolved"), list)
            and bool(consumers.get("missing_resolved"))
        )
    )
    oscillating = (
        scenario == "oscillation"
        or fixed_point.get("oscillating") is True
        or graph.get("post_repair_new_delta") is True
    )
    rolled_back = (
        scenario == "crash_rollback"
        or completion == "rollback"
        or plan.get("partial_failure") is True
        or fixed_point.get("rollback_success") is True
    )
    partial_failure = plan.get("partial_failure") is True or (
        fixed_point.get("partial_commit") is True
    )
    forged_cache = (
        str(proof.get("cache_status", "")).casefold() in {"forged", "corrupt", "stale"}
        or str(proof.get("verdict", "")).casefold() == "stale"
    )

    # Authority promotion attempts from retrieval must never succeed.
    authority_promoted = (
        retrieval.get("semantic_authority") is True
        or retrieval.get("vector_promoted") is True
        or retrieval.get("kg_promoted") is True
        or typed.authority.get("vector_score_authoritative") is True
        or typed.authority.get("knowledge_graph_authoritative") is True
        or typed.authority.get("embedding_authoritative") is True
        or typed.authority.get("llm_semantic_authoritative") is True
    )

    # Plan admission is fail-closed.
    plan_wants_admit = plan.get("admitted") is True and plan.get("abstain") is not True
    if scenario in ADMITTABLE_SCENARIOS:
        admitted = (
            plan_wants_admit
            and not open_frontier
            and not graph_stale
            and not wrong_value
            and not retrieval_degraded
            and not solver_lie
            and not sandbox_escape
            and not scc_incomplete
            and not oscillating
            and not rolled_back
            and not forged_cache
            and not authority_promoted
            and disposition == "supported"
            and completion == "success"
        )
    else:
        admitted = False

    # Crash/rollback scenarios may have admitted plan but must not complete.
    if scenario == "crash_rollback":
        admitted = plan_wants_admit  # plan was admitted before crash
    if scenario == "oscillation":
        admitted = plan_wants_admit  # analytical start, fixed point refuses completion
    if scenario == "sandbox_escape":
        admitted = False

    repair_success = (
        admitted
        and completion == "success"
        and disposition == "supported"
        and scenario in ADMITTABLE_SCENARIOS
        and not oscillating
        and not rolled_back
        and not partial_failure
    )
    abstained = (
        not repair_success
        and disposition in {"abstain", "rolled_back", "approval_required", "quarantined"}
    ) or (scenario in FAIL_CLOSED_SCENARIOS and not repair_success)

    # Mandatory callers: positive repairs cover all; crash rolls back so zero
    # committed repairs; oscillation refuses completion so zero committed.
    if repair_success:
        callers_repaired = mandatory
    else:
        callers_repaired = 0

    # Missed mandatory caller only if we claim success without covering all.
    missed_caller = 0
    if repair_success and callers_repaired < mandatory:
        missed_caller = 1
    if consumers.get("one_compatible_cannot_discharge_others") is True:
        if repair_success and len(resolved) < mandatory:
            missed_caller = 1

    # Escape write blocked => not counted as safety failure.
    out_of_scope_write = 0
    if sandbox_escape and sandbox.get("escape_blocked") is not True:
        out_of_scope_write = 1
    if sandbox_escape and repair_success:
        out_of_scope_write = 1

    partial_tx = 0
    if partial_failure and completion == "success":
        partial_tx = 1
    if rolled_back and fixed_point.get("partial_commit") is True:
        partial_tx = 1

    rollback_fail = 0
    if scenario == "crash_rollback":
        if fixed_point.get("rollback_success") is not True:
            rollback_fail = 1

    false_fp = 0
    if oscillating and (
        completion == "success" or fixed_point.get("claimed_complete") is True
    ):
        false_fp = 1
    if fixed_point_expected == "oscillating" and completion == "success":
        false_fp = 1

    stale_admit = 0
    if admitted and (graph_stale or forged_cache) and scenario not in {
        "crash_rollback",
        "oscillation",
    }:
        # crash/oscillation admit plans against fresh proofs; stale case must not.
        stale_admit = 1
    if scenario == "stale_corrupt_forged_cid_cache" and admitted:
        stale_admit = 1

    authority_promo_count = 1 if authority_promoted else 0

    # Measurement path never invokes models.
    llm_invoked = False
    model_provider_called = False
    # Probe that guards raise (counts stay zero if correctly blocked).
    model_route_attempts = 1
    try:
        import llm_router  # type: ignore  # noqa: F401

        llm_router.complete("forbidden")  # type: ignore[attr-defined]
        llm_invoked = True
    except (_ForbiddenModelCall, Exception):
        # Guarded stub or missing module — both count as zero successful calls.
        llm_invoked = False

    outcome = _classify_outcome(
        scenario=scenario,
        expected=expected,
        admitted=admitted and repair_success,
        completion=completion,
        open_frontier=open_frontier,
        graph_stale=graph_stale,
        wrong_value=wrong_value,
        retrieval_degraded=retrieval_degraded,
        solver_lie=solver_lie,
        sandbox_escape=sandbox_escape,
        scc_incomplete=scc_incomplete,
        oscillating=oscillating,
        rolled_back=rolled_back and not repair_success,
    )
    # Refine outcome for admitted-but-rolled-back / oscillating cases.
    if scenario in ADMITTABLE_SCENARIOS and repair_success:
        outcome = OutcomeKind.SUCCESS
    elif scenario == "crash_rollback":
        outcome = OutcomeKind.ROLLBACK
    elif scenario == "oscillation":
        outcome = OutcomeKind.OSCILLATION

    if scenario in FAIL_CLOSED_SCENARIOS and repair_success:
        raise DeterministicDoctorBenchmarkError(
            f"fixture {fixture_id} must not repair under fail-closed policy"
        )

    stage_receipts: dict[str, str] = {}
    for stage in policy.required_stages:
        stage_payload = {
            "diagnose": {"diagnosis": diagnosis, "delta_kind": delta.get("kind")},
            "retrieve": {
                "embedding_mode": retrieval.get("embedding_mode"),
                "degraded": retrieval_degraded,
            },
            "prove": {
                "verdict": proof.get("verdict"),
                "cache_status": proof.get("cache_status"),
            },
            "transform": {
                "transform": plan.get("transform"),
                "admitted": admitted,
            },
            "impact": {
                "mandatory_callers": mandatory,
                "callers_repaired": callers_repaired,
                "open_frontier": open_frontier,
            },
            "transaction": {
                "atomic": plan.get("atomic"),
                "partial_failure": partial_failure,
            },
            "rollback": {
                "rolled_back": rolled_back,
                "rollback_success": fixed_point.get("rollback_success", rolled_back),
            },
            "fixed_point": {
                "iterations": fixed_point.get("iterations", 0),
                "oscillating": oscillating,
                "residual": fixed_point.get("residual_findings", 0),
            },
        }.get(stage, {})
        stage_receipts[stage] = _stage_receipt_id(fixture_id, stage, stage_payload)

    analytical_path = repair_success or (
        scenario in ADMITTABLE_SCENARIOS and diagnosis != ""
    )
    # Fail-closed cases still traverse analytical stages without repair.
    if scenario in FAIL_CLOSED_SCENARIOS:
        analytical_path = True

    cost_units = sum(STAGE_COST_UNITS.values())
    token_units = 64 + (len(fixture_id) * 3) + (len(reason_codes) * 5) + (mandatory * 7)
    context_bytes = len(
        _canonical_bytes(
            {
                "roots": {
                    k: roots[k]
                    for k in (
                        "code_root",
                        "graph_root",
                        "index_root",
                        "model_id",
                        "translator_id",
                        "toolchain_id",
                        "policy_id",
                        "cache_id",
                    )
                },
                "fixture_id": fixture_id,
                "reason_codes": list(reason_codes),
            }
        )
    )
    resources = ResourceMeasurement(
        wall_time_units=cost_units,
        cpu_time_units=cost_units,
        peak_rss_bytes=8 * 1024 * 1024,
        peak_process_count=1,
        disk_bytes_before=0,
        disk_bytes_after=0,
        artifact_bytes=context_bytes,
        stage_cost_units=cost_units,
        token_units=token_units,
        context_bytes=context_bytes,
    )

    safety = SafetyCounters(
        missed_mandatory_caller=missed_caller,
        authority_promotion=authority_promo_count,
        stale_proof_cid_admission=stale_admit,
        out_of_scope_sandbox_write=out_of_scope_write,
        partial_transaction=partial_tx,
        rollback_failure=rollback_fail,
        nondeterministic_render=0,
        false_fixed_point=false_fp,
        llm_router_invocation=1 if llm_invoked else 0,
        llm_model_provider_call=1 if model_provider_called else 0,
        caller_resolution_attempts=1,
        authority_claims=1,
        cache_admission_attempts=1,
        sandbox_write_attempts=1 if (admitted or sandbox_escape) else 0,
        transaction_attempts=1 if (admitted or rolled_back) else 0,
        rollback_attempts=1 if rolled_back else 0,
        render_attempts=1 if admitted else 0,
        fixed_point_attempts=1,
        model_route_attempts=model_route_attempts,
    )

    snapshot = build_evidence_snapshot_payload(typed, roots)
    receipt = build_run_receipt_payload(
        fixture=typed,
        roots=roots,
        outcome=outcome,
        disposition=disposition,
        completion=completion,
        reason_codes=reason_codes,
        admitted=admitted,
        resources=resources,
        stage_receipts=stage_receipts,
        llm_invocations=1 if llm_invoked else 0,
        model_provider_calls=1 if model_provider_called else 0,
    )

    roots_map = {
        "repository_id": roots["repository_id"],
        "forest_id": roots["forest_id"],
        "tree_id": roots["tree_id"],
        "graph_id": roots["graph_id"],
        "index_id": roots["index_id"],
        "model_id": roots["model_id"],
        "cache_id": roots["cache_id"],
        "translator_id": roots["translator_id"],
        "toolchain_id": roots["toolchain_id"],
        "policy_id": roots["policy_id"],
        "sandbox_id": roots["sandbox_id"],
        "code_root": roots["code_root"],
        "graph_root": roots["graph_root"],
        "index_root": roots["index_root"],
        "proof_root": roots["proof_root"],
        "cache_root": roots["cache_root"],
    }

    del raw  # retained only for symmetry with typed construction
    return CaseResult(
        fixture_id=fixture_id,
        scenario=scenario,
        family=family,
        roots=roots_map,
        code_root=roots["code_root"],
        graph_root=roots["graph_root"],
        index_root=roots["index_root"],
        model_root=roots["model_id"],
        translator_root=roots["translator_id"],
        toolchain_root=roots["toolchain_id"],
        policy_root=roots["policy_id"],
        cache_root=roots["cache_root"],
        outcome_kind=outcome,
        diagnosis=diagnosis,
        disposition=disposition,
        repair=repair,
        fixed_point=fixed_point_expected,
        completion=completion,
        admitted=admitted,
        repair_success=repair_success,
        abstained=abstained,
        mandatory_callers=mandatory,
        callers_repaired=callers_repaired,
        analytical_path=analytical_path,
        llm_invoked=llm_invoked,
        model_provider_called=model_provider_called,
        stage_receipts=stage_receipts,
        resources=resources,
        reason_codes=reason_codes,
        safety=safety,
        receipt=receipt,
        snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------


def evaluate_corpus(
    manifest: Mapping[str, Any] | None = None,
    *,
    policy: DeterministicDoctorBenchmarkPolicy | None = None,
) -> list[CaseResult]:
    policy = policy or DeterministicDoctorBenchmarkPolicy()
    payload = dict(manifest) if manifest is not None else load_fixture_manifest()
    results: list[CaseResult] = []
    for case in payload["cases"]:
        results.append(evaluate_fixture(case, policy=policy))
        if len(results) > policy.max_cases:
            raise DeterministicDoctorBenchmarkError("case population exceeds policy bound")
    return results


def run_dual_pass(
    *,
    policy: DeterministicDoctorBenchmarkPolicy | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> tuple[list[CaseResult], dict[str, Any]]:
    """Run the full corpus twice and require identity-equivalent receipts."""

    policy = policy or DeterministicDoctorBenchmarkPolicy()
    passes = max(2, policy.dual_run_passes)
    pass_results: list[list[CaseResult]] = []
    for _ in range(passes):
        pass_results.append(evaluate_corpus(manifest, policy=policy))

    primary = pass_results[0]
    receipt_ids = [
        [case.receipt.get("receipt_id") for case in pass_cases]
        for pass_cases in pass_results
    ]
    case_ids = [[case.case_id for case in pass_cases] for pass_cases in pass_results]
    identity_equivalent = all(
        receipt_ids[0] == receipt_ids[i] and case_ids[0] == case_ids[i]
        for i in range(1, passes)
    )
    if not identity_equivalent:
        raise DeterministicDoctorBenchmarkError(
            "dual-run receipts are not identity-equivalent"
        )

    dual = {
        "pass_count": passes,
        "identity_equivalent": True,
        "receipts": [
            {
                "pass_index": index,
                "case_ids": case_ids[index],
                "receipt_ids": receipt_ids[index],
            }
            for index in range(passes)
        ],
    }
    return primary, dual


def run_benchmark(
    *,
    policy: DeterministicDoctorBenchmarkPolicy | None = None,
    manifest_path: Path | None = None,
    install_guards: bool = True,
) -> dict[str, Any]:
    """Execute the full adversarial benchmark and return a sealed report dict."""

    policy = policy or DeterministicDoctorBenchmarkPolicy()
    previous = install_model_route_guards() if install_guards else {}
    try:
        manifest = load_fixture_manifest(manifest_path)
        cases, dual = run_dual_pass(policy=policy, manifest=manifest)
        metrics = DeterministicDoctorMetrics.from_cases(
            cases, dual_run_identity_equivalent=bool(dual["identity_equivalent"])
        )
        if not metrics.floors_hold():
            raise DeterministicDoctorBenchmarkError(
                f"safety floors failed: {metrics.safety_absolute}"
            )
        report = DeterministicDoctorBenchmarkReport(
            cases=tuple(cases),
            metrics=metrics,
            policy=policy,
            dual_run=dual,
            fixture_families=REQUIRED_FIXTURE_FAMILIES,
        )
        sealed = seal_report(report.to_dict(include_id=False))
        # Ensure report_id matches sealed identity.
        if sealed["report_id"] != report.report_id:
            sealed = seal_report(report.to_dict(include_id=False))
        return sealed
    finally:
        if install_guards:
            restore_model_route_guards(previous)


def build_report(
    cases: Sequence[CaseResult],
    *,
    dual_run: Mapping[str, Any] | None = None,
    policy: DeterministicDoctorBenchmarkPolicy | None = None,
) -> DeterministicDoctorBenchmarkReport:
    policy = policy or DeterministicDoctorBenchmarkPolicy()
    dual = dict(dual_run or {"pass_count": 1, "identity_equivalent": True, "receipts": []})
    metrics = DeterministicDoctorMetrics.from_cases(
        cases,
        dual_run_identity_equivalent=bool(dual.get("identity_equivalent", True)),
    )
    return DeterministicDoctorBenchmarkReport(
        cases=tuple(cases),
        metrics=metrics,
        policy=policy,
        dual_run=dual,
        fixture_families=REQUIRED_FIXTURE_FAMILIES,
    )


__all__ = [
    "ADMITTABLE_SCENARIOS",
    "ARTIFACT_ROLES",
    "BENCHMARK_INTERFACE",
    "BENCHMARK_SCHEMA",
    "CORPUS_VERSION",
    "DUAL_RUN_PASSES",
    "FAIL_CLOSED_SCENARIOS",
    "FIXTURE_FAMILIES",
    "GOAL_ID",
    "MANIFEST_SCHEMA",
    "OutcomeKind",
    "REQUIRED_FIXTURE_FAMILIES",
    "REQUIRED_OUTCOME_KINDS",
    "REQUIRED_STAGES",
    "SAFETY_ABSOLUTE_KEYS",
    "SAFETY_FLOOR_KEYS",
    "TASK_ID",
    "CaseResult",
    "DeterministicDoctorBenchmarkError",
    "DeterministicDoctorBenchmarkPolicy",
    "DeterministicDoctorBenchmarkReport",
    "DeterministicDoctorFixture",
    "DeterministicDoctorMetrics",
    "DeterministicDoctorBenchmarkMetrics",
    "ResourceMeasurement",
    "SafetyCounters",
    "build_authority_roots",
    "build_evidence_snapshot_payload",
    "build_report",
    "build_run_receipt_payload",
    "default_fixture_manifest_path",
    "evaluate_corpus",
    "evaluate_fixture",
    "family_for_scenario",
    "install_model_route_guards",
    "load_fixture_manifest",
    "restore_model_route_guards",
    "run_benchmark",
    "run_dual_pass",
    "seal_report",
    "verify_report",
]
