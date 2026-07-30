"""Complete symbolic contract assurance baseline pipeline (SCA-200).

Materializes a single-snapshot, zero-LLM shadow baseline over SwissKnife:

1. repository index / coverage ledger
2. expected-contract extraction and catalog normalization
3. mandatory symbolic-contract graph projection
4. expected-versus-actual MCP++ invocation tracing
5. proof / cache verification via McpContractProver + TrustAwareProofCache
   (SCAEV071PROOFCACHE end-to-end orchestration)
6. mismatch classification and vulnerability rule evaluation
7. bounded artifact publication (coverage, findings, summary)

Unhealthy or incomplete stages never grant exhaustive, no-drift, or no-findings
claims. Empty findings under partial health are not parity evidence. Cache hits
revalidate exact snapshot, graph, policy, solver, kernel, and toolchain roots;
counterexamples flow into mismatch/vulnerability records; missing evidence
withholds downstream authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import (
    AssuranceLevel,
    ResourceBudget,
    content_identity,
)
from ..proof.mcp_contract_obligations import (
    McpContractObligation,
    McpContractObligationError,
    compile_contract_claim,
)
from ..proof.mcp_contract_proof_cache import (
    IdentityBinding,
    ProofCacheKey,
    TrustAwareProofCache,
)
from ..proof.mcp_contract_prover import (
    MCP_LOCAL_GRAPH_CHECKER_ID,
    MCP_LOCAL_SCHEMA_CHECKER_ID,
    MCP_NO_KERNEL_ID,
    MCP_PROVIDER_TRANSLATOR_ID,
    ContractProofOutcome,
    ContractProofRoute,
    McpContractProofResult,
    McpContractProver,
)
from .analyzer_health import AnalyzerHealthStatus
from .content_identity_bridge import identify_strict_artifact
from .contract_mismatch_analyzer import (
    ContractFinding,
    ContractMismatchAnalyzer,
    FindingLifecycle,
    MismatchAnalysis,
    MismatchState,
)
from .contract_vulnerability_rules import (
    ContractVulnerabilityFinding,
    ContractVulnerabilityRuleEngine,
)
from .mcp_contract_analysis import (
    ContractParityClaim,
    McpContractAnalysis,
    McpContractAnalyzer,
    ParityState,
)
from .mcp_contract_catalog import McpContractCatalog
from .mcp_invocation_trace import McpInvocationTrace, McpInvocationTracer
from .repository_indexer import RepositoryIndex
from .repository_snapshot import RepositorySnapshot
from .runtime_component_catalog import RuntimeComponentCatalog
from .runtime_contract_evidence_compiler import (
    AnchorResolutionState,
    RuntimeContractEvidenceCompilation,
    RuntimeContractEvidenceCompiler,
    compile_runtime_contract_evidence,
)
from .swissknife_contract_extractor import (
    SwissKnifeContractExtraction,
    SwissKnifeContractExtractor,
)
from .symbolic_contract_graph import (
    GRAPH_VERSION,
    ContractAuthority,
    ContractEdgeKind,
    ContractGraphEdge,
    ContractGraphNode,
    ContractNodeKind,
    ContractProvenance,
    SymbolicContractGraph,
    build_symbolic_contract_graph,
    project_symbolic_contract_graph,
)


CONTRACT_ASSURANCE_BASELINE_INTERFACE: Final = "ContractAssuranceBaseline@1"
CONTRACT_ASSURANCE_BASELINE_VERSION: Final = "1"

# SCA-G071 / SCA-218 end-to-end proof/cache orchestration evidence term.
PROOF_PIPELINE_EVIDENCE: Final = "SCAEV071PROOFCACHE"
PROOF_PIPELINE_INTERFACE: Final = "ContractAssuranceProofPipeline@1"

BASELINE_FINDINGS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-baseline-contract-findings@1"
)
BASELINE_COVERAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-baseline-coverage@1"
)
BASELINE_RUN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-baseline-run@1"
)
BASELINE_STAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-baseline-stage@1"
)
BASELINE_CONTRACT_ROW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-baseline-contract-row@1"
)
PROOF_PIPELINE_OUTCOME_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/sca-baseline-proof-pipeline-outcome@1"
)

TERMINAL_STATUS_DOMAIN: Final[tuple[str, ...]] = (
    "proved",
    "refuted",
    "unknown",
    "unsupported",
    "stale",
)

DEFAULT_REPRODUCTION_COMMAND: Final = (
    "python3 external/ipfs_accelerate/scripts/index_repository_contracts.py "
    "--repo-root . "
    "--scope-config config/swissknife_symbolic_contract_scope.json "
    "--output-root data/agent_supervisor/swissknife_contract_assurance/baseline "
    "--shadow"
)

# Hard envelope for published baseline artifacts (bytes).
DEFAULT_MAX_ARTIFACT_BYTES: Final = 4_000_000

# Stable root identities for local-checker receipts that omit registry IDs.
BASELINE_THEOREM_REGISTRY_ID: Final = "mcp-baseline-theorem-registry@1"
BASELINE_CAPABILITY_REPORT_ID: Final = "mcp-baseline-capability@1"
DEFAULT_PROOF_REPOSITORY_ID: Final = "repository:swissknife-contract-assurance"
DEFAULT_PROOF_TOOLCHAIN_ID: Final = "toolchain:mcp-contract-assurance@1"
DEFAULT_PROOF_POLICY_ID: Final = "policy:mcp-contract-assurance@1"
DEFAULT_PROOF_ASSUMPTION_ID: Final = "assumption:closed-world-catalog"
DEFAULT_PROOF_SCOPE_ID: Final = "scope:reviewed-mcp-surface"


class ContractAssuranceBaselineError(ValueError):
    """Baseline pipeline evidence is incomplete, contradictory, or oversized."""


class TerminalContractStatus(str, Enum):
    """Closed terminal statuses admitted by the baseline population ledger."""

    PROVED = "proved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    STALE = "stale"


class BaselineStageName(str, Enum):
    REPOSITORY_INDEX = "repository_index"
    EXTRACTION = "extraction"
    CATALOG = "catalog"
    GRAPH = "graph"
    INVOCATION_TRACE = "invocation_trace"
    PROOF_CACHE = "proof_cache"
    MISMATCH = "mismatch"
    VULNERABILITY = "vulnerability"
    PUBLISH = "publish"


class StageCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    WITHHELD = "withheld"
    FAILED = "failed"
    SKIPPED = "skipped"


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    allow_multiline: bool = False,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise ContractAssuranceBaselineError(f"{name} must be a string")
    if "\x00" in text:
        raise ContractAssuranceBaselineError(f"{name} must not contain NUL")
    if allow_multiline:
        # Markdown summaries may end with a trailing newline; strip only CR noise.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    elif text != text.strip():
        raise ContractAssuranceBaselineError(
            f"{name} must not contain surrounding whitespace"
        )
    if required and not text.strip():
        raise ContractAssuranceBaselineError(f"{name} is required")
    if len(text.encode("utf-8")) > 262_144:
        raise ContractAssuranceBaselineError(f"{name} is oversized")
    return text


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise ContractAssuranceBaselineError("value exceeds nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ContractAssuranceBaselineError(
            "floating values are not canonical baseline evidence"
        )
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth=depth + 1) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ContractAssuranceBaselineError("raw bytes are not baseline evidence")
    raise ContractAssuranceBaselineError(
        f"unsupported baseline value type: {type(value).__name__}"
    )


def _canonical_measurement(value: Any, *, depth: int = 0) -> Any:
    """Project operational floats into deterministic decimal evidence.

    Analyzer reports use Python floats for ratios and policy thresholds.
    Baseline identities intentionally reject binary floating-point values, so
    this boundary turns finite measurements into exact decimal strings while
    preserving integer counters and all other canonical JSON value types.
    """

    if depth > 32:
        raise ContractAssuranceBaselineError("measurement exceeds nesting bound")
    if isinstance(value, float):
        try:
            decimal = Decimal(str(value))
        except InvalidOperation as exc:
            raise ContractAssuranceBaselineError(
                "measurement is not a valid decimal"
            ) from exc
        if not decimal.is_finite():
            raise ContractAssuranceBaselineError(
                "non-finite measurement is not baseline evidence"
            )
        if decimal == 0:
            return "0"
        text = format(decimal, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_measurement(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_measurement(item, depth=depth + 1)
            for item in value
        ]
    return _plain(value, depth=depth)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _plain(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_label(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _canonical_bytes(value))


def _atomic_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _status_from_parity(state: ParityState | str | None) -> TerminalContractStatus:
    if state is None:
        return TerminalContractStatus.UNKNOWN
    value = state if isinstance(state, ParityState) else ParityState(str(state))
    if value is ParityState.SATISFIED:
        return TerminalContractStatus.PROVED
    if value is ParityState.REFUTED:
        return TerminalContractStatus.REFUTED
    if value is ParityState.UNSUPPORTED:
        return TerminalContractStatus.UNSUPPORTED
    return TerminalContractStatus.UNKNOWN


def _status_from_mismatch(state: MismatchState | str | None) -> TerminalContractStatus:
    if state is None:
        return TerminalContractStatus.UNKNOWN
    value = state if isinstance(state, MismatchState) else MismatchState(str(state))
    if value is MismatchState.REFUTED:
        return TerminalContractStatus.REFUTED
    if value is MismatchState.STALE:
        return TerminalContractStatus.STALE
    if value is MismatchState.UNSUPPORTED:
        return TerminalContractStatus.UNSUPPORTED
    return TerminalContractStatus.UNKNOWN


def _status_from_proof_outcome(
    outcome: ContractProofOutcome | str | None,
) -> TerminalContractStatus:
    """Map a closed prover outcome onto the baseline terminal domain."""

    if outcome is None:
        return TerminalContractStatus.UNKNOWN
    value = (
        outcome
        if isinstance(outcome, ContractProofOutcome)
        else ContractProofOutcome(str(outcome))
    )
    if value is ContractProofOutcome.PROVED:
        return TerminalContractStatus.PROVED
    if value is ContractProofOutcome.REFUTED:
        return TerminalContractStatus.REFUTED
    if value is ContractProofOutcome.UNSUPPORTED:
        return TerminalContractStatus.UNSUPPORTED
    # inconclusive / timed_out remain unknown (not proved authority).
    return TerminalContractStatus.UNKNOWN


def _identity_binding(logical_id: str, *, component: str, **fields: Any) -> IdentityBinding:
    """Build a CID-bound identity with a stable logical id for cache keys."""

    payload = {
        "component": component,
        "logical_id": logical_id,
        **fields,
    }
    return IdentityBinding.from_identity(
        identify_strict_artifact(payload),
        logical_id=logical_id,
    )


def _default_resource_budget() -> ResourceBudget:
    return ResourceBudget(
        wall_time_ms=0,
        cpu_time_ms=0,
        memory_bytes=0,
        max_processes=0,
        max_premises=0,
        network_allowed=False,
    )


def _facts_from_claim(claim: ContractParityClaim) -> dict[str, Any]:
    """Project analyzer observations into local-checker premise facts.

    Satisfied claims supply complete positive premises and schema_valid=True.
    Refuted claims mark premises failed so the local checker emits a
    counterexample. Partial/unknown claims leave premises incomplete so the
    prover returns inconclusive rather than forging a proof.
    """

    premise_ids = tuple(claim.premise_ids)
    if claim.state is ParityState.SATISFIED:
        return {
            "premise_results": {item: True for item in premise_ids},
            "schema_valid": True,
            "graph_valid": True,
            "required_edges": (),
            "observed_edges": (),
        }
    if claim.state is ParityState.REFUTED:
        failed = {item: False for item in premise_ids}
        if not failed and claim.claim_id:
            failed = {claim.claim_id: False}
        return {
            "premise_results": failed,
            "schema_valid": False,
            "graph_valid": False,
            "required_edges": (("expected", "handler"),),
            "observed_edges": (),
            "failed_edges": (("expected", "handler"),),
        }
    # unsupported / ambiguous / unknown — incomplete evidence
    return {
        "premise_results": {},
        "schema_valid": None,
        "graph_valid": None,
        "required_edges": (),
        "observed_edges": (),
    }


def _align_proof_result_for_cache(
    result: McpContractProofResult,
) -> McpContractProofResult:
    """Bind the empty local-checker theorem registry slot to a stable identity.

    Local graph/schema receipts omit ``theorem_registry_id``.  TrustAwareProofCache
    requires a non-empty logical id on every key dimension, so the baseline binds
    the empty slot to :data:`BASELINE_THEOREM_REGISTRY_ID` without changing
    assurance, verdict, or independent evidence.
    """

    receipt = result.receipt
    registry_id = receipt.theorem_registry_id or BASELINE_THEOREM_REGISTRY_ID
    if (
        receipt.theorem_registry_id == registry_id
        and result.outcome is not ContractProofOutcome.PROVED
    ):
        return result
    aligned_receipt = replace(receipt, theorem_registry_id=registry_id)
    return McpContractProofResult(
        obligation_id=result.obligation_id,
        outcome=result.outcome,
        route=result.route,
        reason_codes=result.reason_codes,
        receipt=aligned_receipt,
        counterexample=result.counterexample,
        capability=result.capability,
        fallback_used=result.fallback_used,
    )


def _route_execution_ids(
    route: ContractProofRoute,
) -> tuple[str, str, str]:
    """Return (translator_id, solver_id, kernel_id) matching McpContractProver receipts."""

    if route is ContractProofRoute.LOCAL_GRAPH:
        checker = MCP_LOCAL_GRAPH_CHECKER_ID
        return checker, route.value, checker
    if route is ContractProofRoute.LOCAL_SCHEMA:
        checker = MCP_LOCAL_SCHEMA_CHECKER_ID
        return checker, route.value, checker
    if route is ContractProofRoute.NONE:
        return MCP_NO_KERNEL_ID, route.value, MCP_NO_KERNEL_ID
    return MCP_PROVIDER_TRANSLATOR_ID, route.value, MCP_NO_KERNEL_ID


def _build_baseline_proof_cache_key(
    *,
    obligation: McpContractObligation,
    route: ContractProofRoute,
    catalog: McpContractCatalog,
    graph: SymbolicContractGraph | None,
    capability_root: str,
    resource_budget: ResourceBudget | None = None,
) -> ProofCacheKey:
    """Construct a TrustAwareProofCache key bound to exact pipeline roots."""

    budget = resource_budget or _default_resource_budget()
    scope_ids = tuple(obligation.scope_ids) or (DEFAULT_PROOF_SCOPE_ID,)
    premise_ids = tuple(obligation.premise_ids) or (obligation.obligation_id,)
    assumption_ids = tuple(obligation.assumption_ids) or (
        DEFAULT_PROOF_ASSUMPTION_ID,
    )
    translator_id, solver_id, kernel_id = _route_execution_ids(route)
    toolchain_id = obligation.toolchain_id
    policy_id = obligation.policy_id
    registry_id = BASELINE_THEOREM_REGISTRY_ID
    capability_id = capability_root or BASELINE_CAPABILITY_REPORT_ID
    # Graph root is recorded in capability material so cache keys change when
    # mandatory graph closure changes without detaching receipt scope ids.
    capability_payload = {
        "capability_id": capability_id,
        "graph_root": graph.graph_root if graph is not None else "",
        "catalog_id": catalog.catalog_id,
        "route": route.value,
    }
    return ProofCacheKey(
        snapshot=_identity_binding(
            obligation.snapshot_id,
            component="snapshot",
            snapshot_id=obligation.snapshot_id,
        ),
        scope=tuple(
            _identity_binding(item, component="scope", scope_id=item)
            for item in scope_ids
        ),
        property_catalog=_identity_binding(
            catalog.catalog_id,
            component="property_catalog",
            catalog_id=catalog.catalog_id,
        ),
        obligation=_identity_binding(
            obligation.obligation_id,
            component="obligation",
            obligation_id=obligation.obligation_id,
            property_id=obligation.property_id,
        ),
        premises=tuple(
            _identity_binding(item, component="premise", premise_id=item)
            for item in premise_ids
        ),
        assumptions=tuple(
            _identity_binding(item, component="assumption", assumption_id=item)
            for item in assumption_ids
        ),
        provider=_identity_binding(
            solver_id, component="provider", provider_id=solver_id
        ),
        translator=_identity_binding(
            translator_id, component="translator", translator_id=translator_id
        ),
        solver=_identity_binding(solver_id, component="solver", solver_id=solver_id),
        kernel=_identity_binding(kernel_id, component="kernel", kernel_id=kernel_id),
        toolchain=_identity_binding(
            toolchain_id, component="toolchain", toolchain_id=toolchain_id
        ),
        theorem_registry=_identity_binding(
            registry_id, component="theorem_registry", registry_id=registry_id
        ),
        policy=_identity_binding(policy_id, component="policy", policy_id=policy_id),
        capability_report=_identity_binding(
            capability_id,
            component="capability_report",
            **capability_payload,
        ),
        resource_budget=budget,
        required_assurance=obligation.required_assurance,
        route=route,
    )


def _revalidate_cached_roots(
    *,
    key: ProofCacheKey,
    receipt: Any,
    graph: SymbolicContractGraph | None,
    snapshot_id: str,
) -> tuple[str, ...]:
    """Re-check exact roots on a cache hit; return rejection reason codes."""

    reasons: list[str] = []
    if receipt.repository_tree_id != snapshot_id:
        reasons.append("snapshot_root_mismatch")
    if receipt.repository_tree_id != key.snapshot.logical_id:
        reasons.append("cache_snapshot_binding_mismatch")
    if receipt.policy_id != key.policy.logical_id:
        reasons.append("policy_root_mismatch")
    if receipt.solver_id != key.solver.logical_id:
        reasons.append("solver_root_mismatch")
    if receipt.kernel_id != key.kernel.logical_id:
        reasons.append("kernel_root_mismatch")
    if receipt.toolchain_id != key.toolchain.logical_id:
        reasons.append("toolchain_root_mismatch")
    if graph is not None:
        expected_graph = graph.graph_root
        # Capability report encodes the graph root used at store time.
        try:
            capability_bytes = key.capability_report.canonical_bytes
            capability = json.loads(capability_bytes.decode("utf-8"))
            stored_graph = str(capability.get("graph_root") or "")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            stored_graph = ""
        if stored_graph and stored_graph != expected_graph:
            reasons.append("graph_root_mismatch")
    return tuple(sorted(set(reasons)))


@dataclass(frozen=True)
class ProofPipelineOutcome:
    """One obligation discharged (or withheld) through the baseline pipeline."""

    contract_id: str
    claim_family: str
    operation_id: str
    obligation_id: str
    outcome: str
    terminal_status: str
    route: str
    cache_hit: bool
    reason_codes: tuple[str, ...]
    receipt_content_id: str = ""
    counterexample_id: str = ""
    roots: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contract_id", _text(self.contract_id, "contract_id")
        )
        object.__setattr__(
            self, "claim_family", _text(self.claim_family, "claim_family")
        )
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        object.__setattr__(
            self,
            "obligation_id",
            _text(self.obligation_id, "obligation_id", required=False),
        )
        object.__setattr__(self, "outcome", _text(self.outcome, "outcome"))
        object.__setattr__(
            self,
            "terminal_status",
            _text(self.terminal_status, "terminal_status"),
        )
        if self.terminal_status not in TERMINAL_STATUS_DOMAIN:
            raise ContractAssuranceBaselineError(
                f"illegal proof pipeline terminal: {self.terminal_status}"
            )
        object.__setattr__(
            self, "route", _text(self.route, "route", required=False)
        )
        if not isinstance(self.cache_hit, bool):
            raise ContractAssuranceBaselineError("cache_hit must be boolean")
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                sorted(
                    {
                        _text(item, "reason_code")
                        for item in self.reason_codes
                        if str(item).strip()
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "receipt_content_id",
            _text(self.receipt_content_id, "receipt_content_id", required=False),
        )
        object.__setattr__(
            self,
            "counterexample_id",
            _text(self.counterexample_id, "counterexample_id", required=False),
        )
        object.__setattr__(
            self,
            "roots",
            MappingProxyType(
                {
                    str(key): _text(value, f"roots.{key}", required=False)
                    for key, value in dict(self.roots or {}).items()
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_PIPELINE_OUTCOME_SCHEMA,
            "evidence_id": PROOF_PIPELINE_EVIDENCE,
            "contract_id": self.contract_id,
            "claim_family": self.claim_family,
            "operation_id": self.operation_id,
            "obligation_id": self.obligation_id,
            "outcome": self.outcome,
            "terminal_status": self.terminal_status,
            "route": self.route,
            "cache_hit": self.cache_hit,
            "reason_codes": list(self.reason_codes),
            "receipt_content_id": self.receipt_content_id,
            "counterexample_id": self.counterexample_id,
            "roots": dict(self.roots),
        }


def run_contract_proof_pipeline(
    *,
    analyses: Sequence[McpContractAnalysis],
    catalog: McpContractCatalog,
    snapshot_id: str,
    graph: SymbolicContractGraph | None = None,
    proof_cache: TrustAwareProofCache | None = None,
    proof_cache_dir: str | Path | None = None,
    prover: McpContractProver | None = None,
    repository_id: str = "",
    toolchain_id: str = "",
    policy_id: str = "",
    capability_root: str = "",
    required_assurance: AssuranceLevel | str = AssuranceLevel.KERNEL_VERIFIED,
) -> tuple[ProofPipelineOutcome, ...]:
    """Wire claims through obligations, prover, kernel policy, and proof cache.

    ``TrustAwareProofCache`` remains the sole receipt cache.  Analyzer parity
    states never mint assurance: only independently accepted receipts (local
    kernel-grade checkers or trusted validators) establish proved terminals.
    """

    if proof_cache is None:
        cache_path = (
            Path(proof_cache_dir)
            if proof_cache_dir is not None
            else None
        )
        proof_cache = TrustAwareProofCache(cache_path)
    active_prover = prover or McpContractProver()
    repo_id = _text(
        repository_id or DEFAULT_PROOF_REPOSITORY_ID, "repository_id"
    )
    tool_id = _text(
        toolchain_id or DEFAULT_PROOF_TOOLCHAIN_ID, "toolchain_id"
    )
    pol_id = _text(policy_id or DEFAULT_PROOF_POLICY_ID, "policy_id")
    assurance = (
        required_assurance
        if isinstance(required_assurance, AssuranceLevel)
        else AssuranceLevel(str(required_assurance))
    )
    outcomes: list[ProofPipelineOutcome] = []

    for analysis in analyses:
        contract_id = analysis.expected_contract_id or (
            f"contract:{analysis.operation_id}"
        )
        for claim in analysis.claims:
            reasons: list[str] = []
            try:
                obligation = compile_contract_claim(
                    claim,
                    catalog=catalog,
                    contract=contract_id,
                    repository_id=repo_id,
                    snapshot_id=snapshot_id,
                    scope_ids=tuple(claim.premise_ids)
                    or (DEFAULT_PROOF_SCOPE_ID,),
                    assumption_ids=(DEFAULT_PROOF_ASSUMPTION_ID,),
                    toolchain_id=tool_id,
                    policy_id=pol_id,
                    required_assurance=assurance,
                )
            except (McpContractObligationError, TypeError, ValueError) as exc:
                outcomes.append(
                    ProofPipelineOutcome(
                        contract_id=contract_id,
                        claim_family=claim.family.value,
                        operation_id=analysis.operation_id,
                        obligation_id="",
                        outcome=ContractProofOutcome.UNSUPPORTED.value,
                        terminal_status=TerminalContractStatus.UNSUPPORTED.value,
                        route=ContractProofRoute.NONE.value,
                        cache_hit=False,
                        reason_codes=(
                            "obligation_compile_failed",
                            type(exc).__name__,
                        ),
                        roots={
                            "snapshot": snapshot_id,
                            "graph": graph.graph_root if graph is not None else "",
                            "policy": pol_id,
                            "toolchain": tool_id,
                        },
                    )
                )
                continue

            facts = _facts_from_claim(claim)
            route = active_prover.route(obligation)
            cache_key = _build_baseline_proof_cache_key(
                obligation=obligation,
                route=route,
                catalog=catalog,
                graph=graph,
                capability_root=capability_root or BASELINE_CAPABILITY_REPORT_ID,
                resource_budget=_default_resource_budget(),
            )

            def _prove() -> McpContractProofResult:
                raw = active_prover.prove(
                    obligation,
                    facts=facts,
                    resource_budget=_default_resource_budget(),
                )
                return _align_proof_result_for_cache(raw)

            # Sole authoritative cache: exact-key lookup, then single-flight prove.
            cache_hit = False
            try:
                cached = proof_cache.get_or_prove(cache_key, _prove)
                proof_result = cached.result
                cache_hit = bool(cached.cache_hit)
                reasons.extend(cached.reason_codes)
            except Exception as exc:  # noqa: BLE001 - typed pipeline terminal
                # Detached/invalid provider results fail closed to unknown.
                provisional = _prove()
                proof_result = provisional
                reasons.extend(
                    (
                        "proof_cache_coordination_failed",
                        type(exc).__name__,
                        *provisional.reason_codes,
                    )
                )
            else:
                if cache_hit and proof_result.receipt is not None:
                    root_failures = _revalidate_cached_roots(
                        key=cache_key,
                        receipt=proof_result.receipt,
                        graph=graph,
                        snapshot_id=snapshot_id,
                    )
                    if root_failures:
                        reasons.extend(root_failures)
                        reasons.append("cache_hit_rejected_stale_roots")
                        proof_result = _prove()
                        cache_hit = False
                        if proof_result.outcome is ContractProofOutcome.PROVED:
                            stored = proof_cache.put(cache_key, proof_result)
                            if not stored.stored:
                                reasons.extend(stored.reason_codes)

            reasons.extend(proof_result.reason_codes)
            terminal = _status_from_proof_outcome(proof_result.outcome)
            counterexample_id = ""
            if proof_result.counterexample is not None:
                counterexample_id = str(
                    getattr(
                        proof_result.counterexample,
                        "counterexample_id",
                        "",
                    )
                    or content_identity(proof_result.counterexample.to_dict())
                )
            receipt_id = ""
            try:
                receipt_id = str(
                    getattr(proof_result.receipt, "content_id", "")
                    or content_identity(proof_result.receipt.to_dict())
                )
            except (TypeError, ValueError):
                receipt_id = ""
            outcomes.append(
                ProofPipelineOutcome(
                    contract_id=contract_id,
                    claim_family=claim.family.value,
                    operation_id=analysis.operation_id,
                    obligation_id=obligation.obligation_id,
                    outcome=proof_result.outcome.value,
                    terminal_status=terminal.value,
                    route=proof_result.route.value,
                    cache_hit=cache_hit,
                    reason_codes=tuple(sorted(set(reasons))),
                    receipt_content_id=receipt_id,
                    counterexample_id=counterexample_id,
                    roots={
                        "snapshot": proof_result.receipt.repository_tree_id,
                        "graph": graph.graph_root if graph is not None else "",
                        "policy": proof_result.receipt.policy_id,
                        "solver": proof_result.receipt.solver_id,
                        "kernel": proof_result.receipt.kernel_id,
                        "toolchain": proof_result.receipt.toolchain_id,
                        "catalog": catalog.catalog_id,
                    },
                )
            )
    return tuple(
        sorted(
            outcomes,
            key=lambda item: (
                item.contract_id,
                item.claim_family,
                item.obligation_id,
            ),
        )
    )


@dataclass(frozen=True)
class BaselineStageReceipt:
    """Typed receipt for one pipeline stage bound to a single snapshot."""

    name: BaselineStageName
    completeness: StageCompleteness
    reason_codes: tuple[str, ...] = ()
    root_id: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", BaselineStageName(self.name)
            if not isinstance(self.name, BaselineStageName)
            else self.name
        )
        object.__setattr__(
            self,
            "completeness",
            StageCompleteness(self.completeness)
            if not isinstance(self.completeness, StageCompleteness)
            else self.completeness,
        )
        codes = tuple(
            sorted(
                {
                    _text(item, "reason_code", required=True)
                    for item in self.reason_codes
                }
            )
        )
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "root_id", _text(self.root_id, "root_id", required=False))
        object.__setattr__(
            self,
            "details",
            MappingProxyType(_plain(dict(self.details or {}))),
        )

    @property
    def complete(self) -> bool:
        return self.completeness is StageCompleteness.COMPLETE

    @property
    def healthy_enough_for_authority(self) -> bool:
        return self.completeness is StageCompleteness.COMPLETE and not any(
            code.startswith("withheld") or code.endswith("_unhealthy")
            for code in self.reason_codes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BASELINE_STAGE_SCHEMA,
            "name": self.name.value,
            "completeness": self.completeness.value,
            "reason_codes": list(self.reason_codes),
            "root_id": self.root_id,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class BaselineContractTerminal:
    """One in-scope contract with a closed terminal status."""

    contract_id: str
    claim_family: str
    package_id: str
    status: TerminalContractStatus
    terminal: bool = True
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contract_id", _text(self.contract_id, "contract_id")
        )
        object.__setattr__(
            self, "claim_family", _text(self.claim_family, "claim_family")
        )
        object.__setattr__(
            self,
            "package_id",
            _text(self.package_id, "package_id", required=False),
        )
        status = (
            self.status
            if isinstance(self.status, TerminalContractStatus)
            else TerminalContractStatus(str(self.status))
        )
        if status.value not in TERMINAL_STATUS_DOMAIN:
            raise ContractAssuranceBaselineError(
                f"illegal terminal status: {status.value}"
            )
        object.__setattr__(self, "status", status)
        if not isinstance(self.terminal, bool):
            raise ContractAssuranceBaselineError("terminal must be boolean")
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                sorted(
                    {
                        _text(item, "reason_code", required=True)
                        for item in self.reason_codes
                    }
                )
            ),
        )

    def as_row(self) -> list[Any]:
        return [
            self.contract_id,
            self.claim_family,
            self.package_id,
            self.status.value,
            bool(self.terminal),
            list(self.reason_codes),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BASELINE_CONTRACT_ROW_SCHEMA,
            "contract_id": self.contract_id,
            "claim_family": self.claim_family,
            "package_id": self.package_id,
            "status": self.status.value,
            "terminal": self.terminal,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ContractAssuranceBaselineResult:
    """Published baseline bound to one snapshot with stage receipts."""

    snapshot_id: str
    coverage: Mapping[str, Any]
    findings: Mapping[str, Any]
    summary_markdown: str
    stages: tuple[BaselineStageReceipt, ...]
    llm_call_count: int = 0
    mismatch_analysis: MismatchAnalysis | None = None
    vulnerability_findings: tuple[ContractVulnerabilityFinding, ...] = ()
    proof_pipeline_outcomes: tuple[ProofPipelineOutcome, ...] = ()
    graph: SymbolicContractGraph | None = None
    extraction: SwissKnifeContractExtraction | None = None
    catalog: McpContractCatalog | None = None
    repository_index: RepositoryIndex | None = None
    runtime_catalog: RuntimeComponentCatalog | None = None
    result_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        if not isinstance(self.llm_call_count, int) or self.llm_call_count != 0:
            raise ContractAssuranceBaselineError(
                "baseline runtime must record zero LLM calls"
            )
        object.__setattr__(self, "coverage", MappingProxyType(_plain(dict(self.coverage))))
        object.__setattr__(self, "findings", MappingProxyType(_plain(dict(self.findings))))
        object.__setattr__(
            self,
            "summary_markdown",
            _text(
                self.summary_markdown,
                "summary_markdown",
                allow_multiline=True,
            ),
        )
        stages = tuple(self.stages)
        if not all(isinstance(item, BaselineStageReceipt) for item in stages):
            raise ContractAssuranceBaselineError("stages must be BaselineStageReceipt")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(
            self,
            "vulnerability_findings",
            tuple(self.vulnerability_findings),
        )
        object.__setattr__(
            self,
            "proof_pipeline_outcomes",
            tuple(self.proof_pipeline_outcomes),
        )
        payload = self._identity_payload()
        derived = content_identity(payload)
        claimed = _text(self.result_id, "result_id", required=False)
        if claimed and claimed != derived:
            raise ContractAssuranceBaselineError("result_id does not match content")
        object.__setattr__(self, "result_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": BASELINE_RUN_SCHEMA,
            "interface": CONTRACT_ASSURANCE_BASELINE_INTERFACE,
            "version": CONTRACT_ASSURANCE_BASELINE_VERSION,
            "snapshot_id": self.snapshot_id,
            "coverage_id": self.findings.get("coverage_id", ""),
            "findings_root": self.findings.get("findings_root", ""),
            "stages": [item.to_dict() for item in self.stages],
            "llm_call_count": 0,
        }

    @property
    def claims(self) -> Mapping[str, Any]:
        return dict(self.findings.get("claims") or {})

    @property
    def no_drift_claim(self) -> bool:
        return bool(self.claims.get("no_drift"))

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return bool(
            self.findings.get("analyzer_health", {}).get(
                "safe_for_completion_reasoning"
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "result_id": self.result_id,
            "coverage": dict(self.coverage),
            "findings": dict(self.findings),
            "summary_markdown": self.summary_markdown,
        }


def _index_healthy(index: RepositoryIndex | None) -> bool:
    if index is None:
        return False
    return (
        index.health.status is AnalyzerHealthStatus.HEALTHY
        and index.safe_for_completion_reasoning
    )


def _coverage_from_index(index: RepositoryIndex) -> dict[str, Any]:
    """Bounded coverage ledger bound to the repository index row IDs."""

    return {
        "schema": BASELINE_COVERAGE_SCHEMA,
        "snapshot_id": index.snapshot_id,
        "index_id": index.index_id,
        "ast_index_id": index.ast_index_id,
        "scope_id": index.snapshot.scope_id,
        "scope_policy_id": index.snapshot.scope_policy_id,
        "head_commit_id": index.snapshot.head_commit_id,
        "head_tree_id": index.snapshot.head_tree_id,
        "index_tree_id": index.snapshot.index_tree_id,
        "is_clean": index.snapshot.is_clean,
        "health": _canonical_measurement(index.health.to_dict()),
        "stats": _canonical_measurement(index.build_stats.to_dict()),
        "rows": [
            {
                "path": row.path,
                "row_id": row.row_id,
                "disposition_kind": row.disposition_kind.value,
                "declared_kind": row.declared_kind.value,
                "reason_code": row.reason_code,
                "parser_status": row.parser_status.value,
                "parser_reason": row.parser_reason,
                "language": row.language,
                "tracked": row.tracked,
                "overlay": row.overlay,
            }
            for row in index.rows
        ],
    }


def _coverage_from_snapshot(snapshot: RepositorySnapshot | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(snapshot, RepositorySnapshot):
        payload = snapshot.to_dict()
    else:
        payload = dict(snapshot)
    # Preserve the SCA-120 snapshot ledger shape when it is already materialised.
    if payload.get("schema", "").endswith("sca-repository-snapshot@1"):
        return payload
    return {
        "schema": BASELINE_COVERAGE_SCHEMA,
        "snapshot": payload,
        "snapshot_id": str(
            payload.get("snapshot_id")
            or (snapshot.snapshot_id if isinstance(snapshot, RepositorySnapshot) else "")
        ),
    }


def _health_projection(
    index: RepositoryIndex | None,
    *,
    discovery_complete: bool,
    measurement_complete: bool,
) -> dict[str, Any]:
    if index is None:
        return {
            "status": "missing",
            "reason_code": "repository_index_missing",
            "safe_for_completion_reasoning": False,
            "exhaustive": False,
            "no_drift_claim": False,
            "coverage_complete": False,
            "index_complete": False,
            "contract_discovery_complete": discovery_complete,
            "contract_measurement_complete": measurement_complete,
            "repository_index_health_status": "missing",
            "blocking_errors": [],
            "metrics": {},
            "thresholds": {},
        }

    health = _canonical_measurement(index.health.to_dict())
    status = index.health.status
    reasons = list(index.health.reasons)
    primary_reason = reasons[0] if reasons else (
        "healthy" if status is AnalyzerHealthStatus.HEALTHY else status.value
    )
    overall = (
        "healthy"
        if status is AnalyzerHealthStatus.HEALTHY and measurement_complete
        else (
            "partial"
            if status is AnalyzerHealthStatus.PARTIAL
            or (status is AnalyzerHealthStatus.HEALTHY and not measurement_complete)
            else "unhealthy"
        )
    )
    safe = bool(
        status is AnalyzerHealthStatus.HEALTHY
        and index.safe_for_completion_reasoning
        and measurement_complete
        and discovery_complete
    )
    blocking: list[dict[str, Any]] = []
    for reason in reasons:
        blocking.append(
            {
                "reason_code": reason,
                "error_type": "AnalyzerHealth",
                "affected_path_count": int(
                    getattr(index.build_stats, "parse_failure_count", 0) or 0
                ),
            }
        )
    return {
        "status": overall,
        "reason_code": primary_reason
        if overall != "healthy"
        else ("measurement_incomplete" if not measurement_complete else "healthy"),
        "safe_for_completion_reasoning": safe,
        "exhaustive": safe,
        "no_drift_claim": False if not safe else False,
        "coverage_complete": index.path_count
        == index.snapshot.stats.disposition_count,
        "index_complete": True,
        "contract_discovery_complete": discovery_complete,
        "contract_measurement_complete": measurement_complete,
        "repository_index_health_status": status.value,
        "blocking_errors": blocking,
        "metrics": dict(health.get("metrics") or {}),
        "thresholds": dict(health.get("thresholds") or {}),
    }


def _build_graph_from_catalog(
    *,
    snapshot_id: str,
    catalog: McpContractCatalog,
    extraction: SwissKnifeContractExtraction | None = None,
) -> SymbolicContractGraph:
    nodes: list[ContractGraphNode] = []
    edges: list[ContractGraphEdge] = []
    for contract in catalog.contracts:
        node = ContractGraphNode(
            kind=ContractNodeKind.CONTRACT,
            stable_key=f"contract:{contract.contract_id}",
            snapshot_id=snapshot_id,
            provenance=ContractProvenance.SCHEMA,
            authority=ContractAuthority.REVIEWED_CONTRACT,
            version=GRAPH_VERSION,
            payload={
                "contract_id": contract.contract_id,
                "claim_family": contract.claim_family.value,
                "subject": contract.subject,
                "package_id": contract.package_id,
                "tool_name": contract.tool_name,
            },
            source_refs=tuple(contract.source_ids),
        )
        nodes.append(node)
    if extraction is not None:
        for edge in extraction.invocation_edges:
            edge_path = (
                edge.source_span.path
                if edge.source_span is not None and edge.source_span.path
                else f"edge:{edge.edge_id}"
            )
            source = ContractGraphNode(
                kind=ContractNodeKind.SYMBOL,
                stable_key=f"edge-source:{edge.edge_id}",
                snapshot_id=snapshot_id,
                provenance=ContractProvenance.AST,
                authority=ContractAuthority.SOURCE_OBSERVATION,
                version=GRAPH_VERSION,
                payload={"edge_id": edge.edge_id, "role": "source"},
                source_refs=(edge_path,),
            )
            target = ContractGraphNode(
                kind=ContractNodeKind.SYMBOL,
                stable_key=f"edge-target:{edge.edge_id}",
                snapshot_id=snapshot_id,
                provenance=ContractProvenance.AST,
                authority=ContractAuthority.SOURCE_OBSERVATION,
                version=GRAPH_VERSION,
                payload={"edge_id": edge.edge_id, "role": "target"},
                source_refs=(edge_path,),
            )
            nodes.extend((source, target))
            edges.append(
                ContractGraphEdge(
                    kind=(
                        ContractEdgeKind.RELATED_TO
                        if edge.compatibility or edge.bypass_candidate
                        else ContractEdgeKind.CALLS
                    ),
                    source=source.node_id,
                    target=target.node_id,
                    snapshot_id=snapshot_id,
                    provenance=ContractProvenance.AST,
                    authority=ContractAuthority.SOURCE_OBSERVATION,
                    version=GRAPH_VERSION,
                    mandatory=not (edge.compatibility or edge.bypass_candidate),
                    source_refs=(edge_path,),
                    payload={
                        "edge_id": edge.edge_id,
                        "kind": edge.kind.value,
                        "compatibility": edge.compatibility,
                        "bypass_candidate": edge.bypass_candidate,
                    },
                )
            )
    # Deduplicate nodes by stable_key / node_id.
    by_id = {node.node_id: node for node in nodes}
    return build_symbolic_contract_graph(
        snapshot_id=snapshot_id,
        nodes=tuple(by_id.values()),
        edges=tuple(edges),
    )


def _contract_terminals_from_catalog(
    catalog: McpContractCatalog,
    *,
    analysis_by_contract: Mapping[str, McpContractAnalysis] = MappingProxyType({}),
    mismatch_by_contract: Mapping[str, ContractFinding] = MappingProxyType({}),
    proof_by_contract: Mapping[str, Sequence[ProofPipelineOutcome]] = MappingProxyType(
        {}
    ),
    health_partial: bool,
    measurement_complete: bool,
    stale: bool,
) -> tuple[BaselineContractTerminal, ...]:
    rows: list[BaselineContractTerminal] = []
    for contract in catalog.contracts:
        reasons: list[str] = []
        status = TerminalContractStatus.UNKNOWN
        finding = mismatch_by_contract.get(contract.contract_id)
        analysis = analysis_by_contract.get(contract.contract_id)
        proof_outcomes = tuple(proof_by_contract.get(contract.contract_id) or ())
        if stale:
            status = TerminalContractStatus.STALE
            reasons.append("snapshot_stale")
        elif proof_outcomes:
            # Prefer the worst authoritative proof outcome across claim families.
            ranked = {
                TerminalContractStatus.REFUTED: 0,
                TerminalContractStatus.UNSUPPORTED: 1,
                TerminalContractStatus.STALE: 2,
                TerminalContractStatus.UNKNOWN: 3,
                TerminalContractStatus.PROVED: 4,
            }
            status = TerminalContractStatus.PROVED
            for item in proof_outcomes:
                candidate = TerminalContractStatus(item.terminal_status)
                if ranked[candidate] < ranked[status]:
                    status = candidate
                reasons.extend(item.reason_codes)
                if item.counterexample_id:
                    reasons.append(f"counterexample:{item.counterexample_id}")
                if item.cache_hit:
                    reasons.append("proof_cache_hit")
            if not reasons:
                reasons.append(f"proof_{status.value}")
        elif finding is not None:
            status = _status_from_mismatch(finding.state)
            for revision in finding.evidence:
                reasons.extend(revision.premise_ids)
            if finding.counterexample_id:
                reasons.append(f"counterexample:{finding.counterexample_id}")
            if not reasons:
                reasons.append(f"mismatch_{finding.state.value}")
        elif analysis is not None:
            status = _status_from_parity(analysis.state)
            for claim in analysis.claims:
                reasons.extend(claim.reason_codes)
            if not reasons:
                reasons.append(f"parity_{analysis.state.value}")
        else:
            if not measurement_complete:
                reasons.append("observed_contract_unavailable")
            if health_partial:
                reasons.append("partial_analyzer_health")
            if not reasons:
                reasons.append("not_measured")
            status = TerminalContractStatus.UNKNOWN
        rows.append(
            BaselineContractTerminal(
                contract_id=contract.contract_id,
                claim_family=contract.claim_family.value,
                package_id=contract.package_id,
                status=status,
                terminal=True,
                reason_codes=tuple(reasons),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.contract_id))


def _health_meta_contract(
    *,
    health_status: str,
    reason_code: str,
    health_partial: bool,
) -> BaselineContractTerminal:
    status = (
        TerminalContractStatus.PROVED
        if health_status == "healthy" and not health_partial
        else TerminalContractStatus.UNSUPPORTED
    )
    reasons = [reason_code]
    if health_partial or health_status != "healthy":
        reasons.append("partial_analyzer_health")
    return BaselineContractTerminal(
        contract_id="contract:sca-baseline-analyzer-health",
        claim_family="AnalyzerHealthSufficient",
        package_id="swissknife",
        status=status,
        terminal=True,
        reason_codes=tuple(reasons),
    )


def _status_counts(
    terminals: Sequence[BaselineContractTerminal],
) -> dict[str, int]:
    counts = {name: 0 for name in TERMINAL_STATUS_DOMAIN}
    for item in terminals:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1
    return counts


def _claims_projection(
    *,
    stages: Sequence[BaselineStageReceipt],
    health: Mapping[str, Any],
    terminals: Sequence[BaselineContractTerminal],
) -> dict[str, bool]:
    stages_ok = all(
        stage.completeness is StageCompleteness.COMPLETE
        for stage in stages
        if stage.name
        in {
            BaselineStageName.REPOSITORY_INDEX,
            BaselineStageName.EXTRACTION,
            BaselineStageName.CATALOG,
            BaselineStageName.GRAPH,
            BaselineStageName.PROOF_CACHE,
            BaselineStageName.MISMATCH,
        }
    )
    healthy = bool(health.get("safe_for_completion_reasoning")) and stages_ok
    # Never promote optional / model authority.
    no_findings = healthy and all(
        item.status is TerminalContractStatus.PROVED for item in terminals
    )
    # Explicit withhold: unhealthy or incomplete stages cannot claim no-drift.
    no_drift = False if not healthy else no_findings
    return {
        "exhaustive": bool(healthy and health.get("exhaustive")),
        "no_drift": no_drift,
        "no_findings": no_findings,
        "authority_promoted_from_optional_provider": False,
    }


def _summary_markdown(
    *,
    snapshot_id: str,
    index: RepositoryIndex | None,
    health: Mapping[str, Any],
    terminals: Sequence[BaselineContractTerminal],
    findings_payload: Mapping[str, Any],
    stages: Sequence[BaselineStageReceipt],
    llm_call_count: int,
) -> str:
    counts = _status_counts(terminals)
    stage_lines = [
        f"- Stage `{stage.name.value}`: `{stage.completeness.value}`"
        + (f" ({', '.join(stage.reason_codes)})" if stage.reason_codes else "")
        for stage in stages
    ]
    tracked = (
        index.snapshot.stats.tracked_path_count
        if index is not None
        else findings_payload.get("contract_population", {}).get(
            "tracked_path_count", "unknown"
        )
    )
    disposed = (
        index.snapshot.stats.disposition_count
        if index is not None
        else findings_payload.get("contract_population", {}).get(
            "disposition_count", "unknown"
        )
    )
    health_status = health.get("status", "missing")
    index_status = health.get("repository_index_health_status", "missing")
    safe = str(bool(health.get("safe_for_completion_reasoning"))).lower()
    claims = findings_payload.get("claims") or {}
    return "\n".join(
        [
            "# SwissKnife Symbolic Contract Baseline",
            "",
            (
                f"Snapshot `{snapshot_id}` was materialised by the complete "
                "symbolic contract assurance baseline pipeline. The deterministic "
                f"shadow scan made `{llm_call_count}` LLM calls and mutated no "
                "tracked source or backlog state."
            ),
            "",
            f"- Snapshot ID: `{snapshot_id}`",
            (
                f"- Repository index ID: "
                f"`{findings_payload.get('repository_index_root', '')}`"
            ),
            f"- Graph root: `{findings_payload.get('graph_root', '')}`",
            (
                f"- Extraction / catalog roots: "
                f"`{findings_payload.get('extraction_root', '')}` / "
                f"`{findings_payload.get('catalog_root', '')}`"
            ),
            f"- Analyzer health: `{health_status}` (index `{index_status}`)",
            f"- Safe for completion reasoning: `{safe}`",
            f"- Tracked path dispositions: `{disposed}` (tracked `{tracked}`)",
            (
                "- Contract terminals: "
                f"`{len(terminals)}` total "
                f"({counts.get('proved', 0)} proved, "
                f"{counts.get('refuted', 0)} refuted, "
                f"{counts.get('unknown', 0)} unknown, "
                f"{counts.get('unsupported', 0)} unsupported, "
                f"{counts.get('stale', 0)} stale)"
            ),
            (
                f"- Claims: exhaustive=`{str(claims.get('exhaustive')).lower()}`, "
                f"no_drift=`{str(claims.get('no_drift')).lower()}`, "
                f"no_findings=`{str(claims.get('no_findings')).lower()}`"
            ),
            (
                f"- Findings: `{len(findings_payload.get('findings') or [])}` "
                f"(root `{findings_payload.get('findings_root', '')}`)"
            ),
            f"- Model calls: `{llm_call_count}`",
            "",
            "## Pipeline stages",
            "",
            *stage_lines,
            "",
            (
                "Unhealthy or incomplete stages withhold exhaustive, no-drift, "
                "and no-findings claims. An empty findings list is not evidence of "
                "contract parity while measurement is incomplete."
            ),
            "",
            "Reproduce with:",
            "",
            "```sh",
            DEFAULT_REPRODUCTION_COMMAND,
            "```",
            "",
        ]
    )


def _finding_dict_from_health(
    *,
    snapshot_id: str,
    health: Mapping[str, Any],
    index: RepositoryIndex | None,
) -> dict[str, Any]:
    reason = str(health.get("reason_code") or "analyzer_health_blocker")
    if health.get("status") == "healthy" and health.get("safe_for_completion_reasoning"):
        return {}
    counterexample = {
        "kind": reason,
        "health_status": health.get("repository_index_health_status")
        or health.get("status"),
    }
    if index is not None:
        counterexample["parser_failure_count"] = int(
            getattr(index.build_stats, "parse_failure_count", 0) or 0
        )
        metrics = health.get("metrics") or {}
        if "parser_failure_ratio" in metrics:
            # Keep JSON-canonical ints/bools only in identity; ratios stay as
            # descriptive strings to avoid float canonicalisation issues.
            counterexample["parser_failure_ratio"] = str(metrics["parser_failure_ratio"])
    payload = {
        "affected_paths": ["swissknife"],
        "confidence": 1,
        "contract_id": "contract:sca-baseline-analyzer-health",
        "counterexample": counterexample,
        "lifecycle": FindingLifecycle.ACTIVE.value,
        "reason_code": reason,
        "reproduction": {
            "command": DEFAULT_REPRODUCTION_COMMAND,
            "expected_exit_code": 0,
            "expected_health_status": str(
                health.get("repository_index_health_status") or health.get("status")
            ),
            "expected_snapshot_root": snapshot_id,
        },
        "severity": "high",
        "snapshot_root": snapshot_id,
        "state": TerminalContractStatus.UNSUPPORTED.value
        if health.get("status") != "healthy"
        else TerminalContractStatus.UNKNOWN.value,
    }
    payload["finding_id"] = _sha256_label(
        {
            "schema": "sca-baseline-finding@1",
            "snapshot_root": snapshot_id,
            "contract_id": payload["contract_id"],
            "reason_code": reason,
            "counterexample": counterexample,
        }
    )
    return payload


def _ensure_artifact_size(path_label: str, payload: bytes, maximum: int) -> None:
    if len(payload) > maximum:
        raise ContractAssuranceBaselineError(
            f"{path_label} exceeds artifact envelope "
            f"({len(payload)} > {maximum} bytes)"
        )


def publish_baseline_artifacts(
    result: ContractAssuranceBaselineResult,
    output_root: str | Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> dict[str, Path]:
    """Atomically write coverage.json, contract_findings.json, and summary.md."""

    root = Path(output_root)
    coverage_bytes = _canonical_bytes(dict(result.coverage))
    findings_bytes = _canonical_bytes(dict(result.findings))
    summary_bytes = result.summary_markdown.encode("utf-8")
    _ensure_artifact_size("coverage.json", coverage_bytes, max_file_bytes)
    _ensure_artifact_size("contract_findings.json", findings_bytes, max_file_bytes)
    _ensure_artifact_size("summary.md", summary_bytes, max_file_bytes)
    paths = {
        "coverage": root / "coverage.json",
        "findings": root / "contract_findings.json",
        "summary": root / "summary.md",
    }
    _atomic_write_bytes(paths["coverage"], coverage_bytes)
    _atomic_write_bytes(paths["findings"], findings_bytes)
    _atomic_write_bytes(paths["summary"], summary_bytes)
    return paths


def materialize_contract_assurance_baseline(
    *,
    snapshot_id: str = "",
    repository_index: RepositoryIndex | None = None,
    snapshot: RepositorySnapshot | Mapping[str, Any] | None = None,
    extraction: SwissKnifeContractExtraction | None = None,
    catalog: McpContractCatalog | None = None,
    graph: SymbolicContractGraph | None = None,
    runtime_catalog: RuntimeComponentCatalog | None = None,
    observed_contracts: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]] = (),
    repo_root: str | Path | None = None,
    swissknife_root: str | Path | None = None,
    extract_expected: bool = True,
    project_graph: bool = True,
    run_traces: bool = True,
    run_parity: bool = True,
    run_proof_pipeline: bool = True,
    run_mismatch: bool = True,
    run_vulnerability: bool = True,
    proof_cache: TrustAwareProofCache | None = None,
    proof_cache_dir: str | Path | None = None,
    prover: McpContractProver | None = None,
    repository_id: str = "",
    toolchain_id: str = "",
    policy_id: str = "",
    output_root: str | Path | None = None,
    max_file_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    capability_root: str = "",
    scope_policy_root: str = "",
    allow_proof_without_healthy_index: bool = False,
) -> ContractAssuranceBaselineResult:
    """Run the complete zero-LLM baseline pipeline over one exact snapshot.

    Stages that cannot complete under partial health still emit typed terminals
    and withhold no-drift claims. Model call count is always zero.

    When ``run_proof_pipeline`` is true, reviewed parity claims are compiled into
    obligations, discharged through ``McpContractProver`` (including kernel
    policy for non-local routes), and memoized in the sole
    ``TrustAwareProofCache`` under exact snapshot/graph/policy/solver/kernel/
    toolchain roots (SCAEV071PROOFCACHE).
    """

    llm_call_count = 0
    stages: list[BaselineStageReceipt] = []

    # --- Stage: repository index / coverage ---------------------------------
    if repository_index is not None:
        snapshot_id = snapshot_id or repository_index.snapshot_id
        coverage = _coverage_from_index(repository_index)
        index_complete = StageCompleteness.COMPLETE
        index_reasons: list[str] = []
        if repository_index.health.status is not AnalyzerHealthStatus.HEALTHY:
            index_reasons.extend(repository_index.health.reasons)
            index_reasons.append("repository_index_unhealthy")
            index_complete = StageCompleteness.PARTIAL
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.REPOSITORY_INDEX,
                completeness=index_complete,
                reason_codes=tuple(index_reasons),
                root_id=repository_index.index_id,
                details={
                    "ast_index_id": repository_index.ast_index_id,
                    "row_count": repository_index.path_count,
                    "health_status": repository_index.health.status.value,
                },
            )
        )
    elif snapshot is not None:
        coverage = _coverage_from_snapshot(snapshot)
        snapshot_id = snapshot_id or str(
            coverage.get("snapshot_id")
            or (
                snapshot.snapshot_id
                if isinstance(snapshot, RepositorySnapshot)
                else ""
            )
        )
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.REPOSITORY_INDEX,
                completeness=StageCompleteness.PARTIAL,
                reason_codes=("repository_index_not_provided",),
                root_id=str(coverage.get("snapshot_id") or snapshot_id),
            )
        )
    else:
        raise ContractAssuranceBaselineError(
            "repository_index or snapshot is required"
        )

    snapshot_id = _text(snapshot_id, "snapshot_id")
    if not scope_policy_root:
        if repository_index is not None:
            scope_policy_root = repository_index.snapshot.scope_policy_id
        else:
            scope_policy_root = str(coverage.get("scope_policy_id") or "")

    # --- Stage: extraction --------------------------------------------------
    if extraction is None and extract_expected:
        root = Path(
            swissknife_root
            or (
                Path(repo_root) / "swissknife"
                if repo_root is not None
                else Path("swissknife")
            )
        )
        if root.is_dir():
            tree_id = (
                repository_index.snapshot.head_tree_id
                if repository_index is not None
                else str(coverage.get("head_tree_id") or snapshot_id)
            )
            try:
                extraction = SwissKnifeContractExtractor().extract_repository(
                    root,
                    repository_tree_id=tree_id,
                )
            except Exception as exc:  # noqa: BLE001 - typed withhold
                stages.append(
                    BaselineStageReceipt(
                        name=BaselineStageName.EXTRACTION,
                        completeness=StageCompleteness.FAILED,
                        reason_codes=("extraction_failed", type(exc).__name__),
                        details={"message": str(exc)[:512]},
                    )
                )
                extraction = None
        else:
            stages.append(
                BaselineStageReceipt(
                    name=BaselineStageName.EXTRACTION,
                    completeness=StageCompleteness.WITHHELD,
                    reason_codes=("swissknife_root_missing",),
                    details={"path": str(root)},
                )
            )

    if extraction is not None and not any(
        stage.name is BaselineStageName.EXTRACTION for stage in stages
    ):
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.EXTRACTION,
                completeness=StageCompleteness.COMPLETE,
                reason_codes=(),
                root_id=extraction.extraction_id,
                details={
                    "descriptor_count": len(extraction.descriptors),
                    "expectation_count": len(extraction.expectations),
                    "edge_count": len(extraction.invocation_edges),
                    "canonical_packages": list(extraction.canonical_packages_present),
                },
            )
        )
    elif extraction is None and not any(
        stage.name is BaselineStageName.EXTRACTION for stage in stages
    ):
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.EXTRACTION,
                completeness=StageCompleteness.WITHHELD,
                reason_codes=("extraction_not_provided",),
            )
        )

    # --- Stage: catalog -----------------------------------------------------
    if catalog is None and extraction is not None:
        catalog = extraction.catalog
    if catalog is not None:
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.CATALOG,
                completeness=StageCompleteness.COMPLETE,
                reason_codes=(),
                root_id=catalog.catalog_id,
                details={
                    "contract_count": len(catalog.contracts),
                    "source_count": len(catalog.sources),
                    "contradiction_count": len(catalog.contradictions),
                },
            )
        )
    else:
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.CATALOG,
                completeness=StageCompleteness.WITHHELD,
                reason_codes=("catalog_not_available",),
            )
        )

    # --- Stage: graph -------------------------------------------------------
    graph_reasons: list[str] = []
    if graph is None and project_graph:
        if repository_index is not None:
            try:
                graph = project_symbolic_contract_graph(repository_index)
            except Exception as exc:  # noqa: BLE001
                graph_reasons.extend(("graph_projection_failed", type(exc).__name__))
                graph = None
        if graph is None and catalog is not None:
            try:
                graph = _build_graph_from_catalog(
                    snapshot_id=snapshot_id,
                    catalog=catalog,
                    extraction=extraction,
                )
            except Exception as exc:  # noqa: BLE001
                graph_reasons.extend(("graph_from_catalog_failed", type(exc).__name__))
                graph = None
    if graph is not None:
        completeness = (
            StageCompleteness.COMPLETE
            if graph.complete
            else StageCompleteness.PARTIAL
        )
        if not graph.complete:
            graph_reasons.append("mandatory_closure_incomplete")
        if graph.snapshot_id != snapshot_id:
            raise ContractAssuranceBaselineError(
                "graph snapshot_id does not match baseline snapshot"
            )
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.GRAPH,
                completeness=completeness,
                reason_codes=tuple(graph_reasons),
                root_id=graph.graph_root,
                details={
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                    "complete": graph.complete,
                },
            )
        )
    else:
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.GRAPH,
                completeness=StageCompleteness.WITHHELD,
                reason_codes=tuple(graph_reasons or ("graph_not_available",)),
            )
        )

    # --- Stage: invocation traces + endpoint evidence (SCA-217) ------------
    traces: list[McpInvocationTrace] = []
    trace_reasons: list[str] = []
    evidence_compilation: RuntimeContractEvidenceCompilation | None = None
    evidence_findings: list[dict[str, Any]] = []
    # Caller-supplied observed contracts win; otherwise the evidence compiler
    # projects observed package contracts from reviewed catalog/index facts.
    observed_map: dict[str, Mapping[str, Any]] = {}
    if isinstance(observed_contracts, Mapping):
        observed_map = {
            str(key): value for key, value in observed_contracts.items()
        }
    else:
        for item in observed_contracts:
            op = str(item.get("operation_id") or item.get("name") or "")
            if op:
                observed_map[op] = item

    if catalog is not None and (
        run_traces or not observed_map
    ):
        try:
            evidence_compilation = compile_runtime_contract_evidence(
                catalog,
                snapshot_id=snapshot_id,
                graph=graph if run_traces else None,
                extraction=extraction,
                runtime_catalog=runtime_catalog,
                run_traces=bool(
                    run_traces
                    and graph is not None
                    and graph.complete
                ),
            )
        except Exception as exc:  # noqa: BLE001 - typed stage failure
            trace_reasons.extend(
                ("evidence_compilation_failed", type(exc).__name__)
            )
            evidence_compilation = None

    if evidence_compilation is not None:
        if not observed_map:
            observed_map = {
                key: dict(value)
                for key, value in evidence_compilation.observed_contract_map.items()
            }
        for finding in evidence_compilation.findings:
            evidence_findings.append(finding.to_dict())
        traces.extend(evidence_compilation.traces)

    if run_traces and catalog is not None and graph is not None and graph.complete:
        # Always bind the tracer so the graph remains path-queryable even when
        # every anchor is typed-unknown.
        _ = McpInvocationTracer(graph)
        if evidence_compilation is None:
            # Compilation failed above — stage failed closed with findings.
            stages.append(
                BaselineStageReceipt(
                    name=BaselineStageName.INVOCATION_TRACE,
                    completeness=StageCompleteness.FAILED,
                    reason_codes=tuple(sorted(set(trace_reasons))),
                    details={"trace_count": 0, "tracer_bound": True},
                )
            )
        else:
            unresolved = [
                anchor
                for anchor in evidence_compilation.anchors
                if anchor.resolution_state is not AnchorResolutionState.RESOLVED
            ]
            if evidence_compilation.complete and traces:
                completeness = StageCompleteness.COMPLETE
            elif evidence_compilation.anchors:
                # Missing/ambiguous anchors become typed unknown findings, not
                # a withheld empty-success stage.
                completeness = StageCompleteness.PARTIAL
                if unresolved:
                    trace_reasons.append("endpoint_anchors_partial")
                if evidence_compilation.findings:
                    trace_reasons.append("typed_unknown_anchor_findings")
                if not traces and any(
                    anchor.is_traceable for anchor in evidence_compilation.anchors
                ):
                    trace_reasons.append("traces_empty_for_traceable_anchors")
                elif not traces:
                    trace_reasons.append("no_traceable_anchors")
            else:
                completeness = StageCompleteness.PARTIAL
                trace_reasons.append("no_reviewed_runtime_operations")
            trace_reasons.extend(evidence_compilation.reason_codes)
            stages.append(
                BaselineStageReceipt(
                    name=BaselineStageName.INVOCATION_TRACE,
                    completeness=completeness,
                    reason_codes=tuple(sorted(set(trace_reasons))),
                    root_id=evidence_compilation.compilation_id,
                    details={
                        "trace_count": len(traces),
                        "anchor_count": len(evidence_compilation.anchors),
                        "resolved_anchor_count": sum(
                            1
                            for anchor in evidence_compilation.anchors
                            if anchor.resolution_state
                            is AnchorResolutionState.RESOLVED
                        ),
                        "observed_contract_count": len(
                            evidence_compilation.observed_contracts
                        ),
                        "finding_count": len(evidence_compilation.findings),
                        "tracer_bound": True,
                        "mcp_plus_plus_path_class": "mcp_plus_plus",
                        "direct_path_class": "direct",
                    },
                )
            )
    else:
        if not run_traces:
            trace_reasons.append("trace_stage_disabled")
        if graph is None:
            trace_reasons.append("graph_unavailable")
        elif not graph.complete:
            trace_reasons.append("graph_incomplete")
        if catalog is None:
            trace_reasons.append("catalog_unavailable")
        if not _index_healthy(repository_index) and repository_index is not None:
            trace_reasons.append("analyzer_unhealthy")
        # When anchors compiled but traces were not runnable, still surface the
        # compiled evidence rather than claiming empty success.
        if evidence_compilation is not None and evidence_compilation.anchors:
            stages.append(
                BaselineStageReceipt(
                    name=BaselineStageName.INVOCATION_TRACE,
                    completeness=StageCompleteness.PARTIAL,
                    reason_codes=tuple(sorted(set(trace_reasons))),
                    root_id=evidence_compilation.compilation_id,
                    details={
                        "trace_count": len(traces),
                        "anchor_count": len(evidence_compilation.anchors),
                        "observed_contract_count": len(
                            evidence_compilation.observed_contracts
                        ),
                        "finding_count": len(evidence_compilation.findings),
                        "tracer_bound": graph is not None,
                    },
                )
            )
        else:
            stages.append(
                BaselineStageReceipt(
                    name=BaselineStageName.INVOCATION_TRACE,
                    completeness=StageCompleteness.WITHHELD,
                    reason_codes=tuple(sorted(set(trace_reasons))),
                )
            )

    # --- Stage: proof / cache (SCAEV071PROOFCACHE) -------------------------
    proof_attempted = 0
    proof_proved = 0
    proof_refuted = 0
    proof_cache_hits = 0
    proof_pipeline_outcomes: tuple[ProofPipelineOutcome, ...] = ()
    proof_by_contract: dict[str, list[ProofPipelineOutcome]] = {}
    analyses: list[McpContractAnalysis] = []
    analysis_by_contract: dict[str, McpContractAnalysis] = {}
    measurement_complete = False
    health_partial = not _index_healthy(repository_index)
    proof_health_gate = health_partial and not allow_proof_without_healthy_index

    # Index observed contracts by operation_id, tool name, package:tool, and
    # bound contract_ids so catalog subjects join without name-only synthesis.
    observed_index: dict[str, Mapping[str, Any]] = dict(observed_map)
    for key, value in list(observed_map.items()):
        tool = str(value.get("tool_name") or value.get("name") or "")
        package = str(value.get("package_id") or "")
        if tool:
            observed_index.setdefault(tool, value)
        if package and tool:
            observed_index.setdefault(f"{package}:{tool}", value)
        for contract_id in value.get("contract_ids") or ():
            observed_index.setdefault(str(contract_id), value)
    traces_by_operation = {trace.operation_id: trace for trace in traces}

    if (
        run_parity
        and catalog is not None
        and observed_index
        and not proof_health_gate
    ):
        analyzer = McpContractAnalyzer()
        for contract in catalog.contracts:
            subject = contract.tool_name or contract.subject
            package_tool = (
                f"{contract.package_id}:{contract.tool_name}"
                if contract.package_id and contract.tool_name
                else ""
            )
            observed = (
                observed_index.get(subject)
                or observed_index.get(contract.contract_id)
                or observed_index.get(package_tool)
                or observed_index.get(contract.tool_name)
            )
            if observed is None:
                continue
            operation_id = str(
                observed.get("operation_id")
                or package_tool
                or subject
            )
            expected = {
                "operation_id": operation_id,
                "complete": True,
                "contract_id": contract.contract_id,
                "claim_family": contract.claim_family.value,
                "package_id": contract.package_id,
            }
            # Align observed operation_id with the expected analyzer key.
            observed_payload = dict(observed)
            observed_payload["operation_id"] = operation_id
            trace = traces_by_operation.get(operation_id)
            try:
                analysis = analyzer.analyze(
                    expected, observed_payload, trace=trace
                )
            except Exception:  # noqa: BLE001
                continue
            analyses.append(analysis)
            analysis_by_contract[contract.contract_id] = analysis

        if run_proof_pipeline and analyses:
            active_cache = proof_cache
            if active_cache is None and proof_cache_dir is not None:
                active_cache = TrustAwareProofCache(Path(proof_cache_dir))
            if active_cache is None and output_root is not None:
                active_cache = TrustAwareProofCache(
                    Path(output_root) / "proof-cache"
                )
            if active_cache is None:
                # Ephemeral sole cache for this baseline run.
                active_cache = TrustAwareProofCache()
            proof_pipeline_outcomes = run_contract_proof_pipeline(
                analyses=analyses,
                catalog=catalog,
                snapshot_id=snapshot_id,
                graph=graph,
                proof_cache=active_cache,
                prover=prover,
                repository_id=repository_id
                or (
                    repository_index.snapshot_id
                    if repository_index is not None
                    else DEFAULT_PROOF_REPOSITORY_ID
                ),
                toolchain_id=toolchain_id or DEFAULT_PROOF_TOOLCHAIN_ID,
                policy_id=policy_id
                or scope_policy_root
                or DEFAULT_PROOF_POLICY_ID,
                capability_root=capability_root or BASELINE_CAPABILITY_REPORT_ID,
            )
            for item in proof_pipeline_outcomes:
                proof_by_contract.setdefault(item.contract_id, []).append(item)
                proof_attempted += 1
                if item.terminal_status == TerminalContractStatus.PROVED.value:
                    proof_proved += 1
                elif item.terminal_status == TerminalContractStatus.REFUTED.value:
                    proof_refuted += 1
                if item.cache_hit:
                    proof_cache_hits += 1
        else:
            # Parity-only fallback when proof pipeline is disabled.
            for analysis in analyses:
                proof_attempted += 1
                if analysis.state is ParityState.SATISFIED:
                    proof_proved += 1
                elif analysis.state is ParityState.REFUTED:
                    proof_refuted += 1

        # Measurement is complete only when every tool-bearing reviewed contract
        # has an observed counterpart; interface-only contracts are optional.
        tool_contracts = [
            item
            for item in catalog.contracts
            if item.tool_name
        ]
        measured_targets = tool_contracts or list(catalog.contracts)
        measured_count = len(analysis_by_contract) or proof_attempted
        measurement_complete = measured_count > 0 and measured_count >= len(
            measured_targets
        )
        stage_reasons: list[str] = []
        if not measurement_complete:
            stage_reasons.append("observed_contract_coverage_incomplete")
        if health_partial:
            stage_reasons.append("partial_analyzer_health")
        if run_proof_pipeline and not proof_pipeline_outcomes and analyses:
            stage_reasons.append("proof_pipeline_produced_no_outcomes")
        completeness = (
            StageCompleteness.COMPLETE
            if measurement_complete
            and not health_partial
            and (not run_proof_pipeline or bool(proof_pipeline_outcomes))
            else StageCompleteness.PARTIAL
        )
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.PROOF_CACHE,
                completeness=completeness,
                reason_codes=tuple(stage_reasons),
                root_id=PROOF_PIPELINE_EVIDENCE,
                details={
                    "evidence_id": PROOF_PIPELINE_EVIDENCE,
                    "interface": PROOF_PIPELINE_INTERFACE,
                    "attempted": proof_attempted,
                    "proved": proof_proved,
                    "refuted": proof_refuted,
                    "cache_hits": proof_cache_hits,
                    "parity_analyses": len(analyses),
                    "cache_status": (
                        "computed_degraded" if health_partial else "computed"
                    ),
                    "sole_cache": "TrustAwareProofCache",
                    "prover": "McpContractProver",
                },
            )
        )
    else:
        reasons = []
        if proof_health_gate:
            reasons.append("partial_analyzer_health_proof_not_started")
        if not observed_map:
            reasons.append("observed_contracts_unavailable")
        if catalog is None:
            reasons.append("catalog_unavailable")
        if not run_parity:
            reasons.append("parity_stage_disabled")
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.PROOF_CACHE,
                completeness=StageCompleteness.WITHHELD,
                reason_codes=tuple(reasons),
                root_id=PROOF_PIPELINE_EVIDENCE,
                details={
                    "evidence_id": PROOF_PIPELINE_EVIDENCE,
                    "interface": PROOF_PIPELINE_INTERFACE,
                    "attempted": 0,
                    "proved": 0,
                    "refuted": 0,
                    "cache_hits": 0,
                    "cache_status": (
                        "published_unhealthy" if health_partial else "not_started"
                    ),
                    "sole_cache": "TrustAwareProofCache",
                    "prover": "McpContractProver",
                },
            )
        )

    # --- Stage: mismatch ----------------------------------------------------
    mismatch_findings: list[ContractFinding] = []
    mismatch_by_contract: dict[str, ContractFinding] = {}
    mismatch_analysis: MismatchAnalysis | None = None
    # Mismatch classifies analyzer refutations and proof-pipeline counterexamples.
    # Partial health withholds authority via claims projection, not classification.
    if run_mismatch and analyses and (
        not proof_health_gate or allow_proof_without_healthy_index
    ):
        mismatch_analyzer = ContractMismatchAnalyzer()
        for analysis in analyses:
            contract_id = (
                analysis.expected_contract_id
                or f"contract:{analysis.operation_id}"
            )
            for claim in analysis.claims:
                proof_for_claim = [
                    item
                    for item in proof_pipeline_outcomes
                    if item.contract_id == contract_id
                    and item.claim_family == claim.family.value
                ]
                proof_refuted = any(
                    item.terminal_status == TerminalContractStatus.REFUTED.value
                    for item in proof_for_claim
                )
                claim_for_mismatch = claim
                if claim.state is ParityState.SATISFIED and not proof_refuted:
                    continue
                if claim.state is ParityState.SATISFIED and proof_refuted:
                    # Analyzer observed parity but the proof pipeline refuted;
                    # only promote when a compact counterexample already exists
                    # on the claim (mismatch analyzer requires one for REFUTED).
                    if not claim.counterexamples:
                        continue
                    try:
                        claim_for_mismatch = ContractParityClaim(
                            family=claim.family,
                            state=ParityState.REFUTED,
                            operation_id=claim.operation_id,
                            premise_ids=claim.premise_ids,
                            reason_codes=tuple(
                                sorted(
                                    set(claim.reason_codes)
                                    | {"proof_pipeline_refuted"}
                                )
                            ),
                            counterexamples=claim.counterexamples,
                        )
                    except Exception:  # noqa: BLE001
                        continue
                obligation_ids = tuple(
                    item.obligation_id
                    for item in proof_for_claim
                    if item.obligation_id
                )
                produced = mismatch_analyzer.analyze_claim(
                    claim_for_mismatch,
                    snapshot_id=snapshot_id,
                    contract_id=contract_id,
                    affected_symbols=(f"operation:{analysis.operation_id}",),
                    affected_paths=(),
                    obligation_ids=obligation_ids,
                    cas_handles=(analysis.analysis_id,),
                    reproduction_commands=(DEFAULT_REPRODUCTION_COMMAND,),
                )
                for finding in produced:
                    mismatch_findings.append(finding)
                    mismatch_by_contract[finding.contract_id] = finding
        mismatch_analysis = MismatchAnalysis(
            snapshot_id=snapshot_id,
            findings=tuple(mismatch_findings),
            reason_codes=(
                ("mismatch_complete",)
                if measurement_complete
                else ("mismatch_partial",)
            ),
        )
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.MISMATCH,
                completeness=(
                    StageCompleteness.COMPLETE
                    if measurement_complete
                    else StageCompleteness.PARTIAL
                ),
                reason_codes=tuple(mismatch_analysis.reason_codes),
                root_id=mismatch_analysis.analysis_id,
                details={"finding_count": len(mismatch_findings)},
            )
        )
    else:
        reasons = []
        if health_partial:
            reasons.append("mismatch_withheld_until_analyzer_healthy")
        if not analyses:
            reasons.append("no_parity_claims_to_classify")
        if not run_mismatch:
            reasons.append("mismatch_stage_disabled")
        mismatch_analysis = MismatchAnalysis(
            snapshot_id=snapshot_id,
            findings=(),
            reason_codes=tuple(reasons or ("mismatch_not_run",)),
        )
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.MISMATCH,
                completeness=StageCompleteness.WITHHELD,
                reason_codes=tuple(mismatch_analysis.reason_codes),
                root_id=mismatch_analysis.analysis_id,
            )
        )

    # --- Stage: vulnerability ----------------------------------------------
    vulnerability_findings: tuple[ContractVulnerabilityFinding, ...] = ()
    if run_vulnerability and mismatch_findings:
        engine = ContractVulnerabilityRuleEngine()
        try:
            vulnerability_findings = engine.classify_many(mismatch_findings)
            stages.append(
                BaselineStageReceipt(
                    name=BaselineStageName.VULNERABILITY,
                    completeness=StageCompleteness.COMPLETE,
                    reason_codes=(),
                    details={"finding_count": len(vulnerability_findings)},
                )
            )
        except Exception as exc:  # noqa: BLE001
            stages.append(
                BaselineStageReceipt(
                    name=BaselineStageName.VULNERABILITY,
                    completeness=StageCompleteness.FAILED,
                    reason_codes=("vulnerability_classification_failed", type(exc).__name__),
                )
            )
    else:
        stages.append(
            BaselineStageReceipt(
                name=BaselineStageName.VULNERABILITY,
                completeness=StageCompleteness.WITHHELD,
                reason_codes=(
                    ("no_mismatch_findings",)
                    if not mismatch_findings
                    else ("vulnerability_stage_disabled",)
                ),
            )
        )

    # --- Terminals & findings document -------------------------------------
    discovery_complete = catalog is not None and len(catalog.contracts) > 0
    health = _health_projection(
        repository_index,
        discovery_complete=discovery_complete,
        measurement_complete=measurement_complete,
    )
    # Force no_drift_claim false in health projection until claims computed.
    health["no_drift_claim"] = False

    terminals: list[BaselineContractTerminal] = []
    if catalog is not None:
        terminals.extend(
            _contract_terminals_from_catalog(
                catalog,
                analysis_by_contract=analysis_by_contract,
                mismatch_by_contract=mismatch_by_contract,
                proof_by_contract=proof_by_contract,
                health_partial=health_partial,
                measurement_complete=measurement_complete,
                stale=False,
            )
        )
    terminals.append(
        _health_meta_contract(
            health_status=str(health.get("status") or "missing"),
            reason_code=str(health.get("reason_code") or "analyzer_health"),
            health_partial=health_partial or not bool(health.get("safe_for_completion_reasoning")),
        )
    )
    terminals_tuple = tuple(sorted(terminals, key=lambda item: item.contract_id))
    for item in terminals_tuple:
        if item.status.value not in TERMINAL_STATUS_DOMAIN:
            raise ContractAssuranceBaselineError(
                f"contract {item.contract_id} has illegal status {item.status}"
            )

    claims = _claims_projection(
        stages=stages,
        health=health,
        terminals=terminals_tuple,
    )
    health["no_drift_claim"] = bool(claims["no_drift"])
    health["exhaustive"] = bool(claims["exhaustive"])
    health["safe_for_completion_reasoning"] = bool(
        health.get("safe_for_completion_reasoning") and claims["exhaustive"]
    )

    coverage_id = _sha256_label(coverage)
    contracts_root = _sha256_label(
        [item.as_row() for item in terminals_tuple]
    )
    finding_rows: list[dict[str, Any]] = []
    if mismatch_findings:
        for finding in mismatch_findings:
            finding_rows.append(
                {
                    "finding_id": finding.finding_id,
                    "contract_id": finding.contract_id,
                    "state": finding.state.value,
                    "lifecycle": finding.lifecycle.value,
                    "snapshot_root": snapshot_id,
                    "affected_paths": list(finding.affected_paths),
                    "severity": "medium",
                }
            )
    health_finding = _finding_dict_from_health(
        snapshot_id=snapshot_id,
        health=health,
        index=repository_index,
    )
    if health_finding:
        finding_rows.append(health_finding)
    for evidence_row in evidence_findings:
        finding_rows.append(
            {
                "finding_id": evidence_row.get("finding_id", ""),
                "contract_id": evidence_row.get("operation_id", ""),
                "state": TerminalContractStatus.UNKNOWN.value,
                "lifecycle": "open",
                "snapshot_root": snapshot_id,
                "reason_code": evidence_row.get("reason_code", ""),
                "kind": evidence_row.get("kind", ""),
                "severity": "medium",
                "terminal": True,
            }
        )
    findings_root = _sha256_label(finding_rows)

    proof_reason = "ok"
    proof_stage = next(
        stage for stage in stages if stage.name is BaselineStageName.PROOF_CACHE
    )
    if proof_stage.reason_codes:
        proof_reason = proof_stage.reason_codes[0]

    cache_status = str(
        proof_stage.details.get("cache_status")
        or ("published_unhealthy" if health_partial else "not_started")
    )
    status_counts = _status_counts(terminals_tuple)
    measured = sum(
        1
        for item in terminals_tuple
        if item.status
        in {
            TerminalContractStatus.PROVED,
            TerminalContractStatus.REFUTED,
        }
    )

    findings_payload: dict[str, Any] = {
        "schema": BASELINE_FINDINGS_SCHEMA,
        "schema_version": 1,
        "snapshot_root": snapshot_id,
        "coverage_id": coverage_id,
        "repository_index_root": (
            repository_index.index_id if repository_index is not None else ""
        ),
        "graph_root": graph.graph_root if graph is not None else "",
        "extraction_root": (
            extraction.extraction_id if extraction is not None else ""
        ),
        "catalog_root": catalog.catalog_id if catalog is not None else "",
        "evidence_compilation_root": (
            evidence_compilation.compilation_id
            if evidence_compilation is not None
            else ""
        ),
        "endpoint_anchor_count": (
            len(evidence_compilation.anchors)
            if evidence_compilation is not None
            else 0
        ),
        "invocation_trace_count": len(traces),
        "contracts_root": contracts_root,
        "findings_root": findings_root,
        "scope_policy_root": scope_policy_root,
        "capability_root": capability_root or _sha256_label(
            {
                "schema": "sca-baseline-capability@1",
                "snapshot_id": snapshot_id,
                "stages": [stage.name.value for stage in stages],
            }
        ),
        "terminal_status_domain": list(TERMINAL_STATUS_DOMAIN),
        "analyzer_health": health,
        "authority": {
            "model_results_promoted": 0,
            "optional_provider_results_promoted": 0,
        },
        "claims": claims,
        "cache_outcomes": {
            "status": cache_status,
            "published_index": repository_index is not None,
            "reason_code": proof_reason,
            "reused_path_count": int(
                getattr(
                    getattr(repository_index, "build_stats", None),
                    "reused_path_count",
                    0,
                )
                or 0
            ),
            "cache_hit_ratio": 1 if repository_index is not None else 0,
            "proof_cache_hits": proof_cache_hits,
            "sole_cache": "TrustAwareProofCache",
        },
        "proof_outcomes": {
            "evidence_id": PROOF_PIPELINE_EVIDENCE,
            "interface": PROOF_PIPELINE_INTERFACE,
            "attempted": proof_attempted,
            "proved": proof_proved,
            "refuted": proof_refuted,
            "unknown": max(0, len(terminals_tuple) - proof_proved - proof_refuted),
            "cache_hits": proof_cache_hits,
            "reason_code": proof_reason,
            "pipeline_outcomes": [
                item.to_dict() for item in proof_pipeline_outcomes
            ],
        },
        "contract_population": {
            "contract_fields": [
                "contract_id",
                "claim_family",
                "package_id",
                "status",
                "terminal",
                "reason_codes",
            ],
            "contracts": [item.as_row() for item in terminals_tuple],
            "discovery_complete": discovery_complete,
            "measurement_complete": measurement_complete,
            "emitted_contract_count": len(terminals_tuple),
            "measured_contract_count": measured,
            "status_counts": status_counts,
        },
        "findings": finding_rows,
        "generation": {
            "command": DEFAULT_REPRODUCTION_COMMAND,
            "deterministic": True,
            "llm_call_count": llm_call_count,
            "reproducible": True,
        },
        "stages": [stage.to_dict() for stage in stages],
        "interface": CONTRACT_ASSURANCE_BASELINE_INTERFACE,
        "version": CONTRACT_ASSURANCE_BASELINE_VERSION,
    }

    if runtime_catalog is not None:
        findings_payload["runtime_catalog_root"] = runtime_catalog.to_dict().get(
            "catalog_id", ""
        ) or content_identity(runtime_catalog.to_dict())

    summary = _summary_markdown(
        snapshot_id=snapshot_id,
        index=repository_index,
        health=health,
        terminals=terminals_tuple,
        findings_payload=findings_payload,
        stages=stages,
        llm_call_count=llm_call_count,
    )

    stages.append(
        BaselineStageReceipt(
            name=BaselineStageName.PUBLISH,
            completeness=StageCompleteness.COMPLETE,
            reason_codes=(),
            root_id=findings_root,
            details={
                "coverage_id": coverage_id,
                "findings_root": findings_root,
                "contract_count": len(terminals_tuple),
            },
        )
    )
    findings_payload["stages"] = [stage.to_dict() for stage in stages]

    result = ContractAssuranceBaselineResult(
        snapshot_id=snapshot_id,
        coverage=coverage,
        findings=findings_payload,
        summary_markdown=summary,
        stages=tuple(stages),
        llm_call_count=llm_call_count,
        mismatch_analysis=mismatch_analysis,
        vulnerability_findings=vulnerability_findings,
        proof_pipeline_outcomes=proof_pipeline_outcomes,
        graph=graph,
        extraction=extraction,
        catalog=catalog,
        repository_index=repository_index,
        runtime_catalog=runtime_catalog,
    )

    if output_root is not None:
        publish_baseline_artifacts(
            result, output_root, max_file_bytes=max_file_bytes
        )
    return result


def materialize_baseline_from_repository_index(
    index: RepositoryIndex,
    *,
    output_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    swissknife_root: str | Path | None = None,
    observed_contracts: Sequence[Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]] = (),
    runtime_catalog: RuntimeComponentCatalog | None = None,
    max_file_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    proof_cache_dir: str | Path | None = None,
    proof_cache: TrustAwareProofCache | None = None,
    run_proof_pipeline: bool = True,
) -> ContractAssuranceBaselineResult:
    """Convenience entry used by ``index_repository_contracts``."""

    return materialize_contract_assurance_baseline(
        repository_index=index,
        output_root=output_root,
        repo_root=repo_root,
        swissknife_root=swissknife_root,
        observed_contracts=observed_contracts,
        runtime_catalog=runtime_catalog,
        max_file_bytes=max_file_bytes,
        proof_cache_dir=proof_cache_dir,
        proof_cache=proof_cache,
        run_proof_pipeline=run_proof_pipeline,
    )


__all__ = [
    "BASELINE_COVERAGE_SCHEMA",
    "BASELINE_FINDINGS_SCHEMA",
    "BASELINE_RUN_SCHEMA",
    "CONTRACT_ASSURANCE_BASELINE_INTERFACE",
    "CONTRACT_ASSURANCE_BASELINE_VERSION",
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "DEFAULT_REPRODUCTION_COMMAND",
    "PROOF_PIPELINE_EVIDENCE",
    "PROOF_PIPELINE_INTERFACE",
    "TERMINAL_STATUS_DOMAIN",
    "BaselineContractTerminal",
    "BaselineStageName",
    "BaselineStageReceipt",
    "ContractAssuranceBaselineError",
    "ContractAssuranceBaselineResult",
    "ProofPipelineOutcome",
    "StageCompleteness",
    "TerminalContractStatus",
    "materialize_baseline_from_repository_index",
    "materialize_contract_assurance_baseline",
    "publish_baseline_artifacts",
    "run_contract_proof_pipeline",
    "compile_runtime_contract_evidence",
    "RuntimeContractEvidenceCompiler",
]
